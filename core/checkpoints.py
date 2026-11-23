"""Crash-safe per-row generation artifacts and explicit resume policy."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.profiles import atomic_json

OPTIONS: ContextVar[Mapping[str, Any]] = ContextVar("benchmark_run_options", default=MappingProxyType({}))
ACTIVE: ContextVar[tuple[Path, str, str] | None] = ContextVar("benchmark_row", default=None)
CALLS: ContextVar[dict[str, Any] | None] = ContextVar("benchmark_row_calls", default=None)
logger = logging.getLogger(__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def identity(instance) -> str:
    if isinstance(instance, tuple):
        instance = instance[0]  # BFCL pairs of question and ground truth
    return str(instance.get("id", instance.get("instance_id", "")))


@contextmanager
def options(**values):
    token = OPTIONS.set(values)
    try:
        yield
    finally:
        OPTIONS.reset(token)


@contextmanager
def row_context(directory: Path | None, instance):
    token = ACTIVE.set((directory, identity(instance), digest(instance)) if directory else None)
    calls_token = CALLS.set({"counts": {}, "stack": []})
    try:
        yield
    finally:
        ACTIVE.reset(token)
        CALLS.reset(calls_token)


def cached_call(fn, *, stage=None):
    """Save the complete successful generation before the caller grades it.

    Failed generations aren't cached. A grading retry reuses generation; an API
    failure is retried only with --resume --retry-execution-errors.
    """
    active = ACTIVE.get()
    if active is None:
        return _generate(fn)
    directory, iid, fingerprint = active
    calls = CALLS.get()
    if calls is None:
        raise RuntimeError("checkpoint_call_context_missing")
    parent = calls["stack"][-1] if calls["stack"] else ""
    if stage is None:
        index = calls["counts"].get(parent, 0)
        calls["counts"][parent] = index + 1
        stage = f"solve-{index}"
    stage = parent + "/" + stage
    path = directory / "artifacts" / f"{digest([iid, stage])}.json"
    if path.exists() and OPTIONS.get().get("resume"):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["instance_hash"] != fingerprint:
            raise ValueError(f"checkpoint_instance_mismatch: {iid}")
        if not _execution_failed(saved["output"]):
            calls["last_duration"] = saved["generation_s"]
            return saved["output"]
    started = time.monotonic()
    calls["stack"].append(stage)
    try:
        output = _generate(fn)
    finally:
        calls["stack"].pop()
    calls["last_duration"] = time.monotonic() - started
    if _execution_failed(output):
        return output
    atomic_json(
        path,
        {
            "id": iid,
            "instance_hash": fingerprint,
            "stage": stage,
            "generation_s": calls["last_duration"],
            "output": output,
        },
    )
    return output


def _generate(fn):
    """Remember existing successful work only; never send an extra request or adjust parameters."""
    from core import profiles

    config = None
    if OPTIONS.get().get("remember_model"):
        try:
            config = profiles.current()
        except (OSError, ValueError) as exc:
            logger.warning("model profile snapshot failed: %s", type(exc).__name__)
    output = fn()
    if config is not None and output is not None and not _execution_failed(output):
        try:
            name = profiles.remember(successful=True, config=config)
            logger.debug("successful model configuration saved: %s", name)
        except (OSError, ValueError) as exc:
            # A profile write failure must not discard a paid generation.
            logger.warning("successful model profile could not be saved: %s", type(exc).__name__)
    return output


def _execution_failed(output):
    return isinstance(output, dict) and (
        output.get("status") == "execution_error"
        or (bool(output.get("error")) and classify(dict(output))["status"] == "execution_error")
    )


def call_duration(fallback: float) -> float:
    calls = CALLS.get()
    return calls.get("last_duration", fallback) if calls else fallback


def classify(record: dict) -> dict:
    """Add English status without treating every legacy 'error' as an execution fault."""
    if "status" in record:
        return record
    stage = record.get("stage")
    if record.get("error") and (stage not in {"parse", "api_name"}):
        # API-Bank uses error for ordinary mismatch as well as framework faults.
        grading = record.get("error_category") == "grading_failure" or str(record["error"]).startswith(
            ("API name mismatch", "API result mismatch")
        )

        if not grading:
            record.update(
                status="execution_error",
                error_category="execution_error",
                failure_reason=record.get("failure_reason")
                or (
                    "scoring_failed"
                    if stage == "score"
                    else "api_request_failed"
                    if str(record["error"]).startswith(("BadRequestError:", "API", "AuthenticationError:"))
                    else "execution_failed"
                ),
            )
            return record
    score = next((record[k] for k in ("correct", "valid", "resolved", "em") if k in record), None)
    record["status"] = "completed" if score is None or bool(score) else "grading_failure"
    record["error_category"] = None if record["status"] == "completed" else "grading_failure"
    if record.get("failure_reason") is None and score is not None and not bool(score):
        diagnostic = record.get("diagnostics") or {}
        code_task = "predicted_code" in record
        missing = (code_task and not record["predicted_code"]) or any(
            key in record and record[key] is None for key in ("predicted_answer", "predicted_letter")
        )
        if missing and diagnostic.get("output_truncated"):
            record["failure_reason"] = "output_truncated"
        elif missing and diagnostic.get("turn_limit_reached"):
            record["failure_reason"] = "turn_limit_reached"
        elif missing:
            record["failure_reason"] = "code_not_found" if code_task else "answer_not_found"
        else:
            record["failure_reason"] = "wrong_answer"
    return record
