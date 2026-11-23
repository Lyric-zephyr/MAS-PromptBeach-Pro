"""The runners' shared batch loop.

A runner supplies one callable per concern and :func:`run_batch` does the rest:

* ``row(index, instance) -> record`` solves and scores one instance (usually via
  :func:`attempt`, which times the call and turns a failure into an ``error``);
* ``summarize(records) -> dict`` computes the dataset's batch scores;
* ``header(index, total, instance) -> str`` and ``progress(index, total, record,
  records) -> str`` render the verbose progress before and after each row, logged
  at INFO; ``banner(summary) -> str`` renders the end-of-batch report, printed on
  ``stream`` (default stdout).

Records are written to ``out_path`` as JSON lines as soon as they exist, and each
:class:`Output` gets one line per instance too. JSONL views are rebuilt on each
invocation; resume reuses atomic per-row records and generation artifacts in
the adjacent .state directory. The CLI prevents accidental overwrites. The
summary returned is ``{**summarize(records), "total_s":
..., "per_instance": records}``, or ``summarize(records)`` alone with ``raw_summary``.
A batch in which every row failed on infrastructure (the model server unreachable
or refusing the request) raises :class:`InfrastructureFailure` once it is written.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from core import checkpoints

logger = logging.getLogger(__name__)
_FORK_ROW = None


def _process_row(index, instance, state, options):
    with checkpoints.options(**options), checkpoints.row_context(state, instance):
        return _FORK_ROW(index, instance)


# Exception types, as :func:`attempt` records them, that mean the model server could
# not be reached or refused the request: the row never got to the task.
INFRASTRUCTURE_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ConnectionError",
        "ConnectionRefusedError",
    }
)


class InfrastructureFailure(RuntimeError):
    """Every row of a batch failed on infrastructure (see :data:`INFRASTRUCTURE_ERRORS`)."""


def is_infrastructure_error(error: object) -> bool:
    """True when a record's ``error`` (``"<ExceptionType>: <message>"``) names an infrastructure exception."""
    return isinstance(error, str) and error.partition(":")[0] in INFRASTRUCTURE_ERRORS


@dataclass(frozen=True)
class Output:
    """A JSON-lines file with one line per instance: ``line(instance, record)``."""

    path: Path
    line: Callable[[dict, dict], dict]


def attempt(
    solve: Callable[[], dict],
    *,
    fallback: dict | None = None,
    propagate: bool = False,
) -> tuple[dict, float, str | None]:
    """Call ``solve`` and time it: ``(output, latency_s, error)``.

    On an exception the output is a copy of ``fallback`` (default ``{"answer": None}``)
    and ``error`` is ``"<ExceptionType>: <message>"``; with ``propagate`` the
    exception is re-raised instead.
    """
    start = time.time()
    try:
        out, error = checkpoints.cached_call(solve), None
        duration = checkpoints.call_duration(time.time() - start)
    except Exception as exc:
        if propagate:
            raise
        out, error = dict(fallback or {"answer": None}), f"{type(exc).__name__}: {exc}"
        duration = time.time() - start
    return out, duration, error


@contextmanager
def jsonl_writer(
    path: Path | None,
    *,
    default: Callable[[Any], Any] | None = None,
    ensure_ascii: bool = True,
) -> Iterator[Callable[[dict], None]]:
    """Empty ``path`` and yield ``write(record)`` adding one flushed JSON line to it (a no-op without a path)."""
    if path is None:
        yield lambda record: None
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:

        def write(record: dict) -> None:
            out.write(json.dumps(record, ensure_ascii=ensure_ascii, default=default) + "\n")
            out.flush()

        yield write


def run_batch(
    instances: list[dict],
    row: Callable[[int, dict], dict],
    *,
    summarize: Callable[[list[dict]], dict],
    out_path: Path | None = None,
    outputs: Sequence[Output] = (),
    ensure_ascii: bool = True,
    json_default: Callable[[Any], Any] | None = None,
    verbose: bool = True,
    stream: TextIO | None = None,
    header: Callable[[int, int, dict], str] | None = None,
    progress: Callable[[int, int, dict, list[dict]], str] | None = None,
    banner: Callable[[dict], str] | None = None,
    raw_summary: bool = False,
    resume: bool | None = None,
    retry_execution_errors: bool | None = None,
    workers: int = 1,
) -> dict:
    """Run ``row`` on every instance, write the records and return the summary.

    Raises :class:`InfrastructureFailure`, after the records and the report are
    written, when every row's ``error`` is an infrastructure error.
    """
    global _FORK_ROW
    options = checkpoints.OPTIONS.get()
    if workers not in {1, 2}:
        raise ValueError("workers must be 1 or 2")
    if workers == 2 and "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("parallel evaluation requires WSL/Linux")
    if workers == 2 and _FORK_ROW is not None:
        raise RuntimeError("nested parallel batches are unsupported")
    resume = options.get("resume", False) if resume is None else resume
    retry_execution_errors = (
        options.get("retry_execution_errors", False) if retry_execution_errors is None else retry_execution_errors
    )
    if retry_execution_errors and not resume:
        raise ValueError("retry_execution_errors requires resume")
    state = Path(str(out_path) + ".state") if out_path is not None else None
    if resume and state is None:
        raise ValueError("resume requires an output path")
    paths = [Path(path).resolve() for path in ([out_path] if out_path is not None else []) + [o.path for o in outputs]]
    if len(paths) != len(set(paths)):
        raise ValueError("result and prediction output paths must be different")
    previous = {}
    if state:
        from core import settings
        from core.profiles import atomic_json, current

        ids = [checkpoints.identity(inst) for inst in instances]
        if any(ids) and (len(set(ids)) != len(ids) or any(i in {"", "None"} for i in ids)):
            raise ValueError("dataset_invalid_ids: missing or duplicate IDs")
        manifest = {
            "model": current(),
            "decoding": settings.decoding(),
            "instances_hash": checkpoints.digest(instances),
            "entrypoint": options.get("entrypoint"),
            "arguments": options.get("arguments"),
        }
        manifest_path = state / "run_config.json"
        has_saved_work = any(next((state / name).glob("*.json"), None) is not None for name in ("artifacts", "records"))
        if not resume and has_saved_work:
            if options.get("protect_outputs"):
                raise FileExistsError(f"checkpoint_exists: {state}; use --resume or a new --out-dir")
            # Low-level callers retain their historical 'fresh run' behavior, but
            # keep old artifacts in an archive rather than mixing two attempts.
            parent = state.parent.resolve()
            archive = state.with_name(state.name + ".previous-" + uuid.uuid4().hex[:8])
            if not state.resolve().is_relative_to(parent) or not archive.resolve().is_relative_to(parent):
                raise ValueError("checkpoint directory must remain within the output directory")
            state.rename(archive)
        if manifest_path.exists() and resume:
            import json

            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("resume_config_mismatch: use a new output directory for changed model/data/options")
        else:
            if options.get("protect_outputs") and out_path.exists() and out_path.stat().st_size:
                raise FileExistsError(f"output_exists: {out_path}; use --resume or a new --out-dir")
            if options.get("protect_outputs") and not resume:
                for output in outputs:
                    if output.path.exists() and output.path.stat().st_size:
                        raise FileExistsError(f"output_exists: {output.path}; use --resume or a new --out-dir")
            if resume and out_path.exists() and out_path.stat().st_size:
                raise ValueError("resume_manifest_missing: legacy outputs need a new directory")
            atomic_json(manifest_path, manifest)
        if options.get("remember_model"):
            from core.profiles import remember

            logger.info("model profile remembered: %s", remember())
        if resume:
            # Per-row atomic records survive interruption while rebuilding JSONL views.
            for saved in (state / "records").glob("*.json"):
                record = json.loads(saved.read_text(encoding="utf-8"))
                previous[str(record.get("id", record.get("instance_id", "")))] = record
    records: list[dict] = []
    start = time.time()

    def log(render: Callable[..., str] | None, *args) -> None:
        if verbose and render is not None:
            logger.info(render(*args))

    with ExitStack() as files:
        writer_options = {"default": json_default, "ensure_ascii": ensure_ascii}
        write = files.enter_context(jsonl_writer(out_path, **writer_options))
        extra = [(output.line, files.enter_context(jsonl_writer(output.path, **writer_options))) for output in outputs]
        futures = {}
        submit_next = None
        if workers == 2:
            _FORK_ROW = row
            files.callback(lambda: globals().__setitem__("_FORK_ROW", None))
            pool = files.enter_context(
                ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork"))
            )
            pending = iter(
                (index, instance)
                for index, instance in enumerate(instances)
                if (prior := previous.get(checkpoints.identity(instance))) is None
                or (retry_execution_errors and checkpoints.classify(dict(prior))["status"] == "execution_error")
            )

            def submit_next():
                item = next(pending, None)
                if item is not None:
                    index, instance = item
                    futures[index] = pool.submit(_process_row, index, instance, state, {**options, "resume": resume})

            submit_next()
            submit_next()
        for index, instance in enumerate(instances):
            log(header, index, len(instances), instance)
            iid = checkpoints.identity(instance)
            prior = previous.get(iid)
            retry = (
                prior is not None
                and retry_execution_errors
                and checkpoints.classify(dict(prior))["status"] == "execution_error"
            )
            with checkpoints.options(**{**options, "resume": resume}), checkpoints.row_context(state, instance):
                if prior is not None and not retry:
                    record = prior
                else:
                    record = futures[index].result() if index in futures else row(index, instance)
                    if index in futures:
                        submit_next()
                    checkpoints.classify(record)
                    if state:
                        atomic_json(state / "attempts" / f"{checkpoints.digest(iid)}-{uuid.uuid4().hex}.json", record)
                        atomic_json(state / "records" / f"{checkpoints.digest(iid)}.json", record)
            records.append(record)
            for line, write_line in extra:
                write_line(line(instance, record))
            write(record)
            log(progress, index, len(instances), record, records)
    summary = summarize(records)
    if not raw_summary:
        summary = {**summary, "total_s": round(time.time() - start, 1), "per_instance": records}
    if verbose and banner is not None:
        print(banner(summary), file=stream, flush=True)
    if records and all(is_infrastructure_error(record.get("error")) for record in records):
        raise InfrastructureFailure(
            f"every row ({len(records)}) failed on infrastructure; the first error: {records[0]['error']}"
        )
    return summary


def write_trace(path: Path, sections: Iterable[tuple[str, str]]) -> None:
    """Write a text trace of ``(title, text)`` sections, each as ``=== title ===`` followed by the text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for title, text in sections:
            out.write(f"=== {title} ===\n{text}\n\n")
