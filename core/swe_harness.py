"""Official repo-specific SWE-bench evaluation in cached Singularity images."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from swebench.harness.test_spec.test_spec import TestSpec

from core.checkpoints import digest
from core.profiles import atomic_json


class HarnessError(RuntimeError):
    """Environment/script/launcher failure; never a normal model score."""


def eval_script(instance: dict) -> str:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import make_eval_script_list_py

    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
    commands = make_eval_script_list_py(
        instance, specs, "testbed", "/testbed", instance["base_commit"], instance["test_patch"]
    )
    script = "#!/bin/bash\nset -uxo pipefail\n" + "\n".join(commands) + "\n"
    if instance["repo"] == "sphinx-doc/sphinx":
        script = re.sub(r"(?m)^(tox\b[^\n]*? -- )", r"\1-v --color=no -o console_output_style=classic ", script)
    return script


LAUNCH = r"""
set -e
export HOME=/tmp/swe_home
mkdir -p "$HOME"
export GIT_CONFIG_GLOBAL=/tmp/swe_gitconfig
git config --global --add safe.directory /testbed
cd /testbed
if [ -s /swe_grade/model.patch ]; then
    if ! git apply --verbose --recount /swe_grade/model.patch; then
        echo "__MODEL_PATCH_FAILED__"
        exit 2
    fi
fi
exec bash /swe_grade/eval.sh
"""


def evaluate(instance: dict, patch: str, *, label="model", timeout_s=1800) -> dict:
    from swebench.harness.grading import get_eval_report, get_logs_eval

    from core.tasks import swe

    iid = instance["instance_id"]
    image = swe.ensure_sif(iid)
    script = eval_script(instance)
    directory = swe.eval_log_dir() / iid / label
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "test_output.txt"
    (directory / "eval.sh").write_text(script, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="swe_grade_") as temporary:
        tmp = Path(temporary)
        (tmp / "model.patch").write_text(patch, encoding="utf-8")
        (tmp / "eval.sh").write_text(script, encoding="utf-8")
        with log.open("w", encoding="utf-8") as stream:
            try:
                subprocess.run(
                    [
                        "singularity",
                        "exec",
                        "--writable-tmpfs",
                        "--cleanenv",
                        "--no-home",
                        "--bind",
                        f"{tmp.resolve()}:/swe_grade:ro",
                        str(image),
                        "bash",
                        "-c",
                        LAUNCH,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                # Keep the partial log and generated patch for inspection/regrading.
                raise HarnessError(f"evaluation_timeout: {log}") from exc
    content = log.read_text(encoding="utf-8", errors="replace")
    f2p, p2p = swe.instance_tests(instance)
    if "__MODEL_PATCH_FAILED__" in content:
        report = swe._failed_report(f2p, p2p, "model_patch_apply_failed")
        report.update(resolved=False, failure_reason="model_patch_apply_failed")
    else:
        # Official grading reads only these five fields; image/build fields of
        # the full TestSpec are unnecessary for an already prepared container.
        spec = cast(
            "TestSpec",
            SimpleNamespace(
                instance_id=iid, repo=instance["repo"], version=instance["version"], FAIL_TO_PASS=f2p, PASS_TO_PASS=p2p
            ),
        )
        statuses, found = get_logs_eval(spec, str(log))
        if not found:
            raise HarnessError(f"evaluation_output_missing: {log}")
        # An empty map can be a model-induced collection/import error. Require
        # concrete evidence; an empty log/unknown parser failure is infrastructure.
        collection_error = bool(
            re.search(
                r"ERROR collecting|while loading conftest|ConftestImportFailure|errors? during collection",
                content,
                re.I,
            )
        )
        if not statuses and (label == "reference" or not collection_error):
            raise HarnessError(f"evaluation_status_missing: {log}")
        official = get_eval_report(
            spec, swe.predictions_entry(iid, patch, "benchmark"), str(log), include_tests_status=True
        )[iid]
        buckets = official.get("tests_status", {})
        report = swe._report(
            f2p, p2p, lambda name: name in {test for group in buckets.values() for test in group.get("success", [])}
        )
        report.update(resolved=official["resolved"], grading_report=official)
        if collection_error and not statuses:
            report["failure_reason"] = "model_test_collection_failed"
    report["log"] = str(log)
    atomic_json(directory / "report.json", report)
    return report


def ensure_reference(instance: dict) -> dict:
    """Validate the environment before paid generation; cache by image/script/reference versions."""
    from filelock import FileLock

    from core.tasks import swe

    image = swe.ensure_sif(instance["instance_id"])
    fingerprint = digest(
        {
            "instance": instance,
            "script": eval_script(instance),
            "harness": version("swebench"),
            "image": str(image.resolve()),
            "size": image.stat().st_size,
            "mtime": image.stat().st_mtime_ns,
            "launcher": LAUNCH,
        }
    )
    directory = Path.home() / ".cache/mas-promptbench/swe_reference"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fingerprint}.json"
    with FileLock(str(path) + ".lock", timeout=3600):
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            report = evaluate(instance, instance["patch"], label="reference")
            if report.get("resolved"):
                atomic_json(path, report)
        if not report.get("resolved"):
            raise HarnessError(f"reference_patch_failed: {report.get('log')}")
    return report
