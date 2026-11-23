import subprocess
from pathlib import Path

import pytest

from core import swe_harness
from core.tasks import swe


def instance():
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    return {
        "instance_id": "pylint-dev__pylint-test",
        "repo": "pylint-dev/pylint",
        "version": next(iter(MAP_REPO_VERSION_TO_SPECS["pylint-dev/pylint"])),
        "base_commit": "0" * 40,
        "test_patch": "test patch",
        "patch": "reference patch",
        "FAIL_TO_PASS": '["test_a"]',
        "PASS_TO_PASS": '["test_b"]',
    }


def test_model_collection_failure_uses_official_unresolved_result(tmp_path, monkeypatch):
    from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

    monkeypatch.setenv("SWE_EVAL_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(swe, "ensure_sif", lambda _: tmp_path / "image.sif")
    monkeypatch.setattr(swe_harness, "eval_script", lambda _: "echo eval")

    def run(command, **options):
        options["stdout"].write(
            START_TEST_OUTPUT + "\nImportError while loading conftest.py\n"
            "ValueError: unsupported format character 'A' in %APPDATA%\n" + END_TEST_OUTPUT
        )
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(swe_harness.subprocess, "run", run)
    report = swe_harness.evaluate(instance(), "model patch")
    assert report["resolved"] is False
    assert report["failure_reason"] == "model_test_collection_failed"
    assert report["grading_report"]["resolved"] is False
    assert report["fail_to_pass"]["failure"] == ["test_a"]


def test_empty_container_output_is_infrastructure_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_EVAL_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(swe, "ensure_sif", lambda _: tmp_path / "image.sif")
    monkeypatch.setattr(swe_harness, "eval_script", lambda _: "echo eval")
    monkeypatch.setattr(swe_harness.subprocess, "run", lambda command, **_: subprocess.CompletedProcess(command, 1))
    with pytest.raises(swe_harness.HarnessError, match="evaluation_output_missing"):
        swe_harness.evaluate(instance(), "model patch")


def test_reference_cache_changes_with_image_and_script(tmp_path, monkeypatch):
    # Use a temporary HOME so no real reference cache is read or changed.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    image = tmp_path / "image.sif"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(swe, "ensure_sif", lambda _: image)
    script = ["script-v1"]
    monkeypatch.setattr(swe_harness, "eval_script", lambda _: script[0])
    evaluated = []

    def evaluate(inst, patch, **kwargs):
        evaluated.append(patch)
        return {"resolved": True}

    monkeypatch.setattr(swe_harness, "evaluate", evaluate)
    swe_harness.ensure_reference(instance())
    swe_harness.ensure_reference(instance())
    assert len(evaluated) == 1
    image.write_bytes(b"new fixture")
    swe_harness.ensure_reference(instance())
    script[0] = "script-v2"
    swe_harness.ensure_reference(instance())
    assert len(evaluated) == 3


def test_local_repository_reuse_does_not_reuse_previous_patch(tmp_path, monkeypatch):
    old = tmp_path / "old" / "example__repo-1"
    old.mkdir(parents=True)

    def git(*args):
        return subprocess.run(["git", "-C", str(old), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (old / "source.py").write_text("original\n")
    git("add", "source.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    commit = git("rev-parse", "HEAD")
    (old / "source.py").write_text("old model patch\n")
    monkeypatch.setenv("SWE_REPO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SWE_LOCAL_REPO_ROOTS", str(old.parent))
    monkeypatch.setenv("MAS_OFFLINE", "1")
    target = tmp_path / "new" / old.name
    assert swe.clone_and_checkout("example/repo", commit, target) == ""
    assert (target / "source.py").read_text() == "original\n"
    assert (old / "source.py").read_text() == "old model patch\n"
