"""Local-first loading and judge inputs use temporary fixtures, never a live model."""

import base64
import json
import pickle
import zlib

import datasets
import pytest

from core import data
from core.tasks import lcb


def test_prepared_dataset_avoids_subsequent_hub_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "empty_hf")
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs["download_config"].local_files_only)
        return datasets.DatasetDict({"test": datasets.Dataset.from_dict({"id": ["a"], "question": ["Q"]})})

    monkeypatch.setattr(datasets, "load_dataset", download)
    result = data.load_dataset("test/benchmark")
    assert result["test"][0]["id"] == "a" and calls == [True]
    monkeypatch.setattr(datasets, "load_dataset", lambda *_a, **_k: pytest.fail("cached data must not contact HF"))
    assert data.load_dataset("test/benchmark")["test"][0]["question"] == "Q"


def test_explicit_local_hub_file_is_preferred(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_DATA_DIR", str(tmp_path))
    path = tmp_path / "test/benchmark/revision/file.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"a"}\n')
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *_a, **_k: pytest.fail("local file exists"))
    assert data.hub_file("test/benchmark", "file.jsonl", revision="revision") == str(path)


def test_appps_uses_pinned_raw_test_file_and_raw_id(tmp_path, monkeypatch):
    from core.tasks import apps

    monkeypatch.setenv("MAS_DATA_DIR", str(tmp_path))
    local = tmp_path / "codeparrot/apps" / data.APPS_REVISION / "test.jsonl"
    local.parent.mkdir(parents=True)
    local.write_text(
        json.dumps(
            {
                "id": 7,
                "question": "Read a number",
                "difficulty": "interview",
                "input_output": json.dumps({"inputs": ["1"], "outputs": ["1"]}),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "empty")
    rows = apps.load_instances(only=["7"])
    assert rows[0]["id"] == "7" and rows[0]["input_output"]["inputs"] == ["1"]


def test_lcb_metadata_single_list_argument_typing_and_large_stdin(monkeypatch):
    values = list(range(60000))  # exceeds Linux's per-argument argv limit
    tests = [{"testtype": "functional", "input": json.dumps(values) + "\n", "output": str(sum(values))}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    raw = {
        "question_id": "q1",
        "question_content": "Sum",
        "metadata": '{"func_name":"total"}',
        "private_test_cases": blob,
        "starter_code": "class Solution:",
        "difficulty": "easy",
    }
    monkeypatch.setattr(data, "lcb_rows", lambda: iter([raw]))
    loaded = lcb.load_instances(only=["q1"])
    assert loaded[0]["tests"][0]["fn_name"] == "total"
    code = "class Solution:\n def total(self, nums: List[int]) -> int:\n  return sum(nums)"
    report = lcb.run_tests(code, loaded[0]["tests"], timeout_s=10)
    assert report["pass_rate"] == 1.0, report


def test_lcb_separate_json_arguments(monkeypatch):
    code = "class Solution:\n def plus(self, nums: List[int], n: int) -> int:\n  return sum(nums) + n"
    tests = [
        {
            "fn_name": "plus",
            "testtype": "functional",
            "input_format": "json_lines",
            "input": "[1,2,3]\n4\n",
            "output": "10",
        }
    ]
    assert lcb.run_tests(code, tests)["pass_rate"] == 1.0


def test_apps_reuses_previous_normalized_local_slice_without_download(tmp_path, monkeypatch):
    from core.tasks import apps

    local = tmp_path / "eval_data.json"
    local.write_text(
        json.dumps(
            [
                {
                    "id": "0",
                    "problem": "P",
                    "difficulty": "interview",
                    "input_output": {"inputs": ["1"], "outputs": ["1"]},
                }
            ]
        )
    )
    monkeypatch.setenv("APPS_LOCAL_DATA", str(local))
    monkeypatch.setattr(data, "load_dataset", lambda *_a, **_k: pytest.fail("local APPS data must be used"))
    assert apps.load_instances(only=["0"])[0]["id"] == "0"
    with pytest.raises(ValueError, match="dataset_cache_subset_too_small"):
        apps.load_instances(limit=100)
    with pytest.raises(ValueError, match="dataset_ids_missing"):
        apps.load_instances(only=["1"])
