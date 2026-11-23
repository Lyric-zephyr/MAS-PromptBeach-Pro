"""Validated local-first dataset loading, shared by the benchmark tasks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from core.profiles import atomic_json

logger = logging.getLogger(__name__)
APPS_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"


def root() -> Path:
    return Path(os.environ.get("MAS_DATA_DIR", Path.home() / ".cache/mas-promptbench/data"))


def offline() -> bool:
    return os.environ.get("MAS_OFFLINE", "0").lower() in {"1", "true"}


def hub_file(repo_id: str, filename: str, *, repo_type: str = "dataset", revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    local = root() / repo_id / (revision or "default") / filename
    if local.is_file():
        return str(local)
    try:
        return hf_hub_download(repo_id, filename, local_files_only=True, repo_type=repo_type, revision=revision)
    except (FileNotFoundError, OSError):
        if offline():
            raise FileNotFoundError(f"dataset_cache_missing: {local}; unset MAS_OFFLINE to download") from None
    return hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision)


def load_dataset(path: str, *args, **kwargs):
    """Prepared local copy -> existing HF cache -> download. Never replace a corrupt local copy silently.

    Prepared Arrow snapshots are keyed by dataset/config/split/revision and reused
    without contacting HF. MAS_DATA_DIR can be shared by all model runs.
    """
    import datasets

    spec = {"dataset": path, "args": args, "options": kwargs}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    target = root() / "prepared" / path.replace("/", "--") / digest
    if target.exists():
        manifest = json.loads((target / "source.json").read_text(encoding="utf-8"))
        if manifest != json.loads(json.dumps(spec)):
            raise ValueError(f"dataset_version_mismatch: {target}")
        logger.info("dataset cache: %s", target)
        return datasets.load_from_disk(str(target / "arrow"))
    cached = _arrow_cache(path, args, kwargs)
    if cached is not None:
        dataset = cached
    elif path == "codeparrot/apps":
        split = kwargs.get("split", "test")
        filename = hub_file(path, f"{split}.jsonl", revision=APPS_REVISION)
        dataset = datasets.load_dataset("json", data_files=filename, split="train")
    else:
        from datasets import DownloadConfig

        if offline():
            raise FileNotFoundError(f"dataset_cache_missing: {path}; unset MAS_OFFLINE to prepare the dataset")

        try:
            dataset = datasets.load_dataset(
                path, *args, **kwargs, download_config=DownloadConfig(local_files_only=True)
            )
        except (FileNotFoundError, ConnectionError, OSError):
            if offline():
                raise FileNotFoundError(f"dataset_cache_missing: {path}; unset MAS_OFFLINE to download") from None
            dataset = datasets.load_dataset(path, *args, **kwargs)
    if not isinstance(dataset, (datasets.Dataset, datasets.DatasetDict)):
        raise TypeError("dataset_streaming_unsupported: prepared caches require a materialized dataset")
    # A crash cannot leave a partial prepared dataset looking complete.
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=target.parent))
    try:
        dataset.save_to_disk(str(temporary / "arrow"))
        atomic_json(temporary / "source.json", spec)
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return dataset


def _arrow_cache(path, args, kwargs):
    """Read an unambiguous existing datasets Arrow cache without any Hub requests."""
    import datasets
    from datasets.config import HF_DATASETS_CACHE

    if kwargs.get("revision"):
        return None
    config = args[0] if args else kwargs.get("name", "default")
    cache_root = Path(HF_DATASETS_CACHE)
    name = path.replace("/", "___")
    matches = (
        [p for p in cache_root.iterdir() if p.is_dir() and p.name.casefold() == name.casefold()]
        if cache_root.exists()
        else []
    )
    directory = (matches[0] if len(matches) == 1 else cache_root / name) / config
    candidates = list(directory.glob("*/*/dataset_info.json"))
    if len(candidates) != 1:
        return None  # never guess among different cached versions
    info_path = candidates[0]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    splits = {}
    for split, description in info["splits"].items():
        if kwargs.get("split") and split != kwargs["split"]:
            continue
        files = sorted(info_path.parent.glob(f"*-{split}.arrow"))
        if not files:
            files = sorted(info_path.parent.glob(f"*-{split}-*.arrow"))
        if not files:
            return None
        parts = [datasets.Dataset.from_file(str(file)) for file in files]
        combined = parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)
        if len(combined) != description["num_examples"]:
            raise ValueError(f"dataset_cache_incomplete: {info_path}")
        splits[split] = combined
    logger.info("existing HF Arrow cache: %s", info_path.parent)
    if kwargs.get("split") and kwargs["split"] not in splits:
        return None
    return splits[kwargs["split"]] if kwargs.get("split") else datasets.DatasetDict(splits)


def lcb_rows():
    """Pinned raw JSONL files, streamed rather than copying gigabytes of private tests."""
    release = int(os.environ.get("LCB_RELEASE", "3"))
    if release not in range(1, 7):
        raise ValueError("LCB_RELEASE must be an integer from 1 to 6")
    revision = os.environ.get("LCB_REVISION", LCB_REVISION)
    for number in range(1, release + 1):
        name = "test.jsonl" if number == 1 else f"test{number}.jsonl"
        filename = hub_file("livecodebench/code_generation_lite", name, revision=revision)
        with Path(filename).open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)


def validate_ids(rows: list[dict], requested: list[str] | None = None, *, key: str = "id") -> None:
    ids = [str(row[key]) for row in rows]
    if len(ids) != len(set(ids)) or any(i in {"", "None"} for i in ids):
        raise ValueError("dataset_invalid_ids: missing or duplicate IDs")
    if requested is not None and set(requested) - set(ids):
        raise ValueError(f"dataset_ids_missing: {sorted(set(requested) - set(ids))}")
