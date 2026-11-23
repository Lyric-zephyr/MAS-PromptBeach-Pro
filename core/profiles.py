"""Key-free profiles, including successful settings remembered from ordinary model runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC
from pathlib import Path

FIELDS = (
    "MODEL_ID",
    "VLLM_BASE_URL",
    "MODEL_PROVIDER",
    "TASK_MODEL_TEMPERATURE",
    "TASK_MODEL_TOP_P",
    "TASK_MODEL_MAX_TOKENS",
    "REQUEST_SEED",
    "ENABLE_THINKING",
    "TOKEN_LIMIT_FIELD",
    "MODEL_EXTRA_BODY",
    "MODEL_OMIT_PARAMS",
)


def root() -> Path:
    return Path(os.environ.get("MAS_PROFILE_DIR", Path.home() / ".config/mas-promptbench/models"))


def profile_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("profile name must contain only letters, numbers, dots, underscores or dashes")
    return root() / f"{name}.json"


def read(name: str) -> dict:
    label, separator, revision = name.partition("@")
    field = {"verified": "verified_settings", "successful": "successful_settings"}.get(revision)
    if separator and field is None:
        raise ValueError(f"unknown profile revision: {revision}")
    data = json.loads(profile_path(label).read_text(encoding="utf-8"))
    selected = data.get(field) if field else data["settings"]
    if selected is None:
        raise ValueError(f"profile_not_{revision}: {name}")
    return {key: str(value) for key, value in selected.items() if key in FIELDS}


def value(name: str, default: str) -> str:
    if name in os.environ:
        return os.environ[name]
    profile = os.environ.get("MAS_MODEL_PROFILE")
    if profile:
        return read(profile).get(name, default)
    # Identify the connection explicitly; never guess which model the user wants.
    model, endpoint = os.environ.get("MODEL_ID"), os.environ.get("VLLM_BASE_URL")
    if model and endpoint and name not in {"MODEL_ID", "VLLM_BASE_URL"}:
        path = profile_path(auto_name(model, endpoint))
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8")).get("successful_settings") or {}
            if saved.get("MODEL_ID") == model and saved.get("VLLM_BASE_URL") == endpoint:
                return saved.get(name, default)
    return default


def auto_name(model: str, endpoint: str) -> str:
    import hashlib

    suffix = hashlib.sha256(json.dumps([endpoint, model]).encode()).hexdigest()[:10]
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", model)[:64]
    return f"auto_{label}_{suffix}"


def atomic_json(path: Path, data: dict) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def current() -> dict:
    from urllib.parse import urlsplit

    from core import settings

    url = urlsplit(settings.base_url())
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("profile endpoint must not contain credentials, query parameters or fragments")
    defaults = {
        "MODEL_ID": settings.model_id(),
        "VLLM_BASE_URL": settings.base_url(),
        "TASK_MODEL_TEMPERATURE": str(settings.temperature()),
        "TASK_MODEL_TOP_P": str(settings.top_p()),
        "TASK_MODEL_MAX_TOKENS": str(settings.max_tokens()),
        "REQUEST_SEED": str(settings.request_seed()),
        "ENABLE_THINKING": str(settings.enable_thinking()).lower(),
    }
    return {key: value(key, defaults.get(key, "")) for key in FIELDS}


def remember(*, successful: bool = False, config: dict | None = None) -> str:
    """Keep the last successful generation settings separate from trials and manual verification."""
    from datetime import datetime

    settings = current() if config is None else {key: str(value) for key, value in config.items() if key in FIELDS}
    name = auto_name(settings["MODEL_ID"], settings["VLLM_BASE_URL"])
    path = profile_path(name)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    verified = previous.get("verified_settings")
    if (
        successful
        and previous.get("successful_settings") == settings
        and previous.get("settings") == settings
        and previous.get("status") in {"successful", "verified"}
    ):
        return name
    updates = {}
    if successful:
        updates = {"successful_settings": settings, "last_success_at": datetime.now(UTC).isoformat()}
    atomic_json(
        path,
        {
            **previous,
            **updates,
            "settings": settings,
            "status": "verified" if verified == settings else "successful" if successful else "last_used",
            "verified_settings": verified,
            "last_used_at": datetime.now(UTC).isoformat(),
        },
    )
    return name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    save = sub.add_parser("save", help="save current effective settings (never the API key)")
    save.add_argument("name")
    show = sub.add_parser("show")
    show.add_argument("name")
    verify = sub.add_parser("verify", help="mark a profile verified after your smoke test")
    verify.add_argument("name")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(p.stem for p in sorted(root().glob("*.json"))))
    elif args.command == "save":
        path = profile_path(args.name)
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        # Preserve the last verified settings separately when saving a trial revision.
        atomic_json(
            path,
            {
                **previous,
                "settings": current(),
                "status": "trial",
                "verified_settings": previous.get("verified_settings"),
            },
        )
        print(path)
    elif args.command == "verify":
        path = profile_path(args.name)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(status="verified", verified_settings=data["settings"])
        atomic_json(path, data)
        print(path)
    else:
        print(json.dumps(json.loads(profile_path(args.name).read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
