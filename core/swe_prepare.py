"""Prepare local SWE repositories/images with at most two simultaneous downloads. No model calls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from core.tasks import swe


def prepare(instance):
    iid = instance["instance_id"]
    repository = swe.repository_seed(instance["repo"], instance["base_commit"], iid)
    image = swe.ensure_sif(iid)
    return {"instance_id": iid, "repository": str(repository), "image": str(image)}


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    rows = swe.load_instances(only=args.only, limit=args.limit)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(prepare, rows):
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
