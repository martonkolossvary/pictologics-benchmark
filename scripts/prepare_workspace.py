#!/usr/bin/env python3
"""Prepare or revalidate all three benchmark input pillars without timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.benchmark_workspace import prepare_workspace, validate_workspace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/benchmark")
    parser.add_argument("--ibsi2-source")
    parser.add_argument("--pillar1-profile", default="configs/benchmark/pillar1.json")
    parser.add_argument("--pillar2-profile", default="configs/benchmark/pillar2.json")
    parser.add_argument(
        "--endpoint-contract",
        default="configs/benchmark/main.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Deeply revalidate existing inputs and refresh the workspace manifest",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="Development-only: skip expensive Pillar 1 byte-for-byte recomputation",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    if args.validate_only:
        record = validate_workspace(root, deep=not args.shallow)
    else:
        if not args.ibsi2_source:
            parser.error("--ibsi2-source is required unless --validate-only is used")
        record = prepare_workspace(
            root,
            ibsi2_source=Path(args.ibsi2_source),
            pillar1_profile=Path(args.pillar1_profile),
            pillar2_profile=Path(args.pillar2_profile),
            endpoint_contract_path=Path(args.endpoint_contract),
            resume=args.resume,
        )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
