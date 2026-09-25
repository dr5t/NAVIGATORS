#!/usr/bin/env python3
"""Phase 12 benchmark. Never trains, exports, registers, or promotes models."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.research.runner import benchmark, resolve_config, verify_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps({"verified_report_sha256": verify_report(args.verify)}))
        return 0
    if args.config is None or args.output is None:
        parser.error("--config and --output are required for a new benchmark")
    coverage = benchmark(resolve_config(args.config), args.output, progress=lambda s: print(s, flush=True))
    print(json.dumps(coverage, indent=2))
    return 0 if coverage["jobs"] and coverage["status_counts"].get("completed") == coverage["jobs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
