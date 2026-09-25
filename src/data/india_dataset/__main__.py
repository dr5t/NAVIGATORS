"""Run with python -m src.data.india_dataset (no optional ML dependencies)."""
import argparse
import json
from pathlib import Path

from .collection import SessionCollector
from .evaluation import evaluate_release
from .pipeline import build_release, verify_release
from .schema import DATASET_NAME, Policy


def main(argv=None):
    parser = argparse.ArgumentParser(description="Navigators India Dataset collection, releases and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Append JSON samples from stdin to a new raw recording")
    collect.add_argument("path", type=Path)
    collect.add_argument("--metadata", type=Path, required=True)
    build = sub.add_parser("build", help="Archive, validate, normalize, split and report")
    build.add_argument("inputs", nargs="*", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--source-kind", choices=("navigators_collection", "external_benchmark", "synthetic"), default="navigators_collection")
    build.add_argument("--source-dataset", default=DATASET_NAME)
    build.add_argument("--split-strategy", choices=("session", "vehicle", "device", "joint"), default="joint")
    build.add_argument("--fractions", nargs=3, type=float, metavar=("TRAIN", "VALIDATION", "TEST"), default=(.7, .15, .15))
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--window-size", type=int, default=200)
    build.add_argument("--stride", type=int, default=50)
    build.add_argument("--policy", type=Path, help="JSON Policy overrides; stored in manifest")
    check = sub.add_parser("verify")
    check.add_argument("release", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("release", type=Path)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        import sys
        with SessionCollector(args.path, json.loads(args.metadata.read_text())) as collector:
            for line in sys.stdin:
                collector.append(json.loads(line))
        print(f"Raw recording saved: {args.path}; run build to validate it")
        return 0
    if args.command == "verify":
        manifest = verify_release(args.release)
        print(json.dumps({"verified": True, "manifest_sha256": manifest["manifest_sha256"]}))
        return 0
    if args.command == "evaluate":
        report = evaluate_release(args.release, args.predictions, args.reference, args.split)
        with args.output.open("x") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        print(args.output)
        return 0
    policy = Policy(**json.loads(args.policy.read_text())) if args.policy else Policy()
    release = build_release(args.inputs, args.output, args.version, source_kind=args.source_kind,
                            source_dataset=args.source_dataset, strategy=args.split_strategy,
                            fractions=dict(zip(("train", "validation", "test"), args.fractions)), seed=args.seed,
                            policy=policy, window_size=args.window_size, stride=args.stride)
    manifest = verify_release(release)
    print(json.dumps({"release": str(release), "statistics": manifest["statistics"]}, indent=2))
    return 2 if manifest["statistics"]["rejected_sessions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
