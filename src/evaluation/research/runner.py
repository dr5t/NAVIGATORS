"""Reproducible scheduling, isolated execution, reports and input/output hashes."""
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

from data.india_dataset.schema import canonical, digest
from evaluation.recording import sha256
from . import BENCHMARK_VERSION
from .config import ARCHITECTURES, ENVIRONMENTS, configuration, interventions
from .inputs import read_session, cases, prepare_case, generalization
from .metrics import DEFINITIONS
from .statistics import aggregate, paired_statistics
from .worker import load_model, run_job

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve_config(path):
    path = Path(path).resolve()
    cfg = configuration(json.loads(path.read_text()))
    for session in cfg["sessions"]:
        for key in ("release", "reference", "map"):
            if session.get(key):
                session[key] = str((path.parent / session[key]).resolve())
    for key in ("production_model", "candidate_model"):
        spec = cfg[key]
        if spec:
            for field in ("path", "stats", "training_inventory"):
                if spec.get(field):
                    spec[field] = str((path.parent / spec[field]).resolve())
    if cfg["candidate_model"] is None:
        cfg["candidate_model"] = cfg["production_model"]
    return cfg


def provenance():
    paths = list((ROOT / "evaluation").rglob("*.py")) + list((ROOT / "navigation").rglob("*.py"))
    paths += list((ROOT / "data" / "india_dataset").rglob("*.py"))
    code = {str(path.relative_to(ROOT.parent)): sha256(path) for path in sorted(paths)}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, []
    return dict(benchmark_version=BENCHMARK_VERSION, code_sha256=code, git_commit=commit, working_tree_status=status,
                python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
                timing="perf_counter; CPU inference; all first calls retained; one process per run",
                memory="Worker process peak RSS including imports, model, trajectory storage and scoring")


def audit_models(config):
    audits, files = {}, {}
    for role in ("production_model", "candidate_model"):
        spec = config[role]
        if not spec:
            audits[role] = dict(available=False, error="No model configured")
            continue
        for key in ("path", "stats", "training_inventory"):
            if spec.get(key) and Path(spec[key]).is_file():
                files[spec[key]] = sha256(spec[key])
        try:
            model = load_model(spec)
            meta = model.metadata()
            files.update(meta["files"])
            audits[role] = dict(available=True, metadata=meta)
        except Exception as error:
            audits[role] = dict(available=False, error_type=type(error).__name__, error=str(error))
    return audits, files


def benchmark(config, output, *, isolated=True, model_factory=None, progress=None):
    """Publish a fresh report directory; model_factory is an in-process test seam."""
    config = configuration(config)
    if config["candidate_model"] is None:
        config["candidate_model"] = config["production_model"]
    if isolated and model_factory is not None:
        raise ValueError("A model override is allowed only for explicitly in-process software tests")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Benchmark output already exists; use a new run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=output.parent) as tmp:
        work = Path(tmp) / "report"
        work.mkdir()
        write_json(work / "benchmark_config.json", config)
        write_json(work / "architectures.json", interventions())
        write_json(work / "metric_definitions.json", DEFINITIONS)
        repro = provenance()
        repro["config_sha256"] = digest(config)
        repro["isolated_workers"] = isolated
        if model_factory:
            audits, model_files = {"test_override": {"available": True, "scope": "synthetic software tests only"}}, {}
        else:
            audits, model_files = audit_models(config)
        repro["model_audit"], repro["model_files"] = audits, model_files
        write_json(work / "reproducibility.json", repro)
        results, failures, sessions, seen = [], [], [], set()
        for spec in config["sessions"]:
            try:
                session = read_session(spec)
                key = (session.identity["source_dataset"], session.identity["session_id"])
                if key in seen:
                    raise ValueError("Duplicate benchmark session (including across releases)")
                seen.add(key)
                gen = generalization(session, [config["production_model"], config["candidate_model"]])
                sessions.append(dict(identity=session.identity, metadata=session.entry["metadata"], generalization=gen))
            except Exception as error:
                failures.append(dict(stage="input_validation", session=spec, error_type=type(error).__name__, error=str(error)))
                continue
            for case in cases(session, config):
                eligibility_error = None
                try:
                    prepare_case(session, case, config)
                except Exception as error:
                    eligibility_error = str(error)
                for repeat in range(config["repeats"]):
                    # Seeded order minimizes systematic warm-machine ordering bias.
                    import random
                    architectures = list(config["architectures"])
                    random.Random(config["seed"] + repeat + int(digest([key, case])[:8],16)).shuffle(architectures)
                    for architecture in architectures:
                        job_id = digest([session.identity, case, repeat, architecture, digest(config)])[:24]
                        job = dict(job_id=job_id, architecture=architecture, session=spec, case=case, repeat=repeat,
                                   config=config, input_identity=session.identity, generalization=gen, model_files=model_files)
                        directory = work / "raw_results" / job_id
                        directory.mkdir(parents=True)
                        write_json(directory / "job.json", job)
                        reason = eligibility_error
                        role = "production_model" if architecture == "production" else "candidate_model"
                        if not reason and ARCHITECTURES[architecture].ai_velocity and not model_factory and not audits[role]["available"]:
                            reason = audits[role]["error"]
                        if reason:
                            result = dict(status="unsupported", job_id=job_id, architecture=architecture, case=case,
                                          repeat=repeat, input_identity=session.identity, error=reason)
                        else:
                            if progress:
                                progress(f"Running {architecture}: {key[1]}, {case}, repeat {repeat}")
                            try:
                                if isolated:
                                    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONHASHSEED": str(config["seed"]),
                                           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                                    process = subprocess.run([sys.executable, "-m", "evaluation.research.worker",
                                                              str(directory / "job.json"), str(directory)],
                                                             env=env, cwd=ROOT.parent, capture_output=True, text=True,
                                                             timeout=config["timeout_s"])
                                    (directory / "worker.log").write_text(process.stdout + process.stderr)
                                    result = json.loads((directory / "result.json").read_text())
                                    if process.returncode and result["status"] == "completed":
                                        raise RuntimeError(f"Worker exited {process.returncode} after reporting completion")
                                else:
                                    result = run_job(job, directory, model_factory(architecture) if model_factory else None, isolated=False)
                            except Exception as error:
                                result = dict(status="failed", job_id=job_id, architecture=architecture, case=case,
                                              repeat=repeat, input_identity=session.identity, error=str(error),
                                              error_type=type(error).__name__, traceback=traceback.format_exc())
                        result["artifact_directory"] = str(directory.relative_to(work))
                        write_json(directory / "result.json", result)
                        results.append(result)
                        if result["status"] != "completed":
                            failures.append(dict(stage="execution", job_id=job_id, architecture=architecture,
                                                 status=result["status"], error=result["error"]))
                        elif any(e["censored"] for e in result["recovery"]["episodes"]):
                            failures.append(dict(stage="recovery", job_id=job_id, error="Recovery threshold not demonstrated within observation window",
                                                 episodes=result["recovery"]["episodes"]))
        if not config["sessions"]:
            failures.append(dict(stage="coverage", error="No held-out sessions configured; no navigation results measured"))
        aggregates = aggregate(results)
        statistics = paired_statistics(results, config)
        limitations = [
            "No metric establishes improvement without eligible data and paired comparisons; null is not zero.",
            "Sources, releases, reference kinds, outage durations and environments remain separate aggregation cohorts.",
            "ATE is unaligned translational RMSE in the common ENU frame; it equals position RMSE by definition.",
            "Receiver-proxy agreement is not independent positioning accuracy. Natural outages require independent references.",
            "Confidence currently reports uncertainty only; removing it may leave the trajectory unchanged.",
            "Nearest-road map control remains active when temporal road hypotheses are removed; removing map constraints retains hypothesis telemetry.",
            "Outage prediction is scored before onset with full future coverage; artificial masking cannot demonstrate prediction of natural degradation.",
            "Recovery intervals without sustained reference-supported convergence are censored, not assigned zero recovery time.",
            "Latency/RSS are host measurements, not deployment-device measurements; RSS includes imports and scoring storage.",
            "Production comparison invokes the existing Python replay G adapter; it is not a browser-runtime parity claim.",
            "Generalization requires declared complete development inventories and stable route/vehicle/device identifiers; missing declarations stay unknown.",
            "Bootstrap clusters are release split groups, not overlapping windows. Small group counts limit statistical power.",
            "Material improvement is predeclared for position RMSE only; multiplicity-adjusted evidence is not a universal proof.",
            "No production checkpoint, export or model registry is replaced by this framework.",
        ]
        if config["production_model"] != config["candidate_model"]:
            limitations.append("Candidate and production model specifications differ: production comparison includes a model change; component ablations use the candidate model consistently.")
        if not isolated:
            limitations.append("In-process test execution: memory comparisons are disabled; results are software fixtures only.")
        counts = Counter(r["status"] for r in results)
        coverage = dict(configured_sessions=len(config["sessions"]), validated_sessions=len(sessions), jobs=len(results),
                        status_counts=dict(counts), requested_environments=list(ENVIRONMENTS),
                        environments_with_scores=sorted({r["cohort"]["environment"] for r in aggregates if r["cohort"]["environment"] != "all" and r["metrics"].get("position_rmse_m", {}).get("sessions",0)}),
                        production_matched_comparisons=sum(r["comparator"] == "production" for r in statistics),
                        requested_architectures=config["architectures"], requested_outage_durations_s=config["outage_durations_s"])
        write_json(work / "raw_results.json", results)
        write_json(work / "aggregated_results.json", aggregates)
        write_json(work / "statistical_summary.json", statistics)
        write_json(work / "per_session_results.json", [dict(session=s, results=[r for r in results if r["input_identity"] == s["identity"]]) for s in sessions])
        write_json(work / "per_environment_results.json", [r for r in aggregates if r["cohort"]["environment"] != "all"])
        write_json(work / "ablation_results.json", [r for r in statistics if r["comparator"].startswith("F_without_")])
        write_json(work / "failure_cases.json", failures)
        write_json(work / "limitations.json", limitations)
        write_json(work / "coverage.json", coverage)
        write_json(work / "generalization_results.json", [dict(identity=s["identity"], generalization=s["generalization"]) for s in sessions])
        lines = ["# Navigators research benchmark", "", f"Validated sessions: {len(sessions)}. Scheduled runs: {len(results)}.",
                 f"Run status counts: `{dict(counts)}`.", "", "## Model/input audit", "", "```json", json.dumps(audits, indent=2), "```", "",
                 "## Paired position-RMSE comparisons", "", "Source/release, environment, phase and duration must match. Negative delta favors F.", "",
                 "| Comparator | Source | Environment | Phase | Duration (s) | Groups | F minus comparator (m) | Evidence |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in statistics:
            if row["metric"] == "position_rmse_m":
                lines.append(f"| {row['comparator']} | {row['source_dataset']} | {row['environment']} | {row['phase']} | {row['duration_s']} | {row['independent_groups']} | {row['mean_delta']:.6g} | {row['evidence']} |")
        if not statistics:
            lines += ["", "No compatible paired comparisons available. No component improvement is established."]
        lines += ["", "## Limitations", ""] + ["- " + note for note in limitations]
        lines += ["", "## Reproduce", "", "Use the archived benchmark_config.json with the recorded Python/packages and unchanged input/model/code hashes.",
                  "Run `python scripts/research_benchmark.py --config <report>/benchmark_config.json --output <new-report-directory>`.",
                  "All raw trajectories, sample identities, metric supports and failures are retained. Inspect coverage before interpreting averages.",
                  "Verify `artifact_manifest.json` with `--verify <report-directory>`. Hashes detect changes, not authorship."]
        (work / "report.md").write_text("\n".join(lines) + "\n")
        artifact_manifest = {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob("*")) if p.is_file()}
        write_json(work / "artifact_manifest.json", dict(artifacts=artifact_manifest, sha256=digest(artifact_manifest)))
        work.rename(output)
    return coverage


def verify_report(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "artifact_manifest.json").read_text())
    if manifest["sha256"] != digest(manifest["artifacts"]):
        raise ValueError("Report manifest hash mismatch")
    for name, expected in manifest["artifacts"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file() or sha256(file) != expected:
            raise ValueError(f"Report artifact integrity failure: {name}")
    return manifest["sha256"]
