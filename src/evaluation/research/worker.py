"""One isolated process per system/session/outage/repetition for honest RSS scope."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import traceback
import numpy as np

from data.india_dataset.schema import canonical, digest
from evaluation.replay import VelocityModel, machine_info
from .config import ARCHITECTURES, ENVIRONMENTS
from .inputs import read_session, prepare_case, reference_at
from .metrics import score, recovery_metrics
from .navigation import ResearchNavigation, ProductionNavigation


def load_model(spec):
    if not spec:
        raise ValueError("A real model and preprocessing contract are required; no AI fallback")
    if set(spec) - {"path", "stats", "backend", "training_inventory"}:
        raise ValueError("Unknown model configuration; legacy/incompatible preprocessing is not permitted")
    return VelocityModel(spec["path"], spec["stats"], spec.get("backend", "onnx"))


def run_job(job, output, model_override=None, isolated=True):
    config, case, system = job["config"], job["case"], job["architecture"]
    architecture = ARCHITECTURES[system]
    np.random.seed(config["seed"] + job["repeat"])
    random.seed(config["seed"] + job["repeat"])
    session = read_session(job["session"])
    if session.identity != job["input_identity"]:
        raise ValueError("Input changed after scheduling")
    from evaluation.recording import sha256
    for path, expected in job.get("model_files", {}).items():
        if sha256(path) != expected:
            raise ValueError(f"Model/development artifact changed after scheduling: {path}")
    spec = config["production_model"] if architecture.production else config["candidate_model"]
    model = (model_override or load_model(spec)) if architecture.ai_velocity else None
    model_metadata = model.metadata() if model else None
    first, rotation, origin, mask, allowed = prepare_case(session, case, config,
                    model.window_size if model else 0, model.sample_rate if model else None)
    if (architecture.map_constraints or architecture.road_hypotheses) and not session.map_path:
        raise ValueError("This architecture requires a supplied map; no generated map fallback")
    initial = session.recording.gnss[first]
    nav = (ProductionNavigation(rotation, initial, origin, model, session.map_path) if architecture.production else
           ResearchNavigation(architecture, rotation, initial, origin, model, session.map_path))
    rows, rec = [], session.recording
    onsets = rec.timestamps[np.flatnonzero(np.r_[False, allowed[:-1] & ~allowed[1:]])]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trajectory.jsonl").open("wb") as file:
        for i in range(first, len(rec.timestamps)):
            t = float(rec.timestamps[i])
            if case["kind"] == "masked_outage" and t > case["start_s"] + case["duration_s"] + config["recovery_window_s"]:
                break
            dt = 0. if i == first else float(t - rec.timestamps[i-1])
            fix = rec.gnss[i].copy() if allowed[i] else None
            fresh = bool(allowed[i] and (rec.fresh[i] or i == first))
            receiver = session.rows[i].get("gnss_timestamp")
            receiver = receiver-session.rows[0]["timestamp"] if receiver is not None else None
            result = nav.step(rec.accel[i], rec.gyro[i], dt, t, fix, fresh, receiver)
            reference = reference_at(session, i, origin)
            proxy = session.reference_metadata["reference_kind"] == "receiver_proxy"
            ref_ok = not proxy or (session.rows[i]["gnss_status"] == "normal_gnss" and rec.valid[i] and rec.fresh[i]
                                  and session.rows[i].get("gnss_timestamp") is not None)
            future = None
            if fresh and t + config["prediction_horizon_s"] <= rec.timestamps[-1]:
                future = bool(np.any((onsets > t) & (onsets <= t + config["prediction_horizon_s"])))
            environments = [e for e in session.rows[i]["categories"] if e in ENVIRONMENTS]
            if not allowed[i]:
                environments.append("gnss_fully_unavailable")
            elif session.rows[i]["gnss_status"] == "degraded_gnss":
                environments.append("gnss_degraded")
            phase = "observed"
            if case["kind"] == "masked_outage":
                phase = "pre_outage" if t < case["start_s"] else "outage" if mask[i] else "recovery"
            elif not allowed[i]:
                phase = "outage"
            row = dict(sample_index=i, time_s=t, timestamp_utc_s=session.rows[i]["timestamp_utc_s"],
                       gnss_allowed=bool(allowed[i]), receiver_epoch=fresh, phase=phase,
                       environments=sorted(set(environments)), future_outage=future,
                       reference=reference, reference_eligible=bool(ref_ok), **result)
            file.write(canonical(row) + b"\n")
            rows.append(row)
    regions = []
    for phase in ("whole", "outage", "recovery"):
        selected = rows if phase == "whole" else [r for r in rows if r["phase"] == phase]
        for environment in ("all",) + ENVIRONMENTS:
            subset = selected if environment == "all" else [r for r in selected if environment in r["environments"]]
            scored = score(subset, architecture, config["prediction_threshold"])
            regions.append(dict(phase=phase, environment=environment, **scored))
    recovery = recovery_metrics(rows, config)
    machine = machine_info()
    machine["ram_scope"] = ("Isolated worker peak RSS: imports, model, navigation, raw rows and scoring" if isolated else
                            "In-process test run; peak RSS includes prior work and cannot compare systems")
    whole = next(r for r in regions if r["phase"] == "whole" and r["environment"] == "all")
    for key in ("recovery_time_s", "max_recovery_position_jump_m"):
        whole["metrics"][key] = recovery[key]
        whole["support"][key] = dict(sample_indices=sorted({i for episode in recovery["episodes"] for i in episode["jump_sample_indices"]}),
                                      reason=recovery["reason"], episodes=len(recovery["episodes"]))
    for key, value in (("memory_usage_bytes", machine["process_peak_rss_bytes"] if isolated else None),
                       ("model_size_bytes", model_metadata.get("model_size_bytes") if model_metadata else None)):
        whole["metrics"][key] = value
        whole["support"][key] = dict(sample_indices=[], reason=None if value is not None else "not measured/not applicable",
                                      provenance="worker machine/model metadata")
    errors = [(float(np.linalg.norm(np.array(r["estimate"][:2])-r["reference"]["position"])), r)
              for r in rows if r["reference_eligible"] and r["reference"]["position"] is not None]
    worst = [dict(sample_index=r["sample_index"], time_s=r["time_s"], position_error_m=error)
             for error, r in sorted(errors, key=lambda e: e[0], reverse=True)[:5]]
    return dict(status="completed", job_id=job["job_id"], architecture=system, configuration=asdict(architecture),
                case=case, repeat=job["repeat"], input_identity=session.identity,
                generalization=job.get("generalization", {}), reference_metadata=session.reference_metadata,
                origin=origin.tolist(), calibration_rotation=rotation.tolist(), regions=regions,
                recovery=recovery, resource_metrics=dict(memory_usage_bytes=machine["process_peak_rss_bytes"],
                    model_size_bytes=model_metadata.get("model_size_bytes") if model_metadata else None),
                machine=machine, model=model_metadata, component_counts=rows[-1]["component_counts"] if rows else {},
                worst_position_samples=worst, raw_trajectory="trajectory.jsonl",
                raw_trajectory_sha256=sha256(output / "trajectory.jsonl"))


def main():
    job_path, output = map(Path, sys.argv[1:3])
    job = json.loads(job_path.read_text())
    try:
        result = run_job(job, output)
    except Exception as error:
        result = dict(status="failed", job_id=job["job_id"], architecture=job["architecture"],
                      case=job["case"], repeat=job["repeat"], input_identity=job["input_identity"],
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_bytes(canonical(result) + b"\n")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
