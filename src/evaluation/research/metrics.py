"""Common scoring for every system. No trajectory alignment or invented labels."""
import math
import numpy as np

DEFINITIONS = {
    "position_rmse_m": "sqrt(mean(||estimated_EN-reference_EN||^2)); common fixed ENU frame",
    "absolute_trajectory_error_m": "Unaligned translational ATE RMSE; identical to position RMSE by definition",
    "final_position_error_m": "Position error at the exact final selected sample; null if its reference is absent",
    "cross_track_rmse_m": "RMSE of signed position residual normal to reference heading/moving velocity tangent",
    "along_track_rmse_m": "RMSE of signed position residual along reference heading/moving velocity tangent",
    "velocity_mae_mps": "Mean Euclidean absolute EN velocity error",
    "velocity_rmse_mps": "sqrt(mean(sum(EN velocity component errors squared)))",
    "heading_mae_deg": "Mean absolute wrapped heading error, excluding known stationary reference velocity",
    "heading_rmse_deg": "Root mean square wrapped heading error, excluding known stationary reference velocity",
    "anomaly_precision": "TP/(TP+FP) at labeled receiver epochs; undefined denominator -> null",
    "anomaly_recall": "TP/(TP+FN) at labeled receiver epochs; undefined denominator -> null",
    "outage_prediction_precision": "Epoch precision for outage onset in (t,t+horizon], only while GNSS available with full future coverage",
    "outage_prediction_recall": "Epoch recall for the same future-onset labels; current outage is excluded",
    "road_hypothesis_accuracy": "Exact segment-ID accuracy over all labeled epochs; no-match counts as incorrect",
    "map_matching_accuracy": "Exact selected segment-ID accuracy over all labeled epochs; no-match counts as incorrect",
    "inference_latency_ms": "Mean actual model predict call latency, including first call; host CPU",
    "inference_latency_p95_ms": "95th percentile of actual model predict call latency",
    "navigation_latency_ms": "Mean causal navigation step latency, excluding scoring and output I/O",
    "recovery_time_s": "Worst time from first fresh returning fix to a sustained error-threshold interval; null if any episode censored",
    "max_recovery_position_jump_m": "Maximum norm of actual GNSS measurement correction, excluding predicted vehicle movement and map corrections",
    "memory_usage_bytes": "Isolated worker process peak RSS including imports/model; not tensor-only or device RAM",
    "model_size_bytes": "Actual bytes of loaded model artifact plus external tensor files; null for no AI model",
}


from typing import Dict, Optional

def score(rows, capabilities, prediction_threshold=.65):
    metrics: Dict[str, Optional[float]] = {key: None for key in DEFINITIONS if key not in ("recovery_time_s", "max_recovery_position_jump_m", "memory_usage_bytes", "model_size_bytes")}
    support = {key: {"sample_indices": [], "reason": "required reference or component output unavailable"} for key in metrics}

    def save(key, values, indices, rms=False):
        support[key] = {"sample_indices": indices, "samples": len(indices), "reason": None if values else "no eligible reference/output pairs"}
        if values:
            metrics[key] = float(np.sqrt(np.mean(np.square(values))) if rms else np.mean(values))

    position, pidx, velocity, vidx, angle, hidx, cross, along, tidx = [], [], [], [], [], [], [], [], []
    for row in rows:
        ref, est, i = row["reference"], row["estimate"], row["sample_index"]
        eligible = row["reference_eligible"]
        if not eligible:
            continue
        if ref["position"] is not None:
            residual = np.array(est[:2]) - ref["position"]
            position.append(float(np.linalg.norm(residual)))
            pidx.append(i)
            tangent = None
            if ref["velocity"] is not None and np.linalg.norm(ref["velocity"]) >= .5:
                tangent = np.array(ref["velocity"]) / np.linalg.norm(ref["velocity"])
            elif ref["heading"] is not None and ref["velocity"] is None:
                tangent = np.array([np.sin(ref["heading"]), np.cos(ref["heading"])])
            if tangent is not None:
                along.append(float(residual @ tangent))
                cross.append(float(residual @ np.array([tangent[1], -tangent[0]])))
                tidx.append(i)
        if ref["velocity"] is not None:
            velocity.append(float(np.linalg.norm(np.array(est[2:4]) - ref["velocity"])))
            vidx.append(i)
        if ref["heading"] is not None and (ref["velocity"] is None or np.linalg.norm(ref["velocity"]) >= .5):
            angle.append(float(np.degrees(abs((est[4] - ref["heading"] + np.pi) % (2*np.pi) - np.pi))))
            hidx.append(i)
    for key in ("position_rmse_m", "absolute_trajectory_error_m"):
        save(key, position, pidx, True)
    if rows and pidx and pidx[-1] == rows[-1]["sample_index"]:
        save("final_position_error_m", [position[-1]], [pidx[-1]])
    save("velocity_mae_mps", velocity, vidx)
    save("velocity_rmse_mps", velocity, vidx, True)
    save("heading_mae_deg", angle, hidx)
    save("heading_rmse_deg", angle, hidx, True)
    save("cross_track_rmse_m", cross, tidx, True)
    save("along_track_rmse_m", along, tidx, True)
    for name, truth, prediction in (("anomaly", "anomaly", "anomaly_detected"),
                                     ("outage_prediction", "future_outage", "outage_probability")):
        pairs = []
        for row in rows:
            label = row["reference"][truth] if name == "anomaly" else row[truth]
            pred = row[prediction]
            if label is not None and pred is not None and row["receiver_epoch"]:
                pairs.append((row["sample_index"], bool(label), bool(pred) if name == "anomaly" else pred >= prediction_threshold))
        tp = sum(t and p for _, t, p in pairs)
        fp = sum(not t and p for _, t, p in pairs)
        fn = sum(t and not p for _, t, p in pairs)
        tn = sum(not t and not p for _, t, p in pairs)
        for suffix, denominator in (("precision", tp+fp), ("recall", tp+fn)):
            key = name + "_" + suffix
            metrics[key] = tp/denominator if denominator else None
            support[key] = dict(sample_indices=[i for i, _, _ in pairs], samples=len(pairs), tp=tp, fp=fp, fn=fn, tn=tn,
                                reason=None if denominator else "undefined denominator or missing labeled predictions")
    for key, pred, enabled in (("road_hypothesis_accuracy", "road_hypothesis_id", capabilities.road_hypotheses),
                               ("map_matching_accuracy", "map_match_id", capabilities.map_constraints)):
        labeled = [r for r in rows if r["reference"]["road_segment_id"] is not None] if enabled else []
        save(key, [int(r[pred] == r["reference"]["road_segment_id"]) for r in labeled], [r["sample_index"] for r in labeled])
    for key, field in (("inference_latency_ms", "inference_latency_ms"), ("navigation_latency_ms", "loop_latency_ms")):
        valid = [r for r in rows if r[field] is not None]
        save(key, [r[field] for r in valid], [r["sample_index"] for r in valid])
    valid = [r for r in rows if r["inference_latency_ms"] is not None]
    metrics["inference_latency_p95_ms"] = float(np.percentile([r["inference_latency_ms"] for r in valid], 95)) if valid else None
    support["inference_latency_p95_ms"] = support["inference_latency_ms"].copy()
    return dict(metrics=metrics, support=support, selected_samples=len(rows),
                reference_position_coverage=len(pidx)/len(rows) if rows else None)


def recovery_metrics(rows, config):
    episodes = []
    denied = False
    for i, row in enumerate(rows):
        if not row["gnss_allowed"]:
            denied = True
        if not (denied and row["receiver_epoch"]):
            continue
        denied = False
        start = row["time_s"]
        window = []
        for sample in rows[i:]:
            if sample["time_s"] > start + config["recovery_window_s"] or not sample["gnss_allowed"]:
                break
            window.append(sample)
        streak, recovery = None, None
        threshold_indices = []
        for sample in window:
            ref = sample["reference"]["position"]
            good = sample["reference_eligible"] and ref is not None and np.linalg.norm(np.array(sample["estimate"][:2])-ref) <= config["recovery_error_threshold_m"]
            if good:
                if streak is None:
                    streak = sample["time_s"]
                threshold_indices.append(sample["sample_index"])
                if sample["time_s"] - streak >= config["recovery_hold_s"] - 1e-9:
                    recovery = streak-start
                    break
            else:
                streak = None
                threshold_indices = []
        episodes.append(dict(start_s=start, recovery_time_s=recovery, censored=recovery is None,
                             observed_until_s=window[-1]["time_s"] if window else start,
                             threshold_sample_indices=threshold_indices,
                             jump_sample_indices=[r["sample_index"] for r in window],
                             max_position_jump_m=max((r["gnss_correction_m"] for r in window), default=None)))
    completed = [e["recovery_time_s"] for e in episodes if not e["censored"]]
    return dict(recovery_time_s=max(completed) if episodes and len(completed) == len(episodes) else None,
                max_recovery_position_jump_m=max((e["max_position_jump_m"] for e in episodes), default=None),
                episodes=episodes, reason=None if episodes and len(completed) == len(episodes) else "no return episode, missing reference, or recovery censored")
