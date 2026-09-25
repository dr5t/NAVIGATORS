"""Release-pinned position evaluation against explicitly attributed references.

Independent reference measurements are required to score natural GNSS outages.
Receiver proxies are separately labeled and restricted to normal, fresh fixes.
"""
from collections import defaultdict
import json
import math
from pathlib import Path

from .pipeline import SPLITS, _hash, verify_release
from .schema import CATEGORIES, distance_m, number


def _metrics(errors):
    if not errors:
        return dict(samples=0, mean_position_error_m=None, position_rmse_m=None,
                    p95_position_error_m=None, max_position_error_m=None)
    values = sorted(errors)
    return dict(samples=len(values), mean_position_error_m=sum(values) / len(values),
                position_rmse_m=math.sqrt(sum(e * e for e in values) / len(values)),
                p95_position_error_m=values[max(0, math.ceil(.95 * len(values)) - 1)],
                max_position_error_m=values[-1])


def evaluate_release(release, predictions_path, reference_path, split="test"):
    """Evaluate exact sample identities; no nearest-time pairing/interpolation.

    Both files contain metadata and records with session_id, sample_index,
    timestamp_utc_s, latitude and longitude. Prediction metadata identifies the
    model and declares training_session_ids for an overlap check.
    """
    if split not in SPLITS:
        raise ValueError("Unknown evaluation split")
    manifest = verify_release(release)
    prediction_bytes = Path(predictions_path).read_bytes()
    reference_bytes = Path(reference_path).read_bytes()
    predictions, reference = json.loads(prediction_bytes), json.loads(reference_bytes)
    model_meta, ref_meta = predictions["metadata"], reference["metadata"]
    training_ids = model_meta.get("training_session_ids")
    if not isinstance(model_meta.get("model_id"), str) or not model_meta["model_id"].strip() or not isinstance(training_ids, list) or any(not isinstance(s, str) for s in training_ids):
        raise ValueError("Predictions require model_id and declared training_session_ids")
    if ref_meta.get("reference_kind") not in ("independent", "receiver_proxy"):
        raise ValueError("Reference kind must be independent or receiver_proxy")
    if any(not isinstance(ref_meta.get(k), str) or not ref_meta[k].strip() for k in ("source_dataset", "provenance")):
        raise ValueError("Reference requires source_dataset and provenance")
    sessions = {s["session_id"]: s for s in manifest["sessions"] if s["split"] == split}
    if split != "train" and set(training_ids) & set(sessions):
        raise ValueError("Evaluation leakage: model training includes held-out sessions")
    rows = {key: [json.loads(line) for line in (Path(release) / session["normalized_path"]).read_text().splitlines()]
            for key, session in sessions.items()}

    def keyed(payload):
        result = {}
        if not isinstance(payload.get("records"), list):
            raise ValueError("Evaluation records must be a list")
        for row in payload["records"]:
            if not isinstance(row, dict):
                raise ValueError("Evaluation row must be an object")
            session, index = row.get("session_id"), row.get("sample_index")
            if not isinstance(session, str) or session not in sessions or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(rows[session]):
                raise ValueError("Evaluation sample is outside the selected release/split")
            key = (session, index)
            if key in result:
                raise ValueError("Duplicate evaluation sample")
            if any(not number(row.get(k)) for k in ("timestamp_utc_s", "latitude", "longitude")):
                raise ValueError("Evaluation coordinates/time must be finite")
            if abs(row["latitude"]) > 90 or abs(row["longitude"]) > 180:
                raise ValueError("Invalid evaluation coordinates")
            if abs(row["timestamp_utc_s"] - rows[session][index]["timestamp_utc_s"]) > 1e-6:
                raise ValueError("Evaluation timestamp does not match sample identity")
            result[key] = row
        return result

    estimates, references = keyed(predictions), keyed(reference)
    errors, by_category, by_session = [], defaultdict(list), defaultdict(list)
    excluded_proxy = 0
    for key in sorted(estimates.keys() & references.keys()):
        row = rows[key[0]][key[1]]
        if ref_meta["reference_kind"] == "receiver_proxy":
            age = row["timestamp"] - row.get("gnss_timestamp", -math.inf)
            if row["gnss_status"] != "normal_gnss" or row["GNSS_accuracy"] is None or not 0 <= age <= manifest["policy"]["max_fix_age_s"]:
                excluded_proxy += 1
                continue
        error = distance_m([estimates[key][k] for k in ("latitude", "longitude")],
                           [references[key][k] for k in ("latitude", "longitude")])
        errors.append(error)
        by_session[key[0]].append(error)
        for category in row["categories"]:
            by_category[category].append(error)
    total = sum(len(r) for r in rows.values())
    return dict(dataset_version=manifest["dataset_version"], manifest_sha256=manifest["manifest_sha256"],
                source_dataset=manifest["source_dataset"], source_kind=manifest["source_kind"], split=split,
                prediction_sha256=_hash(prediction_bytes), reference_sha256=_hash(reference_bytes),
                model_metadata=model_meta, reference_metadata=ref_meta,
                metric_basis="independent reference" if ref_meta["reference_kind"] == "independent" else "receiver proxy agreement (not ground-truth accuracy)",
                coverage=dict(eligible_split_samples=total, predicted_samples=len(estimates), reference_samples=len(references),
                              matched_samples=len(estimates.keys() & references.keys()), scored_samples=len(errors),
                              excluded_proxy_samples=excluded_proxy, scored_fraction=len(errors) / total if total else None),
                metrics=_metrics(errors), by_category={c: _metrics(by_category[c]) for c in CATEGORIES},
                by_session={s: _metrics(by_session[s]) for s in sessions},
                limitations=["Training membership and reference independence are declared by the submitter.",
                             "Scores cover only paired samples; missing references/predictions are reported in coverage."])
