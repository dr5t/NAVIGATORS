"""Immutable, source-separated releases and leakage-safe window manifests."""
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

from .schema import (CATEGORIES, CONTEXT, DATASET_NAME, GNSS, IMU, NORMALIZED_UNITS,
                     PIPELINE_VERSION, SCHEMA_VERSION, Policy, canonical, digest, validate_session)

SPLITS = ("train", "validation", "test")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def read_input(path):
    """Decode raw input once; return exact bytes used to validate and archive."""
    path = Path(path)
    raw = path.read_bytes()
    blobs = {f"raw/{_hash(raw)}{path.suffix.lower()}": raw}
    metadata, records = {}, []
    try:
        if path.suffix.lower() == ".json":
            payload = _strict_json(raw)
            if not isinstance(payload, dict):
                raise ValueError("JSON must contain metadata and records")
            metadata, records = payload.get("metadata"), payload.get("records")
        elif path.suffix.lower() in (".jsonl", ".csv"):
            sidecar = Path(str(path) + ".metadata.json").read_bytes()
            blobs[f"raw/{_hash(sidecar)}.metadata.json"] = sidecar
            metadata = _strict_json(sidecar)
            if path.suffix.lower() == ".jsonl":
                records = [_strict_json(line) for line in raw.decode("utf-8").splitlines()]
            else:
                reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
                if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError("CSV has empty or duplicate headers")
                for row in reader:
                    if None in row:
                        raise ValueError("CSV row has extra cells")
                    for key, value in row.items():
                        if value in (None, ""):
                            row[key] = None
                        elif key in ("timestamp", "gnss_timestamp") + IMU + GNSS:
                            row[key] = float(value)
                        elif key == "categories":
                            row[key] = _strict_json(value)
                    records.append(row)
        else:
            raise ValueError("Supported inputs: .json, .jsonl, .csv")
        return metadata, records, blobs, None
    except (ValueError, UnicodeError, OSError) as error:
        return metadata, [], blobs, str(error)


def _error(entry, code, message):
    entry["quality"]["accepted"] = False
    entry["quality"]["issues"].append(dict(code=code, message=message, row=None, severity="error"))


def _duplicates(entries):
    """Reject all conflicting copies, independent of input order or filenames."""
    identities, content, samples = {}, {}, {}
    for entry in entries:
        if not entry["quality"]["accepted"]:
            continue
        metadata, rows = entry["metadata"], entry["rows"]
        identities.setdefault(metadata["session_id"], []).append(entry)
        signature = digest([[r["elapsed_s"]] + [r[k] for k in IMU + GNSS] for r in rows])
        content.setdefault(signature, []).append(entry)
        # Three consecutive physical samples expose overlapping re-exported chunks.
        hashes = [digest([r["timestamp_utc_s"]] + [r[k] for k in IMU + GNSS]) for r in rows]
        for i in range(len(hashes) - 2):
            key = digest(hashes[i:i + 3])
            samples.setdefault(key, set()).add(entry["input_id"])
    for groups, code in ((identities, "duplicate_session"), (content, "duplicate_content")):
        for group in groups.values():
            if len(group) > 1:
                for entry in group:
                    _error(entry, code, "Conflicting inputs: " + ", ".join(e["input_id"] for e in group))
    conflicts = set().union(*(group for group in samples.values() if len(group) > 1)) if samples else set()
    for entry in entries:
        if entry["input_id"] in conflicts:
            _error(entry, "duplicate_segment", "Consecutive sensor samples overlap another input")


def _assign_splits(entries, strategy, fractions, seed, policy):
    """Connected components preserve transitive vehicle/device/recording links."""
    parents = list(range(len(entries)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[root(i)] = root(j)

    keys = ["recording_id"]
    if strategy in ("vehicle", "joint"):
        keys.append("vehicle_id")
    if strategy in ("device", "joint"):
        keys.append("device_id")
    known = {}
    for i, entry in enumerate(entries):
        for key in keys:
            value = entry["metadata"].get(key)
            if value is not None:
                token = (key, value)
                if token in known:
                    union(i, known[token])
                known[token] = i
    # Even session-only splits keep simultaneous/adjacent traces from the same
    # known device or vehicle together. Missing IDs are reported, never invented.
    for i, a in enumerate(entries):
        for j in range(i):
            b = entries[j]
            shared = any(a["metadata"].get(k) is not None and a["metadata"][k] == b["metadata"].get(k)
                         for k in ("device_id", "vehicle_id"))
            if shared and (max(a["rows"][0]["timestamp_utc_s"], b["rows"][0]["timestamp_utc_s"])
                           <= min(a["rows"][-1]["timestamp_utc_s"], b["rows"][-1]["timestamp_utc_s"]) + policy.temporal_guard_s):
                union(i, j)
    groups = {}
    for i, entry in enumerate(entries):
        groups.setdefault(root(i), []).append(entry)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), digest([seed, sorted(e["metadata"]["session_id"] for e in g)])))
    counts = Counter()
    for group in ordered:
        split = max((s for s in SPLITS if fractions[s] > 0),
                    key=lambda s: fractions[s] * len(entries) - counts[s])
        group_id = digest(sorted(e["metadata"]["session_id"] for e in group))
        for entry in group:
            entry["split"], entry["split_group"] = split, group_id
        counts[split] += len(group)
    return len(groups)


def window_index(sessions, window_size=200, stride=50):
    """Generate indices only after session assignment; stop is exclusive."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (window_size, stride)):
        raise ValueError("Window size and stride must be positive integers")
    for session in sessions:
        if session["split"] not in SPLITS:
            raise ValueError("Windows require a valid session split")
        for start in range(0, session["samples"] - window_size + 1, stride):
            yield {"session_id": session["session_id"], "source_dataset": session["source_dataset"],
                   "split": session["split"], "start": start, "stop": start + window_size}


def build_release(inputs, output_root, version, *, source_kind="navigators_collection",
                  source_dataset=DATASET_NAME, strategy="joint", fractions=None, seed=42,
                  policy=None, window_size=200, stride=50):
    """Validate, archive and publish a new release. Existing versions never change.

    Rejected inputs are archived and reported but have no normalized rows/split.
    Each release contains exactly one declared source; no external imports are
    converted into original Navigators observations.
    """
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("Version must be a safe nonempty identifier")
    if source_kind not in ("navigators_collection", "external_benchmark", "synthetic"):
        raise ValueError("Unknown source kind")
    if not isinstance(source_dataset, str) or not source_dataset.strip():
        raise ValueError("Explicit source_dataset required")
    if (source_kind == "navigators_collection") != (source_dataset == DATASET_NAME):
        raise ValueError("Original collections and external/synthetic sources must be separate")
    if strategy not in ("session", "vehicle", "device", "joint"):
        raise ValueError("Unknown split strategy")
    fractions = fractions or dict(train=.7, validation=.15, test=.15)
    from .schema import number
    if set(fractions) != set(SPLITS) or any(not number(x) or x < 0 for x in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("Split fractions must be finite, nonnegative and sum to one")
    list(window_index([], window_size, stride))
    policy = policy or Policy()
    slug = re.sub(r"[^a-z0-9]+", "-", source_dataset.lower()).strip("-")[:60] or "source"
    parent = Path(output_root) / source_kind / (slug + "-" + digest(source_dataset)[:10])
    destination = parent / version
    if destination.exists():
        raise FileExistsError(f"Immutable release already exists: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    paths = sorted([Path(p) for p in inputs], key=lambda p: str(p.resolve()))
    if len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError("Input path listed more than once")
    with tempfile.TemporaryDirectory(prefix=".building-", dir=parent) as staging:
        work = Path(staging) / "release"
        work.mkdir()
        entries = []
        for index, path in enumerate(paths):
            metadata, records, blobs, parse_error = read_input(path)
            quality, rows = ({"accepted": False, "statistics": {}, "issues": [dict(code="parse_error", message=parse_error, severity="error", row=None)]}, []) if parse_error else validate_session(metadata, records, policy)
            entry = dict(input_id=f"input-{index:06d}", input_name=path.name, metadata=metadata,
                         quality=quality, rows=rows, raw_artifacts=sorted(blobs))
            if isinstance(metadata, dict) and (metadata.get("source_kind") != source_kind or metadata.get("source_dataset") != source_dataset):
                _error(entry, "source_mismatch", "Input source differs from this release; import it into a separate source release")
            for name, blob in blobs.items():
                artifact = work / name
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_bytes(blob)
            entries.append(entry)
        _duplicates(entries)
        accepted = [e for e in entries if e["quality"]["accepted"]]
        accepted_raw = {name for entry in accepted for name in entry["raw_artifacts"]}
        rejected_raw = set()
        for entry in entries:
            if entry["quality"]["accepted"]:
                continue
            archived = []
            for name in entry["raw_artifacts"]:
                quarantined = "quarantine/" + name
                artifact = work / quarantined
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes((work / name).read_bytes())
                archived.append(quarantined)
                rejected_raw.add(name)
            entry["raw_artifacts"] = archived
        for name in rejected_raw - accepted_raw:
            (work / name).unlink()  # Only redundant files in our private staging directory.
        group_count = _assign_splits(accepted, strategy, fractions, seed, policy)
        sessions, categories, split_counts = [], Counter(), Counter()
        for entry in accepted:
            meta, rows = entry["metadata"], entry["rows"]
            name = "normalized/" + digest(meta["session_id"]) + ".jsonl"
            file = work / name
            file.parent.mkdir(exist_ok=True)
            with file.open("wb") as stream:
                for row in rows:
                    stream.write(canonical({**row, "source_dataset": source_dataset, "source_kind": source_kind}) + b"\n")
            sessions.append(dict(session_id=meta["session_id"], source_dataset=source_dataset,
                                 source_kind=source_kind, metadata=meta, normalized_path=name,
                                 raw_artifacts=entry["raw_artifacts"], samples=len(rows),
                                 split=entry["split"], split_group=entry["split_group"],
                                 statistics=entry["quality"]["statistics"]))
            categories.update(entry["quality"]["statistics"]["category_samples"])
            split_counts[entry["split"]] += 1
        windows = work / "windows.jsonl"
        window_counts = Counter()
        with windows.open("wb") as file:
            for window in window_index(sessions, window_size, stride):
                file.write(canonical(window) + b"\n")
                window_counts[window["split"]] += 1
        notes = ["Annotations and provenance are contributor declarations; validation does not establish ownership or physical axis calibration."]
        for key in ("vehicle_id", "device_id"):
            if any(e["metadata"].get(key) is None for e in accepted):
                notes.append(f"Some {key} values unavailable: complete {key} isolation cannot be verified.")
        if any(fractions[s] > 0 and not split_counts[s] for s in SPLITS):
            notes.append("Requested splits are empty: collect more independent groups; no sessions were duplicated to fill them.")
        stats = dict(inputs=len(entries), accepted_sessions=len(sessions), rejected_sessions=len(entries) - len(sessions),
                     samples=sum(s["samples"] for s in sessions),
                     duration_s=sum(s["statistics"]["duration_s"] for s in sessions),
                     independent_groups=group_count, sessions_by_split={s: split_counts[s] for s in SPLITS},
                     windows_by_split={s: window_counts[s] for s in SPLITS},
                     samples_by_split={split: sum(s["samples"] for s in sessions if s["split"] == split) for split in SPLITS},
                     duration_s_by_split={split: sum(s["statistics"]["duration_s"] for s in sessions if s["split"] == split) for split in SPLITS},
                     category_samples={c: categories[c] for c in CATEGORIES},
                     category_sessions={c: sum(s["statistics"]["category_samples"].get(c, 0) > 0 for s in sessions) for c in CATEGORIES},
                     issues_by_code=dict(Counter(i["code"] for e in entries for i in e["quality"]["issues"])),
                     missing_categories=[c for c in CATEGORIES if not categories[c]],
                     unique_devices=len({e["metadata"]["device_id"] for e in accepted if e["metadata"].get("device_id")}),
                     unique_vehicles=len({e["metadata"]["vehicle_id"] for e in accepted if e["metadata"].get("vehicle_id")}),
                     limitations=notes)
        _write(work / "statistics.json", stats)
        # Invalid metadata may itself contain NaN; do not copy it into valid JSON reports.
        reports = [{k: e[k] for k in ("input_id", "input_name", "quality", "raw_artifacts")} for e in entries]
        _write(work / "quality_report.json", reports)
        manifest = dict(schema_version=SCHEMA_VERSION, pipeline_version=PIPELINE_VERSION,
                        dataset_version=version, source_dataset=source_dataset, source_kind=source_kind,
                        normalization=dict(units=NORMALIZED_UNITS, timestamp="source clock retained; elapsed_s and timestamp_utc_s added",
                                           axes="declared signed permutation only; no inferred vehicle alignment",
                                           transformations="unit conversion only; no interpolation/filtering/scaling fit"),
                        policy=asdict(policy), split_policy=dict(strategy=strategy, fractions=fractions, seed=seed,
                                                               window_size=window_size, stride=stride),
                        sessions=sessions, statistics=stats,
                        artifacts={str(p.relative_to(work)): _hash(p.read_bytes()) for p in sorted(work.rglob("*")) if p.is_file()})
        manifest["manifest_sha256"] = digest(manifest)
        _write(work / "manifest.json", manifest)
        # Atomic publication on the same filesystem. Nonempty existing releases cannot be replaced.
        work.rename(destination)
    return destination


def verify_release(path):
    """Check pinned manifest and every archived/derived byte before consumption."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_bytes())
    expected = manifest.pop("manifest_sha256")
    if digest(manifest) != expected:
        raise ValueError("Manifest integrity mismatch")
    for name, checksum in manifest["artifacts"].items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path.resolve()) or not artifact.is_file():
            raise ValueError(f"Missing/unsafe release artifact: {name}")
        if _hash(artifact.read_bytes()) != checksum:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    manifest["manifest_sha256"] = expected
    return manifest
