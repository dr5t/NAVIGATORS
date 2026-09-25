# Navigators India Dataset pipeline

The Phase 11 implementation lives in `src/data/india_dataset/`. It collects raw
sample records, validates whole sessions, creates a normalized representation,
publishes immutable source-specific releases, assigns splits before indexing
windows, and evaluates predictions against attributed reference measurements.
It uses the Python standard library; model training dependencies are unnecessary.

**This change includes no collected road dataset and claims no dataset size or
road accuracy.** Test fixtures are synthetic software checks. Existing IO-VNBD,
synthetic trajectories, and diagnostic browser recordings are not original
Navigators India Dataset sessions. Existing administrative session approval and
the older `src/data/pipeline.py` feature exporter do not certify this contract.

## Collection contract (schema 1.0.0)

Supported inputs:

- JSON: `{"metadata": {...}, "records": [{...}, ...]}`.
- JSONL: one sample object per line, with `trip.jsonl.metadata.json` sidecar.
- CSV: named columns with `trip.csv.metadata.json` sidecar. Blank cells mean
  unavailable, numeric columns are parsed explicitly, and `categories` is a
  JSON-encoded array when supplied. Repeated headers/extra cells are rejected.

Original input bytes and sidecars are archived by SHA-256, including rejected
inputs. Extra raw fields remain in this archive. Normalized rows do not replace
raw files. Failed validation never sorts, interpolates, clips, imputes, or deletes
samples. Corrections require a separately documented new input and release.

Every sample uses these fields where available:

| Group | Fields |
| --- | --- |
| Required measurements | `timestamp`, `accelerometer_x/y/z`, `gyroscope_x/y/z` |
| Nullable GNSS | `latitude`, `longitude`, `GNSS_speed`, `GNSS_heading`, `GNSS_accuracy` |
| Context | `device_id`, `vehicle_id`, `session_id`, `phone_mount_position`, `vehicle_type`, `road_type`, `environment`, `activity` |
| Quality annotations | Required `gnss_status`; optional `gnss_timestamp`, `categories` |

Context defaults to the session metadata when absent from a row. Missing device,
vehicle, road and environment information is reported, never invented. Use stable
pseudonyms for device/vehicle identity. A phone or vehicle change starts a new
session. `recording_id` identifies the **whole physical drive**, including chunks
exported to separate files. Never assign fresh recording IDs to windows of a drive.

Metadata example (replace placeholder declarations with actual collection facts):

```json
{
  "schema_version": "1.0.0",
  "session_id": "<unique-session-id>",
  "recording_id": "<physical-drive-id>",
  "source_kind": "navigators_collection",
  "source_dataset": "Navigators India Dataset",
  "country": "IN",
  "consent": true,
  "provenance": {
    "original_collection": true,
    "collector": "<collector-pseudonym>",
    "collection_method": "<sensor-adapter-and-version>",
    "license": "<actual-permission-or-license>"
  },
  "device_id": "<stable-device-pseudonym>",
  "vehicle_id": "<stable-vehicle-pseudonym>",
  "vehicle_type": "car",
  "phone_mount_position": "dashboard",
  "road_type": "urban",
  "environment": "urban",
  "activity": "driving",
  "timestamp_clock": "elapsed",
  "start_time_utc_s": 0,
  "sample_rate_hz": 50,
  "acceleration_includes_gravity": true,
  "units": {
    "timestamp": "s",
    "accelerometer": "m/s^2",
    "gyroscope": "rad/s",
    "coordinates": "deg",
    "GNSS_speed": "m/s",
    "GNSS_heading": "deg",
    "GNSS_accuracy": "m"
  },
  "axes": {
    "frame": "phone",
    "description": "<physical positive x, y, z directions from adapter specification>",
    "accelerometer": ["x", "y", "z"],
    "gyroscope": ["x", "y", "z"]
  }
}
```

`start_time_utc_s` must be the actual UTC epoch corresponding to elapsed zero,
not the placeholder zero. Alternatively use `timestamp_clock: "unix"` and epoch
timestamps. `gnss_timestamp` uses the same clock and units as `timestamp` and
records receiver acquisition time, not the time an old fix was polled again.
Sampling rate is declared for the shared IMU sample stream. Asynchronous adapters
must retain their original logs and expose this contract explicitly; the pipeline
does not silently synchronize or upsample sensor streams.

Allowed input units: seconds/milliseconds/microseconds/nanoseconds; acceleration
`m/s^2` or `g`; angular rate `rad/s` or `deg/s`; coordinates/heading `deg` or `rad`;
speed `m/s` or `km/h`; accuracy `m`. Normalized output uses seconds, m/s², rad/s,
WGS84 degrees and m/s. `timestamp` keeps its declared clock; `elapsed_s` and
`timestamp_utc_s` are added. Axis lists are signed permutations: e.g.
`["y", "-x", "z"]` maps output x from raw y and output y from negative raw x.
Physical orientation and gravity inclusion remain explicit declarations. No
alignment or gravity removal is inferred from future samples. No z-score
statistics are fitted here; any downstream learned scaler must fit train only.

## GNSS and road coverage

`gnss_status` is a collector annotation, not a claimed ground-truth inference:

| Status | Meaning |
| --- | --- |
| `normal_gnss` | Available coordinates consistent with normal-quality policy |
| `degraded_gnss` | Available/partial receiver measurements are unreliable |
| `gnss_outage` | No usable receiver fix; missing GNSS is expected |
| `gnss_recovery` | Fix reacquisition following degradation/outage |

Use nulls for measurements the receiver did not supply. Retained last-known
values during outage are preserved and flagged, never converted into valid
training labels. Whole sessions with no GNSS fixes are supported. In-range jumps
in degraded/recovery intervals are retained with warnings for anomaly research;
impossible numeric ranges and unexplained jumps in normal intervals reject the
session. Sparse optional speed/heading/accuracy is counted in quality reports.
Missing receiver timestamps produce an explicit freshness limitation.

Additional `categories` may contain `urban`, `highway`, `rural`, `hilly`,
`dense_traffic`, `stop_and_go`, `parallel_roads`, `service_roads`, `underpass`,
and `tunnel`. Matching `road_type`, `environment`, or `activity` values are also
counted. Tags can overlap. Tunnel/underpass annotations require
`legal_collection_confirmed: true` in session metadata. Collect only where
permitted and practical, using a secured mount and an operator other than the
driver for any interaction. Do not deliberately create unsafe driving conditions
or interfere with GNSS reception. Missing category coverage stays visible as zero.

## Acquire and build

Sensor adapters can use `SessionCollector(path, metadata).append(sample)`, or send
one JSON sample per line to the command below. It writes exclusively to a new file
and flushes each sample; closing fsyncs the file. It is an acquisition sink, not a
phone sensor driver. Invalid captured sessions are preserved for later rejection.

```bash
python -m src.data.india_dataset collect recordings/drive.jsonl --metadata metadata.json
python -m src.data.india_dataset build recordings/drive.jsonl --output data/dataset_releases --version 0.1.0
```

The collect command reads stdin until EOF. Do not pipe a complete JSON session
object into it; use sample objects. Existing browser array-format exports require
an explicit adapter and verified units/axis declarations before admission.

Build validates ordering, sample rate/gaps/jitter, missing values, units, axes,
finite values/ranges, receiver freshness/time ordering, coordinate jumps, session
identity, source provenance, duplicate samples, duplicate sessions, identical
traces and overlapping re-exported segments. Defaults and overrides are recorded
in the manifest. Supply policy overrides with `--policy policy.json`, e.g.:

```json
{"sample_rate_tolerance": 0.1, "max_gap_periods": 3, "normal_accuracy_m": 15}
```

All invalid sessions receive machine-readable rejection reasons and no normalized
data or split. A mixed release publishes its accepted subset and all quality
reports; the CLI exits **2** when any input is rejected, **0** otherwise. An empty
build produces an honest zero-count report. A missing/unreadable input file aborts
publication, as its raw bytes cannot be preserved.

## Source separation and versioning

Releases are stored at
`<output>/<source_kind>/<source-slug-and-hash>/<version>/`:

```text
manifest.json           # schema/pipeline/dataset versions, policy, attribution, hashes
raw/                    # exact accepted raw input and sidecar bytes
quarantine/raw/         # exact rejected inputs, including source mismatches
normalized/             # accepted session JSONL, with source_dataset on every row
quality_report.json     # validation issues for every input
statistics.json         # actual counts, coverage gaps, split limitations
windows.jsonl           # indices referencing whole sessions and their fixed splits
```

`navigators_collection`, `external_benchmark`, and `synthetic` are distinct source
kinds. Every release contains one declared dataset. Source mismatches are rejected
and archived only for audit, never normalized or counted as accepted observations.
External/synthetic sessions must identify `source_dataset` and provenance with
`source_uri`, `source_version`, `license`, and `attribution`. For example:

```bash
python -m src.data.india_dataset build external/session.json --output data/dataset_releases --version benchmark-1 --source-kind external_benchmark --source-dataset IO-VNBD
```

Renaming external data to the reserved Navigators dataset name is rejected. This
does not mechanically prove ownership: collectors remain responsible for truthful
provenance and permission declarations. Keep original citations and license terms
with external data. Never copy external benchmark counts into original collection
statistics.

Both existing IO-VNBD loaders also expose `source_dataset = "IO-VNBD"`,
`source_kind = "external_benchmark"`, and `get_sample_metadata(index)` without
changing their tensor-pair return API. Saved IO-VNBD normalization statistics carry
these source fields. The legacy phone feature exporter preserves source fields
from input metadata, leaving unknown attribution null rather than assuming it is
original Navigators data. These legacy loaders remain separate from release
validation; their tensors are not automatically admitted to the India Dataset.

Existing versions cannot be overwritten. Publication stages all artifacts and
renames the finished directory atomically. SHA-256 covers raw files, normalized
rows, reports, window indices and the manifest. Verify before consuming a release:

```bash
python -m src.data.india_dataset verify <release-directory>
```

Checksums detect changes relative to the manifest; they are not an authenticity
signature. Pin the manifest hash with every experiment. New data produces a new
release. Split assignment is reproducible for the same inputs/policy/seed, but
adding groups can change assignments across versions: comparisons must pin the
same release and must not pool differently partitioned versions.

## Splits and window leakage

`--split-strategy session|vehicle|device|joint` supports independent experiments.
The default `joint` keeps every connected set of shared recording, vehicle and
device identities together, including transitive links. Every strategy keeps a
whole recording together. Adjacent/overlapping sessions with a shared known
vehicle/device are joined using the configurable temporal guard, even in session
mode. Exact duplicate traces and three-sample overlapping exports are rejected
before splitting. Equal stationary traces can conservatively be flagged as
duplicates; inspect the raw audit evidence rather than silently bypassing rejection.

`--fractions 0.7 0.15 0.15` targets session counts, subject to indivisible groups.
Exact ratios and nonempty splits are not guaranteed for small/connected datasets;
reports explain these limitations. Unknown device/vehicle IDs are not invented:
reports explicitly decline to guarantee isolation for those identities.

`--window-size 200 --stride 50` indexes windows **after** split assignment. Each
window carries session identity, source dataset, split, and exclusive start/stop
indices. No window crosses a session boundary; no part of a session can enter a
different split. Window indices are not training labels or prefiltered tensors.
Consumers must honor the pinned split and quality annotations. Separate natural
outage evaluation from artificially masked GNSS experiments in experiment metadata.

## Evaluation

Submit predictions and independently acquired reference measurements in two JSON
files. Each contains `metadata` and `records`. Every record specifies
`session_id`, `sample_index`, `timestamp_utc_s`, `latitude`, `longitude`. Timestamps
must match the indexed release sample (1 µs tolerance); there is no interpolation
or nearest-neighbor matching. Duplicate keys and records outside the selected
split are rejected.

Prediction metadata requires `model_id` and `training_session_ids` (an empty array
is valid only for a model using no such sessions). Held-out overlap is rejected.
Reference metadata requires `reference_kind` (`independent` or `receiver_proxy`),
`source_dataset`, and a `provenance` string describing the reference acquisition,
accuracy and synchronization. Reference independence is a declaration requiring
researcher review; the evaluator cannot establish it from coordinates alone.

```bash
python -m src.data.india_dataset evaluate <release-directory> --predictions predictions.json --reference reference.json --split test --output evaluation.json
```

The report pins release/model/reference identities and file hashes, counts matched
and missing coverage, and computes mean, RMSE, nearest-rank p95 and maximum
haversine position error, overall, per session and per category. Empty scores are
null. Receiver proxies score only normal samples with accuracy and fresh receiver
timestamps and are labeled **proxy agreement, not ground-truth accuracy**. Natural
outage accuracy requires independent reference measurements. Missing references
cannot be replaced with zeros, interpolation, model output or fabricated results.

## Verification and extension

```bash
python -m pytest tests/test_india_dataset_pipeline.py -q
```

Add future collections by supplying files conforming to the same contract. New
hardware can use an adapter feeding `SessionCollector`; preserve original sensor
logs as part of collection provenance. Additive raw fields survive the archive.
Contract or transformation changes require a schema/pipeline version update.
Current builds use in-memory session validation and duplicate indexes; very large
releases may need a streaming implementation behind the same public contract.
