# Reproducible navigation evaluation

No real phone recording is currently available. Real-data training, held-out accuracy, real-input export parity, and smartphone performance remain **unmeasured**. Existing trip_4 data fails GPS plausibility checks, and its processed copies overlap across train/validation/test. The tools reject these cases; unit and integration tests use explicitly labeled synthetic fixtures only.

## Record and replay

Use **Record Trip Data** in the browser after motion/location permissions are granted. Capture a GPS-available calibration prefix with forward acceleration, the intended outage interval, and a GPS-available recovery interval. Keep a physical device/mounting description alongside each trip. The logger retains missing GPS as null and includes fix timestamps, sensor axis order, units, and user-agent metadata. A recorder label documents capture provenance; it is not proof that a file has never been edited.

Replay JSON uses the recorder's `data.timestamps`, `accel`, `gyro`, `gnss`, and optional `gnss_timestamps` arrays. Timestamps may be Unix or elapsed **seconds**, with GNSS timestamps in the same clock domain. GNSS rows contain `[lat, lon, altitude, speed_mps, heading_degrees, accuracy_m]`. Gyroscope values are XYZ **rad/s**; `--gyro-order alpha-beta-gamma` supports old recorder files. Accelerometer data includes gravity in **m/s²**. Missing GPS rows may be six nulls. IMU timestamps must strictly increase; split a trip at sensor gaps exceeding one second.

CSV columns:

```text
timestamp,ax,ay,az,gx,gy,gz,lat,lon,alt,speed,heading,accuracy,gnss_timestamp
```

`gnss_timestamp` is optional but recommended. Legacy recordings without it infer new fixes from changed GPS values and expire unchanged fixes after three seconds; reports flag that limitation. Unknown speed/heading cannot be a valid velocity/heading reference.

```bash
# Baseline works without a trained model:
python replay.py --dataset trip_07.csv --mode A --gnss-outage 60 --outage-start 10
# Full system after real-data training and verified ONNX export:
python replay.py --dataset trip_07.csv --gnss-outage 60 --outage-start 10
# All seven configurations on identical inputs and outage timing:
python replay.py --dataset trip_07.csv --gnss-outage 60 --ablations
# Equivalent benchmark entry point:
python scripts/benchmark.py --dataset trip_07.csv --gnss-outage 60
```

Run from the project root using the environment with `requirements.txt` installed. Outage seconds are relative to the recording's first sample. The denied interval is `[start, start + duration)` and must leave recovery samples; requests longer than the available trip fail instead of silently shortening. Default calibration is five seconds. Use `--calibration-seconds` and `--outage-start` to leave enough GPS/IMU warm-up for the 200-sample model window. Use `--aligned` **only** for known vehicle-frame recordings. Automatic calibration fails explicitly if it cannot resolve gravity or forward direction.

The downloaded OSM map must cover the initial fix. Set `--map path/to/road_network.json` or download the correct area. Map matching uses the existing geometric matcher, transformed into the same local ENU origin as navigation. It never generates a substitute road grid.

Output in `results/replay/`:

- `trip_07_results.json`: configuration, input/model/map/source hashes, calibration, mode transitions, component invocation counts, timings, and metrics for the whole navigated trip, outage, and recovery.
- `trip_07_A_trajectory.csv` through `G`: timestamped ENU position/velocity, heading, mode, and GPS availability. Calibration rows have blank estimates.
- `trip_07_A_trajectory.json` through `G`: open these in **Saved playback → Open trajectory**. Calibration rows are omitted, trip timestamps are preserved, and unavailable references, confidence, drift, and per-frame ZUPT status remain null. The files retain recording provenance and evaluation configuration.
- `trip_07_ablation.csv`: comparable outage metrics and explicit blocked statuses/reasons.

The default mode is G. A missing or incompatible model blocks C–G; an invalid map blocks G. The CLI returns nonzero if any requested experiment is blocked. `--allow-legacy-model` permits explicitly diagnostic runs of old checkpoints; it does not validate their preprocessing or training provenance.

## What each ablation changes

All configurations use the same permitted GPS fixes outside the outage and the same initial calibration. A/B/C use direct integration and GPS corrections without an EKF; D–G use the production Python EKF.

| Mode | Components |
| --- | --- |
| A | Aligned raw IMU → integration |
| B | Same IMU → causal filtering → integration |
| C | Filtered IMU → normalized TCN → velocity integration |
| D | TCN + EKF |
| E | D + NHC |
| F | E + ZUPT |
| G | F + downloaded OSM map matching |

“Raw” omits noise filtering, not the coordinate conversion and gravity convention needed to integrate phone IMU meaningfully. Filtering uses a trailing median of five samples, a causal 20 Hz first-order low-pass filter, and gravity subtraction for model features. Only the pre-outage prefix determines alignment. The training pipeline and replay use this same frontend. Browser features are buffered from motion events, independent of the approximately 10 Hz inference/UI loop. The browser checks the observed IMU rate against a verified model's recorded training rate; incompatible rates disable AI and show an error. Python replay currently invokes inference on every full-window sample, so its total-loop workload differs from the browser's.

## Metric definitions and limits

GPS is omitted from navigation at the **first** denied sample. It is not reintroduced through GPS speed, GPS heading, alignment, whole-trip filtering, or reference interpolation. Scoring happens after navigation finishes.

Recorded GPS is a **reference**, not independent ground truth. Scores use fresh valid GPS fixes; natural GPS loss provides no hidden position reference, so unscorable metrics are null/unavailable. Final error is the error at the last scorable fix in the named interval, not necessarily its exact endpoint. All position metrics are horizontal ENU meters.

- Final/mean/max position error: final/arithmetic mean/maximum Euclidean reference-relative distance.
- Position RMSE: square root of mean squared Euclidean position error; distinct from mean error.
- Velocity RMSE: square root of mean squared two-dimensional velocity-vector error, m/s.
- Heading RMSE: wrapped angular error in degrees, excluding reference speeds below 0.5 m/s.

Repeat a command with the same file hashes and configuration to reproduce the trajectory and scores. Wall-clock timings and RAM are measurements and will vary. Results may show that a component worsens accuracy; the tool does not select favorable windows or assert that every added component improves results.

## Train → export → measured parity

Collect separate trips for train, validation, and test, update `split_config.json`, then run:

```bash
python src/data/pipeline.py
python scripts/train.py
python scripts/verify_onnx.py --dataset data/raw_trips/held_out_trip.json --export
```

Training checks content hashes and source-trip hashes across splits, requires causal preprocessing sidecars, and computes normalization only from training data. Old generated `.npy` files without these sidecars are rejected. Keep incompatible old outputs separate when rebuilding the dataset. The checkpoint records its normalization/preprocessing contract. Test evaluation reloads the best checkpoint.

The parity script reconstructs the saved model architecture, selects deterministic recording windows, applies the same normalization to both runtimes, and compares actual PyTorch/ONNX outputs with `abs(diff) <= atol + rtol * abs(PyTorch)`. Defaults are `atol=1e-5`, `rtol=1e-4`. JSON includes input hashes/window indices, every compared output, measured max/mean difference, versions, and pass/fail. Non-finite results fail. Without `--export`, it tests the existing ONNX artifact. With `--export`, it stages the export and replaces the artifact only after parity passes, writing `model.contract.json` beside it. This establishes numerical export parity, not navigation accuracy.

The existing checkpoint was not retrained on placeholder data. Its browser contract is marked `legacy-unverified`; it is a diagnostic model. Re-export a properly trained causal model before claiming real performance/accuracy, bump the service-worker cache version, and refresh the offline package.

## Measure the actual target device

Python reports are labeled **Development-machine benchmark** and record OS/architecture/Python, single-thread model startup, actual TCN/EKF/map/total-loop call counts and mean/p50/p95/max timings, whole-process peak RSS, and model bytes including external ONNX weights. The full loop excludes scoring/file output. Initial inference is included; inactive components have zero calls and unavailable latency, not a fabricated measurement.

On the intended phone, open the offline app, start the live engine, complete calibration, run movement/outage/recovery, and select **Export device timings**. Use **Reset timings** to start a fresh measurement interval. The report includes browser/device-identifying metadata, observed IMU rate, model startup and bytes, and the latest 600 real calls per component. Record the phone model, OS, browser, power/thermal conditions, and whether the page stayed foreground alongside this report. Desktop Chrome results are desktop-browser results, not smartphone measurements.

Browsers do not reliably expose process RAM. The report leaves `ram_bytes` null and optionally labels Chromium's heap estimate separately; that estimate can omit WASM/browser allocations. Measure target process RAM with the device's OS profiler and attach that result separately. [MDN documents the heap API's limitations](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory).

## Failure and recovery demo

1. GPS available: initialize from a current fix, complete alignment and the AI buffer; verify **NORMAL / GNSS + INS**.
2. Disconnect internet. Press **Simulate GNSS outage**. The GPS watcher stops, pending fixes are rejected, and the marker follows IMU/AI/EKF navigation. Read the individual AI, EKF, NHC, ZUPT, and map-matching statuses; “no confident match” is shown when no road supports a correction.
3. Press **Restore GPS**. A fresh fix enters **REACQUISITION**. The full correction is capped at 2 meters per fix, including position movement caused by velocity fusion. Normal mode resumes only after at least five good fixes and a residual within 3 meters. This is a correction limit; physical movement between fixes is additional. Persistent disagreement remains visibly in reacquisition.

Saved visual playback is distinct from running navigation on a recording; use `replay.py` for reproducible experiments. Real sensor/road accuracy and convergence under real multipath still require physical validation. No cloud service, extra vehicle hardware, or synthetic distance claim is needed for this workflow.

## Software validation

```bash
python -m pytest tests/test_replay.py tests/test_ekf.py -q
node --test tests/offline_navigation.test.cjs
node tests/browser_offline.mjs
```

Tests cover all seven GPS-leakage gates, repeatability, causal filtering, epoch timestamps, stale fixes, corrupt recordings, split leakage, mean versus RMSE, angle wrapping, bounded recovery, and runtime profiling. The browser test runs actual local ONNX/WASM inference and recovery with controlled sensor inputs and networking disabled. These checks validate software behavior, not real-trip accuracy.

## Demo workspace

Serve `simulator/` locally and open the navigation workspace:

```bash
python3 -m http.server 8000 --directory simulator
```

Open `http://localhost:8000`. The interface has four views:

- **Navigation**: local OSM map, live telemetry, sensor pipeline, recording, and the outage/recovery sequence. It opens in standby; no saved readings are presented as live sensor data. The map's follow toggle and fit-area button control the viewport.
- **Saved playback**: play, pause, reset, and scrub the included illustrative trajectory. Playback follows its actual timestamps and selected speed. The included Delhi trajectory lies outside the Bengaluru map; the coverage message identifies that mismatch.
- **Experiments**: import a `*_results.json` replay report to compare A–G. Imports remain on the device. Empty/blocked results have no invented numbers, and configurations from different recordings/outage intervals are rejected as incomparable.
- **Device timings**: inspect actual timing samples, export them, reset the sample window, or import a measured ONNX parity report.

The field guide is available from the navigation sidebar, including its compact phone layout. The interface and its scripts are cached with the offline package; reconnect and reload once to install an updated service worker before presenting offline.

## Walking with online and offline maps

Choose **Walking · GPS + steps** and start the engine on a phone over trusted HTTPS (or localhost on that device). Allow location, motion, and compass permissions. GPS initializes the marker immediately without the vehicle model's alignment phase. Hold the phone screen-up, with its physical top pointing forward, tilted less than 45 degrees. Keep the page visible; background sensor delivery is not guaranteed.

Choose **Online streets** for the online OSM basemap. At the area you intend to walk, press **Save area offline** while connected. This saves street geometry for a 2 km square around the current map center, replacing the previous user-downloaded area. Wait for the saved-area confirmation and the offline application readiness message, then reload before disconnecting. If internet connectivity or tile loading fails, the view uses saved streets automatically. Outside that area the app can still report coordinates, but offline street coverage is unavailable. Browser storage can be cleared or evicted; verify the offline view before starting.

Online tiles use the standard OSM service for interactive viewing. Offline downloads retrieve road geometry through Overpass; they do not bulk-download standard tiles, in accordance with the [OSM tile policy](https://operations.osmfoundation.org/policies/tiles/).

Internet loss and GPS loss are separate. With GPS available, location updates continue regardless of internet connectivity. With GPS unavailable for more than three seconds (or **Simulate GNSS outage** enabled), walking mode estimates displacement from detected steps and absolute compass heading. The default step length is 0.70 m; set your own average distance per step before starting. Latitude and longitude are converted using the existing map projection. Recovery applies at most 2 m correction per new fix and requires at least five fixes near the estimate before returning to normal GPS tracking.

This is a pedestrian dead-reckoning prototype, not a trained walking AI. Phone handling can create false steps, magnetic interference can bias heading, and step-length error accumulates. Relative orientation is not treated as north; absent or stale compass/motion data pauses sensor-based displacement. Walking mode does not apply vehicle EKF/NHC/ZUPT or force positions onto roads. It reports GPS accuracy only while fixes are available and leaves unmeasured confidence/drift blank. No real walking accuracy has been established. The Python replay configurations remain vehicle-oriented experiments; importing their outputs does not evaluate this new walking tracker.

Absolute orientation handling follows the [W3C orientation coordinate convention](https://www.w3.org/TR/orientation-event/). Browser and physical-phone support must be checked on the target device.
