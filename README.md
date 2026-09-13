# Navigators IDR — Smartphone Intelligent Dead Reckoning

*Smart India Hackathon 2026 (SIH26168) — ISRO Problem Statement*

This project implements smartphone IMU navigation, a TCN velocity model, and a 15-state EKF with an offline browser runtime. Real-trip navigation accuracy and smartphone performance have not yet been validated. No valid real phone recording is currently available; the existing checkpoint is a diagnostic artifact.

## System Architecture

1. **Smartphone-Only Interface**: Records gravity-included acceleration and XYZ gyroscope readings at the device event rate, with timestamps.
2. **Triad Automatic Alignment**: Dynamically estimates the *Down* vector using gravity, and the *Forward* vector using motion, allowing the system to rotate the raw, arbitrary smartphone IMU readings into the vehicle frame; calibration requires valid GPS and forward acceleration.
3. **Temporal Convolutional Network (TCN)**: An ONNX-exported neural network processes rolling 200-sample windows of normalized causal IMU features, predicting velocity in East/North order. Window duration depends on the recorded sample rate.
4. **15-State Offline EKF**: Implemented in JavaScript for local execution. It tracks 3D Position, 3D Velocity, 3D Attitude (Euler angles), and biases.
5. **Dynamic Constraints**: Integrates Non-Holonomic Constraints (NHC) and Zero Velocity Updates (ZUPT) when stationary.

## Core Directives

### 1. Completely Offline Edge Execution
The system strictly honors the SIH requirement: **No server dependencies at runtime.**
The Python FastAPI backend included in this repository is strictly a **development tool** for validating logic and training the AI. The final inference engine is `simulator/offline_engine.js`, which executes the EKF, the alignment mathematics, and the ONNX model locally in the browser using `onnxruntime-web`.

### 2. Zero Ground-Truth Leakage
The replay evaluator removes GPS measurements at the first denied sample, including GPS speed and heading. Calibration uses only the prefix before the outage; filtering uses past and current IMU samples. Recorded GPS is used separately as a scoring reference, not independent ground truth. No distance or accuracy claim is generated without measured reference data.

### 3. Real-World Training Data Collection
Training now uses the official **IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)**.
- See `data/IO-VNBD-SETUP.md` for download instructions.
- The pipeline parses the official V and S synchronized CSV pairs, treating each as an independent session to prevent data leakage.
- During training only, GNSS velocity serves as the ground truth label.

## Quickstart (Development)

**1. Collect Real Trips:**
- Launch the offline PWA simulator.
- Click **Start Data Logger**.
- Drive your car to record a `.json` trip.

**2. Run the Processing Pipeline:**
```bash
python src/data/pipeline.py
```
This validates phone recordings, calibrates using the initial GPS-available prefix, applies a causal median/low-pass filter, removes gravity for model inputs, and saves targets with source/preprocessing metadata.

**3. Train & Export the AI Model:**
```bash
python scripts/train.py
python scripts/verify_onnx.py --dataset data/raw_trips/your_held_out_trip.json --export
```
This trains the TCN model, saves the exact normalization statistics, and exports `model.onnx` back to the JavaScript edge simulator.

**4. Run Validation Benchmark:**
```bash
python scripts/benchmark.py --dataset data/raw_trips/your_held_out_trip.json --gnss-outage 60
```
This runs all seven ablations on the same recording and time-based outage, saving trajectories, component timings, and reference-relative errors.

## Offline map and outage demo

The browser renders roads directly from `simulator/data/road_network.json`, the downloaded OSM database. It uses that same database for map matching. Leaflet, ONNX Runtime Web, WASM, model weights, icons, and application scripts are bundled locally. No tile server, CDN, online routing service, or Python inference server is used by the browser engine.

Prepare before disconnecting:

```bash
# Only needed to fetch/refresh the bundled browser dependencies:
python3 scripts/prepare_offline_assets.py
# Choose your actual demo area (defaults to ISRO headquarters, Bengaluru):
python3 scripts/download_osm_network.py --lat 13.0326 --lon 77.5582 --radius 2000
# Serve locally over HTTPS for Android sensor access (requires LAN connection):
python3 scripts/serve_phone.py --ip <your-local-ip>
```

1. Open the provided `https://<your-local-ip>:8443` on your phone browser. Follow the prompt to install the CA certificate first if needed, to bypass self-signed warnings and allow sensor access. Wait for **Offline files ready**, then reload once. Keep the same URL when reopening offline.
2. Start **Offline Engine**, allow motion/location access, complete orientation calibration while GPS is available, and acquire an initial fix. Wait for the AI buffer to fill.
3. Turn internet access off. Select **Simulate GNSS outage**: this stops the navigation GPS watcher and rejects pending GPS callbacks. Move within the downloaded area; local IMU, ONNX inference, EKF, and map matching continue. **Restore GPS** starts a fresh watch.
4. Reload with internet still off to verify the map and model are cached. A fresh engine session needs alignment and an initial GPS fix again; initialization is not persisted across reloads.

The included map covers Bengaluru; the saved simulation JSON is a Delhi trajectory. Playback is paused initially when it lies outside the downloaded map. Download Delhi (`--lat 28.6139 --lon 77.209 --radius 3000`) to view that replay on local roads. Saved playback displays previously computed results; use the live engine to demonstrate actual on-device inference. This road map supports position tracking and map matching; it does not implement turn-by-turn route planning.

When updating the map, model, or application assets, bump `CACHE_NAME` in `simulator/sw.js` and reload while connected to install the new complete package. Browser storage can be cleared or evicted, so check the ready indicator before each demo.

Offline regression checks:

```bash
node --test tests/offline_navigation.test.cjs
node tests/browser_offline.mjs
```

The browser check uses Chrome (set `CHROME_BIN` on non-macOS systems), a temporary browser profile, and a local static server. It verifies an offline reload, OSM rendering, and actual WASM inference with network access disabled. Sensor/GPS inputs in that check are controlled fixtures, not a physical road-accuracy validation.

## Replay, ablations, export parity, and device measurements

See [the evaluation guide](docs/replay-and-evaluation.md) for the recording schema, exact metric definitions, reproduction commands, model contract, and failure/recovery procedure. Start with a valid recording; `trip_4.json` is rejected because its GPS values are implausible. Nothing in the new reports treats the existing placeholder data as a real road benchmark.
