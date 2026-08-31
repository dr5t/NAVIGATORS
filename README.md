# 🧭 Navigators — AI/ML Intelligent Dead Reckoning System

**SIH Problem Statement ID:** SIH26168 (S. No. 168)  
**Organization:** Indian Space Research Organisation (ISRO)  
**Team:** Navigators

---

## 🎯 Problem Statement

Conventional GPS/GNSS-based navigation fails in tunnels, underpasses, and dense urban environments. When satellite signals are blocked, vehicles lose positioning, causing navigation to freeze or jump erratically.

**Our solution:** An AI/ML-powered Intelligent Dead Reckoning (IDR) system that maintains seamless, accurate vehicle navigation using only smartphone IMU sensors during GNSS denial periods.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMARTPHONE / EDGE DEVICE                     │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐ │
│  │  IMU      │───▶│ Preproc  │───▶│ AI Model │───▶│   EKF     │ │
│  │  Sensors  │    │ Filter   │    │ TCN/LSTM │    │  Fusion   │ │
│  └──────────┘    └──────────┘    └──────────┘    └─────┬─────┘ │
│                                                        │       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │       │
│  │  GNSS    │───▶│  ZUPT    │───▶│   NHC    │─────────┘       │
│  │  Receiver│    │  Detect  │    │ Constrain│                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
│  ┌──────────┐    ┌──────────┐                                  │
│  │  Map     │───▶│  UI      │  ◀── Continuous position output  │
│  │  Match   │    │  Render  │                                   │
│  └──────────┘    └──────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Flow

| GNSS Status | Pipeline |
|---|---|
| **Available** | GNSS + IMU → AI filtering → EKF Fusion → Map Match → Position |
| **Lost** | IMU → AI velocity estimation → Dead Reckoning → NHC → Map Match → Position |
| **Restored** | DR → Re-acquisition → GNSS + INS fusion → Corrected position |

---

## 📁 Project Structure

```
Navigators-SIH/
├── src/
│   ├── data/               # Data loading & preprocessing
│   │   ├── io_vnbd_loader.py   # IO-VNBD dataset parser
│   │   ├── preprocessor.py     # IMU filtering, alignment, windowing
│   │   └── synthetic_data.py   # Synthetic data for testing
│   ├── models/             # AI/ML architectures
│   │   ├── tcn_model.py        # Temporal Convolutional Network
│   │   ├── lstm_model.py       # Bidirectional LSTM with attention
│   │   └── trainer.py          # Training loop & checkpointing
│   ├── navigation/         # Core navigation engine
│   │   ├── ekf.py              # 15-state Extended Kalman Filter
│   │   ├── dead_reckoning.py   # Dead reckoning engine
│   │   ├── nhc.py              # Non-Holonomic Constraints
│   │   ├── zupt.py             # Zero Velocity Update detection
│   │   └── map_matching.py     # Road network snapping (Geometric/HMM)
│   ├── edge/               # Edge deployment
│   │   └── onnx_export.py      # PyTorch → ONNX export
│   └── utils/              # Shared utilities
│       ├── coordinates.py      # WGS84 ↔ ENU conversions
│       └── metrics.py          # Drift %, ATE, CEP evaluation
├── simulator/              # Web-based navigation demo
│   ├── index.html              # UI layout
│   ├── index.css               # Premium dark-mode styling
│   └── app.js                  # Map visualization & telemetry
├── scripts/                # Entry points
│   ├── train.py                # Model training
│   ├── evaluate.py             # Benchmarking
│   └── simulate.py             # Full navigation simulation
├── tests/                  # Unit & integration tests
├── configs/
│   └── default.yaml            # All hyperparameters
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Navigators-SIH
pip install -r requirements.txt
```

### 2. Run Navigation Simulation

```bash
python scripts/simulate.py
```

This generates synthetic sensor data, runs the full navigation pipeline (IMU → AI → EKF → DR → Map Match), and outputs results for the web simulator.

### 3. Open the Web Simulator

Open `simulator/index.html` in your browser. The simulator includes built-in demo data and auto-plays a navigation scenario showing:
- Ground truth trajectory (cyan dashed line)
- Estimated trajectory (green when GNSS available, orange during dead reckoning)
- Real-time telemetry: speed, heading, position error, drift %, confidence

### 4. Train the AI Model

```bash
# Train with synthetic data (no dataset needed)
python scripts/train.py

# Train with IO-VNBD dataset
python scripts/train.py --dataset ./data/io_vnbd

# Use LSTM instead of TCN
python scripts/train.py --model lstm --epochs 50
```

### 5. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🎯 Performance Targets

| Metric | Target | Description |
|---|---|---|
| DR Positional Drift | **< 10%** | Of distance traveled |
| 50m GNSS-denied | **< 5m** drift | In under 1 minute |
| 1km GNSS-denied | **< 100m** drift | At 60 km/h |
| Update Rate | **10 Hz** | Smartphone IMU |
| Edge Engine | **200 Hz** | External FOG-IMU |

---

## 🧠 Key Technologies

- **AI Model:** TCN (Temporal Convolutional Network) for lightweight edge inference, LSTM with attention as alternative
- **Sensor Fusion:** 15-state Extended Kalman Filter (position, velocity, orientation, biases)
- **Constraints:** Non-Holonomic (no sideslip), ZUPT (zero velocity at stops)
- **Map Matching:** Geometric snapping + HMM-based probabilistic road selection
- **Dataset:** IO-VNBD (58 hours, 4400 km, collected in UK/Nigeria/France)

---

## 📊 Web Simulator Features

- 🗺️ Full-screen dark-mode map with Leaflet.js
- 📡 Real-time GNSS status with signal bars
- 📈 Live telemetry: speed, heading, position error
- 🎯 Drift ring gauge with target indicator
- 🎮 Playback controls with speed adjustment
- 🏠 Built-in demo data (works offline)

---

## 📦 Edge Deployment

Export trained model to ONNX for mobile deployment:

```python
from edge.onnx_export import export_to_onnx
export_to_onnx(model, "model.onnx")
```

---

## 📄 License

This project is developed for Smart India Hackathon 2026 (SIH26168).
