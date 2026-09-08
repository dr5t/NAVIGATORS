# Navigators IDR — Implementation Plans

## Phase 1: Architecture Recovery (Completed)
- **Goal**: Invalidate false benchmarks and establish a scientifically sound benchmark script.
- **Outcome**: `benchmark.py` rewritten to remove leakage and explicitly require a trained ML model.

## Phase 2: Real Data Pipeline & Training (Completed)
- **Goal**: Ingest real JSON trip data, align coordinate frames, filter noise, and train the TCN.
- **Outcome**: Created `src/data/pipeline.py` with Triad alignment and Butterworth/Median filters. Generated `best_model.pt` on actual sensor noise.

## Phase 3: ONNX Export & Edge Integration (Completed)
- **Goal**: Ensure the AI model can run offline on edge devices (JavaScript target) without Python.
- **Outcome**: `verify_onnx.py` exports the model to WebAssembly format. The JavaScript simulator now natively hosts the ONNX session.

## Phase 4: Frontend Simulator & Offline EKF (Completed)
- **Goal**: Move the entire 15-state EKF and inference logic out of the Python backend.
- **Outcome**: Built `simulator/offline_engine.js` which natively executes the EKF, Phone-to-Vehicle Triad alignment, and TCN inference on-device in the browser using `onnxruntime-web`. The FastAPI backend is officially completely decoupled.

## Phase 5: Live Testing & Map Matching Integration (Pending)
- **Goal**: Record varied real-world trips to finalize the model weights, and replace the synthetic map grid with OpenStreetMap matching.
- **Method**: Mount the phone in a car, use the Edge Simulator to record JSON trips. Train the pipeline. Use offline spatial trees to snap EKF outputs to known road geometries.
