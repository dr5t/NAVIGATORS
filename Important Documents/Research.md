# Navigators IDR — Research & Literature Review

## 1. The Challenge of Smartphone Pedestrian vs. Vehicle DR
Unlike Pedestrian Dead Reckoning (PDR) which relies on step-counting, vehicle dynamics are continuous. Double integration of noisy smartphone accelerometers causes catastrophic quadratic drift within seconds.

## 2. AI for Velocity Estimation
Recent literature (e.g., ION GNSS+) demonstrates that deep learning models—specifically Temporal Convolutional Networks (TCNs) and LSTMs—can map windowed IMU data directly to velocity vectors, bypassing the double-integration problem.
- **Why TCN over LSTM?**: TCNs offer parallelized convolution, meaning lower latency on mobile edge devices compared to the sequential nature of LSTMs.

## 3. 15-State Extended Kalman Filter
Standard 6-state or 9-state filters cannot track sensor biases. A 15-state EKF tracks position, velocity, attitude, AND the dynamic biases of the accelerometer and gyroscope. By using the AI velocity prediction as an "update" measurement, the EKF can continually correct these biases even when GNSS is lost.

## 4. Map Matching
Offline Map Matching using HMM (Hidden Markov Models) acts as a pseudo-lateral constraint, snapping the trajectory to road graphs to eliminate cross-track drift over multi-minute outages.
