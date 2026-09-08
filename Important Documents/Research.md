# Research & Literature Review

## Core Problems with Smartphone Dead Reckoning
Classical Inertial Navigation Systems (INS) calculate position by double-integrating the acceleration data from IMUs. Because commercial smartphone IMUs suffer from significant MEMS (Micro-Electro-Mechanical Systems) noise, quantization error, and thermal drift, this double integration causes position error to grow quadratically (and eventually cubically due to orientation errors) over time.

## Research Insights
To bound this error during GNSS-denied scenarios, this project synthesized several state-of-the-art approaches:

### 1. AI-Driven Velocity Estimation (TCNs vs LSTMs)
Recent literature indicates that treating velocity estimation as a sequence-to-sequence mapping problem yields better results than raw integration. We chose a Temporal Convolutional Network (TCN) over LSTMs because TCNs:
- Exhibit stable gradients over long sequences.
- Can process sliding windows (e.g., 200 samples) in parallel, dramatically reducing inference latency on edge CPUs.
- Predict absolute 2D vehicle velocities directly, sidestepping integration errors.

### 2. Vehicle Kinematic Constraints (NHC & ZUPT)
Research shows that without constraints, an EKF will rapidly diverge. 
- **Non-Holonomic Constraints (NHC):** Because cars typically cannot slide sideways or fly, we mathematically constrain lateral and vertical velocities to near-zero variance.
- **Zero-Velocity Updates (ZUPT):** By examining the variance of the acceleration vector, we can accurately determine if the vehicle is stopped (e.g., at a red light). When stationary, the EKF covariance can be severely collapsed, eliminating drift while waiting.

### 3. Geometric Map Matching
While Hidden Markov Models (HMMs) are standard for cloud-based map matching (e.g., Google Maps), processing Viterbi algorithms natively on a phone is computationally expensive. Research into geometric and distance-based heuristics proved sufficient when coupled with an already highly-accurate EKF output, satisfying the requirement for an offline, self-contained constraint mechanism.
