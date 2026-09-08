# Research Document

**Project Name:** Navigators  
**Topic:** AI/ML for Intelligent Dead Reckoning using IO-VNBD  

---

## 1. Dataset Analysis: IO-VNBD

The IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark Dataset) is a pivotal resource for this project.

- **Origin:** Collected across the UK, Nigeria, and France.
- **Scale:** Contains ~58 hours of smartphone-recorded data covering ~4,400 km, alongside vehicle-extracted ground truth data.
- **Challenge Addressed:** Low-cost INS sensors (like those in smartphones) suffer from substantial noise and bias, causing rapid, cubic drift when integrated traditionally for position.

---

## 2. State-of-the-Art ML for Dead Reckoning

- **Traditional Approach:** Pure mathematical integration of IMU data (accelerometer for velocity/position, gyroscope for heading) leads to cubic or exponential error growth.
- **AI/ML Approach (End-to-End):** Models like RoNIN (Robust Neural Inertial Navigation) or TLIO (Tight Learning-based Inertial Odometry) map sequences of IMU data directly to velocity vectors or relative displacements.
- **Sequence Models:** LSTMs and Temporal Convolutional Networks (TCNs) are highly effective because they can learn the implicit temporal patterns of human or vehicle motion, correcting for vibration and dynamic biases better than traditional filters.
- **Robust Generalization:** Training must include aggressive data augmentations (impulsive shock, Gaussian noise, varying scale and rotation) to prevent the AI from overfitting to one specific car or mounting angle.

---

## 3. Sensor Fusion and EKF

While AI predicts relative motion well, an absolute reference is needed to prevent unbounded drift over long periods. 

- **Extended Kalman Filter (EKF):** The standard algorithm for fusing the AI's high-frequency relative predictions with low-frequency absolute GNSS updates. We use a 15-state EKF implementation for optimal state tracking.
- **Zero Velocity Updates (ZUPT):** A crucial heuristic where the model detects if the vehicle has stopped (e.g., at a red light) to rest accumulate velocity error to zero.
- **Non-Holonomic Constraints (NHC):** Assumes vehicles do not typically slide sideways, constraining lateral velocity to near zero.

---

## 4. Map Matching

- **Purpose:** To snap the estimated trajectory to known road networks, bounding error drift geometrically.
- **Implementation:** Usually handled using Hidden Markov Models (HMM) for probabilistic snapping or geometry-based snapping using offline maps (OpenStreetMap data) computed directly on the edge.
