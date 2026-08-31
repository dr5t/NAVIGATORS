# Navigators IDR: Comprehensive Project Guide
**Project:** Navigators (SIH26168 for ISRO)
**Focus:** AI/ML-based Navigation System during GNSS Denial

---

## 1. The Problem We Are Solving
In mission-critical scenarios—such as military operations, disaster response, and aerospace navigation (ISRO)—vehicles rely heavily on Global Navigation Satellite Systems (GNSS) like GPS or NavIC. However, GNSS signals are easily compromised. They can be jammed by adversaries, blocked by natural terrain (tunnels, dense forests), or fail due to atmospheric conditions.

When a vehicle enters a "GNSS-denied" environment, traditional navigation systems fail, leaving operators blind. While professional Inertial Navigation Systems (INS) exist, they are extremely expensive and bulky. Modern vehicles and personnel carry smartphones with built-in sensors (accelerometers and gyroscopes), but these commercial sensors are extremely noisy. If you simply integrate mobile sensor data to find position (traditional Dead Reckoning), the error multiplies exponentially, leading to severe drift within seconds.

## 2. Our Solution: Intelligent Dead Reckoning (IDR) Edge AI
We have built a 100% offline, AI-powered navigation engine that runs directly on commercial mobile devices (Edge Inference) without requiring any internet connection.

**How it works:**
1. **AI Velocity Estimation:** Instead of relying on pure physics equations that suffer from sensor noise, we trained a Temporal Convolutional Network (TCN). This neural network analyzes a rolling window of vibration and motion patterns from the phone's sensors to accurately predict the vehicle's speed and heading.
2. **Physical Constraints:** The AI is fused with physical constraints like Zero Velocity Updates (ZUPT) and Non-Holonomic Constraints (NHC) to mathematically eliminate drift when the vehicle is stopped or moving in a straight line.
3. **Seamless Handover:** When GPS is lost, the system instantly switches to AI Dead Reckoning. When GPS returns, it automatically corrects any minor drift and resumes satellite tracking.
4. **100% Offline Edge Architecture:** The entire neural network (compiled via ONNX) and Dead Reckoning engine run in WebAssembly directly on the mobile phone's CPU. **No backend servers or cloud connections are required.**

## 3. Impact
- **Resilience:** Provides uninterrupted navigation in tunnels, urban canyons, and jammed environments.
- **Cost-Efficiency:** Achieves professional-grade navigation tracking using cheap, ubiquitous commercial smartphone sensors.
- **Security & Privacy:** Because the system operates entirely offline on the edge device, zero location data is transmitted over a network, making it highly secure for defense and space agency applications.

## 4. How We Will Scale It
- **Model Generalization:** The current AI model can be retrained with massive datasets (like IO-VNBD) covering different vehicle types (cars, trucks, rovers) and mounting positions. As the model sees more data, its accuracy scales universally.
- **Hardware Acceleration:** By utilizing WebAssembly and ONNX Runtime, the system automatically scales to take advantage of whatever hardware it is running on (e.g., utilizing a phone's Neural Processing Unit (NPU) or GPU for faster, lower-power inference).
- **Platform Agnostic:** Because the core engine is built using Web Standards and ONNX, the exact same AI model and logic can run on an iPhone, an Android tablet, a Raspberry Pi inside a drone, or a dashboard infotainment system without rewriting the core engine.

## 5. How We Will Distribute It
We have architected the solution for frictionless, decentralized distribution:
1. **Progressive Web App (PWA):** The application is hosted as a secure webpage. Users simply navigate to the URL, and the app installs itself (along with the AI model) directly onto the device's home screen. This bypasses the need for Apple App Store or Google Play Store approvals, which is crucial for internal government/ISRO distribution.
2. **Automatic Background Updates:** If a new, more accurate AI model (`model.onnx`) is trained by the central team, the PWA's Service Worker will silently download it the next time the device touches Wi-Fi. The operator doesn't have to manually "update" the app.
3. **Native Wrappers:** For highly secure, air-gapped deployments, the entire HTML/JS/ONNX bundle can be wrapped into a native Android `.apk` using tools like Capacitor, allowing it to be sideloaded via USB onto military or ISRO field devices.
