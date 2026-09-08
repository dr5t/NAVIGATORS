/**
 * Navigators IDR — 100% Offline Edge Inference Engine
 * Captures mobile IMU & GNSS data and runs PyTorch ONNX model via WebAssembly.
 * Performs Dead Reckoning locally on the mobile processor without any network connection.
 */

class OfflineEngine {
    constructor() {
        this.isCapturing = false;
        
        // Sensor Data Buffer
        this.currentAccel = [0, 0, 0];
        this.currentGyro = [0, 0, 0];
        this.currentGnss = null;
        
        // Buffer for ONNX model (window_size = 200, features = 6)
        this.windowSize = 200;
        this.sensorBuffer = []; // stores [ax, ay, az, gx, gy, gz] arrays
        
        // ONNX Runtime Session
        this.session = null;
        this.modelLoading = false;
        
        // Navigation State
        this.initialized = false;
        this.refLat = 0;
        this.refLon = 0;
        
        // Edge EKF Engine
        this.ekf = new ExtendedKalmanFilter();
        
        // Setup simple map matching grid for demo
        this.roadNetwork = new RoadNetwork();
        this.roadNetwork.generateGridNetwork([0, 0], 100.0, 5);
        this.mapMatcher = new GeometricMapMatcher(this.roadNetwork, 30.0);
        
        // EKF Mode Emulation
        this.gnssAvailable = false;
        this.zuptActive = false;
        
        // Loop controls
        this.lastTime = performance.now() / 1000.0;
        this.loopInterval = null;
        
        // Ensure ONNX Runtime is available
        if (typeof ort !== 'undefined') {
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        }
    }

    async initModel() {
        if (this.session) return true;
        if (this.modelLoading) return false;
        
        this.modelLoading = true;
        try {
            console.log("[Edge AI] Loading ONNX Model into WebAssembly...");
            // Load the model exported to the simulator folder
            this.session = await ort.InferenceSession.create('./model.onnx');
            console.log("[Edge AI] Model loaded successfully!");
            this.modelLoading = false;
            return true;
        } catch (e) {
            console.error("[Edge AI] Failed to load ONNX model:", e);
            this.modelLoading = false;
            alert("Failed to load Edge AI Model: " + e.message);
            return false;
        }
    }

    async requestPermissionsAndStart() {
        // iOS requires explicit permission for DeviceMotionEvent
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            try {
                const permission = await DeviceMotionEvent.requestPermission();
                if (permission !== 'granted') {
                    alert('Sensor permission denied.');
                    return false;
                }
            } catch (e) {
                console.error('Error requesting sensor permission:', e);
                alert('Must be in HTTPS to request sensor permissions.');
                return false;
            }
        }
        
        const modelLoaded = await this.initModel();
        if (!modelLoaded) return false;

        this.startCapture();
        return true;
    }

    startCapture() {
        if (this.isCapturing) return;
        console.log('[Edge Engine] Starting sensor capture & local processing...');
        this.isCapturing = true;

        // Reset buffers and timers
        this.sensorBuffer = [];
        this.lastTime = performance.now() / 1000.0;

        // IMU Listeners
        this.handleMotion = (event) => {
            const accel = event.accelerationIncludingGravity;
            if (accel) {
                this.currentAccel = [accel.x || 0, accel.y || 0, accel.z || 0];
            }
            const gyro = event.rotationRate;
            if (gyro) {
                const toRad = Math.PI / 180;
                this.currentGyro = [
                    (gyro.alpha || 0) * toRad, 
                    (gyro.beta || 0) * toRad, 
                    (gyro.gamma || 0) * toRad
                ];
            }
        };
        window.addEventListener('devicemotion', this.handleMotion);

        // GNSS Listener
        if ('geolocation' in navigator) {
            this.watchId = navigator.geolocation.watchPosition(
                (position) => {
                    this.currentGnss = {
                        lat: position.coords.latitude,
                        lon: position.coords.longitude,
                        alt: position.coords.altitude || 0,
                        accuracy: position.coords.accuracy,
                        speed: position.coords.speed || 0,
                        heading: (position.coords.heading || 0) * (Math.PI / 180)
                    };
                },
                (error) => {
                    console.warn('[Edge Engine] GNSS Error:', error.message);
                    this.currentGnss = null;
                },
                { enableHighAccuracy: true, maximumAge: 0 }
            );
        }

        // Run Edge Inference Loop at ~10 Hz
        this.loopInterval = setInterval(() => this.runInferenceLoop(), 100);
    }

    stopCapture() {
        this.isCapturing = false;
        if (this.handleMotion) {
            window.removeEventListener('devicemotion', this.handleMotion);
        }
        if (this.watchId !== undefined) {
            navigator.geolocation.clearWatch(this.watchId);
        }
        if (this.loopInterval) {
            clearInterval(this.loopInterval);
        }
        console.log('[Edge Engine] Stopped.');
    }

    async runInferenceLoop() {
        const currentTime = performance.now() / 1000.0;
        const dt = currentTime - this.lastTime;
        this.lastTime = currentTime;

        // 1. Buffer the IMU data
        this.sensorBuffer.push([...this.currentAccel, ...this.currentGyro]);
        if (this.sensorBuffer.length > this.windowSize) {
            this.sensorBuffer.shift(); // Keep only last 200 samples
        }

        // 2. GNSS Logic (Initialization & Ground Truth update)
        let hasGoodGNSS = false;
        if (this.currentGnss && this.currentGnss.accuracy < 20.0) {
            hasGoodGNSS = true;
        }

        if (hasGoodGNSS && !this.initialized) {
            // First good GNSS fix initializes the map origin
            this.refLat = this.currentGnss.lat;
            this.refLon = this.currentGnss.lon;
            this.ekf.x[0][0] = 0;
            this.ekf.x[1][0] = 0;
            this.ekf.x[8][0] = this.currentGnss.heading;
            this.initialized = true;
            console.log("[Edge Engine] Initialized EKF Map Origin via GNSS", this.refLat, this.refLon);
        }

        if (!this.initialized) {
            // Can't navigate until we know where we are
            this.updateUI({ status: 'waiting_for_gnss' });
            return;
        }

        // 3. EKF Predict (High frequency, 10Hz)
        this.ekf.dt = dt;
        this.ekf.predict(this.currentAccel, this.currentGyro, null, true); // apply NHC

        // 4. Simple ZUPT (Zero Velocity Update) detection
        // If variance of accel/gyro is very low, we are stationary
        const recentAccel = this.sensorBuffer.slice(-10).map(s => Math.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]));
        const variance = recentAccel.length > 0 ? 
            recentAccel.reduce((acc, val) => acc + Math.pow(val - recentAccel.reduce((a,b)=>a+b)/recentAccel.length, 2), 0) / recentAccel.length : 100;
        
        this.zuptActive = variance < 0.05;
        if (this.zuptActive) {
            this.ekf.updateZupt(0.01);
        }

        // 5. Run Edge AI Model if buffer is full
        let aiVelocity = null;
        if (this.sensorBuffer.length === this.windowSize && this.session && !this.zuptActive) {
            try {
                // Flatten the 2D buffer into a 1D Float32Array
                const flatData = new Float32Array(this.windowSize * 6);
                for (let i = 0; i < this.windowSize; i++) {
                    for (let j = 0; j < 6; j++) {
                        flatData[i * 6 + j] = this.sensorBuffer[i][j];
                    }
                }
                
                const tensor = new ort.Tensor('float32', flatData, [1, this.windowSize, 6]);
                const inputName = this.session.inputNames[0];
                const feeds = {};
                feeds[inputName] = tensor;
                
                const results = await this.session.run(feeds);
                const outputName = this.session.outputNames[0];
                const v = results[outputName].data; // Model output: [v_north, v_east]
                
                // Convert to ENU velocity: [v_east, v_north]
                aiVelocity = [v[1], v[0]];
                
                // If GNSS is denied, feed AI velocity into EKF
                if (!hasGoodGNSS) {
                    this.ekf._updateAiVelocity(aiVelocity);
                }
            } catch (e) {
                console.error("[Edge Engine] Inference Error:", e);
            }
        }

        // 6. Navigation Update (GNSS Update)
        const metersPerDegLat = 111320.0;
        const metersPerDegLon = 111320.0 * Math.cos(this.refLat * Math.PI / 180);

        if (hasGoodGNSS) {
            this.gnssAvailable = true;
            this.drDuration = 0;
            this.drDistanceTraveled = 0;
            
            const de = (this.currentGnss.lon - this.refLon) * metersPerDegLon;
            const dn = (this.currentGnss.lat - this.refLat) * metersPerDegLat;
            
            const vE = this.currentGnss.speed * Math.sin(this.currentGnss.heading);
            const vN = this.currentGnss.speed * Math.cos(this.currentGnss.heading);
            
            this.ekf.updateGnss([de, dn], [vE, vN]);
            this.currentGnss = null; // Consume
        } else {
            this.gnssAvailable = false;
            this.drDuration = (this.drDuration || 0) + dt;
            this.ekf.setGnssDenied();
            
            const vel = this.ekf.getVelocity();
            const stepDist = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1]) * dt;
            this.drDistanceTraveled = (this.drDistanceTraveled || 0) + stepDist;
        }

        // 7. Map Matching
        let pos = this.ekf.getPosition();
        let heading = this.ekf.getHeading();
        let match = this.mapMatcher.match(pos, heading);
        
        // Optionally feedback map matching if highly confident (soft map matching)
        if (match.confidence > 0.8 && this.ekf.mode === 'dr') {
            // Apply a small snap to EKF state to bound drift
            this.ekf.x[0][0] = 0.9 * this.ekf.x[0][0] + 0.1 * match.snapped_position[0];
            this.ekf.x[1][0] = 0.9 * this.ekf.x[1][0] + 0.1 * match.snapped_position[1];
        }

        // 8. Extract Final State for UI
        pos = this.ekf.getPosition();
        const vel = this.ekf.getVelocity();
        const estLat = this.refLat + (pos[1] / metersPerDegLat);
        const estLon = this.refLon + (pos[0] / metersPerDegLon);
        const speed = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1]);
        
        // Compute physical uncertainty
        const posUncertainty = this.ekf.P[0][0] + this.ekf.P[1][1]; // trace of pos covariance
        const driftPct = (this.ekf.mode === 'dr' && (this.drDistanceTraveled || 0) > 2.0)
            ? (Math.sqrt(posUncertainty) / this.drDistanceTraveled) * 100.0
            : 0.0;
        const confidenceScore = this.ekf.mode === 'dr'
            ? Math.max(0.1, Math.exp(-(this.drDuration || 0) / 60.0))
            : 1.0;

        this.updateUI({
            status: 'active',
            nav_mode: this.ekf.mode,
            gnss_available: this.gnssAvailable,
            estimated_lat: estLat,
            estimated_lon: estLon,
            speed: speed,
            heading: this.ekf.getHeading(),
            zupt_active: this.zuptActive,
            position_error: Math.sqrt(posUncertainty),
            dr_drift_percent: driftPct,
            confidence: confidenceScore
        });
    }

    updateUI(data) {
        // Hook into the existing app.js functions safely
        if (data.status === 'waiting_for_gnss') {
            if (typeof updateNavMode === 'function') {
                updateNavMode('reacq');
                updateGnssStatus(false);
            }
            return;
        }

        if (data.status === 'active' && typeof updateNavMode === 'function') {
            updateNavMode(data.nav_mode);
            updateGnssStatus(data.gnss_available);
            updateSpeed(data.speed * 3.6, data.heading);
            updatePositionError(data.position_error);
            updateDrift(data.dr_drift_percent);
            updateConfidence(data.confidence);

            if (typeof state !== 'undefined' && state.map) {
                const estLat = data.estimated_lat;
                const estLon = data.estimated_lon;
                
                state.estimatedCoords.push([estLat, estLon]);
                state.estimatedLine.setLatLngs(state.estimatedCoords);
                state.vehicleMarker.setLatLng([estLat, estLon]);
                state.map.panTo([estLat, estLon], { animate: true, duration: 0.1 });
                
                const markerEl = state.vehicleMarker.getElement();
                if (markerEl) {
                    const wrapper = markerEl.querySelector('.vehicle-marker') || markerEl;
                    if (data.nav_mode === 'dr') {
                        wrapper.classList.add('dr-active');
                    } else {
                        wrapper.classList.remove('dr-active');
                    }
                }
            }
        }
    }
}

// Global instance
window.offlineEngine = new OfflineEngine();
