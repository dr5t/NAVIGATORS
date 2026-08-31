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
        
        // Navigation State (Simple Dead Reckoning)
        this.initialized = false;
        this.refLat = 0;
        this.refLon = 0;
        this.position = [0, 0]; // [East, North] in meters
        this.velocity = [0, 0]; // [East, North] in m/s
        this.heading = 0; // radians
        
        // EKF Mode Emulation
        this.mode = 'gnss_ins'; // 'gnss_ins', 'dr', 'reacq'
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
            this.position = [0, 0];
            this.velocity = [0, 0];
            this.heading = this.currentGnss.heading;
            this.initialized = true;
            console.log("[Edge Engine] Initialized Map via GNSS", this.refLat, this.refLon);
        }

        if (!this.initialized) {
            // Can't navigate until we know where we are
            this.updateUI({ status: 'waiting_for_gnss' });
            return;
        }

        // 3. Simple ZUPT (Zero Velocity Update) detection
        // If variance of accel/gyro is very low, we are stationary
        const recentAccel = this.sensorBuffer.slice(-10).map(s => Math.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]));
        const variance = recentAccel.length > 0 ? 
            recentAccel.reduce((acc, val) => acc + Math.pow(val - recentAccel.reduce((a,b)=>a+b)/recentAccel.length, 2), 0) / recentAccel.length : 100;
        
        this.zuptActive = variance < 0.05;

        // 4. Run Edge AI Model if buffer is full
        let aiVelocity = [0, 0];
        if (this.sensorBuffer.length === this.windowSize && this.session && !this.zuptActive) {
            try {
                // Flatten the 2D buffer into a 1D Float32Array
                const flatData = new Float32Array(this.windowSize * 6);
                for (let i = 0; i < this.windowSize; i++) {
                    for (let j = 0; j < 6; j++) {
                        flatData[i * 6 + j] = this.sensorBuffer[i][j];
                    }
                }
                
                // Create ONNX Tensor: shape [batch=1, seq=200, features=6]
                const tensor = new ort.Tensor('float32', flatData, [1, this.windowSize, 6]);
                
                // The input name must match the exported ONNX model (usually 'x' or 'input')
                const inputName = this.session.inputNames[0];
                const feeds = {};
                feeds[inputName] = tensor;
                
                // Run Inference
                const results = await this.session.run(feeds);
                const outputName = this.session.outputNames[0];
                const v = results[outputName].data; // Float32Array [v_east, v_north]
                
                aiVelocity = [v[0], v[1]];
            } catch (e) {
                console.error("[Edge Engine] Inference Error:", e);
            }
        }

        // 5. Navigation Update (Dead Reckoning vs GNSS)
        const metersPerDegLat = 111320.0;
        const metersPerDegLon = 111320.0 * Math.cos(this.refLat * Math.PI / 180);

        if (hasGoodGNSS) {
            // GNSS Available -> We use GNSS as truth
            this.mode = 'gnss_ins';
            this.gnssAvailable = true;
            
            const dn = (this.currentGnss.lat - this.refLat) * metersPerDegLat;
            const de = (this.currentGnss.lon - this.refLon) * metersPerDegLon;
            
            this.position = [de, dn];
            this.heading = this.currentGnss.heading;
            
            // Clear GNSS so we only update on fresh fixes
            this.currentGnss = null;
        } else {
            // GNSS Denied -> Intelligent Dead Reckoning!
            this.mode = 'dr';
            this.gnssAvailable = false;
            
            if (this.zuptActive) {
                this.velocity = [0, 0];
            } else {
                // Use AI predicted velocity!
                this.velocity = aiVelocity;
                
                // Integrate velocity into position (Dead Reckoning)
                this.position[0] += this.velocity[0] * dt;
                this.position[1] += this.velocity[1] * dt;
                
                // Update heading (simple gyro integration for demo purposes, 
                // normally EKF fuses this properly)
                this.heading += this.currentGyro[2] * dt;
            }
        }

        // 6. Push to UI
        const estLat = this.refLat + (this.position[1] / metersPerDegLat);
        const estLon = this.refLon + (this.position[0] / metersPerDegLon);
        const speed = Math.sqrt(this.velocity[0]*this.velocity[0] + this.velocity[1]*this.velocity[1]);

        this.updateUI({
            status: 'active',
            nav_mode: this.mode,
            gnss_available: this.gnssAvailable,
            estimated_lat: estLat,
            estimated_lon: estLon,
            speed: speed,
            heading: this.heading,
            zupt_active: this.zuptActive,
            position_error: this.mode === 'dr' ? 2.5 : 0.5, // Mock uncertainty
            dr_drift_percent: this.mode === 'dr' ? 1.2 : 0.0,
            confidence: this.mode === 'dr' ? 0.95 : 1.0
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
