/**
 * Navigators IDR — 100% Offline Edge Inference Engine
 * Captures mobile IMU & GNSS data and runs PyTorch ONNX model via WebAssembly.
 * Performs Dead Reckoning locally on the mobile processor without any network connection.
 */

class GnssStateMachine {
    constructor() {
        this.state = 'NORMAL';
        this.timeInCurrentState = 0;
    }

    update(gnssData, dt) {
        // gnssData might be null if disconnected
        let acc = gnssData ? gnssData.accuracy : 999.0;
        if (gnssData === null) acc = 999.0;

        this.timeInCurrentState += dt;

        switch (this.state) {
            case 'NORMAL':
                if (acc > 20.0 && this.timeInCurrentState > 1.0) {
                    this.state = 'DEGRADED';
                    this.timeInCurrentState = 0;
                } else if (acc <= 20.0) {
                    this.timeInCurrentState = 0; // reset timer
                }
                break;
            case 'DEGRADED':
                if (acc > 50.0 && this.timeInCurrentState > 3.0) {
                    this.state = 'DEAD_RECKONING';
                    this.timeInCurrentState = 0;
                } else if (acc <= 20.0 && this.timeInCurrentState > 2.0) {
                    this.state = 'NORMAL';
                    this.timeInCurrentState = 0;
                } else if (acc > 20.0 && acc <= 50.0) {
                    // stays in degraded
                } else if (acc <= 20.0) {
                    // slowly transitioning back to normal
                } else {
                    // > 50 but hasn't been 3 seconds yet
                }
                break;
            case 'DEAD_RECKONING':
                if (acc < 50.0) {
                    this.state = 'REACQUISITION';
                    this.timeInCurrentState = 0;
                }
                break;
            case 'REACQUISITION':
                if (acc < 20.0 && this.timeInCurrentState > 3.0) {
                    this.state = 'NORMAL';
                    this.timeInCurrentState = 0;
                } else if (acc > 50.0) {
                    this.state = 'DEAD_RECKONING';
                    this.timeInCurrentState = 0;
                }
                break;
        }

        return this.state;
    }
}

class ZUPTDetector {
    constructor() {
        this.accelThreshold = 0.05;
        this.gyroThreshold = 0.01;
        this.windowSize = 10;
        this.gravity = 9.81;
        this.minStationarySamples = 3;

        this.accelBuffer = [];
        this.gyroBuffer = [];

        this.isStationary = false;
        this.consecutiveStationary = 0;
    }

    update(accel, gyro, speed) {
        this.accelBuffer.push(accel);
        this.gyroBuffer.push(gyro);

        if (this.accelBuffer.length > this.windowSize) {
            this.accelBuffer.shift();
            this.gyroBuffer.shift();
        }

        if (this.accelBuffer.length < this.windowSize) {
            this.isStationary = false;
            return false;
        }

        // 1. Accelerometer variance and mean magnitude
        let accelMagnitudes = this.accelBuffer.map(a => Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]));
        let accelMean = accelMagnitudes.reduce((a, b) => a + b) / accelMagnitudes.length;
        let accelVar = accelMagnitudes.reduce((acc, val) => acc + Math.pow(val - accelMean, 2), 0) / accelMagnitudes.length;

        // 2. Gravity consistency
        let gravityConsistent = Math.abs(accelMean - this.gravity) < 1.0;

        // 3. Gyroscope variance and max rate
        let gyroMags = this.gyroBuffer.map(g => Math.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]));
        let gyroMeanMag = gyroMags.reduce((a, b) => a + b) / gyroMags.length;
        let gyroVar = gyroMags.reduce((acc, val) => acc + Math.pow(val - gyroMeanMag, 2), 0) / gyroMags.length;
        let maxGyroRate = Math.max(...gyroMags);

        // 4. GNSS Speed check
        let speedOk = true;
        if (speed !== null && speed !== undefined) {
            speedOk = speed < 0.5;
        }

        let conditionMet = (
            accelVar < this.accelThreshold &&
            gyroVar < this.gyroThreshold &&
            maxGyroRate < 0.15 &&
            gravityConsistent &&
            speedOk
        );

        if (conditionMet) {
            this.consecutiveStationary++;
        } else {
            this.consecutiveStationary = 0;
        }

        this.isStationary = this.consecutiveStationary >= this.minStationarySamples;
        return this.isStationary;
    }

    reset() {
        this.accelBuffer = [];
        this.gyroBuffer = [];
        this.isStationary = false;
        this.consecutiveStationary = 0;
    }
}

class OfflineEngine {
    constructor() {
        this.isCapturing = false;
        this.navigationMode = 'vehicle';
        this.stepLength = 0.7;

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
        this.profiler = new DeviceProfiler();
        this.modelContract = null;
        this.imuFilter = new CausalIMUFilter();
        this.sensorTimes = [];

        // Navigation State
        this.initialized = false;
        this.refLat = 0;
        this.refLon = 0;

        // Edge EKF Engine
        this.ekf = new ExtendedKalmanFilter();

        // Coordinate Aligner
        this.aligner = new PhoneVehicleAligner();

        // Real downloaded roads share one projection with the map and EKF.
        this.roadNetwork = new RoadNetwork();
        this.localMap = null;
        this.gpsOutage = false;
        this.gnssGeneration = 0;
        this.inferenceRunning = false;
        this.lastGnssTime = -Infinity;
        this.lastMotionTime = -Infinity;
        this.mapMatcher = new GeometricMapMatcher(this.roadNetwork, 30.0);

        // GNSS State Machine
        this.gnssStateMachine = new GnssStateMachine();
        this.zuptDetector = new ZUPTDetector();
        this.gnssAvailable = false;
        this.zuptActive = false;

        // Loop controls
        this.lastTime = performance.now() / 1000.0;
        this.loopInterval = null;

        // Ensure ONNX Runtime is available
        if (typeof ort !== 'undefined') {
            ort.env.wasm.wasmPaths = new URL('./vendor/onnxruntime/', document.baseURI).href;
            ort.env.wasm.numThreads = 1;
        }
    }

    setLocalMap(localMap) {
        const walkingPosition = this.walker?.position && this.localMap?.toLatLon(this.walker.position);
        this.localMap = localMap;
        this.roadNetwork.segments = [];
        for (const road of localMap.data.roads) {
            this.roadNetwork.addRoad(road.points, road.id, road.name, road.speed_limit, road.one_way);
        }
        this.roadNetwork.buildSpatialHash();
        this.refLat = localMap.data.origin.lat;
        this.refLon = localMap.data.origin.lon;
        if (walkingPosition) this.walker.position = localMap.toENU(...walkingPosition);
    }

    setGpsOutage(enabled) {
        if (enabled && (!this.initialized || !this.isCapturing)) return false;
        this.gpsOutage = enabled;
        this.gnssGeneration++;
        this.currentGnss = null;
        this.lastGnssTime = -Infinity;
        if (enabled) {
            this.walker?.loseGps();
            if (this.watchId !== undefined) navigator.geolocation.clearWatch(this.watchId);
            this.watchId = undefined;
            this.gnssStateMachine.state = 'DEAD_RECKONING';
            this.gnssStateMachine.timeInCurrentState = 0;
            this.ekf.setGnssDenied();
            this.gnssAvailable = false;
        } else if (this.isCapturing) {
            this.startGnssWatch();
        }
        return true;
    }

    startGnssWatch() {
        if (!('geolocation' in navigator) || this.gpsOutage) return;
        const generation = ++this.gnssGeneration;
        this.watchId = navigator.geolocation.watchPosition(position => {
            if (this.gpsOutage || !this.isCapturing || generation !== this.gnssGeneration) return;
            if (Date.now() - position.timestamp > 3000) return;
            this.lastGnssTime = performance.now() / 1000;
            this.gpsError = '';
            this.currentGnss = {
                timestamp: position.timestamp,
                lat: position.coords.latitude, lon: position.coords.longitude,
                accuracy: position.coords.accuracy, speed: position.coords.speed || 0,
                heading: (position.coords.heading || 0) * Math.PI / 180,
                speedKnown: Number.isFinite(position.coords.speed),
                headingKnown: Number.isFinite(position.coords.heading),
            };
        }, error => {
            if (generation !== this.gnssGeneration) return;
            console.warn('[Edge Engine] GNSS:', error.message);
            this.gpsError = error.message;
            this.currentGnss = null;
            this.lastGnssTime = -Infinity;
        }, { enableHighAccuracy: true, maximumAge: 0, timeout: 3000 });
    }

    async initModel() {
        if (this.session) return true;
        if (this.modelLoading) return false;

        this.modelLoading = true;
        const startup = performance.now();
        try {
            const contractResponse = await fetch('./model.contract.json');
            if (!contractResponse.ok) throw new Error('Model contract missing: export the trained model first');
            this.modelContract = await contractResponse.json();
            const contract = this.modelContract;
            if (contract.mean?.length !== 6 || contract.std?.length !== 6 ||
                !contract.mean.every(Number.isFinite) || !contract.std.every(v => Number.isFinite(v) && v > 0) ||
                JSON.stringify(contract.output_order) !== '["east","north"]') throw new Error('Invalid model input/output contract');
            this.windowSize = contract.window_size;
            if (window.setLoadingState) window.setLoadingState(40, "DOWNLOADING AI VELOCITY MODEL");

            const response = await fetch('./model.onnx');
            if (!response.ok) throw new Error('Failed to fetch model.onnx');
            
            const contentLength = response.headers.get('content-length');
            // Provide a fallback total if content-length is missing
            const totalBytes = contentLength ? parseInt(contentLength, 10) : 5000000;
            let loadedBytes = 0;
            
            const reader = response.body.getReader();
            const chunks = [];
            while(true) {
                const {done, value} = await reader.read();
                if (done) break;
                chunks.push(value);
                loadedBytes += value.length;
                if (window.setLoadingState) {
                    const progress = 40 + (loadedBytes / totalBytes) * 20;
                    window.setLoadingState(Math.min(60, progress), "DOWNLOADING AI VELOCITY MODEL");
                }
            }
            
            const modelBytes = new Uint8Array(loadedBytes);
            let position = 0;
            for(let chunk of chunks) {
                modelBytes.set(chunk, position);
                position += chunk.length;
            }

            const external = contract.external_data ? new Uint8Array(await (await fetch('./model.onnx.data')).arrayBuffer()) : null;
            this.profiler.modelBytes = modelBytes.byteLength + (external?.byteLength || 0);
            
            if (window.setLoadingState) window.setLoadingState(60, "COMPILING WEBASSEMBLY ENGINE");
            console.log("[Edge AI] Loading ONNX Model into WebAssembly...");
            // Load the model exported to the simulator folder
            // Small delay to allow the UI to render the 60% state before blocking synchronous compilation
            await new Promise(r => setTimeout(r, 50));
            this.session = await ort.InferenceSession.create(modelBytes, {
                executionProviders: ['wasm'],
                externalData: external ? [{ path: 'model.onnx.data', data: external }] : [],
            });
            if (window.setLoadingState) window.setLoadingState(80, "INITIALIZING NAVIGATION ENGINE");
            this.profiler.startupMs = performance.now() - startup;
            document.getElementById('modelValidation').textContent = contract.preprocessing === 'legacy-unverified'
                ? 'Diagnostic model · real-data validation unavailable' : 'Causal model loaded · validate accuracy on held-out trips';
            console.log("[Edge AI] Model loaded successfully!");
            this.modelLoading = false;
            return true;
        } catch (e) {
            console.error("[Edge AI] Failed to load ONNX model:", e);
            if (window.setLoadingState) window.setLoadingState(0, "ERROR: FAILED TO LOAD MODEL");
            this.modelLoading = false;
            this.lastError = 'Failed to load local model: ' + e.message;
            return false;
        }
    }

    async requestPermissionsAndStart() {
        if (window.setLoadingState) window.setLoadingState(0, "INITIALIZING SENSORS");
        this.lastError = '';
        if (window.isSecureContext === false) { this.lastError = 'Open the app using HTTPS or localhost to allow phone location and motion access.'; if (window.setLoadingState) window.setLoadingState(0, "ERROR: INSECURE CONTEXT"); return false; }
        if (!this.localMap) { this.lastError = 'The local road database is not ready. Reload and try again.'; if (window.setLoadingState) window.setLoadingState(0, "ERROR: MAP NOT READY"); return false; }
        if (this.navigationMode === 'walking' && (!Number.isFinite(this.stepLength) || this.stepLength < 0.3 || this.stepLength > 1.2)) {
            this.lastError = 'Set your step length between 0.3 and 1.2 meters.';
            if (window.setLoadingState) window.setLoadingState(0, "ERROR: INVALID STEP LENGTH");
            return false;
        }
        // Request both permissions in the original button gesture on iOS.
        try {
            const requests = [];
            if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') requests.push(DeviceMotionEvent.requestPermission());
            if (this.navigationMode === 'walking' && typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') requests.push(DeviceOrientationEvent.requestPermission(true));
            if ((await Promise.all(requests)).some(permission => permission !== 'granted')) {
                    this.lastError = 'Motion permission denied. Allow sensor access in browser settings and retry.';
                    if (window.setLoadingState) window.setLoadingState(0, "ERROR: PERMISSION DENIED");
                    return false;
            }
        } catch (e) {
                console.error('Error requesting sensor permission:', e);
                this.lastError = 'Motion access failed. Open this app over HTTPS or localhost and retry.';
                if (window.setLoadingState) window.setLoadingState(0, "ERROR: PERMISSION FAILED");
                return false;
        }

        if (window.setLoadingState) window.setLoadingState(20, "CALIBRATING IMU");
        await new Promise(r => setTimeout(r, 200));

        if (this.navigationMode !== 'walking' && !await this.initModel()) return false;

        if (window.setLoadingState) window.setLoadingState(95, "LOADING OFFLINE MAP");
        await new Promise(r => setTimeout(r, 100));

        this.startCapture();
        if (window.setLoadingState) window.setLoadingState(100, "NAVIGATION READY");
        return true;
    }

    startCapture() {
        if (this.isCapturing) return;
        console.log('[Edge Engine] Starting sensor capture & local processing...');
        this.isCapturing = true;
        this.initialized = false;
        this.currentGnss = null;
        this.lastGnssTime = -Infinity;
        this.lastMotionTime = -Infinity;
        this.gpsOutage = false;
        this.gpsError = '';
        this.ekf = new ExtendedKalmanFilter();
        this.gnssStateMachine = new GnssStateMachine();
        this.zuptDetector.reset();
        this.walker = this.navigationMode === 'walking' ? new PedestrianTracker(this.stepLength) : null;
        document.getElementById('btnGpsOutage').textContent = 'Simulate GNSS outage';

        // Reset buffers and timers
        this.sensorBuffer = [];
        this.sensorTimes = [];
        this.imuFilter = new CausalIMUFilter();
        this.profiler.reset();
        this.lastTime = performance.now() / 1000.0;

        // IMU Listeners
        this.handleMotion = (event) => {
            if (this.walker) {
                this.walker.motion(event.accelerationIncludingGravity, performance.now() / 1000);
                this.lastMotionTime = this.walker.lastMotion;
                return;
            }
            if (!event.accelerationIncludingGravity || !event.rotationRate) return;
            const motionTime = performance.now() / 1000;
            const motionDt = Number.isFinite(this.lastMotionTime) ? motionTime - this.lastMotionTime : 0;
            this.lastMotionTime = motionTime;
            const accel = event.accelerationIncludingGravity;
            if (accel) {
                this.currentAccel = [accel.x || 0, accel.y || 0, accel.z || 0];
            }
            const gyro = event.rotationRate;
            if (gyro) {
                const toRad = Math.PI / 180;
                this.currentGyro = [
                    (gyro.beta || 0) * toRad,
                    (gyro.gamma || 0) * toRad,
                    (gyro.alpha || 0) * toRad
                ];
            }
            if (this.aligner.isAligned) {
                const aligned = [...this.aligner.rotate(this.currentAccel), ...this.aligner.rotate(this.currentGyro)];
                this.sensorBuffer.push(this.imuFilter.step(aligned, motionDt));
                this.sensorTimes.push(motionTime);
                if (this.sensorBuffer.length > this.windowSize) { this.sensorBuffer.shift(); this.sensorTimes.shift(); }
                if (this.sensorTimes.length > 1) {
                    this.observedImuRate = (this.sensorTimes.length - 1) / (motionTime - this.sensorTimes[0]);
                }
            }
        };
        window.addEventListener('devicemotion', this.handleMotion);
        if (this.walker) {
            this.handleOrientation = event => this.walker.orientation(event, performance.now() / 1000);
            window.addEventListener('deviceorientationabsolute', this.handleOrientation);
            window.addEventListener('deviceorientation', this.handleOrientation);
            document.getElementById('modelValidation').textContent = 'Walking prototype · step length + absolute compass · no trained walking AI';
        }

        this.startGnssWatch();

        // Run Edge Inference Loop at ~10 Hz
        this.aligner.reset();
        this.loopInterval = setInterval(() => this.runInferenceLoop(), 100);
    }

    stopCapture() {
        this.isCapturing = false;
        this.gnssGeneration++;
        if (this.handleMotion) {
            window.removeEventListener('devicemotion', this.handleMotion);
        }
        if (this.handleOrientation) {
            window.removeEventListener('deviceorientationabsolute', this.handleOrientation);
            window.removeEventListener('deviceorientation', this.handleOrientation);
        }
        if (this.watchId !== undefined) {
            navigator.geolocation.clearWatch(this.watchId);
            this.watchId = undefined;
        }
        if (this.loopInterval) {
            clearInterval(this.loopInterval);
        }
        console.log('[Edge Engine] Stopped.');
    }

    async runInferenceLoop() {
        if (!this.isCapturing || this.inferenceRunning) return;
        this.inferenceRunning = true;
        const loopStart = performance.now();
        try {
            const active = await this.processInferenceStep();
            if (active) {
                this.profiler.record('total_loop', performance.now() - loopStart);
                this.profiler.loops++;
            }
        } finally {
            this.inferenceRunning = false;
        }
    }

    async processInferenceStep() {
        if (this.walker) return this.processWalkingStep();
        const currentTime = performance.now() / 1000.0;
        const dt = currentTime - this.lastTime;
        this.lastTime = currentTime;

        if (currentTime - this.lastMotionTime > 1) {
            document.getElementById('edgeStatus').textContent = 'Waiting for IMU samples';
            this.updateUI({ status: 'waiting_for_imu' });
            return;
        }
        // Snapshot fixes across asynchronous inference; never reuse stale GPS during an outage.
        const gnssGeneration = this.gnssGeneration;
        const gnss = !this.gpsOutage && currentTime - this.lastGnssTime <= 3 ? this.currentGnss : null;
        const currentGnssState = this.gnssStateMachine.update(gnss, dt);

        let hasGoodGNSS = false;
        let speed = 0;

        // In NORMAL or REACQUISITION, we strictly use GNSS
        // In DEGRADED, we use it but it might be jumping
        // In DEAD_RECKONING, we completely ignore it.
        if ((currentGnssState === 'NORMAL' || currentGnssState === 'DEGRADED' || currentGnssState === 'REACQUISITION') && gnss && gnss.accuracy <= 2000) {
            hasGoodGNSS = true;
            speed = gnss.speed;
        }


        // 2. Alignment Phase
        const isAligned = this.aligner.feed(this.currentAccel, speed, currentTime);
        if (!isAligned) {
            // Documenting: update UI here if necessary to say "Calibrating..."
            document.getElementById('edgeStatus').textContent = "Calibrating Orientation...";
            document.getElementById('edgeStatus').style.color = "var(--accent-cyan)";
            this.updateUI({ status: 'calibrating' });
            return; // Wait until aligned
        }

        // 3. Buffer the Aligned IMU data
        const alignedAccel = this.aligner.rotate(this.currentAccel);
        const alignedGyro = this.aligner.rotate(this.currentGyro);

        if (hasGoodGNSS && gnss.accuracy <= 2000 && !this.initialized) {
            // The fix initializes position within the downloaded map's reference frame.
            const initial = this.localMap.toENU(gnss.lat, gnss.lon);
            this.ekf.x[0][0] = initial[0];
            this.ekf.x[1][0] = initial[1];
            this.ekf.x[8][0] = gnss.heading;
            this.initialized = true;
            document.getElementById('btnGpsOutage').disabled = false;
            console.log("[Edge Engine] Initialized EKF Map Origin via GNSS", this.refLat, this.refLon);
        }

        if (!this.initialized) {
            // Can't navigate until we know where we are
            document.getElementById('edgeStatus').textContent = 'Waiting for initial GPS fix';
            this.updateUI({ status: 'waiting_for_gnss' });
            return;
        }

        // 3. EKF Predict (High frequency, 10Hz)
        let ekfStart = performance.now();
        let ekfMs = 0;
        this.ekf.dt = dt;
        this.ekf.predict(alignedAccel, alignedGyro, null, true); // apply NHC

        // 4. Robust ZUPT (Zero Velocity Update) detection
        // Uses accelerometer variance, gyroscope variance, gravity check, and GNSS speed
        let estSpeed = hasGoodGNSS ? speed : null;
        this.zuptActive = this.zuptDetector.update(this.currentAccel, this.currentGyro, estSpeed);

        if (this.zuptActive) {
            this.ekf.updateZupt(0.01);
        }

        ekfMs += performance.now() - ekfStart;
        // 5. Run Edge AI Model if buffer is full
        let aiVelocity = null;
        let aiError = null;
        const expectedRate = this.modelContract?.sample_rate_hz;
        const rateMatches = !expectedRate || (this.observedImuRate && Math.abs(this.observedImuRate / expectedRate - 1) <= 0.1);
        if (!rateMatches) aiError = 'IMU rate differs from trained model';
        if (this.sensorBuffer.length === this.windowSize && this.session && !this.zuptActive && rateMatches) {
            try {
                const aiStart = performance.now();
                // Flatten the 2D buffer into a 1D Float32Array
                const flatData = new Float32Array(this.windowSize * 6);
                for (let i = 0; i < this.windowSize; i++) {
                    for (let j = 0; j < 6; j++) {
                        flatData[i * 6 + j] = (this.sensorBuffer[i][j] - (this.modelContract?.mean[j] || 0)) / (this.modelContract?.std[j] || 1);
                    }
                }

                const tensor = new ort.Tensor('float32', flatData, [1, this.windowSize, 6]);
                const inputName = this.session.inputNames[0];
                const feeds = {};
                feeds[inputName] = tensor;

                const results = await this.session.run(feeds);
                this.profiler.record('tcn', performance.now() - aiStart);
                const outputName = this.session.outputNames[0];
                const v = results[outputName].data;
                if (v.length !== 2 || !Array.from(v).every(Number.isFinite)) throw new Error('Invalid AI velocity');
                aiVelocity = [v[0], v[1]]; // Training targets and EKF use East, North.

                if (!this.isCapturing) return;
                if (this.gpsOutage || gnssGeneration !== this.gnssGeneration) hasGoodGNSS = false;
                // If GNSS is denied, feed AI velocity into EKF
                if (!hasGoodGNSS) {
                    ekfStart = performance.now();
                    this.ekf._updateAiVelocity(aiVelocity);
                    ekfMs += performance.now() - ekfStart;
                }

                // Edge AI Status indicator
                const statusEl = document.getElementById('edgeStatus');
                if (statusEl) {
                    if (currentGnssState === 'DEAD_RECKONING') {
                        statusEl.textContent = `DEAD RECKONING (AI ACTIVE)`;
                        statusEl.style.color = "#ff5252";
                    } else if (currentGnssState === 'NORMAL') {
                        statusEl.textContent = `NORMAL (GNSS + AI)`;
                        statusEl.style.color = "#00e676";
                    } else if (currentGnssState === 'DEGRADED') {
                        statusEl.textContent = `DEGRADED (Fusing)`;
                        statusEl.style.color = "#ffeb3b";
                    } else if (currentGnssState === 'REACQUISITION') {
                        statusEl.textContent = `REACQUISITION (Stabilizing)`;
                        statusEl.style.color = "#ffeb3b";
                    }
                }
            } catch (e) {
                aiError = e.message;
                console.error("[Edge Engine] Inference Error:", e);
            }
        }

        ekfStart = performance.now();
        // 6. Navigation Update (GNSS Update)
        const metersPerDegLat = this.localMap.metersPerDegree;
        const metersPerDegLon = this.localMap.lonScale;

        if (this.gpsOutage || gnssGeneration !== this.gnssGeneration) hasGoodGNSS = false;
        if (hasGoodGNSS) {
            this.gnssAvailable = true;
            this.drDuration = 0;
            this.drDistanceTraveled = 0;

            const de = (gnss.lon - this.refLon) * metersPerDegLon;
            const dn = (gnss.lat - this.refLat) * metersPerDegLat;

            const vE = gnss.speed * Math.sin(gnss.heading);
            const vN = gnss.speed * Math.cos(gnss.heading);

            if (this.lastFusedGnss !== gnss) {
                this.ekf.updateGnss([de, dn], [vE, vN]);
                this.lastFusedGnss = gnss;
            }
        } else {
            this.gnssAvailable = false;
            this.drDuration = (this.drDuration || 0) + dt;
            this.ekf.setGnssDenied();

            const vel = this.ekf.getVelocity();
            const stepDist = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1]) * dt;
            this.drDistanceTraveled = (this.drDistanceTraveled || 0) + stepDist;
        }

        ekfMs += performance.now() - ekfStart;
        this.profiler.record('ekf', ekfMs);
        // 7. Map Matching
        const mapStart = performance.now();
        let pos = this.ekf.getPosition();
        let heading = this.ekf.getHeading();
        let match = this.mapMatcher.match(pos, heading);

        // Optionally feedback map matching if highly confident (soft map matching)
        if (match.confidence > 0.8 && this.ekf.mode === 'dr') {
            // Apply a small snap to EKF state to bound drift
            this.ekf.x[0][0] = 0.9 * this.ekf.x[0][0] + 0.1 * match.snapped_position[0];
            this.ekf.x[1][0] = 0.9 * this.ekf.x[1][0] + 0.1 * match.snapped_position[1];
        }

        this.profiler.record('map_matching', performance.now() - mapStart);
        // 8. Extract Final State for UI
        pos = this.ekf.getPosition();
        const vel = this.ekf.getVelocity();
        const estLat = this.refLat + (pos[1] / metersPerDegLat);
        const estLon = this.refLon + (pos[0] / metersPerDegLon);
        const estimatedSpeed = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1]);

        // Compute physical uncertainty
        const posUncertainty = this.ekf.P[0][0] + this.ekf.P[1][1]; // trace of pos covariance
        const driftPct = (this.ekf.mode === 'dr' && (this.drDistanceTraveled || 0) > 2.0)
            ? (Math.sqrt(posUncertainty) / this.drDistanceTraveled) * 100.0
            : 0.0;
        const confidenceScore = this.ekf.mode === 'dr'
            ? Math.max(0.1, Math.exp(-(this.drDuration || 0) / 60.0))
            : 1.0;

        document.getElementById('mapStatus').textContent = this.localMap.contains(estLat, estLon)
            ? 'Local OSM map · navigation active' : 'Outside downloaded map · map matching unavailable';
        document.getElementById('edgeStatus').textContent = this.zuptActive
            ? 'IMU + EKF · stationary (AI paused)'
            : `IMU + EKF · ${aiVelocity ? 'AI active' : `AI buffering ${this.sensorBuffer.length}/${this.windowSize}`} · ${this.gpsOutage ? 'GPS disabled' : this.ekf.mode === 'reacq' ? 'REACQUISITION' : currentGnssState}`;
        document.getElementById('componentStatus').textContent =
            `AI: ${aiError ? 'ERROR (' + aiError + ')' : aiVelocity ? 'ACTIVE' : this.zuptActive ? 'PAUSED' : 'BUFFERING'} · EKF: ACTIVE · NHC: ${estimatedSpeed >= 0.5 ? 'ACTIVE' : 'LOW SPEED'} · ZUPT: ${this.zuptActive ? 'ACTIVE' : 'MONITORING'} · MAP: ${match.confidence > 0.8 ? 'MATCHED' : 'NO CONFIDENT MATCH'}`;
        this.updateUI({
            status: 'active',
            nav_mode: this.ekf.mode,
            gnss_available: this.gnssAvailable,
            estimated_lat: estLat,
            estimated_lon: estLon,
            speed: estimatedSpeed,
            heading: this.ekf.getHeading(),
            zupt_active: this.zuptActive,
            ai_active: Boolean(aiVelocity),
            ai_error: aiError,
            nhc_active: estimatedSpeed >= 0.5,
            map_matched: match.confidence > 0.8,
            position_error: Math.sqrt(posUncertainty),
            dr_drift_percent: driftPct,
            confidence: confidenceScore
        });
        return true;
    }

    updateUI(data) {
        if (data.status !== 'active') window.updateConsoleTelemetry?.(data);
        // Hook into the existing app.js functions safely
        if (data.status === 'waiting_for_gnss') {
            if (typeof updateNavMode === 'function') {
                updateGnssStatus(false);
                document.getElementById('gnssStatusText').textContent = 'ACQUIRING';
                const indicator = document.getElementById('navModeIndicator');
                indicator.className = 'nav-mode-indicator';
                indicator.querySelector('.mode-label').textContent = 'INITIALIZING';
            }
            return;
        }

        if (data.status === 'active' && typeof updateNavMode === 'function') {
            updateNavMode(data.nav_mode);
            updateGnssStatus(data.gnss_available);
            if (data.nav_mode === 'reacq') document.getElementById('gnssStatusText').textContent = 'REACQUIRED';
            updateSpeed(Number.isFinite(data.speed) ? data.speed * 3.6 : null, data.heading);
            document.getElementById('positionCoordinates').textContent = `${data.estimated_lat.toFixed(6)}, ${data.estimated_lon.toFixed(6)}`;
            updatePositionError(data.position_error);
            updateDrift(data.dr_drift_percent);
            updateConfidence(data.confidence);
            if (typeof window.updateConsoleTelemetry === 'function') window.updateConsoleTelemetry(data);

            if (typeof state !== 'undefined' && state.map) {
                const estLat = data.estimated_lat;
                const estLon = data.estimated_lon;

                state.estimatedCoords.push([estLat, estLon]);
                state.estimatedLine.setLatLngs(state.estimatedCoords);
                state.vehicleMarker.setLatLng([estLat, estLon]).setOpacity(1);
                state.estimatedLine.setStyle({ color: data.nav_mode === 'dr' ? '#ba5b37' : '#4c7b59', dashArray: data.nav_mode === 'dr' ? '5, 6' : null });
                if (state.followPosition) state.map.panTo([estLat, estLon], { animate: false });

                const markerEl = state.vehicleMarker.getElement();
                if (markerEl) {
                    markerEl.querySelector('.vehicle-marker-inner').style.transform = Number.isFinite(data.heading) ? `rotate(${data.heading * 180 / Math.PI}deg)` : '';
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

    processWalkingStep() {
        const now = performance.now() / 1000;
        const walker = this.walker;
        const fix = !this.gpsOutage && now - this.lastGnssTime <= 3 ? this.currentGnss : null;
        if (fix) walker.fix(this.localMap.toENU(fix.lat, fix.lon), fix, now);
        if (!fix || !walker.hasGps(now)) walker.loseGps();
        if (!walker.position) {
            this.updateUI({ status: 'waiting_for_gnss' });
            document.getElementById('edgeStatus').textContent = this.gpsError || 'Walking · waiting for a GPS fix (accuracy ≤ 50 m)';
            return false;
        }
        this.initialized = true;
        this.gnssAvailable = walker.hasGps(now);
        document.getElementById('btnGpsOutage').disabled = false;
        const [lat, lon] = this.localMap.toLatLon(walker.position);
        const sensorsReady = walker.hasHeading(now) && now - walker.lastMotion <= 1;
        const status = !this.gnssAvailable && !sensorsReady ? 'Estimate paused · hold phone screen-up, top pointing forward; check motion and compass access'
            : walker.mode === 'reacq' ? 'GPS returned · correcting the estimated position gradually'
            : this.gnssAvailable ? 'GPS position · walk with phone screen-up, top pointing forward' : 'Estimated position · steps + compass; error grows without GPS';
        document.getElementById('edgeStatus').textContent = status;
        document.getElementById('componentStatus').textContent = `${walker.steps} detected steps · ${walker.stepLength.toFixed(2)} m per step · compass ${walker.hasHeading(now) ? 'ready' : 'unavailable'} · AI/EKF/vehicle constraints off`;
        document.getElementById('mapStatus').textContent = this.localMap.contains(lat, lon)
            ? 'Walking on local map · position is not forced onto a road' : 'Outside downloaded streets · coordinates continue; download this area before the next offline walk';
        this.updateUI({ status: 'active', walking: true, session_hint: status, sensors_ready: sensorsReady,
            nav_mode: walker.mode, gnss_available: this.gnssAvailable, estimated_lat: lat, estimated_lon: lon,
            speed: walker.speed(now), heading: walker.hasHeading(now) || this.gnssAvailable ? walker.heading : null,
            position_error: this.gnssAvailable ? walker.accuracy : null, confidence: null, dr_drift_percent: null });
        return true;
    }
}

// Global instance
window.offlineEngine = new OfflineEngine();
