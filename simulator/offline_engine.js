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

class VehicleMotionDetector {
    constructor() {
        this.accelThreshold = 0.035;   // m2/s4 stationary variance threshold
        this.gyroThreshold = 0.008;    // rad2/s2 stationary gyro variance threshold
        this.maxGyroThreshold = 0.08;  // rad/s stationary max gyro rate
        this.windowSize = 15;          // 1.5s at 10Hz
        this.gravity = 9.81;

        this.accelBuffer = [];
        this.gyroBuffer = [];

        this.isStationary = false;
        this.motionState = 'UNKNOWN'; // 'STATIONARY' | 'MOVING' | 'UNKNOWN'
        this.motionReason = 'insufficient data';
        this.stationaryDwell = 0.0;
        this.movingDwell = 0.0;
        this.lastMovingTime = 0;
        this.consecutiveStationary = 0;
        this.consecutiveMoving = 0;

        this.lastMetrics = {
            accelMag: 9.81,
            dynamicAccel: 0.0,
            accelVar: 0.0,
            gyroMag: 0.0,
            maxGyro: 0.0,
            gyroVar: 0.0
        };
    }

    reset() {
        this.accelBuffer = [];
        this.gyroBuffer = [];
        this.isStationary = false;
        this.motionState = 'UNKNOWN';
        this.motionReason = 'insufficient data';
        this.stationaryDwell = 0.0;
        this.movingDwell = 0.0;
        this.consecutiveStationary = 0;
        this.consecutiveMoving = 0;
    }

    update(accel, gyro, speed, dt = 0.1, currentTime = 0) {
        this.accelBuffer.push([...accel]);
        this.gyroBuffer.push([...gyro]);

        if (this.accelBuffer.length > this.windowSize) {
            this.accelBuffer.shift();
            this.gyroBuffer.shift();
        }

        if (this.accelBuffer.length < 5) {
            this.isStationary = false;
            this.motionState = 'UNKNOWN';
            this.motionReason = 'insufficient data';
            return this.isStationary;
        }

        // 1. Accelerometer variance and mean magnitude
        const accelMags = this.accelBuffer.map(a => Math.hypot(a[0], a[1], a[2]));
        const accelMean = accelMags.reduce((a, b) => a + b, 0) / accelMags.length;
        const accelVar = accelMags.reduce((acc, val) => acc + Math.pow(val - accelMean, 2), 0) / accelMags.length;
        const latestAccelMag = accelMags[accelMags.length - 1];
        const dynamicAccel = Math.abs(latestAccelMag - this.gravity);

        // 2. Gyroscope variance and max rate
        const gyroMags = this.gyroBuffer.map(g => Math.hypot(g[0], g[1], g[2]));
        const gyroMean = gyroMags.reduce((a, b) => a + b, 0) / gyroMags.length;
        const gyroVar = gyroMags.reduce((acc, val) => acc + Math.pow(val - gyroMean, 2), 0) / gyroMags.length;
        const maxGyro = Math.max(...gyroMags);
        const latestGyroMag = gyroMags[gyroMags.length - 1];

        this.lastMetrics = {
            accelMag: latestAccelMag,
            dynamicAccel: dynamicAccel,
            accelVar: accelVar,
            gyroMag: latestGyroMag,
            maxGyro: maxGyro,
            gyroVar: gyroVar
        };

        const hasGnssSpeed = (speed !== null && speed !== undefined && Number.isFinite(speed));

        if (hasGnssSpeed) {
            // Real GNSS speed available as supporting evidence (NOT fed into AI model)
            if (speed >= 0.8) {
                // Vehicle moving (speed >= 0.8 m/s ~ 2.9 km/h)
                this.motionState = 'MOVING';
                this.motionReason = `GNSS speed (${speed.toFixed(1)} m/s)`;
                this.stationaryDwell = 0.0;
                this.movingDwell += dt;
                this.lastMovingTime = currentTime || Date.now() / 1000;
                this.consecutiveStationary = 0;
            } else if (speed < 0.4) {
                // Vehicle speed near zero: check if IMU is also quiet or has idle vibrations
                const imuStationary = (accelVar < this.accelThreshold && maxGyro < this.maxGyroThreshold && dynamicAccel < 0.35);
                this.stationaryDwell += dt;
                if (imuStationary && this.stationaryDwell >= 1.0) {
                    this.motionState = 'STATIONARY';
                    this.motionReason = `GNSS zero speed & IMU quiet (${this.stationaryDwell.toFixed(1)}s)`;
                    this.movingDwell = 0.0;
                } else if (this.stationaryDwell >= 1.5) {
                    this.motionState = 'STATIONARY';
                    this.motionReason = `GNSS zero speed (vehicle idle vibrations)`;
                    this.movingDwell = 0.0;
                }
            } else {
                // Low speed crawling (0.4 - 0.8 m/s)
                this.motionState = 'MOVING';
                this.motionReason = `GNSS crawl (${speed.toFixed(1)} m/s)`;
                this.stationaryDwell = 0.0;
                this.lastMovingTime = currentTime || Date.now() / 1000;
            }
        } else {
            // GNSS Denied / Outage / Tunnel / Offline
            // Rely purely on IMU dynamics without GNSS speed
            if (this.motionState === 'MOVING' || this.motionState === 'UNKNOWN') {
                // Check if vehicle has come to a complete stop
                const isVeryQuiet = (accelVar < 0.025 && maxGyro < 0.06 && dynamicAccel < 0.25);
                if (isVeryQuiet) {
                    this.stationaryDwell += dt;
                    if (this.stationaryDwell >= 1.5) { // 1.5s quiet dwell required to stop
                        this.motionState = 'STATIONARY';
                        this.motionReason = `IMU stationary dwell (${this.stationaryDwell.toFixed(1)}s)`;
                        this.movingDwell = 0.0;
                    } else {
                        // Still in MOVING state during deceleration/settling
                        this.motionState = 'MOVING';
                        this.motionReason = `IMU settling (${this.stationaryDwell.toFixed(1)}s)`;
                    }
                } else {
                    // Road vibrations, steering, or acceleration active
                    this.stationaryDwell = 0.0;
                    this.movingDwell += dt;
                    this.motionState = 'MOVING';
                    this.motionReason = `IMU motion (accel var: ${accelVar.toFixed(3)}, gyro: ${maxGyro.toFixed(2)})`;
                    this.lastMovingTime = currentTime || Date.now() / 1000;
                }
            } else {
                // Currently STATIONARY: check for start of vehicle movement
                const motionDetected = (accelVar > 0.040 || maxGyro > 0.10 || dynamicAccel > 0.40);
                if (motionDetected) {
                    this.consecutiveMoving++;
                    if (this.consecutiveMoving >= 3) { // 0.3s of motion triggers moving
                        this.motionState = 'MOVING';
                        this.motionReason = `IMU motion detected (accel var: ${accelVar.toFixed(3)})`;
                        this.stationaryDwell = 0.0;
                        this.consecutiveMoving = 0;
                        this.lastMovingTime = currentTime || Date.now() / 1000;
                    }
                } else {
                    this.consecutiveMoving = 0;
                    this.motionState = 'STATIONARY';
                    this.motionReason = 'IMU quiet';
                }
            }
        }

        this.isStationary = (this.motionState === 'STATIONARY');
        return this.isStationary;
    }
}
const ZUPTDetector = VehicleMotionDetector;

class OfflineEngine {
    constructor() {
        this.isCapturing = false;
        this.navigationMode = 'vehicle';
        this.stepLength = 0.7;

        // Sensor Data Buffer
        this.currentAccel = [0, 0, 0];
        this.currentGyro = [0, 0, 0];
        this.currentGnss = null;        // Buffer for ONNX model (window_size = 200, features = 6)
        this.windowSize = 200;
        this.sensorBuffer = []; // stores [ax, ay, az, gx, gy, gz] arrays

        // ONNX Runtime Session
        this.session = null;
        this.modelLoading = false;
        this.profiler = new DeviceProfiler();
        this.modelContract = null;
        this.imuFilter = new CausalIMUFilter(false); // removeGravity = false (model trained on raw IMU)
        this.sensorTimes = [];

        // Raw Android sensor event rate tracking (50-100Hz from hardware) vs 10Hz model rate
        this.rawEventTimes = [];
        this.observedRawSensorRate = 10;
        this.observedEffectiveRate = 10;
        this.lastDiagnosticLogTime = 0;
        this.lastLoggedMotionState = null;

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

        // GNSS State Machine & Vehicle Motion Detector
        this.gnssStateMachine = new GnssStateMachine();
        this.motionDetector = new VehicleMotionDetector();
        this.zuptDetector = this.motionDetector;
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
                (JSON.stringify(contract.output_order) !== '["east","north"]' && JSON.stringify(contract.output_order) !== '["north","east"]')) throw new Error('Invalid model input/output contract');
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
                    const pct = 40 + Math.min(35, Math.floor((loadedBytes / totalBytes) * 35));
                    window.setLoadingState(pct, `DOWNLOADING AI VELOCITY MODEL (${Math.round(loadedBytes/1024/1024*10)/10}MB)`);
                }
            }
            const modelBytes = new Uint8Array(loadedBytes);
            let offset = 0;
            for (const chunk of chunks) {
                modelBytes.set(chunk, offset);
                offset += chunk.length;
            }
            
            let external = null;
            if (contract.external_data) {
                if (window.setLoadingState) window.setLoadingState(75, "DOWNLOADING MODEL WEIGHTS");
                const extResponse = await fetch('./model.onnx.data');
                if (!extResponse.ok) throw new Error('Failed to fetch model.onnx.data');
                external = await extResponse.arrayBuffer();
            }

            this.profiler.modelBytes = modelBytes.byteLength + (external?.byteLength || 0);

            if (window.setLoadingState) window.setLoadingState(80, "COMPILING WEBASSEMBLY ENGINE");
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
        this.motionDetector.reset();
        this.zuptDetector = this.motionDetector;
        this.walker = this.navigationMode === 'walking' ? new PedestrianTracker(this.stepLength) : null;
        document.getElementById('btnGpsOutage').textContent = 'Simulate GNSS outage';

        // Reset buffers and timers
        this.sensorBuffer = [];
        this.sensorTimes = [];
        this.rawEventTimes = [];
        this.observedRawSensorRate = 10;
        this.observedEffectiveRate = 10;
        this.lastDiagnosticLogTime = 0;
        this.lastLoggedMotionState = null;
        this.imuFilter = new CausalIMUFilter(false);
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
            this.lastMotionTime = motionTime;

            // Track raw Android hardware sensor event rate (typically 50-100 Hz)
            this.rawEventTimes.push(motionTime);
            if (this.rawEventTimes.length > 50) this.rawEventTimes.shift();
            if (this.rawEventTimes.length > 1) {
                const dtRaw = motionTime - this.rawEventTimes[0];
                if (dtRaw > 0) this.observedRawSensorRate = (this.rawEventTimes.length - 1) / dtRaw;
            }

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
        };
        window.addEventListener('devicemotion', this.handleMotion);
        if (this.walker) {
            this.handleWalkerOrientation = event => this.walker.orientation(event, performance.now() / 1000);
            window.addEventListener('deviceorientationabsolute', this.handleWalkerOrientation);
            window.addEventListener('deviceorientation', this.handleWalkerOrientation);
            document.getElementById('modelValidation').textContent = 'Walking prototype · step length + absolute compass · no trained walking AI';
        }

        // Global compass listener for all modes
        this.handleOrientation = (event) => {
            if (event.webkitCompassHeading !== undefined) {
                // iOS
                this.currentCompassHeading = event.webkitCompassHeading * Math.PI / 180;
            } else if (event.absolute && event.alpha !== null) {
                // Android standard: alpha is CCW from East? Actually, absolute alpha is CCW from North.
                // 360 - alpha gives clockwise from North (standard navigation heading)
                this.currentCompassHeading = (360 - event.alpha) * Math.PI / 180;
            }
        };
        window.addEventListener('deviceorientationabsolute', this.handleOrientation);
        window.addEventListener('deviceorientation', this.handleOrientation);

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
        if (this.handleWalkerOrientation) {
            window.removeEventListener('deviceorientationabsolute', this.handleWalkerOrientation);
            window.removeEventListener('deviceorientation', this.handleWalkerOrientation);
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
        let dt = currentTime - this.lastTime;
        if (!Number.isFinite(dt) || dt <= 0 || dt > 1.0) dt = 0.1;
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

        // 1. Phone-to-Vehicle Alignment (runs in background; does NOT block buffering or navigation)
        this.aligner.feed(this.currentAccel, hasGoodGNSS ? speed : -1, currentTime);

        // 2. Rotate and Filter IMU into vehicle frame
        const alignedAccel = this.aligner.rotate(this.currentAccel);
        const alignedGyro = this.aligner.rotate(this.currentGyro);
        const filteredIMU = this.imuFilter.step([...alignedAccel, ...alignedGyro], dt);

        // 3. Resample into 10 Hz sensor buffer (decoupling Android raw 50-100Hz hardware rate from 10Hz model)
        this.sensorBuffer.push(filteredIMU);
        this.sensorTimes.push(currentTime);
        if (this.sensorBuffer.length > this.windowSize) {
            this.sensorBuffer.shift();
            this.sensorTimes.shift();
        }
        if (this.sensorTimes.length > 1) {
            const timeSpan = currentTime - this.sensorTimes[0];
            if (timeSpan > 0) {
                this.observedEffectiveRate = (this.sensorTimes.length - 1) / timeSpan;
            }
        } else {
            this.observedEffectiveRate = 10.0;
        }
        this.observedImuRate = this.observedEffectiveRate;

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

        // 4. EKF Predict (High frequency, 10Hz)
        let ekfStart = performance.now();
        let ekfMs = 0;
        this.ekf.dt = dt;
        this.ekf.predict(alignedAccel, alignedGyro, null, true); // apply NHC

        // 5. Robust Vehicle Motion Detection (ZUPT)
        // Multi-signal: accelerometer variance, dynamic accel, gyro variance, dwell time, GNSS speed
        let estSpeed = hasGoodGNSS ? speed : null;
        this.zuptActive = this.motionDetector.update(this.currentAccel, this.currentGyro, estSpeed, dt, currentTime);
        const motionState = this.motionDetector.motionState;
        const motionReason = this.motionDetector.motionReason;

        if (this.zuptActive) {
            this.ekf.updateZupt(0.01);
        }

        ekfMs += performance.now() - ekfStart;

        // 6. Run Edge AI Model if buffer is full and vehicle is moving
        let aiVelocity = null;
        let aiError = null;
        let aiStatus = 'INACTIVE';
        const expectedRate = this.modelContract?.sample_rate_hz || 10;
        const effectiveRate = this.observedEffectiveRate || 10;
        const rateMatches = Math.abs(effectiveRate / expectedRate - 1) <= 0.25;
        if (!rateMatches && expectedRate) {
            aiError = `Effective IMU rate (${effectiveRate.toFixed(1)}Hz) differs from trained model (${expectedRate}Hz)`;
        }

        let fallbackReason = null;
        if (!this.session) {
            fallbackReason = 'model unavailable (session not initialized)';
            aiStatus = 'MODEL_UNAVAILABLE';
        } else if (this.sensorBuffer.length < this.windowSize) {
            fallbackReason = `insufficient samples (${this.sensorBuffer.length}/${this.windowSize})`;
            aiStatus = 'BUFFERING';
        } else if (this.zuptActive) {
            fallbackReason = `stationary/ZUPT (${motionReason})`;
            aiStatus = 'STATIONARY_GATED';
        } else if (!rateMatches) {
            fallbackReason = `sensor rate mismatch (effective: ${effectiveRate.toFixed(1)} Hz, expected: ${expectedRate} Hz)`;
            aiStatus = 'RATE_MISMATCH';
        }

        if (this.sensorBuffer.length === this.windowSize && this.session && !this.zuptActive && rateMatches) {
            try {
                aiStatus = 'RUNNING';
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
                const aiLatency = performance.now() - aiStart;
                this.profiler.record('tcn', aiLatency);
                const outputName = this.session.outputNames[0];
                const v = results[outputName].data;
                if (v.length !== 2 || !Array.from(v).every(Number.isFinite)) {
                    throw new Error('Invalid AI velocity: non-finite or dimension mismatch');
                }

                // Map according to contract output order
                let vNorth, vEast;
                if (this.modelContract?.output_order?.[0] === 'north') {
                    vNorth = v[0];
                    vEast = v[1];
                } else {
                    vEast = v[0];
                    vNorth = v[1];
                }
                const predSpeed = Math.hypot(vNorth, vEast);
                aiVelocity = [vEast, vNorth]; // EKF state vector maps index 3 -> vEast, index 4 -> vNorth
                aiStatus = 'ACTIVE';

                if (!this.isCapturing) return;
                if (this.gpsOutage || gnssGeneration !== this.gnssGeneration) hasGoodGNSS = false;
                // If GNSS is denied / outage, feed AI velocity into EKF
                if (!hasGoodGNSS) {
                    ekfStart = performance.now();
                    this.ekf._updateAiVelocity(aiVelocity);
                    ekfMs += performance.now() - ekfStart;
                }
            } catch (e) {
                aiError = e.message;
                aiStatus = 'ERROR';
                fallbackReason = `inference error (${e.message})`;
                console.error("[Diagnostic AI] Inference failure:", e.message);
            }
        }

        // 7. Rate-limited Vehicle Motion & Inference Diagnostics (Requirement 9)
        const nowMs = performance.now();
        const stateChanged = (motionState !== this.lastLoggedMotionState);
        if (stateChanged || (nowMs - this.lastDiagnosticLogTime >= 1000)) {
            this.lastDiagnosticLogTime = nowMs;
            this.lastLoggedMotionState = motionState;
            const metrics = this.motionDetector.lastMetrics;
            const accelMag = metrics ? metrics.accelMag.toFixed(2) : Math.hypot(...this.currentAccel).toFixed(2);
            const dynAccel = metrics ? metrics.dynamicAccel.toFixed(2) : Math.abs(Math.hypot(...this.currentAccel) - 9.81).toFixed(2);
            const gyroMag = metrics ? metrics.gyroMag.toFixed(3) : Math.hypot(...this.currentGyro).toFixed(3);
            const predSpeedStr = aiVelocity ? `${Math.hypot(...aiVelocity).toFixed(1)} m/s` : 'N/A';
            const gnssSpeedStr = hasGoodGNSS ? `${speed.toFixed(1)} m/s` : (this.gpsOutage ? 'OUTAGE' : 'NO FIX');

            console.log(
                `[Vehicle Motion Diagnostic] State: ${motionState} | Reason: ${motionReason} | ` +
                `AccelMag: ${accelMag} m/s² (dyn: ${dynAccel}) | GyroMag: ${gyroMag} rad/s | ` +
                `RawRate: ${(this.observedRawSensorRate || 10).toFixed(1)} Hz | EffRate: ${effectiveRate.toFixed(1)} Hz | ` +
                `Buffer: ${this.sensorBuffer.length}/${this.windowSize} | AI: ${aiStatus} (Pred: ${predSpeedStr}) | ` +
                `GNSS: ${gnssSpeedStr}`
            );
        }

        ekfStart = performance.now();
        // 8. Navigation Update (GNSS Update)
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

            // Use the compass to ensure heading does not hallucinate
            if (this.currentCompassHeading !== undefined) {
                // sigma grows slightly with time since we trust absolute compass less than GNSS,
                // but we trust it much more than integrated drifting gyro!
                const compassSigma = Math.min(1.0, 0.3 + this.drDuration * 0.01);
                this.ekf.updateCompass(this.currentCompassHeading, compassSigma);
            }
        }

        ekfMs += performance.now() - ekfStart;
        this.profiler.record('ekf', ekfMs);
        // 9. Map Matching
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
        // 10. Extract Final State for UI
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
            ? `IMU + EKF · STATIONARY (${motionReason}) · AI gated`
            : `IMU + EKF · ${motionState} (${motionReason}) · ${aiVelocity ? `AI active (${Math.hypot(...aiVelocity).toFixed(1)} m/s)` : `AI buffering ${this.sensorBuffer.length}/${this.windowSize}`} · ${this.gpsOutage ? 'GPS disabled' : this.ekf.mode === 'reacq' ? 'REACQUISITION' : currentGnssState}`;
        document.getElementById('componentStatus').textContent =
            `Motion: ${motionState} · AI: ${aiError ? 'ERROR (' + aiError + ')' : aiVelocity ? `ACTIVE (${(Math.hypot(...aiVelocity)*3.6).toFixed(1)} km/h)` : this.zuptActive ? 'GATED (ZUPT)' : 'BUFFERING'} · EKF: ACTIVE · NHC: ${estimatedSpeed >= 0.5 ? 'ACTIVE' : 'LOW SPEED'} · ZUPT: ${this.zuptActive ? 'ACTIVE' : 'MONITORING'} · MAP: ${match.confidence > 0.8 ? 'MATCHED' : 'NO CONFIDENT MATCH'}`;
        this.updateUI({
            status: 'active',
            motion_state: motionState,
            motion_reason: motionReason,
            raw_sensor_rate: this.observedRawSensorRate || 10,
            effective_sensor_rate: effectiveRate,
            buffered_samples: this.sensorBuffer.length,
            ai_predicted_speed: aiVelocity ? Math.hypot(...aiVelocity) : null,
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
        this.pendingUIData = data;
        if (!this.uiFrameRequested) {
            this.uiFrameRequested = true;
            if (typeof requestAnimationFrame === 'function') {
                requestAnimationFrame(() => this.renderUI());
            } else if (typeof setTimeout === 'function') {
                setTimeout(() => this.renderUI(), 16);
            } else {
                this.renderUI();
            }
        }
    }

    renderUI() {
        this.uiFrameRequested = false;
        const data = this.pendingUIData;
        if (!data) return;
        
        if (data.status !== 'active') window.updateConsoleTelemetry?.(data);
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
            updateNavMode(data.nav_mode, data.position_error);
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
                state.estimatedLine.addLatLng([estLat, estLon]);
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
