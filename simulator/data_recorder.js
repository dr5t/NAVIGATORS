




class DataRecorder {
    constructor() {
        this.isRecording = false;
        
        
        this.currentAccel = [0, 0, 0];
        this.currentGyro = [0, 0, 0];
        this.currentOrient = [0, 0, 0];
        this.currentGnss = [null, null, null, null, null, null];
        this.currentGnssTimestamp = null; 
        
        
        this.buffer = {
            timestamps: [],
            accel: [],
            gyro: [],
            orient: [],
            gnss: [],
            gnss_timestamps: []
        };
        
        this.startTime = 0;
        
        
        this.ws = null;
        this.wsConnected = false;
        this.telemetryInterval = null;
    }

    async connectToServer(host) {
        if (this.ws) this.ws.close();
        const wsUrl = `ws://${host}/ws?role=mobile`;
        const secureWsUrl = window.location.protocol === 'https:' ? `wss://${host}/ws?role=mobile` : wsUrl;
        console.log(`[Sensor] Connecting to ${secureWsUrl}...`);
        
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(secureWsUrl);
                this.ws.onopen = () => {
                    console.log('[Sensor] WebSocket Connected');
                    this.wsConnected = true;
                    resolve(true);
                };
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (window.handleServerMessage) window.handleServerMessage(data);
                };
                this.ws.onclose = () => {
                    console.log('[Sensor] WebSocket Disconnected');
                    this.wsConnected = false;
                };
                this.ws.onerror = (e) => reject(e);
            } catch (e) {
                reject(e);
            }
        });
    }

    async requestPermissionsAndStart() {
        
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
        
        if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
            try {
                await DeviceOrientationEvent.requestPermission();
            } catch (e) {
                console.error('Error requesting orientation permission:', e);
            }
        }
        
        this.startRecording();
        return true;
    }

    startRecording() {
        if (this.isRecording) return;
        console.log('[Data Recorder] Starting recording...');
        this.isRecording = true;
        this.startTime = Date.now();
        
        
        this.buffer = { timestamps: [], accel: [], gyro: [], orient: [], gnss: [], gnss_timestamps: [] };
        this.currentGnss = [null, null, null, null, null, null];
        this.currentGnssTimestamp = null;
        this.recordingEpoch = Date.now() / 1000;
        this.recordingClock = performance.now() / 1000;
        this.metadata = {
            schema_version: 2, provenance: 'phone sensor capture (self-reported by recorder)', gyro_order: 'xyz',
            navigation_mode: window.offlineEngine.isCapturing ? window.offlineEngine.navigationMode : document.getElementById('travelMode').value,
            user_agent: navigator.userAgent, start_time: this.recordingEpoch,
            columns: { accel: 'x, y, z (m/s^2)', gyro: 'x, y, z (rad/s)', orient: 'alpha, beta, gamma (deg)', gnss: 'lat, lon, alt, speed, heading, accuracy' },
        };
        this.syncTrip = window.recordingSync.start(this.buffer, this.metadata);

        
        this.handleMotion = (event) => {
            if (!event.accelerationIncludingGravity || !event.rotationRate) return;
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
            
            
            this.recordFrame();
        };
        window.addEventListener('devicemotion', this.handleMotion);

        
        this.handleOrientation = (event) => {
            
            this.currentOrient = [
                event.alpha || 0,
                event.beta || 0,
                event.gamma || 0
            ];
        };
        window.addEventListener('deviceorientation', this.handleOrientation);

        
        if ('geolocation' in navigator) {
            this.watchId = navigator.geolocation.watchPosition(
                (position) => {
                    if (!this.isRecording) return;
                    this.currentGnssTimestamp = position.timestamp / 1000;
                    this.currentGnss = [
                        position.coords.latitude,
                        position.coords.longitude,
                        position.coords.altitude || 0,
                        position.coords.speed ?? null,
                        position.coords.heading ?? null,
                        position.coords.accuracy ?? null
                    ];
                },
                (error) => {
                    this.currentGnssTimestamp = null;
                    console.warn('[Data Recorder] GNSS Error:', error.message);
                },
                { enableHighAccuracy: true, maximumAge: 0 }
            );
        }

        
        this.telemetryInterval = setInterval(() => {
            if (this.wsConnected && this.ws.readyState === WebSocket.OPEN) {
                const gnssPayload = this.currentGnssTimestamp && (Date.now()/1000 - this.currentGnssTimestamp < 3) ? {
                    lat: this.currentGnss[0],
                    lon: this.currentGnss[1],
                    alt: this.currentGnss[2],
                    speed: this.currentGnss[3],
                    heading: this.currentGnss[4],
                    accuracy: this.currentGnss[5]
                } : null;
                
                this.ws.send(JSON.stringify({
                    timestamp: Date.now() / 1000.0,
                    accel: this.currentAccel,
                    gyro: this.currentGyro,
                    gnss: gnssPayload
                }));
            }
        }, 100);
    }

    recordFrame() {
        if (!this.isRecording) return;
        const ts = this.recordingEpoch + performance.now() / 1000 - this.recordingClock;
        this.buffer.timestamps.push(ts);
        this.buffer.accel.push([...this.currentAccel]);
        this.buffer.gyro.push([...this.currentGyro]);
        this.buffer.orient.push([...this.currentOrient]);
        const fresh = this.currentGnssTimestamp !== null && ts - this.currentGnssTimestamp <= 3;
        this.buffer.gnss.push(fresh ? [...this.currentGnss] : [null, null, null, null, null, null]);
        this.buffer.gnss_timestamps.push(fresh ? this.currentGnssTimestamp : null);
    }

    stopRecording() {
        if (!this.isRecording) return;
        this.isRecording = false;
        
        
        if (this.handleMotion) {
            window.removeEventListener('devicemotion', this.handleMotion);
        }
        if (this.handleOrientation) {
            window.removeEventListener('deviceorientation', this.handleOrientation);
        }
        if (this.watchId !== undefined) {
            navigator.geolocation.clearWatch(this.watchId);
        }
        if (this.telemetryInterval) {
            clearInterval(this.telemetryInterval);
        }
        
        console.log(`[Data Recorder] Stopped. Captured ${this.buffer.timestamps.length} frames.`);
        
        const duration = (Date.now() - this.startTime) / 1000.0;
        
        
        this.metadata.duration_sec = duration;
        this.metadata.num_frames = this.buffer.timestamps.length;
        
        window.recordingSync.finish(this.syncTrip);
    }
}


window.dataRecorder = new DataRecorder();
