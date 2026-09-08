/**
 * Navigators IDR — Data Recorder
 * Captures raw smartphone sensor data for dataset generation.
 */

class DataRecorder {
    constructor() {
        this.isRecording = false;
        
        // Sensor State
        this.currentAccel = [0, 0, 0];
        this.currentGyro = [0, 0, 0];
        this.currentOrient = [0, 0, 0];
        this.currentGnss = [null, null, null, null, null, null];
        this.currentGnssTimestamp = null; // lat, lon, alt, speed, heading, accuracy
        
        // Data Buffer
        this.buffer = {
            timestamps: [],
            accel: [],
            gyro: [],
            orient: [],
            gnss: [],
            gnss_timestamps: []
        };
        
        this.startTime = 0;
    }

    async requestPermissionsAndStart() {
        // iOS requires explicit permission for DeviceMotionEvent and DeviceOrientationEvent
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
        
        // Clear buffer
        this.buffer = { timestamps: [], accel: [], gyro: [], orient: [], gnss: [], gnss_timestamps: [] };
        this.currentGnss = [null, null, null, null, null, null];
        this.currentGnssTimestamp = null;
        this.recordingEpoch = Date.now() / 1000;
        this.recordingClock = performance.now() / 1000;

        // 1. Device Motion (IMU)
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
            
            // Record a frame exactly when IMU updates (typically 50-100Hz)
            this.recordFrame();
        };
        window.addEventListener('devicemotion', this.handleMotion);

        // 2. Device Orientation (Attitude)
        this.handleOrientation = (event) => {
            // Absolute orientation if available, otherwise relative
            this.currentOrient = [
                event.alpha || 0,
                event.beta || 0,
                event.gamma || 0
            ];
        };
        window.addEventListener('deviceorientation', this.handleOrientation);

        // 3. GNSS Location
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

    stopRecordingAndDownload() {
        if (!this.isRecording) return;
        this.isRecording = false;
        
        // Remove listeners
        if (this.handleMotion) {
            window.removeEventListener('devicemotion', this.handleMotion);
        }
        if (this.handleOrientation) {
            window.removeEventListener('deviceorientation', this.handleOrientation);
        }
        if (this.watchId !== undefined) {
            navigator.geolocation.clearWatch(this.watchId);
        }
        
        console.log(`[Data Recorder] Stopped. Captured ${this.buffer.timestamps.length} frames.`);
        
        const duration = (Date.now() - this.startTime) / 1000.0;
        
        // Format final payload
        const payload = {
            metadata: {
                schema_version: 2,
                provenance: 'phone sensor capture (self-reported by recorder)',
                gyro_order: 'xyz',
                user_agent: navigator.userAgent,
                start_time: this.startTime / 1000.0,
                duration_sec: duration,
                num_frames: this.buffer.timestamps.length,
                columns: {
                    accel: "x, y, z (m/s^2)",
                    gyro: "x, y, z (rad/s)",
                    orient: "alpha, beta, gamma (deg)",
                    gnss: "lat, lon, alt, speed, heading, accuracy"
                }
            },
            data: this.buffer
        };
        
        // Trigger file download
        const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        
        const dateStr = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `trip_${dateStr}.json`;
        
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);
    }
}

// Global instance
window.dataRecorder = new DataRecorder();
