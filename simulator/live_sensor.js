/**
 * Navigators IDR — Live Sensor Client
 * Captures mobile IMU & GNSS data and streams it to the FastAPI backend over WebSockets.
 */

class LiveSensorClient {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.isCapturing = false;
        
        // Sensor Data Buffer
        this.currentAccel = [0, 0, 0];
        this.currentGyro = [0, 0, 0];
        this.currentGnss = null;
        
        this.sensorUpdateInterval = null;
        this.updateRateMs = 100; // 10 Hz
    }

    async connectToServer(host) {
        if (this.ws) {
            this.ws.close();
        }

        const wsUrl = `ws://${host}/ws`;
        // Handle HTTPS -> WSS automatically if needed
        const secureWsUrl = window.location.protocol === 'https:' ? `wss://${host}/ws` : wsUrl;

        console.log(`[Sensor] Connecting to ${secureWsUrl}...`);
        
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(secureWsUrl);
                
                this.ws.onopen = () => {
                    console.log('[Sensor] WebSocket Connected');
                    this.isConnected = true;
                    resolve(true);
                };
                
                this.ws.onmessage = (event) => {
                    this.handleServerMessage(JSON.parse(event.data));
                };
                
                this.ws.onclose = () => {
                    console.log('[Sensor] WebSocket Disconnected');
                    this.isConnected = false;
                    this.stopCapture();
                };
                
                this.ws.onerror = (error) => {
                    console.error('[Sensor] WebSocket Error:', error);
                    reject(error);
                };
            } catch (e) {
                reject(e);
            }
        });
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
                alert('Must be in HTTPS or localhost to request sensor permissions.');
                return false;
            }
        }
        
        this.startCapture();
        return true;
    }

    startCapture() {
        if (!this.isConnected) {
            alert('Connect to server first!');
            return;
        }
        if (this.isCapturing) return;

        console.log('[Sensor] Starting sensor capture...');
        this.isCapturing = true;

        // IMU Listeners
        this.handleMotion = (event) => {
            // Acceleration including gravity (we usually need linear accel + gravity for EKF, but let's use what we have)
            const accel = event.accelerationIncludingGravity;
            if (accel) {
                this.currentAccel = [accel.x || 0, accel.y || 0, accel.z || 0];
            }
            
            const gyro = event.rotationRate;
            if (gyro) {
                // Convert deg/s to rad/s
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
                        heading: position.coords.heading || 0
                    };
                },
                (error) => {
                    console.warn('[Sensor] GNSS Error:', error.message);
                },
                { enableHighAccuracy: true, maximumAge: 0 }
            );
        }

        // Send Data Interval
        this.sensorUpdateInterval = setInterval(() => {
            this.sendData();
        }, this.updateRateMs);
    }

    stopCapture() {
        this.isCapturing = false;
        if (this.handleMotion) {
            window.removeEventListener('devicemotion', this.handleMotion);
        }
        if (this.watchId !== undefined) {
            navigator.geolocation.clearWatch(this.watchId);
        }
        if (this.sensorUpdateInterval) {
            clearInterval(this.sensorUpdateInterval);
        }
        console.log('[Sensor] Stopped sensor capture.');
    }

    sendData() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const payload = {
            timestamp: Date.now() / 1000.0,
            accel: this.currentAccel,
            gyro: this.currentGyro,
            gnss: this.currentGnss
        };

        this.ws.send(JSON.stringify(payload));
        
        // Clear GNSS so we only send it when we get a fresh update, or keep sending last known?
        // Let's clear it so backend knows exactly when a new fix arrives.
        this.currentGnss = null; 
    }

    handleServerMessage(data) {
        // Integrate with the existing app.js UI
        if (data.status === 'waiting_for_gnss') {
            updateNavMode('reacq'); // Re-using as "acquiring"
            updateGnssStatus(false);
            return;
        }

        if (data.status === 'active') {
            // This expects the global functions from app.js to be available
            if (typeof updateNavMode === 'function') {
                updateNavMode(data.nav_mode);
                updateGnssStatus(data.gnss_available);
                updateSpeed(data.speed * 3.6, data.heading);
                updatePositionError(data.position_error);
                updateDrift(data.dr_drift_percent);
                updateConfidence(data.confidence);

                // Update Map markers
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
}

// Global instance
window.liveSensorClient = new LiveSensorClient();
