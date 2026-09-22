/**
 * Navigators IDR - Phone-to-Vehicle Alignment
 * 
 * Estimates the orientation of the phone relative to the vehicle frame.
 * Assumes the vehicle travels mostly forward and flat.
 */

class PhoneVehicleAligner {
    constructor() {
        this.isAligned = false;
        
        this.gravityAccumulator = [0, 0, 0];
        this.gravityCount = 0;
        
        this.forwardAccumulator = [0, 0, 0];
        this.forwardCount = 0;
        
        this.lastSpeed = -1;
        this.lastSpeedTime = 0;
        
        // Rotation Matrix (Phone to Vehicle)
        this.R = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
        
        // Configuration
        this.GRAVITY_SAMPLES_REQUIRED = 100; // 10 seconds at 10Hz
        this.FORWARD_SAMPLES_REQUIRED = 50;  // 0.5 seconds of pure acceleration
    }

    reset() {
        this.isAligned = false;
        this.gravityAccumulator = [0, 0, 0];
        this.gravityCount = 0;
        this.forwardAccumulator = [0, 0, 0];
        this.forwardCount = 0;
        this.lastSpeed = -1;
    }

    feed(accel, gnssSpeed, timestamp) {
        if (this.isAligned) return true; // Already calibrated

        // 1. Estimate Gravity (Z-axis)
        if (this.gravityCount < this.GRAVITY_SAMPLES_REQUIRED) {
            this.gravityAccumulator[0] += accel[0];
            this.gravityAccumulator[1] += accel[1];
            this.gravityAccumulator[2] += accel[2];
            this.gravityCount++;
            return false;
        }

        const Z = this._normalize([
            this.gravityAccumulator[0] / this.gravityCount,
            this.gravityAccumulator[1] / this.gravityCount,
            this.gravityAccumulator[2] / this.gravityCount
        ]);

        // 2. Estimate Forward (X-axis)
        if (this.lastSpeed >= 0 && timestamp - this.lastSpeedTime > 0.1) {
            const accelMagnitude = (gnssSpeed - this.lastSpeed) / (timestamp - this.lastSpeedTime);
            
            // If accelerating forward > 0.5 m/s^2
            if (accelMagnitude > 0.5) {
                // Project current acceleration onto horizontal plane
                const dotZ = accel[0]*Z[0] + accel[1]*Z[1] + accel[2]*Z[2];
                const accelHoriz = [
                    accel[0] - dotZ * Z[0],
                    accel[1] - dotZ * Z[1],
                    accel[2] - dotZ * Z[2]
                ];
                
                this.forwardAccumulator[0] += accelHoriz[0];
                this.forwardAccumulator[1] += accelHoriz[1];
                this.forwardAccumulator[2] += accelHoriz[2];
                this.forwardCount++;
            }
        }
        
        this.lastSpeed = gnssSpeed;
        this.lastSpeedTime = timestamp;

        // 3. Finalize Alignment
        if (this.forwardCount >= this.FORWARD_SAMPLES_REQUIRED) {
            let X_raw = [
                this.forwardAccumulator[0] / this.forwardCount,
                this.forwardAccumulator[1] / this.forwardCount,
                this.forwardAccumulator[2] / this.forwardCount
            ];
            
            // Ensure X is perfectly orthogonal to Z
            const dotXZ = X_raw[0]*Z[0] + X_raw[1]*Z[1] + X_raw[2]*Z[2];
            let X = this._normalize([
                X_raw[0] - dotXZ * Z[0],
                X_raw[1] - dotXZ * Z[1],
                X_raw[2] - dotXZ * Z[2]
            ]);
            
            // Y = Z cross X
            let Y = this._normalize([
                Z[1]*X[2] - Z[2]*X[1],
                Z[2]*X[0] - Z[0]*X[2],
                Z[0]*X[1] - Z[1]*X[0]
            ]);
            
            // R = [X, Y, Z]^T
            this.R = [
                [X[0], X[1], X[2]],
                [Y[0], Y[1], Y[2]],
                [Z[0], Z[1], Z[2]]
            ];
            
            this.isAligned = true;
            console.log("[Alignment] Calibration Locked!");
            console.log(`[Alignment] R = ${JSON.stringify(this.R)}`);
            return true;
        }

        return false;
    }

    rotate(vector3) {
        if (!this.isAligned) return vector3;
        
        return [
            this.R[0][0]*vector3[0] + this.R[0][1]*vector3[1] + this.R[0][2]*vector3[2],
            this.R[1][0]*vector3[0] + this.R[1][1]*vector3[1] + this.R[1][2]*vector3[2],
            this.R[2][0]*vector3[0] + this.R[2][1]*vector3[1] + this.R[2][2]*vector3[2]
        ];
    }
    
    _normalize(v) {
        const mag = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (mag === 0) return [0,0,0];
        return [v[0]/mag, v[1]/mag, v[2]/mag];
    }
}
