/**
 * Navigators IDR - Extended Kalman Filter (JavaScript Edge Port)
 * 15-state EKF for vehicle navigation with GNSS, INS, AI-velocity, NHC, and ZUPT fusion.
 */

class ExtendedKalmanFilter {
    constructor(dt = 0.1) {
        this.dt = dt;
        this.STATE_DIM = 15;

        // State vector (15x1)
        // [0:3] Position (E, N, U)
        // [3:6] Velocity (vE, vN, vU)
        // [6:9] Orientation (roll, pitch, yaw)
        // [9:12] Accel Bias
        // [12:15] Gyro Bias
        this.x = Matrix.zeros(this.STATE_DIM, 1);

        // Covariance Matrix (15x15)
        this.P = Matrix.eye(this.STATE_DIM);
        for(let i=0; i<3; i++) this.P[i][i] = 10.0;
        for(let i=3; i<6; i++) this.P[i][i] = 5.0;
        for(let i=6; i<9; i++) this.P[i][i] = 0.1;
        for(let i=9; i<12; i++) this.P[i][i] = 0.5;
        for(let i=12; i<15; i++) this.P[i][i] = 0.01;

        // Process Noise parameters
        this.q_pos = 0.5;
        this.q_vel = 2.0;
        this.q_ori = 0.05;
        this.q_abias = 0.001;
        this.q_gbias = 0.0001;

        // Measurement Noise parameters
        this.r_gnss_pos = 2.5;
        this.r_gnss_vel = 0.5;
        this.r_ai_vel = 0.3;
        this.r_nhc_lateral = 1.0;
        this.r_nhc_vertical = 1.0;

        this.mode = 'gnss_ins';
        this.reacquisition_steps = 0;
        this.consecutive_good_gnss = 0;
    }

    _buildProcessNoise() {
        let Q = Matrix.zeros(this.STATE_DIM, this.STATE_DIM);
        for(let i=0; i<3; i++) Q[i][i] = this.q_pos * this.dt * this.dt;
        for(let i=3; i<6; i++) Q[i][i] = this.q_vel * this.dt;
        for(let i=6; i<9; i++) Q[i][i] = this.q_ori * this.dt;
        for(let i=9; i<12; i++) Q[i][i] = this.q_abias * this.dt;
        for(let i=12; i<15; i++) Q[i][i] = this.q_gbias * this.dt;
        return Q;
    }

    _rotationMatrix(roll, pitch, yaw) {
        let cr = Math.cos(roll), sr = Math.sin(roll);
        let cp = Math.cos(pitch), sp = Math.sin(pitch);
        let cy = Math.sin(yaw), sy = Math.cos(yaw);

        return [
            [cy * cp,   sy * cr + cy * sp * sr,  -sy * sr + cy * sp * cr],
            [sy * cp,  -cy * cr + sy * sp * sr,   cy * sr + sy * sp * cr],
            [sp,       -cp * sr,                 -cp * cr               ]
        ];
    }

    _enforceSymmetry() {
        for (let i = 0; i < this.STATE_DIM; i++) {
            for (let j = i; j < this.STATE_DIM; j++) {
                let avg = 0.5 * (this.P[i][j] + this.P[j][i]);
                this.P[i][j] = avg;
                this.P[j][i] = avg;
            }
            if (this.P[i][i] < 1e-9) this.P[i][i] = 1e-9;
        }
    }

    predict(accel_body, gyro_body, ai_velocity = null, apply_nhc = true) {
        let roll = this.x[6][0], pitch = this.x[7][0], yaw = this.x[8][0];
        
        // Corrected IMU
        let ax = accel_body[0] - this.x[9][0];
        let ay = accel_body[1] - this.x[10][0];
        let az = accel_body[2] - this.x[11][0];

        let gx = gyro_body[0] - this.x[12][0];
        let gy = gyro_body[1] - this.x[13][0];
        let gz = gyro_body[2] - this.x[14][0];

        let R_b2n = this._rotationMatrix(roll, pitch, yaw);
        
        // Navigation frame specific force
        let an_x = R_b2n[0][0]*ax + R_b2n[0][1]*ay + R_b2n[0][2]*az;
        let an_y = R_b2n[1][0]*ax + R_b2n[1][1]*ay + R_b2n[1][2]*az;
        let an_z = (R_b2n[2][0]*ax + R_b2n[2][1]*ay + R_b2n[2][2]*az) + 9.81; // Gravity removal in ENU Z=Up is -9.81 for gravity, so we add 9.81? Wait, in python: accel_nav = R_b2n @ accel_corrected - [0,0,-9.81] which means +9.81.
        
        let accel_nav = [an_x, an_y, an_z];

        // State update
        this.x[0][0] += this.x[3][0]*this.dt + 0.5*accel_nav[0]*(this.dt*this.dt);
        this.x[1][0] += this.x[4][0]*this.dt + 0.5*accel_nav[1]*(this.dt*this.dt);
        this.x[2][0] += this.x[5][0]*this.dt + 0.5*accel_nav[2]*(this.dt*this.dt);

        this.x[3][0] += accel_nav[0]*this.dt;
        this.x[4][0] += accel_nav[1]*this.dt;
        this.x[5][0] += accel_nav[2]*this.dt;

        this.x[6][0] += gx*this.dt;
        this.x[7][0] += gy*this.dt;
        this.x[8][0] += gz*this.dt;
        
        // Normalize yaw
        this.x[8][0] = (this.x[8][0] + Math.PI) % (2.0 * Math.PI);
        if (this.x[8][0] < 0) this.x[8][0] += 2*Math.PI;
        this.x[8][0] -= Math.PI;

        // Jacobian F
        let F = Matrix.eye(this.STATE_DIM);
        for(let i=0; i<3; i++) F[i][i+3] = this.dt;

        let skew_accel = [
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0]
        ];

        let R_skew_dt = Matrix.scale(Matrix.mul(R_b2n, skew_accel), -this.dt);
        let R_dt = Matrix.scale(R_b2n, -this.dt);
        
        for(let i=0; i<3; i++) {
            for(let j=0; j<3; j++) {
                F[3+i][6+j] = R_skew_dt[i][j];
                F[3+i][9+j] = R_dt[i][j];
            }
            F[6+i][12+i] = -this.dt;
        }

        let Q = this._buildProcessNoise();
        if (this.mode === 'dr') {
            for(let i=0; i<3; i++) Q[i][i] *= 4.0;
            for(let i=3; i<6; i++) Q[i][i] *= 2.0;
        }

        let F_T = Matrix.transpose(F);
        this.P = Matrix.add(Matrix.mul(Matrix.mul(F, this.P), F_T), Q);
        this._enforceSymmetry();

        if (ai_velocity && (this.mode === 'dr' || this.mode === 'gnss_degraded')) {
            this._updateAiVelocity(ai_velocity);
        }

        if (apply_nhc) {
            this.updateNhc(gz);
        }
    }

    _updateAiVelocity(ai_vel) {
        let H = Matrix.zeros(2, this.STATE_DIM);
        H[0][3] = 1.0;
        H[1][4] = 1.0;

        let R = [
            [this.r_ai_vel * this.r_ai_vel, 0],
            [0, this.r_ai_vel * this.r_ai_vel]
        ];

        let z = [[ai_vel[0]], [ai_vel[1]]];
        let H_x = Matrix.mul(H, this.x);
        let y = Matrix.sub(z, H_x);

        let H_T = Matrix.transpose(H);
        let S = Matrix.add(Matrix.mul(Matrix.mul(H, this.P), H_T), R);
        let S_inv = Matrix.inv2x2(S);

        let K = Matrix.mul(Matrix.mul(this.P, H_T), S_inv);

        this.x = Matrix.add(this.x, Matrix.mul(K, y));

        let I_KH = Matrix.sub(Matrix.eye(this.STATE_DIM), Matrix.mul(K, H));
        let K_R_KT = Matrix.mul(Matrix.mul(K, R), Matrix.transpose(K));
        this.P = Matrix.add(Matrix.mul(Matrix.mul(I_KH, this.P), Matrix.transpose(I_KH)), K_R_KT);
        this._enforceSymmetry();

        let speed = Math.sqrt(ai_vel[0]*ai_vel[0] + ai_vel[1]*ai_vel[1]);
        if (speed > 2.0) {
            let cog = Math.atan2(ai_vel[0], ai_vel[1]);
            let yaw_err = (cog - this.x[8][0] + Math.PI) % (2.0*Math.PI);
            if (yaw_err < 0) yaw_err += 2*Math.PI;
            yaw_err -= Math.PI;
            
            this.x[8][0] = (this.x[8][0] + 0.15*yaw_err + Math.PI) % (2.0*Math.PI);
            if (this.x[8][0] < 0) this.x[8][0] += 2*Math.PI;
            this.x[8][0] -= Math.PI;
        }
    }

    updateCompass(heading_rad, sigma_rad = 0.5) {
        let H = Matrix.zeros(1, this.STATE_DIM);
        H[0][8] = 1.0;

        let R = [[sigma_rad * sigma_rad]];

        let z = [[heading_rad]];
        
        let yaw_err = (heading_rad - this.x[8][0] + Math.PI) % (2.0*Math.PI);
        if (yaw_err < 0) yaw_err += 2*Math.PI;
        yaw_err -= Math.PI;

        let y = [[yaw_err]];

        let H_T = Matrix.transpose(H);
        let S = Matrix.add(Matrix.mul(Matrix.mul(H, this.P), H_T), R);
        let S_inv = [[1.0 / S[0][0]]];

        let K = Matrix.mul(Matrix.mul(this.P, H_T), S_inv);

        this.x = Matrix.add(this.x, Matrix.mul(K, y));
        
        // Normalize heading state
        this.x[8][0] = (this.x[8][0] + Math.PI) % (2.0*Math.PI);
        if (this.x[8][0] < 0) this.x[8][0] += 2*Math.PI;
        this.x[8][0] -= Math.PI;

        let I_KH = Matrix.sub(Matrix.eye(this.STATE_DIM), Matrix.mul(K, H));
        let K_R_KT = Matrix.mul(Matrix.mul(K, R), Matrix.transpose(K));
        this.P = Matrix.add(Matrix.mul(Matrix.mul(I_KH, this.P), Matrix.transpose(I_KH)), K_R_KT);
        this._enforceSymmetry();
    }

    updateGnss(gnss_pos, gnss_vel = null) {
        const priorX = this.x.map(row => [...row]);
        const priorP = this.P.map(row => [...row]);
        if (this.mode === 'dr') {
            this.mode = 'reacq';
            this.reacquisition_steps = 0;
            this.consecutive_good_gnss = 1;
        } else if (this.mode === 'reacq') {
            this.reacquisition_steps++;
            this.consecutive_good_gnss++;
        } else {
            this.mode = 'gnss_ins';
            this.consecutive_good_gnss++;
        }

        let H_pos = Matrix.zeros(3, this.STATE_DIM);
        H_pos[0][0] = 1; H_pos[1][1] = 1; H_pos[2][2] = 1;

        let R_pos = Matrix.eye(3);
        let r2 = this.r_gnss_pos * this.r_gnss_pos;
        
        let ramp = 1.0;
        if (this.mode === 'reacq') {
            ramp = Math.max(0.1, Math.min(1.0, this.reacquisition_steps / 10.0));
        }
        r2 = r2 * (1.0 / ramp);
        for(let i=0; i<3; i++) R_pos[i][i] = r2;

        let z_pos = [[gnss_pos[0]], [gnss_pos[1]], [0]];
        let y_pos = Matrix.sub(z_pos, Matrix.mul(H_pos, this.x));

        if (this.mode === 'reacq') {
            let y_norm = Math.sqrt(y_pos[0][0]*y_pos[0][0] + y_pos[1][0]*y_pos[1][0]);
            if (y_norm > 10.0) {
                y_pos[0][0] *= 10.0 / y_norm;
                y_pos[1][0] *= 10.0 / y_norm;
            }
        }

        let H_T = Matrix.transpose(H_pos);
        let S_pos = Matrix.add(Matrix.mul(Matrix.mul(H_pos, this.P), H_T), R_pos);
        let S_inv = Matrix.inv3x3(S_pos);

        let K = Matrix.mul(Matrix.mul(this.P, H_T), S_inv);
        this.x = Matrix.add(this.x, Matrix.mul(K, y_pos));

        let I_KH = Matrix.sub(Matrix.eye(this.STATE_DIM), Matrix.mul(K, H_pos));
        let K_R_KT = Matrix.mul(Matrix.mul(K, R_pos), Matrix.transpose(K));
        this.P = Matrix.add(Matrix.mul(Matrix.mul(I_KH, this.P), Matrix.transpose(I_KH)), K_R_KT);
        this._enforceSymmetry();

        if (gnss_vel) {
            let H_v = Matrix.zeros(3, this.STATE_DIM);
            H_v[0][3] = 1; H_v[1][4] = 1; H_v[2][5] = 1;
            
            let R_v = Matrix.eye(3);
            let rv2 = this.r_gnss_vel * this.r_gnss_vel;
            for(let i=0; i<3; i++) R_v[i][i] = rv2;

            let z_v = [[gnss_vel[0]], [gnss_vel[1]], [0]];
            let y_v = Matrix.sub(z_v, Matrix.mul(H_v, this.x));

            let Hv_T = Matrix.transpose(H_v);
            let Sv = Matrix.add(Matrix.mul(Matrix.mul(H_v, this.P), Hv_T), R_v);
            let Sv_inv = Matrix.inv3x3(Sv);

            let Kv = Matrix.mul(Matrix.mul(this.P, Hv_T), Sv_inv);
            this.x = Matrix.add(this.x, Matrix.mul(Kv, y_v));

            let I_KHv = Matrix.sub(Matrix.eye(this.STATE_DIM), Matrix.mul(Kv, H_v));
            let Kv_R_KT = Matrix.mul(Matrix.mul(Kv, R_v), Matrix.transpose(Kv));
            this.P = Matrix.add(Matrix.mul(Matrix.mul(I_KHv, this.P), Matrix.transpose(I_KHv)), Kv_R_KT);
            this._enforceSymmetry();

            let speed = Math.sqrt(gnss_vel[0]*gnss_vel[0] + gnss_vel[1]*gnss_vel[1]);
            if (speed > 2.0) {
                let cog = Math.atan2(gnss_vel[0], gnss_vel[1]);
                let yaw_err = (cog - this.x[8][0] + Math.PI) % (2.0*Math.PI);
                if (yaw_err < 0) yaw_err += 2*Math.PI;
                yaw_err -= Math.PI;
                
                this.x[8][0] = (this.x[8][0] + 0.15*yaw_err + Math.PI) % (2.0*Math.PI);
                if (this.x[8][0] < 0) this.x[8][0] += 2*Math.PI;
                this.x[8][0] -= Math.PI;
            }
        }
        if (this.mode === 'reacq') {
            const correction = this.x.map((row, i) => row[0] - priorX[i][0]);
            correction[8] = Math.atan2(Math.sin(correction[8]), Math.cos(correction[8]));
            const fraction = Math.min(1, 2 / Math.max(Math.hypot(correction[0], correction[1]), 1e-12));
            this.x = priorX.map((row, i) => [row[0] + fraction * correction[i]]);
            this.P = priorP.map((row, i) => row.map((v, j) => (1 - fraction) * v + fraction * this.P[i][j]));
            this._enforceSymmetry();
            const residual = Math.hypot(gnss_pos[0] - this.x[0][0], gnss_pos[1] - this.x[1][0]);
            if (this.consecutive_good_gnss >= 5 && residual <= 3) this.mode = 'gnss_ins';
        }
    }

    updateNhc(yaw_rate = 0.0) {
        let speed = Math.sqrt(this.x[3][0]*this.x[3][0] + this.x[4][0]*this.x[4][0]);
        if (speed < 0.5) return;

        let turn_dilation = 1.0 + 30.0 * (yaw_rate * yaw_rate);
        let r_lat = this.r_nhc_lateral * Math.sqrt(turn_dilation);
        let r_vert = this.r_nhc_vertical;

        let heading = this.x[8][0];
        let cos_h = Math.cos(heading);
        let sin_h = Math.sin(heading);

        let H = Matrix.zeros(2, this.STATE_DIM);
        H[0][3] = cos_h;
        H[0][4] = -sin_h;
        H[1][5] = -1.0;

        let R = [
            [r_lat * r_lat, 0],
            [0, r_vert * r_vert]
        ];

        let z = [[0], [0]];
        let y = Matrix.sub(z, Matrix.mul(H, this.x));

        let H_T = Matrix.transpose(H);
        let S = Matrix.add(Matrix.mul(Matrix.mul(H, this.P), H_T), R);
        let S_inv = Matrix.inv2x2(S);

        let K = Matrix.mul(Matrix.mul(this.P, H_T), S_inv);

        this.x = Matrix.add(this.x, Matrix.mul(K, y));

        let I_KH = Matrix.sub(Matrix.eye(this.STATE_DIM), Matrix.mul(K, H));
        let K_R_KT = Matrix.mul(Matrix.mul(K, R), Matrix.transpose(K));
        this.P = Matrix.add(Matrix.mul(Matrix.mul(I_KH, this.P), Matrix.transpose(I_KH)), K_R_KT);
        this._enforceSymmetry();
    }

    updateZupt(velocity_sigma = 0.01) {
        let H = Matrix.zeros(3, this.STATE_DIM);
        H[0][3] = 1; H[1][4] = 1; H[2][5] = 1;

        let R = Matrix.eye(3);
        let r2 = velocity_sigma * velocity_sigma;
        for(let i=0; i<3; i++) R[i][i] = r2;

        let z = [[0], [0], [0]];
        let y = Matrix.sub(z, Matrix.mul(H, this.x));

        let H_T = Matrix.transpose(H);
        let S = Matrix.add(Matrix.mul(Matrix.mul(H, this.P), H_T), R);
        let S_inv = Matrix.inv3x3(S);

        let K = Matrix.mul(Matrix.mul(this.P, H_T), S_inv);

        this.x = Matrix.add(this.x, Matrix.mul(K, y));

        let I_KH = Matrix.sub(Matrix.eye(this.STATE_DIM), Matrix.mul(K, H));
        let K_R_KT = Matrix.mul(Matrix.mul(K, R), Matrix.transpose(K));
        this.P = Matrix.add(Matrix.mul(Matrix.mul(I_KH, this.P), Matrix.transpose(I_KH)), K_R_KT);
        this._enforceSymmetry();
    }

    setGnssDenied() {
        if (this.mode !== 'dr') {
            this.mode = 'dr';
            this.consecutive_good_gnss = 0;
        }
    }

    getPosition() {
        return [this.x[0][0], this.x[1][0], this.x[2][0]];
    }

    getVelocity() {
        return [this.x[3][0], this.x[4][0], this.x[5][0]];
    }

    getHeading() {
        return this.x[8][0];
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExtendedKalmanFilter;
} else {
    window.ExtendedKalmanFilter = ExtendedKalmanFilter;
}
