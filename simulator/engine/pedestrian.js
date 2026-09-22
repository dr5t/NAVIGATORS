
class PedestrianTracker {
    constructor(stepLength = 0.7) {
        if (!Number.isFinite(stepLength) || stepLength < 0.3 || stepLength > 1.2) throw new Error('Step length must be between 0.3 and 1.2 meters.');
        this.stepLength = stepLength;
        this.position = null;
        this.heading = null;
        this.headingTime = -Infinity;
        this.lastMotion = -Infinity;
        this.lastStep = -Infinity;
        this.lastFix = -Infinity;
        this.lastFixStamp = -Infinity;
        this.gravity = 9.81;
        this.filtered = 0;
        this.armed = true;
        this.steps = 0;
        this.stepSpeed = 0;
        this.gpsSpeed = null;
        this.accuracy = null;
        this.mode = 'gnss_ins';
        this.recoveryFixes = 0;
    }

    hasHeading(now) { return now - this.headingTime <= 2; }
    hasGps(now) { return now - this.lastFix <= 3; }

    orientation(event, now) {
        
        if ((Number.isFinite(event.beta) && Math.abs(event.beta) > 45) ||
            (Number.isFinite(event.gamma) && Math.abs(event.gamma) > 45)) {
            this.headingTime = -Infinity;
            return;
        }
        let degrees = null;
        if (Number.isFinite(event.webkitCompassHeading)) {
            if (Number.isFinite(event.webkitCompassAccuracy) && (event.webkitCompassAccuracy < 0 || event.webkitCompassAccuracy > 30)) return;
            degrees = event.webkitCompassHeading;
        } else if (event.absolute === true && Number.isFinite(event.alpha)) degrees = 360 - event.alpha;
        if (degrees === null) return;
        this.heading = ((degrees % 360 + 360) % 360) * Math.PI / 180;
        this.headingTime = now;
    }

    motion(accel, now) {
        if (!accel || ![accel.x, accel.y, accel.z].every(Number.isFinite)) return false;
        const dt = now - this.lastMotion;
        if (dt <= 0) return false;
        this.lastMotion = now;
        const magnitude = Math.hypot(accel.x, accel.y, accel.z);
        if (dt > 1) {
            this.gravity = magnitude;
            this.filtered = 0;
            this.armed = true;
            return false;
        }
        this.gravity += dt / (0.6 + dt) * (magnitude - this.gravity);
        this.filtered += dt / (0.06 + dt) * (magnitude - this.gravity - this.filtered);
        if (this.filtered < 0.2) this.armed = true;
        if (!this.armed || this.filtered < 1 || now - this.lastStep < 0.32) return false;
        this.armed = false;
        this.stepSpeed = this.stepLength / Math.max(0.32, Math.min(1, now - this.lastStep));
        this.lastStep = now;
        this.steps++;
        if (!this.hasGps(now)) this.loseGps();
        if (this.position && this.hasHeading(now)) {
            this.position[0] += this.stepLength * Math.sin(this.heading);
            this.position[1] += this.stepLength * Math.cos(this.heading);
        }
        return true;
    }

    loseGps() {
        this.lastFix = -Infinity;
        if (this.position) this.mode = 'dr';
        this.recoveryFixes = 0;
    }

    fix(point, fix, now) {
        if (!point.every(Number.isFinite) || !Number.isFinite(fix.timestamp) || fix.timestamp <= this.lastFixStamp ||
            !Number.isFinite(fix.accuracy) || fix.accuracy <= 0 || fix.accuracy > 50) return false;
        if (this.position && !this.hasGps(now)) this.mode = 'dr';
        if (!this.position || this.mode === 'gnss_ins') {
            this.position = [...point];
            this.mode = 'gnss_ins';
        } else {
            const delta = point.map((value, i) => value - this.position[i]);
            const distance = Math.hypot(...delta);
            const fraction = distance ? Math.min(1, 2 / distance) : 1;
            this.position = this.position.map((value, i) => value + delta[i] * fraction);
            this.mode = ++this.recoveryFixes >= 5 && distance <= 3 ? 'gnss_ins' : 'reacq';
        }
        this.lastFix = now;
        this.lastFixStamp = fix.timestamp;
        this.accuracy = fix.accuracy;
        this.gpsSpeed = fix.speedKnown ? fix.speed : null;
        if (!this.hasHeading(now) && fix.headingKnown) this.heading = fix.heading;
        return true;
    }

    speed(now) {
        if (this.hasGps(now) && Number.isFinite(this.gpsSpeed)) return this.gpsSpeed;
        if (!this.hasHeading(now) || now - this.lastMotion > 1) return null;
        return now - this.lastStep <= 1.2 ? this.stepSpeed : 0;
    }
}

if (typeof module !== 'undefined') module.exports = PedestrianTracker;
