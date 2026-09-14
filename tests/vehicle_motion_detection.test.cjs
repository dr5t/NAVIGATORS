const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const LocalMap = require('../simulator/local_map.js');

const root = path.resolve(__dirname, '../simulator');

function setupTestEngine() {
    let successCallback;
    const cleared = [];
    const elements = new Map();
    const context = vm.createContext({
        console,
        Float32Array,
        URL,
        performance: { now: () => 10000 },
        Date,
        navigator: {
            geolocation: {
                watchPosition: callback => { successCallback = callback; return 42; },
                clearWatch: id => cleared.push(id),
            }
        },
        document: {
            baseURI: 'http://localhost/',
            getElementById: id => {
                if (!elements.has(id)) elements.set(id, { style: {} });
                return elements.get(id);
            }
        },
        setInterval: () => 1,
        clearInterval() {},
        window: { addEventListener() {}, removeEventListener() {} },
    });

    for (const file of [
        'engine/matrix.js',
        'engine/ekf.js',
        'engine/map_matcher.js',
        'engine/alignment.js',
        'engine/pedestrian.js',
        'device_profiler.js',
        'engine/preprocessing.js',
        'offline_engine.js'
    ]) {
        vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), context);
    }

    const engine = context.window.offlineEngine;
    engine.setLocalMap(new LocalMap(JSON.parse(fs.readFileSync(path.join(root, 'data/road_network.json')))));
    engine.startCapture();
    engine.initialized = true;
    engine.aligner.isAligned = true;

    // Provide mock ONNX session returning realistic non-zero velocity
    context.ort = { Tensor: class { constructor(type, data, dims) { this.data = data; this.dims = dims; } } };
    engine.session = {
        inputNames: ['imu'],
        outputNames: ['velocity'],
        run: async (feeds) => {
            return {
                velocity: { data: new Float32Array([10.5, 4.2]) } // 10.5 m/s East, 4.2 m/s North
            };
        }
    };
    engine.modelContract = {
        window_size: 200,
        sample_rate_hz: 10,
        output_order: ['east', 'north'],
        mean: [0, 0, 8.93, 0, 0, 0],
        std: [1, 1, 1, 1, 1, 1]
    };

    return {
        engine,
        context,
        sendGnssFix: (speed = 12.0, heading = 0) => {
            if (successCallback) {
                successCallback({
                    timestamp: Date.now(),
                    coords: {
                        latitude: 13.0326,
                        longitude: 77.5582,
                        accuracy: 4,
                        speed: speed,
                        heading: heading
                    }
                });
            }
        }
    };
}

test('Vehicle Motion Transitions: Stationary -> Moving (GNSS) -> GNSS-denied Outage -> Stop (ZUPT) -> Moving (Recovery)', async () => {
    const { engine, sendGnssFix } = setupTestEngine();

    // Fill buffer to 200 samples
    for (let i = 0; i < 200; i++) {
        engine.sensorBuffer.push([0, 0, 9.81, 0, 0, 0]);
        engine.sensorTimes.push(1000 + i * 0.1);
    }

    // ==========================================
    // Phase A: Phone Stationary (on table / parked)
    // ==========================================
    engine.lastMotionTime = 1000;
    engine.currentAccel = [0.01, -0.01, 9.81];
    engine.currentGyro = [0.001, 0.001, -0.001];
    
    // Simulate 20 steps (2 seconds) of stationary state
    for (let t = 0; t < 20; t++) {
        engine.lastMotionTime = 1000 + t * 0.1;
        await engine.processInferenceStep();
    }

    assert.equal(engine.motionDetector.motionState, 'STATIONARY', 'Phase A: Motion detector should detect STATIONARY');
    assert.equal(engine.zuptActive, true, 'Phase A: ZUPT should be active when stationary');
    assert.equal(engine.pendingUIData.ai_active, false, 'Phase A: AI inference must be gated during stationary/ZUPT');

    // ==========================================
    // Phase B: Moving Vehicle with GNSS Available
    // ==========================================
    sendGnssFix(13.5); // 13.5 m/s (~48.6 km/h)
    engine.currentAccel = [0.4, 0.2, 10.1]; // Engine + road vibrations
    engine.currentGyro = [0.02, 0.03, 0.05];

    for (let t = 20; t < 30; t++) {
        engine.lastMotionTime = 1000 + t * 0.1;
        await engine.processInferenceStep();
    }

    assert.equal(engine.motionDetector.motionState, 'MOVING', 'Phase B: Motion detector should detect MOVING');
    assert.equal(engine.zuptActive, false, 'Phase B: ZUPT must NOT be active while vehicle is moving');
    assert.equal(engine.pendingUIData.ai_active, true, 'Phase B: AI inference must execute while moving');
    assert.ok(engine.pendingUIData.ai_predicted_speed > 10.0, 'Phase B: AI predicts realistic non-zero speed');
    assert.equal(engine.gnssAvailable, true, 'Phase B: GNSS available');

    // ==========================================
    // Phase C: Vehicle Enters GNSS-Denied Condition (Tunnel)
    // ==========================================
    engine.setGpsOutage(true);
    assert.equal(engine.gnssAvailable, false, 'Phase C: GNSS is now unavailable (tunnel outage)');

    let aiVelocityFusedCount = 0;
    const origUpdateAiVelocity = engine.ekf._updateAiVelocity.bind(engine.ekf);
    engine.ekf._updateAiVelocity = (vel) => {
        aiVelocityFusedCount++;
        origUpdateAiVelocity(vel);
    };

    // Continue driving inside tunnel with road dynamics
    for (let t = 30; t < 45; t++) {
        engine.lastMotionTime = 1000 + t * 0.1;
        engine.currentAccel = [0.3 * Math.sin(t), 0.2 * Math.cos(t), 9.81 + 0.3 * Math.sin(2 * t)];
        engine.currentGyro = [0.03, 0.02, 0.04];
        await engine.processInferenceStep();
    }

    assert.equal(engine.motionDetector.motionState, 'MOVING', 'Phase C: Vehicle must remain MOVING in GNSS-denied condition');
    assert.equal(engine.zuptActive, false, 'Phase C: ZUPT must remain inactive during driving');
    assert.equal(engine.pendingUIData.ai_active, true, 'Phase C: AI inference must continue running in outage');
    assert.ok(aiVelocityFusedCount > 10, 'Phase C: AI velocity must be continuously fused into EKF during GNSS outage');
    assert.equal(engine.ekf.mode, 'dr', 'Phase C: EKF should be in Dead Reckoning mode');

    // ==========================================
    // Phase D: Vehicle Stops (Inside Tunnel)
    // ==========================================
    // Vehicle brakes and comes to a full stop
    engine.currentAccel = [0.0, 0.0, 9.81];
    engine.currentGyro = [0.001, 0.001, 0.001];

    for (let t = 45; t < 70; t++) { // 2.5 seconds of quiet stop
        engine.lastMotionTime = 1000 + t * 0.1;
        await engine.processInferenceStep();
    }

    assert.equal(engine.motionDetector.motionState, 'STATIONARY', 'Phase D: Vehicle stop must be detected');
    assert.equal(engine.zuptActive, true, 'Phase D: ZUPT should activate at stop');
    assert.equal(engine.pendingUIData.ai_active, false, 'Phase D: AI inference gated at stop');

    // ==========================================
    // Phase E: Vehicle Starts Moving Again (Recovery without App Restart)
    // ==========================================
    // Driver presses throttle, vehicle accelerates!
    for (let t = 70; t < 80; t++) {
        engine.lastMotionTime = 1000 + t * 0.1;
        // Dynamic acceleration from motor throttle + vibration
        engine.currentAccel = [0.8, 0.3, 10.4];
        engine.currentGyro = [0.04, 0.05, 0.06];
        await engine.processInferenceStep();
    }

    assert.equal(engine.motionDetector.motionState, 'MOVING', 'Phase E: Vehicle must transition STATIONARY -> MOVING automatically');
    assert.equal(engine.zuptActive, false, 'Phase E: ZUPT must deactivate automatically');
    assert.equal(engine.pendingUIData.ai_active, true, 'Phase E: AI inference must automatically resume without restart');
    assert.ok(engine.pendingUIData.ai_predicted_speed > 10.0, 'Phase E: AI produces valid speed after recovery');
});
