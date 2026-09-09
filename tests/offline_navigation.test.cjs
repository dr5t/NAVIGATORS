const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const LocalMap = require('../simulator/local_map.js');
const PedestrianTracker = require('../simulator/engine/pedestrian.js');
const { RoadNetwork, GeometricMapMatcher } = require('../simulator/engine/map_matcher.js');
const root = path.resolve(__dirname, '../simulator');

test('downloaded OSM coordinates round-trip and match in the same frame', () => {
    const map = new LocalMap(JSON.parse(fs.readFileSync(path.join(root, 'data/road_network.json'))));
    const point = map.data.roads[0].points[0];
    const restored = map.toENU(...map.toLatLon(point));
    assert.ok(Math.hypot(restored[0] - point[0], restored[1] - point[1]) < 1e-6);
    assert.ok(map.contains(...map.toLatLon(point)));
    assert.equal(map.contains(map.data.origin.lat + 10, map.data.origin.lon + 10), false);
    assert.throws(() => new LocalMap({ origin: map.data.origin, roads: [] }));
});

test('long OSM segments can be matched far from both endpoints', () => {
    const roads = new RoadNetwork();
    roads.addRoad([[0, 0], [2000, 0]], 'long');
    roads.buildSpatialHash();
    const match = new GeometricMapMatcher(roads, 30).match([1000, 2], Math.PI / 2);
    assert.equal(match.matched_segment.id, 'long_0');
    assert.deepEqual(match.snapped_position, [1000, 0]);
});

function engineFixture() {
    let success;
    const cleared = [];
    const elements = new Map();
    const context = vm.createContext({
        console, Float32Array, URL,
        performance: { now: () => 10000 }, Date,
        navigator: { geolocation: {
            watchPosition: callback => { success = callback; return 42; },
            clearWatch: id => cleared.push(id),
        } },
        document: { baseURI: 'http://localhost/', getElementById: id => {
            if (!elements.has(id)) elements.set(id, { style: {} });
            return elements.get(id);
        } },
        setInterval: () => 1, clearInterval() {},
        window: { addEventListener() {}, removeEventListener() {} },
    });
    for (const file of ['engine/matrix.js', 'engine/ekf.js', 'engine/map_matcher.js', 'engine/alignment.js', 'engine/pedestrian.js', 'device_profiler.js', 'engine/preprocessing.js', 'offline_engine.js']) {
        vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), context);
    }
    const engine = context.window.offlineEngine;
    engine.setLocalMap(new LocalMap(JSON.parse(fs.readFileSync(path.join(root, 'data/road_network.json')))));
    engine.startCapture();
    engine.initialized = true;
    engine.aligner.isAligned = true;
    engine.lastMotionTime = 10;
    engine.currentAccel = [0, 0, 9.81];
    engine.currentGyro = [0, 0, 0.2];
    return { engine, context, cleared, fix: () => success({ timestamp: Date.now(), coords: {
        latitude: 13.0326, longitude: 77.5582, accuracy: 4, speed: 5, heading: 0,
    } }) };
}

function walk(tracker, start, seconds, alpha = 0) {
    for (let i = 0; i < seconds * 50; i++) {
        const time = start + i / 50;
        tracker.orientation({ absolute: true, alpha, beta: 0, gamma: 0 }, time);
        tracker.motion({ x: 0, y: 0, z: 9.81 + 2.5 * Math.sin(i / 50 * 4 * Math.PI) }, time);
    }
}

test('walking uses real fixes without IMU calibration, then steps north/east during GPS loss', () => {
    const walker = new PedestrianTracker(0.7);
    const fix = { timestamp: 1, accuracy: 4, speedKnown: false, headingKnown: false };
    walker.fix([12, 20], fix, 0);
    assert.deepEqual(walker.position, [12, 20]);
    walker.loseGps();
    walk(walker, 1, 3);
    assert.equal(walker.mode, 'dr');
    assert.ok(walker.steps >= 4 && walker.steps <= 7);
    assert.equal(walker.position[0], 12);
    assert.ok(walker.position[1] > 22);
    const north = walker.position[1];
    walk(walker, 4, 2, 270);
    assert.ok(walker.position[0] > 13);
    assert.ok(Math.abs(walker.position[1] - north) < 1e-9);
    assert.equal(walker.hasGps(6), false);
});

test('walking stays still without steps or absolute heading and never invents a GPS origin', () => {
    const walker = new PedestrianTracker();
    walk(walker, 0, 2);
    assert.equal(walker.position, null);
    walker.fix([0, 0], { timestamp: 1, accuracy: 4 }, 2);
    walker.loseGps();
    for (let i = 0; i < 100; i++) {
        walker.orientation({ absolute: true, alpha: 0, beta: 0, gamma: 0 }, 3 + i / 50);
        walker.motion({ x: 0, y: 0, z: 9.81 }, 3 + i / 50);
    }
    assert.deepEqual(walker.position, [0, 0]);
    const relative = new PedestrianTracker();
    relative.orientation({ absolute: false, alpha: 90 }, 0);
    assert.equal(relative.hasHeading(0), false);
    relative.fix([0, 0], { timestamp: 1, accuracy: 4 }, 0);
    for (let i = 0; i < 100; i++) relative.motion({ x: 0, y: 0, z: 9.81 + 3 * Math.sin(i / 50 * 4 * Math.PI) }, i / 50);
    assert.deepEqual(relative.position, [0, 0]);
    relative.orientation({ absolute: true, alpha: 0, beta: 90 }, 3);
    assert.equal(relative.hasHeading(3), false);
});

test('walking recovery rejects duplicate fixes and caps each correction at two meters', () => {
    const walker = new PedestrianTracker();
    walker.fix([0, 0], { timestamp: 1, accuracy: 4 }, 0);
    walker.loseGps();
    for (let i = 2; i <= 12; i++) {
        const before = [...walker.position];
        assert.equal(walker.fix([10, 0], { timestamp: i, accuracy: 4 }, i), true);
        assert.ok(Math.hypot(...walker.position.map((value, j) => value - before[j])) <= 2);
        if (i < 6) assert.equal(walker.mode, 'reacq');
        assert.equal(walker.fix([100, 0], { timestamp: i, accuracy: 4 }, i), false);
    }
    assert.equal(walker.mode, 'gnss_ins');
    assert.deepEqual(walker.position, [10, 0]);
    assert.equal(walker.fix([100, 0], { timestamp: 20, accuracy: 100 }, 20), false);
    assert.throws(() => new PedestrianTracker(0));
});

test('walking outage rejects GPS callbacks and map replacement preserves geographic position', async () => {
    const { engine, fix } = engineFixture();
    engine.stopCapture();
    engine.navigationMode = 'walking';
    engine.startCapture();
    fix();
    await engine.runInferenceLoop();
    const position = engine.localMap.toLatLon(engine.walker.position);
    engine.setGpsOutage(true);
    fix();
    assert.equal(engine.currentGnss, null);
    await engine.runInferenceLoop();
    assert.equal(engine.walker.mode, 'dr');
    const replacement = new LocalMap({ origin: { lat: 13.04, lon: 77.55 }, roads: [{ points: [[0, 0], [10, 0]] }] });
    engine.setLocalMap(replacement);
    assert.deepEqual(replacement.toLatLon(engine.walker.position), position);
});

test('outage stops GPS and ignores pending fixes while EKF continues', async () => {
    const { engine, cleared, fix } = engineFixture();
    fix();
    assert.ok(engine.currentGnss);
    assert.equal(engine.setGpsOutage(true), true);
    assert.deepEqual(cleared, [42]);
    fix();
    assert.equal(engine.currentGnss, null);
    engine.ekf.updateGnss = () => assert.fail('GPS fused during outage');
    let predictions = 0;
    const predict = engine.ekf.predict.bind(engine.ekf);
    engine.ekf.predict = (...args) => { predictions++; return predict(...args); };
    await engine.runInferenceLoop();
    assert.equal(predictions, 1);
    assert.equal(engine.ekf.mode, 'dr');
    assert.equal(engine.gnssAvailable, false);
    engine.setGpsOutage(false);
    assert.equal(engine.currentGnss, null);
    fix();
    assert.ok(engine.currentGnss);
});

test('stale fixes cannot feed the EKF and each fresh fix is fused once', async () => {
    const { engine, fix } = engineFixture();
    fix();
    let updates = 0;
    engine.ekf.updateGnss = () => updates++;
    await engine.runInferenceLoop();
    await engine.runInferenceLoop();
    assert.equal(updates, 1);
    engine.lastGnssTime = 0;
    await engine.runInferenceLoop();
    assert.equal(updates, 1);
    assert.equal(engine.gnssAvailable, false);
});

test('service worker package includes every referenced local runtime file', () => {
    const context = vm.createContext({ self: { addEventListener() {} } });
    vm.runInContext(fs.readFileSync(path.join(root, 'sw.js'), 'utf8'), context);
    const assets = vm.runInContext('ASSETS', context);
    for (const asset of assets) {
        assert.ok(!asset.startsWith('http'), asset);
        assert.ok(fs.existsSync(path.join(root, asset)), `Missing offline asset: ${asset}`);
    }
    const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
    assert.doesNotMatch(html, /(?:src|href)="https?:/);
});


test('disabling GPS during pending AI inference prevents that fix from being fused', async () => {
    const { engine, context, fix } = engineFixture();
    fix();
    engine.sensorBuffer = Array.from({ length: 200 }, () => [0, 0, 9.81, 0, 0, 0.2]);
    context.ort = { Tensor: class {} };
    let finish;
    engine.session = {
        inputNames: ['imu'], outputNames: ['velocity'],
        run: () => new Promise(resolve => { finish = resolve; }),
    };
    engine.ekf.updateGnss = () => assert.fail('Pending GPS fix leaked across outage');
    let aiUpdates = 0;
    engine.ekf._updateAiVelocity = () => aiUpdates++;
    const pending = engine.runInferenceLoop();
    assert.ok(finish);
    engine.setGpsOutage(true);
    finish({ velocity: { data: [1, 2] } });
    await pending;
    assert.equal(aiUpdates, 1);
    assert.equal(engine.gnssAvailable, false);
});

test('reacquisition bounds every full GPS correction until convergence', () => {
    const { engine } = engineFixture();
    const ekf = engine.ekf;
    ekf.x[0][0] = 80;
    for (let i = 0; i < 15; i++) ekf.P[i][i] = 1000;
    ekf.P[0][3] = ekf.P[3][0] = 200;
    ekf.setGnssDenied();
    for (let i = 0; i < 200; i++) {
        const before = ekf.getPosition();
        ekf.updateGnss([0, 0], [0, 1]);
        const after = ekf.getPosition();
        assert.ok(Math.hypot(after[0] - before[0], after[1] - before[1]) <= 2.000001);
        if (i < 5) assert.equal(ekf.mode, 'reacq');
        if (ekf.mode === 'gnss_ins') {
            assert.ok(Math.hypot(after[0], after[1]) <= 3);
            return;
        }
    }
    assert.fail('Recovery did not converge');
});

test('causal IMU filtering removes stationary gravity and retains past-only state', () => {
    const Filter = require('../simulator/engine/preprocessing.js');
    const filter = new Filter();
    assert.deepEqual(filter.step([0, 0, 9.81, 0, 0, 0], 0), [0, 0, 0, 0, 0, 0]);
    const previous = filter.step([1, 0, 9.81, 0, 0, 0], 0.01);
    filter.step([100, 0, 9.81, 0, 0, 0], 0.01);
    assert.ok(previous[0] < 0.5);
});

test('device profiler reports actual sample counts and unavailable measurements', () => {
    const Profiler = require('../simulator/device_profiler.js');
    const profiler = new Profiler();
    profiler.record('tcn', 3);
    profiler.record('tcn', 7);
    const report = profiler.report();
    assert.equal(report.timings.tcn.mean_ms, 5);
    assert.equal(report.timings.tcn.count, 2);
    assert.equal(report.timings.ekf.mean_ms, null);
    assert.equal(report.ram_bytes, null);
});
