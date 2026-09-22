
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { resolve, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../simulator/', import.meta.url));
const profile = await mkdtemp(join(tmpdir(), 'navigators-offline-'));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.wasm': 'application/wasm', '.json': 'application/json', '.css': 'text/css',
    '.svg': 'image/svg+xml', '.png': 'image/png' };
const server = createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    const file = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
    try {
        res.setHeader('Content-Type', mime[extname(file)] || 'application/octet-stream');
        res.end(await readFile(file));
    } catch { res.writeHead(404).end(); }
});
let browser, socket;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, message, timeout = 30000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        try { const result = await fn(); if (result) return result; } catch {}
        await delay(100);
    }
    throw new Error(message);
}
try {
    await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', resolve);
    });
    const origin = `http://127.0.0.1:${server.address().port}`;
    browser = spawn(process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
        '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
        '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
    ], { stdio: 'ignore' });
    browser.on('error', error => console.error(error.message));
    const port = await until(async () => (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0], 'Chrome did not start');
    const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    socket = new WebSocket(pages.find(p => p.type === 'page').webSocketDebuggerUrl);
    await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
    let id = 0;
    const pending = new Map(), requests = [], exceptions = [];
    socket.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (message.id) {
            const request = pending.get(message.id);
            pending.delete(message.id);
            if (!request) return;
            clearTimeout(request.timeout);
            if (message.error) request.reject(new Error(JSON.stringify(message.error)));
            else request.resolve(message.result);
        }
        if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
        if (message.method === 'Page.javascriptDialogOpening') send('Page.handleJavaScriptDialog', { accept: false });
        if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    });
    const send = (method, params = {}) => new Promise((resolve, reject) => {
        const requestId = ++id;
        const timeout = setTimeout(() => {
            pending.delete(requestId);
            reject(new Error(`Browser command timed out: ${method}`));
        }, 30000);
        pending.set(requestId, { resolve, reject, timeout });
        socket.send(JSON.stringify({ id: requestId, method, params }));
    });
    const evaluate = async expression => {
        const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
        return result.result.value;
    };
    await send('Runtime.enable');
    await send('Network.enable');
    await send('Page.enable');
    await send('Page.navigate', { url: origin });
    await until(() => evaluate("document.getElementById('offlineStatus')?.textContent.includes('Offline files ready')"), 'Complete cache never became ready', 60000);
    assert.ok(await evaluate('state.localMap.data.roads.length > 0'));
    console.log('Complete offline package installed; local OSM roads rendered.');
    await send('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 });
    await send('Page.reload', { ignoreCache: true });
    await until(() => evaluate("document.getElementById('offlineStatus')?.textContent.includes('Offline files ready') && !!state.localMap"), 'Offline reload failed');
    assert.ok(await evaluate('document.querySelectorAll("#map canvas").length > 0'));
    const inference = await evaluate(`(async () => {
        const loaded = await offlineEngine.initModel();
        if (!loaded) throw new Error('Offline model failed to load');
        const session = offlineEngine.session;
        const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(1200), [1, 200, 6]) });
        return Array.from(result[session.outputNames[0]].data);
    })()`);
    assert.equal(inference.length, 2);
    assert.ok(inference.every(Number.isFinite));
    console.log('Network disabled: page reloaded, map rendered, real ONNX/WASM inference passed.', inference);
    
    const outage = await evaluate(`(async () => {
        const engine = offlineEngine;
        engine.isCapturing = true;
        engine.initialized = true;
        engine.aligner.isAligned = true;
        engine.currentAccel = [0.3, 0.1, 9.81];
        engine.currentGyro = [0, 0, 0.2];
        engine.sensorBuffer = Array.from({length: 200}, () => [0.3, 0.1, 9.81, 0, 0, 0.2]);
        engine.lastTime = performance.now() / 1000 - 0.1;
        engine.lastMotionTime = performance.now() / 1000;
        let gpsUpdates = 0;
        window.restoreTestGnss = engine.ekf.updateGnss.bind(engine.ekf);
        engine.ekf.updateGnss = () => gpsUpdates++;
        engine.setGpsOutage(true);
        await engine.runInferenceLoop();
        return { gpsUpdates, mode: engine.ekf.mode, gps: engine.gnssAvailable,
            position: engine.ekf.getPosition(), status: document.getElementById('edgeStatus').textContent };
    })()`);
    assert.equal(outage.gpsUpdates, 0);
    assert.equal(outage.mode, 'dr');
    assert.equal(outage.gps, false);
    assert.ok(outage.position.every(Number.isFinite));
    assert.match(outage.status, /AI active/);
    const recovery = await evaluate(`(async () => {
        const engine = offlineEngine;
        engine.ekf.updateGnss = window.restoreTestGnss;
        engine.ekf.x[0][0] = 40;
        engine.setGpsOutage(false);
        const modes = [];
        for (let i = 0; i < 100; i++) {
            engine.currentGnss = { lat: engine.refLat, lon: engine.refLon, accuracy: 3, speed: 0, heading: 0 };
            engine.lastGnssTime = engine.lastMotionTime = performance.now() / 1000;
            engine.lastTime = performance.now() / 1000 - 0.1;
            await engine.runInferenceLoop();
            modes.push(engine.ekf.mode);
            if (engine.ekf.mode === 'gnss_ins') break;
        }
        return { modes, profile: engine.profiler.report(), label: document.querySelector('#navModeIndicator .mode-label').textContent };
    })()`);
    assert.equal(recovery.modes[0], 'reacq');
    assert.equal(recovery.modes.at(-1), 'gnss_ins');
    assert.ok(recovery.profile.timings.tcn.count > 0);
    assert.ok(recovery.profile.timings.ekf.count > 0);
    assert.ok(recovery.profile.timings.map_matching.count > 0);
    assert.ok(recovery.profile.timings.total_loop.count > 0);
    assert.ok(recovery.profile.model_size_bytes > 0);
    assert.equal(recovery.profile.ram_bytes, null);
    console.log('Recovery converged through REACQUISITION to NORMAL; browser timing report contains measured component calls.');
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    assert.deepEqual(requests.filter(url => /^https?:/.test(url) && !url.startsWith(origin + '/')), []);
    console.log('GPS outage: real IMU pipeline + AI + EKF step passed with zero GPS updates and zero external requests.');
} finally {
    socket?.close();
    if (browser && browser.exitCode === null) {
        browser.kill();
        await new Promise(resolve => browser.once('exit', resolve));
    }
    await new Promise(resolve => server.close(resolve));
    await rm(profile, { recursive: true, force: true });
}
