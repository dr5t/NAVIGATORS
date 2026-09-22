

import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { resolve, join, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../simulator/', import.meta.url));
const profile = await mkdtemp(join(tmpdir(), 'navigators-ui-'));
const screenshots = process.env.UI_SCREENSHOTS || '/private/tmp/navigators-ui-review';
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.wasm': 'application/wasm', '.json': 'application/json', '.css': 'text/css',
    '.svg': 'image/svg+xml', '.png': 'image/png' };
const server = createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    const file = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
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
    await mkdir(screenshots, { recursive: true });
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
    socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl);
    await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
    let id = 0;
    const pending = new Map(), exceptions = [], requests = [];
    socket.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (message.id) {
            const request = pending.get(message.id);
            if (!request) return;
            pending.delete(message.id);
            clearTimeout(request.timeout);
            if (message.error) request.reject(new Error(JSON.stringify(message.error)));
            else request.resolve(message.result);
        }
        if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
        if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
        if (message.method === 'Fetch.requestPaused') {
            const osm = message.params.request.url.includes('overpass-api.de');
            const body = osm ? Buffer.from(JSON.stringify({ elements: [{ type: 'way', id: 1, tags: { name: 'Synthetic map test fixture' },
                geometry: [{ lat: 13.0326, lon: 77.5582 }, { lat: 13.033, lon: 77.5582 }] }] })).toString('base64')
                : 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=';
            send('Fetch.fulfillRequest', { requestId: message.params.requestId, responseCode: 200,
                responseHeaders: [{ name: 'Content-Type', value: osm ? 'application/json' : 'image/png' },
                    { name: 'Access-Control-Allow-Origin', value: '*' }], body }).catch(error => {
                
                if (!error.message.includes('Invalid InterceptionId')) exceptions.push(error.message);
            });
        }
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
    const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
    const screenshot = async name => {
        const metrics = await send('Page.getLayoutMetrics');
        const content = metrics.cssContentSize;
        const result = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true,
            clip: { x: 0, y: 0, width: content.width, height: content.height, scale: 1 } });
        await writeFile(join(screenshots, name), Buffer.from(result.data, 'base64'));
    };
    const noOverflow = async label => {
        const sizes = await evaluate(`({ document: document.documentElement.scrollWidth, viewport: document.documentElement.clientWidth })`);
        assert.ok(sizes.document <= sizes.viewport + 1, `${label}: horizontal overflow ${JSON.stringify(sizes)}`);
    };
    const select = async view => {
        await click(`[data-view="${view}"]`);
        assert.equal(await evaluate('state.view'), view);
        assert.equal(await evaluate(`document.querySelector('[aria-current="page"]').dataset.view`), view);
        await noOverflow(view);
    };
    const importReport = async (name, content, inputId = 'experimentFile') => {
        await evaluate(`(() => {
            const transfer = new DataTransfer();
            transfer.items.add(new File([${JSON.stringify(content)}], ${JSON.stringify(name)}, { type: 'application/json' }));
            const input = document.getElementById(${JSON.stringify(inputId)});
            input.files = transfer.files;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        })()`);
    };
    await send('Runtime.enable');
    await send('Network.enable');
    await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: origin });
    await until(() => evaluate("document.getElementById('offlineStatus')?.textContent.includes('Offline files ready') && !!state.data && !!state.localMap"), 'Workspace did not become ready', 60000);
    await noOverflow('desktop navigation');
    assert.equal(await evaluate('state.view'), 'console');
    assert.deepEqual(await evaluate(`({ mode: document.querySelector('#navModeIndicator .mode-label').textContent,
        speed: document.getElementById('speedValue').textContent,
        metricsHidden: document.getElementById('metricsCard').hidden,
        playbackHidden: document.getElementById('playbackControls').hidden })`),
    { mode: 'STANDBY', speed: '-', metricsHidden: true, playbackHidden: true });
    await screenshot('desktop.png');
    await click('#btnGuide');
    assert.equal(await evaluate("document.getElementById('guideDialog').open"), true);
    await click('#btnCloseGuide');
    assert.equal(await evaluate("document.getElementById('guideDialog').open"), false);
    await click('#btnFollow');
    assert.equal(await evaluate('state.followPosition'), false);
    assert.equal(await evaluate("document.getElementById('btnFollow').getAttribute('aria-pressed')"), 'false');
    await click('#btnFollow');
    assert.equal(await evaluate('state.followPosition'), true);
    console.log('Desktop: clean standby, local map, guide and follow control passed.');

    await select('replay');
    const target = Math.min(137, await evaluate('state.data.data.timestamps.length - 20'));
    assert.ok(target > 0);
    await evaluate(`(() => { const input = document.getElementById('timeline'); input.value = ${target}; input.dispatchEvent(new Event('input', { bubbles: true })); })()`);
    assert.equal(await evaluate('state.currentIndex'), target, 'Seeking must retain the exact requested frame');
    const playbackStart = await evaluate(`(() => {
        const slider = document.getElementById('speedSlider'); slider.value = '2'; slider.dispatchEvent(new Event('input', { bubbles: true }));
        const timestamp = state.data.data.timestamps[state.currentIndex];
        document.getElementById('btnPlay').click();
        return { timestamp, wall: performance.now() };
    })()`);
    await delay(1100);
    const playbackEnd = await evaluate(`(() => { document.getElementById('btnPause').click(); return { timestamp: state.data.data.timestamps[state.currentIndex], wall: performance.now() }; })()`);
    const actualSeconds = playbackEnd.timestamp - playbackStart.timestamp;
    const expectedSeconds = (playbackEnd.wall - playbackStart.wall) / 1000 * 2;
    assert.ok(Math.abs(actualSeconds - expectedSeconds) < 0.35,
        `Playback advanced ${actualSeconds.toFixed(3)}s; expected about ${expectedSeconds.toFixed(3)}s`);
    console.log(`Replay: exact seek passed; 2× playback advanced ${actualSeconds.toFixed(2)}s over ${(expectedSeconds / 2).toFixed(2)}s wall time.`);
    const trajectory = await evaluate('JSON.stringify(state.data)');
    await importReport('saved-example.json', trajectory, 'replayFile');
    await until(() => evaluate("state.replayName === 'saved-example.json'"), 'Saved trajectory did not load');
    assert.equal(await evaluate('state.currentIndex'), 0);
    assert.match(await evaluate("document.getElementById('replayNotice').textContent"), /provenance unverified/);
    await importReport('raw-trip.json', '{"data":{"timestamps":[0,1]}}', 'replayFile');
    await until(() => evaluate("document.getElementById('replayNotice').classList.contains('error')"), 'Invalid trajectory did not show an error');
    assert.equal(await evaluate('state.replayName'), 'saved-example.json');
    console.log('Trajectory import: saved playback loads; raw recordings are rejected without losing the current trajectory.');
    const evaluated = process.env.REPLAY_TRAJECTORY
        ? JSON.parse(await readFile(process.env.REPLAY_TRAJECTORY, 'utf8')) : JSON.parse(trajectory);
    if (!process.env.REPLAY_TRAJECTORY) {
        evaluated.metadata = { source: 'replay.py', configuration: { ai: false, ekf: false, nhc: false, zupt: false, map_matching: false },
            dataset: { metadata: { provenance: 'synthetic test fixture' } } };
        evaluated.data.timestamps = evaluated.data.timestamps.map(time => time + 5);
        evaluated.data.true_lat_lon[0] = null;
        for (const key of ['position_error', 'confidence', 'dr_drift_percent', 'zupt_active']) evaluated.data[key][0] = null;
    }
    await importReport('evaluated_trajectory.json', JSON.stringify(evaluated), 'replayFile');
    await until(() => evaluate("state.replayName === 'evaluated_trajectory.json'"), 'Evaluated trajectory failed to import');
    assert.match(await evaluate("document.getElementById('replayNotice').textContent"), /synthetic test fixture/);
    const missingIndex = evaluated.data.true_lat_lon.findIndex(point => point === null);
    assert.ok(missingIndex >= 0, 'Test trajectory must exercise missing GPS references');
    await evaluate(`seekFrame(${missingIndex})`);
    assert.equal(await evaluate('state.truthMarker.options.opacity'), 0);
    for (const id of ['posErrorValue', 'confidenceValue', 'driftValue']) {
        assert.equal(await evaluate(`document.getElementById('${id}').textContent`), '-');
    }
    assert.equal(await evaluate("document.getElementById('statusAI').textContent"), 'Off');
    assert.equal(await evaluate("document.getElementById('statusEKF').textContent"), 'Off');
    assert.ok((await evaluate("document.getElementById('timeLabel').textContent")).startsWith(evaluated.data.timestamps[missingIndex].toFixed(1)));
    await click('#btnPlay');
    await delay(250);
    await click('#btnPause');
    console.log('Evaluated replay: original timestamps, disabled components, and missing reference/metrics render correctly.');

    await select('experiments');
    assert.equal(await evaluate("document.querySelectorAll('#experimentRows tr').length"), 7);
    assert.equal(await evaluate("document.querySelectorAll('.experiment-table th').length"), 7);
    assert.ok(await evaluate("Array.from(document.querySelectorAll('#experimentRows tr')).every(row => row.lastElementChild.textContent === 'Not run')"));
    const fixture = ['A', 'G'].map((mode, index) => ({
        mode, status: 'completed',
        dataset: { sha256: 'ui-smoke-fixture-only', metadata: { provenance: 'UI TEST FIXTURE - not a real trip' } },
        outage: { start_s: 20, requested_duration_s: 60 },
        machine: { label: 'UI smoke fixture; no performance claim' },
        metrics: { outage: { mean_position_error_m: index ? 2.25 : 8.5,
            final_position_error_m: index ? 3.5 : 11.25, velocity_rmse_mps: index ? 0.5 : 1.25 } },
    }));
    await importReport('ui-fixture_results.json', JSON.stringify(fixture));
    await until(() => evaluate("document.getElementById('experimentNotice').textContent.includes('2 completed / 2')"), 'Fixture report failed to import');
    assert.match(await evaluate("document.getElementById('experimentRows').textContent"), /8\.50/);
    assert.match(await evaluate("document.getElementById('experimentProvenance').textContent"), /UI TEST FIXTURE/);
    await screenshot('experiments.png');
    await importReport('malformed-fixture.json', '{ broken JSON');
    await until(() => evaluate("document.getElementById('experimentNotice').classList.contains('error')"), 'Malformed JSON did not show an error');
    assert.match(await evaluate("document.getElementById('experimentRows').textContent"), /8\.50/, 'Failed imports must preserve previous results');
    console.log('Experiments: seven empty configurations, fixture import, provenance and malformed JSON recovery passed.');

    await select('performance');
    assert.equal(await evaluate("document.querySelectorAll('.performance-card').length"), 4);
    assert.ok(await evaluate("Array.from(document.querySelectorAll('.performance-card strong')).every(value => value.firstChild.textContent === '-')"));
    assert.match(await evaluate("document.getElementById('deviceIdentity').textContent"), /No navigation steps measured/);
    console.log('Device timings: unmeasured results remain placeholders.');

    await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await select('console');
    await evaluate(`window.updateConsoleTelemetry({ status: 'waiting_for_imu' })`);
    assert.notEqual(await evaluate("getComputedStyle(document.getElementById('sessionHint')).display"), 'none');
    assert.match(await evaluate("document.getElementById('sessionHint').textContent"), /Waiting for motion/);
    await evaluate('window.showStandby()');
    await evaluate('window.scrollTo(0, 0)');
    await delay(200);
    await noOverflow('mobile navigation');
    const guideVisible = await evaluate(`(() => {
        const button = document.getElementById('btnGuide'), rect = button.getBoundingClientRect();
        const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
        return rect.width >= 24 && rect.height >= 24 && rect.left >= 0 && rect.right <= innerWidth && !!hit && button.contains(hit);
    })()`);
    assert.equal(guideVisible, true, 'Mobile field guide must be visible and reachable');
    await click('#btnGuide');
    assert.equal(await evaluate("document.getElementById('guideDialog').open"), true);
    await click('#btnCloseGuide');
    await screenshot('mobile.png');
    for (const view of ['replay', 'experiments', 'performance']) await select(view);
    await evaluate('window.offlineEngine.isCapturing = true');
    await select('performance');
    assert.equal(await evaluate("document.getElementById('btnStartLive').hidden"), false, 'Stop control must remain available in report views');
    await evaluate('window.offlineEngine.isCapturing = false');
    await select('console');
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    assert.deepEqual(requests.filter(url => /^https?:/.test(url) && !url.startsWith(origin + '/')), []);
    console.log('Mobile 390×844: all views fit, field guide opens, and navigation remains usable.');
    await evaluate(`Object.defineProperty(navigator, 'geolocation', { configurable: true, value: {
        watchPosition: callback => { window.walkingFix = callback; return 17; }, clearWatch: () => {} } })`);
    await click('#btnStartLive');
    await until(() => evaluate('offlineEngine.isCapturing && !!offlineEngine.walker'), 'Walking engine did not start');
    assert.equal(await evaluate('offlineEngine.session'), null, 'Walking must not load the vehicle model');
    await evaluate(`window.walkingFix({ timestamp: Date.now(), coords: { latitude: 13.0326, longitude: 77.5582, accuracy: 4, speed: null, heading: null } }); offlineEngine.runInferenceLoop()`);
    assert.equal(await evaluate("document.querySelector('#navModeIndicator .mode-label').textContent"), 'GPS · WALKING');
    await click('#btnGpsOutage');
    const moved = await evaluate(`(async () => {
        const walker = offlineEngine.walker, before = [...walker.position], start = performance.now() / 1000;
        for (let i = 0; i < 100; i++) {
            walker.orientation({ absolute: true, alpha: 270, beta: 0, gamma: 0 }, start + i / 50);
            walker.motion({ x: 0, y: 0, z: 9.81 + 3 * Math.sin(i / 50 * 4 * Math.PI) }, start + i / 50);
        }
        await offlineEngine.runInferenceLoop();
        return walker.position[0] - before[0];
    })()`);
    assert.ok(moved > 1, 'Controlled steps must move the estimate east while GPS is disabled');
    assert.match(await evaluate("document.getElementById('positionCoordinates').textContent"), /13\.032600, 77\.558/);
    await click('#btnStartLive');
    console.log('Walking browser flow: GPS initializes immediately; sensor steps move the estimate during outage without loading AI.');

    
    await send('Fetch.enable', { patterns: [{ urlPattern: 'https://tile.openstreetmap.org/*' }, { urlPattern: 'https://overpass-api.de/*' }] });
    await evaluate("document.getElementById('mapSource').value = 'online'; window.updateMapSource()");
    await until(() => evaluate('state.map.hasLayer(state.onlineTiles) && !!document.querySelector(".leaflet-tile-loaded")'), 'Online map tiles did not render');
    await click('#btnSaveArea');
    await until(() => evaluate("document.getElementById('mapNetworkStatus').textContent.includes('Area saved')"), 'Area download failed');
    assert.equal(await evaluate("state.localMap.data.roads[0].name"), 'Synthetic map test fixture');
    await send('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 });
    await until(() => evaluate('!state.map.hasLayer(state.onlineTiles) && state.map.hasLayer(state.localMap.layer)'), 'Offline fallback did not activate');
    await send('Page.reload', { ignoreCache: true });
    await until(() => evaluate("state.localMap?.data.roads[0].name === 'Synthetic map test fixture'"), 'Downloaded streets did not survive offline reload');
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    console.log('Map integration: online tiles, area download, automatic offline fallback, and saved-area offline reload passed with controlled provider fixtures.');
    console.log(`Screenshots saved to ${screenshots}; imported metrics are test fixtures only.`);
} finally {
    socket?.close();
    if (browser && browser.exitCode === null) {
        browser.kill();
        await new Promise(resolve => browser.once('exit', resolve));
    }
    await new Promise(resolve => server.close(resolve));
    await rm(profile, { recursive: true, force: true });
}
