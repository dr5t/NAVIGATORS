/** Workspace UI. Reports are read locally; no uploads or generated accuracy figures. */
const workspaceViews = {
    console: ['Navigation', 'Navigation console', 'SENSOR FUSION / OFFLINE NAVIGATION', 'A continuous position. Even without a signal.'],
    replay: ['Saved playback', 'Replay the journey', 'SAVED EXAMPLE / TRAJECTORY PLAYBACK', 'Inspect a saved trajectory, one moment at a time.'],
    experiments: ['Experiments', 'Show the evidence', 'EVALUATION / COMPONENT ABLATION', 'The same recording. A clear view of every component.'],
    performance: ['Device timings', 'Know the runtime', 'MEASUREMENT / ON-DEVICE EXECUTION', 'Measured here. Labeled for the device that ran it.'],
};
const configurationNames = {
    A: 'Raw IMU', B: 'Filtered IMU', C: 'AI velocity', D: 'AI + EKF',
    E: 'AI + EKF + NHC', F: 'AI + EKF + NHC + ZUPT', G: 'Full system + map matching',
};
const byId = id => document.getElementById(id);
const readableNumber = (value, suffix = '') => Number.isFinite(value) ? `${value.toFixed(2)}${suffix}` : '—';
const experimentMetrics = ['mean_position_error_m', 'final_position_error_m', 'max_position_error_m', 'velocity_rmse_mps', 'heading_rmse_deg'];

function fitLocalArea() {
    state.localMap?.centerView(state.map);
}

window.showStandby = (clearTrack = false) => {
    for (const id of ['speedValue', 'headingValue', 'posErrorValue', 'driftValue', 'confidenceValue', 'driftBadge']) {
        byId(id).textContent = '—';
        byId(id).style.color = '';
    }
    byId('posErrorBar').style.width = byId('confidenceBar').style.width = '0%';
    byId('gnssStatusText').textContent = 'WAITING';
    byId('gnssStatusText').style.color = '';
    byId('gnssLed').className = 'status-led';
    byId('signalBars').className = 'signal-bars';
    byId('navModeIndicator').className = 'nav-mode-indicator';
    byId('navModeIndicator').querySelector('.mode-label').textContent = 'STANDBY';
    byId('sessionHint').textContent = 'Waiting to start a session';
    byId('gnssDescription').textContent = 'Acquire a fix to initialize position.';
    byId('btnGpsOutage').disabled = true;
    byId('btnGpsOutage').setAttribute('aria-pressed', 'false');
    byId('btnGpsOutage').textContent = 'Simulate GNSS outage';
    document.querySelectorAll('.run-sequence li').forEach(item => item.classList.remove('active'));
    ['statusIMU', 'statusAI', 'statusEKF', 'statusConstraints', 'statusMap'].forEach(id => {
        byId(id).textContent = 'Waiting';
        byId(id).className = '';
    });
    if (clearTrack) {
        state.estimatedCoords = [];
        state.truthCoords = [];
        state.estimatedLine.setLatLngs([]);
        state.truthLine.setLatLngs([]);
        state.vehicleMarker.setOpacity(0);
        state.truthMarker.setOpacity(0);
        state.gnssZones.forEach(zone => state.map.removeLayer(zone));
        state.gnssZones = [];
        fitLocalArea();
        if (state.localMap) byId('mapStatus').textContent = `Local OSM map · ${state.localMap.data.roads.length} roads`;
    }
};

window.selectWorkspace = view => {
    if (!workspaceViews[view]) return;
    if (view === 'replay' && (window.offlineEngine.isCapturing || byId('btnStartLive').getAttribute('aria-busy') === 'true')) {
        byId('sessionHint').textContent = 'Stop the live engine before opening playback';
        byId('btnStartLive').focus();
        return;
    }
    const previous = state.view;
    pause();
    state.view = view;
    document.querySelectorAll('[data-view]').forEach(button => {
        button.classList.toggle('selected', button.dataset.view === view);
        if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
    const [name, title, eyebrow, description] = workspaceViews[view];
    byId('breadcrumbView').textContent = name;
    byId('pageTitle').replaceChildren(document.createTextNode(title), Object.assign(document.createElement('span'), { textContent: '.' }));
    byId('pageEyebrow').textContent = eyebrow;
    byId('pageDescription').textContent = description;
    byId('navigationWorkspace').hidden = view === 'experiments' || view === 'performance';
    byId('experimentsWorkspace').hidden = view !== 'experiments';
    byId('performanceWorkspace').hidden = view !== 'performance';
    byId('playbackControls').hidden = view !== 'replay';
    byId('replaySource').hidden = view !== 'replay';
    byId('metricsCard').hidden = view !== 'replay';
    byId('btnStartLive').hidden = view !== 'console' && !window.offlineEngine.isCapturing && byId('btnStartLive').getAttribute('aria-busy') !== 'true';
    byId('sourceLabel').textContent = view === 'replay' ? (state.replayName ? 'SAVED TRAJECTORY' : 'SAVED EXAMPLE') : 'LIVE SENSORS';
    byId('telemetrySource').textContent = view === 'replay' ? 'Saved values' : 'Live session';
    byId('positionMetricLabel').textContent = view === 'replay' ? 'Reference position error' : 'Position uncertainty';
    byId('driftMetricLabel').textContent = view === 'replay' ? 'Saved DR drift' : 'Estimated drift';
    if (view === 'console' || view === 'replay') {
        state.map.invalidateSize();
        if (view === 'replay') {
            byId('sessionHint').textContent = state.replayName || 'Illustrative trajectory · not a live experiment';
            byId('btnGpsOutage').disabled = true;
            if (state.data && previous !== 'replay') onDataLoaded();
        } else if (!window.offlineEngine.isCapturing && previous !== 'console') window.showStandby(true);
    }
    if (view === 'performance') window.refreshDeviceTimings();
};

window.updateConsoleTelemetry = data => {
    if (data.status && data.status !== 'active') {
        const waiting = data.status === 'waiting_for_imu';
        byId('sessionHint').textContent = waiting ? 'Waiting for motion samples · allow sensors on a supported phone'
            : data.status === 'calibrating' ? 'Calibrating orientation · follow the field guide' : 'Waiting for an accurate initial GPS fix';
        byId('statusIMU').textContent = waiting ? 'No samples' : 'Receiving';
        for (const id of ['statusAI', 'statusEKF', 'statusConstraints', 'statusMap']) {
            byId(id).textContent = 'Waiting';
            byId(id).className = '';
        }
        return;
    }
    const saved = data.source === 'saved';
    const stage = data.nav_mode === 'dr' ? 'dr' : data.nav_mode === 'reacq' ? 'reacq' : 'normal';
    document.querySelectorAll('.run-sequence li').forEach(item => item.classList.toggle('active', item.dataset.stage === stage));
    byId('gnssDescription').textContent = data.nav_mode === 'reacq'
        ? 'Fix reacquired. Position is converging.'
        : data.gnss_available ? 'Position is aided by the current fix.' : 'Position continues from local sensors.';
    if (!saved) {
        byId('sessionHint').textContent = data.nav_mode === 'dr' ? 'GPS denied · navigation continues' : data.nav_mode === 'reacq' ? 'Fresh fix · converging smoothly' : 'Local sensors · session running';
    }
    const components = saved ? {
        statusIMU: 'Saved', statusAI: 'Saved', statusEKF: 'Saved', statusConstraints: data.zupt_active ? 'ZUPT saved' : 'Saved', statusMap: 'Saved',
    } : {
        statusIMU: 'Active', statusAI: data.ai_error ? 'Error' : data.ai_active ? 'Active' : data.zupt_active ? 'Paused' : 'Buffering',
        statusEKF: 'Active', statusConstraints: data.zupt_active ? 'ZUPT active' : data.nhc_active ? 'NHC active' : 'Monitoring',
        statusMap: data.map_matched ? 'Matched' : 'Searching',
    };
    Object.entries(components).forEach(([id, value]) => {
        byId(id).textContent = value;
        byId(id).classList.toggle('active', !saved && (value.includes('active') || value === 'Active' || value === 'Matched'));
    });
};

function renderExperiments(reports = []) {
    const rows = byId('experimentRows');
    rows.replaceChildren();
    Object.entries(configurationNames).forEach(([key, name]) => {
        const report = reports.find(item => item.mode === key);
        const score = report?.status === 'completed' ? report.metrics?.outage : null;
        const row = document.createElement('tr');
        const nameCell = document.createElement('td');
        const letter = Object.assign(document.createElement('span'), { className: 'experiment-letter', textContent: key });
        nameCell.append(letter, document.createTextNode(name));
        row.append(nameCell);
        experimentMetrics.forEach(metric => {
            row.append(Object.assign(document.createElement('td'), { textContent: readableNumber(score?.[metric]) }));
        });
        const status = document.createElement('td');
        status.textContent = report ? report.status === 'completed' ? 'Complete' : 'Blocked' : 'Not run';
        if (report?.error) status.title = String(report.error);
        row.append(status);
        rows.append(row);
    });
}

async function readLocalJSON(input) {
    const file = input.files?.[0];
    if (!file) return null;
    if (file.size > 20 * 1024 * 1024) throw new Error('Choose a file smaller than 20 MB.');
    return { name: file.name, payload: JSON.parse(await file.text()) };
}

window.refreshDeviceTimings = () => {
    const engine = window.offlineEngine;
    const report = engine.profiler.report();
    const cards = byId('performanceCards');
    cards.replaceChildren();
    for (const [key, label] of Object.entries({ tcn: 'TCN inference', ekf: 'EKF update', map_matching: 'Map matching', total_loop: 'Navigation loop' })) {
        const timing = report.timings[key];
        const card = Object.assign(document.createElement('div'), { className: 'performance-card' });
        const heading = Object.assign(document.createElement('h3'), { textContent: label });
        const value = Object.assign(document.createElement('strong'), { textContent: readableNumber(timing.mean_ms) });
        value.append(Object.assign(document.createElement('small'), { textContent: 'ms' }));
        const count = Object.assign(document.createElement('p'), { textContent: `${timing.count} measured calls · mean` });
        const p95 = Object.assign(document.createElement('p'), { textContent: `P95 ${readableNumber(timing.p95_ms, ' ms')}` });
        card.append(heading, value, count, p95);
        cards.append(card);
    }
    byId('modelSizeReadout').textContent = Number.isFinite(report.model_size_bytes) ? `${(report.model_size_bytes / (1024 * 1024)).toFixed(2)} MB` : '—';
    byId('startupReadout').textContent = readableNumber(report.startup_ms, ' ms');
    byId('imuRateReadout').textContent = readableNumber(engine.observedImuRate, ' Hz');
    byId('deviceIdentity').textContent = report.total_navigation_steps
        ? `${report.total_navigation_steps} navigation steps measured in this browser. ${navigator.userAgent}`
        : 'No navigation steps measured yet. Start the engine and complete calibration to collect timings.';
};

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-view]').forEach(button => {
        button.title = workspaceViews[button.dataset.view][0];
        button.addEventListener('click', () => window.selectWorkspace(button.dataset.view));
    });
    byId('btnFitArea').addEventListener('click', fitLocalArea);
    byId('replayFile').addEventListener('change', async event => {
        const notice = byId('replayNotice');
        try {
            const loaded = await readLocalJSON(event.target);
            if (!loaded) return;
            const data = loaded.payload?.data;
            const count = data?.timestamps?.length;
            const numeric = ['timestamps', 'speed_estimated', 'heading_estimated', 'position_error', 'confidence', 'dr_drift_percent'];
            const fields = [...numeric, 'true_lat_lon', 'estimated_lat_lon', 'gnss_available', 'nav_mode', 'zupt_active'];
            if (!count || fields.some(key => !Array.isArray(data[key]) || data[key].length !== count)) {
                throw new Error('Choose a simulator trajectory JSON with estimated positions. Use replay.py to evaluate raw phone recordings.');
            }
            for (let i = 0; i < count; i++) {
                if (numeric.some(key => !Number.isFinite(data[key][i])) ||
                    (i > 0 && data.timestamps[i] <= data.timestamps[i - 1]) ||
                    !['gnss_ins', 'dr', 'reacq'].includes(data.nav_mode[i]) ||
                    typeof data.gnss_available[i] !== 'boolean' || typeof data.zupt_active[i] !== 'boolean' ||
                    data.speed_estimated[i] < 0 || data.position_error[i] < 0 || data.dr_drift_percent[i] < 0 ||
                    data.confidence[i] < 0 || data.confidence[i] > 1 ||
                    ['true_lat_lon', 'estimated_lat_lon'].some(key => {
                        const point = data[key][i];
                        return !Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite) || Math.abs(point[0]) > 85 || Math.abs(point[1]) > 180;
                    })) throw new Error(`Invalid trajectory values at frame ${i + 1}.`);
            }
            if (window.offlineEngine.isCapturing || byId('btnStartLive').getAttribute('aria-busy') === 'true') throw new Error('Stop the engine before loading playback.');
            pause();
            state.data = { data, metadata: loaded.payload.metadata || {} };
            state.replayName = loaded.name;
            onDataLoaded();
            window.selectWorkspace('replay');
            notice.classList.remove('error');
            notice.textContent = `${loaded.name} · ${count.toLocaleString()} frames · ${(data.timestamps.at(-1) - data.timestamps[0]).toFixed(1)} seconds · provenance unverified`;
        } catch (error) {
            notice.classList.add('error');
            notice.textContent = `Import failed: ${error.message}`;
        } finally { event.target.value = ''; }
    });
    byId('btnFollow').addEventListener('click', () => {
        state.followPosition = !state.followPosition;
        byId('btnFollow').setAttribute('aria-pressed', String(state.followPosition));
        if (state.followPosition && state.estimatedCoords.length) state.map.panTo(state.estimatedCoords.at(-1));
    });
    const dialog = byId('guideDialog');
    byId('btnGuide').addEventListener('click', () => dialog.showModal());
    byId('btnCloseGuide').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) {
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    } });
    byId('experimentFile').addEventListener('change', async event => {
        const notice = byId('experimentNotice');
        try {
            const loaded = await readLocalJSON(event.target);
            if (!loaded) return;
            const reports = loaded.payload;
            if (!Array.isArray(reports) || !reports.length || reports.length > 7 || reports.some(report =>
                !report || !Object.hasOwn(configurationNames, report.mode) || !['completed', 'blocked'].includes(report.status))) {
                throw new Error('Choose the *_results.json file produced by replay.py.');
            }
            if (new Set(reports.map(report => report.mode)).size !== reports.length) throw new Error('The report contains duplicate configurations.');
            const completed = reports.filter(report => report.status === 'completed');
            const comparisons = new Set(completed.map(report => JSON.stringify([report.dataset?.sha256, report.outage?.start_s, report.outage?.requested_duration_s])));
            if (comparisons.size > 1) throw new Error('These configurations use different recordings or outage intervals. Import a comparable run.');
            for (const report of completed) {
                if (!report.metrics?.outage || !report.dataset?.sha256 || !Number.isFinite(report.outage?.requested_duration_s)) throw new Error('This report is missing its metrics or recording identity.');
                for (const metric of experimentMetrics) {
                    const value = report.metrics.outage[metric];
                    if (value != null && (!Number.isFinite(value) || value < 0)) throw new Error('The report contains invalid error values.');
                }
            }
            renderExperiments(reports);
            notice.classList.remove('error');
            notice.textContent = `${loaded.name} · ${completed.length} completed / ${reports.length} configurations · loaded locally`;
            const first = completed[0];
            byId('experimentInsight').textContent = first ? `GPS outage: ${first.outage.requested_duration_s} seconds` : 'The experiment did not complete.';
            byId('experimentProvenance').textContent = first
                ? `${first.dataset.metadata?.provenance || 'Recording provenance unverified'}. Results are relative to recorded GNSS. ${first.machine?.label || 'Measurement device unspecified'}.`
                : reports.map(report => `${report.mode}: ${report.error || 'No reason provided'}`).join(' · ');
        } catch (error) {
            notice.classList.add('error');
            notice.textContent = `Import failed: ${error.message} Your previous results are unchanged.`;
        } finally { event.target.value = ''; }
    });
    byId('parityFile').addEventListener('change', async event => {
        try {
            const loaded = await readLocalJSON(event.target);
            if (!loaded) return;
            const report = loaded.payload;
            if (typeof report?.passed !== 'boolean' || !Number.isFinite(report.max_absolute_difference) || report.max_absolute_difference < 0 || !Number.isInteger(report.window_count) || report.window_count < 1) {
                throw new Error('Choose a measured report from verify_onnx.py.');
            }
            byId('parityStatus').textContent = `${report.passed ? 'Reported PASS' : 'Reported FAIL'} · maximum difference ${report.max_absolute_difference.toExponential(3)} · ${report.window_count} windows · ${report.provenance || 'Input provenance unverified'}. Export parity only; navigation accuracy is separate.`;
        } catch (error) { byId('parityStatus').textContent = `Import failed: ${error.message}`; }
        finally { event.target.value = ''; }
    });
    renderExperiments();
    window.refreshDeviceTimings();
    setInterval(() => { if (state.view === 'performance') window.refreshDeviceTimings(); }, 1000);
});
