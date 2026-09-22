
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
const readableNumber = (value, suffix = '') => Number.isFinite(value) ? `${value.toFixed(2)}${suffix}` : '-';
const experimentMetrics = ['mean_position_error_m', 'final_position_error_m', 'max_position_error_m', 'velocity_rmse_mps', 'heading_rmse_deg'];

function fitLocalArea() {
    state.localMap?.centerView(state.map);
}

window.showStandby = (clearTrack = false) => {
    for (const id of ['speedValue', 'headingValue', 'posErrorValue', 'driftValue', 'confidenceValue', 'driftBadge', 'positionCoordinates']) {
        byId(id).textContent = '-';
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
    if (previous === view) return;

    const performSwitch = () => {
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
        byId('navigationSettings').hidden = view !== 'console';
        byId('metricsCard').hidden = view !== 'replay';
        byId('btnStartLive').hidden = view !== 'console' && !window.offlineEngine.isCapturing && byId('btnStartLive').getAttribute('aria-busy') !== 'true';
        byId('sourceLabel').textContent = view === 'replay' ? (state.replayName ? 'SAVED TRAJECTORY' : 'SAVED EXAMPLE') : 'LIVE SENSORS';
        byId('telemetrySource').textContent = view === 'replay' ? 'Saved values' : 'Live session';
        byId('positionMetricLabel').textContent = view === 'replay' ? 'Reference position error' : byId('travelMode').value === 'walking' ? 'GPS accuracy' : 'Position uncertainty';
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

    performSwitch();
};

window.updateConsoleTelemetry = data => {
    if (data.walking) {
        byId('sessionHint').textContent = data.session_hint;
        byId('statusIMU').textContent = data.sensors_ready ? 'Steps + compass' : 'Check sensors';
        for (const id of ['statusAI', 'statusEKF', 'statusConstraints', 'statusMap']) {
            byId(id).textContent = 'Off · walking';
            byId(id).className = '';
        }
        byId('positionMetricLabel').textContent = 'GPS accuracy';
        byId('navModeIndicator').querySelector('.mode-label').textContent = data.nav_mode === 'dr' ? 'ESTIMATED · GPS LOST' : data.nav_mode === 'reacq' ? 'REACQUISITION' : 'GPS · WALKING';
        byId('gnssDescription').textContent = data.gnss_available ? 'Following GPS fixes.' : 'No GPS fix. Steps and compass estimate your position.';
        document.querySelectorAll('.run-sequence li').forEach(item => item.classList.toggle('active', item.dataset.stage === (data.nav_mode === 'gnss_ins' ? 'normal' : data.nav_mode)));
        return;
    }
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
    const config = saved && state.data.metadata.source === 'replay.py' ? state.data.metadata.configuration : null;
    const components = config ? {
        statusIMU: 'Recorded', statusAI: config.ai ? 'Configured' : 'Off', statusEKF: config.ekf ? 'Configured' : 'Off',
        statusConstraints: config.zupt ? 'NHC + ZUPT' : config.nhc ? 'NHC' : 'Off', statusMap: config.map_matching ? 'Configured' : 'Off',
    } : saved ? {
        statusIMU: 'Saved', statusAI: 'Saved', statusEKF: 'Saved', statusConstraints: data.zupt_active ? 'ZUPT saved' : 'Saved', statusMap: 'Saved',
    } : {
        statusIMU: 'Active', statusAI: data.ai_error ? 'Error' : data.ai_active ? 'AI Speed · Active' : data.zupt_active ? 'Gated (ZUPT)' : 'Buffering',
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
    if (!cards) return;

    const metricsMap = { tcn: 'TCN inference', ekf: 'EKF update', map_matching: 'Map matching', total_loop: 'Navigation loop' };

    let existingCards = cards.querySelectorAll('.performance-card');
    if (existingCards.length !== 4) {
        cards.replaceChildren();
        for (const [key, label] of Object.entries(metricsMap)) {
            const card = document.createElement('div');
            card.className = 'performance-card';
            card.dataset.metricKey = key;
            card.innerHTML = `<h3>${label}</h3><strong class="perf-val">0<small>ms</small></strong><p class="perf-count">0 measured calls · mean</p><p class="perf-p95">P95 0 ms</p>`;
            cards.append(card);
        }
        existingCards = cards.querySelectorAll('.performance-card');
    }

    let idx = 0;
    for (const [key] of Object.entries(metricsMap)) {
        const timing = report.timings[key];
        const card = existingCards[idx++];
        if (card && timing) {
            const valEl = card.querySelector('.perf-val');
            const countEl = card.querySelector('.perf-count');
            const p95El = card.querySelector('.perf-p95');
            if (valEl) valEl.innerHTML = `${readableNumber(timing.mean_ms)}<small>ms</small>`;
            if (countEl) countEl.textContent = `${timing.count} measured calls · mean`;
            if (p95El) p95El.textContent = `P95 ${readableNumber(timing.p95_ms, ' ms')}`;
        }
    }

    byId('modelSizeReadout').textContent = Number.isFinite(report.model_size_bytes) ? `${(report.model_size_bytes / (1024 * 1024)).toFixed(2)} MB` : '-';
    byId('startupReadout').textContent = readableNumber(report.startup_ms, ' ms');
    byId('imuRateReadout').textContent = readableNumber(engine.observedImuRate, ' Hz');
    byId('deviceIdentity').textContent = report.total_navigation_steps
        ? `${report.total_navigation_steps} navigation steps measured in this browser. ${navigator.userAgent}`
        : 'No navigation steps measured yet. Start the engine and complete calibration to collect timings.';
};

window.setLoadingState = (percent, message, finalMessage = 'NAVIGATION READY') => {
    const overlay = document.getElementById('compassLoadingOverlay');
    if (!overlay) return;
    
    
    if (!overlay.classList.contains('active')) {
        overlay.classList.remove('fade-out');
        overlay.classList.add('active');
        document.getElementById('compassContainer')?.classList.remove('compass-locked');

        
        if (window._loadingTimeout) clearTimeout(window._loadingTimeout);
        window._loadingTimeout = setTimeout(() => {
            const currentProgress = document.getElementById('loadingProgressText').textContent;
            if (currentProgress !== '100%') {
                
                const statusText = document.getElementById('loadingStatusText');
                statusText.textContent = 'FAILED TO START';
                statusText.style.color = 'red';
                
                
                document.getElementById('compassLoadingBar').style.width = currentProgress;
                
                
                setTimeout(() => {
                    overlay.classList.remove('active');
                    overlay.classList.add('fade-out');
                    statusText.style.color = ''; 
                }, 2000);
            }
        }, 3000);
    }

    percent = Math.min(100, Math.max(0, percent));
    
    
    const statusText = document.getElementById('loadingStatusText');
    if (statusText.textContent === 'FAILED TO START') return; 

    
    statusText.style.color = '';
    
    document.getElementById('loadingProgressText').textContent = Math.round(percent) + '%';
    if (message) document.getElementById('loadingStatusText').textContent = message;
    const bar = document.getElementById('compassLoadingBar');
    if (bar) bar.style.width = percent + '%';
    const arc = document.getElementById('compassProgressArc');
    const circumference = 597; 
    const offset = circumference - (percent / 100) * circumference;
    if (arc) arc.style.strokeDashoffset = offset;

    
    if (percent >= 100) {
        document.getElementById('loadingStatusText').textContent = finalMessage;
        document.getElementById('compassContainer')?.classList.add('compass-locked');
        setTimeout(() => {
            overlay.classList.remove('active');
            overlay.classList.add('fade-out');
        }, 800); 
    }
};

document.addEventListener('DOMContentLoaded', () => {
    byId('btnPairPC').addEventListener('click', async () => {
        const button = byId('btnPairPC');
        button.disabled = true;
        try { await window.recordingSync.pair(byId('pcPairingCode').value.trim()); }
        catch (error) { window.recordingSync.status(error.message, true); }
        finally { button.disabled = false; }
    });
    byId('btnSyncPC').addEventListener('click', () => window.recordingSync.flush());
    byId('btnUpdateApp').addEventListener('click', async () => {
        try {
            if (offlineEngine.isCapturing || dataRecorder.isRecording) throw new Error('Stop navigation and recording before updating.');
            const registration = await navigator.serviceWorker.getRegistration();
            if (!registration) throw new Error('Install the offline package first.');
            window.recordingSync.status('Checking the PC for an updated application…');
            await registration.update();
            const worker = registration.installing || registration.waiting;
            if (worker) await new Promise((resolve, reject) => {
                const check = () => {
                    if (worker.state === 'activated') resolve();
                    if (worker.state === 'redundant') reject(new Error('Update failed; the previous offline app is retained.'));
                };
                worker.addEventListener('statechange', check);
                check();
            });
            location.reload();
        } catch (error) { window.recordingSync.status(error.message, true); }
        finally { byId('btnUpdateApp').disabled = false; }
    });
    byId('travelMode').addEventListener('change', () => {
        const walking = byId('travelMode').value === 'walking';
        byId('stepLengthControl').hidden = byId('walkingHelp').hidden = !walking;
        byId('positionMetricLabel').textContent = walking ? 'GPS accuracy' : 'Position uncertainty';
        document.querySelector('[data-stage="normal"] small').textContent = walking ? 'Follow GPS fixes' : 'Initialize & calibrate';
        document.querySelector('[data-stage="dr"] small').textContent = walking ? 'Steps + compass' : 'Continue with IMU + AI';
    });
    byId('travelMode').dispatchEvent(new Event('change'));
    byId('mapSource').addEventListener('change', () => window.updateMapSource());
    window.addEventListener('online', () => { window.updateMapSource(); window.refreshOfflineMapsUI?.(); });
    window.addEventListener('offline', () => { window.updateMapSource(); window.refreshOfflineMapsUI?.(); });
    byId('btnSaveArea').addEventListener('click', async () => {
        const button = byId('btnSaveArea'), notice = byId('mapNetworkStatus');
        const netBadge = byId('offlineNetStatusBadge');
        button.disabled = true;
        if (netBadge) {
            netBadge.dataset.downloading = 'true';
            netBadge.textContent = 'Downloading';
            netBadge.style.background = 'rgba(6,182,212,0.15)';
            netBadge.style.color = 'var(--accent-cyan)';
        }
        notice.textContent = 'Downloading a 2 km square around the map center…';
        try {
            if (!navigator.onLine) throw new Error('Reconnect to download streets first.');
            const center = state.map.getCenter().wrap();
            const localMap = await LocalMap.download(center.lat, center.lng);
            if (window.offlineEngine.isCapturing && window.offlineEngine.navigationMode !== 'walking') throw new Error('Stop the vehicle engine before replacing its map.');
            const cache = await caches.open('navigators-map-data-v1');
            await cache.put('./data/road_network.json', new Response(JSON.stringify(localMap.data), { headers: { 'Content-Type': 'application/json' } }));
            installLocalMap(localMap);
            notice.textContent = 'Area saved on this device · offline view uses downloaded streets';
            notice.style.color = '';
        } catch (error) { 
            notice.textContent = `Area not saved: ${error.message}`; 
            notice.style.color = 'red';
        }
        finally { 
            button.disabled = false; 
            if (netBadge) delete netBadge.dataset.downloading;
            window.refreshOfflineMapsUI?.();
        }
    });
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
            const numeric = ['timestamps', 'speed_estimated', 'heading_estimated'];
            const optional = ['position_error', 'confidence', 'dr_drift_percent'];
            const fields = [...numeric, ...optional, 'true_lat_lon', 'estimated_lat_lon', 'gnss_available', 'nav_mode', 'zupt_active'];
            if (!count || fields.some(key => !Array.isArray(data[key]) || data[key].length !== count)) {
                throw new Error('Choose a *_trajectory.json exported by replay.py or a simulator trajectory. Evaluate raw recordings with replay.py first.');
            }
            for (let i = 0; i < count; i++) {
                if (numeric.some(key => !Number.isFinite(data[key][i])) ||
                    optional.some(key => data[key][i] !== null && !Number.isFinite(data[key][i])) ||
                    (i > 0 && data.timestamps[i] <= data.timestamps[i - 1]) ||
                    !['gnss_ins', 'dr', 'reacq'].includes(data.nav_mode[i]) ||
                    typeof data.gnss_available[i] !== 'boolean' || (data.zupt_active[i] !== null && typeof data.zupt_active[i] !== 'boolean') ||
                    data.speed_estimated[i] < 0 || data.position_error[i] < 0 || data.dr_drift_percent[i] < 0 ||
                    data.confidence[i] < 0 || data.confidence[i] > 1 ||
                    ['true_lat_lon', 'estimated_lat_lon'].some(key => {
                        const point = data[key][i];
                        if (key === 'true_lat_lon' && point === null) return false;
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
            notice.style.color = '';
            const metadata = state.data.metadata;
            notice.textContent = `${loaded.name} · ${count.toLocaleString()} frames · ${(data.timestamps.at(-1) - data.timestamps[0]).toFixed(1)} seconds · ${metadata.dataset?.metadata?.provenance || 'provenance unverified'}. ${metadata.reference || 'Saved trajectory; not live accuracy.'}`;
        } catch (error) {
            notice.classList.add('error');
            notice.textContent = `Import failed: ${error.message}`;
            notice.style.color = 'red';
        } finally { event.target.value = ''; }
    });
    byId('btnFollow').addEventListener('click', () => {
        state.followPosition = !state.followPosition;
        byId('btnFollow').setAttribute('aria-pressed', String(state.followPosition));
        if (state.followPosition) {
            if (state.estimatedCoords.length) {
                state.map.panTo(state.estimatedCoords.at(-1));
            } else if ('geolocation' in navigator) {
                
                navigator.geolocation.getCurrentPosition(
                    pos => {
                        const currentZoom = state.map.getZoom();
                        state.map.setView([pos.coords.latitude, pos.coords.longitude], Math.max(currentZoom, 17));
                    },
                    err => console.warn('Geolocation error:', err),
                    { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
                );
            }
        }
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
            notice.style.color = '';
            notice.textContent = `${loaded.name} · ${completed.length} completed / ${reports.length} configurations · loaded locally`;
            const first = completed[0];
            byId('experimentInsight').textContent = first ? `GPS outage: ${first.outage.requested_duration_s} seconds` : 'The experiment did not complete.';
            byId('experimentProvenance').textContent = first
                ? `${first.dataset.metadata?.provenance || 'Recording provenance unverified'}. Results are relative to recorded GNSS. ${first.machine?.label || 'Measurement device unspecified'}.`
                : reports.map(report => `${report.mode}: ${report.error || 'No reason provided'}`).join(' · ');
        } catch (error) {
            notice.classList.add('error');
            notice.textContent = `Import failed: ${error.message} Your previous results are unchanged.`;
            notice.style.color = 'red';
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
            byId('parityStatus').style.color = '';
        } catch (error) { 
            byId('parityStatus').textContent = `Import failed: ${error.message}`; 
            byId('parityStatus').style.color = 'red';
        }
        finally { event.target.value = ''; }
    });
    renderExperiments();
    window.refreshDeviceTimings();
    setInterval(() => { if (state.view === 'performance') window.refreshDeviceTimings(); }, 1000);
    
    
    const startupModal = byId('startupModal');
    if (startupModal) {
        const chkAgree = byId('acceptTerms');
        const btnStart = byId('btnStart');
        const btnSkip = byId('btnSkip');

        const runInitialLoading = async () => {
            window.setLoadingState(0, "INITIALIZING ENGINE & WORKSPACE", "WORKSPACE READY");
            if (window.offlineEngine?.initModel) {
                try {
                    await window.offlineEngine.initModel();
                } catch (e) {
                    console.warn('[Engine Init] Engine deferred or offline:', e);
                }
            }
            window.setLoadingState(60, "VERIFYING LOCAL MAP DATA", "WORKSPACE READY");
            await window.refreshOfflineMapsUI?.();
            window.setLoadingState(100, "WORKSPACE READY", "WORKSPACE READY");

            
            setTimeout(() => {
                startupModal.showModal();
            }, 1000);
        };
        runInitialLoading();

        chkAgree.addEventListener('change', () => {
            btnStart.disabled = !chkAgree.checked;
        });

        btnStart.addEventListener('click', () => {
            btnStart.classList.add('btn-accepted-anim');
            btnStart.textContent = 'Accepted';
            setTimeout(() => {
                localStorage.setItem('startupAcknowledged', 'true');
                startupModal.close();
                btnStart.classList.remove('btn-accepted-anim');
                btnStart.textContent = 'Accept & Continue';
            }, 600);
        });


        btnSkip.addEventListener('click', () => {
            btnSkip.classList.remove('btn-shake-anim');
            
            void btnSkip.offsetWidth;
            btnSkip.classList.add('btn-shake-anim');
            
            const originalText = btnSkip.textContent;
            btnSkip.textContent = 'Please accept T&C';
            btnSkip.style.color = '#b45230'; 
            
            setTimeout(() => {
                btnSkip.textContent = 'Reject All';
                btnSkip.style.color = '';
            }, 1500);
        });
    }
});

window.updateMapSource = (failed = false) => {
    if (!state.localMap) return;
    const online = byId('mapSource').value === 'online' && navigator.onLine && !failed;
    if (online) {
        if (!state.onlineTiles) {
            state.onlineTiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
            }).on('tileerror', () => window.updateMapSource(true));
            state.onlineTiles.addTo(state.map);
        }
        state.localMap.layer.remove();
        state.onlineTiles.getContainer().style.filter = 'none';
    } else {
        if (state.onlineTiles) {
            state.onlineTiles.getContainer().style.filter = 'grayscale(100%) opacity(50%)';
        }
        state.localMap.layer.addTo(state.map).bringToFront();
    }
    byId('mapNetworkStatus').textContent = online ? 'Online streets · save this area before disconnecting'
        : failed ? 'Online map unavailable · showing downloaded streets' : 'Offline streets · downloaded area only';
    window.refreshOfflineMapsUI?.();
};

window.refreshOfflineMapsUI = async () => {
    const netBadge = byId('offlineNetStatusBadge');
    const storageVal = byId('offlineStorageValue');
    const lastUpdatedVal = byId('offlineLastUpdated');
    const downloadedList = byId('downloadedAreasList');

    const isOnline = navigator.onLine;
    if (netBadge && !netBadge.dataset.downloading) {
        netBadge.textContent = isOnline ? 'Online' : 'Offline';
        netBadge.style.background = isOnline ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)';
        netBadge.style.color = isOnline ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }

    let cacheItem = null;
    let cacheSize = 0;
    try {
        if (typeof caches !== 'undefined') {
            const cache = await caches.open('navigators-map-data-v1');
            const resp = await cache.match('./data/road_network.json');
            if (resp) {
                const text = await resp.text();
                cacheSize = text.length;
                cacheItem = JSON.parse(text);
            }
        }
    } catch (e) {
        console.warn('Failed checking local map cache:', e);
    }

    if (cacheItem && cacheItem.origin) {
        const dateObj = cacheItem.downloaded_at ? new Date(cacheItem.downloaded_at) : null;
        const dateStr = dateObj ? `${dateObj.toLocaleDateString()} ${dateObj.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}` : 'Unknown';
        const ageDays = dateObj ? (Date.now() - dateObj.getTime()) / (1000 * 3600 * 24) : 0;
        const needsUpdate = ageDays > 30;

        if (storageVal) storageVal.textContent = `${(cacheSize / 1024).toFixed(1)} KB`;
        if (lastUpdatedVal) lastUpdatedVal.textContent = dateStr;

        if (downloadedList) {
            downloadedList.innerHTML = `
                <div class="offline-area-item" style="display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; background: rgba(16,185,129,0.06); border: 1px solid var(--accent-emerald); border-radius: 6px;">
                    <div>
                        <strong style="color: var(--text-primary); font-size: 12px;">Saved Region (${cacheItem.roads?.length || 0} roads)</strong>
                        <div style="font-size: 11px; color: var(--text-muted);">${Math.abs(cacheItem.origin.lat).toFixed(4)}° N, ${Math.abs(cacheItem.origin.lon).toFixed(4)}° E</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span class="badge" style="font-size: 10px; padding: 2px 6px; border-radius: 4px; background: ${needsUpdate ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.15)'}; color: ${needsUpdate ? 'var(--accent-amber)' : 'var(--accent-emerald)'};">
                            ${needsUpdate ? 'Needs Update' : 'Downloaded'}
                        </span>
                    </div>
                </div>
            `;
        }
    } else {
        if (storageVal) storageVal.textContent = '0 KB';
        if (lastUpdatedVal) lastUpdatedVal.textContent = 'Never';
        if (downloadedList) {
            downloadedList.innerHTML = `
                <div style="font-size: 11px; color: var(--text-muted); padding: 8px; background: rgba(255,255,255,0.02); border-radius: 6px;">
                    No downloaded maps on this device. Move map to target region and click Download Area.
                </div>
            `;
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => window.refreshOfflineMapsUI?.(), 200);
});
