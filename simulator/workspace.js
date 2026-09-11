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
    for (const id of ['speedValue', 'headingValue', 'posErrorValue', 'driftValue', 'confidenceValue', 'driftBadge', 'positionCoordinates']) {
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

    // Fast switch if it's the initial load or loading overlay doesn't exist
    if (previous === undefined || !window.setLoadingState) {
        performSwitch();
        return;
    }

    // Play loading animation
    window.setLoadingState(0, "LOADING MODULE", "READY");
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        window.setLoadingState(progress, "LOADING MODULE", "READY");
        if (progress >= 100) {
            clearInterval(interval);
            // Switch UI immediately when progress hits 100% so it's ready when the overlay fades out (800ms transition)
            performSwitch();
        }
    }, 40); // 400ms loading + 800ms lock/fade animation = ~1.2s total wait before user sees the new screen
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

window.setLoadingState = (percent, message, finalMessage = 'NAVIGATION READY') => {
    const overlay = document.getElementById('compassLoadingOverlay');
    if (!overlay) return;
    
    // Show overlay if not active
    if (!overlay.classList.contains('active')) {
        overlay.classList.remove('fade-out');
        overlay.classList.add('active');
        document.getElementById('compassContainer')?.classList.remove('compass-locked');

        // Enforce max 3s loading time to detect failure
        if (window._loadingTimeout) clearTimeout(window._loadingTimeout);
        window._loadingTimeout = setTimeout(() => {
            const currentProgress = document.getElementById('loadingProgressText').textContent;
            if (currentProgress !== '100%') {
                // If not complete in 3 seconds, show failure
                const statusText = document.getElementById('loadingStatusText');
                statusText.textContent = 'FAILED TO START';
                statusText.style.color = 'red';
                
                // Stop any further progression of the bar
                document.getElementById('compassLoadingBar').style.width = currentProgress;
                
                // Reset loading state timeout
                setTimeout(() => {
                    overlay.classList.remove('active');
                    overlay.classList.add('fade-out');
                    statusText.style.color = ''; // reset color
                }, 2000);
            }
        }, 3000);
    }

    percent = Math.min(100, Math.max(0, percent));
    
    // Update text and bar (only if not failed)
    const statusText = document.getElementById('loadingStatusText');
    if (statusText.textContent === 'FAILED TO START') return; // Do not update if failed

    // Reset color in case it was previously failed
    statusText.style.color = '';
    
    document.getElementById('loadingProgressText').textContent = Math.round(percent) + '%';
    if (message) document.getElementById('loadingStatusText').textContent = message;
    const bar = document.getElementById('compassLoadingBar');
    if (bar) bar.style.width = percent + '%';
    const arc = document.getElementById('compassProgressArc');
    const circumference = 597; // 2 * PI * 95
    const offset = circumference - (percent / 100) * circumference;
    if (arc) arc.style.strokeDashoffset = offset;

    // Lock and hide at 100%
    if (percent >= 100) {
        document.getElementById('loadingStatusText').textContent = finalMessage;
        document.getElementById('compassContainer')?.classList.add('compass-locked');
        setTimeout(() => {
            overlay.classList.remove('active');
            overlay.classList.add('fade-out');
        }, 800); // Wait for lock animation to settle before fading out
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
    window.addEventListener('online', () => window.updateMapSource());
    window.addEventListener('offline', () => window.updateMapSource());
    byId('btnSaveArea').addEventListener('click', async () => {
        const button = byId('btnSaveArea'), notice = byId('mapNetworkStatus');
        button.disabled = true;
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
        finally { button.disabled = false; }
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
                // Request real-time location via browser API if no engine path exists
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
    
    // Handle consent modal
    const startupModal = byId('startupModal');
    if (startupModal) {
        const chkAgree = byId('acceptTerms');
        const btnStart = byId('btnStart');
        const btnSkip = byId('btnSkip');

        const runInitialLoading = async () => {
            window.setLoadingState(0, "INITIALIZING WORKSPACE", "WORKSPACE READY");
            await new Promise(r => setTimeout(r, 400));
            window.setLoadingState(40, "LOADING OFFLINE DATA", "WORKSPACE READY");
            await new Promise(r => setTimeout(r, 400));
            window.setLoadingState(80, "CONNECTING TO SENSORS", "WORKSPACE READY");
            await new Promise(r => setTimeout(r, 400));
            window.setLoadingState(100, "WORKSPACE READY", "WORKSPACE READY");
            
            // Wait for the overlay to fade out before showing the modal
            setTimeout(() => {
                startupModal.showModal();
            }, 1200);
        };
        runInitialLoading();

        chkAgree.addEventListener('change', () => {
            btnStart.disabled = !chkAgree.checked;
        });

        btnStart.addEventListener('click', () => {
            btnStart.classList.add('btn-accepted-anim');
            btnStart.textContent = 'Accepted ✓';
            setTimeout(() => {
                localStorage.setItem('startupAcknowledged', 'true');
                startupModal.close();
                btnStart.classList.remove('btn-accepted-anim');
                btnStart.textContent = 'Accept & Continue';
            }, 600);
        });

        btnSkip.addEventListener('click', () => {
            btnSkip.classList.remove('btn-shake-anim');
            // Trigger reflow to restart animation if clicked multiple times
            void btnSkip.offsetWidth;
            btnSkip.classList.add('btn-shake-anim');
            
            const originalText = btnSkip.textContent;
            btnSkip.textContent = 'Please accept T&C';
            btnSkip.style.color = '#b45230'; // Highlight the text in orange/red
            
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
        }
        state.localMap.layer.remove();
        state.onlineTiles.addTo(state.map);
    } else {
        state.onlineTiles?.remove();
        state.localMap.layer.addTo(state.map).bringToBack();
    }
    byId('mapNetworkStatus').textContent = online ? 'Online streets · save this area before disconnecting'
        : failed ? 'Online map unavailable · showing downloaded streets' : 'Offline streets · downloaded area only';
};
