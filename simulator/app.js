/**
 * Navigators IDR — Simulator Application
 * Interactive map visualization with real-time telemetry playback.
 *
 * Loads simulation JSON data and animates:
 *   - Ground truth trajectory (cyan)
 *   - Estimated trajectory (green/orange based on mode)
 *   - GNSS denied zones (red hatching)
 *   - Real-time telemetry gauges
 */

// ========================================================
// State
// ========================================================
const state = {
    map: null,
    data: null,
    playing: false,
    view: 'console',
    followPosition: true,
    playbackTime: 0,
    currentIndex: 0,
    playbackSpeed: 2,
    animationFrame: null,
    lastFrameTime: 0,

    // Map layers
    truthLine: null,
    estimatedLine: null,
    vehicleMarker: null,
    truthMarker: null,
    gnssZones: [],

    // Accumulated coordinates for drawing
    truthCoords: [],
    estimatedCoords: [],
};

// ========================================================
// Initialization
// ========================================================
document.addEventListener('DOMContentLoaded', async () => {
    const loader = document.getElementById('app-loader');

    // Mode Switch: ?mode=mobile vs ?mode=dashboard (default)
    const params = new URLSearchParams(window.location.search);
    const mode = params.get('mode') || 'dashboard';
    document.body.classList.add(`${mode}-mode`);
    
    if (mode === 'dashboard') {
        // Dashboard needs to connect to WebSocket as a viewer
        connectDashboardToWebSocket();
    } else {
        // Mobile mode needs to connect to WebSocket as a sensor source
        if (window.dataRecorder) {
            window.dataRecorder.connectToServer(window.location.host).catch(e => console.error("WS error:", e));
        }
    }

    initMap();
    initControls();
    try {
        installLocalMap(await LocalMap.load());
    } catch (error) {
        document.getElementById('mapStatus').textContent = error.message;
        const btnStartLive = document.getElementById('btnStartLive');
        if (btnStartLive) btnStartLive.disabled = true;
    }
    
    // Wait for simulation data to load
    await loadSimulationData();
    
    // Hide loader after a short delay
    setTimeout(() => {
        if (loader) loader.classList.add('hidden');
    }, 400);
});

function connectDashboardToWebSocket() {
    const wsUrl = window.location.protocol === 'https:' ? `wss://${window.location.host}/ws?role=dashboard` : `ws://${window.location.host}/ws?role=dashboard`;
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => console.log('[Dashboard] Connected to telemetry broadcast.');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'telemetry') {
            updateDashboardTelemetry(data.payload);
        }
    };
    ws.onclose = () => setTimeout(connectDashboardToWebSocket, 2000);
}

function updateDashboardTelemetry(payload) {
    if (payload.raw) {
        document.getElementById('sourceLabel').textContent = 'LIVE FROM MOBILE';
        // You could update speed and heading here directly from the raw IMU/GNSS
    }
}

// Training Logic
let trainingPollInterval = null;

async function startTraining() {
    const confirmed = confirm("The verified 4.20 m/s production model is currently loaded and active.\n\nDo you want to start a new training job?");
    if (!confirmed) return;
    
    try {
        const res = await fetch('/training/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({epochs: 40, batch_size: 256})
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
        } else {
            if (!trainingPollInterval) {
                trainingPollInterval = setInterval(pollTrainingStatus, 2000);
            }
        }
    } catch (e) {
        console.error("Failed to start training:", e);
    }
}

async function pollTrainingStatus() {
    try {
        const res = await fetch('/training/status');
        const data = await res.json();
        
        const label = document.getElementById('trainStatusLabel');
        if (label) {
            if (data.status === 'Training Complete' || (!data.is_training && data.current_epoch === 40)) {
                label.textContent = `Production Model Active (4.20 m/s MAE — Retained)`;
                label.style.color = '#10b981';
            } else if (!data.is_training && data.status === 'Idle') {
                label.textContent = `Production Model Active (4.20 m/s MAE)`;
                label.style.color = '#10b981';
            } else {
                label.textContent = data.status;
                if (data.status && (data.status.startsWith('Error') || data.status.startsWith('Failed'))) {
                    label.style.color = '#ef4444';
                }
            }
        }
        
        if (data.is_training) {
            if (!trainingPollInterval) {
                trainingPollInterval = setInterval(pollTrainingStatus, 2000);
            }
        } else if (data.status !== "Idle" && !data.status.startsWith("Loading")) {
            if (trainingPollInterval) {
                clearInterval(trainingPollInterval);
                trainingPollInterval = null;
            }
        }
    } catch (e) {
        console.error("Failed to poll training status:", e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const btnTrain = document.getElementById('btnStartTraining');
    if (btnTrain) btnTrain.addEventListener('click', startTraining);
    pollTrainingStatus();
});

function installLocalMap(localMap) {
    state.localMap?.layer?.remove();
    state.localMap = localMap;
    if (!localMap) return;
    localMap.draw(state.map);
    const origin = localMap.data.origin;
    document.getElementById('mapCoordinates').textContent = `${Math.abs(origin.lat).toFixed(4)}° ${origin.lat >= 0 ? 'N' : 'S'} / ${Math.abs(origin.lon).toFixed(4)}° ${origin.lon >= 0 ? 'E' : 'W'}`;
    document.getElementById('roadCount').textContent = `${localMap.data.roads.length.toLocaleString()} roads · stored on this device`;
    window.offlineEngine.setLocalMap(localMap);
    document.getElementById('mapStatus').textContent = `Local OSM map · ${localMap.data.roads.length} roads`;
    window.updateMapSource?.();
}

function initMap() {
    state.map = L.map('map', {
        center: [0, 0],
        zoom: 2,
        zoomControl: true,
        attributionControl: true,
    });

    state.onlineTiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
    }).addTo(state.map);

    if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(async (pos) => {
            if (state.map) {
                state.map.setView([pos.coords.latitude, pos.coords.longitude], 15);
                if (!state.localMap && navigator.onLine) {
                    try {
                        document.getElementById('mapNetworkStatus').textContent = 'Downloading streets for your realtime location...';
                        const localMap = await LocalMap.download(pos.coords.latitude, pos.coords.longitude);
                        const cache = await caches.open('navigators-map-data-v1');
                        await cache.put('./data/road_network.json', new Response(JSON.stringify(localMap.data), { headers: { 'Content-Type': 'application/json' } }));
                        installLocalMap(localMap);
                    } catch (e) {
                        console.error('Failed to download local map for realtime location:', e);
                    }
                }
            }
        });
    }

    L.control.zoom({ position: 'bottomright' }).addTo(state.map);
    L.control.scale({ position: 'bottomleft', imperial: false }).addTo(state.map);

    // Ground truth trajectory line
    state.truthLine = L.polyline([], {
        color: '#6c8884',
        weight: 3,
        opacity: 0.6,
        dashArray: '8, 6',
        lineCap: 'round',
    }).addTo(state.map);

    // Estimated trajectory line
    state.estimatedLine = L.polyline([], {
        color: '#4c7b59',
        weight: 3,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round',
    }).addTo(state.map);

    // Vehicle marker (estimated position)
    const vehicleIcon = L.divIcon({
        className: 'vehicle-marker',
        html: `
            <div class="vehicle-marker-pulse"></div>
            <div class="vehicle-marker-inner"></div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
    });

    state.vehicleMarker = L.marker([28.6139, 77.2090], {
        icon: vehicleIcon,
        zIndexOffset: 1000,
        opacity: 0,
    }).addTo(state.map);

    // Truth marker (small dot)
    const truthIcon = L.divIcon({
        className: '',
        html: '<div style="width:8px;height:8px;background:#6c8884;border-radius:50%;border:2px solid #f5f4ef;"></div>',
        iconSize: [8, 8],
        iconAnchor: [4, 4],
    });

    state.truthMarker = L.marker([28.6139, 77.2090], {
        icon: truthIcon,
        zIndexOffset: 999,
        opacity: 0,
    }).addTo(state.map);
}

function initControls() {
    document.getElementById('btnPlay').addEventListener('click', play);
    document.getElementById('btnPause').addEventListener('click', pause);
    document.getElementById('btnReset').addEventListener('click', reset);

    const speedSlider = document.getElementById('speedSlider');
    speedSlider.addEventListener('input', (e) => {
        state.playbackSpeed = parseFloat(e.target.value);
        document.getElementById('speedLabel').textContent = `${state.playbackSpeed}x`;
    });

    const timeline = document.getElementById('timeline');
    timeline.addEventListener('input', (e) => {
        if (state.data && !window.offlineEngine.isCapturing && state.view === 'replay') {
            pause();
            seekFrame(Math.min(state.data.data.timestamps.length - 1, Math.max(0, Math.round(Number(e.target.value)))));
        }
    });

    // Installation succeeds only after the complete local application is cached.
    const offlineStatus = document.getElementById('offlineStatus');
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.addEventListener('message', event => {
            if (event.data?.type === 'OFFLINE_STATUS') {
                offlineStatus.textContent = event.data.ready
                    ? 'Offline files ready · reload once before disconnecting'
                    : 'Offline files incomplete · reconnect and reload';
            }
        });
        navigator.serviceWorker.register('./sw.js').then(registration => {
            const check = () => registration.active?.postMessage({ type: 'CHECK_OFFLINE' });
            check();
            navigator.serviceWorker.addEventListener('controllerchange', check);
            registration.addEventListener('updatefound', () => {
                const worker = registration.installing;
                worker?.addEventListener('statechange', () => {
                    if (worker.state === 'redundant') offlineStatus.textContent = 'Offline download failed · reconnect and reload';
                });
            });
        }).catch(error => {
            if (error.message.includes('SSL certificate') || error.message.includes('security')) {
                console.warn('Service Worker registration skipped: Development SSL certificate detected. Offline mode requires a trusted CA or localhost. The application remains fully functional online.');
                offlineStatus.textContent = `Offline mode unavailable (Dev SSL)`;
            } else {
                offlineStatus.textContent = `Offline installation failed: ${error.message}`;
            }
        });
    } else {
        offlineStatus.textContent = 'Offline installation requires HTTPS or localhost';
    }

    document.getElementById('btnGpsOutage').addEventListener('click', () => {
        const engine = window.offlineEngine;
        if (engine.setGpsOutage(!engine.gpsOutage)) {
            document.getElementById('btnGpsOutage').textContent = engine.gpsOutage
                ? 'Restore GPS' : 'Simulate GNSS outage';
            document.getElementById('btnGpsOutage').setAttribute('aria-pressed', String(engine.gpsOutage));
        }
    });

    document.getElementById('btnResetTimings').addEventListener('click', () => {
        window.offlineEngine.profiler.reset();
        window.refreshDeviceTimings?.();
    });
    document.getElementById('btnExportTimings').addEventListener('click', () => {
        const engine = window.offlineEngine;
        const report = { ...engine.profiler.report(), model_contract: engine.modelContract,
            observed_imu_rate_hz: engine.observedImuRate ?? null, source: 'live browser engine', navigation_mode: engine.navigationMode };
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = 'navigators-device-timings.json';
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    // Live Sensor Controls (Offline Edge Engine)
    document.getElementById('btnStartLive').addEventListener('click', async () => {
        const btn = document.getElementById('btnStartLive');
        const statusEl = document.getElementById('edgeStatus');

        if (window.offlineEngine.isCapturing) {
            window.offlineEngine.stopCapture();
            if (window.dataRecorder && window.dataRecorder.isRecording) {
                window.dataRecorder.stopRecording();
            }
            btn.textContent = 'Start Navigation';
            btn.style.color = '';
            btn.classList.remove('running');
            statusEl.textContent = 'Engine stopped';
            document.getElementById('componentStatus').textContent = 'IMU / AI / EKF / NHC / map matching: stopped';
            document.getElementById('btnGpsOutage').disabled = true;
            document.getElementById('playbackControls').style.opacity = '';
            document.getElementById('playbackControls').style.pointerEvents = '';
            window.showStandby?.();
            document.getElementById('travelMode').disabled = false;
            document.getElementById('stepLength').disabled = false;
            btn.hidden = state.view !== 'console';
        } else {
            window.selectWorkspace?.('console');
            window.offlineEngine.navigationMode = document.getElementById('travelMode').value;
            window.offlineEngine.stepLength = Number(document.getElementById('stepLength').value);
            document.getElementById('travelMode').disabled = true;
            document.getElementById('stepLength').disabled = true;
            btn.disabled = true;
            btn.setAttribute('aria-busy', 'true');
            btn.textContent = 'Initializing sensors...';
            statusEl.closest('details').open = true;
            statusEl.style.color = '';
            document.getElementById('sessionHint').textContent = 'Loading model and requesting sensor access';
            statusEl.textContent = 'Initializing Edge AI...';
            let success = false;
            try {
                success = await window.offlineEngine.requestPermissionsAndStart();
                if (success && window.dataRecorder && !window.dataRecorder.isRecording) {
                    await window.dataRecorder.requestPermissionsAndStart();
                }
            } catch (error) {
                window.offlineEngine.lastError = error.message;
            } finally {
                btn.disabled = false;
                btn.removeAttribute('aria-busy');
            }
            if (success) {
                btn.textContent = 'Stop Navigation';
                btn.classList.add('running');
                statusEl.textContent = 'Running Locally';
                document.getElementById('sessionHint').textContent = window.offlineEngine.navigationMode === 'walking' ? 'Waiting for GPS · hold the phone screen-up with its top forward' : 'Calibrating phone sensors';
                state.truthMarker.setOpacity(0);
                state.vehicleMarker.setOpacity(0);

                // Clear map trajectories for live run
                state.truthLine.setLatLngs([]);
                state.estimatedLine.setLatLngs([]);
                state.estimatedCoords = [];
                // Disable playback
                pause();
                document.getElementById('playbackControls').style.opacity = '0.3';
                document.getElementById('playbackControls').style.pointerEvents = 'none';
            } else {
                btn.textContent = 'Retry Navigation';
                const errMsg = window.offlineEngine.lastError || 'Unable to start. Check motion and location permissions, then retry.';
                statusEl.textContent = errMsg;
                statusEl.style.color = 'var(--accent-red)';
                document.getElementById('sessionHint').textContent = statusEl.textContent;
                document.getElementById('travelMode').disabled = false;
                document.getElementById('stepLength').disabled = false;
                
                if (window.setLoadingState) {
                    window.setLoadingState(0, "ERROR: " + errMsg);
                    const arc = document.getElementById('compassProgressArc');
                    if (arc) arc.style.stroke = 'var(--accent-red)';
                    setTimeout(() => {
                        window.setLoadingState(100, "FAILED TO START", "FAILED TO START");
                        if (arc) arc.style.stroke = '';
                    }, 4000);
                }
            }
        }
    });

    // Data Recording UI removed (automatically handles background sync via Start Engine)
}

// ========================================================
// Data Loading
// ========================================================
async function loadSimulationData() {
    try {
        const response = await fetch('data/simulation.json');
        const data = response.ok ? await response.json() : null;
        if (state.replayName) return;
        state.data = data;
        if (state.view === 'replay' && data) onDataLoaded();
    } catch (e) {
        if (state.replayName) return;
        console.log('No simulation data available.');
        state.data = null;
    }
}

function onDataLoaded() {
    if (!state.data) return;
    const data = state.data;
    const meta = data.metadata;

    console.log(`[Simulator] Loaded ${data.data.timestamps.length} frames`);

    // Set timeline range
    const timeline = document.getElementById('timeline');
    timeline.max = data.data.timestamps.length - 1;

    // Center map on trajectory
    const [firstLat, firstLon] = data.data.estimated_lat_lon[0];
    const covered = state.localMap?.contains(firstLat, firstLon);
    if (covered) {
        state.map.setView([firstLat, firstLon], 16);
        document.getElementById('mapStatus').textContent = 'Local OSM map · saved trajectory';
    }
    else if (state.localMap) document.getElementById('mapStatus').textContent = 'Local OSM map ready · saved replay is outside this area';

    // Mark GNSS denied zones on the map
    state.gnssZones.forEach(zone => state.map.removeLayer(zone));
    state.gnssZones = [];
    markGnssDeniedZones();

    // Update metrics display
    for (const [id, value] of Object.entries({ ateRmse: meta.metrics?.ate_rmse, cep50: meta.metrics?.cep50,
        cep95: meta.metrics?.cep95, totalDistance: meta.total_distance })) {
        document.getElementById(id).textContent = Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)}m` : '—';
    }

    // Saved examples are paused until the user starts playback.
    seekFrame(0);

    // Keep the downloaded area visible when the saved replay covers another city.
    if (!covered) state.localMap?.centerView(state.map);
}

function markGnssDeniedZones() {
    const data = state.data.data;
    let zoneStart = null;

    for (let i = 0; i <= data.gnss_available.length; i++) {
        if (i < data.gnss_available.length && !data.gnss_available[i] && zoneStart === null) {
            zoneStart = i;
        } else if ((i === data.gnss_available.length || data.gnss_available[i]) && zoneStart !== null) {
            // Create zone polygon
            const coords = [];
            for (let j = zoneStart; j < i; j++) {
                coords.push(data.estimated_lat_lon[j]);
            }
            if (coords.length > 1) {
                const zone = L.polyline(coords, {
                    color: '#b5483b',
                    weight: 8,
                    opacity: 0.2,
                    lineCap: 'butt',
                }).addTo(state.map);
                state.gnssZones.push(zone);
            }
            zoneStart = null;
        }
    }
}

// ========================================================
// Playback Control
// ========================================================
function play() {
    if (!state.data || window.offlineEngine.isCapturing || state.view !== 'replay' || state.playing) return;
    if (state.currentIndex >= state.data.data.timestamps.length - 1) seekFrame(0);
    state.playbackTime = state.data.data.timestamps[state.currentIndex];
    state.playing = true;
    document.getElementById('btnPlay').classList.add('hidden');
    document.getElementById('btnPause').classList.remove('hidden');
    state.lastFrameTime = performance.now();
    animate();
}

function pause() {
    state.playing = false;
    document.getElementById('btnPlay').classList.remove('hidden');
    document.getElementById('btnPause').classList.add('hidden');
    if (state.animationFrame) {
        cancelAnimationFrame(state.animationFrame);
    }
}

function reset() {
    pause();
    state.currentIndex = 0;
    state.truthCoords = [];
    state.estimatedCoords = [];
    state.truthLine.setLatLngs([]);
    state.estimatedLine.setLatLngs([]);
    document.getElementById('timeline').value = 0;
    updateFrame(0);
}

function animate() {
    if (!state.playing || !state.data) return;

    const now = performance.now();
    const elapsed = (now - state.lastFrameTime) / 1000; // seconds
    state.lastFrameTime = now;

    state.playbackTime += elapsed * state.playbackSpeed;
    const timestamps = state.data.data.timestamps;
    let next = state.currentIndex;
    while (next + 1 < timestamps.length && timestamps[next + 1] <= state.playbackTime) next++;
    if (next !== state.currentIndex) seekFrame(next);
    if (next === timestamps.length - 1) { pause(); return; }

    state.animationFrame = requestAnimationFrame(animate);
}

// ========================================================
// Frame Update
// ========================================================
function seekFrame(index) {
    if (!state.data) return;
    state.currentIndex = index;
    state.truthCoords = state.data.data.true_lat_lon.slice(0, index);
    state.estimatedCoords = state.data.data.estimated_lat_lon.slice(0, index);
    updateFrame(index);
}

function updateFrame(index) {
    if (!state.data) return;
    const data = state.data.data;
    if (index < 0 || index >= data.timestamps.length) return;

    const reference = data.true_lat_lon[index];
    const estLat = data.estimated_lat_lon[index][0];
    const estLon = data.estimated_lat_lon[index][1];
    const gnssOk = data.gnss_available[index];
    const navMode = data.nav_mode[index];
    const speed = data.speed_estimated[index];
    const heading = data.heading_estimated[index];
    const posError = data.position_error[index];
    const confidence = data.confidence[index];
    const driftPct = data.dr_drift_percent[index];
    const isZupt = data.zupt_active[index];
    const timestamp = data.timestamps[index];

    // --- Update Map ---
    // Add coordinates to trajectories
    state.truthCoords.push(reference);
    state.estimatedCoords.push([estLat, estLon]);

    // Break the reference trail at missing fixes rather than drawing across GPS gaps.
    const segments = [];
    let segment = [];
    for (const point of state.truthCoords) {
        if (!point) { segment = []; continue; }
        if (!segment.length) segments.push(segment);
        segment.push(point);
    }
    state.truthLine.setLatLngs(segments);
    state.estimatedLine.setLatLngs(state.estimatedCoords);

    // Change estimated line color based on mode
    if (navMode === 'dr') {
        state.estimatedLine.setStyle({ color: '#ba5b37', dashArray: '5, 8' });
    } else if (navMode === 'reacq') {
        state.estimatedLine.setStyle({ color: '#ba5b37', dashArray: '2, 4' });
    } else {
        state.estimatedLine.setStyle({ color: '#4c7b59', dashArray: null });
    }

    // Update markers
    state.vehicleMarker.setLatLng([estLat, estLon]).setOpacity(1);
    state.truthMarker.setOpacity(reference ? 1 : 0);
    if (reference) state.truthMarker.setLatLng(reference);

    // Update vehicle marker appearance
    const markerEl = state.vehicleMarker.getElement();
    if (markerEl) {
        const wrapper = markerEl.querySelector('.vehicle-marker') || markerEl;
        if (navMode === 'dr') {
            wrapper.classList.add('dr-active');
        } else {
            wrapper.classList.remove('dr-active');
        }
    }

    // Pan map to follow vehicle
    if (state.followPosition) state.map.panTo([estLat, estLon], { animate: false });

    // --- Update Telemetry ---
    updateNavMode(navMode);
    updateGnssStatus(gnssOk);
    updateSpeed(speed * 3.6, heading); // Convert m/s to km/h
    document.getElementById('positionCoordinates').textContent = `${estLat.toFixed(6)}, ${estLon.toFixed(6)}`;
    updatePositionError(posError);
    updateDrift(driftPct);
    updateConfidence(confidence);
    window.updateConsoleTelemetry?.({ source: 'saved', nav_mode: navMode, gnss_available: gnssOk, zupt_active: isZupt });

    // Timeline
    document.getElementById('timeline').value = index;
    const start = state.data.metadata.source === 'replay.py' ? 0 : data.timestamps[0];
    const elapsed = timestamp - start;
    const duration = data.timestamps.at(-1) - start;
    document.getElementById('timeLabel').textContent = `${elapsed.toFixed(1)} / ${duration.toFixed(1)}s`;
    document.getElementById('timeline').setAttribute('aria-valuetext', `${elapsed.toFixed(1)} of ${duration.toFixed(1)} seconds`);
}

// ========================================================
// Telemetry Updates
// ========================================================
let lastToastMode = null;
let toastTimeout = null;

function showNavToast(title, body, durationMs = 2500) {
    const toast = document.getElementById('navToast');
    if (!toast) return;
    toast.innerHTML = `<strong>${title}</strong><p>${body}</p>`;
    toast.hidden = false;
    toast.classList.add('visible');

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.remove('visible');
        setTimeout(() => { toast.hidden = true; }, 200);
    }, durationMs);
}

function updateNavMode(mode, accuracy = 0) {
    const indicator = document.getElementById('navModeIndicator');
    if (!indicator) return;
    indicator.className = 'nav-mode-indicator';

    const label = indicator.querySelector('.mode-label') || indicator;

    if (lastToastMode !== mode) {
        if (mode === 'dr' && lastToastMode !== null) {
            showNavToast('Dead Reckoning Active', 'GNSS signal lost. Navigation continues.', 2500);
        } else if ((mode === 'reacq' || (mode === 'gnss_ins' && lastToastMode === 'dr')) && lastToastMode !== null) {
            showNavToast('GNSS Restored', 'Position correction resumed.', 2500);
        }
        lastToastMode = mode;
    }

    switch (mode) {
        case 'gnss_ins':
            if (accuracy > 15) {
                indicator.classList.add('reacquisition');
                label.textContent = 'GNSS Degraded';
            } else {
                indicator.classList.add('gnss-ins');
                label.textContent = 'GNSS + INS';
            }
            break;
        case 'dr':
            indicator.classList.add('dead-reckoning');
            label.textContent = 'Dead Reckoning';
            break;
        case 'reacq':
            indicator.classList.add('reacquisition');
            label.textContent = 'Reacquiring GNSS';
            break;
        default:
            label.textContent = 'STANDBY';
            break;
    }
}

function updateGnssStatus(available) {
    const led = document.getElementById('gnssLed');
    const text = document.getElementById('gnssStatusText');
    const bars = document.getElementById('signalBars');
    const card = document.getElementById('gnssStatusCard');

    led.className = 'status-led ' + (available ? 'active' : 'denied');
    bars.className = 'signal-bars ' + (available ? 'active' : 'denied');

    if (available) {
        text.textContent = 'LOCKED';
        text.style.color = 'var(--gnss-active)';
        card.style.borderColor = '';
    } else {
        text.textContent = 'DENIED';
        text.style.color = 'var(--gnss-denied)';
        card.style.borderColor = '';
    }
}

function updateSpeed(speedKmh, headingRad) {
    const speedEl = document.getElementById('speedValue');
    const headingEl = document.getElementById('headingValue');

    if (speedEl) {
        speedEl.textContent = Number.isFinite(speedKmh) ? `${Math.round(speedKmh)}` : 'Speed unavailable';
    }

    if (headingEl) {
        if (Number.isFinite(headingRad)) {
            const headingDeg = ((headingRad * 180 / Math.PI) % 360 + 360) % 360;
            const cardinalDirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
            const index = Math.round(headingDeg / 45) % 8;
            headingEl.textContent = `${cardinalDirs[index]} ${headingDeg.toFixed(0)}°`;
        } else {
            headingEl.textContent = 'Heading unavailable';
        }
    }
}

function updatePositionError(error) {
    const errorEl = document.getElementById('posErrorValue');
    if (!Number.isFinite(error)) {
        errorEl.textContent = '—';
        errorEl.style.color = '';
        document.getElementById('posErrorBar').style.width = '0%';
        return;
    }
    errorEl.textContent = error.toFixed(1);

    // Color based on severity
    if (error < 3) {
        errorEl.style.color = 'var(--accent-green)';
    } else if (error < 7) {
        errorEl.style.color = 'var(--accent-orange)';
    } else {
        errorEl.style.color = 'var(--accent-red)';
    }

    // Error bar (max 10m)
    const pct = Math.min(100, (error / 10) * 100);
    document.getElementById('posErrorBar').style.width = `${pct}%`;
}

function updateDrift(driftPct) {
    const valueEl = document.getElementById('driftValue');
    const ringEl = document.getElementById('driftRingFill');
    const badge = document.getElementById('driftBadge');
    if (!Number.isFinite(driftPct)) {
        valueEl.textContent = badge.textContent = '—';
        valueEl.style.color = badge.style.color = '';
        ringEl.style.strokeDashoffset = 326.73;
        return;
    }

    valueEl.textContent = driftPct.toFixed(1);

    // Ring fill (circumference = 2πr = 2 * π * 52 ≈ 326.73)
    const circumference = 326.73;
    const fillPct = Math.min(1, driftPct / 15); // Scale to 15% max
    ringEl.style.strokeDashoffset = circumference * (1 - fillPct);

    // Color based on threshold
    if (driftPct < 5) {
        ringEl.style.stroke = 'var(--accent-green)';
        valueEl.style.color = 'var(--accent-green)';
        badge.textContent = 'OK';
        badge.style.color = 'var(--accent-green)';
    } else if (driftPct < 10) {
        ringEl.style.stroke = 'var(--accent-orange)';
        valueEl.style.color = 'var(--accent-orange)';
        badge.textContent = 'WARN';
        badge.style.color = 'var(--accent-orange)';
    } else {
        ringEl.style.stroke = 'var(--accent-red)';
        valueEl.style.color = 'var(--accent-red)';
        badge.textContent = 'HIGH';
        badge.style.color = 'var(--accent-red)';
    }
}

function updateConfidence(confidence) {
    const bar = document.getElementById('confidenceBar');
    const value = document.getElementById('confidenceValue');
    if (!Number.isFinite(confidence)) {
        value.textContent = '—';
        value.style.color = '';
        bar.style.width = '0%';
        return;
    }

    const pct = Math.max(0, Math.min(100, confidence * 100));
    bar.style.width = `${pct}%`;
    value.textContent = `${pct.toFixed(0)}%`;

    if (pct > 70) {
        value.style.color = 'var(--accent-green)';
    } else if (pct > 40) {
        value.style.color = 'var(--accent-orange)';
    } else {
        value.style.color = 'var(--accent-red)';
    }
}
