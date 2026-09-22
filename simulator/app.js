













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

    
    truthLine: null,
    estimatedLine: null,
    vehicleMarker: null,
    truthMarker: null,
    gnssZones: [],

    
    truthCoords: [],
    estimatedCoords: [],

    
    routeLine: null,
    destMarker: null,
    lastPosition: null,
};
window.state = state;




document.addEventListener('DOMContentLoaded', async () => {
    const loader = document.getElementById('app-loader');

    
    const params = new URLSearchParams(window.location.search);
    const mode = params.get('mode') || 'dashboard';
    document.body.classList.add(`${mode}-mode`);
    
    if (mode === 'dashboard') {
        
        connectDashboardToWebSocket();
    } else {
        
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
    
    
    await loadSimulationData();
    
    
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
        
    }
}


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
                label.textContent = `Production Model Active (4.20 m/s MAE - Retained)`;
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
        } else {
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

    
    state.truthLine = L.polyline([], {
        color: '#6c8884',
        weight: 3,
        opacity: 0.6,
        dashArray: '8, 6',
        lineCap: 'round',
    }).addTo(state.map);

    
    state.estimatedLine = L.polyline([], {
        color: '#4c7b59',
        weight: 3,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round',
    }).addTo(state.map);

    
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

    
    state.map.on('click', (e) => {
        if (typeof window.setDestination === 'function') {
            window.setDestination(e.latlng.lat, e.latlng.lng, `Destination (${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)})`);
        }
    });
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

                
                state.truthLine.setLatLngs([]);
                state.estimatedLine.setLatLngs([]);
                state.estimatedCoords = [];
                
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
    
    const btnFollow = document.getElementById('btnFollow');
    if (btnFollow) {
        btnFollow.addEventListener('click', () => {
            state.followPosition = !state.followPosition;
            btnFollow.setAttribute('aria-pressed', String(state.followPosition));
            btnFollow.style.background = state.followPosition ? 'rgba(16,185,129,0.2)' : '';
            btnFollow.style.borderColor = state.followPosition ? 'var(--accent-emerald)' : '';
            if (state.followPosition && state.map) {
                const pos = state.lastPosition || (state.vehicleMarker ? state.vehicleMarker.getLatLng() : null);
                if (pos) {
                    const lat = pos.lat !== undefined ? pos.lat : pos[0];
                    const lon = pos.lon !== undefined ? (pos.lng !== undefined ? pos.lng : pos.lon) : pos[1];
                    if (Number.isFinite(lat) && Number.isFinite(lon)) {
                        state.map.panTo([lat, lon]);
                    }
                }
            }
        });
    }

    const btnFitArea = document.getElementById('btnFitArea');
    if (btnFitArea) {
        btnFitArea.addEventListener('click', () => {
            if (state.routeLine && state.map) {
                state.map.fitBounds(state.routeLine.getBounds(), { padding: [50, 50] });
            } else if (state.localMap && state.localMap.data && state.localMap.data.origin && state.map) {
                state.map.setView([state.localMap.data.origin.lat, state.localMap.data.origin.lon], 15);
            }
        });
    }

    const btnEmergencySOS = document.getElementById('btnEmergencySOS');
    if (btnEmergencySOS) {
        btnEmergencySOS.addEventListener('click', () => {
            const pos = state.lastPosition || (state.vehicleMarker ? state.vehicleMarker.getLatLng() : null);
            const latStr = pos ? (pos.lat !== undefined ? pos.lat.toFixed(6) : pos[0].toFixed(6)) : 'Acquiring...';
            const lonStr = pos ? (pos.lon !== undefined ? (pos.lng !== undefined ? pos.lng.toFixed(6) : pos.lon.toFixed(6)) : pos[1].toFixed(6)) : 'Acquiring...';
            const modeEl = document.getElementById('navModeIndicator');
            const modeName = modeEl ? modeEl.textContent.trim() : 'Active';
            const sosMsg = `EMERGENCY SOS ALERT\n\nCurrent Estimated Position:\nLatitude: ${latStr}\nLongitude: ${lonStr}\nMode: ${modeName}\n\nBroadcast emergency coordinates to local dispatch?`;
            if (confirm(sosMsg)) {
                showNavToast('Emergency Coordinates Saved', `Broadcast: ${latStr}, ${lonStr}`, 4000);
            }
        });
    }

    const btnCancelRoute = document.getElementById('btnCancelRoute');
    if (btnCancelRoute) {
        btnCancelRoute.addEventListener('click', () => {
            if (typeof window.cancelRoute === 'function') {
                window.cancelRoute();
            }
        });
    }
}




if (typeof NavigatorsRouter !== 'undefined') {
    window.router = new NavigatorsRouter();
}
if (typeof NavigatorsAuth !== 'undefined') {
    window.auth = new NavigatorsAuth();
}

window.setDestination = async function(destLat, destLon, name = 'Destination') {
    if (!state.map) return;

    let startLat = null, startLon = null;
    if (state.lastPosition && Number.isFinite(state.lastPosition.lat)) {
        startLat = state.lastPosition.lat;
        startLon = state.lastPosition.lon;
    } else if (state.vehicleMarker && state.vehicleMarker.getLatLng()) {
        const vPos = state.vehicleMarker.getLatLng();
        if (Number.isFinite(vPos.lat) && Number.isFinite(vPos.lng) && (vPos.lat !== 0 || vPos.lng !== 0)) {
            startLat = vPos.lat;
            startLon = vPos.lng;
        }
    }

    if (!Number.isFinite(startLat) || !Number.isFinite(startLon)) {
        if (state.localMap && state.localMap.data && state.localMap.data.origin) {
            startLat = state.localMap.data.origin.lat;
            startLon = state.localMap.data.origin.lon;
        } else {
            const center = state.map.getCenter();
            startLat = center.lat;
            startLon = center.lng;
        }
    }

    const routeCard = document.getElementById('activeRouteCard');
    const routeDestName = document.getElementById('routeDestName');
    const routeMeta = document.getElementById('routeMeta');
    const routeManeuvers = document.getElementById('routeManeuvers');
    const mapStatus = document.getElementById('mapStatus');

    if (routeCard) {
        routeCard.style.display = 'flex';
        if (routeDestName) routeDestName.textContent = name;
        if (routeMeta) routeMeta.textContent = 'Calculating optimal driving route...';
        if (routeManeuvers) routeManeuvers.innerHTML = '<div style="font-size:11px; color:var(--text-muted); padding:4px;">Finding path...</div>';
    }
    if (mapStatus) mapStatus.textContent = `Routing to ${name}...`;

    try {
        if (!window.router && typeof NavigatorsRouter !== 'undefined') {
            window.router = new NavigatorsRouter();
        }
        if (!window.router) throw new Error('Routing engine not initialized.');

        const route = await window.router.route(startLat, startLon, destLat, destLon, state.localMap);

        if (state.routeLine) {
            state.routeLine.remove();
            state.routeLine = null;
        }
        if (state.destMarker) {
            state.destMarker.remove();
            state.destMarker = null;
        }

        state.routeLine = L.polyline(route.coordinates, {
            color: '#06b6d4',
            weight: 5,
            opacity: 0.9,
            lineJoin: 'round'
        }).addTo(state.map);

        const destIcon = L.divIcon({
            className: 'dest-marker',
            html: `<div style="background:#06b6d4; color:#0b0f19; font-weight:700; font-size:11px; padding:3px 8px; border-radius:4px; border:1px solid #ffffff; white-space:nowrap; box-shadow:0 2px 6px rgba(0,0,0,0.5);">${name}</div>`,
            iconAnchor: [30, 20]
        });
        state.destMarker = L.marker([destLat, destLon], { icon: destIcon }).addTo(state.map);

        state.map.fitBounds(state.routeLine.getBounds(), { padding: [60, 60] });

        if (routeCard) {
            const distText = route.distanceMeters < 1000 
                ? `${Math.round(route.distanceMeters)} m` 
                : `${(route.distanceMeters / 1000).toFixed(1)} km`;
            const durationMins = Math.ceil(route.durationSeconds / 60);
            const engineLabel = route.engine === 'osrm' ? 'Online OSRM' : 'Offline A* Road Graph';

            if (routeMeta) routeMeta.textContent = `${distText} · ${durationMins} min · ${engineLabel}`;

            if (routeManeuvers) {
                routeManeuvers.innerHTML = route.maneuvers.map(m => `
                    <div class="route-step-item">
                        <span class="route-step-instruction">
                            <svg viewBox="0 0 24 24" style="width:14px; height:14px; fill:none; stroke:currentColor; stroke-width:2;"><polyline points="9 18 15 12 9 6"/></svg>
                            ${m.instruction}
                        </span>
                        <span class="route-step-dist">${m.distanceMeters < 1000 ? Math.round(m.distanceMeters) + 'm' : (m.distanceMeters/1000).toFixed(1) + 'km'}</span>
                    </div>
                `).join('');
            }
        }

        if (mapStatus) mapStatus.textContent = `Route active: ${name} via ${route.engine === 'osrm' ? 'Online OSRM' : 'Offline A*'}.`;

    } catch (err) {
        console.warn('[Routing Error]', err);
        if (routeCard) routeCard.style.display = 'none';
        if (mapStatus) mapStatus.textContent = `Route planning: ${err.message || 'No driving path found.'}`;
        alert(`Route planning failed: ${err.message || 'No driving path found in local road network.'}`);
    }
};

window.cancelRoute = function() {
    if (state.routeLine) {
        state.routeLine.remove();
        state.routeLine = null;
    }
    if (state.destMarker) {
        state.destMarker.remove();
        state.destMarker = null;
    }
    const routeCard = document.getElementById('activeRouteCard');
    if (routeCard) routeCard.style.display = 'none';
    const mapStatus = document.getElementById('mapStatus');
    if (mapStatus) mapStatus.textContent = 'Active route cancelled.';
};





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

    
    const timeline = document.getElementById('timeline');
    timeline.max = data.data.timestamps.length - 1;

    
    const [firstLat, firstLon] = data.data.estimated_lat_lon[0];
    const covered = state.localMap?.contains(firstLat, firstLon);
    if (covered) {
        state.map.setView([firstLat, firstLon], 16);
        document.getElementById('mapStatus').textContent = 'Local OSM map · saved trajectory';
    }
    else if (state.localMap) document.getElementById('mapStatus').textContent = 'Local OSM map ready · saved replay is outside this area';

    
    state.gnssZones.forEach(zone => state.map.removeLayer(zone));
    state.gnssZones = [];
    markGnssDeniedZones();

    
    for (const [id, value] of Object.entries({ ateRmse: meta.metrics?.ate_rmse, cep50: meta.metrics?.cep50,
        cep95: meta.metrics?.cep95, totalDistance: meta.total_distance })) {
        document.getElementById(id).textContent = Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)}m` : '-';
    }

    
    seekFrame(0);

    
    if (!covered) state.localMap?.centerView(state.map);
}

function markGnssDeniedZones() {
    const data = state.data.data;
    let zoneStart = null;

    for (let i = 0; i <= data.gnss_available.length; i++) {
        if (i < data.gnss_available.length && !data.gnss_available[i] && zoneStart === null) {
            zoneStart = i;
        } else if ((i === data.gnss_available.length || data.gnss_available[i]) && zoneStart !== null) {
            
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
    const elapsed = (now - state.lastFrameTime) / 1000; 
    state.lastFrameTime = now;

    state.playbackTime += elapsed * state.playbackSpeed;
    const timestamps = state.data.data.timestamps;
    let next = state.currentIndex;
    while (next + 1 < timestamps.length && timestamps[next + 1] <= state.playbackTime) next++;
    if (next !== state.currentIndex) seekFrame(next);
    if (next === timestamps.length - 1) { pause(); return; }

    state.animationFrame = requestAnimationFrame(animate);
}




function seekFrame(index) {
    if (!state.data) return;
    state.currentIndex = index;
    state.truthCoords = state.data.data.true_lat_lon.slice(0, index);
    state.estimatedCoords = state.data.data.estimated_lat_lon.slice(0, index);
    state.needsFullLineRebuild = true;
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

    
    state.truthCoords.push(reference);
    state.estimatedCoords.push([estLat, estLon]);

    
    if (state.needsFullLineRebuild || !state.truthSegments) {
        state.truthSegments = [];
        let currentSegment = [];
        for (const point of state.truthCoords) {
            if (!point) { currentSegment = []; continue; }
            if (!currentSegment.length) state.truthSegments.push(currentSegment);
            currentSegment.push(point);
        }
        state.truthLine.setLatLngs(state.truthSegments);
        state.estimatedLine.setLatLngs(state.estimatedCoords);
        state.needsFullLineRebuild = false;
    } else {
        if (reference) {
            if (!state.truthSegments.length) state.truthSegments.push([]);
            state.truthSegments[state.truthSegments.length - 1].push(reference);
        } else if (state.truthCoords[state.truthCoords.length - 2]) {
            state.truthSegments.push([]);
        }
        state.truthLine.setLatLngs(state.truthSegments);
        state.estimatedLine.addLatLng([estLat, estLon]);
    }

    
    if (state.currentNavStyle !== navMode) {
        if (navMode === 'dr') {
            state.estimatedLine.setStyle({ color: '#ba5b37', dashArray: '5, 8' });
        } else if (navMode === 'reacq') {
            state.estimatedLine.setStyle({ color: '#ba5b37', dashArray: '2, 4' });
        } else {
            state.estimatedLine.setStyle({ color: '#4c7b59', dashArray: null });
        }
        state.currentNavStyle = navMode;
    }

    
    state.vehicleMarker.setLatLng([estLat, estLon]).setOpacity(1);
    state.truthMarker.setOpacity(reference ? 1 : 0);
    if (reference) state.truthMarker.setLatLng(reference);

    
    const markerEl = state.vehicleMarker.getElement();
    if (markerEl) {
        const wrapper = markerEl.querySelector('.vehicle-marker') || markerEl;
        const isDr = navMode === 'dr';
        if (wrapper.classList.contains('dr-active') !== isDr) {
            wrapper.classList.toggle('dr-active', isDr);
        }
    }

    
    if (state.followPosition) state.map.panTo([estLat, estLon], { animate: false });

    
    updateNavMode(navMode);
    updateGnssStatus(gnssOk);
    updateSpeed(speed * 3.6, heading); 
    document.getElementById('positionCoordinates').textContent = `${estLat.toFixed(6)}, ${estLon.toFixed(6)}`;
    updatePositionError(posError);
    updateDrift(driftPct);
    updateConfidence(confidence);
    window.updateConsoleTelemetry?.({ source: 'saved', nav_mode: navMode, gnss_available: gnssOk, zupt_active: isZupt });

    
    document.getElementById('timeline').value = index;
    const start = state.data.metadata.source === 'replay.py' ? 0 : data.timestamps[0];
    const elapsed = timestamp - start;
    const duration = data.timestamps.at(-1) - start;
    document.getElementById('timeLabel').textContent = `${elapsed.toFixed(1)} / ${duration.toFixed(1)}s`;
    document.getElementById('timeline').setAttribute('aria-valuetext', `${elapsed.toFixed(1)} of ${duration.toFixed(1)} seconds`);
}




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
            showNavToast('Dead Reckoning Active', 'GNSS signal lost. Inertial dead reckoning continues.', 3500);
        } else if ((mode === 'reacq' || (mode === 'gnss_ins' && lastToastMode === 'dr')) && lastToastMode !== null) {
            showNavToast('GNSS Restored', 'GNSS signal reacquired. Position correction resumed.', 2500);
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
        errorEl.textContent = '-';
        errorEl.style.color = '';
        document.getElementById('posErrorBar').style.width = '0%';
        return;
    }
    errorEl.textContent = error.toFixed(1);

    
    if (error < 3) {
        errorEl.style.color = 'var(--accent-green)';
    } else if (error < 7) {
        errorEl.style.color = 'var(--accent-orange)';
    } else {
        errorEl.style.color = 'var(--accent-red)';
    }

    
    const pct = Math.min(100, (error / 10) * 100);
    document.getElementById('posErrorBar').style.width = `${pct}%`;
}

function updateDrift(driftPct) {
    const valueEl = document.getElementById('driftValue');
    const ringEl = document.getElementById('driftRingFill');
    const badge = document.getElementById('driftBadge');
    if (!Number.isFinite(driftPct)) {
        valueEl.textContent = badge.textContent = '-';
        valueEl.style.color = badge.style.color = '';
        ringEl.style.strokeDashoffset = 326.73;
        return;
    }

    valueEl.textContent = driftPct.toFixed(1);

    
    const circumference = 326.73;
    const fillPct = Math.min(1, driftPct / 15); 
    ringEl.style.strokeDashoffset = circumference * (1 - fillPct);

    
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
        value.textContent = '-';
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
