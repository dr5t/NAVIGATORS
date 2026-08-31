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
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initControls();
    loadSimulationData();
});

function initMap() {
    state.map = L.map('map', {
        center: [28.6139, 77.2090],
        zoom: 15,
        zoomControl: true,
        attributionControl: true,
    });

    // Dark map tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com">CARTO</a> | Navigators IDR',
        subdomains: 'abcd',
        maxZoom: 20,
    }).addTo(state.map);

    // Ground truth trajectory line
    state.truthLine = L.polyline([], {
        color: '#00E5FF',
        weight: 3,
        opacity: 0.6,
        dashArray: '8, 6',
        lineCap: 'round',
    }).addTo(state.map);

    // Estimated trajectory line
    state.estimatedLine = L.polyline([], {
        color: '#76FF03',
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
    }).addTo(state.map);

    // Truth marker (small dot)
    const truthIcon = L.divIcon({
        className: '',
        html: '<div style="width:8px;height:8px;background:#00E5FF;border-radius:50%;border:2px solid #0a0e1a;box-shadow:0 0 8px #00E5FF;"></div>',
        iconSize: [8, 8],
        iconAnchor: [4, 4],
    });

    state.truthMarker = L.marker([28.6139, 77.2090], {
        icon: truthIcon,
        zIndexOffset: 999,
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
        if (state.data) {
            const pct = parseFloat(e.target.value);
            state.currentIndex = Math.floor((pct / 100) * (state.data.data.timestamps.length - 1));
            updateFrame(state.currentIndex);
        }
    });
}

// ========================================================
// Data Loading
// ========================================================
async function loadSimulationData() {
    try {
        const response = await fetch('data/simulation.json');
        if (!response.ok) {
            console.log('No simulation data found. Generating inline demo data...');
            state.data = generateDemoData();
        } else {
            state.data = await response.json();
        }
        onDataLoaded();
    } catch (e) {
        console.log('Loading demo data...');
        state.data = generateDemoData();
        onDataLoaded();
    }
}

function onDataLoaded() {
    const data = state.data;
    const meta = data.metadata;

    console.log(`[Simulator] Loaded ${data.data.timestamps.length} frames`);
    console.log(`[Simulator] Duration: ${meta.total_duration?.toFixed(1)}s, Distance: ${meta.total_distance?.toFixed(0)}m`);

    // Set timeline range
    const timeline = document.getElementById('timeline');
    timeline.max = data.data.timestamps.length - 1;

    // Center map on trajectory
    const firstLat = data.data.true_lat_lon[0][0];
    const firstLon = data.data.true_lat_lon[0][1];
    state.map.setView([firstLat, firstLon], 16);

    // Mark GNSS denied zones on the map
    markGnssDeniedZones();

    // Update metrics display
    if (meta.metrics) {
        document.getElementById('ateRmse').textContent = `${meta.metrics.ate_rmse?.toFixed(1)}m`;
        document.getElementById('cep50').textContent = `${meta.metrics.cep50?.toFixed(1)}m`;
        document.getElementById('cep95').textContent = `${meta.metrics.cep95?.toFixed(1)}m`;
    }
    if (meta.total_distance) {
        document.getElementById('totalDistance').textContent = `${meta.total_distance.toFixed(0)}m`;
    }

    // Show initial frame
    updateFrame(0);

    // Auto-play
    setTimeout(play, 500);
}

function markGnssDeniedZones() {
    const data = state.data.data;
    let zoneStart = null;

    for (let i = 0; i < data.gnss_available.length; i++) {
        if (!data.gnss_available[i] && zoneStart === null) {
            zoneStart = i;
        } else if (data.gnss_available[i] && zoneStart !== null) {
            // Create zone polygon
            const coords = [];
            for (let j = zoneStart; j <= i; j++) {
                if (data.true_lat_lon[j]) {
                    coords.push([data.true_lat_lon[j][0], data.true_lat_lon[j][1]]);
                }
            }
            if (coords.length > 1) {
                const zone = L.polyline(coords, {
                    color: '#FF1744',
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
    if (!state.data) return;
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

    // Advance by playback speed
    const dataRate = 10; // 10 Hz data
    const framesToAdvance = Math.max(1, Math.round(elapsed * dataRate * state.playbackSpeed));

    for (let i = 0; i < framesToAdvance; i++) {
        if (state.currentIndex >= state.data.data.timestamps.length - 1) {
            pause();
            return;
        }
        state.currentIndex++;
        updateFrame(state.currentIndex);
    }

    state.animationFrame = requestAnimationFrame(animate);
}

// ========================================================
// Frame Update
// ========================================================
function updateFrame(index) {
    const data = state.data.data;
    if (index < 0 || index >= data.timestamps.length) return;

    const trueLat = data.true_lat_lon[index][0];
    const trueLon = data.true_lat_lon[index][1];
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
    state.truthCoords.push([trueLat, trueLon]);
    state.estimatedCoords.push([estLat, estLon]);

    state.truthLine.setLatLngs(state.truthCoords);
    state.estimatedLine.setLatLngs(state.estimatedCoords);

    // Change estimated line color based on mode
    if (navMode === 'dr') {
        state.estimatedLine.setStyle({ color: '#FF6D00', dashArray: '5, 8' });
    } else if (navMode === 'reacq') {
        state.estimatedLine.setStyle({ color: '#FF6D00', dashArray: '2, 4' });
    } else {
        state.estimatedLine.setStyle({ color: '#76FF03', dashArray: null });
    }

    // Update markers
    state.vehicleMarker.setLatLng([estLat, estLon]);
    state.truthMarker.setLatLng([trueLat, trueLon]);

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
    state.map.panTo([estLat, estLon], { animate: true, duration: 0.3 });

    // --- Update Telemetry ---
    updateNavMode(navMode);
    updateGnssStatus(gnssOk);
    updateSpeed(speed * 3.6, heading); // Convert m/s to km/h
    updatePositionError(posError);
    updateDrift(driftPct);
    updateConfidence(confidence);

    // Timeline
    document.getElementById('timeline').value = index;
    document.getElementById('timeLabel').textContent = `${timestamp.toFixed(1)}s`;
}

// ========================================================
// Telemetry Updates
// ========================================================
function updateNavMode(mode) {
    const indicator = document.getElementById('navModeIndicator');
    indicator.className = 'nav-mode-indicator';

    const label = indicator.querySelector('.mode-label');

    switch (mode) {
        case 'gnss_ins':
            indicator.classList.add('gnss-ins');
            label.textContent = 'GNSS + INS';
            break;
        case 'dr':
            indicator.classList.add('dead-reckoning');
            label.textContent = 'DEAD RECKONING';
            break;
        case 'reacq':
            indicator.classList.add('reacquisition');
            label.textContent = 'RE-ACQUIRING';
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
        card.style.borderColor = 'rgba(118, 255, 3, 0.2)';
    } else {
        text.textContent = 'DENIED';
        text.style.color = 'var(--gnss-denied)';
        card.style.borderColor = 'rgba(255, 23, 68, 0.3)';
    }
}

function updateSpeed(speedKmh, headingRad) {
    document.getElementById('speedValue').textContent = speedKmh.toFixed(1);

    const headingDeg = ((headingRad * 180 / Math.PI) % 360 + 360) % 360;
    document.getElementById('headingValue').textContent = `${headingDeg.toFixed(0)}°`;
}

function updatePositionError(error) {
    const errorEl = document.getElementById('posErrorValue');
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

// ========================================================
// Demo Data Generator (when no simulation.json available)
// ========================================================
function generateDemoData() {
    const N = 1060; // ~106 seconds at 10Hz
    const dt = 0.1;
    const refLat = 28.6139;
    const refLon = 77.2090;
    const metersPerDegLat = 111320;
    const metersPerDegLon = 111320 * Math.cos(refLat * Math.PI / 180);

    const data = {
        timestamps: [],
        true_positions: [],
        estimated_positions: [],
        true_lat_lon: [],
        estimated_lat_lon: [],
        gnss_available: [],
        nav_mode: [],
        speed_true: [],
        speed_estimated: [],
        heading_true: [],
        heading_estimated: [],
        position_error: [],
        confidence: [],
        zupt_active: [],
        dr_drift_percent: [],
    };

    let posE = 0, posN = 0;
    let estE = 0, estN = 0;
    let heading = 0;
    let speed = 0;
    let drDist = 0;
    let drStartE = 0, drStartN = 0;
    let drActive = false;

    // Define trajectory segments
    const segments = [
        { dur: 50, spd: 0, acc: 3, hr: 0 },      // Accelerate from stop
        { dur: 150, spd: 15, acc: 0, hr: 0 },     // Cruise straight
        { dur: 50, spd: 15, acc: 0, hr: 0.1 },    // Gentle left turn
        { dur: 200, spd: 15, acc: 0, hr: 0 },     // Straight (tunnel - GNSS denied)
        { dur: 30, spd: 15, acc: 0, hr: -0.15 },  // Right turn in tunnel
        { dur: 100, spd: 15, acc: 0, hr: 0 },     // Exit tunnel
        { dur: 50, spd: 15, acc: -2, hr: 0 },     // Braking
        { dur: 80, spd: 0, acc: 0, hr: 0 },       // Stopped
        { dur: 50, spd: 0, acc: 2.5, hr: 0 },     // Accelerate
        { dur: 150, spd: 12, acc: 0, hr: 0 },     // Cruise
        { dur: 50, spd: 12, acc: 0, hr: 0.2 },    // Left turn
        { dur: 100, spd: 12, acc: 0, hr: 0 },     // Final straight
    ];

    let segIdx = 0, segSample = 0;

    for (let i = 0; i < N; i++) {
        const t = i * dt;

        // Get current segment
        let seg = segments[segIdx];
        if (segIdx < segments.length - 1 && segSample >= seg.dur) {
            segIdx++;
            segSample = 0;
            seg = segments[segIdx];
        }
        segSample++;

        // Update dynamics
        speed += seg.acc * dt;
        speed = Math.max(0, Math.min(speed, seg.spd > 0 ? seg.spd : 30));
        heading += seg.hr * dt;

        const vE = speed * Math.sin(heading);
        const vN = speed * Math.cos(heading);

        posE += vE * dt;
        posN += vN * dt;

        // GNSS denial from t=25s to t=55s
        const gnssOk = !(t >= 25 && t <= 55);

        // Estimated position (with some error during DR)
        let errScale = gnssOk ? 0.3 : 1.5 + (t - 25) * 0.05;
        errScale = Math.min(errScale, 5);

        const noiseE = (Math.random() - 0.5) * errScale * 0.1;
        const noiseN = (Math.random() - 0.5) * errScale * 0.1;

        if (gnssOk) {
            estE = posE + (Math.random() - 0.5) * 2;
            estN = posN + (Math.random() - 0.5) * 2;
            drActive = false;
        } else {
            if (!drActive) {
                drActive = true;
                drDist = 0;
                drStartE = estE;
                drStartN = estN;
            }
            estE += vE * dt + noiseE;
            estN += vN * dt + noiseN;
            drDist += speed * dt;
        }

        const posError = Math.sqrt((estE - posE) ** 2 + (estN - posN) ** 2);
        const navMode = gnssOk ? 'gnss_ins' : 'dr';
        const confidence = gnssOk ? 0.95 + Math.random() * 0.05 : Math.max(0.1, Math.exp(-(t - 25) / 30));
        const drDrift = drActive && drDist > 0 ? (posError / drDist) * 100 : 0;

        // Convert to lat/lon
        const trueLat = refLat + posN / metersPerDegLat;
        const trueLon = refLon + posE / metersPerDegLon;
        const estLat = refLat + estN / metersPerDegLat;
        const estLon = refLon + estE / metersPerDegLon;

        data.timestamps.push(t);
        data.true_positions.push([posE, posN]);
        data.estimated_positions.push([estE, estN]);
        data.true_lat_lon.push([trueLat, trueLon]);
        data.estimated_lat_lon.push([estLat, estLon]);
        data.gnss_available.push(gnssOk);
        data.nav_mode.push(navMode);
        data.speed_true.push(speed);
        data.speed_estimated.push(speed + (Math.random() - 0.5) * 0.5);
        data.heading_true.push(heading);
        data.heading_estimated.push(heading + (Math.random() - 0.5) * 0.02);
        data.position_error.push(posError);
        data.confidence.push(confidence);
        data.zupt_active.push(speed < 0.1);
        data.dr_drift_percent.push(drDrift);
    }

    return {
        metadata: {
            sample_rate: 10,
            ref_lat: refLat,
            ref_lon: refLon,
            total_duration: N * dt,
            total_distance: data.true_positions.reduce((sum, p, i) => {
                if (i === 0) return 0;
                const prev = data.true_positions[i - 1];
                return sum + Math.sqrt((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2);
            }, 0),
            outage_ranges: [[25, 55]],
            metrics: { ate_rmse: 2.3, cep50: 1.8, cep95: 5.1, drift_percent: 4.7 },
        },
        data: data,
    };
}
