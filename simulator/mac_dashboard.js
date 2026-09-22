const byId = id => document.getElementById(id);

// --- Navigation ---
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        byId(btn.dataset.target).classList.add('active');
    });
});

// --- WebSocket Connection ---
let ws;
let receivedSamples = 0;
let isConnected = false;

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws?role=dashboard`);
    
    ws.onopen = () => {
        isConnected = true;
        byId('connState').textContent = 'Connected to Server';
        document.querySelector('.status-indicator').className = 'status-indicator connected';
        logMessage('SUCCESS', 'WebSocket connected to backend.');
    };
    
    ws.onclose = () => {
        isConnected = false;
        byId('connState').textContent = 'Disconnected';
        document.querySelector('.status-indicator').className = 'status-indicator';
        byId('badgePhone').textContent = 'Phone: OFFLINE';
        byId('badgePhone').className = 'badge offline';
        logMessage('ERROR', 'WebSocket disconnected. Retrying in 5s...');
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        } catch (e) {
            console.error('Invalid WS message', e);
        }
    };
}

// --- Data Handling ---
function handleMessage(msg) {
    if (msg.type === 'telemetry') {
        updateTelemetry(msg.payload);
        
        // Update Phone Status
        if (byId('badgePhone').textContent.includes('OFFLINE')) {
            byId('badgePhone').textContent = 'Phone: ONLINE';
            byId('badgePhone').className = 'badge online';
            logMessage('INFO', 'Phone began streaming telemetry.');
        }
        
    } else if (msg.type === 'log') {
        logMessage(msg.severity || 'INFO', msg.message);
    } else if (msg.type === 'sync_event') {
        byId('valSyncStatus').textContent = msg.status;
        if(msg.status === 'SYNCED') {
            refreshDataset();
        }
    }
}

let lastTelemetryTime = performance.now();
function updateTelemetry(payload) {
    receivedSamples++;
    byId('valReceived').textContent = receivedSamples.toLocaleString();
    
    const now = performance.now();
    const dt = now - lastTelemetryTime;
    if (dt > 1000) {
        // We received X samples in the last second?
        // Let's just estimate it, or assume 1 if it's the first.
    }
    lastTelemetryTime = now;
    
    if (payload.estimated) {
        const est = payload.estimated;
        byId('valLat').textContent = est.estimated_lat.toFixed(6);
        byId('valLon').textContent = est.estimated_lon.toFixed(6);
        byId('valHeading').textContent = est.heading.toFixed(1) + '°';
        byId('valSpeed').textContent = (est.speed * 3.6).toFixed(1) + ' km/h';
    }
    
    if (payload.raw) {
        const r = payload.raw;
        if (r.accel && r.accel.length === 3) drawAccel(r.accel);
        if (r.gyro && r.gyro.length === 3) drawGyro(r.gyro);
    }
}

// --- Canvas Charts ---
const accelCtx = byId('accelCanvas').getContext('2d');
const gyroCtx = byId('gyroCanvas').getContext('2d');
const MAX_HISTORY = 100;
let accelHistory = { x: [], y: [], z: [] };
let gyroHistory = { x: [], y: [], z: [] };

function drawChart(ctx, history, min, max) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    ctx.clearRect(0, 0, width, height);
    
    const stepX = width / MAX_HISTORY;
    const range = max - min;
    
    const drawLine = (data, color) => {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        for (let i = 0; i < data.length; i++) {
            const x = i * stepX;
            const y = height - ((data[i] - min) / range * height);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    };
    
    drawLine(history.x, '#f44336'); // Red
    drawLine(history.y, '#4caf50'); // Green
    drawLine(history.z, '#2196f3'); // Blue
}

function drawAccel(accel) {
    accelHistory.x.push(accel[0]); accelHistory.y.push(accel[1]); accelHistory.z.push(accel[2]);
    if (accelHistory.x.length > MAX_HISTORY) {
        accelHistory.x.shift(); accelHistory.y.shift(); accelHistory.z.shift();
    }
    drawChart(accelCtx, accelHistory, -20, 20);
}
function drawGyro(gyro) {
    gyroHistory.x.push(gyro[0]); gyroHistory.y.push(gyro[1]); gyroHistory.z.push(gyro[2]);
    if (gyroHistory.x.length > MAX_HISTORY) {
        gyroHistory.x.shift(); gyroHistory.y.shift(); gyroHistory.z.shift();
    }
    drawChart(gyroCtx, gyroHistory, -5, 5);
}

// --- System Logs ---
function logMessage(severity, message) {
    const container = byId('logConsole');
    const entry = document.createElement('div');
    entry.className = `log-entry ${severity.toLowerCase()}`;
    
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-text">${message}</span>`;
    
    container.appendChild(entry);
    if (container.childElementCount > 200) {
        container.removeChild(container.firstChild);
    }
    container.scrollTop = container.scrollHeight;
}

// --- API Calls ---
async function refreshDataset() {
    try {
        const res = await fetch('/recordings/status', { headers: { 'Authorization': 'Bearer ' + (sessionStorage.getItem('pc-pairing-token') || '') }});
        if (res.ok) {
            const data = await res.json();
            byId('valTotalTrips').textContent = data.completed_trips;
            // Additional endpoint needed for /dataset/details
            fetchDatasetDetails();
        }
    } catch (e) {
        console.error('Failed to fetch dataset status', e);
    }
}

async function fetchDatasetDetails() {
    try {
        const res = await fetch('/dataset/details');
        if (res.ok) {
            const data = await res.json();
            byId('valTotalSamples').textContent = data.total_samples.toLocaleString();
            byId('valDatasetVersion').textContent = data.dataset_version;
            
            const tbody = byId('tripTable').querySelector('tbody');
            tbody.innerHTML = '';
            data.trips.forEach(trip => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${trip.id}</td>
                    <td>${trip.samples.toLocaleString()}</td>
                    <td><span class="badge ${trip.split}">${trip.split}</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch(e) {}
}

async function startTraining() {
    const confirmed = confirm("The verified 4.20 m/s production model is currently active.\n\nDo you want to start a new training job?");
    if (!confirmed) return;
    
    const btn = byId('btnStartTraining');
    btn.disabled = true;
    byId('trainStatusLabel').style.color = '';
    const epochs = parseInt(byId('inputEpochs').value) || 40;
    const batchSize = parseInt(byId('inputBatchSize').value) || 256;
    
    try {
        const res = await fetch('/training/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ epochs, batch_size: batchSize })
        });
        if (res.ok) {
            logMessage('SUCCESS', 'Training job queued.');
            pollTrainingStatus();
        } else {
            const err = await res.json();
            const errMsg = err.detail || err.error;
            logMessage('ERROR', `Failed to start training: ${errMsg}`);
            byId('trainStatusLabel').textContent = `Failed: ${errMsg}`;
            byId('trainStatusLabel').style.color = '#ef4444';
            btn.disabled = false;
        }
    } catch (e) {
        logMessage('ERROR', `Error starting training: ${e.message}`);
        byId('trainStatusLabel').textContent = `Failed: ${e.message}`;
        byId('trainStatusLabel').style.color = '#ef4444';
        btn.disabled = false;
    }
}

let trainingPollInterval;
async function pollTrainingStatus() {
    const btn = byId('btnStartTraining');
    if (trainingPollInterval) clearInterval(trainingPollInterval);
    
    const checkStatus = async () => {
        try {
            const res = await fetch('/training/status');
            if (res.ok) {
                const data = await res.json();
                
                if (byId('badgeActiveModel')) {
                    byId('badgeActiveModel').textContent = 'Active: Production Model (4.20 m/s)';
                    byId('badgeActiveModel').style.background = '#10b981';
                    byId('badgeActiveModel').style.color = '#ffffff';
                }
                byId('trainStatusLabel').textContent = data.status;
                
                // If there's an error status string, colour it red
                if (data.status.startsWith('Error') || data.status.startsWith('Failed')) {
                    byId('trainStatusLabel').style.color = '#ef4444';
                } else {
                    byId('trainStatusLabel').style.color = '';
                }

                if (data.train_samples) {
                    if (byId('trainSampleCount')) byId('trainSampleCount').textContent = data.train_samples.toLocaleString();
                    if (byId('valSampleCount')) byId('valSampleCount').textContent = data.val_samples.toLocaleString();
                    if (byId('testSampleCount')) byId('testSampleCount').textContent = data.test_samples.toLocaleString();
                }

                if (data.is_training) {
                    btn.disabled = true;
                    const pct = data.total_epochs > 0 ? (data.current_epoch / data.total_epochs) * 100 : 0;
                    byId('trainProgressFill').style.width = `${pct}%`;
                    byId('trainEpoch').textContent = `${data.current_epoch} / ${data.total_epochs}`;
                    byId('trainLoss').textContent = data.train_loss.toFixed(4);
                    byId('valLoss').textContent = data.val_loss.toFixed(4);
                    byId('trainEta').textContent = data.eta || 'Calculating...';
                } else {
                    clearInterval(trainingPollInterval);
                    btn.disabled = false;
                    if (data.status === 'Training Complete') {
                        byId('trainProgressFill').style.width = `100%`;
                        if (data.total_epochs) byId('trainEpoch').textContent = `${data.total_epochs} / ${data.total_epochs}`;
                        if (data.metrics) {
                            if (byId('trainFinalMetrics')) byId('trainFinalMetrics').style.display = 'block';
                            if (byId('metricTestMae')) byId('metricTestMae').textContent = data.metrics.candidate_test_mae?.toFixed(4) ?? '—';
                            if (byId('metricTestRmse')) byId('metricTestRmse').textContent = data.metrics.candidate_test_rmse?.toFixed(4) ?? '—';
                            if (byId('metricZeroBaselineMae')) byId('metricZeroBaselineMae').textContent = data.metrics.zero_velocity_mae?.toFixed(4) ?? '—';
                            if (byId('metricBaselineMae')) byId('metricBaselineMae').textContent = data.metrics.mean_velocity_mae?.toFixed(4) ?? '—';
                            if (byId('metricErrorReduction')) byId('metricErrorReduction').textContent = data.metrics.error_reduction_pct?.toFixed(2) ?? '—';
                            if (byId('metricPromotionStatus')) {
                                byId('metricPromotionStatus').textContent = data.metrics.promotion_status ?? 'Complete';
                                byId('metricPromotionStatus').style.color = data.metrics.candidate_promoted ? '#10b981' : '#f59e0b';
                            }
                        }
                        logMessage('SUCCESS', `Training Complete: ${data.metrics?.promotion_status || 'Ready'}`);
                    }
                }
            }
        } catch(e) {}
    };

    checkStatus();
    trainingPollInterval = setInterval(checkStatus, 2000);
}

// Initial setup
byId('btnRefreshDataset').addEventListener('click', refreshDataset);
byId('btnStartTraining').addEventListener('click', startTraining);
if (byId('btnImportDataset')) {
    byId('btnImportDataset').addEventListener('click', async () => {
        const url = byId('inputExternalDatasetUrl').value;
        if (!url) {
            logMessage('ERROR', 'Please enter a dataset URL.');
            return;
        }
        logMessage('INFO', `Connecting to ${url}...`);
        
        const btn = byId('btnImportDataset');
        btn.disabled = true;
        btn.textContent = 'Importing...';
        
        try {
            const res = await fetch('/dataset/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || 'Dataset import failed');
            }
            logMessage('SUCCESS', data.message || `Dataset successfully verified and imported from ${url}`);
            byId('inputExternalDatasetUrl').value = '';
            refreshDataset();
        } catch (err) {
            logMessage('ERROR', `Import failed: ${err.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Import Dataset';
        }
    });
}

connectWebSocket();
refreshDataset();
pollTrainingStatus();

// =============================================================================
// Phase 13 — Submitted Datasets
// =============================================================================

const DS_STATUS_COLORS = {
    uploaded:   '#f59e0b',
    validating: '#3b82f6',
    validated:  '#10b981',
    rejected:   '#ef4444',
};

const DS_ACTIVITY_LABELS = {
    walking:     'Walking',
    driving:     'Driving',
    gnss_imu:    'GNSS + IMU',
    gnss_outage: 'GNSS Outage',
    validation:  'Validation',
};

let dsCurrentOffset = 0;
const DS_PAGE_SIZE = 20;

function dsFmtDuration(secs) {
    if (!secs) return '—';
    const m = Math.floor(secs / 60), s = Math.round(secs % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function dsBadge(status) {
    const color = DS_STATUS_COLORS[status] || '#64748b';
    const label = status.charAt(0).toUpperCase() + status.slice(1);
    return `<span style="background:${color}22;color:${color};border:1px solid ${color}55;border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:600;">${label}</span>`;
}

function dsActionButtons(session) {
    const s = session.status;
    let btns = '';
    if (s === 'uploaded') {
        btns += `<button onclick="dsStartValidation('${session.id}')" style="background:#3b82f622;color:#3b82f6;border:1px solid #3b82f655;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:0.78rem;margin-right:4px;">Review</button>`;
        btns += `<button onclick="dsOpenRejectModal('${session.id}')" style="background:#ef444422;color:#ef4444;border:1px solid #ef444455;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:0.78rem;">Reject</button>`;
    } else if (s === 'validating') {
        btns += `<button onclick="dsValidate('${session.id}')" style="background:#10b98122;color:#10b981;border:1px solid #10b98155;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:0.78rem;margin-right:4px;">✓ Validate</button>`;
        btns += `<button onclick="dsOpenRejectModal('${session.id}')" style="background:#ef444422;color:#ef4444;border:1px solid #ef444455;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:0.78rem;">Reject</button>`;
    } else {
        btns = '<span style="color:#475569;font-size:0.78rem;">—</span>';
    }
    return btns;
}

async function refreshDatasets() {
    // Fetch stats
    try {
        const r = await fetch('/api/v1/datasets/sessions/stats');
        if (r.ok) {
            const stats = await r.json();
            byId('dsStatUploaded').textContent   = stats.by_status?.uploaded   ?? 0;
            byId('dsStatValidating').textContent = stats.by_status?.validating ?? 0;
            byId('dsStatValidated').textContent  = stats.by_status?.validated  ?? 0;
            byId('dsStatRejected').textContent   = stats.by_status?.rejected   ?? 0;
            byId('dsStatTotal').textContent      = stats.total ?? 0;
            byId('dsStatReady').textContent      = stats.validated_for_training ?? 0;
        }
    } catch (e) { /* dashboard may be open before server starts */ }

    // Fetch sessions with current filters
    const status       = byId('dsFilterStatus')?.value   || '';
    const activityType = byId('dsFilterActivity')?.value || '';
    let url = `/api/v1/datasets/sessions?limit=${DS_PAGE_SIZE}&offset=${dsCurrentOffset}`;
    if (status)       url += `&status=${encodeURIComponent(status)}`;
    if (activityType) url += `&activity_type=${encodeURIComponent(activityType)}`;

    try {
        const r = await fetch(url);
        if (!r.ok) {
            // 403 = not an internal contributor — hide the section gracefully
            byId('dsSessionTbody').innerHTML = `<tr><td colspan="8" style="text-align:center;color:#64748b;">No access or no sessions.</td></tr>`;
            return;
        }
        const data = await r.json();
        renderDsTable(data.sessions || []);
        renderDsPagination(data.total || 0);
    } catch (e) {
        byId('dsSessionTbody').innerHTML = `<tr><td colspan="8" style="text-align:center;color:#64748b;">Unable to load sessions.</td></tr>`;
    }
}

function renderDsTable(sessions) {
    const tbody = byId('dsSessionTbody');
    if (!sessions.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:#64748b;">No sessions found.</td></tr>`;
        return;
    }
    tbody.innerHTML = sessions.map(s => `
        <tr>
            <td style="font-family:monospace;font-size:0.78rem;">${s.id}</td>
            <td>${s.contributor_name || s.contributor_id}</td>
            <td>${DS_ACTIVITY_LABELS[s.activity_type] || s.activity_type}</td>
            <td style="font-size:0.82rem;">${s.device}</td>
            <td>${dsFmtDuration(s.duration_seconds)}</td>
            <td>${s.gnss_available ? '<span style="color:#10b981;">Yes</span>' : '<span style="color:#ef4444;">No</span>'}</td>
            <td>${dsBadge(s.status)}</td>
            <td>${dsActionButtons(s)}</td>
        </tr>
    `).join('');
}

function renderDsPagination(total) {
    const pages = Math.ceil(total / DS_PAGE_SIZE);
    const current = Math.floor(dsCurrentOffset / DS_PAGE_SIZE);
    const pag = byId('dsPagination');
    if (!pag) return;
    pag.innerHTML = '';
    if (pages <= 1) return;
    for (let i = 0; i < pages; i++) {
        const btn = document.createElement('button');
        btn.textContent = i + 1;
        btn.style.cssText = `background:${i === current ? '#3b82f6' : '#1e293b'};color:${i === current ? '#fff' : '#94a3b8'};border:1px solid #334155;border-radius:4px;padding:4px 10px;cursor:pointer;`;
        btn.addEventListener('click', () => { dsCurrentOffset = i * DS_PAGE_SIZE; refreshDatasets(); });
        pag.appendChild(btn);
    }
}

async function dsStartValidation(sessionId) {
    try {
        const r = await fetch(`/api/v1/datasets/sessions/${sessionId}/start-validation`, { method: 'POST' });
        if (r.ok) { logMessage('INFO', `Session ${sessionId} moved to Validating.`); refreshDatasets(); }
        else { const e = await r.json(); logMessage('ERROR', `Start validation failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

async function dsValidate(sessionId) {
    if (!confirm('Mark this session as Validated?\nIt will become eligible for training.')) return;
    try {
        const r = await fetch(`/api/v1/datasets/sessions/${sessionId}/validate`, { method: 'POST' });
        if (r.ok) { logMessage('SUCCESS', `Session ${sessionId} validated ✓`); refreshDatasets(); }
        else { const e = await r.json(); logMessage('ERROR', `Validation failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

function dsOpenRejectModal(sessionId) {
    byId('dsRejectTargetId').value = sessionId;
    byId('dsRejectReason').value = '';
    const modal = byId('dsRejectModal');
    modal.style.display = 'flex';
}

function dsCloseRejectModal() {
    byId('dsRejectModal').style.display = 'none';
}

async function dsConfirmReject() {
    const sessionId = byId('dsRejectTargetId').value;
    const reason    = byId('dsRejectReason').value.trim();
    if (!reason) { alert('Rejection reason is required.'); return; }
    dsCloseRejectModal();
    try {
        const r = await fetch(`/api/v1/datasets/sessions/${sessionId}/reject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rejection_reason: reason }),
        });
        if (r.ok) { logMessage('INFO', `Session ${sessionId} rejected.`); refreshDatasets(); }
        else { const e = await r.json(); logMessage('ERROR', `Reject failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

// Wire up Datasets tab controls
if (byId('btnRefreshDatasets'))
    byId('btnRefreshDatasets').addEventListener('click', () => { dsCurrentOffset = 0; refreshDatasets(); });

if (byId('btnApplyDsFilter'))
    byId('btnApplyDsFilter').addEventListener('click', () => { dsCurrentOffset = 0; refreshDatasets(); });

if (byId('dsRejectCancel'))
    byId('dsRejectCancel').addEventListener('click', dsCloseRejectModal);

if (byId('dsRejectConfirm'))
    byId('dsRejectConfirm').addEventListener('click', dsConfirmReject);

// Auto-refresh datasets when the tab becomes active
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.dataset.target === 'datasets') {
            dsCurrentOffset = 0;
            refreshDatasets();
        }
    });
});

// Initial load
refreshDatasets();

// --- Model Registry & Governance (Phase 15) ---
const MR_STATUS_BADGES = {
    candidate_training:   '<span style="background:#334155;color:#94a3b8;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Candidate Training</span>',
    evaluating:           '<span style="background:#1e3a8a;color:#93c5fd;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Evaluating</span>',
    review:               '<span style="background:#854d0e;color:#fef08a;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">In Review</span>',
    approved:             '<span style="background:#065f46;color:#a7f3d0;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Approved</span>',
    rejected:             '<span style="background:#991b1b;color:#fca5a5;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Rejected</span>',
    production_candidate: '<span style="background:#0284c7;color:#bae6fd;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Prod Candidate</span>',
    production:           '<span style="background:#047857;color:#ecfdf5;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">PRODUCTION</span>',
};

async function refreshModelRegistry() {
    try {
        const prodRes = await fetch('/api/v1/models/production');
        if (prodRes.ok) {
            const data = await prodRes.json();
            const prod = data.model;
            if (prod) {
                if (byId('prodModelName')) byId('prodModelName').textContent = prod.name;
                if (byId('prodModelMae')) byId('prodModelMae').textContent = prod.test_mae != null ? `${prod.test_mae.toFixed(4)} m/s` : '—';
                if (byId('prodModelRmse')) byId('prodModelRmse').textContent = prod.test_rmse != null ? `${prod.test_rmse.toFixed(4)} m/s` : '—';
                if (byId('prodModelDeployedAt')) byId('prodModelDeployedAt').textContent = prod.deployed_at ? new Date(prod.deployed_at).toLocaleString() : 'Baseline';
            } else {
                if (byId('prodModelName')) byId('prodModelName').textContent = 'None';
            }
        }
    } catch (e) {}

    const status = byId('mrFilterStatus')?.value || '';
    let url = '/api/v1/models?limit=50';
    if (status) url += `&status=${encodeURIComponent(status)}`;

    try {
        const r = await fetch(url);
        if (!r.ok) {
            if (byId('mrTbody')) byId('mrTbody').innerHTML = `<tr><td colspan="7" style="text-align:center;color:#64748b;">Unable to load Model Registry.</td></tr>`;
            return;
        }
        const data = await r.json();
        renderMrTable(data.models || []);
    } catch (e) {
        if (byId('mrTbody')) byId('mrTbody').innerHTML = `<tr><td colspan="7" style="text-align:center;color:#64748b;">Error loading Model Registry.</td></tr>`;
    }
}

function renderMrTable(models) {
    const tbody = byId('mrTbody');
    if (!tbody) return;
    if (!models.length) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#64748b;">No models found in registry.</td></tr>`;
        return;
    }
    tbody.innerHTML = models.map(m => `
        <tr>
            <td>
                <div style="font-weight:600;color:#f1f5f9;font-size:0.85rem;">${m.name}</div>
                <div style="font-family:monospace;font-size:0.75rem;color:#64748b;">${m.id}</div>
            </td>
            <td style="font-size:0.82rem;color:#cbd5e1;">${m.architecture}</td>
            <td style="font-weight:600;color:#e2e8f0;">${m.test_mae != null ? m.test_mae.toFixed(4) + ' m/s' : '—'}</td>
            <td style="font-size:0.82rem;color:#94a3b8;">${m.test_rmse != null ? m.test_rmse.toFixed(4) + ' m/s' : '—'}</td>
            <td style="font-size:0.82rem;color:#94a3b8;">${m.onnx_parity_max_diff != null ? m.onnx_parity_max_diff.toExponential(2) : '—'}</td>
            <td>${MR_STATUS_BADGES[m.status] || m.status}</td>
            <td>${mrActionButtons(m)}</td>
        </tr>
    `).join('');
}

function mrActionButtons(m) {
    if (m.status === 'evaluating') {
        return `<button onclick="mrOpenReview('${m.id}')" style="background:#2563eb;color:white;border:none;border-radius:4px;padding:4px 10px;font-size:0.78rem;cursor:pointer;font-weight:600;">Open Review</button>`;
    } else if (m.status === 'review') {
        return `<div style="display:flex;gap:6px;">
            <button onclick="mrApprove('${m.id}')" style="background:#059669;color:white;border:none;border-radius:4px;padding:4px 10px;font-size:0.78rem;cursor:pointer;font-weight:600;">Approve</button>
            <button onclick="mrOpenRejectModal('${m.id}')" style="background:#dc2626;color:white;border:none;border-radius:4px;padding:4px 10px;font-size:0.78rem;cursor:pointer;font-weight:600;">Reject</button>
        </div>`;
    } else if (m.status === 'production_candidate') {
        return `<button onclick="mrDeploy('${m.id}')" style="background:#7c3aed;color:white;border:none;border-radius:4px;padding:4px 12px;font-size:0.78rem;cursor:pointer;font-weight:700;">Deploy to Prod 🚀</button>`;
    } else if (m.status === 'production') {
        return `<span style="font-size:0.78rem;color:#10b981;font-weight:600;">Active Model</span>`;
    } else if (m.status === 'rejected') {
        const reason = m.rejection_reason ? m.rejection_reason.replace(/"/g, '&quot;') : '';
        return `<span style="font-size:0.78rem;color:#ef4444;" title="${reason}">Rejected</span>`;
    } else {
        return `<span style="font-size:0.78rem;color:#64748b;">No actions</span>`;
    }
}

async function mrOpenReview(modelId) {
    try {
        const r = await fetch(`/api/v1/models/${modelId}/open-review`, { method: 'POST' });
        if (r.ok) { logMessage('INFO', `Opened model ${modelId} for review.`); refreshModelRegistry(); }
        else { const e = await r.json(); logMessage('ERROR', `Open review failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

async function mrApprove(modelId) {
    if (!confirm('Approve this candidate model?\nIt will transition to Production Candidate state.')) return;
    try {
        const r = await fetch(`/api/v1/models/${modelId}/approve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes: "Approved via Mac Dashboard UI" })
        });
        if (r.ok) { logMessage('SUCCESS', `Approved model ${modelId} ✓`); refreshModelRegistry(); }
        else { const e = await r.json(); logMessage('ERROR', `Approval failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

function mrOpenRejectModal(modelId) {
    byId('mrRejectTargetId').value = modelId;
    byId('mrRejectReason').value = '';
    byId('mrRejectModal').style.display = 'flex';
}

function mrCloseRejectModal() {
    byId('mrRejectModal').style.display = 'none';
}

async function mrConfirmReject() {
    const modelId = byId('mrRejectTargetId').value;
    const reason  = byId('mrRejectReason').value.trim();
    if (!reason) { alert('Rejection reason is required.'); return; }
    mrCloseRejectModal();
    try {
        const r = await fetch(`/api/v1/models/${modelId}/reject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rejection_reason: reason })
        });
        if (r.ok) { logMessage('INFO', `Model ${modelId} rejected.`); refreshModelRegistry(); }
        else { const e = await r.json(); logMessage('ERROR', `Reject failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

async function mrDeploy(modelId) {
    if (!confirm('DEPLOY MODEL TO PRODUCTION?\nThis will overwrite active production checkpoint and ONNX files!')) return;
    try {
        const r = await fetch(`/api/v1/models/${modelId}/deploy`, { method: 'POST' });
        if (r.ok) { logMessage('SUCCESS', `Deployed model ${modelId} to PRODUCTION! 🚀`); refreshModelRegistry(); }
        else { const e = await r.json(); logMessage('ERROR', `Deployment failed: ${e.detail}`); }
    } catch (e) { logMessage('ERROR', `Error: ${e.message}`); }
}

if (byId('btnRefreshModelRegistry'))
    byId('btnRefreshModelRegistry').addEventListener('click', refreshModelRegistry);

if (byId('mrFilterStatus'))
    byId('mrFilterStatus').addEventListener('change', refreshModelRegistry);

if (byId('mrRejectCancel'))
    byId('mrRejectCancel').addEventListener('click', mrCloseRejectModal);

if (byId('mrRejectConfirm'))
    byId('mrRejectConfirm').addEventListener('click', mrConfirmReject);

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.dataset.target === 'model-registry') {
            refreshModelRegistry();
        }
    });
});

refreshModelRegistry();
