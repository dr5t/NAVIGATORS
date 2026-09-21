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
