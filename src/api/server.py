"""
Navigators IDR - FastAPI Real-time Server
Receives live IMU & GNSS data from mobile web clients over WebSockets,
runs the IDR pipeline, and streams back the estimated navigation state.
"""

import os
import sys
import json
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from navigation.ekf import ExtendedKalmanFilter
from navigation.dead_reckoning import DeadReckoningEngine
from navigation.nhc import NonHolonomicConstraints
from navigation.zupt import ZUPTDetector
from navigation.map_matching import create_map_matcher, RoadNetwork
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Navigators IDR Backend")
from api.recordings import router as recordings_router
from api.training import router as training_router
from api.auth import router as auth_router
from api.contributions import router as contributions_router
from api.places import router as places_router
from api.moderation import router as moderation_router
from api.reports import router as reports_router
from api.sync import router as sync_router
from api.internal_contributors import router as internal_contributors_router
from api.datasets import router as datasets_router
from api.model_registry import router as model_registry_router
from api.audit import router as audit_router
from api.admin import router as admin_router
from api.routing import router as routing_router
from api.search import router as search_router
from api.sos import router as sos_router
from api.privacy import router as privacy_router
app.include_router(recordings_router)
app.include_router(training_router)
app.include_router(auth_router)
app.include_router(contributions_router)
app.include_router(places_router)
app.include_router(moderation_router)
app.include_router(reports_router)
app.include_router(sync_router)
app.include_router(internal_contributors_router)
app.include_router(datasets_router)
app.include_router(model_registry_router)
app.include_router(audit_router)
app.include_router(admin_router)
app.include_router(routing_router)
app.include_router(search_router)
app.include_router(sos_router)
app.include_router(privacy_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Load ML Model if available
try:
    from models.trainer import Trainer
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best_model.pt")
    stats_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "norm_stats.json")
    if os.path.exists(checkpoint_path):
        ml_model, _ = Trainer.load_checkpoint(checkpoint_path)
        ml_model.eval()
        
        import json
        NORM_MEAN = None
        NORM_STD = None
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
                NORM_MEAN = np.array(stats["mean"], dtype=np.float32)
                NORM_STD = np.array(stats["std"], dtype=np.float32)
                NORM_STD[NORM_STD < 1e-6] = 1.0
                
        print("[Server] Loaded ML Model successfully")
    else:
        ml_model = None
        NORM_MEAN = None
        NORM_STD = None
        print("[Server] No ML Model checkpoint found, will use mock inference")
except Exception as e:
    ml_model = None
    NORM_MEAN = None
    NORM_STD = None
    print(f"[Server] Could not load ML model: {e}")

class NavigationSession:
    """Manages the state for a single connected client."""
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dt=0.1) # Default dt, will adapt
        self.dr = DeadReckoningEngine(dt=0.1)
        self.nhc = NonHolonomicConstraints()
        self.zupt = ZUPTDetector()
        
        self.last_time = time.time()
        self.initialized = False
        self.window_buffer = []
        self.window_size = 200
        self.prev_gnss_available = False
        
        # Trajectory history for the API
        self.trajectory = []
        
        # Load real OSM data into RoadNetwork
        self.road_network = RoadNetwork()
        try:
            self.road_network.load_osm_network("data/road_network.json")
        except FileNotFoundError:
            print("[Server] Could not find data/road_network.json, falling back to synthetic grid")
            self.road_network.generate_grid_network()
        self.map_matcher = create_map_matcher(method="geometric", road_network=self.road_network)

    def process_measurement(self, data: dict) -> dict:
        """Process a single measurement packet from client."""
        current_time = data.get("timestamp", time.time())
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.1
        self.last_time = current_time

        # Update EKF and DR dt
        self.ekf.dt = dt
        self.dr.dt = dt

        accel = np.array(data.get("accel", [0, 0, 0]))
        gyro = np.array(data.get("gyro", [0, 0, 0]))
        
        # GNSS processing
        gnss = data.get("gnss")
        gnss_available = False
        if gnss and gnss.get("lat") is not None and gnss.get("lon") is not None:
            # We assume accuracy < 20m means good fix
            accuracy = gnss.get("accuracy", 100)
            if accuracy < 20.0:
                gnss_available = True
        
        # ZUPT
        is_stationary = self.zupt.update(accel, gyro, dt)

        # AI Velocity Estimation
        ai_velocity = np.array([0.0, 0.0])
        self.window_buffer.append(np.concatenate([accel, gyro]))
        if len(self.window_buffer) > self.window_size:
            self.window_buffer.pop(0)
            
        if ml_model is not None and len(self.window_buffer) == self.window_size:
            window = np.array(self.window_buffer, dtype=np.float32)
            if NORM_MEAN is not None and NORM_STD is not None:
                window = (window - NORM_MEAN) / NORM_STD
            window_tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32)
            with torch.no_grad():
                ai_velocity = ml_model(window_tensor).numpy()[0]
        else:
            # Fallback: assume zero or use previous if we have no model
            pass

        # Handle Initialization
        if gnss_available and gnss is not None and not self.initialized:
            # Simple init: assume origin is first GNSS, and speed is 0
            # A real system would use a WGS84->ENU projection relative to a home point.
            # We will use the lat/lon as a pseudo-ENU for visualization purposes, or setup a proper reference.
            self.ref_lat = gnss["lat"]
            self.ref_lon = gnss["lon"]
            self.ekf.initialize_from_gnss(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), heading=0.0)
            self.initialized = True
            self.prev_gnss_available = True

        if not self.initialized:
            return {"status": "waiting_for_gnss"}

        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(self.ref_lat))

        # State transitions based on GNSS
        if gnss_available and gnss is not None:
            if not self.prev_gnss_available and self.dr.is_active:
                self.dr.stop()
            
            dn = (gnss["lat"] - self.ref_lat) * meters_per_deg_lat
            de = (gnss["lon"] - self.ref_lon) * meters_per_deg_lon
            
            self.ekf.update_gnss(np.array([de, dn, 0.0]), timestamp=current_time)
            self.prev_gnss_available = True
        else:
            if self.prev_gnss_available:
                self.ekf.set_gnss_denied(timestamp=current_time)
                pos = self.ekf.get_position()
                v = self.ekf.get_velocity()
                heading = self.ekf.get_heading()
                speed = np.linalg.norm(v[:2])
                self.dr.start(pos[:2], heading, speed, current_time)
                self.prev_gnss_available = False
            
            self.dr.update(ai_speed=np.linalg.norm(ai_velocity), timestamp=current_time, gyro_yaw_rate=gyro[2])
        
        # Prediction
        self.ekf.predict(accel, gyro, ai_velocity)
        
        if is_stationary:
            self.ekf.update_zupt()

        # Output
        pos = self.ekf.get_position()
        vel = self.ekf.get_velocity()
        
        # Map Matching
        map_match = self.map_matcher.match(position=pos[:2], heading=self.ekf.get_heading())
        est_lat = self.ref_lat + map_match.snapped_position[1] / meters_per_deg_lat
        est_lon = self.ref_lon + map_match.snapped_position[0] / meters_per_deg_lon
        
        state_result = {
            "status": "active",
            "nav_mode": self.ekf.mode.value,
            "gnss_available": gnss_available,
            "estimated_lat": est_lat,
            "estimated_lon": est_lon,
            "raw_lat": self.ref_lat + pos[1] / meters_per_deg_lat,
            "raw_lon": self.ref_lon + pos[0] / meters_per_deg_lon,
            "speed": float(np.linalg.norm(vel[:2])),
            "heading": self.ekf.get_heading(),
            "position_error": self.ekf.get_position_uncertainty(),
            "dr_drift_percent": self.dr.get_drift_percentage() if self.dr.is_active else 0.0,
            "zupt_active": is_stationary,
            "confidence": self.dr.get_confidence() if self.dr.is_active else 1.0,
            "map_matched": map_match.confidence > 0.5
        }
        
        self.trajectory.append({
            "timestamp": current_time,
            "lat": est_lat,
            "lon": est_lon,
            "mode": self.ekf.mode.value
        })
        
        return state_result

class ConnectionManager:
    def __init__(self):
        self.mobile_ws: Optional[WebSocket] = None
        self.dashboard_viewers: List[WebSocket] = []
        self.session: Optional[NavigationSession] = None

    async def connect_mobile(self, websocket: WebSocket):
        await websocket.accept()
        if self.mobile_ws:
            try:
                await self.mobile_ws.close()
            except: pass
        self.mobile_ws = websocket
        self.session = NavigationSession()
        print("[WS] Mobile client connected. Re-initialized NavigationSession.")

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboard_viewers.append(websocket)
        print(f"[WS] Dashboard connected. Total viewers: {len(self.dashboard_viewers)}")

    def disconnect_mobile(self):
        self.mobile_ws = None
        print("[WS] Mobile client disconnected.")

    def disconnect_dashboard(self, websocket: WebSocket):
        if websocket in self.dashboard_viewers:
            self.dashboard_viewers.remove(websocket)
        print(f"[WS] Dashboard disconnected. Total viewers: {len(self.dashboard_viewers)}")

    async def broadcast_telemetry(self, raw_data: dict, estimated_state: dict):
        payload = {
            "type": "telemetry",
            "payload": {
                "raw": raw_data,
                "estimated": estimated_state
            }
        }
        dead_viewers = []
        for viewer in self.dashboard_viewers:
            try:
                await viewer.send_json(payload)
            except:
                dead_viewers.append(viewer)
        for viewer in dead_viewers:
            self.disconnect_dashboard(viewer)

    async def broadcast_log(self, severity: str, message: str):
        payload = {
            "type": "log",
            "severity": severity,
            "message": message
        }
        dead_viewers = []
        for viewer in self.dashboard_viewers:
            try:
                await viewer.send_json(payload)
            except:
                dead_viewers.append(viewer)
        for viewer in dead_viewers:
            self.disconnect_dashboard(viewer)

ws_manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, role: str = "dashboard"):
    if role == "mobile":
        await ws_manager.connect_mobile(websocket)
        try:
            while True:
                data_text = await websocket.receive_text()
                data = json.loads(data_text)
                if ws_manager.session:
                    result = ws_manager.session.process_measurement(data)
                    await websocket.send_json(result)
                    await ws_manager.broadcast_telemetry(data, result)
        except WebSocketDisconnect:
            ws_manager.disconnect_mobile()
        except Exception as e:
            print(f"[WS] Mobile Error: {e}")
            ws_manager.disconnect_mobile()
    else:
        await ws_manager.connect_dashboard(websocket)
        try:
            while True:
                await websocket.receive_text() # Keep alive
        except WebSocketDisconnect:
            ws_manager.disconnect_dashboard(websocket)

# Global state for REST API
active_sessions: Dict[str, NavigationSession] = {}

class SessionStartResponse(BaseModel):
    session_id: str

class SensorBatchRequest(BaseModel):
    session_id: str
    measurements: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Navigators IDR Backend"}

@app.get("/status")
def status_check():
    return {
        "active_sessions_count": len(active_sessions),
        "ml_model_loaded": ml_model is not None
    }

@app.post("/session/start", response_model=SessionStartResponse)
def start_session():
    import uuid
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = NavigationSession()
    return {"session_id": session_id}

@app.post("/session/stop")
def stop_session(session_id: str):
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "stopped"}
    return {"error": "session not found"}, 404

@app.post("/sensor/batch")
def process_batch(req: SensorBatchRequest):
    if req.session_id not in active_sessions:
        return {"error": "session not found"}, 404
    
    session = active_sessions[req.session_id]
    results = []
    for m in req.measurements:
        results.append(session.process_measurement(m))
    return {"processed_count": len(results), "last_state": results[-1] if results else None}

@app.get("/navigation/state")
def get_navigation_state(session_id: str):
    if session_id not in active_sessions:
        return {"error": "session not found"}, 404
    
    session = active_sessions[session_id]
    if not session.initialized:
        return {"status": "waiting_for_gnss"}
        
    pos = session.ekf.get_position()
    vel = session.ekf.get_velocity()
    heading = session.ekf.get_heading()
    
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(session.ref_lat))
    raw_lat = session.ref_lat + pos[1] / meters_per_deg_lat
    raw_lon = session.ref_lon + pos[0] / meters_per_deg_lon
    
    map_match = session.map_matcher.match(position=pos[:2], heading=heading)
    est_lat = session.ref_lat + map_match.snapped_position[1] / meters_per_deg_lat
    est_lon = session.ref_lon + map_match.snapped_position[0] / meters_per_deg_lon
    
    return {
        "status": "active",
        "nav_mode": session.ekf.mode.value,
        "raw_lat": raw_lat,
        "raw_lon": raw_lon,
        "estimated_lat": est_lat,
        "estimated_lon": est_lon,
        "speed": float(np.linalg.norm(vel[:2])),
        "heading": heading
    }

@app.get("/navigation/trajectory")
def get_trajectory(session_id: str):
    if session_id not in active_sessions:
        return {"error": "session not found"}, 404
    return {"trajectory": active_sessions[session_id].trajectory}

@app.get("/metrics")
def get_metrics(session_id: str):
    if session_id not in active_sessions:
        return {"error": "session not found"}, 404
    session = active_sessions[session_id]
    return {
        "position_error": session.ekf.get_position_uncertainty(),
        "dr_drift_percent": session.dr.get_drift_percentage() if session.dr.is_active else 0.0,
        "total_trajectory_points": len(session.trajectory)
    }

# Serve the static simulator files
# This must be mounted last so it doesn't override API routes
from fastapi.responses import FileResponse

@app.get("/dashboard")
def get_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "mac_dashboard.html")
    return FileResponse(dashboard_path)

@app.get("/admin")
def get_admin_page():
    admin_path = os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "admin.html")
    return FileResponse(admin_path)

@app.get("/privacy")
def get_privacy_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "privacy.html"))

@app.get("/terms")
def get_terms_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "terms.html"))

@app.get("/cookies")
def get_cookies_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "cookies.html"))

@app.get("/about")
def get_about_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "about.html"))

@app.get("/contact")
def get_contact_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "contact.html"))

@app.get("/404")
def get_404_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "404.html"))

from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
def custom_http_exception_handler(request, exc):
    if exc.status_code == 404:
        return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "simulator", "404.html"), status_code=404)
    return {"detail": exc.detail}


@app.get("/dataset/details")
def get_dataset_details():
    import json
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "phone_recordings")
    total_samples = 0
    trips = []
    
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if f.endswith(".json") and f.startswith("trip_"):
                try:
                    with open(os.path.join(dataset_dir, f), 'r') as file:
                        data = json.load(file)
                        samples = len(data.get("data", {}).get("timestamps", []))
                        total_samples += samples
                        trips.append({
                            "id": f.replace("trip_", "").replace(".json", ""),
                            "samples": samples,
                            "split": "Train"
                        })
                except Exception:
                    pass
                    
    for i, trip in enumerate(trips):
        if i % 5 == 0:
            trip["split"] = "Test"
        elif i % 5 == 1:
            trip["split"] = "Val"
        else:
            trip["split"] = "Train"

    return {
        "dataset_version": "v1.0",
        "total_samples": total_samples,
        "trips": trips
    }

class DatasetImportRequest(BaseModel):
    url: str

@app.post("/dataset/import")
def import_dataset(req: DatasetImportRequest):
    if not req.url or not (req.url.startswith("http://") or req.url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid URL protocol. Must start with http:// or https://")
    import urllib.request
    try:
        req_obj = urllib.request.Request(req.url, headers={'User-Agent': 'Navigators/1.0'})
        with urllib.request.urlopen(req_obj, timeout=5) as resp:
            content_type = resp.headers.get('Content-Type', '')
            status = resp.status
        return {
            "status": "success",
            "url": req.url,
            "http_status": status,
            "content_type": content_type,
            "message": f"Successfully verified external dataset source ({status})"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to dataset URL: {str(e)}")

static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "simulator")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    # Add root to pythonpath for internal imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    uvicorn.run(app, host="0.0.0.0", port=8000)
