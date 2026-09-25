# Navigators :  Production & Local Deployment Guide

```
Document Identifier: DEP-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Prerequisites & Environment Setup

### 1.1 System Requirements
- **Operating System**: macOS (Darwin 24+), Linux (Ubuntu 22.04+ LTS), or Windows with WSL2.
- **Python**: Version 3.11 or higher (developed and verified on Python 3.14).
- **Node.js**: Version 18.0+ LTS (for running client test suites).
- **Git & Git LFS**: Required for repository management and large model weight handling.

### 1.2 Python Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/Navigators/Navigators.git
cd Navigators

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Database Initialization

The relational SQLite database initializes automatically on first server startup, or can be seeded explicitly:

```bash
# Initialize schema and default roles/permissions
python3 -c "from src.db.database import init_db; init_db('data/navigators.db')"
```

---

## 3. Launching the Backend Server

### 3.1 Standard Development Server
```bash
# Run FastAPI with live reloading
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

### 3.2 Secure Mobile Development Server (HTTPS for Phone Testing)
Mobile browsers require HTTPS to grant access to accelerometer and gyroscope APIs:
```bash
# Replace with your workstation's local Wi-Fi IPv4 address
python scripts/serve_phone.py --ip 192.168.1.50
```
This command:
1. Generates local development Root CA certificates in `.phone-dev/`.
2. Starts an HTTP bootstrap server on port `8001` to serve the public CA certificate to the phone.
3. Launches the FastAPI backend over TLS on port `8443` (`https://192.168.1.50:8443`).

---

## 4. Client PWA Offline Deployment

The client navigation interface is fully static and requires no dynamic server-side rendering:
- Serve the `simulator/` directory using any standard static file server (Nginx, Caddy, Cloudflare Pages, or FastAPI static mounts).
- Ensure the following MIME types are correctly configured:
  - `.wasm` $\rightarrow$ `application/wasm`
  - `.json` $\rightarrow$ `application/json`
  - `.onnx` $\rightarrow$ `application/octet-stream`

---

## 5. Model Rollback & Troubleshooting

### Emergency Model Rollback
If a newly promoted model displays abnormal trajectory drift in field evaluation:
```bash
# Authenticated rollback request
curl -X POST https://127.0.0.1:8000/api/v1/models/rollback \
  -H "Authorization: Bearer <team_admin_token>"
```
This immediately reactivates the previous production checkpoint in the database.

### Common Troubleshooting Scenarios
1. **Sensors Not Streaming on Android**: Ensure the phone is connected via HTTPS (port 8443), the custom CA certificate is installed, and motion permissions are enabled in Chrome site settings.
2. **WebAssembly SIMD Error**: Verify that the browser version supports WebAssembly SIMD (Chrome 91+, Safari 16.4+, Firefox 89+).

---

Developed by Navigators
