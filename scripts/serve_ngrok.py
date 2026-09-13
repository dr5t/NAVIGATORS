#!/usr/bin/env python3
"""Serve the app via HTTP on port 8000, suitable for tunneling with ngrok."""
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    os.environ['NAVIGATORS_SYNC_TOKEN'] = secrets.token_urlsafe(24)
    print("=" * 60, flush=True)
    print(f'PC pairing code: {os.environ["NAVIGATORS_SYNC_TOKEN"]}', flush=True)
    print("=" * 60, flush=True)
    
    sys.path.insert(0, str(ROOT))
    import uvicorn
    uvicorn.run('src.api.server:app', host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
