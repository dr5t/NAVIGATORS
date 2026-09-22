#!/usr/bin/env python3
"""Serve the app and recording receiver over local HTTPS for an Android phone."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import os
from pathlib import Path
import secrets
import subprocess
import sys
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', required=True, type=ipaddress.ip_address, help='This computer’s Wi-Fi IPv4 address')
    args = parser.parse_args()
    if args.ip.version != 4:
        parser.error('Use the computer’s Wi-Fi IPv4 address.')
    folder = ROOT / '.phone-dev'
    public = folder / 'public'
    public.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)
    ca_key, ca_cert = folder / 'ca-key.pem', public / 'navigators-ca.crt'
    key, certificate = folder / 'server-key.pem', folder / 'server.pem'
    def openssl(*arguments):
        subprocess.run(['openssl', *map(str, arguments)], check=True, capture_output=True)
    if not ca_key.exists() or not ca_cert.exists():
        openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '365',
                '-keyout', ca_key, '-out', ca_cert, '-subj', '/CN=Navigators Local Development CA',
                '-addext', 'basicConstraints=critical,CA:TRUE', '-addext', 'keyUsage=critical,keyCertSign,cRLSign')
        ca_key.chmod(0o600)
    request = folder / 'server.csr'
    openssl('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', request, '-subj', '/CN=Navigators Local App')
    key.chmod(0o600)
    extensions = folder / 'server.ext'
    extensions.write_text(f'subjectAltName=IP:{args.ip},IP:127.0.0.1,DNS:localhost\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n')
    openssl('x509', '-req', '-in', request, '-CA', ca_cert, '-CAkey', ca_key, '-CAcreateserial',
            '-out', certificate, '-days', '30', '-extfile', extensions)
    os.environ['NAVIGATORS_SYNC_TOKEN'] = secrets.token_urlsafe(24)

    bootstrap = ThreadingHTTPServer(('0.0.0.0', 8001), partial(SimpleHTTPRequestHandler, directory=str(public)))
    Thread(target=bootstrap.serve_forever, daemon=True).start()
    print(f'Android certificate: http://{args.ip}:8001/navigators-ca.crt', flush=True)
    print(f'Phone app: https://{args.ip}:8443', flush=True)
    print(f'PC pairing code: {os.environ["NAVIGATORS_SYNC_TOKEN"]}', flush=True)
    print('Install the public CA on your test phone first. Keep this terminal open. Never share ca-key.pem.', flush=True)
    sys.path.insert(0, str(ROOT))
    import uvicorn
    try:
        uvicorn.run('src.api.server:app', host='0.0.0.0', port=8443,
                    ssl_keyfile=str(key), ssl_certfile=str(certificate))
    finally:
        bootstrap.shutdown()


if __name__ == '__main__':
    main()
