"""Download pinned browser dependencies once, before disconnecting the demo device."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "simulator" / "vendor"
PACKAGES = {
    "leaflet": ("leaflet@1.9.4", {
        "leaflet.js": "dist/leaflet.js", "leaflet.css": "dist/leaflet.css",
        "images/layers.png": "dist/images/layers.png",
        "images/layers-2x.png": "dist/images/layers-2x.png",
        "images/marker-icon.png": "dist/images/marker-icon.png",
        "images/marker-icon-2x.png": "dist/images/marker-icon-2x.png",
        "images/marker-shadow.png": "dist/images/marker-shadow.png",
        "LICENSE": "LICENSE",
    }),
    "onnxruntime": ("onnxruntime-web@1.22.0", {
        "ort.wasm.min.js": "dist/ort.wasm.min.js",
        "ort-wasm-simd-threaded.mjs": "dist/ort-wasm-simd-threaded.mjs",
        "ort-wasm-simd-threaded.wasm": "dist/ort-wasm-simd-threaded.wasm",
        "LICENSE": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE",
    }),
}


def main():
    manifest = {}
    for directory, (package, files) in PACKAGES.items():
        for filename, source in files.items():
            url = source if source.startswith("https://") else f"https://cdn.jsdelivr.net/npm/{package}/{source}"
            target = ROOT / directory / filename
            print(f"Downloading {package}/{source}", flush=True)
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[f"{directory}/{filename}"] = {
                "url": url, "sha256": hashlib.sha256(content).hexdigest(),
            }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Offline browser dependencies saved to simulator/vendor.")


if __name__ == "__main__":
    main()
