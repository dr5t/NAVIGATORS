// Bump the version whenever a bundled asset, model, or downloaded map changes.
const CACHE_NAME = 'navigators-idr-offline-v4';
const ASSETS = [
    './', './index.html', './index.css', './app.js', './local_map.js',
    './offline_engine.js', './device_profiler.js', './engine/preprocessing.js', './model.contract.json', './data_recorder.js', './manifest.json', './icon.svg',
    './engine/matrix.js', './engine/ekf.js', './engine/map_matcher.js', './engine/alignment.js',
    './model.onnx', './model.onnx.data', './data/simulation.json', './data/road_network.json',
    './vendor/leaflet/leaflet.css', './vendor/leaflet/leaflet.js',
    './vendor/leaflet/images/layers.png', './vendor/leaflet/images/layers-2x.png',
    './vendor/leaflet/images/marker-icon.png', './vendor/leaflet/images/marker-icon-2x.png',
    './vendor/leaflet/images/marker-shadow.png',
    './vendor/onnxruntime/ort.wasm.min.js',
    './vendor/onnxruntime/ort-wasm-simd-threaded.mjs',
    './vendor/onnxruntime/ort-wasm-simd-threaded.wasm',
];

self.addEventListener('install', event => {
    // A failed download must never mark a partial package ready.
    event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(
        ASSETS.map(url => new Request(url, { cache: 'reload' }))
    )).then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
    event.waitUntil((async () => {
        for (const key of await caches.keys()) {
            if (key.startsWith('navigators-idr-') && key !== CACHE_NAME) await caches.delete(key);
        }
        await self.clients.claim();
    })());
});

self.addEventListener('message', event => {
    if (event.data?.type !== 'CHECK_OFFLINE') return;
    event.waitUntil((async () => {
        const cache = await caches.open(CACHE_NAME);
        const responses = await Promise.all(ASSETS.map(url => cache.match(url)));
        event.source?.postMessage({ type: 'OFFLINE_STATUS', ready: responses.every(Boolean) });
    })());
});

self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    if (url.origin !== self.location.origin) {
        event.respondWith(Promise.resolve(new Response('External requests are disabled', { status: 403 })));
        return;
    }
    if (event.request.method !== 'GET') return;
    event.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(event.request);
        if (cached) return cached;
        try {
            return await fetch(event.request);
        } catch {
            return new Response('Local asset unavailable. Reconnect and prepare offline files.', { status: 503 });
        }
    })());
});
