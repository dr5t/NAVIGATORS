const CACHE_NAME = 'navigators-idr-v2';
const ASSETS = [
    './',
    './index.html',
    './index.css',
    './app.js',
    './offline_engine.js',
    './manifest.json',
    './model.onnx',
    'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
    'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Return cache if found, else fetch from network
                return response || fetch(event.request);
            })
    );
});
