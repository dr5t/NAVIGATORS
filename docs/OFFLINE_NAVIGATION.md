# Offline-First Navigation and Client Synchronization

```
Client Runtime: PWA Service Worker, IndexedDB, simulator/offline_engine.js
Sync Router: src/api/routers/offline_sync.py
```

## Offline-First Core Design

Navigators is designed to operate completely offline. All core navigation components (motion classification, TCN inference, EKF filtering, map matching, and UI rendering) execute client-side in JavaScript / WebAssembly.

---

## Client Storage and Offline Map Packs

- **IndexedDB**: Stores offline vector map tile packs (GeoJSON polylines, segment bounding boxes, spatial indexes).
- **LocalStorage**: Caches navigation user settings, recent searches, and saved locations.
- **Service Worker Cache**: Caches all static web assets (HTML, CSS, JS, Leaflet canvas tiles, ONNX binary models) to enable application launch with zero network connection.

---

## Opportunistic Offline Telemetry Sync

When network connectivity is restored:

1. **Contribution Sync Queue**: Pending road hazard reports, community contributions, and anonymous sensor telemetry sessions stored in IndexedDB are sent via `POST /api/v1/sync/push`.
2. **Idempotency Verification**: Device-generated UUIDs ensure pushed records are never duplicated on the server.
3. **Incremental Delta Pull**: Clients fetch updated road segment geometry and candidate map updates via `GET /api/v1/sync/pull?since_version=V`.
