# Navigators — Web Application & Interface Specification

```
Document Identifier: WEB-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Application Philosophy & Visual Design Standards

The Navigators web application (`simulator/index.html`) is built as a serious, professional, high-contrast navigation instrument. In accordance with the repository's `DESIGN_PHILOSOPHY.md` and `PRODUCT_RULES.md`:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PROHIBITED ANTI-PATTERNS                                               │
│ ❌ No glowing neon/cyberpunk gradients                                  │
│ ❌ No emoji icons                                                      │
│ ❌ No excessive pill buttons                                           │
│ ❌ No generic AI SaaS landing page tropes                              │
│ ❌ No fake counters, fake reviews, or fake accuracy claims             │
├────────────────────────────────────────────────────────────────────────┤
│ MANDATORY DESIGN STANDARDS                                             │
│ ✅ Clean obsidian dark theme with high contrast ratios                 │
│ ✅ Map-first viewport: Leaflet canvas is the primary surface           │
│ ✅ High information density: Crisp tabular numbers & state indicators  │
│ ✅ Mobile-first touch targets (minimum 44x44px)                        │
│ ✅ Attribution: "Developed by Navigators" in footer                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Interface Component Structure

```mermaid
graph TD
    Root[index.html] --> TopNav[Top Telemetry Bar & Status Badges]
    Root --> MapContainer[Leaflet Vector Map Canvas]
    Root --> BottomHUD[Bottom Navigation & Control HUD]
    Root --> Drawer[Collapsible Workspace Drawer]
    
    TopNav --> GNSSBadge[GNSS Mode: LOCKED / DEGRADED / OUTAGE]
    TopNav --> EKFBadge[Filter Mode: GNSS-INS / DR / ZUPT]
    TopNav --> SpeedGauge[Current Speed (km/h)]
    
    BottomHUD --> StartBtn[Start Navigation]
    BottomHUD --> RecordBtn[Record Trip Data]
    BottomHUD --> ReplayBtn[Saved Playback]
    
    Drawer --> ContribForm[Submit Map Contribution]
    Drawer --> OfflineSyncList[Queued Offline Edits]
    Drawer --> IncidentReporter[Report Hazard / SOS]
```

### 2.1 Top Telemetry HUD
- **GNSS Status Indicator**: Dynamically reflects satellite fix quality (`LOCKED` - green, `DEGRADED` - amber, `OUTAGE` - red).
- **EKF Mode Indicator**: Displays active estimation mode (`GNSS_INS`, `DR`, `ZUPT`, `REACQ`).
- **Speed Gauge**: Real-time vehicle speed computed from fused state velocity vector ($\sqrt{v_E^2 + v_N^2} \times 3.6\text{ km/h}$).

### 2.2 Leaflet Vector Map Viewport
- **Tile Layer**: Cached raster or vector tiles rendered locally without external internet requests.
- **Trajectory Polylines**: Renders the active estimated vehicle path in solid blue, with historic GNSS reference in dotted white during replay modes.
- **Centerline Snapping**: Shows the snapped road point with orientation arrow matching the vehicle azimuth $\psi$.

### 2.3 Collapsible Workspace Drawer (`workspace.js`)
Enables local contributors to add new POIs, verify offline synchronization queues, and trigger distress alerts without leaving the active navigation map.

---

## 3. PWA Configuration & Service Worker Lifecycle

- **Web Manifest (`simulator/manifest.json`)**: Configured with `display: "standalone"`, `orientation: "any"`, high-resolution SVG icons, and theme color `#0f172a`.
- **Service Worker (`simulator/sw.js`)**: Automatically registers on first visit, downloads and caches the complete application bundle (including `model.onnx` and `ort-wasm-simd-threaded.wasm`), enabling instant offline launches.

---

Developed by Navigators
