# Navigators IDR Simulator - Button Documentation

This document provides a comprehensive list of all interactive buttons within the Navigators IDR Simulator interface (`index.html`), detailing their identifiers, their function, and the underlying JavaScript components they are connected to.

## 🧭 Navigation & Sidebar Buttons

### 1. Workspace Tabs
* **Identifier**: Class `.nav-link`
* **Label**: "Navigation", "Saved playback", "Experiments", "Device timings"
* **Function**: Switches between the different application views/workspaces.
* **Connection**: Connected in `workspace.js`. Clicking triggers the `setView(viewId)` function which toggles the `hidden` attribute on the corresponding `<section>` elements (e.g., `#navigationWorkspace`, `#experimentsWorkspace`).

### 2. Demo field guide
* **Identifier**: ID `#btnGuide`
* **Label**: "Demo field guide ↗"
* **Function**: Opens the instructions modal dialog.
* **Connection**: Connected in `workspace.js`. Triggers `dialog.showModal()` on the `#guideDialog` element.

### 3. Close Guide
* **Identifier**: ID `#btnCloseGuide`
* **Label**: "×" (Inside the guide dialog)
* **Function**: Closes the instructions modal dialog.
* **Connection**: Connected in `workspace.js`. Triggers `dialog.close()` on the `#guideDialog` element.

---

## 🗺️ Live Navigation & Map Panel

### 4. Start Offline Engine / Stop Engine
* **Identifier**: ID `#btnStartLive`
* **Label**: "Start Offline Engine" (changes to "Stop Engine" when running)
* **Function**: Toggles the state of the local navigation engine. When started, it requests sensor permissions (GPS, IMU), loads the AI models, begins capturing data, and updates the UI status indicators.
* **Connection**: Connected in `app.js`. Interacts heavily with `window.offlineEngine` (defined in `offline_engine.js`) and UI state logic.

### 5. Simulate GNSS Outage / Restore GPS
* **Identifier**: ID `#btnGpsOutage`
* **Label**: "Simulate GNSS outage" (changes to "Restore GPS")
* **Function**: Toggles the `engine.gpsOutage` boolean flag. When simulated outage is active, the system ignores live GPS coordinates and falls back to local Dead Reckoning (IMU/Steps + EKF) to estimate position.
* **Connection**: Connected in `app.js`. Interacts with `window.offlineEngine` and the Kalman Filter/Pedestrian algorithms.

### 6. Center local map
* **Identifier**: ID `#btnFitArea`
* **Label**: Target icon inside map panel
* **Function**: Centers the Leaflet map view either to the saved offline bounding box or to the current GPS location.
* **Connection**: Connected in `workspace.js`. Triggers `fitLocalArea()`. Interacts with the `Leaflet` map instance.

### 7. Save area offline
* **Identifier**: ID `#btnSaveArea`
* **Label**: "Save area offline"
* **Function**: Grabs the current map bounds, identifies the required map tiles, and downloads them into the browser's IndexedDB. Allows the map to render when internet connectivity is lost.
* **Connection**: Connected in `workspace.js`. Interacts with the `LocalMap` instance (defined in `local_map.js`) and IndexedDB APIs.

### 8. Follow position
* **Identifier**: ID `#btnFollow`
* **Label**: "Follow position"
* **Function**: Toggles whether the map view automatically pans to keep the user's current position marker centered.
* **Connection**: Connected in `workspace.js`. Toggles the `state.followPosition` flag which is checked during map updates in `app.js`.

---

## ▶️ Playback Controls

### 9. Play Trajectory
* **Identifier**: ID `#btnPlay`
* **Label**: Play icon
* **Function**: Starts the playback loop of a loaded trajectory JSON file.
* **Connection**: Connected in `app.js`. Triggers the `play()` function which initiates a `requestAnimationFrame` loop.

### 10. Pause Trajectory
* **Identifier**: ID `#btnPause`
* **Label**: Pause icon
* **Function**: Pauses the currently playing trajectory.
* **Connection**: Connected in `app.js`. Triggers the `pause()` function which clears the animation frame loop.

### 11. Reset Trajectory
* **Identifier**: ID `#btnReset`
* **Label**: "↺" (Reset icon)
* **Function**: Resets the trajectory playback index to `0` and updates the timeline slider to the beginning.
* **Connection**: Connected in `app.js`. Triggers the `reset()` function.

---

## 💻 PC Sync & App Settings

### 12. Pair PC
* **Identifier**: ID `#btnPairPC`
* **Label**: "Pair PC"
* **Function**: Reads the pairing code from the `#pcPairingCode` input and attempts to establish a connection with the local Python server for syncing recorded data.
* **Connection**: Connected in `workspace.js`. Interacts with the `RecordingSync` class (defined in `recording_sync.js`).

### 13. Retry Sync
* **Identifier**: ID `#btnSyncPC`
* **Label**: "Retry sync"
* **Function**: Attempts to manually flush any locally buffered sensor data to the connected PC if automatic background sync failed.
* **Connection**: Connected in `workspace.js`. Calls `window.recordingSync.flush()`.

### 14. Update App
* **Identifier**: ID `#btnUpdateApp`
* **Label**: "Update app"
* **Function**: Interacts with the browser's Service Worker to fetch the latest cached files and reloads the window to apply the update.
* **Connection**: Connected in `workspace.js`. Accesses `navigator.serviceWorker`.

### 15. Start ML Training
* **Identifier**: ID `#btnStartTraining`
* **Label**: "Start ML Training"
* **Function**: Sends a request to the paired backend to initiate model training using the synced recording data.
* **Connection**: Connected in `app.js` and `mac_dashboard.js`. Triggers the `startTraining()` function.

---

## 📊 Performance Profiling

### 16. Reset Timings
* **Identifier**: ID `#btnResetTimings`
* **Label**: "Reset timings"
* **Function**: Clears the currently recorded performance metrics/timings for device profiling.
* **Connection**: Connected in `app.js`. Interacts with `window.deviceProfiler` (defined in `device_profiler.js`).

### 17. Export Device Timings
* **Identifier**: ID `#btnExportTimings`
* **Label**: "Export device timings"
* **Function**: Serializes the device profiler stats into a JSON Blob and triggers a local file download.
* **Connection**: Connected in `app.js`. Calls `window.deviceProfiler.getStats()` and creates a temporary `<a>` element for downloading.

---

## 🛡️ Startup Privacy Modal

### 18. Accept & Continue
* **Identifier**: ID `#btnStart`
* **Label**: "Accept & Continue"
* **Function**: Confirms acceptance of terms, closes the modal, sets a consent flag in `localStorage`, and initializes the workspace.
* **Connection**: Connected in `workspace.js`. Interacts with DOM class lists and `localStorage`.

### 19. Reject All
* **Identifier**: ID `#btnSkip`
* **Label**: "Reject All"
* **Function**: Plays a shake animation and changes text to "Please accept T&C" to enforce consent.
* **Connection**: Connected in `workspace.js`. Modifies DOM styles and classes.
