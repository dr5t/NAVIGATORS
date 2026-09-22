# Navigators — Android Mobile Integration & PWA Deployment

```
Document Identifier: AND-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Mobile Execution Model: Zero Native Overhead

Rather than requiring an invasive native Android APK installation with specialized app store distribution, Navigators operates as an offline **Progressive Web App (PWA)**:
- **Runtime Environment**: Executes within modern mobile browsers (Google Chrome on Android v100+).
- **Zero App Store Gatekeepers**: Drivers access navigation directly via HTTPS URL and tap "Add to Home Screen" to install the app with full-screen standalone display.
- **Hardware Sensor Access**: Uses standardized HTML5 `devicemotion` and `geolocation` APIs without requiring native NDK compilation or binary code signing.

---

## 2. Local Development & Mobile Device Pairing

Modern mobile browsers strictly enforce that Motion and Sensor APIs are accessible **only over secure HTTPS contexts** (or `localhost`). To test physical Android smartphones over a local Wi-Fi development network, Navigators provides an automated HTTPS provisioning script (`scripts/serve_phone.py`):

```
Developer Workstation (Mac / Linux)             Android Smartphone
  │                                                      │
  ├─ Run `python scripts/serve_phone.py --ip <IP>`      │
  │  (Generates Root CA & Server TLS Certificates)       │
  │  (Serves CA Cert on HTTP :8001)                      │
  │  (Serves App on HTTPS :8443)                         │
  │                                                      │
  │◄─ Download CA Cert: http://<IP>:8001/navigators-ca.crt ─┤
  │   (User installs certificate in Android Settings)    │
  │                                                      │
  │◄─ Connect PWA: https://<IP>:8443 ────────────────────┤
  │   (Chrome trusts certificate with zero warnings)     │
  │                                                      │
  ├─ Enter Pairing Code (`NAVIGATORS_SYNC_TOKEN`) ───────┤
  │                                                      │
  ▼                                                      ▼
[Continuous Real-Time Telemetry & Inertial Navigation Active]
```

### Step-by-Step Device Setup:
1. Connect the computer and the Android phone to the same local Wi-Fi network.
2. Launch the server from repository root:
   ```bash
   python scripts/serve_phone.py --ip 192.168.1.50
   ```
3. On the phone, browse to `http://192.168.1.50:8001/navigators-ca.crt` and install the certificate under **Android Settings $\rightarrow$ Security $\rightarrow$ Encryption & Credentials $\rightarrow$ Install CA certificate**.
4. Open Chrome on Android and navigate to `https://192.168.1.50:8443`.
5. Tap the browser menu $\rightarrow$ **Install App / Add to Home Screen**.

---

## 3. Sensor Permissions & Event Tuning

1. **Permission Grant**: Chrome on Android automatically prompts the user to grant motion and location permissions upon launching navigation.
2. **Frequency Lock**: While Android hardware sensors can dispatch up to $100\text{ Hz}$, the browser event loop buffers readings into a stable $10\text{ Hz}$ estimation window ($\Delta t = 0.1\text{ s}$).
3. **Screen Wake Lock**: The client invokes `navigator.wakeLock.request('screen')` to prevent the device from entering low-power display sleep while actively guiding the vehicle.

---

## 4. Current Physical Device Validation Status

> [!IMPORTANT]
> **Status Disclosure**:
> - PWA asset delivery, offline service worker caching, and sensor permission flows have been validated on Android Chrome in development.
> - **Physical Moving Vehicle Road Testing under genuine GNSS loss remains UNVALIDATED**.
> - Test suite coverage (288/288 tests) validates the software algorithms against recorded and synthetic traces; in-vehicle road testing with physical Android hardware is ongoing future work.

---

Developed by Navigators
