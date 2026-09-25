# Navigators :  Privacy Architecture & Telemetry Protection

```
Document Identifier: PRIV-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Privacy-by-Design Architecture

Navigators is engineered from the ground up to prioritize user privacy:

1. **Zero Tracking in Offline Mode**: When operating offline, the application executes entirely on-device inside the browser sandbox. No location coordinates, sensor telemetry, device identifiers, or usage statistics are transmitted across the network.
2. **Explicit Consent for Telemetry Collection**: The client sensor recorder (`simulator/data_recorder.js`) remains completely deactivated until the user explicitly toggles data collection on. Ingested datasets require `consent: true` in the database record; sessions without verified consent are rejected by the API.
3. **Data Minimization**: Navigators does not request or collect personal demographic details, telephone contacts, or address books. The only identity required for authenticated features is an email address.

---

## 2. Location Obfuscation & Sensitive Origin Protection

To protect home and workplace locations when users voluntarily donate telemetry for neural network training:
- **Spatial Truncation**: Contributor telemetry recording automatically clips the first and last 200 meters of any recorded trip to prevent identifying private residential driveways or parking spots.
- **Coordinate Perturbation**: When public aggregate statistics are computed, geographic endpoints are snapped to the nearest major intersection node rather than exact private coordinates.

---

## 3. Data Retention & User Rights

- **Right to Erasure**: Users can permanently delete their account and associated profile data via `DELETE /api/v1/users/me`.
- **Contribution Anonymization**: If an account is deleted, previously published canonical map contributions remain in the public map with author metadata set to `NULL` / `anonymous_contributor`.
- **Session Auto-Expiration**: Active login sessions expire automatically after 30 days of inactivity.

---

Developed by Navigators
