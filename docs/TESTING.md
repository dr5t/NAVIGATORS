# Navigators — Verification & Testing Protocol

```
Document Identifier: TEST-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Test Architecture & Coverage Framework

The Navigators test suite employs an exhaustive multi-tier verification methodology covering mathematical algorithms, state machine transitions, REST API endpoints, security matrices, database constraints, and browser execution:

```mermaid
graph TD
    TestRoot[Complete Test Harness (288 Automated Tests)] --> Pytest[pytest Backend Suite (274 Tests)]
    TestRoot --> NodeRunner[Node.js Edge Engine Suite (14 Tests)]
    
    Pytest --> UnitAlgo[Inertial Algorithms & EKF (57 Tests)]
    Pytest --> SecurityRBAC[RBAC, Auth & Security Matrix (78 Tests)]
    Pytest --> StateMachine[Lifecycle State Machines (22 Tests)]
    Pytest --> CRUD[Database CRUD & Repositories (65 Tests)]
    Pytest --> ML[Neural Models & Parity (52 Tests)]
    
    NodeRunner --> MotionTest[Vehicle Motion Detection (1 Test)]
    NodeRunner --> OfflineNav[Offline Navigation & LocalMap (13 Tests)]
```

---

## 2. Automated Test Suite Inventory (288 Tests)

### 2.1 Python Backend & Algorithm Suite (274 Tests via `pytest`)

| Test File | Test Count | Subsystem Scoped | Key Invariants Verified |
| :--- | :---: | :--- | :--- |
| `tests/test_ekf.py` | 15 | Extended Kalman Filter | Covariance propagation, Joseph symmetry, measurement update validity. |
| `tests/test_dead_reckoning.py` | 21 | Dead Reckoning & NHC | Zero-velocity suppression, lateral drift gating, velocity integration. |
| `tests/test_alignment.py` | 5 | Triad Alignment | Gravity vector leveling, forward acceleration heading resolution, PCA fallback. |
| `tests/test_nav_state.py` | 5 | Navigation State Machine | Transitions between INIT, LOCKED, DEGRADED, DR, ZUPT, REACQ. |
| `tests/test_security_matrix.py` | 8 | Security Matrix | Rejections of unauthorized access across 7 roles and direct routes. |
| `tests/test_authorization_service.py`| 10 | Central Authorization | Permission evaluation, self-moderation prevention, account status checks. |
| `tests/test_database_rbac.py` | 14 | RBAC Database Engine | Role-permission junction joins, cascading revocations, privilege isolation. |
| `tests/test_ownership_rules.py` | 2 | IDOR & Ownership | Draft privacy protection, author-exclusive edit rights. |
| `tests/test_auth_sessions.py` | 15 | Session Security | SHA-256 token hashing, expiration, revocation, guest session bounds. |
| `tests/test_places_crud.py` | 14 | Canonical Places | Spatial queries, bounding box search, soft-deletion, place versioning. |
| `tests/test_state_machine.py` | 10 | State Machine Engine | Deterministic 5-stage contribution transitions; illegal jump rejection. |
| `tests/test_moderation.py` | 8 | Moderation API | Queue filtering, approval, rejection notes, audit trail creation. |
| `tests/test_datasets.py` | 21 | Telemetry Ingestion | Consent validation, continuity verification, session split assignment. |
| `tests/test_models.py` | 17 | TCN Architecture | Causal receptive field, forward tensor dimensions, parameter count. |
| `tests/test_model_registry.py` | 7 | Model Governance | Candidate promotion, deployment atomic swapping, instant rollback. |
| `tests/test_audit_logging.py` | 6 | Audit Ledger | Action formatting, actor recording, immutable ledger retrieval. |
| `tests/test_replay.py` | 18 | Replay Harness | Zero ground-truth leakage, ablation scoring, output trajectory schemas. |
| `tests/test_search.py` | 5 | Place Search | Text normalization, category filtering, proximity distance sorting. |
| `tests/test_offline_sync.py` | 5 | Offline Queue | Idempotency key deduplication, transaction integrity. |
| `tests/test_admin_dashboard.py` | 4 | Admin Endpoints | System telemetry consolidation, user role elevation. |
| `tests/test_internal_contributors.py`| 11 | Internal Role Access | Application submission, staff review, capability upgrade. |
| `tests/test_privacy.py` | 8 | Data Privacy | Coordinate obfuscation, user profile deletion, consent compliance. |
| `tests/test_sos.py` | 5 | Emergency Distress | Incident logging, distress broadcast, resolution status update. |
| `tests/test_ui_integration.py` | 5 | UI Integration | Contract matching between backend models and frontend client state. |

### 2.2 Node.js Edge Engine Suite (14 Tests via `node --test`)

| Test File | Test Count | Key Invariants Verified |
| :--- | :---: | :--- |
| `tests/vehicle_motion_detection.test.cjs` | 1 | Complete motion transition cycle: Stationary $\rightarrow$ Moving (GNSS) $\rightarrow$ Outage $\rightarrow$ ZUPT Stop $\rightarrow$ Motion Recovery. |
| `tests/offline_navigation.test.cjs` | 13 | Coordinate frame round-trips, long road segment projections, causal IMU filtering, zero GPS origin invention, anti-jump reacquisition capping ($\le 2\text{m}$). |

---

## 3. Separation of Automated Software Tests vs. Physical Hardware Tests

> [!IMPORTANT]
> **Strict Operational Distinction**:
> - **Automated Software Validation (PASSED: 288/288)**: Validates that all mathematical equations, EKF routines, TCN models, state transitions, security barriers, and offline engines execute correctly against digital inputs and synthetic scenarios.
> - **Physical Hardware Validation (BLOCKED / PENDING ROAD TRIALS)**: Validates physical smartphone MEMS noise under real vehicle vibration, real thermal dissipation inside a hot car cabin, and real satellite attenuation in physical road tunnels. This remains unmeasured until field road testing is conducted.

---

Developed by Navigators
