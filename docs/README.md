# Navigators — Technical Documentation System

```
System Version: 1.0.0
Status: Complete Production Reference
Attribution: Developed by Navigators
```

Welcome to the central technical documentation repository for **Navigators**: an autonomous, edge-executable intelligent dead-reckoning navigation system designed for smartphone-based positioning during GNSS degradation or complete outage.

---

## Documentation Navigation Directory

### 1. System Specifications & Architecture
- [Product Requirements Document (PRD)](PRODUCT_REQUIREMENTS.md) — Requirement IDs, acceptance criteria, operational limits.
- [System Design Document (SDD)](SYSTEM_DESIGN.md) — Boundaries, components, dataflows, state lifecycles.
- [Architecture Document](ARCHITECTURE.md) — Edge runtime tier vs. engineering/governance tier.
- [Technical Approach](TECHNICAL_APPROACH.md) — End-to-end processing pipeline, coordinate frames, calibration.
- [Technology Stack](TECH_STACK.md) — Detailed engineering justifications for all technologies used.

### 2. Navigation, Kinematics & Mathematics
- [Navigation Engine Specification](NAVIGATION_ENGINE.md) — State machine, dead-reckoning engine, recovery logic.
- [Navigation Mathematics](NAVIGATION_MATHEMATICS.md) — LaTeX formulations for coordinate frames, 15-state EKF, NHC, ZUPT.
- [Sensor Processing & Conditioning](SENSOR_PROCESSING.md) — Causal low-pass filtering, median filter, dynamic gravity removal.
- [Map Matching & Centerline Snapping](MAP_MATCHING.md) — Vector graph representations, orthogonal projection, scoring.
- [Offline Architecture & Synchronization](OFFLINE_ARCHITECTURE.md) — Independent GNSS/Internet states, PWA cache, sync queue.

### 3. AI / Machine Learning & Empirical Data
- [AI / ML Neural Pipeline](AI_ML_PIPELINE.md) — TCN architecture, 5.4M parameters, causal receptive field, ONNX export.
- [Dataset & Data Provenance](DATASET_AND_DATA_PROVENANCE.md) — IO-VNBD benchmark, session splitting, synthetic trajectories.
- [Navigators India Dataset](NAVIGATORS_INDIA_DATASET.md) — Collection contract, validation, immutable releases, source separation, splits and evaluation.
- [Experimental Methodology](EXPERIMENTAL_METHODOLOGY.md) — Seven-stage ablation protocol (A–G), zero-leakage constraints.
- [Quantitative Evaluation & Results](EVALUATION.md) — Benchmark latencies, memory footprint, test suite breakdown.
- [Academic Research Paper](RESEARCH_AND_METHODOLOGY.md) — Formal research paper and methodological formulation.

### 4. Backend, Database, Security & Contributor Workflows
- [Backend Architecture](BACKEND_ARCHITECTURE.md) — FastAPI service structure, dependency injection, router catalog.
- [Database Design & Schema](DATABASE_DESIGN.md) — SQLite schema, 19 tables, foreign key cascades, WAL mode.
- [Authentication & RBAC](AUTHENTICATION_AND_RBAC.md) — SHA-256 token hashing, 7 roles, 40+ atomic permissions.
- [Contributor System](CONTRIBUTOR_SYSTEM.md) — Local Contributor (map POIs) and Internal Contributor (sensor telemetry).
- [Model Lifecycle & Governance](MODEL_LIFECYCLE.md) — Model registry states, approval gates, atomic rollbacks.
- [Security Architecture](SECURITY.md) — Threat modeling, IDOR elimination, privilege escalation prevention.
- [Privacy Architecture](PRIVACY.md) — Telemetry minimization, coordinate perturbation, explicit consent.

### 5. Client Applications & Operation
- [Android Integration & Mobile PWA](ANDROID.md) — Chrome on Android, HTTPS development certificates, sensor tuning.
- [Web Application Specification](WEB_APPLICATION.md) — PWA design tenets, Leaflet canvas, telemetry HUD.
- [Engineering Dashboard](DASHBOARD.md) — Desktop telemetry console, model promotion, moderation queue.
- [REST API Reference](API_REFERENCE.md) — Complete endpoint schemas, request/response models, status codes.

### 6. Operations, Testing & Governance
- [Verification & Testing Protocol](TESTING.md) — Automated test suite inventory (288 tests), hardware status.
- [Deployment Guide](DEPLOYMENT.md) — Local installation, HTTPS certificates, static asset packaging.
- [Reproducibility Guide](REPRODUCIBILITY.md) — Exact commands to replicate training, export, and evaluation.
- [System Limitations](LIMITATIONS.md) — Transparent disclosure of physical, sensor, and operational bounds.
- [Future Work & Roadmap](FUTURE_WORK.md) — In-vehicle trials, barometric altitude, visual-inertial odometry.
- [Bibliographic References](REFERENCES.md) — Formal academic citations for datasets and navigation literature.
- [Phase 38 E2E Test Report](reports/PHASE_38_E2E_TEST_REPORT.md) — Official verification audit (288/288 passed, hardware blocked).

---

Developed by Navigators
