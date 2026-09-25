# Navigators: Technical Documentation System

```
System Version: 1.0.0
Status: Complete Production Reference
Attribution: Developed by Navigators Team
```

Welcome to the central technical documentation repository for **Navigators**: an autonomous, edge-executable intelligent dead-reckoning navigation system designed for smartphone-based multi-modal positioning during GNSS degradation or complete outages.

---

## Documentation Navigation Directory

### 1. Multi-Modal Navigation Phases (Phases 1-12)
- [Phase 1: Navigation Data Audit](NAVIGATION_AUDIT.md) - Baseline sensor, coordinate, and EKF audit.
- [Phase 2: Dataset Registry & Provenance](DATASET_REGISTRY_PROVENANCE.md) - Registry architecture, third-party dataset quarantine, and provenance rules.
- [Phase 3: Cross-Dataset Sensor Normalization](CROSS_DATASET_SENSOR_NORMALIZATION.md) - Zero-leakage unit conversion, coordinate alignment, and filtering.
- [Phase 4: Vehicle Dead-Reckoning Training](VEHICLE_DEAD_RECKONING_TRAINING.md) - Vehicle TCN velocity estimation and split validation.
- [Phase 5: Indian Vehicle Domain Adaptation](INDIAN_VEHICLE_DOMAIN_ADAPTATION.md) - Indian domain generalization and fine-tuning results.
- [Phase 6: Pedestrian Dead-Reckoning Model](PEDESTRIAN_DEAD_RECKONING_MODEL.md) - Pedestrian motion model and target formulations.
- [Phase 7: Navigators India Pedestrian Dataset](NAVIGATORS_INDIA_PEDESTRIAN_DATASET.md) - Ground truth verification and multi-placement walking dataset.
- [Phase 8: Motion Classification Layer](MOTION_CLASSIFICATION_LAYER.md) - Real-time IMU classifier, hysteresis buffer, and motion router.
- [Phase 9: Multi-Modal Adaptive Dead-Reckoning Fusion](MULTI_MODAL_ADAPTIVE_FUSION.md) - Adaptive EKF, map constraints, and network-independent operation.
- [Phase 10: Confidence-Aware GNSS & Dead Reckoning](CONFIDENCE_AWARE_GNSS_AND_DEAD_RECKONING.md) - GNSS trust states, 6-axis anomaly detection, and anti-jump recovery.
- [Phase 11: Accuracy Optimization and Ablation](ACCURACY_OPTIMIZATION_AND_ABLATION.md) - 17-dimension systematic ablation and Pareto trade-off analysis.
- [Phase 12: Final Vehicle & Pedestrian Benchmark Report](FINAL_BENCHMARK_REPORT.md) - Reproducible 5-baseline evaluation across 5 outage durations (10s to 300s).

---

### 2. System Specifications & Architecture
- [Product Requirements Document (PRD)](PRODUCT_REQUIREMENTS.md) - Requirement IDs, acceptance criteria, operational limits.
- [System Design Document (SDD)](SYSTEM_DESIGN.md) - Boundaries, components, dataflows, state lifecycles.
- [Architecture Document](ARCHITECTURE.md) - Edge runtime tier vs engineering/governance tier.
- [Technical Approach](TECHNICAL_APPROACH.md) - End-to-end processing pipeline, coordinate frames, calibration.
- [Technology Stack](TECH_STACK.md) - Detailed engineering justifications for all technologies used.

---

### 3. Navigation, Kinematics & Mathematics
- [Navigation Engine Specification](NAVIGATION_ENGINE.md) - State machine, dead-reckoning engine, recovery logic.
- [Navigation Mathematics](NAVIGATION_MATHEMATICS.md) - Formulations for coordinate frames, 15-state EKF, NHC, ZUPT.
- [Sensor Processing & Conditioning](SENSOR_PROCESSING.md) - Causal low-pass filtering, median filter, dynamic gravity removal.
- [Map Matching & Centerline Snapping](MAP_MATCHING.md) - Vector graph representations, orthogonal projection, scoring.
- [Adaptive Sensor Fusion Engine](ADAPTIVE_SENSOR_FUSION.md) - Adaptive measurement covariance scaling and innovation gating.
- [GNSS Anomaly Detection](GNSS_ANOMALY_DETECTION.md) - Position jump, speed, heading, AI, and map disagreement detectors.
- [GNSS Trust Engine](GNSS_TRUST_ENGINE.md) - Signal integrity state machine (TRUSTED, DEGRADED, SUSPICIOUS, UNUSABLE).
- [GNSS Degradation Prediction](GNSS_DEGRADATION_PREDICTION.md) - Early warning system for approaching tunnels and urban canyons.
- [Validated GNSS Reacquisition](VALIDATED_GNSS_REACQUISITION.md) - Anti-jump N=3 fix verification during satellite return.
- [Map Constrained Dead Reckoning](MAP_CONSTRAINED_DEAD_RECKONING.md) - Segment geometry projection and cross-track error bounds.
- [Road Hypothesis Engine](ROAD_HYPOTHESIS_ENGINE.md) - Multi-road hypothesis tracking during ambiguity.
- [Offline Architecture & Synchronization](OFFLINE_ARCHITECTURE.md) - Independent GNSS/Internet states, PWA cache, sync queue.

---

### 4. AI / Machine Learning & Empirical Data
- [AI / ML Neural Pipeline](AI_ML_PIPELINE.md) - TCN architecture, causal receptive field, ONNX export.
- [AI Velocity Integration](AI_VELOCITY_INTEGRATION.md) - Coupling deep velocity inference with EKF measurement updates.
- [Navigators India Dataset](NAVIGATORS_INDIA_DATASET.md) - Collection contract, validation, immutable releases, source separation.
- [India Dataset Pipeline](INDIA_DATASET_PIPELINE.md) - Ingestion, quality report, ground truth verification.
- [Experimental Methodology](EXPERIMENTAL_METHODOLOGY.md) - Ablation protocol, zero-leakage constraints.
- [Quantitative Evaluation & Results](EVALUATION.md) - Benchmark latencies, memory footprint, test suite breakdown.
- [Academic Research Paper](RESEARCH_AND_METHODOLOGY.md) - Formal research paper and methodological formulation.

---

### 5. Backend, Database, Security & Contributor Workflows
- [Backend Architecture](BACKEND_ARCHITECTURE.md) - FastAPI service structure, dependency injection, router catalog.
- [Database Design & Schema](DATABASE_DESIGN.md) - SQLite schema, 19 tables, foreign key cascades, WAL mode.
- [Authentication & RBAC](AUTHENTICATION_AND_RBAC.md) - SHA-256 token hashing, 7 roles, 40+ atomic permissions.
- [Contributor System](CONTRIBUTOR_SYSTEM.md) - Local Contributor (map POIs) and Internal Contributor (sensor telemetry).
- [Model Lifecycle & Governance](MODEL_LIFECYCLE.md) - Model registry states, approval gates, atomic rollbacks.
- [Security Architecture](SECURITY.md) - Threat modeling, IDOR elimination, privilege escalation prevention.
- [Privacy Architecture](PRIVACY.md) - Telemetry minimization, coordinate perturbation, explicit consent.

---

### 6. Client Applications & Operations
- [Android Integration & Mobile PWA](ANDROID.md) - Chrome on Android, HTTPS development certificates, sensor tuning.
- [Web Application Specification](WEB_APPLICATION.md) - PWA design tenets, Leaflet canvas, telemetry HUD.
- [Engineering Dashboard](DASHBOARD.md) - Desktop telemetry console, model promotion, moderation queue.
- [REST API Reference](API_REFERENCE.md) - Complete endpoint schemas, request/response models, status codes.

---

### 7. Governance, Operations & Reproducibility
- [Verification & Testing Protocol](TESTING.md) - Automated test suite inventory (464 unit/integration tests).
- [Deployment Guide](DEPLOYMENT.md) - Local installation, HTTPS certificates, static asset packaging.
- [Reproducibility Guide](REPRODUCIBILITY.md) - Commands to replicate training, export, and benchmark suite.
- [System Limitations](LIMITATIONS.md) - Transparent disclosure of physical, sensor, and operational bounds.
- [Future Work & Roadmap](FUTURE_WORK.md) - In-vehicle trials, barometric altitude, visual-inertial odometry.
- [Bibliographic References](REFERENCES.md) - Formal academic citations for datasets and navigation literature.

---

Developed by Navigators Team
