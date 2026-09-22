-- Navigators IDR - Relational Database Schema & RBAC Data Model
-- Enforces relational integrity, foreign key constraints, and dynamic permission evaluation.

PRAGMA foreign_keys = ON;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    avatar TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'pending_verification', 'deleted')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_login_at TEXT
);

-- 2. Roles Table
CREATE TABLE IF NOT EXISTS roles (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL
);

-- 3. Permissions Table
CREATE TABLE IF NOT EXISTS permissions (
    id TEXT PRIMARY KEY,
    resource TEXT NOT NULL,
    action TEXT NOT NULL,
    description TEXT,
    UNIQUE (resource, action)
);

-- 4. Role Permissions Mapping Table
CREATE TABLE IF NOT EXISTS role_permissions (
    role_id TEXT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id TEXT NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

-- 5. User Roles Mapping Table
CREATE TABLE IF NOT EXISTS user_roles (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id TEXT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (user_id, role_id)
);

-- 6. Authentication Identities (Extensible multi-provider credentials)
CREATE TABLE IF NOT EXISTS auth_identities (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL, -- 'local_password', 'google', 'apple', 'github'
    identifier TEXT NOT NULL, -- email or external subject id
    credential_hash TEXT, -- hashed password / secret
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (provider, identifier)
);

-- 7. Sessions Table (Secure server-managed sessions)
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    token_hash TEXT UNIQUE NOT NULL,
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE, -- NULL for anonymous guest sessions
    is_guest INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    user_agent TEXT,
    ip_address TEXT
);

-- 8. Canonical Places Table (Live map data with versioning and soft-delete)
CREATE TABLE IF NOT EXISTS places (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL, -- 'fuel', 'hospital', 'ev_charging', 'atm', 'pharmacy', 'restaurant', etc.
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    address TEXT,
    opening_hours TEXT,
    phone TEXT,
    website TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'published' CHECK (status IN ('published', 'archived', 'soft_deleted')),
    version INTEGER NOT NULL DEFAULT 1,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    created_by TEXT REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 9. Place Version History Table (Immutable audit trail of place mutations)
CREATE TABLE IF NOT EXISTS place_history (
    id TEXT PRIMARY KEY,
    place_id TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    action TEXT NOT NULL, -- 'created', 'updated', 'archived', 'soft_deleted', 'restored'
    changed_by TEXT REFERENCES users(id),
    snapshot_json TEXT NOT NULL,
    change_summary TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 10. Community Contributions Table (Ownership & moderation lifecycle)
CREATE TABLE IF NOT EXISTS contributions (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL, -- 'place', 'road_hazard', 'amenity'
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'submitted', 'pending_review', 'pending', 'changes_requested', 'approved', 'published', 'rejected', 'withdrawn')),
    title TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    target_resource_id TEXT REFERENCES places(id) ON DELETE SET NULL,
    action TEXT NOT NULL DEFAULT 'create' CHECK (action IN ('create', 'update', 'delete')),
    reviewed_by TEXT REFERENCES users(id),
    review_notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    reviewed_at TEXT,
    published_at TEXT
);

-- 11. Platform Audit Logs Table (Full mutation and state transition traceability)
CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    actor_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    old_state TEXT,
    new_state TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    ip_hash TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 12. Community Reports & Flagging Table (User issue reporting on places and contributions)
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    reporter_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL CHECK (target_type IN ('place', 'contribution', 'user')),
    target_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    details TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'resolved', 'dismissed')),
    resolved_by TEXT REFERENCES users(id),
    resolution_notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT
);

-- ========================================================
-- Canonical Map Pipeline & Offline Synchronization Tables
-- ========================================================

-- Canonical Changelog: Monotonically increasing sequence stream for incremental delta synchronization
CREATE TABLE IF NOT EXISTS canonical_changelog (
    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_type TEXT NOT NULL DEFAULT 'place',
    resource_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('create', 'update', 'delete', 'restore')),
    version INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Offline Map Packages: Pre-compiled standalone map packages for offline sync
CREATE TABLE IF NOT EXISTS offline_map_packages (
    id TEXT PRIMARY KEY,
    package_version INTEGER NOT NULL,
    region TEXT NOT NULL DEFAULT 'global',
    format TEXT NOT NULL DEFAULT 'sqlite' CHECK (format IN ('sqlite', 'json_bundle')),
    file_path TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Device Sync Queue: Tracking offline sync queues submitted by devices
CREATE TABLE IF NOT EXISTS device_sync_queue (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    client_sequence INTEGER NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('add_place', 'suggest_edit', 'report')),
    payload_json TEXT NOT NULL,
    base_version INTEGER,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'synced', 'conflict', 'rejected')),
    conflict_reason TEXT,
    server_resource_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    synced_at TEXT
);

-- ========================================================
-- Phase 12: Internal Contributor Request & Approval Tables
-- ========================================================

CREATE TABLE IF NOT EXISTS internal_contributor_requests (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    experience TEXT NOT NULL,
    requested_scope TEXT NOT NULL DEFAULT 'trajectories_and_models',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    rejection_reason TEXT,
    reviewed_by TEXT REFERENCES users(id),
    reviewed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ========================================================
-- Phase 13: Dataset Sessions (Internal Contributor Uploads)
-- ========================================================

CREATE TABLE IF NOT EXISTS dataset_sessions (
    id TEXT PRIMARY KEY,
    contributor_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    activity_type TEXT NOT NULL CHECK (activity_type IN (
        'walking', 'driving', 'gnss_imu', 'gnss_outage', 'validation'
    )),
    device TEXT NOT NULL,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    sensor_data_path TEXT,
    gnss_available INTEGER NOT NULL DEFAULT 1 CHECK (gnss_available IN (0, 1)),
    consent INTEGER NOT NULL DEFAULT 1 CHECK (consent IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'uploaded'
        CHECK (status IN ('uploaded', 'validating', 'validated', 'rejected')),
    rejection_reason TEXT,
    validated_by TEXT REFERENCES users(id),
    validated_at TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_icr_user_id ON internal_contributor_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_icr_status ON internal_contributor_requests(status);
CREATE INDEX IF NOT EXISTS idx_dataset_sessions_contributor ON dataset_sessions(contributor_id);
CREATE INDEX IF NOT EXISTS idx_dataset_sessions_status ON dataset_sessions(status);
CREATE INDEX IF NOT EXISTS idx_dataset_sessions_activity ON dataset_sessions(activity_type);
CREATE INDEX IF NOT EXISTS idx_dataset_sessions_created ON dataset_sessions(created_at);

-- ========================================================
-- Phase 15: Model Registry (Approval Workflow)
-- ========================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    architecture TEXT NOT NULL DEFAULT 'TCNVelocityEstimator',
    status TEXT NOT NULL DEFAULT 'candidate_training'
        CHECK (status IN (
            'candidate_training', 'evaluating', 'review',
            'approved', 'rejected', 'production_candidate', 'production'
        )),
    -- Evaluation metrics
    test_mae REAL,
    test_rmse REAL,
    val_loss REAL,
    onnx_parity_max_diff REAL,
    error_reduction_pct REAL,
    epochs INTEGER,
    batch_size INTEGER,
    window_size INTEGER,
    total_training_time_s REAL,
    -- File artifact paths
    checkpoint_path TEXT,
    onnx_path TEXT,
    norm_stats_path TEXT,
    -- Governance
    registered_by TEXT REFERENCES users(id),
    reviewed_by TEXT REFERENCES users(id),
    deployed_by TEXT REFERENCES users(id),
    rejection_reason TEXT,
    review_notes TEXT,
    -- Timestamps
    evaluated_at TEXT,
    reviewed_at TEXT,
    deployed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_model_registry_status  ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_created ON model_registry(created_at);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_permissions_resource ON permissions(resource);
CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_perm ON role_permissions(permission_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id);
CREATE INDEX IF NOT EXISTS idx_auth_identities_user ON auth_identities(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_identities_lookup ON auth_identities(provider, identifier);
CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(revoked_at, expires_at);
CREATE INDEX IF NOT EXISTS idx_places_category ON places(category);
CREATE INDEX IF NOT EXISTS idx_places_coords ON places(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_places_status_deleted ON places(status, is_deleted);
CREATE INDEX IF NOT EXISTS idx_place_history_place ON place_history(place_id);
CREATE INDEX IF NOT EXISTS idx_contributions_owner ON contributions(owner_id);
CREATE INDEX IF NOT EXISTS idx_contributions_status ON contributions(status);
CREATE INDEX IF NOT EXISTS idx_contributions_target ON contributions(target_resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_reporter ON reports(reporter_id);
CREATE INDEX IF NOT EXISTS idx_reports_target ON reports(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_canonical_changelog_seq ON canonical_changelog(sequence_id);
CREATE INDEX IF NOT EXISTS idx_canonical_changelog_res ON canonical_changelog(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_offline_map_packages_ver ON offline_map_packages(package_version);
CREATE INDEX IF NOT EXISTS idx_offline_map_packages_region ON offline_map_packages(region);
CREATE INDEX IF NOT EXISTS idx_device_sync_queue_dev ON device_sync_queue(device_id);
CREATE INDEX IF NOT EXISTS idx_device_sync_queue_status ON device_sync_queue(status);

-- Trigger: auto-update updated_at on user modification
CREATE TRIGGER IF NOT EXISTS trg_users_updated_at
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
END;

-- Trigger: auto-update updated_at on place modification
CREATE TRIGGER IF NOT EXISTS trg_places_updated_at
AFTER UPDATE ON places
FOR EACH ROW
BEGIN
    UPDATE places SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
END;

-- Trigger: auto-update updated_at on contribution modification
CREATE TRIGGER IF NOT EXISTS trg_contributions_updated_at
AFTER UPDATE ON contributions
FOR EACH ROW
BEGIN
    UPDATE contributions SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
END;

-- Trigger: auto-update updated_at on internal_contributor_requests modification
CREATE TRIGGER IF NOT EXISTS trg_icr_updated_at
AFTER UPDATE ON internal_contributor_requests
FOR EACH ROW
BEGIN
    UPDATE internal_contributor_requests SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
END;

-- Trigger: auto-update updated_at on dataset_sessions modification
CREATE TRIGGER IF NOT EXISTS trg_dataset_sessions_updated_at
AFTER UPDATE ON dataset_sessions
FOR EACH ROW
BEGIN
    UPDATE dataset_sessions SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
END;

-- Trigger: auto-update updated_at on model_registry modification
CREATE TRIGGER IF NOT EXISTS trg_model_registry_updated_at
AFTER UPDATE ON model_registry
FOR EACH ROW
BEGIN
    UPDATE model_registry SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
END;


-- ========================================================
-- Seed Initial Roles
-- ========================================================
INSERT OR IGNORE INTO roles (id, name, description) VALUES
    ('guest', 'Guest', 'Unauthenticated visitor with read-only navigation and map access'),
    ('user', 'Registered User', 'Standard member with saved places and personal contribution drafting'),
    ('local_contributor', 'Local Contributor', 'Verified local community editor with place and road contribution permissions'),
    ('internal_contributor', 'Internal Contributor', 'Technical field contributor with trajectory recording and ML evaluation access'),
    ('moderator', 'Community Moderator', 'Reviewer authorized to approve or reject community submissions and manage place data'),
    ('team_admin', 'Engineering Team Admin', 'Full engineering lifecycle administrator with dataset, model deployment, and role privileges'),
    ('super_admin', 'Super Administrator', 'Unrestricted administrative governance and security control');

-- ========================================================
-- Seed Permissions Catalog
-- ========================================================
INSERT OR IGNORE INTO permissions (id, resource, action, description) VALUES
    ('place:create', 'place', 'create', 'Create and submit community place contributions'),
    ('place:read', 'place', 'read', 'Read published canonical places and public map metadata'),
    ('place:update', 'place', 'update', 'Suggest or execute modifications to existing places'),
    ('place:delete', 'place', 'delete', 'Archive, soft delete, or remove place listings'),

    ('contribution:create', 'contribution', 'create', 'Draft community places or edit proposals'),
    ('contribution:read', 'contribution', 'read', 'View own contributions or access moderation review queue'),
    ('contribution:update', 'contribution', 'update', 'Modify content of active draft submissions'),
    ('contribution:withdraw', 'contribution', 'withdraw', 'Withdraw or cancel submitted contribution proposals'),
    ('contribution:approve', 'contribution', 'approve', 'Accept community contribution into canonical map'),
    ('contribution:reject', 'contribution', 'reject', 'Reject community contribution with stated reason'),

    ('dataset:create', 'dataset', 'create', 'Upload or record new sensor trajectory datasets'),
    ('dataset:read', 'dataset', 'read', 'Inspect and download trajectory recordings and ground truth'),
    ('dataset:update', 'dataset', 'update', 'Tag session metadata, vehicle parameters, and sensor mounting info'),
    ('dataset:delete', 'dataset', 'delete', 'Delete datasets from local or cloud storage'),

    ('training:create', 'training', 'create', 'Dispatch neural network model training jobs'),
    ('training:read', 'training', 'read', 'Monitor training status, epoch loss curves, and profiler timings'),

    ('model:create', 'model', 'create', 'Register newly trained ONNX model checkpoints'),
    ('model:read', 'model', 'read', 'Inspect model architectures, weights, and evaluation metrics'),
    ('model:review', 'model', 'review', 'Open a candidate model for human evaluation review'),
    ('model:approve', 'model', 'approve', 'Approve candidate model for production qualification'),
    ('model:deploy', 'model', 'deploy', 'Deploy qualified model to live client navigation runtime'),

    ('user:read', 'user', 'read', 'View user profiles, identifiers, and role statuses'),
    ('user:update', 'user', 'update', 'Modify user profile settings and account states'),

    ('role:assign', 'role', 'assign', 'Assign or revoke roles and permissions for accounts'),

    ('contribution:request_changes', 'contribution', 'request_changes', 'Request revisions on pending community contributions'),
    ('report:create', 'report', 'create', 'Submit community flag or report on map data'),
    ('report:read', 'report', 'read', 'Inspect community reports triage queue'),
    ('report:resolve', 'report', 'resolve', 'Resolve or dismiss community reports'),
    ('audit:read', 'audit', 'read', 'Inspect immutable platform audit logs'),
    ('sync:pull', 'sync', 'pull', 'Pull canonical map updates and changelog deltas'),
    ('sync:push', 'sync', 'push', 'Push local device queue changes to server'),
    ('package:build', 'package', 'build', 'Build and release offline map packages'),

    ('internal_contributor:request', 'contributor', 'request', 'Submit application for internal contributor role'),
    ('internal_contributor:review', 'contributor', 'review', 'View and triage internal contributor applications'),
    ('internal_contributor:approve', 'contributor', 'approve', 'Approve internal contributor access request and promote role'),
    ('internal_contributor:reject', 'contributor', 'reject', 'Reject internal contributor access request with reason'),

    ('dataset:validate', 'dataset', 'validate', 'Validate or reject uploaded dataset sessions for training eligibility');

-- ========================================================
-- Seed Role Permissions Mapping
-- ========================================================

-- 1. Guest permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES
    ('guest', 'place:read');

-- 2. Registered User permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES
    ('user', 'place:read'),
    ('user', 'contribution:create'),
    ('user', 'contribution:read'),
    ('user', 'contribution:update'),
    ('user', 'contribution:withdraw'),
    ('user', 'report:create'),
    ('user', 'sync:pull'),
    ('user', 'sync:push'),
    ('user', 'internal_contributor:request'),
    ('user', 'user:read');

-- 3. Local Contributor permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES
    ('local_contributor', 'place:read'),
    ('local_contributor', 'place:create'),
    ('local_contributor', 'place:update'),
    ('local_contributor', 'contribution:create'),
    ('local_contributor', 'contribution:read'),
    ('local_contributor', 'contribution:update'),
    ('local_contributor', 'contribution:withdraw'),
    ('local_contributor', 'report:create'),
    ('local_contributor', 'sync:pull'),
    ('local_contributor', 'sync:push'),
    ('local_contributor', 'internal_contributor:request'),
    ('local_contributor', 'user:read');

-- 4. Internal Contributor permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES
    ('internal_contributor', 'place:read'),
    ('internal_contributor', 'place:create'),
    ('internal_contributor', 'place:update'),
    ('internal_contributor', 'contribution:create'),
    ('internal_contributor', 'contribution:read'),
    ('internal_contributor', 'contribution:update'),
    ('internal_contributor', 'contribution:withdraw'),
    ('internal_contributor', 'dataset:create'),
    ('internal_contributor', 'dataset:read'),
    ('internal_contributor', 'dataset:update'),
    ('internal_contributor', 'training:create'),
    ('internal_contributor', 'training:read'),
    ('internal_contributor', 'model:create'),
    ('internal_contributor', 'model:read'),
    ('internal_contributor', 'report:create'),
    ('internal_contributor', 'sync:pull'),
    ('internal_contributor', 'sync:push'),
    ('internal_contributor', 'user:read');
    -- NOTE: internal_contributor does NOT have dataset:validate or model:deploy

-- 5. Moderator permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES
    ('moderator', 'place:read'),
    ('moderator', 'place:create'),
    ('moderator', 'place:update'),
    ('moderator', 'place:delete'),
    ('moderator', 'contribution:create'),
    ('moderator', 'contribution:read'),
    ('moderator', 'contribution:update'),
    ('moderator', 'contribution:withdraw'),
    ('moderator', 'contribution:approve'),
    ('moderator', 'contribution:reject'),
    ('moderator', 'contribution:request_changes'),
    ('moderator', 'report:create'),
    ('moderator', 'report:read'),
    ('moderator', 'report:resolve'),
    ('moderator', 'audit:read'),
    ('moderator', 'sync:pull'),
    ('moderator', 'sync:push'),
    ('moderator', 'package:build'),
    ('moderator', 'user:read');

-- 6. Team Admin permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES
    ('team_admin', 'place:read'),
    ('team_admin', 'place:create'),
    ('team_admin', 'place:update'),
    ('team_admin', 'place:delete'),
    ('team_admin', 'contribution:create'),
    ('team_admin', 'contribution:read'),
    ('team_admin', 'contribution:update'),
    ('team_admin', 'contribution:withdraw'),
    ('team_admin', 'contribution:approve'),
    ('team_admin', 'contribution:reject'),
    ('team_admin', 'contribution:request_changes'),
    ('team_admin', 'report:create'),
    ('team_admin', 'report:read'),
    ('team_admin', 'report:resolve'),
    ('team_admin', 'audit:read'),
    ('team_admin', 'dataset:create'),
    ('team_admin', 'dataset:read'),
    ('team_admin', 'dataset:update'),
    ('team_admin', 'dataset:delete'),
    ('team_admin', 'training:create'),
    ('team_admin', 'training:read'),
    ('team_admin', 'model:create'),
    ('team_admin', 'model:read'),
    ('team_admin', 'model:update'),
    ('team_admin', 'model:review'),
    ('team_admin', 'model:approve'),
    ('team_admin', 'model:deploy'),
    ('team_admin', 'sync:pull'),
    ('team_admin', 'sync:push'),
    ('team_admin', 'package:build'),
    ('team_admin', 'internal_contributor:review'),
    ('team_admin', 'internal_contributor:approve'),
    ('team_admin', 'internal_contributor:reject'),
    ('team_admin', 'dataset:validate'),
    ('team_admin', 'user:read'),
    ('team_admin', 'user:update'),
    ('team_admin', 'role:assign');

-- 7. Super Admin permissions (Universal grant)
INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
SELECT 'super_admin', id FROM permissions;
