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

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_permissions_resource ON permissions(resource);
CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_perm ON role_permissions(permission_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id);

-- Trigger: auto-update updated_at on user modification
CREATE TRIGGER IF NOT EXISTS trg_users_updated_at
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = OLD.id;
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
    ('team_admin', 'Team Administrator', 'Engineering lead with ML training, model deployment, and role assignment permissions'),
    ('super_admin', 'Super Administrator', 'Platform governance administrator with universal access across all resources');

-- ========================================================
-- Seed Permissions Catalog (24 granular capabilities)
-- ========================================================
INSERT OR IGNORE INTO permissions (id, resource, action, description) VALUES
    ('place:create', 'place', 'create', 'Add new places, amenities, and POIs to map'),
    ('place:read', 'place', 'read', 'Query and view places, amenities, and POIs'),
    ('place:update', 'place', 'update', 'Modify place details, metadata, and operating hours'),
    ('place:delete', 'place', 'delete', 'Remove places from canonical map data'),

    ('contribution:create', 'contribution', 'create', 'Submit new place edits, hazard reports, or road change proposals'),
    ('contribution:read', 'contribution', 'read', 'View community contributions and review statuses'),
    ('contribution:update', 'contribution', 'update', 'Edit pending contributions before review'),
    ('contribution:withdraw', 'contribution', 'withdraw', 'Withdraw own pending contribution'),
    ('contribution:approve', 'contribution', 'approve', 'Approve contribution and synchronize to canonical map data'),
    ('contribution:reject', 'contribution', 'reject', 'Reject contribution with moderation rationale'),

    ('dataset:create', 'dataset', 'create', 'Upload or record new sensor trajectory datasets'),
    ('dataset:read', 'dataset', 'read', 'Inspect and download trajectory recordings and ground truth'),
    ('dataset:update', 'dataset', 'update', 'Tag session metadata, vehicle parameters, and sensor mounting info'),
    ('dataset:delete', 'dataset', 'delete', 'Delete datasets from local or cloud storage'),

    ('training:create', 'training', 'create', 'Dispatch neural network model training jobs'),
    ('training:read', 'training', 'read', 'Monitor training status, epoch loss curves, and profiler timings'),

    ('model:create', 'model', 'create', 'Register newly trained ONNX model checkpoints'),
    ('model:read', 'model', 'read', 'Inspect model architectures, weights, and evaluation metrics'),
    ('model:update', 'model', 'update', 'Update model metadata and evaluation notes'),
    ('model:approve', 'model', 'approve', 'Approve candidate model for production qualification'),
    ('model:deploy', 'model', 'deploy', 'Deploy qualified model to live client navigation runtime'),

    ('user:read', 'user', 'read', 'View user profiles, identifiers, and role statuses'),
    ('user:update', 'user', 'update', 'Modify user profile settings and account states'),

    ('role:assign', 'role', 'assign', 'Assign or revoke roles and permissions for accounts');

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
    ('user', 'contribution:withdraw'),
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
    ('internal_contributor', 'user:read');

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
    ('team_admin', 'dataset:create'),
    ('team_admin', 'dataset:read'),
    ('team_admin', 'dataset:update'),
    ('team_admin', 'dataset:delete'),
    ('team_admin', 'training:create'),
    ('team_admin', 'training:read'),
    ('team_admin', 'model:create'),
    ('team_admin', 'model:read'),
    ('team_admin', 'model:update'),
    ('team_admin', 'model:approve'),
    ('team_admin', 'model:deploy'),
    ('team_admin', 'user:read'),
    ('team_admin', 'user:update'),
    ('team_admin', 'role:assign');

-- 7. Super Admin permissions (Universal grant)
INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
SELECT 'super_admin', id FROM permissions;
