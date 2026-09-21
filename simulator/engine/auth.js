/**
 * Navigators IDR - Dynamic Role-Based Access Control (RBAC) System
 * Evaluates fine-grained resource:action permissions dynamically without hardcoding role names.
 * Mirrors the canonical database schema (roles, permissions, role_permissions, user_roles).
 */

class NavigatorsAuth {
    constructor() {
        this.STORAGE_KEY = 'navigators_auth_user';
        this.MATRIX_STORAGE_KEY = 'navigators_rbac_matrix';

        // Canonical default matrix (matches src/db/schema.sql)
        this.matrix = {
            roles: [
                { id: 'guest', name: 'Guest', description: 'Unauthenticated visitor' },
                { id: 'user', name: 'Registered User', description: 'Standard member' },
                { id: 'local_contributor', name: 'Local Contributor', description: 'Community map editor' },
                { id: 'internal_contributor', name: 'Internal Contributor', description: 'Field sensor & ML contributor' },
                { id: 'moderator', name: 'Community Moderator', description: 'Reviewer for community edits' },
                { id: 'team_admin', name: 'Team Administrator', description: 'Engineering lead with ML training & deployment' },
                { id: 'super_admin', name: 'Super Administrator', description: 'Universal access across all resources' }
            ],
            role_permissions: {
                guest: ['place:read'],
                user: [
                    'place:read',
                    'contribution:create',
                    'contribution:read',
                    'contribution:withdraw',
                    'user:read'
                ],
                local_contributor: [
                    'place:read',
                    'place:create',
                    'place:update',
                    'contribution:create',
                    'contribution:read',
                    'contribution:update',
                    'contribution:withdraw',
                    'user:read'
                ],
                internal_contributor: [
                    'place:read',
                    'place:create',
                    'place:update',
                    'contribution:create',
                    'contribution:read',
                    'contribution:update',
                    'contribution:withdraw',
                    'dataset:create',
                    'dataset:read',
                    'dataset:update',
                    'training:create',
                    'training:read',
                    'model:create',
                    'model:read',
                    'user:read'
                ],
                moderator: [
                    'place:read',
                    'place:create',
                    'place:update',
                    'place:delete',
                    'contribution:create',
                    'contribution:read',
                    'contribution:update',
                    'contribution:withdraw',
                    'contribution:approve',
                    'contribution:reject',
                    'user:read'
                ],
                team_admin: [
                    'place:read',
                    'place:create',
                    'place:update',
                    'place:delete',
                    'contribution:create',
                    'contribution:read',
                    'contribution:update',
                    'contribution:withdraw',
                    'contribution:approve',
                    'contribution:reject',
                    'dataset:create',
                    'dataset:read',
                    'dataset:update',
                    'dataset:delete',
                    'training:create',
                    'training:read',
                    'model:create',
                    'model:read',
                    'model:update',
                    'model:approve',
                    'model:deploy',
                    'user:read',
                    'user:update',
                    'role:assign'
                ],
                super_admin: [
                    'place:create', 'place:read', 'place:update', 'place:delete',
                    'contribution:create', 'contribution:read', 'contribution:update', 'contribution:withdraw',
                    'contribution:approve', 'contribution:reject',
                    'dataset:create', 'dataset:read', 'dataset:update', 'dataset:delete',
                    'training:create', 'training:read',
                    'model:create', 'model:read', 'model:update', 'model:approve', 'model:deploy',
                    'user:read', 'user:update',
                    'role:assign'
                ]
            }
        };

        this.loadStoredMatrix();
        this.currentUser = this.loadUser();
    }

    loadStoredMatrix() {
        if (typeof localStorage === 'undefined') return;
        try {
            const raw = localStorage.getItem(this.MATRIX_STORAGE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed.role_permissions) {
                    this.matrix = parsed;
                }
            }
        } catch (e) {
            console.warn('[NavigatorsAuth] Failed loading custom RBAC matrix:', e);
        }
    }

    setMatrix(matrix) {
        if (!matrix || !matrix.role_permissions) return;
        this.matrix = matrix;
        if (typeof localStorage !== 'undefined') {
            try {
                localStorage.setItem(this.MATRIX_STORAGE_KEY, JSON.stringify(matrix));
            } catch (e) {}
        }
    }

    loadUser() {
        if (typeof localStorage === 'undefined') {
            return { id: 'usr_guest', name: 'Guest Explorer', email: null, role: 'guest', status: 'active' };
        }
        try {
            const raw = localStorage.getItem(this.STORAGE_KEY);
            if (raw) return JSON.parse(raw);
        } catch (e) {
            console.warn('[NavigatorsAuth] Failed to load user:', e);
        }
        return {
            id: 'usr_guest',
            name: 'Guest Explorer',
            email: null,
            role: 'guest',
            status: 'active',
            created_at: new Date().toISOString()
        };
    }

    saveUser(user) {
        this.currentUser = user;
        if (typeof localStorage !== 'undefined') {
            try {
                localStorage.setItem(this.STORAGE_KEY, JSON.stringify(user));
            } catch (e) {
                console.warn('[NavigatorsAuth] Failed to save user:', e);
            }
        }
        if (typeof window !== 'undefined' && typeof window.updateAuthUI === 'function') {
            window.updateAuthUI(this.currentUser);
        }
    }

    setRole(roleId) {
        if (!this.matrix.role_permissions[roleId]) {
            throw new Error(`Role "${roleId}" not defined in active RBAC matrix.`);
        }
        const roleMeta = this.matrix.roles?.find(r => r.id === roleId);
        const user = {
            ...this.currentUser,
            role: roleId,
            name: roleMeta ? `${roleMeta.name} Session` : `${roleId} User`,
            updated_at: new Date().toISOString()
        };
        this.saveUser(user);
        return user;
    }

    /**
     * Core dynamic security evaluation: checks whether active user has permission
     * without any hardcoded role name checks.
     */
    hasPermission(permissionId) {
        if (this.currentUser.status !== 'active') return false;
        const role = this.currentUser.role || 'guest';
        const granted = this.matrix.role_permissions[role] || [];

        // Direct check
        if (granted.includes(permissionId)) return true;

        // Legacy compatibility aliases
        const aliases = {
            navigate: 'place:read',
            offline_maps: 'place:read',
            add_place: 'place:create',
            suggest_road_change: 'contribution:create',
            report_hazard: 'contribution:create',
            save_places: 'contribution:create',
            view_my_contributions: 'contribution:read',
            approve_places: 'contribution:approve',
            record_trajectory: 'dataset:create',
            tag_sessions: 'dataset:update',
            submit_training: 'training:create',
            view_model_evaluation: 'model:read',
            manage_internal_access: 'role:assign',
            approve_production_model: 'model:deploy'
        };

        const canonical = aliases[permissionId];
        return canonical ? granted.includes(canonical) : false;
    }

    getUserPermissions() {
        if (this.currentUser.status !== 'active') return [];
        const role = this.currentUser.role || 'guest';
        return this.matrix.role_permissions[role] || [];
    }

    requestInternalAccess(reason, deviceHardware) {
        if (typeof localStorage === 'undefined') return null;
        const requests = JSON.parse(localStorage.getItem('navigators_internal_requests') || '[]');
        const existing = requests.find(r => r.userId === this.currentUser.id);
        const reqData = {
            id: 'req_' + Date.now(),
            userId: this.currentUser.id,
            userName: this.currentUser.name,
            reason: reason || 'Sensor data collection and model evaluation',
            device: deviceHardware || (typeof navigator !== 'undefined' ? navigator.userAgent : 'Unknown'),
            state: 'Pending Review',
            created_at: new Date().toISOString()
        };
        if (existing) {
            Object.assign(existing, reqData);
        } else {
            requests.unshift(reqData);
        }
        localStorage.setItem('navigators_internal_requests', JSON.stringify(requests));
        localStorage.setItem('navigators_internal_access_state', 'Pending Review');
        return reqData;
    }

    approveInternalAccess(requestId) {
        if (typeof localStorage === 'undefined') return;
        const requests = JSON.parse(localStorage.getItem('navigators_internal_requests') || '[]');
        const req = requests.find(r => r.id === requestId);
        if (req) {
            req.state = 'Approved';
            localStorage.setItem('navigators_internal_requests', JSON.stringify(requests));
            if (this.currentUser.id === req.userId) {
                this.setRole('internal_contributor');
                localStorage.setItem('navigators_internal_access_state', 'Approved');
            }
        }
    }

    getInternalRequests() {
        if (typeof localStorage === 'undefined') return [];
        return JSON.parse(localStorage.getItem('navigators_internal_requests') || '[]');
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = NavigatorsAuth;
} else if (typeof window !== 'undefined') {
    window.NavigatorsAuth = NavigatorsAuth;
}
