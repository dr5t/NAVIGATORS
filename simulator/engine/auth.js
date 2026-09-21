/**
 * Navigators IDR - Dynamic Role-Based Access Control (RBAC) & Session Authentication
 * Evaluates fine-grained resource:action permissions dynamically without hardcoding role names.
 * Mirrors the canonical database schema and resolves authenticated sessions from server.
 */

class NavigatorsAuth {
    constructor() {
        this.STORAGE_KEY = 'navigators_auth_user';
        this.TOKEN_STORAGE_KEY = 'navigators_session_token';
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
                    'contribution:update',
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

        // Attempt asynchronous session verification if token is present
        if (typeof window !== 'undefined' && this.getToken()) {
            this.fetchSession().catch(e => {
                console.log('[NavigatorsAuth] Offline or local session initialized:', e?.message || e);
            });
        }
    }

    getToken() {
        if (typeof localStorage === 'undefined') return null;
        return localStorage.getItem(this.TOKEN_STORAGE_KEY);
    }

    setToken(token) {
        if (typeof localStorage === 'undefined') return;
        if (token) {
            localStorage.setItem(this.TOKEN_STORAGE_KEY, token);
        } else {
            localStorage.removeItem(this.TOKEN_STORAGE_KEY);
        }
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
            return {
                id: 'usr_guest',
                name: 'Guest Explorer',
                email: null,
                role: 'guest',
                permissions: ['place:read'],
                is_guest: true,
                status: 'active'
            };
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
            permissions: ['place:read'],
            is_guest: true,
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

    // -------------------------------------------------------------------------
    // Server-Side Authentication Operations
    // -------------------------------------------------------------------------

    async login(email, password) {
        const response = await fetch('/api/v1/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Authentication failed');
        }

        this.setToken(data.token);
        const session = data.session;
        const primaryRole = (session.roles && session.roles.length > 0) ? session.roles[0].id : 'user';

        const userObj = {
            id: session.user ? session.user.id : 'usr_' + Date.now().toString(36),
            name: session.user ? session.user.name : email.split('@')[0],
            email: session.user ? session.user.email : email,
            role: primaryRole,
            roles: session.roles || [],
            permissions: session.permissions || [],
            is_guest: false,
            status: 'active',
            session_id: session.session_id,
            updated_at: new Date().toISOString()
        };

        this.saveUser(userObj);
        return { user: userObj, token: data.token };
    }

    async register(name, email, password) {
        const response = await fetch('/api/v1/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Registration failed');
        }

        this.setToken(data.token);
        const session = data.session;
        const primaryRole = (session.roles && session.roles.length > 0) ? session.roles[0].id : 'user';

        const userObj = {
            id: session.user ? session.user.id : 'usr_' + Date.now().toString(36),
            name: session.user ? session.user.name : name,
            email: session.user ? session.user.email : email,
            role: primaryRole,
            roles: session.roles || [],
            permissions: session.permissions || [],
            is_guest: false,
            status: 'active',
            session_id: session.session_id,
            updated_at: new Date().toISOString()
        };

        this.saveUser(userObj);
        return { user: userObj, token: data.token };
    }

    async startGuestSession() {
        try {
            const response = await fetch('/api/v1/auth/guest', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                this.setToken(data.token);
                const guestUser = {
                    id: data.session.session_id ? `guest_${data.session.session_id}` : 'usr_guest',
                    name: 'Guest Explorer',
                    email: null,
                    role: 'guest',
                    roles: [{ id: 'guest', name: 'Guest' }],
                    permissions: data.session.permissions || ['place:read'],
                    is_guest: true,
                    status: 'active'
                };
                this.saveUser(guestUser);
                return guestUser;
            }
        } catch (e) {
            console.log('[NavigatorsAuth] Using offline guest session fallback');
        }

        // Offline fallback
        this.setToken(null);
        const offlineGuest = {
            id: 'usr_guest',
            name: 'Guest Explorer',
            email: null,
            role: 'guest',
            roles: [{ id: 'guest', name: 'Guest' }],
            permissions: ['place:read'],
            is_guest: true,
            status: 'active'
        };
        this.saveUser(offlineGuest);
        return offlineGuest;
    }

    async fetchSession() {
        const token = this.getToken();
        if (!token) return null;

        try {
            const response = await fetch('/api/v1/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (!response.ok) {
                // Token invalid or expired
                this.setToken(null);
                return null;
            }

            const session = await response.json();
            const primaryRole = (session.roles && session.roles.length > 0) ? session.roles[0].id : (session.is_guest ? 'guest' : 'user');

            const userObj = {
                id: session.user ? session.user.id : `guest_${session.session_id}`,
                name: session.user ? session.user.name : 'Guest Explorer',
                email: session.user ? session.user.email : null,
                role: primaryRole,
                roles: session.roles || [],
                permissions: session.permissions || [],
                is_guest: session.is_guest,
                status: session.user ? session.user.status : 'active',
                session_id: session.session_id,
                updated_at: new Date().toISOString()
            };

            this.saveUser(userObj);
            return userObj;
        } catch (e) {
            console.log('[NavigatorsAuth] Backend unreachable during fetchSession');
            return null;
        }
    }

    async logout() {
        const token = this.getToken();
        if (token) {
            try {
                await fetch('/api/v1/auth/logout', {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` }
                });
            } catch (e) {
                console.log('[NavigatorsAuth] Logout network error:', e);
            }
        }
        this.setToken(null);
        return this.startGuestSession();
    }

    setRole(roleId) {
        if (!this.matrix.role_permissions[roleId]) {
            throw new Error(`Role "${roleId}" not defined in active RBAC matrix.`);
        }
        const roleMeta = this.matrix.roles?.find(r => r.id === roleId);
        const perms = this.matrix.role_permissions[roleId] || [];
        const user = {
            ...this.currentUser,
            role: roleId,
            permissions: perms,
            name: roleMeta ? `${roleMeta.name} Session` : `${roleId} User`,
            updated_at: new Date().toISOString()
        };
        this.saveUser(user);
        return user;
    }

    /**
     * Core dynamic security evaluation: checks whether active user has permission.
     * Evaluates server-provided permissions dynamically without hardcoding role names.
     */
    hasPermission(permissionId) {
        if (this.currentUser.status !== 'active') return false;

        // 1. If server session returned verified permissions, use them directly
        if (Array.isArray(this.currentUser.permissions) && this.currentUser.permissions.length > 0) {
            if (this.currentUser.permissions.includes(permissionId)) return true;
        }

        // 2. Otherwise fall back to local RBAC matrix for current role
        const role = this.currentUser.role || 'guest';
        const granted = this.matrix.role_permissions[role] || [];
        if (granted.includes(permissionId)) return true;

        // 3. Compatibility aliases
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
        if (Array.isArray(this.currentUser.permissions) && this.currentUser.permissions.length > 0) {
            return this.currentUser.permissions;
        }
        const role = this.currentUser.role || 'guest';
        return this.matrix.role_permissions[role] || [];
    }

    /**
     * Centralized Authorization Service API (Phase 4):
     * Answers: Can user X perform action Y on resource Z?
     * Evaluates:
     *   1. Authentication (active vs guest vs inactive)
     *   2. Role & Permissions (granted capability)
     *   3. Ownership (author/owner constraints for personal actions)
     *   4. Resource State (lifecycle state: pending, approved, rejected, archived)
     * Never scatter authorization logic across frontend components.
     */
    can(action, resource = null) {
        // 1. Authentication
        if (!this.currentUser || this.currentUser.status !== 'active') {
            return { allowed: false, reason: 'Account is inactive or not logged in.', code: 'ACCOUNT_INACTIVE' };
        }
        const isGuest = Boolean(this.currentUser.is_guest || this.currentUser.role === 'guest');
        if (isGuest && action !== 'place:read') {
            return { allowed: false, reason: 'Action requires an authenticated user account.', code: 'UNAUTHENTICATED' };
        }

        // 2. Role & Permission
        if (!this.hasPermission(action)) {
            return { allowed: false, reason: `Missing required permission '${action}'.`, code: 'PERMISSION_DENIED' };
        }

        if (!resource) {
            return { allowed: true, reason: `Permission '${action}' is granted.`, code: 'AUTHORIZED' };
        }

        // 3. Ownership
        const ownerId = resource.user_id || resource.owner_id || resource.created_by || resource.author_id;
        const currentUserId = this.currentUser.id;
        const isOwner = Boolean(ownerId && currentUserId && (ownerId === currentUserId));
        const roles = this.currentUser.roles ? this.currentUser.roles.map(r => r.id || r) : [this.currentUser.role];
        const isAdmin = roles.includes('team_admin') || roles.includes('super_admin');

        const authorOnlyActions = ['contribution:withdraw', 'contribution:update'];
        if (authorOnlyActions.includes(action)) {
            if (!isOwner) {
                return { allowed: false, reason: `Only the creator can perform '${action}'.`, code: 'NOT_OWNER' };
            }
        }

        if (action === 'user:update' && ownerId) {
            if (!isOwner && !isAdmin) {
                return { allowed: false, reason: 'Cannot update profile of another user without admin rights.', code: 'NOT_OWNER' };
            }
        }

        if ((action === 'dataset:update' || action === 'dataset:delete') && ownerId) {
            if (!isOwner && !isAdmin) {
                return { allowed: false, reason: 'User is not the owner of this dataset.', code: 'NOT_OWNER' };
            }
        }

        // 4. Resource State
        const state = resource.status || resource.state;
        if (state) {
            const stateNorm = String(state).toLowerCase();
            if (['contribution:update', 'contribution:withdraw'].includes(action)) {
                if (['approved', 'rejected', 'withdrawn'].includes(stateNorm)) {
                    return { allowed: false, reason: `Cannot perform '${action}' on contribution with status '${state}'.`, code: 'INVALID_RESOURCE_STATE' };
                }
            }
            if (['contribution:approve', 'contribution:reject'].includes(action)) {
                if (stateNorm !== 'pending') {
                    return { allowed: false, reason: `Cannot review contribution with status '${state}'. Must be pending.`, code: 'INVALID_RESOURCE_STATE' };
                }
            }
            if (action === 'model:deploy') {
                if (!['approved', 'qualified', 'ready'].includes(stateNorm)) {
                    return { allowed: false, reason: `Cannot deploy model with status '${state}'. Must be approved or qualified.`, code: 'INVALID_RESOURCE_STATE' };
                }
            }
            if (['dataset:update', 'dataset:delete'].includes(action)) {
                if (['archived', 'locked', 'read_only'].includes(stateNorm)) {
                    return { allowed: false, reason: `Cannot mutate dataset in '${state}' state.`, code: 'INVALID_RESOURCE_STATE' };
                }
            }
        }

        return { allowed: true, reason: `Action '${action}' is authorized.`, code: 'AUTHORIZED' };
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
