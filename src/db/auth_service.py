from __future__ import annotations
"""
Navigators IDR - Authentication & Session Management Service
Implements PBKDF2-HMAC-SHA256 password hashing, extensible identity management,
secure session handling, and dynamic RBAC resolution.

Security Architecture:
  User -> Login -> Identity Verified -> Session/Token -> User Record -> Role Lookup -> Permissions Loaded
  The backend unconditionally determines: authenticated_user -> database -> role -> permissions.
  Client-supplied role or permission assertions are never trusted.
"""

import os
import hmac
import hashlib
import secrets
import sqlite3
from typing import Optional, List, Set, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.db.database import get_db
from src.db.rbac import RBACRepository, User, Role, Permission






def hash_password(password: str, iterations: int = 600_000) -> str:
    """
    Hash a password using salted PBKDF2-HMAC-SHA256 with 600,000 iterations.
    Returns format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters in length.")
    
    salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def verify_password(password: str, hashed_str: str) -> bool:
    """
    Verify a plaintext password against a stored PBKDF2 hash using constant-time comparison.
    """
    if not password or not hashed_str:
        return False
    
    parts = hashed_str.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    
    try:
        iterations = int(parts[1])
        salt = parts[2]
        expected_hash = parts[3]
    except (ValueError, IndexError):
        return False

    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return hmac.compare_digest(dk.hex(), expected_hash)


def generate_session_token() -> Tuple[str, str]:
    """
    Generate a 32-byte cryptographically secure session token (64 hex characters)
    and compute its SHA-256 digest for database storage.
    """
    raw_token = secrets.token_hex(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    return raw_token, token_hash


def compute_token_hash(raw_token: str) -> str:
    """Compute SHA-256 digest of raw token for database query."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()






@dataclass
class AuthIdentity:
    id: str
    user_id: str
    provider: str
    identifier: str
    credential_hash: Optional[str]
    metadata_json: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionRecord:
    id: str
    token_hash: str
    user_id: Optional[str]
    is_guest: int
    created_at: str
    expires_at: str
    revoked_at: Optional[str]
    last_seen_at: str
    user_agent: Optional[str]
    ip_address: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionContext:
    """
    Authenticated security context resolved exclusively by the server database.
    Contains user record, roles, and granular permissions.
    """
    session_id: str
    user: Optional[User]
    roles: List[Role]
    permissions: Set[str]
    is_guest: bool
    created_at: str
    expires_at: str
    last_seen_at: str
    db_path: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "session_id": self.session_id,
            "user": self.user.to_dict() if self.user else None,
            "roles": [r.to_dict() for r in self.roles],
            "permissions": sorted(list(self.permissions)),
            "is_guest": self.is_guest,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_seen_at": self.last_seen_at,
        }
        if self.db_path is not None:
            d["db_path"] = str(self.db_path)
        return d






class AuthService:
    """
    Handles user authentication, extensible multi-provider identity records,
    session tokens, and dynamic permission resolution.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        rbac_repo: Optional[RBACRepository] = None,
    ):
        self.db_path = db_path
        self.rbac = rbac_repo or RBACRepository(db_path)





    def create_identity(
        self,
        user_id: str,
        provider: str,
        identifier: str,
        credential_hash: Optional[str] = None,
        metadata_json: Optional[str] = None,
    ) -> AuthIdentity:
        """Store an authentication identity mapping for a user."""
        identity_id = f"idn_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO auth_identities (id, user_id, provider, identifier, credential_hash, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (identity_id, user_id, provider, identifier.lower().strip(), credential_hash, metadata_json, now, now),
            )
        return AuthIdentity(
            id=identity_id,
            user_id=user_id,
            provider=provider,
            identifier=identifier.lower().strip(),
            credential_hash=credential_hash,
            metadata_json=metadata_json,
            created_at=now,
            updated_at=now,
        )

    def get_identity(self, provider: str, identifier: str) -> Optional[AuthIdentity]:
        """Fetch identity by provider and identifier."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM auth_identities WHERE provider = ? AND identifier = ?",
                (provider, identifier.lower().strip()),
            )
            row = cur.fetchone()
            if not row:
                return None
            return AuthIdentity(**dict(row))





    def register(
        self,
        email: str,
        password: str,
        name: str,
        avatar: Optional[str] = None,
        role_id: str = "user",
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[User, SessionContext, str]:
        """
        Register a new user account with local password identity,
        assign default role, and issue an authenticated session token.
        """
        clean_email = email.lower().strip()
        clean_name = name.strip()

        if not clean_email or "@" not in clean_email:
            raise ValueError("A valid email address is required.")
        if not clean_name:
            raise ValueError("A display name is required.")


        if self.rbac.get_user_by_email(clean_email):
            raise ValueError(f"User with email '{clean_email}' already exists.")
        if self.get_identity("local_password", clean_email):
            raise ValueError(f"Identity with email '{clean_email}' already exists.")

        user_id = f"usr_{secrets.token_hex(8)}"
        pwd_hash = hash_password(password)


        user = self.rbac.create_user(
            user_id=user_id,
            email=clean_email,
            name=clean_name,
            avatar=avatar,
            status="active",
            initial_role_ids=[role_id],
        )


        self.create_identity(
            user_id=user.id,
            provider="local_password",
            identifier=clean_email,
            credential_hash=pwd_hash,
        )


        session_context, raw_token = self.create_session(
            user_id=user.id,
            is_guest=False,
            duration_days=30,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return user, session_context, raw_token






    def login(
        self,
        email: str,
        password: str,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[User, SessionContext, str]:
        """
        Authenticate user credentials, verify identity, create session,
        and dynamically load roles and permissions from database.
        """
        clean_email = email.lower().strip()
        identity = self.get_identity("local_password", clean_email)
        if not identity or not identity.credential_hash:
            raise ValueError("Invalid email or password.")

        if not verify_password(password, identity.credential_hash):
            raise ValueError("Invalid email or password.")

        user = self.rbac.get_user(identity.user_id)
        if not user:
            raise ValueError("Associated user record not found.")

        if user.status != "active":
            raise ValueError(f"User account status is '{user.status}'. Access denied.")


        self.rbac.record_login(user.id)
        user = self.rbac.get_user(user.id) or user


        session_context, raw_token = self.create_session(
            user_id=user.id,
            is_guest=False,
            duration_days=30,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return user, session_context, raw_token





    def create_guest_session(
        self,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[SessionContext, str]:
        """
        Issue an anonymous guest session without creating an account in the users table.
        Guest sessions receive the 'guest' role and 'place:read' permissions.
        """
        return self.create_session(
            user_id=None,
            is_guest=True,
            duration_days=7,
            user_agent=user_agent,
            ip_address=ip_address,
        )





    def create_session(
        self,
        user_id: Optional[str],
        is_guest: bool = False,
        duration_days: int = 30,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[SessionContext, str]:
        """Create and store a session record in the database."""
        session_id = f"ses_{secrets.token_hex(8)}"
        raw_token, token_hash = generate_session_token()

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=duration_days)

        now_str = now.isoformat()
        expires_str = expires_at.isoformat()

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, token_hash, user_id, is_guest, created_at, expires_at, revoked_at, last_seen_at, user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    session_id,
                    token_hash,
                    user_id,
                    1 if is_guest else 0,
                    now_str,
                    expires_str,
                    now_str,
                    user_agent,
                    ip_address,
                ),
            )

        context = self.resolve_session(raw_token)
        if not context:
            raise RuntimeError("Failed to resolve freshly created session.")
        return context, raw_token

    def resolve_session(self, raw_token: str) -> Optional[SessionContext]:
        """
        Primary security gate: Validates session token against database,
        updates last_seen_at, and dynamically resolves user, roles, and permissions.
        
        The server unconditionally determines roles and permissions via relational joins.
        """
        if not raw_token:
            return None

        token_hash = compute_token_hash(raw_token)
        now_dt = datetime.now(timezone.utc)
        now_str = now_dt.isoformat()

        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM sessions WHERE token_hash = ?",
                (token_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None

            session = SessionRecord(**dict(row))


            if session.revoked_at is not None:
                return None


            try:
                expires_dt = datetime.fromisoformat(session.expires_at)
                if expires_dt <= now_dt:
                    return None
            except ValueError:
                return None


            conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                (now_str, session.id),
            )


        if session.is_guest == 1 or session.user_id is None:
            guest_role = self.rbac.get_role("guest") or Role(
                id="guest",
                name="Guest",
                description="Unauthenticated visitor",
            )
            guest_perms = self.rbac.get_role_permissions("guest")
            guest_perm_ids = {p.id for p in guest_perms} if guest_perms else {"place:read"}

            virtual_user = User(
                id=f"guest_{session.id}",
                email="",
                name="Guest Explorer",
                avatar=None,
                status="active",
                created_at=session.created_at,
                updated_at=session.last_seen_at,
                last_login_at=None,
            )

            return SessionContext(
                session_id=session.id,
                user=virtual_user,
                roles=[guest_role],
                permissions=guest_perm_ids,
                is_guest=True,
                created_at=session.created_at,
                expires_at=session.expires_at,
                last_seen_at=now_str,
                db_path=str(self.db_path) if self.db_path else None,
            )



        user = self.rbac.get_user(session.user_id)
        if not user or user.status != "active":
            return None

        roles = self.rbac.get_user_roles(user.id)
        permissions = self.rbac.get_user_permissions(user.id)

        return SessionContext(
            session_id=session.id,
            user=user,
            roles=roles,
            permissions=permissions,
            is_guest=False,
            created_at=session.created_at,
            expires_at=session.expires_at,
            last_seen_at=now_str,
            db_path=str(self.db_path) if self.db_path else None,
        )

    def revoke_session(self, raw_token: str) -> bool:
        """Revoke a session immediately by raw token."""
        if not raw_token:
            return False
        token_hash = compute_token_hash(raw_token)
        now_str = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
                (now_str, token_hash),
            )
            return cur.rowcount > 0

    def revoke_session_by_id(self, session_id: str) -> bool:
        """Revoke a session by session id."""
        now_str = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (now_str, session_id),
            )
            return cur.rowcount > 0

    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all active sessions for a user (e.g. upon password reset or suspension)."""
        now_str = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                (now_str, user_id),
            )
            return cur.rowcount
