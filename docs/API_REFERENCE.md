# Navigators — REST API Reference Specification

```
Document Identifier: API-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Global API Standards

- **Base URL**: `/api/v1`
- **Protocol**: HTTPS / REST
- **Payload Format**: `application/json` (UTF-8)
- **Authentication**: HTTP Bearer Header (`Authorization: Bearer <raw_token>`)
- **Error Response Schema**:
  ```json
  {
    "detail": "Descriptive error message"
  }
  ```

---

## 2. Authentication Endpoints (`/api/v1/auth`)

### 2.1 Register New Account
- **Endpoint**: `POST /api/v1/auth/register`
- **Access**: Public
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "name": "Jane Doe",
    "password": "SecurePassword123!"
  }
  ```
- **Response (201 Created)**:
  ```json
  {
    "message": "Account created successfully.",
    "token": "raw_bearer_token_string",
    "user": { "id": "usr_abc123", "email": "user@example.com", "name": "Jane Doe", "roles": ["user"] }
  }
  ```

### 2.2 Login
- **Endpoint**: `POST /api/v1/auth/login`
- **Access**: Public
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "SecurePassword123!"
  }
  ```
- **Response (200 OK)**: Returns new bearer token and user profile.

### 2.3 Logout & Revoke Session
- **Endpoint**: `POST /api/v1/auth/logout`
- **Access**: Authenticated (Bearer)
- **Response (200 OK)**: `{"message": "Session revoked successfully."}`

---

## 3. Places & Cartography (`/api/v1/places`)

### 3.1 Search Canonical Places
- **Endpoint**: `GET /api/v1/places/search`
- **Access**: Public (Optional Bearer)
- **Query Parameters**:
  - `q` (string): Text search query.
  - `category` (string, optional): Filter by category (`fuel`, `hospital`, `ev_charging`, etc.).
  - `lat`, `lon` (float, optional): Center coordinates for proximity sorting.
  - `radius_km` (float, optional, default: 10.0).
- **Response (200 OK)**:
  ```json
  {
    "count": 2,
    "places": [
      {
        "id": "plc_123",
        "name": "Central Hospital EV Hub",
        "category": "ev_charging",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "verified": 1
      }
    ]
  }
  ```

---

## 4. Community Contributions (`/api/v1/contributions`)

### 4.1 Create Contribution Draft
- **Endpoint**: `POST /api/v1/contributions`
- **Access**: Authenticated (`contribution:create`)
- **Request Body**:
  ```json
  {
    "resource_type": "place",
    "title": "New Community Pharmacy",
    "data": { "name": "Apollo Pharmacy", "category": "pharmacy", "latitude": 12.95, "longitude": 77.58 },
    "submit_now": false
  }
  ```
- **Response (201 Created)**: Returns created contribution object in `draft` or `pending` status.

### 4.2 Submit Draft for Moderation
- **Endpoint**: `POST /api/v1/contributions/{id}/submit`
- **Access**: Author Only (`contribution:update`)
- **Response (200 OK)**: Transitions status to `pending_review`.

### 4.3 Moderator Review (Approve / Reject)
- **Endpoint**: `POST /api/v1/contributions/{id}/review`
- **Access**: Moderator Only (`contribution:approve` / `contribution:reject`)
- **Request Body**:
  ```json
  {
    "decision": "approved",
    "notes": "Coordinates verified via OpenStreetMap orthophoto."
  }
  ```
- **Response (200 OK)**: Returns updated contribution in `approved` or `rejected` status.

---

## 5. Model Registry & Governance (`/api/v1/models`)

### 5.1 List Models
- **Endpoint**: `GET /api/v1/models`
- **Access**: Authenticated (`model:read`)
- **Response (200 OK)**: Array of candidate and production models with architecture, parameter counts, and scores.

### 5.2 Promote Candidate to Production
- **Endpoint**: `POST /api/v1/models/{id}/deploy`
- **Access**: Team Admin (`model:deploy`)
- **Response (200 OK)**: Sets `is_production = 1` and logs an audit record.

### 5.3 Rollback Production Model
- **Endpoint**: `POST /api/v1/models/rollback`
- **Access**: Team Admin (`model:rollback`)
- **Response (200 OK)**: Restores previous production model.

---

## 6. Offline Synchronization (`/api/v1/sync`)

### 6.1 Replay Offline Queue
- **Endpoint**: `POST /api/v1/sync/queue`
- **Access**: Authenticated (Bearer)
- **Request Body**:
  ```json
  {
    "items": [
      {
        "client_sync_id": "c1f7a22e-5034-4b55-b461-9f939e08a0d1",
        "action": "contribution:create",
        "payload": { "title": "Offline Landmark", "data": { ... } }
      }
    ]
  }
  ```
- **Response (200 OK)**: Returns array of sync results mapped by `client_sync_id`. Duplicate sync IDs are acknowledged without re-execution.

---

Developed by Navigators
