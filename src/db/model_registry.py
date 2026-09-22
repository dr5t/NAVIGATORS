"""
Navigators IDR - Model Registry Repository (Phase 15)
Source of truth for all model candidates, their evaluation metrics, approval
decisions, and deployment history.

State machine:
  CANDIDATE_TRAINING
        ↓  record_evaluation()
    EVALUATING
        ↓  open_for_review()
      REVIEW
      /    \\
 REJECTED  APPROVED
               ↓  mark_production_candidate()
     PRODUCTION_CANDIDATE
               ↓  deploy()  [model:deploy gate]
          PRODUCTION

Terminal states: REJECTED, PRODUCTION
"""

import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.db.database import get_db, DEFAULT_DB_PATH
from src.db.audit import AuditRepository

# ─── State machine ────────────────────────────────────────────────────────────

MODEL_STATUSES = frozenset({
    "candidate_training",
    "evaluating",
    "review",
    "approved",
    "rejected",
    "production_candidate",
    "production",
})

VALID_TRANSITIONS: Dict[str, frozenset] = {
    "candidate_training":   frozenset({"evaluating"}),
    "evaluating":           frozenset({"review"}),
    "review":               frozenset({"approved", "rejected"}),
    "approved":             frozenset({"production_candidate"}),
    "rejected":             frozenset(),        # terminal
    "production_candidate": frozenset({"production"}),
    "production":           frozenset(),        # terminal
}


@dataclass
class ModelEntry:
    id: str
    name: str
    architecture: str
    status: str
    # Metrics
    test_mae: Optional[float]
    test_rmse: Optional[float]
    val_loss: Optional[float]
    onnx_parity_max_diff: Optional[float]
    error_reduction_pct: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    window_size: Optional[int]
    total_training_time_s: Optional[float]
    # Artifacts
    checkpoint_path: Optional[str]
    onnx_path: Optional[str]
    norm_stats_path: Optional[str]
    # Governance
    registered_by: Optional[str]
    reviewed_by: Optional[str]
    deployed_by: Optional[str]
    rejection_reason: Optional[str]
    review_notes: Optional[str]
    # Timestamps
    evaluated_at: Optional[str]
    reviewed_at: Optional[str]
    deployed_at: Optional[str]
    created_at: str
    updated_at: str
    # Joined display names
    registered_by_name: Optional[str] = None
    reviewed_by_name: Optional[str] = None
    deployed_by_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "architecture": self.architecture,
            "status": self.status,
            "test_mae": self.test_mae,
            "test_rmse": self.test_rmse,
            "val_loss": self.val_loss,
            "onnx_parity_max_diff": self.onnx_parity_max_diff,
            "error_reduction_pct": self.error_reduction_pct,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "total_training_time_s": self.total_training_time_s,
            "checkpoint_path": self.checkpoint_path,
            "onnx_path": self.onnx_path,
            "norm_stats_path": self.norm_stats_path,
            "registered_by": self.registered_by,
            "registered_by_name": self.registered_by_name,
            "reviewed_by": self.reviewed_by,
            "reviewed_by_name": self.reviewed_by_name,
            "deployed_by": self.deployed_by,
            "deployed_by_name": self.deployed_by_name,
            "rejection_reason": self.rejection_reason,
            "review_notes": self.review_notes,
            "evaluated_at": self.evaluated_at,
            "reviewed_at": self.reviewed_at,
            "deployed_at": self.deployed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ModelRegistryRepository:
    """Source of truth for all model lifecycle events."""

    _SELECT_FULL = """
        SELECT
            m.id, m.name, m.architecture, m.status,
            m.test_mae, m.test_rmse, m.val_loss,
            m.onnx_parity_max_diff, m.error_reduction_pct,
            m.epochs, m.batch_size, m.window_size, m.total_training_time_s,
            m.checkpoint_path, m.onnx_path, m.norm_stats_path,
            m.registered_by, m.reviewed_by, m.deployed_by,
            m.rejection_reason, m.review_notes,
            m.evaluated_at, m.reviewed_at, m.deployed_at,
            m.created_at, m.updated_at,
            r.name AS registered_by_name,
            v.name AS reviewed_by_name,
            d.name AS deployed_by_name
        FROM model_registry m
        LEFT JOIN users r ON m.registered_by = r.id
        LEFT JOIN users v ON m.reviewed_by   = v.id
        LEFT JOIN users d ON m.deployed_by   = d.id
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH

    @property
    def db_path(self) -> Path:
        return self._db_path

    @db_path.setter
    def db_path(self, val: Path) -> None:
        self._db_path = val
        self.audit = AuditRepository(val)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _row_to_obj(self, row) -> ModelEntry:
        def _f(v):
            return float(v) if v is not None else None

        def _i(v):
            return int(v) if v is not None else None

        return ModelEntry(
            id=row["id"], name=row["name"], architecture=row["architecture"],
            status=row["status"],
            test_mae=_f(row["test_mae"]), test_rmse=_f(row["test_rmse"]),
            val_loss=_f(row["val_loss"]),
            onnx_parity_max_diff=_f(row["onnx_parity_max_diff"]),
            error_reduction_pct=_f(row["error_reduction_pct"]),
            epochs=_i(row["epochs"]), batch_size=_i(row["batch_size"]),
            window_size=_i(row["window_size"]),
            total_training_time_s=_f(row["total_training_time_s"]),
            checkpoint_path=row["checkpoint_path"],
            onnx_path=row["onnx_path"],
            norm_stats_path=row["norm_stats_path"],
            registered_by=row["registered_by"],
            reviewed_by=row["reviewed_by"],
            deployed_by=row["deployed_by"],
            rejection_reason=row["rejection_reason"],
            review_notes=row["review_notes"],
            evaluated_at=row["evaluated_at"],
            reviewed_at=row["reviewed_at"],
            deployed_at=row["deployed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            registered_by_name=row["registered_by_name"],
            reviewed_by_name=row["reviewed_by_name"],
            deployed_by_name=row["deployed_by_name"],
        )

    def _assert_transition(self, current: str, target: str) -> None:
        allowed = VALID_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            terminal = not allowed
            raise ValueError(
                f"Invalid state transition: '{current}' → '{target}'. "
                + (f"Allowed: {sorted(allowed)}" if not terminal
                   else f"'{current}' is a terminal state and cannot transition.")
            )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ────────────────────────────────────────────────────────────

    def register_candidate(
        self,
        name: str,
        registered_by: Optional[str] = None,
        architecture: str = "TCNVelocityEstimator",
        checkpoint_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
    ) -> ModelEntry:
        """
        Create a new registry entry in candidate_training state.
        Called by the training pipeline when a run begins.
        """
        if not name.strip():
            raise ValueError("Model name cannot be empty")
        model_id = f"mdl_{secrets.token_hex(8)}"
        now = self._now()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO model_registry (
                    id, name, architecture, status,
                    checkpoint_path, onnx_path, norm_stats_path,
                    registered_by, created_at, updated_at
                ) VALUES (?, ?, ?, 'candidate_training', ?, ?, ?, ?, ?, ?)
                """,
                (model_id, name.strip(), architecture,
                 checkpoint_path, onnx_path, norm_stats_path,
                 registered_by, now, now),
            )
        self.audit.log(
            actor_id=registered_by,
            action="REGISTER_MODEL_CANDIDATE",
            resource_type="model_registry",
            resource_id=model_id,
            old_state=None,
            new_state="candidate_training",
            metadata={"name": name, "architecture": architecture},
        )
        return self.get_model(model_id)

    def record_evaluation(self, model_id: str, metrics: Dict[str, Any]) -> ModelEntry:
        """
        Attach evaluation metrics and transition candidate_training → evaluating.
        Called automatically after training completes.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model '{model_id}' not found")
        self._assert_transition(entry.status, "evaluating")

        now = self._now()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE model_registry SET
                    status = 'evaluating',
                    test_mae = ?, test_rmse = ?, val_loss = ?,
                    onnx_parity_max_diff = ?, error_reduction_pct = ?,
                    epochs = ?, batch_size = ?, window_size = ?,
                    total_training_time_s = ?,
                    checkpoint_path = COALESCE(?, checkpoint_path),
                    onnx_path       = COALESCE(?, onnx_path),
                    norm_stats_path = COALESCE(?, norm_stats_path),
                    evaluated_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    metrics.get("candidate_test_mae"),
                    metrics.get("candidate_test_rmse"),
                    metrics.get("best_val_loss"),
                    metrics.get("onnx_parity_max_diff"),
                    metrics.get("error_reduction_pct"),
                    metrics.get("total_epochs"),
                    metrics.get("batch_size"),
                    metrics.get("window_size"),
                    metrics.get("total_time_s"),
                    metrics.get("checkpoint_path"),
                    metrics.get("onnx_path"),
                    metrics.get("norm_stats_path"),
                    now, now, model_id,
                ),
            )
        self.audit.log(
            actor_id=None,
            action="RECORD_MODEL_EVALUATION",
            resource_type="model_registry",
            resource_id=model_id,
            old_state="candidate_training",
            new_state="evaluating",
            metadata={"test_mae": metrics.get("candidate_test_mae")},
        )
        return self.get_model(model_id)

    def open_for_review(self, model_id: str, reviewer_id: str) -> ModelEntry:
        """Transition evaluating → review. Called when a team admin opens the entry."""
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model '{model_id}' not found")
        self._assert_transition(entry.status, "review")

        now = self._now()
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE model_registry SET status='review', reviewed_by=?, reviewed_at=?, updated_at=? WHERE id=?",
                (reviewer_id, now, now, model_id),
            )
        self.audit.log(
            actor_id=reviewer_id,
            action="OPEN_MODEL_REVIEW",
            resource_type="model_registry",
            resource_id=model_id,
            old_state="evaluating",
            new_state="review",
            metadata={},
        )
        return self.get_model(model_id)

    def approve_model(
        self,
        model_id: str,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> ModelEntry:
        """Transition review → approved → production_candidate (two-step in one call)."""
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model '{model_id}' not found")
        self._assert_transition(entry.status, "approved")
        self._assert_transition("approved", "production_candidate")  # pre-validate

        now = self._now()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE model_registry SET
                    status = 'production_candidate',
                    reviewed_by = ?, reviewed_at = ?,
                    review_notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (reviewer_id, now, notes, now, model_id),
            )
        self.audit.log(
            actor_id=reviewer_id,
            action="APPROVE_MODEL",
            resource_type="model_registry",
            resource_id=model_id,
            old_state="review",
            new_state="production_candidate",
            metadata={"notes": notes},
        )
        return self.get_model(model_id)

    def reject_model(
        self,
        model_id: str,
        reviewer_id: str,
        reason: str,
    ) -> ModelEntry:
        """Transition review → rejected. Reason is mandatory."""
        clean = reason.strip()
        if not clean:
            raise ValueError("Rejection reason cannot be empty")
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model '{model_id}' not found")
        self._assert_transition(entry.status, "rejected")

        now = self._now()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE model_registry SET
                    status = 'rejected', reviewed_by = ?,
                    reviewed_at = ?, rejection_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (reviewer_id, now, clean, now, model_id),
            )
        self.audit.log(
            actor_id=reviewer_id,
            action="REJECT_MODEL",
            resource_type="model_registry",
            resource_id=model_id,
            old_state="review",
            new_state="rejected",
            metadata={"reason": clean},
        )
        return self.get_model(model_id)

    def deploy(
        self,
        model_id: str,
        deployer_id: str,
        production_checkpoint_dest: str = "checkpoints/best_model.pt",
        production_onnx_dest: str = "simulator/model.onnx",
        production_stats_dest: str = "checkpoints/norm_stats.json",
        production_sim_stats_dest: str = "simulator/norm_stats.json",
    ) -> ModelEntry:
        """
        Transition production_candidate → production.
        Atomically copies checkpoint, ONNX, and norm_stats to production paths.
        Only the deployment step ever writes to production file locations.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError(f"Model '{model_id}' not found")
        self._assert_transition(entry.status, "production")

        # Validate source files exist before touching production
        src_pt    = Path(entry.checkpoint_path)   if entry.checkpoint_path   else None
        src_onnx  = Path(entry.onnx_path)         if entry.onnx_path         else None
        src_stats = Path(entry.norm_stats_path)   if entry.norm_stats_path   else None

        if src_pt and not src_pt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {src_pt}")
        if src_onnx and not src_onnx.exists():
            raise FileNotFoundError(f"ONNX model not found: {src_onnx}")
        if src_stats and not src_stats.exists():
            raise FileNotFoundError(f"Norm stats not found: {src_stats}")

        # Atomic copy: write to .tmp then rename (avoids partial writes)
        def atomic_copy(src: Path, dest_str: str):
            dest = Path(dest_str)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            shutil.copy2(src, tmp)
            tmp.replace(dest)

        if src_pt:
            atomic_copy(src_pt, production_checkpoint_dest)
        if src_onnx:
            atomic_copy(src_onnx, production_onnx_dest)
        if src_stats:
            atomic_copy(src_stats, production_stats_dest)
            atomic_copy(src_stats, production_sim_stats_dest)

        now = self._now()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE model_registry SET
                    status = 'production', deployed_by = ?,
                    deployed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (deployer_id, now, now, model_id),
            )
        self.audit.log(
            actor_id=deployer_id,
            action="DEPLOY_MODEL",
            resource_type="model_registry",
            resource_id=model_id,
            old_state="production_candidate",
            new_state="production",
            metadata={
                "checkpoint_dest": production_checkpoint_dest,
                "onnx_dest": production_onnx_dest,
            },
        )
        return self.get_model(model_id)

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Fetch single model entry with joined display names."""
        with get_db(self.db_path) as conn:
            row = conn.execute(
                f"{self._SELECT_FULL} WHERE m.id = ?", (model_id,)
            ).fetchone()
            return self._row_to_obj(row) if row else None

    def get_production_model(self) -> Optional[ModelEntry]:
        """Return the current production model (status='production'), most recent first."""
        with get_db(self.db_path) as conn:
            row = conn.execute(
                f"{self._SELECT_FULL} WHERE m.status = 'production' ORDER BY m.deployed_at DESC LIMIT 1"
            ).fetchone()
            return self._row_to_obj(row) if row else None

    def list_models(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[ModelEntry], int]:
        """Paginated list, optionally filtered by status."""
        where = "WHERE m.status = ?" if status else ""
        params: List[Any] = ([status] if status else [])

        with get_db(self.db_path) as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM model_registry m {where}", params
            ).fetchone()["n"]
            rows = conn.execute(
                f"{self._SELECT_FULL} {where} ORDER BY m.created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        return [self._row_to_obj(r) for r in rows], total

    def seed_production_model(
        self,
        name: str = "TCN v1.0 — Verified Production (4.2039 m/s)",
        test_mae: float = 4.2039,
        test_rmse: float = 6.0735,
        checkpoint_path: str = "checkpoints/best_model.pt",
        onnx_path: str = "simulator/model.onnx",
        norm_stats_path: str = "checkpoints/norm_stats.json",
    ) -> Optional[ModelEntry]:
        """
        Insert the verified production model as the baseline registry entry.
        Idempotent: only inserts if no production entry exists.
        """
        existing = self.get_production_model()
        if existing:
            return existing

        model_id = "mdl_production_baseline"
        now = self._now()
        with get_db(self.db_path) as conn:
            existing_row = conn.execute(
                "SELECT id FROM model_registry WHERE id = ?", (model_id,)
            ).fetchone()
            if existing_row:
                return self.get_model(model_id)

            conn.execute(
                """
                INSERT INTO model_registry (
                    id, name, architecture, status,
                    test_mae, test_rmse,
                    checkpoint_path, onnx_path, norm_stats_path,
                    deployed_at, created_at, updated_at
                ) VALUES (?, ?, 'TCNVelocityEstimator', 'production',
                          ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (model_id, name, test_mae, test_rmse,
                 checkpoint_path, onnx_path, norm_stats_path,
                 now, now, now),
            )
        return self.get_model(model_id)
