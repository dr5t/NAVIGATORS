"""
Navigators IDR - Navigation Engine State Machine
Manages formal navigation system states and transitions driven strictly
by the underlying GNSS/INS sensor fusion engine.

States:
  - STANDBY: Navigation engine initialized, awaiting initial GNSS fix or start.
  - NORMAL: Continuous GNSS + INS sensor fusion active with high accuracy.
  - GNSS_DEGRADED: High DOP or reduced satellite visibility; fallback filters active.
  - DR: GNSS unavailable/denied; TCN + EKF Dead Reckoning active.
  - REACQUIRING: GNSS restored; smoothing trajectory re-convergence.
  - STOPPED: Navigation session terminated normally.
  - ERROR: Critical sensor or system fault.

Architectural Contract:
  The UI client consumes state updates emitted by this state machine.
  The UI NEVER independently decides to transition into DR or change states.
"""

from enum import Enum
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone


class NavState(str, Enum):
    STANDBY = "STANDBY"
    NORMAL = "NORMAL"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    DR = "DR"
    REACQUIRING = "REACQUIRING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


VALID_NAV_TRANSITIONS: Dict[NavState, set[NavState]] = {
    NavState.STANDBY: {NavState.NORMAL, NavState.GNSS_DEGRADED, NavState.DR, NavState.STOPPED, NavState.ERROR},
    NavState.NORMAL: {NavState.GNSS_DEGRADED, NavState.DR, NavState.STOPPED, NavState.ERROR},
    NavState.GNSS_DEGRADED: {NavState.NORMAL, NavState.DR, NavState.STOPPED, NavState.ERROR},
    NavState.DR: {NavState.REACQUIRING, NavState.STOPPED, NavState.ERROR},
    NavState.REACQUIRING: {NavState.NORMAL, NavState.GNSS_DEGRADED, NavState.DR, NavState.STOPPED, NavState.ERROR},
    NavState.STOPPED: {NavState.STANDBY, NavState.NORMAL},
    NavState.ERROR: {NavState.STANDBY, NavState.STOPPED},
}


class NavigationStateEngine:
    """
    State machine owned by the backend navigation engine.
    Processes sensor inputs (GNSS fix status, HDOP, IMU health)
    and emits authoritative navigation states.
    """

    def __init__(self, initial_state: NavState = NavState.STANDBY):
        self._current_state = initial_state
        self._last_state_change = datetime.now(timezone.utc).isoformat()
        self._transition_history: list[dict[str, Any]] = []

    @property
    def current_state(self) -> NavState:
        return self._current_state

    def can_transition(self, target_state: NavState) -> bool:
        """Check if transitioning from current_state to target_state is valid."""
        allowed = VALID_NAV_TRANSITIONS.get(self._current_state, set())
        return target_state in allowed

    def transition(self, target_state: NavState, reason: str = "") -> NavState:
        """
        Execute an authoritative state transition.
        Raises ValueError if transition is invalid.
        """
        if target_state == self._current_state:
            return self._current_state

        if not self.can_transition(target_state):
            raise ValueError(
                f"Invalid navigation state transition from '{self._current_state.value}' to '{target_state.value}'."
            )

        now_str = datetime.now(timezone.utc).isoformat()
        old_state = self._current_state
        self._current_state = target_state
        self._last_state_change = now_str

        self._transition_history.append({
            "from_state": old_state.value,
            "to_state": target_state.value,
            "reason": reason,
            "timestamp": now_str,
        })

        return self._current_state

    def process_sensor_health(
        self,
        gnss_available: bool,
        hdop: float = 1.0,
        imu_healthy: bool = True,
        reacquiring_progress: float = 1.0,
    ) -> NavState:
        """
        Evaluates sensor status telemetry and drives automatic state transitions.
        """
        if not imu_healthy:
            return self.transition(NavState.ERROR, reason="IMU sensor failure")

        if self._current_state == NavState.STANDBY:
            if gnss_available and hdop <= 2.5:
                return self.transition(NavState.NORMAL, reason="GNSS fix acquired")
            elif gnss_available and hdop > 2.5:
                return self.transition(NavState.GNSS_DEGRADED, reason="GNSS fix acquired with high HDOP")
            else:
                return self.transition(NavState.DR, reason="Navigation started without GNSS fix")

        if self._current_state == NavState.NORMAL:
            if not gnss_available:
                return self.transition(NavState.DR, reason="GNSS signal lost")
            elif hdop > 3.0:
                return self.transition(NavState.GNSS_DEGRADED, reason="GNSS precision degraded")

        if self._current_state == NavState.GNSS_DEGRADED:
            if not gnss_available:
                return self.transition(NavState.DR, reason="GNSS signal lost")
            elif hdop <= 2.0:
                return self.transition(NavState.NORMAL, reason="GNSS precision restored")

        if self._current_state == NavState.DR:
            if gnss_available:
                return self.transition(NavState.REACQUIRING, reason="GNSS signal restored, reacquiring position")

        if self._current_state == NavState.REACQUIRING:
            if not gnss_available:
                return self.transition(NavState.DR, reason="GNSS lost during re-acquisition")
            elif reacquiring_progress >= 0.95:
                if hdop <= 2.5:
                    return self.transition(NavState.NORMAL, reason="Re-acquisition converged to NORMAL")
                else:
                    return self.transition(NavState.GNSS_DEGRADED, reason="Re-acquisition converged to DEGRADED")

        return self._current_state

    def get_ui_notification(self) -> Optional[dict[str, str]]:
        """
        Formats user-facing minimalist toast notifications when GNSS/DR states change.
        Strict UI rules:
          - No popup gradient
          - No giant modal
          - No unnecessary animation
          - No interruption to navigation
        """
        if self._current_state == NavState.DR:
            return {
                "title": "Dead Reckoning Active",
                "body": "GNSS signal lost. Navigation continues.",
                "type": "info",
                "dismiss_after_ms": 3000,
            }
        elif self._current_state in (NavState.NORMAL, NavState.REACQUIRING) and len(self._transition_history) > 0:
            last = self._transition_history[-1]
            if last["from_state"] in (NavState.DR.value, NavState.REACQUIRING.value):
                return {
                    "title": "GNSS Restored",
                    "body": "Position correction resumed.",
                    "type": "success",
                    "dismiss_after_ms": 3000,
                }
        return None

    def get_status_payload(self) -> dict[str, Any]:
        """Format current navigation engine state payload for UI consumption."""
        return {
            "state": self._current_state.value,
            "last_change": self._last_state_change,
            "history_count": len(self._transition_history),
            "notification": self.get_ui_notification(),
        }
