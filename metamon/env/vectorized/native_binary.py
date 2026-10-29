"""Binary Gen 1 state transport for high-throughput self-play collection.

This module deliberately stops below Metamon observation construction.  It
captures the Zig engine's authoritative 384-byte battle image, legal choices,
actions, and terminal result without routing through Showdown text.  That is a
complete lossless simulator trajectory and the stable input for the native
observation decoder.  The existing text adapter remains the correctness oracle
while that decoder is differential-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


GEN1_BATTLE_BYTES = 384
RESULT_NONE = 0
RESULT_WIN = 1
RESULT_LOSE = 2
RESULT_TIE = 3
RESULT_ERROR = 4


def result_type(encoded: int) -> int:
    """Return the low-nibble terminal type from a native encoded Result."""
    return int(encoded) & 0x0F


@dataclass(slots=True)
class NativeBinaryTrajectory:
    """Lossless native simulator trajectory, before policy featurization."""

    p1: dict[str, Any]
    p2: dict[str, Any]
    states: list[bytes] = field(default_factory=list)
    actions: list[tuple[int, int]] = field(default_factory=list)
    result: int = RESULT_NONE


class NativeBinaryLane:
    """Lane handler for ``MSG_NATIVE_STATE`` frames."""

    def __init__(self, lane_id: int):
        self.lane_id = int(lane_id)
        self.ready = False
        self.state = b""
        self.result = RESULT_NONE
        self.p1_choices = b""
        self.p2_choices = b""
        self.trajectory: Optional[NativeBinaryTrajectory] = None

    def begin(self, p1: dict[str, Any], p2: dict[str, Any]) -> None:
        self.ready = False
        self.state = b""
        self.result = RESULT_NONE
        self.p1_choices = b""
        self.p2_choices = b""
        self.trajectory = NativeBinaryTrajectory(p1=dict(p1), p2=dict(p2))

    def handle_chunk(self, stream: str, data: str) -> None:
        raise RuntimeError(
            f"binary lane {self.lane_id} unexpectedly received text stream {stream}"
        )

    def handle_native_state(
        self,
        battle: bytes,
        result: int,
        p1_choices: bytes,
        p2_choices: bytes,
    ) -> None:
        if len(battle) != GEN1_BATTLE_BYTES:
            raise RuntimeError(
                f"lane {self.lane_id}: expected {GEN1_BATTLE_BYTES} native bytes, "
                f"got {len(battle)}"
            )
        if self.trajectory is None:
            raise RuntimeError(f"lane {self.lane_id}: state arrived before begin")
        self.state = battle
        self.result = result_type(result)
        self.p1_choices = p1_choices
        self.p2_choices = p2_choices
        self.trajectory.states.append(battle)
        self.trajectory.result = self.result
        self.ready = True

    @property
    def terminal(self) -> bool:
        return self.result != RESULT_NONE

    def commit(self, p1_choice: int, p2_choice: int) -> None:
        if self.trajectory is None:
            raise RuntimeError(f"lane {self.lane_id}: action committed before begin")
        if not self.ready or self.terminal:
            raise RuntimeError(f"lane {self.lane_id}: action committed when not actionable")
        if p1_choice not in self.p1_choices or p2_choice not in self.p2_choices:
            raise RuntimeError(
                f"lane {self.lane_id}: illegal encoded choices "
                f"{p1_choice}/{p2_choice}"
            )
        self.trajectory.actions.append((int(p1_choice), int(p2_choice)))
        self.ready = False

