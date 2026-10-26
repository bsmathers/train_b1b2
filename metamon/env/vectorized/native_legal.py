"""Reduce Metamon's action mask for native Zig automatic move turns.

The native host uses Metamon's existing request-parser aliases:
  recharge: automatic Hyper Beam recharge
  fight: automatic Rage, Thrash, or two-turn charge continuation
  struggle: the sole allowed fallback after exhausting available moves

These aliases are *not* arbitrary trainer choices. Map automatic decisions to
canonical action 0; only the Struggle case may still allow regular switches.
"""


def canonical_legal_indices(regular_legal, request):
    active = (request or {}).get("active") or []
    moves = active[0].get("moves") or [] if active else []
    if len(moves) != 1:
        return list(regular_legal)
    pseudo = str(moves[0].get("id", "")).lower()
    if pseudo in ("recharge", "fight"):
        return [0]
    if pseudo == "struggle":
        return sorted({0, *(int(idx) for idx in regular_legal if 4 <= int(idx) <= 8)})
    return list(regular_legal)
