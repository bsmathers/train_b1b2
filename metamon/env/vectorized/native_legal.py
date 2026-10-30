"""Reduce Metamon's action mask for native Zig automatic move turns.

The native host uses Metamon's existing request-parser aliases:
  recharge: automatic Hyper Beam recharge
  fight: automatic Rage, Thrash, or two-turn charge continuation
  struggle: the sole allowed fallback after exhausting available moves

These aliases are *not* arbitrary trainer choices. Map automatic decisions to
canonical action 0; only the Struggle case may still allow regular switches.
"""

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.interface import consistent_move_order, consistent_pokemon_order


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


def default_action_legal_indices(state, request):
    """Fast equivalent of ``UniversalAction.definitely_valid_actions``.

    The native request already contains the enabled move ids, switch slots,
    trap state, and force-switch state, while ``state`` defines the model's
    stable sorted action ordering. Compute their intersection directly instead
    of materializing a complete poke-env Pokemon/team graph. The optional
    verifier in the caller still compares this result to the generic path.
    """
    request = request or {}
    active = request.get("active") or []
    request_moves = active[0].get("moves") or [] if active else []
    valid_moves = {
        move_name(move.get("id", ""))
        for move in request_moves
        if not move.get("disabled", False)
    }
    pseudo = (
        str(request_moves[0].get("id", "")).lower()
        if len(request_moves) == 1
        else ""
    )
    if pseudo in ("recharge", "fight"):
        return [0]

    legal = []
    if pseudo == "struggle" or valid_moves == {"struggle"}:
        legal.append(0)
    elif not state.forced_switch:
        move_options = consistent_move_order(state.player_active_pokemon.moves)
        for index, move in enumerate(move_options[:4]):
            if move.name in valid_moves:
                legal.append(index)
                if state.can_tera:
                    legal.append(index + 9)

    trapped = bool(active and active[0].get("trapped", False))
    valid_switches = set()
    if not trapped:
        for pokemon in request.get("side", {}).get("pokemon", ()):
            condition = str(pokemon.get("condition", ""))
            if pokemon.get("active", False) or "fnt" in condition.split():
                continue
            species = str(pokemon.get("details", "")).split(",", 1)[0]
            valid_switches.add(pokemon_name(species))
    switch_pool = consistent_pokemon_order(state.available_switches)
    for index, pokemon in enumerate(switch_pool[:5]):
        if pokemon.name in valid_switches:
            legal.append(index + 4)
    return sorted(legal)
