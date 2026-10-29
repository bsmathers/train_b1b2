"""Reduce Metamon's action mask for native Zig automatic move turns.

The native host uses Metamon's existing request-parser aliases:
  recharge: automatic Hyper Beam recharge
  fight: automatic Rage, Thrash, or two-turn charge continuation
  struggle: the sole allowed fallback after exhausting available moves

These aliases are *not* arbitrary trainer choices. Map automatic decisions to
canonical action 0; only the Struggle case may still allow regular switches.
"""

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


def default_action_legal_indices(state, battle, request):
    """Fast equivalent of ``UniversalAction.definitely_valid_actions``.

    The generic implementation builds a ``BattleOrder`` for every candidate
    action. Native collection uses the fixed 13-way DefaultActionSpace, so the
    same membership checks can be performed once per decision without creating
    temporary orders. ``METAMON_NATIVE_VERIFY_FAST_LEGAL=1`` makes the caller
    compare this result with the generic path on every state.
    """
    valid_moves = {move.id for move in battle.available_moves}
    active = (request or {}).get("active") or []
    request_moves = active[0].get("moves") or [] if active else []
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
    elif not battle.force_switch:
        move_options = consistent_move_order(
            list(battle.active_pokemon.moves.values())
        )
        for index, move in enumerate(move_options[:4]):
            if move.id in valid_moves:
                legal.append(index)
                if state.can_tera:
                    legal.append(index + 9)

    valid_switches = {pokemon.name for pokemon in battle.available_switches}
    if not battle.reviving:
        switch_pool = [
            pokemon
            for pokemon in battle.team.values()
            if not pokemon.fainted and not pokemon.active
        ]
    else:
        switch_pool = [
            pokemon
            for pokemon in battle.team.values()
            if pokemon.fainted and not pokemon.active
        ]
    for index, pokemon in enumerate(consistent_pokemon_order(switch_pool)[:5]):
        if pokemon.name in valid_switches:
            legal.append(index + 4)
    return sorted(legal)
