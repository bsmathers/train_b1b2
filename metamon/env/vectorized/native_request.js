"use strict";

const { originalPartyOrder, privateIdent } = require("./native_pov");

/** Construct a POV-specific Showdown private request from the real Zig state. */
function createEngineRequest(engineBattle, gen, sideId, playerName, res, originalTeam) {
  const isP1 = sideId === "p1";
  const playerChoiceType = isP1 ? res.p1 : res.p2;
  const side = engineBattle.side(sideId);
  const pokemonList = [];

  // Zig lists Pokemon in CURRENT switch order (the active mon is always
  // dynamic slot 1). Showdown private |request| party entries MUST remain in
  // ORIGINAL order throughout the entire battle. Pokemon.position is the
  // stable original position; side.slot(position) maps it back for choices.
  for (const mon of originalPartyOrder(side)) {
    const speciesData = gen.species.get(mon.stored.species);
    const speciesName = speciesData ? speciesData.name : mon.stored.species;
    const maxHp = mon.stored.stats.hp;
    const condition =
      mon.hp === 0
        ? "0 fnt"
        : mon.status
        ? `${mon.hp}/${maxHp} ${mon.status}`
        : `${mon.hp}/${maxHp}`;
    const moveIds = Array.from(mon.stored.moves).map((m) => m.id);

    pokemonList.push({
      ident: privateIdent(sideId, mon, originalTeam, speciesName),
      details: speciesName,
      condition: condition,
      active: mon.active,
      stats: mon.stored.stats,
      moves: moveIds,
      baseAbility: "",
      item: "",
      pokeball: "pokeball",
    });
  }

  const sideObj = { name: playerName, id: sideId, pokemon: pokemonList };

  if (playerChoiceType === "pass" || !playerChoiceType) {
    if (res.type) return null;
    return { wait: true, side: sideObj };
  }

  if (playerChoiceType === "switch") {
    return { forceSwitch: [true], side: sideObj };
  }

  // IMPORTANT: The policy's legality mask is derived from this synthetic
  // private Showdown request. Reflect *native Zig* choices, not simply the
  // Pokemon's stored moves. Zig disallows most/all moves during recharge,
  // Thrash/Rage/charging, Bide, Bind/Wrap, Disable, and depleted PP.
  //
  // In Showdown-compatibility mode an automatic forced action is exposed as
  // the single native Choice.move(1). A regular move with the same index is
  // NOT necessarily legal from the player's point of view.
  const legal = engineBattle.choices(sideId, res);
  const legalMoveSlots = new Set(
    legal.filter(c => c.type === "move").map(c => c.data)
  );
  const volatiles = side.active.volatiles;
  const isForced = !!(volatiles.recharging || volatiles.rage ||
    volatiles.thrashing || volatiles.charging);
  if (isForced) {
    // 'recharge' and 'fight' are supported pseudo moves by Metamon's
    // request parser and UniversalAction.action_idx_to_BattleOrder. They
    // preserve the model's complete underlying moveset and make all of
    // its move action indices resolve to the sole native forced move.
    if (legal.length !== 1 || legal[0].type !== "move" || legal[0].data !== 1) {
      throw new Error(
        `Unexpected native forced choices for ${sideId}: ${JSON.stringify(legal)}`
      );
    }
    const alias = volatiles.recharging ? "recharge" : "fight";
    return {
      active: [{ trapped: true, moves: [{
        move: alias === "recharge" ? "Recharge" : "Fight",
        id: alias,
        pp: 1,
        maxpp: 1,
        target: "normal",
        disabled: false,
      }] }],
      side: sideObj,
    };
  }

  // Struggle is a native move-0 special case. The stored moves are all
  // unusable; advertising them as selectable produces an all-illegal actor
  // mask. Metamon already understands the synthetic 'struggle' move.
  if (legalMoveSlots.has(0)) {
    if (legalMoveSlots.size !== 1) {
      throw new Error("Native Struggle choice must be the sole move option");
    }
    return {
      active: [{ moves: [{
        move: "Struggle",
        id: "struggle",
        pp: 1,
        maxpp: 1,
        target: "normal",
        disabled: false,
      }] }],
      side: sideObj,
    };
  }

  const activeMoves = [];
  for (const [i, m] of Array.from(side.active.moves).entries()) {
    const moveData = gen.moves.get(m.id);
    const moveName = moveData ? moveData.name : m.id;
    activeMoves.push({
      move: moveName,
      id: m.id,
      pp: m.pp,
      maxpp: moveData ? Math.floor((moveData.pp * 8) / 5) : m.pp,
      target: moveData?.target || "normal",
      // Never advertise a move that Zig would reject. This matters for
      // Bide / Wrap / Bind locks, Disable and exhausted PP.
      disabled: !legalMoveSlots.has(i + 1),
    });
  }
  // Zig has a dedicated Choice.move(0) for Struggle in non-Showdown mode.
  // In showdown=true mode Struggle is represented as move 1. Do not
  // arbitrarily substitute moves if the actual engine offers no choice.
  return { active: [{ moves: activeMoves }], side: sideObj };
}


module.exports = { createEngineRequest };
