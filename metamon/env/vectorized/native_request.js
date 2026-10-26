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

  const activeMoves = [];
  for (const m of Array.from(side.active.moves)) {
    const moveData = gen.moves.get(m.id);
    const moveName = moveData ? moveData.name : m.id;
    activeMoves.push({
      move: moveName,
      id: m.id,
      pp: m.pp,
      maxpp: Math.floor((moveData.pp * 8) / 5),
      target: moveData.target,
      disabled: !!m.disabled,
    });
  }

  return { active: [{ moves: activeMoves }], side: sideObj };
}


module.exports = { createEngineRequest };
