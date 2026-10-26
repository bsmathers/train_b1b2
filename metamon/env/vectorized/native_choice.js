"use strict";

/**
 * Translate an explicit Showdown choice into an @pkmn/engine Choice.
 *
 * Action indices are decoded in Python using Metamon's canonical
 * UniversalAction.action_idx_to_BattleOrder(), then converted to a Showdown
 * choice ("move <name/slot>" or "switch <party slot>").
 *
 * Numeric switches are 1-based **party** slots 1..6, including the lead
 * after it has switched out. Numeric moves are 1-based active move slots 1..4.
 * Every explicit choice must match an entry in battle.choices(); a mismatch is
 * an adapter error and MUST NOT silently execute validChoices[0].
 *
 * "default" is the sole intentional auto-choice and uses the first legal
 * native choice. Forced aliases (recharge/struggle/fight) may similarly resolve
 * to the sole legal move, but are rejected if ambiguous.
 */
function normalizedId(s) {
  return String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function resolveNativeChoice(choiceStr, validChoices, side, Choice) {
  if (!Array.isArray(validChoices)) {
    throw new TypeError("native choices must be an array");
  }
  const command = String(choiceStr || "").trim();
  if (validChoices.length === 0) {
    if (command === "pass" || command === "default" || command === "") {
      return Choice.pass;
    }
    throw new Error(`No legal native choices for explicit action "${command}"`);
  }
  if (command === "default") return validChoices[0];
  if (command === "pass" || command === "") {
    throw new Error(`Unexpected "${command || "<empty>"}" with native legal choices`);
  }

  const parts = /^(move|switch)\s+(.+)$/i.exec(command);
  if (!parts) throw new Error(`Unsupported explicit Showdown action "${command}"`);
  const verb = parts[1].toLowerCase();
  const target = parts[2].trim().toLowerCase();

  let slot;
  if (verb === "move") {
    const moves = Array.from(side.active?.moves || []);
    if (/^[1-4]$/.test(target)) {
      slot = Number(target);
    } else {
      const matches = moves.flatMap((move, i) =>
        normalizedId(move.id) === normalizedId(target) ? [i + 1] : []
      );
      if (matches.length === 1) {
        slot = matches[0];
      } else if (["recharge", "struggle", "fight"].includes(normalizedId(target))) {
        const legalMoves = validChoices.filter(v => v.type === Choice.move(1).type);
        if (legalMoves.length === 1) return legalMoves[0];
      }
    }
  } else {
    const mons = Array.from(side.pokemon || []);
    // Both "switch 1" and "switch 6" are valid 1-based party slots.
    if (/^[1-6]$/.test(target)) {
      slot = Number(target);
    } else {
      const matches = mons.flatMap((mon, i) =>
        normalizedId(mon.stored?.species) === normalizedId(target) ? [i + 1] : []
      );
      if (matches.length === 1) slot = matches[0];
    }
  }

  if (!slot) {
    throw new Error(`Cannot uniquely resolve native ${verb} target "${target}"`);
  }
  const proposed = verb === "move" ? Choice.move(slot) : Choice.switch(slot);
  const matched = validChoices.find(
    v => v.type === proposed.type && v.data === proposed.data
  );
  if (!matched) {
    throw new Error(
      `Native engine rejected explicit action "${command}" (resolved ${verb} slot ${slot}); ` +
      `legal choices: ${validChoices.map(v => `${v.type}:${v.data}`).join(", ")}`
    );
  }
  return matched;
}

module.exports = { normalizedId, resolveNativeChoice };
