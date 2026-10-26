"use strict";

/**
 * The Zig engine iterates party members in dynamically rearranged active-first
 * order, while a Showdown private |request| must expose the original stable
 * team order. Pokemon.position is the fixed original one-based party ID.
 */
function originalPartyOrder(side) {
  return Array.from(side.pokemon).sort((a, b) => a.position - b.position);
}

/**
 * Preserve each original Showdown nickname (used by public |switch| logs),
 * rather than using the species name for private request identities.
 */
function privateIdent(sideId, mon, originalTeam, speciesName) {
  const set = originalTeam?.[mon.position - 1];
  const nickname = set?.name || speciesName;
  return `${sideId}: ${nickname}`;
}

module.exports = { originalPartyOrder, privateIdent };
