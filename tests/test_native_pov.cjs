"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { originalPartyOrder, privateIdent } = require("../metamon/env/vectorized/native_pov");

test("private requests remain in original party order after Zig dynamic switch", () => {
  const party = [
    { position: 2, stored: { species: "snorlax" } },
    { position: 1, stored: { species: "tauros" } },
    { position: 3, stored: { species: "chansey" } },
  ];
  const result = originalPartyOrder({ pokemon: party });
  assert.deepEqual(result.map(p => p.position), [1, 2, 3]);
});

test("private request identifiers preserve per-side original nicknames", () => {
  const mon = { position: 1, stored: { species: "tauros" } };
  const originalTeam = [
    { name: "Angry Bull", species: "Tauros" },
    { name: "Snorlax", species: "Snorlax" },
  ];
  assert.equal(privateIdent("p1", mon, originalTeam, "Tauros"), "p1: Angry Bull");
  assert.equal(privateIdent("p2", mon, originalTeam, "Tauros"), "p2: Angry Bull");
  assert.equal(privateIdent("p1", mon, [], "Tauros"), "p1: Tauros");
});
