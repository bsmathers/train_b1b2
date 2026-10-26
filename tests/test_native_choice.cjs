"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { resolveNativeChoice } = require("../metamon/env/vectorized/native_choice");

const Choice = {
  pass: { type: "pass", data: 0 },
  move(slot) { return { type: "move", data: slot }; },
  switch(slot) { return { type: "switch", data: slot }; },
};
const side = {
  slot(id) { return id; },
  active: { moves: [
    { id: "bodySlam" }, { id: "Earthquake" },
    { id: "Hyper Beam" }, { id: "Rest" },
  ] },
  pokemon: [
    { position: 1, stored: { species: "Tauros" } },
    { position: 2, stored: { species: "Snorlax" } },
    { position: 3, stored: { species: "Alakazam" } },
    { position: 4, stored: { species: "Chansey" } },
    { position: 5, stored: { species: "Exeggutor" } },
    { position: 6, stored: { species: "Starmie" } },
  ],
};

test("all four numeric move slots round-trip to explicit legal Zig choices", () => {
  for (let slot = 1; slot <= 4; slot++) {
    const legal = [Choice.move(slot)];
    assert.deepEqual(resolveNativeChoice(`move ${slot}`, legal, side, Choice), legal[0]);
  }
});

test("all six original party slots map to the current legal Zig slot", () => {
  const shifted = { ...side, slot(id) { return id === 1 ? 2 : id === 2 ? 1 : id; } };
  for (let original = 1; original <= 6; original++) {
    const dynamic = original === 1 ? 2 : original === 2 ? 1 : original;
    // Choice.switch(1) is never native-legal while that mon is active.
    if (dynamic === 1) continue;
    const legal = [Choice.switch(dynamic)];
    assert.deepEqual(resolveNativeChoice(`switch ${original}`, legal, shifted, Choice), legal[0]);
  }
  const legal = [Choice.switch(2)];
  assert.deepEqual(resolveNativeChoice("switch 1", legal, shifted, Choice), legal[0]);
});

test("Metamon move and switch names preserve intended choices", () => {
  for (let slot = 1; slot <= 4; slot++) {
    const legal = [Choice.move(slot)];
    assert.deepEqual(
      resolveNativeChoice(`move ${side.active.moves[slot - 1].id}`, legal, side, Choice),
      legal[0]
    );
  }
  for (let slot = 2; slot <= 6; slot++) {
    const legal = [Choice.switch(slot)];
    assert.deepEqual(
      resolveNativeChoice(`switch ${side.pokemon[slot - 1].stored.species}`, legal, side, Choice),
      legal[0]
    );
  }
  const shifted = { ...side, slot(id) { return id === 1 ? 2 : id === 2 ? 1 : id; } };
  assert.deepEqual(
    resolveNativeChoice("switch Tauros", [Choice.switch(2)], shifted, Choice),
    Choice.switch(2)
  );
  assert.deepEqual(
    resolveNativeChoice("move Hyper Beam", [Choice.move(3)], side, Choice),
    Choice.move(3)
  );
});

test("illegal/ambiguous moves and switches cannot silently become first legal action", () => {
  const legal = [Choice.move(1), Choice.switch(2)];
  for (const cmd of [
    "move 4", "move Hypnosis", "move 0", "move 5",
    "switch 1", "switch 7", "switch MissingNo", "switch 0",
    "switch", "move", "forfeit", "pass", "",
  ]) {
    assert.throws(() => resolveNativeChoice(cmd, legal, side, Choice), undefined, cmd);
  }
});

test("forced move aliases use exactly one legal move or fail closed", () => {
  for (const alias of ["recharge", "struggle", "fight"]) {
    assert.deepEqual(
      resolveNativeChoice(`move ${alias}`, [Choice.move(3)], side, Choice),
      Choice.move(3)
    );
    assert.throws(() =>
      resolveNativeChoice(`move ${alias}`, [Choice.move(1), Choice.move(2)], side, Choice)
    );
  }
});

test("explicit default is the only permitted auto-choice", () => {
  const legal = [Choice.switch(1), Choice.switch(2)];
  assert.deepEqual(resolveNativeChoice("default", legal, side, Choice), legal[0]);
  assert.deepEqual(resolveNativeChoice("pass", [], side, Choice), Choice.pass);
  assert.throws(() => resolveNativeChoice("switch 1", [], side, Choice));
});

test("duplicate species name is ambiguous and fails closed", () => {
  const duplicate = { ...side, pokemon: [
    { position: 1, stored: { species: "Tauros" } }, { position: 2, stored: { species: "Tauros" } },
  ] };
  assert.throws(() => resolveNativeChoice("switch Tauros", [
    Choice.switch(1), Choice.switch(2),
  ], duplicate, Choice));
});
