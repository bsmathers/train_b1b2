"use strict";

// Native CPU integration smoke: exercises the same native addon, package version,
// team structures, and Choice API used by battle_host_engine.js. No model weights.
// Run after ./tools/setup_pkmn_engine.sh, from any working directory.
const assert = require("node:assert/strict");
const path = require("node:path");
const host = path.resolve(__dirname, "../metamon/env/vectorized");
const engine = require(path.join(host, "node_modules/@pkmn/engine"));
const data = require(path.join(host, "node_modules/@pkmn/data"));
const sim = require(path.join(host, "node_modules/@pkmn/sim"));
const { resolveNativeChoice } = require(path.join(host, "native_choice.js"));

const roster = [
  ["Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]],
  ["Snorlax", ["Body Slam", "Earthquake", "Rest", "Self-Destruct"]],
  ["Chansey", ["Soft-Boiled", "Thunder Wave", "Ice Beam", "Seismic Toss"]],
  ["Exeggutor", ["Sleep Powder", "Psychic", "Stun Spore", "Explosion"]],
  ["Alakazam", ["Psychic", "Recover", "Thunder Wave", "Seismic Toss"]],
  ["Starmie", ["Surf", "Recover", "Thunderbolt", "Blizzard"]],
];
const sets = roster.map(([species, moves]) => ({
  name: species,
  species,
  moves,
  level: 100,
  evs: {hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252},
  ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
}));
(async () => {
  const addon = path.join(host, "pkmn-showdown.node");
  const native = require(addon);
  assert.equal(native.engine.options.showdown, true);
  assert.equal(native.engine.options.log, true);
  await engine.initialize(true, addon);
  const gen1 = new data.Generations(sim.Dex).get(1);
  const packed = sim.Teams.pack(sets);
  const decoded = sim.Teams.unpack(packed);
  const battle = engine.Battle.create(gen1, {
    p1: {name: "p1", team: decoded},
    p2: {name: "p2", team: decoded},
    seed: [1, 2, 3, 4],
    showdown: true, log: true,
  });
  let res = battle.update(undefined, undefined);
  const choice = (side, command) => resolveNativeChoice(
    command, battle.choices(side, res), battle.side(side), engine.Choice
  );
  // Exhaustively round-trip every *actually legal* Zig choice through the
  // Showdown interface, on both physical player sides and after switches.
  function assertChoiceParity(sideId) {
    const side = battle.side(sideId);
    const legal = battle.choices(sideId, res);
    const mons = Array.from(side.pokemon);
    const moves = Array.from(side.active?.moves || []);
    for (const native of legal) {
      let command;
      if (native.type === "switch") {
        const mon = mons[native.data - 1]; // current dynamic slot
        assert.ok(mon, `Missing dynamic slot ${native.data}`);
        command = `switch ${mon.position}`; // stable Showdown party slot
      } else if (native.type === "move") {
        if (native.data < 1 || native.data > moves.length) continue;
        command = `move ${moves[native.data - 1].id}`;
      } else {
        continue; // native pass is a wait, not a model action
      }
      assert.deepEqual(choice(sideId, command), native,
        `Mismatch side=${sideId} choice=${command}`);
    }
  }
  assertChoiceParity("p1");
  assertChoiceParity("p2");

  // Verify a *real* native engine accepts both the party slot and move
  // conventions. Switch out of the lead, then switch back into party slot 1.
  const legal1 = battle.choices("p1", res);
  assert.ok(legal1.length > 0);
  const switch2 = choice("p1", "switch 2");
  const other = choice("p2", "move 1");
  assert.equal(switch2.type, engine.Choice.switch(2).type);
  assert.equal(switch2.data, engine.Choice.switch(2).data);
  res = battle.update(switch2, other);
  assertChoiceParity("p1");
  assertChoiceParity("p2");

  // Zig keeps the active mon in CURRENT slot 1. Its Side.slot(originalId)
  // translates a stable Showdown party position to the native 2..6 slot.
  const sideAfter = battle.side("p1");
  const byCurrentSlot = Array.from(sideAfter.pokemon);
  const byOriginalPosition = byCurrentSlot.slice().sort((a, b) => a.position - b.position);
  assert.equal(byOriginalPosition[0].stored.species, "tauros");
  assert.notEqual(byCurrentSlot[0].position, byOriginalPosition[0].position,
    "A switch must reorder dynamic slots without changing original identities");
  const dynamicLead = sideAfter.slot(1);
  assert.ok(dynamicLead >= 2 && dynamicLead <= 6,
    "Original party slot 1 should now be a switchable bench Pokemon");
  const wanted = engine.Choice.switch(dynamicLead);
  const legalAfter = battle.choices("p1", res);
  assert.ok(legalAfter.some(v => v.type === wanted.type && v.data === wanted.data),
    "Native battle must allow switching back to the original lead via a dynamic slot");
  // Request is fixed Showdown position 1; resolver must emit the dynamic choice.
  const switch1 = choice("p1", "switch 1");
  assert.deepEqual(switch1, wanted);
  const c2 = choice("p2", "move 1");
  res = battle.update(switch1, c2);
  assertChoiceParity("p1");
  assertChoiceParity("p2");

  let steps = 2;
  for (; steps < 18 && !res.type; steps++) {
    const p1legal = battle.choices("p1", res);
    const p2legal = battle.choices("p2", res);
    const c1 = p1legal.length ? p1legal[0] : engine.Choice.pass;
    const c2 = p2legal.length ? p2legal[0] : engine.Choice.pass;
    res = battle.update(c1, c2);
  }
  assert.ok(steps >= 2);
  console.log(`Native Zig integration smoke passed (${steps} steps, switch1 and switch2)`);
})().catch(e => {
  console.error(e);
  process.exitCode = 1;
});
