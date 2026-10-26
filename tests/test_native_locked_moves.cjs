"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { createEngineRequest } = require("../metamon/env/vectorized/native_request");
const { resolveNativeChoice } = require("../metamon/env/vectorized/native_choice");

const Choice = {
  pass: { type: "pass", data: 0 },
  move(data) { return { type: "move", data }; },
  switch(data) { return { type: "switch", data }; },
};
const moves = ["earthquake", "bodyslam", "rest", "blizzard"];
const gen = {
  species: { get: id => ({ name: id }) },
  moves: { get: id => ({ name: id, pp: 10, target: "normal" }) },
};
function fixture(allowed, volatiles={}) {
  const active = {
    position: 1, active: true, stored: {
      species: "tauros",
      stats: { hp: 353 }, moves: moves.map(id => ({ id })),
    },
    hp: 353, status: undefined,
    moves: moves.map(id => ({ id, pp: 10, disabled: false })),
    volatiles,
  };
  const bench = {
    position: 2, active: false, stored: {
      species: "snorlax", stats: { hp: 523 },
      moves: moves.map(id => ({ id })),
    },
    hp: 523, status: undefined,
  };
  const side = {
    pokemon: [active, bench],
    active,
  };
  const battle = {
    side() { return side; },
    choices() { return allowed; },
  };
  return { battle, side };
}
function request(allowed, volatiles={}) {
  return createEngineRequest(
    fixture(allowed, volatiles).battle, gen, "p1", "p1",
    {p1: "move", p2: "move"}, [{name:"Bull",species:"Tauros"}]
  );
}

test("locked recharge and other automatic modes expose one pseudo-action", () => {
  for (const name of ["recharging", "rage", "thrashing", "charging"]) {
    const out = request([Choice.move(1)], {[name]: true});
    assert.equal(out.active[0].trapped, true, name);
    assert.deepEqual(out.active[0].moves.map(m=>m.id),
      [name === "recharging" ? "recharge" : "fight"]);
    const command = name === "recharging" ? "move recharge" : "move fight";
    const side = fixture([Choice.move(1)], {[name]: true}).side;
    assert.deepEqual(
      resolveNativeChoice(command, [Choice.move(1)], side, Choice),
      Choice.move(1)
    );
  }
});

test("limited moves and PP reflect actual native legal choices", () => {
  const out = request([Choice.switch(2), Choice.move(1)]);
  assert.deepEqual(out.active[0].moves.map(m=>m.disabled),
    [false, true, true, true]);
  assert.equal(out.active[0].trapped, undefined);
});

test("native move zero exposes special Struggle rather than empty legal mask", () => {
  const out = request([Choice.switch(2), Choice.move(0)]);
  assert.deepEqual(out.active[0].moves.map(m=>m.id), ["struggle"]);
  const side = fixture([Choice.switch(2), Choice.move(0)]).side;
  assert.deepEqual(
    resolveNativeChoice("move struggle", [Choice.switch(2), Choice.move(0)], side, Choice),
    Choice.move(0)
  );
});

test("ordinary unforced moves preserve all normal legal choices", () => {
  const out = request(moves.map((m,i)=>Choice.move(i+1)));
  assert.deepEqual(out.active[0].moves.map(m=>m.id), moves);
  assert.deepEqual(out.active[0].moves.map(m=>m.disabled), [false,false,false,false]);
});

test("forced modes without exact sole move-1 native result fail", () => {
  assert.throws(() => request([Choice.move(2)], {recharging:true}), /forced choices/);
});
