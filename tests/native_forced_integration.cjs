"use strict";
// Exercise forced-turn requests against real pinned @pkmn/engine Zig code.
// Matches observed crash: an ordinary named move (Blizzard) was offered while
// native choices contained only Choice.move(1) due to Hyper Beam / Rage lock.
const assert = require("node:assert/strict");
const path = require("node:path");
const host = path.resolve(__dirname, "../metamon/env/vectorized");
const engine = require(path.join(host, "node_modules/@pkmn/engine"));
const sim = require(path.join(host, "node_modules/@pkmn/sim"));
const data = require(path.join(host, "node_modules/@pkmn/data"));
const { createEngineRequest } = require(path.join(host, "native_request.js"));
const { resolveNativeChoice } = require(path.join(host, "native_choice.js"));

const evs = {hp:252,atk:252,def:252,spa:252,spd:252,spe:252};
const ivs = {hp:31,atk:31,def:31,spa:31,spd:31,spe:31};
const set = (species,moves) => ({name:species,species,moves,level:100,evs,ivs});
const filler = [
 set("Chansey",["Soft-Boiled","Thunder Wave","Ice Beam","Seismic Toss"]),
 set("Exeggutor",["Sleep Powder","Psychic","Stun Spore","Explosion"]),
 set("Alakazam",["Psychic","Recover","Thunder Wave","Seismic Toss"]),
 set("Starmie",["Surf","Recover","Thunderbolt","Blizzard"]),
 set("Golem",["Earthquake","Rock Slide","Body Slam","Explosion"]),
];
(async()=>{
  const addon = path.join(host, "pkmn-showdown.node");
  await engine.initialize(true,addon);
  const gen1 = new data.Generations(sim.Dex).get(1);

  function checkForced(moveName, expectedPseudo, volatile) {
    const p1 = sim.Teams.unpack(sim.Teams.pack([
      set("Tauros",[moveName,"Body Slam","Earthquake","Blizzard"]), ...filler,
    ]));
    const p2 = sim.Teams.unpack(sim.Teams.pack([
      set("Snorlax",["Rest","Body Slam","Earthquake","Amnesia"]), ...filler,
    ]));
    for(let trial=1;trial<=32;trial++){
      const battle = engine.Battle.create(gen1,{
        p1:{name:"p1",team:p1},p2:{name:"p2",team:p2},
        seed:[trial,trial*37,trial*43,trial*59],
        showdown:true,log:true
      });
      let res=battle.update(undefined,undefined);
      const side=battle.side("p1"), foe=battle.side("p2");
      const chosen=Array.from(side.active.moves).findIndex(m=>m.id===moveName.toLowerCase().replace(/[^a-z0-9]/g,""))+1;
      assert.ok(chosen>=1,`Test set missing ${moveName}`);
      const restSlot=Array.from(foe.active.moves).findIndex(m=>m.id==="rest")+1;
      assert.ok(restSlot>=1);
      res=battle.update(engine.Choice.move(chosen),engine.Choice.move(restSlot));
      if(res.type || !side.active.volatiles[volatile]) continue;
      const legal=battle.choices("p1",res);
      assert.deepEqual(legal,[engine.Choice.move(1)],
        `Native forced ${moveName} should offer precisely Choice.move(1)`);
      const req=createEngineRequest(battle,gen1,"p1","p1",res,p1);
      assert.equal(req.active[0].trapped,true);
      assert.deepEqual(req.active[0].moves.map(m=>m.id),[expectedPseudo]);
      const choice=resolveNativeChoice(
        `move ${expectedPseudo}`,legal,side,engine.Choice
      );
      assert.deepEqual(choice,engine.Choice.move(1));
      const other=battle.choices("p2",res)[0] || engine.Choice.pass;
      res=battle.update(choice,other);
      console.log(`Native ${moveName} forced mode verified on trial ${trial}`);
      return;
    }
    throw new Error(`Could not exercise ${volatile} forced native move in 32 trials`);
  }
  checkForced("Hyper Beam","recharge","recharging");
  checkForced("Rage","fight","rage");
})().catch(e=>{console.error(e);process.exitCode=1;});
