"use strict";
const { resolveNativeChoice } = require("./native_choice");

/**
 * High-speed vectorized battle host adapter powered by @pkmn/engine.
 *
 * For Gen 1 OU battles, runs via the Zig-compiled @pkmn/engine native addon
 * (~50x faster than standard Node Showdown).
 * Seamlessly falls back to standard pokemon-showdown for other generations
 * or when the native engine addon is not installed.
 *
 * Implements the exact same binary IPC protocol as battle_host.js:
 *   stdin: JSON-lines commands (start, choose, choose_batch, reset, ping, close)
 *   stdout: binary frames (READY=0, CHUNK=1, HOST_ERROR=2, LANE_ERROR=3, PONG=4)
 */

const readline = require("readline");
const path = require("path");
const fs = require("fs");

const MSG = { READY: 0, CHUNK: 1, HOST_ERROR: 2, LANE_ERROR: 3, PONG: 4 };
const STREAM = { p1: 0, p2: 1, error: 2 };

function emitReady() {
  const buf = Buffer.alloc(1);
  buf.writeUInt8(MSG.READY, 0);
  process.stdout.write(buf);
}

function emitPong() {
  const buf = Buffer.alloc(1);
  buf.writeUInt8(MSG.PONG, 0);
  process.stdout.write(buf);
}

function emitHostError(message) {
  const payload = Buffer.from(String(message), "utf8");
  const frame = Buffer.alloc(5 + payload.length);
  frame.writeUInt8(MSG.HOST_ERROR, 0);
  frame.writeUInt32LE(payload.length, 1);
  payload.copy(frame, 5);
  process.stdout.write(frame);
}

function emitLaneError(lane, epoch, message) {
  const payload = Buffer.from(String(message), "utf8");
  const frame = Buffer.alloc(13 + payload.length);
  frame.writeUInt8(MSG.LANE_ERROR, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt32LE(payload.length, 9);
  payload.copy(frame, 13);
  process.stdout.write(frame);
}

function emitChunk(lane, epoch, streamName, data) {
  const payload = Buffer.from(String(data), "utf8");
  const streamId = STREAM[streamName] !== undefined ? STREAM[streamName] : STREAM.error;
  const frame = Buffer.alloc(14 + payload.length);
  frame.writeUInt8(MSG.CHUNK, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt8(streamId, 9);
  frame.writeUInt32LE(payload.length, 10);
  payload.copy(frame, 14);
  process.stdout.write(frame);
}

// ---------------------------------------------------------------------------
// Load engines
// ---------------------------------------------------------------------------

let Showdown = null;
try {
  Showdown = require("pokemon-showdown");
} catch (e) {
  const devPath = process.env.METAMON_SHOWDOWN_DIST;
  if (devPath) {
    try { Showdown = require(devPath); } catch (_) {}
  }
}

let pkmnEngine = null;
let pkmnData = null;
let pkmnSim = null;
let gen1 = null;
let engineReady = false;
const requireNativeGen1 = process.env.METAMON_REQUIRE_PKMN_ENGINE !== "0";

function findEngineAddon() {
  const candidates = [
    process.env.METAMON_PKMN_ENGINE,
    path.join(__dirname, "pkmn-showdown.node"),
    path.join(__dirname, "node_modules", "@pkmn", "engine", "pkmn-showdown.node"),
    path.join(__dirname, "..", "..", "..", "build", "lib", "pkmn-showdown.node"),
  ];
  for (const c of candidates) {
    if (c && fs.existsSync(c)) return path.resolve(c);
  }
  return null;
}

function ensureCommonModule() {
  const commonSrc = path.join(__dirname, "pkmn_engine_common.js");
  if (!fs.existsSync(commonSrc)) return;
  const targetDirs = [
    path.join(__dirname, "node_modules", "@pkmn", "engine", "build", "pkg"),
    path.join(__dirname, "node_modules", "@pkmn", "engine", "build", "lib"),
  ];
  for (const tDir of targetDirs) {
    if (fs.existsSync(tDir)) {
      const targetFile = path.join(tDir, "common.js");
      if (!fs.existsSync(targetFile)) {
        try { fs.copyFileSync(commonSrc, targetFile); } catch (_) {}
      }
    }
  }
}

async function tryInitEngine() {
  try {
    ensureCommonModule();
    pkmnEngine = require("@pkmn/engine");
    pkmnData = require("@pkmn/data");
    pkmnSim = require("@pkmn/sim");

    const addonPath = findEngineAddon();
    if (addonPath && pkmnEngine.initialize) {
      await pkmnEngine.initialize(true, addonPath);
      const gens = new pkmnData.Generations(pkmnSim.Dex);
      gen1 = gens.get(1);
      engineReady = true;
      process.stderr.write(`[@pkmn/engine] HIGH-SPEED NATIVE ZIG ENGINE ACTIVE (addon: ${addonPath})\n`);
    } else {
      const action = requireNativeGen1 ? "Gen 1 battles will be rejected" : "Falling back to slow Showdown";
      process.stderr.write(`[@pkmn/engine WARN] Addon not found (addonPath=${addonPath}). ${action}.\n`);
    }
  } catch (err) {
    engineReady = false;
    const action = requireNativeGen1 ? "Gen 1 battles will be rejected" : "Falling back to slow Showdown";
    process.stderr.write(`[@pkmn/engine ERROR] Failed to initialize native engine: ${err.message}. ${action}.\n`);
  }
}

// ---------------------------------------------------------------------------
// Standard Showdown Lane (Fallback)
// ---------------------------------------------------------------------------

class ShowdownLane {
  constructor(id) {
    this.id = id;
    this.battleStream = null;
    this.streams = null;
    this.epoch = 0;
  }

  start(spec, p1spec, p2spec, epoch) {
    if (!Showdown) {
      throw new Error("pokemon-showdown is required for fallback lane");
    }
    this.battleStream = new Showdown.BattleStream();
    this.streams = Showdown.getPlayerStreams(this.battleStream);
    this.epoch = epoch;

    this._pump(this.streams.p1, "p1", epoch);
    this._pump(this.streams.p2, "p2", epoch);

    const initMessage =
      `>start ${JSON.stringify(spec)}\n` +
      `>player p1 ${JSON.stringify(p1spec)}\n` +
      `>player p2 ${JSON.stringify(p2spec)}`;
    void this.streams.omniscient.write(initMessage);
  }

  async _pump(stream, name, epoch) {
    const id = this.id;
    try {
      for await (const chunk of stream) {
        if (chunk) emitChunk(id, epoch, name, chunk);
      }
    } catch (err) {
      emitLaneError(id, epoch, `${name}: ${err.message}`);
    }
  }

  choose(side, choice, epoch) {
    if (!this.streams || (epoch !== undefined && epoch !== this.epoch)) return;
    const stream = side === "p1" ? this.streams.p1 : this.streams.p2;
    void stream.write(String(choice));
  }

  destroy() {
    if (this.battleStream) {
      try { void this.battleStream.destroy(); } catch (_) {}
    }
    this.battleStream = null;
    this.streams = null;
  }
}

// ---------------------------------------------------------------------------
// Fast Native @pkmn/engine Lane (Gen 1)
// ---------------------------------------------------------------------------

function createEngineRequest(engineBattle, gen, sideId, playerName, res) {
  const isP1 = sideId === "p1";
  const playerChoiceType = isP1 ? res.p1 : res.p2;
  const side = engineBattle.side(sideId);
  const pokemonList = [];

  // Zig lists Pokemon in CURRENT switch order (the active mon is always
  // dynamic slot 1). Showdown private |request| party entries MUST remain in
  // ORIGINAL order throughout the entire battle. Pokemon.position is the
  // stable original position; side.slot(position) maps it back for choices.
  for (const mon of Array.from(side.pokemon).sort((a, b) => a.position - b.position)) {
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
      ident: `${sideId}: ${speciesName}`,
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

function stringifyProtocolLine(line) {
  let s = "|" + line.args.join("|");
  for (const [k, v] of Object.entries(line.kwArgs || {})) {
    s += "|[" + k + "]" + (v ? " " + v : "");
  }
  return s;
}

class EngineLane {
  constructor(id) {
    this.id = id;
    this.battle = null;
    this.logParser = null;
    this.epoch = 0;
    this.pendingChoices = { p1: null, p2: null };
    this.playerNames = { p1: "p1", p2: "p2" };
    this.sentBattleHeader = false;
  }

  start(spec, p1spec, p2spec, epoch) {
    this.epoch = epoch;
    this.pendingChoices = { p1: null, p2: null };
    this.sentBattleHeader = false;

    try {
      const { Battle, Log, Lookup, Info } = pkmnEngine;

      const p1Team = Showdown.Teams.unpack(p1spec.team);
      const p2Team = Showdown.Teams.unpack(p2spec.team);
      this.playerNames = {
        p1: String(p1spec.name || "p1"),
        p2: String(p2spec.name || "p2"),
      };

      let seed = [1, 2, 3, 4];
      if (spec.seed && Array.isArray(spec.seed)) {
        seed = spec.seed.slice(0, 4);
      }

      this.battle = Battle.create(gen1, {
        p1: { name: this.playerNames.p1, team: p1Team },
        p2: { name: this.playerNames.p2, team: p2Team },
        seed: seed,
        showdown: true,
        log: true,
      });

      this.logParser = new Log(
        gen1,
        Lookup.get(gen1),
        new Info(gen1, {
          p1: { name: this.playerNames.p1, team: p1Team },
          p2: { name: this.playerNames.p2, team: p2Team },
        })
      );

      const res0 = this.battle.update(undefined, undefined);
      this.lastResult = res0;
      this._emitTurn(res0);
    } catch (err) {
      emitLaneError(this.id, epoch, `Engine start error: ${err.message}`);
    }
  }

  _emitTurn(res) {
    try {
      // The engine Result is authoritative and is always from p1's
      // perspective.  Do not infer the winner from Side fields: @pkmn/engine
      // Side objects have no `fainted` property, so that inference appended a
      // bogus tie. Strip any terminal record decoded from the binary log and
      // append exactly one unambiguous record from Result below.
      const logLines = Array.from(this.logParser.parse(this.battle.log))
        .map(stringifyProtocolLine)
        .filter(line => line !== "|tie" && !line.startsWith("|win|"));
      const reqP1 = createEngineRequest(
        this.battle, gen1, "p1", this.playerNames.p1, res
      );
      const reqP2 = createEngineRequest(
        this.battle, gen1, "p2", this.playerNames.p2, res
      );

      // @pkmn/engine's compact battle log omits the simulator's player
      // declarations.  Metamon's replay parser needs those declarations to
      // map a later winner name back to Player 1/2.  Emit them once per battle
      // on both POV streams before any state/request traffic.
      const headerLines = this.sentBattleHeader ? [] : [
        `|player|p1|${this.playerNames.p1}`,
        `|player|p2|${this.playerNames.p2}`,
      ];
      this.sentBattleHeader = true;

      // Format p1 chunk
      const p1Parts = [...headerLines];
      if (reqP1) p1Parts.push(`|request|${JSON.stringify(reqP1)}`);
      p1Parts.push(...logLines);

      // Format p2 chunk
      const p2Parts = [...headerLines];
      if (reqP2) p2Parts.push(`|request|${JSON.stringify(reqP2)}`);
      p2Parts.push(...logLines);

      // Check battle conclusion
      if (res.type) {
        let endLine;
        if (res.type === "win") endLine = `|win|${this.playerNames.p1}`;
        else if (res.type === "lose") endLine = `|win|${this.playerNames.p2}`;
        else if (res.type === "tie") endLine = "|tie";
        else throw new Error(`native engine returned terminal result ${res.type}`);

        p1Parts.push(endLine);
        p2Parts.push(endLine);
      }

      if (p1Parts.length > 0) emitChunk(this.id, this.epoch, "p1", p1Parts.join("\n"));
      if (p2Parts.length > 0) emitChunk(this.id, this.epoch, "p2", p2Parts.join("\n"));
    } catch (err) {
      emitLaneError(this.id, this.epoch, `Engine emit error: ${err.message}`);
    }
  }

  _resolveChoice(sideId, choiceStr) {
    const { Choice } = pkmnEngine;
    if (!this.lastResult) return Choice.pass;
    const choices = this.battle.choices(sideId, this.lastResult) || [];
    // A requested action must agree with native legal choices; never replace
    // an unmatched move/switch with the first legal choice.
    return resolveNativeChoice(choiceStr, choices, this.battle.side(sideId), Choice);
  }

  choose(side, choiceStr, epoch) {
    if (!this.battle || (epoch !== undefined && epoch !== this.epoch)) return;
    this.pendingChoices[side] = String(choiceStr);

    const needP1 = !this.lastResult || this.lastResult.p1 !== "pass";
    const needP2 = !this.lastResult || this.lastResult.p2 !== "pass";

    const p1Ready = !needP1 || this.pendingChoices.p1 !== null;
    const p2Ready = !needP2 || this.pendingChoices.p2 !== null;

    if (p1Ready && p2Ready) {
      this._step();
    }
  }

  _step() {
    try {
      const c1Str = this.pendingChoices.p1;
      const c2Str = this.pendingChoices.p2;
      this.pendingChoices = { p1: null, p2: null };

      const needP1 = !this.lastResult || this.lastResult.p1 !== "pass";
      const needP2 = !this.lastResult || this.lastResult.p2 !== "pass";

      const c1 = needP1 ? this._resolveChoice("p1", c1Str) : pkmnEngine.Choice.pass;
      const c2 = needP2 ? this._resolveChoice("p2", c2Str) : pkmnEngine.Choice.pass;

      const res = this.battle.update(c1, c2);
      this.lastResult = res;
      this._emitTurn(res);
    } catch (err) {
      emitLaneError(this.id, this.epoch, `Engine step error: ${err.message}`);
    }
  }

  destroy() {
    this.battle = null;
    this.logParser = null;
    this.pendingChoices = { p1: null, p2: null };
    this.playerNames = { p1: "p1", p2: "p2" };
    this.sentBattleHeader = false;
  }
}

// ---------------------------------------------------------------------------
// Host Routing
// ---------------------------------------------------------------------------

const lanes = new Map();

function getLane(id, formatid) {
  let lane = lanes.get(id);
  if (!lane) {
    const isGen1 = formatid && formatid.toLowerCase().startsWith("gen1");
    if (isGen1 && engineReady) {
      lane = new EngineLane(id);
    } else if (isGen1 && requireNativeGen1) {
      throw new Error(
        "native @pkmn/engine addon is required for Gen 1; run tools/setup_pkmn_engine.sh"
      );
    } else {
      lane = new ShowdownLane(id);
    }
    lanes.set(id, lane);
  }
  return lane;
}

function handleCommand(msg) {
  switch (msg.cmd) {
    case "start": {
      let lane = lanes.get(msg.lane);
      if (lane) lane.destroy();
      lane = getLane(msg.lane, msg.formatid);
      const spec = { formatid: msg.formatid };
      if (msg.seed !== undefined && msg.seed !== null) spec.seed = msg.seed;
      const epoch = msg.epoch !== undefined ? msg.epoch : lane.epoch + 1;
      lane.start(spec, msg.p1 || { name: "p1" }, msg.p2 || { name: "p2" }, epoch);
      break;
    }
    case "choose": {
      const lane = lanes.get(msg.lane);
      if (lane) lane.choose(msg.side, msg.choice, msg.epoch);
      break;
    }
    case "choose_batch": {
      for (const c of msg.choices || []) {
        const lane = lanes.get(c.lane);
        if (lane) lane.choose(c.side, c.choice, c.epoch);
      }
      break;
    }
    case "reset": {
      const lane = lanes.get(msg.lane);
      if (lane) lane.destroy();
      break;
    }
    case "ping": {
      emitPong();
      break;
    }
    case "close": {
      for (const lane of lanes.values()) lane.destroy();
      process.exit(0);
      break;
    }
    default:
      emitHostError(`unknown cmd: ${JSON.stringify(msg)}`);
  }
}

async function main() {
  await tryInitEngine();

  const rl = readline.createInterface({ input: process.stdin });
  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg;
    try {
      msg = JSON.parse(trimmed);
    } catch (err) {
      emitHostError(`bad json: ${err.message}`);
      return;
    }
    try {
      handleCommand(msg);
    } catch (err) {
      emitLaneError(
        msg && msg.lane !== undefined ? msg.lane : 0,
        msg && msg.epoch !== undefined ? msg.epoch : 0,
        `cmd ${msg && msg.cmd}: ${err.message}`
      );
    }
  });
  rl.on("close", () => process.exit(0));

  emitReady();
}

main().catch((err) => {
  emitHostError(`Fatal initialization error: ${err.message}`);
});
