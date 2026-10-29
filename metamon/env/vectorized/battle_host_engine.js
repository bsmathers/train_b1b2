"use strict";
const { resolveNativeChoice } = require("./native_choice");
const { createEngineRequest } = require("./native_request");

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
 *   stdout: binary frames (READY=0, CHUNK=1, HOST_ERROR=2, LANE_ERROR=3,
 *           PONG=4, PROFILE=5, NATIVE_STATE=6)
 */

const readline = require("readline");
const path = require("path");
const fs = require("fs");

const MSG = {
  READY: 0,
  CHUNK: 1,
  HOST_ERROR: 2,
  LANE_ERROR: 3,
  PONG: 4,
  PROFILE: 5,
  NATIVE_STATE: 6,
};
const STREAM = { p1: 0, p2: 1, error: 2 };
const BINARY_STATE = process.env.METAMON_NATIVE_BINARY_STATE === "1";
// Differential-test mode emits the authoritative native frame immediately
// before the legacy text frames for the *same* engine update.  Production
// collection never enables this: it exists solely to prove that a separately
// executed binary run saw identical engine bytes, choices, and results.
const BINARY_MIRROR = process.env.METAMON_NATIVE_BINARY_MIRROR === "1";
const ENCODED_CHOICES =
  BINARY_STATE || process.env.METAMON_NATIVE_ENCODED_CHOICES === "1";
let nativeFrameBatch = null;

// These counters answer a narrower question than the Python wall profiler:
// exactly how much synchronous host time is spent inside the native Zig
// battle.update call, versus adapting its result into Showdown protocol.
// Keep them disabled outside explicit profiles so hrtime calls cannot reduce
// production collector throughput.
const PROFILE = process.env.METAMON_VEC_PROFILE === "1";
const profile = {
  startedNs: process.hrtime.bigint(),
  jsonParseNs: 0n,
  commandHandleNs: 0n,
  engineUpdateNs: 0n,
  engineStartUpdateNs: 0n,
  engineStartNs: 0n,
  teamUnpackNs: 0n,
  battleCreateNs: 0n,
  logSetupNs: 0n,
  choiceResolutionNs: 0n,
  emitTurnNs: 0n,
  logDecodeNs: 0n,
  requestBuildNs: 0n,
  frameWriteNs: 0n,
  commands: 0,
  starts: 0,
  steps: 0,
  chunks: 0,
  payloadBytes: 0,
  teamCacheHits: 0,
  teamCacheMisses: 0,
};

function profileNow() {
  return PROFILE ? process.hrtime.bigint() : 0n;
}

function profileAdd(key, started) {
  if (PROFILE) profile[key] += process.hrtime.bigint() - started;
}

function emitProfile() {
  const wallNs = process.hrtime.bigint() - profile.startedNs;
  const lines = [
    `binary_state=${+BINARY_STATE}`,
    `wall_ns=${wallNs}`,
    `json_parse_ns=${profile.jsonParseNs}`,
    `command_handle_ns=${profile.commandHandleNs}`,
    `engine_update_ns=${profile.engineUpdateNs}`,
    `engine_start_update_ns=${profile.engineStartUpdateNs}`,
    `engine_start_ns=${profile.engineStartNs}`,
    `team_unpack_ns=${profile.teamUnpackNs}`,
    `battle_create_ns=${profile.battleCreateNs}`,
    `log_setup_ns=${profile.logSetupNs}`,
    `choice_resolution_ns=${profile.choiceResolutionNs}`,
    `emit_turn_ns=${profile.emitTurnNs}`,
    `log_decode_ns=${profile.logDecodeNs}`,
    `request_build_ns=${profile.requestBuildNs}`,
    `frame_write_ns=${profile.frameWriteNs}`,
    `commands=${profile.commands}`,
    `starts=${profile.starts}`,
    `steps=${profile.steps}`,
    `chunks=${profile.chunks}`,
    `payload_bytes=${profile.payloadBytes}`,
    `team_cache_hits=${profile.teamCacheHits}`,
    `team_cache_misses=${profile.teamCacheMisses}`,
  ];
  const payload = Buffer.from(lines.join("\n"), "utf8");
  const frame = Buffer.alloc(5 + payload.length);
  frame.writeUInt8(MSG.PROFILE, 0);
  frame.writeUInt32LE(payload.length, 1);
  payload.copy(frame, 5);
  process.stdout.write(frame);
}

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
  const started = profileNow();
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
  if (PROFILE) {
    profile.frameWriteNs += process.hrtime.bigint() - started;
    profile.chunks += 1;
    profile.payloadBytes += payload.length;
  }
}

/**
 * Emit the native Gen 1 state without converting it to Showdown protocol.
 *
 * Payload layout (little endian):
 *   u8 type, u32 lane, u32 epoch, u8 encoded result,
 *   u8 p1 choice count, u8 p2 choice count, u16 battle byte length,
 *   battle bytes, p1 encoded choices, p2 encoded choices.
 *
 * The 384-byte battle is the source of truth owned by the Zig engine. Choice
 * bytes use the engine ABI (low two bits are pass/move/switch; the remaining
 * bits are the 1-based slot). This removes the JS log/request adapter from the
 * hot path. The text path remains the default pending observation differential
 * tests.
 */
function emitNativeState(lane, epoch, battle, result, p1Choices, p2Choices) {
  const started = profileNow();
  const view = battle.bytes();
  const battleBytes = Buffer.from(view.buffer, view.byteOffset, view.byteLength);
  const n1 = p1Choices.length;
  const n2 = p2Choices.length;
  if (n1 > 255 || n2 > 255 || battleBytes.length > 65535) {
    throw new Error("native state frame exceeds binary protocol limits");
  }
  const headerLength = 14;
  const frame = Buffer.allocUnsafe(headerLength + battleBytes.length + n1 + n2);
  frame.writeUInt8(MSG.NATIVE_STATE, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt8(pkmnEngine.Result.encode(result), 9);
  frame.writeUInt8(n1, 10);
  frame.writeUInt8(n2, 11);
  frame.writeUInt16LE(battleBytes.length, 12);
  battleBytes.copy(frame, headerLength);
  let offset = headerLength + battleBytes.length;
  for (const choice of p1Choices) frame.writeUInt8(pkmnEngine.Choice.encode(choice), offset++);
  for (const choice of p2Choices) frame.writeUInt8(pkmnEngine.Choice.encode(choice), offset++);
  if (nativeFrameBatch === null) process.stdout.write(frame);
  else nativeFrameBatch.push(frame);
  if (PROFILE) {
    profile.frameWriteNs += process.hrtime.bigint() - started;
    profile.chunks += 1;
    profile.payloadBytes += frame.length - headerLength;
  }
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


function stringifyProtocolLine(line) {
  let s = "|" + line.args.join("|");
  for (const [k, v] of Object.entries(line.kwArgs || {})) {
    s += "|[" + k + "]" + (v ? " " + v : "");
  }
  return s;
}

// The Python team cache avoids reparsing team files, but every battle still
// sends the same packed strings to this host. Showdown's unpack is pure and
// @pkmn/engine copies each set into a fresh native battle buffer, so these
// decoded inputs can be shared safely. Bound the map for nonstandard callers
// that stream an effectively unlimited set of teams.
const unpackedTeamCache = new Map();
const MAX_UNPACKED_TEAMS = 4096;

function unpackTeam(packed) {
  const cached = unpackedTeamCache.get(packed);
  if (cached !== undefined) {
    if (PROFILE) profile.teamCacheHits += 1;
    return cached;
  }
  const team = Showdown.Teams.unpack(packed);
  unpackedTeamCache.set(packed, team);
  if (PROFILE) profile.teamCacheMisses += 1;
  if (unpackedTeamCache.size > MAX_UNPACKED_TEAMS) {
    unpackedTeamCache.delete(unpackedTeamCache.keys().next().value);
  }
  return team;
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
    const startStarted = profileNow();
    this.epoch = epoch;
    this.pendingChoices = { p1: null, p2: null };
    this.sentBattleHeader = false;

    try {
      if (PROFILE) profile.starts += 1;
      const { Battle, Log, Lookup, Info } = pkmnEngine;

      const unpackStarted = profileNow();
      const p1Team = unpackTeam(p1spec.team);
      const p2Team = unpackTeam(p2spec.team);
      profileAdd("teamUnpackNs", unpackStarted);
      this.originalTeams = { p1: p1Team, p2: p2Team };
      this.playerNames = {
        p1: String(p1spec.name || "p1"),
        p2: String(p2spec.name || "p2"),
      };

      let seed = [1, 2, 3, 4];
      if (spec.seed && Array.isArray(spec.seed)) {
        seed = spec.seed.slice(0, 4);
      }

      const createStarted = profileNow();
      this.battle = Battle.create(gen1, {
        p1: { name: this.playerNames.p1, team: p1Team },
        p2: { name: this.playerNames.p2, team: p2Team },
        seed: seed,
        showdown: true,
        // Binary state is authoritative and does not need the compact protocol
        // log. Avoid asking Zig to generate bytes that no consumer reads.
        log: !BINARY_STATE,
      });
      profileAdd("battleCreateNs", createStarted);

      const logSetupStarted = profileNow();
      this.logParser = BINARY_STATE ? null : new Log(
          gen1,
          Lookup.get(gen1),
          new Info(gen1, {
            p1: { name: this.playerNames.p1, team: p1Team },
            p2: { name: this.playerNames.p2, team: p2Team },
          })
        );
      profileAdd("logSetupNs", logSetupStarted);

      const updateStarted = profileNow();
      const res0 = this.battle.update(undefined, undefined);
      if (PROFILE) {
        const elapsed = process.hrtime.bigint() - updateStarted;
        profile.engineUpdateNs += elapsed;
        profile.engineStartUpdateNs += elapsed;
      }
      this.lastResult = res0;
      this._emitTurn(res0);
    } catch (err) {
      emitLaneError(this.id, epoch, `Engine start error: ${err.message}`);
    } finally {
      profileAdd("engineStartNs", startStarted);
    }
  }

  _emitTurn(res) {
    const emitStarted = profileNow();
    try {
      if (BINARY_STATE || BINARY_MIRROR) {
        const p1Choices = this.battle.choices("p1", res) || [];
        const p2Choices = this.battle.choices("p2", res) || [];
        emitNativeState(
          this.id, this.epoch, this.battle, res, p1Choices, p2Choices
        );
        if (BINARY_STATE) return;
      }
      // The engine Result is authoritative and is always from p1's
      // perspective.  Do not infer the winner from Side fields: @pkmn/engine
      // Side objects have no `fainted` property, so that inference appended a
      // bogus tie. Strip any terminal record decoded from the binary log and
      // append exactly one unambiguous record from Result below.
      const logStarted = profileNow();
      const logLines = Array.from(this.logParser.parse(this.battle.log))
        .map(stringifyProtocolLine)
        .filter(line => line !== "|tie" && !line.startsWith("|win|"));
      profileAdd("logDecodeNs", logStarted);
      const requestStarted = profileNow();
      const reqP1 = createEngineRequest(
        this.battle, gen1, "p1", this.playerNames.p1, res, this.originalTeams.p1
      );
      const reqP2 = createEngineRequest(
        this.battle, gen1, "p2", this.playerNames.p2, res, this.originalTeams.p2
      );
      profileAdd("requestBuildNs", requestStarted);

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
    } finally {
      profileAdd("emitTurnNs", emitStarted);
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
      if (PROFILE) profile.steps += 1;
      const c1Str = this.pendingChoices.p1;
      const c2Str = this.pendingChoices.p2;
      this.pendingChoices = { p1: null, p2: null };

      const needP1 = !this.lastResult || this.lastResult.p1 !== "pass";
      const needP2 = !this.lastResult || this.lastResult.p2 !== "pass";

      const choiceStarted = profileNow();
      const c1 = needP1 ? this._resolveChoice("p1", c1Str) : pkmnEngine.Choice.pass;
      const c2 = needP2 ? this._resolveChoice("p2", c2Str) : pkmnEngine.Choice.pass;
      profileAdd("choiceResolutionNs", choiceStarted);

      const updateStarted = profileNow();
      const res = this.battle.update(c1, c2);
      profileAdd("engineUpdateNs", updateStarted);
      this.lastResult = res;
      this._emitTurn(res);
    } catch (err) {
      emitLaneError(this.id, this.epoch, `Engine step error: ${err.message}`);
    }
  }

  stepEncoded(c1, c2, epoch) {
    if (!ENCODED_CHOICES) {
      throw new Error("encoded choices require binary or differential-test mode");
    }
    if (!this.battle || (epoch !== undefined && epoch !== this.epoch)) return;
    try {
      if (PROFILE) profile.steps += 1;
      const updateStarted = profileNow();
      const res = this.battle.update(
        pkmnEngine.Choice.decode(Number(c1)),
        pkmnEngine.Choice.decode(Number(c2))
      );
      profileAdd("engineUpdateNs", updateStarted);
      this.lastResult = res;
      this._emitTurn(res);
    } catch (err) {
      emitLaneError(this.id, this.epoch, `Engine encoded step error: ${err.message}`);
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
    case "choose_native_batch": {
      const frames = [];
      nativeFrameBatch = frames;
      for (const c of msg.choices || []) {
        const lane = lanes.get(c[0]);
        if (lane) lane.stepEncoded(c[2], c[3], c[1]);
      }
      nativeFrameBatch = null;
      if (frames.length) process.stdout.write(Buffer.concat(frames));
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
    case "profile": {
      emitProfile();
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
      const parseStarted = profileNow();
      msg = JSON.parse(trimmed);
      profileAdd("jsonParseNs", parseStarted);
    } catch (err) {
      emitHostError(`bad json: ${err.message}`);
      return;
    }
    try {
      const handleStarted = profileNow();
      handleCommand(msg);
      profileAdd("commandHandleNs", handleStarted);
      if (PROFILE) profile.commands += 1;
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
