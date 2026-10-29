#!/usr/bin/env python3
"""Differential-test legacy text and binary native Gen 1 adapters.

Every game has an immutable team pair, four-word engine seed, and independent
policy seed for each player.  The binary run chooses from sorted native legal
choices with those player PRNGs.  All legacy runs then replay the exact encoded
joint actions.  The mirrored legacy run must reproduce every authoritative
384-byte battle image, legal-choice vector, result byte, and action.  Finally,
one uninstrumented run rechecks every parsed UniversalState replay record and
another supplies the fair legacy timing measurement without hashing overhead.

Progress and results are human-readable plain text.  No artifacts are written.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metamon.env import get_metamon_teams
from metamon.env.vectorized.lane import StreamBattleLane
from metamon.env.vectorized.native_binary import (
    RESULT_LOSE,
    RESULT_NONE,
    RESULT_TIE,
    RESULT_WIN,
    NativeBinaryLane,
    result_type,
)
from metamon.env.vectorized.sim_process import make_sim_process
from metamon.env.vectorized.team_adapter import player_spec


@dataclass(frozen=True, slots=True)
class GameSpec:
    number: int
    p1: dict[str, Any]
    p2: dict[str, Any]
    engine_seed: tuple[int, int, int, int]
    p1_policy_seed: int
    p2_policy_seed: int


@dataclass(frozen=True, slots=True)
class NativeFrame:
    battle: bytes
    result: int
    p1_choices: bytes
    p2_choices: bytes


@dataclass(slots=True)
class ExpectedGame:
    frames: list[NativeFrame] = field(default_factory=list)
    actions: list[tuple[int, int]] = field(default_factory=list)
    result: int = RESULT_NONE


class MirroredTextLane:
    """Pair a native mirror frame with the legacy parser's same update."""

    def __init__(self, lane_id: int, battle_format: str):
        self.text = StreamBattleLane(lane_id, battle_format)
        self.native: Optional[NativeFrame] = None
        self.native_serial = 0
        self.consumed_native_serial = 0

    def reset_state(self) -> None:
        self.text.reset_state()
        self.native = None
        self.native_serial = 0
        self.consumed_native_serial = 0

    def handle_native_state(
        self,
        battle: bytes,
        result: int,
        p1_choices: bytes,
        p2_choices: bytes,
    ) -> None:
        self.native = NativeFrame(
            bytes(battle), result_type(result), bytes(p1_choices), bytes(p2_choices)
        )
        self.native_serial += 1

    def handle_chunk(self, stream: str, data: str) -> None:
        self.text.handle_chunk(stream, data)

    @property
    def ready(self) -> bool:
        return (
            self.native_serial > self.consumed_native_serial
            and self.text.decision_ready()
        )

    def consume(self) -> None:
        if not self.ready:
            raise RuntimeError("attempted to consume incomplete mirrored update")
        self.consumed_native_serial = self.native_serial


def _set_transport(binary: bool, mirror: bool) -> None:
    for key in (
        "METAMON_NATIVE_BINARY_STATE",
        "METAMON_NATIVE_BINARY_MIRROR",
        "METAMON_NATIVE_ENCODED_CHOICES",
        "METAMON_VEC_PROFILE",
    ):
        os.environ.pop(key, None)
    if binary:
        os.environ["METAMON_NATIVE_BINARY_STATE"] = "1"
    else:
        os.environ["METAMON_NATIVE_ENCODED_CHOICES"] = "1"
        if mirror:
            os.environ["METAMON_NATIVE_BINARY_MIRROR"] = "1"
    os.environ["METAMON_NATIVE_STRICT_PARSER"] = "1"


def _manifest(args: argparse.Namespace) -> tuple[GameSpec, ...]:
    rng = random.Random(args.seed)
    teams = get_metamon_teams(args.format, args.team_set)
    teams.seed(args.seed ^ 0x5445414D)
    if args.raw_team_files:
        for file in teams.team_files:
            raw = Path(file).read_text(encoding="utf-8")
            teams._packed_team_cache[file] = teams.join_team(
                teams.parse_showdown_team(raw)
            )
    games: list[GameSpec] = []
    for number in range(args.games):
        p1, _ = player_spec(f"seeded-p1-{number}", teams, args.format)
        p2, _ = player_spec(f"seeded-p2-{number}", teams, args.format)
        games.append(
            GameSpec(
                number=number,
                p1=dict(p1),
                p2=dict(p2),
                engine_seed=tuple(rng.randrange(65536) for _ in range(4)),
                p1_policy_seed=rng.getrandbits(64),
                p2_policy_seed=rng.getrandbits(64),
            )
        )
    return tuple(games)


def _native_digest(games: list[ExpectedGame]) -> str:
    digest = hashlib.sha256()
    for game in games:
        digest.update(struct.pack("<I", len(game.frames)))
        for frame in game.frames:
            digest.update(frame.battle)
            digest.update(bytes((frame.result, len(frame.p1_choices), len(frame.p2_choices))))
            digest.update(frame.p1_choices)
            digest.update(frame.p2_choices)
        digest.update(struct.pack("<I", len(game.actions)))
        for p1, p2 in game.actions:
            digest.update(bytes((p1, p2)))
        digest.update(bytes((game.result,)))
    return digest.hexdigest()


def _outcome_counts(results: list[int]) -> tuple[int, int, int, int]:
    return (
        results.count(RESULT_WIN),
        results.count(RESULT_LOSE),
        results.count(RESULT_TIE),
        results.count(RESULT_NONE),
    )


def _run_binary(
    manifest: tuple[GameSpec, ...], args: argparse.Namespace
) -> tuple[list[ExpectedGame], float, int]:
    _set_transport(binary=True, mirror=False)
    lane_count = min(args.lanes, len(manifest))
    lanes = [NativeBinaryLane(i) for i in range(lane_count)]
    proc = make_sim_process(lane_count, n_workers=args.workers)
    for lane in lanes:
        proc.register_lane(lane.lane_id, lane)

    expected = [ExpectedGame() for _ in manifest]
    active: dict[int, int] = {}
    next_game = 0
    completed = 0
    state_count = 0
    policy_rngs: dict[int, tuple[random.Random, random.Random]] = {}
    next_report = args.progress_every

    def start(lane_id: int, game_id: int) -> None:
        spec = manifest[game_id]
        lane = lanes[lane_id]
        lane.begin(spec.p1, spec.p2)
        active[lane_id] = game_id
        policy_rngs[game_id] = (
            random.Random(spec.p1_policy_seed),
            random.Random(spec.p2_policy_seed),
        )
        proc.start_battle(
            lane_id,
            args.format,
            spec.p1,
            spec.p2,
            seed=list(spec.engine_seed),
        )

    started = time.perf_counter()
    try:
        while next_game < lane_count:
            start(next_game, next_game)
            next_game += 1
        proc.pump_until(lambda: all(lanes[i].ready for i in active))
        while completed < len(manifest):
            actions: list[tuple[int, int, int]] = []
            restart: list[tuple[int, int]] = []
            for lane_id, game_id in list(active.items()):
                lane = lanes[lane_id]
                game = expected[game_id]
                frame = NativeFrame(
                    bytes(lane.state),
                    int(lane.result),
                    bytes(lane.p1_choices),
                    bytes(lane.p2_choices),
                )
                game.frames.append(frame)
                state_count += 1
                if lane.terminal or len(game.actions) >= args.turn_limit:
                    game.result = int(lane.result)
                    completed += 1
                    del active[lane_id]
                    if next_game < len(manifest):
                        restart.append((lane_id, next_game))
                        next_game += 1
                    continue
                if not lane.p1_choices or not lane.p2_choices:
                    raise RuntimeError(
                        f"binary game {game_id} has no legal choice before terminal"
                    )
                p1_rng, p2_rng = policy_rngs[game_id]
                p1_legal = tuple(sorted(lane.p1_choices))
                p2_legal = tuple(sorted(lane.p2_choices))
                p1 = p1_legal[p1_rng.randrange(len(p1_legal))]
                p2 = p2_legal[p2_rng.randrange(len(p2_legal))]
                lane.commit(p1, p2)
                game.actions.append((p1, p2))
                actions.append((lane_id, p1, p2))
            for lane_id, game_id in restart:
                start(lane_id, game_id)
            proc.choose_native_batch(actions)
            if active:
                proc.pump_until(lambda: all(lanes[i].ready for i in active))
            if args.progress_every and completed >= next_report:
                print(f"Binary progress: {completed}/{len(manifest)} games")
                while next_report <= completed:
                    next_report += args.progress_every
    finally:
        elapsed = time.perf_counter() - started
        proc.close()
    return expected, elapsed, state_count


def _snapshot_bytes(lane: StreamBattleLane) -> bytes:
    # Protocol 5 is deterministic for these same-process dataclasses/dicts and
    # avoids JSON conversion of numpy-backed observation fields.
    record = (
        lane.universal_state("p1").to_dict(),
        lane.universal_state("p2").to_dict(),
        lane.last_request["p1"],
        lane.last_request["p2"],
        lane.ended,
        lane.winner,
    )
    return pickle.dumps(record, protocol=5)


def _legacy_result(lane: StreamBattleLane, spec: GameSpec) -> int:
    if not lane.ended:
        return RESULT_NONE
    if lane.winner is None:
        return RESULT_TIE
    if lane.winner == spec.p1["name"]:
        return RESULT_WIN
    if lane.winner == spec.p2["name"]:
        return RESULT_LOSE
    raise RuntimeError(
        f"game {spec.number}: unknown legacy winner {lane.winner!r}"
    )


def _run_legacy(
    manifest: tuple[GameSpec, ...],
    expected: list[ExpectedGame],
    args: argparse.Namespace,
    *,
    mirror: bool,
    capture_replay: bool = True,
    expected_replay_digests: Optional[list[bytes]] = None,
) -> tuple[float, int, list[int], Optional[list[bytes]]]:
    _set_transport(binary=False, mirror=mirror)
    lane_count = min(args.lanes, len(manifest))
    handlers: list[Any]
    if mirror:
        handlers = [MirroredTextLane(i, args.format) for i in range(lane_count)]
    else:
        handlers = [StreamBattleLane(i, args.format) for i in range(lane_count)]
    proc = make_sim_process(lane_count, n_workers=args.workers)
    for i, handler in enumerate(handlers):
        proc.register_lane(i, handler)

    active: dict[int, int] = {}
    frame_index: dict[int, int] = {}
    digests = [hashlib.sha256() for _ in manifest]
    results = [RESULT_NONE for _ in manifest]
    completed = 0
    next_game = 0
    state_count = 0
    next_report = args.progress_every

    def text_lane(lane_id: int) -> StreamBattleLane:
        handler = handlers[lane_id]
        return handler.text if mirror else handler

    def ready(lane_id: int) -> bool:
        handler = handlers[lane_id]
        return handler.ready if mirror else handler.decision_ready()

    def start(lane_id: int, game_id: int) -> None:
        handler = handlers[lane_id]
        if mirror:
            handler.reset_state()
        else:
            handler.reset_state()
        active[lane_id] = game_id
        frame_index[lane_id] = 0
        spec = manifest[game_id]
        proc.start_battle(
            lane_id,
            args.format,
            spec.p1,
            spec.p2,
            seed=list(spec.engine_seed),
        )

    started = time.perf_counter()
    try:
        while next_game < lane_count:
            start(next_game, next_game)
            next_game += 1
        proc.pump_until(lambda: all(ready(i) for i in active))
        while completed < len(manifest):
            actions: list[tuple[int, int, int]] = []
            restart: list[tuple[int, int]] = []
            for lane_id, game_id in list(active.items()):
                lane = text_lane(lane_id)
                game = expected[game_id]
                index = frame_index[lane_id]
                if index >= len(game.frames):
                    raise RuntimeError(
                        f"legacy game {game_id} produced extra state {index}"
                    )
                if mirror:
                    actual = handlers[lane_id].native
                    wanted = game.frames[index]
                    if actual != wanted:
                        raise RuntimeError(
                            f"native trajectory mismatch at game {game_id}, state {index}: "
                            f"result {getattr(actual, 'result', None)}/{wanted.result}, "
                            f"battle_equal={getattr(actual, 'battle', None) == wanted.battle}, "
                            f"p1_choices_equal={getattr(actual, 'p1_choices', None) == wanted.p1_choices}, "
                            f"p2_choices_equal={getattr(actual, 'p2_choices', None) == wanted.p2_choices}"
                        )
                    handlers[lane_id].consume()
                digest = digests[game_id]
                if capture_replay:
                    snapshot = _snapshot_bytes(lane)
                    digest.update(struct.pack("<I", len(snapshot)))
                    digest.update(snapshot)
                state_count += 1
                terminal_here = index == len(game.frames) - 1
                if terminal_here:
                    actual_result = _legacy_result(lane, manifest[game_id])
                    if actual_result != game.result:
                        raise RuntimeError(
                            f"result mismatch game {game_id}: legacy={actual_result}, "
                            f"binary={game.result}"
                        )
                    results[game_id] = actual_result
                    completed += 1
                    del active[lane_id]
                    if next_game < len(manifest):
                        restart.append((lane_id, next_game))
                        next_game += 1
                    continue
                if lane.ended:
                    raise RuntimeError(
                        f"legacy game {game_id} ended before binary state {index}"
                    )
                p1, p2 = game.actions[index]
                if capture_replay:
                    digest.update(bytes((p1, p2)))
                lane.mark_settled()
                actions.append((lane_id, p1, p2))
                frame_index[lane_id] = index + 1
            for lane_id, game_id in restart:
                start(lane_id, game_id)
            proc.choose_native_batch(actions)
            if active:
                proc.pump_until(lambda: all(ready(i) for i in active))
            if args.progress_every and completed >= next_report:
                label = "Legacy mirror" if mirror else "Legacy timing"
                print(f"{label} progress: {completed}/{len(manifest)} games")
                while next_report <= completed:
                    next_report += args.progress_every
    finally:
        elapsed = time.perf_counter() - started
        proc.close()

    final_digests = [digest.digest() for digest in digests] if capture_replay else None
    if expected_replay_digests is not None:
        if final_digests is None:
            raise RuntimeError("replay digest comparison requested without capture")
        for game_id, (actual, wanted) in enumerate(
            zip(final_digests, expected_replay_digests, strict=True)
        ):
            if actual != wanted:
                raise RuntimeError(
                    f"legacy replay data mismatch at game {game_id}: "
                    f"{actual.hex()} != {wanted.hex()}"
                )
    return elapsed, state_count, results, final_digests


def _rate(games: int, seconds: float) -> float:
    return games * 60.0 / seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--format", default="gen1ou")
    parser.add_argument("--team-set", default="competitive")
    parser.add_argument("--turn-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--raw-team-files", action="store_true")
    args = parser.parse_args()
    if min(args.games, args.lanes, args.workers, args.turn_limit) <= 0:
        parser.error("games, lanes, workers, and turn-limit must be positive")

    print(
        f"Building immutable manifest: {args.games} games, master seed {args.seed}, "
        f"{args.lanes} lanes, {args.workers} host worker"
        f"{'s' if args.workers != 1 else ''}"
    )
    manifest = _manifest(args)

    print("Running binary adapter with independently seeded player PRNGs")
    expected, binary_seconds, binary_states = _run_binary(manifest, args)
    native_hash = _native_digest(expected)

    print("Running legacy adapter with native mirror for exact differential check")
    mirror_seconds, mirror_states, mirror_results, replay_digests = _run_legacy(
        manifest, expected, args, mirror=True
    )
    assert replay_digests is not None

    print("Rechecking every parsed legacy replay record without native mirroring")
    _, _, recheck_results, _ = _run_legacy(
        manifest,
        expected,
        args,
        mirror=False,
        capture_replay=True,
        expected_replay_digests=replay_digests,
    )

    print("Running uninstrumented legacy adapter for timing")
    legacy_seconds, legacy_states, legacy_results, _ = _run_legacy(
        manifest, expected, args, mirror=False, capture_replay=False
    )

    expected_results = [game.result for game in expected]
    if (
        mirror_results != expected_results
        or recheck_results != expected_results
        or legacy_results != expected_results
    ):
        raise RuntimeError("aggregate result vectors differ despite per-game checks")
    wins, losses, ties, timeouts = _outcome_counts(expected_results)
    replay_hash = hashlib.sha256(b"".join(replay_digests)).hexdigest()

    print("\nExact differential result: PASS")
    print(f"Games: {args.games}")
    print(f"P1 wins: {wins}")
    print(f"P2 wins: {losses}")
    print(f"Natural ties: {ties}")
    print(f"Turn-limit timeouts: {timeouts}")
    print(f"Native trajectory SHA-256: {native_hash}")
    print(f"Legacy parsed replay SHA-256: {replay_hash}")
    print(
        "Matched exactly: engine bytes, native legal choices, seeded actions, "
        "terminal results, and repeated legacy UniversalState replay records"
    )
    print(f"\nPerformance on the identical {args.games}-game manifest")
    print(
        f"Binary adapter: {binary_seconds:.3f}s, "
        f"{_rate(args.games, binary_seconds):.0f} games/min, "
        f"{binary_states / binary_seconds:.0f} states/s"
    )
    print(
        f"Legacy adapter: {legacy_seconds:.3f}s, "
        f"{_rate(args.games, legacy_seconds):.0f} games/min, "
        f"{legacy_states / legacy_seconds:.0f} states/s"
    )
    print(f"Binary speedup: {legacy_seconds / binary_seconds:.2f}x")
    print(
        f"Instrumented legacy mirror (not used for speedup): {mirror_seconds:.3f}s, "
        f"{mirror_states} states"
    )
    print(
        "Boundary: the binary path is a lossless native trajectory transport. "
        "It still lacks the binary-to-UniversalState decoder, so this test does "
        "not claim equality for a new model-facing observation that does not yet exist."
    )


if __name__ == "__main__":
    main()
