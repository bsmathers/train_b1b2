#!/usr/bin/env python3
"""Benchmark lossless Zig-state self-play without the Node text adapter.

This measures the engine + binary IPC + battle restart ceiling.  It also checks
that every collected trajectory has exactly one more state than joint action,
that every action was advertised legal, and that every completed game has an
unambiguous native terminal result.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metamon.env import get_metamon_team_mixture, get_metamon_teams
from metamon.env.vectorized.native_binary import (
    RESULT_LOSE,
    RESULT_TIE,
    RESULT_WIN,
    NativeBinaryLane,
)
from metamon.env.vectorized.sim_process import make_sim_process
from metamon.env.vectorized.team_adapter import player_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--format", default="gen1ou")
    parser.add_argument("--team-set", default="competitive")
    parser.add_argument("--turn-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--raw-team-files",
        action="store_true",
        help="Parse fully specified local teams without usage-stat normalization.",
    )
    args = parser.parse_args()
    if args.games <= 0 or args.lanes <= 0 or args.workers <= 0 or args.turn_limit <= 0:
        parser.error("games, lanes, workers, and turn-limit must be positive")

    # Must be set before the subprocess is created.  The normal environment
    # never opts into this until the observation decoder is verified.
    os.environ["METAMON_NATIVE_BINARY_STATE"] = "1"
    os.environ.setdefault("METAMON_VEC_PROFILE", "1")

    rng = random.Random(args.seed)
    team_names = [name.strip() for name in args.team_set.split(",") if name.strip()]
    teams = (
        get_metamon_teams(args.format, team_names[0])
        if len(team_names) == 1
        else get_metamon_team_mixture(args.format, team_names)
    )
    teams.seed(args.seed)
    if args.raw_team_files:
        for file in teams.team_files:
            raw = Path(file).read_text(encoding="utf-8")
            teams._packed_team_cache[file] = teams.join_team(
                teams.parse_showdown_team(raw)
            )
    lanes = [NativeBinaryLane(i) for i in range(args.lanes)]
    proc = make_sim_process(num_lanes=args.lanes, n_workers=args.workers)
    for lane in lanes:
        proc.register_lane(lane.lane_id, lane)

    game_number = [0] * args.lanes

    def start_lane(lane_id: int) -> None:
        game = game_number[lane_id]
        p1, _ = player_spec(f"binary-p1-{lane_id}", teams, args.format)
        p2, _ = player_spec(f"binary-p2-{lane_id}", teams, args.format)
        seed = [rng.randrange(65536) for _ in range(4)]
        lane = lanes[lane_id]
        lane.begin(p1, p2)
        proc.start_battle(lane_id, args.format, p1, p2, seed=seed)
        game_number[lane_id] = game + 1

    wins = losses = draws = timeouts = completed = lane_steps = 0
    last_report = 0
    try:
        for lane_id in range(args.lanes):
            start_lane(lane_id)
        proc.pump_until(lambda: all(lane.ready for lane in lanes))
        started = time.perf_counter()
        while completed < args.games:
            encoded_actions: list[tuple[int, int, int]] = []
            restart: list[int] = []
            for lane in lanes:
                if completed >= args.games:
                    break
                if lane.terminal:
                    trajectory = lane.trajectory
                    assert trajectory is not None
                    if len(trajectory.states) != len(trajectory.actions) + 1:
                        raise RuntimeError(
                            f"lane {lane.lane_id}: state/action alignment failed"
                        )
                    completed += 1
                    if lane.result == RESULT_WIN:
                        wins += 1
                    elif lane.result == RESULT_LOSE:
                        losses += 1
                    elif lane.result == RESULT_TIE:
                        draws += 1
                    else:
                        raise RuntimeError(
                            f"lane {lane.lane_id}: invalid terminal {lane.result}"
                        )
                    if completed < args.games:
                        restart.append(lane.lane_id)
                    continue
                trajectory = lane.trajectory
                assert trajectory is not None
                if len(trajectory.actions) >= args.turn_limit:
                    completed += 1
                    timeouts += 1
                    if completed < args.games:
                        restart.append(lane.lane_id)
                    continue
                if not lane.p1_choices or not lane.p2_choices:
                    raise RuntimeError(
                        f"lane {lane.lane_id}: nonterminal state has no legal choice"
                    )
                p1 = rng.choice(lane.p1_choices)
                p2 = rng.choice(lane.p2_choices)
                lane.commit(p1, p2)
                encoded_actions.append((lane.lane_id, p1, p2))
                lane_steps += 1

            for lane_id in restart:
                start_lane(lane_id)
            proc.choose_native_batch(encoded_actions)
            active = [
                lane for lane in lanes
                if lane.trajectory is not None and not lane.ready
            ]
            if active:
                proc.pump_until(lambda: all(lane.ready for lane in active))
            if completed - last_report >= args.progress_every:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"Progress: {completed} games, "
                    f"{60.0 * completed / elapsed:.1f} games/min",
                    flush=True,
                )
                last_report = completed

        elapsed = time.perf_counter() - started
        print("\nNative binary self-play benchmark")
        print(f"Completed games: {completed}")
        print(f"P1 wins / losses: {wins} / {losses}")
        print(f"Natural draws: {draws}")
        print(f"Decision-limit timeouts: {timeouts}")
        print(f"Elapsed seconds: {elapsed:.3f}")
        print(f"Games per minute: {60.0 * completed / elapsed:.1f}")
        print(f"Lane steps per second: {lane_steps / elapsed:.1f}")
        print("\nNative boundary profile")
        print(proc.profile_report())
    finally:
        proc.close()


if __name__ == "__main__":
    main()
