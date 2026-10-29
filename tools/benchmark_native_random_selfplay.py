#!/usr/bin/env python3
"""Benchmark native Gen 1 random-vs-random battles through the RL adapter.

Both sides sample uniformly from the action mask supplied by Metamon.  This is
deliberately the same VectorizedShowdownEnv boundary used by the Ataraxos
collector: two POV parsers, legal-action conversion, rewards, lane restarts,
and optional native timing frames are all included in the measurement.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metamon.env import get_metamon_teams
from metamon.env.vectorized import RandomBatchedOpponent, VectorizedShowdownEnv
from metamon.interface import (
    DefaultActionSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)


def _prepack_explicit_teams(team_set, limit: int | None) -> None:
    if limit is not None:
        team_set.team_files = team_set.team_files[: max(1, int(limit))]
    for file in team_set.team_files:
        raw = Path(file).read_text(encoding="utf-8")
        team_set._packed_team_cache[file] = team_set.join_team(
            team_set.parse_showdown_team(raw)
        )


def _legal_random_actions(
    legal_actions: list[list[int]], rng: np.random.Generator
) -> np.ndarray:
    actions = np.empty(len(legal_actions), dtype=np.int64)
    for lane, legal in enumerate(legal_actions):
        if not legal:
            raise RuntimeError(
                f"evaluated random player has no legal action in lane {lane}"
            )
        actions[lane] = int(rng.choice(legal))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--format", default="gen1ou")
    parser.add_argument("--team-set", default="competitive")
    parser.add_argument("--turn-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--raw-team-files",
        action="store_true",
        help="Use only for explicit local teams; skips expensive normalization.",
    )
    parser.add_argument("--team-limit", type=int)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable VectorizedShowdownEnv and native-boundary timing reports.",
    )
    args = parser.parse_args()

    if args.games <= 0 or args.lanes <= 0 or args.workers <= 0:
        parser.error("games, lanes, and workers must all be positive")
    if not args.no_profile:
        os.environ["METAMON_VEC_PROFILE"] = "1"

    obs_space = DefaultObservationSpace()
    action_space = DefaultActionSpace()
    reward = DefaultShapedReward()
    teams = get_metamon_teams(args.format, args.team_set)
    if args.raw_team_files:
        _prepack_explicit_teams(teams, args.team_limit)
    elif args.team_limit is not None:
        teams.team_files = teams.team_files[: max(1, args.team_limit)]

    opponent = RandomBatchedOpponent(
        num_lanes=args.lanes,
        action_dim=action_space.gym_space.n,
        rng=np.random.default_rng(args.seed + 1),
    )
    env = VectorizedShowdownEnv(
        player_team_set=teams,
        opponent_team_set=teams,
        opponent=opponent,
        opponent_obs_space=obs_space,
        opponent_action_space=action_space,
        eval_obs_space=obs_space,
        eval_action_space=action_space,
        eval_reward_function=reward,
        opponent_reward_function=reward,
        batched_envs=args.lanes,
        battle_format=args.format,
        turn_limit=args.turn_limit,
        opponent_model_name="legal-random",
        n_workers=args.workers,
        seed=args.seed,
    )

    wins = losses = draws = timeouts = vector_steps = lane_steps = 0
    completed = 0
    last_report = 0
    rng = np.random.default_rng(args.seed)
    try:
        reset_started = time.perf_counter()
        _obs, info = env.reset(seed=args.seed)
        reset_seconds = time.perf_counter() - reset_started
        started = time.perf_counter()
        while completed < args.games:
            actions = _legal_random_actions(info["legal_actions"], rng)
            _obs, _rewards, terminated, truncated, info = env.step(actions)
            vector_steps += 1
            lane_steps += args.lanes
            done = np.logical_or(terminated, truncated)
            if done.any() and ("won" not in info or "draw" not in info):
                raise RuntimeError(
                    "native env returned a terminal lane without outcome metadata: "
                    f"terminated={np.flatnonzero(terminated).tolist()} "
                    f"truncated={np.flatnonzero(truncated).tolist()} "
                    f"info_keys={sorted(info)}"
                )
            for lane in np.flatnonzero(done):
                if completed >= args.games:
                    break
                lane = int(lane)
                completed += 1
                if bool(truncated[lane]):
                    timeouts += 1
                elif bool(info["draw"][lane]):
                    draws += 1
                elif bool(info["won"][lane]):
                    wins += 1
                else:
                    losses += 1
            if completed - last_report >= args.progress_every:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"Progress: {completed} games, "
                    f"{60.0 * completed / elapsed:.1f} games/min",
                    flush=True,
                )
                last_report = completed

        elapsed = time.perf_counter() - started
        decisive = wins + losses
        print("\nNative legal-random self-play benchmark")
        print(f"Format: {args.format}")
        print(f"Team set: {args.team_set}")
        print(f"Lanes / native workers: {args.lanes} / {args.workers}")
        print(f"Reset seconds: {reset_seconds:.3f}")
        print(f"Measured seconds: {elapsed:.3f}")
        print(f"Completed games: {completed}")
        print(f"Decisive games: {decisive}")
        print(f"Player-one wins / losses: {wins} / {losses}")
        print(f"Natural draws: {draws}")
        print(f"Decision-limit timeouts: {timeouts}")
        print(f"Completed games per minute: {60.0 * completed / elapsed:.1f}")
        print(f"Decisive games per minute: {60.0 * decisive / elapsed:.1f}")
        print(f"Vector steps: {vector_steps}")
        print(f"Lane-steps per second: {lane_steps / elapsed:.1f}")
        if not args.no_profile:
            print("\nAdapter and native boundary profile")
            print(env.profile_report())
            env._profile_reported = True
    finally:
        env.close()


if __name__ == "__main__":
    main()
