"""Profile VectorizedShowdownEnv step() throughput at high lane counts.

Usage:
    cd metamon/env/vectorized && npm ci   # once
    python -m metamon.env.vectorized._profile_step --lanes 128 --steps 100

Set METAMON_VEC_PROFILE=1 (on by default here) for a per-section timing breakdown
printed when the env closes.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from metamon.interface import (
    DefaultActionSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)
from metamon.env.vectorized import RandomBatchedOpponent, VectorizedShowdownEnv
from metamon.env import get_metamon_teams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument(
        "--team-set",
        help="Named team cache used by formats that do not provide random teams.",
    )
    parser.add_argument(
        "--raw-team-files",
        action="store_true",
        help="Profile-only shortcut for explicit team files; bypass prediction normalization.",
    )
    parser.add_argument(
        "--team-limit",
        type=int,
        help="Restrict profiling to the first N team files (useful for cache tests).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of Node processes (lanes split across workers)",
    )
    parser.add_argument(
        "--no-section-profile",
        action="store_true",
        help="Disable METAMON_VEC_PROFILE section breakdown",
    )
    args = parser.parse_args()

    if not args.no_section_profile:
        os.environ["METAMON_VEC_PROFILE"] = "1"

    n = int(args.lanes)
    obs_space = DefaultObservationSpace()
    act_space = DefaultActionSpace()
    reward = DefaultShapedReward()
    action_dim = act_space.gym_space.n

    opponent = RandomBatchedOpponent(
        num_lanes=n, action_dim=action_dim, rng=np.random.default_rng(args.seed)
    )
    teams = get_metamon_teams(args.format, args.team_set) if args.team_set else None
    if teams is not None and args.team_limit is not None:
        teams.team_files = teams.team_files[: max(1, args.team_limit)]
    if teams is not None and args.raw_team_files:
        for file in teams.team_files:
            raw = Path(file).read_text(encoding="utf-8")
            teams._packed_team_cache[file] = teams.join_team(
                teams.parse_showdown_team(raw)
            )
    env = VectorizedShowdownEnv(
        player_team_set=teams,
        opponent_team_set=teams,
        opponent=opponent,
        opponent_obs_space=obs_space,
        opponent_action_space=act_space,
        eval_obs_space=obs_space,
        eval_action_space=act_space,
        eval_reward_function=reward,
        opponent_reward_function=reward,
        batched_envs=n,
        battle_format=args.format,
        turn_limit=200,
        seed=args.seed,
        n_workers=args.n_workers,
    )

    try:
        t_reset0 = time.perf_counter()
        obs, info = env.reset()
        reset_s = time.perf_counter() - t_reset0
        print(f"reset: {reset_s:.3f}s ({n} lanes, {args.format})")

        rng = np.random.default_rng(args.seed)
        total_done = 0
        t_steps0 = time.perf_counter()
        for step in range(args.steps):
            actions = []
            for la in info["legal_actions"]:
                actions.append(int(rng.choice(la)) if la else 0)
            obs, rewards, terminated, truncated, info = env.step(np.array(actions))
            total_done += int((terminated | truncated).sum())

        steps_s = time.perf_counter() - t_steps0
        print(
            f"step: {args.steps} steps in {steps_s:.3f}s "
            f"({args.steps / steps_s:.2f} steps/s, "
            f"{args.steps * n / steps_s:.1f} lane-steps/s); "
            f"battles finished={total_done}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
