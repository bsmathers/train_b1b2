"""Run a B1/B2-vs-TaurosV0 matchup with the local Zig @pkmn/engine.

No Pokémon Showdown websocket/server. Two AMAGO policies act in the same
VectorizedShowdownEnv, whose Gen 1 host uses the native Zig addon.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def require_native_engine() -> None:
    """Fail before downloading model weights if the Zig engine is absent."""
    root = Path(__file__).resolve().parents[3]
    host_dir = root / "metamon" / "env" / "vectorized"
    addon = os.environ.get("METAMON_PKMN_ENGINE")
    native = Path(addon) if addon else host_dir / "pkmn-showdown.node"
    if not native.is_file():
        raise RuntimeError(
            f"Native Zig addon not found: {native}. "
            "Run ./tools/setup_pkmn_engine.sh from the repository root."
        )
    if not (host_dir / "node_modules" / "@pkmn" / "engine").exists():
        raise RuntimeError(
            "Missing Node dependencies for the Zig host. "
            "Run ./tools/setup_pkmn_engine.sh."
        )
    # These are read at import time by sim_process.py / battle_host_engine.js.
    os.environ["METAMON_BATTLE_HOST"] = str(host_dir / "battle_host_engine.js")
    os.environ["METAMON_PKMN_ENGINE"] = str(native.resolve())
    os.environ["METAMON_REQUIRE_PKMN_ENGINE"] = "1"
    # Fail rather than scoring model actions silently repaired by an adapter.
    # Parser errors must not discard historical battle events during selection.
    os.environ.setdefault("METAMON_NATIVE_STRICT_ACTIONS", "1")
    os.environ.setdefault("METAMON_NATIVE_STRICT_PARSER", "1")


def get_candidate_model(model_name: str, checkpoint: int):
    from metamon.rl.pretrained import (
        LocalFinetunedModel,
        TaurosV1,
        get_pretrained_model,
    )

    if model_name not in ("Stage1B1", "Stage1B2"):
        return get_pretrained_model(model_name)
    prefix = "B2" if model_name == "Stage1B2" else "B1"
    return LocalFinetunedModel(
        base_model=TaurosV1,
        amago_ckpt_dir=os.environ.get(
            prefix + "_SAVE_DIR", str(Path("checkpoints").resolve())
        ),
        model_name=os.environ.get(
            prefix + "_RUN_NAME",
            "stage1_b2_selection" if prefix == "B2" else "stage1_b1",
        ),
        default_checkpoint=checkpoint,
        model_gin_config="grouped_v2_medium_cuda.gin",
        train_gin_config="stage1_b1_imitation.gin",
    )


def finish_score(done, truncated, info, lane: int) -> float:
    """Score B1 outcome; timeout/tie is half a point, not a B1 loss."""
    if not done:
        raise ValueError("Cannot score a running battle")
    draw = info.get("draw", [False] * len(truncated))[lane]
    if truncated[lane] or draw:
        return 0.5
    won = info.get("won")
    if won is None or won[lane] is None:
        raise RuntimeError("Native engine did not report a completed battle outcome")
    return 1.0 if won[lane] else 0.0


def validate_native_inputs(env, obs, action_dim: int) -> None:
    """Check private request ownership and the exact model legality mask.

    These invariants are inexpensive enough to run on every decision batch.
    They catch p1/p2 request mixups and differences between the legal actions
    the model sees and those used by the Python action adapter.
    """
    masks = obs.get("illegal_actions")
    if masks is None or len(masks) != env.batched_envs:
        raise RuntimeError("Native evaluation observation has no per-lane action mask")
    for i, lane in enumerate(env.lanes):
        for side in ("p1", "p2"):
            request = lane.last_request.get(side)
            own = (request or {}).get("side") or {}
            if own.get("id") != side:
                raise RuntimeError(
                    f"Native input ownership mismatch lane={i} side={side}: "
                    f"request side.id={own.get('id')!r}"
                )
            for pokemon in own.get("pokemon", []):
                if not pokemon.get("ident", "").startswith(side + ": "):
                    raise RuntimeError(
                        f"Native private team leaked/mislabeled lane={i} side={side}: "
                        f"{pokemon.get('ident')!r}"
                    )
        legal = set(lane.legal_action_indices(env.eval_side, env.eval_action_space))
        if not legal:
            raise RuntimeError(
                f"Native evaluation lane={i} has no legal action at decision"
            )
        mask = masks[i]
        if len(mask) != action_dim:
            raise RuntimeError(
                f"Native action mask width={len(mask)}; expected {action_dim}"
            )
        for idx in range(action_dim):
            if bool(mask[idx]) != (idx not in legal):
                raise RuntimeError(
                    f"Native action mask mismatch lane={i} action={idx}; "
                    f"mask={bool(mask[idx])}, legal={idx in legal}"
                )


def run(args):
    require_native_engine()
    import numpy as np
    import torch

    from metamon.env import get_metamon_teams
    from metamon.env.vectorized.vector_env import VectorizedShowdownEnv
    from metamon.env.vectorized.opponent import AmagoBatchedOpponent
    from metamon.env.vectorized.obs_utils import unstack_obs_dicts
    from metamon.rl.pretrained import get_pretrained_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lanes = min(args.lanes, args.battles)
    candidate_model = get_candidate_model(args.model, args.checkpoint)
    opponent_model = get_pretrained_model("TaurosV0")
    print(f"[native] Loading {args.model} epoch {args.checkpoint}", flush=True)
    candidate_exp = candidate_model.initialize_agent(
        checkpoint=args.checkpoint, action_temperature=1.0, log=False
    )
    print("[native] Loading TaurosV0 epoch 62", flush=True)
    opponent_exp = opponent_model.initialize_agent(
        checkpoint=62, action_temperature=1.0, log=False
    )
    candidate_policy = candidate_exp.policy.to(device).eval()
    opponent_policy = opponent_exp.policy.to(device).eval()
    # Policy actions are sampled, not argmax. Seed PyTorch after both model
    # loads so the same test configuration has reproducible sampling streams.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    candidate = AmagoBatchedOpponent(
        policy=candidate_policy,
        device=device,
        num_lanes=lanes,
        action_dim=candidate_model.action_space.gym_space.n,
        sample=True,
    )
    opponent = AmagoBatchedOpponent(
        policy=opponent_policy,
        device=device,
        num_lanes=lanes,
        action_dim=opponent_model.action_space.gym_space.n,
        sample=True,
    )
    env = VectorizedShowdownEnv(
        player_team_set=get_metamon_teams("gen1ou", args.candidate_team),
        opponent_team_set=get_metamon_teams("gen1ou", args.opponent_team),
        opponent=opponent,
        opponent_obs_space=opponent_model.observation_space,
        opponent_action_space=opponent_model.action_space,
        eval_obs_space=candidate_model.observation_space,
        eval_action_space=candidate_model.action_space,
        eval_reward_function=candidate_model.reward_function,
        opponent_reward_function=opponent_model.reward_function,
        battle_format="gen1ou",
        batched_envs=lanes,
        eval_player_side=args.side,
        turn_limit=args.turn_limit,
        n_workers=args.workers,
        seed=args.seed,
    )
    # Fix the number of games *per lane*: stopping at the first N completions
    # overweights quick finishes (a systematic outcome-dependent selection bias).
    quotas = np.full(lanes, args.battles // lanes, dtype=np.int64)
    quotas[: args.battles % lanes] += 1
    counts = np.zeros(lanes, dtype=np.int64)
    scores = []
    steps = 0
    try:
        obs, _ = env.reset(seed=args.seed)
        while np.any(counts < quotas):
            steps += 1
            if steps > args.max_steps:
                raise RuntimeError(
                    f"Native matchup exceeded {args.max_steps} vector steps; "
                    f"finished {len(scores)} of {args.battles} games"
                )
            validate_native_inputs(env, obs, candidate_model.action_space.gym_space.n)
            active = np.ones(lanes, dtype=bool)
            actions = candidate.act(active, unstack_obs_dicts(obs))
            obs, reward, terminated, truncated, info = env.step(actions)
            for i in range(lanes):
                committed = env._committed_side_actions.get(
                    (i, env.eval_side), int(actions[i])
                )
                candidate.observe(i, float(reward[i]), int(committed))
            done = np.logical_or(terminated, truncated)
            for i in np.flatnonzero(done):
                if counts[i] >= quotas[i]:
                    continue
                scores.append(finish_score(True, truncated, info, int(i)))
                counts[i] += 1
            candidate.reset_lanes(done)
            if (
                len(scores)
                and (len(scores) % 10 == 0 or len(scores) == args.battles)
                and getattr(run, "_last_printed", None) != len(scores)
            ):
                print(
                    f"[native] {len(scores)}/{args.battles} battles finished, "
                    f"score={sum(scores) / len(scores):.3f}",
                    flush=True,
                )
                run._last_printed = len(scores)
    finally:
        env.close()

    wins = scores.count(1.0)
    draws = scores.count(0.5)
    losses = scores.count(0.0)
    assert wins + draws + losses == args.battles
    output = {
        "battles": args.battles,
        "candidate_wins": wins,
        "opponent_wins": losses,
        "draws": draws,
        "candidate_score": (wins + 0.5 * draws) / args.battles,
        "engine": "zig-pkmn",
        "side": args.side,
        "steps": steps,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(output_path)
    print(f"[native] Completed: {json.dumps(output)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("Stage1B1", "Stage1B2"), required=True)
    p.add_argument("--checkpoint", type=int, required=True)
    p.add_argument("--candidate-team", required=True)
    p.add_argument("--opponent-team", required=True)
    p.add_argument("--side", type=int, choices=(0, 1), required=True)
    p.add_argument("--battles", type=int, required=True)
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--turn-limit", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.battles < 1 or args.lanes < 1 or args.workers < 1:
        p.error("battles, lanes and workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
