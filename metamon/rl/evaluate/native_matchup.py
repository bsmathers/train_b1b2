"""Run a B1/TaurosV1-vs-TaurosV0 matchup with local Zig @pkmn/engine.

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

    model_gin = portable_model_config("grouped_v2_medium_cuda.gin")
    if model_name == "TaurosV1":
        # LocalFinetunedModel instantiates TaurosV1() to obtain observation
        # and actor metadata; ensure that base constructor sees this checkout.
        os.environ.setdefault("TAUROSV1_SAVE_DIR", str(Path("checkpoints").resolve()))
        # The *unmodified* parent of B1. Use the same CUDA-safe model and
        # stage-1 inference gin as B1; only the checkpoint weights differ.
        # The original TaurosV1 registry entry defaults to the scratchpad
        # training path and FlashAttention model gin, neither of which is
        # appropriate for this standalone B1 native baseline.
        return LocalFinetunedModel(
            base_model=TaurosV1,
            amago_ckpt_dir=os.environ.get(
                "TAUROSV1_SAVE_DIR", str(Path("checkpoints").resolve())
            ),
            model_name="mediumg1online",
            default_checkpoint=checkpoint,
            model_gin_config=model_gin,
            train_gin_config="stage1_b1_imitation.gin",
        )
    if model_name != "Stage1B1":
        return get_pretrained_model(model_name)
    return LocalFinetunedModel(
        base_model=TaurosV1,
        amago_ckpt_dir=os.environ.get(
            "B1_SAVE_DIR", str(Path("checkpoints").resolve())
        ),
        model_name=os.environ.get("B1_RUN_NAME", "stage1_b1"),
        default_checkpoint=checkpoint,
        model_gin_config=model_gin,
        train_gin_config="stage1_b1_imitation.gin",
    )


def portable_model_config(config_name: str) -> str:
    """Use the parameter-free PyTorch attention fallback when FA2 is absent."""
    from amago.nets import transformer

    if transformer.flash_attn is not None:
        return config_name
    portable = {
        "grouped_v2_medium_cuda.gin": "grouped_v2_medium_portable.gin",
        "grouped_v2_50m.gin": "grouped_v2_50m_portable.gin",
    }
    try:
        return portable[config_name]
    except KeyError as exc:
        raise RuntimeError(
            f"No portable attention config is defined for {config_name}"
        ) from exc


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


def finish_outcome(done, truncated, info, lane: int) -> str:
    """Return an auditable terminal category instead of conflating all draws."""
    if not done:
        raise ValueError("Cannot classify a running battle")
    if truncated[lane]:
        return "timeout"
    draw = info.get("draw", [False] * len(truncated))[lane]
    if draw:
        return "draw"
    won = info.get("won")
    if won is None or won[lane] is None:
        raise RuntimeError("Native engine did not report a completed battle outcome")
    return "win" if won[lane] else "loss"


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
    portable_opponent_config = portable_model_config(opponent_model.model_gin_config)
    if portable_opponent_config != opponent_model.model_gin_config:
        opponent_model.model_gin_config = portable_opponent_config
        opponent_model.model_gin_config_path = str(
            Path(opponent_model.model_gin_config_path).with_name(
                portable_opponent_config
            )
        )
        print(
            "[native] FlashAttention unavailable; using checkpoint-compatible "
            "portable sliding-window attention",
            flush=True,
        )
    checkpoint_label = args.checkpoint_path or f"epoch {args.checkpoint}"
    print(f"[native] Loading {args.model} {checkpoint_label}", flush=True)
    # AMAGO's temperature rescales logits before constructing the action
    # distribution.  Temperature zero would divide by zero, so represent the
    # requested zero-temperature limit exactly by disabling sampling and using
    # argmax from the unscaled logits.
    sample_actions = args.temperature > 0.0
    distribution_temperature = args.temperature if sample_actions else 1.0
    candidate_exp = candidate_model.initialize_agent(
        checkpoint=(0 if args.checkpoint_path else args.checkpoint),
        action_temperature=distribution_temperature,
        log=False,
    )
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "policy" in state:
            state = state["policy"]
        candidate_model._validate_checkpoint(
            state, candidate_exp.policy.state_dict()
        )
        candidate_exp.policy.load_state_dict(state, strict=True)
        candidate_exp.policy.on_checkpoint_loaded(is_resume=False)
    print("[native] Loading TaurosV0 epoch 62", flush=True)
    opponent_exp = opponent_model.initialize_agent(
        checkpoint=62,
        action_temperature=distribution_temperature,
        log=False,
    )
    candidate_policy = candidate_exp.policy.to(device).eval()
    opponent_policy = opponent_exp.policy.to(device).eval()
    # Seed after both model loads so sampled runs are reproducible. Argmax runs
    # remain deterministic while using the same simulator/team seed handling.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    candidate = AmagoBatchedOpponent(
        policy=candidate_policy,
        device=device,
        num_lanes=lanes,
        action_dim=candidate_model.action_space.gym_space.n,
        sample=sample_actions,
    )
    opponent = AmagoBatchedOpponent(
        policy=opponent_policy,
        device=device,
        num_lanes=lanes,
        action_dim=opponent_model.action_space.gym_space.n,
        sample=sample_actions,
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
    outcomes = []
    steps = 0
    game_log = None
    if args.game_log:
        game_log_path = Path(args.game_log).expanduser().resolve()
        game_log_path.parent.mkdir(parents=True, exist_ok=True)
        game_log = game_log_path.open("a", buffering=1)
        game_log.write(
            f"Starting native matchup: candidate={args.model}, "
            f"candidate_side=p{args.side + 1}, games={args.battles}, "
            f"seed={args.seed}, action_mode="
            f"{'sample' if sample_actions else 'argmax'}, "
            f"temperature={args.temperature:g}\n"
        )
    try:
        obs, _ = env.reset(seed=args.seed)
        while np.any(counts < quotas):
            steps += 1
            if steps > args.max_steps:
                raise RuntimeError(
                    f"Native matchup exceeded {args.max_steps} vector steps; "
                    f"finished {len(outcomes)} of {args.battles} games"
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
                outcome = finish_outcome(True, truncated, info, int(i))
                outcomes.append(outcome)
                counts[i] += 1
                game_line = (
                    f"Game {len(outcomes)}/{args.battles}: "
                    f"candidate_side=p{args.side + 1}, lane={int(i)}, "
                    f"lane_game={int(counts[i])}, result={outcome}"
                )
                print(f"[native-game] {game_line}", flush=True)
                if game_log is not None:
                    game_log.write(game_line + "\n")
            candidate.reset_lanes(done)
            if (
                len(outcomes)
                and (len(outcomes) % 10 == 0 or len(outcomes) == args.battles)
                and getattr(run, "_last_printed", None) != len(outcomes)
            ):
                wins_so_far = outcomes.count("win")
                half_so_far = outcomes.count("draw") + outcomes.count("timeout")
                print(
                    f"[native] {len(outcomes)}/{args.battles} battles finished, "
                    f"score={(wins_so_far + 0.5 * half_so_far) / len(outcomes):.3f}",
                    flush=True,
                )
                run._last_printed = len(outcomes)
    finally:
        env.close()
        if game_log is not None:
            game_log.close()

    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    natural_draws = outcomes.count("draw")
    timeouts = outcomes.count("timeout")
    draws = natural_draws + timeouts
    assert wins + losses + draws == args.battles
    output = {
        "battles": args.battles,
        "candidate_wins": wins,
        "opponent_wins": losses,
        "draws": draws,
        "natural_draws": natural_draws,
        "timeouts": timeouts,
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
    print(
        f"[native] Completed {args.battles} battles: wins={wins}, "
        f"losses={losses}, natural_draws={natural_draws}, "
        f"timeouts={timeouts}, score={output['candidate_score']:.4f}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", choices=("Stage1B1", "TaurosV1"), required=True
    )
    checkpoint = p.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=int)
    checkpoint.add_argument("--checkpoint-path")
    p.add_argument("--candidate-team", required=True)
    p.add_argument("--opponent-team", required=True)
    p.add_argument("--side", type=int, choices=(0, 1), required=True)
    p.add_argument("--battles", type=int, required=True)
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Action temperature; zero selects deterministic argmax for both policies.",
    )
    p.add_argument("--turn-limit", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--output", required=True)
    p.add_argument("--game-log")
    args = p.parse_args()
    if args.battles < 1 or args.lanes < 1 or args.workers < 1:
        p.error("battles, lanes and workers must be positive")
    if args.temperature < 0:
        p.error("temperature must be non-negative")
    if args.checkpoint_path and not Path(args.checkpoint_path).expanduser().is_file():
        p.error(f"checkpoint path does not exist: {args.checkpoint_path}")
    # A path load still needs an architecture descriptor; zero means construct
    # the model without fetching/loading an epoch checkpoint.
    if args.checkpoint is None:
        args.checkpoint = 0
    run(args)


if __name__ == "__main__":
    main()
