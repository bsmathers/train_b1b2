"""High-throughput native collection infrastructure for Ataraxos self-play.

The collector supports the repository's conservative reference updater as well
as a replaceable synchronous update hook. This module owns the contracts that
must remain correct regardless of later Ataraxos loss refinements:

* every battle runs through the native Zig ``@pkmn/engine`` host;
* both policies use batched transformer inference with persistent KV state;
* the learner-side policy performs one device-to-host transfer per vector step;
* every action is stored with its exact behavior distribution, value and mask;
* replay compression overlaps simulation through a bounded writer pool; and
* each round is tied to an immutable, content-addressed behavior checkpoint.

``--update-command`` is a synchronous one-shot hook.  It receives paths through
environment variables and must write the next checkpoint path as one plain-text
line to ``$ATARAXOS_NEXT_CHECKPOINT_FILE``.  No scheduler or background service
is involved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from metamon.rl.ataraxos_replay import validate_buffer


def _require_native_engine() -> None:
    from metamon.rl.evaluate.native_matchup import require_native_engine

    require_native_engine()


def _checkpoint_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"sha256-{digest.hexdigest()}"


def _infer_local_run(checkpoint: Path) -> tuple[Path, str]:
    """Infer ``save root`` and ``run name`` from an AMAGO policy path."""
    parts = checkpoint.resolve().parts
    try:
        ckpts_index = len(parts) - 1 - parts[::-1].index("ckpts")
    except ValueError as exc:
        raise ValueError(
            "checkpoint path must be inside <save-root>/<run-name>/ckpts; "
            "pass a standard actor-training checkpoint"
        ) from exc
    if ckpts_index < 2:
        raise ValueError(f"cannot infer run directory from {checkpoint}")
    run_name = parts[ckpts_index - 1]
    save_root = Path(*parts[: ckpts_index - 1])
    return save_root, run_name


def _load_stage1_policy(
    checkpoint: Path,
    temperature: float,
    device: torch.device,
    *,
    architecture_save_root: str | None = None,
    architecture_run_name: str | None = None,
):
    """Construct the B1 architecture without loading any Epoch475 weights."""
    from metamon.rl.evaluate.native_matchup import get_candidate_model

    if architecture_save_root is None:
        save_root, inferred_run_name = _infer_local_run(checkpoint)
    else:
        save_root = Path(architecture_save_root).expanduser().resolve()
        inferred_run_name = "stage1_b1"
    run_name = architecture_run_name or inferred_run_name
    if not (save_root / run_name / "ckpts").is_dir():
        raise FileNotFoundError(
            "architecture run directory was not found: "
            f"{save_root / run_name / 'ckpts'}"
        )
    os.environ["B1_SAVE_DIR"] = str(save_root)
    os.environ["B1_RUN_NAME"] = run_name
    # TaurosV1 is consulted only for observation/action/architecture metadata by
    # LocalFinetunedModel.  Point it at the same provisioned checkpoint root.
    os.environ.setdefault("TAUROSV1_SAVE_DIR", str(save_root))
    model = get_candidate_model("Stage1B1", 0)
    experiment = model.initialize_agent(
        checkpoint=0,
        action_temperature=temperature,
        log=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "policy" in state:
        state = state["policy"]
    model._validate_checkpoint(state, experiment.policy.state_dict())
    experiment.policy.load_state_dict(state, strict=True)
    # Do not call Stage1B1Agent.on_checkpoint_loaded(): it creates frozen
    # training anchors that collection never uses and needlessly duplicates the
    # backbone in host memory.
    policy = experiment.policy.eval().to(device)
    if not hasattr(policy, "wld_value"):
        raise TypeError("Ataraxos collection requires a categorical W/L/D value head")
    return model, policy


class NativeBehaviorDriver:
    """Direct batched policy adapter with one GPU-to-host boundary per step."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        lanes: int,
        action_dim: int,
        *,
        sample: bool,
        mixed_precision: str,
    ):
        self.policy = policy
        self.device = device
        self.lanes = int(lanes)
        self.action_dim = int(action_dim)
        self.sample = bool(sample)
        self.mixed_precision = mixed_precision
        self.rl2s = torch.zeros(
            (self.lanes, 1, self.action_dim + 1),
            device=device,
            dtype=torch.float32,
        )
        self.time_idxs = torch.zeros(
            (self.lanes, 1, 1), device=device, dtype=torch.long
        )
        self.hidden_state = policy.traj_encoder.init_hidden_state(self.lanes, device)

    def _caster(self):
        enabled = self.device.type == "cuda" and self.mixed_precision != "no"
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def act(self, obs: dict[str, np.ndarray]):
        torch_obs = {
            key: torch.from_numpy(value).to(self.device, non_blocking=True).unsqueeze(1)
            for key, value in obs.items()
        }
        with torch.inference_mode(), self._caster():
            states, self.hidden_state = self.policy.get_state_embedding(
                obs=torch_obs,
                rl2s=self.rl2s,
                time_idxs=self.time_idxs,
                hidden_state=self.hidden_state,
            )
            dist = self.policy.actor(
                states,
                straight_from_obs={
                    key: torch_obs[key]
                    for key in self.policy.pass_obs_keys_to_actor
                },
            )
            probs = dist.probs[..., -1, :][:, -1, :].float()
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-30)
            if self.sample:
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                actions = probs.argmax(-1)
            value_probs = self.policy.wld_value(states[:, -1, :]).float().softmax(-1)
            legal = ~torch_obs["illegal_actions"][:, -1].bool()
            log_probs = torch.where(
                probs > 0,
                probs.clamp_min(1e-30).log(),
                torch.full_like(probs, -70.0),
            )
            # This is the only device synchronization in the evaluated policy's
            # decision path.  Four small results share one packed transfer.
            packed = torch.cat(
                (
                    actions[:, None].float(),
                    log_probs,
                    value_probs,
                    legal.float(),
                ),
                dim=-1,
            ).cpu().numpy()
        action_end = 1
        logp_end = action_end + self.action_dim
        value_end = logp_end + 3
        return (
            packed[:, 0].astype(np.int64, copy=False),
            packed[:, action_end:logp_end],
            packed[:, logp_end:value_end],
            packed[:, value_end:].astype(bool, copy=False),
        )

    def observe(self, rewards: np.ndarray, committed_actions: np.ndarray) -> None:
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        actions_t = torch.as_tensor(
            committed_actions, device=self.device, dtype=torch.long
        )
        self.rl2s.zero_()
        self.rl2s[:, 0, 0] = rewards_t
        self.rl2s[:, 0, 1:].scatter_(1, actions_t[:, None], 1.0)
        self.time_idxs.add_(1)

    def reset_lanes(self, done: np.ndarray) -> None:
        if not np.any(done):
            return
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self.rl2s[done_t] = 0
        self.time_idxs[done_t] = 0
        self.hidden_state = self.policy.traj_encoder.reset_hidden_state(
            self.hidden_state, done_t
        )


class JointNativePolicyDriver:
    """One policy forward for both physical players with independent state."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        lanes: int,
        action_dim: int,
        *,
        sample: bool,
        mixed_precision: str,
    ):
        self.policy = policy
        self.device = device
        self.lanes = int(lanes)
        self.action_dim = int(action_dim)
        self.sample = bool(sample)
        self.mixed_precision = mixed_precision
        self.rl2s = np.zeros(
            (2 * self.lanes, self.action_dim + 1), dtype=np.float32
        )
        self.time_idxs = np.zeros((2 * self.lanes,), dtype=np.int64)
        self.hidden_state = policy.traj_encoder.init_hidden_state(
            2 * self.lanes, device
        )
        self._prefetched_active: np.ndarray | None = None
        self._prefetched_actions: np.ndarray | None = None

    def _caster(self):
        enabled = self.device.type == "cuda" and self.mixed_precision != "no"
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        return torch.autocast(
            device_type=self.device.type, dtype=dtype, enabled=enabled
        )

    @staticmethod
    def _hidden_view(hidden_state, start: int, stop: int):
        if hidden_state is None:
            return None
        key_cache = copy.copy(hidden_state.key_cache)
        val_cache = copy.copy(hidden_state.val_cache)
        key_cache.data = hidden_state.key_cache.data[:, start:stop]
        val_cache.data = hidden_state.val_cache.data[:, start:stop]
        return type(hidden_state)(
            key_cache, val_cache, hidden_state.seq_lens[start:stop]
        )

    @staticmethod
    def _snapshot_hidden(hidden_state, inactive: np.ndarray):
        if hidden_state is None or not inactive.any():
            return None
        idx = torch.as_tensor(
            np.where(inactive)[0], device=hidden_state.device, dtype=torch.long
        )
        return (
            idx,
            hidden_state.seq_lens[idx].clone(),
            hidden_state.key_cache.data[:, idx].clone(),
            hidden_state.val_cache.data[:, idx].clone(),
        )

    @staticmethod
    def _restore_hidden(hidden_state, saved) -> None:
        if hidden_state is None or saved is None:
            return
        idx, seq_lens, keys, values = saved
        hidden_state.seq_lens[idx] = seq_lens
        hidden_state.key_cache.data[:, idx] = keys
        hidden_state.val_cache.data[:, idx] = values

    def _forward(self, obs, start: int, stop: int, active: np.ndarray):
        torch_obs = {
            key: torch.from_numpy(value).to(self.device, non_blocking=True).unsqueeze(1)
            for key, value in obs.items()
        }
        rl2s = torch.from_numpy(self.rl2s[start:stop]).to(self.device).unsqueeze(1)
        time_idxs = (
            torch.from_numpy(self.time_idxs[start:stop])
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        hidden = self._hidden_view(self.hidden_state, start, stop)
        saved = self._snapshot_hidden(hidden, ~active)
        with torch.inference_mode(), self._caster():
            states, hidden = self.policy.get_state_embedding(
                obs=torch_obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=hidden,
            )
            dist = self.policy.actor(
                states,
                straight_from_obs={
                    key: torch_obs[key]
                    for key in self.policy.pass_obs_keys_to_actor
                },
            )
            probs = dist.probs[..., -1, :][:, -1, :].float()
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-30)
            actions = (
                torch.multinomial(probs, num_samples=1).squeeze(-1)
                if self.sample
                else probs.argmax(-1)
            )
        self._restore_hidden(hidden, saved)
        return states, probs, actions, torch_obs

    def act_both(
        self,
        eval_obs: dict[str, np.ndarray],
        eval_active: np.ndarray,
        opponent_active: np.ndarray,
        opponent_obs_list: list[dict],
    ):
        from metamon.env.vectorized.obs_utils import stack_obs_dicts

        opponent_obs = stack_obs_dicts(opponent_obs_list)
        combined_obs = {
            key: np.concatenate((eval_obs[key], opponent_obs[key]), axis=0)
            for key in eval_obs
        }
        active = np.concatenate((eval_active, opponent_active))
        states, probs, actions, torch_obs = self._forward(
            combined_obs, 0, 2 * self.lanes, active
        )
        with torch.inference_mode(), self._caster():
            eval_probs = probs[: self.lanes]
            value_probs = self.policy.wld_value(
                states[: self.lanes, -1, :]
            ).float().softmax(-1)
            legal = ~torch_obs["illegal_actions"][: self.lanes, -1].bool()
            log_probs = torch.where(
                eval_probs > 0,
                eval_probs.clamp_min(1e-30).log(),
                torch.full_like(eval_probs, -70.0),
            )
            packed = torch.cat(
                (
                    actions[: self.lanes, None].float(),
                    log_probs,
                    value_probs,
                    legal.float(),
                    actions[self.lanes :, None].float(),
                ),
                dim=-1,
            ).cpu().numpy()
        action_end = 1
        logp_end = action_end + self.action_dim
        value_end = logp_end + 3
        legal_end = value_end + self.action_dim
        if opponent_active.any():
            self._prefetched_active = opponent_active.copy()
            self._prefetched_actions = packed[:, legal_end].astype(
                np.int64, copy=False
            )
        else:
            self._prefetched_active = None
            self._prefetched_actions = None
        return (
            packed[:, 0].astype(np.int64, copy=False),
            packed[:, action_end:logp_end],
            packed[:, logp_end:value_end],
            packed[:, value_end:legal_end].astype(bool, copy=False),
        )

    def has_prefetched_actions(self, active: np.ndarray) -> bool:
        return self._prefetched_active is not None and np.array_equal(
            active, self._prefetched_active
        )

    def act(self, active: np.ndarray, obs_list: list[dict]) -> np.ndarray:
        if self._prefetched_active is not None:
            if not np.array_equal(active, self._prefetched_active):
                raise RuntimeError("prefetched opponent active mask changed before step")
            actions = self._prefetched_actions
            self._prefetched_active = None
            self._prefetched_actions = None
            assert actions is not None
            return actions
        from metamon.env.vectorized.obs_utils import stack_obs_dicts

        _states, _probs, actions, _torch_obs = self._forward(
            stack_obs_dicts(obs_list), self.lanes, 2 * self.lanes, active
        )
        return actions.cpu().numpy().astype(np.int64, copy=False)

    def _observe_slice(
        self,
        lanes: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
        offset: int,
    ) -> None:
        idx = np.asarray(lanes, dtype=np.int64) + offset
        rewards = np.asarray(rewards, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        self.time_idxs[idx] += 1
        self.rl2s[idx] = 0.0
        self.rl2s[idx, 0] = rewards
        valid = (actions >= 0) & (actions < self.action_dim)
        self.rl2s[idx[valid], 1 + actions[valid]] = 1.0

    def observe_eval(self, rewards: np.ndarray, actions: np.ndarray) -> None:
        self._observe_slice(
            np.arange(self.lanes), rewards, actions, offset=0
        )

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self.observe_batch(
            np.asarray([lane_idx]),
            np.asarray([reward]),
            np.asarray([action_idx]),
        )

    def observe_batch(
        self, lane_indices: np.ndarray, rewards: np.ndarray, action_indices: np.ndarray
    ) -> None:
        self._observe_slice(
            lane_indices, rewards, action_indices, offset=self.lanes
        )

    def _reset_slice(self, done: np.ndarray, offset: int) -> None:
        if not np.any(done):
            return
        idx = np.where(done)[0] + offset
        self.rl2s[idx] = 0.0
        self.time_idxs[idx] = 0
        reset = np.zeros((2 * self.lanes,), dtype=bool)
        reset[idx] = True
        self.hidden_state = self.policy.traj_encoder.reset_hidden_state(
            self.hidden_state, torch.as_tensor(reset, device=self.device)
        )

    def reset_eval_lanes(self, done: np.ndarray) -> None:
        self._reset_slice(done, 0)

    def reset_lanes(self, done: np.ndarray) -> None:
        self._reset_slice(done, self.lanes)

    def reset_all(self) -> None:
        self.rl2s.fill(0.0)
        self.time_idxs.fill(0)
        self.hidden_state = self.policy.traj_encoder.init_hidden_state(
            2 * self.lanes, self.device
        )
        self._prefetched_active = None
        self._prefetched_actions = None


def _publish_behavior_checkpoint(checkpoint: Path, policy_root: Path) -> tuple[str, Path]:
    policy_id = _checkpoint_identity(checkpoint)
    policy_root.mkdir(parents=True, exist_ok=True)
    published = policy_root / f"{policy_id}.pt"
    if not published.exists():
        temporary = published.with_suffix(".pt.tmp")
        shutil.copy2(checkpoint, temporary)
        os.replace(temporary, published)
    return policy_id, published


def _write_round_text(path: Path, fields: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{key}: {value}\n" for key, value in fields.items()),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_round_text(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(": ")
            if separator:
                fields[key] = value
    return fields


def _update_round_text(path: Path, **updates: object) -> None:
    fields: dict[str, object] = dict(_read_round_text(path))
    fields.update(updates)
    _write_round_text(path, fields)


def collect_round(
    args, checkpoint: Path, round_index: int
) -> tuple[Path, dict, Path]:
    profile: dict[str, float] = {}

    def timed(name: str, started_at: float) -> None:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started_at)

    if args.profile:
        os.environ["METAMON_VEC_PROFILE"] = "1"
    _require_native_engine()
    from metamon.env import get_metamon_team_mixture, get_metamon_teams
    from metamon.env.vectorized.opponent import AmagoBatchedOpponent
    from metamon.env.vectorized.vector_env import VectorizedShowdownEnv

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA collection requested but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    sample = args.temperature > 0
    policy_temperature = args.temperature if sample else 1.0
    tick = time.perf_counter()
    policy_id, immutable_checkpoint = _publish_behavior_checkpoint(
        checkpoint, Path(args.output_dir) / "behavior_policies"
    )
    timed("checkpoint_hash_and_publish", tick)
    round_dir = Path(args.output_dir) / f"round_{round_index:06d}"
    replay_root = round_dir / "replays"
    existing = list((replay_root / args.battle_format).glob("*.json.lz4"))
    existing_games = 0
    if existing:
        existing_summary = validate_buffer(
            replay_root, workers=args.validation_workers
        )
        existing_games = int(existing_summary["games"])
        if existing_summary["policy_id"] != policy_id:
            raise RuntimeError(
                f"round {round_index} already contains a different behavior policy"
            )
        if existing_games >= args.games:
            print(
                f"[Ataraxos] round={round_index} already complete "
                f"games={existing_summary['games']}",
                flush=True,
            )
            return round_dir, existing_summary, immutable_checkpoint
        remaining = args.games - existing_games
    else:
        remaining = args.games
    round_dir.mkdir(parents=True, exist_ok=True)

    tick = time.perf_counter()
    model, policy = _load_stage1_policy(
        checkpoint,
        temperature=policy_temperature,
        device=device,
        architecture_save_root=args.architecture_save_root,
        architecture_run_name=args.architecture_run_name,
    )
    timed("model_construction_and_load", tick)
    action_dim = model.action_space.gym_space.n
    # Sharing read-only weights between the two policy drivers halves model
    # memory.  Each side owns independent RL2/KV state.
    if args.joint_inference:
        driver = JointNativePolicyDriver(
            policy,
            device,
            args.lanes,
            action_dim,
            sample=sample,
            mixed_precision=args.mixed_precision,
        )
        opponent = driver
    else:
        opponent = AmagoBatchedOpponent(
            policy=policy,
            device=device,
            num_lanes=args.lanes,
            action_dim=action_dim,
            sample=sample,
            mixed_precision=args.mixed_precision,
        )
        driver = NativeBehaviorDriver(
            policy,
            device,
            args.lanes,
            action_dim,
            sample=sample,
            mixed_precision=args.mixed_precision,
        )
    tick = time.perf_counter()
    team_set_names = [
        name.strip() for name in args.team_set.split(",") if name.strip()
    ]
    if not team_set_names:
        raise ValueError("Ataraxos team-set mixture is empty")
    teams = (
        get_metamon_teams(args.battle_format, team_set_names[0])
        if len(team_set_names) == 1
        else get_metamon_team_mixture(
            args.battle_format,
            team_set_names,
            weights=[1.0] * len(team_set_names),
        )
    )
    timed("team_load", tick)
    tick = time.perf_counter()
    env = VectorizedShowdownEnv(
        player_team_set=teams,
        opponent_team_set=copy.deepcopy(teams),
        opponent=opponent,
        opponent_obs_space=model.observation_space,
        opponent_action_space=model.action_space,
        eval_obs_space=model.observation_space,
        eval_action_space=model.action_space,
        eval_reward_function=model.reward_function,
        opponent_reward_function=model.reward_function,
        batched_envs=args.lanes,
        battle_format=args.battle_format,
        turn_limit=args.turn_limit,
        opponent_model_name="behavior-self-play",
        eval_player_side=(
            round_index % 2 if args.side == "alternate" else int(args.side)
        ),
        save_trajectories_to=str(replay_root),
        n_workers=args.workers,
        seed=args.seed + round_index,
        trajectory_writer_workers=args.writer_workers,
    )
    timed("environment_construction", tick)
    env.set_behavior_policy_id(policy_id)
    env.set_collection_game_target(remaining)
    started = time.monotonic()
    vector_steps = 0
    try:
        tick = time.perf_counter()
        obs, _ = env.reset(seed=args.seed + round_index)
        timed("environment_reset", tick)
        while not env.collection_game_target_reached():
            if args.joint_inference:
                tick = time.perf_counter()
                eval_active, opponent_active, opponent_obs = (
                    env.opponent_decision_inputs()
                )
                timed("opponent_observation", tick)
                tick = time.perf_counter()
                actions, log_probs, values, legal = driver.act_both(
                    obs, eval_active, opponent_active, opponent_obs
                )
            else:
                tick = time.perf_counter()
                actions, log_probs, values, legal = driver.act(obs)
            timed("evaluated_policy", tick)
            tick = time.perf_counter()
            env.record_policy_metadata(log_probs, values, legal)
            timed("metadata_validation", tick)
            tick = time.perf_counter()
            obs, rewards, terminated, truncated, _ = env.step(actions)
            timed("environment_step", tick)
            tick = time.perf_counter()
            committed = np.asarray(
                [
                    env._committed_side_actions.get((lane, env.eval_side), actions[lane])
                    for lane in range(args.lanes)
                ],
                dtype=np.int64,
            )
            if args.joint_inference:
                driver.observe_eval(rewards, committed)
                driver.reset_eval_lanes(np.logical_or(terminated, truncated))
            else:
                driver.observe(rewards, committed)
                driver.reset_lanes(np.logical_or(terminated, truncated))
            timed("evaluated_policy_feedback", tick)
            vector_steps += 1
            if vector_steps % args.progress_interval == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                games = env.collection_game_count()
                print(
                    f"[Ataraxos] round={round_index} games={games}/{remaining} "
                    f"rate={60.0 * games / elapsed:.1f} games/min",
                    flush=True,
                )
            if vector_steps >= args.max_vector_steps:
                raise RuntimeError(
                    f"round exceeded {args.max_vector_steps} vector steps"
                )
        tick = time.perf_counter()
        env.flush_trajectory_writes()
        timed("writer_flush", tick)
    finally:
        tick = time.perf_counter()
        env.close()
        timed("environment_close", tick)

    tick = time.perf_counter()
    summary = validate_buffer(replay_root, workers=args.validation_workers)
    timed("replay_validation", tick)
    elapsed = time.monotonic() - started
    collected_games = int(summary["games"]) - existing_games
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    fields = {
        "round": round_index,
        "behavior_policy_id": policy_id,
        "behavior_checkpoint": immutable_checkpoint,
        "games": summary["games"],
        "games_collected_this_process": collected_games,
        "wins": summary["wins"],
        "draws": summary["draws"],
        "losses": summary["losses"],
        "decisions": summary["steps"],
        "vector_steps": vector_steps,
        "elapsed_seconds": f"{elapsed:.3f}",
        "games_per_minute": f"{60.0 * collected_games / max(elapsed, 1e-6):.3f}",
        "decisions_per_second": f"{int(summary['steps']) / max(elapsed, 1e-6):.3f}",
        "vector_steps_per_second": f"{vector_steps / max(elapsed, 1e-6):.3f}",
        "evaluated_policy_seconds": f"{profile.get('evaluated_policy', 0.0):.3f}",
        "environment_step_seconds": f"{profile.get('environment_step', 0.0):.3f}",
        "device": device,
        "device_name": device_name,
        "host_logical_cpus": os.cpu_count() or 0,
        "lanes": args.lanes,
        "simulator_workers": args.workers,
        "writer_workers": args.writer_workers,
        "joint_inference": args.joint_inference,
        "team_set_mixture": args.team_set,
        "behavior_temperature": args.temperature,
        "engine": "native-zig-pkmn",
        "update_status": "pending",
    }
    _write_round_text(round_dir / "round-summary.txt", fields)
    print(
        f"[Ataraxos] round={round_index} complete games={summary['games']} "
        f"wins={summary['wins']} losses={summary['losses']} "
        f"rate={fields['games_per_minute']} games/min",
        flush=True,
    )
    if args.profile:
        measured = sum(profile.values())
        print(
            f"[AtaraxosProfile] round={round_index} vector_steps={vector_steps} "
            f"measured_seconds={measured:.3f}",
            flush=True,
        )
        for name, seconds in sorted(
            profile.items(), key=lambda item: item[1], reverse=True
        ):
            per_step_ms = 1000.0 * seconds / max(vector_steps, 1)
            print(
                f"[AtaraxosProfile] {name} seconds={seconds:.3f} "
                f"share={100.0 * seconds / max(measured, 1e-9):.1f}% "
                f"per_step_ms={per_step_ms:.3f}",
                flush=True,
            )
    return round_dir, summary, immutable_checkpoint


def _run_update_hook(
    command: str,
    round_dir: Path,
    checkpoint: Path,
    *,
    parent_checkpoint: Path,
    architecture_save_root: str,
    architecture_run_name: str,
    behavior_temperature: float,
) -> Path:
    next_file = round_dir / "next-checkpoint.txt"
    if next_file.exists():
        next_file.unlink()
    env = os.environ.copy()
    env.update(
        {
            "ATARAXOS_ROUND_DIR": str(round_dir),
            "ATARAXOS_REPLAY_DIR": str(round_dir / "replays"),
            "ATARAXOS_BEHAVIOR_CHECKPOINT": str(checkpoint),
            "ATARAXOS_PARENT_CHECKPOINT": str(parent_checkpoint),
            "ATARAXOS_NEXT_CHECKPOINT_FILE": str(next_file),
            "ATARAXOS_ARCHITECTURE_SAVE_ROOT": architecture_save_root,
            "ATARAXOS_ARCHITECTURE_RUN_NAME": architecture_run_name,
            "ATARAXOS_BEHAVIOR_TEMPERATURE": str(behavior_temperature),
        }
    )
    subprocess.run(shlex.split(command), env=env, check=True)
    if not next_file.is_file():
        raise RuntimeError(
            "update command succeeded but did not write ATARAXOS_NEXT_CHECKPOINT_FILE"
        )
    next_checkpoint = Path(next_file.read_text(encoding="utf-8").strip()).resolve()
    if not next_checkpoint.is_file():
        raise RuntimeError(f"update hook returned missing checkpoint: {next_checkpoint}")
    return next_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--architecture-save-root",
        help="AMAGO save root used only to construct the Stage1B1 architecture; "
        "inferred from a standard checkpoint path when omitted.",
    )
    parser.add_argument(
        "--architecture-run-name",
        help="Architecture run below --architecture-save-root (default stage1_b1).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("collect", "pipeline", "inspect"), default="collect")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Terminal games per round; zero chooses two games per lane.",
    )
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--battle-format", default="gen1ou")
    parser.add_argument(
        "--team-set",
        default="b1_good_807,hl_05_26",
        help="One team set or a comma-separated uniform per-game mixture.",
    )
    parser.add_argument(
        "--side",
        choices=("0", "1", "alternate"),
        default="alternate",
        help="Physical side for the recorded policy; alternate balances rounds.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--turn-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="bf16")
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--max-vector-steps", type=int, default=1_000_000)
    parser.add_argument("--update-command")
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument(
        "--joint-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Batch both self-play sides into one policy forward.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print collector and native-adapter timing breakdowns.",
    )
    args = parser.parse_args()
    if args.games == 0:
        # Complete-game approximation to Ataraxos's ~100 fresh decisions per
        # player/environment. Local rollouts average about 50 recorded POV
        # decisions, so two terminal games per lane is the closest simple
        # cadence until live streaming is implemented.
        args.games = 2 * args.lanes
    if min(args.games, args.lanes, args.workers, args.rounds) < 1:
        parser.error("games, lanes, workers and rounds must be positive")
    if min(args.writer_workers, args.validation_workers) < 0 or args.temperature < 0:
        parser.error(
            "writer-workers, validation-workers and temperature must be non-negative"
        )

    if args.mode == "inspect":
        summary = validate_buffer(
            Path(args.output_dir) / f"round_{args.round:06d}" / "replays",
            workers=args.validation_workers,
        )
        print(
            "[Ataraxos] "
            + " ".join(f"{key}={value}" for key, value in summary.items()),
            flush=True,
        )
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required for collect and pipeline modes")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    # Preserve architecture identity across pipeline rounds. Updated policies
    # are published below round directories, where the standard AMAGO path can
    # no longer be inferred from the checkpoint itself.
    if args.architecture_save_root is None:
        save_root, run_name = _infer_local_run(checkpoint)
        args.architecture_save_root = str(save_root)
        args.architecture_run_name = args.architecture_run_name or run_name

    if args.mode == "collect":
        collect_round(args, checkpoint, args.round)
        return

    if not args.update_command:
        parser.error("pipeline mode requires --update-command")
    if args.temperature <= 0:
        parser.error(
            "pipeline mode requires temperature > 0 so actions are sampled "
            "from the stored behavior distribution"
        )
    for round_index in range(args.round, args.round + args.rounds):
        parent_checkpoint = checkpoint
        round_dir, _, immutable_checkpoint = collect_round(
            args, checkpoint, round_index
        )
        round_fields = _read_round_text(round_dir / "round-summary.txt")
        if round_fields.get("update_status") == "complete":
            checkpoint = Path(round_fields.get("next_checkpoint", "")).resolve()
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"round {round_index} records a missing next checkpoint: "
                    f"{checkpoint}"
                )
            print(
                f"[Ataraxos] round={round_index} update already complete "
                f"checkpoint={checkpoint}",
                flush=True,
            )
            continue
        checkpoint = _run_update_hook(
            args.update_command,
            round_dir,
            immutable_checkpoint,
            parent_checkpoint=parent_checkpoint,
            architecture_save_root=args.architecture_save_root,
            architecture_run_name=args.architecture_run_name,
            behavior_temperature=args.temperature,
        )
        _update_round_text(
            round_dir / "round-summary.txt",
            update_status="complete",
            next_checkpoint=checkpoint,
        )
        print(
            f"[Ataraxos] round={round_index} update complete checkpoint={checkpoint}",
            flush=True,
        )


if __name__ == "__main__":
    main()
