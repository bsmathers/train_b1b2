"""One Ataraxos PPO/TD(lambda) update over a validated native round.

The learner follows the original Ataraxos objective while consuming complete
Pokemon trajectories. Complete recurrent histories keep transformer states
consistent; live-game streaming is intentionally deferred until KV histories
can be reconstructed after every policy update.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from amago.loading import RLData_pad_collate

from metamon.data.parsed_replay_dset import MetamonDataset
from metamon.rl.ataraxos import _load_stage1_policy
from metamon.rl.ataraxos_replay import AtaraxosRollout, load_rollout, rollout_paths
from metamon.rl.metamon_to_amago import MetamonAMAGODataset


@dataclass
class UpdateTrajectory:
    data: object
    behavior_log_probs: torch.Tensor
    behavior_values: torch.Tensor
    behavior_legal_masks: torch.Tensor
    outcome: int
    path: Path

    @property
    def steps(self) -> int:
        return int(self.behavior_values.shape[0])


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_text(path: Path, fields: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{key}: {value}\n" for key, value in fields.items()),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_trajectories(replay_dir: Path, model) -> list[UpdateTrajectory]:
    dataset = MetamonDataset(
        dset_root=str(replay_dir),
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        formats=["gen1ou"],
        max_seq_len=None,
        verbose=False,
        shuffle=False,
        use_cached_filenames=False,
        write_index_cache=False,
    )
    converter = MetamonAMAGODataset(dataset)
    output: list[UpdateTrajectory] = []
    for raw_path in rollout_paths(replay_dir):
        rollout: AtaraxosRollout = load_rollout(raw_path)
        obs, action_infos, rewards, dones = dataset.load_filename(str(raw_path))
        data = converter._process_data((obs, action_infos, rewards, dones))
        if len(data.actions) != rollout.steps:
            raise RuntimeError(f"{raw_path}: tokenized action length changed")
        tokenized_actions = data.actions.argmax(-1).cpu().numpy()
        if not np.array_equal(tokenized_actions, rollout.actions):
            raise RuntimeError(f"{raw_path}: action-space conversion changed actions")

        # The native private request is the authority on legality.  Rebuilding
        # legality from a public UniversalState can be less restrictive during
        # forced-choice edge cases and would change the policy distribution.
        data.obs["illegal_actions"][: rollout.steps] = torch.from_numpy(
            ~rollout.behavior_legal_masks
        )
        output.append(
            UpdateTrajectory(
                data=data,
                behavior_log_probs=torch.from_numpy(rollout.behavior_log_probs),
                behavior_values=torch.from_numpy(rollout.behavior_values),
                behavior_legal_masks=torch.from_numpy(rollout.behavior_legal_masks),
                outcome=rollout.outcome,
                path=raw_path,
            )
        )
    if not output:
        raise RuntimeError(f"no Ataraxos trajectories found under {replay_dir}")
    return output


def _to_device(batch, device: torch.device):
    batch.obs = {key: value.to(device) for key, value in batch.obs.items()}
    batch.rl2s = batch.rl2s.to(device)
    batch.rews = batch.rews.to(device)
    batch.dones = batch.dones.to(device)
    batch.actions = batch.actions.to(device)
    batch.time_idxs = batch.time_idxs.to(device)
    return batch


def _decision_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average every valid decision equally, as in the Ataraxos minibatches."""
    return values[valid].mean()


def _pad_metadata(
    samples: list[UpdateTrajectory], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    longest = max(sample.steps for sample in samples)
    action_dim = samples[0].behavior_log_probs.shape[-1]
    old_log = torch.full((len(samples), longest, action_dim), -70.0, device=device)
    old_value = torch.zeros((len(samples), longest, 3), device=device)
    valid = torch.zeros((len(samples), longest), dtype=torch.bool, device=device)
    legal = torch.zeros(
        (len(samples), longest, action_dim), dtype=torch.bool, device=device
    )
    outcomes = torch.empty((len(samples),), device=device)
    for row, sample in enumerate(samples):
        steps = sample.steps
        old_log[row, :steps] = sample.behavior_log_probs.to(device)
        old_value[row, :steps] = sample.behavior_values.to(device)
        legal[row, :steps] = sample.behavior_legal_masks.to(device)
        valid[row, :steps] = True
        outcomes[row] = float(sample.outcome)
    return old_log, old_value, legal, valid, outcomes


def categorical_gae_targets(
    old_value_probs: torch.Tensor,
    outcomes: torch.Tensor,
    valid: torch.Tensor,
    *,
    gae_lambda: float,
    td_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached GAE and categorical TD(lambda) targets.

    ``t + 1`` is the next decision in the same stored player trajectory. Since
    this first implementation uses complete games, the final bootstrap is the
    observed terminal W/L/D one-hot vector rather than a truncated bootstrap.
    """
    if old_value_probs.shape[:2] != valid.shape or old_value_probs.shape[-1] != 3:
        raise ValueError("old W/L/D probabilities are not decision-aligned")
    advantages = torch.zeros_like(valid, dtype=old_value_probs.dtype)
    targets = torch.zeros_like(old_value_probs)
    aggregation = old_value_probs.new_tensor((-1.0, 0.0, 1.0))
    lengths = valid.sum(1)
    if torch.any(lengths < 1):
        raise ValueError("empty trajectory in Ataraxos update")
    terminal = F.one_hot((outcomes.long() + 1).clamp(0, 2), num_classes=3).to(
        old_value_probs.dtype
    )
    next_advantage = old_value_probs.new_zeros((old_value_probs.shape[0],))
    next_target = terminal
    for step in range(old_value_probs.shape[1] - 1, -1, -1):
        active = valid[:, step]
        terminating = lengths == step + 1
        next_prediction = (
            terminal
            if step + 1 == old_value_probs.shape[1]
            else old_value_probs[:, step + 1]
        )
        bootstrap = torch.where(terminating[:, None], terminal, next_prediction)
        delta = ((bootstrap - old_value_probs[:, step]) * aggregation).sum(-1)
        advantage = delta + gae_lambda * next_advantage * (~terminating)
        target = torch.where(
            terminating[:, None],
            terminal,
            (1.0 - td_lambda) * bootstrap + td_lambda * next_target,
        )
        advantages[:, step] = torch.where(active, advantage, 0.0)
        targets[:, step] = torch.where(active[:, None], target, 0.0)
        next_advantage = torch.where(active, advantage, next_advantage)
        next_target = torch.where(active[:, None], target, next_target)
    return advantages.detach(), targets.detach()


def ataraxos_loss(
    *,
    new_log_probs: torch.Tensor,
    value_logits: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_value_probs: torch.Tensor,
    legal: torch.Tensor,
    outcomes: torch.Tensor,
    valid: torch.Tensor,
    clip_ratio: float,
    value_coef: float,
    magnet_coef: float,
    kl_coef: float,
    gae_lambda: float,
    td_lambda: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute equal-decision Ataraxos PPO and categorical TD(lambda) loss."""
    chosen_new = new_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    chosen_old = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    advantages, value_targets = categorical_gae_targets(
        old_value_probs,
        outcomes,
        valid,
        gae_lambda=gae_lambda,
        td_lambda=td_lambda,
    )
    selected_advantages = advantages[valid]
    advantage_mean = selected_advantages.mean()
    advantage_std = selected_advantages.std(unbiased=False)

    log_ratio = (chosen_new - chosen_old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
    )
    policy_loss = -_decision_mean(surrogate, valid)

    value_steps = -(value_targets * F.log_softmax(value_logits.float(), -1)).sum(-1)
    value_loss = _decision_mean(value_steps, valid)

    new_probs = new_log_probs.exp()
    # Original Ataraxos direction: D_KL(new policy || behavior policy).
    kl_steps = (new_probs * (new_log_probs - old_log_probs)).sum(-1)
    behavior_kl = _decision_mean(kl_steps, valid)
    entropy_steps = -(new_probs * new_log_probs).sum(-1)
    entropy = _decision_mean(entropy_steps, valid)
    # Verify the critic as well as the actor before any on-policy update.
    behavior_value_l1 = _decision_mean(
        (value_logits.float().softmax(-1) - old_value_probs).abs().sum(-1), valid
    )
    legal_count = legal.sum(-1).clamp_min(1).to(new_probs.dtype)
    magnet_steps = (new_probs * new_log_probs).sum(-1) + legal_count.log()
    magnet_kl = _decision_mean(magnet_steps, valid)
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio)[valid].float().mean()
    total = policy_loss + value_coef * value_loss + kl_coef * behavior_kl
    total = total + magnet_coef * magnet_kl
    return total, {
        "loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "behavior_kl": behavior_kl.detach(),
        "behavior_value_l1": behavior_value_l1.detach(),
        "magnet_kl": magnet_kl.detach(),
        "entropy": entropy.detach(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_mean": ratio[valid].mean().detach(),
        "advantage_mean": advantage_mean.detach(),
        "advantage_std": advantage_std.detach(),
    }


def _prepare_batch(samples, device):
    batch = _to_device(RLData_pad_collate([sample.data for sample in samples]), device)
    old_log, old_value, legal, valid, outcomes = _pad_metadata(samples, device)
    return (
        batch.obs,
        batch.rl2s,
        batch.time_idxs,
        batch.actions,
        old_log,
        old_value,
        legal,
        valid,
        outcomes,
    )


def _forward_loss(
    policy,
    obs,
    rl2s,
    time_idxs,
    batch_actions,
    old_log,
    old_value,
    legal,
    valid,
    outcomes,
    device_type,
    mixed_precision_dtype,
    mixed_precision_enabled,
    args,
):
    with torch.autocast(
        device_type=device_type,
        dtype=mixed_precision_dtype,
        enabled=mixed_precision_enabled,
    ):
        step_embeddings = policy.tstep_encoder(obs=obs, rl2s=rl2s)
        states, _ = policy.traj_encoder(
            seq=step_embeddings, time_idxs=time_idxs, hidden_state=None
        )
        distribution = policy.actor(
            states,
            straight_from_obs={key: obs[key] for key in policy.pass_obs_keys_to_actor},
        )
        probs = distribution.probs[..., -1, :].float()[:, : valid.shape[1]]
        new_log = torch.where(
            probs > 0,
            probs.clamp_min(1e-30).log(),
            torch.full_like(probs, -70.0),
        )
        value_logits = policy.wld_value(states[:, : valid.shape[1]])
        actions = batch_actions.argmax(-1)[:, : valid.shape[1]]
        return ataraxos_loss(
            new_log_probs=new_log,
            value_logits=value_logits,
            actions=actions,
            old_log_probs=old_log,
            old_value_probs=old_value,
            legal=legal,
            outcomes=outcomes,
            valid=valid,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            magnet_coef=args.magnet_coef,
            kl_coef=args.kl_coef,
            gae_lambda=args.gae_lambda,
            td_lambda=args.td_lambda,
        )


class AtaraxosLearner(torch.nn.Module):
    """DDP/compile boundary around the recurrent policy/value objective."""

    def __init__(self, policy, device_type, mixed_precision_dtype, enabled, args):
        super().__init__()
        self.policy = policy
        self.device_type = device_type
        self.mixed_precision_dtype = mixed_precision_dtype
        self.mixed_precision_enabled = enabled
        self.args = args

    def forward(
        self,
        obs,
        rl2s,
        time_idxs,
        actions,
        old_log,
        old_value,
        legal,
        valid,
        outcomes,
    ):
        return _forward_loss(
            self.policy,
            obs,
            rl2s,
            time_idxs,
            actions,
            old_log,
            old_value,
            legal,
            valid,
            outcomes,
            self.device_type,
            self.mixed_precision_dtype,
            self.mixed_precision_enabled,
            self.args,
        )


def _batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _length_bucket_order(items, rng: random.Random, bucket_size: int):
    """Shuffle full histories while limiting recurrent padding waste."""
    ordered = sorted(items, key=lambda item: item.steps)
    buckets = [
        ordered[start : start + bucket_size]
        for start in range(0, len(ordered), bucket_size)
    ]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    return [item for bucket in buckets for item in bucket]


def _decision_batches(items, target_decisions: int, min_trajectories: int = 1):
    """Pack whole recurrent histories into nearly fixed-decision update groups.

    Do not split an episode into individual states: the transformer must replay
    the complete prefix for every decision. One oversized history is one group.
    """
    if target_decisions < 1 or min_trajectories < 1:
        raise ValueError("decision budget and min_trajectories must be positive")
    groups = []
    group = []
    count = 0
    for item in items:
        n = item.steps
        if n < 1:
            raise ValueError("empty player trajectory")
        if (
            group
            and len(group) >= min_trajectories
            and abs(count - target_decisions) <= abs(count + n - target_decisions)
        ):
            groups.append(group)
            group, count = [], 0
        group.append(item)
        count += n
        if count >= target_decisions and len(group) >= min_trajectories:
            groups.append(group)
            group, count = [], 0
    if group:
        if groups and (count < target_decisions // 2 or len(group) < min_trajectories):
            groups[-1].extend(group)
        else:
            groups.append(group)
    return groups


def _rank_decision_groups(ordered, target_decisions: int, world_size: int, rank: int):
    """Identical update count on every DDP rank; balance decisions across ranks.

    DDP averages rank gradients, so callers weight each local decision by its
    fraction of the *global* update group's decision count.
    """
    groups = _decision_batches(
        ordered, target_decisions * world_size, min_trajectories=world_size
    )
    output = []
    for group in groups:
        shards = [[] for _ in range(world_size)]
        weights = [0] * world_size
        for item in sorted(group, key=lambda row: row.steps, reverse=True):
            destination = min(range(world_size), key=lambda i: weights[i])
            shards[destination].append(item)
            weights[destination] += item.steps
        if any(not shard for shard in shards):
            raise RuntimeError("DDP update group has no trajectories on one rank")
        output.append((shards[rank], sum(weights)))
    return output


def power_schedule(
    coefficient: float,
    iteration: int,
    decay: float,
    floor: float,
    ceiling: float,
) -> float:
    return min(max(coefficient / ((iteration + 1) ** decay), floor), ceiling)


def _ema_update(
    ema_state: dict[str, torch.Tensor],
    live_state: dict[str, torch.Tensor],
    parameter_names: set[str],
    decay: float,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, live in live_state.items():
        previous = ema_state.get(name)
        if (
            name in parameter_names
            and live.is_floating_point()
            and previous is not None
        ):
            output[name] = (
                previous.to(live.device, live.dtype)
                .mul(decay)
                .add(live.detach(), alpha=1.0 - decay)
            )
        else:
            output[name] = live.detach().clone()
    return output


def update(args) -> Path:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Ataraxos DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA update requested but CUDA is unavailable")
    model, policy = _load_stage1_policy(
        args.checkpoint,
        temperature=args.temperature,
        device=device,
        architecture_save_root=args.architecture_save_root,
        architecture_run_name=args.architecture_run_name,
    )
    # Eval mode is intentional: PPO ratios must not contain dropout noise.
    # eval() leaves autograd enabled.
    policy.eval().requires_grad_(True)
    trajectories = _load_trajectories(args.replay_dir, model)
    if args.max_games:
        trajectories = trajectories[: args.max_games]
    source_games = len(trajectories) // 2

    enabled = device.type == "cuda" and args.mixed_precision != "no"
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    learner = AtaraxosLearner(policy, device.type, dtype, enabled, args)
    if args.compile and device.type == "cuda":
        learner = torch.compile(learner, dynamic=True, fullgraph=False)
    if distributed:
        learner = torch.nn.parallel.DistributedDataParallel(
            learner,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    state_path = args.parent_checkpoint.with_name("ataraxos_training_state.pt")
    resumed_state = None
    update_iteration = 0
    ema_state = {
        name: tensor.detach().clone() for name, tensor in policy.state_dict().items()
    }
    if state_path.is_file():
        resumed_state = torch.load(state_path, map_location=device, weights_only=True)
        if not isinstance(resumed_state, dict) or "optimizer" not in resumed_state:
            raise RuntimeError(f"invalid Ataraxos training state: {state_path}")
        update_iteration = int(resumed_state.get("next_iteration", 0))
        stored_ema = resumed_state.get("ema_policy")
        if isinstance(stored_ema, dict):
            ema_state = stored_ema
    if args.training_iteration is not None:
        update_iteration = args.training_iteration
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else power_schedule(
            args.lr_coefficient,
            update_iteration,
            args.lr_decay,
            args.lr_floor,
            args.lr_ceiling,
        )
    )
    magnet_coef = (
        args.magnet_coef
        if args.magnet_coef is not None
        else power_schedule(
            args.magnet_coefficient,
            update_iteration,
            args.magnet_decay,
            args.magnet_floor,
            args.magnet_ceiling,
        )
    )
    # _forward_loss reads the resolved per-iteration coefficient.
    args.magnet_coef = magnet_coef

    # Before changing a single weight, verify that replaying the complete
    # recurrent inputs recovers the behavior distribution that was collected.
    initial_kl_sum = 0.0
    initial_value_l1_sum = 0.0
    initial_decisions = 0
    with torch.no_grad():
        # Weight the safety check by decisions, not variable-length histories.
        for samples in _batches(trajectories[rank::world_size], args.batch_size):
            _, metrics = learner(*_prepare_batch(samples, device))
            n = sum(sample.steps for sample in samples)
            initial_kl_sum += float(metrics["behavior_kl"].cpu()) * n
            initial_value_l1_sum += float(metrics["behavior_value_l1"].cpu()) * n
            initial_decisions += n
    if distributed:
        totals = torch.tensor(
            [initial_kl_sum, initial_value_l1_sum, float(initial_decisions)],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(totals)
        initial_kl_sum, initial_value_l1_sum, initial_decisions = totals.tolist()
    initial_kl = initial_kl_sum / max(initial_decisions, 1)
    initial_value_l1 = initial_value_l1_sum / max(initial_decisions, 1)
    if not math.isfinite(initial_kl) or initial_kl > args.max_initial_kl:
        raise RuntimeError(
            f"replayed policy does not match behavior policy: KL={initial_kl:.6g} "
            f"> {args.max_initial_kl:.6g}"
        )
    if (
        not math.isfinite(initial_value_l1)
        or initial_value_l1 > args.max_initial_value_l1
    ):
        raise RuntimeError(
            f"replayed W/L/D values do not match behavior predictions: "
            f"L1={initial_value_l1:.6g} > {args.max_initial_value_l1:.6g}"
        )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    optimizer_resumed = False
    if resumed_state is not None:
        optimizer.load_state_dict(resumed_state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["weight_decay"] = args.weight_decay
        optimizer_resumed = True
    rng = random.Random(args.seed)
    final_metrics: dict[str, float] = {}
    optimizer_steps = 0
    decisions_per_update = []
    for epoch in range(args.update_epochs):
        order = _length_bucket_order(
            trajectories, rng, max(args.batch_size * world_size * 8, 32)
        )
        update_groups = _rank_decision_groups(
            order, args.target_batch_decisions, world_size, rank
        )
        for samples_for_rank, global_decisions in update_groups:
            optimizer.zero_grad(set_to_none=True)
            metrics_accum: dict[str, float] = {}
            local_decisions = sum(sample.steps for sample in samples_for_rank)
            if not local_decisions:
                raise RuntimeError("empty decision-budget update")
            microbatches = list(_batches(samples_for_rank, args.batch_size))
            for micro_index, samples in enumerate(microbatches):
                # DDP must synchronize exactly once per optimizer step even if
                # ranks have different numbers of recurrent microbatches.
                sync = not distributed or micro_index == len(microbatches) - 1
                context = nullcontext() if sync else learner.no_sync()
                with context:
                    loss, metrics = learner(*_prepare_batch(samples, device))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Ataraxos update produced a non-finite loss")
                    micro_decisions = sum(sample.steps for sample in samples)
                    # DDP averages gradients across ranks. The world_size factor
                    # recovers an exact global per-decision mean in that reduction.
                    weight = world_size * micro_decisions / global_decisions
                    (loss * weight).backward()
                for key, value in metrics.items():
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + (
                        float(value.detach().cpu()) * micro_decisions / local_decisions
                    )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.max_grad_norm
            )
            optimizer.step()
            optimizer_steps += 1
            decisions_per_update.append(global_decisions)
            final_metrics = metrics_accum
            final_metrics["grad_norm"] = float(grad_norm.detach().float().cpu())
        if rank == 0:
            print(
                f"[AtaraxosUpdate] epoch={epoch} steps={optimizer_steps} "
                f"decisions={sum(item.steps for item in trajectories)} "
                f"mean_decisions_per_update="
                f"{sum(decisions_per_update) / max(len(decisions_per_update), 1):.1f} "
                f"loss={final_metrics['loss']:.6f} "
                f"kl={final_metrics['behavior_kl']:.6f} "
                f"ratio={final_metrics['ratio_mean']:.6f}",
                flush=True,
            )

    ema_state = _ema_update(
        ema_state,
        policy.state_dict(),
        set(dict(policy.named_parameters())),
        args.ema_decay,
    )
    output = args.output_checkpoint
    if distributed:
        torch.distributed.barrier()
    if rank != 0:
        torch.distributed.destroy_process_group()
        return output
    _atomic_torch_save(policy.state_dict(), output)
    optimizer_output = output.with_name("optimizer_after_update.pt")
    _atomic_torch_save(optimizer.state_dict(), optimizer_output)
    ema_output = output.with_name("ema_policy.pt")
    _atomic_torch_save(
        {name: value.detach().cpu() for name, value in ema_state.items()}, ema_output
    )
    training_state_output = output.with_name("ataraxos_training_state.pt")
    _atomic_torch_save(
        {
            "optimizer": optimizer.state_dict(),
            "ema_policy": {
                name: value.detach().cpu() for name, value in ema_state.items()
            },
            "next_iteration": update_iteration + 1,
        },
        training_state_output,
    )
    summary = {
        "behavior_checkpoint": args.checkpoint,
        "output_checkpoint": output,
        "optimizer_checkpoint": optimizer_output,
        "training_state": training_state_output,
        "ema_checkpoint": ema_output,
        "optimizer_resumed": optimizer_resumed,
        "games": source_games,
        "player_trajectories": len(trajectories),
        "decisions": sum(item.steps for item in trajectories),
        "update_epochs": args.update_epochs,
        "optimizer_steps": optimizer_steps,
        "target_batch_decisions": args.target_batch_decisions,
        "min_decisions_per_update": min(decisions_per_update),
        "max_decisions_per_update": max(decisions_per_update),
        "mean_decisions_per_update": f"{sum(decisions_per_update) / len(decisions_per_update):.2f}",
        "decisions_per_update": ",".join(str(n) for n in decisions_per_update),
        "training_iteration": update_iteration,
        "learning_rate": f"{learning_rate:.10g}",
        "magnet_coefficient": f"{magnet_coef:.10g}",
        "clip_ratio": args.clip_ratio,
        "gae_lambda": args.gae_lambda,
        "td_lambda": args.td_lambda,
        "ema_decay": args.ema_decay,
        "behavior_temperature": args.temperature,
        "distributed_world_size": world_size,
        "compiled": bool(args.compile and device.type == "cuda"),
        "initial_behavior_kl": f"{initial_kl:.8f}",
        "initial_behavior_value_l1": f"{initial_value_l1:.8f}",
    }
    summary.update({key: f"{value:.8f}" for key, value in final_metrics.items()})
    _write_text(args.summary_file, summary)
    temporary = args.next_checkpoint_file.with_suffix(".tmp")
    temporary.write_text(str(output.resolve()) + "\n", encoding="utf-8")
    os.replace(temporary, args.next_checkpoint_file)
    if distributed:
        torch.distributed.destroy_process_group()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--next-checkpoint-file")
    parser.add_argument("--architecture-save-root")
    parser.add_argument("--architecture-run-name")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--summary-file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=("no", "bf16"), default="bf16")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the learner graph on CUDA (disable for debugging).",
    )
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum full histories per GPU microbatch",
    )
    parser.add_argument("--target-batch-decisions", type=int, default=400)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--training-iteration", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-coefficient", type=float, default=0.5)
    parser.add_argument("--lr-decay", type=float, default=1.1)
    parser.add_argument("--lr-floor", type=float, default=5e-6)
    parser.add_argument("--lr-ceiling", type=float, default=1e-4)
    parser.add_argument("--magnet-coef", type=float)
    parser.add_argument("--magnet-coefficient", type=float, default=0.05)
    parser.add_argument("--magnet-decay", type=float, default=0.3)
    parser.add_argument("--magnet-floor", type=float, default=0.001)
    parser.add_argument("--magnet-ceiling", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=0.10)
    parser.add_argument("--gae-lambda", type=float, default=0.5)
    parser.add_argument("--td-lambda", type=float, default=0.8)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=0.267)
    parser.add_argument("--max-initial-kl", type=float, default=0.02)
    parser.add_argument("--max-initial-value-l1", type=float, default=0.05)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    round_dir = Path(os.environ.get("ATARAXOS_ROUND_DIR", ".")).resolve()
    replay_value = args.replay_dir or os.environ.get("ATARAXOS_REPLAY_DIR")
    checkpoint_value = args.checkpoint or os.environ.get("ATARAXOS_BEHAVIOR_CHECKPOINT")
    parent_value = args.parent_checkpoint or os.environ.get(
        "ATARAXOS_PARENT_CHECKPOINT"
    )
    next_value = args.next_checkpoint_file or os.environ.get(
        "ATARAXOS_NEXT_CHECKPOINT_FILE"
    )
    if not replay_value or not checkpoint_value or not next_value:
        parser.error(
            "replay-dir, checkpoint, and next-checkpoint-file are required "
            "as arguments or Ataraxos hook environment variables"
        )
    args.replay_dir = Path(replay_value).resolve()
    args.checkpoint = Path(checkpoint_value).resolve()
    args.parent_checkpoint = Path(parent_value or checkpoint_value).resolve()
    args.next_checkpoint_file = Path(next_value).resolve()
    args.architecture_save_root = args.architecture_save_root or os.environ.get(
        "ATARAXOS_ARCHITECTURE_SAVE_ROOT"
    )
    args.architecture_run_name = args.architecture_run_name or os.environ.get(
        "ATARAXOS_ARCHITECTURE_RUN_NAME"
    )
    if args.temperature is None:
        args.temperature = float(os.environ.get("ATARAXOS_BEHAVIOR_TEMPERATURE", "1.0"))
    args.output_checkpoint = Path(
        args.output_checkpoint or round_dir / "checkpoints" / "policy_after_update.pt"
    ).resolve()
    args.summary_file = Path(
        args.summary_file or round_dir / "update-summary.txt"
    ).resolve()
    if not args.replay_dir.is_dir():
        parser.error(f"--replay-dir is not a directory: {args.replay_dir}")
    if not args.checkpoint.is_file():
        parser.error(f"--checkpoint is not a file: {args.checkpoint}")
    if not args.parent_checkpoint.is_file():
        parser.error(f"--parent-checkpoint is not a file: {args.parent_checkpoint}")
    if min(args.update_epochs, args.batch_size, args.target_batch_decisions) < 1:
        parser.error(
            "update-epochs, batch-size and target-batch-decisions must be positive"
        )
    if args.max_games < 0:
        parser.error("max-games must be non-negative")
    if args.training_iteration is not None and args.training_iteration < 0:
        parser.error("training-iteration must be non-negative")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and non-negative")
    if not 0 < args.clip_ratio < 1:
        parser.error("clip-ratio must be between zero and one")
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.value_coef, args.kl_coef)
    ):
        parser.error("value and KL coefficients must be non-negative")
    schedule_values = (
        args.lr_coefficient,
        args.lr_decay,
        args.lr_floor,
        args.lr_ceiling,
        args.magnet_coefficient,
        args.magnet_decay,
        args.magnet_floor,
        args.magnet_ceiling,
    )
    if any(not math.isfinite(value) or value < 0 for value in schedule_values):
        parser.error("schedule values must be finite and non-negative")
    if args.lr_floor > args.lr_ceiling or args.magnet_floor > args.magnet_ceiling:
        parser.error("schedule floor cannot exceed its ceiling")
    if args.magnet_coef is not None and (
        not math.isfinite(args.magnet_coef) or args.magnet_coef < 0
    ):
        parser.error("magnet-coef must be finite and non-negative")
    if not 0 <= args.gae_lambda <= 1 or not 0 <= args.td_lambda <= 1:
        parser.error("GAE and TD lambda must be between zero and one")
    if not 0 <= args.ema_decay < 1:
        parser.error("EMA decay must be in [0, 1)")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("max-grad-norm must be finite and positive")
    if not math.isfinite(args.max_initial_kl) or args.max_initial_kl < 0:
        parser.error("max-initial-kl must be finite and non-negative")
    if not math.isfinite(args.max_initial_value_l1) or args.max_initial_value_l1 < 0:
        parser.error("max-initial-value-l1 must be finite and non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error(
            "temperature must be positive; deterministic argmax rollouts are "
            "not valid samples from their stored probability distribution"
        )
    update(args)


if __name__ == "__main__":
    main()
