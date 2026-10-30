"""One conservative Ataraxos policy/value update over a validated native round.

This is deliberately a small reference learner: terminal Monte Carlo advantages,
clipped PPO, a categorical W/L/D value loss, exact behavior-policy KL, and atomic
checkpoint publication.  It consumes complete recurrent trajectories so the
transformer sees the same history as it did during native collection.
"""

from __future__ import annotations

import argparse
import math
import os
import random
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
    outcome: int
    path: Path

    @property
    def steps(self) -> int:
        return int(self.behavior_values.numel())


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


def _trajectory_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_f = valid.to(values.dtype)
    return ((values * valid_f).sum(1) / valid_f.sum(1).clamp_min(1)).mean()


def _pad_metadata(
    samples: list[UpdateTrajectory], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    longest = max(sample.steps for sample in samples)
    action_dim = samples[0].behavior_log_probs.shape[-1]
    old_log = torch.full(
        (len(samples), longest, action_dim), -70.0, device=device
    )
    old_value = torch.zeros((len(samples), longest), device=device)
    valid = torch.zeros((len(samples), longest), dtype=torch.bool, device=device)
    outcomes = torch.empty((len(samples),), device=device)
    for row, sample in enumerate(samples):
        steps = sample.steps
        old_log[row, :steps] = sample.behavior_log_probs.to(device)
        old_value[row, :steps] = sample.behavior_values.to(device)
        valid[row, :steps] = True
        outcomes[row] = float(sample.outcome)
    return old_log, old_value, valid, outcomes


def ataraxos_loss(
    *,
    new_log_probs: torch.Tensor,
    value_logits: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    outcomes: torch.Tensor,
    valid: torch.Tensor,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute trajectory-balanced clipped PPO and categorical W/L/D loss."""
    chosen_new = new_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    chosen_old = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    advantages = outcomes[:, None] - old_values
    selected_advantages = advantages[valid]
    advantage_mean = selected_advantages.mean()
    advantage_std = selected_advantages.std(unbiased=False).clamp_min(1e-4)
    advantages = (advantages - advantage_mean) / advantage_std

    log_ratio = (chosen_new - chosen_old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
    )
    policy_loss = -_trajectory_mean(surrogate, valid)

    targets = (outcomes.long() + 1)[:, None].expand_as(valid)
    value_steps = F.cross_entropy(
        value_logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(valid)
    value_loss = _trajectory_mean(value_steps, valid)

    old_probs = old_log_probs.exp()
    kl_steps = (old_probs * (old_log_probs - new_log_probs)).sum(-1)
    behavior_kl = _trajectory_mean(kl_steps, valid)
    new_probs = new_log_probs.exp()
    entropy_steps = -(new_probs * new_log_probs).sum(-1)
    entropy = _trajectory_mean(entropy_steps, valid)
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio)[valid].float().mean()
    total = policy_loss + value_coef * value_loss + kl_coef * behavior_kl
    total = total - entropy_coef * entropy
    return total, {
        "loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "behavior_kl": behavior_kl.detach(),
        "entropy": entropy.detach(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_mean": ratio[valid].mean().detach(),
        "advantage_mean": advantage_mean.detach(),
        "advantage_std": advantage_std.detach(),
    }


def _forward_loss(policy, samples, device, caster, args):
    batch = _to_device(RLData_pad_collate([sample.data for sample in samples]), device)
    old_log, old_value, valid, outcomes = _pad_metadata(samples, device)
    with caster():
        step_embeddings = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        states, _ = policy.traj_encoder(
            seq=step_embeddings, time_idxs=batch.time_idxs, hidden_state=None
        )
        distribution = policy.actor(
            states,
            straight_from_obs={
                key: batch.obs[key] for key in policy.pass_obs_keys_to_actor
            },
        )
        probs = distribution.probs[..., -1, :].float()[:, : valid.shape[1]]
        new_log = torch.where(
            probs > 0,
            probs.clamp_min(1e-30).log(),
            torch.full_like(probs, -70.0),
        )
        value_logits = policy.wld_value(states[:, : valid.shape[1]])
        actions = batch.actions.argmax(-1)[:, : valid.shape[1]]
        return ataraxos_loss(
            new_log_probs=new_log,
            value_logits=value_logits,
            actions=actions,
            old_log_probs=old_log,
            old_values=old_value,
            outcomes=outcomes,
            valid=valid,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
        )


def _batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def update(args) -> Path:
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

    enabled = device.type == "cuda" and args.mixed_precision != "no"
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    def caster():
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)

    # Before changing a single weight, verify that replaying the complete
    # recurrent inputs recovers the behavior distribution that was collected.
    initial_kl_sum = 0.0
    initial_batches = 0
    with torch.no_grad():
        for samples in _batches(trajectories, args.batch_size):
            _, metrics = _forward_loss(policy, samples, device, caster, args)
            initial_kl_sum += float(metrics["behavior_kl"].cpu())
            initial_batches += 1
    initial_kl = initial_kl_sum / max(initial_batches, 1)
    if not math.isfinite(initial_kl) or initial_kl > args.max_initial_kl:
        raise RuntimeError(
            f"replayed policy does not match behavior policy: KL={initial_kl:.6g} "
            f"> {args.max_initial_kl:.6g}"
        )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer_resumed = False
    parent_optimizer = args.parent_checkpoint.with_name("optimizer_after_update.pt")
    if parent_optimizer.is_file():
        optimizer.load_state_dict(
            torch.load(parent_optimizer, map_location=device, weights_only=True)
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay
        optimizer_resumed = True
    rng = random.Random(args.seed)
    final_metrics: dict[str, float] = {}
    optimizer_steps = 0
    for epoch in range(args.update_epochs):
        order = list(trajectories)
        rng.shuffle(order)
        for samples in _batches(order, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _forward_loss(policy, samples, device, caster, args)
            if not torch.isfinite(loss):
                raise RuntimeError("Ataraxos update produced a non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.max_grad_norm
            )
            optimizer.step()
            optimizer_steps += 1
            final_metrics = {key: float(value.cpu()) for key, value in metrics.items()}
            final_metrics["grad_norm"] = float(grad_norm.detach().float().cpu())
        print(
            f"[AtaraxosUpdate] epoch={epoch} steps={optimizer_steps} "
            f"loss={final_metrics['loss']:.6f} "
            f"kl={final_metrics['behavior_kl']:.6f} "
            f"ratio={final_metrics['ratio_mean']:.6f}",
            flush=True,
        )

    output = args.output_checkpoint
    _atomic_torch_save(policy.state_dict(), output)
    optimizer_output = output.with_name("optimizer_after_update.pt")
    _atomic_torch_save(optimizer.state_dict(), optimizer_output)
    summary = {
        "behavior_checkpoint": args.checkpoint,
        "output_checkpoint": output,
        "optimizer_checkpoint": optimizer_output,
        "optimizer_resumed": optimizer_resumed,
        "games": len(trajectories),
        "decisions": sum(item.steps for item in trajectories),
        "update_epochs": args.update_epochs,
        "optimizer_steps": optimizer_steps,
        "learning_rate": args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "behavior_temperature": args.temperature,
        "initial_behavior_kl": f"{initial_kl:.8f}",
    }
    summary.update({key: f"{value:.8f}" for key, value in final_metrics.items()})
    _write_text(args.summary_file, summary)
    temporary = args.next_checkpoint_file.with_suffix(".tmp")
    temporary.write_text(str(output.resolve()) + "\n", encoding="utf-8")
    os.replace(temporary, args.next_checkpoint_file)
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
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.10)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-initial-kl", type=float, default=0.02)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    round_dir = Path(os.environ.get("ATARAXOS_ROUND_DIR", ".")).resolve()
    replay_value = args.replay_dir or os.environ.get("ATARAXOS_REPLAY_DIR")
    checkpoint_value = args.checkpoint or os.environ.get(
        "ATARAXOS_BEHAVIOR_CHECKPOINT"
    )
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
        args.temperature = float(
            os.environ.get("ATARAXOS_BEHAVIOR_TEMPERATURE", "1.0")
        )
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
        parser.error(
            f"--parent-checkpoint is not a file: {args.parent_checkpoint}"
        )
    if min(args.update_epochs, args.batch_size) < 1:
        parser.error("update-epochs and batch-size must be positive")
    if args.max_games < 0:
        parser.error("max-games must be non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and non-negative")
    if not 0 < args.clip_ratio < 1:
        parser.error("clip-ratio must be between zero and one")
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.value_coef, args.entropy_coef, args.kl_coef)
    ):
        parser.error("value, entropy, and KL coefficients must be non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("max-grad-norm must be finite and positive")
    if not math.isfinite(args.max_initial_kl) or args.max_initial_kl < 0:
        parser.error("max-initial-kl must be finite and non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error(
            "temperature must be positive; deterministic argmax rollouts are "
            "not valid samples from their stored probability distribution"
        )
    update(args)


if __name__ == "__main__":
    main()
