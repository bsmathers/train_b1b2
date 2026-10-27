"""Staged offline B1 value training and conservative actor refinement.

The value pipeline has two stages:
  1. fit a categorical W/L/D value head on a frozen Epoch475 policy;
  2. fit value + backbone while the actor head is frozen, with Epoch475
     policy KL and L2-SP protection;

Actor refinement is a separate run from the best stage-2 checkpoint. It uses
ordinary behavior cloning on strong-model actions, plus functional anchors to
the Epoch475 policy and stage-2 value model. It deliberately does not use
trajectory outcomes as per-action advantage estimates.
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import random
from pathlib import Path

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from amago.agent import MultiTaskAgent
from amago.loading import RLData_pad_collate
from torch.utils.data import DataLoader


def _trajectory_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average states within each trajectory, then average trajectories."""
    values = values.float()
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    per_traj_count = mask.sum(dim=1).clamp_min(1.0)
    per_traj = (values * mask).sum(dim=1) / per_traj_count
    valid_traj = (mask.sum(dim=1) > 0).to(values.dtype)
    return (per_traj * valid_traj).sum() / valid_traj.sum().clamp_min(1.0)


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


@gin.configurable
class Stage1B1Agent(MultiTaskAgent):
    """Epoch475 actor/backbone with a categorical W/L/D head and no Q critics."""

    PHASE_VALUE_HEAD = 1
    PHASE_VALUE_BACKBONE = 2
    PHASE_ACTOR = 3

    def __init__(
        self,
        *args,
        policy_kl_coef: float = 1.0,
        value_kl_coef: float = 1.0,
        l2sp_coef: float = 1e-3,
        l2sp_layer_weights: dict[str, float] | None = None,
        actor_bc_coef: float = 0.55,
        actor_policy_kl_coef: float = 0.25,
        actor_value_kl_coef: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wld_value = nn.Linear(self.traj_encoder.emb_dim, 3)
        nn.init.xavier_uniform_(self.wld_value.weight)
        nn.init.zeros_(self.wld_value.bias)
        self.policy_kl_coef = float(policy_kl_coef)
        self.value_kl_coef = float(value_kl_coef)
        self.l2sp_coef = float(l2sp_coef)
        self.l2sp_layer_weights = {
            str(prefix): float(weight)
            for prefix, weight in (l2sp_layer_weights or {}).items()
        }
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self.l2sp_layer_weights.values()
        ):
            raise ValueError("L2-SP layer weights must be finite and non-negative")
        self.actor_bc_coef = float(actor_bc_coef)
        self.actor_policy_kl_coef = float(actor_policy_kl_coef)
        self.actor_value_kl_coef = float(actor_value_kl_coef)
        actor_coefficients = (
            self.actor_bc_coef,
            self.actor_policy_kl_coef,
            self.actor_value_kl_coef,
        )
        if any(not math.isfinite(value) or value < 0 for value in actor_coefficients):
            raise ValueError("actor refinement coefficients must be finite and non-negative")
        if sum(actor_coefficients) <= 0:
            raise ValueError("at least one actor refinement coefficient must be positive")
        self._b1_phase = self.PHASE_VALUE_HEAD
        self._epoch475_teacher = None
        self._phase2_reference = None
        self._backbone_anchor = None

        # The old action-value ensemble is incompatible with the W/L/D target
        # and is intentionally absent from all new B1 checkpoints.
        del self.critics
        del self.target_critics
        del self.maximized_critics
        del self.target_actor
        del self.popart
        self.set_b1_phase(self.PHASE_VALUE_HEAD)

    @property
    def trainable_params(self):
        # All parameters must be present in the optimizer before stages toggle
        # requires_grad. Frozen parameters simply receive no gradient.
        return itertools.chain(
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.actor.parameters(),
            self.wld_value.parameters(),
        )

    def hard_sync_targets(self):
        # Called by MultiTaskAgent.__init__ before obsolete target modules are
        # deleted. Preserve the normal initialization at that point.
        if hasattr(self, "target_actor"):
            return super().hard_sync_targets()

    def soft_sync_targets(self):
        return None

    def get_grad_norms(self):
        return {
            "Actor Grad Norm": _module_grad_norm(self.actor),
            "Value Grad Norm": _module_grad_norm(self.wld_value),
            "TrajEncoder Grad Norm": _module_grad_norm(self.traj_encoder),
            "TstepEncoder Grad Norm": _module_grad_norm(self.tstep_encoder),
        }

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = dict(state_dict)
        obsolete = (
            "critics.",
            "target_critics.",
            "maximized_critics.",
            "target_actor.",
            "popart.",
        )
        loading_legacy = any(name.startswith(obsolete) for name in state_dict)
        for name in list(state_dict):
            if name.startswith(obsolete):
                del state_dict[name]
        current = self.state_dict()
        # Only the immutable Epoch475 layout is allowed to omit the new head.
        # A damaged native B1 checkpoint must still fail strict loading.
        if loading_legacy:
            for name in ("wld_value.weight", "wld_value.bias"):
                state_dict.setdefault(name, current[name])
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @staticmethod
    def _copy_modules(*modules):
        copied = tuple(
            copy.deepcopy(module).eval().requires_grad_(False) for module in modules
        )
        return copied

    def on_checkpoint_loaded(self, is_resume: bool = False):
        if is_resume:
            raise RuntimeError(
                "B1 staged full-state resume is not supported; restart from a saved phase checkpoint"
            )
        self._epoch475_teacher = self._copy_modules(
            self.tstep_encoder, self.traj_encoder, self.actor
        )
        self._capture_backbone_anchor()

    def _capture_backbone_anchor(self):
        self._backbone_anchor = {
            name: parameter.detach().clone()
            for name, parameter in itertools.chain(
                (
                    (f"tstep_encoder.{n}", p)
                    for n, p in self.tstep_encoder.named_parameters()
                ),
                (
                    (f"traj_encoder.{n}", p)
                    for n, p in self.traj_encoder.named_parameters()
                ),
            )
        }

    def capture_phase2_reference(self):
        self._phase2_reference = self._copy_modules(
            self.tstep_encoder, self.traj_encoder, self.wld_value
        )
        self._capture_backbone_anchor()

    def set_b1_phase(self, phase: int):
        if phase not in (
            self.PHASE_VALUE_HEAD,
            self.PHASE_VALUE_BACKBONE,
            self.PHASE_ACTOR,
        ):
            raise ValueError(f"invalid B1 phase: {phase}")
        self._b1_phase = int(phase)
        backbone_trainable = phase >= self.PHASE_VALUE_BACKBONE
        actor_trainable = phase == self.PHASE_ACTOR
        for module in (self.tstep_encoder, self.traj_encoder):
            module.requires_grad_(backbone_trainable)
        self.actor.requires_grad_(actor_trainable)
        self.wld_value.requires_grad_(phase != self.PHASE_ACTOR)
        self._enforce_phase_modes()

    def _enforce_phase_modes(self):
        """Keep frozen modules from mutating training-mode buffers or dropout."""
        if not self.training:
            return
        if self._b1_phase == self.PHASE_VALUE_HEAD:
            self.tstep_encoder.eval()
            self.traj_encoder.eval()
        if self._b1_phase != self.PHASE_ACTOR:
            self.actor.eval()
        if self._b1_phase == self.PHASE_ACTOR:
            self.wld_value.eval()

    def train(self, mode: bool = True):
        # Experiment.learn() recursively enables training mode each epoch.
        # Reapply the phase-specific frozen-module modes afterwards so stage 1
        # cannot drift SigmaReparam buffers and stage 2 keeps the actor exact.
        super().train(mode)
        self._enforce_phase_modes()
        return self

    def _l2sp(self):
        if self._backbone_anchor is None:
            raise RuntimeError("B1 backbone anchor has not been captured")
        weighted_squared_error = None
        weighted_parameter_count = 0.0
        for prefix, module in (
            ("tstep_encoder", self.tstep_encoder),
            ("traj_encoder", self.traj_encoder),
        ):
            for name, parameter in module.named_parameters():
                full_name = f"{prefix}.{name}"
                matching_prefixes = (
                    key
                    for key in self.l2sp_layer_weights
                    if full_name == key or full_name.startswith(key + ".")
                )
                match = max(matching_prefixes, key=len, default=None)
                weight = self.l2sp_layer_weights.get(match, 1.0)
                if weight == 0:
                    continue
                term = (
                    parameter - self._backbone_anchor[full_name]
                ).square().sum() * weight
                weighted_squared_error = (
                    term
                    if weighted_squared_error is None
                    else weighted_squared_error + term
                )
                weighted_parameter_count += weight * parameter.numel()
        if weighted_squared_error is None or weighted_parameter_count <= 0:
            raise RuntimeError("L2-SP layer weights exclude the entire backbone")
        # A parameter-count-normalized penalty has a stable interpretation when
        # layer widths change. Prefix weights can then deliberately protect
        # selected modules without accidentally over-weighting every small bias.
        return weighted_squared_error / weighted_parameter_count

    def _teacher_policy(self, batch):
        if self._epoch475_teacher is None:
            raise RuntimeError("Epoch475 teacher has not been captured")
        tstep, traj, actor = self._epoch475_teacher
        with torch.no_grad():
            teacher_o = tstep(obs=batch.obs, rl2s=batch.rl2s)
            teacher_s, _ = traj(seq=teacher_o, time_idxs=batch.time_idxs, hidden_state=None)
            teacher = actor(
                teacher_s,
                straight_from_obs={k: batch.obs[k] for k in self.pass_obs_keys_to_actor},
            ).probs[..., -1, :].float()
        return teacher

    def _phase2_value_distribution(self, batch):
        if self._phase2_reference is None:
            raise RuntimeError("phase-2 value reference has not been captured")
        tstep, traj, value = self._phase2_reference
        with torch.no_grad():
            ref_o = tstep(obs=batch.obs, rl2s=batch.rl2s)
            ref_s, _ = traj(seq=ref_o, time_idxs=batch.time_idxs, hidden_state=None)
            return value(ref_s).softmax(-1).float()

    def forward(self, batch, log_step: bool):
        self.update_info = {}
        if "trajectory_wld" not in batch.obs:
            raise RuntimeError("B1 batch is missing per-state trajectory_wld labels")
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        s, _ = self.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
        value_logits = self.wld_value(s)
        targets = batch.obs["trajectory_wld"].squeeze(-1).long()
        valid_states = ~((batch.rl2s == self.pad_val).all(-1))
        # RLData padding uses pad_val for every observation, which is outside
        # the three W/L/D classes. Cross entropy still indexes masked entries,
        # so replace only padding targets before computing the unreduced loss.
        safe_targets = targets.masked_fill(~valid_states, 1)
        value_ce_step = F.cross_entropy(
            value_logits.flatten(0, 1), safe_targets.flatten(), reduction="none"
        ).view_as(targets)
        value_ce = _trajectory_mean(value_ce_step, valid_states)
        value_probs = value_logits.float().softmax(-1)
        expected_value = value_probs[..., 2] - value_probs[..., 0]

        policy_kl = value_kl = actor_loss = torch.zeros((), device=s.device)
        l2sp = torch.zeros((), device=s.device)
        loss = value_ce

        if self._b1_phase >= self.PHASE_VALUE_BACKBONE:
            dist = self.actor(
                s,
                straight_from_obs={k: batch.obs[k] for k in self.pass_obs_keys_to_actor},
            )
            probs = dist.probs[..., -1, :].float()
            log_probs = torch.where(
                probs > 0,
                probs.clamp_min(1e-30).log(),
                torch.full_like(probs, -70.0),
            )
            teacher_probs = self._teacher_policy(batch)
            teacher_log = torch.where(
                teacher_probs > 0,
                teacher_probs.clamp_min(1e-30).log(),
                torch.full_like(teacher_probs, -70.0),
            )
            policy_kl_step = (teacher_probs * (teacher_log - log_probs)).sum(-1)
            policy_kl = _trajectory_mean(policy_kl_step, valid_states)
            if self._b1_phase == self.PHASE_VALUE_BACKBONE:
                l2sp = self._l2sp()
                loss = loss + self.policy_kl_coef * policy_kl + self.l2sp_coef * l2sp

        if self._b1_phase == self.PHASE_ACTOR:
            ref_value_probs = self._phase2_value_distribution(batch)
            value_kl_step = (
                ref_value_probs
                * (
                    ref_value_probs.clamp_min(1e-30).log()
                    - value_probs.clamp_min(1e-30).log()
                )
            ).sum(-1)
            value_kl = _trajectory_mean(value_kl_step, valid_states)

            T = batch.actions.shape[1]
            chosen = batch.actions.argmax(-1)
            chosen_lp = log_probs[:, :T].gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            valid_actions = valid_states[:, :T] & (
                ~batch.obs["missing_action_mask"][:, :T, 0].bool()
            )
            actor_loss = _trajectory_mean(-chosen_lp, valid_actions)
            loss = (
                self.actor_bc_coef * actor_loss
                + self.actor_policy_kl_coef * policy_kl
                + self.actor_value_kl_coef * value_kl
            )

        if log_step:
            self.update_info = {
                "B1/Phase": self._b1_phase,
                "B1/WLD CE": value_ce.detach(),
                "B1/Policy KL": policy_kl.detach(),
                "B1/Value KL": value_kl.detach(),
                "B1/Actor BC Loss": actor_loss.detach(),
                "B1/L2-SP": l2sp.detach(),
                "B1/Expected Value": _trajectory_mean(expected_value, valid_states).detach(),
            }
        return loss


@torch.no_grad()
def _validate(experiment, loader, phase: int, seed: int):
    model = experiment.policy_aclr
    policy = experiment.accelerator.unwrap_model(model)
    model.eval()
    totals = {}
    batches = 0
    # RLDataset ignores sampler indices and samples with Python's random module.
    # The caller supplies a new deterministic seed for every validation pass,
    # so early stopping observes fresh held-out trajectories and windows without
    # perturbing the training sampler.
    random_state = random.getstate()
    random.seed(seed + experiment.accelerator.process_index)
    try:
        for batch in loader:
            with experiment.caster():
                loss = model(batch, log_step=True)
            metrics = {"Loss": loss.detach()} | policy.update_info
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[name] = totals.get(name, 0.0) + float(value.float().cpu())
            batches += 1
    finally:
        random.setstate(random_state)
    model.train()
    if batches == 0:
        raise RuntimeError("B1 validation loader produced no batches")
    averaged = {name: value / batches for name, value in totals.items()}
    # Phase 1 selects pure WLD accuracy. Later phases select their complete
    # constrained objective rather than silently ignoring drift penalties.
    metric = averaged["B1/WLD CE"] if phase == 1 else averaged["Loss"]
    return metric, averaged


def _atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _save_best_state(experiment, policy, policy_path: Path, training_path: Path):
    if not experiment.accelerator.is_main_process:
        return
    policy_state = policy.state_dict()
    # The training bundle is authoritative: putting policy, optimizer, and
    # scheduler in one atomic file guarantees they always describe the same
    # accepted validation candidate, even if the process is interrupted while
    # writing the convenient policy-only export below.
    _atomic_torch_save(
        {
            "policy": policy_state,
            "optimizer": experiment.optimizer.state_dict(),
            "lr_schedule": experiment.lr_schedule.state_dict(),
        },
        training_path,
    )
    _atomic_torch_save(policy_state, policy_path)


def _restore_best_state(experiment, policy, training_path: Path):
    training_state = torch.load(
        training_path, map_location=experiment.DEVICE, weights_only=True
    )
    policy.load_state_dict(training_state["policy"])
    experiment.optimizer.load_state_dict(training_state["optimizer"])
    experiment.lr_schedule.load_state_dict(training_state["lr_schedule"])


def _save_numbered_policy(experiment, policy, epoch: int, interval: int | None):
    if not interval or epoch % interval != 0:
        return
    if experiment.accelerator.is_main_process:
        _atomic_torch_save(
            policy.state_dict(),
            Path(experiment.ckpt_dir) / "policy_weights" / f"policy_epoch_{epoch}.pt",
        )


def run_staged_b1(
    experiment,
    validation_dataset,
    validation_batches: int,
    validation_seed: int,
    phase1_epochs: int,
    phase2_epochs: int,
    phase1_steps_per_epoch: int,
    phase2_steps_per_epoch: int,
    phase1_patience: int,
    phase2_patience: int,
    min_delta: float,
):
    """Run all B1 phases, restoring the best validation checkpoint each time."""
    policy = experiment.accelerator.unwrap_model(experiment.policy_aclr)
    if not isinstance(policy, Stage1B1Agent):
        raise TypeError("staged B1 requires Stage1B1Agent")
    validation_dataset.configure_from_experiment(experiment)
    validation_dataset.items_per_epoch = validation_batches * experiment.batch_size
    loader = DataLoader(
        validation_dataset,
        batch_size=experiment.batch_size,
        num_workers=0,
        collate_fn=RLData_pad_collate,
        generator=torch.Generator().manual_seed(validation_seed),
    )
    loader = experiment.accelerator.prepare(loader)
    # AMAGO's normal checkpoint duplicates policy weights inside a full
    # Accelerate state that this staged pipeline intentionally cannot resume.
    # Keep its interval for lightweight tournament policies, while best-state
    # bundles below retain the matching optimizer and scheduler needed within
    # the current run.
    numbered_policy_interval = experiment.ckpt_interval
    experiment.ckpt_interval = None
    phase_specs = (
        (1, "value_head", phase1_epochs, phase1_patience, phase1_steps_per_epoch),
        (2, "value_backbone", phase2_epochs, phase2_patience, phase2_steps_per_epoch),
    )
    global_epoch = 0
    validation_round = 0
    for phase, name, max_epochs, patience, steps_per_epoch in phase_specs:
        if steps_per_epoch < 1:
            raise ValueError(f"{name} steps_per_epoch must be positive")
        experiment.train_batches_per_epoch = (
            steps_per_epoch * experiment.batches_per_update
        )
        print(
            f"[B1Phase] phase={name} optimizer_steps_per_epoch={steps_per_epoch} "
            f"max_epochs={max_epochs} patience={patience}"
        )
        policy.set_b1_phase(phase)
        best_path = Path(experiment.ckpt_dir) / "policy_weights" / f"policy_{name}_best.pt"
        best_training_path = (
            Path(experiment.ckpt_dir) / "training_states" / f"training_{name}_best.pt"
        )
        # The phase-entry policy is a real candidate. This prevents the first
        # update after unfreezing the backbone or actor from becoming an
        # automatically accepted regression.
        best_metric, metrics = _validate(
            experiment,
            loader,
            phase,
            seed=validation_seed + validation_round * 1_000_003,
        )
        validation_round += 1
        experiment.log(metrics, key=f"b1-validation-{name}")
        _save_best_state(
            experiment, policy, best_path, best_training_path
        )
        print(
            f"[B1Validation] phase={name} epoch=initial "
            f"global_epoch={global_epoch} metric={best_metric:.6f} "
            f"best={best_metric:.6f}"
        )
        stale = 0
        for local_epoch in range(max_epochs):
            experiment.epoch = global_epoch
            experiment.epochs = global_epoch + 1
            experiment.learn()
            _save_numbered_policy(
                experiment, policy, global_epoch, numbered_policy_interval
            )
            metric, metrics = _validate(
                experiment,
                loader,
                phase,
                seed=validation_seed + validation_round * 1_000_003,
            )
            validation_round += 1
            experiment.log(metrics, key=f"b1-validation-{name}")
            print(
                f"[B1Validation] phase={name} epoch={local_epoch} "
                f"global_epoch={global_epoch} metric={metric:.6f} best={best_metric:.6f}"
            )
            if metric < best_metric - min_delta:
                best_metric = metric
                stale = 0
                _save_best_state(
                    experiment, policy, best_path, best_training_path
                )
            else:
                stale += 1
            global_epoch += 1
            if patience > 0 and stale >= patience:
                print(
                    f"[B1EarlyStop] phase={name} epoch={local_epoch} "
                    f"best={best_metric:.6f} patience={patience}"
                )
                break
        experiment.accelerator.wait_for_everyone()
        _restore_best_state(experiment, policy, best_training_path)
        if phase == 2:
            policy.capture_phase2_reference()
    # Publish the best stage-2 value model. Actor refinement intentionally runs
    # separately so none of its optimizer or validation state can leak into the
    # value checkpoint.
    experiment.accelerator.wait_for_everyone()
    if experiment.accelerator.is_main_process:
        final_state = policy.state_dict()
        policy_dir = Path(experiment.ckpt_dir) / "policy_weights"
        _atomic_torch_save(final_state, policy_dir / "policy_phase2_final.pt")
        _atomic_torch_save(final_state, policy_dir / f"policy_epoch_{global_epoch}.pt")
        _atomic_torch_save(
            final_state, Path(experiment.ckpt_dir) / "latest" / "policy.pt"
        )
        print(
            f"[B1Final] published best phase-2 policy as policy_phase2_final.pt, "
            f"policy_epoch_{global_epoch}.pt, and latest/policy.pt"
        )
    experiment.accelerator.wait_for_everyone()
    experiment.epoch = global_epoch
    experiment.epochs = global_epoch
    experiment.ckpt_interval = numbered_policy_interval


def _load_policy_state(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "policy" in state:
        state = state["policy"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"invalid B1 policy checkpoint: {path}")
    return state


def run_epoch20_actor_refinement(
    experiment,
    validation_dataset,
    phase2_checkpoint: str,
    validation_batches: int,
    validation_seed: int,
    max_epochs: int,
    steps_per_epoch: int,
    patience: int,
    min_delta: float,
):
    """Refine Phase 2 with plain BC and fixed functional anchors.

    The experiment has already loaded Epoch475, causing ``on_checkpoint_loaded``
    to capture the immutable policy teacher. Loading Phase 2 here changes only
    the live model. The optimizer and scheduler are new for this run.
    """
    if steps_per_epoch < 1:
        raise ValueError("actor refinement steps_per_epoch must be positive")
    phase2_path = Path(phase2_checkpoint).expanduser().resolve()
    if not phase2_path.is_file():
        raise FileNotFoundError(f"Phase-2 checkpoint was not found: {phase2_path}")

    policy = experiment.accelerator.unwrap_model(experiment.policy_aclr)
    if not isinstance(policy, Stage1B1Agent):
        raise TypeError("B1 actor refinement requires Stage1B1Agent")
    if policy._epoch475_teacher is None:
        raise RuntimeError("Epoch475 teacher must be captured before loading Phase 2")
    policy.load_state_dict(_load_policy_state(phase2_path, experiment.DEVICE), strict=True)
    policy.capture_phase2_reference()
    policy.set_b1_phase(policy.PHASE_ACTOR)

    validation_dataset.configure_from_experiment(experiment)
    validation_dataset.items_per_epoch = validation_batches * experiment.batch_size
    loader = DataLoader(
        validation_dataset,
        batch_size=experiment.batch_size,
        num_workers=0,
        collate_fn=RLData_pad_collate,
        generator=torch.Generator().manual_seed(validation_seed),
    )
    loader = experiment.accelerator.prepare(loader)
    experiment.train_batches_per_epoch = steps_per_epoch * experiment.batches_per_update
    numbered_policy_interval = experiment.ckpt_interval
    experiment.ckpt_interval = None

    policy_dir = Path(experiment.ckpt_dir) / "policy_weights"
    training_dir = Path(experiment.ckpt_dir) / "training_states"
    best_path = policy_dir / "policy_actor_bc_best.pt"
    best_training_path = training_dir / "training_actor_bc_best.pt"
    best_metric, metrics = _validate(
        experiment, loader, policy.PHASE_ACTOR, seed=validation_seed
    )
    experiment.log(metrics, key="b1-validation-actor-bc")
    _save_best_state(experiment, policy, best_path, best_training_path)
    print(
        "[B1ActorValidation] epoch=initial global_epoch=0 "
        f"metric={best_metric:.6f} best={best_metric:.6f}"
    )

    stale = 0
    completed_epochs = 0
    for epoch in range(max_epochs):
        experiment.epoch = epoch
        experiment.epochs = epoch + 1
        experiment.learn()
        _save_numbered_policy(experiment, policy, epoch, numbered_policy_interval)
        # Use the same held-out sample every epoch. Unlike the old actor phase,
        # validation noise cannot choose a checkpoint by sampling luck.
        metric, metrics = _validate(
            experiment, loader, policy.PHASE_ACTOR, seed=validation_seed
        )
        experiment.log(metrics, key="b1-validation-actor-bc")
        print(
            f"[B1ActorValidation] epoch={epoch} global_epoch={epoch} "
            f"metric={metric:.6f} best={best_metric:.6f}"
        )
        if metric < best_metric - min_delta:
            best_metric = metric
            stale = 0
            _save_best_state(experiment, policy, best_path, best_training_path)
        else:
            stale += 1
        completed_epochs = epoch + 1
        if patience > 0 and stale >= patience:
            print(
                f"[B1ActorEarlyStop] epoch={epoch} best={best_metric:.6f} "
                f"patience={patience}"
            )
            break

    experiment.accelerator.wait_for_everyone()
    _restore_best_state(experiment, policy, best_training_path)
    if experiment.accelerator.is_main_process:
        final_state = policy.state_dict()
        _atomic_torch_save(final_state, policy_dir / "policy_actor_bc_final.pt")
        _atomic_torch_save(final_state, policy_dir / f"policy_epoch_{completed_epochs}.pt")
        _atomic_torch_save(final_state, Path(experiment.ckpt_dir) / "latest" / "policy.pt")
        print(
            "[B1ActorFinal] published restored validation-best plain-BC actor as "
            f"policy_actor_bc_final.pt, policy_epoch_{completed_epochs}.pt, "
            "and latest/policy.pt"
        )
    experiment.accelerator.wait_for_everyone()
    experiment.epoch = completed_epochs
    experiment.epochs = completed_epochs
    experiment.ckpt_interval = numbered_policy_interval
