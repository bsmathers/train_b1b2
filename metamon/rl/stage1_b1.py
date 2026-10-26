"""Staged offline B1 training without the inherited Epoch475 Q critics.

The three stages are:
  1. fit a categorical W/L/D value head on a frozen Epoch475 policy;
  2. fit value + backbone while the actor head is frozen, with Epoch475
     policy KL and L2-SP protection;
  3. train actor + full backbone by conservative advantage-weighted imitation,
     while preserving both Epoch475 policy outputs and the stage-2 value model.
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
        actor_value_coef: float = 1.0,
        awr_temperature: float = 0.5,
        awr_min_weight: float = 0.25,
        awr_max_weight: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wld_value = nn.Linear(self.traj_encoder.emb_dim, 3)
        nn.init.xavier_uniform_(self.wld_value.weight)
        nn.init.zeros_(self.wld_value.bias)
        self.policy_kl_coef = float(policy_kl_coef)
        self.value_kl_coef = float(value_kl_coef)
        self.l2sp_coef = float(l2sp_coef)
        self.actor_value_coef = float(actor_value_coef)
        self.awr_temperature = float(awr_temperature)
        self.awr_min_weight = float(awr_min_weight)
        self.awr_max_weight = float(awr_max_weight)
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
        self.wld_value.requires_grad_(True)
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
        terms = []
        for prefix, module in (
            ("tstep_encoder", self.tstep_encoder),
            ("traj_encoder", self.traj_encoder),
        ):
            for name, parameter in module.named_parameters():
                terms.append(
                    (parameter - self._backbone_anchor[f"{prefix}.{name}"])
                    .square()
                    .mean()
                )
        return torch.stack(terms).mean()

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
        mean_awr_weight = torch.ones((), device=s.device)
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
            l2sp = self._l2sp()
            loss = loss + self.policy_kl_coef * policy_kl + self.l2sp_coef * l2sp

        if self._b1_phase == self.PHASE_ACTOR:
            ref_value_probs = self._phase2_value_distribution(batch)
            ref_expected = ref_value_probs[..., 2] - ref_value_probs[..., 0]
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
            outcome = targets[:, :T].float() - 1.0
            advantage = (outcome - ref_expected[:, :T]).detach()
            weights = (advantage / self.awr_temperature).exp().clamp(
                self.awr_min_weight, self.awr_max_weight
            )
            actor_loss = _trajectory_mean(-weights * chosen_lp, valid_actions)
            mean_awr_weight = _trajectory_mean(weights, valid_actions)
            loss = (
                actor_loss
                + self.actor_value_coef * value_ce
                + self.policy_kl_coef * policy_kl
                + self.value_kl_coef * value_kl
                + self.l2sp_coef * l2sp
            )

        if log_step:
            self.update_info = {
                "B1/Phase": self._b1_phase,
                "B1/WLD CE": value_ce.detach(),
                "B1/Policy KL": policy_kl.detach(),
                "B1/Value KL": value_kl.detach(),
                "B1/Actor AWR Loss": actor_loss.detach(),
                "B1/Mean AWR Weight": mean_awr_weight.detach(),
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
    # Re-seed and restore it so early stopping compares the same held-out replay
    # windows without perturbing the training sampler.
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


def run_staged_b1(
    experiment,
    validation_dataset,
    validation_batches: int,
    validation_seed: int,
    phase1_epochs: int,
    phase2_epochs: int,
    actor_epochs: int,
    phase1_patience: int,
    phase2_patience: int,
    actor_patience: int,
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
    phase_specs = (
        (1, "value_head", phase1_epochs, phase1_patience),
        (2, "value_backbone", phase2_epochs, phase2_patience),
        (3, "actor", actor_epochs, actor_patience),
    )
    global_epoch = 0
    for phase, name, max_epochs, patience in phase_specs:
        policy.set_b1_phase(phase)
        if phase == 3:
            policy.capture_phase2_reference()
        best_path = Path(experiment.ckpt_dir) / "policy_weights" / f"policy_{name}_best.pt"
        best_metric = float("inf")
        stale = 0
        for local_epoch in range(max_epochs):
            experiment.epoch = global_epoch
            experiment.epochs = global_epoch + 1
            experiment.learn()
            metric, metrics = _validate(
                experiment, loader, phase, seed=validation_seed
            )
            experiment.log(metrics, key=f"b1-validation-{name}")
            print(
                f"[B1Validation] phase={name} epoch={local_epoch} "
                f"global_epoch={global_epoch} metric={metric:.6f} best={best_metric:.6f}"
            )
            if metric < best_metric - min_delta:
                best_metric = metric
                stale = 0
                if experiment.accelerator.is_main_process:
                    tmp = best_path.with_suffix(".pt.tmp")
                    torch.save(policy.state_dict(), tmp)
                    os.replace(tmp, best_path)
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
        policy.load_state_dict(
            torch.load(best_path, map_location=experiment.DEVICE, weights_only=True)
        )
        if phase == 2:
            policy.capture_phase2_reference()
    experiment.epochs = global_epoch
