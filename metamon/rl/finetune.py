"""Iterative finetuning with MetamonFinetuneAgent (tortoise EMA + IS correction).

Two independent knobs (example: finetuning ``Kakuna``):

- **Data** — ``--dataset_config`` YAML (``replay_weight``, ``self_play``,
  ``custom_replays``, optional ``prev_dataset`` / ``anneal_epochs``).
- **Weights** — omit ``--prev_run_*`` to start from HuggingFace; set
  ``--prev_run_dir/name/checkpoint`` to continue a prior finetune run.

``--base_model`` is always required (architecture, obs/action spaces, tokenizer).
See ``metamon.rl.dataset_config`` and ``metamon/rl/configs/datasets/`` for YAML
details. The repo ships ``self_play_dset.yaml`` as a starting mix.

Typical loop: finetune from HF → collect self-play → finetune again with new
YAML + ``--prev_run_*``.

**Iter 1 — HuggingFace weights**::

    python -m metamon.rl.finetune \\
        --run_name kakuna_iter1 --save_dir /path/to/ckpts \\
        --base_model Kakuna --dataset_config self_play_dset.yaml --log

**Collect self-play** (point ``custom_replays[].dir`` at the output pile in the
next YAML)::

    python -m metamon.rl.evaluate.ladder_self_play \\
        --config metamon/rl/evaluate/ladder_self_play/example_config.yaml \\
        --format gen1ou --gpus 0 1 2 3 \\
        --save_trajectories_to /path/to/kakuna_iter2_pile

**Iter 2+ — local weights + new YAML**::

    python -m metamon.rl.finetune \\
        --run_name kakuna_iter2 --save_dir /path/to/ckpts \\
        --base_model Kakuna \\
        --prev_run_dir /path/to/ckpts --prev_run_name kakuna_iter1 \\
        --prev_checkpoint 10 --dataset_config my_iter2.yaml --log

Example ``my_iter2.yaml`` (save under ``metamon/rl/configs/datasets/`` or pass
an absolute path)::

    replay_weight: 0.05
    prev_dataset: /path/to/ckpts/kakuna_iter1/dataset_config.yaml
    prev_weight: 0.75
    custom_replays:
      - dir: /path/to/kakuna_iter2_pile
        weight: 0.20
    anneal_epochs: 10
    formats:
      - gen1ou

Each run saves a flattened config to
``{save_dir}/{run_name}/dataset_config.yaml`` — reuse that path as ``prev_dataset``
on the next iteration.
"""

import os
import random

import torch
import torch.nn.functional as F
import wandb
from einops import repeat
from torch.utils.data import DataLoader

from amago.loading import RLData_pad_collate
from amago.nets.policy_dists import DiscreteLikeContinuous

import metamon
from metamon.rl.train import (
    create_offline_rl_trainer,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from metamon.rl.dataset_config import (
    load_dataset_config,
    save_dataset_config,
    flatten_config,
    build_dataset,
)
from metamon.rl.pretrained import get_pretrained_model_names, get_pretrained_model
from metamon.interface import get_reward_function_names, get_reward_function
from metamon.rl.stage1_b1 import Stage1B1Agent, run_staged_b1


def add_cli(parser):
    # Identity
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # Base model (architecture source)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        choices=get_pretrained_model_names(),
        help="Registered pretrained model that defines the architecture.",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=int,
        default=None,
        help="Checkpoint epoch of the base model. Only used on the first "
        "iteration (ignored when --prev_run_dir is set). "
        "Defaults to the model's default checkpoint.",
    )

    # Previous iteration (optional -- omit for the first round)
    parser.add_argument(
        "--prev_run_dir",
        type=str,
        default=None,
        help="--save_dir of the previous finetuning iteration. "
        "If omitted, initialises from --base_model directly.",
    )
    parser.add_argument(
        "--prev_run_name",
        type=str,
        default=None,
        help="--run_name of the previous finetuning iteration.",
    )
    parser.add_argument(
        "--prev_checkpoint",
        type=int,
        default=None,
        help="Checkpoint epoch to load from the previous iteration. "
        "Required when --prev_run_dir is set.",
    )

    # Training
    parser.add_argument(
        "--model_gin_config",
        type=str,
        default=None,
        help="Optional architecture gin override; defaults to the base model config.",
    )
    parser.add_argument("--train_gin_config", type=str, default="finetune.gin")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps_per_epoch", type=int, default=1_000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default="no",
        help="Accelerate mixed-precision mode. CUDA Stage-1 uses bf16.",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.0,
        help="Held-out trajectory fraction for actor imitation validation. Zero disables it.",
    )
    parser.add_argument("--validation_batches", type=int, default=50)
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs. Zero disables it.",
    )
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-3)
    parser.add_argument("--stage1_b1_pipeline", action="store_true")
    parser.add_argument("--b1_value_head_epochs", type=int, default=100)
    parser.add_argument("--b1_value_backbone_epochs", type=int, default=1000)
    parser.add_argument("--b1_actor_epochs", type=int, default=1000)
    parser.add_argument("--b1_value_head_steps_per_epoch", type=int, default=1000)
    parser.add_argument("--b1_value_backbone_steps_per_epoch", type=int, default=None)
    parser.add_argument("--b1_actor_steps_per_epoch", type=int, default=None)
    parser.add_argument("--b1_value_head_patience", type=int, default=3)
    parser.add_argument("--b1_value_backbone_patience", type=int, default=5)
    parser.add_argument("--b1_actor_patience", type=int, default=5)

    # Data
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to a dataset config YAML file. Can include prev_dataset / "
        "anneal_epochs for iterative dataset transitions.",
    )

    # Reward
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        choices=get_reward_function_names(),
    )

    # Eval / infra
    parser.add_argument("--dloader_workers", type=int, default=10)
    parser.add_argument("--async_env_mp_context", type=str, default="forkserver")
    parser.add_argument(
        "--eval_gens",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 9],
    )
    parser.add_argument("--log", action="store_true")
    return parser


def _resolve_checkpoint_path(args, pretrained):
    """Return the path to the policy weights file to load."""
    if args.prev_run_dir is not None:
        assert (
            args.prev_run_name is not None
        ), "--prev_run_name is required when --prev_run_dir is set"
        assert (
            args.prev_checkpoint is not None
        ), "--prev_checkpoint is required when --prev_run_dir is set"
        return os.path.join(
            args.prev_run_dir,
            args.prev_run_name,
            "ckpts",
            "policy_weights",
            f"policy_epoch_{args.prev_checkpoint}.pt",
        )
    ckpt = args.base_checkpoint or pretrained.default_checkpoint
    return pretrained.get_path_to_checkpoint(ckpt)


def _set_dataset_items_per_epoch(dataset, items_per_epoch: int) -> None:
    """Resize an already-configured AMAGO dataset, including mixtures."""
    dataset.items_per_epoch = int(items_per_epoch)
    for child in getattr(dataset, "all_datasets", []):
        _set_dataset_items_per_epoch(child, items_per_epoch)


def _configure_validation_dataset(dataset, experiment, validation_batches: int) -> None:
    dataset.configure_from_experiment(experiment)
    global_batch = experiment.batch_size * experiment.accelerator.num_processes
    _set_dataset_items_per_epoch(dataset, validation_batches * global_batch)


@torch.no_grad()
def _actor_imitation_loss(experiment, validation_dataset, validation_seed: int) -> float:
    """Evaluate masked actor NLL without running critics or mutating PopArt."""
    local_seed = int(validation_seed) + 1_000_003 * experiment.accelerator.process_index
    generator = torch.Generator().manual_seed(local_seed)
    loader = DataLoader(
        validation_dataset,
        batch_size=experiment.batch_size,
        num_workers=experiment.dloader_workers,
        collate_fn=RLData_pad_collate,
        pin_memory=experiment.DEVICE.type == "cuda",
        generator=generator,
    )
    loader = experiment.accelerator.prepare(loader)
    policy = experiment.policy
    was_training = policy.training
    python_rng_state = random.getstate()
    random.seed(local_seed)
    policy.eval()
    # Keep reduction on-device and portable; float32 is easily precise enough
    # for the small, fixed validation sample.
    loss_sum = torch.zeros((), device=experiment.DEVICE, dtype=torch.float32)
    count = torch.zeros((), device=experiment.DEVICE, dtype=torch.float32)

    try:
        for batch in loader:
            obs_embedding = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            state, _ = policy.traj_encoder(
                seq=obs_embedding, time_idxs=batch.time_idxs, hidden_state=None
            )
            straight_from_obs = {
                key: batch.obs[key] for key in policy.pass_obs_keys_to_actor
            }
            action_dist = policy.actor(state, straight_from_obs=straight_from_obs)
            if policy.discrete:
                action_dist = DiscreteLikeContinuous(action_dist)

            actions = batch.actions.clamp(0, 1.0) if policy.discrete else batch.actions
            actions = F.pad(actions, (0, 0, 0, 1), "replicate")
            actions = repeat(actions, "b l a -> b l g a", g=len(policy.gammas))
            actions = policy.actor.policy_dist.action_from_buffer(actions)
            if policy.discrete:
                log_prob = action_dist.log_prob(actions).unsqueeze(-1)
            elif policy.multibinary:
                log_prob = action_dist.log_prob(actions).mean(-1, keepdim=True)
            else:
                log_prob = action_dist.log_prob(actions).sum(-1, keepdim=True)
            log_prob = log_prob[:, :-1]

            valid_state = ~((batch.rl2s == policy.pad_val).all(-1, keepdim=True))
            mask = repeat(
                valid_state[:, 1:], "b l 1 -> b l g 1", g=len(policy.gammas)
            )
            mask = mask & repeat(
                ~batch.obs["missing_action_mask"][:, :-1],
                "b l 1 -> b l g 1",
                g=len(policy.gammas),
            )
            loss_sum += (-log_prob.float() * mask).sum()
            count += mask.sum()
    finally:
        random.setstate(python_rng_state)
        policy.train(was_training)

    totals = torch.stack((loss_sum, count))
    totals = experiment.accelerator.reduce(totals, reduction="sum")
    if totals[1].item() == 0:
        raise RuntimeError("validation split contains no recorded actions")
    return (totals[0] / totals[1]).item()


def _learn_with_early_stopping(
    experiment,
    validation_dataset,
    total_epochs: int,
    validation_seed: int,
    patience: int,
    min_delta: float,
) -> None:
    """Run AMAGO one epoch at a time and stop on held-out actor NLL."""
    def save_best_policy() -> None:
        if not experiment.accelerator.is_main_process:
            return
        best_path = os.path.join(
            experiment.ckpt_dir, "policy_weights", "policy_best_validation.pt"
        )
        tmp_path = best_path + ".tmp"
        torch.save(experiment.policy.state_dict(), tmp_path)
        os.replace(tmp_path, best_path)

    # Establish the loaded B0 checkpoint as the baseline.  Otherwise an epoch-0
    # regression would become the first (and therefore automatically "best")
    # checkpoint seen by early stopping.
    best_loss = _actor_imitation_loss(
        experiment, validation_dataset, validation_seed
    )
    best_epoch = -1
    stale_epochs = 0
    start_epoch = experiment.epoch
    experiment.log({"Actor Imitation Loss": best_loss}, key="validation")
    print(f"[B1Validation] epoch=initial actor_nll={best_loss:.6f}")
    save_best_policy()

    for epoch in range(start_epoch, total_epochs):
        experiment.epoch = epoch
        experiment.epochs = epoch + 1
        experiment.learn()
        val_loss = _actor_imitation_loss(experiment, validation_dataset, validation_seed)
        experiment.log({"Actor Imitation Loss": val_loss}, key="validation")
        print(
            f"[B1Validation] epoch={epoch} actor_nll={val_loss:.6f} "
            f"best={best_loss:.6f}"
        )

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            save_best_policy()
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            print(
                f"[B1EarlyStop] epoch={epoch} best_epoch={best_epoch} "
                f"best_actor_nll={best_loss:.6f} patience={patience}"
            )
            break

    experiment.epochs = total_epochs


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Iterative finetuning with MetamonFinetuneAgent."
    )
    add_cli(parser)
    args = parser.parse_args()

    metamon.print_banner()
    is_continuation = args.prev_run_dir is not None
    if is_continuation:
        print(
            f"  Finetune (iter): {args.prev_run_name} @ epoch {args.prev_checkpoint}"
            f"  →  {args.run_name}"
        )
    else:
        print(f"  Finetune (init): {args.base_model}  →  {args.run_name}")
    print(f"  Dataset config: {args.dataset_config}")
    print()

    pretrained = get_pretrained_model(args.base_model)

    dataset_config = load_dataset_config(args.dataset_config)

    # Auto-fill prev_dataset from the base model when the config declares
    # prev_weight (iterative finetuning) but omits prev_dataset.
    if dataset_config.prev_weight is not None and dataset_config.prev_dataset is None:
        if pretrained.dataset_config is None:
            raise ValueError(
                f"Base model '{args.base_model}' has no known dataset_config. "
                f"Set prev_dataset explicitly in your dataset config YAML."
            )
        dataset_config.prev_dataset = pretrained.dataset_config
        print(
            f"  prev_dataset inferred from {args.base_model}: "
            f"{pretrained.dataset_config}"
        )

    split_enabled = args.validation_fraction > 0.0
    if split_enabled and not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be in (0, 1)")
    if args.early_stopping_patience > 0 and not split_enabled:
        raise ValueError("early stopping requires --validation_fraction > 0")

    amago_dataset = build_dataset(
        config=dataset_config,
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        split="train" if split_enabled else None,
        test_fraction=args.validation_fraction if split_enabled else 0.1,
        split_seed=args.validation_seed,
    )
    validation_dataset = None
    if split_enabled:
        validation_dataset = build_dataset(
            config=dataset_config,
            obs_space=pretrained.observation_space,
            action_space=pretrained.action_space,
            reward_function=pretrained.reward_function,
            verbose=False,
            split="test",
            test_fraction=args.validation_fraction,
            split_seed=args.validation_seed,
        )

    # auto-save effective config to checkpoint directory
    config_save_path = os.path.join(args.save_dir, args.run_name, "dataset_config.yaml")
    save_dataset_config(flatten_config(dataset_config), config_save_path)
    print(f"  Dataset config saved to: {config_save_path}\n")

    reward_function = (
        get_reward_function(args.reward_function)
        if args.reward_function is not None
        else pretrained.reward_function
    )

    experiment = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=args.model_gin_config or pretrained.model_gin_config_path,
        train_gin_config=args.train_gin_config,
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        amago_dataset=amago_dataset,
        eval_gens=args.eval_gens,
        async_env_mp_context=args.async_env_mp_context,
        dloader_workers=args.dloader_workers,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        grad_accum=args.grad_accum,
        batch_size_per_gpu=args.batch_size_per_gpu,
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        manual_gin_overrides=pretrained.gin_overrides,
        ckpt_interval=args.ckpt_interval,
        mixed_precision=args.mixed_precision,
    )

    experiment.start()

    try:
        ckpt_path = _resolve_checkpoint_path(args, pretrained)
        print(f"  Loading weights from: {ckpt_path}")
        experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)

        if args.stage1_b1_pipeline:
            if validation_dataset is None:
                raise ValueError("staged B1 requires --validation_fraction > 0")
            run_staged_b1(
                experiment=experiment,
                validation_dataset=validation_dataset,
                validation_batches=args.validation_batches,
                validation_seed=args.validation_seed,
                phase1_epochs=args.b1_value_head_epochs,
                phase2_epochs=args.b1_value_backbone_epochs,
                actor_epochs=args.b1_actor_epochs,
                phase1_steps_per_epoch=args.b1_value_head_steps_per_epoch,
                phase2_steps_per_epoch=(
                    args.b1_value_backbone_steps_per_epoch or args.steps_per_epoch
                ),
                actor_steps_per_epoch=(
                    args.b1_actor_steps_per_epoch or args.steps_per_epoch
                ),
                phase1_patience=args.b1_value_head_patience,
                phase2_patience=args.b1_value_backbone_patience,
                actor_patience=args.b1_actor_patience,
                min_delta=args.early_stopping_min_delta,
            )
        elif validation_dataset is None:
            experiment.learn()
        else:
            if args.ckpt_interval != 1:
                print(
                    "[B1Validation] warning: --ckpt_interval is not 1; "
                    "some validated epochs will not have historical checkpoints"
                )
            _configure_validation_dataset(
                validation_dataset, experiment, args.validation_batches
            )
            _learn_with_early_stopping(
                experiment=experiment,
                validation_dataset=validation_dataset,
                total_epochs=args.epochs,
                validation_seed=args.validation_seed,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
            )
    finally:
        for env_name in ("train_envs", "val_envs"):
            envs = getattr(experiment, env_name, None)
            if envs is not None:
                try:
                    envs.close()
                except Exception:
                    pass
        wandb.finish()
