#!/usr/bin/env python3
"""Evaluate a saved B1 W/L/D head on held-out trajectories."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from amago.loading import RLData_pad_collate
from torch.utils.data import DataLoader

from metamon.rl.dataset_config import (
    CustomReplaySource,
    DatasetConfig,
    build_dataset,
    load_dataset_config,
)
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.train import create_offline_rl_trainer


def source_configs(config: DatasetConfig):
    if config.prev_dataset is not None:
        raise ValueError("value evaluation currently requires a flat dataset config")
    common = {"formats": config.formats}
    if config.replay_weight > 0:
        yield "human", DatasetConfig(replay_weight=1.0, **common)
    for name, weight in (config.self_play or {}).items():
        if weight > 0:
            yield name, DatasetConfig(
                replay_weight=0.0, self_play={name: 1.0}, **common
            )
    for index, source in enumerate(config.custom_replays or []):
        if source.weight > 0:
            label = Path(source.dir).name or f"custom_{index}"
            yield label, DatasetConfig(
                replay_weight=0.0,
                custom_replays=[CustomReplaySource(source.dir, 1.0)],
                **common,
            )


def weighted_metrics(probs, labels, weights, bins: int = 15):
    weights = weights / weights.sum().clamp_min(1e-12)
    chosen = probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-30)
    predictions = probs.argmax(1)
    correct = predictions.eq(labels).float()
    nll = float((-chosen.log() * weights).sum())
    one_hot = F.one_hot(labels, 3).float()
    brier = float(((probs - one_hot).square().sum(1) * weights).sum())
    accuracy = float((correct * weights).sum())
    class_mass = torch.stack([weights[labels == i].sum() for i in range(3)])
    class_acc = []
    for i in range(3):
        mask = labels == i
        denom = weights[mask].sum()
        class_acc.append(
            float((correct[mask] * weights[mask]).sum() / denom)
            if float(denom) > 0
            else float("nan")
        )
    balanced = sum(x for x in class_acc if math.isfinite(x)) / sum(
        math.isfinite(x) for x in class_acc
    )
    confidence = probs.max(1).values
    ece = 0.0
    for lo in torch.linspace(0.0, 1.0, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        mask = (confidence >= lo) & (
            (confidence <= hi) if float(hi) >= 1.0 else (confidence < hi)
        )
        mass = weights[mask].sum()
        if float(mass) > 0:
            mean_conf = (confidence[mask] * weights[mask]).sum() / mass
            mean_acc = (correct[mask] * weights[mask]).sum() / mass
            ece += float(mass * (mean_conf - mean_acc).abs())
    confusion = torch.zeros(3, 3)
    for target in range(3):
        for predicted in range(3):
            mask = (labels == target) & (predictions == predicted)
            confusion[target, predicted] = weights[mask].sum()
    entropy_baseline = float(
        -(class_mass * class_mass.clamp_min(1e-30).log()).sum()
    )
    return {
        "nll": nll,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "brier": brier,
        "ece": ece,
        "class_mass": class_mass.tolist(),
        "class_accuracy": class_acc,
        "confusion": confusion.tolist(),
        "frequency_nll": entropy_baseline,
    }


@torch.no_grad()
def evaluate_source(experiment, dataset, batches: int, seed: int):
    dataset.configure_from_experiment(experiment)
    dataset.items_per_epoch = batches * experiment.batch_size
    for child in getattr(dataset, "all_datasets", []):
        child.items_per_epoch = dataset.items_per_epoch
    random.seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=experiment.batch_size,
        num_workers=0,
        collate_fn=RLData_pad_collate,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = experiment.accelerator.prepare(loader)
    policy = experiment.policy
    policy.eval()
    all_probs, all_labels, all_weights = [], [], []
    time_parts = {"early": ([], [], []), "middle": ([], [], []), "late": ([], [], [])}
    for batch in loader:
        with experiment.caster():
            encoded = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            states, _ = policy.traj_encoder(
                seq=encoded, time_idxs=batch.time_idxs, hidden_state=None
            )
            probs = policy.wld_value(states).float().softmax(-1)
        targets = batch.obs["trajectory_wld"].squeeze(-1).long()
        valid = ~((batch.rl2s == policy.pad_val).all(-1))
        for row in range(probs.shape[0]):
            indices = valid[row].nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            row_probs = probs[row, indices].detach().cpu()
            row_labels = targets[row, indices].detach().cpu()
            row_weights = torch.full((indices.numel(),), 1.0 / indices.numel())
            all_probs.append(row_probs)
            all_labels.append(row_labels)
            all_weights.append(row_weights)
            for name, part in zip(time_parts, torch.tensor_split(torch.arange(indices.numel()), 3)):
                if part.numel() == 0:
                    continue
                p, y, w = time_parts[name]
                p.append(row_probs[part])
                y.append(row_labels[part])
                w.append(torch.full((part.numel(),), 1.0 / part.numel()))
    overall = weighted_metrics(
        torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_weights)
    )
    temporal = {}
    for name, (p, y, w) in time_parts.items():
        temporal[name] = weighted_metrics(torch.cat(p), torch.cat(y), torch.cat(w))
    return overall, temporal, len(all_probs)


def print_metrics(label, metrics, trajectories, temporal=None):
    print(f"\nValue evaluation: {label} ({trajectories:,} sampled trajectories)")
    print(
        f"  NLL {metrics['nll']:.4f}  frequency-baseline {metrics['frequency_nll']:.4f}  "
        f"accuracy {metrics['accuracy']:.3%}  balanced {metrics['balanced_accuracy']:.3%}"
    )
    print(f"  Brier {metrics['brier']:.4f}  ECE {metrics['ece']:.4f}")
    print("  Class mass L/D/W: " + " ".join(f"{x:.3%}" for x in metrics["class_mass"]))
    print("  Class accuracy L/D/W: " + " ".join(f"{x:.3%}" for x in metrics["class_accuracy"]))
    print("  Weighted confusion rows=true L/D/W, columns=predicted L/D/W")
    for row in metrics["confusion"]:
        print("    " + " ".join(f"{x:.4f}" for x in row))
    if temporal:
        for name, values in temporal.items():
            print(
                f"  {name}: NLL {values['nll']:.4f}  accuracy {values['accuracy']:.3%}  "
                f"Brier {values['brier']:.4f}  ECE {values['ece']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--batches-per-source", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batches_per_source < 1:
        parser.error("--batches-per-source must be positive")

    pretrained = get_pretrained_model("TaurosV1")
    config = load_dataset_config(args.dataset_config)
    sources = list(source_configs(config))
    if not sources:
        raise RuntimeError("dataset config has no evaluable sources")
    first_dataset = build_dataset(
        sources[0][1], pretrained.observation_space, pretrained.action_space,
        pretrained.reward_function, split="test", test_fraction=0.10,
        split_seed=args.seed,
    )
    experiment = create_offline_rl_trainer(
        ckpt_dir=str(args.save_dir), run_name="stage1_b1_value_evaluation",
        model_gin_config="grouped_v2_medium_cuda.gin",
        train_gin_config="stage1_b1_imitation.gin",
        obs_space=pretrained.observation_space, action_space=pretrained.action_space,
        reward_function=pretrained.reward_function, amago_dataset=first_dataset,
        eval_gens=[], dloader_workers=0, epochs=1, steps_per_epoch=0,
        grad_accum=1, batch_size_per_gpu=args.batch_size, log=False,
        mixed_precision="bf16", manual_gin_overrides=pretrained.gin_overrides,
    )
    experiment.start()
    experiment.load_checkpoint_from_path(
        str(args.checkpoint.resolve()), is_accelerate_state=False
    )
    for offset, (name, source_config) in enumerate(sources):
        dataset = first_dataset if offset == 0 else build_dataset(
            source_config, pretrained.observation_space, pretrained.action_space,
            pretrained.reward_function, split="test", test_fraction=0.10,
            split_seed=args.seed, verbose=False,
        )
        overall, temporal, count = evaluate_source(
            experiment, dataset, args.batches_per_source, args.seed + offset * 1_000_003
        )
        print_metrics(name, overall, count, temporal)


if __name__ == "__main__":
    main()
