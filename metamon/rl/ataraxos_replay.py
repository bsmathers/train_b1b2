"""Validated native rollouts for the Ataraxos online-training pipeline.

This module intentionally stops at the data contract.  Advantage estimation and
the policy/value objectives belong to the update implementation, while the
collector guarantees that every stored action is paired with the exact behavior
distribution, value estimate, legal mask, and terminal outcome that produced it.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import lz4.frame
import numpy as np


COLLECTOR_VERSION = 1


@dataclass(frozen=True)
class AtaraxosRollout:
    path: Path
    policy_id: str
    states: list[dict]
    actions: np.ndarray
    behavior_log_probs: np.ndarray
    behavior_values: np.ndarray
    behavior_legal_masks: np.ndarray
    outcome: int

    @property
    def steps(self) -> int:
        return int(self.actions.size)


def _read_payload(path: Path) -> dict:
    if path.name.endswith(".lz4"):
        with lz4.frame.open(path, "rb") as stream:
            return json.loads(stream.read())
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_rollout(path: str | Path) -> AtaraxosRollout:
    """Load one rollout and fail closed on any policy/action misalignment."""
    path = Path(path)
    payload = _read_payload(path)
    version = payload.get("ataraxos_collector_version")
    if version != COLLECTOR_VERSION:
        raise ValueError(
            f"{path}: collector version {version!r}, expected {COLLECTOR_VERSION}"
        )
    policy_id = payload.get("behavior_policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"{path}: missing behavior_policy_id")

    states = payload.get("states")
    raw_actions = np.asarray(payload.get("actions"))
    if not isinstance(states, list) or len(states) < 2:
        raise ValueError(f"{path}: rollout must contain at least two states")
    if raw_actions.ndim != 1 or raw_actions.size != len(states):
        raise ValueError(f"{path}: states/actions are not trajectory-aligned")
    if int(raw_actions[-1]) != -1:
        raise ValueError(f"{path}: terminal action sentinel must be -1")
    actions = raw_actions[:-1].astype(np.int64, copy=False)

    log_probs = np.asarray(payload.get("behavior_log_probs"), dtype=np.float32)
    values = np.asarray(payload.get("behavior_values"), dtype=np.float32)
    legal = np.asarray(payload.get("behavior_legal_masks"), dtype=bool)
    if log_probs.ndim != 2 or log_probs.shape[0] != actions.size:
        raise ValueError(f"{path}: behavior log probabilities are not action-aligned")
    if values.shape != (actions.size,):
        raise ValueError(f"{path}: behavior values are not action-aligned")
    if legal.shape != log_probs.shape:
        raise ValueError(f"{path}: behavior legal masks have the wrong shape")
    if not np.isfinite(log_probs).all() or not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite behavior metadata")
    if np.any(np.abs(values) > 1.0001):
        raise ValueError(f"{path}: behavior values are outside [-1, 1]")
    if np.any(actions < 0) or np.any(actions >= log_probs.shape[1]):
        raise ValueError(f"{path}: action index outside behavior distribution")
    if not legal[np.arange(actions.size), actions].all():
        raise ValueError(f"{path}: stored action is illegal under its behavior mask")
    probs = np.exp(log_probs)
    if not np.allclose(probs.sum(-1), 1.0, atol=2e-3, rtol=0.0):
        raise ValueError(f"{path}: behavior probabilities do not sum to one")
    if np.any(probs[~legal] > 1e-8):
        raise ValueError(f"{path}: illegal action has nonzero behavior probability")

    final = states[-1]
    won = bool(final.get("battle_won"))
    lost = bool(final.get("battle_lost"))
    if won == lost:
        raise ValueError(f"{path}: rollout has no decisive terminal outcome")
    outcome = 1 if won else -1
    return AtaraxosRollout(
        path=path,
        policy_id=policy_id,
        states=states,
        actions=actions,
        behavior_log_probs=log_probs,
        behavior_values=values,
        behavior_legal_masks=legal,
        outcome=outcome,
    )


def rollout_paths(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    yield from sorted(root.rglob("*.json.lz4"))


def validate_buffer(
    root: str | Path, *, workers: int = 0
) -> dict[str, int | str]:
    """Validate an entire round and return a compact human-loggable summary."""
    games = wins = losses = steps = 0
    policy_id: str | None = None
    paths = list(rollout_paths(root))
    if workers > 1 and len(paths) > 1:
        executor = ThreadPoolExecutor(
            max_workers=int(workers), thread_name_prefix="ataraxos-validator"
        )
        rollouts = executor.map(load_rollout, paths)
    else:
        executor = None
        rollouts = map(load_rollout, paths)
    try:
        for rollout in rollouts:
            if policy_id is None:
                policy_id = rollout.policy_id
            elif rollout.policy_id != policy_id:
                raise ValueError(
                    f"{rollout.path}: mixed behavior policies {policy_id!r} and "
                    f"{rollout.policy_id!r} in one collection round"
                )
            games += 1
            wins += rollout.outcome == 1
            losses += rollout.outcome == -1
            steps += rollout.steps
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if games == 0:
        raise ValueError(f"no Ataraxos rollouts found under {Path(root)}")
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "steps": steps,
        "policy_id": policy_id or "",
    }
