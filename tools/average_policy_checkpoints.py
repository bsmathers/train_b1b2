#!/usr/bin/env python3
"""Safely average compatible policy-only PyTorch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a nonempty state dictionary: {path}")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError(f"checkpoint contains non-tensor state entries: {path}")
    return state


def average_checkpoints(
    checkpoints: list[Path], weights: list[float], output: Path
) -> Path:
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint paths must be unique")
    if len(weights) != len(checkpoints) or any(weight < 0 for weight in weights):
        raise ValueError("provide one non-negative weight per checkpoint")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("checkpoint weights must have a positive sum")
    normalized = [weight / total_weight for weight in weights]

    reference = _load_state(checkpoints[0])
    key_order = list(reference)
    output_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in reference.items():
        if tensor.is_floating_point():
            output_state[key] = tensor.float().mul(normalized[0])
        elif tensor.is_complex():
            output_state[key] = tensor.to(torch.complex64).mul(normalized[0])
        else:
            output_state[key] = tensor.clone()

    for path, weight in zip(checkpoints[1:], normalized[1:]):
        state = _load_state(path)
        if list(state) != key_order:
            missing = sorted(set(reference) - set(state))
            extra = sorted(set(state) - set(reference))
            raise ValueError(
                f"state keys differ in {path}; missing={missing[:5]} extra={extra[:5]}"
            )
        for key, reference_tensor in reference.items():
            tensor = state[key]
            if tensor.shape != reference_tensor.shape or tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"incompatible tensor {key!r} in {path}: "
                    f"{tuple(tensor.shape)} {tensor.dtype} versus "
                    f"{tuple(reference_tensor.shape)} {reference_tensor.dtype}"
                )
            if reference_tensor.is_floating_point():
                output_state[key].add_(tensor.float(), alpha=weight)
            elif reference_tensor.is_complex():
                output_state[key].add_(tensor.to(torch.complex64), alpha=weight)
            elif not torch.equal(tensor, reference_tensor):
                raise ValueError(
                    f"non-floating state {key!r} differs in {path}; refusing to average"
                )
        del state

    for key, reference_tensor in reference.items():
        tensor = output_state[key]
        if reference_tensor.is_floating_point() or reference_tensor.is_complex():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"averaged tensor {key!r} contains non-finite values")
            output_state[key] = tensor.to(reference_tensor.dtype)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(output_state, temporary)
    os.replace(temporary, output)

    manifest = output.with_suffix(output.suffix + ".manifest.txt")
    lines = [
        "Policy checkpoint weight average",
        f"Output: {output}",
        f"Output SHA-256: {_sha256(output)}",
        "Sources:",
    ]
    for path, weight in zip(checkpoints, normalized):
        lines.append(f"  weight={weight:.12g} sha256={_sha256(path)} path={path}")
    manifest.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        parser.error(f"missing checkpoint: {missing[0]}")
    weights = args.weight if args.weight is not None else [1.0] * len(checkpoints)
    try:
        output = average_checkpoints(checkpoints, weights, args.output.expanduser().resolve())
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote averaged checkpoint: {output}")
    print(f"Manifest: {output.with_suffix(output.suffix + '.manifest.txt')}")


if __name__ == "__main__":
    main()
