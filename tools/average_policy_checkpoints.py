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


def _normalize(vector: torch.Tensor, *, key: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"cannot normalize SigmaReparam buffer {key!r}")
    return vector / norm


def _recalibrate_sigma_reparam(
    state: dict[str, torch.Tensor],
    power_iterations: int,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> int:
    """Make SigmaReparam spectral buffers consistent with averaged weights.

    SigmaReparam stores unit left/right power-iteration vectors as floating-point
    buffers named ``u`` and ``v``.  Numerically averaging those buffers is invalid:
    their average is generally neither unit length nor a singular-vector estimate
    for the averaged weight.  Recompute them here before publishing the soup.
    """
    if power_iterations < 1:
        raise ValueError("SigmaReparam power iterations must be positive")

    prefixes = sorted(key[:-2] for key in state if key.endswith(".u"))
    recalibrated = 0
    for prefix in prefixes:
        u_key = f"{prefix}.u"
        v_key = f"{prefix}.v"
        weight_key = f"{prefix}.weight"
        if weight_key not in state:
            # A non-SigmaReparam module may legitimately own a buffer named ``u``.
            continue
        if v_key not in state:
            raise ValueError(f"SigmaReparam buffer {u_key!r} has no matching {v_key!r}")

        weight = state[weight_key]
        u_reference = state[u_key]
        v_reference = state[v_key]
        if weight.ndim != 2 or tuple(u_reference.shape) != (weight.shape[0],) or tuple(
            v_reference.shape
        ) != (weight.shape[1],):
            raise ValueError(
                f"invalid SigmaReparam shapes for {prefix!r}: "
                f"weight={tuple(weight.shape)} u={tuple(u_reference.shape)} "
                f"v={tuple(v_reference.shape)}"
            )

        matrix = weight.float()
        # Start from a real checkpoint buffer, never from an averaged spectral
        # buffer. Every iteration below then targets the averaged weight.
        v = (
            initial_state[v_key].float()
            if initial_state is not None and v_key in initial_state
            else v_reference.float()
        )
        if not torch.isfinite(v).all() or torch.linalg.vector_norm(v) <= 1e-12:
            v = torch.ones(matrix.shape[1], dtype=torch.float32)
        v = _normalize(v, key=v_key)
        for _ in range(power_iterations):
            u = _normalize(matrix @ v, key=u_key)
            v = _normalize(matrix.T @ u, key=v_key)
        u = _normalize(matrix @ v, key=u_key)

        sigma = torch.dot(u, matrix @ v)
        if not torch.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                f"invalid SigmaReparam spectral estimate for {prefix!r}: {sigma.item()}"
            )
        state[u_key] = u.to(u_reference.dtype)
        state[v_key] = v.to(v_reference.dtype)
        recalibrated += 1
    return recalibrated


def average_checkpoints(
    checkpoints: list[Path],
    weights: list[float],
    output: Path,
    *,
    sigma_power_iterations: int = 32,
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

    sigma_layers = _recalibrate_sigma_reparam(
        output_state,
        power_iterations=sigma_power_iterations,
        initial_state=reference,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(output_state, temporary)
    os.replace(temporary, output)

    manifest = output.with_suffix(output.suffix + ".manifest.txt")
    lines = [
        "Policy checkpoint weight average",
        f"Output: {output}",
        f"Output SHA-256: {_sha256(output)}",
        f"SigmaReparam layers recalibrated: {sigma_layers}",
        f"SigmaReparam power iterations: {sigma_power_iterations}",
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
    parser.add_argument("--sigma-power-iterations", type=int, default=32)
    args = parser.parse_args()

    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        parser.error(f"missing checkpoint: {missing[0]}")
    weights = args.weight if args.weight is not None else [1.0] * len(checkpoints)
    try:
        output = average_checkpoints(
            checkpoints,
            weights,
            args.output.expanduser().resolve(),
            sigma_power_iterations=args.sigma_power_iterations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote averaged checkpoint: {output}")
    print(f"Manifest: {output.with_suffix(output.suffix + '.manifest.txt')}")


if __name__ == "__main__":
    main()
