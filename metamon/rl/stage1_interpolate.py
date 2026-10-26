"""Stage-1 B2: weight interpolation between a base checkpoint and a B1 finetune.

Writes theta = (1 - alpha) * theta0 + alpha * theta1 for each alpha to
``{out_dir}/policy_epoch_{round(100 * alpha)}.pt``. Point ``out_dir`` at
``{save_dir}/{run_name}/ckpts/policy_weights`` and load the results with the
``TaurosV1Stage1`` model (``STAGE1_SAVE_DIR=save_dir STAGE1_RUN_NAME=run_name``).

Two details beyond plain averaging:

* SigmaReparam layers store ``u``/``v`` estimates of each weight's top singular
  vectors. Averaging two models' ``u``/``v`` can cancel them (singular vectors
  are only defined up to sign), so they are recomputed exactly from the
  interpolated weight.
* Target networks are copied from the interpolated online networks.

Example::

    python -m metamon.rl.stage1_interpolate \\
        --theta0 $TAUROSV1_SAVE_DIR/mediumg1online/ckpts/policy_weights/policy_epoch_475.pt \\
        --theta1 $STAGE1_SAVE_DIR/stage1_b1/ckpts/policy_weights/policy_epoch_30.pt \\
        --out_dir $STAGE1_SAVE_DIR/stage1_b2/ckpts/policy_weights \\
        --alphas 0.2 0.4 0.5 0.6 0.8
"""

import argparse
import os

import torch

TARGETS = {"target_critics.": "critics.", "maximized_critics.": "critics.", "target_actor.": "actor."}


def top_singular_pair(weight, v_ref):
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    u, v = U[:, 0], Vh[0]
    if torch.dot(v, v_ref.float()) < 0:  # keep the sign of the stored estimate
        u, v = -u, -v
    return u.to(weight.dtype), v.to(weight.dtype)


def interpolate(theta0, theta1, alpha):
    assert set(theta0) == set(theta1), "checkpoints have different keys"
    out = {}
    for key, a in theta0.items():
        b = theta1[key]
        assert a.shape == b.shape, f"shape mismatch for {key}"
        if a.is_floating_point() and key != "popart._t":
            out[key] = torch.lerp(a.float(), b.float(), alpha).to(b.dtype)
        else:  # integer buffers and step counters
            out[key] = b.clone()
    for key in out:
        if key.endswith(".u") and key[:-2] + ".v" in out and key[:-2] + ".weight" in out:
            prefix = key[:-2]
            out[prefix + ".u"], out[prefix + ".v"] = top_singular_pair(
                out[prefix + ".weight"], theta1[prefix + ".v"]
            )
    for key in out:
        for target, source in TARGETS.items():
            if key.startswith(target):
                out[key] = out[source + key[len(target):]].clone()
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage-1 B2 weight interpolation.")
    parser.add_argument("--theta0", required=True, help="Base checkpoint (alpha = 0).")
    parser.add_argument("--theta1", required=True, help="B1 checkpoint (alpha = 1).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.2, 0.4, 0.5, 0.6, 0.8])
    args = parser.parse_args()

    theta0 = torch.load(args.theta0, map_location="cpu")
    theta1 = torch.load(args.theta1, map_location="cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    for alpha in args.alphas:
        # Checkpoint 0 means "untrained, skip loading" to initialize_agent.
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        path = os.path.join(args.out_dir, f"policy_epoch_{round(100 * alpha)}.pt")
        torch.save(interpolate(theta0, theta1, alpha), path)
        print(f"alpha={alpha:.2f} -> {path}")


if __name__ == "__main__":
    main()
