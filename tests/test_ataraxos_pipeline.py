import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lz4.frame
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault(
    "METAMON_CACHE_DIR", str(Path(tempfile.gettempdir()) / "metamon-test-cache")
)

from metamon.rl.ataraxos import JointNativePolicyDriver, NativeBehaviorDriver
from metamon.rl.ataraxos_replay import load_rollout, validate_buffer


class _FakeTrajEncoder:
    def init_hidden_state(self, batch_size, device):
        return None

    def reset_hidden_state(self, hidden_state, done):
        return hidden_state


class _MaskedActor(nn.Module):
    def forward(self, states, straight_from_obs):
        illegal = straight_from_obs["illegal_actions"].bool()
        logits = torch.arange(
            illegal.shape[-1], device=states.device, dtype=states.dtype
        ).expand_as(illegal)
        probs = logits.masked_fill(illegal, -1e9).softmax(-1).unsqueeze(-2)
        return SimpleNamespace(probs=probs)


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.traj_encoder = _FakeTrajEncoder()
        self.actor = _MaskedActor()
        self.wld_value = nn.Linear(2, 3)
        self.pass_obs_keys_to_actor = ["illegal_actions"]

    def get_state_embedding(self, obs, rl2s, time_idxs, hidden_state):
        batch = rl2s.shape[0]
        return torch.zeros((batch, 1, 2), device=rl2s.device), hidden_state


def _valid_payload(policy_id="sha256-test"):
    log_probs = np.asarray(
        [[np.log(0.25), np.log(0.75), -70.0]], dtype=np.float32
    )
    return {
        "ataraxos_collector_version": 1,
        "behavior_policy_id": policy_id,
        "states": [
            {"battle_won": False, "battle_lost": False},
            {"battle_won": True, "battle_lost": False},
        ],
        "actions": [1, -1],
        "behavior_log_probs": log_probs.tolist(),
        "behavior_values": [0.125],
        "behavior_legal_masks": [[True, True, False]],
    }


class AtaraxosReplayTests(unittest.TestCase):
    def _write(self, root: Path, name: str, payload: dict) -> Path:
        path = root / "gen1ou" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with lz4.frame.open(path, "wb") as stream:
            stream.write(json.dumps(payload).encode("utf-8"))
        return path

    def test_valid_round_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write(root, "one.json.lz4", _valid_payload())
            rollout = load_rollout(path)
            self.assertEqual(rollout.steps, 1)
            self.assertEqual(rollout.outcome, 1)
            summary = validate_buffer(root)
            self.assertEqual(summary["games"], 1)
            self.assertEqual(summary["wins"], 1)
            self.assertEqual(summary["policy_id"], "sha256-test")

    def test_rejects_illegal_recorded_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = _valid_payload()
            payload["behavior_legal_masks"] = [[True, False, False]]
            path = self._write(Path(tmp), "bad.json.lz4", payload)
            with self.assertRaisesRegex(ValueError, "stored action is illegal"):
                load_rollout(path)

    def test_rejects_mixed_behavior_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "one.json.lz4", _valid_payload("first"))
            self._write(root, "two.json.lz4", _valid_payload("second"))
            with self.assertRaisesRegex(ValueError, "mixed behavior policies"):
                validate_buffer(root)


class NativeBehaviorDriverTests(unittest.TestCase):
    def test_batched_shapes_legality_and_rl2(self):
        policy = _FakePolicy()
        driver = NativeBehaviorDriver(
            policy,
            torch.device("cpu"),
            lanes=2,
            action_dim=3,
            sample=False,
            mixed_precision="no",
        )
        obs = {
            "illegal_actions": np.asarray(
                [[False, True, False], [True, False, True]], dtype=bool
            )
        }
        actions, log_probs, values, legal = driver.act(obs)
        np.testing.assert_array_equal(actions, np.asarray([2, 1]))
        self.assertEqual(log_probs.shape, (2, 3))
        self.assertEqual(values.shape, (2,))
        np.testing.assert_array_equal(legal, ~obs["illegal_actions"])
        self.assertTrue(np.allclose(np.exp(log_probs).sum(-1), 1.0))
        self.assertTrue(np.all(np.exp(log_probs)[~legal] <= 1e-8))

        driver.observe(
            rewards=np.asarray([1.0, -0.5], dtype=np.float32),
            committed_actions=actions,
        )
        self.assertEqual(driver.time_idxs[:, 0, 0].tolist(), [1, 1])
        self.assertEqual(driver.rl2s[0, 0].tolist(), [1.0, 0.0, 0.0, 1.0])
        driver.reset_lanes(np.asarray([True, False]))
        self.assertEqual(driver.time_idxs[:, 0, 0].tolist(), [0, 1])

    def test_joint_driver_prefetches_opponent_without_metadata_drift(self):
        driver = JointNativePolicyDriver(
            _FakePolicy(),
            torch.device("cpu"),
            lanes=2,
            action_dim=3,
            sample=False,
            mixed_precision="no",
        )
        eval_obs = {
            "illegal_actions": np.asarray(
                [[False, True, False], [True, False, True]], dtype=bool
            )
        }
        opponent_obs = [
            {"illegal_actions": np.asarray([False, False, True], dtype=bool)},
            {"illegal_actions": np.asarray([False, True, True], dtype=bool)},
        ]
        opponent_active = np.asarray([True, True])
        actions, log_probs, values, legal = driver.act_both(
            eval_obs, np.asarray([True, True]), opponent_active, opponent_obs
        )
        np.testing.assert_array_equal(actions, np.asarray([2, 1]))
        self.assertEqual(log_probs.shape, (2, 3))
        self.assertEqual(values.shape, (2,))
        np.testing.assert_array_equal(legal, ~eval_obs["illegal_actions"])
        self.assertTrue(driver.has_prefetched_actions(opponent_active))
        np.testing.assert_array_equal(
            driver.act(opponent_active, []), np.asarray([1, 0])
        )
        self.assertFalse(driver.has_prefetched_actions(opponent_active))

        no_opponent = np.asarray([False, False])
        driver.act_both(
            eval_obs, np.asarray([True, True]), no_opponent, opponent_obs
        )
        self.assertFalse(driver.has_prefetched_actions(no_opponent))
        np.testing.assert_array_equal(
            driver.act(opponent_active, opponent_obs), np.asarray([1, 0])
        )


if __name__ == "__main__":
    unittest.main()
