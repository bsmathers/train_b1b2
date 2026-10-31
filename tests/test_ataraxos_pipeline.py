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
from metamon.rl.ataraxos_update import (
    _ema_update,
    _decision_batches,
    _rank_decision_groups,
    ataraxos_loss,
    categorical_gae_targets,
    power_schedule,
)
from metamon.env.vectorized.opponent import RandomBatchedOpponent
from metamon.env.vectorized.vector_env import VectorizedShowdownEnv


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
    log_probs = np.asarray([[np.log(0.25), np.log(0.75), -70.0]], dtype=np.float32)
    return {
        "ataraxos_collector_version": 2,
        "behavior_policy_id": policy_id,
        "terminal_outcome": 1,
        "states": [
            {"battle_won": False, "battle_lost": False},
            {"battle_won": True, "battle_lost": False},
        ],
        "actions": [1, -1],
        "behavior_log_probs": log_probs.tolist(),
        "behavior_values": [[0.25, 0.125, 0.625]],
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
            self.assertEqual(summary["draws"], 0)
            self.assertEqual(summary["policy_id"], "sha256-test")

    def test_rejects_illegal_recorded_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = _valid_payload()
            payload["behavior_legal_masks"] = [[True, False, False]]
            path = self._write(Path(tmp), "bad.json.lz4", payload)
            with self.assertRaisesRegex(ValueError, "stored action is illegal"):
                load_rollout(path)

    def test_accepts_natural_draw_but_rejects_mislabeled_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = _valid_payload()
            payload["terminal_outcome"] = 0
            payload["states"][-1]["battle_won"] = False
            path = self._write(Path(tmp), "draw.json.lz4", payload)
            self.assertEqual(load_rollout(path).outcome, 0)
            payload["terminal_outcome"] = -1
            bad = self._write(Path(tmp), "bad.json.lz4", payload)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_rollout(bad)

    def test_rejects_mixed_behavior_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "one.json.lz4", _valid_payload("first"))
            self._write(root, "two.json.lz4", _valid_payload("second"))
            with self.assertRaisesRegex(ValueError, "mixed behavior policies"):
                validate_buffer(root)


class NativeBehaviorDriverTests(unittest.TestCase):
    def test_random_opponent_samples_only_legal_actions(self):
        opponent = RandomBatchedOpponent(
            num_lanes=3, action_dim=4, rng=np.random.default_rng(7)
        )
        active = np.asarray([True, False, True])
        obs = [
            {"illegal_actions": np.asarray([True, False, True, False])},
            {"illegal_actions": np.asarray([False, False, False, False])},
            {"illegal_actions": np.asarray([True, True, False, True])},
        ]
        for _ in range(20):
            actions = opponent.act(active, obs)
            self.assertIn(int(actions[0]), (1, 3))
            self.assertEqual(int(actions[1]), 0)
            self.assertEqual(int(actions[2]), 2)

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
        self.assertEqual(values.shape, (2, 3))
        self.assertTrue(np.allclose(values.sum(-1), 1.0))
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
        self.assertEqual(values.shape, (2, 3))
        self.assertTrue(np.allclose(values.sum(-1), 1.0))
        np.testing.assert_array_equal(legal, ~eval_obs["illegal_actions"])
        self.assertTrue(driver.has_prefetched_actions(opponent_active))
        np.testing.assert_array_equal(
            driver.act(opponent_active, []), np.asarray([1, 0])
        )
        opponent_metadata = driver.consume_policy_metadata(np.asarray([0, 1]))
        self.assertEqual([item[0] for item in opponent_metadata], [1, 0])
        self.assertTrue(
            all(np.isclose(np.exp(item[1]).sum(), 1.0) for item in opponent_metadata)
        )
        self.assertFalse(driver.has_prefetched_actions(opponent_active))

        no_opponent = np.asarray([False, False])
        driver.act_both(eval_obs, np.asarray([True, True]), no_opponent, opponent_obs)
        self.assertFalse(driver.has_prefetched_actions(no_opponent))
        np.testing.assert_array_equal(
            driver.act(opponent_active, opponent_obs), np.asarray([1, 0])
        )
        self.assertEqual(
            [item[0] for item in driver.consume_policy_metadata(np.asarray([0, 1]))],
            [1, 0],
        )


class AtaraxosDecisionBudgetTests(unittest.TestCase):
    @staticmethod
    def trajectories(lengths):
        return [SimpleNamespace(steps=n, ident=i) for i, n in enumerate(lengths)]

    def test_400_decisions_means_200_updates_for_80000_states(self):
        trajectories = self.trajectories([50] * 1600)
        batches = _decision_batches(trajectories, 400)
        self.assertEqual(len(batches), 200)
        self.assertTrue(all(sum(item.steps for item in b) == 400 for b in batches))
        self.assertEqual(
            sorted(item.ident for batch in batches for item in batch),
            list(range(1600)),
        )

    def test_variable_length_histories_preserved_and_never_dropped(self):
        lengths = [10, 20, 190, 70, 200, 40, 85, 30, 300, 50, 45]
        trajectories = self.trajectories(lengths)
        batches = _decision_batches(trajectories, 400)
        self.assertEqual(
            sorted(item.ident for batch in batches for item in batch),
            list(range(len(lengths))),
        )
        self.assertTrue(all(batch for batch in batches))

    def test_ddp_groups_match_and_preserve_all_decisions(self):
        trajectories = self.trajectories([50] * 1600)
        a = _rank_decision_groups(trajectories, 400, 2, 0)
        b = _rank_decision_groups(trajectories, 400, 2, 1)
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), 100)
        for (r0, total0), (r1, total1) in zip(a, b):
            self.assertEqual(total0, total1)
            self.assertEqual(sum(item.steps for item in r0 + r1), total0)
        all_items = [
            item.ident for rank in (a, b) for group, _ in rank for item in group
        ]
        self.assertEqual(sorted(all_items), list(range(1600)))

    def test_microbatch_weighting_matches_flat_decision_mean(self):
        # Two recurrent microbatches of different sizes. Every decision must
        # retain equal gradient weight after accumulation.
        weight = torch.tensor(1.0, requires_grad=True)
        first = torch.tensor([2.0] * 30)
        second = torch.tensor([3.0] * 370)
        for data in (first, second):
            (weight * data.mean() * (data.numel() / 400)).backward()
        expected = torch.cat([first, second]).mean()
        torch.testing.assert_close(weight.grad, expected)

    def test_decision_collection_drains_inflight_lanes(self):
        # Exercise target bookkeeping without constructing the native engine.
        env = VectorizedShowdownEnv.__new__(VectorizedShowdownEnv)
        env.batched_envs = 3
        env._parked_lanes = np.zeros(3, dtype=bool)
        env._collection_game_target = None
        env._collection_games_saved = 0
        env._collection_decision_target = None
        env._collection_decisions_saved = 0
        env._collection_stop_restarts = False
        env.set_collection_decision_target(400)
        self.assertFalse(env.collection_game_target_reached())
        env._collection_decisions_saved = 440
        env._collection_stop_restarts = True
        self.assertFalse(env.collection_game_target_reached())
        env._parked_lanes[:] = True
        self.assertTrue(env.collection_game_target_reached())


class AtaraxosUpdateTests(unittest.TestCase):
    def test_original_power_schedule_and_parameter_only_ema(self):
        self.assertEqual(power_schedule(0.5, 0, 1.1, 5e-6, 1e-4), 1e-4)
        self.assertEqual(power_schedule(0.05, 0, 0.3, 0.001, 0.1), 0.05)
        updated = _ema_update(
            {"weight": torch.tensor([1.0]), "spectral_u": torch.tensor([1.0])},
            {"weight": torch.tensor([3.0]), "spectral_u": torch.tensor([5.0])},
            {"weight"},
            0.5,
        )
        torch.testing.assert_close(updated["weight"], torch.tensor([2.0]))
        torch.testing.assert_close(updated["spectral_u"], torch.tensor([5.0]))

    def test_complete_game_gae_and_categorical_td_targets(self):
        old_values = torch.tensor(
            [[[0.60, 0.0, 0.40], [0.20, 0.0, 0.80]]], dtype=torch.float32
        )
        advantages, targets = categorical_gae_targets(
            old_values,
            torch.tensor([1.0]),
            torch.tensor([[True, True]]),
            gae_lambda=0.5,
            td_lambda=0.8,
        )
        torch.testing.assert_close(advantages, torch.tensor([[1.0, 0.4]]))
        torch.testing.assert_close(
            targets,
            torch.tensor([[[0.04, 0.0, 0.96], [0.0, 0.0, 1.0]]]),
        )

    def test_categorical_targets_respect_each_padded_length(self):
        old_values = torch.tensor(
            [
                [[0.5, 0.0, 0.5], [0.4, 0.0, 0.6]],
                [[0.7, 0.0, 0.3], [0.0, 0.0, 0.0]],
            ]
        )
        advantages, targets = categorical_gae_targets(
            old_values,
            torch.tensor([1.0, -1.0]),
            torch.tensor([[True, True], [True, False]]),
            gae_lambda=0.5,
            td_lambda=0.8,
        )
        torch.testing.assert_close(advantages[1], torch.tensor([-0.6, 0.0]))
        torch.testing.assert_close(
            targets[1], torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )

    def test_identity_policy_has_unit_ratio_and_zero_kl(self):
        old_probs = torch.tensor(
            [[[0.25, 0.75, 0.0], [0.60, 0.40, 0.0]]], dtype=torch.float32
        )
        old_log = torch.where(
            old_probs > 0,
            old_probs.clamp_min(1e-30).log(),
            torch.full_like(old_probs, -70.0),
        )
        new_log = old_log.clone().requires_grad_(True)
        value_logits = torch.zeros((1, 2, 3), requires_grad=True)
        loss, metrics = ataraxos_loss(
            new_log_probs=new_log,
            value_logits=value_logits,
            actions=torch.tensor([[1, 0]]),
            old_log_probs=old_log,
            old_value_probs=torch.tensor([[[0.45, 0.0, 0.55], [0.35, 0.0, 0.65]]]),
            legal=torch.tensor([[[True, True, False], [True, True, False]]]),
            outcomes=torch.tensor([1.0]),
            valid=torch.tensor([[True, True]]),
            clip_ratio=0.2,
            value_coef=1.0,
            magnet_coef=0.05,
            kl_coef=0.1,
            gae_lambda=0.5,
            td_lambda=0.8,
        )
        self.assertAlmostEqual(float(metrics["ratio_mean"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["behavior_kl"]), 0.0, places=6)
        self.assertGreater(float(metrics["behavior_value_l1"]), 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(new_log.grad)
        self.assertIsNotNone(value_logits.grad)


if __name__ == "__main__":
    unittest.main()
