"""Pure-Python contract tests for the tournament's per-policy private inputs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "metamon/rl/evaluate/native_matchup.py"
spec = importlib.util.spec_from_file_location("_native_matchup_contract", source)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class FakeLane:
    def __init__(self):
        self.last_request = {
            "p1": {"side": {"id": "p1", "pokemon": [
                {"ident": "p1: Tauros"}, {"ident": "p1: Snorlax"}
            ]}},
            "p2": {"side": {"id": "p2", "pokemon": [
                {"ident": "p2: Chansey"}, {"ident": "p2: Starmie"}
            ]}},
        }

    def legal_action_indices(self, side, action_space):
        return [0, 2, 4]


class NativeInputContractTests(unittest.TestCase):
    def setUp(self):
        self.lane = FakeLane()
        self.env = SimpleNamespace(
            batched_envs=1,
            lanes=[self.lane],
            eval_side="p1",
            eval_action_space=object(),
        )
        self.obs = {"illegal_actions": [
            [False, True, False, True, False] + [True] * 8
        ]}

    def test_correct_private_requests_and_mask(self):
        native.validate_native_inputs(self.env, self.obs, 13)

    def test_wrong_pov_is_fatal(self):
        self.lane.last_request["p2"]["side"]["id"] = "p1"
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            native.validate_native_inputs(self.env, self.obs, 13)

    def test_private_team_leak_is_fatal(self):
        self.lane.last_request["p1"]["side"]["pokemon"][0]["ident"] = "p2: Tauros"
        with self.assertRaisesRegex(RuntimeError, "leaked"):
            native.validate_native_inputs(self.env, self.obs, 13)

    def test_bad_legality_mask_is_fatal(self):
        self.obs["illegal_actions"][0][4] = True
        with self.assertRaisesRegex(RuntimeError, "mask mismatch"):
            native.validate_native_inputs(self.env, self.obs, 13)

    def test_missing_legal_actions_is_fatal(self):
        self.lane.legal_action_indices = lambda *_: []
        with self.assertRaisesRegex(RuntimeError, "no legal action"):
            native.validate_native_inputs(self.env, self.obs, 13)


if __name__ == "__main__":
    unittest.main()
