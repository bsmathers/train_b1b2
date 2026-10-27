"""Accounting checks for the exact-total native B1/TaurosV0 runner."""

import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "exact_native_match", ROOT / "tools" / "run_b1_vs_taurosv0_500.py"
)
match = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match)


class ExactNativeMatchTests(unittest.TestCase):
    def test_team_sampling_has_a_private_reproducible_rng(self):
        from metamon.env.wrappers import TeamSet

        team_set = TeamSet.__new__(TeamSet)
        team_set.team_files = ["a", "b", "c"]
        team_set._packed_team_cache = {"a": "A", "b": "B", "c": "C"}
        team_set._most_recent_team_file = None
        team_set._rng = random.Random()
        team_set.seed(123)
        first = [team_set.yield_team() for _ in range(12)]
        team_set.seed(123)
        second = [team_set.yield_team() for _ in range(12)]
        self.assertEqual(first, second)

    def test_result_requires_exact_disjoint_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            data = {
                "battles": 250,
                "candidate_wins": 120,
                "opponent_wins": 125,
                "draws": 5,
                "natural_draws": 2,
                "timeouts": 3,
                "candidate_score": 122.5 / 250,
                "engine": "zig-pkmn",
                "side": 0,
            }
            path.write_text(json.dumps(data))
            self.assertEqual(match._read_result(path, 250, 0), data)
            data["timeouts"] = 4
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                match._read_result(path, 250, 0)


if __name__ == "__main__":
    unittest.main()
