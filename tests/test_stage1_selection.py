"""CPU-only smoke tests for Stage-1 tournament selection."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stage1_selector", ROOT / "tools" / "select_stage1_b1_epoch.py"
)
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)


class SelectionTests(unittest.TestCase):
    def test_archive_and_extraction(self):
        teams = select.good_team_bytes()
        self.assertEqual(len(teams), 807)
        self.assertEqual(len(set(teams)), 807)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            destination = select.install_good_teams(root)
            self.assertEqual(len(list(destination.glob("*.gen1ou_team"))), 807)
            self.assertEqual(select.install_good_teams(root), destination)

    def test_balanced_four_condition_scoring(self):
        entries, results = [], []
        for own, other in select.CONDITIONS:
            for reverse in (False, True):
                key = f"b1-{own}-{other}-{reverse}"
                entries.append(
                    (
                        "B1_e1",
                        own,
                        other,
                        reverse,
                        SimpleNamespace(matchup_id=key, n_battles=10),
                    )
                )
                # Candidate wins 7/10 on each side regardless of player order.
                results.append(
                    SimpleNamespace(
                        matchup_id=key,
                        total_battles=10,
                        policy_a_wins=3 if reverse else 7,
                        policy_b_wins=7 if reverse else 3,
                    )
                )
        with tempfile.TemporaryDirectory() as t:
            rows = select.score_tournament(entries, results, Path(t))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["complete"])
            self.assertAlmostEqual(rows[0]["macro_win_rate"], 0.7)
            # A half-finished bout must not produce a selected epoch.
            results[0].total_battles = 9
            rows = select.score_tournament(entries, results, Path(t))
            self.assertFalse(rows[0]["complete"])
            self.assertIsNone(rows[0]["macro_win_rate"])

    def test_draws_and_manifest(self):
        entries, results = [], []
        for own, other in select.CONDITIONS:
            for reverse in (False, True):
                key = f"{own}-{other}-{reverse}"
                entries.append(
                    (
                        "B1_e2",
                        own,
                        other,
                        reverse,
                        SimpleNamespace(matchup_id=key, n_battles=10),
                    )
                )
                results.append(
                    SimpleNamespace(
                        matchup_id=key,
                        total_battles=10,
                        policy_a_wins=4,
                        policy_b_wins=4,
                    )
                )
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            rows = select.score_tournament(entries, results, directory)
            self.assertAlmostEqual(rows[0]["macro_win_rate"], 0.5)
            select.validate_manifest(directory, {"a": 1})
            select.validate_manifest(directory, {"a": 1})
            with self.assertRaises(ValueError):
                select.validate_manifest(directory, {"a": 2})

    def test_native_outcome_scoring(self):
        # Score draws and turn-limit truncations as half a point; not losses.
        native_spec = importlib.util.spec_from_file_location(
            "native_matchup",
            ROOT / "metamon" / "rl" / "evaluate" / "native_matchup.py",
        )
        native = importlib.util.module_from_spec(native_spec)
        native_spec.loader.exec_module(native)
        trunc = [False, False]
        self.assertEqual(
            native.finish_score(True, trunc, {"won": [True, False]}, 0), 1.0
        )
        self.assertEqual(
            native.finish_score(True, trunc, {"won": [True, False]}, 1), 0.0
        )
        self.assertEqual(
            native.finish_score(
                True, trunc, {"won": [False, True], "draw": [True, False]}, 0
            ),
            0.5,
        )
        self.assertEqual(
            native.finish_score(True, [True], {"won": [False], "draw": [True]}, 0), 0.5
        )
        self.assertEqual(
            native.finish_outcome(True, [False], {"won": [True], "draw": [False]}, 0),
            "win",
        )
        self.assertEqual(
            native.finish_outcome(True, [False], {"won": [False], "draw": [False]}, 0),
            "loss",
        )
        self.assertEqual(
            native.finish_outcome(True, [False], {"won": [False], "draw": [True]}, 0),
            "draw",
        )
        self.assertEqual(
            native.finish_outcome(True, [True], {"won": [False], "draw": [True]}, 0),
            "timeout",
        )
        with self.assertRaises(ValueError):
            native.finish_score(False, [False], {"won": [True]}, 0)

if __name__ == "__main__":
    unittest.main()
