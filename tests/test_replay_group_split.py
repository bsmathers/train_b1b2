"""Regression checks for battle-grouped replay train/validation splits."""

import unittest

from metamon.data.parsed_replay_dset import ParsedReplayDataset


class ReplayGroupSplitTests(unittest.TestCase):
    def test_perspective_trajectories_never_cross_split(self):
        names = []
        for battle in range(20):
            names.extend(
                (
                    f"/data/gen1ou/gen1ou-{battle}_2200_A_vs_B_01-01-2026_WIN.json.lz4",
                    f"/data/gen1ou/gen1ou-{battle}_2210_B_vs_A_01-01-2026_LOSS.json.lz4",
                )
            )

        train = object.__new__(ParsedReplayDataset)
        train.filenames = names.copy()
        train.split = "train"
        train.test_fraction = 0.25
        train.split_seed = 42
        train.verbose = False
        train._apply_split()

        test = object.__new__(ParsedReplayDataset)
        test.filenames = names.copy()
        test.split = "test"
        test.test_fraction = 0.25
        test.split_seed = 42
        test.verbose = False
        test._apply_split()

        train_groups = {train._split_group(name) for name in train.filenames}
        test_groups = {test._split_group(name) for name in test.filenames}
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(train_groups | test_groups, {f"gen1ou-{i}" for i in range(20)})
        self.assertEqual(len(train.filenames), 30)
        self.assertEqual(len(test.filenames), 10)

    def test_nonstandard_filename_remains_its_own_group(self):
        self.assertEqual(
            ParsedReplayDataset._split_group("/data/custom_name.json.lz4"),
            "custom_name.json.lz4",
        )


if __name__ == "__main__":
    unittest.main()
