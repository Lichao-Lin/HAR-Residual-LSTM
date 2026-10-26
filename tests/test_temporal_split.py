"""Check excluded observations and separation of five-day target windows."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from temporal_split import SPLIT_GAP, split_indices


class TemporalSplitTests(unittest.TestCase):
    def test_four_excluded_origins_at_each_boundary(self):
        train, validation, test = split_indices(100)
        np.testing.assert_array_equal(train, np.arange(70))
        np.testing.assert_array_equal(validation, np.arange(74, 85))
        np.testing.assert_array_equal(test, np.arange(89, 100))
        self.assertEqual(validation[0] - train[-1] - 1, SPLIT_GAP)
        self.assertEqual(test[0] - validation[-1] - 1, SPLIT_GAP)

    def test_five_day_targets_do_not_overlap_across_holdouts(self):
        for n in (100, 4913, 4721):
            with self.subTest(n=n):
                splits = split_indices(n)
                for earlier, later in zip(splits, splits[1:]):
                    last_target = set(range(earlier[-1] + 1, earlier[-1] + 6))
                    first_target = set(range(later[0] + 1, later[0] + 6))
                    self.assertTrue(last_target.isdisjoint(first_target))
                self.assertEqual(sum(map(len, splits)), n - 2 * SPLIT_GAP)

    def test_rejects_empty_holdouts(self):
        for n in (0, 1, 10, 30):
            with self.subTest(n=n), self.assertRaises(ValueError):
                split_indices(n)


if __name__ == "__main__":
    unittest.main()
