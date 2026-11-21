import unittest

import numpy as np
import pandas as pd

from qqq_v15_walkforward_ml import build_labels


class QQQV15LabelTests(unittest.TestCase):
    def test_full_horizon_tail_is_missing_not_false(self):
        close = pd.Series(np.arange(100.0, 120.0))
        labels = build_labels(close, horizon=5)

        self.assertTrue(labels.iloc[-6:].isna().all())
        self.assertTrue(labels.iloc[:-6].notna().all())

    def test_label_uses_next_close_entry_and_five_session_exit(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0, 111.0])
        labels = build_labels(close, horizon=5)

        expected = 110.0 / 101.0 - 1.0 > 0
        self.assertEqual(labels.iloc[0], float(expected))


if __name__ == "__main__":
    unittest.main()
