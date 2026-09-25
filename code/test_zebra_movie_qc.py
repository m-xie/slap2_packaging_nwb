import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from zebra_movie_qc import (
    _finite_correlation,
    _repeat_windows,
    _save_roi_correlation_plot,
)


class ZebraMovieQcTests(unittest.TestCase):
    def test_finite_correlation_measures_repeat_shape(self):
        first = np.asarray([1.0, 2.0, np.nan, 4.0])
        second = np.asarray([3.0, 5.0, 99.0, 9.0])
        inverted = np.asarray([9.0, 7.0, 99.0, 3.0])

        self.assertAlmostEqual(_finite_correlation(first, second), 1.0)
        self.assertAlmostEqual(_finite_correlation(first, inverted), -1.0)

    def test_splits_one_combined_zebra_interval(self):
        rows = pd.DataFrame({"start_time": [10.0], "stop_time": [610.0]})

        starts, duration = _repeat_windows(rows)

        self.assertEqual(starts, [10.0, 310.0])
        self.assertEqual(duration, 300.0)

    def test_uses_two_explicit_zebra_intervals(self):
        rows = pd.DataFrame(
            {"start_time": [10.0, 310.0], "stop_time": [310.0, 610.0]}
        )

        starts, duration = _repeat_windows(rows)

        self.assertEqual(starts, [10.0, 310.0])
        self.assertEqual(duration, 300.0)

    def test_saves_one_correlation_png_for_a_series(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "DMD1_green_correlations.png"

            _save_roi_correlation_plot(
                "DMD1_dFF_green",
                np.asarray([0.9, 0.1, -0.5, np.nan]),
                output_path,
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()