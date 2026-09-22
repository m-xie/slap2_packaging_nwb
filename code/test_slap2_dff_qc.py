import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from slap2_dff_qc import (
    _bin_activity,
    _is_raw_fluorescence_series,
    _save_dff_plot,
    _save_raw_fluorescence_plot,
)


class BinActivityTests(unittest.TestCase):
    def test_identifies_raw_f0_series_without_matching_dff(self):
        self.assertTrue(_is_raw_fluorescence_series('DMD1_F0_green'))
        self.assertTrue(_is_raw_fluorescence_series('DMD2_F0_red'))
        self.assertFalse(_is_raw_fluorescence_series('DMD1_dFF_green'))

    def test_averages_activity_in_100_ms_bins_for_every_roi(self):
        data = np.array([
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
            [7.0, 22.0],
        ])
        timestamps = np.array([5.00, 5.04, 5.11, 5.19])

        binned, bin_centers = _bin_activity(data, timestamps)

        np.testing.assert_allclose(binned, [[2.0, 12.0], [6.0, 20.0]])
        np.testing.assert_allclose(bin_centers, [5.05, 5.15])

    def test_preserves_empty_bins_and_ignores_nan_values(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0], [9.0, 8.0]])
        timestamps = np.array([0.0, 0.05, 0.31])

        binned, _ = _bin_activity(data, timestamps, chunk_size=1)

        np.testing.assert_allclose(binned[0], [2.0, 4.0])
        self.assertTrue(np.isnan(binned[1:3]).all())
        np.testing.assert_allclose(binned[3], [9.0, 8.0])

    def test_saves_all_series_in_one_png(self):
        timestamps = np.linspace(0.0, 10.0, 20)
        series = [
            ('DMD1_dFF_green', np.ones((20, 3)), timestamps),
            ('DMD2_dFF_green', np.zeros((20, 2)), timestamps),
        ]

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / 'dff_all_rois.png'
            _save_dff_plot(series, output_path)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_saves_raw_fluorescence_png(self):
        timestamps = np.linspace(0.0, 10.0, 20)
        series = [
            ('DMD1_F0_green', np.arange(60).reshape(20, 3), timestamps),
            ('DMD2_F0_red', np.arange(40).reshape(20, 2), timestamps),
        ]

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / 'raw_fluorescence_all_rois.png'
            _save_raw_fluorescence_plot(series, output_path)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()