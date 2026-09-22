import unittest

import numpy as np
import pandas as pd

from stim_tuning_qc import _calculate_orientation_tuning, _orientation_heatmap


class OrientationTuningTests(unittest.TestCase):
    def test_heatmap_rows_are_rois_and_columns_are_directions(self):
        normalized = np.arange(2 * 14 * 60).reshape(2, 14, 60)
        roi_order = np.arange(60)[::-1]

        heatmap = _orientation_heatmap(normalized, 0, roi_order)

        self.assertEqual(heatmap.shape, (60, 14))
        np.testing.assert_array_equal(heatmap[0], normalized[0, :, 59])

    def test_separates_blocks_and_recovers_direction_response(self):
        timestamps = np.arange(0, 12, 0.05)
        dff = np.zeros((len(timestamps), 1))
        rows = []
        onsets = [1.0, 2.5, 4.0, 7.0, 8.5, 10.0]
        blocks = [1, 1, 1, 3, 3, 3]
        orientations = [0.0, np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0]
        for onset, block, orientation in zip(onsets, blocks, orientations):
            rows.append({
                'start_time': onset,
                'TrialType': 'single',
                'BlockNumber': block,
                'Orientation': orientation,
                'slap2_trial_idx': 0,
            })
            amplitude = 2.0 if orientation == 0.0 else 1.0
            dff[(timestamps >= onset + 0.1) & (timestamps < onset + 0.6), 0] = amplitude

        metrics = _calculate_orientation_tuning(pd.DataFrame(rows), dff, timestamps)

        np.testing.assert_array_equal(metrics['block_values'], np.array([1, 3]))
        np.testing.assert_allclose(metrics['orientations'], np.array([0.0, 90.0]))
        np.testing.assert_array_equal(
            metrics['presentation_counts'], np.array([[2, 1], [2, 1]])
        )
        np.testing.assert_allclose(metrics['tuning_mean'][:, :, 0], [[2, 1], [2, 1]])

    def test_excludes_omissions_and_unassigned_presentations(self):
        stim_df = pd.DataFrame([
            {'start_time': 1.0, 'TrialType': 'single', 'BlockNumber': 1,
             'Orientation': 0.0, 'slap2_trial_idx': 0},
            {'start_time': 2.0, 'TrialType': 'omission', 'BlockNumber': 1,
             'Orientation': 0.0, 'slap2_trial_idx': 0},
            {'start_time': 3.0, 'TrialType': 'single', 'BlockNumber': 3,
             'Orientation': 0.0, 'slap2_trial_idx': -1},
            {'start_time': 4.0, 'TrialType': 'single', 'BlockNumber': 3,
             'Orientation': 0.0, 'slap2_trial_idx': 1},
        ])
        timestamps = np.arange(0, 6, 0.05)
        dff = np.zeros((len(timestamps), 1))

        metrics = _calculate_orientation_tuning(stim_df, dff, timestamps)

        self.assertEqual(metrics['n_presentations'], 2)
        np.testing.assert_array_equal(metrics['presentation_counts'], [[1], [1]])


if __name__ == '__main__':
    unittest.main()