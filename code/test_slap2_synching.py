import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from slap2_synching import (
    _build_trial_line_time_map,
    get_slap2_secondary_plane_timestamps,
    normalize_continued_trial_line_indices,
    reconcile_cycle_clock_trials,
)


def _build_cycle_clock(group_sizes, period=0.02, gap=1.0):
    cycle_starts = []
    next_start = 0.0
    for group_size in group_sizes:
        group = next_start + np.arange(group_size) * period
        cycle_starts.extend(group)
        next_start = group[-1] + gap
    cycle_starts = np.asarray(cycle_starts)
    return cycle_starts, np.diff(cycle_starts)


class ReconcileCycleClockTrialsTests(unittest.TestCase):
    def test_matching_harp_and_summary_counts_are_unchanged(self):
        starts, periods = _build_cycle_clock([4, 5, 6])

        adjusted, _, qc = reconcile_cycle_clock_trials(starts, periods, 3, 3)

        np.testing.assert_array_equal(adjusted, starts)
        self.assertEqual(qc['removed_leading_cycle_count'], 0)

    def test_removes_one_leading_group_when_all_sources_agree(self):
        starts, periods = _build_cycle_clock([2, 4, 5, 6])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            adjusted, _, qc = reconcile_cycle_clock_trials(starts, periods, 3, 3)

        np.testing.assert_array_equal(adjusted, starts[2:])
        self.assertEqual(qc['removed_leading_cycle_count'], 2)

    def test_does_not_remove_group_when_dat_span_is_short(self):
        starts, periods = _build_cycle_clock([2, 4, 5, 6])

        adjusted, _, qc = reconcile_cycle_clock_trials(starts, periods, 3, 2)

        np.testing.assert_array_equal(adjusted, starts)
        self.assertEqual(qc['removed_leading_cycle_count'], 0)

    def test_does_not_remove_multiple_extra_groups(self):
        starts, periods = _build_cycle_clock([2, 3, 4, 5, 6])

        adjusted, _, qc = reconcile_cycle_clock_trials(starts, periods, 3, 3)

        np.testing.assert_array_equal(adjusted, starts)
        self.assertEqual(qc['removed_leading_cycle_count'], 0)


class NormalizeContinuedTrialLineIndicesTests(unittest.TestCase):
    def test_rebases_clear_continued_counter_and_prints_change(self):
        counts = np.array([4, 4, 4])
        lines = np.array([
            1, 30, 60, 100,
            101, 130, 160, 200,
            1, 30, 60, 100,
        ])
        output = StringIO()

        with redirect_stdout(output):
            normalized, corrections = normalize_continued_trial_line_indices(
                lines, counts, plane_name="DMD2"
            )

        np.testing.assert_array_equal(
            normalized,
            np.array([1, 30, 60, 100, 1, 30, 60, 100, 1, 30, 60, 100]),
        )
        self.assertEqual(corrections[0]['trial_number'], 2)
        self.assertEqual(corrections[0]['offset'], 100)
        self.assertIn("DMD2 TRIAL LINE-INDEX ISSUE: trial 2", output.getvalue())
        self.assertIn("changing range [101, 200] to [1, 100]", output.getvalue())

    def test_leaves_in_range_partial_trial_unchanged(self):
        counts = np.array([4, 3, 4])
        lines = np.array([
            1, 30, 60, 100,
            20, 40, 60,
            1, 30, 60, 100,
        ])

        normalized, corrections = normalize_continued_trial_line_indices(lines, counts)

        np.testing.assert_array_equal(normalized, lines)
        self.assertEqual(corrections, [])

    def test_leaves_nonmonotone_out_of_range_trial_unchanged(self):
        counts = np.array([4, 4, 4])
        lines = np.array([
            1, 30, 60, 100,
            101, 150, 140, 200,
            1, 30, 60, 100,
        ])

        normalized, corrections = normalize_continued_trial_line_indices(lines, counts)

        np.testing.assert_array_equal(normalized, lines)
        self.assertEqual(corrections, [])


class TrialLineTimeMapTests(unittest.TestCase):
    def test_uses_one_control_point_per_cycle_boundary(self):
        control_lines, control_times = _build_trial_line_time_map(
            np.array([1.0, 2.0, 3.0]), effective_lpc=100.0, n_detected_cycles=3
        )

        np.testing.assert_array_equal(control_lines, np.array([1.0, 101.0, 201.0, 301.0]))
        np.testing.assert_array_equal(control_times, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_preserves_variable_harp_cycle_periods(self):
        control_lines, control_times = _build_trial_line_time_map(
            np.array([1.0, 2.2, 3.1]), effective_lpc=100.0, n_detected_cycles=3
        )

        np.testing.assert_array_equal(control_lines, np.array([1.0, 101.0, 201.0, 301.0]))
        np.testing.assert_allclose(control_times, np.array([1.0, 2.2, 3.1, 4.15]))

    def test_secondary_interpolation_is_strictly_increasing(self):
        line_time_maps = [_build_trial_line_time_map(
            np.array([1.0, 2.0, 3.0]), effective_lpc=100.0, n_detected_cycles=3
        )]
        secondary_lines = np.array([1, 51, 97, 103, 151, 201, 251, 300])

        timestamps = get_slap2_secondary_plane_timestamps(
            secondary_lines, np.array([len(secondary_lines)]), line_time_maps
        )

        self.assertTrue(np.all(np.diff(timestamps) > 0))
        np.testing.assert_allclose(
            timestamps, np.array([1.0, 1.5, 1.96, 2.02, 2.5, 3.0, 3.5, 3.99])
        )


if __name__ == '__main__':
    unittest.main()