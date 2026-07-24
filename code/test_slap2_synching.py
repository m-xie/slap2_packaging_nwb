import unittest
import warnings

import numpy as np

from slap2_synching import reconcile_cycle_clock_trials


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


if __name__ == '__main__':
    unittest.main()