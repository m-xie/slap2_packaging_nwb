"""Boundary and input-validation tests for standalone continuous DI3 timing."""

import copy
import json
import unittest
import warnings

import numpy as np

from slap2_continuous_sync import build_continuous_clock, map_continuous_lines


def _sample_pulses(pulses):
    """Use irregular low-to-high sample spacing; edges occur at supplied times."""
    pulses = np.asarray(pulses, dtype=float)
    times = np.empty(2 * len(pulses), dtype=float)
    times[1::2] = pulses
    times[0] = pulses[0] - 0.1
    times[2::2] = pulses[:-1] + np.diff(pulses) * 0.37
    return np.tile([False, True], len(pulses)), times


class ContinuousClockTests(unittest.TestCase):
    def build(self, pulses, length, cycles, **kwargs):
        signal, times = _sample_pulses(pulses)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clock = build_continuous_clock(signal, times, length, cycles, **kwargs)
        return clock, caught

    def test_measured_extra_boundary_uses_actual_header_and_fractional_lines(self):
        clock, caught = self.build([2, 3, 6], np.int64(10), np.int64(2))
        self.assertEqual(caught, [])
        np.testing.assert_array_equal(clock["cycle_starts"], [2, 3, 6])
        np.testing.assert_array_equal(clock["control_line_idxs"], [1, 11, 21])
        np.testing.assert_array_equal(clock["control_timestamps"], [2, 3, 6])
        # Sparse sample spacing and maxline / C = 9, not the header's L = 10.
        lines = [1, 2, 8, 10, 11, 12, 18]
        np.testing.assert_allclose(
            map_continuous_lines(lines, clock), [2, 2.1, 2.7, 2.9, 3, 3.3, 5.1]
        )
        self.assertIs(type(clock["lines_per_cycle"]), int)
        self.assertIs(type(clock["total_cycles"]), int)
        self.assertEqual(clock["qc"]["last_cycle_policy"], "measured_end")
        self.assertEqual(clock["qc"]["extra_final_pulse_count"], 1)
        self.assertEqual(clock["qc"]["estimated_duration_cycle_count"], 0)
        self.assertIsNone(clock["qc"]["estimated_last_cycle_end_timestamp"])

    def test_equal_counts_estimate_only_last_cycle_using_mean_not_last_period(self):
        clock, caught = self.build([2, 3, 6], 10, 3)
        self.assertEqual(len(caught), 1)
        self.assertIn("Estimating only the end", str(caught[0].message))
        self.assertIs(caught[0].category, RuntimeWarning)
        np.testing.assert_array_equal(clock["cycle_starts"], [2, 3, 6])
        np.testing.assert_array_equal(clock["control_line_idxs"], [1, 11, 21, 31])
        np.testing.assert_array_equal(clock["control_timestamps"], [2, 3, 6, 8])
        np.testing.assert_allclose(
            map_continuous_lines([1, 11, 21, 26, 30], clock), [2, 3, 6, 7, 7.8]
        )
        self.assertEqual(clock["qc"]["estimated_mean_period_seconds"], 2)
        self.assertEqual(clock["qc"]["estimated_last_cycle_end_timestamp"], 8)
        self.assertEqual(clock["qc"]["estimated_duration_cycle_count"], 1)
        self.assertEqual(clock["qc"]["unsupported_raw_tail_cycle_count"], 0)

    def test_raw_tail_including_estimated_boundary_is_unsupported(self):
        clock, caught = self.build([2, 3, 6], 10, 5)
        self.assertEqual(len(caught), 2)
        self.assertIn("Unsupported raw tail: 2", str(caught[1].message))
        np.testing.assert_allclose(
            map_continuous_lines([21, 30, 31, 32, 41, 50], clock),
            [6, 7.8, np.nan, np.nan, np.nan, np.nan], equal_nan=True,
        )
        self.assertEqual(clock["qc"]["unsupported_raw_tail_cycle_count"], 2)
        self.assertEqual(clock["qc"]["observed_raw_cycle_count"], 3)
        self.assertEqual(clock["qc"]["measured_duration_cycle_count"], 2)

    def test_only_one_extra_pulse_is_allowed(self):
        for cycles in [0, 1, 2]:
            with self.subTest(cycles=cycles), self.assertRaisesRegex(
                ValueError, "at most one extra"
            ):
                self.build([1, 2, 3, 4], 10, cycles)

    def test_one_raw_cycle_with_measured_end(self):
        clock, caught = self.build([5, 7], 10, 1)
        self.assertEqual(caught, [])
        np.testing.assert_allclose(map_continuous_lines([1, 6, 10], clock), [5, 6, 6.8])

    def test_single_pulse_supports_only_exact_start(self):
        for cycles in [1, 4]:
            with self.subTest(cycles=cycles):
                clock, caught = self.build([5], 10, cycles)
                self.assertTrue(any("Only one DI3 pulse" in str(w.message) for w in caught))
                np.testing.assert_array_equal(clock["control_line_idxs"], [1])
                np.testing.assert_array_equal(clock["control_timestamps"], [5])
                np.testing.assert_allclose(
                    map_continuous_lines([1, 2, 10], clock), [5, np.nan, np.nan],
                    equal_nan=True,
                )
                self.assertTrue(np.isnan(map_continuous_lines([2], clock)[0]))
                self.assertEqual(clock["qc"]["last_cycle_policy"], "start_only")
                self.assertIsNone(clock["qc"]["estimated_mean_period_seconds"])
                self.assertEqual(len(caught), 1 + int(cycles > 1))

    def test_zero_cycles_allow_single_boundary_but_no_primary_lines(self):
        clock, caught = self.build([5], 10, 0)
        self.assertEqual(caught, [])
        self.assertEqual(map_continuous_lines([], clock).size, 0)
        with self.assertWarnsRegex(RuntimeWarning, "recorded_line_count"):
            self.assertTrue(np.isnan(map_continuous_lines([1], clock)[0]))

    def test_no_pulses_raise(self):
        for signal, times in [([0], [5]), ([0, 0, 0], [0, 1, 2])]:
            with self.subTest(signal=signal), self.assertRaisesRegex(ValueError, "No DI3"):
                build_continuous_clock(signal, times, 10, 2)

    def test_pre_zero_pulses_and_large_gaps_are_retained_sequentially(self):
        clock, caught = self.build([-20, -19, 1, 2], 10, 3)
        self.assertEqual(caught, [])
        np.testing.assert_array_equal(clock["cycle_starts"], [-20, -19, 1, 2])
        np.testing.assert_allclose(
            map_continuous_lines([1, 11, 16, 21, 26], clock), [-20, -19, -9, 1, 1.5]
        )

    def test_initial_high_after_recording_onset_warns_and_counts_as_first_pulse(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clock = build_continuous_clock([1, 0, 1, 0, 1], [4.0, 4.1, 5.0, 5.1, 6.0], 10, 2, recording_start=-1.0)
        self.assertFalse(clock["qc"]["onset_warning"])
        self.assertTrue(clock["qc"]["initial_signal_high"])
        self.assertEqual(len(caught), 1)
        self.assertIn("DI3 initially starts high", str(caught[0].message))
        np.testing.assert_array_equal(clock["cycle_starts"], [4.0, 5.0, 6.0])

    def test_initial_high_counts_as_cycle_one(self):
        with self.assertWarnsRegex(
            RuntimeWarning, "DI3 initially starts high"
        ):
            clock = build_continuous_clock([1, 0, 1], [-2, -1, 0], 10, 1)
        np.testing.assert_array_equal(clock["cycle_starts"], [-2, 0])
        np.testing.assert_allclose(map_continuous_lines([1, 6, 10], clock), [-2, -1, -0.2])
        self.assertEqual(clock["qc"]["detected_pulse_count"], 2)
        self.assertTrue(clock["qc"]["initial_signal_high"])
        self.assertTrue(clock["qc"]["onset_warning"])
        self.assertTrue(clock["qc"]["first_rising_edge_within_onset_tolerance"])
        self.assertEqual(clock["qc"]["recording_start"], -2)

    def test_high_only_single_sample_is_a_valid_first_pulse(self):
        for signal, times in [([True], [5]), ([1, 1, 1], [5, 6, 7]), ([1, 0], [5, 6])]:
            with self.subTest(signal=signal), self.assertWarnsRegex(RuntimeWarning, "DI3 initially starts high"):
                clock = build_continuous_clock(signal, times, 10, 1)
                np.testing.assert_array_equal(clock["cycle_starts"], [times[0]])

    def test_onset_tolerance_includes_exact_boundary(self):
        for edge, expected in [(0.0005, True), (0.001, True), (0.001001, False)]:
            with self.subTest(edge=edge), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clock = build_continuous_clock([0, 1, 0, 1], [0, edge, 0.5, 1], 10, 1)
                self.assertEqual(len(caught), int(expected))
                self.assertEqual(clock["qc"]["onset_warning"], expected)
                self.assertEqual(clock["qc"]["first_rising_edge_within_onset_tolerance"], expected)
                self.assertEqual(clock["qc"]["onset_tolerance_seconds"], 0.001)
                self.assertEqual(clock["cycle_starts"][0], edge)
                if expected:
                    self.assertIn("SLAP2 may have started before HARP began recording", str(caught[0].message))

    def test_explicit_recording_start_uses_supplied_time_base(self):
        signal = [0, 1, 0, 1]
        times = [-100, -99.9995, -99.5, -99]
        for start, expected in [(None, True), (-101, False), (-99.9996, True), (0, True)]:
            with self.subTest(start=start), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clock = build_continuous_clock(signal, times, 10, 1, recording_start=start)
                self.assertEqual(len(caught), int(expected))
                self.assertEqual(clock["qc"]["onset_warning"], expected)
                np.testing.assert_array_equal(clock["cycle_starts"], [-99.9995, -99])
        # A first edge near normalized zero is NOT near an earlier recording onset.
        clock, caught = self.build([0.0005, 1], 10, 1, recording_start=-100)
        self.assertEqual(caught, [])
        self.assertFalse(clock["qc"]["onset_warning"])

    def test_clock_qc_is_json_safe_for_each_policy_and_mapping_does_not_mutate(self):
        for pulses, cycles in [([1, 2, 4], 2), ([1, 2], 2), ([1, 2], 4), ([1], 4)]:
            with self.subTest(pulses=pulses, cycles=cycles):
                clock, _ = self.build(pulses, 10, cycles)
                before = copy.deepcopy(clock)
                encoded = json.dumps(clock["qc"], allow_nan=False)
                self.assertEqual(json.loads(encoded)["segmentation_method"], "sequential_di3")
                self.assertEqual(clock["qc"]["lines_per_cycle_source"], "actual_header")
                lines = np.array([1, 2, cycles * 10])
                saved_lines = lines.copy()
                result = map_continuous_lines(lines, clock)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_array_equal(lines, saved_lines)
                map_continuous_lines([1, 100], clock, recorded_line_count=100)
                self.assertEqual(json.dumps(clock["qc"], allow_nan=False), encoded)
                for key in ("cycle_starts", "control_line_idxs", "control_timestamps"):
                    np.testing.assert_array_equal(clock[key], before[key])


class ContinuousLineMappingTests(unittest.TestCase):
    def setUp(self):
        signal, times = _sample_pulses([1, 2, 4])
        self.clock = build_continuous_clock(signal, times, 10, 2)

    def test_default_primary_limit_masks_even_measured_next_boundary(self):
        np.testing.assert_allclose(map_continuous_lines([20], self.clock), [3.8])
        with self.assertWarnsRegex(RuntimeWarning, "recorded_line_count"):
            self.assertTrue(np.isnan(map_continuous_lines([21], self.clock)[0]))

    def test_secondary_longer_limit_includes_measured_boundary_but_no_clamping(self):
        result = map_continuous_lines([1, 20, 21, 22, 30], self.clock, recorded_line_count=30)
        np.testing.assert_allclose(result, [1, 3.8, 4, np.nan, np.nan], equal_nan=True)
        self.assertEqual(np.count_nonzero(np.isnan(result)), 2)
        with self.assertWarnsRegex(RuntimeWarning, "recorded_line_count"):
            self.assertTrue(np.isnan(map_continuous_lines([31], self.clock, recorded_line_count=30)[0]))

    def test_secondary_shorter_limit_is_enforced_without_changing_clock(self):
        np.testing.assert_allclose(
            map_continuous_lines([6, 11, 12], self.clock, recorded_line_count=12),
            [1.5, 2, 2.2],
        )
        with self.assertWarnsRegex(RuntimeWarning, "recorded_line_count"):
            result = map_continuous_lines([6, 12, 13, 14], self.clock, recorded_line_count=12)
        np.testing.assert_allclose(result, [1.5, 2.2, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(map_continuous_lines([13], self.clock), [2.4])

    def test_secondary_cannot_use_estimated_end_as_next_cycle_start(self):
        signal, times = _sample_pulses([1, 2])
        with self.assertWarnsRegex(RuntimeWarning, "Estimating"):
            clock = build_continuous_clock(signal, times, 10, 2)
        np.testing.assert_allclose(
            map_continuous_lines([20, 21, 22], clock, recorded_line_count=30),
            [2.9, np.nan, np.nan], equal_nan=True,
        )

    def test_unit_cycle_length(self):
        signal, times = _sample_pulses([1, 3, 4])
        with self.assertWarnsRegex(RuntimeWarning, "Estimating"):
            clock = build_continuous_clock(signal, times, 1, 3)
        np.testing.assert_array_equal(map_continuous_lines([1, 2, 3], clock), [1, 3, 4])

    def test_empty_lines_are_typed_float_arrays(self):
        for lines in [[], np.array([], dtype=np.int64), np.array([], dtype=float)]:
            result = map_continuous_lines(lines, self.clock, recorded_line_count=0)
            self.assertEqual(result.shape, (0,))
            self.assertEqual(result.dtype, np.dtype("float64"))

    def test_integral_float_lines_are_not_rebased(self):
        np.testing.assert_allclose(map_continuous_lines([6.0, 16.0], self.clock), [1.5, 3])

    def test_bad_line_coordinates_raise(self):
        bad_lines = [
            [0], [-1], [np.nan], [np.inf], [-np.inf], [1.1], [1, 1], [2, 1],
            [1, 11, 1], [[1, 2]], 1, [True], ["1"], [1 + 0j], [2**53 + 1],
        ]
        for lines in bad_lines:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                map_continuous_lines(lines, self.clock)

    def test_bad_recorded_line_limit_raises_even_for_empty_samples(self):
        for limit in [-1, 2.5, np.nan, np.inf, True, "20", [20], 2**53]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                map_continuous_lines([], self.clock, recorded_line_count=limit)


class ContinuousInputValidationTests(unittest.TestCase):
    def test_bad_signal_and_time_inputs(self):
        cases = [
            ([], []), ([0], [0, 1]), ([0, 1], [0]),
            ([[0, 1]], [0, 1]), ([0, 1], [[0, 1]]), (1, [0]),
            ([0, 2], [0, 1]), ([-1, 1], [0, 1]), ([0, 0.5], [0, 1]),
            ([0, np.nan], [0, 1]), ([0, np.inf], [0, 1]),
            ([0, 1], [0, np.nan]), ([0, 1], [0, np.inf]),
            ([0, 1], [1, 1]), ([0, 1], [1, 0]),
            (["0", "1"], [0, 1]), ([0, 1], ["0", "1"]),
            ([0, 1j], [0, 1]), ([0, 1], [0, 1j]),
        ]
        for signal, times in cases:
            with self.subTest(signal=signal, times=times), self.assertRaises(ValueError):
                build_continuous_clock(signal, times, 10, 1)

    def test_bad_cycle_lengths_and_counts(self):
        for field in ["lines_per_cycle", "total_cycles"]:
            invalid = [-1, 1.5, np.nan, np.inf, True, "2", [2], 1j, 2**53]
            if field == "lines_per_cycle":
                invalid.append(0)
            for value in invalid:
                kwargs = {"lines_per_cycle": 10, "total_cycles": 1, field: value}
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    build_continuous_clock([0, 1], [0, 1], **kwargs)

    def test_rejects_unrepresentable_cycle_boundaries(self):
        with self.assertRaisesRegex(ValueError, "exact float64"):
            build_continuous_clock([0, 1], [0, 1], 2**52, 2)

    def test_bad_recording_start(self):
        for start in [np.nan, np.inf, -np.inf, [0], "0", True, 1j]:
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, "recording_start"):
                build_continuous_clock([0, 1], [0, 1], 10, 1, recording_start=start)

    def test_nonfinite_estimated_end_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Estimated last-cycle end"):
            build_continuous_clock([0, 1, 0, 1], [0, 1e308, 1.1e308, 1.7e308], 10, 2)

    def test_builder_does_not_mutate_signal_or_times(self):
        signal, times = _sample_pulses([1, 2, 3])
        original_signal, original_times = signal.copy(), times.copy()
        clock = build_continuous_clock(signal, times, 10.0, 2.0)
        clock["cycle_starts"][0] = 42
        clock["control_timestamps"][0] = 43
        np.testing.assert_array_equal(signal, original_signal)
        np.testing.assert_array_equal(times, original_times)


if __name__ == "__main__":
    unittest.main()