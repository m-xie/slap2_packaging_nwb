"""Semantic tests for movie timing; all generated inputs live in temp directories.

The end-to-end oracle is a known display-frame/HARP clock, not the mapping
implementation. Only interrupted-session tests replace photodiode alignment.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

import random_natural_movies as movies
import stimulus_sync
from tests.test_random_natural_movies import example_session


def synthetic_harp(logger, *, slope=1.0008 / 60, offset=0.023, sample_period=0.0001):
    """Sample a two-level ADC trace independently of the production extractor.

    HARP times are already normalized to DO0. Logger timestamps are deliberately
    irrelevant. The initial low state is a baseline, not a physical edge.
    """
    reference = logger.loc[logger.Value.eq("STARTSLAP"), "Frame"].iloc[0]
    diode = logger.loc[logger.Value.isin(("Photodiode-0", "Photodiode-1"))]
    states = diode.Value.eq("Photodiode-1").to_numpy()
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    edge_frames = diode.Frame.to_numpy()[changes]
    edge_times = offset + slope * (edge_frames - reference)
    recording_end = offset + slope * (logger.Frame.max() - reference) + 0.1
    analog_times = np.arange(0, recording_end, sample_period)
    high = np.searchsorted(edge_times, analog_times, side="right") % 2 == 1
    harp = {
        "analog_times": analog_times,
        "photodiode": np.where(high, 3.7, 0.2),
        "normalized_slap2_start": np.array([0.0]),
        "normalized_slap2_end": np.array([analog_times[-1]]),
        "time_reference": 12345.0,
    }
    return harp, edge_frames, edge_times


class MovieFrameMappingTests(unittest.TestCase):
    def assert_mapping(self, frames, anchors, times, gap, expected_times, expected_status):
        actual_times, actual_status = movies.map_movie_frames(frames, anchors, times, gap)
        np.testing.assert_allclose(actual_times, expected_times, rtol=0, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(actual_status, expected_status)
        self.assertEqual(actual_times.dtype, np.dtype("float64"))
        self.assertEqual(actual_status.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(np.isnan(actual_times), actual_status == 3)
        return actual_times, actual_status

    def test_nonlinear_drift_uses_local_interpolation_not_global_affine_fit(self):
        anchors = np.array([10, 20, 30, 40])
        times = np.array([0.1, 0.2, 0.5, 0.6])
        frames = np.array([10, 15, 20, 25, 30, 35, 40])
        expected = [0.1, 0.15, 0.2, 0.35, 0.5, 0.55, 0.6]
        self.assertGreater(np.max(np.abs(np.polyval(np.polyfit(anchors, times, 1), frames) - expected)), 0.02)
        self.assert_mapping(frames, anchors, times, 10, expected, [0, 1, 0, 1, 0, 1, 0])

    def test_large_gap_rejects_interior_but_preserves_both_exact_anchor_edges(self):
        self.assert_mapping(
            [0, 5, 10, 11, 50, 99, 100, 105, 110],
            [0, 10, 100, 110], [0, 1, 10, 11], 10,
            [0, 0.5, 1, np.nan, np.nan, np.nan, 10, 10.5, 11],
            [0, 1, 0, 3, 3, 3, 0, 1, 0],
        )

    def test_gap_threshold_is_inclusive(self):
        for gap, expected, status in ((10, [0.5], [1]), (9.999, [np.nan], [3])):
            with self.subTest(gap=gap):
                self.assert_mapping([5], [0, 10], [0, 1], gap, expected, status)

    def test_two_frame_endpoint_limit_inclusive_and_beyond_is_nan_not_clamped(self):
        self.assert_mapping(
            [7, 7.999, 8, 9, 10, 15, 20, 21, 22, 22.001, 23],
            [10, 20], [1, 2], 10,
            [np.nan, np.nan, 0.8, 0.9, 1, 1.5, 2, 2.1, 2.2, np.nan, np.nan],
            [3, 3, 2, 2, 0, 1, 0, 2, 2, 3, 3],
        )

    def test_endpoint_extension_uses_local_not_session_wide_slope(self):
        anchors = np.arange(50, dtype=float)
        times = np.where(anchors <= 24, 1 + anchors * 0.01, 1.24 + (anchors - 24) * 0.03)
        self.assert_mapping([-2, 0, 49, 51], anchors, times, 1, [0.98, 1, 1.99, 2.05], [2, 0, 0, 2])

    def test_entirely_unsupported_frames_return_nan_without_raising(self):
        self.assert_mapping([-3, 5, 13], [0, 10], [0, 1], 1, [np.nan] * 3, [3] * 3)

    def test_empty_frames_return_typed_empty_arrays(self):
        times, status = self.assert_mapping([], [0, 10], [0, 1], 10, [], [])
        self.assertEqual(times.shape, (0,))
        self.assertEqual(status.shape, (0,))

    def test_invalid_anchors_are_rejected_even_without_movie_frames(self):
        cases = [
            ([], []), ([0], [0]), ([0, 1], [0]),
            ([[0, 1]], [[0, 1]]), ([0, 1], [[0, 1]]),
            ([0, 0], [0, 1]), ([1, 0], [0, 1]),
            ([0, 1], [0, 0]), ([0, 1], [1, 0]),
            ([0, np.nan], [0, 1]), ([0, np.inf], [0, 1]),
            ([0, 1], [0, np.nan]), ([0, 1], [0, np.inf]),
        ]
        for anchors, times in cases:
            for frames in ([], [0.5]):
                with self.subTest(anchors=anchors, times=times, frames=frames):
                    with self.assertRaisesRegex(ValueError, "anchors"):
                        movies.map_movie_frames(frames, anchors, times, 10)

    def test_invalid_gap_parameters_are_rejected(self):
        for gap in (0, -1, np.nan, np.inf, -np.inf):
            with self.subTest(gap=gap):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    movies.map_movie_frames([], [0, 1], [0, 1], gap)

    def test_invalid_movie_display_coordinates_are_rejected(self):
        for frames in ([1, 1], [2, 1], [np.nan], [np.inf], [[1, 2]], 1):
            with self.subTest(frames=frames):
                with self.assertRaisesRegex(ValueError, "Movie display frames"):
                    movies.map_movie_frames(frames, [0, 10], [0, 1], 10)

    def test_inputs_are_not_modified(self):
        arrays = [np.array([0, 5, 10]), np.array([0, 10]), np.array([0.0, 1.0])]
        originals = [array.copy() for array in arrays]
        for array in arrays:
            array.setflags(write=False)
        movies.map_movie_frames(*arrays, maximum_gap_frames=10)
        for array, original in zip(arrays, originals):
            np.testing.assert_array_equal(array, original)


class MovieFrameSynchronizationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "logger.csv"
        self.table, self.logger = example_session()

    def write_logger(self):
        self.logger.to_csv(self.path, index=False)

    def test_real_photodiode_alignment_affine_drift_and_irregular_movie_spacing(self):
        # Remap all global coordinates consistently; content counters still
        # reset 1..4 and cannot substitute for the display frame coordinate.
        rng = np.random.default_rng(20261004)
        remap = 700 + np.r_[0, np.cumsum(rng.integers(1, 4, int(self.logger.Frame.max())))]
        self.logger["Frame"] = remap[self.logger.Frame.to_numpy()]
        self.logger["Timestamp"] = -987654  # Not a clock source.
        self.table.loc[self.table.TrialType.eq("movie"), "TextureName"] = "repeated_movie"
        self.write_logger()
        slope, offset, sample_period = 1.0008 / 60, 0.023, 0.0001
        harp, edge_frames, edge_times = synthetic_harp(self.logger, slope=slope, offset=offset)
        saved_table = self.table.copy(deep=True)
        # Exercise both DO1 cutoff and analog-tail fallback with real matching.
        for has_do1 in (True, False):
            with self.subTest(has_do1=has_do1):
                data = dict(harp)
                if not has_do1:
                    data["normalized_slap2_end"] = np.array([])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    blocks, gratings, metadata = movies.synchronize_presentations(self.table, self.path, data)
                self.assertEqual([str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)], [])
                alignment = blocks.attrs["movie_frame_alignment"]
                np.testing.assert_array_equal(alignment["anchor_frames"], edge_frames)
                np.testing.assert_allclose(alignment["anchor_times"], edge_times, rtol=0, atol=sample_period)
                self.assertEqual(metadata["matched_transition_count"], len(edge_frames))
                self.assertAlmostEqual(metadata["frame_rate_hz"], 1 / slope, delta=0.01)
                self.assertLess(metadata["maximum_absolute_residual_ms"], 0.2)
                for table in (blocks, gratings):
                    for coordinate, timestamp in (("start_frame", "start_time"), ("stop_frame", "stop_time")):
                        expected = offset + slope * (table[coordinate].to_numpy() - 700)
                        np.testing.assert_allclose(table[timestamp], expected, rtol=0, atol=sample_period)
                    np.testing.assert_allclose(table.Duration, table.stop_time - table.start_time)
                event_frames = self.logger.loc[self.logger.Value.str.startswith("MovieFrame-"), "Frame"].to_numpy().reshape(3, 4)
                self.assertGreater(len(np.unique(np.diff(event_frames, axis=1))), 1)
                movie_rows = blocks.loc[blocks.TrialType.eq("movie")]
                for row, frames in zip(movie_rows.itertuples(), event_frames):
                    np.testing.assert_array_equal(row.movie_frame_numbers, [1, 2, 3, 4])
                    np.testing.assert_array_equal(row.movie_display_frames, frames)
                    np.testing.assert_allclose(row.movie_frame_timestamps, offset + slope * (frames - 700), rtol=0, atol=sample_period)
                    expected_status = np.where(np.isin(frames, edge_frames), 0, 1)
                    np.testing.assert_array_equal(row.movie_frame_timing_status, expected_status)
                    self.assertEqual(row.movie_frame_count, 4)
                    self.assertTrue(np.all(np.diff(row.movie_frame_timestamps) > 0))
                    self.assertTrue(np.all(row.movie_frame_timestamps >= row.start_time))
                    self.assertTrue(np.all(row.movie_frame_timestamps <= row.stop_time))
                    self.assertEqual(row.movie_frame_numbers.dtype, np.dtype("int64"))
                    self.assertEqual(row.movie_display_frames.dtype, np.dtype("int64"))
                    self.assertEqual(row.movie_frame_timestamps.dtype, np.dtype("float64"))
                    self.assertEqual(row.movie_frame_timing_status.dtype, np.dtype("uint8"))
                for row in blocks.loc[blocks.TrialType.eq("gratings")].itertuples():
                    for values in (row.movie_frame_numbers, row.movie_display_frames, row.movie_frame_timestamps, row.movie_frame_timing_status):
                        self.assertEqual(len(values), 0)
                timing = metadata["movie_frame_timing"]
                self.assertEqual(timing["logged_frame_count"], 12)
                self.assertEqual(timing["stored_frame_count"], 12)
                self.assertEqual(timing["omitted_frame_count"], 0)
                anchored = int(np.isin(event_frames, edge_frames).sum())
                self.assertGreater(anchored, 0)
                self.assertLess(anchored, 12)
                self.assertEqual(timing["status_counts"], {"anchored": anchored, "interpolated": 12 - anchored, "extrapolated": 0, "unsupported": 0})
                self.assertEqual(timing["maximum_interpolation_gap_frames"], 3 * np.median(np.diff(edge_frames)))
                self.assertEqual(timing["harp_time_reference_seconds"], 12345.0)
                self.assertEqual(timing["status_codes"], {"anchored": 0, "interpolated": 1, "extrapolated": 2, "unsupported": 3})
        pd.testing.assert_frame_equal(self.table, saved_table)

    def test_explicit_gap_override_marks_unanchored_frames_unsupported(self):
        self.write_logger()
        harp, edge_frames, _ = synthetic_harp(self.logger)
        with self.assertWarnsRegex(RuntimeWarning, "unsupported timing"):
            blocks, _, metadata = movies.synchronize_presentations(self.table, self.path, harp, maximum_interpolation_gap_frames=2)
        frames = np.concatenate(blocks.movie_display_frames.to_list())
        times = np.concatenate(blocks.movie_frame_timestamps.to_list())
        statuses = np.concatenate(blocks.movie_frame_timing_status.to_list())
        anchored = np.isin(frames, edge_frames)
        np.testing.assert_array_equal(statuses, np.where(anchored, 0, 3))
        np.testing.assert_array_equal(np.isnan(times), ~anchored)
        timing = metadata["movie_frame_timing"]
        self.assertEqual(timing["maximum_interpolation_gap_frames"], 2)
        self.assertEqual(timing["large_anchor_gap_count"], len(edge_frames) - 1)
        self.assertEqual(timing["status_counts"]["unsupported"], int((~anchored).sum()))
        self.assertTrue(np.isfinite(blocks[["start_time", "stop_time"]]).all().all())

    def test_recovery_keeps_counters_beyond_clipped_interval_and_counts_dropped_block(self):
        self.logger = self.logger.loc[~self.logger.Value.isin(("END", "EndFrame"))]
        self.write_logger()
        harp, _, _ = synthetic_harp(self.logger)
        anchors = np.array([3, 12, 18, 20, 22], dtype=float)
        times = anchors / 60
        qc = stimulus_sync.AlignmentQC(str(self.path), 50, 50, len(anchors), 60, 0, 0, 0)
        with patch.object(stimulus_sync, "align_logger_frames_to_harp", return_value=(anchors, times, qc)) as align:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                blocks, gratings, metadata = movies.synchronize_presentations(self.table, self.path, harp)
        align.assert_called_once()
        np.testing.assert_array_equal(blocks.stimulus_table_row, [0, 1])
        self.assertTrue(gratings.empty)
        row = blocks.iloc[1]
        self.assertTrue(row.is_partial)
        self.assertEqual(row.stop_frame_source, "photodiode_tail_censored")
        self.assertEqual(row.stop_frame, 22)
        self.assertEqual(row.stop_time, 22 / 60)
        self.assertEqual(row.movie_frame_count, 4)
        np.testing.assert_array_equal(row.movie_frame_numbers, [1, 2, 3, 4])
        np.testing.assert_array_equal(row.movie_display_frames, [18, 20, 22, 24])
        np.testing.assert_allclose(row.movie_frame_timestamps, [18 / 60, 20 / 60, 22 / 60, np.nan], equal_nan=True)
        np.testing.assert_array_equal(row.movie_frame_timing_status, [0, 0, 0, 3])
        # Frame 24 is within ordinary endpoint extrapolation, but outside the
        # recovered interval. Censoring must override its otherwise valid map.
        ordinary_times, ordinary_status = movies.map_movie_frames([24], anchors, times, 9)
        self.assertTrue(np.isfinite(ordinary_times[0]))
        self.assertEqual(ordinary_status[0], 2)
        timing = metadata["movie_frame_timing"]
        self.assertEqual(timing["logged_frame_count"], 12)
        self.assertEqual(timing["stored_frame_count"], 8)
        self.assertEqual(timing["omitted_frame_count"], 4)
        self.assertEqual(timing["status_counts"], {"anchored": 4, "interpolated": 3, "extrapolated": 0, "unsupported": 1})
        # Default is based on the complete logged transitions, not sparse or
        # truncated matched anchors (whose median spacing differs here).
        self.assertEqual(timing["maximum_interpolation_gap_frames"], 9)
        self.assertNotEqual(3 * np.median(np.diff(anchors)), 9)
        self.assertEqual(metadata["omitted_stimulus_table_row_count"], 3)
        self.assertEqual(metadata["partial_stimulus_table_row_count"], 1)
        self.assertFalse(metadata["has_session_end"])
        messages = [str(w.message) for w in caught]
        for fragment in ("no END or EndFrame", "beyond photodiode coverage", "1 logged movie frames"):
            self.assertTrue(any(fragment in message for message in messages), messages)
            self.assertTrue(any(fragment in message for message in metadata["recovery_warnings"]))

    def test_missing_and_duplicate_counters_rejected_before_alignment(self):
        original = self.logger.copy(deep=True)
        for corruption in ("missing", "duplicate"):
            with self.subTest(corruption=corruption):
                self.logger = original.copy(deep=True)
                index = self.logger.index[self.logger.Value.eq("MovieFrame-3")][0]
                if corruption == "missing":
                    self.logger = self.logger.drop(index)
                else:
                    # Distinct display coordinate: failure must be the content
                    # counter, not duplicate presentation/global timestamps.
                    self.logger.loc[index, "Value"] = "MovieFrame-2"
                self.write_logger()
                with patch.object(stimulus_sync, "align_logger_frames_to_harp") as align:
                    with self.assertRaisesRegex(ValueError, "Missing or duplicate MovieFrame"):
                        movies.synchronize_presentations(self.table, self.path, {})
                align.assert_not_called()

    def test_repeat_counters_reset_without_merging_adjacent_presentations(self):
        self.table.loc[self.table.TrialType.eq("movie"), "TextureName"] = "same_movie"
        self.write_logger()
        harp, _, _ = synthetic_harp(self.logger)
        blocks, _, _ = movies.synchronize_presentations(self.table, self.path, harp)
        repeats = blocks.loc[blocks.TrialType.eq("movie")]
        self.assertEqual(repeats.stimulus_table_row.tolist(), [0, 1, 4])
        self.assertEqual(blocks.iloc[0].stop_time, blocks.iloc[1].start_time)
        for row in repeats.itertuples():
            np.testing.assert_array_equal(row.movie_frame_numbers, [1, 2, 3, 4])
            self.assertEqual(len(row.movie_frame_timestamps), 4)
        self.assertTrue(np.all(np.diff(np.concatenate(repeats.movie_frame_timestamps.to_list())) > 0))

    def test_invalid_integration_gap_override_is_rejected(self):
        self.write_logger()
        harp, _, _ = synthetic_harp(self.logger)
        for gap in (0, -1, np.nan, np.inf, -np.inf):
            with self.subTest(gap=gap):
                with self.assertRaisesRegex(ValueError, "maximum_interpolation_gap_frames.*finite and positive"):
                    movies.synchronize_presentations(self.table, self.path, harp, maximum_interpolation_gap_frames=gap)

    def test_grating_only_session_has_empty_movie_arrays_and_zero_frame_counts(self):
        self.table, self.logger = example_session(("gratings",))
        self.write_logger()
        harp, _, _ = synthetic_harp(self.logger)
        blocks, gratings, metadata = movies.synchronize_presentations(self.table, self.path, harp)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(gratings), 9)
        row = blocks.iloc[0]
        for column, dtype in (
            ("movie_frame_numbers", "int64"), ("movie_display_frames", "int64"),
            ("movie_frame_timestamps", "float64"), ("movie_frame_timing_status", "uint8"),
        ):
            self.assertEqual(row[column].shape, (0,))
            self.assertEqual(row[column].dtype, np.dtype(dtype))
        timing = metadata["movie_frame_timing"]
        for key in ("logged_frame_count", "stored_frame_count", "omitted_frame_count"):
            self.assertEqual(timing[key], 0)
        self.assertEqual(timing["status_counts"], {"anchored": 0, "interpolated": 0, "extrapolated": 0, "unsupported": 0})


if __name__ == "__main__":
    unittest.main()