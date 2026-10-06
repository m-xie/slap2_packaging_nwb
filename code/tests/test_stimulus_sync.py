import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from stimulus_sync import (
    LoggerEvents,
    align_logger_frames_to_harp,
    extract_harp_photodiode_transitions,
    extract_logger_events,
    resolve_stimulus_start_times,
    select_stimulus_logger,
)


class StimulusSyncTests(unittest.TestCase):
    def test_uses_analog_recording_bound_when_end_pulse_is_missing(self):
        transition_times, transition_states, _ = (
            extract_harp_photodiode_transitions({
                "analog_times": np.arange(7.0),
                "photodiode": np.asarray([0, 0, 1, 0, 1, 0, 0]),
                "normalized_slap2_start": np.asarray([0.0]),
                "normalized_slap2_end": np.asarray([]),
            })
        )

        np.testing.assert_array_equal(transition_times, [2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(
            transition_states, [True, False, True, False]
        )

    def test_selects_dup_logger_then_first_sorted(self):
        self.assertEqual(
            select_stimulus_logger([Path("z.csv"), Path("a_dup.csv")]),
            Path("a_dup.csv"),
        )
        self.assertEqual(
            select_stimulus_logger([Path("z.csv"), Path("a.csv")]),
            Path("a.csv"),
        )

    def test_extracts_starts_and_state_changes(self):
        rows = [
            (10, 1.0, "STARTSLAP"),
            (10, 1.0, "StimStart-one"),
            (10, 1.0, "Photodiode-0"),
            (11, 1.1, "Photodiode-0"),
            (12, 1.2, "Photodiode-1"),
            (13, 1.3, "Photodiode-0"),
            (14, 1.4, "Photodiode-1"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logger.csv"
            pd.DataFrame(rows, columns=["Frame", "Timestamp", "Value"]).to_csv(
                path, index=False
            )
            result = extract_logger_events(path)

        self.assertEqual(result.reference_frame, 10)
        np.testing.assert_array_equal(result.stimulus_frames, [10])
        np.testing.assert_array_equal(result.transition_frames, [10, 12, 13, 14])
        np.testing.assert_array_equal(
            result.transition_states, [False, True, False, True]
        )

    def test_alignment_rejects_missing_and_extra_transitions(self):
        rng = np.random.default_rng(12)
        run_lengths = rng.integers(4, 45, size=180)
        frames = np.cumsum(run_lengths) + 100
        states = np.arange(len(frames)) % 2 == 0
        slope = 1 / 59.997
        intercept = 0.064
        harp_times = slope * (frames - 100) + intercept
        harp_times += rng.normal(0, 0.002, size=len(harp_times))
        keep_logger = np.ones(len(frames), dtype=bool)
        keep_logger[[30, 91]] = False
        logger_data = type("Logger", (), {
            "path": Path("synthetic.csv"),
            "reference_frame": 100,
            "transition_frames": frames[keep_logger],
            "transition_states": states[keep_logger],
        })()
        harp_times = np.insert(harp_times, 75, harp_times[74] + 0.08)
        harp_states = np.insert(states, 75, ~states[74])

        with self.assertRaisesRegex(ValueError, "count mismatch: logger=178, HARP=181"):
            align_logger_frames_to_harp(logger_data, harp_times, harp_states)

    def test_full_analog_edges_are_independent_of_do_bounds(self):
        times = np.arange(-3.0, 7.0)
        signal = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        for markers in ({}, {"normalized_slap2_start": np.array([0.0]),
                             "normalized_slap2_end": np.array([2.0])},
                        {"normalized_slap2_start": np.array([100.0]),
                         "normalized_slap2_end": np.array([])}):
            with self.subTest(markers=markers):
                result, states, threshold = extract_harp_photodiode_transitions(
                    {"analog_times": times, "photodiode": signal, **markers})
                np.testing.assert_array_equal(result, times[1:9])
                np.testing.assert_array_equal(states, signal[1:9].astype(bool))
                self.assertEqual(threshold, 0.5)
        np.testing.assert_array_equal(times, np.arange(-3.0, 7.0))
        np.testing.assert_array_equal(signal, [0, 1, 0, 1, 0, 1, 0, 1, 0, 0])

    def test_non_increasing_analog_times_warn_without_changing_samples(self):
        for times in ([0, 1, 0.5, 3, 4, 5], [0, 1, 1, 3, 4, 5]):
            with self.subTest(times=times):
                times = np.array(times, dtype=float)
                signal = np.array([0, 1, 0, 1, 0, 0])
                times.setflags(write=False)
                signal.setflags(write=False)
                with self.assertWarnsRegex(RuntimeWarning, "1 non-increasing steps; first at sample index 2"):
                    edges, states, _ = extract_harp_photodiode_transitions(
                        {"analog_times": times, "photodiode": signal})
                np.testing.assert_array_equal(edges, times[1:5])
                np.testing.assert_array_equal(states, [True, False, True, False])
                logger = LoggerEvents(Path("clock.csv"), 0, np.array([]),
                                      np.arange(4), states)
                with self.assertRaisesRegex(ValueError, "HARP photodiode transitions"):
                    align_logger_frames_to_harp(logger, edges, states)

    def test_one_to_one_retains_every_edge_despite_large_timing_steps(self):
        frames = np.arange(100, 2100, 10)
        states = np.arange(len(frames)) % 2 == 0
        times = frames / 60 + 0.25
        times[50:] += 0.5
        times[100:] += 0.75
        logger = LoggerEvents(Path("steps.csv"), 99999, np.array([]), frames, states)
        for array in (frames, times, states):
            array.setflags(write=False)
        with patch("stimulus_sync.np.polyfit", side_effect=AssertionError("No fit should run")):
            anchor_frames, anchor_times, qc = align_logger_frames_to_harp(logger, times, states)
        np.testing.assert_array_equal(anchor_frames, frames)
        np.testing.assert_array_equal(anchor_times, times)
        self.assertEqual(qc.matched_transition_count, len(frames))
        self.assertEqual(qc.matching_method, "strict_one_to_one")
        self.assertEqual(qc.residual_reference, "endpoint_secant_descriptive_only")
        self.assertGreater(qc.p95_absolute_residual_ms, 40)
        midpoint = (frames[49] + frames[50]) / 2
        self.assertEqual(np.interp(midpoint, anchor_frames, anchor_times), (times[49] + times[50]) / 2)

    def test_equal_counts_with_wrong_polarity_fail(self):
        logger = LoggerEvents(Path("polarity.csv"), 0, np.array([]),
                              np.array([10, 20, 30, 40]), np.array([1, 0, 1, 0]))
        with self.assertRaisesRegex(ValueError, "polarity mismatch at index 0"):
            align_logger_frames_to_harp(logger, [1, 2, 3, 4], [0, 1, 0, 1])

    def test_extra_terminal_fall_is_not_dropped(self):
        logger = LoggerEvents(Path("tail.csv"), 0, np.array([]),
                              np.array([10, 20, 30]), np.array([1, 0, 1]))
        times, states, _ = extract_harp_photodiode_transitions({
            "analog_times": np.arange(6.0), "photodiode": np.array([0, 1, 0, 1, 0, 0]),
            "normalized_slap2_start": np.array([1.5]), "normalized_slap2_end": np.array([2.5]),
        })
        np.testing.assert_array_equal(times, [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "count mismatch: logger=3, HARP=4"):
            align_logger_frames_to_harp(logger, times, states)

    def test_invalid_transition_arrays_fail_without_filtering(self):
        logger = LoggerEvents(Path("invalid.csv"), 0, np.array([]),
                              np.array([10, 20, 30]), np.array([1, 0, 1]))
        for times, states in (([1, 1, 3], [1, 0, 1]), ([1, np.nan, 3], [1, 0, 1]),
                              ([1, 2, 3], [1, 0]), ([1, 2, 3], [1, 2, 1])):
            with self.subTest(times=times, states=states):
                with self.assertRaisesRegex(ValueError, "HARP photodiode transitions"):
                    align_logger_frames_to_harp(logger, times, states)

    def test_low_baseline_only_inserts_initial_high_state(self):
        for initial_high in (False, True):
            for first_frame in (9, 10, 11, 30):
                with self.subTest(initial_high=initial_high, first_frame=first_frame):
                    rows = [(10, 1.0, "STARTSLAP"), (10, 1.0, "StimStart-one")]
                    states = [initial_high, initial_high, not initial_high, initial_high,
                              not initial_high, not initial_high]
                    rows.extend((first_frame + i, 1 + i / 60, f"Photodiode-{int(state)}") for i, state in enumerate(states))
                    rows.sort(key=lambda row: row[0])
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "logger.csv"
                        pd.DataFrame(rows, columns=["Frame", "Timestamp", "Value"]).to_csv(path, index=False)
                        result = extract_logger_events(path, initial_low_baseline=True)
                        legacy = extract_logger_events(path)
                    changes = np.array([2, 3, 4]) + first_frame
                    expected = np.insert(changes, 0, first_frame) if initial_high else changes
                    np.testing.assert_array_equal(result.transition_frames, expected)
                    np.testing.assert_array_equal(result.transition_states,
                                                  [True, False, True, False] if initial_high else [True, False, True])
                    # Legacy mode still inserts the first state only at STARTSLAP.
                    legacy_expected = np.insert(changes, 0, first_frame) if first_frame == 10 else changes
                    np.testing.assert_array_equal(legacy.transition_frames, legacy_expected)

    def test_observed_initial_falling_edge_is_not_removed(self):
        rows = [(9, 0.9, "Photodiode-1"), (10, 1, "STARTSLAP"), (10, 1, "StimStart-one")]
        rows.extend((10 + i, 1 + i / 60, f"Photodiode-{i % 2}") for i in range(4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logger.csv"
            pd.DataFrame(rows, columns=["Frame", "Timestamp", "Value"]).to_csv(path, index=False)
            result = extract_logger_events(path, initial_low_baseline=True)
        np.testing.assert_array_equal(result.transition_frames, [9, 10, 11, 12, 13])
        np.testing.assert_array_equal(result.transition_states, [True, False, True, False, True])

    def test_do2_is_fallback_without_logger(self):
        expected = np.asarray([1.0, 2.0])
        result, metadata = resolve_stimulus_start_times(
            2, {"normalized_start_gratings": expected}
        )
        self.assertIs(result, expected)
        self.assertEqual(metadata["source"], "harp_do2_fallback")
        self.assertIn("no stimulus logger", metadata["logger_fallback_reason"])

    def test_end_frame_implies_low_on_next_frame_only_when_final_state_high(self):
        for final_high in (False, True):
            for has_end_frame in (False, True):
                with self.subTest(final_high=final_high, has_end_frame=has_end_frame):
                    rows = [(0, 0, "STARTSLAP"), (0, 0, "StimStart-one")]
                    rows += [(i, i / 60, f"Photodiode-{state}")
                             for i, state in enumerate([0, 1, 0, 1] + ([] if final_high else [0]))]
                    rows.append((8, 8 / 60, "END"))
                    if has_end_frame:
                        rows.append((8, 8 / 60, "EndFrame"))
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "logger.csv"
                        pd.DataFrame(rows, columns=["Frame", "Timestamp", "Value"]).to_csv(path, index=False)
                        original = path.read_bytes()
                        result = extract_logger_events(path, initial_low_baseline=True,
                                                       terminal_low_after_end_frame=True)
                        legacy = extract_logger_events(path, initial_low_baseline=True)
                        self.assertEqual(path.read_bytes(), original)
                    if has_end_frame and final_high:
                        np.testing.assert_array_equal(result.transition_frames, [1, 2, 3, 9])
                        np.testing.assert_array_equal(result.transition_states, [True, False, True, False])
                        self.assertEqual(result.terminal_low_frame, 9)
                    else:
                        np.testing.assert_array_equal(result.transition_frames, legacy.transition_frames)
                        self.assertIsNone(result.terminal_low_frame)
                    self.assertIsNone(legacy.terminal_low_frame)

    def test_terminal_rule_ignores_end_events(self):
        base = [(0, 0, "STARTSLAP"), (0, 0, "StimStart-one"),
                (1, 0, "Photodiode-1"), (2, 0, "Photodiode-0"),
                (3, 0, "Photodiode-1"), (4, 0, "EndFrame")]
        for markers in ([], [(4, 0, "END")], [(5, 0, "END")],
                        [(5, 0, "END"), (6, 0, "END")], [("invalid", 0, "END")]):
            with self.subTest(markers=markers), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "logger.csv"
                pd.DataFrame(base + markers, columns=["Frame", "Timestamp", "Value"]).to_csv(path, index=False)
                result = extract_logger_events(path, initial_low_baseline=True,
                                               terminal_low_after_end_frame=True)
                np.testing.assert_array_equal(result.transition_frames, [1, 2, 3, 5])
                np.testing.assert_array_equal(result.transition_states, [True, False, True, False])
                self.assertEqual(result.terminal_low_frame, 5)

    def test_terminal_rule_rejects_invalid_end_frame_and_late_events(self):
        base = [(0, 0, "STARTSLAP"), (0, 0, "StimStart-one"),
                (1, 0, "Photodiode-1"), (2, 0, "Photodiode-0"), (3, 0, "Photodiode-1")]
        for markers, message in (([(4, 0, "EndFrame"), (5, 0, "EndFrame")], "at most one"),
                                 ([(2, 0, "EndFrame")], "beyond EndFrame"),
                                 ([(4.5, 0, "EndFrame")], "nonnegative integer")):
            with self.subTest(markers=markers), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "logger.csv"
                pd.DataFrame(base + markers, columns=["Frame", "Timestamp", "Value"]).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    extract_logger_events(path, initial_low_baseline=True, terminal_low_after_end_frame=True)

    @patch("stimulus_sync.synchronize_stimulus_frames")
    def test_logger_photodiode_is_preferred(self, synchronize):
        expected = np.asarray([1.1, 2.1])
        qc = type("QC", (), {"__dict__": {"matched_transition_count": 100}})()
        synchronize.return_value = (expected, qc)

        result, metadata = resolve_stimulus_start_times(
            2,
            {"normalized_start_gratings": np.asarray([1.0, 2.0])},
            Path("logger.csv"),
        )

        self.assertIs(result, expected)
        self.assertEqual(metadata["source"], "logger_photodiode_aligned")

    @patch("stimulus_sync.synchronize_stimulus_frames")
    def test_complete_do2_falls_back_after_logger_failure(self, synchronize):
        synchronize.side_effect = ValueError("poor alignment")
        expected = np.asarray([1.0, 2.0])

        result, metadata = resolve_stimulus_start_times(
            2,
            {"normalized_start_gratings": expected},
            Path("logger.csv"),
        )

        self.assertIs(result, expected)
        self.assertEqual(metadata["source"], "harp_do2_fallback")
        self.assertEqual(metadata["logger_fallback_reason"], "poor alignment")

    @patch("stimulus_sync.synchronize_stimulus_frames")
    def test_partial_do2_cannot_rescue_logger_failure(self, synchronize):
        synchronize.side_effect = ValueError("poor alignment")
        with self.assertRaisesRegex(ValueError, "HARP DO2 has 1 events"):
            resolve_stimulus_start_times(
                2,
                {"normalized_start_gratings": np.asarray([1.0])},
                Path("logger.csv"),
            )


if __name__ == "__main__":
    unittest.main()