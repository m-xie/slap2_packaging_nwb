import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from stimulus_sync import (
    align_logger_frames_to_harp,
    extract_logger_events,
    resolve_stimulus_start_times,
    select_stimulus_logger,
)


class StimulusSyncTests(unittest.TestCase):
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

    def test_alignment_tolerates_missing_and_extra_transitions(self):
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

        anchor_frames, anchor_times, qc = align_logger_frames_to_harp(
            logger_data, harp_times, harp_states
        )

        self.assertGreater(qc.matched_transition_count, 160)
        self.assertAlmostEqual(qc.frame_rate_hz, 59.997, places=2)
        self.assertLess(qc.p95_absolute_residual_ms, 10)
        self.assertEqual(len(anchor_frames), len(anchor_times))

    def test_do2_is_fallback_without_logger(self):
        expected = np.asarray([1.0, 2.0])
        result, metadata = resolve_stimulus_start_times(
            2, {"normalized_start_gratings": expected}
        )
        self.assertIs(result, expected)
        self.assertEqual(metadata["source"], "harp_do2_fallback")
        self.assertIn("no stimulus logger", metadata["logger_fallback_reason"])

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