"""Test the actual plotted edges, axis mapping, and observed DI3 intervals."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qc import random_natural_movies_sync_qc as sync_qc
from stimulus_sync import LoggerEvents, align_logger_frames_to_harp


class RandomNaturalMoviesSyncQCTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(plt.close, "all")

    def photodiode_figure(self, frames=None, times=None):
        if frames is None:
            frames = [600, 618, 690, 900, 1100, 1160, 1190]
        if times is None:
            times = [10, 10.2, 12, 15, 18, 19.7, 20]
        frames = np.asarray(frames)
        output = self.root / "syncing" / "photodiode_sync.png"
        with patch.object(sync_qc.plt, "close"):
            sync_qc.plot_photodiode_sync(frames, times, output)
            fig = plt.gcf()
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        return fig

    def test_analog_windows_select_equal_count_endpoint_logger_edges(self):
        fig = self.photodiode_figure()
        first, last, first_time, last_time = fig.axes
        np.testing.assert_allclose(first.get_xlim(), [600, 720])
        np.testing.assert_allclose(last.get_xlim(), [1070, 1190])
        np.testing.assert_allclose(first_time.get_xlim(), [10, 12])
        np.testing.assert_allclose(last_time.get_xlim(), [18, 20])
        def edge_positions(ax):
            return [segment[0, 0] for segment in ax.collections[0].get_segments()]
        np.testing.assert_array_equal(edge_positions(first), [600, 618, 690])
        np.testing.assert_array_equal(edge_positions(last), [1100, 1160, 1190])
        np.testing.assert_allclose(edge_positions(first_time), [10, 10.2, 12])
        np.testing.assert_allclose(edge_positions(last_time), [18, 19.7, 20])
        self.assertIn("First anchors", first.get_title())
        self.assertIn("Last anchors", last.get_title())

    def test_totals_count_all_edges_not_only_visible_edges(self):
        fig = self.photodiode_figure()
        text = "\n".join(item.get_text() for item in fig.texts)
        self.assertIn("Matched anchor pairs used for alignment: 7", text)
        self.assertNotIn("Detected", text)
        # Each window only displays three edges; the title still reports all
        # seven. Axis scaling is tested numerically, not via optional prose.
        for ax in fig.axes:
            self.assertEqual(len(ax.collections[0].get_segments()), 3)

    def test_long_logger_span_is_not_stretched_or_clipped(self):
        fig = self.photodiode_figure(frames=[600, 660, 780], times=[10, 10.5, 11])
        first, last, first_time, last_time = fig.axes
        np.testing.assert_allclose(first_time.get_xlim(), [10, 13])
        np.testing.assert_allclose(last_time.get_xlim(), [8, 11])
        for frame_ax, time_ax in ((first, first_time), (last, last_time)):
            # The span ratio must always be exactly 60 frames/second.
            self.assertAlmostEqual(np.diff(frame_ax.get_xlim())[0] /
                                   np.diff(time_ax.get_xlim())[0], 60)
            np.testing.assert_allclose(frame_ax.get_xlim(), [600, 780])

    def test_short_recording_has_overlapping_windows(self):
        fig = self.photodiode_figure(frames=[600, 612], times=[10, 10.2])
        for ax in fig.axes:
            self.assertEqual(len(ax.collections[0].get_segments()), 2)
        np.testing.assert_allclose(fig.axes[2].get_xlim(), [10, 12])
        np.testing.assert_allclose(fig.axes[3].get_xlim(), [8.2, 10.2])

    def test_invalid_anchor_pairs_are_rejected(self):
        for frames, times in (
            ([600], [10, 10.2]), ([], []),
            ([600, 600], [10, 11]), ([600, 601], [11, 10]),
            ([600, np.nan], [10, 11]), ([600, 601], [10, np.inf]),
            ([[600, 601]], [[10, 11]]),
        ):
            with self.subTest(frames=frames, times=times):
                with self.assertRaisesRegex(ValueError, "paired finite strictly increasing anchors"):
                    self.photodiode_figure(frames=frames, times=times)

    def test_last_panel_excludes_unmatched_terminal_blip(self):
        logger = LoggerEvents(self.root / "unused.csv", 0, np.array([]),
                              np.array([10, 20, 30, 40, 55, 56]),
                              np.array([True, False, True, False, True, False]),
                              terminal_low_frame=56)
        with patch("builtins.print"):
            frames, times, _ = align_logger_frames_to_harp(
                logger, np.array([1., 2., 3., 4.]), np.array([True, False, True, False]))
        fig = self.photodiode_figure(frames, times)
        last, last_time = fig.axes[1], fig.axes[3]
        np.testing.assert_array_equal(
            [segment[0, 0] for segment in last.collections[0].get_segments()], [20, 30, 40])
        np.testing.assert_array_equal(
            [segment[0, 0] for segment in last_time.collections[0].get_segments()], [2, 3, 4])
        self.assertIn("Matched anchor pairs used for alignment: 4", fig._suptitle.get_text())

    def test_di3_counts_raw_cycles_and_only_detected_pulse_intervals(self):
        for pulses, controls in (
            ([-0.5, 0.5, 2.5], [-0.5, 0.5, 2.5, 4.0]),  # estimated end excluded
            ([-0.5, 0.5, 2.5, 3.5], [-0.5, 0.5, 2.5, 3.5]),  # measured end retained
        ):
            with self.subTest(pulses=pulses), patch.object(sync_qc.plt, "close"):
                stream = StringIO()
                output = self.root / "syncing" / "slap2_di3_sync.png"
                with redirect_stdout(stream):
                    sync_qc.plot_di3_sync(
                        {"cycle_starts": pulses, "control_timestamps": controls},
                        {"DMD1": 3, "DMD2": 2}, output,
                    )
                fig = plt.gcf()
                text = "\n".join(item.get_text() for item in fig.texts)
                self.assertIn("Total SLAP2 cycles (DMD1, DI3 primary): 3", text)
                self.assertIn("Total SLAP2 cycles (DMD2): 2", text)
                self.assertIn(f"Total detected DI3 pulses: {len(pulses)}", text)
                self.assertIn(f"Total detected DI3 pulses: {len(pulses)}", stream.getvalue())
                self.assertEqual(sum(bar.get_height() for bar in fig.axes[0].patches), len(pulses) - 1)
                self.assertIn("(ms)", fig.axes[0].get_xlabel())
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_single_di3_pulse_has_no_invented_interval_and_closes_figure(self):
        before = plt.get_fignums()
        with redirect_stdout(StringIO()):
            sync_qc.plot_di3_sync({"cycle_starts": [1.0]}, {"DMD1": 1}, self.root / "single.png")
        self.assertEqual(plt.get_fignums(), before)


if __name__ == "__main__":
    unittest.main()