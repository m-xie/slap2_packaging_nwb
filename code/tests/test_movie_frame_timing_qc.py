import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from qc.movie_frame_timing_qc import plot_movie_frame_timing


def make_blocks(rows, anchor_frames=(), anchor_times=()):
    blocks = pd.DataFrame(
        [
            {
                "movie_frame_timestamps": np.asarray(times, dtype=float),
                "movie_display_frames": np.asarray(frames, dtype=int),
                "movie_frame_timing_status": np.asarray(status, dtype=np.uint8),
            }
            for times, frames, status in rows
        ],
        columns=[
            "movie_frame_timestamps", "movie_display_frames", "movie_frame_timing_status"
        ],
    )
    blocks.attrs["movie_frame_alignment"] = {
        "anchor_frames": np.asarray(anchor_frames),
        "anchor_times": np.asarray(anchor_times),
        "metadata": {
            "maximum_anchor_gap_frames": 13,
            "maximum_interpolation_gap_frames": 10,
            "clock_reference": "normalized HARP",
            "mapping_method": "piecewise linear",
        },
    }
    return blocks


class MovieFrameTimingQcTests(unittest.TestCase):
    def render(self, blocks):
        """Capture the actual saved figure while writing only to a temporary folder."""
        figures = []
        savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            figures.append(figure)
            return savefig(figure, *args, **kwargs)

        before = plt.get_fignums()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "timing.png"
            with patch.object(Figure, "savefig", capture):
                result = plot_movie_frame_timing(blocks, str(output))
            self.assertIsInstance(result, Path)
            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(output.stat().st_size, 1000)
        self.assertEqual(plt.get_fignums(), before)
        self.assertEqual(len(figures[0].axes), 4)
        return figures[0]

    def test_photodiode_scatter_uses_anchor_differences_in_frames_and_ms(self):
        blocks = make_blocks(
            [([100, 105], [900, 950], [0, 1])],
            [10, 12, 17, 20], [1.0, 1.034, 1.120, 1.169],
        )
        axis = self.render(blocks).axes[3]
        self.assertEqual(len(axis.collections), 1)
        np.testing.assert_allclose(
            axis.collections[0].get_offsets(), [[2, 34], [5, 86], [3, 49]],
        )
        self.assertIn("(frames)", axis.get_xlabel())
        self.assertIn("(ms)", axis.get_ylabel())
        self.assertIn("matched transitions", axis.get_title())

    def test_photodiode_scatter_does_not_bridge_invalid_anchors(self):
        blocks = make_blocks(
            [], [0, 2, 4, np.nan, 9, 12, 12, 15, 14, 17],
            [0, 0.03, np.nan, 0.10, 0.15, 0.20, 0.21, 0.19, 0.24, 0.29],
        )
        axis = self.render(blocks).axes[3]
        np.testing.assert_allclose(
            axis.collections[0].get_offsets(), [[2, 30], [3, 50], [3, 50]],
        )

    def test_photodiode_scatter_empty_and_singleton(self):
        for frames, times in (([], []), ([1], [0.1]), ([1, 1], [0.1, 0.2])):
            with self.subTest(frames=frames):
                axis = self.render(make_blocks([], frames, times)).axes[3]
                self.assertEqual(len(axis.collections), 0)
                self.assertIn("No valid consecutive photodiode matches",
                              [text.get_text() for text in axis.texts])

    def test_irregular_intervals_do_not_bridge_nan_gaps_or_blocks(self):
        blocks = make_blocks(
            [
                ([1, 1.02, np.nan, 1.1, 1.14, 1.21], [100, 101, 102, 103, 105, 108],
                 [0, 1, 3, 1, 1, 2]),
                ([], [], []),
                ([20, 20.03], [1000, 1002], [0, 2]),
            ],
            [100, 105, 118], [1, 1.14, 1.4],
        )
        figure = self.render(blocks)
        lines = figure.axes[0].lines
        self.assertEqual(len(lines), 2)
        np.testing.assert_allclose(lines[0].get_xdata(), [1.02, 1.14, 1.21])
        np.testing.assert_allclose(lines[0].get_ydata(), [0.02, 0.04, 0.07])
        np.testing.assert_allclose(lines[1].get_xdata(), [20.03])
        np.testing.assert_allclose(lines[1].get_ydata(), [0.03])
        coverage = {line.get_label(): line for line in figure.axes[2].lines}
        np.testing.assert_array_equal(
            coverage["Unsupported movie frames"].get_xdata(), [102]
        )
        np.testing.assert_array_equal(
            coverage["Extrapolated movie frames"].get_xdata(), [108, 1002]
        )
        np.testing.assert_array_equal(coverage["Anchor spacing"].get_xdata(), [105, 118])
        np.testing.assert_array_equal(coverage["Anchor spacing"].get_ydata(), [5, 13])
        for value in ("normalized HARP", "piecewise linear", "13", "10"):
            self.assertIn(value, figure._suptitle.get_text())
        # Input arrays and attrs remain untouched.
        self.assertTrue(np.isnan(blocks.iloc[0].movie_frame_timestamps[2]))
        self.assertEqual(blocks.attrs["movie_frame_alignment"]["metadata"]["maximum_anchor_gap_frames"], 13)

    def test_affine_residual_values_and_honest_label(self):
        # Symmetric curvature has zero mean and zero projection onto frame slope.
        frames = np.array([0, 10, 20, 30]) + 1_000_000_000
        expected = np.array([0.01, -0.01, -0.01, 0.01])
        times = 50 + np.arange(4) * 0.2 + expected
        figure = self.render(make_blocks([([], [], [])], frames, times))
        axis = figure.axes[1]
        residual = next(line for line in axis.lines if line.get_label() == "Affine residual")
        np.testing.assert_allclose(residual.get_xdata(), times)
        np.testing.assert_allclose(residual.get_ydata(), expected, atol=1e-12)
        self.assertIn("Affine residual", axis.get_ylabel())
        self.assertIn("not a measure of timing accuracy", axis.get_title())

    def test_grating_only_and_small_anchor_arrays(self):
        for frames, times in (([], []), ([5], [1.0]), ([5, 5], [1.0, 1.1]),
                              ([5, 8], [1.0, 1.1])):
            with self.subTest(frames=frames):
                figure = self.render(make_blocks([([], [], []), ([], [], [])], frames, times))
                self.assertEqual(len(figure.axes[0].lines), 0)
                self.assertIn("No movie frames", [text.get_text() for text in figure.axes[2].texts])
                if len(set(frames)) < 2:
                    self.assertEqual(len(figure.axes[1].lines), 0)

    def test_zero_rows_without_alignment_metadata(self):
        blocks = make_blocks([])
        blocks.attrs.clear()
        self.render(blocks)

    def test_unsupported_and_singleton_have_no_intervals(self):
        figure = self.render(make_blocks([
            ([np.nan, np.nan], [10, 11], [3, 3]),
            ([2.0], [20], [0]),
            ([3.0, 3.1, 3.2], [30, 31, 32], [1, 3, 1]),
        ]))
        self.assertEqual(len(figure.axes[0].lines), 0)

    def test_figure_closed_if_save_fails(self):
        before = plt.get_fignums()
        with TemporaryDirectory() as directory:
            with patch.object(Figure, "savefig", side_effect=OSError("Cannot save")):
                with self.assertRaisesRegex(OSError, "Cannot save"):
                    plot_movie_frame_timing(make_blocks([]), Path(directory) / "timing.png")
        self.assertEqual(plt.get_fignums(), before)


if __name__ == "__main__":
    unittest.main()