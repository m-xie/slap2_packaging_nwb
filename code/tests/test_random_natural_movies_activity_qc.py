"""Numerical alignment, missing-data policy, and real NWB activity-QC coverage."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import hdmf_zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb

from qc import random_natural_movies_activity_qc as activity


def movie(start, times=None, *, name="natural_movie_TOE_1", partial=False, statuses=None):
    times = np.asarray(times if times is not None else start + np.arange(6) / 30, dtype=float)
    return dict(TextureName=name, TrialType="movie", start_time=float(start),
                stop_time=float(start + len(times) / 30), is_partial=partial,
                movie_frame_numbers=np.arange(1, len(times) + 1),
                movie_frame_timestamps=times,
                movie_frame_timing_status=np.zeros(len(times), dtype=np.uint8) if statuses is None else np.asarray(statuses))


class ActivityAlignmentTests(unittest.TestCase):
    def test_analyze_hz_aliases_require_positive_scalar_and_agreement(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.h5"
            for spelling in ("analyze_hz", "analyzeHz"):
                with h5py.File(path, "w") as f:
                    f.create_dataset(f"params/{spelling}", data=[[200.]])
                self.assertEqual(activity.read_analyze_hz(path), 200.)
            for value in (0, -1, np.nan, np.inf, [20, 30], b"200"):
                with self.subTest(value=value):
                    with h5py.File(path, "w") as f:
                        f.create_dataset("params/analyzeHz", data=value)
                    with self.assertRaisesRegex(ValueError, "positive numeric scalar"):
                        activity.read_analyze_hz(path)
            with h5py.File(path, "w") as f:
                f.create_dataset("params/analyze_hz", data=100.)
                f.create_dataset("params/analyzeHz", data=200.)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                activity.read_analyze_hz(path)
            with h5py.File(path, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "requires params"):
                activity.read_analyze_hz(path)

    def test_interpolation_preserves_nan_gaps_and_does_not_clamp(self):
        query = np.array([[-1, 0, 0.5, 1, 1.5], [2, 2.5, 3, 8, np.nan]])
        actual = activity.interpolate_supported([0, 1, 2, 3, 8], [0, 1, np.nan, 3, 8], query, 2)
        np.testing.assert_allclose(actual, [[np.nan, 0, 0.5, 1, np.nan],
                                             [np.nan, np.nan, 3, 8, np.nan]])
        self.assertTrue(np.isnan(activity.interpolate_supported([0, 1, 8], [0, 1, 8], [4, 9], 2)).all())
        np.testing.assert_allclose(activity.interpolate_supported([2], [4], [1, 2, 3], 1), [np.nan, 4, np.nan])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            activity.interpolate_supported([0, 0], [1, 2], [0], 1)

    def test_movies_align_by_content_not_onset_elapsed_time(self):
        first = movie(1.)
        second = movie(4., 4 + np.arange(6) / 25)
        second["stop_time"] = 4 + 6 / 25
        blocks = pd.DataFrame([second, first, movie(8., name="natural_movie_TOE_1_shuffle")])
        groups = activity.build_conditions(blocks, pd.DataFrame(), 120)
        condition = groups[0][0]
        self.assertEqual(condition.targets.shape, (2, 24))
        np.testing.assert_allclose(np.diff(condition.axis), 1 / 120)
        np.testing.assert_allclose(condition.targets[:, ::4], [1 + np.arange(6) / 30,
                                                              4 + np.arange(6) / 25])
        self.assertEqual(groups[0][1].targets.shape[0], 1)  # shuffle stays separate
        # Activity ramps in content coordinates despite differing playback speed.
        timestamps = np.r_[np.linspace(1, 1.2, 121), np.linspace(4, 4.24, 121)]
        trace = np.r_[(timestamps[:121] - 1) * 30, (timestamps[121:] - 4) * 25]
        aligned = activity.align_trace(groups, timestamps, trace, 120)[0][0]
        np.testing.assert_allclose(aligned[0], aligned[1], atol=1e-10)
        np.testing.assert_allclose(aligned[0], condition.axis * 30, atol=1e-10)

    def test_shared_movie_timestamps_preserve_content_and_activity(self):
        times = np.array([1., 1., 1 + 2/30])
        groups = activity.build_conditions(pd.DataFrame([movie(1., times)]), pd.DataFrame(), 120)
        condition = groups[0][0]
        np.testing.assert_allclose(condition.targets[0, ::4], times)
        np.testing.assert_allclose(condition.targets[0, :5], 1.)
        self.assertAlmostEqual(condition.targets[0, 6], 1 + 1/30)
        self.assertTrue(np.isfinite(condition.targets).all())
        timestamps = np.linspace(1., 1.1, 121)
        aligned = activity.align_trace(groups, timestamps, timestamps * 2, 120)[0][0]
        np.testing.assert_allclose(aligned, condition.targets * 2)

    def test_shared_movie_timestamps_do_not_bridge_unsupported_or_partial_coverage(self):
        row = movie(1., [1., 1., 1., 1.], partial=True, statuses=[0, 1, 3, 2])
        targets = activity.build_conditions(pd.DataFrame([row]), pd.DataFrame(), 120)[0][0].targets[0]
        np.testing.assert_allclose(targets[:5], 1.)
        self.assertTrue(np.isnan(targets[5:12]).all())
        self.assertEqual(targets[12], 1.)
        self.assertTrue(np.isnan(targets[13:]).all())

    def test_decreasing_supported_movie_timestamps_still_rejected(self):
        for times, statuses in (([1., 1.05, 1.04], [0, 1, 2]),
                                ([1.05, np.nan, 1.04], [0, 3, 1])):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "must not decrease"):
                activity.build_conditions(pd.DataFrame([movie(1., times, statuses=statuses)]),
                                          pd.DataFrame(), 120)

    def test_partial_movie_nan_tail_and_unsupported_interior_preserved(self):
        complete = movie(1.)
        partial = movie(4., [4., np.nan, 4 + 2/30], partial=True, statuses=[0, 3, 1])
        groups = activity.build_conditions(pd.DataFrame([complete, partial]), pd.DataFrame(), 120)
        condition = groups[0][0]
        self.assertEqual(condition.targets.shape, (2, 24))
        self.assertEqual(condition.targets[1, 0], 4.)
        self.assertTrue(np.isnan(condition.targets[1, 1:8]).all())
        self.assertAlmostEqual(condition.targets[1, 8], 4 + 2/30)
        self.assertTrue(np.isnan(condition.targets[1, 9:]).all())
        self.assertTrue(np.isfinite(condition.targets[0]).all())
        np.testing.assert_array_equal(condition.partial, [False, True])

    def test_grating_onsets_real_durations_flanks_and_censoring(self):
        blocks = pd.DataFrame([movie(1.)])
        gratings = pd.DataFrame([
            dict(start_time=10., stop_time=12., logger_orientation=0, is_partial=False),
            dict(start_time=20., stop_time=21., logger_orientation=0, is_partial=False),
            dict(start_time=30., stop_time=30.75, logger_orientation=0, is_partial=True),
            dict(start_time=40., stop_time=42., logger_orientation=359, is_partial=False),
        ])
        condition = activity.build_conditions(blocks, gratings, 20)[2][0]
        np.testing.assert_allclose(condition.axis[[0, -1]], [-0.5, 2.5])
        zero = np.flatnonzero(condition.axis == 0)[0]
        np.testing.assert_allclose(condition.targets[:, zero], [10, 20, 30])
        np.testing.assert_allclose(condition.targets[0], 10 + condition.axis)
        self.assertTrue(np.isnan(condition.targets[1, condition.axis > 1.5]).all())
        self.assertTrue(np.isnan(condition.targets[2, condition.axis > .75]).all())
        np.testing.assert_allclose(condition.offsets, [2, 1, np.nan])
        self.assertEqual(activity.build_conditions(blocks, gratings, 20)[2][-1].title, "Blank")

    def test_population_ratio_of_sums_is_paired_and_chunked(self):
        baseline = np.array([[1, 3], [1, 3], [1, 3], [np.nan, 2],
                             [0, 0], [1, -1], [1e-6, 3e-6]])
        df = np.array([[1, 9], [np.nan, 12], [np.nan, np.nan], [5, 14],
                       [1, 2], [2, 3], [1e-6, 9e-6]])
        rois = SimpleNamespace(table=object(), data=np.arange(2))
        series = SimpleNamespace(name="green", timestamps=np.arange(7.),
                                 data=df / (baseline + 1e-6), rois=rois)
        f0 = SimpleNamespace(data=baseline, timestamps=series.timestamps.copy(), rois=rois)
        times, trace = activity.read_trace(series, chunk_size=2, f0_series=f0)
        np.testing.assert_allclose(times, np.arange(7))
        np.testing.assert_allclose(trace, [2.5, 4, np.nan, 7, np.nan, np.nan, 2.5])
        # Individual soma traces are unchanged and do not require F0.
        np.testing.assert_allclose(activity.read_trace(series, roi_index=1)[1], series.data[:, 1])
        with self.assertRaisesRegex(ValueError, "requires paired F0"):
            activity.read_trace(series)
        f0.timestamps = series.timestamps + 1
        with self.assertRaisesRegex(ValueError, "matching shape and timestamps"):
            activity.read_trace(series, f0_series=f0)
        f0.timestamps = series.timestamps
        f0.rois = SimpleNamespace(table=rois.table, data=np.array([1, 0]))
        with self.assertRaisesRegex(ValueError, "same ROIs in the same order"):
            activity.read_trace(series, f0_series=f0)

    def test_activity_gaps_are_missing_and_repeat_mean_has_variable_n(self):
        condition = activity.Condition("test", "test", "grating", np.array([0, 1, 2]),
                                       np.array([[0, .5, 1.], [4., 4.01, 8.]]),
                                       np.array([False, True]), np.array([1., np.nan]))
        timestamps = np.array([0, .01, .02, 1, 4, 4.01, 4.02])
        aligned = activity.align_trace([[condition]], timestamps, np.array([1, 2, 2.5, 3, 5, 7, 8]), 100)[0][0]
        np.testing.assert_allclose(aligned, [[1, np.nan, 3], [5, 7, np.nan]])
        mean, n = activity.repeat_statistics(aligned)
        np.testing.assert_allclose(mean, [3, 7, 3])
        np.testing.assert_array_equal(n, [2, 1, 1])
        mean, n = activity.repeat_statistics(np.full((2, 3), np.nan))
        self.assertTrue(np.isnan(mean).all())
        np.testing.assert_array_equal(n, 0)


class ActivityNWBTests(unittest.TestCase):
    def make_nwb(self):
        nwb = pynwb.NWBFile("activity", "test", datetime.now(timezone.utc))  # pyright: ignore[reportAbstractUsage]
        blocks = nwb.create_time_intervals(name="stimulus_blocks", description="movies")
        for col in ("TextureName", "TrialType", "is_partial"):
            blocks.add_column(col, col)
        for col in ("movie_frame_numbers", "movie_frame_timestamps", "movie_frame_timing_status"):
            blocks.add_column(col, col, index=True)
        shared = movie(1., [1., 1., 1 + 2/30, 1 + 3/30, 1 + 4/30, 1 + 5/30])
        for row in (shared, movie(2., partial=True), movie(3., name=activity.ZEBRA[0][0])):
            blocks.add_row(**row)
        gratings = nwb.create_time_intervals(name="gratings", description="gratings")
        for col in ("logger_orientation", "is_partial"):
            gratings.add_column(col, col)
        gratings.add_row(start_time=4., stop_time=5., logger_orientation=0, is_partial=False)
        gratings.add_row(start_time=6., stop_time=7., logger_orientation=359, is_partial=False)
        ophys = nwb.create_processing_module("ophys", "test")
        segmentation = pynwb.ophys.ImageSegmentation()
        ophys.add(segmentation)
        device = nwb.create_device("test")
        for dmd in ("DMD1", "DMD2"):
            channel = pynwb.ophys.OpticalChannel(name=dmd, description="test", emission_lambda=510.)
            plane = nwb.create_imaging_plane(dmd, optical_channel=channel, description="test", device=device,
                                             excitation_lambda=920., indicator="test", location="test")
            table = segmentation.create_plane_segmentation(name=dmd, description="sources", imaging_plane=plane)
            table.add_roi(pixel_mask=[(0, 0, 1.)])
            table.add_roi(pixel_mask=[(1, 0, 1.)])
            rois = table.create_roi_table_region(region=[0, 1], description="sources")
            times = np.arange(0, 8, .01)
            data = np.column_stack([times, times + 2])
            container = pynwb.ophys.Fluorescence(name=f"Fluorescence_{dmd}")
            ophys.add(container)
            for color in ("green", "red"):
                baseline = np.broadcast_to([1., 3.], data.shape).copy()
                container.add_roi_response_series(pynwb.ophys.RoiResponseSeries(
                    name=f"{dmd}_F0_{color}", data=baseline,
                    timestamps=times, rois=rois, unit="a.u."))
                container.add_roi_response_series(pynwb.ophys.RoiResponseSeries(
                    name=f"{dmd}_dFF_{color}",
                    data=data * baseline / (baseline + 1e-6) if color == "green" else data * 100,
                    timestamps=times, rois=rois, unit="dimensionless"))
            if dmd == "DMD2":
                continue
            soma_table = segmentation.create_plane_segmentation(name="soma", description="soma", imaging_plane=plane)
            soma_table.add_column("user_roi_index", "original index")
            soma_table.add_column("label", "original label")
            for idx in (2, 7):
                soma_table.add_roi(pixel_mask=[(idx, 0, 1.)], user_roi_index=idx, label=f"Soma/{idx}")
            soma_rois = soma_table.create_roi_table_region(region=[1, 0], description="reverse order")
            soma = pynwb.ophys.Fluorescence(name="SomaFluorescence_DMD1")
            ophys.add(soma)
            soma.add_roi_response_series(pynwb.ophys.RoiResponseSeries(
                name="DMD1_soma_dFF_red", data=data * 10, timestamps=times,
                rois=soma_rois, unit="dimensionless"))
        return nwb

    def test_hdf5_and_zarr_emit_per_dmd_and_per_soma_pngs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.h5"
            for suffix, io_class, analyze_hz in ((".nwb", pynwb.NWBHDF5IO, 200.),
                                                (".nwb.zarr", hdmf_zarr.NWBZarrIO, 60.)):
                with self.subTest(storage=suffix):
                    with h5py.File(summary, "w") as f:
                        f.create_dataset("params/analyzeHz", data=[[analyze_hz]])
                    path = root / ("activity" + suffix)
                    with io_class(str(path), "w") as io:
                        io.write(self.make_nwb())
                    before = plt.get_fignums()
                    with patch.object(activity, "save_activity_plot", wraps=activity.save_activity_plot) as save:
                        outputs = activity.compute_activity_qc(root / suffix, path, summary)
                    self.assertEqual({p.name for p in outputs}, {
                        "DMD1_sources_green_mean.png", "DMD2_sources_green_mean.png",
                        "DMD1_soma_red_roi_2.png", "DMD1_soma_red_roi_7.png",
                    })
                    self.assertEqual(plt.get_fignums(), before)
                    for output in outputs:
                        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                        self.assertEqual(output.parent.name, "activity")
                    self.assertEqual(save.call_count, 4)
                    # Ratio of green dF/F0 sums with unequal baselines, not mean dFF.
                    np.testing.assert_allclose(save.call_args_list[0].args[1][0][0][:, 0], [2.5, 3.5])
                    self.assertIn("sum(dF)/sum(F0)", save.call_args_list[0].args[2])
                    # Region order [1,0] makes data column 0 soma index 7.
                    self.assertIn("red soma 7", save.call_args_list[2].args[2])
                    np.testing.assert_allclose(save.call_args_list[2].args[1][0][0][:, 0], [10., 20.])
                    for call in save.call_args_list:
                        sampling_hz = min(100., analyze_hz)
                        self.assertEqual(call.args[3], sampling_hz)
                        for group in call.args[0]:
                            for condition in group:
                                if len(condition.axis) > 1:
                                    np.testing.assert_allclose(np.diff(condition.axis), 1 / sampling_hz)

    def test_plot_handles_all_missing_activity_and_no_conditions(self):
        groups = activity.build_conditions(pd.DataFrame([movie(1., partial=True)]), pd.DataFrame(), 30)
        aligned = [[np.full(c.targets.shape, np.nan) for c in group] for group in groups]
        with TemporaryDirectory() as directory, patch.object(activity.plt, "close"):
            path = activity.save_activity_plot(groups, aligned, "Missing", 30, Path(directory) / "missing.png")
            fig = plt.gcf()
            self.assertTrue(path.is_file())
            self.assertTrue(any("Repeat" in ax.get_ylabel() for ax in fig.axes))
            self.assertTrue(any(ax.get_ylabel() == "N" for ax in fig.axes))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()