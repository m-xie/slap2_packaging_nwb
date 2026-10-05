"""Integration coverage for forced Random Natural Movies SLAP2 packaging.

Only raw-header I/O and unrelated image/ROI construction are mocked. The HDF5
summary, acquisition filtering, sequential clock, line mapping, fluorescence
containers, and QC JSON are real. Empty raw/.meta placeholders deliberately
cannot supply an effective lines-per-cycle value or any timestamps.

The integration uses the installed PyNWB API without compatibility shims.
"""

import json
import unittest
import warnings
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pynwb

import run_capsule as packaging
from tests.test_slap2_synching import _sample_cycle_clock


class ContinuousSlap2PackagingTests(unittest.TestCase):
    PREFIX = "acquisition_20260917_130000"
    SOURCE = "slap2_utils.utils.file_header.load_file_header_v2"

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.session = self.root / "session"
        self.processed = self.root / "processed"
        self.qc = self.root / "qc"
        self.session.mkdir()
        self.processed.mkdir()
        self.summary_path = self.processed / "experiment_summary.h5"
        self.summary = self.stack.enter_context(
            h5py.File(self.summary_path, "w", track_order=True)
        )
        self.lines = {
            "Path1": np.asarray([3, 5, 8, 12]),
            "Path2": np.asarray([3, 6, 9, 13]),
        }
        self.counts = {"Path1": [2, 2], "Path2": [1, 2, 1]}
        self.traces = {}
        self.dat_paths = {}
        self.meta_paths = {}
        # Intentionally reverse numeric DMD order, including the HDF5 iterator.
        for number in (2, 1):
            plane = f"Path{number}"
            group = self.summary.create_group(plane)
            info = group.create_group("frame_info")
            info.create_dataset("frame_line_idxs", data=[self.lines[plane]])
            info.create_dataset("trial_num_frames", data=[self.counts[plane]])
            temporal = group.create_group("sources/temporal")
            self.traces[plane] = {}
            for index, name in enumerate(("F0", "dF_denoised", "events")):
                values = (np.arange(4, dtype=float) + number * 10 + index).reshape(4, 1, 1)
                temporal.create_dataset(name, data=values)
                self.traces[plane][name] = values
            dat = self.session / f"{self.PREFIX}_DMD{number}-TRIAL000001.dat"
            meta = self.session / f"{self.PREFIX}_DMD{number}.meta"
            dat.touch()
            meta.touch()
            self.dat_paths[plane] = dat
            self.meta_paths[plane] = meta
        self.summary.flush()
        self.path_metadata = {
            "Path1": dict(lines_per_cycle=4, total_cycles=3, total_lines=12,
                          chunk_count=1, source=self.SOURCE),
            "Path2": dict(lines_per_cycle=7, total_cycles=2, total_lines=14,
                          chunk_count=1, source=self.SOURCE),
        }
        self.harp = self.make_harp()
        self.nwb = pynwb.NWBFile(  # type: ignore[abstract]  # HDMF generates concrete members at runtime.
            session_description="synthetic continuous recording",
            identifier="continuous-slap2-test",
            session_start_time=datetime(2026, 9, 17, tzinfo=timezone.utc),
        )
        self.instrument = {"instrument_id": "test-slap2", "notes": "synthetic"}
        self.read_metadata = self.stack.enter_context(patch.object(
            packaging, "read_path_metadata", side_effect=self.mock_path_metadata
        ))
        self.build_clock = self.stack.enter_context(patch.object(
            packaging, "build_continuous_clock", wraps=packaging.build_continuous_clock
        ))
        self.map_lines = self.stack.enter_context(patch.object(
            packaging, "map_continuous_lines", wraps=packaging.map_continuous_lines
        ))
        self.synced = {}
        real_sync = packaging.sync_slap2_fluorescence

        def capture_sync(*args, **kwargs):
            result = real_sync(*args, **kwargs)
            self.synced[args[0]] = result
            return result

        self.sync = self.stack.enter_context(patch.object(
            packaging, "sync_slap2_fluorescence", side_effect=capture_sync
        ))
        self.imaging = self.stack.enter_context(patch.object(
            packaging, "create_imaging_plane", side_effect=self.mock_imaging_plane
        ))
        self.images = self.stack.enter_context(patch.object(packaging, "add_mean_images"))
        self.rois = self.stack.enter_context(patch.object(
            packaging, "add_image_segmentation", side_effect=self.mock_segmentation
        ))
        self.fluorescence = self.stack.enter_context(patch.object(
            packaging, "add_fluorescence", wraps=packaging.add_fluorescence
        ))
        self.forbidden = []
        for owner, names in (
            (packaging, ("infer_continuous_slap2_mode", "resolve_slap2_acquisition")),
            (packaging.slap2_sync, (
                "normalize_continued_trial_line_indices",
                "get_slap2_primary_plane_timestamps",
                "get_slap2_secondary_plane_timestamps", "get_trial_num_cycles",
                "read_dat_num_cycles", "assign_cycles_by_line_count",
                "reconcile_cycle_clock_trials", "plot_slap2_sync_qc",
            )),
        ):
            for name in names:
                mock = self.stack.enter_context(patch.object(
                    owner, name, side_effect=AssertionError(f"Legacy call: {name}")
                ))
                self.forbidden.append(mock)

    def tearDown(self):
        for mock in self.forbidden:
            mock.assert_not_called()

    @staticmethod
    def make_harp(pulses=(-0.5, 0.5, 2.5, 3.5), starts=(0.0,), ends=(4.0,)):
        signal, times = _sample_cycle_clock(pulses)
        return {
            "normalized_slap2_start": np.asarray(starts),
            "normalized_slap2_end": np.asarray(ends),
            "slap2_cycle_clock_signal": signal,
            "slap2_cycle_clock_times": times,
            "recording_start_time": -2.0,
            "analog_times": np.asarray([-2.0, 5.0]),
        }

    def mock_path_metadata(self, paths):
        self.assertEqual(len(paths), 1)
        parsed = packaging.parse_dat_file(paths[0])
        self.assertIsNone(parsed.cycle_offset, "Fixture must be an unchunked raw trial")
        return self.path_metadata[f"Path{parsed.dmd_number}"]

    @staticmethod
    def mock_imaging_plane(nwb, name, device, acquisition_json, plane_key=None):
        channel = pynwb.ophys.OpticalChannel(
            name=f"{name}_green", description="synthetic", emission_lambda=510.0
        )
        return nwb.create_imaging_plane(
            name=name, optical_channel=channel, description="synthetic",
            device=device, excitation_lambda=920.0, indicator="GCaMP", location="V1",
        )

    @staticmethod
    def mock_segmentation(summary, imaging_plane, name, segmentation, plane_key=None):
        table = segmentation.create_plane_segmentation(
            name=f"PlaneSegmentation_{name}", description="synthetic",
            imaging_plane=imaging_plane,
        )
        table.add_roi(pixel_mask=[(0, 0, 1.0)])
        return table.create_roi_table_region(region=[0], description="one synthetic ROI")

    def plane_inputs(self):
        return [
            (plane, f"DMD{plane[-1]}", plane[-1], [self.meta_paths[plane]],
             [self.dat_paths[plane]], len(self.counts[plane]))
            for plane in self.summary
        ]

    def package(self, **kwargs):
        with redirect_stdout(StringIO()):
            packaging.add_ophys_to_nwb(
                self.summary, self.nwb, self.instrument, {}, self.harp,
                self.session, qc_folder=self.qc,
                logger_format="Random Natural Movies", **kwargs,
            )

    def assert_packaged(self, name, expected):
        fluorescence, timestamps, maps, qc = self.synced[name]
        plane = f"Path{name[-1]}"
        np.testing.assert_allclose(timestamps, expected, rtol=0, atol=1e-14)
        self.assertIsNone(maps)
        for key, values in self.traces[plane].items():
            np.testing.assert_array_equal(fluorescence[key], values)
        np.testing.assert_array_equal(qc["raw_frame_line_idxs"], self.lines[plane])
        np.testing.assert_array_equal(qc["frame_line_idxs"], self.lines[plane])
        np.testing.assert_array_equal(qc["trial_num_frames"], [4])
        np.testing.assert_array_equal(qc["processing_chunk_num_frames"], self.counts[plane])
        self.assertEqual(qc["path_metadata"], self.path_metadata[plane])
        self.assertEqual(qc["lines_per_cycle"], self.path_metadata[plane]["lines_per_cycle"])
        self.assertEqual(qc["acquisition_prefix"], self.PREFIX)
        self.assertEqual(qc["excluded_trial_count"], 0)
        calls = [call for call in self.fluorescence.call_args_list if call.args[2] == name]
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0].args[1], timestamps)
        container = self.nwb.processing["ophys"][f"Fluorescence_{name}"]
        self.assertEqual(set(container.roi_response_series),
                         {f"{name}_F0_green", f"{name}_dFF_green"})
        valid = np.isfinite(expected)
        for series in container.roi_response_series.values():
            np.testing.assert_allclose(series.timestamps, np.asarray(expected)[valid], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(
            container.roi_response_series[f"{name}_F0_green"].data,
            self.traces[plane]["F0"][valid, 0, :],
        )
        np.testing.assert_allclose(
            container.roi_response_series[f"{name}_dFF_green"].data,
            self.traces[plane]["dF_denoised"][valid, 0, :]
            / (self.traces[plane]["F0"][valid, 0, :] + 1e-6),
        )

    def assert_qc_json(self, name):
        path = self.qc / "syncing" / f"{name}_continuous_sync.json"

        def reject_constant(value):
            self.fail(f"Non-finite JSON constant: {value}")

        report = json.loads(path.read_text(), parse_constant=reject_constant)
        # Also reject overflowed JSON numbers (e.g. 1e999), not just NaN tokens.
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["dmd"], name)
        self.assertEqual(report["sample_count"], 4)
        plane = f"Path{name[-1]}"
        self.assertEqual(report["path_metadata"], self.path_metadata[plane])
        self.assertEqual(report["processing_chunk_num_frames"], self.counts[plane])
        clock = report["clock"]
        self.assertEqual(clock["primary_dmd"], "DMD1")
        self.assertEqual(clock["path_metadata_source"], self.SOURCE)
        self.assertEqual(clock["segmentation_method"], "sequential_di3")
        self.assertEqual(clock["lines_per_cycle"], 4)
        self.assertEqual(clock["total_cycles"], 3)
        self.assertEqual(clock["raw_line_count"], 12)
        self.assertIn("normalized HARP", clock["clock_reference"])
        for key in ("cycle_starts", "control_line_idxs", "control_timestamps"):
            self.assertNotIn(key, report)
            self.assertNotIn(key, clock)
        return report

    def test_prepare_uses_dmd1_and_library_dimensions_in_reverse_order(self):
        inputs = self.plane_inputs()
        self.assertEqual([entry[1] for entry in inputs], ["DMD2", "DMD1"])
        resolution, clock, metadata = packaging.prepare_continuous_slap2(inputs, self.harp)
        self.assertEqual(resolution, {
            "acquisition_prefix": self.PREFIX, "excluded_trial_count": 0,
            "excluded_trailing_trials": 0, "retained_trial_count": 1,
            "highest_dat_trial": 1,
        })
        self.assertEqual(metadata, self.path_metadata)
        self.assertEqual(self.read_metadata.call_count, 2)
        self.assertEqual(self.build_clock.call_count, 1)
        self.assertEqual(self.build_clock.call_args.args[2:], (4, 3))
        self.assertEqual(self.build_clock.call_args.kwargs["recording_start"], -2.0)
        np.testing.assert_array_equal(clock["control_line_idxs"], [1, 5, 9, 13])
        np.testing.assert_array_equal(clock["cycle_starts"], [-0.5, 0.5, 2.5, 3.5])
        self.assertEqual(clock["qc"]["primary_dmd"], "DMD1")

    def test_forced_mode_different_summary_chunks_preserves_fractional_mapping(self):
        self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])
        self.assertEqual(self.build_clock.call_count, 1)
        calls = self.sync.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["DMD1", "DMD2"])
        self.assertIs(calls[0].kwargs["continuous_clock"], calls[1].kwargs["continuous_clock"])
        for call in calls:
            self.assertTrue(call.kwargs["continuous_mode"])
            self.assertEqual(call.kwargs["acquisition_resolution"]["retained_trial_count"], 1)
        self.assertEqual([call.kwargs["recorded_line_count"]
                          for call in self.map_lines.call_args_list], [12, 14])
        for name in ("DMD1", "DMD2"):
            report = self.assert_qc_json(name)
            self.assertEqual(report["unsupported_sample_count"], 0)
            self.assertEqual(report["clock"]["extra_final_pulse_count"], 1)
            self.assertEqual(report["clock"]["last_cycle_policy"], "measured_end")

    def test_forced_mode_accepts_single_unchunked_summary(self):
        for plane in self.summary:
            info = self.summary[plane]["frame_info"]
            del info["trial_num_frames"]
            info.create_dataset("trial_num_frames", data=[[4]])
            self.counts[plane] = [4]
        self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])

    def test_soma_uses_continuous_clock_and_excludes_unsupported_samples(self):
        group = self.summary["Path1"].create_group("user_rois")
        group.create_dataset("labels", data=[[b"soma"]])
        group.create_dataset("mask", data=np.ones((2, 3, 1, 1), dtype=np.uint8))
        values = np.arange(8, dtype=float).reshape(4, 2, 1)
        for key, offset in (("F", 0), ("Fsvd", 100)):
            group.create_dataset(key, data=values + offset)
        self.path_metadata["Path1"]["total_lines"] = 8
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.package()
        ophys = self.nwb.processing["ophys"]
        container = ophys["SomaFluorescence_DMD1"]
        for key, offset in (("F", 0), ("Fsvd", 100)):
            for channel, color in enumerate(("green", "red")):
                series = container.roi_response_series[f"DMD1_soma_{key}_{color}"]
                np.testing.assert_array_equal(series.data, (values + offset)[:3, channel, :])
                np.testing.assert_allclose(series.timestamps, [0.0, 0.5, 2.0])
                self.assertEqual(series.rois.table.name, "SomaPlaneSegmentation_DMD1")
        self.assertNotIn("SomaFluorescence_DMD2", ophys.data_interfaces)
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, np.nan])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])

    def test_dat_names_without_trial_are_discovered_as_trial_one(self):
        for path in self.dat_paths.values():
            path.rename(path.with_name(path.name.replace("-TRIAL000001", "")))
        # Do not accidentally collect another numbered DMD via a prefix match.
        (self.session / f"{self.PREFIX}_DMD10.dat").touch()
        self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])

    def test_missing_dmd1_errors_before_reading_metadata(self):
        inputs = [entry for entry in self.plane_inputs() if entry[1] != "DMD1"]
        with self.assertRaisesRegex(ValueError, "DMD1 is required"):
            packaging.prepare_continuous_slap2(inputs, self.harp)
        del self.summary["Path1"]
        with self.assertRaisesRegex(ValueError, "DMD1 is required"):
            self.package()
        self.read_metadata.assert_not_called()
        self.sync.assert_not_called()

    def test_missing_dmd1_raw_data_errors(self):
        self.dat_paths["Path1"].unlink()
        with self.assertRaisesRegex(ValueError, "DMD1 is required"):
            self.package()
        self.read_metadata.assert_not_called()

    def test_initial_high_is_not_used_as_first_cycle(self):
        self.harp["slap2_cycle_clock_signal"] = np.insert(self.harp["slap2_cycle_clock_signal"], 0, True)
        self.harp["slap2_cycle_clock_times"] = np.insert(self.harp["slap2_cycle_clock_times"], 0, -1.5)
        with self.assertWarnsRegex(RuntimeWarning, "DI3 initially starts high"):
            self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])
        self.assertEqual(self.assert_qc_json("DMD1")["clock"]["detected_pulse_count"], 4)

    def test_sourceless_primary_still_controls_secondary_clock_and_raw_limit(self):
        del self.summary["Path1"]["sources"]
        with self.assertWarnsRegex(UserWarning, "DMD1 is missing sources"):
            self.package()
        self.assertEqual(list(self.synced), ["DMD2"])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])
        self.assertEqual(self.read_metadata.call_count, 2)
        self.assertEqual(self.build_clock.call_args.args[2:], (4, 3))
        self.assertEqual(self.map_lines.call_args.kwargs["recorded_line_count"], 14)
        self.assertEqual(self.imaging.call_count, 2)
        self.assertEqual(self.images.call_count, 2)
        self.assertEqual(self.rois.call_count, 1)
        self.assertEqual(self.assert_qc_json("DMD2")["clock"]["primary_dmd"], "DMD1")
        self.assertFalse((self.qc / "syncing" / "DMD1_continuous_sync.json").exists())

    def test_single_start_without_end_is_accepted(self):
        self.harp["normalized_slap2_end"] = np.asarray([])
        self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])

    def test_estimated_last_cycle_uses_mean_period_and_secondary_tail_is_nan(self):
        self.harp = self.make_harp(pulses=(-0.5, 0.5, 2.5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.package()
        self.assertTrue(any("Estimating only the end" in str(item.message) for item in caught))
        self.assertTrue(any("DMD2: 1 fluorescence samples" in str(item.message) for item in caught))
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.625])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, np.nan])
        for name, unsupported in (("DMD1", 0), ("DMD2", 1)):
            report = self.assert_qc_json(name)
            self.assertEqual(report["unsupported_sample_count"], unsupported)
            self.assertEqual(report["stored_sample_count"], 4 - unsupported)
            self.assertEqual(report["clock"]["estimated_mean_period_seconds"], 1.5)
            self.assertEqual(report["clock"]["extra_final_pulse_count"], 0)

    def test_more_than_one_extra_di3_pulse_fails_before_mapping(self):
        self.harp = self.make_harp(pulses=(-0.5, 0.5, 2.5, 3.5, 4.5))
        with self.assertRaisesRegex(ValueError, "5 DI3 pulses for 3 raw cycles"):
            self.package()
        self.map_lines.assert_not_called()
        self.fluorescence.assert_not_called()

    def test_reset_summary_line_indices_fail_without_normalization(self):
        reset = np.asarray([3, 6, 1, 4])
        self.summary["Path2/frame_info/frame_line_idxs"][0] = reset
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.package()
        np.testing.assert_array_equal(self.summary["Path2/frame_info/frame_line_idxs"][0], reset)
        self.assertNotIn("Fluorescence_DMD2", self.nwb.processing["ophys"].data_interfaces)

    def test_secondary_line_beyond_own_raw_limit_warns_and_is_removed_from_nwb(self):
        # Line 13 has a measured timestamp on DMD1's clock, but lies outside
        # this DMD's raw data. It must be NaN before NWB filtering.
        self.path_metadata["Path2"].update(lines_per_cycle=6, total_lines=12)
        with self.assertWarnsRegex(RuntimeWarning, "exceed recorded_line_count"):
            self.package()
        self.assertEqual(self.map_lines.call_args.kwargs["recorded_line_count"], 12)
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, np.nan])
        self.assertEqual(self.assert_qc_json("DMD2")["stored_sample_count"], 3)
        self.assert_filtered_roundtrip("DMD2", [0.0, 1.0, 2.5], np.arange(3))

    def test_primary_out_of_range_samples_are_removed_without_changing_secondary_clock(self):
        self.lines["Path1"] = np.array([3, 5, 13, 14])
        self.summary["Path1/frame_info/frame_line_idxs"][0] = self.lines["Path1"]
        with self.assertWarnsRegex(RuntimeWarning, "exceed recorded_line_count"):
            self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, np.nan, np.nan])
        self.assert_packaged("DMD2", [0.0, 1.0, 2.5, 3.5])
        self.assert_filtered_roundtrip("DMD1", [0.0, 0.5], np.arange(2))

    def test_all_samples_outside_raw_limit_save_empty_fluorescence(self):
        self.lines["Path2"] = np.array([15, 16, 17, 18])
        self.summary["Path2/frame_info/frame_line_idxs"][0] = self.lines["Path2"]
        with self.assertWarnsRegex(RuntimeWarning, "exceed recorded_line_count"):
            self.package()
        self.assert_packaged("DMD2", [np.nan] * 4)
        self.assert_filtered_roundtrip("DMD2", [], np.array([], dtype=int))

    def assert_filtered_roundtrip(self, dmd, expected_times, indices):
        for io_class, filename in ((pynwb.NWBHDF5IO, "filtered.nwb"),
                                   (packaging.hdmf_zarr.NWBZarrIO, "filtered.nwb.zarr")):
            with self.subTest(format=filename):
                path = self.root / filename
                with io_class(str(path), "w") as io:
                    # HDMF binds a written container to its destination;
                    # each backend needs a fresh in-memory graph.
                    io.write(deepcopy(self.nwb))
                with io_class(str(path), "r") as io:
                    series = io.read().processing["ophys"][f"Fluorescence_{dmd}"].roi_response_series
                    plane = f"Path{dmd[-1]}"
                    for name in ("F0", "dFF"):
                        np.testing.assert_allclose(series[f"{dmd}_{name}_green"].timestamps[:], expected_times)
                        self.assertEqual(len(series[f"{dmd}_{name}_green"].data), len(expected_times))
                    np.testing.assert_array_equal(series[f"{dmd}_F0_green"].data[:], self.traces[plane]["F0"][indices, 0, :])

    def test_inconsistent_acquisition_including_sourceless_primary_fails(self):
        del self.summary["Path1"]["sources"]
        original = self.dat_paths["Path1"]
        original.rename(original.with_name(original.name.replace("130000", "120000")))
        with self.assertWarnsRegex(UserWarning, "DMD1 is missing sources"):
            with self.assertRaisesRegex(ValueError, "exactly one raw SLAP2 acquisition"):
                self.package()
        self.read_metadata.assert_not_called()
        self.build_clock.assert_not_called()

    def test_raw_trial_two_fails_instead_of_becoming_a_processing_chunk(self):
        original = self.dat_paths["Path2"]
        original.rename(original.with_name(original.name.replace("TRIAL000001", "TRIAL000002")))
        with self.assertRaisesRegex(ValueError, "requires raw TRIAL1"):
            self.package()
        self.read_metadata.assert_not_called()
        self.build_clock.assert_not_called()

    def test_continuous_markers_validate_without_mutating_recordings(self):
        harp = self.make_harp(starts=(1.0,), ends=(4.0,))
        harp['time_reference'] = 123.0
        original = dict(harp)
        self.assertIsNone(packaging.harp_utils.qc_continuous_harp(harp))
        self.assertEqual(harp.keys(), original.keys())
        for key in original:
            self.assertIs(harp[key], original[key])

    def test_trial_markers_do_not_segment_or_filter_the_di3_clock(self):
        for starts, ends in (((), ()), ((0.0, 1.0), (4.0,)), ((0.0,), (3.0, 4.0))):
            with self.subTest(starts=starts, ends=ends):
                self.build_clock.reset_mock()
                harp = self.make_harp(starts=starts, ends=ends)
                _, clock, _ = packaging.prepare_continuous_slap2(self.plane_inputs(), harp)
                np.testing.assert_array_equal(clock['cycle_starts'], [-0.5, 0.5, 2.5, 3.5])
            self.build_clock.assert_called_once()

    def test_excluding_final_trial_is_forbidden_in_forced_mode(self):
        with self.assertRaisesRegex(ValueError, "Cannot exclude acquisition trials"):
            self.package(exclude_final_trial=True)
        self.read_metadata.assert_not_called()
        self.sync.assert_not_called()

    def test_recording_onset_fallback_retains_negative_first_pulse(self):
        del self.harp["recording_start_time"]
        self.package()
        self.assertEqual(self.build_clock.call_args.kwargs["recording_start"], -2.0)
        report = self.assert_qc_json("DMD1")
        self.assertEqual(report["clock"]["first_pulse_timestamp"], -0.5)
        self.assertFalse(report["clock"]["onset_warning"])
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])

    def test_nwb_round_trip_has_fluorescence_but_no_clock_anchor_containers(self):
        self.package()
        path = self.root / "continuous.nwb"
        with pynwb.NWBHDF5IO(str(path), "w") as io:
            io.write(self.nwb)
        with pynwb.NWBHDF5IO(str(path), "r") as io:
            restored = io.read()
            self.assertEqual(set(restored.processing), {"ophys"})
            self.assertEqual(set(restored.processing["ophys"].data_interfaces), {
                "ImageSegmentation", "Fluorescence_DMD1", "Fluorescence_DMD2",
            })
            for collection in (restored.acquisition, restored.stimulus,
                               restored.intervals, restored.scratch, restored.analysis):
                self.assertEqual(len(collection), 0)
            for obj in restored.all_children():
                self.assertNotRegex(obj.name.lower(), "anchor|control_line|control_time|cycle_start")
            series = restored.processing["ophys"]["Fluorescence_DMD2"].roi_response_series
            np.testing.assert_allclose(series["DMD2_F0_green"].timestamps[:], [0, 1, 2.5, 3.5])

    def test_unshimmed_packaging_regression_removed_pynwb_add_data_interface(self):
        """Packaging uses ProcessingModule.add, supported by current PyNWB."""
        self.package()
        self.assert_packaged("DMD1", [0.0, 0.5, 2.0, 3.25])

    def test_main_bypasses_trial_trimming_and_passes_original_harp_to_packaging(self):
        self.check_main_continuous_packaging(has_end=True)

    def test_main_packages_continuous_acquisition_without_end(self):
        self.check_main_continuous_packaging(has_end=False)
        self.assertEqual(packaging.find_slap2_trial_index(100, np.array([3.0]), np.array([])), 0)

    def check_main_continuous_packaging(self, has_end):
        # main clears its results directory: ALWAYS replace it with a temporary one.
        self.harp = self.make_harp(
            pulses=(3.5, 4.5, 6.5, 7.5),
            starts=(0.0, 3.0) if has_end else (3.0,),
            ends=(0.002, 8.0) if has_end else (),
        )
        self.harp['time_reference'] = 100.0
        results = self.root / "results"
        results.mkdir()
        for directory in (self.session, self.processed):
            (directory / "data_description.json").write_text(json.dumps({"name": "synthetic"}))
        (self.session / "instrument.json").write_text(json.dumps(self.instrument))
        (self.session / "acquisition.json").write_text("{}")
        (self.session / "synthetic.harp").mkdir()
        original_harp = {key: np.copy(value) for key, value in self.harp.items()}
        io = MagicMock()
        io.__enter__.return_value = io
        io.read.return_value = self.nwb
        argv = [
            "run_capsule", "--logger_format", "Random Natural Movies",
            "--input_session_dir", str(self.session),
            "--input_processed_dir", str(self.processed),
        ]
        with ExitStack() as stack, redirect_stdout(StringIO()):
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(packaging, "results_folder", results))
            stack.enter_context(patch.object(packaging, "data_folder", self.root))
            stack.enter_context(patch.object(packaging.hdmf_zarr, "NWBZarrIO", return_value=io))
            stack.enter_context(patch.object(packaging.nwb_utils, "create_base_nwb_file", return_value=self.nwb))
            stack.enter_context(patch.object(packaging.harp_utils, "extract_harp", return_value=self.harp))
            trim_leading = stack.enter_context(patch.object(
                packaging.harp_utils, "trim_leading_trial_pulse_artifact",
                wraps=packaging.harp_utils.trim_leading_trial_pulse_artifact,
            ))
            trim_trailing = stack.enter_context(patch.object(
                packaging.harp_utils, "trim_unterminated_harp_trial",
                side_effect=AssertionError("must not trim continuous recording"),
            ))
            stack.enter_context(patch.object(packaging, "find_stimulus_table", return_value=self.root / "unused.csv"))
            stack.enter_context(patch.object(packaging.stimulus_sync, "select_stimulus_logger", return_value=None))
            stack.enter_context(patch.object(packaging, "find_eye_tracking_paths", return_value=[]))
            consumers = {}
            for name in ("ensure_was_generated_by", "add_stim_table", "package_running_or_skip",
                         "package_eye_or_skip", "run_stimulus_qc", "write_data_process"):
                consumers[name] = stack.enter_context(patch.object(packaging, name))
            for name in ("compute_dff_qc", "compute_raw_fluorescence_qc"):
                stack.enter_context(patch.object(packaging.slap2_dff_qc, name))
            add_ophys = stack.enter_context(patch.object(
                packaging, "add_ophys_to_nwb", wraps=packaging.add_ophys_to_nwb
            ))
            if has_end:
                stack.enter_context(self.assertWarnsRegex(RuntimeWarning, 'Removed erroneous leading'))
            packaging.main()
        trim_leading.assert_called_once_with(self.harp)
        trim_trailing.assert_not_called()
        add_ophys.assert_called_once()
        prepared_harp = add_ophys.call_args.args[4]
        self.assertEqual(prepared_harp['time_reference'], 100.0)
        self.assertEqual(prepared_harp['recording_start_time'], -2.0)
        np.testing.assert_array_equal(prepared_harp['normalized_slap2_start'], [3.0])
        np.testing.assert_array_equal(prepared_harp['normalized_slap2_end'], [8.0] if has_end else [])
        self.assertEqual(prepared_harp.keys(), original_harp.keys())
        for name, index in (('add_stim_table', 3), ('package_running_or_skip', 1),
                            ('package_eye_or_skip', 3)):
            self.assertIs(consumers[name].call_args.args[index], prepared_harp)
        for key in ('slap2_cycle_clock_signal', 'slap2_cycle_clock_times', 'analog_times'):
            self.assertIs(add_ophys.call_args.args[4][key], self.harp[key])
        self.assertEqual(add_ophys.call_args.kwargs["logger_format"], "Random Natural Movies")
        self.assertIs(add_ophys.call_args.kwargs["exclude_final_trial"], False)
        for key, original in original_harp.items():
            np.testing.assert_array_equal(self.harp[key], original)
        self.assert_packaged("DMD1", [4.0, 4.5, 6.0, 7.25])
        self.assert_packaged("DMD2", [4.0, 5.0, 6.5, 7.5])
        self.assertTrue((results / "qc" / "syncing" / "DMD1_continuous_sync.json").is_file())
        with (results / "qc" / "syncing" / "DMD1_continuous_sync.json").open() as stream:
            self.assertNotIn('startup_handling', json.load(stream)['clock'])


class LegacyPrimaryPackagingTests(unittest.TestCase):
    def test_legacy_uses_source_free_dmd1_not_first_source_bearing_plane(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            summary = stack.enter_context(h5py.File(root / "summary.h5", "w", track_order=True))
            prefix = "acquisition_20260917_130000"
            for number in (2, 1):
                group = summary.create_group(f"Path{number}")
                group.create_dataset("frame_info/frame_line_idxs", data=[[1, 5, 9]])
                group.create_dataset("frame_info/trial_num_frames", data=[[3]])
                if number == 2:
                    for name in ("F0", "dF_denoised", "events"):
                        group.create_dataset(f"sources/temporal/{name}", data=np.ones((3, 1, 1)))
                (root / f"{prefix}_DMD{number}-TRIAL000001.dat").touch()
                # Trial-less files must not enter legacy acquisition filtering.
                (root / f"{prefix}_DMD{number}.dat").touch()
                with h5py.File(root / f"{prefix}_DMD{number}.meta", "w") as meta:
                    meta.create_dataset("AcquisitionContainer/ParsePlan/linesPerCycle", data=4)
            nwb = pynwb.NWBFile(  # type: ignore[abstract]  # HDMF generates concrete members at runtime.
                session_description="legacy primary test", identifier="legacy-primary",
                session_start_time=datetime(2026, 9, 17, tzinfo=timezone.utc),
            )
            harp = ContinuousSlap2PackagingTests.make_harp(pulses=(1, 2, 3), ends=(4,))
            stack.enter_context(patch.object(packaging, "create_imaging_plane",
                                            side_effect=ContinuousSlap2PackagingTests.mock_imaging_plane))
            stack.enter_context(patch.object(packaging, "add_mean_images"))
            stack.enter_context(patch.object(packaging, "add_image_segmentation",
                                            side_effect=ContinuousSlap2PackagingTests.mock_segmentation))
            reader = stack.enter_context(patch.object(packaging.slap2_sync, "read_dat_num_cycles", return_value=3))
            primary = stack.enter_context(patch.object(packaging.slap2_sync, "get_slap2_primary_plane_timestamps",
                                                      wraps=packaging.slap2_sync.get_slap2_primary_plane_timestamps))
            secondary = stack.enter_context(patch.object(packaging.slap2_sync, "get_slap2_secondary_plane_timestamps",
                                                        wraps=packaging.slap2_sync.get_slap2_secondary_plane_timestamps))
            sync = stack.enter_context(patch.object(packaging, "sync_slap2_fluorescence", wraps=packaging.sync_slap2_fluorescence))
            with self.assertWarnsRegex(UserWarning, "DMD1 is missing sources"), redirect_stdout(StringIO()):
                packaging.add_ophys_to_nwb(summary, nwb, {"instrument_id": "test", "notes": ""}, {}, harp, root)
            self.assertEqual([call.args[0] for call in sync.call_args_list], ["DMD1", "DMD2"])
            primary.assert_called_once()
            secondary.assert_called_once()
            self.assertEqual(reader.call_count, 2)
            for call in sync.call_args_list:
                self.assertEqual(len(call.kwargs["dat_paths"]), 1)
                self.assertIn("-TRIAL000001.dat", call.kwargs["dat_paths"][0].name)
            self.assertIsNone(sync.call_args_list[0].kwargs["trial_line_time_maps"])
            self.assertIsNotNone(sync.call_args_list[1].kwargs["trial_line_time_maps"])
            self.assertNotIn("Fluorescence_DMD1", nwb.processing["ophys"].data_interfaces)
            self.assertIn("Fluorescence_DMD2", nwb.processing["ophys"].data_interfaces)


if __name__ == "__main__":
    unittest.main()