"""Soma traces keep their own ROI identities and share source timing filters."""

import unittest
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import hdmf_zarr
import numpy as np
import pynwb

import run_capsule as packaging
import slap2_soma_packaging as soma
from qc.slap2_dff_qc import _is_dff_series, _is_raw_fluorescence_series


class SomaPackagingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.summary = self.stack.enter_context(h5py.File(self.root / "summary.h5", "w"))
        self.plane = self.summary.create_group("Path1")
        group = self.plane.create_group("user_rois")
        group.create_dataset("labels", data=np.asarray([[b"neuropil", b"soma", b" Soma "]]))
        self.values = np.arange(36, dtype=float).reshape(6, 2, 3)
        self.values[2, 1, 1] = np.nan
        group.create_dataset("F", data=self.values)
        group.create_dataset("Fsvd", data=self.values + 100)
        masks = np.zeros((5, 4, 2, 3), dtype=np.uint8)
        masks[1, 2, 0, 1] = 1
        masks[3, 1, 1, 2] = 1
        group.create_dataset("mask", data=masks)

    def filter_traces(self, traces):
        prefix = "acquisition_20260917_130000"
        return packaging.filter_slap2_acquisition(
            1, [Path(f"{prefix}_DMD1-TRIAL000001.dat")],
            [Path(f"{prefix}_DMD1.meta")], np.array([1, 3, 2]),
            np.arange(6), self.values, self.values, self.values,
            dict(acquisition_prefix=prefix, excluded_trial_count=1,
                 excluded_trailing_trials=1, retained_trial_count=1),
            additional_traces=traces,
        )["additional_traces"]

    def test_extracts_only_labeled_somas_and_filters_both_trial_ends(self):
        traces = soma.read_soma_traces(self.plane, 6)
        np.testing.assert_array_equal(traces["soma_F"], self.values[:, :, [1, 2]])
        filtered = self.filter_traces(traces)
        for name, offset in (("soma_F", 0), ("soma_Fsvd", 100)):
            np.testing.assert_array_equal(filtered[name], (self.values + offset)[1:4][:, :, [1, 2]])

    def test_optional_user_rois_and_fsvd(self):
        del self.plane["user_rois/Fsvd"]
        self.assertEqual(set(soma.read_soma_traces(self.plane, 6)), {"soma_F"})
        del self.plane["user_rois"]
        self.assertEqual(soma.read_soma_traces(self.plane, 6), {})

    def test_no_soma_label_does_not_require_trace_datasets(self):
        del self.plane["user_rois"]
        self.plane.create_dataset("user_rois/labels", data=[[b"neuropil"]])
        self.assertEqual(soma.read_soma_traces(self.plane, 6), {})

    def test_rejects_misaligned_samples_rois_and_channels(self):
        for shape in ((5, 2, 3), (6, 2, 2), (6, 3, 3), (6, 3)):
            with self.subTest(shape=shape):
                del self.plane["user_rois/F"]
                self.plane.create_dataset("user_rois/F", data=np.ones(shape))
                with self.assertRaisesRegex(ValueError, "must have shape"):
                    soma.read_soma_traces(self.plane, 6)

    def test_trial_filter_rejects_mismatched_soma_sample_count(self):
        with self.assertRaisesRegex(ValueError, "soma_F has 5 samples"):
            self.filter_traces({"soma_F": self.values[:5]})

    def test_legacy_sync_returns_trial_filtered_soma_traces(self):
        self.plane.create_dataset("frame_info/frame_line_idxs", data=[[1, 1, 2, 3, 1, 2]])
        self.plane.create_dataset("frame_info/trial_num_frames", data=[[1, 3, 2]])
        for name in ("F0", "dF_denoised", "events"):
            self.plane.create_dataset(f"sources/temporal/{name}", data=self.values)
        prefix = "acquisition_20260917_130000"
        meta = self.root / f"{prefix}_DMD1.meta"
        with h5py.File(meta, "w") as handle:
            handle.create_dataset("AcquisitionContainer/ParsePlan/linesPerCycle", data=[[4]])
        with patch.object(packaging.slap2_sync, "get_trial_num_cycles", return_value=[1]), patch.object(
            packaging.slap2_sync, "get_slap2_primary_plane_timestamps",
            return_value=(np.array([0.0, 1.0, 2.0]), [], {}),
        ):
            traces, timestamps, _, _ = packaging.sync_slap2_fluorescence(
                "DMD1", "1", self.summary, [meta],
                dict(slap2_cycle_clock_signal=[], slap2_cycle_clock_times=[],
                     normalized_slap2_start=[0.0]),
                dat_paths=[self.root / f"{prefix}_DMD1-TRIAL000001.dat"],
                plane_key="Path1",
                acquisition_resolution=dict(acquisition_prefix=prefix,
                    excluded_trial_count=1, excluded_trailing_trials=1,
                    retained_trial_count=1, highest_dat_trial=1),
            )
        np.testing.assert_array_equal(traces["soma_F"], self.values[1:4][:, :, [1, 2]])
        self.assertEqual(len(timestamps), len(traces["soma_F"]))

    def test_separate_soma_masks_traces_and_hdf5_zarr_roundtrip(self):
        nwb = pynwb.NWBFile(  # type: ignore[abstract]  # HDMF supplies concrete members at runtime.
            session_description="soma test", identifier="soma-test",
            session_start_time=datetime(2026, 10, 5, tzinfo=timezone.utc),
        )
        device = nwb.create_device(name="test")
        channel = pynwb.ophys.OpticalChannel(
            name="green", description="test", emission_lambda=510.0,
        )
        plane = nwb.create_imaging_plane(
            name="DMD1", optical_channel=channel, device=device,
            excitation_lambda=920.0, indicator="test", location="V1",
        )
        ophys = nwb.create_processing_module(name="ophys", description="test")
        segmentation = pynwb.ophys.ImageSegmentation()
        ophys.add(segmentation)
        traces = self.filter_traces(soma.read_soma_traces(self.plane, 6))
        timestamps = np.array([0.0, 1.0, np.nan])
        soma.add_soma_fluorescence(
            self.plane, traces, timestamps, "DMD1", plane,
            segmentation, ophys, packaging.get_pixel_mask,
        )
        table = segmentation["SomaPlaneSegmentation_DMD1"]
        self.assertEqual(list(table["user_roi_index"].data), [1, 2])
        self.assertEqual(list(table["label"].data), ["soma", " Soma "])
        self.assertEqual(list(table["z_min"].data), [0, 1])
        np.testing.assert_array_equal(table["pixel_mask"][0], [(2, 1, 1.0)])
        container = ophys["SomaFluorescence_DMD1"]
        self.assertEqual(len(container.roi_response_series), 4)
        for name in container.roi_response_series:
            self.assertFalse(_is_dff_series(name))
            self.assertFalse(_is_raw_fluorescence_series(name))
        for io_class, suffix in ((pynwb.NWBHDF5IO, ".nwb"), (hdmf_zarr.NWBZarrIO, ".nwb.zarr")):
            with self.subTest(format=suffix):
                path = self.root / f"test{suffix}"
                with io_class(str(path), mode="w") as io:
                    io.write(deepcopy(nwb))
                with io_class(str(path), mode="r") as io:
                    read = io.read()
                    saved = read.processing["ophys"]["SomaFluorescence_DMD1"]
                    for name, series in saved.roi_response_series.items():
                        kind = "Fsvd" if "Fsvd" in name else "F"
                        channel = 1 if name.endswith("red") else 0
                        np.testing.assert_array_equal(series.data[:], traces[f"soma_{kind}"][:2, channel, :])
                        np.testing.assert_array_equal(series.timestamps[:], [0.0, 1.0])
                        self.assertEqual(series.rois.table.name, "SomaPlaneSegmentation_DMD1")
                        np.testing.assert_array_equal(series.rois.data[:], [0, 1])


if __name__ == "__main__":
    unittest.main()