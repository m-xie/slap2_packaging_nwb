import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb

from slap2_eye_tracking_packaging import (
    add_eye_tracking_to_nwbfile,
    align_eye_tracking_frames,
    compute_tracking_metrics,
    dilate_blink_frames,
    load_camera_frame_times,
    load_eye_tracking_hdf,
)


class EyeTrackingPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.frames = pd.Index([0, 1, 2, 3], name="frame")
        self.base_table = pd.DataFrame({
            "center_x": [10.0, 11.0, 12.0, 13.0],
            "center_y": [20.0, 21.0, 22.0, 23.0],
            "height": [2.0, 2.0, np.nan, 2.0],
            "width": [3.0, 3.0, np.nan, 3.0],
            "phi": [0.0, 0.1, 0.2, 0.3],
        }, index=self.frames)

    def write_hdf(self):
        path = self.root / "ellipses_processed_test.h5"
        for key in ("cr", "eye", "pupil"):
            self.base_table.to_hdf(path, key=key)
        return path

    def write_metadata(self):
        path = self.root / "metadata.csv"
        pd.DataFrame({
            "ReferenceTime": [np.nan] * 4,
            "CameraFrameNumber": self.frames,
            "CameraFrameTime": [100.0, 100.03, 100.07, 100.10],
        }).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def test_loads_and_aligns_by_frame_number(self):
        eye_data = load_eye_tracking_hdf(self.write_hdf())
        frame_times = load_camera_frame_times(self.write_metadata(), 99.0)
        aligned = align_eye_tracking_frames(eye_data, frame_times)

        np.testing.assert_allclose(aligned["timestamps"], [1.0, 1.03, 1.07, 1.10])
        self.assertEqual(aligned.index.tolist(), [0, 1, 2, 3])

    def test_truncates_up_to_three_extra_camera_frames(self):
        eye_data = load_eye_tracking_hdf(self.write_hdf())
        frame_times = pd.Series(
            np.arange(6, dtype=float),
            index=pd.Index(range(6), name="frame"),
            name="timestamps",
        )

        aligned = align_eye_tracking_frames(eye_data, frame_times)

        self.assertEqual(aligned.index.tolist(), [0, 1, 2, 3])

    def test_truncates_up_to_three_extra_fit_frames(self):
        eye_data = pd.DataFrame(
            index=pd.Index(range(6), name="frame"),
        )
        frame_times = pd.Series(
            np.arange(4, dtype=float),
            index=pd.Index(range(4), name="frame"),
            name="timestamps",
        )

        aligned = align_eye_tracking_frames(eye_data, frame_times)

        self.assertEqual(aligned.index.tolist(), [0, 1, 2, 3])

    def test_rejects_more_than_three_extra_camera_frames(self):
        eye_data = load_eye_tracking_hdf(self.write_hdf())
        frame_times = pd.Series(
            np.arange(8, dtype=float),
            index=pd.Index(range(8), name="frame"),
            name="timestamps",
        )

        with self.assertRaisesRegex(ValueError, "length discrepancy is 4"):
            align_eye_tracking_frames(eye_data, frame_times)

    def test_marks_missing_fit_and_adjacent_frames_as_blinks(self):
        eye_data = load_eye_tracking_hdf(self.write_hdf())
        frame_times = load_camera_frame_times(self.write_metadata(), 99.0)
        aligned = align_eye_tracking_frames(eye_data, frame_times)
        result = compute_tracking_metrics(aligned, dilation_frames=1)

        self.assertEqual(result["likely_blink"].tolist(), [False, True, True, True])
        self.assertEqual(result.loc[2, "pupil_area"], -1.0)
        self.assertTrue(np.isnan(result.loc[2, "pupil_area_raw"]))

    def test_dilates_blinks_without_scipy(self):
        result = dilate_blink_frames(
            [False, False, True, False, False, False], dilation_frames=2
        )

        self.assertEqual(result.tolist(), [True, True, True, True, True, False])

    def test_marks_any_missing_pupil_parameter_as_a_blink(self):
        eye_data = load_eye_tracking_hdf(self.write_hdf())
        frame_times = load_camera_frame_times(self.write_metadata(), 99.0)
        aligned = align_eye_tracking_frames(eye_data, frame_times)
        aligned.loc[0, "pupil_width"] = np.nan
        aligned.loc[1, "pupil_center_x"] = np.nan

        result = compute_tracking_metrics(aligned, dilation_frames=0)

        self.assertTrue(result.loc[0, "likely_blink"])
        self.assertTrue(np.isnan(result.loc[0, "pupil_area_raw"]))
        self.assertTrue(result.loc[1, "likely_blink"])

    def test_adds_reference_compatible_nwb_interfaces(self):
        eye_data = load_eye_tracking_hdf(self.write_hdf())
        frame_times = load_camera_frame_times(self.write_metadata(), 99.0)
        result = compute_tracking_metrics(align_eye_tracking_frames(eye_data, frame_times))
        nwbfile = pynwb.NWBFile(
            session_description="test",
            identifier="test",
            session_start_time=datetime.now(timezone.utc),
        )

        add_eye_tracking_to_nwbfile(nwbfile, result)

        module = nwbfile.processing["eye_tracking"]
        for name in ("ellipse", "pupil", "corneal_reflection", "likely_blink_times"):
            self.assertIn(name, module.data_interfaces)


if __name__ == "__main__":
    unittest.main()