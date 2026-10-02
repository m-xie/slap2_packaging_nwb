import unittest

import numpy as np
import pynwb

from slap2_running_packaging import (
    COUNTS_PER_REVOLUTION,
    add_raw_running_data_to_nwbfile,
    add_running_speed_to_nwbfile,
    extract_running_speeds,
    unwrap_quadrature_counts,
)


class RunningConversionTests(unittest.TestCase):
    def test_unwraps_signed_16_bit_counter(self):
        raw = np.array([32766, 32767, -32768, -32767], dtype=np.int16)
        np.testing.assert_array_equal(
            unwrap_quadrature_counts(raw),
            np.array([32766, 32767, 32768, 32769]),
        )

    def test_converts_counts_to_interval_speed(self):
        result = extract_running_speeds(
            timestamps=np.array([0.0, 0.5, 1.0]),
            raw_counts=np.array(
                [0, COUNTS_PER_REVOLUTION // 4, COUNTS_PER_REVOLUTION // 2],
                dtype=np.int16,
            ),
            wheel_radius_cm=10.0,
            subject_position=1.0,
        )
        np.testing.assert_allclose(result["net_rotation"], np.pi / 2)
        np.testing.assert_allclose(result["velocity"], 10 * np.pi)

    def test_discards_nonincreasing_timestamp_intervals(self):
        result = extract_running_speeds(
            timestamps=np.array([0.0, 0.1, 0.1, 0.2]),
            raw_counts=np.array([0, 1, 2, 3], dtype=np.int16),
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(np.all(result["end_time"] > result["start_time"]))

    def test_truncates_up_to_three_unpaired_samples(self):
        result = extract_running_speeds(
            timestamps=np.arange(6, dtype=float),
            raw_counts=np.arange(4, dtype=np.int16),
        )

        self.assertEqual(len(result), 3)

        result = extract_running_speeds(
            timestamps=np.arange(4, dtype=float),
            raw_counts=np.arange(7, dtype=np.int16),
        )

        self.assertEqual(len(result), 3)

    def test_rejects_more_than_three_unpaired_samples(self):
        with self.assertRaisesRegex(ValueError, "differ by 4 samples"):
            extract_running_speeds(
                timestamps=np.arange(8, dtype=float),
                raw_counts=np.arange(4, dtype=np.int16),
            )


class RunningNWBTests(unittest.TestCase):
    def setUp(self):
        from datetime import datetime, timezone

        self.nwbfile = pynwb.NWBFile(
            session_description="test",
            identifier="test",
            session_start_time=datetime.now(timezone.utc),
        )

    def test_adds_compatible_processed_and_raw_series(self):
        running_speed = extract_running_speeds(
            timestamps=np.array([0.0, 0.1, 0.2]),
            raw_counts=np.array([0, 1, 3], dtype=np.int16),
        )
        add_running_speed_to_nwbfile(self.nwbfile, running_speed)
        add_raw_running_data_to_nwbfile(
            self.nwbfile,
            np.array([0.0, 0.1, 0.2]),
            np.array([0, 1, 3], dtype=np.int16),
        )

        self.assertIn("running", self.nwbfile.processing)
        running_interfaces = self.nwbfile.processing["running"].data_interfaces
        self.assertIn("running_speed", running_interfaces)
        self.assertIn("running_wheel_rotation", running_interfaces)
        self.assertIn("raw_running_wheel_counts", self.nwbfile.acquisition)


if __name__ == "__main__":
    unittest.main()