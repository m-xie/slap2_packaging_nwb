import unittest

import numpy as np

from harp_utils import trim_leading_trial_pulse_artifact


class TrimLeadingTrialPulseArtifactTests(unittest.TestCase):
    @staticmethod
    def harp_data(starts, ends):
        return {
            "slap2_start_signal": np.ones(len(starts)),
            "slap2_start_times": np.asarray(starts),
            "normalized_slap2_start": np.asarray(starts),
            "slap2_end_signal": np.ones(len(ends)),
            "slap2_end_times": np.asarray(ends),
            "normalized_slap2_end": np.asarray(ends),
        }

    def test_preserves_normal_first_trial(self):
        harp_data = self.harp_data([0.0, 30.0], [29.0, 59.0])

        result = trim_leading_trial_pulse_artifact(harp_data)

        np.testing.assert_array_equal(result["normalized_slap2_start"], [0.0, 30.0])
        np.testing.assert_array_equal(result["normalized_slap2_end"], [29.0, 59.0])

    def test_warns_and_removes_short_leading_pulse_pair(self):
        harp_data = self.harp_data(
            [0.0, 16.5, 46.5, 76.5],
            [0.001984, 46.0, 76.0],
        )

        warning = (
            "Removed erroneous leading SLAP2 pulse pair: duration was "
            "0.001984 seconds versus a typical later trial duration of "
            "29.500000 seconds"
        )
        with self.assertWarnsRegex(RuntimeWarning, warning):
            result = trim_leading_trial_pulse_artifact(harp_data)

        np.testing.assert_array_equal(
            result["normalized_slap2_start"], [16.5, 46.5, 76.5]
        )
        np.testing.assert_array_equal(result["normalized_slap2_end"], [46.0, 76.0])
        self.assertEqual(len(result["slap2_start_signal"]), 3)
        self.assertEqual(len(result["slap2_end_signal"]), 2)


if __name__ == "__main__":
    unittest.main()