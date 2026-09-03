import unittest

import pandas as pd

from slap2_receptive_fields_qc import _filter_assigned_trials


class FilterAssignedTrialsTests(unittest.TestCase):
    def test_excludes_intertrial_and_discarded_trial_stimuli(self):
        stim_df = pd.DataFrame({
            "start_time": [1.0, 2.0, 3.0, 4.0],
            "slap2_trial_idx": [0, -1, 1, -1],
        })

        result = _filter_assigned_trials(stim_df)

        self.assertEqual(result["start_time"].tolist(), [1.0, 3.0])
        self.assertEqual(result["slap2_trial_idx"].tolist(), [0, 1])
        self.assertEqual(result.index.tolist(), [0, 1])

    def test_preserves_legacy_table_without_trial_indices(self):
        stim_df = pd.DataFrame({"start_time": [1.0, 2.0]})

        result = _filter_assigned_trials(stim_df)

        self.assertIs(result, stim_df)


if __name__ == "__main__":
    unittest.main()