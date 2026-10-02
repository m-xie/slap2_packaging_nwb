import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qc.slap2_receptive_fields_qc import (
    _filter_assigned_trials,
    compute_receptive_field_qc,
)


class FilterAssignedTrialsTests(unittest.TestCase):
    @patch("qc.slap2_receptive_fields_qc.pynwb.NWBHDF5IO")
    def test_does_not_create_folder_without_rf_intervals(self, io_class):
        io_class.return_value.__enter__.return_value.read.return_value = (
            SimpleNamespace(intervals={}, processing={})
        )
        with TemporaryDirectory() as directory:
            qc_folder = Path(directory)

            compute_receptive_field_qc(qc_folder, qc_folder / "input.nwb")

            self.assertFalse((qc_folder / "receptive_fields").exists())

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