import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from harp_utils import trim_unterminated_harp_trial

from run_capsule import (
    ensure_was_generated_by,
    filter_slap2_acquisition,
    filter_planes_with_sources,
    find_eye_tracking_paths,
    find_slap2_trial_index,
    infer_continuous_slap2_mode,
    read_stim_csv,
    resolve_slap2_acquisition,
)


class ReadStimCsvTests(unittest.TestCase):
    def test_normalizes_block_type(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stimulus.csv"
            pd.DataFrame(
                {"Duration": [1.0], "Block_Type": ["movie"]}
            ).to_csv(path, index=False)

            result = read_stim_csv(path)

        self.assertIn("BlockType", result.columns)
        self.assertNotIn("Block_Type", result.columns)
        self.assertEqual(result.loc[0, "BlockType"], "movie")

    def test_normalizes_columns_used_by_zebra_and_tuning_qc(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stimulus.csv"
            pd.DataFrame(
                {
                    "Duration": [300.0],
                    "Block_Number": [2],
                    "Block_Label": ["Zebra"],
                    "Trial_Type": ["movie"],
                    "Block_Type": ["movie"],
                }
            ).to_csv(path, index=False)

            result = read_stim_csv(path)

        for column in ("BlockNumber", "BlockLabel", "TrialType", "BlockType"):
            self.assertIn(column, result.columns)


class FakeNWBFile:
    def __init__(self, was_generated_by):
        self.fields = {"was_generated_by": was_generated_by}

    @property
    def was_generated_by(self):
        return self.fields.get("was_generated_by")

    @was_generated_by.setter
    def was_generated_by(self, value):
        if "was_generated_by" in self.fields:
            raise AttributeError("was_generated_by is already set")
        self.fields["was_generated_by"] = value


class EnsureWasGeneratedByTests(unittest.TestCase):
    @patch("run_capsule.package_version", return_value="0.2.5")
    def test_replaces_empty_provenance_with_installed_package_version(
        self, package_version
    ):
        nwbfile = FakeNWBFile([])

        ensure_was_generated_by(nwbfile)

        self.assertEqual(
            nwbfile.was_generated_by,
            [["aind-nwb-utils", "0.2.5"]],
        )
        package_version.assert_called_once_with("aind-nwb-utils")

    @patch("run_capsule.package_version")
    def test_preserves_existing_provenance(self, package_version):
        provenance = [["existing-package", "1.0.0"]]
        nwbfile = FakeNWBFile(provenance)

        ensure_was_generated_by(nwbfile)

        self.assertIs(nwbfile.was_generated_by, provenance)
        package_version.assert_not_called()


class FindEyeTrackingPathsTests(unittest.TestCase):
    def test_prefers_dedicated_eye_tracking_asset(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            eye_path = root / "eye_tracking"
            processed_path = root / "processed"
            eye_path.mkdir()
            processed_path.mkdir()
            dedicated = eye_path / "ellipses_processed.h5"
            fallback = processed_path / "ellipses_processed_old.h5"
            dedicated.touch()
            fallback.touch()

            self.assertEqual(
                find_eye_tracking_paths(eye_path, processed_path), [dedicated]
            )

    def test_falls_back_to_processed_asset(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "processed" / "ellipses_processed.h5"
            fallback.parent.mkdir()
            fallback.touch()

            self.assertEqual(
                find_eye_tracking_paths(root / "missing", fallback.parent),
                [fallback],
            )


class FilterPlanesWithSourcesTests(unittest.TestCase):
    def setUp(self):
        self.plane_inputs = [
            ("DMD1", "DMD1", "1"),
            ("DMD2", "DMD2", "2"),
        ]

    def test_warns_and_skips_dmd_with_missing_sources(self):
        experiment_summary = {
            "DMD1": {"sources": {"spatial": object()}},
            "DMD2": {},
        }

        with self.assertWarnsRegex(UserWarning, "DMD2 is missing sources"):
            result = filter_planes_with_sources(
                self.plane_inputs, experiment_summary
            )

        self.assertEqual(result, [self.plane_inputs[0]])

    def test_warns_and_skips_dmd_with_empty_sources(self):
        experiment_summary = {
            "DMD1": {"sources": {}},
            "DMD2": {"sources": {"temporal": object()}},
        }

        with self.assertWarnsRegex(UserWarning, "DMD1 is missing sources"):
            result = filter_planes_with_sources(
                self.plane_inputs, experiment_summary
            )

        self.assertEqual(result, [self.plane_inputs[1]])

    def test_raises_when_all_dmds_lack_sources(self):
        experiment_summary = {
            "DMD1": {},
            "DMD2": {"sources": {}},
        }

        with self.assertRaisesRegex(ValueError, "all DMDs"):
            filter_planes_with_sources(self.plane_inputs, experiment_summary)


class TrimUnterminatedHarpTrialTests(unittest.TestCase):
    @staticmethod
    def harp_data(starts, ends):
        clock_times = np.arange(0.0, 90.0)
        return {
            "slap2_start_signal": np.ones(len(starts)),
            "slap2_start_times": np.asarray(starts),
            "normalized_slap2_start": np.asarray(starts),
            "slap2_end_times": np.asarray(ends),
            "normalized_slap2_end": np.asarray(ends),
            "slap2_cycle_clock_signal": np.ones(len(clock_times)),
            "slap2_cycle_clock_times": clock_times,
            "normalized_slap2_cycle_clock_times": clock_times,
        }

    def test_balanced_trials_are_unchanged(self):
        harp_data = self.harp_data([0.0, 30.0], [29.0, 59.0])

        result, excluded = trim_unterminated_harp_trial(harp_data)

        np.testing.assert_array_equal(result["normalized_slap2_start"], [0.0, 30.0])
        np.testing.assert_array_equal(result["normalized_slap2_end"], [29.0, 59.0])
        self.assertFalse(excluded)

    def test_excludes_final_start_and_cycle_clock_without_end(self):
        harp_data = self.harp_data([0.0, 30.0, 60.0], [29.0, 59.0])

        with self.assertWarnsRegex(RuntimeWarning, "final SLAP2 trial"):
            result, excluded = trim_unterminated_harp_trial(harp_data)

        np.testing.assert_array_equal(result["normalized_slap2_start"], [0.0, 30.0])
        np.testing.assert_array_equal(result["normalized_slap2_end"], [29.0, 59.0])
        self.assertEqual(result["slap2_cycle_clock_times"][-1], 59.0)
        self.assertTrue(excluded)

    def test_preserves_single_start_without_end_for_continuous_inference(self):
        harp_data = self.harp_data([0.0], [])

        result, excluded = trim_unterminated_harp_trial(harp_data)

        np.testing.assert_array_equal(result["normalized_slap2_start"], [0.0])
        self.assertEqual(len(result["normalized_slap2_end"]), 0)
        np.testing.assert_array_equal(
            result["slap2_cycle_clock_times"],
            harp_data["slap2_cycle_clock_times"],
        )
        self.assertFalse(excluded)

    def test_rejects_other_pulse_count_mismatches(self):
        harp_data = self.harp_data([0.0, 30.0, 60.0], [29.0])

        with self.assertRaisesRegex(ValueError, "Unsupported SLAP2 trial pulse mismatch"):
            trim_unterminated_harp_trial(harp_data)


class FindSlap2TrialIndexTests(unittest.TestCase):
    def test_assigns_time_to_single_open_continuous_trial(self):
        self.assertEqual(
            find_slap2_trial_index(120.0, np.asarray([0.0]), np.asarray([])),
            0,
        )

    def test_rejects_time_before_single_open_continuous_trial(self):
        with self.assertRaisesRegex(Exception, "out of trial bounds"):
            find_slap2_trial_index(-1.0, np.asarray([0.0]), np.asarray([]))


class InferContinuousSlap2ModeTests(unittest.TestCase):
    @staticmethod
    def experiment_summary(first_chunk_start=1, second_chunk_start=101):
        return {
            "Path1": {
                "frame_info": {
                    "trial_num_frames": np.asarray([[3, 3]]),
                    "frame_line_idxs": np.asarray(
                        [[
                            first_chunk_start,
                            first_chunk_start + 49,
                            first_chunk_start + 99,
                            second_chunk_start,
                            second_chunk_start + 49,
                            second_chunk_start + 99,
                        ]]
                    ),
                }
            },
            "Path2": {
                "frame_info": {
                    "trial_num_frames": np.asarray([[3, 3]]),
                    "frame_line_idxs": np.asarray(
                        [[
                            first_chunk_start,
                            first_chunk_start + 49,
                            first_chunk_start + 99,
                            second_chunk_start,
                            second_chunk_start + 49,
                            second_chunk_start + 99,
                        ]]
                    ),
                }
            },
        }

    @staticmethod
    def plane_inputs(trial_number=1):
        return [
            (
                f"Path{dmd_number}",
                f"DMD{dmd_number}",
                str(dmd_number),
                [],
                [Path(
                    "acquisition_20260917_130000_"
                    f"DMD{dmd_number}-TRIAL{trial_number:06d}-CYCLE-000000.dat"
                )],
                2,
            )
            for dmd_number in (1, 2)
        ]

    @staticmethod
    def harp_data(starts=(0.0,), ends=(10.0,)):
        return {
            "normalized_slap2_start": np.asarray(starts),
            "normalized_slap2_end": np.asarray(ends),
        }

    def test_detects_one_raw_trial_with_continued_summary_chunks(self):
        self.assertTrue(infer_continuous_slap2_mode(
            self.experiment_summary(), self.plane_inputs(), self.harp_data()
        ))

    def test_detects_continuous_session_without_end_pulse(self):
        self.assertTrue(infer_continuous_slap2_mode(
            self.experiment_summary(), self.plane_inputs(),
            self.harp_data(ends=())
        ))

    def test_detects_partial_first_chunk_with_continued_line_indices(self):
        self.assertTrue(infer_continuous_slap2_mode(
            self.experiment_summary(
                first_chunk_start=200,
                second_chunk_start=300,
            ),
            self.plane_inputs(),
            self.harp_data(),
        ))

    def test_detects_continuous_session_with_removed_summary_chunk(self):
        summary = self.experiment_summary()
        for plane in summary.values():
            plane["frame_info"]["trial_num_frames"] = np.asarray([[3, 0, 3]])
            plane["frame_info"]["frame_line_idxs"] = np.asarray(
                [[1, 50, 100, 5000, 5050, 5100]]
            )

        self.assertTrue(infer_continuous_slap2_mode(
            summary, self.plane_inputs(), self.harp_data()
        ))

    def test_rejects_line_index_reset_after_removed_summary_chunk(self):
        summary = self.experiment_summary()
        for plane in summary.values():
            plane["frame_info"]["trial_num_frames"] = np.asarray([[3, 0, 3]])
            plane["frame_info"]["frame_line_idxs"] = np.asarray(
                [[1, 50, 100, 1, 50, 100]]
            )

        self.assertFalse(infer_continuous_slap2_mode(
            summary, self.plane_inputs(), self.harp_data()
        ))

    def test_preserves_trial_based_mode_when_raw_trials_differ(self):
        self.assertFalse(infer_continuous_slap2_mode(
            self.experiment_summary(), self.plane_inputs(trial_number=2),
            self.harp_data()
        ))

    def test_rejects_summary_chunks_whose_line_indices_reset(self):
        self.assertFalse(infer_continuous_slap2_mode(
            self.experiment_summary(second_chunk_start=1), self.plane_inputs(),
            self.harp_data()
        ))


class TrimUnterminatedProcessedTrialTests(unittest.TestCase):
    @patch("run_capsule.slap2_sync.read_dat_num_cycles", return_value=3000)
    def test_resolution_accepts_cycle_chunked_dat_files(self, read_dat_num_cycles):
        dat_paths = [
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-000000.dat"),
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-003000.dat"),
            Path("acquisition_20260917_153810_DMD2-TRIAL000001-CYCLE-000000.dat"),
        ]

        result = resolve_slap2_acquisition(dat_paths, n_summary_trials=1)

        self.assertEqual(result["retained_trial_count"], 1)
        self.assertEqual(result["highest_dat_trial"], 1)

    @patch("run_capsule.slap2_sync.read_dat_num_cycles")
    def test_only_validates_selected_acquisition(self, read_dat_num_cycles):
        dat_paths = [
            Path("acquisition_20260917_120000_DMD1-TRIAL000001-CYCLE-000100.dat"),
            Path("acquisition_20260917_130000_DMD1-TRIAL000001.dat"),
            Path("acquisition_20260917_130000_DMD1-TRIAL000002.dat"),
        ]

        with self.assertWarnsRegex(RuntimeWarning, "Multiple SLAP2 acquisitions"):
            result = resolve_slap2_acquisition(dat_paths, n_summary_trials=3)

        self.assertEqual(
            result["acquisition_prefix"], "acquisition_20260917_130000"
        )
        read_dat_num_cycles.assert_not_called()

    def test_resolution_excludes_final_dat_evidence(self):
        dat_paths = [
            Path("acquisition_20251111_144104_DMD1-TRIAL000001.dat"),
            Path("acquisition_20251111_144104_DMD1-TRIAL000002.dat"),
        ]

        result = resolve_slap2_acquisition(
            dat_paths, n_summary_trials=2, excluded_trailing_trials=1
        )

        self.assertEqual(result["retained_trial_count"], 1)
        self.assertEqual(result["highest_dat_trial"], 1)

    def test_does_not_validate_excluded_trailing_trial_chunks(self):
        dat_paths = [
            Path("acquisition_20260917_130000_DMD1-TRIAL000001-CYCLE-000000.dat"),
            Path("acquisition_20260917_130000_DMD1-TRIAL000002-CYCLE-003000.dat"),
        ]

        result = resolve_slap2_acquisition(
            dat_paths, n_summary_trials=2, excluded_trailing_trials=1
        )

        self.assertEqual(result["retained_trial_count"], 1)
        self.assertEqual(result["highest_dat_trial"], 1)

    def test_rejects_malformed_chunks_when_trailing_trial_is_retained(self):
        dat_paths = [
            Path("acquisition_20260917_130000_DMD1-TRIAL000001-CYCLE-000000.dat"),
            Path("acquisition_20260917_130000_DMD1-TRIAL000002-CYCLE-003000.dat"),
        ]

        with self.assertRaisesRegex(ValueError, "trial 2 start at 3000"):
            resolve_slap2_acquisition(dat_paths, n_summary_trials=2)

    def test_filters_final_trial_from_all_processed_arrays(self):
        dat_paths = [
            Path("acquisition_20251111_144104_DMD1-TRIAL000001.dat"),
            Path("acquisition_20251111_144104_DMD1-TRIAL000002.dat"),
        ]
        values = np.arange(5)
        acquisition_resolution = {
            "acquisition_prefix": "acquisition_20251111_144104",
            "excluded_trial_count": 0,
            "excluded_trailing_trials": 1,
            "retained_trial_count": 1,
        }

        result = filter_slap2_acquisition(
            1,
            dat_paths,
            [Path("acquisition_20251111_144104_DMD1.meta")],
            np.asarray([2, 3]),
            values,
            values,
            values,
            values,
            acquisition_resolution,
        )

        np.testing.assert_array_equal(result["trial_num_frames"], [2])
        for key in ("frame_line_idxs", "F0", "dF_denoised", "events"):
            np.testing.assert_array_equal(result[key], [0, 1])
        self.assertEqual(result["dat_paths"], dat_paths[:1])


if __name__ == "__main__":
    unittest.main()