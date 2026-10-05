import unittest
from pathlib import Path
from unittest.mock import Mock

from slap2_dat_utils import parse_dat_file, validate_dat_files


class ParseDatFileTests(unittest.TestCase):
    def test_omitted_trial_defaults_to_one_with_or_without_chunks(self):
        for suffix, offset in [("", None), ("-CYCLE-000000", 0), ("-CYCLE-003000", 3000)]:
            with self.subTest(suffix=suffix):
                result = parse_dat_file(Path(f"acquisition_20260917_153810_DMD1{suffix}.dat"))
                self.assertEqual(result.trial_number, 1)
                self.assertEqual(result.dmd_number, 1)
                self.assertEqual(result.cycle_offset, offset)

    def test_parses_cycle_chunk_context(self):
        path = Path(
            "acquisition_20260917_153810_DMD2-TRIAL000003-CYCLE-006000.dat"
        )

        result = parse_dat_file(path)

        self.assertEqual(result.acquisition_prefix, "acquisition_20260917_153810")
        self.assertEqual(result.dmd_number, 2)
        self.assertEqual(result.trial_number, 3)
        self.assertEqual(result.cycle_offset, 6000)


class ValidateDatFilesTests(unittest.TestCase):
    def test_omitted_and_explicit_trial_one_chunks_share_trial(self):
        reader = Mock(return_value=3000)
        validate_dat_files([
            Path("acquisition_20260917_153810_DMD1-CYCLE-000000.dat"),
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-003000.dat"),
        ], reader)
        reader.assert_called_once()

    def test_accepts_contiguous_cycle_chunks(self):
        dat_paths = [
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-006000.dat"),
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-000000.dat"),
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-003000.dat"),
        ]
        read_cycle_count = Mock(side_effect=[3000, 3000])

        validate_dat_files(dat_paths, read_cycle_count)

        self.assertEqual(read_cycle_count.call_count, 2)

    def test_accepts_one_legacy_file_per_trial(self):
        validate_dat_files([
            Path("acquisition_20251111_144104_DMD1-TRIAL000001.dat"),
            Path("acquisition_20251111_144104_DMD1-TRIAL000002.dat"),
        ], Mock())

    def test_rejects_cycle_chunk_gap(self):
        dat_paths = [
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-000000.dat"),
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-006000.dat"),
        ]

        with self.assertRaisesRegex(ValueError, "cycle-chunk gap"):
            validate_dat_files(dat_paths, Mock(return_value=3000))

    def test_rejects_mixed_chunked_and_legacy_files(self):
        dat_paths = [
            Path("acquisition_20260917_153810_DMD1-TRIAL000001.dat"),
            Path("acquisition_20260917_153810_DMD1-TRIAL000001-CYCLE-003000.dat"),
        ]

        with self.assertRaisesRegex(ValueError, "Mixed chunked and unchunked"):
            validate_dat_files(dat_paths, Mock())


if __name__ == "__main__":
    unittest.main()