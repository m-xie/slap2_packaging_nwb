from importlib import import_module
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import slap2_path_metadata as metadata


MAGIC = 322379495
SOURCE = "slap2_utils.utils.file_header.load_file_header_v2"


class SyntheticDatTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_dat(self, offset=None, cycles=2, lines=3, bytes_per_cycle=8,
                  trailing=b"", prefix="acquisition_20260917_153810",
                  dmd=1, trial=1, fields=None):
        suffix = "" if offset is None else f"-CYCLE-{offset:06d}"
        path = self.root / f"{prefix}_DMD{dmd}-TRIAL{trial:06d}{suffix}.dat"
        # Minimal real v2 header: first offset, cycle size, line count, channel
        # count and channel mask. The library translates these pairs to floats.
        pairs = [(0, 56), (3, bytes_per_cycle), (4, lines), (8, 1), (9, 1)]
        if fields is not None:
            pairs = fields
        header_size = (4 + 2 * len(pairs)) * 4
        words = [MAGIC, 2, header_size]
        for field, value in pairs:
            words.extend([field, value])
        words.append(MAGIC)
        path.write_bytes(
            np.asarray(words, dtype=np.uint32).tobytes()
            + bytes(cycles * bytes_per_cycle) + trailing
        )
        return path

    def rewrite_word(self, path, index, value):
        data = bytearray(path.read_bytes())
        data[index * 4:(index + 1) * 4] = np.uint32(value).tobytes()
        path.write_bytes(data)

    def parser(self, **kwargs):
        patcher = patch.object(metadata, "_load_file_header_v2", **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock


class ReadPathMetadataTests(SyntheticDatTests):
    def test_legacy_result_uses_library_values_not_payload_or_filename(self):
        path = self.write_dat(cycles=2)
        parser = self.parser(return_value=({"linesPerCycle": np.float64(7)}, np.int64(11)))
        result = metadata.read_path_metadata([str(path)])
        self.assertEqual(result, {
            "lines_per_cycle": 7, "total_cycles": 11, "total_lines": 77,
            "source": SOURCE, "chunk_count": 1,
        })
        for key in ("lines_per_cycle", "total_cycles", "total_lines", "chunk_count"):
            self.assertIs(type(result[key]), int)
        parser.assert_called_once()

    def test_readonly_memmap_namespace_floor_shape_and_cleanup(self):
        path = self.write_dat(trailing=b"\x00\x00")
        mappings = []

        def parse(obj, raw):
            self.assertIsInstance(obj, SimpleNamespace)
            self.assertEqual(vars(obj), {"MAGIC_NUMBER": np.uint32(MAGIC)})
            self.assertIsInstance(raw, np.memmap)
            self.assertEqual(raw.dtype, np.dtype(np.uint32))
            self.assertEqual(raw.mode, "r")
            self.assertFalse(raw.flags.writeable)
            self.assertEqual(raw.shape, (path.stat().st_size // 4,))
            mappings.append(raw._mmap)
            return {"linesPerCycle": 3.0}, 2

        self.parser(side_effect=parse)
        self.assertEqual(metadata.read_path_metadata([path])["total_cycles"], 2)
        self.assertTrue(mappings[0].closed)

    def test_out_of_order_chunks_use_cached_actual_counts(self):
        paths = [self.write_dat(offset=5), self.write_dat(offset=0), self.write_dat(offset=2)]
        parser = self.parser(side_effect=[
            ({"linesPerCycle": 3.0}, 1),
            ({"linesPerCycle": 3.0}, 2),
            ({"linesPerCycle": 3.0}, 3),
        ])
        with patch.object(metadata, "validate_dat_files", wraps=metadata.validate_dat_files) as validate:
            result = metadata.read_path_metadata(iter(paths))
        self.assertEqual(result["total_cycles"], 6)
        self.assertEqual(result["total_lines"], 18)
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(parser.call_count, 3)
        validate.assert_called_once()
        callback = validate.call_args.args[1]
        self.assertEqual([callback(path) for path in paths], [1, 2, 3])
        self.assertEqual(parser.call_count, 3)

    def test_accepts_zero_complete_cycles_only_in_final_chunk(self):
        paths = [self.write_dat(offset=2, cycles=0, trailing=b"\x00\x00"), self.write_dat(offset=0)]
        self.parser(side_effect=[({"linesPerCycle": 3.0}, 0), ({"linesPerCycle": 3.0}, 2)])
        self.assertEqual(metadata.read_path_metadata(paths)["total_cycles"], 2)

    def test_rejects_zero_cycles_in_nonfinal_chunk(self):
        paths = [self.write_dat(offset=0), self.write_dat(offset=2)]
        self.parser(side_effect=[({"linesPerCycle": 3.0}, 0), ({"linesPerCycle": 3.0}, 2)])
        # A nonfinal empty chunk necessarily also violates unique contiguity.
        with self.assertRaisesRegex(ValueError, "cycle-chunk gap|Only the final"):
            metadata.read_path_metadata(paths)

    def test_rejects_no_complete_cycles(self):
        self.parser(return_value=({"linesPerCycle": 3.0}, 0))
        for offset in (None, 0):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "total_cycles must be > 0"):
                metadata.read_path_metadata([self.write_dat(offset=offset, cycles=0)])

    def test_rejects_missing_or_empty_path_collection(self):
        parser = self.parser()
        for value in (None, [], (), iter(()), "", Path("example.dat")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty iterable"):
                metadata.read_path_metadata(value)
        parser.assert_not_called()

    def test_rejects_missing_file(self):
        path = self.write_dat()
        path.unlink()
        parser = self.parser()
        with self.assertRaises(FileNotFoundError):
            metadata.read_path_metadata([path])
        parser.assert_not_called()

    def test_rejects_unsupported_filename(self):
        parser = self.parser()
        with self.assertRaisesRegex(ValueError, "Unsupported SLAP2 .dat filename"):
            metadata.read_path_metadata([self.root / "bad.dat"])
        parser.assert_not_called()

    def test_rejects_multiple_acquisition_dmd_or_trial(self):
        first = self.write_dat(offset=0)
        parser = self.parser()
        for identity in ({"prefix": "other_20260917_153810"}, {"dmd": 2}, {"trial": 2}):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "single acquisition prefix, DMD, and trial"):
                metadata.read_path_metadata([first, self.write_dat(offset=2, **identity)])
        parser.assert_not_called()

    def test_layout_errors_use_existing_validator(self):
        self.parser(return_value=({"linesPerCycle": 3.0}, 2))
        for offsets, message in [
            ([1], "expected 0"), ([0, 3], "cycle-chunk gap"),
            ([0, 1], "cycle-chunk overlap"), ([0, 0], "Duplicate cycle offsets"),
            ([None, None], "Duplicate unchunked"), ([None, 0], "Mixed chunked and unchunked"),
        ]:
            with self.subTest(offsets=offsets), self.assertRaisesRegex(ValueError, message):
                metadata.read_path_metadata([self.write_dat(offset=o) for o in offsets])

    def test_rejects_inconsistent_lines_per_cycle(self):
        paths = [self.write_dat(offset=0), self.write_dat(offset=2)]
        self.parser(side_effect=[({"linesPerCycle": 3.0}, 2), ({"linesPerCycle": 4.0}, 1)])
        with self.assertRaisesRegex(ValueError, "same positive integer linesPerCycle"):
            metadata.read_path_metadata(paths)

    def test_rejects_invalid_library_dimensions(self):
        path = self.write_dat()
        parser = self.parser()
        invalid = [0, -1, 1.5, float("nan"), float("inf"), "3", None, True, np.bool_(True), [3]]
        for value in invalid:
            parser.return_value = ({"linesPerCycle": value}, 2)
            with self.subTest(lines=value), self.assertRaisesRegex(ValueError, "linesPerCycle must be an integer"):
                metadata.read_path_metadata([path])
        for value in invalid[1:]:
            parser.return_value = ({"linesPerCycle": 3}, value)
            with self.subTest(cycles=value), self.assertRaisesRegex(ValueError, "numCycles must be an integer"):
                metadata.read_path_metadata([path])

    def test_clear_parser_errors_close_mapping(self):
        path = self.write_dat()
        mappings = []

        def fail(obj, raw):
            mappings.append(raw._mmap)
            raise AssertionError("channel mismatch")

        self.parser(side_effect=fail)
        with self.assertRaisesRegex(ValueError, f"{path.name}: channel mismatch"):
            metadata.read_path_metadata([path])
        self.assertTrue(mappings[0].closed)

    def test_actionable_lazy_import_error(self):
        with patch.object(metadata, "import_module", side_effect=ModuleNotFoundError("test: unavailable library")):
            with self.assertRaisesRegex(ImportError, "requires.*load_file_header_v2.*pinned"):
                metadata.read_path_metadata([self.write_dat()])

    def test_actionable_error_for_incompatible_library(self):
        with patch.object(metadata, "import_module", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImportError, "requires.*load_file_header_v2.*pinned"):
                metadata.read_path_metadata([self.write_dat()])


class PreambleGuardTests(SyntheticDatTests):
    def test_empty_short_or_raw_zero_files_never_reach_library(self):
        parser = self.parser()
        path = self.write_dat()
        for payload in (b"", bytes(2), bytes(12), bytes(128)):
            path.write_bytes(payload)
            with self.subTest(size=len(payload)), self.assertRaisesRegex(ValueError, "truncated|magic number"):
                metadata.read_path_metadata([path])
        parser.assert_not_called()

    def test_invalid_preamble_never_reaches_library(self):
        parser = self.parser()
        for index, value, message in [
            (0, 0, "magic number"), (1, 1, "version"), (1, 3, "version"),
            (2, 0, "header size"), (2, 12, "header size"), (2, 55, "header size"),
            (2, 4096, "exceeds file size"), (2, 52, "field/value pair"),
            (13, 0, "end magic"), (6, 0, "zero bytesPerCycle"),
            (4, 0, "firstCycleOffsetBytes"), (4, 54, "firstCycleOffsetBytes"),
            (4, 57, "firstCycleOffsetBytes"), (4, 1000, "firstCycleOffsetBytes"),
        ]:
            path = self.write_dat()
            self.rewrite_word(path, index, value)
            with self.subTest(index=index, value=value), self.assertRaisesRegex(ValueError, message):
                metadata.read_path_metadata([path])
        parser.assert_not_called()

    def test_missing_arithmetic_fields_never_reach_library(self):
        parser = self.parser()
        for index, message in [(3, "firstCycleOffsetBytes"), (5, "bytesPerCycle")]:
            path = self.write_dat()
            self.rewrite_word(path, index, 99)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                metadata.read_path_metadata([path])
        parser.assert_not_called()


class LibraryParserIntegrationTests(SyntheticDatTests):
    @classmethod
    def setUpClass(cls):
        try:
            module = import_module("slap2_utils.utils.file_header")
        except ImportError as exc:
            raise unittest.SkipTest(f"SLAP2_Utils parser is not installed: {exc}")
        cls.library_parser = staticmethod(module.load_file_header_v2)

    def test_real_v2_header_matches_direct_library_counts(self):
        # With 10-byte cycles, the uint32 view can discard the last cycle's
        # final two bytes. Preserve the library's result, not a filesize formula.
        for trailing, expected in [(b"", 2), (b"\x00\x00", 3)]:
            with self.subTest(trailing=trailing):
                path = self.write_dat(cycles=3, lines=7, bytes_per_cycle=10, trailing=trailing)
                with patch("slap2_utils.utils.file_header.load_file_header_v2", wraps=self.library_parser) as parser:
                    result = metadata.read_path_metadata([path])
                parser.assert_called_once()
                raw = np.frombuffer(path.read_bytes(), dtype=np.uint32, count=path.stat().st_size // 4)
                header, cycles = self.library_parser(SimpleNamespace(MAGIC_NUMBER=np.uint32(MAGIC)), raw)
                self.assertEqual(cycles, expected)
                self.assertEqual(result, {
                    "lines_per_cycle": int(header["linesPerCycle"]), "total_cycles": cycles,
                    "total_lines": cycles * 7, "source": SOURCE, "chunk_count": 1,
                })

    def test_real_parser_ignores_two_byte_tail_and_incomplete_final_cycle(self):
        first = self.write_dat(offset=0, cycles=2, trailing=b"\x00\x00")
        for trailing in (b"", bytes(2), bytes(6)):
            with self.subTest(trailing=trailing):
                last = self.write_dat(offset=2, cycles=0, trailing=trailing)
                result = metadata.read_path_metadata([last, first])
                self.assertEqual(result["total_cycles"], 2)
                self.assertEqual(result["total_lines"], 6)
                self.assertEqual(result["chunk_count"], 2)

    def test_real_parser_reads_all_chunks_without_trial_token(self):
        paths = []
        for offset, cycles in [(0, 2), (2, 3)]:
            original = self.write_dat(offset=offset, cycles=cycles)
            renamed = original.with_name(original.name.replace("-TRIAL000001", ""))
            original.rename(renamed)
            paths.append(renamed)
        with patch("slap2_utils.utils.file_header.load_file_header_v2", wraps=self.library_parser) as parser:
            result = metadata.read_path_metadata(paths[::-1])
        self.assertEqual(parser.call_count, 2)
        self.assertEqual(result["total_cycles"], 5)
        self.assertEqual(result["total_lines"], 15)

    def test_real_parser_rejects_gap_and_overlap_using_payload_counts(self):
        first = self.write_dat(offset=0, cycles=3)
        for offset, relation in [(2, "overlap"), (4, "gap")]:
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, f"cycle-chunk {relation}"):
                metadata.read_path_metadata([first, self.write_dat(offset=offset)])

    def test_real_parser_rejects_invalid_channel_metadata(self):
        path = self.write_dat()
        self.rewrite_word(path, 10, 2)
        with self.assertRaisesRegex(ValueError, "numChannels.*channelMask"):
            metadata.read_path_metadata([path])


if __name__ == "__main__":
    unittest.main()