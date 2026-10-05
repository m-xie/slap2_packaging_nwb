"""Round-trip the ragged frame schema and synchronization provenance."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import hdmf_zarr
import numpy as np
import pynwb

import random_natural_movies as movies
import run_capsule
import stimulus_sync
from tests.test_random_natural_movies import example_session


class MovieFramePackagingTests(unittest.TestCase):
    def test_ragged_columns_provenance_and_qc_round_trip(self):
        scenarios = {
            "mixed": ("movie", "movie", "gratings", "movie"),
            "gratings_only": ("gratings",),
            "movie_only_with_gap": ("movie", "movie"),
            "interrupted": ("movie", "movie", "gratings", "movie"),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario, kinds in scenarios.items():
                table, logger = example_session(kinds)
                if scenario == "interrupted":
                    logger = logger.loc[~logger.Value.isin(("END", "EndFrame"))]
                table_path, logger_path = root / "stim.csv", root / "logger.csv"
                table.to_csv(table_path, index=False)
                logger.to_csv(logger_path, index=False)
                frames = np.arange(0, logger.Frame.max() + 1, dtype=float)
                if scenario == "movie_only_with_gap":
                    frames = frames[(frames <= 10) | (frames >= 22)]
                elif scenario == "interrupted":
                    frames = np.arange(23, dtype=float)
                times = 0.5 + frames * 0.02
                qc = stimulus_sync.AlignmentQC(str(logger_path), len(frames), len(frames), len(frames), 50, 0, 0, 0)
                harp = {"normalized_slap2_start": np.array([0.0]), "normalized_slap2_end": np.array([]), "time_reference": 12345.0}
                for suffix, io_class in ((".nwb", pynwb.NWBHDF5IO), (".nwb.zarr", hdmf_zarr.NWBZarrIO)):
                    with self.subTest(scenario=scenario, storage=suffix):
                        # HDMF supplies external_resources dynamically; NWBFile
                        # is concrete at runtime despite the upstream type warning.
                        nwb = pynwb.NWBFile("test", scenario, datetime.now(timezone.utc))  # pyright: ignore[reportAbstractUsage]
                        plot_path = root / scenario / "photodiode_sync.png"
                        with patch.object(stimulus_sync, "align_logger_frames_to_harp", return_value=(frames, times, qc)) as align:
                            with patch.object(stimulus_sync, "extract_harp_photodiode_transitions", return_value=(times, np.zeros(len(times)), 0.5)):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", RuntimeWarning)
                                    metadata = run_capsule.add_stim_table(
                                        nwb, table_path, logger_path, harp, movies.LOGGER_FORMAT,
                                        photodiode_qc_path=plot_path,
                                    )
                        align.assert_called_once()
                        self.assertEqual(plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                        path = root / (scenario + suffix)
                        with io_class(str(path), "w") as io:
                            io.write(nwb)
                        if io_class is pynwb.NWBHDF5IO:
                            self.assertEqual(pynwb.validate(path=path), [])
                        with io_class(str(path), "r") as io:
                            result = io.read()
                            intervals = result.intervals["stimulus_blocks"]
                            blocks = intervals.to_dataframe()
                            self.assertNotIn("stimulus_synchronization", result.processing)
                            self.assertNotIn("anchor_frames", intervals.colnames)
                            self.assertNotIn("anchor_times", intervals.colnames)
                            # Summaries remain JSON-serializable for external
                            # processing provenance; anchor arrays are not saved.
                            provenance = json.loads(json.dumps(metadata["movie_frame_timing"]))
                            self.assertNotIn("anchor_frames", provenance)
                            self.assertNotIn("anchor_times", provenance)
                            self.assertEqual(provenance["harp_time_reference_seconds"], 12345.0)
                            self.assertNotIn("startup_handling", provenance)
                            for column, (dtype, _) in movies.MOVIE_FRAME_COLUMNS.items():
                                self.assertEqual(intervals[column].target.data.dtype, np.dtype(dtype))
                                self.assertIn("Empty for non-movie rows", intervals[column].target.description)
                                for value in blocks[column]:
                                    self.assertEqual(np.asarray(value).ndim, 1)
                            for row in blocks.itertuples():
                                count = row.movie_frame_count
                                for column in movies.MOVIE_FRAME_COLUMNS:
                                    self.assertEqual(len(getattr(row, column)), count)
                                np.testing.assert_array_equal(row.movie_frame_numbers, np.arange(1, count + 1))
                                finite = np.isfinite(row.movie_frame_timestamps)
                                np.testing.assert_array_equal(~finite, row.movie_frame_timing_status == 3)
                                np.testing.assert_allclose(
                                    row.movie_frame_timestamps[finite], 0.5 + row.movie_display_frames[finite] * 0.02,
                                )
                                self.assertTrue(np.all(np.diff(row.movie_frame_timestamps[finite]) > 0))
                                self.assertTrue(np.all(row.movie_frame_timestamps[finite] >= row.start_time))
                                self.assertTrue(np.all(row.movie_frame_timestamps[finite] <= row.stop_time))
                            if scenario == "movie_only_with_gap":
                                np.testing.assert_array_equal(blocks.iloc[0].movie_frame_timing_status, [0, 3, 3, 3])
                                self.assertTrue(np.isnan(blocks.iloc[0].movie_frame_timestamps[1:]).all())
                            elif scenario == "interrupted":
                                self.assertEqual(len(blocks), 2)
                                self.assertTrue(np.isnan(blocks.iloc[1].movie_frame_timestamps[-1]))
                                self.assertEqual(provenance["omitted_frame_count"], 4)
                            elif scenario == "gratings_only":
                                self.assertEqual(provenance["stored_frame_count"], 0)
                            if "gratings" in result.intervals:
                                for column in movies.MOVIE_FRAME_COLUMNS:
                                    self.assertNotIn(column, result.intervals["gratings"].colnames)

    def test_ragged_column_names_are_reserved(self):
        table, logger = example_session()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "logger.csv"
            logger.to_csv(path, index=False)
            for column in movies.MOVIE_FRAME_COLUMNS:
                with self.subTest(column=column):
                    source = table.assign(**{column: "input_collision"})
                    with self.assertRaisesRegex(ValueError, "reserved output columns"):
                        movies.read_presentation_frames(source, path)


if __name__ == "__main__":
    unittest.main()