"""Playback parsing, NWB round-trip, and legacy isolation regression tests."""

from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pynwb
import hdmf_zarr

import random_natural_movies as movies
import run_capsule
import stimulus_sync


def example_acquisition():
    return {"stimulus_epochs": [{"code": {"parameters": {"StimulusParameters": {
        "GratingSpatialFrequency": [0.08], "GratingSpatialFrequencyUnit": "cycle/degree",
        "GratingTemporalFrequency": [3.0], "GratingTemporalFrequencyUnit": "Hz",
        "GratingDiameter": [270.0], "GratingDiameterUnit": "degree",
        "GratingX": [5.0], "GratingXUnit": "degree",
        "GratingY": [-2.0], "GratingYUnit": "degree",
        "GratingContrast": [0.0, 0.75],
        "GratingDuration": [99.0], "GratingDelay": [88.0],
    }}}}]}


def example_session(types=("movie", "movie", "gratings", "gratings", "movie")):
    events = [(0, "STARTSLAP")]
    rows = []
    frame = 10
    for index, kind in enumerate(types):
        rows.append({
            "TextureName": f"movie_{index}" if kind == "movie" else "gratings",
            "TrialType": kind,
            # Deliberately unrelated to actual playback durations.
            "TrialDuration": 9999,
            "ExtentX": 180.0,
            "ExtentY": 100.0,
        })
        if kind == "movie":
            for counter in range(1, 5):
                events.append((frame, f"MovieFrame-{counter}"))
                frame += 2
        else:
            frame += 2
            for angle in (315, 359, 225, 90, 135, 270, 0, 45, 180):
                events.append((frame, f"GratingStart-{angle}"))
                events.append((frame + 4, f"GratingEnd-{angle}"))
                frame += 6
            frame += 2
    events.extend([(frame, "END"), (frame, "EndFrame")])
    for display_frame in range(frame + 1):
        events.append((display_frame, "Frame"))
        events.append((display_frame, f"Photodiode-{(display_frame // 3) % 2}"))
    events.sort(key=lambda event: event[0])
    logger = pd.DataFrame(
        [(frame, 1000 + frame * 42, value) for frame, value in events],
        columns=["Frame", "Timestamp", "Value"],
    )
    return pd.DataFrame(rows), logger


class RandomNaturalMoviesTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "logger.csv"
        self.table, self.logger = example_session()

    def read_frames(self):
        self.logger.to_csv(self.path, index=False)
        return movies.read_presentation_frames(self.table, self.path)

    def test_shared_movie_display_frames_flag_every_affected_counter(self):
        for counters, expected in (((2, 3), [0, 1, 1, 0]),
                                   ((1, 2, 3), [1, 1, 1, 0]),
                                   ((1, 2, 3, 4), [1, 1, 1, 1])):
            with self.subTest(counters=counters):
                self.table, self.logger = example_session(("movie", "gratings"))
                mask = self.logger.Value.isin([f"MovieFrame-{n}" for n in counters])
                self.logger.loc[mask, "Frame"] = self.logger.loc[mask, "Frame"].min()
                if counters == (1, 2, 3):
                    self.logger.loc[self.logger.Value.eq("MovieFrame-4"), "Frame"] = 12
                self.logger = self.logger.sort_values("Frame", kind="stable")
                with self.assertWarnsRegex(RuntimeWarning, "sharing display frames"):
                    blocks, _ = self.read_frames()
                np.testing.assert_array_equal(blocks.iloc[0].movie_frame_numbers, [1, 2, 3, 4])
                np.testing.assert_array_equal(blocks.iloc[0].movie_frame_playback_status, expected)
                self.assertEqual(blocks.iloc[1].movie_frame_playback_status.size, 0)
                self.assertTrue(any("sharing display frames" in w for w in blocks.attrs["recovery_warnings"]))

    def test_grating_events_cannot_share_display_frames(self):
        self.table, self.logger = example_session(("gratings",))
        start = self.logger.loc[self.logger.Value.eq("GratingStart-315"), "Frame"].iloc[0]
        self.logger.loc[self.logger.Value.eq("GratingEnd-315"), "Frame"] = start
        self.logger = self.logger.sort_values("Frame", kind="stable")
        with self.assertRaisesRegex(ValueError, "Only consecutive movie counters"):
            self.read_frames()

    def test_preserves_rows_and_separates_adjacent_grating_blocks(self):
        blocks, gratings = self.read_frames()
        pd.testing.assert_frame_equal(blocks[self.table.columns], self.table)
        self.assertEqual(len(blocks), 5)
        self.assertEqual(len(gratings), 18)
        self.assertEqual(gratings.groupby("stimulus_table_row").size().to_dict(), {2: 9, 3: 9})
        self.assertEqual(gratings.is_blank.sum(), 2)
        self.assertTrue(gratings.loc[gratings.is_blank, "Orientation"].isna().all())
        self.assertTrue(gratings.loc[gratings.is_blank, "logger_orientation"].eq(359).all())
        self.assertEqual(blocks.loc[2, "start_frame"], gratings.iloc[0].start_frame)
        self.assertEqual(blocks.loc[2, "stop_frame"], gratings.iloc[8].stop_frame)

    def test_movie_boundaries_follow_playback_not_trial_duration_or_timestamp(self):
        blocks, _ = self.read_frames()
        self.assertEqual(blocks.loc[0, "stop_frame"], blocks.loc[1, "start_frame"])
        self.assertEqual(blocks.loc[0, "stop_frame_source"], "next_movie_first_frame")
        self.assertEqual(blocks.loc[1, "stop_frame"] - blocks.loc[1, "start_frame"], 8)
        self.assertEqual(blocks.loc[1, "stop_frame_source"], "last_movie_frame_plus_observed_cadence")
        self.assertEqual(blocks.iloc[-1].stop_frame_source, "session_end")
        self.assertEqual(blocks.iloc[-1].stop_frame, self.logger.Frame.max())
        self.table["TrialDuration"] = 0.001
        self.logger["Timestamp"] = -10
        changed, _ = self.read_frames()
        pd.testing.assert_frame_equal(
            blocks[["start_frame", "stop_frame"]], changed[["start_frame", "stop_frame"]],
        )

    def test_movie_urls_match_all_supplied_assets(self):
        filenames = [
            "Natural_Movie_TOE_1.mp4", "Natural_Movie_TOE_2.mp4",
            "Natural_Movie_TOE_3.mp4", "natural_movie_TOE_1_shuffle.mp4",
            "natural_movie_TOE_2_shuffle.mp4", "natural_movie_TOE_3_shuffle.mp4",
            "zebra_allen_screen_tscale_30_scale_10.mp4",
        ]
        base = (
            "https://github.com/AllenNeuralDynamics/ophys-passive-visual-stim/blob/"
            "37c9c03611f6e16b285dac6aae589f6a285e78ad/src/Movies/"
        )
        self.table, self.logger = example_session(("movie",) * 8 + ("gratings",))
        self.table["TextureName"] = [
            filename.removesuffix(".mp4").lower() for filename in filenames
        ] + ["unknown_movie", "Natural_Movie_TOE_1"]
        original = self.table.copy(deep=True)
        blocks, gratings = self.read_frames()
        self.assertEqual(blocks.movie_url.tolist(), [base + name for name in filenames] + ["", ""])
        self.assertNotIn("movie_url", gratings)
        pd.testing.assert_frame_equal(self.table, original)

    def test_movie_url_is_reserved_output_column(self):
        self.table["movie_url"] = "untrusted_input"
        with self.assertRaisesRegex(ValueError, "reserved output columns"):
            self.read_frames()

    def test_missing_movie_counter_is_rejected(self):
        self.logger = self.logger.drop(self.logger.index[self.logger.Value.eq("MovieFrame-3")][0])
        with self.assertRaisesRegex(ValueError, "Missing or duplicate MovieFrame"):
            self.read_frames()

    def test_truncated_and_mismatched_grating_pairs_are_rejected(self):
        for replacement in ("MovieFrame-1", "GratingEnd-0", "GratingEnd-bad"):
            with self.subTest(replacement=replacement):
                self.table, self.logger = example_session()
                index = self.logger.index[self.logger.Value.eq("GratingEnd-315")][0]
                self.logger.loc[index, "Value"] = replacement
                with self.assertRaises(ValueError):
                    self.read_frames()

    def test_duplicate_direction_cannot_replace_blank(self):
        self.logger["Value"] = self.logger.Value.replace({
            "GratingStart-359": "GratingStart-315", "GratingEnd-359": "GratingEnd-315",
        })
        with self.assertRaisesRegex(ValueError, "eight directions and one blank"):
            self.read_frames()

    def test_table_order_and_counts_are_validated(self):
        for indices in ([1, 2, 0, 3, 4], [0, 1, 2, 3]):
            with self.subTest(indices=indices):
                self.table, self.logger = example_session()
                self.table = self.table.iloc[indices].reset_index(drop=True)
                with self.assertRaises(ValueError):
                    self.read_frames()

    def test_interrupted_last_repeat_is_retained_with_warning(self):
        self.table.loc[4, "TextureName"] = self.table.loc[0, "TextureName"]
        self.logger = self.logger.drop(self.logger.index[self.logger.Value.eq("MovieFrame-4")][-1])
        with self.assertWarnsRegex(RuntimeWarning, "Inconsistent playback frame counts"):
            blocks, _ = self.read_frames()
        self.assertTrue(blocks.iloc[-1].is_partial)
        self.assertEqual(blocks.iloc[-1].movie_frame_count, 3)

    def test_either_end_marker_is_sufficient(self):
        for absent in ("END", "EndFrame"):
            with self.subTest(absent=absent):
                self.table, self.logger = example_session()
                self.logger = self.logger.loc[~self.logger.Value.eq(absent)]
                blocks, _ = self.read_frames()
                self.assertEqual(blocks.iloc[-1].stop_frame_source, "session_end")
                self.assertFalse(blocks.iloc[-1].is_partial)

    def test_no_end_markers_retains_censored_movie(self):
        self.logger = self.logger.loc[~self.logger.Value.isin(("END", "EndFrame"))]
        with self.assertWarnsRegex(RuntimeWarning, "no END or EndFrame"):
            blocks, _ = self.read_frames()
        self.assertEqual(len(blocks), 5)
        self.assertTrue(blocks.iloc[-1].is_partial)
        self.assertEqual(blocks.iloc[-1].stop_frame_source, "logger_tail_censored")
        self.assertLessEqual(blocks.iloc[-1].stop_frame, self.logger.Frame.max())

    def test_unobserved_table_suffix_is_omitted_with_warning(self):
        self.table = pd.concat([self.table, self.table.iloc[[-1]]], ignore_index=True)
        with self.assertWarnsRegex(RuntimeWarning, "Logger ended before stimulus table row 5"):
            blocks, gratings = self.read_frames()
        self.assertEqual(len(blocks), 5)
        self.assertEqual(len(gratings), 18)
        self.assertEqual(blocks.stimulus_table_row.tolist(), list(range(5)))

    def test_cutoff_inside_movie_keeps_observed_frames(self):
        cutoff = self.logger.loc[self.logger.Value.eq("MovieFrame-3"), "Frame"].iloc[1]
        self.logger = self.logger.loc[self.logger.Frame <= cutoff]
        with self.assertWarns(RuntimeWarning):
            blocks, gratings = self.read_frames()
        self.assertEqual(len(blocks), 2)
        self.assertTrue(gratings.empty)
        self.assertTrue(blocks.iloc[-1].is_partial)
        self.assertEqual(blocks.iloc[-1].movie_frame_count, 3)
        self.assertEqual(blocks.iloc[-1].stop_frame, cutoff)

    def test_cutoff_inside_grating_preserves_pairs_and_censored_start(self):
        cutoff = self.logger.loc[self.logger.Value.eq("GratingStart-225"), "Frame"].iloc[0] + 1
        self.logger = self.logger.loc[self.logger.Frame <= cutoff]
        with self.assertWarns(RuntimeWarning):
            blocks, gratings = self.read_frames()
        self.assertEqual(len(blocks), 3)
        self.assertEqual(len(gratings), 3)
        self.assertEqual(gratings.is_partial.tolist(), [False, False, True])
        self.assertTrue(blocks.iloc[-1].is_partial)
        self.assertEqual(gratings.iloc[-1].stop_frame, cutoff)

    def test_cutoff_after_grating_end_preserves_completed_pairs(self):
        cutoff = self.logger.loc[self.logger.Value.eq("GratingEnd-359"), "Frame"].iloc[0]
        self.logger = self.logger.loc[self.logger.Frame <= cutoff]
        with self.assertWarnsRegex(RuntimeWarning, "Incomplete grating block"):
            blocks, gratings = self.read_frames()
        self.assertEqual(len(gratings), 2)
        self.assertFalse(gratings.is_partial.any())
        self.assertTrue(blocks.iloc[-1].is_partial)
        self.assertEqual(blocks.iloc[-1].stop_frame_source, "last_grating_end")

    def test_single_movie_frame_is_retained_with_warning(self):
        self.table, self.logger = example_session(("movie", "gratings"))
        self.logger = self.logger.loc[~self.logger.Value.isin(("MovieFrame-2", "MovieFrame-3", "MovieFrame-4"))]
        with self.assertWarnsRegex(RuntimeWarning, "only one frame"):
            blocks, _ = self.read_frames()
        self.assertEqual(blocks.iloc[0].movie_frame_count, 1)
        self.assertTrue(blocks.iloc[0].is_partial)
        self.assertEqual(blocks.iloc[0].stop_frame, blocks.iloc[0].start_frame + 1)

    @patch("stimulus_sync.align_logger_frames_to_harp")
    @patch("stimulus_sync.extract_harp_photodiode_transitions", return_value=([], [], 0.5))
    def test_single_terminal_frame_preserves_onset_without_inventing_duration(self, extract, align):
        self.table, self.logger = example_session(("movie",))
        self.logger = self.logger.loc[self.logger.Frame <= 10]
        self.logger.to_csv(self.path, index=False)
        frames = np.arange(11, dtype=float)
        qc = stimulus_sync.AlignmentQC(str(self.path), 10, 10, 10, 60, 0, 0, 0)
        align.return_value = (frames, frames / 60, qc)
        with self.assertWarnsRegex(RuntimeWarning, "only one frame"):
            blocks, _, metadata = movies.synchronize_presentations(self.table, self.path, {})
        self.assertEqual(blocks.iloc[0].Duration, 0)
        self.assertTrue(blocks.iloc[0].is_partial)
        self.assertEqual(metadata["partial_stimulus_table_row_count"], 1)
        table_path = self.root / "single_frame.csv"
        self.table.to_csv(table_path, index=False)
        for name, io_class in (("single.nwb", pynwb.NWBHDF5IO), ("single.nwb.zarr", hdmf_zarr.NWBZarrIO)):
            with self.subTest(storage=name):
                nwb = pynwb.NWBFile("interrupted", name, datetime.now(timezone.utc))
                with self.assertWarns(RuntimeWarning):
                    run_capsule.add_stim_table(
                        nwb, table_path, self.path,
                        {"normalized_slap2_start": np.array([0.0]), "normalized_slap2_end": np.array([])},
                        movies.LOGGER_FORMAT,
                    )
                path = self.root / name
                with io_class(str(path), "w") as io:
                    io.write(nwb)
                with io_class(str(path), "r") as io:
                    recovered = io.read().intervals["stimulus_blocks"].to_dataframe()
                    self.assertEqual(len(recovered), 1)
                    self.assertEqual(recovered.iloc[0].Duration, 0)
                    self.assertTrue(recovered.iloc[0].is_partial)
                if io_class is pynwb.NWBHDF5IO:
                    self.assertEqual(pynwb.validate(path=path), [])

    @patch("stimulus_sync.align_logger_frames_to_harp")
    @patch("stimulus_sync.extract_harp_photodiode_transitions", return_value=([], [], 0.5))
    def test_unanchored_interrupted_suffix_does_not_discard_supported_prefix(self, extract, align):
        self.logger = self.logger.loc[~self.logger.Value.isin(("END", "EndFrame"))]
        self.logger.to_csv(self.path, index=False)
        frames = np.arange(21, dtype=float)
        qc = stimulus_sync.AlignmentQC(str(self.path), 10, 10, 10, 60, 0, 0, 0)
        align.return_value = (frames, frames / 60, qc)
        with self.assertWarnsRegex(RuntimeWarning, "beyond photodiode coverage"):
            blocks, gratings, metadata = movies.synchronize_presentations(self.table, self.path, {})
        self.assertEqual(len(blocks), 2)
        self.assertTrue(gratings.empty)
        self.assertEqual(blocks.iloc[-1].stop_frame, 20)
        self.assertEqual(blocks.iloc[-1].stop_frame_source, "photodiode_tail_censored")
        self.assertEqual(metadata["omitted_stimulus_table_row_count"], 3)
        self.assertGreater(len(metadata["recovery_warnings"]), 0)

    def test_invalid_global_frame_is_rejected(self):
        self.logger["Frame"] = self.logger.Frame.astype(float)
        self.logger.loc[10, "Frame"] = 0.5
        with self.assertRaisesRegex(ValueError, "nondecreasing nonnegative integers"):
            self.read_frames()

    def test_movie_only_and_grating_only_tables(self):
        for kind in ("movie", "gratings"):
            with self.subTest(kind=kind):
                self.table, self.logger = example_session((kind,))
                blocks, gratings = self.read_frames()
                self.assertEqual(len(blocks), 1)
                self.assertEqual(len(gratings), 9 if kind == "gratings" else 0)

    @patch("stimulus_sync.align_logger_frames_to_harp", side_effect=ValueError("poor alignment"))
    @patch("stimulus_sync.extract_harp_photodiode_transitions", return_value=([], [], 0.5))
    def test_alignment_failure_does_not_use_do2(self, extract, align):
        self.read_frames()
        with self.assertRaisesRegex(ValueError, "poor alignment"):
            movies.synchronize_presentations(
                self.table, self.path, {"normalized_start_gratings": np.arange(len(self.table))},
            )

    def test_missing_logger_cannot_fall_back_to_do2(self):
        with self.assertRaisesRegex(ValueError, "DO2 alone is insufficient"):
            movies.synchronize_presentations(self.table, None, {"normalized_start_gratings": np.arange(5)})

    def test_legacy_parser_still_requires_stimstart(self):
        self.read_frames()
        with self.assertRaisesRegex(ValueError, "no StimStart"):
            stimulus_sync.extract_logger_events(self.path)

    def test_bounded_endpoint_extension_is_not_clamped(self):
        anchors = np.arange(0, 101, 5, dtype=float)
        times, extended, maximum = movies._map_boundary_frames(
            [1, 50, 101, 102], anchors, 0.1 + anchors * 0.02,
        )
        np.testing.assert_allclose(times, [0.12, 1.1, 2.12, 2.14])
        np.testing.assert_array_equal(extended, [False, False, True, True])
        self.assertEqual(maximum, 2)
        with self.assertRaisesRegex(ValueError, "do not cover playback boundaries"):
            movies._map_boundary_frames([103], anchors, anchors * 0.02)

    def test_grating_metadata_leaves_playback_and_input_unchanged(self):
        _, original = self.read_frames()
        saved = original.copy(deep=True)
        enriched, descriptions = movies.add_grating_parameters(original, example_acquisition())
        pd.testing.assert_frame_equal(original, saved)
        pd.testing.assert_frame_equal(enriched[original.columns], original)
        self.assertEqual(set(enriched) - set(original), set(descriptions))
        self.assertTrue(enriched.loc[enriched.is_blank, "Orientation"].isna().all())
        for acquisition in (None, {}):
            if acquisition is None:
                unchanged, descriptions = movies.add_grating_parameters(original, acquisition)
            else:
                with self.assertWarnsRegex(RuntimeWarning, "properties omitted"):
                    unchanged, descriptions = movies.add_grating_parameters(original, acquisition)
            pd.testing.assert_frame_equal(unchanged, original)
            self.assertEqual(descriptions, {})
        empty, descriptions = movies.add_grating_parameters(original.iloc[:0], {})
        self.assertTrue(empty.empty)
        self.assertEqual(descriptions, {})

    def test_ambiguous_or_invalid_grating_metadata_is_rejected(self):
        _, gratings = self.read_frames()
        for key, value, message in (
            ("GratingSpatialFrequency", [0.04, 0.08], "one constant value"),
            ("GratingTemporalFrequency", [float("nan")], "Invalid grating metadata"),
            ("GratingDiameter", None, "Missing grating metadata"),
            ("GratingXUnit", None, "Missing grating metadata unit"),
            ("GratingContrast", [0, 0.5, 1], "one nonblank contrast"),
            ("GratingContrast", [0.5], "zero for blanks"),
        ):
            with self.subTest(key=key, value=value):
                acquisition = example_acquisition()
                acquisition["stimulus_epochs"][0]["code"]["parameters"]["StimulusParameters"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    movies.add_grating_parameters(gratings, acquisition)
        acquisition = example_acquisition()
        acquisition["stimulus_epochs"].append(deepcopy(acquisition["stimulus_epochs"][0]))
        enriched, descriptions = movies.add_grating_parameters(gratings, acquisition)
        self.assertTrue(enriched.SpatialFrequency.eq(0.08).all())
        self.assertIn("stimulus_epochs[1]", descriptions["SpatialFrequency"])
        acquisition["stimulus_epochs"][1]["code"]["parameters"]["StimulusParameters"]["GratingX"] = [7]
        with self.assertRaisesRegex(ValueError, "Ambiguous grating parameters"):
            movies.add_grating_parameters(gratings, acquisition)

    @patch("stimulus_sync.align_logger_frames_to_harp")
    @patch("stimulus_sync.extract_harp_photodiode_transitions")
    def test_nwb_round_trip_and_provenance(self, extract, align):
        self.table.loc[0, "TextureName"] = "natural_movie_TOE_1"
        self.table.loc[1, "TextureName"] = "natural_movie_TOE_2_shuffle"
        self.read_frames()
        table_path = self.root / "stim_table.csv"
        self.table.to_csv(table_path, index=False)
        frames = np.arange(0, self.logger.Frame.max() + 1, dtype=float)
        qc = stimulus_sync.AlignmentQC(str(self.path), 100, 100, 100, 50, 0, 0, 0)
        extract.return_value = (frames, frames % 2 == 0, 0.5)
        align.return_value = (frames, 0.5 + frames * 0.02, qc)
        nwb = pynwb.NWBFile("test", "test", datetime.now(timezone.utc))
        harp_data = {"normalized_slap2_start": np.array([0.0]), "normalized_slap2_end": np.array([])}
        metadata = run_capsule.add_stim_table(
            nwb, table_path, self.path, harp_data, movies.LOGGER_FORMAT,
            acquisition_json=example_acquisition(),
        )
        align.assert_called_once()
        self.assertEqual(metadata["stimulus_table_row_count"], 5)
        self.assertEqual(metadata["grating_presentation_count"], 18)
        self.assertEqual(metadata["blank_presentation_count"], 2)
        self.assertEqual(metadata["stimulus_qc"], "random_natural_movies_activity_qc")
        self.assertEqual(metadata["extrapolated_boundary_count"], 0)
        for name, io_class in (("test.nwb", pynwb.NWBHDF5IO), ("test.nwb.zarr", hdmf_zarr.NWBZarrIO)):
            with self.subTest(storage=name):
                if io_class is hdmf_zarr.NWBZarrIO:
                    # HDMF binds a written container to its original source.
                    # Build a fresh file to test a second storage backend.
                    nwb = pynwb.NWBFile("test", "test-zarr", datetime.now(timezone.utc))
                    run_capsule.add_stim_table(
                        nwb, table_path, self.path, harp_data, movies.LOGGER_FORMAT,
                        acquisition_json=example_acquisition(),
                    )
                path = self.root / name
                with io_class(str(path), "w") as io:
                    io.write(nwb)
                if io_class is pynwb.NWBHDF5IO:
                    self.assertEqual(pynwb.validate(path=path), [])
                with io_class(str(path), "r") as io:
                    result = io.read()
                    blocks = result.intervals["stimulus_blocks"].to_dataframe()
                    gratings = result.intervals["gratings"].to_dataframe()
                    self.assertEqual(len(blocks), 5)
                    self.assertEqual(len(gratings), 18)
                    self.assertEqual(blocks.movie_url.tolist(), [
                        movies.MOVIE_URLS["natural_movie_toe_1"],
                        movies.MOVIE_URLS["natural_movie_toe_2_shuffle"],
                        "", "", "",
                    ])
                    self.assertNotIn("movie_url", gratings)
                    for column, value, unit in (
                        ("SpatialFrequency", 0.08, "cycle/degree"),
                        ("TemporalFrequency", 3.0, "Hz"),
                        ("DiameterX", 270.0, "degree"),
                        ("DiameterY", 270.0, "degree"),
                        ("X", 5.0, "degree"), ("Y", -2.0, "degree"),
                    ):
                        self.assertTrue(gratings[column].eq(value).all())
                        self.assertNotIn(column, blocks)
                        description = result.intervals["gratings"][column].description
                        self.assertIn(unit, description)
                        self.assertIn("acquisition.json stimulus_epochs[0]", description)
                    self.assertTrue(gratings.loc[gratings.is_blank, "Contrast"].eq(0).all())
                    self.assertTrue(gratings.loc[~gratings.is_blank, "Contrast"].eq(0.75).all())
                    self.assertTrue(gratings.loc[gratings.is_blank, "Orientation"].isna().all())
                    np.testing.assert_allclose(gratings.Duration, 0.08)
                    np.testing.assert_allclose(gratings.Duration, gratings.stop_time - gratings.start_time)
                    for column in ("GratingDuration", "GratingDelay", "NominalDuration", "NominalDelay", "Delay"):
                        self.assertNotIn(column, gratings)
                    self.assertTrue(blocks.TrialDuration.eq(9999).all())
                    np.testing.assert_allclose(blocks.start_time, 0.5 + blocks.start_frame * 0.02)
                    np.testing.assert_allclose(blocks.stop_time, 0.5 + blocks.stop_frame * 0.02)
                    self.assertTrue(blocks.slap2_trial_idx.eq(0).all())
                    for row in gratings.itertuples():
                        parent = blocks.loc[row.stimulus_table_row]
                        self.assertGreaterEqual(row.start_time, parent.start_time)
                        self.assertLessEqual(row.stop_time, parent.stop_time)

    @patch("stimulus_sync.resolve_stimulus_start_times")
    def test_legacy_interval_packaging_is_unchanged(self, resolve):
        resolve.return_value = (np.array([1.0, 3.0]), {"source": "harp_do2_fallback"})
        for format_name, column in ((None, None), ("Legacy Drifting Gratings", None), ("OpenScope P3", "Block_Type")):
            with self.subTest(format_name=format_name):
                table = pd.DataFrame({"Duration": [1.0, 2.0], "Orientation": [0.0, 90.0]})
                if column:
                    table[column] = ["movie", "standard_control"]
                path = self.root / "legacy.csv"
                table.to_csv(path, index=False)
                nwb = pynwb.NWBFile("test", "legacy", datetime.now(timezone.utc))
                harp_data = {"normalized_slap2_start": np.array([0.0]), "normalized_slap2_end": np.array([10.0])}
                metadata = run_capsule.add_stim_table(
                    nwb, path, None, harp_data, format_name, acquisition_json=example_acquisition(),
                )
                resolve.assert_called_with(2, harp_data, None)
                self.assertEqual(metadata, {"source": "harp_do2_fallback"})
                expected_names = {"movie", "standard_control"} if column else {"gratings"}
                self.assertEqual(set(nwb.intervals), expected_names)
                output = pd.concat([t.to_dataframe() for t in nwb.intervals.values()])
                self.assertNotIn("SpatialFrequency", output)
                np.testing.assert_allclose(output.start_time, [1.0, 3.0])
                np.testing.assert_allclose(output.stop_time, [2.0, 5.0])

    def test_discovery_fallback_only_applies_to_new_format(self):
        behavior = self.root / "behavior"
        behavior.mkdir()
        path = behavior / "stim_table.csv"
        path.touch()
        self.assertEqual(run_capsule.find_stimulus_table(self.root, "orientations_orientations0", movies.LOGGER_FORMAT), path)
        with self.assertRaises(StopIteration):
            run_capsule.find_stimulus_table(self.root, "orientations_orientations0", "OpenScope P3")
        with self.assertRaisesRegex(ValueError, "exactly one stimulus table"):
            run_capsule.find_stimulus_table(self.root, "explicit_missing", movies.LOGGER_FORMAT)
        (behavior / "stim_table_copy.csv").touch()
        with self.assertRaisesRegex(ValueError, "exactly one stimulus table"):
            run_capsule.find_stimulus_table(self.root, "orientations_orientations0", movies.LOGGER_FORMAT)

    @patch("run_capsule.stim_tuning_qc.compute_orientation_tuning_qc")
    @patch("run_capsule.stim_tuning_qc.compute_stim_tuning_qc")
    @patch("run_capsule.slap2_rf_qc.compute_receptive_field_qc")
    @patch("run_capsule.zebra_movie_qc.plot_zebra_repeats")
    def test_new_format_activity_qc_keeps_legacy_dispatch(self, zebra, rf, tuning, orientation):
        with patch.object(run_capsule.random_natural_movies_activity_qc, "compute_activity_qc") as activity:
            run_capsule.run_stimulus_qc("input", self.root, movies.LOGGER_FORMAT, 0.2,
                                      experiment_summary_path="summary.h5")
            activity.assert_called_once_with(self.root, "input", "summary.h5")
            for format_name in ("OpenScope P3", "Legacy Drifting Gratings"):
                run_capsule.run_stimulus_qc("input", self.root, format_name, 0.2)
            self.assertEqual(activity.call_count, 1)
        # Exactly the two legacy calls, never the Random Natural Movies call.
        for mock in (zebra, rf, tuning, orientation):
            self.assertEqual(mock.call_count, 2)
        zebra.assert_called_with("input", self.root / "zebra_movie")
        rf.assert_called_with(self.root, "input", onset_delay=0.2)
        tuning.assert_called_with(self.root, "input")
        orientation.assert_called_with(self.root, "input")
        self.assertEqual(zebra.call_count, 2)


ATTACHED_BEHAVIOR = Path(__file__).resolve().parents[2] / "data" / "878030_2026-08-31_14-17-19" / "behavior"


class AttachedRandomMoviesTests(unittest.TestCase):
    @unittest.skipUnless((ATTACHED_BEHAVIOR.parent / "acquisition.json").is_file(), "Session asset not attached")
    def test_attached_grating_acquisition_metadata(self):
        with (ATTACHED_BEHAVIOR.parent / "acquisition.json").open() as stream:
            acquisition = json.load(stream)
        gratings, _ = movies.add_grating_parameters(
            pd.DataFrame({"is_blank": [False, True]}), acquisition,
        )
        self.assertEqual(gratings.SpatialFrequency.tolist(), [0.04, 0.04])
        self.assertEqual(gratings.TemporalFrequency.tolist(), [2.0, 2.0])
        self.assertEqual(gratings.DiameterX.tolist(), [360.0, 360.0])
        self.assertEqual(gratings.DiameterY.tolist(), [360.0, 360.0])
        self.assertEqual(gratings.X.tolist(), [0.0, 0.0])
        self.assertEqual(gratings.Y.tolist(), [0.0, 0.0])
        self.assertEqual(gratings.Contrast.tolist(), [1.0, 0.0])

    @unittest.skipUnless((ATTACHED_BEHAVIOR / "bonvision_logger.csv").is_file(), "Session asset not attached")
    def test_attached_session_counts_and_sequence(self):
        source = pd.read_csv(ATTACHED_BEHAVIOR / "stim_table.csv")
        blocks, gratings = movies.read_presentation_frames(source, ATTACHED_BEHAVIOR / "bonvision_logger.csv")
        pd.testing.assert_frame_equal(source, blocks[source.columns])
        self.assertEqual(len(blocks), 55)
        self.assertEqual(blocks.TrialType.eq("movie").sum(), 35)
        self.assertEqual(len(gratings), 180)
        self.assertEqual(gratings.is_blank.sum(), 20)
        self.assertTrue(gratings.groupby("stimulus_table_row").size().eq(9).all())
        self.assertTrue((blocks.stop_frame.to_numpy()[:-1] <= blocks.start_frame.to_numpy()[1:]).all())


if __name__ == "__main__":
    unittest.main()