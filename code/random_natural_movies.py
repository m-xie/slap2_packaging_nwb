"""Opt-in adapter for the RandomNaturalMovies BonVision workflow.

MovieFrame counters identify content, while the global Frame column supplies
the display coordinate for HARP alignment. TrialDuration is metadata only.
This format has no movie-off or grating-block boundary events. Movie offsets
before gratings are explicitly cadence-derived; grating blocks are the envelope
of their nine logged presentations, not an estimate of unlogged blank periods.
Interrupted suffixes are recovered with warnings and explicit censoring flags.
"""

import warnings

import numpy as np
import pandas as pd

import stimulus_sync
from qc.random_natural_movies_sync_qc import plot_photodiode_sync


LOGGER_FORMAT = "Random Natural Movies"
GRATING_ORIENTATIONS = frozenset((0, 45, 90, 135, 180, 225, 270, 315, 359))
MAX_ENDPOINT_EXTRAPOLATION_FRAMES = 10
# Quality policy, not an accuracy guarantee: reject interpolation across more
# than three typical *logged* photodiode periods (not matched-anchor periods).
MAX_INTERPOLATION_GAP_FACTOR = 3.0
MOVIE_FRAME_STATUS = {"anchored": 0, "interpolated": 1, "extrapolated": 2, "unsupported": 3}
MOVIE_FRAME_COLUMNS = {
    "movie_frame_timestamps": (np.float64,
        "Estimated movie-frame onsets in seconds relative to the first recorded SLAP2 DO0 pulse, "
        "on the normalized HARP clock. Piecewise-linear photodiode alignment; NaN means "
        "unsupported. Not independent optical measurements of every movie frame."),
    "movie_frame_numbers": (np.int64,
        "Original 1-based MovieFrame-N logger counters, reset per presentation; "
        "not independently verified decoded MP4 frame indices."),
    "movie_display_frames": (np.int64,
        "Global logger Frame coordinates for MovieFrame events, not movie frame indices."),
    "movie_frame_timing_status": (np.uint8,
        "Per-frame timing quality: 0=photodiode anchor, 1=interpolated, "
        "2=bounded endpoint extrapolation, 3=unsupported (timestamp NaN). "
        "Unsupported includes large anchor gaps and frames beyond a censored block."),
    "movie_frame_playback_status": (np.uint8,
        "Per-event playback status: 0=no anomaly detected, 1=shared display frame. "
        "All content events sharing a display frame are flagged; which content was "
        "physically displayed is unknown. Shared timestamps describe logged events, "
        "not distinct measured visual onsets. Independent of timing quality."),
}
MOVIE_URL_BASE = (
    "https://github.com/AllenNeuralDynamics/ophys-passive-visual-stim/blob/"
    "37c9c03611f6e16b285dac6aae589f6a285e78ad/src/Movies/"
)
MOVIE_URLS = {
    filename.removesuffix(".mp4").lower(): MOVIE_URL_BASE + filename
    for filename in (
        "Natural_Movie_TOE_1.mp4",
        "Natural_Movie_TOE_2.mp4",
        "Natural_Movie_TOE_3.mp4",
        "natural_movie_TOE_1_shuffle.mp4",
        "natural_movie_TOE_2_shuffle.mp4",
        "natural_movie_TOE_3_shuffle.mp4",
        "zebra_allen_screen_tscale_30_scale_10.mp4",
    )
}


def add_grating_parameters(gratings, acquisition_json):
    """Add constant acquisition settings, never nominal duration or delay.

    DiameterX/Y follow the existing NWB grating column convention. Constants
    also describe the configured stimulus on blank trials (Contrast=0), not
    a visible grating. Missing metadata leaves older callers unchanged; varying
    settings are rejected because logger events cannot disambiguate them.
    """
    gratings = gratings.copy()
    if gratings.empty or acquisition_json is None:
        return gratings, {}
    fields = {
        "SpatialFrequency": "GratingSpatialFrequency",
        "TemporalFrequency": "GratingTemporalFrequency",
        "DiameterX": "GratingDiameter",
        "DiameterY": "GratingDiameter",
        "X": "GratingX",
        "Y": "GratingY",
    }
    keys = set(fields.values()) | {"GratingContrast"}
    keys |= {key + "Unit" for key in fields.values()}
    candidates = []
    for index, epoch in enumerate(acquisition_json.get("stimulus_epochs") or []):
        parameters = ((epoch.get("code") or {}).get("parameters") or {}).get("StimulusParameters") or {}
        if any(key in parameters for key in fields.values()):
            candidates.append((index, {key: parameters.get(key) for key in keys}))
    if not candidates:
        warnings.warn(
            "No grating stimulus parameters in acquisition metadata; grating properties omitted.",
            RuntimeWarning, stacklevel=2,
        )
        return gratings, {}
    parameters = candidates[0][1]
    if any(candidate != parameters for _, candidate in candidates[1:]):
        raise ValueError("Ambiguous grating parameters across stimulus epochs")
    sources = ", ".join(
        f"acquisition.json stimulus_epochs[{index}].code.parameters.StimulusParameters"
        for index, _ in candidates
    )

    def numeric_values(key):
        raw = parameters.get(key)
        if raw is None:
            raise ValueError(f"Missing grating metadata: {key}")
        try:
            values = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid grating metadata: {key}") from exc
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid grating metadata: {key}")
        return np.unique(values)

    descriptions = {}
    for column, key in fields.items():
        values = numeric_values(key)
        if len(values) != 1:
            raise ValueError(f"Expected one constant value for {key}")
        unit = parameters.get(key + "Unit")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Missing grating metadata unit: {key}Unit")
        gratings[column] = values[0]
        descriptions[column] = (
            f"{column} ({unit}), from {sources}.{key}; "
            "configured value also retained for blank presentations."
        )
    contrasts = numeric_values("GratingContrast")
    nonblank = contrasts[contrasts != 0]
    if len(nonblank) != 1 or (gratings["is_blank"].any() and 0 not in contrasts):
        raise ValueError("GratingContrast must specify one nonblank contrast and zero for blanks")
    gratings["Contrast"] = np.where(gratings["is_blank"], 0.0, nonblank[0])
    descriptions["Contrast"] = (
        f"Contrast (dimensionless), from {sources}.GratingContrast; "
        "zero for logger_orientation=359 blanks, otherwise the unique nonzero contrast."
    )
    return gratings, descriptions


def read_presentation_frames(stimulus_table, logger_path):
    """Return table-level and individual-grating intervals in display frames.

    Rows are matched sequentially, never using nominal durations. Each grating
    complete block contains the eight directions and blank sentinel (359).
    An interrupted suffix is retained with warnings and explicit censoring;
    malformed events or mismatches within the recorded sequence still fail.
    """
    required = {"TextureName", "TrialType", "TrialDuration", "ExtentX", "ExtentY"}
    missing = required.difference(stimulus_table.columns)
    if missing:
        raise ValueError(f"Random Natural Movies table is missing columns: {sorted(missing)}")
    if stimulus_table.empty:
        raise ValueError("Random Natural Movies stimulus table is empty")
    if stimulus_table[list(required)].isna().any().any():
        raise ValueError("Random Natural Movies table contains missing metadata")
    if not stimulus_table["TrialType"].isin(("movie", "gratings")).all():
        raise ValueError("Random Natural Movies TrialType must be movie or gratings")
    reserved = {
        "stimulus_table_row", "start_frame", "stop_frame", "stop_frame_source",
        "movie_frame_count", "start_time", "stop_time", "Duration", "slap2_trial_idx",
        "is_partial", "movie_url",
    } | set(MOVIE_FRAME_COLUMNS)
    if reserved.intersection(stimulus_table.columns):
        raise ValueError("Random Natural Movies table contains reserved output columns")
    if logger_path is None:
        raise ValueError("Random Natural Movies requires a playback logger; DO2 alone is insufficient")

    logger = pd.read_csv(logger_path)
    missing = {"Frame", "Timestamp", "Value"}.difference(logger.columns)
    if missing:
        raise ValueError(f"Random Natural Movies logger is missing columns: {sorted(missing)}")
    frames = pd.to_numeric(logger["Frame"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(frames).all() or np.any(frames < 0)
        or np.any(frames != np.floor(frames)) or np.any(np.diff(frames) < 0)
    ):
        raise ValueError("Logger Frame must contain nondecreasing nonnegative integers")
    logger["Frame"] = frames.astype(np.int64)
    values = logger["Value"].fillna("").astype(str)
    recovery_warnings = []

    def warn(message):
        recovery_warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    starts = logger.loc[values.eq("STARTSLAP"), "Frame"].to_numpy()
    ends = logger.loc[values.eq("END"), "Frame"].to_numpy()
    end_frames = logger.loc[values.eq("EndFrame"), "Frame"].to_numpy()
    if len(starts) != 1:
        raise ValueError("Random Natural Movies requires one STARTSLAP event")
    if len(ends) > 1 or len(end_frames) > 1:
        raise ValueError("Random Natural Movies allows at most one END and EndFrame event")
    if len(ends) and len(end_frames) and end_frames[0] != ends[0]:
        raise ValueError("Logger END and EndFrame disagree")
    session_end = float(ends[0]) if len(ends) else (float(end_frames[0]) if len(end_frames) else None)
    if session_end is not None and session_end <= starts[0]:
        raise ValueError("Session end must follow STARTSLAP")
    if session_end is None:
        warn("Logger has no END or EndFrame; recovering observed presentations with a censored final interval.")
    display_frames = logger.loc[values.eq("Frame"), "Frame"].to_numpy()
    if len(display_frames) < 2 or np.any(np.diff(display_frames) <= 0):
        raise ValueError("Logger display Frame events must strictly increase")
    observed_end = float(display_frames[-1])
    if session_end is not None:
        if session_end > observed_end:
            raise ValueError("Session end extends beyond the recorded display frames")
        observed_end = session_end
    if values.str.startswith("StimStart-").any():
        raise ValueError("Unexpected StimStart events in Random Natural Movies logger")

    relevant = values.str.startswith(("MovieFrame", "GratingStart", "GratingEnd"))
    payloads = values.loc[relevant].str.extract(r"^(MovieFrame|GratingStart|GratingEnd)-(\d+)$")
    if payloads.isna().any().any():
        raise ValueError("Malformed movie or grating event in logger")
    events = list(zip(
        payloads[0], payloads[1].astype(int), logger.loc[relevant, "Frame"],
    ))
    if not events:
        raise ValueError("Random Natural Movies logger has no presentation events")
    event_frames = np.asarray([event[2] for event in events])
    shared_frames = np.flatnonzero(np.diff(event_frames) == 0)
    if any(
        events[i][0] != "MovieFrame" or events[i + 1][0] != "MovieFrame"
        or events[i + 1][1] != events[i][1] + 1
        for i in shared_frames
    ):
        raise ValueError("Only consecutive movie counters may share a presentation frame")
    if (
        np.any(np.diff(event_frames) < 0)
        or event_frames[0] < starts[0] or event_frames[-1] > observed_end
    ):
        raise ValueError("Presentation frames must increase within STARTSLAP/END")

    blocks = stimulus_table.copy().reset_index(drop=True)
    block_rows = []
    gratings = []
    movie_counts = {}
    position = 0
    for row_id, row in blocks.iterrows():
        if position == len(events):
            warn(f"Logger ended before stimulus table row {row_id}; {len(blocks) - row_id} unobserved table rows omitted.")
            break
        first_frame = events[position][2]
        is_partial = False
        movie_frames = []
        movie_numbers = []
        playback_status = np.array([], dtype=np.uint8)
        if row["TrialType"] == "movie":
            if events[position][:2] != ("MovieFrame", 1):
                raise ValueError(f"Expected MovieFrame-1 for stimulus table row {row_id}")
            movie_frames = [first_frame]
            movie_numbers = [events[position][1]]
            position += 1
            while position < len(events):
                kind, counter, frame = events[position]
                if kind != "MovieFrame" or counter == 1:
                    break
                if counter != len(movie_frames) + 1:
                    raise ValueError(f"Missing or duplicate MovieFrame counter at table row {row_id}")
                movie_frames.append(frame)
                movie_numbers.append(counter)
                position += 1
            shared = np.diff(movie_frames) == 0
            playback_status = (np.r_[shared, False] | np.r_[False, shared]).astype(np.uint8)
            if playback_status.any():
                warn(
                    f"Movie at table row {row_id} has {int(playback_status.sum())} content events "
                    "sharing display frames; preserving counters and flagging ambiguous playback."
                )
            single_frame = len(set(movie_frames)) == 1
            if single_frame:
                warn(f"Movie at table row {row_id} has only one frame coordinate; playback cadence cannot be estimated.")
                is_partial = True
            # Flag shortened terminal repeats without using TrialDuration.
            # Interior multi-frame inconsistencies still indicate a sequence
            # problem, rather than a safely recoverable interrupted suffix.
            texture = row["TextureName"]
            previous_count = movie_counts.setdefault(texture, len(movie_frames))
            if previous_count != len(movie_frames):
                if position != len(events) and not single_frame and previous_count != 1:
                    raise ValueError(f"Inconsistent playback frame counts for {texture}")
                warn(f"Inconsistent playback frame counts for {texture} at table row {row_id}; retaining observed frames as partial.")
                is_partial = True
            positive_steps = np.diff(movie_frames)
            positive_steps = positive_steps[positive_steps > 0]
            cadence = float(np.median(positive_steps)) if positive_steps.size else None
            if position == len(events) and session_end is not None:
                stop_frame = session_end
                stop_source = "session_end"
            elif position < len(events) and events[position][0] == "MovieFrame":
                stop_frame = float(events[position][2])
                stop_source = "next_movie_first_frame"
            elif position == len(events):
                # No completion event: do not call the recording end a movie
                # offset. Preserve only the observed prefix (a lower bound).
                stop_frame = min(observed_end, movie_frames[-1] + cadence) if cadence else float(movie_frames[-1])
                stop_source = "logger_tail_censored"
                is_partial = True
                warn(f"Movie at table row {row_id} has no observed completion; retaining a censored playback interval.")
            elif single_frame:
                # The next recorded display tick proves a minimum observed
                # prefix, not the content frame's full display duration.
                later_frames = display_frames[display_frames > first_frame]
                stop_frame = float(later_frames[0]) if len(later_frames) else float(first_frame)
                stop_source = "single_movie_frame_censored"
            else:
                stop_frame = float(movie_frames[-1]) + cadence
                stop_source = "last_movie_frame_plus_observed_cadence"
            if stop_source in ("session_end", "next_movie_first_frame") and cadence is not None:
                if not 0 <= stop_frame - movie_frames[-1] <= 2 * cadence:
                    raise ValueError(f"Ambiguous movie playback offset at table row {row_id}")
            if position < len(events) and stop_frame > events[position][2]:
                raise ValueError(f"Movie playback overlaps gratings at table row {row_id}")
            frame_count = len(movie_frames)
        else:
            block_gratings = []
            for trial_index in range(len(GRATING_ORIENTATIONS)):
                if position == len(events):
                    is_partial = True
                    break
                kind, orientation, start = events[position]
                censored = position + 1 == len(events)
                if censored:
                    end_kind, end_orientation, stop = "GratingEnd", orientation, observed_end
                else:
                    end_kind, end_orientation, stop = events[position + 1]
                if (
                    kind != "GratingStart" or end_kind != "GratingEnd"
                    or orientation != end_orientation
                    or orientation not in GRATING_ORIENTATIONS
                ):
                    raise ValueError(f"Unpaired or invalid grating events at table row {row_id}")
                block_gratings.append({
                    "stimulus_table_row": row_id,
                    "grating_in_block": trial_index,
                    "start_frame": float(start),
                    "stop_frame": float(stop),
                    "logger_orientation": orientation,
                    "Orientation": np.nan if orientation == 359 else float(orientation),
                    "is_blank": orientation == 359,
                    "is_partial": censored,
                })
                position += 1 if censored else 2
                if censored:
                    is_partial = True
                    warn(f"Grating at table row {row_id} has no GratingEnd; retaining its observed prefix as censored.")
            directions = [trial["logger_orientation"] for trial in block_gratings]
            if len(set(directions)) != len(directions) or (not is_partial and set(directions) != GRATING_ORIENTATIONS):
                raise ValueError(f"Grating block must contain eight directions and one blank at row {row_id}")
            if is_partial:
                warn(f"Incomplete grating block at table row {row_id}; retaining {len(block_gratings)} observed presentations.")
            gratings.extend(block_gratings)
            stop_frame = block_gratings[-1]["stop_frame"]
            stop_source = "logger_tail_censored" if block_gratings[-1]["is_partial"] else "last_grating_end"
            frame_count = 0
        block_rows.append({
            "stimulus_table_row": row_id,
            "start_frame": float(first_frame),
            "stop_frame": stop_frame,
            "stop_frame_source": stop_source,
            "movie_frame_count": frame_count,
            "is_partial": is_partial,
            "movie_display_frames": np.asarray(movie_frames, dtype=np.int64),
            "movie_frame_numbers": np.asarray(movie_numbers, dtype=np.int64),
            "movie_frame_playback_status": playback_status,
        })
    if position != len(events):
        raise ValueError("Logger contains extra presentation events after the stimulus table")

    block_frames = pd.DataFrame(block_rows)
    for column in ("start_frame", "stop_frame"):
        if (
            block_frames[column].min() < display_frames[0]
            or block_frames[column].max() > display_frames[-1]
        ):
            raise ValueError("Playback intervals extend beyond the recorded display frames")
    blocks = pd.concat([blocks.iloc[:len(block_frames)], block_frames], axis=1)
    # Preserve exact URL filename casing while matching source texture names.
    # Empty strings keep non-movie and unknown entries serializable in NWB.
    blocks["movie_url"] = (
        blocks["TextureName"].str.lower().map(MOVIE_URLS).fillna("")
        .where(blocks["TrialType"].eq("movie"), "")
    )
    blocks.attrs["recovery_warnings"] = recovery_warnings
    blocks.attrs["has_session_end"] = session_end is not None
    grating_columns = [
        "stimulus_table_row", "grating_in_block", "start_frame", "stop_frame",
        "logger_orientation", "Orientation", "is_blank", "is_partial",
    ]
    return blocks, pd.DataFrame(gratings, columns=grating_columns)


def _map_boundary_frames(frames, anchor_frames, anchor_times):
    """Interpolate boundaries, with at most MAX_ENDPOINT_EXTRAPOLATION_FRAMES
    display ticks of extrapolation.

    END may follow the final physical photodiode edge by one display tick.
    Never silently clamp it (np.interp's default) or extrapolate an absent tail.
    The small, explicitly reported extension uses the local measured slope.
    """
    frames = np.asarray(frames, dtype=float)
    distances = np.maximum(anchor_frames[0] - frames, frames - anchor_frames[-1])
    if np.any(distances > MAX_ENDPOINT_EXTRAPOLATION_FRAMES):
        raise ValueError(
            "Photodiode anchors do not cover playback boundaries "
            f"(beyond {MAX_ENDPOINT_EXTRAPOLATION_FRAMES} display frames)"
            )
    times = np.interp(frames, anchor_frames, anchor_times)
    for outside, subset, endpoint in (
        (frames < anchor_frames[0], slice(0, 20), 0),
        (frames > anchor_frames[-1], slice(-20, None), -1),
    ):
        if np.any(outside):
            slope = np.polyfit(anchor_frames[subset], anchor_times[subset], 1)[0]
            if not np.isfinite(slope) or slope <= 0:
                raise ValueError("Invalid photodiode endpoint slope")
            times[outside] = anchor_times[endpoint] + slope * (frames[outside] - anchor_frames[endpoint])
    return times, distances > 0, max(0.0, float(distances.max(initial=0)))


def _validate_frame_anchors(anchor_frames, anchor_times):
    """Reject invalid maps before interpolation can hide nonmonotonic timing."""
    anchor_frames = np.asarray(anchor_frames, dtype=float)
    anchor_times = np.asarray(anchor_times, dtype=float)
    if (
        anchor_frames.ndim != 1 or anchor_times.shape != anchor_frames.shape
        or len(anchor_frames) < 2
        or not np.isfinite(anchor_frames).all() or not np.isfinite(anchor_times).all()
        or np.any(np.diff(anchor_frames) <= 0) or np.any(np.diff(anchor_times) <= 0)
    ):
        raise ValueError("Photodiode anchors must be finite, strictly increasing paired arrays")
    return anchor_frames, anchor_times


def map_movie_frames(frames, anchor_frames, anchor_times, maximum_gap_frames):
    """Map logged events without fabricating timestamps outside usable coverage.

    Exact anchors remain supported even at the edge of a large gap. Endpoint
    extension uses the same fit as interval boundaries, limited to two ticks.
    """
    anchor_frames, anchor_times = _validate_frame_anchors(anchor_frames, anchor_times)
    if not np.isfinite(maximum_gap_frames) or maximum_gap_frames <= 0:
        raise ValueError("maximum_gap_frames must be finite and positive")
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 1 or not np.isfinite(frames).all() or np.any(np.diff(frames) < 0):
        raise ValueError("Movie display frames must be finite and nondecreasing")
    times = np.full(len(frames), np.nan, dtype=np.float64)
    status = np.full(len(frames), MOVIE_FRAME_STATUS["unsupported"], dtype=np.uint8)
    right = np.searchsorted(anchor_frames, frames)
    clipped_right = np.minimum(right, len(anchor_frames) - 1)
    anchored = frames == anchor_frames[clipped_right]
    interior = (right > 0) & (right < len(anchor_frames)) & ~anchored
    gaps = anchor_frames[clipped_right] - anchor_frames[np.maximum(right - 1, 0)]
    interpolated = interior & (gaps <= maximum_gap_frames)
    distance = np.maximum(anchor_frames[0] - frames, frames - anchor_frames[-1])
    extrapolated = (distance > 0) & (distance <= MAX_ENDPOINT_EXTRAPOLATION_FRAMES)
    supported = anchored | interpolated | extrapolated
    times[supported], _, _ = _map_boundary_frames(frames[supported], anchor_frames, anchor_times)
    status[anchored] = MOVIE_FRAME_STATUS["anchored"]
    status[interpolated] = MOVIE_FRAME_STATUS["interpolated"]
    status[extrapolated] = MOVIE_FRAME_STATUS["extrapolated"]
    return times, status


def synchronize_presentations(stimulus_table, logger_path, harp_data, maximum_interpolation_gap_frames=None, *, photodiode_qc_path=None):
    """Align all playback boundaries once; DO2 cannot replace missing offsets."""
    blocks, gratings = read_presentation_frames(stimulus_table, logger_path)
    recovery_warnings = list(blocks.attrs["recovery_warnings"])
    has_session_end = blocks.attrs["has_session_end"]
    logger_data = stimulus_sync.extract_logger_events(
        logger_path, stimulus_frames=blocks["start_frame"].to_numpy(),
        initial_low_baseline=True,
        terminal_low_after_end_frame=True,
    )
    harp_times, harp_states, _ = stimulus_sync.extract_harp_photodiode_transitions(harp_data)
    # Keep raw-edge diagnostics available even when strict pairing rejects a
    # count/polarity mismatch. This plot never selects synchronization anchors.
    if photodiode_qc_path is not None:
        plot_photodiode_sync(logger_data, harp_times, photodiode_qc_path)
    anchor_frames, anchor_times, qc = stimulus_sync.align_logger_frames_to_harp(
        logger_data, harp_times, harp_states,
    )
    anchor_frames, anchor_times = _validate_frame_anchors(anchor_frames, anchor_times)
    logged_frame_count = int(blocks["movie_frame_count"].sum())
    if maximum_interpolation_gap_frames is None:
        maximum_interpolation_gap_frames = MAX_INTERPOLATION_GAP_FACTOR * float(
            np.median(np.diff(logger_data.transition_frames))
        )
    if not np.isfinite(maximum_interpolation_gap_frames) or maximum_interpolation_gap_frames <= 0:
        raise ValueError("maximum_interpolation_gap_frames must be finite and positive")
    # Recover the supported portion of an interrupted recording even when
    # its final photodiode state has no closing edge. Do not extend through
    # an unanchored tail or discard earlier well-aligned presentations.
    if not has_session_end or blocks["is_partial"].any() or len(blocks) < len(stimulus_table):
        for table in (blocks, gratings):
            unsupported = table["stop_frame"] > anchor_frames[-1] + MAX_ENDPOINT_EXTRAPOLATION_FRAMES
            if unsupported.any():
                message = (
                    f"Interrupted session: {int(unsupported.sum())} intervals extend beyond photodiode coverage; "
                    "clipping observed prefixes to the last anchor and omitting starts after it."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                recovery_warnings.append(message)
                table.loc[unsupported, "stop_frame"] = anchor_frames[-1]
                table.loc[unsupported, "is_partial"] = True
                if "stop_frame_source" in table:
                    table.loc[unsupported, "stop_frame_source"] = "photodiode_tail_censored"
                table.drop(index=table.index[table["start_frame"] > table["stop_frame"]], inplace=True)
        gratings = gratings.loc[gratings["stimulus_table_row"].isin(blocks["stimulus_table_row"])].copy()
    if blocks.empty:
        raise ValueError("No observed presentations have usable photodiode coverage")
    extrapolated_count = 0
    maximum_extrapolation = 0.0
    for table in (blocks, gratings):
        for frame_column, time_column in (("start_frame", "start_time"), ("stop_frame", "stop_time")):
            times, extrapolated, distance = _map_boundary_frames(
                table[frame_column].to_numpy(), anchor_frames, anchor_times,
            )
            table[time_column] = times
            extrapolated_count += int(extrapolated.sum())
            maximum_extrapolation = max(maximum_extrapolation, distance)
        table["Duration"] = table["stop_time"] - table["start_time"]
        if (
            not np.isfinite(table["Duration"]).all()
            or (table["Duration"] < 0).any()
            or ((table["Duration"] == 0) & ~table["is_partial"]).any()
        ):
            raise ValueError("Aligned intervals must have positive durations, except zero-length censored observations")
    timestamp_rows = []
    status_rows = []
    for row in blocks.itertuples():
        times, status = map_movie_frames(
            row.movie_display_frames, anchor_frames, anchor_times,
            maximum_interpolation_gap_frames,
        )
        # Preserve every event in retained blocks, even after an offset was
        # clipped. Unsupported events must not acquire a time beyond that block.
        outside_block = (row.movie_display_frames < row.start_frame) | (row.movie_display_frames > row.stop_frame)
        times[outside_block] = np.nan
        status[outside_block] = MOVIE_FRAME_STATUS["unsupported"]
        timestamp_rows.append(times)
        status_rows.append(status)
    blocks["movie_frame_timestamps"] = timestamp_rows
    blocks["movie_frame_timing_status"] = status_rows
    all_status = np.concatenate(status_rows)
    status_counts = {name: int(np.count_nonzero(all_status == code)) for name, code in MOVIE_FRAME_STATUS.items()}
    if status_counts["unsupported"]:
        message = f"{status_counts['unsupported']} logged movie frames have unsupported timing; retaining counters with NaN timestamps."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        recovery_warnings.append(message)
    frame_metadata = {
        "clock_reference": "seconds relative to first recorded SLAP2 DO0 pulse (normalized HARP)",
        "mapping_method": "piecewise_linear_matched_photodiode_anchors",
        "frame_identity": "original 1-based MovieFrame-N logger counter; decoded MP4 index unverified",
        "accuracy_note": "Aligned onset estimates, not per-frame optical measurements; descriptive timing residuals are not timing uncertainty and do not select anchors.",
        "maximum_endpoint_extrapolation_frames": MAX_ENDPOINT_EXTRAPOLATION_FRAMES,
        "maximum_interpolation_gap_frames": float(maximum_interpolation_gap_frames),
        "default_gap_policy": "3 times median logger photodiode transition spacing",
        "median_anchor_gap_frames": float(np.median(np.diff(anchor_frames))),
        "maximum_anchor_gap_frames": float(np.max(np.diff(anchor_frames))),
        "maximum_anchor_gap_seconds": float(np.max(np.diff(anchor_times))),
        "large_anchor_gap_count": int(np.count_nonzero(np.diff(anchor_frames) > maximum_interpolation_gap_frames)),
        "status_codes": MOVIE_FRAME_STATUS.copy(),
        "status_counts": status_counts,
        "logged_frame_count": logged_frame_count,
        "stored_frame_count": len(all_status),
        "omitted_frame_count": logged_frame_count - len(all_status),
        "omission_policy": "Existing interval recovery omits whole blocks starting beyond usable coverage; all events in retained blocks are preserved.",
        "alignment_qc": qc.__dict__.copy(),
    }
    if "time_reference" in harp_data:
        frame_metadata["harp_time_reference_seconds"] = float(harp_data["time_reference"])
    blocks.attrs["movie_frame_alignment"] = {
        "anchor_frames": anchor_frames, "anchor_times": anchor_times,
        "metadata": frame_metadata,
    }
    metadata = {
        "source": "logger_photodiode_aligned",
        "logger_format": LOGGER_FORMAT,
        **qc.__dict__,
        "stimulus_table_row_count": len(blocks),
        "input_stimulus_table_row_count": len(stimulus_table),
        "omitted_stimulus_table_row_count": len(stimulus_table) - len(blocks),
        "partial_stimulus_table_row_count": int(blocks["is_partial"].sum()),
        "partial_grating_presentation_count": int(gratings["is_partial"].sum()),
        "has_session_end": has_session_end,
        "recovery_warnings": recovery_warnings,
        "initial_photodiode_baseline": "low",
        "terminal_photodiode_rule": "if_high_next_frame_after_EndFrame_is_low",
        "terminal_photodiode_transition_inferred": logger_data.terminal_low_frame is not None,
        "terminal_photodiode_transition_frame": logger_data.terminal_low_frame,
        "photodiode_threshold_method": "full_recording_10_90_percentile_midpoint",
        "photodiode_signal_window": "full_analog_recording_no_DO0_DO1_cropping",
        "movie_presentation_count": int(blocks["TrialType"].eq("movie").sum()),
        "grating_presentation_count": len(gratings),
        "blank_presentation_count": int(gratings["is_blank"].sum()),
        "movie_identity_source": "stimulus_table_row_order",
        "block_stop_frame_sources": {str(k): int(v) for k, v in blocks["stop_frame_source"].value_counts().items()},
        "grating_block_bounds": "first_grating_start_to_last_grating_end_excluding_unlogged_outer_blanks",
        "extrapolated_boundary_count": extrapolated_count,
        "maximum_endpoint_extrapolation_frames": maximum_extrapolation,
        "stimulus_qc": "random_natural_movies_activity_qc",
        "movie_frame_timing": frame_metadata,
    }
    return blocks, gratings, metadata