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


LOGGER_FORMAT = "Random Natural Movies"
GRATING_ORIENTATIONS = frozenset((0, 45, 90, 135, 180, 225, 270, 315, 359))
MAX_ENDPOINT_EXTRAPOLATION_FRAMES = 2.0


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
        "is_partial",
    }
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
    if (
        np.any(np.diff(event_frames) <= 0)
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
        if row["TrialType"] == "movie":
            if events[position][:2] != ("MovieFrame", 1):
                raise ValueError(f"Expected MovieFrame-1 for stimulus table row {row_id}")
            movie_frames = [first_frame]
            position += 1
            while position < len(events):
                kind, counter, frame = events[position]
                if kind != "MovieFrame" or counter == 1:
                    break
                if counter != len(movie_frames) + 1:
                    raise ValueError(f"Missing or duplicate MovieFrame counter at table row {row_id}")
                movie_frames.append(frame)
                position += 1
            single_frame = len(movie_frames) == 1
            if single_frame:
                warn(f"Movie at table row {row_id} has only one frame; playback cadence cannot be estimated.")
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
            cadence = None if single_frame else float(np.median(np.diff(movie_frames)))
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
    blocks.attrs["recovery_warnings"] = recovery_warnings
    blocks.attrs["has_session_end"] = session_end is not None
    grating_columns = [
        "stimulus_table_row", "grating_in_block", "start_frame", "stop_frame",
        "logger_orientation", "Orientation", "is_blank", "is_partial",
    ]
    return blocks, pd.DataFrame(gratings, columns=grating_columns)


def _map_boundary_frames(frames, anchor_frames, anchor_times):
    """Interpolate boundaries, with at most two display ticks of extrapolation.

    END may follow the final physical photodiode edge by one display tick.
    Never silently clamp it (np.interp's default) or extrapolate an absent tail.
    The small, explicitly reported extension uses the local measured slope.
    """
    frames = np.asarray(frames, dtype=float)
    distances = np.maximum(anchor_frames[0] - frames, frames - anchor_frames[-1])
    if np.any(distances > MAX_ENDPOINT_EXTRAPOLATION_FRAMES):
        raise ValueError("Photodiode anchors do not cover playback boundaries (beyond two display frames)")
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


def synchronize_presentations(stimulus_table, logger_path, harp_data):
    """Align all playback boundaries once; DO2 cannot replace missing offsets."""
    blocks, gratings = read_presentation_frames(stimulus_table, logger_path)
    recovery_warnings = list(blocks.attrs["recovery_warnings"])
    has_session_end = blocks.attrs["has_session_end"]
    logger_data = stimulus_sync.extract_logger_events(
        logger_path, stimulus_frames=blocks["start_frame"].to_numpy(),
        initial_low_baseline=True,
    )
    harp_times, harp_states, _ = stimulus_sync.extract_harp_photodiode_transitions(harp_data)
    anchor_frames, anchor_times, qc = stimulus_sync.align_logger_frames_to_harp(
        logger_data, harp_times, harp_states,
    )
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
        "photodiode_threshold_method": "acquisition_10_90_percentile_midpoint",
        "movie_presentation_count": int(blocks["TrialType"].eq("movie").sum()),
        "grating_presentation_count": len(gratings),
        "blank_presentation_count": int(gratings["is_blank"].sum()),
        "movie_identity_source": "stimulus_table_row_order",
        "block_stop_frame_sources": {str(k): int(v) for k, v in blocks["stop_frame_source"].value_counts().items()},
        "grating_block_bounds": "first_grating_start_to_last_grating_end_excluding_unlogged_outer_blanks",
        "extrapolated_boundary_count": extrapolated_count,
        "maximum_endpoint_extrapolation_frames": maximum_extrapolation,
        "stimulus_qc": "skipped_for_random_natural_movies",
    }
    return blocks, gratings, metadata