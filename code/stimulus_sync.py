from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoggerEvents:
    """Logger events expressed in display-frame coordinates, not logger time."""

    path: Path
    reference_frame: int
    stimulus_frames: np.ndarray
    transition_frames: np.ndarray
    transition_states: np.ndarray
    terminal_low_frame: int | None = None


@dataclass(frozen=True)
class AlignmentQC:
    """Strict pairing metadata with descriptive, non-gating timing statistics.

    frame_rate_hz is the endpoint-to-endpoint average rate. Residual fields
    describe departures from that endpoint line, NOT a fitted alignment or
    timing uncertainty. They never select, reject, or move an anchor.
    """

    logger_path: str
    logger_transition_count: int
    harp_transition_count: int
    matched_transition_count: int
    frame_rate_hz: float
    median_absolute_residual_ms: float
    p95_absolute_residual_ms: float
    maximum_absolute_residual_ms: float
    matching_method: str = "strict_one_to_one"
    residual_reference: str = "endpoint_secant_descriptive_only"


def select_stimulus_logger(candidate_paths):
    """Select deterministically, preferring the known complete ``dup`` logger."""
    paths = sorted(Path(path) for path in candidate_paths)
    if not paths:
        raise FileNotFoundError("No stimulus logger CSV files were found")
    duplicate_paths = [path for path in paths if "dup" in path.name.lower()]
    return duplicate_paths[0] if duplicate_paths else paths[0]


def extract_logger_events(logger_path, *, stimulus_frames=None, initial_low_baseline=False,
                          terminal_low_after_end_frame=False):
    """Extract expected photodiode changes and stimulus display frames.

    Format adapters may supply validated stimulus frames; by default the
    existing StimStart-* parser and its validation are used unchanged.
    With initial_low_baseline=True, a high first state is always inserted as a
    rising edge at its logged frame, regardless of the STARTSLAP frame. A low
    first state is not an edge. This assumes the pre-logging patch was low;
    later state changes (including observed high-to-low changes) are unchanged.
    With terminal_low_after_end_frame=True, EndFrame declares that its next
    frame is low. If the last explicit state is high, append that implied
    falling transition at EndFrame + 1. Without EndFrame, infer nothing.
    END events are not used by this rule.
    """
    logger_path = Path(logger_path)
    table = pd.read_csv(logger_path)
    required_columns = {"Frame", "Timestamp", "Value"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"Logger {logger_path} is missing columns: {sorted(missing_columns)}"
        )

    values = table["Value"].fillna("").astype(str)
    # STARTSLAP identifies frame zero for alignment. The logger Timestamp column
    # is intentionally not used as the primary synchronization coordinate.
    reference_rows = table.loc[values.eq("STARTSLAP")]
    if len(reference_rows) == 0:
        raise ValueError(
            f"Logger {logger_path} must contain at least one STARTSLAP event"
        )
    # Multi-trial legacy sessions emit STARTSLAP once per acquisition trial.
    # The first marker shares the origin used by HARP normalization; subsequent
    # markers remain ordinary frames within the global display-frame sequence.
    reference_frame = int(reference_rows.iloc[0]["Frame"])

    if stimulus_frames is None:
        stimulus_frames = table.loc[
            values.str.startswith("StimStart-"), "Frame"
        ].to_numpy(dtype=int)
        if len(stimulus_frames) == 0:
            raise ValueError(f"Logger {logger_path} contains no StimStart-* events")
    else:
        stimulus_frames = np.asarray(stimulus_frames, dtype=int)

    # Frame, wheel, and photodiode rows are interleaved. Only explicit
    # Photodiode-* rows describe the expected state of the display patch.
    photodiode_rows = table.loc[
        values.isin(("Photodiode-0", "Photodiode-1")), ["Frame", "Value"]
    ]
    states = photodiode_rows["Value"].eq("Photodiode-1").to_numpy()
    transition_indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    photodiode_frames = photodiode_rows["Frame"].to_numpy(dtype=int)
    if (
        len(photodiode_frames)
        and (states[0] if initial_low_baseline else photodiode_frames[0] == reference_frame)
    ):
        # Preserve the historical first-state anchor by default. For a known
        # low baseline, count the first high even when STARTSLAP is elsewhere.
        transition_indices = np.insert(transition_indices, 0, 0)
    transition_frames = photodiode_frames[transition_indices]
    transition_states = states[transition_indices]
    terminal_low_frame = None
    if terminal_low_after_end_frame:
        end_rows = table.loc[values.eq("EndFrame"), "Frame"]
        if len(end_rows) > 1:
            raise ValueError("Terminal photodiode rule requires at most one EndFrame")
        if len(end_rows):
            end_frame = float(end_rows.iloc[0])
            if not np.isfinite(end_frame) or end_frame < 0 or end_frame != np.floor(end_frame):
                raise ValueError("EndFrame must be a nonnegative integer")
            if len(photodiode_frames) and photodiode_frames[-1] > end_frame:
                raise ValueError("Photodiode events extend beyond EndFrame")
            if len(states) and states[-1]:
                terminal_low_frame = int(end_frame) + 1
                transition_frames = np.append(transition_frames, terminal_low_frame)
                transition_states = np.append(transition_states, False)
    if len(transition_frames) < 3:
        raise ValueError(
            f"Logger {logger_path} has too few photodiode transitions: "
            f"{len(transition_frames)}"
        )

    if np.any(np.diff(transition_frames) <= 0):
        raise ValueError("Logger photodiode transition frames must increase")

    return LoggerEvents(
        path=logger_path,
        reference_frame=reference_frame,
        stimulus_frames=stimulus_frames,
        transition_frames=transition_frames,
        transition_states=transition_states,
        terminal_low_frame=terminal_low_frame,
    )


def extract_harp_photodiode_transitions(harp_data):
    """Detect every photodiode transition across the full analog recording.

    DO0/DO1 are not optical bounds. Neither samples nor edges are cropped to
    them, to DI3, or to a desired logger count. Estimate the threshold from the
    full recording; retain leading/trailing and noisy edges so mismatches are
    visible to the strict pairing check rather than silently repaired.
    """
    analog_times = np.asarray(harp_data["analog_times"], dtype=float)
    photodiode = np.asarray(harp_data["photodiode"], dtype=float)
    if (
        analog_times.ndim != 1 or photodiode.shape != analog_times.shape
        or len(analog_times) < 3
        or not np.isfinite(analog_times).all() or not np.isfinite(photodiode).all()
    ):
        raise ValueError("HARP analog samples must be finite paired arrays (at least 3 samples)")
    time_diffs = np.diff(analog_times)
    non_increasing = np.flatnonzero(time_diffs <= 0)
    if non_increasing.size:
        warnings.warn(
            f"HARP analog timestamps are not strictly increasing: {non_increasing.size} "
            f"non-increasing steps; first at sample index {non_increasing[0] + 1} "
            f"(zero-based), minimum delta {time_diffs.min():.9f} s. "
            "Preserving recorded timestamps and sample order; photodiode anchors "
            "must still be strictly increasing for alignment.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Percentiles are robust to brief transitions and isolated analog outliers.
    low, high = np.quantile(photodiode, (0.1, 0.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("HARP photodiode levels cannot be separated")
    threshold = (low + high) / 2
    states = photodiode > threshold
    transition_indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    if len(transition_indices) < 3:
        raise ValueError(
            f"HARP photodiode has too few transitions: {len(transition_indices)}"
        )
    return analog_times[transition_indices], states[transition_indices], threshold


def align_logger_frames_to_harp(
    logger_data,
    harp_times,
    harp_states,
):
    """Pair all logger and measured photodiode edges one-to-one, in order.

    Counts and polarities must agree exactly. No fit, tolerance, event search,
    outlier rejection, or cropping selects the anchors. Their original frames
    and measured HARP times define the downstream piecewise-linear mapping.
    """
    anchor_frames = np.asarray(logger_data.transition_frames, dtype=float)
    anchor_times = np.asarray(harp_times, dtype=float)
    logger_states = np.asarray(logger_data.transition_states)
    harp_states = np.asarray(harp_states)
    for label, coordinates, states in (
        ("Logger", anchor_frames, logger_states),
        ("HARP", anchor_times, harp_states),
    ):
        if (
            coordinates.ndim != 1 or states.shape != coordinates.shape
            or not np.isfinite(coordinates).all()
            or np.any(np.diff(coordinates) <= 0)
            or not np.isin(states, (False, True)).all()
        ):
            raise ValueError(f"{label} photodiode transitions must have finite strictly increasing coordinates and paired binary states")
    if len(anchor_frames) != len(anchor_times):
        raise ValueError(
            "Photodiode transition count mismatch: "
            f"logger={len(anchor_frames)}, HARP={len(anchor_times)}. "
            "Strict one-to-one pairing requires equal counts; all detected "
            "transitions are retained without DO0/DO1 cropping or edge skipping."
        )
    if len(anchor_frames) < 2:
        raise ValueError("At least two paired photodiode transitions are required")
    mismatch = np.flatnonzero(logger_states != harp_states)
    if mismatch.size:
        index = int(mismatch[0])
        raise ValueError(
            f"Photodiode transition polarity mismatch at index {index} (zero-based): "
            f"logger frame {anchor_frames[index]:g}, state {int(logger_states[index])}; "
            f"HARP time {anchor_times[index]:.9f} s, state {int(harp_states[index])}. "
            "Strict one-to-one pairing does not shift or skip edges."
        )
    if np.any(logger_states[1:] == logger_states[:-1]):
        raise ValueError("Photodiode transition states must alternate; repeated states are not transitions")

    # Descriptive metadata only: departures from the line joining the first
    # and last anchors. No line is fitted and these values cannot reject data.
    seconds_per_frame = (anchor_times[-1] - anchor_times[0]) / (anchor_frames[-1] - anchor_frames[0])
    endpoint_line = anchor_times[0] + seconds_per_frame * (anchor_frames - anchor_frames[0])
    absolute_residuals = np.abs(anchor_times - endpoint_line)

    qc = AlignmentQC(
        logger_path=str(logger_data.path),
        logger_transition_count=len(logger_data.transition_frames),
        harp_transition_count=len(harp_times),
        matched_transition_count=len(anchor_frames),
        frame_rate_hz=1.0 / seconds_per_frame,
        median_absolute_residual_ms=float(np.median(absolute_residuals) * 1000),
        p95_absolute_residual_ms=float(
            np.quantile(absolute_residuals, 0.95) * 1000
        ),
        maximum_absolute_residual_ms=float(np.max(absolute_residuals) * 1000),
    )
    return anchor_frames, anchor_times, qc


def synchronize_stimulus_frames(logger_path, harp_data, expected_stim_count):
    """Map logger StimStart frames onto authoritative HARP timestamps."""
    logger_data = extract_logger_events(logger_path)
    if len(logger_data.stimulus_frames) != expected_stim_count:
        raise ValueError(
            f"Logger stimulus count {len(logger_data.stimulus_frames)} does not "
            f"match stimulus table row count {expected_stim_count}"
        )
    harp_times, harp_states, _ = extract_harp_photodiode_transitions(harp_data)
    anchor_frames, anchor_times, qc = align_logger_frames_to_harp(
        logger_data, harp_times, harp_states
    )
    if (
        logger_data.stimulus_frames[0] < anchor_frames[0]
        or logger_data.stimulus_frames[-1] > anchor_frames[-1]
    ):
        raise ValueError("Photodiode anchors do not cover all stimulus starts")
    # A StimStart need not coincide with a photodiode edge. Interpolate its
    # display frame between neighboring matched physical edges in HARP time.
    stimulus_times = np.interp(
        logger_data.stimulus_frames.astype(float), anchor_frames, anchor_times
    )
    return stimulus_times, qc


def resolve_stimulus_start_times(
    stimulus_count, harp_data, logger_path=None
):
    """Prefer physical logger/photodiode timing and fall back to complete DO2."""
    do2_times = np.asarray(harp_data["normalized_start_gratings"], dtype=float)
    logger_error = None
    if logger_path is not None:
        try:
            stimulus_times, qc = synchronize_stimulus_frames(
                logger_path, harp_data, stimulus_count
            )
            metadata = {"source": "logger_photodiode_aligned", **qc.__dict__}
            return stimulus_times, metadata
        except (OSError, ValueError) as error:
            # A complete DO2 stream remains a safe legacy fallback. Preserve the
            # reason in provenance instead of silently changing timing sources.
            logger_error = error

    if len(do2_times) == stimulus_count:
        metadata = {"source": "harp_do2_fallback"}
        if logger_error is not None:
            metadata["logger_fallback_reason"] = str(logger_error)
        elif logger_path is None:
            metadata["logger_fallback_reason"] = "no stimulus logger was found"
        return do2_times, metadata

    if logger_error is not None:
        raise ValueError(
            f"Logger/photodiode synchronization failed ({logger_error}) and "
            f"HARP DO2 has {len(do2_times)} events for {stimulus_count} stimuli"
        ) from logger_error
    if len(do2_times) != 0:
        raise ValueError(
            f"Stimulus table has {stimulus_count} rows but HARP DO2 has "
            f"{len(do2_times)} events"
        )
    raise ValueError("No usable stimulus logger or HARP DO2 timing was found")