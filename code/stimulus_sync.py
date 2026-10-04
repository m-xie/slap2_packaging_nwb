from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class AlignmentQC:
    """Summary statistics for a logger-frame to HARP-time alignment."""

    logger_path: str
    logger_transition_count: int
    harp_transition_count: int
    matched_transition_count: int
    frame_rate_hz: float
    median_absolute_residual_ms: float
    p95_absolute_residual_ms: float
    maximum_absolute_residual_ms: float


def select_stimulus_logger(candidate_paths):
    """Select deterministically, preferring the known complete ``dup`` logger."""
    paths = sorted(Path(path) for path in candidate_paths)
    if not paths:
        raise FileNotFoundError("No stimulus logger CSV files were found")
    duplicate_paths = [path for path in paths if "dup" in path.name.lower()]
    return duplicate_paths[0] if duplicate_paths else paths[0]


def extract_logger_events(logger_path, *, stimulus_frames=None, initial_low_baseline=False):
    """Extract expected photodiode changes and stimulus display frames.

    Format adapters may supply validated stimulus frames; by default the
    existing StimStart-* parser and its validation are used unchanged.
    With initial_low_baseline=True, only a high first state at STARTSLAP is
    inserted as an initial edge. This assumes the pre-stimulus patch was low;
    later state changes (including observed high-to-low changes) are unchanged.
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
        len(photodiode_frames) and photodiode_frames[0] == reference_frame
        and (not initial_low_baseline or states[0])
    ):
        # Preserve the historical first-state anchor by default. For a known
        # low baseline, a first low state is not a physical transition.
        transition_indices = np.insert(transition_indices, 0, 0)
    if len(transition_indices) < 3:
        raise ValueError(
            f"Logger {logger_path} has too few photodiode transitions: "
            f"{len(transition_indices)}"
        )

    transition_frames = photodiode_frames[transition_indices]
    transition_states = states[transition_indices]
    if np.any(np.diff(transition_frames) <= 0):
        raise ValueError("Logger photodiode transition frames must increase")

    return LoggerEvents(
        path=logger_path,
        reference_frame=reference_frame,
        stimulus_frames=stimulus_frames,
        transition_frames=transition_frames,
        transition_states=transition_states,
    )


def extract_harp_photodiode_transitions(harp_data):
    """Binarize the physical photodiode and return its HARP-timed edges.

    DO1 normally bounds the useful analog interval. If DO1 is absent, the HARP
    analog recording end is used instead: DI3 can stop before a stimulus block
    ends, so its final cycle is not a safe photodiode cutoff.
    """
    analog_times = np.asarray(harp_data["analog_times"], dtype=float)
    photodiode = np.asarray(harp_data["photodiode"], dtype=float)
    starts = np.asarray(harp_data["normalized_slap2_start"], dtype=float)
    ends = np.asarray(harp_data["normalized_slap2_end"], dtype=float)
    if len(starts) == 0:
        raise ValueError("HARP DO0 pulse is required for photodiode sync")

    acquisition_start = float(starts[0])
    if len(ends):
        acquisition_end = float(ends[-1])
    else:
        # Logger matching and residual gates below reject an unusable analog
        # tail; keeping it is preferable to dropping valid post-imaging stimuli.
        acquisition_end = float(analog_times[-1])
    if acquisition_end <= acquisition_start:
        raise ValueError("HARP acquisition end must follow acquisition start")
    # Long pre-acquisition recordings contain baseline ADC noise but no useful
    # bright state. Restricting level estimation to the acquisition interval
    # prevents that noise from collapsing the threshold toward the baseline.
    in_acquisition = (
        (analog_times >= acquisition_start) & (analog_times <= acquisition_end)
    )
    if np.count_nonzero(in_acquisition) < 3:
        raise ValueError("HARP photodiode has too few in-acquisition samples")

    times = analog_times[in_acquisition]
    signal = photodiode[in_acquisition]
    # Percentiles are robust to brief transitions and isolated analog outliers.
    low, high = np.quantile(signal, (0.1, 0.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("HARP photodiode levels cannot be separated")
    threshold = (low + high) / 2
    states = signal > threshold
    transition_indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    if len(transition_indices) < 3:
        raise ValueError(
            f"HARP photodiode has too few transitions: {len(transition_indices)}"
        )
    return times[transition_indices], states[transition_indices], threshold


def _ordered_transition_matches(
    logger_frames,
    logger_states,
    harp_times,
    harp_states,
    slope,
    intercept,
    tolerance,
):
    """Greedily pair same-state edges without reusing or reordering HARP edges."""
    matches = []
    last_harp_index = -1
    for logger_index, (frame, state) in enumerate(
        zip(logger_frames, logger_states)
    ):
        predicted_time = slope * frame + intercept
        candidates = np.flatnonzero(
            (harp_states == state)
            & (np.abs(harp_times - predicted_time) <= tolerance)
        )
        # Advancing this lower bound makes every accepted match one-to-one and
        # order-preserving while allowing either trace to omit occasional edges.
        candidates = candidates[candidates > last_harp_index]
        if len(candidates) == 0:
            continue
        harp_index = int(
            candidates[np.argmin(np.abs(harp_times[candidates] - predicted_time))]
        )
        matches.append((logger_index, harp_index))
        last_harp_index = harp_index
    return np.asarray(matches, dtype=int).reshape(-1, 2)


def align_logger_frames_to_harp(
    logger_data,
    harp_times,
    harp_states,
    nominal_frame_rate_hz=60.0,
    initial_tolerance=0.12,
):
    """Align expected logger edges to physical HARP edges in frame coordinates.

    The fitted line initializes and validates correspondence; returned anchors
    retain the measured HARP edge times for subsequent piecewise interpolation.
    """
    relative_frames = (
        logger_data.transition_frames.astype(float) - logger_data.reference_frame
    )
    slope = 1.0 / nominal_frame_rate_hz

    # STARTSLAP/DO0 provide an approximate common origin, but retained HARP time
    # may remain offset after removal of a leading acquisition artifact. Center
    # the search on the first edges, then let the full random pattern determine
    # the best sub-frame-scale offset.
    offset_center = harp_times[0] - slope * relative_frames[0]
    candidate_offsets = np.linspace(
        offset_center - 0.15, offset_center + 0.15, 301
    )
    best_score = None
    intercept = 0.0
    for offset in candidate_offsets:
        predicted = slope * relative_frames + offset
        errors = []
        for state in (False, True):
            source = predicted[logger_data.transition_states == state]
            target = harp_times[harp_states == state]
            if len(source) == 0 or len(target) == 0:
                continue
            positions = np.searchsorted(target, source)
            left = target[np.clip(positions - 1, 0, len(target) - 1)]
            right = target[np.clip(positions, 0, len(target) - 1)]
            errors.extend(np.minimum(np.abs(source - left), np.abs(source - right)))
        errors = np.asarray(errors)
        score = (
            np.count_nonzero(errors <= 0.025),
            -float(np.median(np.minimum(errors, 0.25))),
        )
        if best_score is None or score > best_score:
            best_score = score
            intercept = float(offset)

    # Alternate ordered matching and affine fitting. The first pass is loose
    # enough for the coarse offset; later passes tighten around the fitted map.
    matches = np.empty((0, 2), dtype=int)
    for iteration in range(8):
        tolerance = initial_tolerance if iteration == 0 else 0.05
        new_matches = _ordered_transition_matches(
            relative_frames,
            logger_data.transition_states,
            harp_times,
            harp_states,
            slope,
            intercept,
            tolerance,
        )
        if len(new_matches) < 10:
            raise ValueError(
                f"Too few logger/HARP photodiode matches: {len(new_matches)}"
            )
        x = relative_frames[new_matches[:, 0]]
        y = harp_times[new_matches[:, 1]]
        new_slope, new_intercept = np.polyfit(x, y, 1)
        residuals = y - (new_slope * x + new_intercept)
        # Median/MAD rejection removes occasional incorrect or noisy edges
        # without imposing a sub-millisecond requirement on display hardware.
        center = np.median(residuals)
        mad = np.median(np.abs(residuals - center))
        keep = np.abs(residuals - center) <= max(0.035, 6 * 1.4826 * mad)
        new_matches = new_matches[keep]
        slope, intercept = float(new_slope), float(new_intercept)
        if np.array_equal(new_matches, matches):
            matches = new_matches
            break
        matches = new_matches

    logger_indices = matches[:, 0]
    harp_indices = matches[:, 1]
    anchor_frames = logger_data.transition_frames[logger_indices].astype(float)
    anchor_times = harp_times[harp_indices]
    residuals = anchor_times - (
        slope * (anchor_frames - logger_data.reference_frame) + intercept
    )
    absolute_residuals = np.abs(residuals)
    # These are session-level quality gates. Observed good sessions retain over
    # 95% of edges with p95 residuals around 15-20 ms.
    if len(matches) < 0.5 * min(
        len(logger_data.transition_frames), len(harp_times)
    ):
        raise ValueError("Photodiode alignment matched fewer than half the transitions")
    if np.quantile(absolute_residuals, 0.95) > 0.04:
        raise ValueError("Photodiode alignment p95 residual exceeds 40 ms")

    qc = AlignmentQC(
        logger_path=str(logger_data.path),
        logger_transition_count=len(logger_data.transition_frames),
        harp_transition_count=len(harp_times),
        matched_transition_count=len(matches),
        frame_rate_hz=1.0 / slope,
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