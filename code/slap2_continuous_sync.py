"""Sequential DI3 synchronization for continuous Random Natural Movies.

The first detected pulse belongs to the first primary DMD path cycle. All
timestamps retain the caller's time base, including negative timestamps. Scan
lines retain their original, global 1-based coordinates on both DMDs. This
module deliberately does not use trial segmentation or pulse reconciliation.
"""

import warnings

import numpy as np


_ONSET_TOLERANCE_SECONDS = 0.001
_MAX_EXACT_LINE = 2**53 - 1  # np.interp uses float64 coordinates.


def _real_vector(value, name, allow_bool=False):
    """Validate before converting, so strings and complex values are not coerced."""
    array = np.asarray(value)
    kinds = "biuf" if allow_bool else "iuf"
    if array.ndim != 1 or array.dtype.kind not in kinds:
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_scalar(value, name, minimum=0):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an integer scalar >= {minimum}")
    number = float(array)
    if (not np.isfinite(number) or number != np.floor(number)
            or number < minimum or number > _MAX_EXACT_LINE):
        raise ValueError(
            f"{name} must be an integer between {minimum} and {_MAX_EXACT_LINE}"
        )
    return int(number)


def build_continuous_clock(
    signal, times, lines_per_cycle, total_cycles, recording_start=None
):
    """Build a shared scan-line clock from sequential primary DI3 pulses.

    Parameters
    ----------
    signal, times : one-dimensional array-like
        Nonempty binary DI3 samples and finite, strictly increasing timestamps.
        Rising edges use the timestamp of the first high sample. An initially
        high signal warns but is not counted: a pulse requires a low-to-high edge.
    lines_per_cycle : positive integer
        Actual constant primary path cycle length from the raw header. Never
        infer it from fluorescence sample counts or their maximum line index.
    total_cycles : nonnegative integer
        Total raw primary cycles, including any concatenated raw chunks.
    recording_start : finite scalar, optional
        HARP recording onset in the SAME time base as ``times``. Defaults to
        ``times[0]``; zero is not assumed to be recording onset.

    Returns
    -------
    dict
        ``cycle_starts`` contains ALL detected pulses, including an allowed
        extra final pulse. ``control_line_idxs`` and ``control_timestamps``
        describe boundaries at ``1 + arange(N) * lines_per_cycle``. ``qc`` is
        JSON-safe clock-level provenance; it is never updated by mapping.

    Notes
    -----
    For P pulses and C raw cycles, P > C + 1 is an error. When P = C + 1,
    the last pulse is the measured end of the last raw cycle. Otherwise, for
    P >= 2, append only the estimated end of cycle P using the mean observed
    inter-pulse duration. That boundary is an interpolation control, NOT a
    measured start of cycle P + 1. Cycles without a start pulse are unsupported.
    With one pulse, only the exact first start is supported; no duration is
    invented. Zero raw cycles are allowed (at most one boundary pulse), but
    their default mapping accepts no sample lines.
    """
    signal = _real_vector(signal, "signal", allow_bool=True)
    times = _real_vector(times, "times")
    if not times.size or signal.size != times.size:
        raise ValueError("signal and times must have the same nonzero length")
    if not np.all((signal == 0) | (signal == 1)):
        raise ValueError("signal must be binary (0 or 1)")
    if np.any(times[1:] <= times[:-1]):
        raise ValueError("times must be strictly increasing")
    lines_per_cycle = _integer_scalar(lines_per_cycle, "lines_per_cycle", 1)
    total_cycles = _integer_scalar(total_cycles, "total_cycles")
    raw_line_count = total_cycles * lines_per_cycle
    if raw_line_count + 1 > _MAX_EXACT_LINE:
        raise ValueError("raw cycle boundaries exceed exact float64 line coordinates")

    if recording_start is None:
        recording_start = float(times[0])
    else:
        onset = np.asarray(recording_start)
        if (onset.ndim != 0 or onset.dtype.kind not in "iuf"
                or not np.isfinite(onset)):
            raise ValueError("recording_start must be a finite real scalar")
        recording_start = float(onset)

    rising_idxs = np.flatnonzero((signal[:-1] == 0) & (signal[1:] == 1)) + 1
    initial_high = bool(signal[0] == 1)
    if initial_high:
        warnings.warn(
            "DI3 initially starts high; ignoring the initial high state because "
            "no low-to-high transition was observed.",
            RuntimeWarning, stacklevel=2,
        )
    cycle_starts = times[rising_idxs].copy()
    pulse_count = int(cycle_starts.size)
    if not pulse_count:
        raise ValueError("No DI3 pulses detected")
    if pulse_count > total_cycles + 1:
        raise ValueError(
            f"Detected {pulse_count} DI3 pulses for {total_cycles} raw cycles; "
            "at most one extra final boundary pulse is allowed"
        )

    first_rising_edge = float(times[rising_idxs[0]]) if rising_idxs.size else None
    early_edge = bool(
        first_rising_edge is not None
        and first_rising_edge <= recording_start + _ONSET_TOLERANCE_SECONDS
    )
    # DigitalInputState is event-driven: its first record may legitimately be
    # high several seconds after HARP recording began. Do not confuse the first
    # DI3 event with device recording onset when that onset is known.
    initial_high_at_onset = initial_high and bool(
        times[0] <= recording_start + _ONSET_TOLERANCE_SECONDS
    )
    onset_warning = initial_high_at_onset or early_edge
    if onset_warning:
        warnings.warn(
            "SLAP2 may have started before HARP began recording: DI3 is "
            "high at recording onset or its first rising edge is within 0.001 seconds "
            "of recording onset. Only observed low-to-high pulses are counted; "
            "line indices are unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )

    extra_final_pulse = pulse_count == total_cycles + 1
    estimate_end = not extra_final_pulse and pulse_count >= 2
    mean_period = None
    estimated_end = None
    control_timestamps = cycle_starts.copy()
    if estimate_end:
        with np.errstate(over="ignore", invalid="ignore"):
            mean_period = float(np.mean(np.diff(cycle_starts)))
            estimated_end = float(cycle_starts[-1] + mean_period)
        if (not np.isfinite(mean_period) or not np.isfinite(estimated_end)
                or estimated_end <= cycle_starts[-1]):
            raise ValueError("Estimated last-cycle end is not finite and increasing")
        control_timestamps = np.append(control_timestamps, estimated_end)
        last_cycle_policy = "mean_period_estimate"
        warnings.warn(
            "Estimating only the end of the last observed DI3 cycle using "
            "the mean inter-pulse duration; no later cycle starts are inferred.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif extra_final_pulse:
        last_cycle_policy = "measured_end"
    else:
        last_cycle_policy = "start_only"
        warnings.warn(
            "Only one DI3 pulse: only the exact first cycle start can be mapped; "
            "later lines are unsupported NaN because no duration is available.",
            RuntimeWarning,
            stacklevel=2,
        )

    raw_tail_cycles = max(total_cycles - pulse_count, 0)
    if raw_tail_cycles:
        warnings.warn(
            f"Unsupported raw tail: {raw_tail_cycles} raw cycles have no DI3 "
            "start pulse; their sample timestamps remain NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    control_line_idxs = (
        1 + np.arange(control_timestamps.size, dtype=np.int64) * lines_per_cycle
    )
    qc = {
        "segmentation_method": "sequential_di3",
        "lines_per_cycle": lines_per_cycle,
        "lines_per_cycle_source": "actual_header",
        "total_cycles": total_cycles,
        "raw_line_count": raw_line_count,
        "detected_pulse_count": pulse_count,
        "observed_raw_cycle_count": min(pulse_count, total_cycles),
        "measured_duration_cycle_count": min(pulse_count - 1, total_cycles),
        "extra_final_pulse_count": int(extra_final_pulse),
        "estimated_duration_cycle_count": int(estimate_end),
        "unsupported_raw_tail_cycle_count": raw_tail_cycles,
        "control_point_count": int(control_timestamps.size),
        "initial_signal_high": initial_high,
        "initial_signal_high_at_recording_onset": initial_high_at_onset,
        "first_rising_edge_within_onset_tolerance": early_edge,
        "onset_warning": onset_warning,
        "onset_tolerance_seconds": _ONSET_TOLERANCE_SECONDS,
        "recording_start": recording_start,
        "first_rising_edge_timestamp": first_rising_edge,
        "first_pulse_timestamp": float(cycle_starts[0]),
        "last_cycle_policy": last_cycle_policy,
        "last_cycle_end_estimated": estimate_end,
        "estimated_mean_period_seconds": mean_period,
        "estimated_last_cycle_end_timestamp": estimated_end,
        "unsupported_sample_policy": "NaN; count per mapping, not in clock QC",
    }
    return {
        "cycle_starts": cycle_starts,
        "control_line_idxs": control_line_idxs,
        "control_timestamps": control_timestamps,
        "lines_per_cycle": lines_per_cycle,
        "total_cycles": total_cycles,
        "qc": qc,
    }


def map_continuous_lines(frame_line_idxs, clock, recorded_line_count=None):
    """Interpolate original 1-based scan lines without clamping or rebasing.

    ``clock`` must be a result of :func:`build_continuous_clock`. Sample lines
    must be positive, finite integers in strictly increasing order. The default
    raw line limit is C * L for the primary DMD; secondary callers must supply
    their own raw total line count when it differs. A valid raw line beyond
    clock coverage returns NaN. Lines beyond the raw limit warn and return NaN,
    even when the shared clock could otherwise interpolate them.

    A measured final control is inclusive (useful for secondary recordings).
    An estimated end is exclusive: it cannot time the first line of a cycle
    with no observed start pulse. Empty inputs return a float64 empty array.
    Neither the inputs nor clock/QC are mutated. Count unsupported samples per
    call using ``np.count_nonzero(np.isnan(timestamps))`` on the returned array.
    """
    lines = _real_vector(frame_line_idxs, "frame_line_idxs")
    if (np.any(lines < 1) or np.any(lines > _MAX_EXACT_LINE)
            or np.any(lines != np.floor(lines))):
        raise ValueError("frame_line_idxs must be positive finite integer coordinates")
    if np.any(lines[1:] <= lines[:-1]):
        raise ValueError("frame_line_idxs must be strictly increasing")
    if recorded_line_count is None:
        recorded_line_count = clock["total_cycles"] * clock["lines_per_cycle"]
    recorded_line_count = _integer_scalar(recorded_line_count, "recorded_line_count")
    beyond_recorded = lines > recorded_line_count
    if np.any(beyond_recorded):
        warnings.warn(
            f"{np.count_nonzero(beyond_recorded)} frame_line_idxs exceed "
            f"recorded_line_count={recorded_line_count}; setting timestamps to NaN.",
            RuntimeWarning, stacklevel=2,
        )
    if not lines.size:
        return np.empty(0, dtype=np.float64)

    control_lines = clock["control_line_idxs"]
    timestamps = np.interp(
        lines, control_lines, clock["control_timestamps"], left=np.nan, right=np.nan
    )
    if clock["qc"]["last_cycle_end_estimated"]:
        timestamps[lines >= control_lines[-1]] = np.nan
    timestamps[beyond_recorded] = np.nan
    return timestamps