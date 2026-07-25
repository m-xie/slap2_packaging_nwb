import re
import warnings
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SLAP2 / HARP Timing Synchronization
# =============================================================================
#
# --- The SLAP2 imaging system ---
#
# SLAP2 (Scanned Line Angular Projection 2-photon) uses two Digital Micromirror
# Devices (DMDs) to define spatial regions of interest (ROIs) on two independent
# imaging planes. The microscope scans lines continuously across both planes
# at a fixed rate (lineRateHz). Each DMD defines which scan lines belong to its
# ROIs; when the laser hits an ROI line, a fluorescence sample is recorded for
# that plane.
#
# Because both DMDs share the same scanner, they experience the same
# physical scan lines at the same physical times. This shared scan-line clock is
# the key bridge between the two planes.
#
# Each DMD completes one "imaging cycle" after its full set of ROI lines has been
# scanned once. Because the two planes have different ROI configurations, they
# generally have different numbers of scan lines per cycle (lines_per_cycle) and
# therefore cycle at different rates. Only the primary plane's cycle timing is
# recorded by HARP; the secondary plane has no direct HARP signal.
#
#
# --- Input data: fluorescence side ---
#
# frame_line_idxs : 1-D integer array, one entry per fluorescence sample.
#   Each value is the 1-based scan line index at which that sample was acquired.
#   This index counts up monotonically within each trial (it does NOT reset
#   between cycles within a trial), but resets to 1 at the start of each trial.
#   The maximum value in a trial is approximately n_cycles * lines_per_cycle.
#
# trial_num_frames : 1-D integer array, one entry per trial.
#   The number of fluorescence samples recorded in each trial. Summing this
#   array gives the total length of frame_line_idxs.
#
# lines_per_cycle : scalar (int or float), from the .meta h5 file.
#   The nominal number of scan lines per imaging cycle. Used to estimate how
#   many cycles fit in a trial. In practice this value may be slightly inaccurate
#   (see "effective_lpc" below).
#
# Each trial is a contiguous slice of frame_line_idxs defined by cumulative sums
# of trial_num_frames:
#
#   boundaries = [0, trial_num_frames[0],
#                    trial_num_frames[0] + trial_num_frames[1], ...]
#   trial_i_line_idxs = frame_line_idxs[boundaries[i] : boundaries[i+1]]
#
#
# --- Input data: HARP side (primary plane only) ---
#
# slap2_cycle_clock_signal : square wave sampled at the HARP clock rate.
#   A value of 1 means the primary plane is actively imaging; 0 means it is
#   between cycles. Only rising edges (0->1) are used; they mark the start of
#   each primary plane imaging cycle. Falling edges are present in the signal
#   but do not reliably mark cycle ends and are ignored.
#
# slap2_cycle_clock_times : timestamps corresponding to each sample of the
#   square wave, in the HARP absolute time base.
#
# Parsing the square wave yields one array:
#   cycle_starts[c]  — absolute time at which primary plane cycle c began
#
# Each cycle's time span is defined as [cycle_starts[c], cycle_starts[c+1]).
# For the final cycle in a trial, the end is estimated as
# cycle_starts[-1] + mean(diff(cycle_starts)).
#
#
# --- Session types ---
#
# Trial-based sessions consist of multiple trials separated by long inter-trial
# pauses (typically hundreds of milliseconds) during which imaging stops
# entirely. These pauses appear in the HARP data as abnormally large gaps
# between consecutive cycle start times. Gap detection exploits this contrast
# to segment the cycle stream into per-trial groups without needing to count
# cycles precisely:
#
#   threshold = gap_multiplier * median(cycle_starts[1:] - cycle_starts[:-1])
#   trial boundaries = consecutive-start gaps > threshold
#
# Continuous sessions have a single trial with no inter-trial pause. Gap
# detection is inapplicable; cycles are assigned directly via line-count.
#
#
# --- Primary plane alignment algorithm ---
#
#   1. Parse the HARP cycle clock square wave to extract cycle start timestamps
#      from rising edges. Falling edges are ignored.
#
#   2. Assign HARP cycles to trials.
#      For trial-based sessions, attempt gap detection first: find inter-cycle
#      gaps that exceed gap_multiplier * median(all_gaps) and treat them as
#      trial boundaries. This method is robust to pre-bug-fix data because it
#      relies on the shape of the gap distribution, not on exact pulse counts.
#      If gap detection fails (wrong number of long gaps found), or for
#      continuous (single-trial) sessions, fall back to line-count assignment:
#      estimate cycles per trial as ceil(max(trial_line_idxs) / lines_per_cycle)
#      and consume that many cycles sequentially from the cycle stream.
#
#   3. Assign each fluorescence sample to its cycle within the trial.
#      Compute a data-derived lines-per-cycle value:
#
#        effective_lpc = max(trial_line_idxs) / n_detected_cycles
#        sample_cycle_idxs = ceil(trial_line_idxs / effective_lpc)  # 1-based
#
#      The nominal lines_per_cycle from metadata is intentionally NOT used here.
#      It is often slightly larger than the actual lines scanned per cycle,
#      causing ceil(max / nominal_lpc) to undercount by 1 and break the
#      per-cycle grouping. effective_lpc is derived from the data itself and
#      guarantees max(sample_cycle_idxs) == n_detected_cycles exactly.
#
#   4. Interpolate timestamps within each cycle.
#      Each cycle c spans from cycle_starts[c] to cycle_starts[c+1]. For the
#      final cycle in a trial, the end is estimated from the mean cycle period.
#      Samples in cycle c receive uniformly spaced timestamps across that span:
#
#        timestamps[mask_c] = linspace(cycle_starts[c], cycle_starts[c+1], n_c)
#
#      effective_lpc is defined from the detected HARP cycle count, so ordinary
#      positive line indices are distributed across exactly that many cycle
#      buckets. A trial-span linspace remains as a defensive fallback if those
#      derived bucket counts ever disagree.
#
#      A trial with only one detected cycle has no following HARP cycle start
#      from which to infer its duration. The current implementation therefore
#      uses a zero-duration final cycle. Such a trial cannot produce strictly
#      increasing secondary timestamps and is not supported for secondary-plane
#      alignment.
#
#
# --- Secondary plane alignment algorithm ---
#
#   1. Run get_slap2_primary_plane_timestamps to obtain primary_timestamps and
#      trial_line_time_maps (the per-trial cycle-boundary control points produced
#      during primary alignment).
#
#   2. For each trial, the primary alignment produces a set of (line, time)
#      control points. Each HARP cycle start is paired with the estimated scan
#      line where that cycle begins:
#
#        control_line_idxs[c]  = 1 + c * effective_lpc
#        control_timestamps[c] = cycle_starts[c]
#
#      One final boundary is appended after the last cycle using the mean HARP
#      cycle period. Representing each boundary once produces strictly increasing
#      line and time coordinates and avoids flat intervals between cycles.
#
#   3. Interpolate secondary plane timestamps from the control points:
#
#        secondary_timestamps = interp(secondary_line_idxs,
#                                      control_line_idxs, control_timestamps)
#
#      Secondary samples whose scan line indices fall outside the estimated
#      cycle-boundary range are clamped (np.interp default) with a warning.
#      Each non-empty trial is then required to have strictly increasing
#      timestamps; equal or decreasing values raise ValueError.
#
#
# --- Known data quality issues ---
#
# Pre-bug-fix cycle clock frequency: in some older data, the HARP cycle clock
# pulses were emitted at an incorrect frequency. The total pulse count is lower
# than the true number of imaging cycles. Gap detection handles this gracefully
# because it relies on the shape of the gap distribution between cycle starts,
# not the count. The line-count fallback will issue a warning if fewer than 90%
# of the expected pulses are present.
#
# Inaccurate lines_per_cycle metadata: the .meta h5 value for lines_per_cycle
# may not match the actual lines scanned per cycle in a given recording. This
# affects the line-count cycle assignment (step 1 fallback) and the sanity-check
# warnings, but NOT the per-cycle timestamp interpolation, which uses effective_lpc.
# If a large discrepancy is observed in sanity-check warnings, the metadata field
# being read should be verified.
#
# frame_line_idxs not resetting to 1: some processed summaries contain a trial
# whose scan-line counter continued from the previous trial. The caller should
# run normalize_continued_trial_line_indices independently for each plane before
# either timestamp entry point. In-range non-1 starts are retained because they
# may represent valid partial trials. validate_inputs warns about any remaining
# non-1 starts.
#
# Signal starts high: if the HARP square wave is already high at the first
# recorded sample (i.e. recording began mid-cycle), there is no rising edge at
# the front of the signal. parse_cycle_clock detects this via signal[0] and
# synthesises a rising edge at times[0], treating the start of recording as the
# effective cycle start.
#
#
# --- Entry points ---
#
#   normalize_continued_trial_line_indices(...)
#       -> normalized_frame_line_idxs, correction_records
#
#   get_slap2_primary_plane_timestamps(...)
#       -> primary_timestamps, trial_line_time_maps, sync_qc_values
#
#   get_slap2_secondary_plane_timestamps(secondary_..., trial_line_time_maps)
#       -> secondary_timestamps
#
#   plot_slap2_sync_qc(slap2_cycle_clock_signal, slap2_cycle_clock_times,
#                      primary_frame_line_idxs, primary_trial_num_frames,
#                      trial_line_time_maps, primary_timestamps, ...)
#       -> matplotlib.figure.Figure
#
# =============================================================================


# =============================================================================
# .dat file helpers
# =============================================================================

SLAP2_DAT_MAGIC = np.uint32(322379495)

def read_dat_num_cycles(dat_path):
    """Read numCycles from a SLAP2 v2 .dat binary file (header only)."""
    with open(dat_path, 'rb') as f:
        # Read first 12 bytes to get magic, version, and header size
        preamble = np.frombuffer(f.read(12), dtype=np.uint32)
        assert preamble[0] == SLAP2_DAT_MAGIC, f"Not a SLAP2 dat file: {dat_path}"
        assert preamble[1] == 2, f"Unsupported dat file version {preamble[1]}: {dat_path}"
        n_header_bytes = int(preamble[2])
        # Seek back and read only the header
        f.seek(0)
        raw_header = np.frombuffer(f.read(n_header_bytes), dtype=np.uint32)
    file_size_bytes = dat_path.stat().st_size
    n_header_entries = n_header_bytes // 4
    pairs = raw_header[3:n_header_entries - 1].reshape(-1, 2)
    field_map = {int(p[0]): float(p[1]) for p in pairs}
    first_cycle_offset = field_map[0]  # firstCycleOffsetBytes
    bytes_per_cycle    = field_map[3]  # bytesPerCycle
    return int((file_size_bytes - first_cycle_offset) / bytes_per_cycle)


def get_trial_num_cycles(dat_paths, n_trials):
    """
    Read numCycles from each .dat file and return a per-trial array of length n_trials.

    Trial index is parsed from the TRIAL\\d+ token in the filename (0-based).
    Any trial index with no corresponding .dat file is filled with 0.

    Parameters
    ----------
    dat_paths : list of Path
        Unsorted list of .dat file paths for one DMD.
    n_trials : int
        Total number of trials expected.

    Returns
    -------
    trial_num_cycles : np.ndarray of int, shape (n_trials,)
    """
    trial_num_cycles = np.zeros(n_trials, dtype=int)
    for dat_path in dat_paths:
        m = re.search(r'TRIAL(\d+)', dat_path.name, re.IGNORECASE)
        if m is None:
            print(f'WARNING: could not parse trial index from {dat_path.name}, skipping.')
            continue
        trial_idx = int(m.group(1)) - 1  # MATLAB 1-based -> Python 0-based
        if trial_idx < 0 or trial_idx >= n_trials:
            print(f'WARNING: trial index {trial_idx} in {dat_path.name} exceeds n_trials={n_trials}, skipping.')
            continue
        trial_num_cycles[trial_idx] = read_dat_num_cycles(dat_path)
    n_missing = int(np.sum(trial_num_cycles == 0))
    if n_missing > 0:
        print(f'WARNING: {n_missing}/{n_trials} trials have no .dat file; numCycles set to 0 for those trials.')
    return trial_num_cycles


def parse_cycle_clock(signal, times):
    """
    Extract primary plane cycle start timestamps from the HARP cycle clock
    square wave.

    Only rising edges (0 -> 1) are used; they mark the start of each imaging
    cycle. Falling edges are present in the signal but do not reliably mark
    cycle ends and are ignored.

    If the signal is already high at the first sample (recording began
    mid-cycle), a synthetic rising edge is prepended at times[0].

    Parameters
    ----------
    signal : np.ndarray
        Square wave of length hrp_smp; 1 = cycle active, 0 = cycle inactive.
    times : np.ndarray
        HARP timestamps of length hrp_smp corresponding to each sample in signal.

    Returns
    -------
    cycle_starts : np.ndarray
        Timestamp of each rising edge (start of cycle).
    inter_cycle_periods : np.ndarray
        Time elapsed between the start of cycle i and the start of cycle i+1.
        Length is len(cycle_starts) - 1.
    """
    transitions = np.diff(signal.astype(int))

    # +1 offset: np.diff index i corresponds to the transition between sample i and i+1,
    # so the new state begins at sample i+1
    rising_idxs = np.where(transitions == 1)[0] + 1
    cycle_starts = times[rising_idxs]

    # Falling edges are detected only to check whether the signal started high;
    # they are otherwise unused.
    if signal[0] == 1:
        cycle_starts = np.concatenate([[times[0]], cycle_starts])

    inter_cycle_periods = cycle_starts[1:] - cycle_starts[:-1]

    return cycle_starts, inter_cycle_periods


def find_inter_trial_gap_indices(inter_cycle_periods, n_expected_trials, gap_multiplier=5.0):
    """
    Identify which inter-cycle periods correspond to inter-trial pauses.

    In trial-based sessions, imaging pauses for several hundred milliseconds
    between trials, producing outlier periods between cycle starts that are much
    longer than the typical intra-trial cycle period. This function finds those
    long periods and returns the cycle index at which each trial begins.

    This method is robust to pre-bug-fix data because it relies on the shape of
    the period distribution, not on the total count of cycle pulses.

    Parameters
    ----------
    inter_cycle_periods : np.ndarray
        Time between consecutive cycle starts (output of parse_cycle_clock).
    n_expected_trials : int
        Expected number of trials, used to validate detection results.
    gap_multiplier : float
        A period is considered an inter-trial pause if it exceeds
        gap_multiplier * median(inter_cycle_periods). Default is 5.0.

    Returns
    -------
    trial_start_cycle_idxs : np.ndarray or None
        Index into cycle_starts for the first cycle of each trial.
        Returns None if the detected number of long periods does not match
        n_expected_trials - 1 (detection failure).
    """
    if n_expected_trials == 1:
        # Continuous / single-trial session: no inter-trial gaps to find
        return None

    threshold = gap_multiplier * np.median(inter_cycle_periods)
    long_gap_positions = np.where(inter_cycle_periods > threshold)[0]

    n_expected_long_gaps = n_expected_trials - 1
    if len(long_gap_positions) != n_expected_long_gaps:
        warnings.warn(
            f"Gap detection found {len(long_gap_positions)} inter-trial gaps "
            f"but expected {n_expected_long_gaps}. "
            f"Falling back to line-count method."
        )
        return None

    # long_gap_positions[i] is the index of the period between cycle i and cycle i+1,
    # so the next trial begins at cycle index long_gap_positions[i] + 1
    trial_start_cycle_idxs = np.concatenate([[0], long_gap_positions + 1])
    return trial_start_cycle_idxs


def reconcile_cycle_clock_trials(
    cycle_starts,
    inter_cycle_periods,
    n_summary_trials,
    highest_dat_trial,
    gap_multiplier=5.0,
):
    """Remove one corroborated leading HARP cycle group before trial assignment.

    The processed summary supplies the expected trial count, HARP gaps supply
    candidate trial groups, and the highest present .dat trial number confirms
    that the selected acquisition reaches the end of the processed summary.
    No correction is made unless all three sources support exactly one extra
    leading HARP group.
    """
    threshold = gap_multiplier * np.median(inter_cycle_periods)
    long_gap_positions = np.where(inter_cycle_periods > threshold)[0]
    n_harp_groups = len(long_gap_positions) + 1
    reconciliation_qc = {
        'harp_group_count_before_reconciliation': n_harp_groups,
        'summary_trial_count': int(n_summary_trials),
        'highest_dat_trial': highest_dat_trial,
        'removed_leading_cycle_count': 0,
    }

    if n_harp_groups == n_summary_trials:
        return cycle_starts, inter_cycle_periods, reconciliation_qc

    if (
        n_harp_groups == n_summary_trials + 1
        and highest_dat_trial == n_summary_trials
    ):
        first_real_group_idx = int(long_gap_positions[0] + 1)
        reconciliation_qc['removed_leading_cycle_count'] = first_real_group_idx
        warnings.warn(
            f"HARP contains {n_harp_groups} cycle groups for {n_summary_trials} "
            f"processed trials, while .dat files reach trial {highest_dat_trial}. "
            f"Removing the unmatched leading group of {first_real_group_idx} "
            f"cycle pulses before gap-based trial assignment.",
            RuntimeWarning,
        )
        cycle_starts = cycle_starts[first_real_group_idx:]
        inter_cycle_periods = np.diff(cycle_starts)

    return cycle_starts, inter_cycle_periods, reconciliation_qc


def get_expected_cycle_count(frame_line_idxs_chunk, lines_per_cycle):
    """
    Estimate the number of imaging cycles in a trial from its scan line indices.

    linesPerCycle is the number of scan lines per imaging cycle. frame_line_idxs
    maps each fluorescence sample to a 1-based scan line index and resets to 1
    at the start of each trial. The maximum line index in a trial chunk, divided
    by lines_per_cycle, gives the number of cycles that occurred.

    Parameters
    ----------
    frame_line_idxs_chunk : np.ndarray
        Slice of frame_line_idxs for a single trial (1-based scan line indices).
    lines_per_cycle : int or float
        Number of scan lines per imaging cycle, from the .meta h5 file.

    Returns
    -------
    int
        Estimated number of imaging cycles in this trial.
    """
    if len(frame_line_idxs_chunk) == 0:
        return 0
    return int(np.ceil(np.max(frame_line_idxs_chunk) / lines_per_cycle))


def assign_cycles_by_gap_detection(
    cycle_starts, inter_cycle_periods,
    trial_num_frames, frame_line_idxs, lines_per_cycle,
    gap_multiplier=5.0,
    trial_num_cycles=None
):
    """
    Primary method: segment the cycle clock into per-trial groups using
    inter-trial gap detection.

    After segmenting, each trial's detected cycle count is compared against the
    estimate from get_expected_cycle_count() as a sanity check. Any mismatch
    triggers a warning but does not abort.

    Parameters
    ----------
    cycle_starts : np.ndarray
        Timestamps of each cycle start (rising edge).
    inter_cycle_periods : np.ndarray
        Time between consecutive cycle starts.
    trial_num_frames : np.ndarray
        Number of fluorescence samples in each trial.
    frame_line_idxs : np.ndarray
        Scan line indices for all fluorescence samples (1-based, resets per trial).
    lines_per_cycle : int or float
        Number of scan lines per imaging cycle.
    gap_multiplier : float
        Threshold multiplier for inter-trial gap detection.
    trial_num_cycles : np.ndarray or None
        Per-trial cycle counts from .dat files. Where non-zero, used in place of
        get_expected_cycle_count() for the sanity check.

    Returns
    -------
    trial_cycle_groups : list of np.ndarray or None
        Per-trial list of cycle_starts arrays.
        Returns None if gap detection fails.
    """
    n_trials = len(trial_num_frames)
    trial_start_cycle_idxs = find_inter_trial_gap_indices(
        inter_cycle_periods, n_trials, gap_multiplier
    )

    if trial_start_cycle_idxs is None:
        return None, None

    # End of trial i is the start of trial i+1; last trial ends at the final cycle
    trial_end_cycle_idxs = np.concatenate([trial_start_cycle_idxs[1:], [len(cycle_starts)]])

    # Precompute per-trial frame index boundaries for the sanity check
    frame_boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])

    trial_cycle_groups = []
    trial_expected_cycles = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start_idx = trial_start_cycle_idxs[i]
        end_idx = trial_end_cycle_idxs[i]

        trial_cycle_starts = cycle_starts[start_idx:end_idx]

        # Record the expected cycle count used for this trial
        trial_line_idxs = frame_line_idxs[frame_boundaries[i]:frame_boundaries[i + 1]]
        if trial_num_cycles is not None and trial_num_cycles[i] > 0:
            expected_cycles = int(trial_num_cycles[i])
        else:
            expected_cycles = get_expected_cycle_count(trial_line_idxs, lines_per_cycle)
        trial_expected_cycles[i] = expected_cycles
        detected_cycles = len(trial_cycle_starts)

        if detected_cycles != expected_cycles:
            warnings.warn(
                f"Trial {i}: gap detection found {detected_cycles} cycles but "
                f"expected {expected_cycles}. Results may be inaccurate."
            )

        trial_cycle_groups.append(trial_cycle_starts)

    return trial_cycle_groups, trial_expected_cycles


def assign_cycles_by_line_count(
    cycle_starts,
    trial_num_frames, frame_line_idxs, lines_per_cycle,
    trial_num_cycles=None
):
    """
    Fallback method: assign cycles to trials by consuming the expected number of
    cycles per trial, as estimated from scan line indices via get_expected_cycle_count().

    Used when inter-trial gap detection fails, or for continuous (single-trial)
    sessions where there are no inter-trial pauses to detect.

    Pre-bug-fix data warning: before the SLAP2 cycle clock bug fix, cycle pulses
    were sent at an incorrect frequency, so the total pulse count may be lower
    than the true number of cycles. If total available pulses are substantially
    below the total expected cycle count, a warning is emitted.

    Parameters
    ----------
    cycle_starts : np.ndarray
        Timestamps of each cycle start (rising edge).
    trial_num_frames : np.ndarray
        Number of fluorescence samples in each trial.
    frame_line_idxs : np.ndarray
        Scan line indices for all fluorescence samples (1-based, resets per trial).
    lines_per_cycle : int or float
        Number of scan lines per imaging cycle.
    trial_num_cycles : np.ndarray or None
        Per-trial cycle counts from .dat files. Where non-zero, used directly
        instead of get_expected_cycle_count().

    Returns
    -------
    trial_cycle_groups : list of np.ndarray
        Per-trial list of cycle_starts arrays.
    """
    frame_boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])

    # Check total available pulses vs total expected cycles as a pre-bug-fix indicator
    total_expected_cycles = sum(
        int(trial_num_cycles[i]) if (trial_num_cycles is not None and trial_num_cycles[i] > 0)
        else get_expected_cycle_count(
            frame_line_idxs[frame_boundaries[i]:frame_boundaries[i + 1]],
            lines_per_cycle
        )
        for i in range(len(trial_num_frames))
    )
    total_available_pulses = len(cycle_starts)

    if total_available_pulses < 0.9 * total_expected_cycles:
        warnings.warn(
            f"Only {total_available_pulses} cycle pulses detected but "
            f"{total_expected_cycles} cycles expected from line indices. "
            f"Possible causes: session ended early (last trial will be clamped to "
            f"available pulses), or pre-bug-fix data where cycle pulses were emitted "
            f"at an incorrect frequency."
        )

    cycle_idx = 0
    trial_cycle_groups = []
    trial_expected_cycles = np.zeros(len(trial_num_frames), dtype=int)

    for i in range(len(trial_num_frames)):
        trial_line_idxs = frame_line_idxs[frame_boundaries[i]:frame_boundaries[i + 1]]
        if trial_num_cycles is not None and trial_num_cycles[i] > 0:
            n_cycles = int(trial_num_cycles[i])
        else:
            n_cycles = get_expected_cycle_count(trial_line_idxs, lines_per_cycle)
        trial_expected_cycles[i] = n_cycles

        if cycle_idx + n_cycles > total_available_pulses:
            remaining = total_available_pulses - cycle_idx
            # HARP stopped recording before this trial finished (or before it
            # started). Clamp to whatever pulses remain rather than crashing.
            # This can happen on the last recorded trial when HARP stops a few
            # cycles short of what the .dat file reports.
            if remaining > 0:
                warnings.warn(
                    f"Trial {i}: HARP recording ended during this trial "
                    f"(expected {n_cycles} cycles, only {remaining} pulses remain). "
                    f"Clamping to {remaining} available pulses."
                )
            else:
                warnings.warn(
                    f"Trial {i}: HARP recording ended before this trial started "
                    f"(0 pulses remain). Timestamps for this trial will be NaN."
                )
            n_cycles = remaining
            trial_expected_cycles[i] = n_cycles

        trial_cycle_groups.append(cycle_starts[cycle_idx:cycle_idx + n_cycles])
        cycle_idx += n_cycles

    return trial_cycle_groups, trial_expected_cycles


def assign_samples_to_cycles(frame_line_idxs_chunk, lines_per_cycle):
    """
    Assign each fluorescence sample in a trial to its imaging cycle.

    Uses the scan line index of each sample and the known number of scan lines
    per cycle to determine which cycle (1-based) produced each sample. This
    assignment is derived entirely from the SLAP2 line indices and is independent
    of the HARP clock.

    Parameters
    ----------
    frame_line_idxs_chunk : np.ndarray
        Slice of frame_line_idxs for a single trial (1-based scan line indices).
    lines_per_cycle : int or float
        Number of scan lines per imaging cycle.

    Returns
    -------
    sample_cycle_idxs : np.ndarray of int
        1-based cycle index for each fluorescence sample in the trial.
    """
    return np.ceil(frame_line_idxs_chunk / lines_per_cycle).astype(int)


def normalize_continued_trial_line_indices(
    frame_line_idxs,
    trial_num_frames,
    plane_name="SLAP2",
    continuation_gap_multiplier=1.5,
):
    """Fix scan-line counters that locally continue across trial boundaries.

    ``frame_line_idxs`` should normally restart at 1 for each trial. For
    example, three trials might look like this::

        expected:  [1, 30, 60, 100] [1, 30, 60, 100] [1, 30, 60, 100]

    Occasionally the counter continues into the next trial instead. The small
    positive boundary gap is comparable to the ordinary spacing between samples::

        observed:  [1, 30, 60, 100] [101, 130, 160, 200] [1, 30, 60, 100]
        boundary:                    100 -> 101 (gap 1)

    The samples still belong to the correct trials because ``trial_num_frames``
    defines the trial boundaries. Only the second trial's line coordinates are
    wrong. This function subtracts 100 from that trial to restore::

        corrected: [1, 30, 60, 100] [1, 30, 60, 100] [1, 30, 60, 100]

    This matters for both planes. A bad primary-plane index can distort cycle
    assignment and the line-to-time map. A bad secondary-plane index can fall
    outside that map, causing many samples to receive the same clamped timestamp.

    The function estimates ordinary sample spacing from all positive within-trial
    line differences. A non-1 trial start is rebased only when it follows the
    previous non-empty trial's raw endpoint by a positive gap no larger than
    ``continuation_gap_multiplier`` times the 99th-percentile spacing. Raw
    endpoints are used so consecutive failed resets can be corrected in order.

    A large jump is left unchanged because it can represent a genuine partial
    trial whose early samples were not recorded::

        previous: [1, 30, 60, 100]
        partial:  [5000, 5030, 5060]  # gap 4900, not local continuation

    Empty trials are skipped when finding the previous observed endpoint. This
    can still detect a counter that paused across one or more empty trial entries.
    Ambiguous non-1 starts are preserved rather than modified.

    Parameters
    ----------
    frame_line_idxs : np.ndarray
        Concatenated scan-line indices for one plane.
    trial_num_frames : np.ndarray
        Number of samples in each trial.
    plane_name : str
        Plane label included in correction messages.
    continuation_gap_multiplier : float
        Multiplier applied to the 99th percentile of positive within-trial line
        differences to define the largest plausible continuation gap.

    Returns
    -------
    normalized : np.ndarray
        A copy of ``frame_line_idxs`` with clear continued-counter trials
        rebased to start at 1.
    corrections : list of dict
        One diagnostic record per corrected trial. Trial numbers are 1-based.
    """
    frame_line_idxs = np.asarray(frame_line_idxs)
    trial_num_frames = np.asarray(trial_num_frames)
    if int(np.sum(trial_num_frames)) != len(frame_line_idxs):
        raise ValueError(
            f"sum(trial_num_frames) = {np.sum(trial_num_frames)} does not equal "
            f"len(frame_line_idxs) = {len(frame_line_idxs)}."
        )

    if continuation_gap_multiplier <= 0:
        raise ValueError("continuation_gap_multiplier must be positive.")

    frame_boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])
    positive_within_trial_steps = []
    for i, n_frames in enumerate(trial_num_frames):
        if n_frames < 2:
            continue
        trial_lines = frame_line_idxs[frame_boundaries[i]:frame_boundaries[i + 1]]
        trial_diffs = np.diff(trial_lines.astype(np.int64, copy=False))
        positive_within_trial_steps.extend(trial_diffs[trial_diffs > 0])

    if not positive_within_trial_steps:
        warnings.warn(
            f"{plane_name}: cannot detect continued trial line indices because "
            f"no positive within-trial line differences are available. "
            f"No indices were changed."
        )
        return frame_line_idxs.copy(), []

    reference_step = float(np.percentile(positive_within_trial_steps, 99))
    max_continuation_gap = continuation_gap_multiplier * reference_step
    normalized = frame_line_idxs.copy()
    corrections = []
    previous_raw_last = None
    for i, n_frames in enumerate(trial_num_frames):
        if n_frames == 0:
            continue
        start = frame_boundaries[i]
        stop = frame_boundaries[i + 1]
        raw_trial_lines = frame_line_idxs[start:stop]
        raw_trial_lines_int = raw_trial_lines.astype(np.int64, copy=False)
        original_first = int(raw_trial_lines_int[0])
        original_last = int(raw_trial_lines_int[-1])
        boundary_gap = (
            original_first - previous_raw_last
            if previous_raw_last is not None
            else None
        )
        previous_raw_last = original_last

        if (
            original_first == 1
            or boundary_gap is None
            or not 0 < boundary_gap <= max_continuation_gap
            or not np.all(np.diff(raw_trial_lines_int) >= 0)
        ):
            continue

        offset = original_first - 1
        normalized[start:stop] = raw_trial_lines - offset
        corrected_last = int(normalized[stop - 1])
        correction = {
            'trial_number': i + 1,
            'offset': offset,
            'original_range': (original_first, original_last),
            'corrected_range': (1, corrected_last),
            'boundary_gap': boundary_gap,
            'max_continuation_gap': max_continuation_gap,
        }
        corrections.append(correction)
        print(
            f"{plane_name} TRIAL LINE-INDEX ISSUE: trial {i + 1} started at "
            f"{original_first}, only {boundary_gap} lines after the previous "
            f"observed endpoint (continuation threshold "
            f"{max_continuation_gap:.1f}); the scan-line counter appears not to "
            f"have reset. Subtracted offset {offset} from all {int(n_frames)} "
            f"samples, changing range [{original_first}, {original_last}] to "
            f"[1, {corrected_last}]."
        )

    return normalized, corrections


def _build_trial_line_time_map(
    trial_cycle_starts, effective_lpc, n_detected_cycles
):
    """
    Build one line-to-time control point at each scanner cycle boundary.

    Each HARP cycle start is paired with the estimated scan line where that
    cycle begins. One final boundary is added using the mean cycle period. This
    represents a boundary shared by adjacent cycles only once, avoiding the
    flat time intervals produced when one cycle's last sampled line and the
    next cycle's first sampled line are both assigned the same timestamp.

    Parameters
    ----------
    trial_cycle_starts : np.ndarray
        HARP timestamps of each cycle's rising edge within this trial.
    effective_lpc : float
        Average scan lines per cycle for this trial
        (= primary_trial_line_idxs.max() / n_detected_cycles).
    n_detected_cycles : int
        Number of HARP cycles detected for this trial.

    Returns
    -------
    control_line_idxs : np.ndarray
        Strictly increasing estimated cycle-boundary line indices.
        Length == n_detected_cycles + 1.
    control_timestamps : np.ndarray
        HARP timestamps corresponding to each cycle boundary.
        Length == n_detected_cycles + 1.
    """
    if n_detected_cycles > 1:
        mean_cycle_period = float(np.mean(np.diff(trial_cycle_starts)))
    else:
        mean_cycle_period = 0.0
    control_line_idxs = 1.0 + np.arange(n_detected_cycles + 1) * effective_lpc
    control_timestamps = np.append(
        trial_cycle_starts, trial_cycle_starts[-1] + mean_cycle_period
    )

    return control_line_idxs, control_timestamps


def validate_cycle_groups(trial_cycle_groups):
    """
    Assert that the cycle timing structure of each trial is internally consistent.

    Checks per trial:
    - at least one cycle start is present
    - cycle start times are strictly increasing (no duplicates or reversals)

    Parameters
    ----------
    trial_cycle_groups : list of np.ndarray
        Per-trial list of cycle_starts arrays.
    """
    for i, trial_cycle_starts in enumerate(trial_cycle_groups):
        if len(trial_cycle_starts) == 0:
            warnings.warn(f"Trial {i}: no cycle starts found (HARP likely stopped before this trial).")
            continue
        if len(trial_cycle_starts) > 1:
            assert np.all(np.diff(trial_cycle_starts) > 0), (
                f"Trial {i}: cycle start times are not strictly increasing."
            )


def build_timestamps(trial_cycle_groups, trial_num_frames, primary_frame_line_idxs):
    """
    Construct the primary plane timestamp array and per-trial line-to-time maps.

    For each trial, attempts per-cycle interpolation: samples are assigned to
    their cycle via assign_samples_to_cycles(), and each group of samples is
    interpolated uniformly between that cycle's own HARP start and end timestamp.
    This uses the full precision of the HARP cycle clock.

    ``effective_lpc`` is derived from the detected HARP cycle count, so ordinary
    positive line indices span exactly that many cycle buckets. A trial-span
    interpolation remains as a defensive fallback if the derived bucket count
    nevertheless differs.

    The final cycle duration is estimated from the mean period between detected
    cycle starts. With only one detected cycle there is no measured period, so
    the current estimate is zero. That case cannot support strictly increasing
    secondary-plane timestamps.

    Also builds a per-trial scan-line-to-time map (via _build_trial_line_time_map)
    for use by get_slap2_secondary_plane_timestamps.

    Parameters
    ----------
    trial_cycle_groups : list of np.ndarray
        Per-trial list of cycle_starts arrays, as returned by
        assign_cycles_by_gap_detection or assign_cycles_by_line_count.
    trial_num_frames : np.ndarray
        Number of primary plane fluorescence samples in each trial.
    primary_frame_line_idxs : np.ndarray
        Scan line indices for all primary plane fluorescence samples
        (1-based, resets per trial).

    Returns
    -------
    timestamps : np.ndarray
        Timestamp for each primary plane fluorescence sample.
        Length == sum(trial_num_frames).
    trial_line_time_maps : list of (np.ndarray, np.ndarray) or None
        Per-trial scan-line-to-time control point maps for secondary plane
        alignment. Each entry is a (control_line_idxs, control_timestamps) tuple,
        or None for trials where the primary plane had no samples.
        Length == len(trial_cycle_groups).
    """
    validate_cycle_groups(trial_cycle_groups)

    frame_boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])
    all_timestamps = []
    trial_line_time_maps = []

    for i, trial_cycle_starts in enumerate(trial_cycle_groups):
        primary_trial_line_idxs = primary_frame_line_idxs[
            frame_boundaries[i]:frame_boundaries[i + 1]
        ]
        n_frames = trial_num_frames[i]
        n_detected_cycles = len(trial_cycle_starts)

        # Trials with no frames contribute nothing to the timestamp array.
        # A None sentinel is stored so the secondary plane function can detect
        # and handle this trial explicitly.
        if n_frames == 0:
            trial_line_time_maps.append(None)
            continue

        # 0 detected cycles with non-zero frames: HARP ran out before this trial.
        # No timing anchor exists; emit NaN timestamps.
        if n_detected_cycles == 0:
            warnings.warn(
                f"Trial {i}: 0 HARP cycles for {n_frames} frames "
                f"(HARP likely stopped before this trial). "
                f"Assigning NaN timestamps."
            )
            all_timestamps.append(np.full(n_frames, np.nan))
            trial_line_time_maps.append(None)
            continue
        # Use effective_lpc derived from this trial's actual max line index rather
        # than the nominal lines_per_cycle from metadata. The nominal value may be
        # slightly larger than the actual lines scanned per cycle, causing cumulative
        # drift that makes ceil(max_line_idx / nominal_lpc) undercount by 1.
        # effective_lpc = max(primary_trial_line_idxs) / n_detected_cycles guarantees
        # that n_cycle_buckets always equals n_detected_cycles exactly.
        effective_lpc = int(primary_trial_line_idxs.max()) / n_detected_cycles
        sample_cycle_idxs = assign_samples_to_cycles(primary_trial_line_idxs, effective_lpc)
        n_cycle_buckets = int(sample_cycle_idxs.max())

        # Derive end time for each cycle: start of the next cycle, or estimated
        # from the mean period for the final cycle.
        if n_detected_cycles > 1:
            mean_cycle_period = float(np.mean(np.diff(trial_cycle_starts)))
        else:
            mean_cycle_period = 0.0
        cycle_period_ends = np.append(
            trial_cycle_starts[1:], trial_cycle_starts[-1] + mean_cycle_period
        )

        if n_cycle_buckets == n_detected_cycles:
            # Per-cycle interpolation: each cycle spans [cycle_starts[c], cycle_starts[c+1])
            trial_timestamps = np.empty(n_frames)
            for c in range(1, n_detected_cycles + 1):
                mask = sample_cycle_idxs == c
                # Use endpoint=False for all but the last cycle so that the
                # end of cycle c (== start of cycle c+1) is not duplicated in
                # the timestamp array. The final cycle uses endpoint=True so
                # its last sample lands exactly at the estimated trial end.
                trial_timestamps[mask] = np.linspace(
                    trial_cycle_starts[c - 1], cycle_period_ends[c - 1], int(np.sum(mask)),
                    endpoint=(c == n_detected_cycles),
                )
            all_timestamps.append(trial_timestamps)

            # Build one control point per scanner cycle boundary so interpolation
            # remains continuous and strictly increasing between HARP starts.
            control_line_idxs, control_timestamps = _build_trial_line_time_map(
                trial_cycle_starts, effective_lpc, n_detected_cycles
            )
            trial_line_time_maps.append((control_line_idxs, control_timestamps))
        else:
            # Fallback: interpolate uniformly across the full trial span
            warnings.warn(
                f"Trial {i}: {n_cycle_buckets} cycle buckets from line indices but "
                f"{n_detected_cycles} HARP cycles detected. "
                f"Falling back to trial-span interpolation."
            )
            all_timestamps.append(
                np.linspace(trial_cycle_starts[0], cycle_period_ends[-1], n_frames)
            )
            # Fallback line→time map: two control points spanning the full trial
            trial_line_time_maps.append((
                np.array([float(primary_trial_line_idxs.min()),
                          float(primary_trial_line_idxs.max())]),
                np.array([trial_cycle_starts[0], cycle_period_ends[-1]])
            ))

    return np.concatenate(all_timestamps), trial_line_time_maps


def validate_inputs(frame_line_idxs, trial_num_frames):
    """
    Sanity check inputs before running alignment.

    Verifies:
    - sum(trial_num_frames) == len(frame_line_idxs)
    - frame_line_idxs resets to 1 at the start of each trial, consistent with
      the 1-based MATLAB indexing used by SLAP2 control software

    Parameters
    ----------
    frame_line_idxs : np.ndarray
        Scan line indices for each fluorescence sample (1-based, resets per trial).
    trial_num_frames : np.ndarray
        Number of fluorescence samples per trial.
    """
    if np.sum(trial_num_frames) != len(frame_line_idxs):
        raise ValueError(
            f"sum(trial_num_frames) = {np.sum(trial_num_frames)} does not equal "
            f"len(frame_line_idxs) = {len(frame_line_idxs)}."
        )

    frame_boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])
    for i in range(len(trial_num_frames)):
        if trial_num_frames[i] == 0:
            continue  # no frames for this trial; nothing to check
        first_line_idx = frame_line_idxs[frame_boundaries[i]]
        if first_line_idx != 1:
            warnings.warn(
                f"Trial {i}: frame_line_idxs does not reset to 1 at trial start "
                f"(found {first_line_idx}). Expected 1-based indexing from SLAP2."
            )


def get_slap2_primary_plane_timestamps(
    primary_frame_line_idxs,
    primary_trial_num_frames,
    primary_lines_per_cycle,
    slap2_cycle_clock_signal,
    slap2_cycle_clock_times,
    gap_multiplier=5.0,
    trial_num_cycles=None,
    highest_dat_trial=None,
):
    """
    Align primary plane SLAP2 fluorescence samples to absolute HARP timestamps.

    The primary plane is the DMD whose cycle clock is recorded by HARP. Uses the
    HARP cycle clock square wave to assign a timestamp to every fluorescence
    sample. Also produces per-trial scan-line-to-time maps required to align the
    secondary plane via get_slap2_secondary_plane_timestamps.

    Call ``normalize_continued_trial_line_indices`` on this plane's line indices
    before calling this function. The primary and secondary planes must each be
    normalized independently because their counters can fail to reset on
    different trials.

    The alignment strategy depends on session type:

    - Continuous (single-trial) session:
        Gap detection is not applicable. Cycles are assigned directly using
        the line-count method (assign_cycles_by_line_count).

    - Trial-based (multi-trial) session:
        1. (Primary) Gap detection: segment the cycle stream at long inter-trial
           pauses (assign_cycles_by_gap_detection). Robust to pre-bug-fix data
           since it does not rely on cycle pulse counts.
        2. (Fallback) Line-count: if gap detection fails, estimate cycles per
           trial from scan line indices (assign_cycles_by_line_count). May be
           unreliable for pre-bug-fix data; a warning is emitted in that case.

    Trials with one detected HARP cycle do not provide enough information to
    estimate that cycle's duration. The current implementation gives such a
    trial zero duration; its line-to-time map is unsuitable for secondary-plane
    alignment.

    Parameters
    ----------
    primary_frame_line_idxs : np.ndarray
        Scan line index for each primary plane fluorescence sample
        (1-based, resets per trial). Length == primary_flr_smp.
    primary_trial_num_frames : np.ndarray
        Number of primary plane fluorescence samples per trial.
        sum == primary_flr_smp.
    primary_lines_per_cycle : int or float
        Number of scan lines per primary plane imaging cycle,
        from the .meta h5 file.
    slap2_cycle_clock_signal : np.ndarray
        Square wave of length hrp_smp; rising edges mark primary plane
        imaging cycle starts.
    slap2_cycle_clock_times : np.ndarray
        HARP timestamps of length hrp_smp for each sample in the cycle
        clock signal.
    gap_multiplier : float
        Multiplier over the median cycle period used to detect inter-trial
        pauses. Increase if trials are incorrectly split; decrease if pauses
        are not being detected. Default is 5.0.
    trial_num_cycles : np.ndarray or None
        Per-trial cycle counts read from .dat files (e.g. via get_trial_num_cycles).
        Where non-zero, used in place of the LPC-based estimate for both the
        gap-detection sanity check and the line-count cycle assignment.
        Trials with a value of 0 fall back to the LPC estimate.
    highest_dat_trial : int or None
        Highest trial number present among the selected acquisition's .dat files
        across all DMDs. Used only to corroborate removal of one unmatched leading
        HARP cycle group.

    Returns
    -------
    primary_timestamps : np.ndarray
        HARP-referenced timestamp for each primary plane fluorescence sample.
        Length == primary_flr_smp.
    trial_line_time_maps : list of (np.ndarray, np.ndarray) or None
        Per-trial scan-line-to-time control point maps. Each entry is a tuple of
        (control_line_idxs, control_timestamps), or None for trials where the
        primary plane had no samples. Pass directly to
        get_slap2_secondary_plane_timestamps.
        Length == len(primary_trial_num_frames).
    sync_qc_values : dict
        Intermediate data structures for use with plot_slap2_sync_qc. Keys:
          'cycle_starts'              : np.ndarray
          'inter_cycle_periods'       : np.ndarray
          'trial_cycle_groups'        : list of np.ndarray
          'trial_expected_cycles_qc'  : np.ndarray of int
              Per-trial expected cycle count as actually used during processing
              (from .dat numCycles where available, otherwise LPC-based estimate).
          'segmentation_method_qc'    : str
              One of 'gap_detection', 'line_count_fallback',
              'line_count_continuous'.
          'gap_threshold_qc'          : float
              gap_multiplier * median(inter_cycle_periods).
          'eff_lpc_per_trial_qc'      : np.ndarray of float
              Effective lines-per-cycle per trial: max scan line / detected cycles.
          'cycle_clock_reconciliation_qc' : dict
              Counts from the HARP groups, processed summary, and .dat trial span,
              plus the number of leading cycle pulses removed before assignment.
    """
    validate_inputs(primary_frame_line_idxs, primary_trial_num_frames)

    cycle_starts, inter_cycle_periods = parse_cycle_clock(
        slap2_cycle_clock_signal, slap2_cycle_clock_times
    )
    print(f"Primary plane: {len(cycle_starts)} cycles detected from HARP clock.")

    n_trials = len(primary_trial_num_frames)
    cycle_starts, inter_cycle_periods, reconciliation_qc = reconcile_cycle_clock_trials(
        cycle_starts,
        inter_cycle_periods,
        n_trials,
        highest_dat_trial,
        gap_multiplier=gap_multiplier,
    )
    gap_threshold_qc = gap_multiplier * float(np.median(inter_cycle_periods))

    # For multi-trial sessions, attempt gap-based segmentation first
    trial_cycle_groups = None
    segmentation_method_qc = None
    trial_expected_cycles = None
    if n_trials > 1:
        trial_cycle_groups, trial_expected_cycles = assign_cycles_by_gap_detection(
            cycle_starts, inter_cycle_periods,
            primary_trial_num_frames, primary_frame_line_idxs, primary_lines_per_cycle,
            gap_multiplier=gap_multiplier,
            trial_num_cycles=trial_num_cycles
        )

    # Fall back to line-count if gap detection failed or session is continuous
    if trial_cycle_groups is None:
        if n_trials == 1:
            segmentation_method_qc = 'line_count_continuous'
            print(
                f"Continuous session: assigning {len(cycle_starts)} cycles to "
                f"1 trial using line-count method."
            )
        else:
            segmentation_method_qc = 'line_count_fallback'
            print(
                f"Gap detection failed: assigning {len(cycle_starts)} cycles "
                f"across {n_trials} trials using line-count method."
            )
        trial_cycle_groups, trial_expected_cycles = assign_cycles_by_line_count(
            cycle_starts,
            primary_trial_num_frames, primary_frame_line_idxs, primary_lines_per_cycle,
            trial_num_cycles=trial_num_cycles
        )
    else:
        segmentation_method_qc = 'gap_detection'
        print(
            f"Trial-based session: gap detection assigned {len(cycle_starts)} "
            f"cycles across {n_trials} trials."
        )

    primary_timestamps, trial_line_time_maps = build_timestamps(
        trial_cycle_groups, primary_trial_num_frames, primary_frame_line_idxs
    )

    expected_length = int(np.sum(primary_trial_num_frames))
    assert len(primary_timestamps) == expected_length, (
        f"Output timestamp length {len(primary_timestamps)} does not match "
        f"expected {expected_length}."
    )
    print(f"Primary plane alignment complete: {expected_length} samples timestamped.")

    # Compute effective LPC per trial for QC: max scan line / detected cycle count.
    # This matches the value used inside build_timestamps and avoids recomputing it there.
    frame_boundaries_qc = np.concatenate([[0], np.cumsum(primary_trial_num_frames)])
    eff_lpc_per_trial_qc = np.array([
        float(primary_frame_line_idxs[frame_boundaries_qc[i]:frame_boundaries_qc[i + 1]].max())
        / len(trial_cycle_groups[i])
        if len(trial_cycle_groups[i]) > 0
        and frame_boundaries_qc[i] < frame_boundaries_qc[i + 1]
        else np.nan
        for i in range(n_trials)
    ])

    sync_qc_values = {
        'cycle_starts': cycle_starts,
        'inter_cycle_periods': inter_cycle_periods,
        'trial_cycle_groups': trial_cycle_groups,
        'trial_expected_cycles_qc': trial_expected_cycles,
        'segmentation_method_qc': segmentation_method_qc,
        'gap_threshold_qc': gap_threshold_qc,
        'eff_lpc_per_trial_qc': eff_lpc_per_trial_qc,
        'cycle_clock_reconciliation_qc': reconciliation_qc,
    }
    return primary_timestamps, trial_line_time_maps, sync_qc_values


def get_slap2_secondary_plane_timestamps(
    secondary_frame_line_idxs,
    secondary_trial_num_frames,
    trial_line_time_maps
):
    """
    Align secondary plane SLAP2 fluorescence samples to absolute HARP timestamps.

    The secondary plane is the DMD whose cycle clock is not recorded by HARP.
    Its samples are aligned by mapping their scan line indices onto the
    scan-line-to-time maps derived from the primary plane's HARP timestamps by
    get_slap2_primary_plane_timestamps.

    Since both planes share the same scanner, the scan line index is a
    universal time coordinate across both planes. For each trial, the primary
    plane's (control_line_idxs, control_timestamps) control points define a
    piecewise-linear mapping from scan line to absolute time. Secondary plane
    samples are aligned by interpolating their scan line indices onto this map
    using np.interp.

    Call ``normalize_continued_trial_line_indices`` on the secondary plane's
    line indices before calling this function. The supplied maps must come from
    primary indices that were normalized independently before primary alignment.
    A non-empty secondary trial requires at least two distinct boundary times;
    equal or decreasing interpolated timestamps raise ``ValueError``.

    Parameters
    ----------
    secondary_frame_line_idxs : np.ndarray
        Scan line index for each secondary plane fluorescence sample
        (1-based, resets per trial). Length == secondary_flr_smp.
    secondary_trial_num_frames : np.ndarray
        Number of secondary plane fluorescence samples per trial.
        sum == secondary_flr_smp.
    trial_line_time_maps : list of (np.ndarray, np.ndarray) or None
        Per-trial scan-line-to-time control point maps produced by
        get_slap2_primary_plane_timestamps. Each entry is a tuple of
        (control_line_idxs, control_timestamps), or None if the primary plane
        had no samples in that trial.

    Returns
    -------
    secondary_timestamps : np.ndarray
        HARP-referenced timestamp for each secondary plane fluorescence sample.
        Length == secondary_flr_smp.
    """
    validate_inputs(secondary_frame_line_idxs, secondary_trial_num_frames)

    n_trials = len(secondary_trial_num_frames)
    if len(trial_line_time_maps) < n_trials:
        raise ValueError(
            f"trial_line_time_maps has {len(trial_line_time_maps)} entries but "
            f"secondary_trial_num_frames implies {n_trials} trials. "
            f"There must be at least one map entry per secondary trial."
        )

    print(
        f"Secondary plane: interpolating {int(np.sum(secondary_trial_num_frames))} samples "
        f"across {n_trials} trials onto primary plane timeline."
    )
    frame_boundaries = np.concatenate([[0], np.cumsum(secondary_trial_num_frames)])
    all_secondary_timestamps = []

    for i, line_time_map in enumerate(trial_line_time_maps[:n_trials]):
        secondary_trial_line_idxs = secondary_frame_line_idxs[
            frame_boundaries[i]:frame_boundaries[i + 1]
        ]
        n_frames = secondary_trial_num_frames[i]

        if n_frames == 0:
            continue

        if line_time_map is None:
            # Primary plane had 0 HARP cycles for this trial (HARP ran out).
            # Emit NaN timestamps rather than raising so the pipeline completes.
            warnings.warn(
                f"Trial {i}: secondary plane has {n_frames} samples but primary "
                f"plane had no HARP cycles. Assigning NaN timestamps."
            )
            all_secondary_timestamps.append(np.full(n_frames, np.nan))
            continue

        control_line_idxs, control_timestamps = line_time_map

        # Warn if any secondary plane scan lines fall outside the primary plane's
        # mapped line range for this trial; those samples will be clamped.
        map_min_line = control_line_idxs[0]
        map_max_line = control_line_idxs[-1]
        n_out_of_range = int(np.sum(
            (secondary_trial_line_idxs < map_min_line) |
            (secondary_trial_line_idxs > map_max_line)
        ))
        if n_out_of_range > 0:
            warnings.warn(
                f"Trial {i}: {n_out_of_range} secondary plane samples have scan "
                f"line indices outside the primary plane's mapped range "
                f"[{map_min_line:.0f}, {map_max_line:.0f}]. "
                f"Timestamps for these samples will be clamped to the nearest "
                f"boundary."
            )

        trial_secondary_timestamps = np.interp(
            secondary_trial_line_idxs, control_line_idxs, control_timestamps
        )
        timestamp_diffs = np.diff(trial_secondary_timestamps)
        if np.any(timestamp_diffs <= 0):
            raise ValueError(
                f"Trial {i}: secondary timestamp interpolation produced "
                f"{int(np.sum(timestamp_diffs == 0))} equal and "
                f"{int(np.sum(timestamp_diffs < 0))} decreasing adjacent values."
            )
        all_secondary_timestamps.append(trial_secondary_timestamps)

    secondary_timestamps = np.concatenate(all_secondary_timestamps)

    expected_length = int(np.sum(secondary_trial_num_frames))
    assert len(secondary_timestamps) == expected_length, (
        f"Output timestamp length {len(secondary_timestamps)} does not match "
        f"expected {expected_length}."
    )
    print(f"Secondary plane alignment complete: {expected_length} samples timestamped.")

    return secondary_timestamps


def plot_slap2_inputs_qc(
    slap2_cycle_clock_signal,
    slap2_cycle_clock_times,
    primary_frame_line_idxs,
    primary_trial_num_frames,
    sync_qc_values,
    secondary_frame_line_idxs=None,
    secondary_trial_num_frames=None,
    primary_raw_frame_line_idxs=None,
    secondary_raw_frame_line_idxs=None,
    clock_signal_window_s=0.5,
    qc_folder=None,
):
    """
    Produce a 4×3 QC figure showing inputs to the SLAP2 sync pipeline.

    Panels:
      Row 0 — Cycle clock:
        [0,0] Clock signal snippet with detected cycle starts
        [0,1] Cycle period distribution (log–log histogram)
        [0,2] Cycle periods over time (scatter with threshold)
      Row 1 — Frame line indices:
        [1,0] Frame line index ramp (first 5 trials)
        [1,1] First frame line index per trial (histogram)
        [1,2] Final frame line index per trial (histogram)
      Row 2 — Per-trial frame counts:
        [2,0] Primary frames per trial (line plot)
        [2,1] Secondary frames per trial (line plot, orange; if provided)
            Row 3 — Line-index normalization:
                [3,0] DMD1 raw and normalized final scan-line index per trial
                [3,1] DMD2 raw and normalized final scan-line index per trial

    Parameters
    ----------
    slap2_cycle_clock_signal : np.ndarray
    slap2_cycle_clock_times : np.ndarray
    primary_frame_line_idxs : np.ndarray
    primary_trial_num_frames : np.ndarray
    sync_qc_values : dict
        Third return value of get_slap2_primary_plane_timestamps.
    secondary_frame_line_idxs : np.ndarray, optional
        Normalized secondary-plane scan-line indices.
    secondary_trial_num_frames : np.ndarray, optional
        Number of secondary plane fluorescence samples per trial.
        If provided, plotted in orange in [2,1].
    primary_raw_frame_line_idxs : np.ndarray, optional
        Primary scan-line indices before trial normalization.
    secondary_raw_frame_line_idxs : np.ndarray, optional
        Secondary scan-line indices before trial normalization.
    clock_signal_window_s : float
        Duration (seconds) of the clock signal window shown in [0,0]. Default 0.5.
    qc_folder : str or Path, optional
        If provided, saved as ``slap2_inputs_qc.png`` inside this folder
        (dpi=150, tight layout) and closed. Returns None in this case.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    cycle_starts        = sync_qc_values['cycle_starts']
    inter_cycle_periods = sync_qc_values['inter_cycle_periods']
    gap_threshold_qc    = sync_qc_values['gap_threshold_qc']
    segmentation_method_qc = sync_qc_values['segmentation_method_qc']

    n_trials = len(primary_trial_num_frames)
    frame_boundaries = np.concatenate([[0], np.cumsum(primary_trial_num_frames)])

    # Shared reference lines: session-wide min/max of per-trial first and peak
    # scan line values, used identically on [1,0] and [1,1].
    trial_first_vals = [int(primary_frame_line_idxs[frame_boundaries[i]])
                        for i in range(n_trials) if frame_boundaries[i] < frame_boundaries[i + 1]]
    trial_max_vals   = [int(primary_frame_line_idxs[frame_boundaries[i]:frame_boundaries[i + 1]].max())
                        for i in range(n_trials) if frame_boundaries[i] < frame_boundaries[i + 1]]
    ref_lines = [
        (min(trial_first_vals), f'min first ({min(trial_first_vals)})', 'tab:blue'),
        (max(trial_first_vals), f'max first ({max(trial_first_vals)})', 'tab:cyan'),
        (min(trial_max_vals),   f'min peak ({min(trial_max_vals)})',    'tab:orange'),
        (max(trial_max_vals),   f'max peak ({max(trial_max_vals)})',    'tab:red'),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('SLAP2 Sync — Raw Inputs', fontsize=13, fontweight='bold')

    # ------------------------------------------------------------------
    # Row 0: Cycle clock
    # ------------------------------------------------------------------

    # [0,0] Clock signal snippet
    ax = axes[0, 0]
    t_anchor = cycle_starts[0] if len(cycle_starts) > 0 else slap2_cycle_clock_times[0]
    t_win_start = max(slap2_cycle_clock_times[0], t_anchor - 0.05)
    t_win_end = min(t_win_start + clock_signal_window_s, slap2_cycle_clock_times[-1])
    win = (slap2_cycle_clock_times >= t_win_start) & (slap2_cycle_clock_times <= t_win_end)
    ax.plot(slap2_cycle_clock_times[win] - t_win_start,
            slap2_cycle_clock_signal[win],
            'k-', linewidth=0.7, drawstyle='steps-post')
    cs_in_win = cycle_starts[(cycle_starts >= t_win_start) & (cycle_starts <= t_win_end)]
    for cs in cs_in_win:
        ax.axvline(cs - t_win_start, color='tab:blue', alpha=0.5, linewidth=0.7)
    ax.set_xlim(0, t_win_end - t_win_start)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    ax.set_yticks([0, 1])
    ax.set_title(f'Clock signal ({clock_signal_window_s:.0f}s from first cycle); '
                 f'{len(cs_in_win)} cycle starts shown')

    # [0,1] Cycle period histogram (log-log)
    ax = axes[0, 1]
    periods_ms = inter_cycle_periods * 1e3
    if len(periods_ms) > 0:
        n_bins = min(80, max(20, len(periods_ms) // 20))
        log_min = np.log10(max(periods_ms.min(), 1e-6))
        log_max = np.log10(periods_ms.max())
        bins = np.logspace(log_min, log_max, n_bins + 1)
        ax.hist(periods_ms, bins=bins, color='steelblue', edgecolor='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axvline(gap_threshold_qc * 1e3, color='black', linewidth=1.5,
                   linestyle='--',
                   label=f'Gap threshold ({gap_threshold_qc * 1e3:.1f} ms)')
    ax.set_xlabel('Period (ms)')
    ax.set_ylabel('Count (log)')
    ax.set_title('Cycle period distribution (log-log)')
    ax.legend(fontsize=8)

    # [0,2] Cycle periods over time (scatter with threshold, log y)
    ax = axes[0, 2]
    cyc_idx = np.arange(len(inter_cycle_periods))
    ax.scatter(cyc_idx, inter_cycle_periods * 1e3,
               s=4, c='steelblue', alpha=0.6)
    ax.axhline(gap_threshold_qc * 1e3, color='black', linewidth=1.2,
               linestyle='--', label='Threshold')
    ax.set_yscale('log')
    ax.set_xlabel('Cycle index')
    ax.set_ylabel('Period (ms)')
    ax.set_title(f'Cycle periods  [method: {segmentation_method_qc}]')
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Row 1: Frame line indices
    # ------------------------------------------------------------------

    # [1,0] Frame line index ramp — first 5 trials
    ax = axes[1, 0]
    n_show_trials = min(5, n_trials)
    show_end = frame_boundaries[n_show_trials]
    show_idx = np.arange(show_end)
    step = max(1, show_end // 5000)
    ax.plot(show_idx[::step], primary_frame_line_idxs[:show_end][::step],
            'k-', linewidth=0.4)
    for val, label, color in ref_lines:
        ax.axhline(val, linestyle='--', linewidth=0.9, color=color, alpha=0.8, label=label)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Scan line (1-based)')
    ax.set_title(f'Frame line indices — first {n_show_trials} of {n_trials} trial(s)')

    # [1,1] First frame line index per trial — distribution of the first
    # scan line value recorded in each trial, with min/max first reference lines.
    ax = axes[1, 1]
    trial_first_arr = np.array(trial_first_vals)
    ax.hist(trial_first_arr,
            bins=min(50, max(5, len(trial_first_arr) // 3)),
            color='steelblue', edgecolor='none')
    for val, label, color in ref_lines[:2]:   # only min/max first lines
        ax.axvline(val, linestyle='--', linewidth=0.9, color=color, alpha=0.8, label=label)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('First scan line index (1-based)')
    ax.set_ylabel('Trial count')
    ax.set_yscale('log')
    ax.set_title('First frame line index per trial')

    # [1,2] Final frame line index per trial — distribution of the last
    # (= peak) scan line value recorded in each trial, with min/max peak reference lines.
    ax = axes[1, 2]
    trial_final_vals = np.array([
        int(primary_frame_line_idxs[frame_boundaries[i + 1] - 1])
        for i in range(n_trials)
        if frame_boundaries[i] < frame_boundaries[i + 1]
    ])
    ax.hist(trial_final_vals,
            bins=min(50, max(5, len(trial_final_vals) // 3)),
            color='steelblue', edgecolor='none')
    for val, label, color in ref_lines[2:]:   # only min/max peak lines
        ax.axvline(val, linestyle='--', linewidth=0.9, color=color, alpha=0.8, label=label)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Final scan line index (1-based)')
    ax.set_ylabel('Trial count')
    ax.set_yscale('log')
    ax.set_title('Final frame line index per trial')

    # ------------------------------------------------------------------
    # Row 2: Per-trial frame counts
    # ------------------------------------------------------------------

    # [2,0] Primary frames per trial
    ax = axes[2, 0]
    ax.plot(np.arange(n_trials), primary_trial_num_frames,
            '.-', color='steelblue', linewidth=0.8, markersize=3)
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Frame count')
    ax.set_title('Primary frames per trial')

    # [2,1] Secondary frames per trial (optional)
    if secondary_trial_num_frames is not None:
        ax = axes[2, 1]
        ax.plot(np.arange(len(secondary_trial_num_frames)), secondary_trial_num_frames,
                '.-', color='tab:orange', linewidth=0.8, markersize=3)
        ax.set_xlabel('Trial index')
        ax.set_ylabel('Frame count')
        ax.set_title('Secondary frames per trial')
    else:
        axes[2, 1].axis('off')

    axes[2, 2].axis('off')

    # ------------------------------------------------------------------
    # Row 3: Raw vs normalized per-trial scan-line ranges
    # ------------------------------------------------------------------

    has_line_normalization_inputs = (
        primary_raw_frame_line_idxs is not None
        and secondary_raw_frame_line_idxs is not None
        and secondary_frame_line_idxs is not None
        and secondary_trial_num_frames is not None
    )
    if has_line_normalization_inputs:
        _plot_trial_line_normalization(
            axes[3, 0],
            primary_raw_frame_line_idxs,
            primary_frame_line_idxs,
            primary_trial_num_frames,
            'DMD1',
            'steelblue',
        )
        _plot_trial_line_normalization(
            axes[3, 1],
            secondary_raw_frame_line_idxs,
            secondary_frame_line_idxs,
            secondary_trial_num_frames,
            'DMD2',
            'tab:orange',
        )
    else:
        axes[3, 0].axis('off')
        axes[3, 1].axis('off')
    axes[3, 2].axis('off')

    plt.tight_layout()
    if qc_folder is not None:
        fig.savefig(qc_folder / 'slap2_inputs_qc.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def _plot_trial_line_normalization(
    ax,
    raw_frame_line_idxs,
    normalized_frame_line_idxs,
    trial_num_frames,
    plane_name,
    color,
):
    """Compare raw and normalized final scan-line indices by trial."""
    boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])
    trial_numbers = []
    raw_final_lines = []
    normalized_final_lines = []
    corrected_trials = []
    for trial_idx in range(len(trial_num_frames)):
        start = boundaries[trial_idx]
        stop = boundaries[trial_idx + 1]
        if start == stop:
            continue
        raw_trial_lines = raw_frame_line_idxs[start:stop]
        normalized_trial_lines = normalized_frame_line_idxs[start:stop]
        trial_number = trial_idx + 1
        raw_final = float(raw_trial_lines[-1])
        normalized_final = float(normalized_trial_lines[-1])

        trial_numbers.append(trial_number)
        raw_final_lines.append(raw_final)
        normalized_final_lines.append(normalized_final)
        if not np.array_equal(raw_trial_lines, normalized_trial_lines):
            corrected_trials.append(trial_number)

    ax.plot(
        trial_numbers,
        raw_final_lines,
        color='0.2',
        linestyle='-',
        linewidth=0.6,
        label='Raw final line',
        zorder=2,
    )
    ax.plot(
        trial_numbers,
        normalized_final_lines,
        color=color,
        linestyle=(0, (1, 0.6)),
        linewidth=0.8,
        dash_capstyle='round',
        label='Normalized final line',
        zorder=3,
    )
    for trial_number in corrected_trials:
        ax.annotate(
            '',
            xy=(trial_number, 0.91),
            xytext=(trial_number, 0.99),
            xycoords=ax.get_xaxis_transform(),
            textcoords=ax.get_xaxis_transform(),
            arrowprops={
                'arrowstyle': '-|>',
                'color': 'crimson',
                'linewidth': 1.2,
            },
            zorder=4,
        )

    ax.set_yscale('log')
    ax.set_xlabel('Trial number (1-based)')
    ax.set_ylabel('Final scan-line index (log)')
    ax.set_title(f'{plane_name} line-index normalization — '
                 f'{len(corrected_trials)} corrected trial(s)')
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# plot_slap2_sync_qc row helpers
# ---------------------------------------------------------------------------

def _plot_sync_qc_row0_cycle_detection(
    axes_row,
    trial_cycle_groups,
    trial_expected_cycles_qc,
    segmentation_method_qc,
    n_trials,
    t_idx,
):
    """Row 0: detected vs. expected cycles per trial and discrepancy."""
    is_gap_detection = segmentation_method_qc == 'gap_detection'
    trials_detected = np.array([len(g) for g in trial_cycle_groups])

    # [0] Detected vs expected cycles per trial
    ax = axes_row[0]
    ax.plot(t_idx, trials_detected, '.-', markersize=2, linewidth=0.8,
            color='steelblue', alpha=0.7, label='Detected')
    if trial_expected_cycles_qc is not None:
        ax.plot(t_idx, trial_expected_cycles_qc, '.-', markersize=2, linewidth=0.8,
                color='coral', alpha=0.7, label='Expected')
        ax.legend(fontsize=8)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Cycle count')
    if is_gap_detection:
        ax.set_title('Cycles per trial\n(gap-detected vs. predicted)')
    else:
        ax.set_title(f'Cycles per trial\n({segmentation_method_qc or "unknown"} assignment)')
    if n_trials > 30:
        ax.set_xticks([])

    # [1] Detected − expected cycles per trial
    ax = axes_row[1]
    if trial_expected_cycles_qc is not None:
        diff = trials_detected - trial_expected_cycles_qc
        ax.plot(t_idx, diff, '.-', markersize=2, linewidth=0.8, color='steelblue')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Detected \u2212 Expected')
        if is_gap_detection:
            ax.set_title('Cycle count discrepancy per trial\n(gap-detected \u2212 predicted)')
        else:
            ax.set_title(f'Cycle count discrepancy\n({segmentation_method_qc or "unknown"} assignment)')
        if n_trials > 30:
            ax.set_xticks([])
    else:
        ax.axis('off')

    axes_row[2].axis('off')


def _plot_sync_qc_row1_trial_metrics(
    axes_row,
    eff_lpc_per_trial_qc,
    primary_lines_per_cycle,
    primary_timestamps,
    frame_boundaries,
    n_trials,
    t_idx,
):
    """Row 1: effective LPC per trial and per-trial sampling rate."""
    # [0] effective_lpc per trial
    ax = axes_row[0]
    ax.plot(t_idx, eff_lpc_per_trial_qc, 'o-', markersize=3,
            color='steelblue', label='effective_lpc')
    if primary_lines_per_cycle is not None:
        ax.axhline(primary_lines_per_cycle, color='tab:red',
                   linewidth=1.5, linestyle='--',
                   label=f'Metadata ({primary_lines_per_cycle})')
        ax.legend(fontsize=8)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Lines per cycle')
    ax.set_title('effective_lpc per trial')
    if n_trials > 30:
        ax.set_xticks([])

    # [1] Effective sampling rate per trial
    ax = axes_row[1]
    fps_vals = []
    for i in range(n_trials):
        trial_ts = primary_timestamps[frame_boundaries[i]:frame_boundaries[i + 1]]
        if len(trial_ts) > 1:
            fps_vals.append((len(trial_ts) - 1) / (trial_ts[-1] - trial_ts[0]))
        else:
            fps_vals.append(np.nan)
    fps_arr = np.array(fps_vals)
    ax.plot(t_idx, fps_arr, 'o-', markersize=3, color='steelblue')
    med_fps = float(np.nanmedian(fps_arr))
    ax.axhline(med_fps, color='tab:green', linewidth=1.2,
               linestyle='--', label=f'Median {med_fps:.1f} Hz')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Effective rate (Hz)')
    ax.set_title('Primary sampling rate per trial')
    ax.legend(fontsize=8)
    if n_trials > 30:
        ax.set_xticks([])

    axes_row[2].axis('off')


def _plot_sync_qc_timestamp_detail(axes_row, timestamps, color, label):
    """
    Shared by primary (row 2) and secondary (row 4) timestamp detail panels.

    Parameters
    ----------
    axes_row : array of 3 Axes
    timestamps : np.ndarray
    color : str  — matplotlib colour string for all three panels
    label : str  — plane label used in titles (e.g. 'Primary', 'Secondary')
    """
    intervals_ms = np.diff(timestamps) * 1e3

    # [0] Timestamps vs sample index (monotonicity annotated)
    ax = axes_row[0]
    step = max(1, len(timestamps) // 5000)
    ax.plot(np.arange(len(timestamps))[::step],
            (timestamps - timestamps[0])[::step],
            color=color, linewidth=0.6)
    diffs = np.diff(timestamps)
    n_equal = int(np.sum(diffs == 0))
    n_decr  = int(np.sum(diffs < 0))
    if n_equal == 0 and n_decr == 0:
        mono_label = 'monotone \u2713'
    else:
        parts = []
        if n_equal > 0:
            parts.append(f'{n_equal} equal')
        if n_decr > 0:
            parts.append(f'{n_decr} decreasing')
        mono_label = 'NOT MONOTONE \u2717 (' + ', '.join(parts) + ')'
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Time since start (s)')
    ax.set_title(f'{label} timestamps\n' + mono_label)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # [1] Sampling diffs over sample index (max-within-window downsampled)
    ax = axes_row[1]
    step_d = max(1, len(intervals_ms) // 5000)
    if step_d > 1:
        n_w = len(intervals_ms) // step_d
        iv_ds  = intervals_ms[:n_w * step_d].reshape(n_w, step_d).max(axis=1)
        idx_ds = np.arange(n_w) * step_d + step_d // 2
    else:
        iv_ds  = intervals_ms
        idx_ds = np.arange(len(intervals_ms))
    ax.plot(idx_ds, iv_ds, color=color, linewidth=0.4, alpha=0.7)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Diff (ms)')
    ax.set_yscale('log')
    ax.set_title(f'{label} sampling diffs over time')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # [2] Sampling diffs distribution (log y)
    ax = axes_row[2]
    ax.hist(intervals_ms,
            bins=min(500, max(10, len(intervals_ms) // 10)),
            color=color, edgecolor='none')
    ax.set_xlabel('Interval (ms)')
    ax.set_ylabel('Count (log)')
    ax.set_yscale('log')
    ax.set_title(f'{label} sampling diffs distribution (log y)')


def _plot_sync_qc_row3_control_points(
    axes_row,
    trial_line_time_maps,
    primary_timestamps,
    n_trials,
):
    """Row 3: secondary plane interpolation control points."""
    # [0] Line→time control points (up to 10 trials, coloured by trial)
    ax = axes_row[0]
    n_show = min(n_trials, 10)
    cmap = plt.cm.tab10
    for i in range(n_show):
        ltm = trial_line_time_maps[i]
        if ltm is None:
            continue
        ctrl_lines, ctrl_ts = ltm
        ax.plot(ctrl_lines,
                (ctrl_ts - primary_timestamps[0]) * 1e3,
                '-o', markersize=2, linewidth=0.8,
                color=cmap(i / n_show), label=f'T{i}')
    ax.set_xlabel('Scan line index')
    ax.set_ylabel('Time since start (ms)')
    ax.set_title('Line\u2192time control points (first 10 trials)')
    if n_trials <= 10:
        ax.legend(fontsize=7, ncol=2)

    # [1] Normalised control point shape — all trials overlaid
    ax = axes_row[1]
    n_plotted = 0
    for i in range(n_trials):
        ltm = trial_line_time_maps[i]
        if ltm is None:
            continue
        ctrl_lines, ctrl_ts = ltm
        line_range = ctrl_lines[-1] - ctrl_lines[0]
        time_range = ctrl_ts[-1]  - ctrl_ts[0]
        if line_range > 0 and time_range > 0:
            rel_lines = (ctrl_lines - ctrl_lines[0]) / line_range
            rel_ts    = (ctrl_ts    - ctrl_ts[0])    / time_range
            ax.plot(rel_lines, rel_ts, '-', linewidth=0.4, alpha=0.2, color='steelblue')
            n_plotted += 1
    ax.set_xlabel('Relative scan line position (0=first, 1=last)')
    ax.set_ylabel('Relative time (0=trial start, 1=trial end)')
    ax.set_title(f'Control point shape \u2014 all {n_plotted} trials\n(normalized per trial)')

    axes_row[2].axis('off')


def plot_slap2_sync_qc(
    slap2_cycle_clock_signal,
    slap2_cycle_clock_times,
    primary_frame_line_idxs,
    primary_trial_num_frames,
    trial_line_time_maps,
    primary_timestamps,
    sync_qc_values,
    primary_lines_per_cycle=None,
    secondary_frame_line_idxs=None,
    secondary_trial_num_frames=None,
    secondary_timestamps=None,
    primary_raw_frame_line_idxs=None,
    secondary_raw_frame_line_idxs=None,
    clock_signal_window_s=0.5,
    qc_folder=None,
):
    """
    Produce a multi-panel QC figure for SLAP2 sync algorithm outputs.

    Raw inputs are shown separately by plot_slap2_inputs_qc.

    Panels (3 columns per row):

      Row 0 — Cycle detection (primary):
        [0,0] Detected vs. expected cycles per trial
        [0,1] Cycle count discrepancy (detected − expected)
        [0,2] blank

      Row 1 — Per-trial quality metrics (primary):
        [1,0] effective_lpc per trial vs. metadata reference
        [1,1] Primary sampling rate per trial
        [1,2] blank

      Row 2 — Primary timestamp detail:
        [2,0] Primary timestamps vs. sample index (monotonicity annotated)
        [2,1] Sampling diffs over time
        [2,2] Sampling diffs distribution (log-y)

      Row 3 (secondary only) — Interpolation control points:
        [3,0] Line→time control points for up to 10 trials (absolute, colored by trial)
        [3,1] Line→time control point shape — all trials normalized to [0,1]
                [3,2] blank

      Row 4 (secondary only) — Secondary timestamp detail:
        [4,0] Secondary timestamps vs. sample index (monotonicity annotated)
        [4,1] Secondary sampling diffs over time
        [4,2] Secondary sampling diffs distribution (log-y)

    Parameters
    ----------
    primary_frame_line_idxs : np.ndarray
    primary_trial_num_frames : np.ndarray
    trial_line_time_maps : list
        Second return value of get_slap2_primary_plane_timestamps.
    primary_timestamps : np.ndarray
        First return value of get_slap2_primary_plane_timestamps.
    sync_qc_values : dict
        Third return value of get_slap2_primary_plane_timestamps.
    primary_lines_per_cycle : int or float, optional
        Metadata value for reference lines on cycle-count and lpc plots.
    secondary_frame_line_idxs : np.ndarray, optional
    secondary_trial_num_frames : np.ndarray, optional
    secondary_timestamps : np.ndarray, optional
        If all three secondary arguments are provided, two additional rows are
        added for secondary alignment diagnostics.
    primary_raw_frame_line_idxs : np.ndarray, optional
        Primary scan-line indices before trial normalization, passed through to
        the inputs QC figure.
    secondary_raw_frame_line_idxs : np.ndarray, optional
        Secondary scan-line indices before trial normalization, passed through
        to the inputs QC figure.
    qc_folder : str or Path, optional
        If provided, the figure is saved as ``slap2_sync_qc.png`` inside this
        folder (dpi=150, tight layout) and closed before returning. The return
        value is None in this case.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure, or None if qc_folder was provided and the figure was closed.
    """
    plot_slap2_inputs_qc(
        slap2_cycle_clock_signal,
        slap2_cycle_clock_times,
        primary_frame_line_idxs,
        primary_trial_num_frames,
        sync_qc_values,
        secondary_frame_line_idxs=secondary_frame_line_idxs,
        secondary_trial_num_frames=secondary_trial_num_frames,
        primary_raw_frame_line_idxs=primary_raw_frame_line_idxs,
        secondary_raw_frame_line_idxs=secondary_raw_frame_line_idxs,
        clock_signal_window_s=clock_signal_window_s,
        qc_folder=qc_folder,
    )

    trial_cycle_groups       = sync_qc_values['trial_cycle_groups']
    trial_expected_cycles_qc = sync_qc_values.get('trial_expected_cycles_qc')
    segmentation_method_qc   = sync_qc_values.get('segmentation_method_qc')
    eff_lpc_per_trial_qc     = sync_qc_values.get('eff_lpc_per_trial_qc')

    n_trials = len(primary_trial_num_frames)
    frame_boundaries = np.concatenate([[0], np.cumsum(primary_trial_num_frames)])
    t_idx = np.arange(n_trials)

    has_secondary = (
        secondary_timestamps is not None
        and secondary_frame_line_idxs is not None
        and secondary_trial_num_frames is not None
    )
    n_rows = 5 if has_secondary else 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    fig.suptitle('SLAP2 Sync QC', fontsize=13, fontweight='bold')

    _plot_sync_qc_row0_cycle_detection(
        axes[0], trial_cycle_groups, trial_expected_cycles_qc,
        segmentation_method_qc, n_trials, t_idx,
    )
    _plot_sync_qc_row1_trial_metrics(
        axes[1], eff_lpc_per_trial_qc, primary_lines_per_cycle,
        primary_timestamps, frame_boundaries, n_trials, t_idx,
    )
    _plot_sync_qc_timestamp_detail(axes[2], primary_timestamps, 'steelblue', 'Primary')

    if has_secondary:
        _plot_sync_qc_row3_control_points(
            axes[3], trial_line_time_maps, primary_timestamps, n_trials,
        )
        _plot_sync_qc_timestamp_detail(axes[4], secondary_timestamps, 'tab:orange', 'Secondary')

    plt.tight_layout()
    if qc_folder is not None:
        fig.savefig(qc_folder / 'slap2_sync_qc.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig
