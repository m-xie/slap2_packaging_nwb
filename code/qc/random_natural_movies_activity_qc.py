"""Repeat heatmaps and averages from already synchronized Random Natural Movies NWB.

Movies use content coordinates (MovieFrame-1 at zero, nominal 30 fps), sampled
at min(100, params/analyzeHz) Hz. Gratings use elapsed HARP seconds, with +/-0.5 s flanks.
Source populations use sum(dF)/sum(F0); soma dF/F and stored NWB values are unchanged. Missing
coverage stays NaN; interpolation never bridges invalid samples or large gaps.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import h5py
import hdmf_zarr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb


MOVIE_FPS = 30.0  # Content-coordinate convention, not measured acquisition rate.
MOVIES = [
    (f"natural_movie_toe_{i}{suffix}", f"Movie {i}{label}")
    for i in (1, 2, 3)
    for suffix, label in (("", ""), ("_shuffle", " (shuf.)"))
]
ZEBRA = [("zebra_allen_screen_tscale_30_scale_10", "Zebra noise")]
ORIENTATIONS = (0, 45, 90, 135, 180, 225, 270, 315, 359)
GAP_FACTOR = 3.0


def read_analyze_hz(summary_path):
    """Read an explicit positive scalar rate; never silently assume 120 Hz."""
    with h5py.File(summary_path, "r") as summary:
        rates = []
        for key in ("params/analyze_hz", "params/analyzeHz"):
            if key not in summary:
                continue
            value = np.asarray(summary[key][()])
            if value.size != 1 or value.dtype.kind not in "iuf":
                raise ValueError(f"{key} must be a positive numeric scalar")
            rate = float(value.reshape(-1)[0])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"{key} must be a positive numeric scalar")
            rates.append(rate)
    if not rates:
        raise ValueError("Activity QC requires params/analyze_hz or params/analyzeHz in experiment_summary")
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError("Conflicting analyze_hz and analyzeHz parameters")
    return rates[0]


def interpolate_supported(x, y, query, max_gap):
    """Linear interpolation only between adjacent finite samples; no clamping.

    Exact valid samples remain usable next to missing samples or long gaps.
    Importantly, do not drop NaNs before interpolation: that would bridge holes.
    """
    x, y, query = (np.asarray(a, dtype=float) for a in (x, y, query))
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Interpolation coordinates and values must be paired vectors")
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Interpolation coordinates must be finite and strictly increasing")
    result = np.full(query.shape, np.nan)
    if not len(x):
        return result
    right = np.searchsorted(x, query)
    clipped = np.minimum(right, len(x) - 1)
    exact = np.isfinite(query) & (x[clipped] == query)
    result[exact] = y[clipped[exact]]
    interior = np.isfinite(query) & ~exact & (right > 0) & (right < len(x))
    idx = right[interior]
    span = x[idx] - x[idx - 1]
    valid = (span <= max_gap) & np.isfinite(y[idx - 1]) & np.isfinite(y[idx])
    values = np.full(len(idx), np.nan)
    weight = (query[interior][valid] - x[idx[valid] - 1]) / span[valid]
    values[valid] = y[idx[valid] - 1] * (1 - weight) + y[idx[valid]] * weight
    result[interior] = values
    return result


@dataclass
class Condition:
    key: str
    title: str
    kind: str
    axis: np.ndarray
    targets: np.ndarray  # repeat x display sample, in actual HARP time
    partial: np.ndarray
    offsets: np.ndarray  # observed grating offsets relative to onset, else NaN


def _movie_targets(row, content_frames):
    numbers = np.asarray(row.movie_frame_numbers, dtype=float)
    times = np.asarray(row.movie_frame_timestamps, dtype=float).copy()
    status = np.asarray(row.movie_frame_timing_status)
    if times.shape != numbers.shape or status.shape != numbers.shape:
        raise ValueError("Movie frame numbers, timestamps and timing status must be paired")
    if not len(numbers):
        return np.full(content_frames.shape, np.nan)
    if np.any(numbers < 1) or np.any(numbers != np.floor(numbers)):
        raise ValueError("Movie frame numbers must be positive integers")
    times[(status == 3) | (times < row.start_time) | (times > row.stop_time)] = np.nan
    finite = times[np.isfinite(times)]
    # Content events can share a display frame and therefore a timestamp.
    # Interpolation uses strictly increasing content numbers as coordinates,
    # not these times; preserve ties without inventing distinct visual onsets.
    if np.any(np.diff(finite) < 0):
        raise ValueError("Supported movie frame timestamps must not decrease")
    # A complete interval supplies the end of its last content frame. Do not
    # invent this boundary for a censored trial or unsupported final frame.
    if not row.is_partial and np.isfinite(times[-1]) and row.stop_time > times[-1]:
        numbers = np.append(numbers, numbers[-1] + 1)
        times = np.append(times, row.stop_time)
    return interpolate_supported(numbers, times, content_frames, max_gap=1.0)


def build_conditions(blocks, gratings, analyze_hz):
    """Build common grids once, keeping texture identity and all partial repeats."""
    if not np.isfinite(analyze_hz) or analyze_hz <= 0:
        raise ValueError("analyze_hz must be finite and positive")
    groups = [[], [], []]
    movies = blocks.loc[blocks["TrialType"].eq("movie")].copy()
    movies["texture_key"] = movies["TextureName"].astype(str).str.strip().str.lower().str.removesuffix(".mp4")
    known = {key for key, _ in MOVIES + ZEBRA}
    extra = [(key, key) for key in sorted(set(movies["texture_key"]) - known)]
    for group_idx, order in ((0, MOVIES + extra), (1, ZEBRA)):
        for key, title in order:
            rows = movies.loc[movies["texture_key"].eq(key)].sort_values("start_time")
            if rows.empty:
                groups[group_idx].append(Condition(key, title, "movie", np.array([]),
                                                   np.empty((0, 0)), np.array([], bool), np.array([])))
                continue
            last_frame = max((np.max(row.movie_frame_numbers, initial=0) for row in rows.itertuples()), default=0)
            # t=0 corresponds to MovieFrame-1. Fractional frames interpolate
            # within a repeat using its own physical frame-onset timestamps.
            axis = np.arange(int(np.ceil(last_frame / MOVIE_FPS * analyze_hz))) / analyze_hz
            targets = np.vstack([_movie_targets(row, 1 + axis * MOVIE_FPS) for row in rows.itertuples()])
            groups[group_idx].append(Condition(key, title, "movie", axis, targets,
                                               rows["is_partial"].to_numpy(bool), np.full(len(rows), np.nan)))

    for orientation in ORIENTATIONS:
        title = "Blank" if orientation == 359 else f"Grating {orientation}°"
        rows = (gratings.loc[gratings["logger_orientation"].eq(orientation)].sort_values("start_time")
                if not gratings.empty else gratings)
        if rows.empty:
            groups[2].append(Condition(str(orientation), title, "grating", np.array([]),
                                       np.empty((0, 0)), np.array([], bool), np.array([])))
            continue
        durations = (rows["stop_time"] - rows["start_time"]).to_numpy(float)
        if not np.isfinite(durations).all() or np.any(durations < 0):
            raise ValueError("Grating durations must be finite and nonnegative")
        axis = np.arange(-int(np.floor(0.5 * analyze_hz)),
                         int(np.floor((durations.max() + 0.5) * analyze_hz)) + 1) / analyze_hz
        partial = rows["is_partial"].to_numpy(bool)
        targets = rows["start_time"].to_numpy(float)[:, None] + axis
        # Censored stop times are coverage bounds, not observed offsets. Never
        # synthesize a post-offset segment for a partial presentation.
        upper = durations + np.where(partial, 0.0, 0.5)
        targets[axis[None, :] > upper[:, None]] = np.nan
        groups[2].append(Condition(str(orientation), title, "grating", axis, targets,
                                   partial, np.where(partial, np.nan, durations)))
    return groups


def read_trace(series, roi_index=None, chunk_size=10000, *, f0_series=None):
    """Read one soma, or sum(dF)/sum(F0) over paired finite source ROIs.

    Source NWB dFF is dF_denoised/(F0 + 1e-6); undo that denominator
    before summing. A missing or zero summed baseline leaves the sample NaN.
    """
    timestamps = np.asarray(series.timestamps[:], dtype=float)
    if series.data.shape[0] != len(timestamps) or len(series.data.shape) != 2:
        raise ValueError(f"{series.name}: expected (samples, ROIs) paired with timestamps")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{series.name}: timestamps must be finite and strictly increasing")
    if roi_index is None:
        if f0_series is None:
            raise ValueError(f"{series.name}: population activity requires paired F0")
        if f0_series.data.shape != series.data.shape or not np.array_equal(
                np.asarray(f0_series.timestamps[:], dtype=float), timestamps):
            raise ValueError(f"{series.name}: F0 must have matching shape and timestamps")
        if (f0_series.rois.table is not series.rois.table
                or not np.array_equal(f0_series.rois.data[:], series.rois.data[:])):
            raise ValueError(f"{series.name}: F0 must reference the same ROIs in the same order")
    trace = np.full(len(timestamps), np.nan)
    for start in range(0, len(trace), chunk_size):
        stop = min(start + chunk_size, len(trace))
        if roi_index is not None:
            trace[start:stop] = np.asarray(series.data[start:stop, roi_index], dtype=float)
        else:
            values = np.asarray(series.data[start:stop, :], dtype=float)
            baseline = np.asarray(f0_series.data[start:stop, :], dtype=float)
            df = values * (baseline + 1e-6)
            finite = np.isfinite(values) & np.isfinite(baseline) & np.isfinite(df)
            numerator = np.where(finite, df, 0.0).sum(axis=1)
            denominator = np.where(finite, baseline, 0.0).sum(axis=1)
            np.divide(numerator, denominator, out=trace[start:stop],
                      where=finite.any(axis=1) & np.isfinite(denominator) & (denominator != 0))
    trace[~np.isfinite(trace)] = np.nan
    return timestamps, trace


def align_trace(groups, timestamps, trace, analyze_hz):
    """Resample without crossing NaNs or gaps >3 expected/measured sample periods."""
    period = max(1 / analyze_hz, float(np.median(np.diff(timestamps)))) if len(timestamps) > 1 else 1 / analyze_hz
    return [[interpolate_supported(timestamps, trace, condition.targets, GAP_FACTOR * period)
             for condition in group] for group in groups]


def repeat_statistics(trials):
    finite = np.isfinite(trials)
    count = finite.sum(axis=0)
    mean = np.full(trials.shape[1], np.nan)
    np.divide(np.where(finite, trials, 0.0).sum(axis=0), count, out=mean, where=count > 0)
    return mean, count


def save_activity_plot(groups, aligned, title, sampling_hz, output_path):
    """Three stimulus bands, each with repeat heatmap, mean, and contributing N."""
    finite_values = [a[np.isfinite(a)] for group in aligned for a in group if np.isfinite(a).any()]
    if finite_values:
        limits = np.quantile(np.concatenate(finite_values), [0.01, 0.99])
        if limits[0] == limits[1]:
            limits = limits + np.array([-1, 1]) * max(1e-6, abs(limits[0]) * 0.01)
    else:
        limits = [0, 1]
    means = [repeat_statistics(a)[0] for group in aligned for a in group if a.shape[0]]
    finite_means = [a[np.isfinite(a)] for a in means if np.isfinite(a).any()]
    mean_ylim = None
    if finite_means:
        all_means = np.concatenate(finite_means)
        low, high = float(all_means.min()), float(all_means.max())
        pad = max((high - low) * 0.05, 1e-6)
        mean_ylim = (low - pad, high + pad)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 8}):
        fig = plt.figure(figsize=(22, 12), layout="constrained")
        try:
            outer = fig.add_gridspec(3, 2, width_ratios=[1, 0.015])
            image = None
            cmap = plt.get_cmap("bone").copy()
            cmap.set_bad("#dddddd")
            for band, (conditions, arrays) in enumerate(zip(groups, aligned)):
                grid = outer[band, 0].subgridspec(3, len(conditions), height_ratios=[2, 1, 0.45])
                for col, (condition, trials) in enumerate(zip(conditions, arrays)):
                    heat = fig.add_subplot(grid[0, col])
                    avg = fig.add_subplot(grid[1, col], sharex=heat)
                    coverage = fig.add_subplot(grid[2, col], sharex=heat)
                    heat.set_title(condition.title + f" (n={len(trials)})")
                    if trials.size == 0:
                        heat.text(0.5, 0.5, "No data", ha="center", va="center", transform=heat.transAxes)
                        for ax in (heat, avg, coverage):
                            ax.set_axis_off()
                        continue
                    x = condition.axis
                    dt = 1 / sampling_hz
                    image = heat.imshow(np.ma.masked_invalid(trials), origin="lower", aspect="auto",
                                        extent=[x[0] - dt / 2, x[-1] + dt / 2, 0.5, len(trials) + 0.5],
                                        cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
                    ticks = np.unique(np.linspace(1, len(trials), min(len(trials), 6), dtype=int))
                    heat.set_yticks(ticks, [f"{i}{'*' if condition.partial[i - 1] else ''}" for i in ticks])
                    mean, count = repeat_statistics(trials)
                    avg.plot(x, mean, color="#a52131", linewidth=0.9)
                    if mean_ylim is not None:
                        avg.set_ylim(mean_ylim)
                    coverage.fill_between(x, 0, count, color="#447a99", step="mid", alpha=0.7)
                    coverage.set_ylim(0, max(1, len(trials)))
                    coverage.set_yticks([0, len(trials)])
                    coverage.set_xlabel("Movie time (s; (frame−1)/30)" if condition.kind == "movie"
                                        else "Time from onset (s)")
                    for ax in (heat, avg):
                        ax.tick_params(labelbottom=False)
                    if condition.kind == "grating":
                        valid_offsets = np.isfinite(condition.offsets)
                        heat.scatter(condition.offsets[valid_offsets], np.flatnonzero(valid_offsets) + 1,
                                     marker="|", color="#ea8735", s=45)
                        for ax in (heat, avg, coverage):
                            ax.axvline(0, color="#447a99", linewidth=0.7, linestyle=":")
                        if valid_offsets.any():
                            avg.axvline(np.median(condition.offsets[valid_offsets]), color="#ea8735",
                                        linewidth=0.8, linestyle="--")
                    if col == 0:
                        heat.set_ylabel("Repeat (* partial)")
                        avg.set_ylabel("Mean dF/F")
                        coverage.set_ylabel("N")
            if image is not None:
                fig.colorbar(image, cax=fig.add_subplot(outer[:, 1]),
                             label="dF/F (shared 1st–99th percentile color limits; gray = missing)", extend="both")
            fig.suptitle(
                f"{title}\nSampling grid: {sampling_hz:g} Hz (min(100 Hz, analyzeHz)); existing dF/F, no added baseline subtraction\n"
                "Movies: content-aligned at nominal 30 fps; gratings: real-time onset alignment, ±0.5 s flanks. "
                "Orange: observed offsets (mean panel: median). N: finite contributing repeats.", fontsize=11,
            )
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)
    return output_path


def compute_activity_qc(qc_folder, nwb_path, experiment_summary_path):
    """Write one green sum(dF)/sum(F0) PNG per DMD and one red PNG per soma."""
    analyze_hz = read_analyze_hz(experiment_summary_path)
    sampling_hz = min(100.0, analyze_hz)
    nwb_path = Path(nwb_path)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO
    outputs = []
    with io_class(str(nwb_path), "r") as io:
        nwb = io.read()
        if "ophys" not in nwb.processing or "stimulus_blocks" not in nwb.intervals:
            warnings.warn("Random Natural Movies activity QC: missing ophys or stimulus_blocks; skipping.")
            return outputs
        blocks = nwb.intervals["stimulus_blocks"].to_dataframe()
        gratings = nwb.intervals["gratings"].to_dataframe() if "gratings" in nwb.intervals else pd.DataFrame()
        groups = build_conditions(blocks, gratings, sampling_hz)
        for name, container in sorted(nwb.processing["ophys"].data_interfaces.items()):
            match = re.fullmatch(r"(Soma)?Fluorescence_(DMD\d+)", name)
            if match is None:
                continue
            soma, dmd = match.groups()
            series_name = f"{dmd}_soma_dFF_red" if soma else f"{dmd}_dFF_green"
            if series_name not in container.roi_response_series:
                warnings.warn(f"Activity QC: {series_name} unavailable; skipping this channel.")
                continue
            series = container.roi_response_series[series_name]
            f0_series = None
            if not soma:
                f0_name = f"{dmd}_F0_green"
                if f0_name not in container.roi_response_series:
                    warnings.warn(f"Activity QC: {f0_name} unavailable; skipping population activity.")
                    continue
                f0_series = container.roi_response_series[f0_name]
            if not series.data.shape[0] or not series.data.shape[1]:
                warnings.warn(f"Activity QC: {series_name} is empty; skipping.")
                continue
            roi_indices = range(series.data.shape[1]) if soma else [None]
            for index in roi_indices:
                if soma:
                    row = int(series.rois.data[index])
                    roi_table = series.rois.table
                    source_index = int(roi_table["user_roi_index"][row])
                    label = str(roi_table["label"][row])
                    title = f"{dmd} — red soma {source_index}: {label}"
                    filename = f"{dmd}_soma_red_roi_{source_index}.png"
                else:
                    title = f"{dmd} — green sum(dF)/sum(F0) across {series.data.shape[1]} extracted-source ROIs (excluding user-soma series)"
                    filename = f"{dmd}_sources_green_mean.png"
                times, trace = read_trace(series, index, f0_series=f0_series)
                aligned = align_trace(groups, times, trace, analyze_hz)
                output = save_activity_plot(groups, aligned, title, sampling_hz,
                                            Path(qc_folder) / "activity" / filename)
                outputs.append(output)
                print(f"Random Natural Movies activity QC: saved {output}")
    return outputs