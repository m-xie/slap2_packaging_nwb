"""Compact synchronization plots for Random Natural Movies only."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot_photodiode_sync(anchor_frames, anchor_times, output_path):
    """Compare the actual synchronization anchor pairs at both endpoints.

    Select pairs within 2 s of the first/last matched HARP anchor. Convert
    their display frames at fixed nominal 60 Hz and align only the endpoint
    pair. Do not apply the piecewise mapping to the plotted frame positions:
    that would force every anchor to overlap and conceal timing differences.
    Expand the limits when necessary without stretching either sequence.
    """
    frame_edges = np.asarray(anchor_frames, dtype=float)
    time_edges = np.asarray(anchor_times, dtype=float)
    if (
        frame_edges.ndim != 1 or time_edges.shape != frame_edges.shape
        or not frame_edges.size
        or not np.isfinite(frame_edges).all() or not np.isfinite(time_edges).all()
        or np.any(np.diff(frame_edges) <= 0) or np.any(np.diff(time_edges) <= 0)
    ):
        raise ValueError("Photodiode QC requires paired finite strictly increasing anchors")
    frame_rate = 60.0
    windows = ((time_edges[0], time_edges[0] + 2.0),
               (time_edges[-1] - 2.0, time_edges[-1]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    try:
        for ax, bounds, label in zip(axes, windows, ("First", "Last")):
            time_left, time_right = bounds
            selected = (time_edges >= time_left) & (time_edges <= time_right)
            times = time_edges[selected]
            count = len(times)
            frames = frame_edges[selected]
            endpoint = 0 if label == "First" else -1
            shift = times[endpoint] - frames[endpoint] / frame_rate
            shifted_times = frames / frame_rate + shift
            # Keep all N frame edges visible even if their span exceeds 2 s.
            time_left = min(time_left, shifted_times[0])
            time_right = max(time_right, shifted_times[-1])
            left, right = (np.array([time_left, time_right]) - shift) * frame_rate
            ax.vlines(frames, 0, 1, color="b", alpha=0.6)
            ax.set_xlim(left, right)
            ax.set_ylim(0, 1)
            ax.set_xlabel("BonVision display frame (fixed 60 Hz; horizontal shift only)")
            ax.set_ylabel("Photodiode anchor")
            ax.set_yticks([])
            ax.set_title(f"{label} anchors: {count} matched pairs within 2 s of endpoint")
            time_ax = ax.twiny()
            time_ax.vlines(times, 0, 1, color="g", alpha=0.6, linewidth=2, linestyle="--")
            time_ax.set_xlim(time_left, time_right)
            time_ax.set_xlabel("Harp time (s, normalized to first recorded DO0)")
            time_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            note = (f"Selected: {count} matched anchor pairs\n"
                    f"{label} anchor pair aligned; shift = {shift:+.6f} s")
            ax.text(0.01, 0.03, note, transform=ax.transAxes, fontsize=9,
                    va="bottom", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
            ax.legend(handles=[
                Line2D([], [], color="b", alpha=0.6, label="BonVision frame anchors"),
                Line2D([], [], color="g", alpha=0.6, linestyle="--", label="Harp photodiode anchors"),
            ], loc="upper right")
        fig.suptitle(
            "Visual stimulus / Harp synchronization\n"
            f"Matched anchor pairs used for alignment: {len(time_edges):,}\n",
            fontsize=11,
        )
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
    return output_path


def plot_di3_sync(clock, cycle_counts, output_path):
    """Report raw per-DMD cycles and ALL detected DI3 intervals, not controls.

    DI3 belongs to DMD1: other DMD counts are reported separately, never summed
    into that comparison. An extra detected final boundary is retained; an
    estimated terminal interpolation boundary is not a detected pulse.
    """
    pulse_times = np.asarray(clock["cycle_starts"], dtype=float)
    intervals = np.diff(pulse_times)
    lines = [
        f"Total SLAP2 cycles ({name}{', DI3 primary' if name == 'DMD1' else ''}): {count:,}"
        for name, count in cycle_counts.items()
    ]
    lines.append(f"Total detected DI3 pulses: {len(pulse_times):,}")
    summary = "\n".join(lines)
    print(summary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    try:
        fig.suptitle("SLAP2 / Harp synchronization\n" + summary)
        if len(intervals):
            ax.hist(intervals * 1000, bins=100, color="tab:green", edgecolor="white")
            # Keep rare long gaps visible without excluding any intervals.
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "Fewer than two detected DI3 pulses; no intervals",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time between consecutive detected DI3 pulses (ms)")
        ax.set_ylabel("Interval count (log scale)" if len(intervals) else "Interval count")
        ax.set_title("All detected pulses; no estimated boundaries")
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
    return output_path