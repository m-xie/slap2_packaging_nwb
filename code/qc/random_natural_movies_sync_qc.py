"""Compact synchronization plots for Random Natural Movies only."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot_photodiode_sync(logger_data, harp_times, output_path):
    """Compare endpoint edge sequences using translation only, not fitted timing.

    Take N analog edges within 2 s of the first/last detected analog edge and
    the first/last N logger edges. Convert display frames at fixed nominal 60 Hz.
    Align the first pair in the first panel and the last pair in the last panel.
    No matched anchors, fitted frame rate, logger timestamps, or stretching are
    used. Expand the display limits if needed to show all selected logger edges;
    this does not change the frame-to-second conversion. Analog counts include
    all detected edges across the recording, without any DO0/DO1 cutoff.
    """
    frame_edges = np.asarray(logger_data.transition_frames, dtype=float)
    time_edges = np.asarray(harp_times, dtype=float)
    if not len(frame_edges) or not len(time_edges):
        raise ValueError("Photodiode QC requires both logger and analog edges")
    frame_rate = 60.0
    windows = ((time_edges[0], time_edges[0] + 2.0),
               (time_edges[-1] - 2.0, time_edges[-1]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    try:
        for ax, bounds, label in zip(axes, windows, ("First", "Last")):
            time_left, time_right = bounds
            times = time_edges[(time_edges >= time_left) & (time_edges <= time_right)]
            count = len(times)
            frames = frame_edges[:count] if label == "First" else frame_edges[-count:]
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
            ax.set_ylabel("Photodiode edge")
            ax.set_yticks([])
            ax.set_title(f"{label} 2 s of analog edges vs {label.lower()} {count} BonVision edges")
            time_ax = ax.twiny()
            time_ax.vlines(times, 0, 1, color="g", alpha=0.6, linewidth=2, linestyle="--")
            time_ax.set_xlim(time_left, time_right)
            time_ax.set_xlabel("Harp time (s, normalized to first recorded DO0)")
            time_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            note = (f"Selected: {count} analog / {len(frames)} BonVision edges\n"
                    f"{label} edge pair aligned; shift = {shift:+.6f} s")
            if len(frames) < count:
                note += f"\nOnly {len(frames)} BonVision edges available (requested {count})"
            ax.text(0.01, 0.03, note, transform=ax.transAxes, fontsize=9,
                    va="bottom", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
            ax.legend(handles=[
                Line2D([], [], color="b", alpha=0.6, label="BonVision frame edges"),
                Line2D([], [], color="g", alpha=0.6, linestyle="--", label="Analog photodiode edges"),
            ], loc="upper right")
        fig.suptitle(
            "Visual stimulus / Harp synchronization\n"
            f"Detected analog photodiode edges: {len(time_edges):,}  |  "
            f"Detected BonVision frame edges: {len(frame_edges):,}\n",
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