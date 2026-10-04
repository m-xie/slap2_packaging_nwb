"""Diagnostics for movie frame timing in normalized HARP coordinates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_movie_frame_timing(blocks, output_path):
    """Save a three-panel timing diagnostic as PNG and return its ``Path``.

    ``blocks`` contains ragged movie_frame_timestamps (seconds),
    movie_display_frames (global logger frames), and movie_frame_timing_status
    arrays. Status codes are 0 anchored, 1 interpolated, 2 extrapolated, and
    3 unsupported. Nonmovie rows contain empty arrays. Alignment anchors and
    metadata are read from ``blocks.attrs['movie_frame_alignment']``.

    Differences use only adjacent supported, finite timestamps within a row;
    they never bridge blocks or missing timestamps. Affine anchor residuals
    describe deviation from a constant frame clock, not timing accuracy.
    Coverage rugs use axis-relative vertical lanes, not gap values, so even
    unsupported frames without timestamps are shown on the global frame axis.
    No global matplotlib backend is selected or changed.
    """
    rows = []
    columns = (
        "movie_frame_timestamps",
        "movie_display_frames",
        "movie_frame_timing_status",
    )
    for timestamps, frames, status in blocks.loc[:, columns].itertuples(
        index=False, name=None
    ):
        timestamps = np.asarray(timestamps, dtype=float)
        frames = np.asarray(frames)
        status = np.asarray(status)
        if any(array.ndim != 1 for array in (timestamps, frames, status)) or not (
            len(timestamps) == len(frames) == len(status)
        ):
            raise ValueError("Movie timing arrays must be one-dimensional and equal length")
        rows.append((timestamps, frames, status))

    alignment = blocks.attrs.get("movie_frame_alignment", {})
    metadata = alignment.get("metadata", {})
    anchor_frames = np.asarray(alignment.get("anchor_frames", []), dtype=float)
    anchor_times = np.asarray(alignment.get("anchor_times", []), dtype=float)
    if anchor_frames.ndim != 1 or anchor_times.shape != anchor_frames.shape:
        raise ValueError("Anchor frame and time arrays must be one-dimensional and equal length")
    finite = np.isfinite(anchor_frames) & np.isfinite(anchor_times)
    order = np.argsort(anchor_frames[finite], kind="stable")
    anchor_frames = anchor_frames[finite][order]
    anchor_times = anchor_times[finite][order]

    output_path = Path(output_path)
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    try:
        interval_axis, residual_axis, coverage_axis = axes
        has_intervals = False
        for timestamps, _, status in rows:
            supported = np.isfinite(timestamps) & (status != 3)
            adjacent = supported[:-1] & supported[1:]
            if adjacent.any():
                interval_axis.plot(
                    timestamps[1:][adjacent],
                    np.diff(timestamps)[adjacent],
                    linestyle="none", marker=".", markersize=3,
                    color="#1876a3",
                    label=None if has_intervals else "Adjacent movie frames",
                )
                has_intervals = True
        interval_axis.set(
            title="Movie inter-frame intervals (within blocks; no missing-time bridges)",
            xlabel="Normalized HARP time of later frame (s)",
            ylabel="Inter-frame interval (s)",
        )
        if not has_intervals:
            interval_axis.text(
                0.5, 0.5, "No valid adjacent movie timestamps",
                ha="center", transform=interval_axis.transAxes,
            )

        # Center before fitting to avoid conditioning problems with large counters.
        if len(anchor_frames) >= 2 and np.ptp(anchor_frames) > 0:
            centered_frames = anchor_frames - anchor_frames.mean()
            centered_times = anchor_times - anchor_times.mean()
            slope = np.dot(centered_frames, centered_times) / np.dot(
                centered_frames, centered_frames
            )
            residuals = centered_times - slope * centered_frames
            residual_axis.plot(
                anchor_times, residuals, linestyle="none", marker=".",
                color="#1876a3", label="Affine residual",
            )
            residual_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        else:
            residual_axis.text(
                0.5, 0.5, "Affine fit requires at least two distinct anchor frames",
                ha="center", transform=residual_axis.transAxes,
            )
        residual_axis.set(
            title="Anchor affine residuals — not a measure of timing accuracy",
            xlabel="Normalized HARP anchor time (s)",
            ylabel="Affine residual (s)",
        )

        if len(anchor_frames) >= 2:
            coverage_axis.plot(
                anchor_frames[1:], np.diff(anchor_frames), ".-",
                color="#1876a3", linewidth=0.8, label="Anchor spacing",
            )
        if len(anchor_frames):
            coverage_axis.plot(
                anchor_frames, np.full(len(anchor_frames), 0.26),
                linestyle="none", marker="|", color="black",
                transform=coverage_axis.get_xaxis_transform(), label="Alignment anchors",
            )
        styles = (
            (0, "Anchored movie frames", "#1876a3", "|"),
            (1, "Interpolated movie frames", "#338855", "|"),
            (2, "Extrapolated movie frames", "#dd8822", "^"),
            (3, "Unsupported movie frames", "#b33b2e", "x"),
        )
        for code, label, color, marker in styles:
            selected = [frames[status == code] for _, frames, status in rows]
            frames = np.concatenate(selected) if selected else np.array([])
            if len(frames):
                coverage_axis.plot(
                    frames, np.full(len(frames), 0.05 + 0.05 * code),
                    linestyle="none", marker=marker, markersize=4, alpha=0.7,
                    color=color, transform=coverage_axis.get_xaxis_transform(),
                    label=label,
                )
        if not any(len(frames) for _, frames, _ in rows):
            coverage_axis.text(
                0.5, 0.5, "No movie frames",
                ha="center", transform=coverage_axis.transAxes,
            )
        coverage_axis.set(
            title="Anchor spacing and global-frame coverage (coverage rugs in lower lanes)",
            xlabel="Global logger display frame",
            ylabel="Gap between adjacent anchors (frames)",
        )
        # Leave room below the spacing curve for the axis-relative coverage rugs.
        coverage_axis.set_ylim(bottom=0)
        coverage_axis.set_ylim(top=coverage_axis.get_ylim()[1] * 1.3)
        for axis in axes:
            axis.grid(alpha=0.2, linewidth=0.5)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8, loc="upper right")
        figure.suptitle(
            "Movie frame timing diagnostics\n"
            f"clock_reference: {metadata.get('clock_reference', 'unknown')}; "
            f"mapping_method: {metadata.get('mapping_method', 'unknown')}\n"
            f"maximum_anchor_gap_frames: {metadata.get('maximum_anchor_gap_frames', 'unknown')}; "
            "maximum_interpolation_gap_frames: "
            f"{metadata.get('maximum_interpolation_gap_frames', 'unknown')}",
            fontsize=10, fontweight="bold",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="png", dpi=150)
    finally:
        plt.close(figure)
    return output_path