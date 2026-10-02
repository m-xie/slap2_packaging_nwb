"""Plot dF/F traces aligned to repeated Zebra movie presentations."""

from pathlib import Path

import hdmf_zarr
import matplotlib.pyplot as plt
import numpy as np
import pynwb


def _bin_repeat(data, timestamps, start_time, duration, bin_size):
    """Return mean dF/F percent in fixed bins relative to one repeat onset."""
    edges = np.arange(0.0, duration + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2
    start_index, stop_index = np.searchsorted(
        timestamps, (start_time, start_time + duration)
    )
    local_timestamps = timestamps[start_index:stop_index] - start_time
    local_data = np.asarray(data[start_index:stop_index]) * 100.0
    bin_indices = np.searchsorted(edges, local_timestamps, side="right") - 1
    valid = (bin_indices >= 0) & (bin_indices < len(centers))

    sums = np.zeros((len(centers), data.shape[1]), dtype=float)
    counts = np.zeros_like(sums, dtype=int)
    finite = np.isfinite(local_data[valid])
    np.add.at(sums, bin_indices[valid], np.where(finite, local_data[valid], 0.0))
    np.add.at(counts, bin_indices[valid], finite)
    binned = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=binned, where=counts > 0)
    return centers, binned


def _save_series_plot(series_name, times, repeats, output_path):
    """Write one panel per ROI with both repeats overlaid."""
    n_rois = repeats[0].shape[1]
    n_columns = min(6, n_rois)
    n_rows = (n_rois + n_columns - 1) // n_columns
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.2 * n_columns, 2.2 * n_rows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    for roi_index, axis in enumerate(axes.flat):
        if roi_index >= n_rois:
            axis.axis("off")
            continue
        axis.plot(times, repeats[0][:, roi_index], linewidth=0.8, label="Repeat 1")
        axis.plot(times, repeats[1][:, roi_index], linewidth=0.8, label="Repeat 2")
        axis.set_title(f"ROI {roi_index}", fontsize=8)
        axis.grid(alpha=0.2, linewidth=0.5)

    for axis in axes[-1]:
        if axis.axison:
            axis.set_xlabel("Time since Zebra start (s)")
    for axis in axes[:, 0]:
        if axis.axison:
            axis.set_ylabel("dF/F (%)")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    figure.suptitle(f"{series_name}: Zebra movie repeats", fontweight="bold")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _finite_correlation(first, second):
    """Return Pearson r using finite paired time bins.

    Pearson r is the normalized sum of products
    ``(repeat1 - mean1) * (repeat2 - mean2)``. Subtracting the two centered
    traces would measure difference, not similarity of response shape.
    """
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return np.nan
    first_values = first[finite]
    second_values = second[finite]
    if np.std(first_values) == 0 or np.std(second_values) == 0:
        return np.nan
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _repeat_windows(zebra_rows):
    """Return starts and duration for combined or explicitly separated repeats."""
    if len(zebra_rows) == 1:
        zebra = zebra_rows.iloc[0]
        repeat_duration = float(zebra["stop_time"] - zebra["start_time"]) / 2
        repeat_starts = [
            float(zebra["start_time"]),
            float(zebra["start_time"]) + repeat_duration,
        ]
    elif len(zebra_rows) == 2:
        durations = (
            zebra_rows["stop_time"].to_numpy(dtype=float)
            - zebra_rows["start_time"].to_numpy(dtype=float)
        )
        if not np.isclose(durations[0], durations[1], rtol=0.01):
            raise ValueError(
                "The two Zebra repeat intervals must have equal durations"
            )
        repeat_duration = float(np.mean(durations))
        repeat_starts = zebra_rows["start_time"].to_numpy(dtype=float).tolist()
    else:
        raise ValueError(
            f"Expected one combined or two separate Zebra intervals, "
            f"found {len(zebra_rows)}"
        )
    if repeat_duration <= 0:
        raise ValueError("Zebra repeat duration must be positive")
    return repeat_starts, repeat_duration


def _save_roi_correlation_plot(series_name, roi_correlations, output_path):
    """Plot repeat-to-repeat Pearson correlation for every ROI."""
    roi_correlations = np.asarray(roi_correlations, dtype=float)
    roi_indices = np.arange(len(roi_correlations))
    finite = np.isfinite(roi_correlations)

    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.scatter(
        roi_indices[finite],
        roi_correlations[finite],
        s=12,
        color="#1876a3",
        alpha=0.8,
        edgecolors="none",
    )
    if np.any(~finite):
        axis.scatter(
            roi_indices[~finite],
            np.zeros(np.count_nonzero(~finite)),
            marker="x",
            s=18,
            color="#888888",
            label="Undefined correlation",
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    if finite.any():
        median_correlation = float(np.median(roi_correlations[finite]))
        axis.axhline(
            median_correlation,
            color="#b33b2e",
            linestyle="--",
            linewidth=1.2,
            label=f"Median r = {median_correlation:.3f}",
        )
    axis.set(xlabel="ROI index", ylabel="Repeat 1 vs repeat 2 Pearson r", ylim=(-1.05, 1.05))
    axis.set_title(f"{series_name}: Zebra repeat response reliability")
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    if finite.any() or np.any(~finite):
        axis.legend(loc="lower right", fontsize=8)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_repeat_rasters(series_name, times, repeats, output_path):
    """Write matched repeat heatmaps and their signed difference."""
    first, second = repeats
    difference = second - first
    repeat_values = np.concatenate((first[np.isfinite(first)], second[np.isfinite(second)]))
    color_min, color_max = np.percentile(repeat_values, (1, 99))
    difference_values = np.abs(difference[np.isfinite(difference)])
    difference_limit = float(np.percentile(difference_values, 99))
    if difference_limit == 0:
        difference_limit = 1.0

    roi_correlations = np.array([
        _finite_correlation(first[:, roi_index], second[:, roi_index])
        for roi_index in range(first.shape[1])
    ])
    global_correlation = _finite_correlation(first, second)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        constrained_layout=True,
    )
    extent = (times[0], times[-1], first.shape[1], 0)
    panels = (
        (first.T, "Repeat 1", "viridis", color_min, color_max),
        (second.T, "Repeat 2", "viridis", color_min, color_max),
        (difference.T, "Repeat 2 - Repeat 1", "RdBu_r", -difference_limit, difference_limit),
    )
    for axis, (values, title, color_map, minimum, maximum) in zip(axes, panels):
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=color_map,
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_ylabel("ROI index")
        figure.colorbar(image, ax=axis, label="dF/F (%)", pad=0.01)
    axes[-1].set_xlabel("Time since Zebra start (s)")
    median_correlation = float(np.nanmedian(roi_correlations))
    figure.suptitle(
        f"{series_name}: Zebra repeat comparison\n"
        f"global r = {global_correlation:.3f}; median per-ROI r = {median_correlation:.3f}",
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return global_correlation, roi_correlations


def plot_zebra_repeats(nwb_path, output_folder, bin_size=0.1):
    """Plot two equal Zebra repeats and their per-ROI response reliability."""
    nwb_path = Path(nwb_path)
    output_folder = Path(output_folder)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO

    with io_class(str(nwb_path), mode="r") as io:
        nwbfile = io.read()
        if "movie" not in nwbfile.intervals:
            print("Zebra QC: no movie interval table found; skipping.")
            return
        movie_table = nwbfile.intervals["movie"].to_dataframe()
        if "BlockLabel" not in movie_table.columns:
            print("Zebra QC: movie table has no BlockLabel column; skipping.")
            return
        zebra_rows = movie_table.loc[movie_table["BlockLabel"] == "Zebra"]
        if len(zebra_rows) == 0:
            print("Zebra QC: no Zebra movie intervals found; skipping.")
            return
        repeat_starts, repeat_duration = _repeat_windows(zebra_rows)
        output_folder.mkdir(parents=True, exist_ok=True)

        for interface in nwbfile.processing["ophys"].data_interfaces.values():
            if not isinstance(interface, pynwb.ophys.Fluorescence):
                continue
            for series_name, series in interface.roi_response_series.items():
                if "dff" not in series_name.lower():
                    continue
                timestamps = np.asarray(series.timestamps)
                repeats = []
                for repeat_start in repeat_starts:
                    times, values = _bin_repeat(
                        series.data,
                        timestamps,
                        repeat_start,
                        repeat_duration,
                        bin_size,
                    )
                    repeats.append(values)
                output_path = output_folder / f"{series_name}_zebra_repeats.png"
                _save_series_plot(series_name, times, repeats, output_path)
                print(f"Zebra QC: saved {output_path}")
                raster_path = output_folder / f"{series_name}_zebra_rasters.png"
                global_correlation, roi_correlations = _save_repeat_rasters(
                    series_name, times, repeats, raster_path
                )
                print(
                    f"Zebra QC: saved {raster_path} "
                    f"(global r={global_correlation:.3f}, "
                    f"median ROI r={np.nanmedian(roi_correlations):.3f})"
                )
                correlation_path = (
                    output_folder
                    / f"{series_name}_zebra_roi_correlations.png"
                )
                _save_roi_correlation_plot(
                    series_name, roi_correlations, correlation_path
                )
                print(f"Zebra QC: saved {correlation_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nwb_path", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--bin-size", type=float, default=0.1)
    arguments = parser.parse_args()
    plot_zebra_repeats(
        arguments.nwb_path,
        arguments.output_folder,
        bin_size=arguments.bin_size,
    )