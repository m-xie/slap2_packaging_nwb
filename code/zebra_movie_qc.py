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
    """Return Pearson correlation using finite paired samples."""
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return np.nan
    first_values = first[finite]
    second_values = second[finite]
    if np.std(first_values) == 0 or np.std(second_values) == 0:
        return np.nan
    return float(np.corrcoef(first_values, second_values)[0, 1])


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
    """Plot two equal repeats contained in the single Zebra movie interval."""
    nwb_path = Path(nwb_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO

    with io_class(str(nwb_path), mode="r") as io:
        nwbfile = io.read()
        movie_table = nwbfile.intervals["movie"].to_dataframe()
        zebra_rows = movie_table.loc[movie_table["BlockLabel"] == "Zebra"]
        if len(zebra_rows) != 1:
            raise ValueError(f"Expected one Zebra interval, found {len(zebra_rows)}")

        zebra = zebra_rows.iloc[0]
        repeat_duration = float(zebra["stop_time"] - zebra["start_time"]) / 2
        repeat_starts = [
            float(zebra["start_time"]),
            float(zebra["start_time"]) + repeat_duration,
        ]

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