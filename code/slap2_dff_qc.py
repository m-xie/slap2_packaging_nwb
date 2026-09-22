"""Whole-session dF/F QC plotting for packaged SLAP2 NWB files."""

from pathlib import Path

import hdmf_zarr
import matplotlib.pyplot as plt
import numpy as np
import pynwb


_DFF_TAGS = ('dff', 'dfoverf', 'df_over_f', 'delta_f_over_f')


def _is_dff_series(name):
    normalized = name.lower().replace(' ', '_')
    return any(tag in normalized for tag in _DFF_TAGS)


def _is_raw_fluorescence_series(name):
    normalized = name.lower().replace(' ', '_')
    return '_f0_' in f'_{normalized}_'


def _bin_activity(data, timestamps, bin_size=0.1, chunk_size=50000):
    """Compute per-ROI means in fixed-width bins while preserving empty gaps."""
    timestamps = np.asarray(timestamps)
    if len(timestamps) == 0:
        return np.empty((0, data.shape[1])), np.empty(0)

    start_time = float(timestamps[0])
    bin_indices = np.floor((timestamps - start_time) / bin_size).astype(np.int64)
    n_bins = int(bin_indices[-1]) + 1
    n_rois = data.shape[1]
    sums = np.zeros((n_bins, n_rois), dtype=np.float64)
    counts = np.zeros((n_bins, n_rois), dtype=np.int64)

    for chunk_start in range(0, len(timestamps), chunk_size):
        chunk_stop = min(chunk_start + chunk_size, len(timestamps))
        chunk = np.asarray(data[chunk_start:chunk_stop])
        chunk_bins = bin_indices[chunk_start:chunk_stop]
        boundaries = np.r_[0, np.flatnonzero(np.diff(chunk_bins)) + 1]
        unique_bins = chunk_bins[boundaries]
        finite = np.isfinite(chunk)
        sums[unique_bins] += np.add.reduceat(
            np.where(finite, chunk, 0.0), boundaries, axis=0
        )
        counts[unique_bins] += np.add.reduceat(finite, boundaries, axis=0)

    binned = np.full((n_bins, n_rois), np.nan)
    np.divide(sums, counts, out=binned, where=counts > 0)
    bin_centers = start_time + (np.arange(n_bins) + 0.5) * bin_size
    return binned, bin_centers


def _save_dff_plot(series, output_path):
    """Save one whole-session heatmap panel for each dF/F series."""
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(16, max(3, 2.5 * len(series))),
        squeeze=False,
        constrained_layout=True,
    )

    for axis, (series_name, dff, timestamps) in zip(axes[:, 0], series):
        finite_values = np.abs(dff[np.isfinite(dff)])
        color_limit = (
            float(np.percentile(finite_values, 99))
            if finite_values.size
            else 1.0
        )
        if color_limit == 0:
            color_limit = 1.0

        image = axis.imshow(
            dff.T,
            aspect='auto',
            interpolation='nearest',
            cmap='RdBu_r',
            vmin=-color_limit,
            vmax=color_limit,
            extent=(timestamps[0], timestamps[-1], dff.shape[1], 0),
        )
        axis.set_title(series_name, fontsize=10)
        axis.set_ylabel('ROI index')
        figure.colorbar(image, ax=axis, label='dF/F', pad=0.01)

    axes[-1, 0].set_xlabel('Session time (s)')
    figure.suptitle(
        'Whole-session dF/F for all ROIs (100 ms means)', fontweight='bold'
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_raw_fluorescence_plot(series, output_path):
    """Save one whole-session heatmap panel for each raw fluorescence series."""
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(16, max(3, 2.5 * len(series))),
        squeeze=False,
        constrained_layout=True,
    )

    for axis, (series_name, fluorescence, timestamps) in zip(axes[:, 0], series):
        finite_values = fluorescence[np.isfinite(fluorescence)]
        if finite_values.size:
            color_min, color_max = np.percentile(finite_values, (1, 99))
        else:
            color_min, color_max = 0.0, 1.0
        if color_min == color_max:
            color_max = color_min + 1.0

        image = axis.imshow(
            fluorescence.T,
            aspect='auto',
            interpolation='nearest',
            cmap='viridis',
            vmin=color_min,
            vmax=color_max,
            extent=(timestamps[0], timestamps[-1], fluorescence.shape[1], 0),
        )
        axis.set_title(series_name, fontsize=10)
        axis.set_ylabel('ROI index')
        figure.colorbar(image, ax=axis, label='Raw fluorescence (F0)', pad=0.01)

    axes[-1, 0].set_xlabel('Session time (s)')
    figure.suptitle(
        'Whole-session raw fluorescence for all ROIs (100 ms means)',
        fontweight='bold',
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def compute_dff_qc(qc_folder, nwb_path, bin_size=0.1):
    """Write a 100 ms-binned whole-session heatmap for all dF/F series."""
    nwb_path = Path(nwb_path)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO

    with io_class(str(nwb_path), mode='r') as io:
        nwbfile = io.read()
        if 'ophys' not in nwbfile.processing:
            print('dF/F QC: no ophys processing module found; skipping.')
            return

        series = []
        ophys = nwbfile.processing['ophys']
        for interface in ophys.data_interfaces.values():
            if not isinstance(interface, pynwb.ophys.Fluorescence):
                continue
            for series_name, response_series in interface.roi_response_series.items():
                if not _is_dff_series(series_name):
                    continue
                dff, timestamps = _bin_activity(
                    response_series.data,
                    response_series.timestamps,
                    bin_size=bin_size,
                )
                series.append((series_name, dff, timestamps))

        if not series:
            print('dF/F QC: no dF/F ROI response series found; skipping.')
            return

        series.sort(key=lambda item: item[0])
        output_path = Path(qc_folder) / 'dff_all_rois_100ms.png'
        _save_dff_plot(series, output_path)
        print(f'dF/F QC: saved {output_path}')


def compute_raw_fluorescence_qc(qc_folder, nwb_path, bin_size=0.1):
    """Write a 100 ms-binned whole-session heatmap for all raw F0 series."""
    nwb_path = Path(nwb_path)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO

    with io_class(str(nwb_path), mode='r') as io:
        nwbfile = io.read()
        if 'ophys' not in nwbfile.processing:
            print('Raw fluorescence QC: no ophys processing module found; skipping.')
            return

        series = []
        ophys = nwbfile.processing['ophys']
        for interface in ophys.data_interfaces.values():
            if not isinstance(interface, pynwb.ophys.Fluorescence):
                continue
            for series_name, response_series in interface.roi_response_series.items():
                if not _is_raw_fluorescence_series(series_name):
                    continue
                fluorescence, timestamps = _bin_activity(
                    response_series.data,
                    response_series.timestamps,
                    bin_size=bin_size,
                )
                series.append((series_name, fluorescence, timestamps))

        if not series:
            print('Raw fluorescence QC: no raw F0 response series found; skipping.')
            return

        series.sort(key=lambda item: item[0])
        output_path = Path(qc_folder) / 'raw_fluorescence_all_rois_100ms.png'
        _save_raw_fluorescence_plot(series, output_path)
        print(f'Raw fluorescence QC: saved {output_path}')