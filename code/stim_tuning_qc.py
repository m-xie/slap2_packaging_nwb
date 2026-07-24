"""Stimulus-context tuning QC for packaged SLAP2 NWB files."""

from pathlib import Path
import warnings

import hdmf_zarr
import matplotlib.pyplot as plt
import numpy as np
import pynwb


_DFF_TAGS = ('dff', 'dfoverf', 'df_over_f', 'delta_f_over_f')


def _is_dff_series(name):
    normalized = name.lower().replace(' ', '_')
    return any(tag in normalized for tag in _DFF_TAGS)


def _find_stimulus_epochs(nwbfile):
    rows = []
    for table_name, table in nwbfile.intervals.items():
        stim_df = table.to_dataframe()
        rows.extend(
            (float(start), float(stop), table_name)
            for start, stop in zip(stim_df['start_time'], stim_df['stop_time'])
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2]))

    epochs = []
    for start, stop, table_name in rows:
        if epochs and epochs[-1]['name'] == table_name:
            epochs[-1]['stop'] = max(epochs[-1]['stop'], stop)
            epochs[-1]['presentations'] += 1
        else:
            epochs.append({
                'name': table_name,
                'start': start,
                'stop': stop,
                'presentations': 1,
            })

    name_totals = {}
    for epoch in epochs:
        name_totals[epoch['name']] = name_totals.get(epoch['name'], 0) + 1
    name_counts = {}
    for epoch in epochs:
        name = epoch['name']
        name_counts[name] = name_counts.get(name, 0) + 1
        epoch['label'] = (
            f"{name} {name_counts[name]}" if name_totals[name] > 1 else name
        )
    return epochs


def _window_median(dff, timestamps, start, stop):
    mask = (timestamps >= start) & (timestamps < stop)
    if not np.any(mask):
        return np.full(dff.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(dff[mask], axis=0)


def _time_coverage(timestamps, finite_rows, start, stop, bin_size=1.0):
    n_bins = max(1, int(np.ceil((stop - start) / bin_size)))
    valid_times = timestamps[
        finite_rows & (timestamps >= start) & (timestamps < stop)
    ]
    if valid_times.size == 0:
        return 0.0
    occupied_bins = np.unique(((valid_times - start) // bin_size).astype(int))
    occupied_bins = occupied_bins[(occupied_bins >= 0) & (occupied_bins < n_bins)]
    return len(occupied_bins) / n_bins


def _interpolate_time_for_display(values, factor=8):
    if values.shape[1] < 2 or factor <= 1:
        return values

    interpolated = np.full(
        (values.shape[0], (values.shape[1] - 1) * factor + 1),
        np.nan,
    )
    interpolated[:, ::factor] = values
    for time_index in range(values.shape[1] - 1):
        left = values[:, time_index]
        right = values[:, time_index + 1]
        finite = np.isfinite(left) & np.isfinite(right)
        finite_indices = np.flatnonzero(finite)
        if finite_indices.size == 0:
            continue
        columns = time_index * factor + np.arange(1, factor)
        weights = np.arange(1, factor) / factor
        interpolated[np.ix_(finite_indices, columns)] = (
            left[finite_indices, None] * (1 - weights)
            + right[finite_indices, None] * weights
        )
    return interpolated


def _roi_axis_ticks(count, maximum_ticks=5):
    return np.unique(
        np.linspace(1, count, min(count, maximum_ticks), dtype=int)
    )


def _calculate_stim_tuning(
    epochs,
    dff,
    timestamps,
    peri_window,
    peri_bin_size,
):
    n_rois = dff.shape[1]
    epoch_coverage = np.zeros(len(epochs))
    finite_rows = np.any(np.isfinite(dff), axis=1)

    for epoch_index, epoch in enumerate(epochs):
        epoch_coverage[epoch_index] = _time_coverage(
            timestamps,
            finite_rows,
            epoch['start'],
            epoch['stop'],
        )

    boundaries = [epoch['start'] for epoch in epochs[1:]]
    peri_times = np.arange(peri_window[0], peri_window[1], peri_bin_size)
    peri_activity = np.full((len(boundaries), len(peri_times), n_rois), np.nan)

    for boundary_index, boundary in enumerate(boundaries):
        local_mask = (
            np.isfinite(timestamps)
            & (timestamps >= boundary + peri_window[0])
            & (timestamps < boundary + peri_window[1])
        )
        local_dff = dff[local_mask]
        local_timestamps = timestamps[local_mask]
        for time_index, relative_time in enumerate(peri_times):
            peri_activity[boundary_index, time_index] = _window_median(
                local_dff,
                local_timestamps,
                boundary + relative_time,
                boundary + relative_time + peri_bin_size,
            )

    baseline_bins = peri_times < 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        baseline_mean = np.nanmean(
            peri_activity[:, baseline_bins], axis=1, keepdims=True
        )
        baseline_std = np.nanstd(
            peri_activity[:, baseline_bins], axis=1, keepdims=True
        )
    baseline_std[baseline_std == 0] = np.nan
    peri_zscore = (peri_activity - baseline_mean) / baseline_std

    return {
        'epoch_coverage': epoch_coverage,
        'peri_times': peri_times,
        'peri_bin_size': peri_bin_size,
        'peri_zscore': peri_zscore,
    }


def _save_stim_tuning_dashboard(channel_name, epochs, metrics, roi_groups, output_path):
    epoch_labels = [epoch['label'] for epoch in epochs]
    transition_labels = [f'{index + 1} -> {index + 2}' for index in range(len(epochs) - 1)]
    colors = plt.get_cmap('tab10')(np.arange(len(epochs)) % 10)

    fig = plt.figure(figsize=(17, 16), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=(0.75, 1, 1, 1))
    timeline_ax = fig.add_subplot(grid[0, :])
    transition_axes = [
        fig.add_subplot(grid[row, column])
        for row, column in (
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
            (3, 0),
        )
    ]
    key_ax = fig.add_subplot(grid[3, 1:])
    session_start = epochs[0]['start']
    for index, epoch in enumerate(epochs):
        timeline_ax.axvspan(epoch['start'], epoch['stop'], color=colors[index], alpha=0.65)
        timeline_ax.text(
            (epoch['start'] + epoch['stop']) / 2,
            0.82,
            str(index + 1),
            ha='center',
            va='center',
            fontsize=8,
        )
        epoch_midpoint = (epoch['start'] + epoch['stop']) / 2
        for group_index, (group_name, _) in enumerate(roi_groups):
            timeline_ax.plot(
                epoch_midpoint,
                metrics['epoch_coverage'][group_index, index],
                marker=('o', 's', '^', 'D')[group_index % 4],
                color='black',
                markerfacecolor=colors[group_index],
                markersize=5,
                label=group_name if index == 0 else None,
            )
    timeline_ax.set_xlim(session_start, epochs[-1]['stop'])
    timeline_ax.set_ylim(0, 1.05)
    timeline_ax.set_ylabel('Valid sample coverage')
    timeline_ax.set_xlabel('Time (s)')
    timeline_ax.set_title('Stimulus epochs and fluorescence coverage')
    timeline_ax.legend(loc='upper right', title='Fluorescence')
    timeline_ax.text(
        0.01,
        0.02,
        '\n'.join(f"{index + 1}: {label}" for index, label in enumerate(epoch_labels)),
        transform=timeline_ax.transAxes,
        fontsize=7,
        va='bottom',
    )

    zscore_values = metrics['peri_zscore']
    transition_image = None
    total_rois = zscore_values.shape[2]
    group_starts = np.r_[0, np.cumsum([count for _, count in roi_groups])[:-1]]
    group_boundaries = np.cumsum([count for _, count in roi_groups])[:-1]
    for index, (axis, label) in enumerate(zip(transition_axes, transition_labels)):
        display_values = _interpolate_time_for_display(
            zscore_values[index].T
        )
        transition_image = axis.imshow(
            display_values,
            aspect='auto',
            interpolation='nearest',
            cmap='viridis',
            extent=[
                metrics['peri_times'][0],
                metrics['peri_times'][-1] + metrics['peri_bin_size'],
                total_rois + 0.5,
                0.5,
            ],
            vmin=-3,
            vmax=3,
        )
        axis.axvline(0, color='black', linestyle='--', linewidth=1)
        for boundary in group_boundaries:
            axis.axhline(boundary + 0.5, color='black', linewidth=1.25)
        axis.set_title(
            f'{label}: {epoch_labels[index]} to {epoch_labels[index + 1]}',
            fontsize=9,
        )
        axis.set_xlabel('Time from context change (s)')
        tick_positions = []
        tick_labels = []
        for (group_name, group_count), group_start in zip(
            roi_groups, group_starts
        ):
            local_ticks = _roi_axis_ticks(group_count)
            tick_positions.extend(group_start + local_ticks)
            tick_labels.extend(local_ticks)
            axis.text(
                -0.16,
                group_start + (group_count + 1) / 2,
                f'{group_name} ROI index',
                transform=axis.get_yaxis_transform(),
                rotation=90,
                ha='center',
                va='center',
                fontweight='bold',
            )
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels)

    key_ax.axis('off')
    key_ax.text(
        0.02,
        0.95,
        'Reading the transition heatmaps',
        fontsize=11,
        fontweight='bold',
        va='top',
    )
    key_ax.text(
        0.02,
        0.82,
        'Each row is one ROI, with the same concatenated row order in every panel.\n'
        + '\n'.join(
            f'{group_name}: ROI indices 1-{count}'
            for (group_name, count), start in zip(
                roi_groups,
                group_starts,
            )
        )
        + '\nThe horizontal line marks the boundary between DMDs.\n'
        f"Analysis bin width: {metrics['peri_bin_size'] * 1000:.2f} ms.\n"
        'Colors are interpolated between adjacent finite bins for display only.\n'
        'Each ROI is z-scored to its own -5 to 0 s pre-transition baseline.\n'
        'Dark purple is z = -3; green is z = 0; bright yellow is z = +3.\n'
        'Values beyond three baseline standard deviations are clipped.\n'
        'The dashed line is the recorded context-change time.\n'
        'A vertical response band shifted from zero suggests a timing offset.\n'
        'Blank regions indicate inadequate fluorescence coverage.',
        fontsize=9,
        va='top',
        linespacing=1.5,
    )
    if transition_image is not None:
        fig.colorbar(
            transition_image,
            ax=transition_axes,
            label='dF/F z-score',
            shrink=0.75,
        )

    fig.suptitle(
        f'Combined DMD {channel_name} dF/F stimulus-context tuning QC',
        fontweight='bold',
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_stim_tuning_qc(
    qc_folder,
    nwb_path,
    peri_window=(-5.0, 5.0),
    peri_bin_size=0.25,
    peri_bin_percentile=25.0,
):
    """Compute stimulus-context tuning dashboards for all packaged dF/F series."""
    output_folder = Path(qc_folder) / 'stim_tuning'
    output_folder.mkdir(exist_ok=True)

    nwb_path = Path(nwb_path)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO
    with io_class(str(nwb_path), mode='r') as io:
        nwbfile = io.read()
        epochs = _find_stimulus_epochs(nwbfile)
        if len(epochs) < 2:
            print(f"Stim tuning QC: found {len(epochs)} stimulus epoch(s); skipping.")
            return
        if 'ophys' not in nwbfile.processing:
            print('Stim tuning QC: no ophys processing module found; skipping.')
            return

        print(f"Stim tuning QC: found {len(epochs)} chronological epochs")
        for index, epoch in enumerate(epochs, 1):
            print(
                f"  {index}: {epoch['label']} "
                f"[{epoch['start']:.3f}, {epoch['stop']:.3f}]"
            )

        ophys = nwbfile.processing['ophys']
        series_by_channel = {}
        for interface in ophys.data_interfaces.values():
            if not isinstance(interface, pynwb.ophys.Fluorescence):
                continue
            for series_name, series in interface.roi_response_series.items():
                if not _is_dff_series(series_name):
                    continue
                channel_name = series_name.rsplit('_', 1)[-1]
                dmd_name = series_name.split('_', 1)[0]
                series_by_channel.setdefault(channel_name, []).append(
                    (dmd_name, series_name, series)
                )

        for channel_name, channel_series in sorted(series_by_channel.items()):
            channel_metrics = []
            roi_groups = []
            loaded_series = []
            for dmd_name, series_name, series in sorted(channel_series):
                dff = np.asarray(series.data)
                timestamps = np.asarray(series.timestamps)
                loaded_series.append((dmd_name, dff, timestamps))

            analysis_bin_size = peri_bin_size
            if analysis_bin_size is None:
                positive_intervals = []
                for _, _, timestamps in loaded_series:
                    intervals = np.diff(timestamps)
                    positive_intervals.append(
                        intervals[np.isfinite(intervals) & (intervals > 0)]
                    )
                positive_intervals = np.concatenate(positive_intervals)
                if positive_intervals.size == 0:
                    raise ValueError(
                        f'No positive timestamp intervals found for {channel_name}.'
                    )
                analysis_bin_size = float(
                    np.percentile(positive_intervals, peri_bin_percentile)
                )
                print(
                    f'Stim tuning QC: {channel_name} bin width '
                    f'{analysis_bin_size * 1000:.3f} ms '
                    f'(timestamp interval percentile {peri_bin_percentile:g})'
                )

            for dmd_name, dff, timestamps in loaded_series:
                channel_metrics.append(
                    _calculate_stim_tuning(
                        epochs,
                        dff,
                        timestamps,
                        peri_window,
                        analysis_bin_size,
                    )
                )
                roi_groups.append((dmd_name, dff.shape[1]))

            metrics = {
                'epoch_coverage': np.stack([
                    item['epoch_coverage'] for item in channel_metrics
                ]),
                'peri_times': channel_metrics[0]['peri_times'],
                'peri_bin_size': channel_metrics[0]['peri_bin_size'],
                'peri_zscore': np.concatenate([
                    item['peri_zscore'] for item in channel_metrics
                ], axis=2),
            }
            output_path = output_folder / f'combined_dFF_{channel_name}_stim_tuning.png'
            _save_stim_tuning_dashboard(
                channel_name,
                epochs,
                metrics,
                roi_groups,
                output_path,
            )
            print(f'Stim tuning QC: saved {output_path}')