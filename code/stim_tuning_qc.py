"""Stimulus-context tuning QC for packaged SLAP2 NWB files."""

from pathlib import Path
import warnings

import hdmf_zarr
import matplotlib.pyplot as plt
import numpy as np
import pynwb


_DFF_TAGS = ('dff', 'dfoverf', 'df_over_f', 'delta_f_over_f')
_ORIENTATION_TABLE = 'standard_control'


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


def _calculate_orientation_tuning(
    stim_df,
    dff,
    timestamps,
    baseline_window=(-0.3, 0.0),
    response_window=(0.1, 0.6),
    peri_window=(-0.5, 1.5),
    peri_bin_size=0.05,
):
    """Calculate baseline-subtracted direction responses for two stimulus blocks."""
    presentations = stim_df.loc[stim_df['TrialType'] == 'single'].copy()
    if 'slap2_trial_idx' in presentations:
        presentations = presentations.loc[presentations['slap2_trial_idx'] >= 0]
    presentations = presentations.loc[np.isfinite(presentations['Orientation'])]
    presentations['orientation_degrees'] = np.mod(
        np.degrees(presentations['Orientation'].astype(float)), 360.0
    )

    block_values = np.sort(presentations['BlockNumber'].unique())
    orientations = np.sort(presentations['orientation_degrees'].unique())
    peri_times = np.arange(peri_window[0], peri_window[1], peri_bin_size)
    responses = np.full((len(presentations), dff.shape[1]), np.nan)
    peri_responses = np.full(
        (len(presentations), len(peri_times), dff.shape[1]), np.nan
    )

    for presentation_index, (_, presentation) in enumerate(presentations.iterrows()):
        onset = float(presentation['start_time'])
        baseline = _window_median(
            dff,
            timestamps,
            onset + baseline_window[0],
            onset + baseline_window[1],
        )
        responses[presentation_index] = (
            _window_median(
                dff,
                timestamps,
                onset + response_window[0],
                onset + response_window[1],
            )
            - baseline
        )
        for time_index, relative_time in enumerate(peri_times):
            peri_responses[presentation_index, time_index] = (
                _window_median(
                    dff,
                    timestamps,
                    onset + relative_time,
                    onset + relative_time + peri_bin_size,
                )
                - baseline
            )

    tuning_mean = np.full(
        (len(block_values), len(orientations), dff.shape[1]), np.nan
    )
    tuning_sem = np.full_like(tuning_mean, np.nan)
    presentation_counts = np.zeros((len(block_values), len(orientations)), dtype=int)
    peri_mean = np.full(
        (len(block_values), len(peri_times), dff.shape[1]), np.nan
    )

    for block_index, block_value in enumerate(block_values):
        block_mask = presentations['BlockNumber'].to_numpy() == block_value
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            peri_mean[block_index] = np.nanmean(peri_responses[block_mask], axis=0)
        for orientation_index, orientation in enumerate(orientations):
            mask = block_mask & np.isclose(
                presentations['orientation_degrees'].to_numpy(), orientation
            )
            presentation_counts[block_index, orientation_index] = int(mask.sum())
            if not np.any(mask):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tuning_mean[block_index, orientation_index] = np.nanmean(
                    responses[mask], axis=0
                )
                tuning_sem[block_index, orientation_index] = np.nanstd(
                    responses[mask], axis=0, ddof=1
                ) / np.sqrt(mask.sum())

    return {
        'block_values': block_values,
        'orientations': orientations,
        'presentation_counts': presentation_counts,
        'tuning_mean': tuning_mean,
        'tuning_sem': tuning_sem,
        'peri_times': peri_times,
        'peri_mean': peri_mean,
        'n_presentations': len(presentations),
        'baseline_window': baseline_window,
        'response_window': response_window,
    }


def _orientation_labels(orientations):
    return [f'{value:g}°' for value in orientations]


def _normalize_orientation_tuning(tuning_mean):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        center = np.nanmean(tuning_mean, axis=1, keepdims=True)
        scale = np.nanstd(tuning_mean, axis=1, keepdims=True)
    scale[scale == 0] = np.nan
    return (tuning_mean - center) / scale


def _orientation_heatmap(normalized_tuning, block_index, roi_order):
    """Return an ROI-by-direction matrix without advanced-index axis reordering."""
    return normalized_tuning[block_index][:, roi_order].T


def _save_orientation_summary(series_name, metrics, output_path):
    tuning_mean = metrics['tuning_mean']
    normalized = _normalize_orientation_tuning(tuning_mean)
    orientations = metrics['orientations']
    block_values = metrics['block_values']
    preferred = np.nanargmax(np.nan_to_num(normalized[0], nan=-np.inf), axis=0)
    roi_order = np.argsort(preferred, kind='stable')

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    image = None
    for block_index, axis in enumerate(axes[0]):
        image = axis.imshow(
            _orientation_heatmap(normalized, block_index, roi_order),
            aspect='auto',
            interpolation='nearest',
            cmap='coolwarm',
            vmin=-2.5,
            vmax=2.5,
        )
        axis.set_title(f'Block {block_values[block_index]} tuning, sorted by block 1')
        axis.set_xlabel('Direction')
        axis.set_ylabel('ROI (sorted)')
        axis.set_xticks(np.arange(len(orientations)))
        axis.set_xticklabels(_orientation_labels(orientations), rotation=45, ha='right')
    fig.colorbar(image, ax=axes[0], label='Within-ROI response z-score', shrink=0.8)

    colors = ('tab:blue', 'tab:orange')
    for block_index, block_value in enumerate(block_values):
        population_trace = np.nanmean(metrics['peri_mean'][block_index], axis=1)
        population_sem = np.nanstd(metrics['peri_mean'][block_index], axis=1) / np.sqrt(
            metrics['peri_mean'].shape[2]
        )
        axes[1, 0].plot(
            metrics['peri_times'], population_trace,
            color=colors[block_index], label=f'Block {block_value}',
        )
        axes[1, 0].fill_between(
            metrics['peri_times'],
            population_trace - population_sem,
            population_trace + population_sem,
            color=colors[block_index], alpha=0.2,
        )
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].axvspan(0, 0.343, color='0.8', alpha=0.4, label='Stimulus')
    axes[1, 0].set(
        title='Population response to grating onset',
        xlabel='Time from onset (s)',
        ylabel='Baseline-subtracted dF/F',
    )
    axes[1, 0].legend()

    correlations = np.full(tuning_mean.shape[2], np.nan)
    for roi_index in range(tuning_mean.shape[2]):
        first = tuning_mean[0, :, roi_index]
        second = tuning_mean[1, :, roi_index]
        finite = np.isfinite(first) & np.isfinite(second)
        if finite.sum() >= 3 and np.std(first[finite]) > 0 and np.std(second[finite]) > 0:
            correlations[roi_index] = np.corrcoef(first[finite], second[finite])[0, 1]
    axes[1, 1].hist(correlations[np.isfinite(correlations)], bins=np.linspace(-1, 1, 21))
    median_correlation = np.nanmedian(correlations)
    axes[1, 1].axvline(median_correlation, color='black', linestyle='--')
    axes[1, 1].set(
        title=f'Cross-block tuning reliability (median r={median_correlation:.2f})',
        xlabel='Pearson r across directions',
        ylabel='ROI count',
        xlim=(-1, 1),
    )

    count_range = (
        int(metrics['presentation_counts'].min()),
        int(metrics['presentation_counts'].max()),
    )
    fig.suptitle(
        f'{series_name} orientation tuning | {metrics["n_presentations"]} presentations | '
        f'{count_range[0]}-{count_range[1]} repeats/direction/block\n'
        f'baseline {metrics["baseline_window"]} s, response {metrics["response_window"]} s',
        fontweight='bold',
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_orientation_roi_grid(series_name, metrics, output_path):
    n_rois = metrics['tuning_mean'].shape[2]
    n_columns = 6
    n_rows = int(np.ceil(n_rois / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(18, 2.6 * n_rows),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    colors = ('tab:blue', 'tab:orange')
    for roi_index, axis in enumerate(axes.flat):
        if roi_index >= n_rois:
            axis.axis('off')
            continue
        for block_index, block_value in enumerate(metrics['block_values']):
            axis.errorbar(
                metrics['orientations'],
                metrics['tuning_mean'][block_index, :, roi_index],
                yerr=metrics['tuning_sem'][block_index, :, roi_index],
                color=colors[block_index],
                marker='o',
                markersize=2.5,
                linewidth=1,
                label=f'Block {block_value}' if roi_index == 0 else None,
            )
        axis.axhline(0, color='0.7', linewidth=0.7)
        axis.set_title(f'ROI {roi_index}', fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel('Direction (deg)', fontsize=8)
    fig.supylabel('Baseline-subtracted dF/F')
    fig.suptitle(f'{series_name} per-ROI orientation tuning', fontweight='bold')
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def compute_orientation_tuning_qc(qc_folder, nwb_path):
    """Compute two-block orientation tuning plots for all packaged dF/F series."""
    output_folder = Path(qc_folder) / 'orientation_tuning'
    output_folder.mkdir(exist_ok=True)

    nwb_path = Path(nwb_path)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO
    with io_class(str(nwb_path), mode='r') as io:
        nwbfile = io.read()
        if _ORIENTATION_TABLE not in nwbfile.intervals:
            print(f'Orientation tuning QC: no {_ORIENTATION_TABLE} table; skipping.')
            return
        if 'ophys' not in nwbfile.processing:
            print('Orientation tuning QC: no ophys processing module; skipping.')
            return

        stim_df = nwbfile.intervals[_ORIENTATION_TABLE].to_dataframe()
        block_values = np.sort(stim_df['BlockNumber'].unique())
        if len(block_values) != 2:
            print(
                f'Orientation tuning QC: expected two control blocks, found '
                f'{len(block_values)}; skipping.'
            )
            return

        ophys = nwbfile.processing['ophys']
        for interface in ophys.data_interfaces.values():
            if not isinstance(interface, pynwb.ophys.Fluorescence):
                continue
            for series_name, series in interface.roi_response_series.items():
                if not _is_dff_series(series_name):
                    continue
                metrics = _calculate_orientation_tuning(
                    stim_df,
                    np.asarray(series.data),
                    np.asarray(series.timestamps),
                )
                summary_path = output_folder / f'{series_name}_orientation_summary.png'
                roi_path = output_folder / f'{series_name}_orientation_rois.png'
                _save_orientation_summary(series_name, metrics, summary_path)
                _save_orientation_roi_grid(series_name, metrics, roi_path)
                print(
                    f'Orientation tuning QC: saved {summary_path} and {roi_path}'
                )