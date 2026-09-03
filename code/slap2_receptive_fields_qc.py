"""
SLAP2 receptive field QC utilities.
Called from the NWB packaging pipeline; not intended as a standalone script.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pynwb
import hdmf_zarr

# Candidate column names for stimulus position, in priority order
_X_COL_CANDIDATES = ('X', 'x_position', 'X_Position', 'x_pos', 'x_Pos')
_Y_COL_CANDIDATES = ('Y', 'y_position', 'Y_Position', 'y_pos', 'y_Pos')

# Substrings that identify a dF/F RoiResponseSeries (matched case-insensitively)
_DFF_TAGS = ('dff', 'dfoverf', 'df_over_f', 'delta_f_over_f')


def _is_dff_series(name):
    """Return True if name indicates a dF/F trace (case-insensitive)."""
    normalised = name.lower().replace(' ', '_')
    return any(tag in normalised for tag in _DFF_TAGS)


def _find_xy_cols(stim_df):
    """Return (xcol, ycol) names from stim_df, raising if not found."""
    xcol = next((c for c in _X_COL_CANDIDATES if c in stim_df.columns), None)
    ycol = next((c for c in _Y_COL_CANDIDATES if c in stim_df.columns), None)
    if xcol is None or ycol is None:
        raise ValueError(
            f"Could not find X/Y position columns in stim table. "
            f"Columns present: {list(stim_df.columns)}"
        )
    return xcol, ycol


def _sorted_unique_positions(stim_df, col):
    """Return sorted list of unique float positions from a column (handles str/int/float values)."""
    return sorted(set(float(v) for v in stim_df[col]))


def _filter_assigned_trials(stim_df):
    """Exclude stimuli outside retained SLAP2 imaging trials when available."""
    if 'slap2_trial_idx' not in stim_df.columns:
        return stim_df

    assigned = stim_df['slap2_trial_idx'] >= 0
    n_excluded = int((~assigned).sum())
    if n_excluded:
        print(
            f"RF QC: excluding {n_excluded}/{len(stim_df)} stimuli not assigned "
            "to a retained SLAP2 trial"
        )
    return stim_df.loc[assigned].reset_index(drop=True)


def _calculate_receptive_fields(stim_df, dff, timestamps, onset_delay, xcol, ycol):
    """
    Compute per-ROI receptive fields.

    Parameters
    ----------
    stim_df : pd.DataFrame
        Must have 'start_time', 'stop_time', xcol, ycol columns.
        Position values may be str, int, or float.
    dff : np.ndarray, shape (n_timestamps, n_rois)
    timestamps : np.ndarray, shape (n_timestamps,)
    onset_delay : float
        Seconds to shift the response window relative to stim onset.
    xcol, ycol : str

    Returns
    -------
    receptive_fields : np.ndarray, shape (n_rois, height, width)
        Per-ROI RF maps, each normalized to [0, 1].
    """
    sorted_x = _sorted_unique_positions(stim_df, xcol)
    sorted_y = _sorted_unique_positions(stim_df, ycol)
    height, width = len(sorted_y), len(sorted_x)
    n_rois = dff.shape[1]

    # Diagnostic: check timing overlap between stim table and fluorescence
    ts_min, ts_max = float(timestamps[0]), float(timestamps[-1])
    st_min = float(stim_df['start_time'].min())
    st_max = float(stim_df['stop_time'].max())
    print(f"  RF calc: fluorescence timestamps span [{ts_min:.3f}, {ts_max:.3f}]")
    print(f"  RF calc: stim table times span        [{st_min:.3f}, {st_max:.3f}]")
    if st_max < ts_min or st_min > ts_max:
        print("  RF calc: WARNING — stim times and fluorescence timestamps do not overlap! "
              "RF maps will be empty. Check that both use the same time reference.")

    receptive_fields = np.zeros((n_rois, height, width))
    stim_counts = np.zeros((height, width))
    n_valid = 0

    sorted_x_arr = np.array(sorted_x)
    sorted_y_arr = np.array(sorted_y)

    for i in range(len(stim_df)):
        t_start = stim_df['start_time'].iloc[i] + onset_delay
        t_end   = stim_df['stop_time'].iloc[i]  + onset_delay
        i0, i1  = np.searchsorted(timestamps, (t_start, t_end))
        if i0 >= i1:
            continue
        n_valid += 1
        response = np.nanmean(dff[i0:i1], axis=0)
        # Replace per-ROI NaN (all frames in window were NaN) with 0 so they
        # don't poison the accumulation for the entire position.
        response = np.where(np.isnan(response), 0.0, response)
        # Use nearest-neighbour matching to avoid float precision mismatches
        xi = int(np.argmin(np.abs(sorted_x_arr - float(stim_df[xcol].iloc[i]))))
        yi = int(np.argmin(np.abs(sorted_y_arr - float(stim_df[ycol].iloc[i]))))
        stim_counts[yi, xi] += 1
        receptive_fields[:, yi, xi] += response

    print(f"  RF calc: {n_valid}/{len(stim_df)} stimuli had valid response windows")
    rf_range = receptive_fields.max() - receptive_fields.min()
    print(f"  RF calc: RF value range before normalisation = {rf_range:.6g}")

    stim_counts[stim_counts == 0] = 1
    receptive_fields /= stim_counts[None, :, :]

    # Normalize each ROI map to [0, 1]
    mins = np.nanmin(receptive_fields, axis=(1, 2), keepdims=True)
    maxs = np.nanmax(receptive_fields, axis=(1, 2), keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    receptive_fields = (receptive_fields - mins) / ranges

    return receptive_fields


def _save_rf_grid(rfs, output_path, supertitle=None):
    """Save a grid of per-ROI RF heatmaps to a PNG file."""
    n_rois = len(rfs)
    n_cols = min(8, n_rois)
    n_rows = max(1, (n_rois + n_cols - 1) // n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), squeeze=False)
    if supertitle:
        fig.suptitle(supertitle, fontsize=10, fontweight='bold')
    for i, ax in enumerate(axs.flatten()):
        if i >= n_rois:
            ax.axis('off')
            continue
        ax.imshow(rfs[i], aspect='auto', interpolation='nearest')
        ax.set_title(f'ROI {i}', fontsize=7)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_receptive_field_qc(qc_folder, nwb_path, onset_delay=0.2):
    """
    Compute and save receptive field QC images for all DMD planes/channels.

    Writes images to qc_folder/receptive_fields/{table}_{dmd}_{channel}_rfs.png

    Parameters
    ----------
    qc_folder : str or Path
    nwb_path  : str or Path  — path to the NWB file (.nwb or .nwb.zarr)
    onset_delay : float
    """
    rf_folder = Path(qc_folder) / 'receptive_fields'
    rf_folder.mkdir(exist_ok=True)

    nwb_path = Path(nwb_path)
    if nwb_path.is_dir():
        io_class = hdmf_zarr.NWBZarrIO
    else:
        io_class = pynwb.NWBHDF5IO

    with io_class(str(nwb_path), mode='r') as io:
        nwbfile = io.read()
        _compute_receptive_field_qc(rf_folder, nwbfile, onset_delay)


def _compute_receptive_field_qc(rf_folder, nwbfile, onset_delay):

    # Locate RF stimulus intervals tables by name
    _RF_TAGS = ('rf_', 'rf ', 'receptive_field')
    rf_table_names = [
        k for k in nwbfile.intervals.keys()
        if any(tag in k.lower() for tag in _RF_TAGS)
    ]
    if not rf_table_names:
        print(f"RF QC: no matching intervals tables found among {list(nwbfile.intervals.keys())}; skipping.")
        return
    print(f"RF QC: found RF table(s): {rf_table_names}")

    if 'ophys' not in nwbfile.processing:
        print("RF QC: no ophys processing module found; skipping.")
        return
    ophys = nwbfile.processing['ophys']

    for table_name in rf_table_names:
        stim_df = nwbfile.intervals[table_name].to_dataframe()
        print(f"RF QC: processing table '{table_name}' ({len(stim_df)} rows)")
        stim_df = _filter_assigned_trials(stim_df)
        if stim_df.empty:
            print(f"RF QC: table '{table_name}' has no stimuli in retained SLAP2 trials; skipping.")
            continue

        xcol, ycol = _find_xy_cols(stim_df)
        n_x = len(_sorted_unique_positions(stim_df, xcol))
        n_y = len(_sorted_unique_positions(stim_df, ycol))
        print(f"RF QC: position columns '{xcol}'/'{ycol}', grid {n_x}x{n_y}")

        # Iterate over all Fluorescence interfaces; pick any dF/F series
        for interface in ophys.data_interfaces.values():
            if not isinstance(interface, pynwb.ophys.Fluorescence):
                continue
            for series_name, rrs in interface.roi_response_series.items():
                if not _is_dff_series(series_name):
                    continue

                dff        = np.array(rrs.data)        # (n_timestamps, n_rois)
                timestamps = np.array(rrs.timestamps)

                print(f"RF QC: computing RFs for {series_name} {dff.shape}")
                rfs = _calculate_receptive_fields(stim_df, dff, timestamps, onset_delay, xcol, ycol)

                out_path = rf_folder / f'{table_name}_{series_name}_rfs.png'
                _save_rf_grid(rfs, out_path, supertitle=f'{series_name}  | {table_name}')
                print(f"RF QC: saved {out_path}")