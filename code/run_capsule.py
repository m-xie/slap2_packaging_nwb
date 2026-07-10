import h5py
import numpy as np
import pynwb
import hdmf_zarr
from datetime import datetime
from pathlib import Path
import harp_utils
import slap2_synching as slap2_sync
import slap2_receptive_fields_qc as slap2_rf_qc
import json
import pandas as pd
import argparse
import shutil
import re
import csv
import matplotlib.pyplot as plt


data_folder = Path("../data")
results_folder = Path("../results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_session_dir", type=str, default="slap2_session")
    parser.add_argument("--input_processed_dir", type=str, default="slap2_processed")
    parser.add_argument("--input_nwb_dir", type=str, default=f'nwb')
    parser.add_argument("--expected_n_trials", type=int, default=None)
    parser.add_argument("--rf_onset_delay", type=float, default=0.2,
                        help="Onset delay in seconds for RF response windows (default: 0.2)")
    args = parser.parse_args()
    expected_n_trials = args.expected_n_trials
    rf_onset_delay = args.rf_onset_delay
    input_nwb_dir = data_folder / Path(args.input_nwb_dir)
    session_path = data_folder / Path(args.input_session_dir)
    processed_path = data_folder / Path(args.input_processed_dir)
    print("SESSION PATH", session_path)
    print("PROCESSED PATH", processed_path)

    print('INPUT NWB DIR', input_nwb_dir)
    assert input_nwb_dir.exists(), "Input NWB dir does not exist"
    nwb_files = [p for p in input_nwb_dir.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, f"Attach one base NWB file data at a time. {len(nwb_files)} found"
    input_nwb_path = nwb_files[0]
    print('INPUT NWB', input_nwb_path)

    for file in results_folder.iterdir():
        shutil.rmtree(file)
    print(f'cleared results folder: {list(results_folder.iterdir())}')

    # determine if file is zarr or hdf5, and copy it to results
    result_nwb_path = results_folder / input_nwb_path.name
    if input_nwb_path.is_dir():
        assert (input_nwb_path / ".zattrs").is_file(), f"{input_nwb_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = hdmf_zarr.NWBZarrIO
        shutil.copytree(input_nwb_path, result_nwb_path, dirs_exist_ok=True)
    else:
        NWB_BACKEND = "hdf5"
        io_class = pynwb.NWBHDF5IO
        shutil.copyfile(input_nwb_path, result_nwb_path)
    print(f"NWB backend: {NWB_BACKEND}")

    nwb_path = results_folder / f"{session_path.name}.nwb"
    instrument_json_path = next(session_path.glob("instrument.json"))
    acquisition_json_path = next(session_path.glob("acquisition.json"))
    harp_path = next(session_path.rglob('*.harp'))
    experiment_summary_path = next(processed_path.rglob('*experiment_summary.h5'))
    orientations_csv = next((session_path / 'behavior').rglob('orientations_orientations0.csv'))
    log_csv = next((session_path / 'behavior').rglob('orientations_logger.csv'))

    print('using instrument json:', instrument_json_path)
    with open(instrument_json_path, "r") as f:
        instrument_json = json.load(f)
    print('using acquisition json:', acquisition_json_path)
    with open(acquisition_json_path, "r") as f:
        acquisition_json = json.load(f)

    with io_class(str(result_nwb_path), "r+") as nwb_io:
        nwbfile = nwb_io.read()
        with h5py.File(experiment_summary_path, "r") as experiment_summary:
            harp_data = harp_utils.extract_harp(harp_path)
            qc_folder = results_folder / 'qc'
            qc_folder.mkdir(exist_ok=True)
            (qc_folder / 'syncing').mkdir(exist_ok=True)
            add_stim_table(nwbfile, orientations_csv, log_csv, harp_data)
            add_ophys_to_nwb(experiment_summary, nwbfile, instrument_json, acquisition_json, harp_data, session_path, qc_folder)
        nwb_io.write(nwbfile)
    slap2_rf_qc.compute_receptive_field_qc(qc_folder, result_nwb_path, onset_delay=rf_onset_delay)
    print(f'Wrote output slap2 nwb to {result_nwb_path}')


def get_dmd_name(plane_name_str):
    m = re.search(r'(?i)((?:path|dmd)(\d+))', plane_name_str)
    if m:
        number = m.group(2)       # e.g. "1"
        full_name = f"DMD{number}"  # always normalize to DMD1, DMD2, ...
        return full_name, number
    return None, None


def get_expected_n_frames(experiment_summary):
    trial_num_frames = experiment_summary['DMD1']['frame_info']['trial_num_frames']
    assert len(trial_num_frames) == len(experiment_summary['DMD2']['frame_info']['trial_num_frames']), 'DMDs have different numbers of trials'
    return len(trial_num_frames)


def find_slap2_trial_index(time, start_trials, end_trials):
    # Find index where `time` would be inserted to keep start_trials sorted
    idx = np.searchsorted(start_trials, time, side='right') - 1
    
    # Check bounds and if time fits in the trial interval
    if idx < 0 or idx >= len(start_trials):
        raise Exception(f"Time {time} is out of trial bounds")
    if time > end_trials[idx]:
        # raise Exception(f"Time {time} not within trial end {end_trials[idx]}")
        return -1
    return idx


def read_stim_csv(filepath):
    default_column_names = ['stim_id','delay','duration','diameter','x','y','contrast','spatial_frequency','temporal_frequency','orientation']
    _OLD_TO_NEW_RENAMES = {
        'stim_id':            'Id',
        'delay':              'Delay',
        'duration':           'Duration',
        'x':                  'X',
        'y':                  'Y',
        'contrast':           'Contrast',
        'spatial_frequency':  'SpatialFrequency',
        'temporal_frequency': 'TemporalFrequency',
        'orientation':        'Orientation',
    }

    with open(filepath, newline='') as f:
        first_row = next(csv.reader(f))

    # If the first value is numeric, there's no header
    try:
        float(first_row[0])
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(filepath)
    else:
        if len(first_row) != len(default_column_names):
            raise ValueError(
                f"Column count mismatch: CSV has {len(first_row)} columns, "
                f"expected {len(default_column_names)} from column_names"
            )
        df = pd.read_csv(filepath, header=None, names=default_column_names)

    # Normalize old-format columns to new schema
    if 'diameter' in df.columns:
        df = df.copy()
        df['DiameterX'] = df['diameter']
        df['DiameterY'] = df['diameter']
        df = df.drop(columns=['diameter'])
        df = df.rename(columns=_OLD_TO_NEW_RENAMES)

    return df


def add_stim_table(nwbfile, orientations_table, log_csv, harp_data):
    gratings_df = read_stim_csv(orientations_table)

    log_df = pd.read_csv(log_csv)
    # spacebar_time = np.array(log_df.loc[log_df['Value'] == 'SPACEBAR', 'Timestamp'].tolist())[0]

    slap2_start_times = harp_data['normalized_slap2_start']
    slap2_end_times = harp_data['normalized_slap2_end']
    # any gratings before the spacebar are erroneous (digital line is noisy perhaps)
    # start_gratings_times = np.array([t for t in harp_data['normalized_start_gratings'] if t >= spacebar_time])
    start_gratings_times = harp_data['normalized_start_gratings']

    # We check there are as many gratings presentation as there are timing data in HARP, tolerate a small difference and truncate
    diff = len(gratings_df) - len(start_gratings_times)
    if abs(diff) > 3:
        raise ValueError(
            f"Mismatch between number of grating presentations {len(gratings_df)} "
            f"and HARP timing data {len(start_gratings_times)}"
        )
    elif diff > 0:
        # gratings_df has more entries — trim from the beginning
        gratings_df = gratings_df.iloc[diff:].reset_index(drop=True)
    elif diff < 0:
        # start_gratings_times has more entries — trim from the beginning
        start_gratings_times = start_gratings_times[-diff:]
    
    start_time = []
    stop_time = []
    slap2_trial_idxs = []
    for i, row in gratings_df.iterrows():
        start_time.append(start_gratings_times[i])
        stop_time.append(start_gratings_times[i] + row['Duration'])
        slap2_trial_idxs.append(find_slap2_trial_index(start_gratings_times[i], slap2_start_times, slap2_end_times))
    gratings_df['slap2_trial_idx'] = slap2_trial_idxs

    # Attach timing columns so they travel with the df during splitting
    gratings_df['start_time'] = start_time
    gratings_df['stop_time']  = stop_time

    # Split by BlockType and add each block as its own TimeIntervals table
    if 'BlockType' in gratings_df.columns:
        block_groups = gratings_df.groupby('BlockType', sort=False)
    else:
        # No BlockType column — fall back to a single 'gratings' table
        block_groups = [('gratings', gratings_df)]

    for block_name, block_df in block_groups:
        block_df = block_df.reset_index(drop=True)

        # Drop columns that are entirely NaN in this sub-table
        block_df = block_df.dropna(axis='columns', how='all')

        block_start = block_df.pop('start_time').tolist()
        block_stop  = block_df.pop('stop_time').tolist()

        table = pynwb.file.TimeIntervals(
            name=str(block_name),
            description=f'Stimulus intervals for block type: {block_name}'
        )

        # Establish row count before add_column
        table.id.data.extend(range(len(block_df)))
        table['start_time'].data.extend(block_start)
        table['stop_time'].data.extend(block_stop)

        for col in block_df.columns:
            data = block_df[col].tolist()
            if col == 'id':
                data = [int(v) for v in data]
            table.add_column(name=col, description=f'{col} from stimulus dataframe', data=data)

        nwbfile.add_time_intervals(table)
        print(f"Added intervals table '{block_name}' with {len(block_df)} rows and {len(block_df.columns)} columns")


def create_optical_channel(acquisition, imaging_channel, dmd_name):
    """
    Create an OpticalChannel object for a DMD using acquisition.json fields.
    nwbfile: NWBFile object (not used, for interface consistency).
    acquisition: Dictionary with acquisition metadata.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    Returns: OpticalChannel object.
    """
    emission_lambda = float(imaging_channel["emission_wavelength"])
    detector_name = imaging_channel["detector"]["device_name"]
    target = imaging_channel["intended_measurement"]

    description = f"{detector_name} detector, targetting {target}"
    optical_channel = pynwb.ophys.OpticalChannel(
        name=f"OpticalChannel_{dmd_name}",
        description=description,
        emission_lambda=emission_lambda
    )
    return optical_channel


def channel_belongs_to_plane(dmd_name, plane_key, channel):
    # The acquisition JSON may refer to the plane by either its normalized NWB name
    # (e.g. 'DMD1') or the original source name (e.g. 'Path1') — check both.
    channel_name = channel["channel_name"].lower().replace(" ", "")
    return any(alias.lower() in channel_name for alias in (dmd_name, plane_key))


def create_imaging_plane(nwbfile, dmd_name, device, acquisition_json, plane_key=None):
    plane_key = plane_key or dmd_name
    """
    Create an ImagingPlane object for a given DMD and add it to the NWBFile.
    nwbfile: NWBFile object to add the imaging plane to.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    device: Device object for this plane.
    instrument_json: Dictionary with instrument metadata (expects 'light_sources' and 'detectors').
    Returns: ImagingPlane object.
    """
    this_plane_channels = [
        channel
        for stream in acquisition_json["data_streams"]
        for config in stream["configurations"]
        if config["object_type"] == "Imaging config"
        for channel in config["channels"]
        if channel_belongs_to_plane(dmd_name, plane_key, channel)
    ]
    assert this_plane_channels, f"No channels found in metadata for plane {dmd_name}"
    this_plane_images = [
        image
        for stream in acquisition_json["data_streams"]
        for config in stream["configurations"]
        if config["object_type"] == "Imaging config"
        for image in config["images"]
        if channel_belongs_to_plane(dmd_name, plane_key, image)
    ]

    excitation_lambda = this_plane_channels[0]["light_sources"][0]["wavelength"]
    imaging_rate = min(this_plane_images[0]["planes"][0]["frame_rates"])

    indicators = [channel["intended_measurement"] for channel in this_plane_channels]
    location = this_plane_images[0]["planes"][0]["targeted_structure"]["acronym"]

    imaging_plane = nwbfile.create_imaging_plane(
        name=f"ImagingPlane_{dmd_name}",
        optical_channel=[create_optical_channel(acquisition_json, c, dmd_name) for c in this_plane_channels],
        description=f"Imaging plane for {dmd_name}",
        device=device,
        excitation_lambda=float(excitation_lambda) if excitation_lambda is not None else 920.0,
        imaging_rate=float(imaging_rate) if imaging_rate is not None else 30.0,
        indicator=", ".join(indicators),
        location=location,
        manifold=None
    )
    return imaging_plane


def get_pixel_mask(mask):
    """
    Given a 3D mask (z x rows x cols), return a list of (row, col, weight) triplets
    from a max-projection along z, plus the z_min and z_max indices of the ROI.
    This is the NWB-compliant, space-efficient way to store ROIs for large FOVs.
    Returns: pixel_mask (list of (row, col, weight)), z_min (int), z_max (int)
    """
    projected = np.nanmax(mask, axis=0)
    valid = np.logical_and(projected != 0, ~np.isnan(projected))
    rows, cols = np.where(valid)
    weights = projected[rows, cols]
    pixel_mask = [(int(r), int(c), float(w)) for r, c, w in zip(rows, cols, weights)]
    z_active = np.any(np.logical_and(mask != 0, ~np.isnan(mask)), axis=(1, 2))
    z_indices = np.where(z_active)[0]
    z_min = int(z_indices.min())
    z_max = int(z_indices.max())
    return pixel_mask, z_min, z_max


def add_image_segmentation(experiment_summary, imaging_plane, dmd_name, image_segmentation, plane_key=None):
    plane_key = plane_key or dmd_name
    """
    Add a PlaneSegmentation for a DMD to the shared ImageSegmentation object.
    For each ROI, store a pixel_mask (list of (row, col, weight)) for space-efficient NWB compliance.
    Also stores z_min and z_max columns indicating the z-axis extent of each ROI.
    experiment_summary: Open HDF5 file handle.
    imaging_plane: ImagingPlane object for this DMD.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    image_segmentation: Shared ImageSegmentation object for the ophys module.
    Returns: DynamicTableRegion referencing all ROIs for this plane.
    """
    ps = image_segmentation.create_plane_segmentation(
        name=f"PlaneSegmentation_{dmd_name}",
        description=f"ROIs for {dmd_name}",
        imaging_plane=imaging_plane
    )
    ps.add_column(name='z_min', description='Minimum z-index of the ROI in the DMD z-axis')
    ps.add_column(name='z_max', description='Maximum z-index of the ROI in the DMD z-axis')
    # matlab saves the data column-major format, so we must transpose
    fp_masks = experiment_summary[plane_key]['sources']['spatial']['profiles'][()].transpose()
    # after transposition, shape should be (n_sources, z, x, y)
    print("SHAPE OF ROIS FROM EXPERIMENT SUMMARY:",fp_masks.shape)
    n_rois = fp_masks.shape[0]
    for mask in fp_masks:
        pixel_mask, z_min, z_max = get_pixel_mask(mask)
        ps.add_roi(pixel_mask=pixel_mask, z_min=z_min, z_max=z_max)
    roi_table_region = ps.create_roi_table_region(
        region=list(range(n_rois)),
        description=f"All ROIs for {dmd_name}"
    )
    return roi_table_region


def sync_slap2_fluorescence(dmd_name, dmd_num, experiment_summary, slap2_meta_h5, harp_data, trial_line_time_maps=None, primary_qc=None, qc_folder=None, dat_paths=None, plane_key=None):
    plane_key = plane_key or dmd_name
    """
    Extract fluorescence traces and compute HARP-aligned timestamps for one SLAP2 DMD plane.

    Parameters
    ----------
    dmd_name : str
        Name of the DMD/plane (e.g. 'DMD1').
    dmd_num : str or int
        DMD number (e.g. '1' or '2').
    experiment_summary : h5py.File
        Open HDF5 file handle for the experiment summary.
    slap2_meta_h5 : h5py.File
        Open HDF5 file handle for this DMD's .meta file.
    harp_data : dict
        Dict from harp_utils.extract_harp containing clock signal arrays.
    trial_line_time_maps : list or None
        None for the primary plane (DMD1); for secondary planes, pass the
        trial_line_time_maps returned from the primary plane call.
    primary_qc : dict or None
        plane_qc dict returned from the primary DMD call. Required for QC
        plotting on secondary planes; ignored for the primary plane.
    qc_folder : Path or None
        If provided and this is the secondary plane, QC figures are saved here
        after both planes have been synced.

    Returns
    -------
    fluorescence : dict
        {'F0': array, 'dF_denoised': array, 'events': array}, each (n_tp, n_ch, n_roi).
    timestamps : np.ndarray
        HARP-referenced timestamp per timepoint.
    trial_line_time_maps : list or None
        Primary plane only: per-trial line-to-time maps needed to sync the secondary
        plane. Pass directly into the next DMD's call. None for secondary planes.
    plane_qc : dict
        Data needed for QC plotting. Primary plane includes 'sync_qc_values' and
        'lines_per_cycle'; all planes include 'frame_line_idxs', 'trial_num_frames',
        and 'timestamps'.
    """
    dmd_group = experiment_summary[plane_key]
    temporal_sources = dmd_group['sources']['temporal']

    f0_data          = temporal_sources['F0'][()]
    df_denoised_data = temporal_sources['dF_denoised'][()]
    events_data      = temporal_sources['events'][()]

    assert df_denoised_data.shape == f0_data.shape == events_data.shape, "Traces must have same shape"

    frame_line_idxs  = dmd_group['frame_info']['frame_line_idxs'][0]
    trial_num_frames = dmd_group['frame_info']['trial_num_frames'][()][0]
    lines_per_cycle  = slap2_meta_h5['AcquisitionContainer']['ParsePlan']['linesPerCycle'][0]

    print("###: extracted traces with shape:", f0_data.shape)

    fluorescence = {
        'F0':          f0_data,
        'dF_denoised': df_denoised_data,
        'events':      events_data,
    }

    if dat_paths is not None:
        trial_num_cycles = slap2_sync.get_trial_num_cycles(dat_paths, len(trial_num_frames))
    else:
        trial_num_cycles = None

    print(f"{dmd_name}: trial_num_frames = {trial_num_frames}")
    print(f"{dmd_name}: trial_num_cycles = {trial_num_cycles}")

    if int(dmd_num) == 1:
        timestamps, out_maps, sync_qc_values = slap2_sync.get_slap2_primary_plane_timestamps(
            frame_line_idxs, trial_num_frames, lines_per_cycle,
            harp_data['slap2_cycle_clock_signal'],
            harp_data['slap2_cycle_clock_times'],
            trial_num_cycles=trial_num_cycles,
        )
        print(f"PRODUCED {len(timestamps)} SLAP2 TIMESTAMPS for {len(f0_data)} (DMD{dmd_num})")
        plane_qc = {
            'frame_line_idxs': frame_line_idxs,
            'trial_num_frames': trial_num_frames,
            'lines_per_cycle': lines_per_cycle,
            'timestamps': timestamps,
            'trial_line_time_maps': out_maps,
            'sync_qc_values': sync_qc_values,
        }
        return fluorescence, timestamps, out_maps, plane_qc
    else:
        timestamps = slap2_sync.get_slap2_secondary_plane_timestamps(
            frame_line_idxs, trial_num_frames, trial_line_time_maps
        )
        print(f"PRODUCED {len(timestamps)} SLAP2 TIMESTAMPS for {len(f0_data)} (DMD{dmd_num})")
        plane_qc = {
            'frame_line_idxs': frame_line_idxs,
            'trial_num_frames': trial_num_frames,
            'timestamps': timestamps,
        }
        if qc_folder is not None and primary_qc is not None:
            slap2_sync.plot_slap2_sync_qc(
                harp_data['slap2_cycle_clock_signal'],
                harp_data['slap2_cycle_clock_times'],
                primary_qc['frame_line_idxs'],
                primary_qc['trial_num_frames'],
                primary_qc['trial_line_time_maps'],
                primary_qc['timestamps'],
                primary_qc['sync_qc_values'],
                primary_lines_per_cycle=primary_qc['lines_per_cycle'],
                secondary_frame_line_idxs=frame_line_idxs,
                secondary_trial_num_frames=trial_num_frames,
                secondary_timestamps=timestamps,
                qc_folder=qc_folder,
            )
        return fluorescence, timestamps, None, plane_qc


def add_fluorescence(fluorescence, timestamps, dmd_name, roi_table, ophys_mod):
    """
    Package pre-synced fluorescence traces into NWB Fluorescence/RoiResponseSeries objects.

    Parameters
    ----------
    fluorescence : dict
        {'F0': array, 'dF_denoised': array} — arrays shaped (n_tp, n_ch, n_roi).
    timestamps : np.ndarray
        HARP-aligned timestamps, one per timepoint.
    dmd_name : str
        Name of the DMD/plane (e.g. 'DMD1').
    roi_table : DynamicTableRegion
        Region referencing ROIs for this plane.
    ophys_mod : pynwb.ProcessingModule
        Ophys module to add the Fluorescence interface to.
    """
    f0_data         = fluorescence['F0']
    df_denoised_data = fluorescence['dF_denoised']

    eps = 1e-6
    dff_data = df_denoised_data / (f0_data + eps)

    fluorescence_obj = pynwb.ophys.Fluorescence(name=f"Fluorescence_{dmd_name}")
    ophys_mod.add_data_interface(fluorescence_obj)

    for ch_idx, ch_name in enumerate(["green", "red"]):
        if ch_idx >= f0_data.shape[1]:
            print("Channel {ch_idx}:{ch_name} not in temporal sources, skipping")
            continue

        rrs_f0 = pynwb.ophys.RoiResponseSeries(
            name=f"{dmd_name}_F0_{ch_name}",
            data=f0_data[:, ch_idx, :],
            rois=roi_table,
            unit='a.u.',
            timestamps=timestamps,
            description=f"F0 (raw fluorescence) traces from {dmd_name}, {ch_name} channel"
        )
        fluorescence_obj.add_roi_response_series(rrs_f0)

        rrs_dff = pynwb.ophys.RoiResponseSeries(
            name=f"{dmd_name}_dFF_{ch_name}",
            data=dff_data[:, ch_idx, :],
            rois=roi_table,
            unit='a.u.',
            timestamps=timestamps,
            description=f"dFF traces from {dmd_name}, {ch_name} channel"
        )
        fluorescence_obj.add_roi_response_series(rrs_dff)

def add_mean_images(experiment_summary, dmd_name, ophys_mod, plane_key=None):
    plane_key = plane_key or dmd_name
    """
    Add registered and motion-corrected mean and activity images for a DMD to the ophys ProcessingModule as ImageSeries objects.
    mean_im: shape (channels, height, width) from HDF5 — channel axis is inferred as the smallest dimension.
    act_im: shape (height, width) from HDF5.
    Each channel of the mean image is stored as a single-frame ImageSeries.
    The activity image is stored as a single-frame ImageSeries.
    """
    mean_im = experiment_summary[plane_key]['visualizations']['mean_im'][()]
    act_im = experiment_summary[plane_key]['visualizations']['act_im'][()]

    if mean_im.ndim == 2:
        mean_im = mean_im[None, ...]  # add channel dim: (1, height, width)
    elif mean_im.ndim == 4:
        # Newer format: (H, W, Z, C) — max-project along Z (axis 2) to get (H, W, C)
        print(f"add_mean_images: {dmd_name} mean_im is 4D {mean_im.shape}, max-projecting along Z axis")
        mean_im = mean_im.max(axis=2)
        # fall through to 3D handling below
    
    if mean_im.ndim == 3:
        # Channel axis is whichever dimension is smallest (2 channels << image pixels)
        ch_axis = int(np.argmin(mean_im.shape))
        if ch_axis != 0:
            print(f"add_mean_images: reordering {dmd_name} mean_im axes from {mean_im.shape} "
                  f"(channel axis detected at position {ch_axis})")
            mean_im = np.moveaxis(mean_im, ch_axis, 0)
    elif mean_im.ndim != 2:
        raise ValueError(f"Unexpected mean_im ndim={mean_im.ndim} for {dmd_name}, shape={mean_im.shape}")

    n_channels = mean_im.shape[0]
    if n_channels > 10:
        raise ValueError(f"Inferred {n_channels} channels for {dmd_name} mean_im — "
                         f"shape {mean_im.shape} looks wrong after axis reorder")

    for ch in range(n_channels):
        mean_image_series = pynwb.image.ImageSeries(
            name=f"{dmd_name}_mean_image_channel{ch}",
            data=mean_im[ch][None, ...],  # shape (1, height, width)
            unit='a.u.',
            format='raw',
            timestamps=[0.0],
            description=f"Registered mean image for {dmd_name}, channel {ch} (motion corrected)"
        )
        ophys_mod.add_data_interface(mean_image_series)
    act_image_series = pynwb.image.ImageSeries(
        name=f"{dmd_name}_activity_image",
        data=act_im[None, ...],  # shape (1, height, width)
        unit='a.u.',
        format='raw',
        timestamps=[0.0],
        description=f"Registered activity image for {dmd_name} (motion corrected)"
    )
    ophys_mod.add_data_interface(act_image_series)


def create_device(nwbfile, instrument_json):
    """
    Create and return a Device object using all relevant fields from instrument.json.
    nwbfile: NWBFile object to add the device to.
    instrument_json: Dictionary with instrument metadata.
    Returns: Device object.
    """
    name = instrument_json["instrument_id"]
    manufacturer = "MBF Bioscience"
    calibrations = instrument_json.get("calibrations",[])
    last_calibrated = calibrations[-1] if calibrations else None
    description = f"Slap2 microscope. Temperature control: {instrument_json.get('temperature_control',None)}, Humidity control: {instrument_json.get('humidity_control',None)}. Calibration date: {last_calibrated}. Notes: {instrument_json['notes']}"
    device = nwbfile.create_device(
        name=name,
        description=description,
        manufacturer=manufacturer
    )
    return device


def add_ophys_to_nwb(experiment_summary, nwbfile, instrument_json, acquisition_json, harp_data, session_path, qc_folder=None):
    """
    Build the full ophys structure in the NWB file, iterating over DMDs (planes).
    For each DMD, create imaging plane, ROI table, add fluorescence, and add mean images.
    nwbfile: NWBFile object to populate.
    instrument_json: Dictionary with instrument metadata.
    """
    device = create_device(nwbfile, instrument_json)
    ophys_mod = pynwb.ProcessingModule('ophys', 'Ophys processing module')
    nwbfile.add_processing_module(ophys_mod)
    image_segmentation = pynwb.ophys.ImageSegmentation()
    ophys_mod.add_data_interface(image_segmentation)
    trial_line_time_maps = None
    primary_qc = None
    secondary_qc = {}

    for plane in experiment_summary.keys():
        dmd_name, dmd_num = get_dmd_name(plane)
        if not dmd_name:
            continue
        slap2_meta_path = next(session_path.rglob(f"*DMD{dmd_num}.meta"))
        print('using slap2 meta file:', slap2_meta_path)
        dat_paths = list(session_path.rglob(f"*DMD{dmd_num}-TRIAL*.dat"))
        print(f'found {len(dat_paths)} .dat files for DMD{dmd_num}')
        sync_qc_folder = (qc_folder / 'syncing') if qc_folder is not None else None
        with h5py.File(slap2_meta_path) as slap2_meta_h5:
            fluorescence, timestamps, out_maps, plane_qc = sync_slap2_fluorescence(
                dmd_name, dmd_num, experiment_summary, slap2_meta_h5, harp_data,
                trial_line_time_maps=trial_line_time_maps,
                primary_qc=primary_qc,
                qc_folder=sync_qc_folder,
                dat_paths=dat_paths,
                plane_key=plane,
            )
        if int(dmd_num) == 1:
            trial_line_time_maps = out_maps
            primary_qc = plane_qc
        else:
            secondary_qc = {
                'secondary_frame_line_idxs': plane_qc['frame_line_idxs'],
                'secondary_trial_num_frames': plane_qc['trial_num_frames'],
                'secondary_timestamps': plane_qc['timestamps'],
            }

        imaging_plane = create_imaging_plane(nwbfile, dmd_name, device, acquisition_json, plane_key=plane)
        roi_table = add_image_segmentation(experiment_summary, imaging_plane, dmd_name, image_segmentation, plane_key=plane)
        add_mean_images(experiment_summary, dmd_name, ophys_mod, plane_key=plane)
        add_fluorescence(fluorescence, timestamps, dmd_name, roi_table, ophys_mod)


if __name__ == "__main__":
    main()