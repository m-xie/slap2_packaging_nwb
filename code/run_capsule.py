import h5py
import numpy as np
import pynwb
import hdmf_zarr
from datetime import datetime
from pathlib import Path
from openscope_upload import harp_utils
import json
import pandas as pd
import argparse
import shutil


data_folder = Path("../data")
results_folder = Path("../results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nwb_dir", type=str, default=f'nwb')
    args = parser.parse_args()
    input_nwb_dir = data_folder / Path(args.input_nwb_dir)

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


    asset_paths = [path for path in data_folder.iterdir() if path.name.lower().startswith('slap2')]
    if len(asset_paths) > 1:
        raise Exception(f"{len(asset_paths)}. asset paths found. There can be only one. ")
    elif len(asset_paths) == 0:
        raise Exception(f"No asset paths found.")
    else:
        asset_path = asset_paths[0]

    h5_path = next(asset_path.rglob("experiment_summary.h5"))
    print("Found h5 file:", h5_path)
    nwb_path = results_folder / f"{asset_path.name}.nwb"
    session_json_path = next(asset_path.glob("session.json"))
    rig_json_path = next(asset_path.glob("rig.json"))
    harp_path = next(asset_path.rglob('.harp'))
    orientations_csv = next((asset_path / 'behavior').rglob('orientations_orientations0.csv'))

    with open(session_json_path, "r") as f:
        session_json = json.load(f)
    with open(rig_json_path, "r") as f:
        rig_json = json.load(f)
    with io_class(str(result_nwb_path), "r+") as nwb_io:
        nwbfile = nwb_io.read()
        with h5py.File(h5_path, "r") as h5:
            harp_data = harp_utils.extract_harp(harp_path)
            add_ophys_to_nwb(nwbfile, h5, rig_json, harp_data)
            add_stim_table(nwbfile, orientations_csv, harp_data)
        nwb_io.write(nwbfile)
        nwb_io.close()
    print(f'Wrote output slap2 nwb to {result_nwb_path}')


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


def add_stim_table(nwbfile, orientations_table, harp_data):
    column_names = [
        'stim_id',
        'delay',
        'duration',
        'diameter',
        'x',
        'y',
        'contrast',
        'spatial_frequency',
        'temporal_frequency',
        'orientation'
    ]
    gratings_df = pd.read_csv(orientations_table, header=None, names=column_names)

    # csv_log = pd.read_csv(asset_path / 'behavior/orientations_logger.csv')
    # start_gratings_times_csv = np.array(csv_log.loc[csv_log['Value'] == 'StartGrating', 'Timestamp'].tolist())

    slap2_start_times = harp_data['normalized_slap2_start']
    slap2_end_times = harp_data['normalized_slap2_end']
    start_gratings_times = harp_data['normalized_start_gratings'][1:] # first time is erroneous (perhaps?)

    # We check there are as many gratings presentation as there are timing data in HARP
    if len(gratings_df) != len(start_gratings_times):
        raise ValueError(f"Mismatch between number of grating presentations {len(gratings_df)} and HARP timing data {len(start_gratings_times)}")

    start_time = []
    stop_time = []
    slap2_trial_idxs = []
    for i, row in gratings_df.iterrows():
        start_time.append(start_gratings_times[i])
        stop_time.append(start_gratings_times[i] + row['duration'])
        slap2_trial_idxs.append(find_slap2_trial_index(start_gratings_times[i], slap2_start_times, slap2_end_times))
    gratings_df['slap2_trial_idx'] = slap2_trial_idxs

    stim_table = pynwb.file.TimeIntervals(name='gratings', description='Gratings presentation intervals')

    # Add all dataframe columns to stim table
    for col in gratings_df.columns:
        stim_table.add_column(name=col, description=f'{col} from stimulus dataframe')

    # Add intervals (rows) to stim table
    for i, row in gratings_df.iterrows():
        stim_table.add_interval(
            start_time=start_time[i],
            stop_time=stop_time[i],
            **{
                col: int(row[col]) if col == 'id' else row[col]
                for col in gratings_df.columns
            }
        )

    # Add the TimeIntervals table to the NWB file
    nwbfile.add_time_intervals(stim_table)


def create_imaging_plane(nwbfile, dmd_name, optical_channel, device, rig_json):
    """
    Create an ImagingPlane object for a given DMD and add it to the NWBFile.
    nwbfile: NWBFile object to add the imaging plane to.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    optical_channel: OpticalChannel object for this plane.
    device: Device object for this plane.
    rig_json: Dictionary with rig metadata (expects 'light_sources' and 'detectors').
    Returns: ImagingPlane object.
    """
    excitation_lambda = rig_json["light_sources"][0]["wavelength"]
    imaging_rate = rig_json["detectors"][0]["frame_rate"]
    imaging_plane = nwbfile.create_imaging_plane(
        name=f"ImagingPlane_{dmd_name}",
        optical_channel=[optical_channel],
        description=f"Imaging plane for {dmd_name}",
        device=device,
        excitation_lambda=float(excitation_lambda) if excitation_lambda is not None else 920.0,
        imaging_rate=float(imaging_rate) if imaging_rate is not None else 30.0,
        indicator="unknown",
        location="unknown",
        manifold=None
    )
    return imaging_plane


def get_pixel_mask(mask):
    """
    Given a 2D mask (full FOV), return a list of (row, col, weight) triplets for all nonzero and non-nan pixels.
    This is the NWB-compliant, space-efficient way to store ROIs for large FOVs.
    Returns: pixel_mask (list of (row, col, weight))
    """
    valid = np.logical_and(mask != 0, ~np.isnan(mask))
    rows, cols = np.where(valid)
    weights = mask[rows, cols]
    pixel_mask = [(int(r), int(c), float(w)) for r, c, w in zip(rows, cols, weights)]
    return pixel_mask


def add_image_segmentation(h5, imaging_plane, dmd_name, image_segmentation):
    """
    Add a PlaneSegmentation for a DMD to the shared ImageSegmentation object.
    For each ROI, store a pixel_mask (list of (row, col, weight)) for space-efficient NWB compliance.
    h5: Open HDF5 file handle.
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
    fp_masks = h5[dmd_name]['sources']['spatial']['fp_masks'][()]
    n_rois = fp_masks.shape[0]
    for mask in fp_masks:
        pixel_mask = get_pixel_mask(mask)
        ps.add_roi(pixel_mask=pixel_mask)
    roi_table_region = ps.create_roi_table_region(
        region=list(range(n_rois)),
        description=f"All ROIs for {dmd_name}"
    )
    return roi_table_region


def add_fluorescence(h5, dmd_name, roi_table, ophys_mod, harp_data):
    """
    Add Fluorescence and dFF traces for a DMD to the NWBFile.
    h5: Open HDF5 file handle.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    roi_table: DynamicTableRegion referencing ROIs for this plane.
    ophys_mod: Ophys ProcessingModule to add the Fluorescence interface to.
    Expects F0 and dFF data as (num_timepoints, num_rois).
    """
    dmd_group = h5[dmd_name]
    temporal_sources = dmd_group['sources']['temporal']
    print("temporal sources:", temporal_sources, temporal_sources.keys())
    f0_data = temporal_sources['F0'][()]
    dff_data = temporal_sources['dFF'][()]
    print(dmd_group, dmd_group.keys())
    trial_start_idxs = dmd_group['frame_info']['trial_start_idxs'][()]
    timestamps = harp_utils.get_concatenated_timestamps(f0_data, trial_start_idxs, harp_data)
    assert len(timestamps) == f0_data.shape[0], "Timestamps and fluorescence trace must have same length"
    fluorescence = pynwb.ophys.Fluorescence(name=f"Fluorescence_{dmd_name}")
    ophys_mod.add_data_interface(fluorescence)
    rrs_f0 = pynwb.ophys.RoiResponseSeries(
        name=f"{dmd_name}_F0",
        data=f0_data,
        rois=roi_table,
        unit='a.u.',
        timestamps=timestamps,
        description=f"F0 (raw fluorescence) traces from {dmd_name}"
    )
    fluorescence.add_roi_response_series(rrs_f0)
    rrs_dff = pynwb.ophys.RoiResponseSeries(
        name=f"{dmd_name}_dFF",
        data=dff_data,
        rois=roi_table,
        unit='a.u.',
        timestamps=timestamps,
        description=f"dFF traces from {dmd_name}"
    )
    fluorescence.add_roi_response_series(rrs_dff)


def add_mean_images(h5, dmd_name, ophys_mod):
    """
    Add registered and motion-corrected mean and activity images for a DMD to the ophys ProcessingModule as ImageSeries objects.
    mean_im: shape (channels, height, width) from HDF5.
    act_im: shape (height, width) from HDF5.
    Each channel of the mean image is stored as a single-frame ImageSeries.
    The activity image is stored as a single-frame ImageSeries.
    """
    mean_im = h5[dmd_name]['visualizations']['mean_im'][()]
    act_im = h5[dmd_name]['visualizations']['act_im'][()]
    if mean_im.ndim == 2: # should be (channels, height, width), add dim if only one channel
        mean_im = mean_im[None, ...]
    for ch in range(mean_im.shape[0]):
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


def create_device(nwbfile, rig_json):
    """
    Create and return a Device object using all relevant fields from rig.json.
    nwbfile: NWBFile object to add the device to.
    rig_json: Dictionary with instrument metadata.
    Returns: Device object.
    """
    name = rig_json["instrument_id"]
    manufacturer = rig_json["manufacturer"]["name"] if isinstance(rig_json["manufacturer"], dict) else rig_json["manufacturer"]
    description = f"{rig_json['instrument_type']} microscope. Temperature control: {rig_json['temperature_control']}, Humidity control: {rig_json['humidity_control']}. Calibration date: {rig_json['calibration_date']}. Notes: {rig_json['notes']}"
    device = nwbfile.create_device(
        name=name,
        description=description,
        manufacturer=manufacturer
    )
    return device


def create_optical_channel(rig_json, dmd_name):
    """
    Create an OpticalChannel object for a DMD using rig.json fields.
    nwbfile: NWBFile object (not used, for interface consistency).
    rig_json: Dictionary with instrument metadata.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    Returns: OpticalChannel object.
    """
    emission_lambda = 510.0  # GCaMP emission peak
    det = rig_json["detectors"][0]
    obj = rig_json["objectives"][0]
    description = f"{det['manufacturer']} {det['model']} detector, {obj['magnification']}x {obj['manufacturer']} objective, GCaMP emission"
    return pynwb.ophys.OpticalChannel(
        name=f"OpticalChannel_{dmd_name}",
        description=description,
        emission_lambda=emission_lambda
    )


def add_ophys_to_nwb(nwbfile, h5, rig_json, harp_data):
    """
    Build the full ophys structure in the NWB file, iterating over DMDs (planes).
    For each DMD, create imaging plane, ROI table, add fluorescence, and add mean images.
    nwbfile: NWBFile object to populate.
    h5: Open HDF5 file handle.
    rig_json: Dictionary with instrument metadata.
    """
    device = create_device(nwbfile, rig_json)
    ophys_mod = pynwb.ProcessingModule('ophys', 'Ophys processing module')
    nwbfile.add_processing_module(ophys_mod)
    image_segmentation = pynwb.ophys.ImageSegmentation()
    ophys_mod.add_data_interface(image_segmentation)
    for dmd_name in sorted([k for k in h5.keys() if k.startswith("DMD")]):
        optical_channel = create_optical_channel(rig_json, dmd_name)
        imaging_plane = create_imaging_plane(nwbfile, dmd_name, optical_channel, device, rig_json)
        roi_table = add_image_segmentation(h5, imaging_plane, dmd_name, image_segmentation)
        add_fluorescence(h5, dmd_name, roi_table, ophys_mod, harp_data)
        add_mean_images(h5, dmd_name, ophys_mod)


if __name__ == "__main__":
    main()