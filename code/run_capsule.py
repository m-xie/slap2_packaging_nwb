import h5py
import numpy as np
import pynwb
import hdmf_zarr
from aind_nwb_utils import utils as nwb_utils
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from datetime import datetime
from pathlib import Path
import harp_utils
import stimulus_sync
import random_natural_movies
from slap2_dat_utils import parse_dat_file, validate_dat_files
import slap2_synching as slap2_sync
import slap2_running_packaging as running_packaging
import slap2_eye_tracking_packaging as eye_tracking_packaging
from qc import slap2_dff_qc
from qc import slap2_receptive_fields_qc as slap2_rf_qc
from qc import stim_tuning_qc
from qc import zebra_movie_qc
import json
import pandas as pd
import argparse
import shutil
import re
import csv
import warnings
import matplotlib.pyplot as plt
import os
from importlib.metadata import version as package_version


data_folder = Path("../data")
results_folder = Path("../results")


def ensure_was_generated_by(nwbfile):
    """Populate empty NWB software provenance with the installed package version."""
    was_generated_by = nwbfile.was_generated_by
    if was_generated_by is not None and len(was_generated_by) > 0:
        return

    nwbfile.fields.pop("was_generated_by", None)
    nwbfile.was_generated_by = [
        ["aind-nwb-utils", package_version("aind-nwb-utils")]
    ]


def package_running_or_skip(
    nwbfile, harp_data, qc_output_path, allow_skip_running=False
):
    """Package running data, optionally skipping when required HARP inputs are absent."""
    missing_inputs = [
        name for name in ("wheel", "analog_times") if name not in harp_data
    ]
    if missing_inputs:
        message = f"Missing required running inputs: {', '.join(missing_inputs)}"
        if allow_skip_running:
            print(f"{message}; skipping running packaging.")
            return None
        raise KeyError(message)

    return running_packaging.package_harp_running_data(
        nwbfile,
        harp_data,
        qc_output_path=qc_output_path,
    )


def package_eye_or_skip(
    nwbfile,
    eye_tracking_paths,
    eye_camera_metadata_path,
    harp_data,
    qc_output_path,
    allow_skip_eye=False,
):
    """Package eye tracking, optionally skipping when required inputs are absent."""
    problems = []
    if not eye_tracking_paths:
        problems.append("no ellipses_processed*.h5 file was found")
    elif len(eye_tracking_paths) > 1:
        problems.append(
            f"expected one ellipses_processed*.h5 file, found {len(eye_tracking_paths)}"
        )
    if not eye_camera_metadata_path.is_file():
        problems.append(f"EyeCamera metadata was not found at {eye_camera_metadata_path}")
    if "time_reference" not in harp_data:
        problems.append("HARP time_reference is missing")

    if problems:
        message = "Missing or ambiguous eye-tracking inputs: " + "; ".join(problems)
        if allow_skip_eye:
            print(f"{message}; skipping eye tracking packaging.")
            return None
        raise FileNotFoundError(message)

    return eye_tracking_packaging.package_eye_tracking(
        nwbfile,
        eye_tracking_paths[0],
        eye_camera_metadata_path,
        harp_data["time_reference"],
        qc_output_path=qc_output_path,
    )


def find_eye_tracking_paths(eye_tracking_path, processed_path):
    """Find ellipse fits in the dedicated eye asset, then the processed asset."""
    if eye_tracking_path.exists():
        paths = list(eye_tracking_path.rglob("ellipses_processed*.h5"))
        if paths:
            return paths
    return list(processed_path.rglob("ellipses_processed*.h5"))


def write_data_process(
    session_path,
    processed_path,
    nwb_path,
    output_dir,
    stimulus_start_time,
    stimulus_end_time,
    ophys_start_time,
    ophys_end_time,
    packaging_end_time,
    parameters,
):
    """Write provenance for SLAP2 synchronization and NWB packaging."""
    code_version = os.getenv("VERSION", "")
    code_url = "https://github.com/AllenNeuralDynamics/aind-slap2-nwb-packaging"
    experimenters = ["AIND Scientific Computing"]

    def capsule_code(process_parameters):
        return Code(
            url=code_url,
            name="SLAP2 NWB packaging",
            version=code_version,
            run_script=Path("code/run_capsule.py"),
            language="Python",
            parameters=process_parameters,
        )

    synchronization = DataProcess(
        process_type=ProcessName.OTHER,
        name="SLAP2-HARP synchronization",
        stage=ProcessStage.PROCESSING,
        code=capsule_code({
            "session_path": str(session_path),
            "harp_clock_source": "HARP digital and cycle-clock signals",
        }),
        experimenters=experimenters,
        start_date_time=ophys_start_time,
        end_date_time=ophys_end_time,
        notes=(
            "Synchronized SLAP2 fluorescence frames and trials to HARP time, "
            "including primary and secondary DMD timestamp alignment."
        ),
    )
    stimulus_packaging = DataProcess(
        process_type=ProcessName.FILE_FORMAT_CONVERSION,
        name="Stimulus interval NWB packaging",
        stage=ProcessStage.PROCESSING,
        code=capsule_code({
            "session_path": str(session_path),
            "processed_path": str(processed_path),
        }),
        experimenters=experimenters,
        start_date_time=stimulus_start_time,
        end_date_time=stimulus_end_time,
        notes="Converted visual stimulus tables and HARP timestamps to NWB TimeIntervals.",
    )
    ophys_packaging = DataProcess(
        process_type=ProcessName.FILE_FORMAT_CONVERSION,
        name="SLAP2 ophys NWB packaging",
        stage=ProcessStage.PROCESSING,
        code=capsule_code({
            "processed_path": str(processed_path),
            "nwb_path": str(nwb_path),
            **parameters,
        }),
        experimenters=experimenters,
        start_date_time=ophys_start_time,
        end_date_time=packaging_end_time,
        output_path=Path(nwb_path).name,
        notes=(
            "Packaged SLAP2 imaging planes, ROI segmentation, fluorescence, "
            "mean images, and synchronized timestamps into NWB."
        ),
    )
    processing = Processing(
        data_processes=[stimulus_packaging, synchronization, ophys_packaging],
        dependency_graph={
            stimulus_packaging.name: [],
            synchronization.name: [],
            ophys_packaging.name: [
                stimulus_packaging.name,
                synchronization.name,
            ],
        },
    )
    output_path = Path(output_dir) / "slap2-nwb-packaging_data_process.json"
    with open(output_path, "w") as f:
        json.dump(json.loads(processing.model_dump_json()), f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logger_format",
        choices=["OpenScope P3", "Random Natural Movies", "Legacy Drifting Gratings"],
        default="Legacy Drifting Gratings",
        help="Visual stimulus format; selects playback parsing for Random Natural Movies and is recorded in provenance.",
    )
    parser.add_argument("--input_session_dir", type=str, default="slap2_session")
    parser.add_argument("--input_processed_dir", type=str, default="slap2_processed")
    parser.add_argument("--stim_table_pattern", type=str, default="orientations_orientations0")
    parser.add_argument("--input_eye_tracking_dir", type=str, default="eye_tracking")
    parser.add_argument("--input_nwb_dir", type=str, default=f'nwb')
    parser.add_argument("--use_input_nwb", type=str, default="false")
    parser.add_argument("--allow_skip_running", type=str, default="false")
    parser.add_argument("--allow_skip_eye", type=str, default="false")
    parser.add_argument("--expected_n_trials", type=int, default=None)
    parser.add_argument("--rf_onset_delay", type=float, default=0.2,
                        help="Onset delay in seconds for RF response windows (default: 0.2)")
    parser.add_argument("--qc_folder_name", type=str, default="qc")
    args = parser.parse_args()
    logger_format = args.logger_format
    use_input_nwb = args.use_input_nwb.lower() in ('t', 'true')
    allow_skip_running = args.allow_skip_running.lower() in ('t', 'true')
    allow_skip_eye = args.allow_skip_eye.lower() in ('t', 'true')
    expected_n_trials = args.expected_n_trials
    rf_onset_delay = args.rf_onset_delay
    qc_folder_name = args.qc_folder_name
    input_nwb_dir = data_folder / Path(args.input_nwb_dir)
    session_path = Path(args.input_session_dir)
    stim_table_pattern = args.stim_table_pattern
    print("stim table name substring to look for:",stim_table_pattern)
    if not session_path.is_absolute():
        session_path = data_folder / session_path
    processed_path = Path(args.input_processed_dir)
    if not processed_path.is_absolute():
        processed_path = data_folder / processed_path
    eye_tracking_path = Path(args.input_eye_tracking_dir)
    if not eye_tracking_path.is_absolute():
        eye_tracking_path = data_folder / eye_tracking_path
    with open(session_path / "data_description.json", "r") as f:
        session_name = json.load(f)["name"]
    with open(processed_path / "data_description.json", "r") as f:
        processed_session_name = json.load(f)["name"]
    print("SESSION PATH", session_path, "SESSION NAME", session_name)
    print("PROCESSED PATH", processed_path, "SESSION NAME", processed_session_name)

    for entry in results_folder.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f'cleared results folder: {list(results_folder.iterdir())}')

    if use_input_nwb:
        print('INPUT NWB DIR', input_nwb_dir)
        assert input_nwb_dir.exists(), "Input NWB dir does not exist"
        nwb_files = [p for p in input_nwb_dir.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
        assert len(nwb_files) == 1, f"Attach one base NWB file data at a time. {len(nwb_files)} found"
        input_nwb_path = nwb_files[0]
        print('INPUT NWB', input_nwb_path)

        # determine if file is zarr or hdf5, and copy it to results
        result_nwb_path = results_folder / input_nwb_path.name
        if input_nwb_path.is_dir():
            assert (input_nwb_path / ".zattrs").is_file(), f"{input_nwb_path.name} is not a valid Zarr folder"
            io_class = hdmf_zarr.NWBZarrIO
            shutil.copytree(input_nwb_path, result_nwb_path, dirs_exist_ok=True)
        else:
            io_class = pynwb.NWBHDF5IO
            shutil.copyfile(input_nwb_path, result_nwb_path)
    else:
        io_class = hdmf_zarr.NWBZarrIO
        nwb_file_obj = nwb_utils.create_base_nwb_file(session_path)
        ensure_was_generated_by(nwb_file_obj)
        result_nwb_path = results_folder / f"{nwb_file_obj.session_id}.nwb"
        with io_class(str(result_nwb_path), "w") as nwb_io:
            nwb_io.write(nwb_file_obj)

    print("Using NWB:", result_nwb_path)
    instrument_json_path = next(session_path.glob("instrument.json"))
    acquisition_json_path = next(session_path.glob("acquisition.json"))
    harp_path = next(session_path.rglob('*.harp'))
    experiment_summary_path = next(processed_path.rglob('*experiment_summary.h5'))
    stim_table_csv = find_stimulus_table(session_path, stim_table_pattern, logger_format)
    try:
        log_csv = stimulus_sync.select_stimulus_logger(
            (session_path / 'behavior').rglob('*logger*.csv')
        )
    except FileNotFoundError:
        log_csv = None
    print('using stimulus logger:', log_csv)
    eye_tracking_paths = find_eye_tracking_paths(eye_tracking_path, processed_path)
    eye_camera_metadata_path = session_path / 'behavior-videos' / 'EyeCamera' / 'metadata.csv'

    if eye_tracking_paths:
        print('using eye tracking data:', eye_tracking_paths)

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
            harp_data = harp_utils.trim_leading_trial_pulse_artifact(harp_data)
            harp_data, exclude_final_trial = trim_unterminated_harp_trial(harp_data)
            qc_folder = results_folder / qc_folder_name
            qc_folder.mkdir(exist_ok=True)
            (qc_folder / 'syncing').mkdir(exist_ok=True)
            stimulus_start_time = datetime.now().astimezone()
            stimulus_timing_metadata = add_stim_table(
                nwbfile, stim_table_csv, log_csv, harp_data, logger_format,
                acquisition_json=acquisition_json,
            )
            stimulus_end_time = datetime.now().astimezone()
            ophys_start_time = datetime.now().astimezone()
            add_ophys_to_nwb(
                experiment_summary,
                nwbfile,
                instrument_json,
                acquisition_json,
                harp_data,
                session_path,
                qc_folder,
                exclude_final_trial=exclude_final_trial,
            )
            ophys_end_time = datetime.now().astimezone()
            # Wheel counts already share the normalized HARP clock with SLAP2.
            # Package them while this NWBFile is still open and writable.
            package_running_or_skip(
                nwbfile,
                harp_data,
                qc_output_path=qc_folder / "running_speed.png",
                allow_skip_running=allow_skip_running,
            )
            package_eye_or_skip(
                nwbfile,
                eye_tracking_paths,
                eye_camera_metadata_path,
                harp_data,
                qc_output_path=qc_folder / "eye_tracking.png",
                allow_skip_eye=allow_skip_eye,
            )
        ensure_was_generated_by(nwbfile)
        nwb_io.write(nwbfile)
    packaging_end_time = datetime.now().astimezone()
    slap2_dff_qc.compute_dff_qc(qc_folder, result_nwb_path)
    slap2_dff_qc.compute_raw_fluorescence_qc(qc_folder, result_nwb_path)
    run_stimulus_qc(result_nwb_path, qc_folder, logger_format, rf_onset_delay)
    write_data_process(
        session_path=session_path,
        processed_path=processed_path,
        nwb_path=result_nwb_path,
        output_dir=results_folder,
        stimulus_start_time=stimulus_start_time,
        stimulus_end_time=stimulus_end_time,
        ophys_start_time=ophys_start_time,
        ophys_end_time=ophys_end_time,
        packaging_end_time=packaging_end_time,
        parameters={
            "visual_stimulus_logger_format": logger_format,
            "use_input_nwb": use_input_nwb,
            "input_eye_tracking_dir": str(eye_tracking_path),
            "allow_skip_running": allow_skip_running,
            "allow_skip_eye": allow_skip_eye,
            "expected_n_trials": expected_n_trials,
            "rf_onset_delay": rf_onset_delay,
            "stimulus_timing": stimulus_timing_metadata,
        },
    )
    print(f'Wrote output slap2 nwb to {result_nwb_path}')


def run_stimulus_qc(nwb_path, qc_folder, logger_format, rf_onset_delay):
    """Retain existing stimulus QC only for the formats it supports."""
    if logger_format == random_natural_movies.LOGGER_FORMAT:
        print("Random Natural Movies: skipping stimulus-specific QC; general fluorescence QC is unchanged.")
        return
    zebra_movie_qc.plot_zebra_repeats(nwb_path, qc_folder / 'zebra_movie')
    slap2_rf_qc.compute_receptive_field_qc(qc_folder, nwb_path, onset_delay=rf_onset_delay)
    stim_tuning_qc.compute_stim_tuning_qc(qc_folder, nwb_path)
    stim_tuning_qc.compute_orientation_tuning_qc(qc_folder, nwb_path)


def find_stimulus_table(session_path, pattern, logger_format):
    """Keep legacy selection; require an unambiguous table for the new format."""
    behavior_path = Path(session_path) / "behavior"
    candidates = behavior_path.rglob(f"*{pattern}*.csv")
    if logger_format != random_natural_movies.LOGGER_FORMAT:
        return next(candidates)
    paths = sorted(candidates)
    # The existing CLI/app-panel default names a legacy orientations table.
    # Only the new format may fall back to the RandomNaturalMovies filename.
    if not paths and pattern == "orientations_orientations0":
        paths = sorted(behavior_path.rglob("*stim_table*.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"Random Natural Movies requires exactly one stimulus table, found {len(paths)}. "
            "Set --stim_table_pattern to identify the intended table."
        )
    return paths[0]


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


def infer_continuous_slap2_mode(experiment_summary, plane_inputs, harp_data):
    """Identify one raw SLAP2 trial split into source-extraction chunks.

    DO1 is optional here because continuous fluorescence timing comes from DI3
    cycle starts and SLAP2 line indices. The remaining structural checks are
    intentionally strict so a missing end pulse cannot turn trial-based data
    into a continuous acquisition by itself.
    """
    if (
        len(harp_data['normalized_slap2_start']) != 1
        or len(harp_data['normalized_slap2_end']) not in (0, 1)
    ):
        return False

    parsed_dat_files = [
        parse_dat_file(dat_path)
        for _, _, _, _, dat_paths, _ in plane_inputs
        for dat_path in dat_paths
    ]
    if not parsed_dat_files:
        return False
    if len({dat_file.acquisition_prefix.lower() for dat_file in parsed_dat_files}) != 1:
        return False
    if any(dat_file.cycle_offset is None for dat_file in parsed_dat_files):
        return False

    dmd_trial_numbers = {}
    for dat_file in parsed_dat_files:
        dmd_trial_numbers.setdefault(dat_file.dmd_number, set()).add(
            dat_file.trial_number
        )
    if not dmd_trial_numbers or any(
        trial_numbers != {1} for trial_numbers in dmd_trial_numbers.values()
    ):
        return False

    for plane, _, _, _, _, n_summary_trials in plane_inputs:
        if n_summary_trials <= 1:
            return False
        frame_info = experiment_summary[plane]['frame_info']
        trial_num_frames = np.asarray(
            frame_info['trial_num_frames'][()]
        ).reshape(-1)
        frame_line_idxs = np.asarray(
            frame_info['frame_line_idxs'][()]
        ).reshape(-1)
        if int(np.sum(trial_num_frames)) != len(frame_line_idxs):
            return False

        boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])
        nonempty_chunks = []
        positive_steps = []
        for chunk_idx, n_frames in enumerate(trial_num_frames):
            if n_frames == 0:
                continue
            chunk = frame_line_idxs[
                boundaries[chunk_idx]:boundaries[chunk_idx + 1]
            ].astype(np.int64, copy=False)
            nonempty_chunks.append((chunk_idx, chunk))
            chunk_steps = np.diff(chunk)
            positive_steps.extend(chunk_steps[chunk_steps > 0])

        if len(nonempty_chunks) <= 1 or not positive_steps:
            return False
        max_continuation_gap = 1.5 * float(np.percentile(positive_steps, 99))
        if int(nonempty_chunks[0][1][0]) < 1:
            return False
        for (previous_idx, previous), (current_idx, current) in zip(
            nonempty_chunks, nonempty_chunks[1:]
        ):
            boundary_gap = int(current[0]) - int(previous[-1])
            if boundary_gap <= 0:
                return False
            if current_idx == previous_idx + 1 and boundary_gap > max_continuation_gap:
                return False

    return True


def trim_unterminated_harp_trial(harp_data):
    """Exclude an unterminated trailing trial unless it may be continuous.

    A sole DO0 without DO1 is preserved provisionally. Continuous-mode
    inference later validates it using chunked .dat files and continued line
    indices. An unmatched final DO0 in a multi-trial session is still removed.
    """
    starts = harp_data['normalized_slap2_start']
    ends = harp_data['normalized_slap2_end']
    excluded_trailing_trials = 0

    # Do not erase the only acquisition before continuous-mode inference has
    # inspected its .dat chunks and source-extraction line indices.
    if len(starts) == 1 and len(ends) == 0:
        return dict(harp_data), False
    if len(starts) == len(ends) + 1 and starts[-1] > ends[-1]:
        excluded_trailing_trials = 1
    elif len(starts) != len(ends):
        raise ValueError(
            f"Unsupported SLAP2 trial pulse mismatch: {len(starts)} starts and "
            f"{len(ends)} ends."
        )

    trimmed = dict(harp_data)
    if not excluded_trailing_trials:
        return trimmed, False

    for key in ('slap2_start_signal', 'slap2_start_times', 'normalized_slap2_start'):
        trimmed[key] = harp_data[key][:-1]
    last_complete_end = ends[-1]
    keep_clock = harp_data['slap2_cycle_clock_times'] <= last_complete_end
    trimmed['slap2_cycle_clock_signal'] = harp_data['slap2_cycle_clock_signal'][keep_clock]
    trimmed['slap2_cycle_clock_times'] = harp_data['slap2_cycle_clock_times'][keep_clock]
    trimmed['normalized_slap2_cycle_clock_times'] = trimmed['slap2_cycle_clock_times']
    warnings.warn(
        "The final SLAP2 trial has a start pulse but no end pulse; excluding the "
        "final trial from fluorescence packaging.",
        RuntimeWarning,
        stacklevel=2,
    )
    return trimmed, True


def find_slap2_trial_index(time, start_trials, end_trials):
    # Find index where `time` would be inserted to keep start_trials sorted
    idx = np.searchsorted(start_trials, time, side='right') - 1
    
    # Check bounds and if time fits in the trial interval
    if idx < 0 or idx >= len(start_trials):
        raise Exception(f"Time {time} is out of trial bounds")
    # A validated continuous acquisition without DO1 is an open interval. Its
    # fluorescence end remains cycle-derived; this only labels stimulus rows.
    if len(start_trials) == 1 and len(end_trials) == 0:
        return idx
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
        'Block_Number':       'BlockNumber',
        'Block_Label':        'BlockLabel',
        'Block_Duration_Minutes': 'BlockDurationMinutes',
        'Trial_Number':       'TrialNumber',
        'Sequence_Number':    'SequenceNumber',
        'Trial_In_Sequence':  'TrialInSequence',
        'Spatial_Frequency':  'SpatialFrequency',
        'Temporal_Frequency': 'TemporalFrequency',
        'Trial_Type':         'TrialType',
        'Block_Type':         'BlockType',
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

    # Normalize the newer underscore-delimited table schema once rather than
    # branching throughout NWB interval construction and downstream QC.
    conflicting_columns = [
        old_name
        for old_name, new_name in _OLD_TO_NEW_RENAMES.items()
        if old_name in df.columns and new_name in df.columns
    ]
    if conflicting_columns:
        raise ValueError(
            "Stimulus table contains both legacy and canonical columns for: "
            f"{conflicting_columns}"
        )
    df = df.rename(columns=_OLD_TO_NEW_RENAMES)

    return df


def add_stim_table(nwbfile, orientations_table, log_csv, harp_data, logger_format=None, acquisition_json=None):
    if logger_format == random_natural_movies.LOGGER_FORMAT:
        return add_stim_table_movies(
            nwbfile, orientations_table, log_csv, harp_data, acquisition_json,
        )
    stimulus_df = read_stim_csv(orientations_table)

    slap2_start_times = harp_data['normalized_slap2_start']
    slap2_end_times = harp_data['normalized_slap2_end']
    stimulus_start_times, timing_metadata = (
        stimulus_sync.resolve_stimulus_start_times(
            len(stimulus_df), harp_data, log_csv
        )
    )
    print('stimulus timing:', timing_metadata)
    
    start_time = []
    stop_time = []
    slap2_trial_idxs = []
    for i, row in stimulus_df.iterrows():
        stimulus_start = stimulus_start_times[i]
        start_time.append(stimulus_start)
        stop_time.append(stimulus_start + row['Duration'])
        slap2_trial_idxs.append(
            find_slap2_trial_index(
                stimulus_start, slap2_start_times, slap2_end_times
            )
        )
    stimulus_df['slap2_trial_idx'] = slap2_trial_idxs

    # Attach timing columns so they travel with the df during splitting
    stimulus_df['start_time'] = start_time
    stimulus_df['stop_time'] = stop_time

    # Split by BlockType and add each block as its own TimeIntervals table
    if 'BlockType' in stimulus_df.columns:
        block_groups = stimulus_df.groupby('BlockType', sort=False)
    else:
        # No BlockType column — fall back to a single 'gratings' table
        block_groups = [('gratings', stimulus_df)]

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

    return timing_metadata

def add_stim_table_movies(nwbfile, stim_table_csv, log_csv, harp_data, acquisition_json=None):
    """Preserve observed source table rows and their individual grating trials."""
    blocks, gratings, timing_metadata = random_natural_movies.synchronize_presentations(
        pd.read_csv(stim_table_csv), log_csv, harp_data,
    )
    gratings, grating_descriptions = random_natural_movies.add_grating_parameters(
        gratings, acquisition_json,
    )
    for name, frame_table, description in (
        (
            "stimulus_blocks", blocks,
            "Random Natural Movies: one interval per observed stimulus-table row, in order. "
            "Movie offsets follow playback; grating blocks span first onset to last offset "
            "and exclude unlogged outer blanks. TrialDuration is nominal metadata only. "
            "is_partial marks censored observations; their stop_time is a lower bound, not a measured offset.",
        ),
        (
            "gratings", gratings,
            "Individual gratings, including blanks, presented with Random Natural Movies workflow. "
            "stimulus_table_row refers to the zero-based stimulus_blocks id. "
            "logger_orientation=359 denotes a blank with Orientation=NaN and is_blank=True. "
            "is_partial marks a missing end event or photodiode-censored offset.",
        ),
    ):
        if frame_table.empty:
            continue
        frame_table = frame_table.copy()
        frame_table["slap2_trial_idx"] = [
            find_slap2_trial_index(
                time, harp_data["normalized_slap2_start"], harp_data["normalized_slap2_end"],
            )
            for time in frame_table["start_time"]
        ]
        table = pynwb.file.TimeIntervals(name=name, description=description)
        table.id.data.extend(
            frame_table["stimulus_table_row"].tolist()
            if name == "stimulus_blocks" else range(len(frame_table))
        )
        table["start_time"].data.extend(frame_table.pop("start_time").tolist())
        table["stop_time"].data.extend(frame_table.pop("stop_time").tolist())
        for column in frame_table:
            column_description = f"{column}: Random Natural Movies source metadata or playback-derived value"
            if name == "stimulus_blocks" and column == "movie_url":
                column_description = (
                    "Commit-pinned GitHub movie URL matched by TextureName; "
                    "empty for non-movie rows or unrecognized textures."
                )
            if name == "gratings":
                column_description = grating_descriptions.get(column, column_description)
            table.add_column(
                name=column,
                description=column_description,
                data=frame_table[column].tolist(),
            )
        nwbfile.add_time_intervals(table)
        print(f"Added intervals table '{name}' with {len(frame_table)} rows")
    print("stimulus timing:", timing_metadata)
    return timing_metadata


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


def resolve_slap2_acquisition(
    dat_paths, n_summary_trials, excluded_trailing_trials=0
):
    """Resolve which SLAP2 acquisition and processed trial range to package.

    Some summaries contain an aborted acquisition followed by the intended one,
    with trial numbering restarted in each acquisition. This function selects the
    later acquisition and excludes summary trials belonging to the earlier run.

    Interrupted sessions can also leave HARP with an unmatched final trial-start
    pulse and SLAP2 with an incomplete or malformed final trial. This function
    applies the resulting trailing-trial exclusion, keeping the processed trial
    count and .dat evidence consistent for downstream synchronization across all
    DMDs.
    """
    acquisition_groups = {}
    for dat_path in dat_paths:
        dat_file = parse_dat_file(dat_path)
        acquisition_groups.setdefault(
            dat_file.acquisition_prefix, []
        ).append(dat_file)

    if not acquisition_groups:
        raise ValueError("No SLAP2 .dat files were found.")
    if len(acquisition_groups) > 2:
        raise ValueError(
            f"SLAP2 data contains {len(acquisition_groups)} acquisitions "
            f"({sorted(acquisition_groups)}). Automatic filtering only supports "
            f"one earlier acquisition followed by one retained acquisition."
        )

    ordered_groups = sorted(
        acquisition_groups.items(),
        key=lambda item: datetime.strptime(
            item[1][0].acquisition_timestamp, '%Y%m%d_%H%M%S'
        ),
    )
    selected_prefix, selected_dat_files = ordered_groups[-1]

    if len(ordered_groups) == 1:
        excluded_trial_count = 0
    else:
        excluded_prefix, excluded_dat_files = ordered_groups[0]
        excluded_trial_count = max(
            dat_file.trial_number for dat_file in excluded_dat_files
        )
        retained_trial_count = (
            n_summary_trials - excluded_trial_count - excluded_trailing_trials
        )
        if retained_trial_count <= 0:
            raise ValueError(
                f"Earlier acquisition {excluded_prefix} reaches trial "
                f"{excluded_trial_count}, leaving no retained trials in the "
                f"{n_summary_trials}-trial experiment summary."
            )
        if excluded_trial_count >= retained_trial_count:
            raise ValueError(
                f"Earlier acquisition {excluded_prefix} is not shorter than retained "
                f"acquisition {selected_prefix}; automatic excision is unsafe."
            )
        warnings.warn(
            f"Multiple SLAP2 acquisitions detected: earlier {excluded_prefix} "
            f"(observed through trial {excluded_trial_count}) and later "
            f"{selected_prefix}. Excluding the first {excluded_trial_count} summary "
            f"trials and their samples, and using only {selected_prefix} .dat/.meta files.",
            RuntimeWarning,
            stacklevel=2,
        )

    retained_trial_count = (
        n_summary_trials - excluded_trial_count - excluded_trailing_trials
    )
    # Aborted acquisitions and excluded trailing trials are often incomplete;
    # only retained files can influence downstream synchronization.
    validate_dat_files([
        dat_file.path
        for dat_file in selected_dat_files
        if dat_file.trial_number <= retained_trial_count
    ], slap2_sync.read_dat_num_cycles)
    selected_trial_numbers = [
        dat_file.trial_number for dat_file in selected_dat_files
    ]
    invalid_trial_numbers = [
        trial_num for trial_num in selected_trial_numbers
        if trial_num < 1
        or trial_num > retained_trial_count + excluded_trailing_trials
    ]
    if invalid_trial_numbers:
        raise ValueError(
            f"Retained acquisition {selected_prefix} has trial numbers outside "
            f"the retained summary's 1..{retained_trial_count} range: "
            f"{sorted(set(invalid_trial_numbers))}."
        )

    retained_trial_numbers = [
        trial_num
        for trial_num in selected_trial_numbers
        if trial_num <= retained_trial_count
    ]
    if not retained_trial_numbers:
        raise ValueError(f"No retained .dat trials remain in {selected_prefix}.")
    observed_retained_span = max(retained_trial_numbers)
    if observed_retained_span < retained_trial_count:
        warnings.warn(
            f"Retained acquisition {selected_prefix} has {retained_trial_count} "
            f"processed trial slots, but .dat files across all DMDs only reach trial "
            f"{observed_retained_span}. Assuming the remaining trailing .dat files "
            f"are missing and retaining all processed trials.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        'acquisition_prefix': selected_prefix,
        'excluded_trial_count': excluded_trial_count,
        'excluded_trailing_trials': excluded_trailing_trials,
        'retained_trial_count': retained_trial_count,
        'highest_dat_trial': observed_retained_span,
    }


def filter_slap2_acquisition(
    dmd_num,
    dat_paths,
    meta_paths,
    trial_num_frames,
    frame_line_idxs,
    f0_data,
    df_denoised_data,
    events_data,
    acquisition_resolution,
):
    """Apply shared leading and trailing trial exclusions to one DMD."""
    selected_prefix = acquisition_resolution['acquisition_prefix']
    n_excluded_trials = acquisition_resolution['excluded_trial_count']
    n_excluded_trailing_trials = acquisition_resolution['excluded_trailing_trials']
    retained_trial_count = acquisition_resolution['retained_trial_count']
    selected_dat_files = []
    for dat_path in dat_paths:
        dat_file = parse_dat_file(dat_path)
        if dat_file.dmd_number != int(dmd_num):
            raise ValueError(
                f"Expected a DMD{dmd_num} .dat file, found DMD"
                f"{dat_file.dmd_number} in {dat_path.name}."
            )
        if dat_file.acquisition_prefix.lower() == selected_prefix.lower():
            selected_dat_files.append(dat_file)

    selected_trial_cycles = [
        (dat_file.trial_number, dat_file.cycle_offset)
        for dat_file in selected_dat_files
    ]
    if len(set(selected_trial_cycles)) != len(selected_trial_cycles):
        raise ValueError(
            f"Duplicate trial/cycle numbers found within {selected_prefix} "
            f"for DMD{dmd_num}."
        )
    selected_trial_numbers = [
        dat_file.trial_number for dat_file in selected_dat_files
    ]
    available_trial_count = retained_trial_count + n_excluded_trailing_trials
    invalid_trial_numbers = [
        trial for trial in selected_trial_numbers
        if trial < 1 or trial > available_trial_count
    ]
    if invalid_trial_numbers:
        raise ValueError(
            f"Acquisition {selected_prefix} has DMD{dmd_num} trial numbers outside "
            f"the available summary's 1..{available_trial_count} range: "
            f"{invalid_trial_numbers}."
        )

    expected_samples = int(np.sum(trial_num_frames))
    sample_arrays = {
        'frame_line_idxs': frame_line_idxs,
        'F0': f0_data,
        'dF_denoised': df_denoised_data,
        'events': events_data,
    }
    for name, values in sample_arrays.items():
        if len(values) != expected_samples:
            raise ValueError(
                f"DMD{dmd_num} {name} has {len(values)} samples, but trial_num_frames "
                f"describes {expected_samples}."
            )

    sample_start = int(np.sum(trial_num_frames[:n_excluded_trials]))
    if n_excluded_trailing_trials:
        sample_end = len(frame_line_idxs) - int(
            np.sum(trial_num_frames[-n_excluded_trailing_trials:])
        )
        trial_end = -n_excluded_trailing_trials
    else:
        sample_end = len(frame_line_idxs)
        trial_end = None
    selected_meta_paths = [
        path for path in meta_paths
        if path.name.lower() == f'{selected_prefix}_DMD{int(dmd_num)}.meta'.lower()
    ]
    if len(selected_meta_paths) != 1:
        raise ValueError(
            f"Expected exactly one DMD{dmd_num} .meta file for {selected_prefix}, "
            f"found {len(selected_meta_paths)}."
        )

    return {
        'acquisition_prefix': selected_prefix,
        'excluded_trial_count': n_excluded_trials,
        'dat_paths': [
            dat_file.path
            # Synchronization expects chunks grouped by logical trial and then
            # ordered by their starting cycle, regardless of filesystem order.
            for dat_file in sorted(
                selected_dat_files,
                key=lambda item: (
                    item.trial_number,
                    -1 if item.cycle_offset is None else item.cycle_offset,
                    item.path.name,
                ),
            )
            if dat_file.trial_number <= retained_trial_count
        ],
        'meta_path': selected_meta_paths[0],
        'trial_num_frames': trial_num_frames[n_excluded_trials:trial_end],
        'frame_line_idxs': frame_line_idxs[sample_start:sample_end],
        'F0': f0_data[sample_start:sample_end],
        'dF_denoised': df_denoised_data[sample_start:sample_end],
        'events': events_data[sample_start:sample_end],
    }


def sync_slap2_fluorescence(dmd_name, dmd_num, experiment_summary, meta_paths, harp_data, trial_line_time_maps=None, primary_qc=None, qc_folder=None, dat_paths=None, plane_key=None, acquisition_resolution=None, continuous_mode=False):
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
    meta_paths : list of Path
        Candidate .meta files for this DMD. The file matching the selected
        acquisition is chosen during input filtering.
    harp_data : dict
        Dict from harp_utils.extract_harp containing clock signal arrays.
    trial_line_time_maps : list or None
        None for the first source-bearing plane; for secondary planes, pass the
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
    processing_chunk_num_frames = trial_num_frames.copy()
    if continuous_mode:
        trial_num_frames = np.asarray(
            [np.sum(trial_num_frames)], dtype=trial_num_frames.dtype
        )
    if acquisition_resolution is None:
        acquisition_resolution = resolve_slap2_acquisition(dat_paths, len(trial_num_frames))
    filtered = filter_slap2_acquisition(
        dmd_num,
        dat_paths,
        meta_paths,
        trial_num_frames,
        frame_line_idxs,
        f0_data,
        df_denoised_data,
        events_data,
        acquisition_resolution,
    )
    trial_num_frames = filtered['trial_num_frames']
    frame_line_idxs = filtered['frame_line_idxs']
    f0_data = filtered['F0']
    df_denoised_data = filtered['dF_denoised']
    events_data = filtered['events']
    dat_paths = filtered['dat_paths']

    print('using slap2 acquisition:', filtered['acquisition_prefix'])
    print('using slap2 meta file:', filtered['meta_path'])
    with h5py.File(filtered['meta_path']) as slap2_meta_h5:
        lines_per_cycle_raw = slap2_meta_h5['AcquisitionContainer']['ParsePlan']['linesPerCycle'][()]
    lines_per_cycle_arr = np.asarray(lines_per_cycle_raw).squeeze()
    if lines_per_cycle_arr.size != 1:
        raise ValueError(
            f"Expected scalar linesPerCycle but got shape {np.asarray(lines_per_cycle_raw).shape}"
        )
    lines_per_cycle = float(lines_per_cycle_arr.item())

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
    raw_frame_line_idxs = frame_line_idxs.copy()
    frame_line_idxs, line_index_corrections = slap2_sync.normalize_continued_trial_line_indices(
        frame_line_idxs, trial_num_frames, plane_name=dmd_name
    )

    if trial_line_time_maps is None:
        timestamps, out_maps, sync_qc_values = slap2_sync.get_slap2_primary_plane_timestamps(
            frame_line_idxs, trial_num_frames, lines_per_cycle,
            harp_data['slap2_cycle_clock_signal'],
            harp_data['slap2_cycle_clock_times'],
            trial_num_cycles=trial_num_cycles,
            highest_dat_trial=acquisition_resolution['highest_dat_trial'],
            first_trial_start=harp_data['normalized_slap2_start'][0],
        )
        print(f"PRODUCED {len(timestamps)} SLAP2 TIMESTAMPS for {len(f0_data)} (DMD{dmd_num})")
        plane_qc = {
            'raw_frame_line_idxs': raw_frame_line_idxs,
            'frame_line_idxs': frame_line_idxs,
            'trial_num_frames': trial_num_frames,
            'processing_chunk_num_frames': processing_chunk_num_frames,
            'lines_per_cycle': lines_per_cycle,
            'timestamps': timestamps,
            'trial_line_time_maps': out_maps,
            'sync_qc_values': sync_qc_values,
            'line_index_corrections': line_index_corrections,
            'acquisition_prefix': filtered['acquisition_prefix'],
            'excluded_trial_count': filtered['excluded_trial_count'],
        }
        return fluorescence, timestamps, out_maps, plane_qc
    else:
        timestamps = slap2_sync.get_slap2_secondary_plane_timestamps(
            frame_line_idxs, trial_num_frames, trial_line_time_maps
        )
        print(f"PRODUCED {len(timestamps)} SLAP2 TIMESTAMPS for {len(f0_data)} (DMD{dmd_num})")
        plane_qc = {
            'raw_frame_line_idxs': raw_frame_line_idxs,
            'frame_line_idxs': frame_line_idxs,
            'trial_num_frames': trial_num_frames,
            'processing_chunk_num_frames': processing_chunk_num_frames,
            'timestamps': timestamps,
            'line_index_corrections': line_index_corrections,
            'acquisition_prefix': filtered['acquisition_prefix'],
            'excluded_trial_count': filtered['excluded_trial_count'],
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
                primary_raw_frame_line_idxs=primary_qc['raw_frame_line_idxs'],
                secondary_raw_frame_line_idxs=raw_frame_line_idxs,
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


def filter_planes_with_sources(plane_inputs, experiment_summary):
    """Skip DMDs without source data, unless no DMD has sources."""
    planes_with_sources = []
    skipped_dmds = []
    for plane_input in plane_inputs:
        plane, dmd_name = plane_input[:2]
        plane_group = experiment_summary[plane]
        if 'sources' in plane_group and len(plane_group['sources']) > 0:
            planes_with_sources.append(plane_input)
        else:
            skipped_dmds.append(dmd_name)

    if not planes_with_sources:
        raise ValueError(
            "Cannot package ophys data because all DMDs are missing sources or "
            "have empty sources."
        )

    for dmd_name in skipped_dmds:
        warnings.warn(
            f"{dmd_name} is missing sources or has empty sources; skipping "
            "source-dependent packaging for this DMD.",
            UserWarning,
        )
    return planes_with_sources


def add_ophys_to_nwb(
    experiment_summary,
    nwbfile,
    instrument_json,
    acquisition_json,
    harp_data,
    session_path,
    qc_folder=None,
    exclude_final_trial=False,
):
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
    selected_acquisition_prefix = None
    excluded_trial_count = None

    plane_inputs = []
    for plane in experiment_summary.keys():
        dmd_name, dmd_num = get_dmd_name(plane)
        if not dmd_name:
            continue
        meta_paths = list(session_path.rglob(f"*DMD{dmd_num}.meta"))
        dat_paths = list(session_path.rglob(f"*DMD{dmd_num}-TRIAL*.dat"))
        n_summary_trials = len(experiment_summary[plane]['frame_info']['trial_num_frames'][()][0])
        plane_inputs.append((plane, dmd_name, dmd_num, meta_paths, dat_paths, n_summary_trials))

    source_plane_inputs = filter_planes_with_sources(plane_inputs, experiment_summary)
    source_planes = {plane_input[0] for plane_input in source_plane_inputs}
    summary_trial_counts = {plane_input[-1] for plane_input in source_plane_inputs}
    if len(summary_trial_counts) != 1:
        raise ValueError(
            f"DMD experiment summaries have different trial counts: "
            f"{sorted(summary_trial_counts)}"
        )
    all_dat_paths = [
        dat_path
        for _, _, _, _, dat_paths, _ in source_plane_inputs
        for dat_path in dat_paths
    ]
    continuous_mode = infer_continuous_slap2_mode(
        experiment_summary, source_plane_inputs, harp_data
    )
    # Missing DO1 is supported only after the independent continuous-acquisition
    # evidence above succeeds. Trial-based data still require closing pulses.
    if (
        len(harp_data['normalized_slap2_start']) == 1
        and len(harp_data['normalized_slap2_end']) == 0
        and not continuous_mode
    ):
        raise ValueError(
            "SLAP2 end pulse is missing, but the session was not recognized as "
            "a continuous acquisition."
        )
    effective_trial_count = 1 if continuous_mode else summary_trial_counts.pop()
    if continuous_mode:
        print(
            "Continuous SLAP2 mode inferred: treating source-extraction chunks "
            "as one acquisition trial."
        )
    acquisition_resolution = resolve_slap2_acquisition(
        all_dat_paths,
        effective_trial_count,
        excluded_trailing_trials=int(exclude_final_trial),
    )

    for plane, dmd_name, dmd_num, meta_paths, dat_paths, _ in plane_inputs:
        imaging_plane = create_imaging_plane(nwbfile, dmd_name, device, acquisition_json, plane_key=plane)
        add_mean_images(experiment_summary, dmd_name, ophys_mod, plane_key=plane)
        if plane not in source_planes:
            continue

        print(f'found {len(dat_paths)} .dat files for DMD{dmd_num}')
        sync_qc_folder = (qc_folder / 'syncing') if qc_folder is not None else None
        fluorescence, timestamps, out_maps, plane_qc = sync_slap2_fluorescence(
            dmd_name, dmd_num, experiment_summary, meta_paths, harp_data,
            trial_line_time_maps=trial_line_time_maps,
            primary_qc=primary_qc,
            qc_folder=sync_qc_folder,
            dat_paths=dat_paths,
            plane_key=plane,
            acquisition_resolution=acquisition_resolution,
            continuous_mode=continuous_mode,
        )
        if selected_acquisition_prefix is None:
            selected_acquisition_prefix = plane_qc['acquisition_prefix']
            excluded_trial_count = plane_qc['excluded_trial_count']
        elif (
            plane_qc['acquisition_prefix'] != selected_acquisition_prefix
            or plane_qc['excluded_trial_count'] != excluded_trial_count
        ):
            raise ValueError(
                f"DMD acquisition filtering disagrees across planes: expected "
                f"{selected_acquisition_prefix} with {excluded_trial_count} excluded "
                f"trials, but DMD{dmd_num} selected {plane_qc['acquisition_prefix']} "
                f"with {plane_qc['excluded_trial_count']} excluded trials."
            )
        if out_maps is not None:
            trial_line_time_maps = out_maps
            primary_qc = plane_qc
        else:
            secondary_qc = {
                'secondary_frame_line_idxs': plane_qc['frame_line_idxs'],
                'secondary_trial_num_frames': plane_qc['trial_num_frames'],
                'secondary_timestamps': plane_qc['timestamps'],
            }

        roi_table = add_image_segmentation(experiment_summary, imaging_plane, dmd_name, image_segmentation, plane_key=plane)
        add_fluorescence(fluorescence, timestamps, dmd_name, roi_table, ophys_mod)


if __name__ == "__main__":
    main()