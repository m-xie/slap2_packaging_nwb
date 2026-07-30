import numpy as np
import harp
import requests
import yaml
import pathlib as pl
import io


def _get_yml_from_who_am_i(who_am_i: int, release: str = "main") -> io.BytesIO:
    try:
        device = _get_who_am_i_list()[who_am_i]
    except KeyError as e:
        raise KeyError(f"WhoAmI {who_am_i} not found in whoami.yml") from e

    repository_url = device.get("repositoryUrl", None)

    if repository_url is None:
        raise ValueError("Device's repositoryUrl not found in whoami.yml")
    else:  # attempt to get the device.yml from the repository
        _repo_hint_paths = [
            "{repository_url}/{release}/device.yml",
            "{repository_url}/{release}/software/bonsai/device.yml",
        ]

        yml = None
        for hint in _repo_hint_paths:
            url = hint.format(repository_url=repository_url, release=release)
            if "github.com" in url:
                url = url.replace("github.com", "raw.githubusercontent.com")
            response = requests.get(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                yml = io.BytesIO(response.content)
                break
        if yml is None:
            raise FileNotFoundError("device.yml not found in any repository")
        else:
            return yml

def _get_who_am_i_list(url: str = "https://raw.githubusercontent.com/harp-tech/protocol/main/whoami.yml"):
    response = requests.get(url, allow_redirects=True, timeout=5)
    content = response.content.decode("utf-8")
    content = yaml.safe_load(content)
    devices = content["devices"]
    return devices


def fetch_yml(harp_path):
    with open(harp_path / 'Behavior_0.bin',mode='rb') as reg_0:
        who_am_i = int(harp.read(reg_0).values[0][0])
        yml_bytes = _get_yml_from_who_am_i(who_am_i)
    yaml_content = yml_bytes.getvalue()
    with open(harp_path / "device.yml", "wb") as f:
        f.write(yaml_content)
    return harp_path / "device.yml"


def extract_harp(harp_path, expected_n_trials=None):
    harp_path = pl.Path(harp_path)
    if not (harp_path / "device.yml").exists():
        fetch_yml(harp_path)
    reader = harp.create_reader(harp_path)
    analog_data = reader.AnalogData.read()
    analog_times = analog_data.index.to_numpy()
    photodiode = analog_data["AnalogInput0"].to_numpy()
    # Native signed int16 quadrature counts, paired one-to-one with analog_times.
    # Conversion to angle and speed belongs in slap2_running_packaging.
    wheel = analog_data["Encoder"].to_numpy()
    slap2_cycle_clock_signal = reader.DigitalInputState.read()['DI3'].to_numpy()
    slap2_cycle_clock_times = reader.DigitalInputState.read()['DI3'].index.to_numpy()
    slap2_start_signal = reader.PulseDO0.read()["PulseDO0"].to_numpy()
    slap2_start_times = reader.PulseDO0.read()["PulseDO0"].index.to_numpy()
    slap2_end_signal = reader.PulseDO1.read()["PulseDO1"].to_numpy()
    slap2_end_times = reader.PulseDO1.read()["PulseDO1"].index.to_numpy()
    grating_signal = reader.PulseDO2.read()["PulseDO2"].to_numpy()
    grating_times = reader.PulseDO2.read()["PulseDO2"].index.to_numpy()

    # All times are normalized to the first SLAP2 start pulse
    time_reference = slap2_start_times[0]
    slap2_cycle_clock_times   -= time_reference
    slap2_start_times         -= time_reference
    slap2_end_times           -= time_reference
    analog_times              -= time_reference
    grating_times             -= time_reference

    time_dict = {
        "analog_times": analog_times,
        "photodiode": photodiode,
        "wheel": wheel,
        "slap2_cycle_clock_signal": slap2_cycle_clock_signal,
        "slap2_cycle_clock_times": slap2_cycle_clock_times,
        "slap2_start_signal": slap2_start_signal,
        "slap2_start_times": slap2_start_times,
        "slap2_end_signal": slap2_end_signal,
        "slap2_end_times": slap2_end_times,
        "grating_signal": grating_signal,
        "grating_times": grating_times,
        "time_reference": time_reference,
        # Aliases kept for backward compatibility
        "normalized_start_gratings": grating_times,
        "normalized_slap2_start": slap2_start_times,
        "normalized_slap2_end": slap2_end_times,
        "normalized_analog_times": analog_times,
        "normalized_slap2_cycle_clock_times": slap2_cycle_clock_times,
    }

    if expected_n_trials is not None:
        print('given expected n trials:', expected_n_trials)
        for key in ("slap2_start_signal", "slap2_start_times", "slap2_end_signal", "slap2_end_times", "normalized_slap2_start", "normalized_slap2_end"):
            time_arr = time_dict[key]
            trials_to_slice = len(time_arr) - expected_n_trials
            if trials_to_slice < 0:
                raise Exception('There are more expected trials than recorded harp trials, something is wrong...')
            print(f"slicing {trials_to_slice} from {key}:{len(time_arr)} to get {expected_n_trials}")
            time_dict[key] = time_dict[key][trials_to_slice:]
    return time_dict


def get_concatenated_timestamps(trace, trial_start_idxs, harp_data):
    start_trial_times = harp_data["normalized_slap2_start"]
    end_trial_times = harp_data["normalized_slap2_end"]
    assert len(start_trial_times) == len(end_trial_times) == len(trial_start_idxs), "Length of start times, end times, and trial start indices must be equal"
    # print(len(start_trial_times), len(end_trial_times), len(trial_start_idxs), trace.shape[0])

    timestamps = np.empty(trace.shape[0])
    for i in range(len(trial_start_idxs)):
        start_idx = trial_start_idxs[i]
        end_idx = trial_start_idxs[i+1] if i < len(trial_start_idxs) - 1 else len(trace)
        num_frames = end_idx - start_idx
        trial_timestamps = np.linspace(start_trial_times[i], end_trial_times[i], num_frames)
        timestamps[start_idx:end_idx] = trial_timestamps
    return timestamps


def get_concatenated_timestamps_from_num_frames(trace, trial_num_frames, harp_data):
    start_trial_times = harp_data["normalized_slap2_start"]
    end_trial_times = harp_data["normalized_slap2_end"]
    print(len(start_trial_times), len(end_trial_times), len(trial_num_frames), trace.shape[0])

    assert len(start_trial_times) == len(end_trial_times) == len(trial_num_frames), "Length of start times, end times, and trial start indices must be equal"

    timestamps = []
    for i, n_frames in enumerate(trial_num_frames):
        if n_frames == 0:
            continue
        trial_timestamps = np.linspace(start_trial_times[i], end_trial_times[i], n_frames)
        timestamps.extend(trial_timestamps)

    assert len(timestamps) == len(trace), f"Length of generated timestamps {len(timestamps)} don't match length of trace recording {len(trace)}"
    return np.array(timestamps)