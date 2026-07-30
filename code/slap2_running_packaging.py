"""Package HARP running-wheel data into an open SLAP2 NWB file.

The SLAP2 rig uses a CUI Devices AMT102-V encoder and a HARP Behavior device.
``AnalogData.Encoder`` is the raw signed 16-bit quadrature counter, not an
angle in radians. The acquisition workflow converts it with
``degrees = 360 * counts / 8192``, establishing 8192 counts per revolution.

HARP samples the counter independently of wheel movement at approximately
1 kHz. Every counter sample and its ``AnalogData`` index timestamp therefore
form a native one-to-one pair. ``harp_utils.extract_harp`` expresses those
timestamps on the same clock used by SLAP2 and stimulus events, relative to the
first SLAP2 start pulse; no additional running-to-SLAP2 synchronization is
needed here.

The conversion performed by this module is::

    radians = count_delta * 2*pi / counts_per_revolution
    speed_cm_s = radians * (wheel_radius_cm * subject_position) / delta_time

The 8.255 cm disc radius is recorded in the session instrument metadata. The
default subject position of 2/3 follows the existing AIND running package and
represents an effective running radius of about 5.5 cm, not a fraction of a
mouse's body position. Parameters remain overrideable through this Python API,
but the capsule entry point intentionally uses the rig defaults.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb


DEFAULT_RUNNING_SPEED_UNITS = {
    "velocity": "cm/s",
    "rotation": "radians",
    "counts": "counts",
}
COUNTS_PER_REVOLUTION = 8192
FRAME_COUNT_TOLERANCE = 3


def unwrap_quadrature_counts(raw_counts, counter_bits=16):
    """Convert a wrapping signed quadrature counter to continuous counts.

    HARP stores this counter as ``int16``, so forward motion through 32767 is
    represented by ``32767, -32768, -32767``. Modular differences recover the
    physical increments before cumulative summation. This assumes that motion
    between adjacent approximately 1 ms samples is less than half the complete
    counter range, which is far beyond physically possible wheel movement.
    """
    raw_counts = np.asarray(raw_counts)
    if raw_counts.ndim != 1:
        raise ValueError("raw_counts must be one-dimensional")
    if raw_counts.size == 0:
        return np.array([], dtype=np.int64)

    modulus = 1 << counter_bits
    half_modulus = modulus // 2
    deltas = np.diff(raw_counts.astype(np.int64))
    deltas = (deltas + half_modulus) % modulus - half_modulus
    return raw_counts.astype(np.int64)[0] + np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(deltas, dtype=np.int64),
    ])


def extract_running_speeds(
    timestamps,
    raw_counts,
    counts_per_revolution=COUNTS_PER_REVOLUTION,
    wheel_radius_cm=8.255,
    subject_position=2 / 3,
    direction=1.0,
    counter_bits=16,
):
    """Convert HARP quadrature counts into interval rotation and speed data.

    Parameters
    ----------
    timestamps : array-like
        Native HARP ``AnalogData`` timestamps, already normalized to the first
        SLAP2 start by ``harp_utils.extract_harp``.
    raw_counts : array-like
        Signed wrapping quadrature counter samples. These are integer counts,
        not radians or degrees, and must pair one-to-one with ``timestamps``.
    counts_per_revolution : float
        Counter increments in one full disc revolution. The SLAP2 workflow uses
        8192, equivalent to 0.0439453125 degrees per count.
    wheel_radius_cm : float
        Physical disc radius. The MindScope Running Disc metadata records
        8.255 cm.
    subject_position : float
        Fraction of disc radius used as the animal's effective running radius.
        The AIND convention is 2/3, giving approximately 5.503 cm.
    direction : float
        Sign convention for forward running; use 1 or -1 as appropriate.
    counter_bits : int
        Width of the signed wrapping HARP counter, normally 16.

    Returns
    -------
    pandas.DataFrame
        One row per valid adjacent sample interval with start/end time, signed
        speed in cm/s, and angular displacement in radians. Non-finite, zero,
        or reversed timestamp intervals are omitted from processed data. The
        If counts and timestamps differ by at most three samples, the longer
        array is tail-truncated before both processed and raw NWB data are made.

    Notes
    -----
    Speed is computed from each measured HARP interval; constant wheel speed is
    not assumed. At native sampling resolution, low speeds may appear as zeros
    interspersed with one-count steps because of encoder quantization.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    raw_counts = np.asarray(raw_counts)
    if timestamps.ndim != 1 or raw_counts.ndim != 1:
        raise ValueError("timestamps and raw_counts must be one-dimensional")
    timestamps, raw_counts = align_running_samples(timestamps, raw_counts)
    if len(timestamps) < 2:
        raise ValueError("at least two wheel samples are required")
    if counts_per_revolution <= 0:
        raise ValueError("counts_per_revolution must be positive")
    if wheel_radius_cm <= 0:
        raise ValueError("wheel_radius_cm must be positive")
    if not 0 < subject_position <= 1:
        raise ValueError("subject_position must be in (0, 1]")

    unwrapped_counts = unwrap_quadrature_counts(raw_counts, counter_bits)
    durations = np.diff(timestamps)
    valid = np.isfinite(durations) & (durations > 0)
    if not np.any(valid):
        raise ValueError("wheel timestamps contain no increasing intervals")

    count_deltas = np.diff(unwrapped_counts).astype(float)
    rotation = direction * count_deltas * (2 * np.pi / counts_per_revolution)
    effective_radius_cm = wheel_radius_cm * subject_position
    velocity = rotation * effective_radius_cm / durations

    return pd.DataFrame({
        "start_time": timestamps[:-1][valid],
        "end_time": timestamps[1:][valid],
        "velocity": velocity[valid],
        "net_rotation": rotation[valid],
    })


def align_running_samples(
    timestamps, raw_counts, tolerance=FRAME_COUNT_TOLERANCE
):
    """Tail-truncate paired running arrays when their lengths differ slightly."""
    timestamps = np.asarray(timestamps)
    raw_counts = np.asarray(raw_counts)
    difference = abs(len(timestamps) - len(raw_counts))
    if difference > tolerance:
        raise ValueError(
            f"timestamps and raw_counts differ by {difference} samples; "
            f"maximum tolerated discrepancy is {tolerance}"
        )
    paired_length = min(len(timestamps), len(raw_counts))
    return timestamps[:paired_length], raw_counts[:paired_length]


def add_running_speed_to_nwbfile(nwbfile, running_speed, units=None):
    """Add processed speed and interval rotation to the NWB running module.

    This mirrors the established AIND running capsule layout using
    ``processing/running/running_speed`` and ``running_wheel_rotation``.
    Timestamps identify the start of each measured encoder interval.
    """
    units = units or DEFAULT_RUNNING_SPEED_UNITS
    if "running" in nwbfile.processing:
        running_module = nwbfile.processing["running"]
    else:
        running_module = pynwb.ProcessingModule("running", "running speed data")
        nwbfile.add_processing_module(running_module)

    speed_series = pynwb.TimeSeries(
        name="running_speed",
        timestamps=running_speed["start_time"].to_numpy(),
        data=running_speed["velocity"].to_numpy(),
        unit=units["velocity"],
        description="Linear running speed over each HARP encoder sample interval.",
    )
    rotation_series = pynwb.TimeSeries(
        name="running_wheel_rotation",
        timestamps=speed_series,
        data=running_speed["net_rotation"].to_numpy(),
        unit=units["rotation"],
        description="Wheel rotation over each HARP encoder sample interval.",
    )
    running_module.add(speed_series)
    running_module.add(rotation_series)
    return nwbfile


def add_raw_running_data_to_nwbfile(nwbfile, timestamps, raw_counts, units=None):
    """Add every native HARP count/timestamp pair as an NWB acquisition.

    Raw values intentionally remain signed wrapping counts. Preserving them
    makes the source representation explicit and permits later recalculation
    without incorrectly labeling the hardware register as radians.
    """
    units = units or DEFAULT_RUNNING_SPEED_UNITS
    raw_series = pynwb.TimeSeries(
        name="raw_running_wheel_counts",
        timestamps=np.asarray(timestamps, dtype=float),
        data=np.asarray(raw_counts),
        unit=units["counts"],
        description="Signed wrapping quadrature counter values from HARP AnalogData Encoder.",
    )
    nwbfile.add_acquisition(raw_series)
    return nwbfile


def plot_running_qc(running_speed, output_path):
    """Write a speed trace for checking scale, direction, wraps, and dropouts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(running_speed["start_time"], running_speed["velocity"], linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(xlabel="Time from first SLAP2 start (s)", ylabel="Speed (cm/s)", title="Running speed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def package_harp_running_data(
    nwbfile,
    harp_data,
    counts_per_revolution=COUNTS_PER_REVOLUTION,
    wheel_radius_cm=8.255,
    subject_position=2 / 3,
    direction=1.0,
    counter_bits=16,
    qc_output_path=None,
):
    """Package synchronized wheel arrays into an already-open ``NWBFile``.

    ``harp_data`` must be the mapping returned by ``harp_utils.extract_harp``;
    specifically, ``wheel`` contains raw counts and ``analog_times`` contains
    their corresponding timestamps. Processed speed/rotation are placed in the
    ``running`` processing module, paired raw counts are placed in acquisition,
    and an optional QC plot is written without reopening the NWB file. A length
    discrepancy of up to three samples is resolved by tail-truncating the
    longer input.
    """
    timestamps, raw_counts = align_running_samples(
        harp_data["analog_times"], harp_data["wheel"]
    )
    running_speed = extract_running_speeds(
        timestamps=timestamps,
        raw_counts=raw_counts,
        counts_per_revolution=counts_per_revolution,
        wheel_radius_cm=wheel_radius_cm,
        subject_position=subject_position,
        direction=direction,
        counter_bits=counter_bits,
    )
    add_running_speed_to_nwbfile(nwbfile, running_speed)
    add_raw_running_data_to_nwbfile(nwbfile, timestamps, raw_counts)
    if qc_output_path is not None:
        plot_running_qc(running_speed, qc_output_path)
    return running_speed