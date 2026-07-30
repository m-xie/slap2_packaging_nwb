"""Package processed EyeCamera ellipse fits on the shared SLAP2 HARP clock.

Video inference is performed upstream. Its output is a pandas HDF5 file with
``cr``, ``eye``, and ``pupil`` keys. Each key contains one row per source video
frame and the columns ``center_x``, ``center_y``, ``height``, ``width``, and
``phi``. The integer DataFrame index is the source ``CameraFrameNumber``.

EyeCamera ``metadata.csv`` supplies authoritative per-frame HARP timestamps.
They are joined to fits by frame number and normalized by the same first-SLAP2
start timestamp used for fluorescence, stimuli, and running data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb


EYE_TRACKING_FIELDS = ("cr", "eye", "pupil")
ELLIPSE_COLUMNS = ("center_x", "center_y", "height", "width", "phi")
FRAME_COUNT_TOLERANCE = 3


def dilate_blink_frames(blinks, dilation_frames):
    """Expand blink flags to neighboring frames without requiring SciPy."""
    blinks = np.asarray(blinks, dtype=bool)
    if dilation_frames == 0 or len(blinks) == 0:
        return blinks.copy()

    padded = np.pad(blinks, dilation_frames, mode="constant")
    window = np.ones(2 * dilation_frames + 1, dtype=np.int64)
    return np.convolve(padded, window, mode="valid") > 0


def load_eye_tracking_hdf(eye_tracking_path):
    """Load and validate the three frame-indexed upstream ellipse tables."""
    eye_tracking_path = Path(eye_tracking_path)
    tables = {}
    for field_name in EYE_TRACKING_FIELDS:
        table = pd.read_hdf(eye_tracking_path, key=field_name)
        missing = set(ELLIPSE_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(
                f"Eye tracking key '{field_name}' is missing columns: "
                f"{sorted(missing)}"
            )
        if not table.index.is_unique:
            raise ValueError(f"Eye tracking key '{field_name}' has duplicate frame indices")
        try:
            frame_index = table.index.to_numpy(dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Eye tracking key '{field_name}' must use integer video frame indices"
            ) from exc
        if not np.array_equal(frame_index, table.index.to_numpy()):
            raise ValueError(
                f"Eye tracking key '{field_name}' contains non-integer frame indices"
            )

        table = table.loc[:, ELLIPSE_COLUMNS].copy()
        table.index = pd.Index(frame_index, name="frame")
        for column in ELLIPSE_COLUMNS:
            table[column] = np.real(table[column].to_numpy()).astype(float)
        tables[field_name] = table

    reference_index = tables[EYE_TRACKING_FIELDS[0]].index
    for field_name in EYE_TRACKING_FIELDS[1:]:
        if not tables[field_name].index.equals(reference_index):
            raise ValueError("cr, eye, and pupil tables must have identical frame indices")

    return pd.concat(
        [tables[name].add_prefix(f"{name}_") for name in EYE_TRACKING_FIELDS],
        axis=1,
    )


def load_camera_frame_times(camera_metadata_path, harp_time_reference):
    """Load frame-numbered EyeCamera timestamps and place them on NWB time."""
    metadata = pd.read_csv(camera_metadata_path, encoding="utf-8-sig")
    required = {"CameraFrameNumber", "CameraFrameTime"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"EyeCamera metadata is missing columns: {sorted(missing)}")

    frames = pd.to_numeric(metadata["CameraFrameNumber"], errors="raise").to_numpy()
    frame_numbers = frames.astype(np.int64)
    if not np.array_equal(frames, frame_numbers):
        raise ValueError("CameraFrameNumber values must be integers")
    if len(np.unique(frame_numbers)) != len(frame_numbers):
        raise ValueError("EyeCamera metadata contains duplicate frame numbers")

    absolute_times = pd.to_numeric(
        metadata["CameraFrameTime"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(absolute_times)):
        raise ValueError("EyeCamera metadata contains non-finite timestamps")
    if len(absolute_times) > 1 and not np.all(np.diff(absolute_times) > 0):
        raise ValueError("EyeCamera timestamps must be strictly increasing")

    return pd.Series(
        absolute_times - float(harp_time_reference),
        index=pd.Index(frame_numbers, name="frame"),
        name="timestamps",
    )


def align_eye_tracking_frames(
    eye_data,
    frame_times,
    require_all_frames=True,
    tolerance=FRAME_COUNT_TOLERANCE,
):
    """Join processed fits to HARP timestamps by source video frame number."""
    missing_times = eye_data.index.difference(frame_times.index)
    missing_fits = frame_times.index.difference(eye_data.index)
    discrepancy = abs(len(eye_data) - len(frame_times))
    indices_are_nested = not len(missing_times) or not len(missing_fits)
    if discrepancy > tolerance or not indices_are_nested:
        raise ValueError(
            f"Eye tracking fits and camera timestamps cannot be aligned: "
            f"length discrepancy is {discrepancy} frame(s), with "
            f"{len(missing_times)} fit-only and {len(missing_fits)} timestamp-only "
            f"frame ID(s); maximum tolerated discrepancy is {tolerance}"
        )
    if require_all_frames and discrepancy:
        print(
            f"Eye tracking frame counts differ by {discrepancy}; "
            f"truncating to {min(len(eye_data), len(frame_times))} shared frames."
        )

    aligned = eye_data.loc[eye_data.index.intersection(frame_times.index)].copy()
    aligned.insert(0, "timestamps", frame_times.loc[aligned.index].to_numpy())
    return aligned


def compute_tracking_metrics(eye_data, z_threshold=3.0, dilation_frames=2):
    """Compute reference-compatible areas, outliers, and likely blink flags.

    The area formulas intentionally match the existing AIND eye packager. Eye
    and corneal-reflection fits use ``pi * width * height``; pupil area uses
    ``pi * max(width, height)^2``. This assumes the upstream fit dimensions use
    the radius convention expected by that pipeline.
    """
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    if dilation_frames < 0:
        raise ValueError("dilation_frames must be non-negative")

    result = eye_data.copy()
    raw_areas = {
        "cr": np.pi * result["cr_width"] * result["cr_height"],
        "eye": np.pi * result["eye_width"] * result["eye_height"],
        "pupil": np.pi * result[["pupil_width", "pupil_height"]].max(
            axis=1, skipna=False
        ) ** 2,
    }
    area_values = pd.DataFrame({"eye": raw_areas["eye"], "pupil": raw_areas["pupil"]})
    means = area_values.mean(axis=0, skipna=True)
    standard_deviations = area_values.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    z_scores = (area_values - means) / standard_deviations
    outliers = z_scores.abs().gt(z_threshold).any(axis=1)
    tracked_fit_columns = [
        f"{field_name}_{column}"
        for field_name in ("eye", "pupil")
        for column in ELLIPSE_COLUMNS
    ]
    missing_fits = ~np.isfinite(result[tracked_fit_columns]).all(axis=1)
    blinks = (outliers | missing_fits).to_numpy(dtype=bool)
    if dilation_frames:
        blinks = dilate_blink_frames(blinks, dilation_frames)

    result.insert(1, "likely_blink", blinks)
    for field_name in EYE_TRACKING_FIELDS:
        result[f"{field_name}_area_raw"] = raw_areas[field_name]
        result[f"{field_name}_area"] = raw_areas[field_name].mask(blinks, -1.0)
    return result


def add_eye_tracking_to_nwbfile(nwbfile, eye_data):
    """Add ellipse tables and blink flags to ``processing/eye_tracking``."""
    if "eye_tracking" in nwbfile.processing:
        raise ValueError("NWBFile already contains an eye_tracking processing module")

    eye_module = pynwb.ProcessingModule("eye_tracking", "eye tracking data")
    nwbfile.add_processing_module(eye_module)
    table_specs = (
        ("ellipse", "eye"),
        ("pupil", "pupil"),
        ("corneal_reflection", "cr"),
    )
    for table_name, prefix in table_specs:
        table_data = pd.DataFrame({
            "frame": eye_data.index.to_numpy(dtype=np.int64),
            "reference_frame": ["nose"] * len(eye_data),
            "data_x": eye_data[f"{prefix}_center_x"].to_numpy(),
            "data_y": eye_data[f"{prefix}_center_y"].to_numpy(),
            "area": eye_data[f"{prefix}_area"].to_numpy(),
            "area_raw": eye_data[f"{prefix}_area_raw"].to_numpy(),
            "width": eye_data[f"{prefix}_width"].to_numpy(),
            "height": eye_data[f"{prefix}_height"].to_numpy(),
            "angle": eye_data[f"{prefix}_phi"].to_numpy(),
            "timestamps": eye_data["timestamps"].to_numpy(),
        })
        table_data.index = pd.RangeIndex(len(table_data))
        eye_module.add(
            pynwb.core.DynamicTable.from_dataframe(name=table_name, df=table_data)
        )

    eye_module.add(
        pynwb.TimeSeries(
            name="likely_blink_times",
            timestamps=eye_data["timestamps"].to_numpy(),
            data=eye_data["likely_blink"].to_numpy(dtype=bool),
            unit="is_blink",
            description="Frames with missing or outlying eye/pupil fits, dilated in time.",
        )
    )
    return nwbfile


def plot_eye_tracking_qc(eye_data, output_path):
    """Plot timestamp continuity, pupil geometry, and likely blink coverage."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = eye_data["timestamps"].to_numpy()
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(timestamps[1:], np.diff(timestamps) * 1000, linewidth=0.5)
    axes[0].set(ylabel="Frame interval (ms)", title="EyeCamera HARP timing")
    axes[1].plot(timestamps, eye_data["pupil_area_raw"], linewidth=0.5)
    axes[1].set(ylabel="Pupil area (pixel^2)", title="Raw pupil area")
    axes[2].plot(timestamps, eye_data["pupil_center_x"], linewidth=0.5, label="x")
    axes[2].plot(timestamps, eye_data["pupil_center_y"], linewidth=0.5, label="y")
    blink_times = timestamps[eye_data["likely_blink"].to_numpy(dtype=bool)]
    finite_center_y = eye_data.loc[
        np.isfinite(eye_data["pupil_center_y"]), "pupil_center_y"
    ]
    blink_y = finite_center_y.min() if len(finite_center_y) else 0.0
    axes[2].scatter(
        blink_times,
        np.full(len(blink_times), blink_y),
        s=1,
        label="blink",
    )
    axes[2].set(xlabel="Time from first SLAP2 start (s)", ylabel="Pupil center (pixel)")
    axes[2].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def package_eye_tracking(
    nwbfile,
    eye_tracking_path,
    camera_metadata_path,
    harp_time_reference,
    z_threshold=3.0,
    dilation_frames=2,
    require_all_frames=True,
    qc_output_path=None,
):
    """Load, align, derive, and package upstream eye-tracking results."""
    eye_data = load_eye_tracking_hdf(eye_tracking_path)
    frame_times = load_camera_frame_times(camera_metadata_path, harp_time_reference)
    eye_data = align_eye_tracking_frames(
        eye_data, frame_times, require_all_frames=require_all_frames
    )
    eye_data = compute_tracking_metrics(
        eye_data, z_threshold=z_threshold, dilation_frames=dilation_frames
    )
    add_eye_tracking_to_nwbfile(nwbfile, eye_data)
    if qc_output_path is not None:
        plot_eye_tracking_qc(eye_data, qc_output_path)
    return eye_data