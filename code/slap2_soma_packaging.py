"""Package labeled user-drawn soma ROIs independently of extracted sources."""

import numpy as np
import pynwb
from scbc.utils.signal import compute_f0


def compute_soma_dff(
    fsvd, timestamps, *, trial_num_frames=None,
    denoise_window_s=1.0, baseline_window_s=4.0,
):
    """Return F0 and (Fsvd - F0) / F0, preserving time × channel × soma axes.

    Estimate the sample rate from finite, positive adjacent HARP intervals
    within retained acquisition trials. Compute each trial independently;
    continuous source-extraction chunks must be passed as one trial. Baselines
    are computed before removing samples with unsupported timestamps, so those
    removals never compress the input time axis. Zero/non-finite baselines and
    non-finite Fsvd samples produce NaN dF/F, not infinities.
    """
    fsvd = np.asarray(fsvd, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if fsvd.ndim != 3 or timestamps.shape != (len(fsvd),):
        raise ValueError("Soma Fsvd must be (samples, channels, ROIs) and match timestamps")
    windows = np.asarray([denoise_window_s, baseline_window_s], dtype=float)
    if not np.all(np.isfinite(windows) & (windows > 0)):
        raise ValueError("Soma baseline windows must be finite and positive")
    counts = np.asarray([len(fsvd)] if trial_num_frames is None else trial_num_frames)
    if (
        counts.ndim != 1 or not np.all(np.isfinite(counts))
        or np.any(counts < 0) or np.any(counts != np.floor(counts))
        or counts.sum() != len(fsvd)
    ):
        raise ValueError("Soma trial frame counts must be nonnegative integers summing to samples")
    boundaries = np.r_[0, np.cumsum(counts.astype(int))]
    intervals = np.diff(timestamps)
    # Exclude inter-trial gaps, even when they happen to resemble a sample interval.
    trial_ends = boundaries[1:-1]
    trial_ends = trial_ends[(trial_ends > 0) & (trial_ends < len(fsvd))]
    intervals[trial_ends - 1] = np.nan
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    f0 = np.full_like(fsvd, np.nan)
    dff = np.full_like(fsvd, np.nan)
    if not fsvd.size or not np.any(np.isfinite(timestamps)):
        return f0, dff
    if not len(intervals):
        raise ValueError("Cannot estimate soma sample rate from adjacent HARP timestamps")
    sample_windows = np.maximum(1, np.ceil(windows / np.median(intervals))).astype(int)
    cleaned = np.where(np.isfinite(fsvd), fsvd, np.nan)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop == start or not np.any(np.isfinite(cleaned[start:stop])):
            continue
        trial = cleaned[start:stop]
        # SCBC's moving mean needs at least three decimated samples. Its
        # short-trace fallback only handles T < 4; edge-pad the remaining
        # undersized inputs instead of reimplementing its baseline estimator.
        minimum_samples = int(np.floor(max(4.0, sample_windows[0] / 6.0))) + 1
        if 4 <= len(trial) < minimum_samples:
            trial = np.pad(trial, ((0, minimum_samples - len(trial)), (0, 0), (0, 0)), mode="edge")
        f0[start:stop] = compute_f0(
            trial, denoise_window=int(sample_windows[0]),
            hull_window=int(sample_windows[1]),
        )[:stop - start]
    valid = np.isfinite(fsvd) & np.isfinite(f0) & (f0 != 0)
    np.divide(fsvd - f0, f0, out=dff, where=valid)
    return f0, dff


def soma_roi_selection(plane_group):
    """Return original labels and soma indices; absent user ROIs are optional."""
    if "user_rois" not in plane_group or not len(plane_group["user_rois"]):
        return [], np.array([], dtype=int)
    group = plane_group["user_rois"]
    labels = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(group["labels"][()]).reshape(-1)
    ]
    indices = np.array([
        index for index, label in enumerate(labels)
        if label.strip().casefold() == "soma"
    ], dtype=int)
    return labels, indices


def read_soma_traces(plane_group, sample_count):
    """Read time × channel × soma arrays without interpreting Fsvd as dF/F."""
    labels, indices = soma_roi_selection(plane_group)
    if not len(indices):
        return {}
    group = plane_group["user_rois"]
    traces = {}
    for name in ("F", "Fsvd"):
        if name == "Fsvd" and name not in group:
            continue
        dataset = group[name]
        if (
            dataset.ndim != 3 or dataset.shape[0] != sample_count
            or dataset.shape[2] != len(labels) or not 1 <= dataset.shape[1] <= 2
        ):
            raise ValueError(
                f"user_rois/{name} must have shape (samples={sample_count}, "
                f"channels=1 or 2, ROIs={len(labels)}); got {dataset.shape}"
            )
        traces[f"soma_{name}"] = dataset[:, :, indices]
    if "soma_Fsvd" in traces and traces["soma_Fsvd"].shape != traces["soma_F"].shape:
        raise ValueError("user_rois/F and Fsvd must have matching shapes")
    return traces


def add_soma_fluorescence(
    plane_group, fluorescence, timestamps, dmd_name, imaging_plane,
    image_segmentation, ophys_mod, pixel_mask_from_volume,
    *, trial_num_frames=None,
):
    """Store soma masks, measured traces, and Fsvd-derived F0/dF/F on the plane clock."""
    if "soma_F" not in fluorescence:
        return
    fluorescence = dict(fluorescence)
    if "soma_Fsvd" in fluorescence:
        fluorescence["soma_F0"], fluorescence["soma_dFF"] = compute_soma_dff(
            fluorescence["soma_Fsvd"], timestamps, trial_num_frames=trial_num_frames,
        )
    labels, indices = soma_roi_selection(plane_group)
    # Same MATLAB axis convention as sources/spatial/profiles.
    masks = plane_group["user_rois"]["mask"]
    if masks.ndim != 4 or masks.shape[-1] != len(labels):
        raise ValueError("user_rois/mask must have shape (y, x, z, user ROIs)")
    masks = masks[:, :, :, indices].transpose()
    table = image_segmentation.create_plane_segmentation(
        name=f"SomaPlaneSegmentation_{dmd_name}",
        description=f"User-drawn soma ROIs from {plane_group.name}/user_rois/mask",
        imaging_plane=imaging_plane,
    )
    table.add_column(name="label", description="Original user ROI label")
    table.add_column(name="user_roi_index", description="Zero-based index in summary user_rois")
    table.add_column(name="z_min", description="Minimum z-index of the soma ROI")
    table.add_column(name="z_max", description="Maximum z-index of the soma ROI")
    for index, mask in zip(indices, masks):
        pixel_mask, z_min, z_max = pixel_mask_from_volume(mask)
        table.add_roi(
            pixel_mask=pixel_mask, z_min=z_min, z_max=z_max,
            label=labels[index], user_roi_index=int(index),
        )
    rois = table.create_roi_table_region(
        region=list(range(len(indices))), description=f"Soma ROIs for {dmd_name}"
    )
    container = pynwb.ophys.Fluorescence(name=f"SomaFluorescence_{dmd_name}")
    ophys_mod.add(container)
    valid = np.isfinite(timestamps)
    for name in ("F", "Fsvd", "F0", "dFF"):
        key = f"soma_{name}"
        if key not in fluorescence:
            continue
        values = fluorescence[key]
        if values.shape[0] != len(timestamps) or values.shape[2] != len(indices):
            raise ValueError(f"{key} does not match soma ROIs and synchronized timestamps")
        if name in ("F0", "dFF"):
            description = (
                f"Soma {name} derived from {plane_group.name}/user_rois/Fsvd using "
                "scbc.utils.signal.compute_f0 (SCBC commit "
                "0307c3fa7e3cc2a0611e0f796c0c4f9d675db76c); "
                "1 s median denoising, 4 s baseline window, converted to samples "
                "using median within-trial HARP sample spacing. "
                "F0 estimated independently per retained acquisition trial before "
                "timestamp exclusions; continuous processing chunks are one trial. "
                "Undersized traces are edge-padded for SCBC's decimated smoother. "
                "dFF = (Fsvd - F0) / F0; zero/non-finite baselines yield NaN dFF. "
            )
        else:
            description = (
                f"Soma {name} from {plane_group.name}/user_rois/{name}; "
                "source values unchanged (not F0 or dF/F). "
            )
        for channel, color in enumerate(("green", "red")[:values.shape[1]]):
            container.add_roi_response_series(pynwb.ophys.RoiResponseSeries(
                name=f"{dmd_name}_soma_{name}_{color}",
                data=values[valid, channel, :], rois=rois,
                unit="dimensionless" if name == "dFF" else "a.u.",
                timestamps=np.asarray(timestamps)[valid],
                description=description + f"{color} channel. "
                "Same HARP clock and sample exclusions as extracted-source fluorescence.",
            ))