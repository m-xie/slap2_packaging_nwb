"""Package labeled user-drawn soma ROIs independently of extracted sources."""

import numpy as np
import pynwb


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
):
    """Store soma masks and measured traces with the synchronized plane clock."""
    if "soma_F" not in fluorescence:
        return
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
    for name in ("F", "Fsvd"):
        key = f"soma_{name}"
        if key not in fluorescence:
            continue
        values = fluorescence[key]
        if values.shape[0] != len(timestamps) or values.shape[2] != len(indices):
            raise ValueError(f"{key} does not match soma ROIs and synchronized timestamps")
        for channel, color in enumerate(("green", "red")[:values.shape[1]]):
            container.add_roi_response_series(pynwb.ophys.RoiResponseSeries(
                name=f"{dmd_name}_soma_{name}_{color}",
                data=values[valid, channel, :], rois=rois, unit="a.u.",
                timestamps=np.asarray(timestamps)[valid],
                description=f"Soma {name} from {plane_group.name}/user_rois/{name}, "
                f"{color} channel; source values unchanged (not F0 or dF/F). "
                "Same HARP clock and sample exclusions as extracted-source fluorescence.",
            ))