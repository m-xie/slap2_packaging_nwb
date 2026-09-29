from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import re

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import harp_utils
import slap2_synching


DATA_ROOT = Path("/root/capsule/data")
RESULTS_ROOT = Path("/root/capsule/results")


def session_key(path):
    match = re.match(r"(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", path.name)
    if match is None:
        raise ValueError(f"Cannot parse session name: {path.name}")
    return match.groups()


def pair_sessions():
    raw_sessions = [
        path for path in DATA_ROOT.iterdir()
        if path.is_dir() and "_processed_" not in path.name
    ]
    for processed in sorted(DATA_ROOT.glob("*_processed_*")):
        subject, date, processed_time = session_key(processed)
        candidates = [
            path for path in raw_sessions
            if session_key(path)[:2] == (subject, date)
        ]
        if not candidates:
            raise ValueError(f"No raw session matches {processed.name}")
        processed_seconds = sum(
            value * scale
            for value, scale in zip(map(int, processed_time.split("-")), (3600, 60, 1))
        )
        raw = min(
            candidates,
            key=lambda path: abs(
                sum(
                    value * scale
                    for value, scale in zip(
                        map(int, session_key(path)[2].split("-")), (3600, 60, 1)
                    )
                ) - processed_seconds
            ),
        )
        yield raw, processed


def source_plane(summary):
    for plane_name in sorted(summary):
        plane = summary[plane_name]
        if (
            re.match(r"(?i)(?:path|dmd)\d+", plane_name)
            and "frame_info" in plane
            and "frame_line_idxs" in plane["frame_info"]
        ):
            dmd_number = int(re.search(r"\d+", plane_name).group())
            return plane_name, dmd_number
    raise ValueError("Experiment summary has no source-bearing DMD plane")


def lines_per_cycle(raw_session, dmd_number, dat_paths):
    acquisition_prefixes = {
        path.name.split(f"_DMD{dmd_number}-TRIAL", 1)[0].lower()
        for path in dat_paths
    }
    values = []
    for meta_path in raw_session.rglob(f"*DMD{dmd_number}.meta"):
        meta_prefix = meta_path.name.split(f"_DMD{dmd_number}.meta", 1)[0].lower()
        if meta_prefix not in acquisition_prefixes:
            continue
        with h5py.File(meta_path, "r") as meta:
            value = float(np.asarray(
                meta["AcquisitionContainer"]["ParsePlan"]["linesPerCycle"][()]
            ).squeeze())
        values.append((meta_path, value))
    if not values:
        raise ValueError(f"No acquisition DMD{dmd_number} metadata in {raw_session}")
    unique_values = {value for _, value in values}
    if len(unique_values) != 1:
        detail = ", ".join(f"{path.name}={value}" for path, value in values)
        raise ValueError(f"DMD metadata disagree on linesPerCycle: {detail}")
    return values[0][1]


def summary_cycle_count(summary_path, plane_name, lpc):
    with h5py.File(summary_path, "r") as summary:
        frame_info = summary[plane_name]["frame_info"]
        trial_num_frames = np.asarray(frame_info["trial_num_frames"][()]).reshape(-1)
        frame_line_idxs = np.asarray(frame_info["frame_line_idxs"][()]).reshape(-1)
    frame_line_idxs, _ = slap2_synching.normalize_continued_trial_line_indices(
        frame_line_idxs, trial_num_frames, plane_name=plane_name
    )
    boundaries = np.concatenate([[0], np.cumsum(trial_num_frames)])
    return int(sum(
        slap2_synching.get_expected_cycle_count(
            frame_line_idxs[boundaries[index]:boundaries[index + 1]], lpc
        )
        for index in range(len(trial_num_frames))
    ))


def count_session(raw_session, processed_session):
    summary_path = processed_session / "source_extraction" / "experiment_summary.h5"
    with h5py.File(summary_path, "r") as summary:
        plane_name, dmd_number = source_plane(summary)
    dat_paths = list(raw_session.rglob(f"*DMD{dmd_number}-TRIAL*.dat"))
    lpc = lines_per_cycle(raw_session, dmd_number, dat_paths)

    harp_path = next((raw_session / "behavior").glob("*.harp"))
    harp_data = harp_utils.extract_harp(harp_path)
    harp_cycles, _ = slap2_synching.parse_cycle_clock(
        harp_data["slap2_cycle_clock_signal"],
        harp_data["slap2_cycle_clock_times"],
    )

    with ThreadPoolExecutor(max_workers=16) as executor:
        dat_cycles = sum(executor.map(slap2_synching.read_dat_num_cycles, dat_paths))
    subject, date, _ = session_key(raw_session)
    return {
        "session": raw_session.name,
        "label": f"{subject}\n{date}",
        "source_plane": plane_name,
        "dmd_number": dmd_number,
        "lines_per_cycle": lpc,
        "harp_cycles": len(harp_cycles),
        "summary_cycles": summary_cycle_count(summary_path, plane_name, lpc),
        "dat_cycles": dat_cycles,
    }


def write_outputs(rows):
    RESULTS_ROOT.mkdir(exist_ok=True)
    csv_path = RESULTS_ROOT / "p3_cycle_count_comparison.csv"
    fieldnames = [key for key in rows[0] if key != "label"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    x_positions = np.arange(len(rows))
    width = 0.25
    fig, axis = plt.subplots(figsize=(16, 7.5))
    series = (
        ("harp_cycles", "HARP DI3", "#2878B5"),
        ("summary_cycles", "Experiment summary", "#E9A23B"),
        ("dat_cycles", ".dat files", "#2E8B57"),
    )
    for offset, (key, label, color) in zip((-width, 0, width), series):
        values = [row[key] for row in rows]
        bars = axis.bar(x_positions + offset, values, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:,}" for value in values], rotation=90,
                       padding=3, fontsize=7)

    axis.set_title("OpenScope P3 cycle totals by session", fontsize=15, pad=16)
    axis.set_ylabel("Cycle count")
    axis.set_xticks(x_positions, [row["label"] for row in rows], fontsize=8)
    axis.ticklabel_format(axis="y", style="plain")
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    axis.margins(x=0.02)
    fig.tight_layout()
    figure_path = RESULTS_ROOT / "p3_cycle_count_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return csv_path, figure_path


def main():
    rows = [count_session(raw, processed) for raw, processed in pair_sessions()]
    csv_path, figure_path = write_outputs(rows)
    for row in rows:
        print(
            f"{row['session']}: HARP={row['harp_cycles']:,}, "
            f"summary={row['summary_cycles']:,}, dat={row['dat_cycles']:,} "
            f"({row['source_plane']}/DMD{row['dmd_number']})"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()