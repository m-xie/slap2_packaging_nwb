from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DatFileInfo:
    path: Path
    acquisition_prefix: str
    acquisition_timestamp: str
    dmd_number: int
    trial_number: int
    cycle_offset: int | None


SLAP2_DAT_PATTERN = re.compile(
    r'^(?P<prefix>.+_(?P<timestamp>\d{8}_\d{6}))_DMD'
    r'(?P<dmd>\d+)-TRIAL(?P<trial>\d+)'
    # Newer acquisitions can split one logical trial into cycle-based chunks.
    r'(?:-CYCLE-(?P<cycle>\d+))?\.dat$',
    re.IGNORECASE,
)


def parse_dat_file(dat_path):
    """Parse legacy and cycle-chunked SLAP2 .dat filenames."""
    match = SLAP2_DAT_PATTERN.match(dat_path.name)
    if match is None:
        raise ValueError(
            f"Unsupported SLAP2 .dat filename format: {dat_path.name}. Expected "
            f"<label>_YYYYMMDD_HHMMSS_DMD<number>-TRIAL<number>.dat or "
            f"<label>_YYYYMMDD_HHMMSS_DMD<number>-TRIAL<number>-"
            f"CYCLE-<offset>.dat."
        )
    cycle = match.group('cycle')
    return DatFileInfo(
        path=dat_path,
        acquisition_prefix=match.group('prefix'),
        acquisition_timestamp=match.group('timestamp'),
        dmd_number=int(match.group('dmd')),
        trial_number=int(match.group('trial')),
        cycle_offset=None if cycle is None else int(cycle),
    )


def validate_dat_files(dat_paths, read_cycle_count):
    """Validate that each retained SLAP2 trial has a coherent .dat layout."""
    groups = {}
    for dat_path in dat_paths:
        dat_file = parse_dat_file(dat_path)
        key = (
            dat_file.acquisition_prefix.lower(),
            dat_file.dmd_number,
            dat_file.trial_number,
        )
        groups.setdefault(key, []).append(dat_file)

    for (prefix, dmd_number, trial_number), dat_files in groups.items():
        chunked = [dat_file.cycle_offset is not None for dat_file in dat_files]
        if not any(chunked):
            # Legacy acquisitions store each DMD/trial pair in one unsuffixed file.
            if len(dat_files) > 1:
                raise ValueError(
                    f"Duplicate unchunked .dat files for {prefix}, DMD{dmd_number}, "
                    f"trial {trial_number}."
                )
            continue
        if not all(chunked):
            raise ValueError(
                f"Mixed chunked and unchunked .dat files for {prefix}, "
                f"DMD{dmd_number}, trial {trial_number}."
            )

        # TRIAL identifies the logical trial; CYCLE identifies each chunk's
        # zero-based starting cycle within that trial, not its cycle count.
        ordered = sorted(dat_files, key=lambda dat_file: dat_file.cycle_offset)
        offsets = [dat_file.cycle_offset for dat_file in ordered]
        if len(set(offsets)) != len(offsets):
            raise ValueError(
                f"Duplicate cycle offsets for {prefix}, DMD{dmd_number}, trial "
                f"{trial_number}: {offsets}."
            )
        if offsets[0] != 0:
            raise ValueError(
                f"Cycle chunks for {prefix}, DMD{dmd_number}, trial {trial_number} "
                f"start at {offsets[0]}, expected 0."
            )

        for dat_file, next_dat_file in zip(ordered, ordered[1:]):
            # Chunk length comes from the binary data, so the next filename's
            # offset must equal this chunk's offset plus its stored cycle count.
            cycle_count = read_cycle_count(dat_file.path)
            expected_next_offset = dat_file.cycle_offset + cycle_count
            if next_dat_file.cycle_offset != expected_next_offset:
                relation = (
                    "overlap"
                    if next_dat_file.cycle_offset < expected_next_offset
                    else "gap"
                )
                raise ValueError(
                    f"SLAP2 cycle-chunk {relation} for {prefix}, DMD{dmd_number}, "
                    f"trial {trial_number}: {dat_file.path.name} starts at "
                    f"{dat_file.cycle_offset} "
                    f"and contains {cycle_count} cycles, so the next chunk should "
                    f"start at {expected_next_offset}, not "
                    f"{next_dat_file.cycle_offset}."
                )