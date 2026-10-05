"""Read logical SLAP2 path dimensions without loading metadata or line indices."""

from importlib import import_module
from numbers import Real
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from slap2_dat_utils import parse_dat_file, validate_dat_files


# DataFile.MAGIC_NUMBER in the Docker-pinned SLAP2_Utils commit
# d5a7cbc1476c52dd57d7530935776ca9b7bc26d5.
_MAGIC_NUMBER = np.uint32(322379495)
_SOURCE = "slap2_utils.utils.file_header.load_file_header_v2"


def _load_file_header_v2(obj, raw_uint32):
    """Lazy, mockable boundary to the same parser used internally by DataFile."""
    try:
        load_file_header_v2 = import_module(
            "slap2_utils.utils.file_header"
        ).load_file_header_v2
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Reading SLAP2 path metadata requires "
            "slap2_utils.utils.file_header.load_file_header_v2. Use the capsule "
            "environment with the SLAP2_Utils revision pinned in "
            "environment/Dockerfile (d5a7cbc1476c52dd57d7530935776ca9b7bc26d5) "
            "and its dependencies."
        ) from exc
    return load_file_header_v2(obj, raw_uint32)


def _integer(value, name, minimum):
    """Accept integral library floats as well as Python/NumPy integers."""
    message = f"{name} must be an integer >= {minimum}; got {value!r}."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        result = int(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if result != value or result < minimum:
        raise ValueError(message)
    return result


def _validate_preamble(raw_uint32):
    """Guard the library's unchecked indexing and division, not parse metadata."""
    if len(raw_uint32) < 4:
        raise ValueError("Truncated v2 header: expected at least 16 bytes.")
    if raw_uint32[0] != _MAGIC_NUMBER:
        raise ValueError("Invalid SLAP2 magic number.")
    if raw_uint32[1] != 2:
        raise ValueError(f"Unsupported SLAP2 file version {raw_uint32[1]}; expected 2.")
    header_bytes = int(raw_uint32[2])
    if header_bytes < 16 or header_bytes % 4:
        raise ValueError("Invalid v2 header size: must be >= 16 and divisible by 4.")
    if header_bytes > raw_uint32.nbytes:
        raise ValueError("Truncated v2 header: declared header size exceeds file size.")
    header_words = header_bytes // 4
    if (header_words - 4) % 2:
        raise ValueError("Invalid v2 header size: incomplete field/value pair.")
    if raw_uint32[header_words - 1] != _MAGIC_NUMBER:
        raise ValueError("Invalid v2 header end magic number.")

    # Only inspect the two fields needed to protect the library's cycle-count
    # arithmetic. Field IDs 0 and 3 are defined by file_header.py at the pinned
    # revision. All metadata translation and cycle counting remain in the library.
    safety_fields = {}
    for index in range(3, header_words - 1, 2):
        field_id = int(raw_uint32[index])
        if field_id in (0, 3):
            safety_fields[field_id] = int(raw_uint32[index + 1])
    if 3 not in safety_fields or safety_fields[3] == 0:
        raise ValueError("Missing or zero bytesPerCycle in v2 header.")
    first_offset = safety_fields.get(0)
    if (
        first_offset is None
        or first_offset < header_bytes
        or first_offset > raw_uint32.nbytes
        or first_offset % 2
    ):
        raise ValueError(
            "Invalid firstCycleOffsetBytes: must be present, aligned to 2 bytes, "
            "and between the header end and the uint32-mapped file end."
        )


def _read_chunk_metadata(path):
    size = path.stat().st_size
    if size < 16:
        raise ValueError(f"Invalid SLAP2 .dat file {path}: empty or truncated v2 header.")
    # DataFile forms a uint32 view of floor(nbytes / 4) words. An explicit
    # shape preserves that behavior when a file ends with two extra bytes.
    raw_uint32 = np.memmap(path, dtype=np.uint32, mode="r", shape=(size // 4,))
    try:
        _validate_preamble(raw_uint32)
        header, num_cycles = _load_file_header_v2(
            SimpleNamespace(MAGIC_NUMBER=_MAGIC_NUMBER), raw_uint32
        )
        return (
            _integer(header["linesPerCycle"], "linesPerCycle", 1),
            _integer(num_cycles, "numCycles", 0),
        )
    except (AssertionError, KeyError, IndexError, TypeError, ValueError,
            ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Invalid SLAP2 .dat file {path}: {exc}") from exc
    finally:
        # Only scalar metadata escapes this function, so the mapping can close
        # immediately even on parser failures or when processing many chunks.
        raw_uint32._mmap.close()


def read_path_metadata(dat_paths):
    """Return dimensions for one acquisition-prefix/DMD/trial logical path.

    ``dat_paths`` is a nonempty iterable of paths, in any order. A legacy path
    has one unsuffixed file; chunked paths must begin at cycle zero and be
    contiguous according to the library's actual complete-cycle counts. Only
    the final chunk may contain zero complete cycles, and the aggregate must
    contain at least one complete cycle. Incomplete trailing cycles are not
    counted, matching the library parser.

    Returns a dict with integer ``lines_per_cycle``, ``total_cycles``,
    ``total_lines``, ``chunk_count`` and a ``source`` naming the library parser.
    No .meta file, DataFile instance, or per-line index arrays are needed.
    """
    if dat_paths is None or isinstance(dat_paths, (str, bytes, Path)):
        raise ValueError("dat_paths must be a nonempty iterable of .dat paths.")
    paths = [Path(path) for path in dat_paths]
    if not paths:
        raise ValueError("dat_paths must be a nonempty iterable of .dat paths.")
    files = [parse_dat_file(path) for path in paths]
    identities = {
        (file.acquisition_prefix.lower(), file.dmd_number, file.trial_number)
        for file in files
    }
    if len(identities) != 1:
        raise ValueError("dat_paths must describe a single acquisition prefix, DMD, and trial.")

    # Cache each result once; validate_dat_files must use library counts, never
    # infer them from the following chunk's filename.
    metadata = {}
    for path in paths:
        if path not in metadata:
            metadata[path] = _read_chunk_metadata(path)
    validate_dat_files(paths, lambda path: metadata[path][1])
    line_counts = {value[0] for value in metadata.values()}
    if len(line_counts) != 1:
        raise ValueError("All SLAP2 chunks must have the same positive integer linesPerCycle.")
    ordered = sorted(files, key=lambda file: file.cycle_offset or 0)
    for file in ordered[:-1]:
        if metadata[file.path][1] == 0:
            raise ValueError("Only the final SLAP2 chunk may contain zero complete cycles.")
    total_cycles = sum(value[1] for value in metadata.values())
    if total_cycles <= 0:
        raise ValueError("SLAP2 path total_cycles must be > 0 (no complete cycles found).")
    lines_per_cycle = line_counts.pop()
    return {
        "lines_per_cycle": lines_per_cycle,
        "total_cycles": total_cycles,
        "total_lines": lines_per_cycle * total_cycles,
        "source": _SOURCE,
        "chunk_count": len(paths),
    }