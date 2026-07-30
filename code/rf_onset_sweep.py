"""
Sweep the receptive-field QC over a range of onset delays.

Runs slap2_receptive_fields_qc over every onset delay from -3.0 s to +3.0 s in
100 ms steps, writing each result into its own subfolder named after the delay:

    ../results/qc/rf_onset_sweep/onset_+0p000/receptive_fields/...
    ../results/qc/rf_onset_sweep/onset_-0p300/receptive_fields/...

The NWB file is opened once and reused across all delays for speed.
"""
import numpy as np
from pathlib import Path
import pynwb
import hdmf_zarr
import slap2_receptive_fields_qc as rf


def main():
    results = Path('../results')
    qc_root = results / 'qc' / 'rf_onset_sweep'
    qc_root.mkdir(parents=True, exist_ok=True)

    nwb_path = next(
        p for p in results.iterdir()
        if p.name.endswith('.nwb') or p.name.endswith('.nwb.zarr')
    )
    print('Using NWB:', nwb_path, flush=True)
    io_class = hdmf_zarr.NWBZarrIO if nwb_path.is_dir() else pynwb.NWBHDF5IO

    delays = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.1), 3)

    with io_class(str(nwb_path), mode='r') as io:
        nwbfile = io.read()
        for d in delays:
            sign = '+' if d >= 0 else '-'
            name = f'onset_{sign}{abs(d):.3f}'.replace('.', 'p')
            sub = qc_root / name
            rf_folder = sub / 'receptive_fields'
            if len(list(rf_folder.glob('rf_mapping_*_rfs.png'))) == 4:
                print(f'=== onset_delay={d:+.3f}s already complete; skipping ===', flush=True)
                continue
            rf_folder.mkdir(parents=True, exist_ok=True)
            print(f'=== onset_delay={d:+.3f}s -> {sub} ===', flush=True)
            rf._compute_receptive_field_qc(rf_folder, nwbfile, float(d))

    print('DONE. Sweep output under', qc_root, flush=True)


if __name__ == '__main__':
    main()
