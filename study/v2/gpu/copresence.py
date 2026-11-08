"""Second corpora (remakes): how many optional units two films share, counted on the GPU machine.

The remakes corpus has thousands of candidate films, too many to bring every film's presence bits home. v2's heredity
rate needs only counts: for a work a and a candidate c,

    rate[a, c] = |optional & present(c) & present(a)| / |optional & present(c)|

where a unit is optional when it is present in at least 2 of the films read and in at most half of them (v2's band,
2 to half of the lineage). This script reads the packed presence planes written by presence_pass.py and writes, for
each plane, the two counts for chosen rows a against every film c:

    python copresence.py <planes-dir> <rows.json> <out-dir> [--planes-list top1_30 ...] [--without over100] [--chunk 131072] [--lo 2] [--hi-share 0.5]

<rows.json> is a list of row numbers of the planes (the remakes). Writes, per plane,
<out-dir>/<plane>.npz with
    shared      int32 [rows, films]   optional units present in both
    n_optional  int32 [films]         optional units present in the film (the denominator for candidate c)
    n_present   int32 [films]         all units present in the film
    band        [lo, hi]              the band in films, and n_units_optional, n_units_present_anywhere
and <out-dir>/<plane>.unit_counts.npy, uint16 [units]: the number of films each unit is present in.
It does not read any text and runs no model: only bit counts. Uses a GPU if there is one.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("planes"); ap.add_argument("rows"); ap.add_argument("out")
    ap.add_argument("--planes-list", nargs="*", default=None, help="plane names; default: every top1_* and top3_* plane in the folder")
    ap.add_argument("--chunk", type=int, default=131072, help="units per step (a multiple of 8)")
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--hi-share", type=float, default=0.5)
    ap.add_argument("--without", default=None, help="a plane (for example over100): units present in it in any film are left out, and the "
                                                    "outputs are named <plane>_without_<this>.npz")
    a = ap.parse_args()
    src, out = Path(a.planes), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = np.array(json.load(open(a.rows)), dtype=np.int64)
    names = a.planes_list or sorted(p.name[:-len(".bits.npy")] for p in src.glob("top*.bits.npy"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert a.chunk % 8 == 0
    drop = None
    if a.without:                                                                  # one bit per unit: set in any film of that plane
        w = np.load(src / f"{a.without}.bits.npy", mmap_mode="r")
        drop = np.zeros(w.shape[1], dtype=np.uint8)
        for r0 in range(0, w.shape[0], 512):
            drop |= np.bitwise_or.reduce(np.ascontiguousarray(w[r0:r0 + 512]), axis=0)
        print(f"{a.without}: {int(np.unpackbits(drop).sum()):,} units are left out", flush=True)
    for name in names:
        bits = np.load(src / f"{name}.bits.npy", mmap_mode="r")                   # [films, 2U / 8]
        F, B = bits.shape
        lo, hi = a.lo, int(np.floor(a.hi_share * F))
        shared = torch.zeros(len(rows), F, dtype=torch.float64, device=dev)       # float64 holds whole numbers exactly far past these counts
        n_opt = torch.zeros(F, dtype=torch.float64, device=dev)
        n_pres = torch.zeros(F, dtype=torch.float64, device=dev)
        n_units_opt = n_units_any = 0
        unit_counts = np.zeros(B * 8, dtype=np.uint16)                            # films each unit is present in (for checking the band elsewhere)
        t0 = time.time()
        for b0 in range(0, B, a.chunk // 8):
            block = np.ascontiguousarray(bits[:, b0:b0 + a.chunk // 8])
            if drop is not None:
                block &= ~drop[None, b0:b0 + a.chunk // 8]
            P = torch.from_numpy(np.unpackbits(block, axis=1)).to(dev)             # [films, units in this step] 0/1 uint8
            nb = P.sum(0, dtype=torch.int64)
            unit_counts[b0 * 8:b0 * 8 + nb.shape[0]] = nb.clamp(max=65535).cpu().numpy().astype(np.uint16)
            n_pres += P.sum(1, dtype=torch.float64)
            opt = (nb >= lo) & (nb <= hi)
            n_units_any += int((nb > 0).sum())
            n_units_opt += int(opt.sum())
            if not bool(opt.any()):
                continue
            Q = P[:, opt].to(torch.float32)                                        # float32 products of 0/1 sum exactly below 2**24 per step
            n_opt += Q.sum(1, dtype=torch.float64)
            shared += (Q[torch.from_numpy(rows).to(dev)] @ Q.T).to(torch.float64)
            del P, Q
        name = f"{name}_without_{a.without}" if a.without else name
        np.savez(out / f"{name}.npz", shared=shared.round().to(torch.int32).cpu().numpy(), n_optional=n_opt.round().to(torch.int32).cpu().numpy(),
                 n_present=n_pres.round().to(torch.int32).cpu().numpy(), rows=rows, band=np.array([lo, hi]),
                 n_units_optional=n_units_opt, n_units_present_anywhere=n_units_any, n_films=F)
        np.save(out / f"{name}.unit_counts.npy", unit_counts)
        print(f"{time.strftime('%H:%M:%S')} {name}: {F} films, {len(rows)} rows, {n_units_opt:,} optional of {n_units_any:,} units present anywhere, band {lo} to {hi}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
