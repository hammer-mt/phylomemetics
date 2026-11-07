"""Download one model's compact results from the GPU machine into v2/acts/<tag>/ (run from study/).

    ../.venv/bin/python v2/gpu/pull.py <model-tag> [--only] [--primary] [--raw] [--sentences[=prefix]] [--selected] [--franchises] [--file=<path>]

By default: the scales, the uint8 score matrices, the background counts, top shares and presence bits, and the
metadata. --raw adds the three raw top-3 score matrices needed for the v1-rule comparison (float16, large).
--sentences adds the sentence-level activations of the chosen units; --selected the per-plot background shares;
--franchises the comparison franchises; --file any other output. --only skips the default set, and --primary
fetches only what the main experiments read, leaving the files of the sensitivity checks for later.

Every file is checked against its sha256 on the GPU machine, so a pass that was run again there is fetched again;
the hashes of what the analyses used are kept in v2/acts/<tag>/pulled.json.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

BASE = [("study/scale.npy", "scale.npy"), ("study/scale_ref.npy", "scale_ref.npy"), ("study/scale_ref5.npy", "scale_ref5.npy"),
        ("study/scale_raw.npy", "scale_raw.npy"), ("study/scale_source.npy", "scale_source.npy"), ("study/scale_summary.json", "scale_summary.json"),
        ("study/meta.json", "study_meta.json"), ("ref/meta.json", "meta.json"), ("ref_sup/meta.json", "sup_meta.json"), ("background/counts.npy", "bg_counts.npy"),
        ("background/counts_ref_scale.npy", "bg_counts_ref_scale.npy"), ("background/counts_ref5_scale.npy", "bg_counts_ref5_scale.npy"),
        ("background/counts_raw.npy", "bg_counts_raw.npy"), ("background/top_shares.u8.npy", "bg_top_shares.u8.npy"),
        ("background/meta.json", "bg_meta.json"), ("background/present30.bits.npy", "bg_present30.bits.npy")]
SCORES = ["orig_main_top1", "orig_main_top3", "orig_cut_main_top1", "orig_raw_top1", "anon_main_top1", "anon_main_top3", "anon_cut_main_top1",
          "anon_codes_main_top1", "anon_after_main_top1", "anon_icon_main_top1", "anon_icon_main_top3", "anon_raw_top1"]
BASE += [(f"study/{n}.u8.npy", f"{n}.u8.npy") for n in SCORES]
BASE += [(f"study/{n}.{alt}.u8.npy", f"{n}.{alt}.u8.npy") for n in ("orig_main_top1", "anon_main_top1", "anon_icon_main_top1") for alt in ("ref", "ref5")]
RAW = [(f"study/{t}_raw_top3.npy", f"{t}_raw_top3.npy") for t in ("orig", "anon", "anon_icon")]
FRANCHISES = [("franchises/present30.bits.npy", "fr_present30.bits.npy"), ("franchises/meta.json", "fr_meta.json"), ("franchises/counts.npy", "fr_counts.npy")]
def sentence_files(prefix):
    return [(f"study/{prefix}_{v}{x}", f"{prefix}_{v}{x}") for v in ("orig", "anon") for x in (".u8.npy", "_where.npy")] + \
           [(f"study/{prefix}_units.npy", f"{prefix}_units.npy"), (f"study/{prefix}_tokens.json", f"{prefix}_tokens.json")]
SEL = [("background/selected.u8.npy", "bg_selected.u8.npy"), ("background/selected_units.npy", "bg_selected_units.npy")]


def main():
    tag = sys.argv[1]
    files = [] if "--only" in sys.argv else list(BASE)
    if "--primary" in sys.argv:                           # leave out what only the sensitivity checks read (fetch it later without this flag)
        files = [f for f in files if not any(x in f[1] for x in ("_top3.", ".ref.", ".ref5.", "anon_after_", "_raw_top1."))]
    files += RAW if "--raw" in sys.argv else []
    files += FRANCHISES if "--franchises" in sys.argv else []
    for a in sys.argv:
        if a.startswith("--sentences"):
            files += sentence_files(a.split("=")[1] if "=" in a else "sent")
        if a.startswith("--file="):                       # any other file under out/<tag>/, e.g. --file=steer_units.json
            files.append((a.split("=", 1)[1], Path(a.split("=", 1)[1]).name))
    files += SEL if "--selected" in sys.argv else []
    out = Path("v2/acts") / tag
    out.mkdir(parents=True, exist_ok=True)
    shas = rc.remote_sha256([f"out/{tag}/{remote}" for remote, _ in files])
    t0, total = time.time(), 0

    def fetch(pair):
        remote, local = pair
        return local, rc.get(f"out/{tag}/{remote}", out / local, chunk=32 << 20, sha=shas[f"out/{tag}/{remote}"])

    with ThreadPoolExecutor(6) as pool:                    # several files at once: one connection is slow through the proxy
        for local, n in pool.map(fetch, files):
            total += n
            print(f"  {local:36s} {'unchanged' if not n else f'{n / 1e6:8.1f} MB'}  ({total / 1e6 / max(time.time() - t0, 1):.0f} MB/s overall)", flush=True)
    manifest_path = out / "pulled.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
    manifest.update({local: shas[f"out/{tag}/{remote}"] for remote, local in files})
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(f"{tag}: {total / 1e9:.2f} GB in {time.time() - t0:.0f}s; hashes in {manifest_path}")


if __name__ == "__main__":
    main()
