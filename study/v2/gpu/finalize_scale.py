"""After passes 1 and 2: fix each unit's scale and express the study scores as shares of it.

scale = the strongest main-regime activation of the unit anywhere in the reference corpus, the scale supplement or
the study's original texts ("the neuron's strongest example in the corpus"). Kept beside it: the reference-only
scale (reference corpus and supplement, no study text), the fifth strongest of the per-batch reference maxima (a
scale that one freak token cannot set), and the scale of the raw product. Five reference documents turned out to be
study works under older titles (OVERLAP below). They count toward the scale, which includes the study's texts
anyway, but not toward the reference-only scale or the fifth example.

Scores are stored as uint8 shares of the scale in steps of 0.005, rounded DOWN (60 means at least 30% of the unit's
maximum, never 29.9%), capped at 1.275. The three full-text strongest-sentence scores are also stored against the
reference-only scale and the fifth-example scale (<name>.ref.u8.npy, <name>.ref5.u8.npy).

    python finalize_scale.py <model-tag> <out-root> [doc doc ...]      (documents to leave out; default OVERLAP)

Writes <out-root>/<tag>/ref_all/top_{pos,neg}_{v,i}.npy (the merged lists of strongest examples) and
<out-root>/<tag>/study/: scale.npy, scale_ref.npy, scale_ref5.npy, scale_raw.npy [2 * neurons], scale_source.npy
(0 reference corpus, 1 supplement, 2 study texts), the .u8.npy score matrices and scale_summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

STEP = 200          # uint8 units per 1.0 of scale
MAIN = ["orig_main_top1", "orig_main_top3", "orig_cut_main_top1", "anon_main_top1", "anon_main_top3", "anon_cut_main_top1",
        "anon_codes_main_top1", "anon_after_main_top1", "anon_icon_main_top1", "anon_icon_main_top3"]
ALT = ["orig_main_top1", "anon_main_top1", "anon_icon_main_top1"]          # also stored against the two alternative scales
RAW = ["orig_raw_top1", "anon_raw_top1"]
CHUNK = 1 << 18
OVERLAP = [843, 7175, 6895, 3597, 5284]        # The Princess Bride, The Dark Knight, The Notebook, The Wizard of Oz, The Exorcist
FIRST_SUPPLEMENT_DOC = 46000


def column_max(path):
    a = np.load(path, mmap_mode="r")
    out = np.zeros(a.shape[1], np.float32)
    for c in range(0, a.shape[1], CHUNK):
        out[c:c + CHUNK] = a[:, c:c + CHUNK].astype(np.float32).max(0)
    return out


def to_shares(src, dst, scale):
    """floor(score / scale * STEP) as uint8, so that value >= 60 means score >= 0.30 * scale exactly."""
    S = np.load(src, mmap_mode="r")
    safe = np.where(scale > 0, scale, np.inf)
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.uint8, shape=S.shape)
    for c in range(0, S.shape[1], CHUNK):
        share = S[:, c:c + CHUNK].astype(np.float32) / safe[c:c + CHUNK]
        out[:, c:c + CHUNK] = np.clip(np.floor(share * STEP + 1e-4), 0, 255).astype(np.uint8)
    out.flush()
    return out


def merge_tops(root):
    """Merge the strongest examples of the reference corpus and of the supplement (if it was run) into ref_all/."""
    folders = [root / "ref"] + ([root / "ref_sup"] if (root / "ref_sup" / "top_pos_v.npy").exists() else [])
    out = root / "ref_all"
    out.mkdir(exist_ok=True)
    for pole in ("pos", "neg"):
        V = [np.load(f / f"top_{pole}_v.npy", mmap_mode="r") for f in folders]
        I = [np.load(f / f"top_{pole}_i.npy", mmap_mode="r") for f in folders]
        K = V[0].shape[1]
        mv = np.lib.format.open_memmap(out / f"top_{pole}_v.npy", mode="w+", dtype=np.float32, shape=V[0].shape)
        mi = np.lib.format.open_memmap(out / f"top_{pole}_i.npy", mode="w+", dtype=np.int32, shape=V[0].shape)
        for l in range(V[0].shape[0]):
            v = np.concatenate([np.asarray(x[l], dtype=np.float32) for x in V], 0)
            i = np.concatenate([np.asarray(x[l]) for x in I], 0)
            order = np.argsort(-v, axis=0, kind="stable")[:K]
            mv[l], mi[l] = np.take_along_axis(v, order, 0), np.take_along_axis(i, order, 0)
        mv.flush(); mi.flush()
    raw_pos = np.maximum.reduce([np.load(f / "raw_pos_max.npy") for f in folders])
    raw_neg = np.minimum.reduce([np.load(f / "raw_neg_min.npy") for f in folders])
    return out, raw_pos.reshape(-1), -raw_neg.reshape(-1), len(folders) > 1


def reference_tops(ref, exclude):
    """Per unit: the strongest example, the document it is in, and the strongest and fifth strongest outside the excluded documents."""
    full, doc, first, fifth = [], [], [], []
    for pole in ("pos", "neg"):
        V, I = np.load(ref / f"top_{pole}_v.npy", mmap_mode="r"), np.load(ref / f"top_{pole}_i.npy", mmap_mode="r")      # [layers, K, width]
        for l in range(V.shape[0]):
            v, loc = np.asarray(V[l], dtype=np.float32), np.asarray(I[l]).astype(np.int64)
            kept = -np.sort(-np.where(np.isin(loc // 2048, exclude), 0, v), axis=0)          # the lists are stored strongest first
            full.append(v[0]); doc.append(loc[0] // 2048); first.append(kept[0]); fifth.append(kept[4])
    return np.concatenate(full), np.concatenate(doc), np.concatenate(first), np.concatenate(fifth)


def main(tag, out_root, exclude=OVERLAP):
    root = Path(out_root) / tag
    st = root / "study"
    ref, raw_pos, raw_neg, with_supplement = merge_tops(root)
    ref_all, top_doc, ref_scale, ref5 = reference_tops(ref, exclude)
    study_max = column_max(st / "orig_main_top1.npy")
    assert np.isfinite(study_max).all() and np.isfinite(ref_all).all(), "a score is not finite (float16 overflow?)"
    scale = np.maximum(ref_all, study_max)
    raw_ref = np.concatenate([np.maximum(raw_pos, 0), np.maximum(raw_neg, 0)]).astype(np.float32)
    scale_raw = np.maximum(raw_ref, column_max(st / "orig_raw_top1.npy"))
    source = np.where(study_max > ref_all, 2, np.where(top_doc >= FIRST_SUPPLEMENT_DOC, 1, 0)).astype(np.uint8)
    np.save(st / "scale.npy", scale)
    np.save(st / "scale_ref.npy", ref_scale)
    np.save(st / "scale_ref5.npy", ref5)
    np.save(st / "scale_raw.npy", scale_raw)
    np.save(st / "scale_source.npy", source)
    U = len(scale) // 2
    other = np.concatenate([scale[U:], scale[:U]])
    summary = {"n_units": int(len(scale)), "supplement_included": bool(with_supplement),
               "documents_left_out_of_the_reference_only_scale": [int(x) for x in exclude],
               "units_whose_strongest_reference_example_is_in_a_left_out_document": int((ref_all > ref_scale).sum()),
               "units_never_active_in_reference": int((ref_scale <= 0).sum()),
               "units_with_zero_scale": int((scale <= 0).sum()),
               "units_whose_scale_comes_from_the_reference_corpus": int((source == 0).sum()),
               "units_whose_scale_comes_from_the_supplement": int((source == 1).sum()),
               "units_whose_scale_comes_from_the_study_corpus": int((source == 2).sum()),
               "units_study_max_over_twice_reference": int((study_max > 2 * ref_scale).sum()),
               "positive_units_whose_strongest_raw_activation_is_an_echo": int((raw_pos > 1.001 * ref_all[:U]).sum()),
               "negative_units_whose_strongest_raw_activation_is_an_echo": int((raw_neg > 1.001 * ref_all[U:]).sum()),
               "units_whose_fifth_reference_example_is_under_half_the_first": int(((ref5 < 0.5 * ref_scale) & (ref_scale > 0)).sum()),
               "poles_under_a_tenth_of_the_neurons_other_pole": int(((scale < 0.1 * other) & (scale > 0)).sum()),
               "poles_under_a_quarter_of_the_neurons_other_pole": int(((scale < 0.25 * other) & (scale > 0)).sum()),
               "largest_scale": float(scale.max()), "scale_quantiles": {str(q): float(np.quantile(scale, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}}
    for name in MAIN:
        out = to_shares(st / f"{name}.npy", st / f"{name}.u8.npy", scale)
        summary[f"{name}_share_of_cells_at_or_above_30pct"] = float((np.asarray(out[:, ::97]) >= 0.3 * STEP).mean())
    for name in ALT:
        to_shares(st / f"{name}.npy", st / f"{name}.ref.u8.npy", ref_scale)
        to_shares(st / f"{name}.npy", st / f"{name}.ref5.u8.npy", ref5)
    for name in RAW:
        to_shares(st / f"{name}.npy", st / f"{name}.u8.npy", scale_raw)
    json.dump(summary, open(st / "scale_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or OVERLAP)
