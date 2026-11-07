"""Pass 3: how often does each unit fire strongly in an ordinary story?

Runs the background film plots (reference documents with src == "bgplot", each read whole like a study work) and,
for every unit, counts the plots in which its strongest activation reaches 20%, 30%, 40% and 50% of its scale.
The counts are taken four ways: main regime on the unit's scale (the one the analyses use), on the reference-only
scale and on the fifth-example scale, and the raw product on its own scale (to show what leaving the echo regime in
would do). Each plot is read on its own, unpadded, exactly as a study work is: in bfloat16 a padded batch gives
slightly different activations, and the base rate has to be measured the way presence is.
It also keeps, for the main regime,

  present30.bits.npy   which units are present (30% of scale) in which plot, packed 8 units to a byte, so that
                       sets of plots (a genre, say) can play the franchise and the rest give the base rate
  top_shares.u8.npy    every unit's 160 highest per-plot shares (the top 5.3% of plots)
  selected.u8.npy      with --select, each selected unit's share of scale in every plot

    python background_pass.py <model-tag> <reference.jsonl> <out-root> [--select units.npy] [--exclude-docs 843]
    python background_pass.py <model-tag> <franchises.jsonl> <out-root> --src franchise --name franchises

The second form reads the comparison franchises' films (v2/build_franchises.py) the same way.

Shares are in steps of 0.005, rounded down. Writes to <out-root>/<tag>/background/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, encode_document, geometry, load, now, provenance, run_hidden, save_json
from ref_pass import batches_by_length

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
STEP = 200
KEEP = 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("reference"); ap.add_argument("out")
    ap.add_argument("--select", default=None)
    ap.add_argument("--exclude-docs", type=int, nargs="*", default=[], help="reference document indices to leave out (a study work under another title)")
    ap.add_argument("--batch-tokens", type=int, default=0, help="0: one plot at a time, as the study works are read (the default)")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--no-bits", action="store_true")
    ap.add_argument("--src", default="bgplot", help="which documents of the file to read (franchise: the comparison franchises' films)")
    ap.add_argument("--name", default="background", help="output folder under <out-root>/<tag>/")
    a = ap.parse_args()
    root = Path(a.out) / a.tag
    out = root / a.name
    out.mkdir(parents=True, exist_ok=True)
    docs = [d for d in map(json.loads, open(a.reference)) if d["src"] == a.src and d["i"] not in set(a.exclude_docs)][: a.limit_docs]
    model, tok = load(a.tag)
    L, D = geometry(model)
    U = L * D

    def scales(name):
        s = torch.tensor(np.load(root / "study" / name)).reshape(2, L, D)
        return torch.where(s > 0, s, torch.full_like(s, float("inf")))

    def steps(share):
        return torch.clamp(torch.floor(share * STEP + 1e-4), 0, 255)

    safe = {"main": scales("scale.npy"), "ref": scales("scale_ref.npy"), "ref5": scales("scale_ref5.npy"), "raw": scales("scale_raw.npy")}
    counts = {k: torch.zeros(len(THRESHOLDS), 2, L, D, dtype=torch.int32) for k in safe}
    sel = np.load(a.select).astype(np.int64) if a.select else np.zeros(0, np.int64)
    sel_pole, sel_layer, sel_col = sel // U, (sel % U) // D, sel % D
    sel_at = {(p, l): np.where((sel_pole == p) & (sel_layer == l))[0] for p in (0, 1) for l in range(L)}
    dense = np.zeros((len(docs), len(sel)), dtype=np.uint8)
    bits = None if a.no_bits else np.zeros((len(docs), 2 * U // 8), dtype=np.uint8)
    devs = [l.mlp.gate_proj.weight.device for l in model.model.layers]
    tops = [[torch.zeros(KEEP, D, dtype=torch.float16, device=d) for d in devs] for _ in (0, 1)]
    ids_all = [encode_document(tok, d["sentences"])[0] for d in docs]
    lens = np.array([len(x) for x in ids_all])
    batches = batches_by_length(lens, a.batch_tokens) if a.batch_tokens else [[i] for i in range(len(docs))]
    print(f"{now()} {a.tag}: {len(docs)} {a.src} plots, {int(lens.sum()):,} tokens, {len(sel):,} selected units", flush=True)
    ctx = {}

    def fn(layer, g, h):
        h = h * ctx["mask"].to(h.device)
        main = torch.where(g > 0, h, torch.zeros_like(h))
        rows = ctx["rows"]
        for pole in (0, 1):
            peak = (main.amax(1) if pole == 0 else -main.amin(1)).float()                 # [B, D] strongest activation in each plot
            raw = (h.amax(1) if pole == 0 else -h.amin(1)).float()
            # every share is put on the storage grid first (steps of 0.005, rounded down), so that counts, bits, top
            # shares and the study scores all mean the same thing by "at least 30%"
            q = {"main": steps(peak / safe["main"][pole, layer].to(h.device)), "ref": steps(peak / safe["ref"][pole, layer].to(h.device)),
                 "ref5": steps(peak / safe["ref5"][pole, layer].to(h.device)), "raw": steps(raw / safe["raw"][pole, layer].to(h.device))}
            for name in q:
                for k, t in enumerate(THRESHOLDS):
                    counts[name][k, pole, layer] += (q[name] >= round(t * STEP)).sum(0).to(torch.int32).cpu()
            qm = q["main"]
            tops[pole][layer] = torch.cat([tops[pole][layer], qm.to(torch.float16)], 0).topk(KEEP, dim=0).values      # whole numbers up to 255: exact
            if bits is not None:
                lo = (pole * U + layer * D) // 8
                bits[rows, lo:lo + D // 8] = np.packbits((qm >= round(0.3 * STEP)).cpu().numpy(), axis=1)
            idx = sel_at[(pole, layer)]
            if len(idx):
                dense[np.ix_(rows, idx)] = qm[:, sel_col[idx]].to(torch.uint8).cpu().numpy()
        return None

    assert D % 8 == 0
    tap = Tap(model, fn)
    t0, n_tok = time.time(), 0
    for bi, b in enumerate(batches):
        width = int(lens[b].max())
        ids = torch.full((len(b), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(b), width), dtype=torch.long)
        for r, j in enumerate(b):
            ids[r, :lens[j]] = torch.tensor(ids_all[j])
            att[r, :lens[j]] = 1
        m = att.clone()
        m[:, 0] = 0
        ctx.update(mask=m[:, :, None].to(model.dtype), rows=np.array(b))
        run_hidden(model, ids, att if len(b) > 1 else None)          # a single plot: no mask, the same call as the study pass
        n_tok += int(lens[b].sum())
        if bi % 200 == 0 or bi == len(batches) - 1:
            print(f"{now()} batch {bi + 1}/{len(batches)}  {n_tok / (time.time() - t0):,.0f} tok/s", flush=True)
    tap.remove()
    np.save(out / "counts.npy", counts["main"].reshape(len(THRESHOLDS), 2 * U).numpy())
    np.save(out / "counts_ref_scale.npy", counts["ref"].reshape(len(THRESHOLDS), 2 * U).numpy())
    np.save(out / "counts_ref5_scale.npy", counts["ref5"].reshape(len(THRESHOLDS), 2 * U).numpy())
    np.save(out / "counts_raw.npy", counts["raw"].reshape(len(THRESHOLDS), 2 * U).numpy())
    top = torch.cat([torch.cat([t.float().cpu() for t in tops[pole]], 1) for pole in (0, 1)], 1)          # [KEEP, 2U]
    np.save(out / "top_shares.u8.npy", top.to(torch.uint8).numpy())
    if bits is not None:
        np.save(out / "present30.bits.npy", bits)
    if len(sel):
        np.save(out / "selected.u8.npy", dense)
        np.save(out / "selected_units.npy", sel)
    save_json({"tag": a.tag, "provenance": provenance(model), "one_plot_at_a_time": not a.batch_tokens, "n_plots": len(docs), "thresholds": THRESHOLDS, "doc_index": [d["i"] for d in docs], "excluded_docs": a.exclude_docs,
               "titles": [d["title"] for d in docs], "franchise": [d.get("franchise") for d in docs], "years": [d.get("year") for d in docs], "genres": [d.get("genre") for d in docs],
               "n_sentences": [len(d["sentences"]) for d in docs], "n_tokens": int(lens.sum()), "seconds": round(time.time() - t0)}, out / "meta.json")
    print(f"{now()} saved to {out}", flush=True)


if __name__ == "__main__":
    main()
