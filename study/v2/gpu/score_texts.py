"""Score any texts on chosen units: explanation tests, rewordings, published examples.

    python score_texts.py <model-tag> <texts.jsonl> <units.npy> <out.npz> [--batch-tokens 8000] [--no-lead-in]

texts.jsonl: one {"id": ..., "text": ...} per line. units.npy: unit indices (positive pole = layer * width +
neuron; negative pole = that + n_neurons). For every text and unit the output holds
  peak   the strongest main-regime activation in the text, as a magnitude (position 0 excluded)
  where  the token position of that peak, and where_end, the character offset in the text just after that token
         (text[:where_end] is the text up to and including the peak token)
  rest   the same peak with the text's first word left out too (a check: after a lead-in token the first word is in
         a position the reference pass never scored)
  last   the raw activation at the text's last token (signed, all sign regimes), for checks against
         activations published with a marked peak token
and tokens.json lists each text's tokens so a peak can be read back as a word.

Position 0 is never scored. So that a text's first word is not lost, tokenizers that add no begin-of-text token
(Qwen) get one lead-in token before the text (neurons.lead_in); --no-lead-in keeps the text as it is, for contexts
that begin mid-document. By default each text is read on its own, unpadded, as the study works are.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, geometry, lead_in, load, now, run_hidden
from ref_pass import batches_by_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("texts"); ap.add_argument("units"); ap.add_argument("out")
    ap.add_argument("--batch-tokens", type=int, default=0, help="0: one text at a time (the default)")
    ap.add_argument("--no-lead-in", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    texts = [json.loads(l) for l in open(a.texts)]
    units = np.load(a.units).astype(np.int64)
    model, tok = load(a.tag)
    L, D = geometry(model)
    U = L * D
    pole, layer, col = units // U, (units % U) // D, units % D
    at = {l: np.where(layer == l)[0] for l in np.unique(layer)}
    lead = [] if a.no_lead_in else lead_in(tok)
    enc = tok([t["text"] for t in texts], add_special_tokens=True, truncation=True, max_length=a.max_tokens - len(lead), return_offsets_mapping=True)
    ids_all = [lead + x for x in enc["input_ids"]]
    ends = [np.array([0] * len(lead) + [b for _, b in off], dtype=np.int32) for off in enc["offset_mapping"]]
    lens = np.array([len(x) for x in ids_all])
    peak = np.zeros((len(texts), len(units)), np.float32)
    where = np.zeros((len(texts), len(units)), np.int16)
    last = np.zeros((len(texts), len(units)), np.float32)
    rest = np.zeros((len(texts), len(units)), np.float32)
    ctx = {}

    def fn(l, g, h):
        idx = at[l]
        c = torch.tensor(col[idx], device=h.device)
        hh, gg = h[:, :, c].float(), g[:, :, c]
        sign = torch.tensor(np.where(pole[idx] == 0, 1.0, -1.0), device=h.device)
        main = torch.where(gg > 0, hh, torch.zeros_like(hh)) * sign * ctx["mask"].to(h.device)       # magnitudes on each unit's own pole
        v, t = main.max(1)
        rows = ctx["rows"]
        peak[np.ix_(rows, idx)] = v.clamp(min=0).cpu().numpy()
        rest[np.ix_(rows, idx)] = main[:, len(lead) + 1:].amax(1).clamp(min=0).cpu().numpy() if main.shape[1] > len(lead) + 1 else 0
        where[np.ix_(rows, idx)] = t.cpu().numpy()
        end = ctx["end"].to(h.device)
        last[np.ix_(rows, idx)] = hh[torch.arange(hh.shape[0], device=h.device), end].cpu().numpy()
        return None

    tap = Tap(model, fn, layers=set(int(l) for l in at))
    for b in (batches_by_length(lens, a.batch_tokens) if a.batch_tokens else [[i] for i in range(len(texts))]):
        width = int(lens[b].max())
        ids = torch.full((len(b), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(b), width), dtype=torch.long)
        for r, j in enumerate(b):
            ids[r, :lens[j]] = torch.tensor(ids_all[j])
            att[r, :lens[j]] = 1
        m = att.clone()
        m[:, 0] = 0
        ctx.update(mask=m[:, :, None].float(), rows=np.array(b), end=torch.tensor(lens[b] - 1))
        run_hidden(model, ids, att if len(b) > 1 else None)
    tap.remove()
    where_end = np.stack([ends[r][where[r]] for r in range(len(texts))]) if len(texts) else np.zeros_like(where, dtype=np.int32)
    np.savez(a.out, peak=peak, rest=rest, where=where, where_end=where_end, last=last, units=units, ids=np.array([str(t["id"]) for t in texts]), lead_in_tokens=len(lead))
    json.dump([tok.convert_ids_to_tokens(x) for x in ids_all], open(str(Path(a.out).with_suffix("")) + ".tokens.json", "w"))
    print(f"{now()} scored {len(texts)} texts on {len(units)} units -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
