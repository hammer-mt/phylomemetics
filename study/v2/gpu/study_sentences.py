"""Sentence-level activations of chosen units on the study's works (original and name-free).

    python study_sentences.py <model-tag> <study-dir> <out-root> <units.npy> [prefix]

Writes <out-root>/<tag>/study/: <prefix>_{orig,anon}.u8.npy [sentences, units] (share of scale in steps of 0.005,
rounded down) and <prefix>_{orig,anon}_where.npy [sentences, units] (token offset of the peak inside the sentence),
<prefix>_units.npy and <prefix>_tokens.json (each sentence's tokens, so the word a unit fired on can be shown).
The prefix defaults to "sent". In the name-free texts the placeholder codes' own tokens cannot carry the peak, as
in study_pass.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from neurons import Tap, encode_document, geometry, load, now, run_hidden
from study_pass import code_tokens

STEP = 200


def main(tag, study_dir, out_root, units_path, prefix="sent"):
    st = Path(out_root) / tag / "study"
    units = np.load(units_path).astype(np.int64)
    model, tok = load(tag)
    L, D = geometry(model)
    U = L * D
    scale = np.load(st / "scale.npy")[units]
    pole, layer, col = units // U, (units % U) // D, units % D
    at = {int(l): np.where(layer == l)[0] for l in np.unique(layer)}
    tokens = {}
    for ver, name in (("orig", "corpus.jsonl"), ("anon", "corpus_neutral.jsonl")):
        works = [json.loads(l) for l in open(Path(study_dir) / name)]
        n_sent = sum(len(w["sentences"]) for w in works)
        frac = np.zeros((n_sent, len(units)), np.uint8)
        where = np.zeros((n_sent, len(units)), np.int16)
        ctx = {}

        def fn(l, g, h):
            idx = at[l]
            c = torch.tensor(col[idx], device=h.device)
            hh, gg = h[0][:, c].float(), g[0][:, c]
            sign = torch.tensor(np.where(pole[idx] == 0, 1.0, -1.0), device=h.device)
            main = torch.where(gg > 0, hh, torch.zeros_like(hh)) * sign
            if len(ctx["codes"]):
                main[ctx["codes"].to(main.device)] = 0
            sc = torch.tensor(np.where(scale[idx] > 0, scale[idx], np.inf), device=h.device)
            for k, (s, e) in enumerate(ctx["spans"]):
                v, t = main[s:e].max(0)
                frac[ctx["row0"] + k, idx] = torch.clamp(torch.floor(v.clamp(min=0) / sc * STEP + 1e-4), 0, 255).to(torch.uint8).cpu().numpy()
                where[ctx["row0"] + k, idx] = t.cpu().numpy()
            return None

        tap = Tap(model, fn, layers=set(at))
        row0, toks = 0, []
        for w in works:
            ids, spans, text, offsets = encode_document(tok, w["sentences"])
            ctx.update(spans=spans, row0=row0, codes=torch.tensor(code_tokens(text, offsets) if ver == "anon" else [], dtype=torch.long))
            run_hidden(model, torch.tensor([ids]))
            toks += [tok.convert_ids_to_tokens(ids[s:e]) for s, e in spans]
            row0 += len(spans)
        tap.remove()
        np.save(st / f"{prefix}_{ver}.u8.npy", frac)
        np.save(st / f"{prefix}_{ver}_where.npy", where)
        tokens[ver] = toks
        print(f"{now()} {ver}: {n_sent} sentences x {len(units)} units", flush=True)
    np.save(st / f"{prefix}_units.npy", units)
    json.dump(tokens, open(st / f"{prefix}_tokens.json", "w"))


if __name__ == "__main__":
    main(*sys.argv[1:6])
