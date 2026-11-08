"""Checks presence_pass.py bit for bit, on the tiny model on CPU (float32): against a slow, obvious re-computation
of every plane, and against the v2 pipeline's own code (study_pass.py's float16 scores, put on the storage grid by
finalize_scale.to_shares and thresholded as analysis/common.py does).

    cd study && bash v2/heavy.sh ../.venv/bin/python v2/gpu/test_presence_pass.py [work ids, default WORKS]

It also reads a file whose records have neither "cut" nor "drop" (the remakes' kind) into a folder that already holds
another run, and hands presence_pass.py four bad files, which it must turn down before it loads the model.

The model reads only Batman texts (data/corpus.jsonl, data/corpus_neutral.jsonl) and made-up sentences. Outputs go
to v2/cache/test_presence/ (the float16 score files are deleted at the end). Every check prints PASS or FAIL with
the number of differing bits; the script exits non-zero if any fails. About 15 minutes and 5 GB on a busy 4-core machine.
"""
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")      # set before torch is imported, and inherited by the scripts run below:
import numpy as np                                 # on a busy machine four threads are ten times slower than two
import torch

from finalize_scale import to_shares
from neurons import encode_document, layers_of, load
from study_pass import ICON, cut_length

TAG, HERE = "qwen3-0.6b", Path(__file__).resolve().parent
STUDY = HERE.parents[1]
OUT = STUDY / "v2" / "cache" / "test_presence"
# Zorro, the 1943 serial, Lost in Translation: the opening is the whole text (9, 13, 20 sentences). Green Hornet: the
# opening ends at the 20-sentence limit. Gotham by Gaslight, Arkham Asylum (comic), Batman & Robin, Arkham Knight: it
# ends at the 400-word limit (19 of 23, 17 of 24, 15 of 29, 9 of 40 sentences). The name-free texts of the last five
# hold 74, 74, 47, 116 and 149 codes (the most of any work) and lose 0, 1, 9, 0 and 0 iconography sentences; the
# originals lose 0, 16, 14, 20 and 36. No work has an opening under 3 sentences (the shortest is 9): a made-up text has.
WORKS = [37, 0, 111, 44, 6, 7, 14, 25]
AGAIN = ", then down again and up again" * 65       # makes a sentence of 399 words: the opening (at most 400 words) stops after it
MADE_UP = [      # one sentence: dropped as iconography, which leaves nothing, so it is kept. Two: the first is the whole opening, and the one dropped
    ("one-sentence", ["Bruce Wayne waited in the cave under Gotham until the rain stopped."], ["X1 waited in the cave under X2 until the rain stopped."]),
    ("two-sentences", ["Selina Kyle climbed to the cave with a rope" + AGAIN + ".", "Dick saw the lamp from the harbour and told Alfred that she was gone."],
     ["X1 climbed to the cave with a rope" + AGAIN + ".", "X2 saw the lamp from the harbour and told X3 that X1 was gone."])]
PARAGRAPHS = [   # made up; the test scale starts from each unit's strongest sentence in them
    ["The ferry left the island at dawn with nine passengers and a crate of oranges.", "A woman in a green coat asked the captain if the storm would reach them by noon.",
     "He laughed, checked the radio and told her to keep an eye on the gulls."],
    ["In 1987 the council voted to close the old tram line, and the depot became a market.", "Two brothers who sold spices there argued for years over forty dollars.",
     "When the younger one died, the elder paid it to his widow in coins."],
    ["A masked thief climbed the tower at night to steal the mayor's diamond, but a detective was waiting in the dark.",
     "They fought on the roof until the police arrived with lamps and dogs.", "The thief escaped through the sewers and swore revenge on the whole city."]]
SEED, FACTOR, LIMIT = 20261108, 3.0, 4
CODE = re.compile(r"\bX\d+\b")
CUTS = {20: 40, 30: 60, 40: 80, 50: 100}      # percent of the scale: the smallest stored share that counts (common.py: ceil(theta * 200 - 1e-6))
PLANES = [f"top1_{p}" for p in CUTS] + [f"top3_{p}" for p in CUTS] + ["cut_top1_30", "icon_top1_30", "over100"]
PRIMARY = ["top1_30", "cut_top1_30", "icon_top1_30", "over100"]
PLAIN = [p for p in PLANES if p not in ("cut_top1_30", "icon_top1_30")]      # what records with neither "cut" nor "drop" get (the remakes)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def test_set(works):
    """Original and name-free records, made-up texts first. cut comes from the original text, drop from each version's own sentences."""
    corpus = {ver: [json.loads(l) for l in open(STUDY / "data" / name)] for ver, name in (("orig", "corpus.jsonl"), ("anon", "corpus_neutral.jsonl"))}
    texts = MADE_UP + [(corpus["orig"][i]["label"], corpus["orig"][i]["sentences"], corpus["anon"][i]["sentences"]) for i in works]
    return {ver: [{"id": label, "label": label, "sentences": both[k], "cut": cut_length(both[0]), "drop": [j for j, s in enumerate(both[k]) if ICON.search(s)]}
                  for label, *both in texts] for k, ver in enumerate(("orig", "anon"))}


def activations(model, ids):
    """Per layer, the gate pre-activation and the neuron activation act_fn(gate) * up, each [tokens, width], from plain forward hooks."""
    seen, hooks = {}, []
    for i, layer in enumerate(layers_of(model)):
        for part in ("gate_proj", "up_proj"):
            hooks.append(getattr(layer.mlp, part).register_forward_hook(lambda m, a, out, key=(i, part): seen.__setitem__(key, out.detach())))
    with torch.no_grad():
        model.model(input_ids=torch.tensor([ids]).to(model.model.embed_tokens.weight.device))
        acts = [layer.mlp.act_fn(seen[i, "gate_proj"]) * seen[i, "up_proj"] for i, layer in enumerate(layers_of(model))]
    for h in hooks:
        h.remove()
    return [(seen[i, "gate_proj"][0].float().cpu().numpy(), act[0].float().cpu().numpy()) for i, act in enumerate(acts)]


def sentence_maxima(model, tok, sentences, mask_codes):
    """Each unit's strongest main-regime activation in each sentence, [sentences, 2 * neurons] (positive poles, then negative
    poles as magnitudes), and how many tokens the text has and how many of them belong to codes."""
    ids, spans, text, offsets = encode_document(tok, sentences)
    assert all(0 < b < e <= len(ids) for b, e in spans), "position 0 belongs to no sentence"
    found = [m.span() for m in CODE.finditer(text)] if mask_codes else []
    codes = [i for i, (a, b) in enumerate(offsets) if b > a and any(a < e and s < b for s, e in found)]
    pos, neg = [], []
    for gate, act in activations(model, ids):
        main = np.where(gate > 0, act, np.float32(0))                # the main sign regime: the gate is open
        main[codes] = 0                                              # a code's own tokens carry no peak
        pos.append(np.stack([main[b:e].max(0) for b, e in spans]))
        neg.append(np.stack([(-main[b:e]).max(0) for b, e in spans]))
    return np.concatenate(pos + neg, axis=1), len(ids), len(codes)


def strongest(S, k=1):
    """Mean of each unit's k strongest sentences (of all, if there are fewer), at the float16 precision study_pass.py stores."""
    top = -np.sort(-S, axis=0)[:k]
    total = sum(top[1:], top[0])                                     # added one by one, strongest first, as torch does
    return (total / np.float32(len(top))).astype(np.float16).astype(np.float32)


def shares(score, scale):
    """The rule of finalize_scale.to_shares, written out again: the share of the scale in steps of 0.005, rounded down, at most 255; 0 where the scale is not positive."""
    safe = np.where(scale > 0, scale, np.float32(np.inf))
    return np.clip(np.floor(score.astype(np.float32) / safe * np.float32(200) + np.float32(1e-4)), 0, 255).astype(np.uint8)


def slow_scores(model, tok, records, mask_codes):
    """Each text's four scores, the obvious way, [texts, 4, 2 * neurons]: strongest sentence, mean of the three strongest,
    strongest sentence of the opening, strongest of the sentences not dropped; and the token and code-token counts."""
    scores, n_tokens, n_codes = [], [], []
    for r in records:
        S, n, c = sentence_maxima(model, tok, r["sentences"], mask_codes)
        kept = [k for k in range(len(S)) if k not in r["drop"]] or list(range(len(S)))      # nothing left: every sentence is kept
        scores.append(np.stack([strongest(S), strongest(S, 3), strongest(S[:r["cut"]]), strongest(S[kept])]))
        n_tokens.append(n); n_codes.append(c)
        print(f"slow: {r['label'][:24]:24s} {len(S):2d} sentences, opening {r['cut']:2d}, {len(r['drop']):2d} dropped, {n:4d} tokens, {c:3d} in codes", flush=True)
    return np.stack(scores), n_tokens, n_codes


def planes_of(scores, scale):
    """Every plane as booleans [texts, 2 * neurons], from the four scores."""
    top1, top3, cut, icon = scores.transpose(1, 0, 2)
    planes = {f"top1_{p}": shares(top1, scale) >= least for p, least in CUTS.items()}
    planes.update({f"top3_{p}": shares(top3, scale) >= least for p, least in CUTS.items()})
    planes.update(cut_top1_30=shares(cut, scale) >= 60, icon_top1_30=shares(icon, scale) >= 60, over100=(top1 > scale) & (scale > 0))
    return planes


def make_scale(base, scores):
    """A scale that tries every edge of the rule: FACTOR times each unit's strongest main-regime sentence in PARAGRAPHS; then
    2% of the units get scale 0 and 0.5% a negative scale (never present), 2% a tiny positive scale (present wherever they fire
    at all), and 4% a scale that puts one score of one test text at 20, 30, 40, 50 or 100% of it: half exactly, half within
    five millionths on either side (base: [2 * neurons]; scores: [any, 2 * neurons], rounded to float16)."""
    scale, rng = np.float32(FACTOR) * base, np.random.default_rng(SEED)
    order, n = rng.permutation(len(scale)), len(scale) // 50
    scale[order[:n]], scale[order[n:n + n // 4]], scale[order[2 * n:3 * n]] = 0, -1, 1e-6
    edge = order[3 * n:5 * n]
    share = rng.choice([0.2, 0.3, 0.4, 0.5, 1.0], len(edge)) * (1 + rng.uniform(-5e-6, 5e-6, len(edge)) * rng.integers(0, 2, len(edge)))
    near = scores[rng.integers(0, len(scores), len(edge)), edge] / share
    scale[edge] = np.where(near > 0, near, scale[edge])
    return scale.astype(np.float32)


def process(script, *args):
    return subprocess.run([sys.executable, str(HERE / script), *map(str, args)], capture_output=True, text=True)


def run(script, *args):
    """Run one of the pipeline's scripts in its own process; stop the test if it fails."""
    t0 = time.time()
    done = process(script, *args)
    check(f"{script} {' '.join(str(a).replace(str(OUT) + '/', '') for a in args)} ran", done.returncode == 0, f"{time.time() - t0:.0f} s")
    if done.returncode:
        sys.exit(done.stdout[-2000:] + done.stderr[-4000:])


def refused(what, *records):
    """presence_pass.py must turn a bad texts file down before it loads the model, and write nothing."""
    (OUT / "bad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    shutil.rmtree(OUT / "bad", ignore_errors=True)
    done = process("presence_pass.py", TAG, OUT / "bad.jsonl", OUT / "bad", "--scale", OUT / "scale.npy")
    check(f"presence_pass.py turns down {what}", done.returncode != 0 and "AssertionError" in done.stderr and not list((OUT / "bad").glob("*")))


def compare(name, path, want):
    """One plane file against the bits it should hold: uint8 [texts, units / 8], 8 units to a byte, the first in the highest bit."""
    got = np.load(path)
    packed = (want.reshape(len(want), -1, 8) * (128 >> np.arange(8))).sum(2).astype(np.uint8)
    diff = int(np.unpackbits(got ^ packed).sum()) if got.dtype == np.uint8 and got.shape == packed.shape else want.size
    check(name, diff == 0, f"{diff} of {want.size:,} bits differ ({want.mean():.1%} set)")


def against_slow(name, folder, names, records, planes, n_tokens, n_codes, scale, texts, masked):
    """One run of presence_pass.py against the slow re-computation: every plane file bit for bit, and meta.json."""
    meta, N, digest = json.load(open(folder / "meta.json")), len(records), lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for p in names:
        compare(f"{name} {p} = slow re-computation", folder / f"{p}.bits.npy", planes[p][:N])
    dead = sum(int(np.unpackbits(np.load(folder / f"{p}.bits.npy"), axis=1)[:, scale <= 0].sum()) for p in names)
    check(f"{name} no unit without a positive scale is present in any plane", dead == 0, f"{dead} bits set")
    check(f"{name} planes written, and meta.json planes, ids and sentence counts", meta["planes"] == names and meta["ids"] == [r["id"] for r in records]
          and meta["n_sentences"] == [len(r["sentences"]) for r in records] and sorted(f.name for f in folder.glob("*.npy")) == sorted(f"{p}.bits.npy" for p in names))
    check(f"{name} meta.json token counts = slow re-computation", meta["n_tokens"] == n_tokens[:N], f"{sum(n_tokens[:N])} tokens")
    check(f"{name} meta.json code-token counts = slow re-computation", meta["n_code_tokens"] == n_codes[:N], f"{sum(n_codes[:N])} code tokens")
    check(f"{name} meta.json present counts = slow re-computation", all(meta["present"][p] == planes[p][:N].sum(1).tolist() for p in names))
    check(f"{name} meta.json hashes of the scale and of the texts, and whether codes were masked",
          (meta["scale_sha256"], meta["texts_sha256"], meta["mask_codes"]) == (digest(OUT / "scale.npy"), digest(texts), masked))


def main():
    records = test_set([int(x) for x in sys.argv[1:]] or WORKS)
    shutil.rmtree(OUT, ignore_errors=True)
    study, scale_file = OUT / "study", OUT / "scale.npy"
    files = {"orig": study / "corpus.jsonl", "anon": study / "corpus_neutral.jsonl"}      # presence_pass.py and study_pass.py read the same two files
    study.mkdir(parents=True)
    for ver, path in files.items():
        path.write_text("".join(json.dumps(r) + "\n" for r in records[ver]))
    t0 = time.time()
    model, tok = load(TAG)
    base = np.max([sentence_maxima(model, tok, p, False)[0].max(0) for p in PARAGRAPHS], axis=0)
    slow = {ver: slow_scores(model, tok, records[ver], ver == "anon") for ver in files}
    del model                                                        # the scripts below load their own copy: hand the memory back
    getattr(ctypes.CDLL(None), "malloc_trim", lambda keep: 0)(0)     # first (glibc would keep about 3 GB of it)
    scale = make_scale(base, np.concatenate([slow[ver][0].reshape(-1, len(base)) for ver in files]))
    np.save(scale_file, scale)
    want = {ver: planes_of(slow[ver][0], scale) for ver in files}
    print(f"scale: {len(scale):,} units, {(scale <= 0).mean():.1%} not positive; slow re-computation took {time.time() - t0:.0f} s", flush=True)
    for ver, path in files.items():
        run("presence_pass.py", TAG, path, OUT / ver, "--scale", scale_file, "--planes", "all", *(["--mask-codes"] if ver == "anon" else []))
    run("presence_pass.py", TAG, files["anon"], OUT / "primary", "--scale", scale_file, "--planes", "primary", "--mask-codes", "--limit", LIMIT)
    # the remakes' kind of file: records with neither "cut" nor "drop", codes not masked; read into a folder that already holds
    # another run (a copy of the one just made), whose meta.json and cut and iconography planes must not survive
    plain = OUT / "plain.jsonl"
    plain.write_text("".join(json.dumps({"id": r["id"], "sentences": r["sentences"]}) + "\n" for r in records["orig"]))
    shutil.copytree(OUT / "primary", OUT / "plain")
    run("presence_pass.py", TAG, plain, OUT / "plain", "--scale", scale_file, "--planes", "all", "--limit", LIMIT)
    two = records["orig"][0]["sentences"] * 2
    refused("a cut of 0", {"id": 0, "sentences": two, "cut": 0, "drop": []})
    refused("a negative drop index", {"id": 0, "sentences": two, "cut": 1, "drop": [-1]})
    refused("a drop index past the last sentence", {"id": 0, "sentences": two, "cut": 1, "drop": [2]})
    refused("a file in which only some records have a cut", {"id": 0, "sentences": two, "cut": 1}, {"id": 1, "sentences": two})
    run("study_pass.py", TAG, study, OUT / "out")
    scores = OUT / "out" / TAG / "study"
    study_meta = json.load(open(scores / "meta.json"))["versions"]
    for ver in files:
        against_slow(ver, OUT / ver, PLANES, records[ver], want[ver], *slow[ver][1:], scale, files[ver], ver == "anon")
        check(f"{ver} planes are neither empty nor full", all(0.01 < want[ver][p].mean() < 0.99 for p in PLANES))
        check(f"{ver} token and code-token counts = study_pass.py", (study_meta[ver]["n_tokens"], study_meta[ver]["n_code_tokens"]) == slow[ver][1:])
        pairs = [(f"top1_{p}", "main_top1", least) for p, least in CUTS.items()] + [(f"top3_{p}", "main_top3", least) for p, least in CUTS.items()]
        pairs += [("cut_top1_30", "cut_main_top1", 60)] + ([("icon_top1_30", "icon_main_top1", 60)] if ver == "anon" else [])
        for p, score, least in pairs:                                # study_pass.py scores the iconography control in the name-free texts only
            stored = to_shares(scores / f"{ver}_{score}.npy", scores / "stored.u8.npy", scale)       # the pipeline's own function
            compare(f"{ver} {p} = study_pass.py {ver}_{score}, stored by finalize_scale.to_shares, >= {least}", OUT / ver / f"{p}.bits.npy", np.asarray(stored) >= least)
        top1 = np.load(scores / f"{ver}_main_top1.npy").astype(np.float32)
        compare(f"{ver} over100 = study_pass.py {ver}_main_top1 above a positive scale", OUT / ver / "over100.bits.npy", (top1 > scale) & (scale > 0))
    against_slow("primary", OUT / "primary", PRIMARY, records["anon"][:LIMIT], want["anon"], *slow["anon"][1:], scale, files["anon"], True)
    against_slow("plain", OUT / "plain", PLAIN, records["orig"][:LIMIT], want["orig"], *slow["orig"][1:], scale, plain, False)
    shutil.rmtree(OUT / "out")                                       # the float16 score files; the rest is small
    print(f"{len(fails)} checks failed" if fails else "all checks passed", f"in {time.time() - t0:.0f} s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
