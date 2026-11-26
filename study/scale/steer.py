"""Part C. python steer.py gen TAG FEATS RESULTS   then   python steer.py judge RESULTS (Qwen3-32B, unsteered)"""
import json, sys
from pathlib import Path
import numpy as np, torch
from items import HERE, STUDY, items
from feats import load, SAE, DEV

CONCEPTS = {"funeral": 17909, "coin_flip": 765185, "secret_identity": 168352, "morbid_joke": 31044, "revenge_vow": None}
SETTINGS = ["a family bakery in a mountain town", "a research station in Antarctica", "a train crossing the prairie",
            "two sisters who run a seaside hotel", "a young teacher's first week at a village school",
            "a lighthouse keeper and a passing ship", "a chess tournament in a big city", "a farm during a long drought",
            "a museum guard on the night shift", "a band on its first tour"]
UP, DOWN, SAMPLES = [2, 5, 10, 20], [0, -5], 2


def pick_units(tag, feats, results, L, H, nl, sae_layers):
    """[{concept, source, role, kind, layer, index, sign, max}] targets and activity-matched random controls."""
    r = json.load(open(results / f"{tag}_features.json"))
    c = {k: np.load(feats / tag / f"corpus_{k}.npy", mmap_mode="r") for k in ("pos", "neg", "sae")}
    from analyze import UNITS_8B
    rng, out = np.random.default_rng(7), []
    for name, u8 in CONCEPTS.items():
        part_a = r["revenge"]["neuron"]["unit"] if u8 is None else r["A"]["neuron"]["units"][UNITS_8B.index(u8)]
        for src, kind, u in (("A", "neuron", part_a), ("B", "neuron", r["B"]["neuron"]["units"][name][0]), ("B", "sae", r["B"]["sae"]["units"][name][0])):
            if kind == "neuron":
                pole, ch = ("pos", u) if u < L * H else ("neg", u - L * H)
                lo = ch // H * H; W = H
            else:
                pole, ch = "sae", u
                lo = ch // nl * nl; W = nl
            mx = np.nan_to_num(np.asarray(c[pole][:, lo:lo + W], np.float32).max(0), posinf=65504)
            t = mx[ch - lo]
            near = np.argsort(np.abs(np.log(np.maximum(mx, 1e-6)) - np.log(max(t, 1e-6))))[1:200]
            k = int(rng.choice(near))
            for role, j in (("target", ch - lo), ("random", k)):
                out.append(dict(concept=name, source=src, kind=kind, role=role, layer=int(lo // W) if kind == "neuron" else sae_layers[lo // W],
                                index=int(j), sign=-1 if pole == "neg" else 1, max=float(mx[j])))
    return out


class Steer:
    """Sets one unit on every token: a neuron's activation, or an SAE latent through its decoder direction."""
    def __init__(self, model, u, alpha, saes):
        lay = model.model.layers[u["layer"]]
        if u["kind"] == "neuron":
            def hook(m, a):
                x = a[0].clone(); v = u["sign"] * alpha * u["max"]
                if alpha == 0:                                    # ablate this pole only
                    x[..., u["index"]] = torch.where(x[..., u["index"]] * u["sign"] > 0, 0, x[..., u["index"]])
                else:
                    x[..., u["index"]] = v
                return (x,)
            self.h = lay.mlp.down_proj.register_forward_pre_hook(hook)
        else:
            s = saes[u["layer"]]; dirn = s.D[:, u["index"]]
            def hook(m, a, o):
                x = o[0] if isinstance(o, tuple) else o
                act = torch.relu((x.float() - s.bd) @ s.W[u["index"]] + s.b[u["index"]])
                act = act * (act > s.th)
                y = (x + ((alpha * u["max"] - act)[..., None] * dirn).to(x.dtype))
                return (y,) + o[1:] if isinstance(o, tuple) else y
            self.h = lay.register_forward_hook(hook)
    def __enter__(self): return self
    def __exit__(self, *a): self.h.remove()


@torch.no_grad()
def gen(tag, feats, results):
    feats, results = Path(feats), Path(results)
    model, tok, c = load(tag)
    tok.padding_side = "left"
    L, H = len(model.model.layers), model.config.intermediate_size
    saes = {l: SAE(c, l) for l in c["sae_layers"]}
    units = pick_units(tag, feats, results, L, H, saes[c["sae_layers"][0]].W.shape[0], c["sae_layers"])
    sel = json.load(open(HERE / "data/select.json"))
    chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    up = [(i, chat(f"Write the first 80 words of a short story about {s}.")) for i, s in enumerate(SETTINGS) for _ in range(SAMPLES)]
    down = {k: [(i, chat(f"Continue this story with the next 80 words. Write only the continuation.\n\n{s}")) for i, s in enumerate(sel[k]["reworded"]) for _ in range(SAMPLES)]
            for k in CONCEPTS}
    out = open(results / f"steer_{tag}.jsonl", "w")

    def run(prompts, seed):
        torch.manual_seed(seed)
        b = tok([p for _, p in prompts], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = model.generate(**b, max_new_tokens=110, do_sample=True, temperature=0.8, top_p=0.95, top_k=0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)

    def write(prompts, texts, **kw):
        for (i, _), t in zip(prompts, texts):
            out.write(json.dumps(kw | dict(prompt=i, text=t)) + "\n")
        out.flush()

    write(up, run(up, 1), mode="up", alpha=None, role="none", concept="none", source="", kind="")
    for k in CONCEPTS:
        write(down[k], run(down[k], 1), mode="down", alpha=None, role="none", concept=k, source="", kind="")
    for n, u in enumerate(units):
        for mode, alphas in (("up", UP), ("down", DOWN)):
            for a in alphas:
                p = up if mode == "up" else down[u["concept"]]
                with Steer(model, u, a, saes):
                    write(p, run(p, 1), mode=mode, alpha=a, **u)
        print(n + 1, len(units), u["concept"], u["kind"], u["role"], flush=True)


def judge(results, tag="qwen3-32b"):
    from baselines import yes_no, DEFS
    results = Path(results)
    model, tok, _ = load(tag)
    for f in sorted(results.glob("steer_*.jsonl")):
        if f.name.endswith("_judged.jsonl"):
            continue
        rows = [json.loads(x) for x in open(f)]
        for k in CONCEPTS:
            q = [f"Text:\n{r['text']}\n\nStory element: {DEFS[k]['label']}. {DEFS[k]['definition']}\n\nDoes the text show this element, or something very close to it? Answer yes or no." for r in rows]
            for r, v in zip(rows, yes_no(model, tok, q, 64)):
                r[f"j_{k}"] = round(float(1 / (1 + np.exp(-v))), 4)
        q = [f"Text:\n{r['text']}\n\nIs this text coherent, readable English (it may stop mid-sentence)? Answer yes or no." for r in rows]
        for r, v in zip(rows, yes_no(model, tok, q, 64)):
            r["j_coherent"] = round(float(1 / (1 + np.exp(-v))), 4)
        with open(results / f"{f.stem}_judged.jsonl", "w") as o:
            for r in rows:
                o.write(json.dumps(r) + "\n")
        print("judged", f.name, len(rows), flush=True)


if __name__ == "__main__":
    gen(*sys.argv[2:5]) if sys.argv[1] == "gen" else judge(*sys.argv[2:])
