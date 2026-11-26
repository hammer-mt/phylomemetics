"""Record SwiGLU neurons, SAE latents and residual means for a model (torch, bf16, one GPU).

python feats.py qwen3-8b OUT     -> OUT/<tag>/{corpus,alone}_{pos,neg}.npy (f16), *_sae.npy, *_resid.npy, meta.json
Same unit and token rules as study/extract.py and study/wording_test.py: sentence max/min over tokens, first token
of each forward pass left out. SAE tokens with residual norm > 10x the pass median are dropped (the SAEs' own rule).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CFG = {
    "qwen3-8b": dict(repo="Qwen/Qwen3-8B", rev="b968826d9c46dd6066d109eabc6255188de91218",
                     sae="adamkarvonen/qwen3-8b-saes", sae_rev="460a11d556a0c58ffd65909555f38b33e9d71aea",
                     sae_path="saes_Qwen_Qwen3-8B_batch_top_k", sae_layers=[9, 18, 27]),
    "qwen3-32b": dict(repo="Qwen/Qwen3-32B", rev="9216db5781bf21249d130ec9da846c4624c16137",
                      sae="adamkarvonen/qwen3-32b-saes", sae_rev="92e82d40f0c48f502dfdf08a7925e0c34aa4ee24",
                      sae_path="saes_Qwen_Qwen3-32B_batch_top_k", sae_layers=[16, 32, 48]),
    "tiny": dict(repo="Qwen/Qwen3-0.6B", rev=None, sae=None, sae_layers=[7, 14, 21]),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SAE:
    def __init__(self, c, layer):
        if c["sae"] is None:                                    # code test only
            g = torch.Generator().manual_seed(layer)
            self.W, self.b, self.bd, self.th = torch.randn(256, 1024, generator=g) / 30, torch.zeros(256), torch.zeros(1024), torch.tensor(0.1)
            self.D = self.W.T.clone()
        else:
            f = hf_hub_download(c["sae"], f"{c['sae_path']}/resid_post_layer_{layer}/trainer_2/ae.pt", revision=c["sae_rev"])
            s = torch.load(f, map_location="cpu", weights_only=True)
            self.W, self.b, self.D, self.bd, self.th = (s["encoder.weight"], s["encoder.bias"], s["decoder.weight"],
                                                        s["b_dec"], s["threshold"])
        for k in ("W", "b", "bd", "th", "D"):
            setattr(self, k, getattr(self, k).to(DEV, torch.float32))

    def encode(self, x):
        a = torch.relu((x.float() - self.bd) @ self.W.T + self.b)
        return a * (a > self.th)


def load(tag):
    c = CFG[tag]
    tok = AutoTokenizer.from_pretrained(c["repo"], revision=c["rev"])
    m = AutoModelForCausalLM.from_pretrained(c["repo"], revision=c["rev"], dtype=torch.bfloat16 if DEV == "cuda" else torch.float32)
    return m.to(DEV).eval(), tok, c


def spans_of(tok, sentences):
    text, cs = "", []
    for s in sentences:
        if text:
            text += " "
        a = len(text); text += s; cs.append((a, len(text)))
    e = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    off = e["offset_mapping"]
    out = []
    for a0, b0 in cs:
        t = [i for i, (a, b) in enumerate(off) if b > a and a < b0 and b > a0 and i > 0]
        out.append((t[0], t[-1] + 1))
    return e["input_ids"], out


def alone_sentences():
    """Every sentence run on its own, with a key, in a fixed order."""
    sys.path.insert(0, str(HERE))
    from items import items
    return items()


def main(tag, out):
    model, tok, c = load(tag)
    L, H, d = len(model.model.layers), model.config.intermediate_size, model.config.hidden_size
    saes = [SAE(c, l) for l in c["sae_layers"]]
    nl = saes[0].W.shape[0]
    h, r = [None] * L, {}
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: h.__setitem__(i, a[0][0]))
        if i in c["sae_layers"]:
            layer.register_forward_hook(lambda m, a, o, i=i: r.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0]))
    works = [json.loads(x) for x in open(STUDY / "data/corpus.jsonl")]
    docs = [(w["sentences"], w["id"]) for w in works]
    alone = [([t], k) for k, t in alone_sentences()]
    o = Path(out) / tag; o.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run(name, docs, whole_span):
        n = sum(len(s) for s, _ in docs)
        mm = lambda nm, shape: np.lib.format.open_memmap(o / f"{name}_{nm}.npy", mode="w+", dtype=np.float16, shape=shape)
        P, N = mm("pos", (n, L * H)), mm("neg", (n, L * H))
        A, R = mm("sae", (n, nl * len(saes))), mm("resid", (n, d * len(saes)))
        row, keys, sinks, toks, t0 = 0, [], 0, 0, time.time()
        for k_doc, (sents, key) in enumerate(docs):
            if whole_span:
                ids = tok(sents[0], add_special_tokens=True)["input_ids"]; sp = [(1, len(ids))]
            else:
                ids, sp = spans_of(tok, sents)
            model.model(torch.tensor([ids], device=DEV))
            for a, b in sp:
                P[row] = torch.cat([x[a:b].amax(0) for x in h]).float().cpu().numpy().astype(np.float16)
                N[row] = -torch.cat([x[a:b].amin(0) for x in h]).float().cpu().numpy().astype(np.float16)
                for j, l in enumerate(c["sae_layers"]):
                    nm = r[l].float().norm(dim=-1)
                    keep = nm[a:b] <= 10 * nm[1:].median()
                    sinks += int((~keep).sum())
                    x = r[l][a:b][keep] if keep.any() else r[l][a:b]
                    A[row, j * nl:(j + 1) * nl] = saes[j].encode(x).amax(0).cpu().numpy().astype(np.float16)
                    R[row, j * d:(j + 1) * d] = x.float().mean(0).cpu().numpy().astype(np.float16)
                keys.append(key); row += 1
            toks += len(ids)
            if k_doc % 200 == 0:
                print(name, k_doc, len(docs), f"{toks / (time.time() - t0):.0f} tok/s", flush=True)
        return keys, sinks

    ck, s1 = run("corpus", docs, False)
    ak, s2 = run("alone", alone, True)
    json.dump(dict(tag=tag, repo=c["repo"], rev=c["rev"], sae=c["sae"], sae_layers=c["sae_layers"], L=L, H=H, d=d, n_latents=nl,
                   corpus_work=ck, alone_key=ak, sink_tokens_dropped=s1 + s2), open(o / "meta.json", "w"))
    print("DONE", tag)


if __name__ == "__main__":
    main(*sys.argv[1:3])
