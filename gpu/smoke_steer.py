"""Smoke test, part 2: steer one neuron with Hugging Face transformers and check the outputs make sense.

    python gpu/smoke_steer.py --model Qwen/Qwen3-32B --layer 32            # first candidate from smoke_record.py
    python gpu/smoke_steer.py --model Qwen/Qwen3-32B --layer 32 --neuron 1234

Steering adds a push along the neuron's down-projection column W_down[:, n] to the MLP output at every position, prompt
and generated. Doses are in neuron-explorer's relative unit: a dose d pushes the residual stream by d times the median
residual norm leaving the layer (measured by smoke_record.py), so +0.8 is a large push and -0.7 a large pull. Each
prompt is generated greedily at every dose, and a random direction is pushed just as hard as a control.

What "makes sense" means here, and what the script checks:
  1. dose 0 reproduces the unsteered model exactly (the hook does nothing when it should do nothing);
  2. adding to the neuron's activation (a pre-hook on down_proj) and adding the scaled W_down column to the MLP output
     give the same text, which is the equivalence the vLLM fork relies on;
  3. a positive dose moves the text towards the neuron's word more than the random control pushed equally hard does.
Writes $PHYLO_DATA/runs/smoke_steer_L<layer>_N<neuron>.json, which smoke_vllm_steer.py reads to compare against vLLM.
"""
import argparse
import json
import os
import time

import torch

from common import load, pick_device, steer_hook

PROMPTS = ["Who are you?", "Tell me a short story.", "What should I do this weekend?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--neuron", type=int, default=None)
    ap.add_argument("--doses", default="-0.7,0.5,0.8")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = pick_device(a.device)
    data = os.path.expanduser(os.environ.get("PHYLO_DATA", "~/phylo-data"))
    smoke = json.load(open(os.path.join(data, "dense", a.model.replace("/", "--"), f"smoke_L{a.layer}.json")))
    word = None
    if a.neuron is None:
        if not smoke["candidates"]:
            raise SystemExit("smoke_record.py found no lexical candidate; pass --neuron")
        a.neuron, word = smoke["candidates"][0]["neuron"], smoke["candidates"][0]["word"]
    else:
        word = next((c["word"] for c in smoke["candidates"] if c["neuron"] == a.neuron), None)
    resid_p50 = smoke["resid_norm_q10_50_90"][1]
    doses = [float(x) for x in a.doses.split(",")]

    tok, model, layers = load(a.model, dev)
    layer = layers[a.layer]
    w_col = layer.mlp.down_proj.weight[:, a.neuron].detach().float()
    down_norm = w_col.norm().item()
    g = torch.Generator().manual_seed(0)
    rand_dir = torch.randn(w_col.shape, generator=g).to(dev)
    rand_dir /= rand_dir.norm()
    print(f"L{a.layer} N{a.neuron} ('{word}'): |W_down col| = {down_norm:.3f}, median residual norm = {resid_p50:.0f}; "
          f"dose 1.0 adds {resid_p50 / down_norm:.1f} to the neuron's activation")

    def add_vector(vec):
        return layer.mlp.register_forward_hook(lambda m, i, o: o + vec.to(o.dtype))

    def generate(prompt, hook=None):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
        h = hook() if hook else None
        try:
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False)
        finally:
            if h:
                h.remove()
        new = out[0, ids.shape[1]:].tolist()
        return new, tok.decode(new, skip_special_tokens=True)

    def mentions(text):
        return text.lower().count(word) if word else 0

    rows, checks = [], {}
    t0 = time.time()
    for p in PROMPTS:
        base_ids, base = generate(p)
        rows.append({"prompt": p, "kind": "baseline", "dose": 0.0, "add": 0.0, "text": base, "ids": base_ids})
        zero_ids, _ = generate(p, lambda: steer_hook(layer, a.neuron, 0.0))
        checks.setdefault("dose0_identical", []).append(zero_ids == base_ids)
        for d in doses:
            add = d * resid_p50 / down_norm
            ids_n, text_n = generate(p, lambda: add_vector(add * w_col.to(dev)))
            rows.append({"prompt": p, "kind": "neuron", "dose": d, "add": add, "text": text_n, "ids": ids_n})
            if d == max(doses):
                ids_pre, _ = generate(p, lambda: steer_hook(layer, a.neuron, add))
                same = next((i for i, (x, y) in enumerate(zip(ids_pre, ids_n)) if x != y), min(len(ids_pre), len(ids_n)))
                checks.setdefault("prehook_vs_vector_common_prefix", []).append(same)
                ids_r, text_r = generate(p, lambda: add_vector(d * resid_p50 * rand_dir))
                rows.append({"prompt": p, "kind": "random", "dose": d, "add": add, "text": text_r, "ids": ids_r})
    print(f"generated {len(rows) + 2 * len(PROMPTS)} rollouts in {time.time() - t0:.0f}s\n")

    for p in PROMPTS:
        print(f"### {p}")
        for r in (r for r in rows if r["prompt"] == p):
            label = r["kind"] if r["kind"] == "baseline" else f"{r['kind']} {r['dose']:+g}"
            print(f"[{label:>14}] ({mentions(r['text'])} x '{word}') {r['text'][:300]!r}")
        print()

    pos = sum(mentions(r["text"]) for r in rows if r["kind"] == "neuron" and r["dose"] > 0)
    ctrl = sum(mentions(r["text"]) for r in rows if r["kind"] == "random")
    base = sum(mentions(r["text"]) for r in rows if r["kind"] == "baseline")
    print(f"check: dose 0 reproduces the unsteered text: {all(checks['dose0_identical'])}")
    print(f"check: pre-hook on the activation vs W_down vector on the MLP output, tokens identical before the first "
          f"difference: {checks['prehook_vs_vector_common_prefix']} of {a.max_new} (bf16 rounding can split them late)")
    print(f"check: '{word}' mentions: baseline {base}, positive doses {pos}, random control {ctrl}. "
          f"{'Looks right.' if pos > max(base, ctrl) else 'No clear effect: try a bigger dose or another candidate.'}")

    out = os.path.join(data, "runs", f"smoke_steer_L{a.layer}_N{a.neuron}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": a.model, "layer": a.layer, "neuron": a.neuron, "word": word, "resid_p50": resid_p50,
               "down_norm": down_norm, "w_down_col": w_col.cpu().tolist(), "prompts": PROMPTS, "max_new": a.max_new,
               "rows": rows, "checks": checks}, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
