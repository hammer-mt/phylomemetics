"""Cost of one neuron pass on a MADE-UP story (no corpus or TEST text): forward pass with extract.py's Recorder,
then (a) per-work top-3-mean aggregate per unit computed on the GPU (what a compact T3 extraction would store),
(b) copy of the full per-sentence fp16 arrays to the host (what extract.py stores). Activations are not analysed.
usage: HF_HUB_OFFLINE=1 python neuron_timing.py <tag>"""
import json
import sys
import time

sys.path.insert(0, "<repo>")
sys.path.insert(0, "<scratchpad>/prereg/scripts")
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402
from extract import MODELS, Recorder, sentence_spans  # noqa: E402
from llm_timing import STORY  # noqa: E402

OUT = "<scratchpad>/prereg/scout/tools/neuron_timing_{}.json"


def main(tag):
    model, tokenizer = load(MODELS[tag])
    hf = tokenizer._tokenizer
    layers = model.model.layers
    sink = {"pos": [None] * len(layers), "neg": [None] * len(layers), "gather": None}
    for i, layer in enumerate(layers):
        layer.mlp = Recorder(layer.mlp, sink, i)
    ids, spans = sentence_spans(hf, STORY)
    longest = max(e - s for s, e in spans)
    sink["gather"] = mx.array([[min(s + j, e - 1) for j in range(longest)] for s, e in spans])
    res = {"model": MODELS[tag], "n_sentences": len(STORY), "n_tokens": len(ids)}
    for rep in range(3):
        t0 = time.time()
        model.model(mx.array([ids]))
        p = mx.concatenate(sink["pos"], axis=1)
        q = mx.concatenate(sink["neg"], axis=1)
        top3 = mx.concatenate([mx.sort(p, axis=0)[-3:].mean(0), -mx.sort(q, axis=0)[:3].mean(0)])   # per-unit work score
        mx.eval(top3)
        t_agg = time.time() - t0
        t1 = time.time()
        full = np.array(mx.concatenate([p, q], axis=1).astype(mx.float16))
        t_copy = time.time() - t1
        res[f"rep{rep}"] = dict(forward_plus_aggregate_s=round(t_agg, 3), full_copy_to_host_s=round(t_copy, 3))
        del p, q
        mx.clear_cache()
    res["n_units"] = int(top3.shape[0])
    res["aggregate_bytes_fp16"] = int(top3.shape[0]) * 2
    res["per_sentence_bytes_int8"] = int(full.shape[1])
    res["sentences_per_s_aggregate_only"] = round(len(STORY) / res["rep2"]["forward_plus_aggregate_s"], 1)
    res["peak_GB"] = round(mx.get_peak_memory() / 1e9, 2)
    json.dump(res, open(OUT.format(tag), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
