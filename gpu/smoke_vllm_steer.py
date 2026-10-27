"""Smoke test, part 3: repeat smoke_steer.py's rollouts with the vLLM steering fork and compare them with Hugging Face.

    python gpu/smoke_vllm_steer.py --run ~/phylo-data/runs/smoke_steer_L32_N1234.json

Sends W_down[:, n] with scale `add` to the fork's `mlp_out` hook, as neuron-explorer's sweep/steer_sweep.py does. HF
and vLLM use different bf16 kernels, so greedy texts can part after a few dozen tokens even unsteered; the test is that
the steered pairs agree about as long as the unsteered pair does (neuron-explorer's sweep/check_equivalence.py makes the
same comparison). Run it in its own process: HF and vLLM each hold a full bf16 copy of the model.
"""
import argparse
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def common_prefix(x, y):
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="json written by smoke_steer.py")
    ap.add_argument("--gpu-mem", type=float, default=0.80, help="share of (unified) memory vLLM may take")
    a = ap.parse_args()
    run = json.load(open(a.run))
    tok = AutoTokenizer.from_pretrained(run["model"])
    prompt_ids = {p: list(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                                  enable_thinking=False, tokenize=True, return_dict=True)["input_ids"])
                  for p in run["prompts"]}
    rows = [r for r in run["rows"] if r["kind"] in ("baseline", "neuron")]

    llm = LLM(model=run["model"], dtype="bfloat16", enable_steering=True, max_steering_configs=16, max_num_seqs=16,
              gpu_memory_utilization=a.gpu_mem, max_model_len=max(map(len, prompt_ids.values())) + run["max_new"] + 16)

    def params(r):
        kw = {"temperature": 0.0, "max_tokens": run["max_new"]}
        if r["kind"] == "neuron":
            kw["steering_vectors"] = {"mlp_out": {run["layer"]: {"vector": run["w_down_col"], "scale": r["add"]}}}
        return SamplingParams(**kw)

    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[r["prompt"]]) for r in rows],
                        [params(r) for r in rows], use_tqdm=False)
    agree = {"baseline": [], "neuron": []}
    for r, o in zip(rows, outs):
        ids = list(o.outputs[0].token_ids)
        n = common_prefix(ids, r["ids"])
        agree[r["kind"]].append(n)
        label = r["kind"] if r["kind"] == "baseline" else f"neuron {r['dose']:+g}"
        print(f"[{r['prompt'][:24]:>24} | {label:>12}] HF and vLLM agree for {n:3d} tokens. vLLM: {o.outputs[0].text[:160]!r}")
    print(f"\nunsteered agreement {agree['baseline']}, steered agreement {agree['neuron']} (tokens, of {run['max_new']})")
    print("Steered agreement in the same range as unsteered means the fork steers the same neuron the same way.")


if __name__ == "__main__":
    main()
