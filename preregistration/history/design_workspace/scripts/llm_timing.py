"""Cost of LLM reading with the local models (generation mode) on a MADE-UP story (no corpus or TEST text).
Measures: one yes/no judge call per meme with no cache; per-meme calls that reuse a cached story prefix; one
multi-meme JSON call; flat summary; two-level recursive summary. Reports seconds, prompt and generation tokens/s,
and peak memory. usage: HF_HUB_OFFLINE=1 python llm_timing.py <qwen3-8b|llama3.1-8b>"""
import json
import sys
import time

sys.path.insert(0, "<repo>")
import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402
from extract import MODELS  # noqa: E402

OUT = "<scratchpad>/prereg/scout/tools/llm_timing_{}.json"
STORY = [
    "A glassblower named Ilse inherits a failing workshop on a canal island after her mother drowns in the spring flood.",
    "The island council, led by a smiling harbour clerk, wants the workshop's furnace for a new customs house.",
    "Ilse discovers that her mother kept ledgers recording bribes paid to the clerk by a salt merchant.",
    "She hides the ledgers inside a cracked chandelier and tells no one, not even her apprentice Tomas.",
    "At night she begins blowing coloured glass floats and sets them adrift with coded warnings for the fishermen.",
    "The fishermen start avoiding the merchant's nets, and his salt barges rot at the quay.",
    "The clerk suspects the apprentice and has him beaten by two dockhands in an alley behind the bell tower.",
    "Ilse nurses Tomas and, guilty, finally shows him the ledgers.",
    "Tomas recognises a signature in the ledgers as belonging to his own uncle, the island's magistrate.",
    "The two quarrel; Tomas leaves the workshop and moves in with his uncle.",
    "A winter storm closes the harbour, and the island runs short of lamp oil and bread.",
    "The merchant offers grain to anyone who will sign away their boat moorings.",
    "Ilse melts down her mother's last chandelier to make lenses for the lighthouse, whose lamp has failed.",
    "With the new lenses the lighthouse guides a relief ship through the storm, and the island cheers the keeper, not her.",
    "Tomas, uneasy, searches his uncle's study and finds a second ledger listing the drowned mother's name with a payment beside it.",
    "He realises the flood gates were left open on purpose the night she drowned.",
    "He brings the second ledger to Ilse, and they reconcile on the frozen canal.",
    "At the spring festival, Ilse unveils a great glass mask that projects the ledger pages onto the town hall wall.",
    "The crowd reads the names in silence, and the merchant tries to flee on a salt barge.",
    "The barge, rotten from a winter at the quay, breaks apart in the harbour mouth.",
    "The magistrate resigns and confesses that he only signed to protect his nephew's apprenticeship.",
    "The clerk is exiled to the mainland, and the customs house is never built.",
    "Ilse takes Tomas as her partner rather than her apprentice, and they rebuild the furnace together.",
    "On the anniversary of the flood, they set one clear glass float adrift with no message inside.",
]
MEMES = ["secret identity", "origin wound: a parent's death", "heir to a dead parent's fortune", "masked or disguised vigilante",
         "loyal servant or mentor who raised the hero", "orphan ward", "coin flip decides fate", "fall into a vat of chemicals",
         "a funeral", "corrupt official", "revenge vow", "coded signal to allies", "hidden ledger or evidence",
         "storm or disaster", "betrayal by a relative", "public unmasking of villains", "exile of the villain",
         "mentor-apprentice quarrel", "lighthouse", "inheritance of a workshop", "bribery", "drowning", "festival", "beating in an alley",
         "reconciliation"]


def chat(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def run(model, tok, prompt, max_tokens, cache=None):
    t0 = time.time()
    text, last = "", None
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, prompt_cache=cache):
        text += r.text
        last = r
    return dict(seconds=round(time.time() - t0, 3), prompt_tokens=last.prompt_tokens, prompt_tps=round(last.prompt_tps),
                gen_tokens=last.generation_tokens, gen_tps=round(last.generation_tps, 1), peak_GB=round(last.peak_memory, 2)), text


def main(tag):
    model, tok = load(MODELS[tag])
    story = " ".join(STORY)
    res = {"model": MODELS[tag], "story_sentences": len(STORY), "n_memes": len(MEMES)}
    run(model, tok, chat(tok, "Say OK."), 3)      # warm-up
    q = lambda m: f"Story:\n{story}\n\nDoes this story contain the following story element: \"{m}\"? Answer only yes or no."
    # 1) no cache: full prompt per meme (first 5 memes, extrapolate)
    calls = [run(model, tok, chat(tok, q(m)), 2)[0] for m in MEMES[:5]]
    res["judge_nocache_per_meme_s"] = round(sum(c["seconds"] for c in calls) / len(calls), 3)
    res["judge_nocache_example"] = calls[0]
    # 2) cached story prefix: prefill once, then per meme only the question suffix
    full = chat(tok, q("XXXX"))
    prefix = full.split("Does this story contain")[0]
    cache = make_prompt_cache(model)
    pre = run(model, tok, prefix, 1, cache)[0]
    n_after_prefix = cache[0].offset
    per, answers = [], []
    for m in MEMES:
        suffix = full[len(prefix):].replace("XXXX", m)
        r, a = run(model, tok, suffix, 2, cache)
        per.append(r["seconds"])
        answers.append(a.strip())
        trim_prompt_cache(cache, cache[0].offset - n_after_prefix)
    res["judge_cached"] = dict(prefill=pre, per_meme_s=round(sum(per) / len(per), 3), total_25_memes_s=round(pre["seconds"] + sum(per), 2))
    res["judge_cached_answers_sample"] = dict(zip(MEMES[:8], answers[:8]))
    # 3) one multi-meme call
    lst = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(MEMES))
    r, a = run(model, tok, chat(tok, f"Story:\n{story}\n\nFor each story element below, answer yes or no: does the story contain it? "
                                     f"Reply as a numbered list with yes or no only.\n{lst}"), 200)
    res["judge_multimeme_one_call"] = r
    res["judge_multimeme_sample"] = a[:300]
    # 4) flat summary
    r, a = run(model, tok, chat(tok, f"Summarise this story in about 80 words, keeping the key plot events.\n\n{story}"), 160)
    res["summary_flat"] = r
    # 5) two-level recursive summary: 4 chunks -> 4 summaries -> 1 summary
    chunks = [" ".join(STORY[i:i + 6]) for i in range(0, len(STORY), 6)]
    t0 = time.time()
    subs = []
    for c in chunks:
        r, a = run(model, tok, chat(tok, f"Summarise this part of a story in about 40 words.\n\n{c}"), 80)
        subs.append(a.strip())
    r, a = run(model, tok, chat(tok, "Combine these consecutive part-summaries into one summary of about 80 words.\n\n" + "\n".join(subs)), 160)
    res["summary_recursive_2level"] = dict(seconds=round(time.time() - t0, 2), final=r)
    res["summary_recursive_sample"] = a[:400]
    res["peak_GB"] = round(mx.get_peak_memory() / 1e9, 2)
    json.dump(res, open(OUT.format(tag), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
