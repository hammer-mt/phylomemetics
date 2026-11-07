"""Ablation: does the model use a set of units to recognise whose story it is reading?

    python ablate.py <model-tag> <jobs.json> <out.json> [scale.npy]

jobs.json: {"question": "...", "sets": {"name": [unit, ...], ...}, "reply_for": ["name", ...], "reply_tokens": 8,
            "items": [{"id", "text", "answers": ["Batman", ...], "conditions": ["name", ...], "question": "..."}]}

An item's own "question" replaces the shared one. Matched random sets are too many to list: with "partner_file"
(an .npz holding units [n], partners [K, n] and partners_distinctive [K2, n]) an item names its own units by
"members" (indices into units), and its conditions may be "canon" (those units), "rand:<k>" (their partners in
matched random set k) and "dist:<k>" (their partners in set k of the second control). With no sets and no conditions the script simply puts each text
and question to the untouched model (the recognition probe of PLAN.md Section 7, and the multiple-choice judge).

For each item the text and the question are put to the model in its chat format (thinking off where the template
has it). The outcome is the log of the probability that the reply starts with one of the item's answers (the sum
over the answers, each teacher-forced over its own tokens). It is measured untouched and with each listed set of
units switched off. A positive-pole unit is switched off by zeroing its activation wherever it is positive with the
gate open; a negative-pole unit likewise where it is negative.

Units are switched off only on the tokens of the story text itself: not on the chat template, the question, or
the reply. So a unit that merely writes the name "Batman" cannot move the outcome by being silenced while the
answer is produced; only what the model took from the story can. (With "where": "all" in the jobs file the units
are off at every position, as a secondary check.) The model's own greedy reply ("reply_tokens" tokens, 8 unless
given) is kept for the untouched run and for the sets named in "reply_for".

"mode" in the jobs file narrows what is switched off: "strong" zeroes a unit only where it is at or above 30% of its
scale, "weak" only where it is below (both need scale.npy); the default zeroes all of its open-gate activity. For
every condition the summed size of the activations that were zeroed is kept ("removed"), so that sets can be
compared on how much they took away.
"""
import copy
import json
import sys

import numpy as np
import torch

from neurons import Tap, geometry, load, now


def main(tag, jobs_path, out_path, scale_path=None):
    jobs = json.load(open(jobs_path))
    mode = jobs.get("mode", "all")
    scale = np.load(scale_path) if scale_path else None
    assert mode == "all" or scale is not None, "strong and weak modes need the scale file"
    everywhere = jobs.get("where", "story") == "all"
    reply_tokens = int(jobs.get("reply_tokens", 8))
    sets = jobs.get("sets", {})
    partner = np.load(jobs["partner_file"]) if jobs.get("partner_file") else None

    def units_for(name, item):
        if name in sets:
            return sets[name]
        members = np.asarray(item["members"], dtype=np.int64)
        if name == "canon":
            return partner["units"][members]
        kind, k = name.split(":")
        return partner["partners" if kind == "rand" else "partners_distinctive"][int(k)][members]
    model, tok = load(tag)
    L, D = geometry(model)
    U = L * D
    dev = model.model.embed_tokens.weight.device
    ctx = {"by_layer": {}, "lo": 0, "hi": 0, "story_pass": False, "removed": 0.0}

    def fn(l, g, h):
        hit = ctx["by_layer"].get(l)
        if hit is None or not (everywhere or ctx["story_pass"]):
            return None
        if not everywhere and h.shape[1] < ctx["hi"]:
            return None                            # a later decoding step inside generate(): the story is already behind us
        lo, hi = (0, h.shape[1]) if everywhere else (ctx["lo"], ctx["hi"])
        h = h.clone()
        for cols, sign, thr in ((hit[0], 1, hit[2]), (hit[1], -1, hit[3])):
            if len(cols):
                c = cols.to(h.device)
                hc, gc = h[:, lo:hi][..., c], g[:, lo:hi][..., c]
                off = (gc > 0) & (hc * sign > 0)
                if mode != "all":
                    strong = hc.abs().float() >= thr.to(h.device)
                    off &= strong if mode == "strong" else ~strong
                ctx["removed"] += float((hc.abs().float() * off).sum())
                h[:, lo:hi, c] = torch.where(off, torch.zeros_like(hc), hc)
        return h

    def set_units(units):
        units = np.asarray(units, dtype=np.int64)
        by = {}
        for l in np.unique((units % U) // D):
            s = units[(units % U) // D == l]
            thr = [torch.tensor(0.3 * scale[x], dtype=torch.float32) if scale is not None else None for x in (s[s < U], s[s >= U])]
            by[int(l)] = (torch.tensor(s[s < U] % D, dtype=torch.long), torch.tensor(s[s >= U] % D, dtype=torch.long), thr[0], thr[1])
        ctx["by_layer"] = by

    def measure(prompt_ids, answers, reply):
        """One pass over the prompt (units off on the story tokens), then each answer's own tokens from the cache."""
        logps = []
        with torch.no_grad():
            ctx["story_pass"], ctx["removed"] = True, 0.0
            first = model(input_ids=prompt_ids, use_cache=True)
            ctx["story_pass"] = False
            removed = ctx["removed"]
            lp0 = torch.log_softmax(first.logits[0, -1].float(), -1)
            for a_ids in answers:
                lp = float(lp0[a_ids[0, 0]])
                if a_ids.shape[1] > 1:
                    more = model(input_ids=a_ids[:, :-1], past_key_values=copy.deepcopy(first.past_key_values), use_cache=True)
                    l2 = torch.log_softmax(more.logits[0].float(), -1)
                    lp += float(l2[torch.arange(a_ids.shape[1] - 1), a_ids[0, 1:]].sum())
                logps.append(lp)
            res = {"logp": float(np.logaddexp.reduce(logps)), "logp_each": [round(x, 4) for x in logps], "removed": round(removed, 2)}
            if reply:
                ctx["story_pass"] = True          # generate() runs the prompt first; later steps see one token at a time
                gen = model.generate(prompt_ids, max_new_tokens=reply_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
                ctx["story_pass"] = False
                res["reply"] = tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        return res

    tap = Tap(model, fn)
    out = {"model": tag, "question": jobs["question"], "where": "all" if everywhere else "story", "mode": mode, "set_sizes": {k: len(v) for k, v in sets.items()}, "items": []}
    for k, item in enumerate(jobs["items"]):
        msgs = [{"role": "user", "content": item["text"] + "\n\n" + item.get("question", jobs["question"])}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
        start = prompt.index(item["text"])
        end = start + len(item["text"])
        inside = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end and b > start]
        ctx["lo"], ctx["hi"] = inside[0], inside[-1] + 1
        p_ids = torch.tensor([enc["input_ids"]], device=dev)
        answers = [tok(a, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev) for a in item["answers"]]
        set_units([])
        res = {"id": item["id"], "answers": item["answers"], "n_prompt_tokens": int(p_ids.shape[1]), "story_tokens": [ctx["lo"], ctx["hi"]],
               "untouched": measure(p_ids, answers, True), "conditions": {}}
        for name in item.get("conditions", []):
            set_units(units_for(name, item))
            res["conditions"][name] = measure(p_ids, answers, name in jobs.get("reply_for", []))
        out["items"].append(res)
        if k % 5 == 0 or k == len(jobs["items"]) - 1:
            print(f"{now()} {k + 1}/{len(jobs['items'])} {item['id'][:30]:30s} untouched {res['untouched']['logp']:.2f} reply {res['untouched']['reply']!r}", flush=True)
            json.dump(out, open(out_path, "w"), ensure_ascii=False)
    tap.remove()
    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    print(f"{now()} wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
