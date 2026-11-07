"""Ablation: does the model use a set of units to recognise whose story it is reading?

    python ablate.py <model-tag> <jobs.json> <out.json> [scale.npy]

jobs.json: {"question": "...", "sets": {"name": [unit, ...], ...}, "reply_for": ["name", ...], "reply_tokens": 8,
            "items": [{"id", "text", "answers": ["Batman", ...], "conditions": ["name", ...], "question": "..."}]}

An item's own "question" replaces the shared one. Matched random sets are too many to list: with "partner_file"
(an .npz holding units [n] and one or more partner arrays [K, n]) an item names its own units by "members" (indices
into units), and its conditions may be "canon" (those units) and "<control>:<k>" (their partners in set k of that
control; "rand" reads the array partners, any other name reads partners_<name>). With no sets and no conditions the
script simply puts each text and question to the untouched model (the recognition probe of PLAN.md Section 7, and
the multiple-choice judge).

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

Conditions are run in batches ("batch" in the jobs file, 16 unless given): every row of a batch is the same prompt
with a different set switched off, and the untouched prompt is one of the rows.
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
    batch = int(jobs.get("batch", 16))
    sets = jobs.get("sets", {})
    partner = np.load(jobs["partner_file"]) if jobs.get("partner_file") else None

    def units_for(name, item):
        if name in sets:
            return np.asarray(sets[name], dtype=np.int64)
        members = np.asarray(item["members"], dtype=np.int64)
        if name == "canon":
            return partner["units"][members]
        kind, k = name.split(":")
        return partner["partners" if kind == "rand" else f"partners_{kind}"][int(k)][members]

    model, tok = load(tag)
    L, D = geometry(model)
    U = L * D
    dev = model.model.embed_tokens.weight.device
    thr = torch.tensor(0.3 * scale.reshape(2, L, D), dtype=torch.float32) if scale is not None else None
    ctx = {"mask": None, "lo": 0, "hi": 0, "story_pass": False, "removed": None}

    def fn(l, g, h):
        m = ctx["mask"]
        if m is None or not (everywhere or ctx["story_pass"]):
            return None
        if not everywhere and h.shape[1] < ctx["hi"]:
            return None                            # a later decoding step: the story is already behind us
        lo, hi = (0, h.shape[1]) if everywhere else (ctx["lo"], ctx["hi"])
        mp, mn = m[0][l].to(h.device)[:, None, :], m[1][l].to(h.device)[:, None, :]          # [rows, 1, D]: which units are off in which row
        if not (mp.any() or mn.any()):
            return None
        sh, sg = h[:, lo:hi], g[:, lo:hi]
        off = (sg > 0) & (((sh > 0) & mp) | ((sh < 0) & mn))
        if mode != "all":
            t = torch.where(sh > 0, thr[0, l].to(h.device)[None, None, :], thr[1, l].to(h.device)[None, None, :])
            strong = sh.abs().float() >= t
            off &= strong if mode == "strong" else ~strong
        if ctx["removed"] is not None and off.shape[0] == len(ctx["removed"]):
            ctx["removed"] += (sh.abs().float() * off).sum((1, 2)).double().cpu()
        h = h.clone()
        h[:, lo:hi] = torch.where(off, torch.zeros_like(sh), sh)
        return h

    def masks(unit_sets):
        """[2, L, rows, D] bool: row r switches off unit_sets[r]."""
        m = np.zeros((2, L, len(unit_sets), D), dtype=bool)
        for r, units in enumerate(unit_sets):
            units = np.asarray(units, dtype=np.int64)
            if len(units):
                m[units // U, (units % U) // D, r, units % D] = True
        return torch.from_numpy(m)

    def measure(prompt_ids, answers, unit_sets):
        """One batched pass over the prompt, a different set off in each row; then each answer's own tokens from the cache."""
        B = len(unit_sets)
        with torch.no_grad():
            ctx.update(mask=masks(unit_sets), story_pass=True, removed=torch.zeros(B, dtype=torch.float64))
            first = model(input_ids=prompt_ids.repeat(B, 1), use_cache=True)
            ctx["story_pass"] = False
            removed = ctx["removed"].clone()
            lp0 = torch.log_softmax(first.logits[:, -1].float(), -1)                        # [B, vocabulary]
            each = []
            for a_ids in answers:
                lp = lp0[:, a_ids[0, 0]].clone()
                n = a_ids.shape[1]
                if n > 1:
                    more = model(input_ids=a_ids[:, :-1].repeat(B, 1), past_key_values=copy.deepcopy(first.past_key_values), use_cache=True)
                    l2 = torch.log_softmax(more.logits.float(), -1)                          # [B, n - 1, vocabulary]
                    lp = lp + l2[:, torch.arange(n - 1), a_ids[0, 1:]].sum(1)
                each.append(lp.cpu().numpy())
            del first
        each = np.stack(each, 1)                                                              # [B, answers]
        return [{"logp": float(np.logaddexp.reduce(each[r])), "logp_each": [round(float(x), 4) for x in each[r]], "removed": round(float(removed[r]), 2)}
                for r in range(B)]

    def reply(prompt_ids, units):
        with torch.no_grad():
            ctx.update(mask=masks([units]), story_pass=True, removed=None)
            gen = model.generate(prompt_ids, max_new_tokens=reply_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            ctx["story_pass"] = False
        return tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()

    tap = Tap(model, fn)
    out = {"model": tag, "question": jobs["question"], "where": "all" if everywhere else "story", "mode": mode,
           "set_sizes": {k: len(v) for k, v in sets.items()}, "items": []}
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
        names = ["untouched"] + list(item.get("conditions", []))
        unit_sets = [np.zeros(0, np.int64)] + [units_for(n, item) for n in names[1:]]
        got = []
        for s in range(0, len(names), batch):
            got += measure(p_ids, answers, unit_sets[s:s + batch])
        res = {"id": item["id"], "answers": item["answers"], "n_prompt_tokens": int(p_ids.shape[1]), "story_tokens": [ctx["lo"], ctx["hi"]],
               "untouched": dict(got[0], reply=reply(p_ids, unit_sets[0])), "conditions": {}}
        for n, r, units in zip(names[1:], got[1:], unit_sets[1:]):
            res["conditions"][n] = dict(r, reply=reply(p_ids, units)) if n in jobs.get("reply_for", []) else r
        out["items"].append(res)
        if k % 5 == 0 or k == len(jobs["items"]) - 1:
            print(f"{now()} {k + 1}/{len(jobs['items'])} {item['id'][:30]:30s} untouched {res['untouched']['logp']:.2f} reply {res['untouched']['reply']!r}", flush=True)
            json.dump(out, open(out_path, "w"), ensure_ascii=False)
    tap.remove()
    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    print(f"{now()} wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
