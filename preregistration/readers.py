"""LLM reading, NLI and embeddings for the pre-registration (frozen prompts; prereg-v2).

The LLM readers are the same local models as the neurons, in generation mode (Qwen3-8B with thinking off, Llama 3.1 8B
Instruct), so the comparison is reading against looking inside; Qwen3-1.7B is the small, cheap reader; Gemma 2 9B is a
cross-family CHECK reader for the data-generation protocols and is never scored as a method. Every scored answer is a
next-token probability: p_yes = P(yes) / (P(yes) + P(no)), summed over the tokenisations of yes/Yes and no/No, and
logit = log P(yes) - log P(no) (the lineage similarity). Generation is greedy.

Every method sees the same meme inputs: label, definition and the main example set (explicit + reworded examples);
the T3 judge uses the far set (--examples far).

Commands (from the project root; all write into preregistration/features/<set>/ unless noted):
  judge        --set S --model M [--tier A] [--examples main|far]   llm_<M>_judge.jsonl: story cached, one full question
                                                                    per meme (label, definition, examples)
  judge-shared --set S --model M        llm_<M>_judge-shared.jsonl: the meme sheet (every meme's label, definition and
                                        examples) cached ONCE for the whole set, then the story, then a short question
                                        per meme
  judge-1pass  --set S --model M        llm_<M>_judge-1pass.jsonl: sheet + story + one answer sheet read in ONE pass
  summarise    --set S --model M        llm_<M>_summary.jsonl     two-level recursive summary, in English
  sjudge       --set S --model M        llm_<M>_sjudge.jsonl      the full-question judge on the summary
  pairs        --rows S[:screen|primary|drift|all] --cands C[:comics|all] --model M   llm_<M>_pairs.jsonl (in S), from summaries
  pairsfull    --rows ... --cands ... --model M                   llm_<M>_pairsfull.jsonl, full texts, candidate cached
  likeness     --set S --model M        llm_<M>_likeness.jsonl    full masked text against 3 DEV Batman exemplar summaries
  recognise    --set S --model M        llm_<M>_recognise.jsonl   title or franchise of a name-free / masked text
  keywords     --model M|tfidf [--expanded]   t2_examples/keywords_<M>.json or keywords_expanded-<M>.json
  zh-keywords  --source SRC --model qwen3-8b  t2_examples/keywords_<SRC>_zh.json
  mt           --set S --model qwen3-8b       preregistration/data/<S>_mt.jsonl (Chinese to English, sentence-aligned)
  nli          --set S                        nli_deberta-v3-base.npy (sentence x meme entailment probability)
  embed        --set S --encoder E [--unit sentence|win3|doc]   emb_<E>[_win3|_doc].npy
  check        --task xread|xannotate|xretain|backtranslate --input FILE --output FILE   (cross-family check reader)
Each command writes cost_<command>_<model>.json: seconds (excluding model load), load_seconds, calls, tokens, MLX peak
memory, process max RSS, load averages at start and end.
"""
import argparse
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

PRE = Path(__file__).resolve().parent
sys.path.insert(0, str(PRE))
import analysis as A  # noqa: E402

MODELS = {"qwen3-8b": ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),       # the DEV study's snapshots
          "llama3.1-8b": ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "241a666dad6cb93c8ff213d39a7f34a36bf26db4"),
          "qwen3-1.7b": ("mlx-community/Qwen3-1.7B-4bit", "3b1b1768f8f8cf8351c712464f906e86c2b8269e"),
          "gemma2-9b": ("mlx-community/gemma-2-9b-it-4bit", "ff12eb39da2cd9b3b0f4b4f9ffb274603f05bb29")}
CHECK_READER = "gemma2-9b"
CHECK_READER_FALLBACK = ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "a4b8f870474b0eb527f466a03fbc187830d271f5")
ENCODERS = {"qwen3emb-4b": ("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b", "", "last"),
            "e5-large": ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3", "query: ", "mean"),
            "e5-base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", "query: ", "mean"),
            "labse": ("sentence-transformers/LaBSE", "836121a0533e5664b21c7aacc5d22951f2b8b25b", "", "st"),
            "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "", "mean")}
NLI_MODEL = ("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8")

# ------------------------------------------------------------------------------------------- frozen prompts
JUDGE_STORY = "Here is a plot summary of a story.\n\n<story>\n{text}\n</story>\n\n"
JUDGE_QUESTION = ("Question: does this story contain the following story element?\n"
                  "Element: {label}. {definition}\n"
                  "Examples of the element from other stories:\n{examples}\n"
                  "The element counts if it happens in this story, even when it is described in different words or is only "
                  "shown through events. Answer with one word: yes or no.")
SHEET_HEAD = ("Below is a list of story elements, each with an id, a definition and examples from other stories. An element "
              "counts if it happens in a story, even when it is described in different words or only shown through events.\n\n")
SHEET_ITEM = "[{id}] {label}. {definition}\nExamples:\n{examples}\n\n"
SHEET_STORY = "Here is a plot summary of a story.\n\n<story>\n{text}\n</story>\n\n"
SHEET_QUESTION = "Question: does this story contain element [{id}]? Answer with one word: yes or no."
ONEPASS_HEAD = "For each element id below, the answer (yes or no) says whether the story contains it.\n"
ONEPASS_LINE = "[{id}]:"
SUMMARY_CHUNK = ("Summarise this part of a plot summary in at most 60 words. Keep every event, relationship and motive. "
                 "Keep codes such as X1 unchanged. Do not add anything that is not in the text. Write the summary in English."
                 "\n\n{text}")
SUMMARY_MERGE = ("These are summaries of consecutive parts of one plot summary. Combine them into a single plot summary of at "
                 "most 150 words. Keep every event, relationship and motive. Keep codes such as X1 unchanged. Do not add "
                 "anything. Write the summary in English.\n\n{text}")
PAIR = ("Story A:\n{a}\n\nStory B:\n{b}\n\nQuestion: is story B an adaptation or retelling of story A, sharing its plot "
        "(possibly with different names, setting or details)? Answer with one word: yes or no.")
PAIR_FULL_A = "Story A:\n{a}\n\n"
PAIR_FULL_B = ("Story B:\n{b}\n\nQuestion: is story B an adaptation or retelling of story A, sharing its plot (possibly with "
               "different names, setting or details)? Answer with one word: yes or no.")
LIKENESS_HEAD = ("Here are three plot summaries of stories of one kind. Names have been replaced by codes such as X1.\n\n"
                 "Story 1:\n{e1}\n\nStory 2:\n{e2}\n\nStory 3:\n{e3}\n\n")
LIKENESS_Q = ("Here is another plot summary (names replaced by codes):\n<story>\n{text}\n</story>\n\nQuestion: apart from the "
              "names, is this story of the same kind as the three stories above? Answer with one word: yes or no.")
LIKENESS_SALT = "phylomemetics-likeness-v2|"
RECOGNISE = ("Here is a plot summary in which names have been replaced by codes such as X1.\n\n<story>\n{text}\n</story>\n\n"
             "Which work is this, or which franchise does it belong to? If you do not know, answer 'unknown'. Answer in at most "
             "ten words.")
KEYWORDS = ("You are an analyst who must find every plot summary that contains a story element, using keyword search.\n"
            "Element: {label}. {definition}\nExamples of the element:\n{examples}\n\n"
            "List up to 10 search keywords or short phrases (one to three words each) that would find this element in other "
            "plot summaries. Reply with the keywords only, one per line.")
KEYWORDS_EXPANDED = ("You are a content-moderation analyst. You must find every plot summary that contains a story element, "
                     "using keyword search, and you know that many texts will AVOID the usual words for it: they use "
                     "synonyms, euphemisms, slang, indirect descriptions, or only describe what happens.\n"
                     "Element: {label}. {definition}\nExamples of the element:\n{examples}\n\n"
                     "List 30 to 40 search terms, mostly single words, that together would catch the element however it is "
                     "worded: the usual words, synonyms, euphemisms, slang and words for the actions, objects and situations "
                     "that typically come with it. Reply with the terms only, one per line.")
KEYWORDS_REVISE = ("Here is your list with how often each term matches the {k} example sentences of the element and {n} "
                   "ordinary plot-summary sentences that do NOT contain it:\n{table}\n\nExample sentences that no term "
                   "matches:\n{missed}\n\nRevise the list once: drop terms that match many ordinary sentences, add terms that "
                   "would catch the unmatched examples and other wordings. Reply with 30 to 40 terms, one per line, and "
                   "nothing else.")
ZH_KEYWORDS = ("Translate each of these search keywords into Simplified Chinese as the words would appear in a Chinese plot "
               "summary. Reply with one translation per line, in the same order, and nothing else.\n\n{terms}")
MT = "Translate this Chinese sentence into English. Keep codes such as X1 unchanged. Reply with the translation only.\n\n{text}"
NLI_HYPOTHESIS = "This text describes the following: {label}. {definition}"
# cross-family check reader (protocol checks only; never a method)
XREAD = ("Read this plot summary. Below is a list of story elements. List every element that happens in the story, however "
         "it is described, and only those. Answer with the element ids, one per line, or 'none'.\n<story>{story}</story>\n{items}")
XRETAIN = ("For each numbered pair, does the REWRITE tell the same events, relationships and motives as the ORIGINAL? Answer "
           "with one line per pair: the number, then same, partly or different.\n{pairs}")
BACKTRANSLATE = "Translate this Chinese sentence into English, sentence for sentence. Keep codes such as X1. Reply with the translation only.\n\n{text}"
CHUNK_SENTENCES = 8


# ------------------------------------------------------------------------------------------- model helpers
def chat(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


MLX_FILES = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.txt", "*.jsonl", "*.jinja"]   # the files mlx_lm loads


def _local(repo, rev):
    """The local path of a pinned snapshot (offline after step 0; only the files mlx_lm loads are required)."""
    if rev is None:
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev, allow_patterns=MLX_FILES)


def cost_stamp():
    return {"loadavg": list(os.getloadavg()), "time": time.strftime("%Y-%m-%d %H:%M:%S")}


class Reader:
    def __init__(self, model_tag):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.tag = model_tag
        t0 = time.time()
        repo, rev = MODELS[model_tag]
        try:
            self.model, self.tok = load(_local(repo, rev))
        except Exception:
            if model_tag != CHECK_READER:
                raise
            self.model, self.tok = load(_local(*CHECK_READER_FALLBACK))
            self.tag = "mistral7b-fallback"
        self.load_seconds = time.time() - t0
        self.hf = self.tok._tokenizer
        self.yes = self._variants(["yes", "Yes", " yes", " Yes", "YES"])
        self.no = self._variants(["no", "No", " no", " No", "NO"])
        self.calls = 0
        self.tokens = 0
        self.start = cost_stamp()

    def _variants(self, words):
        ids = set()
        for w in words:
            t = self.hf.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        return sorted(ids)

    def encode(self, text):
        return self.hf.encode(text, add_special_tokens=False)

    def _logits(self, ids, cache=None, all_positions=False):
        out = self.model(self.mx.array([ids]), cache=cache)
        return out[0].astype(self.mx.float32) if all_positions else out[0, -1].astype(self.mx.float32)

    def yes_no(self, logits):
        """(p_yes, logit) from one position's logits."""
        lp = logits - self.mx.logsumexp(logits)
        self.mx.eval(lp)
        lp = np.array(lp)
        y = np.logaddexp.reduce(lp[self.yes])
        n = np.logaddexp.reduce(lp[self.no])
        return float(1 / (1 + math.exp(n - y))), float(y - n)

    def ask(self, prompt_text):
        ids = self.encode(chat(self.tok, prompt_text))
        self.calls += 1
        self.tokens += len(ids)
        return self.yes_no(self._logits(ids))

    def ask_many(self, prefix_text, suffixes):
        """One prefix (cached once) and many question suffixes (the cache is trimmed back after each)."""
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        full0 = chat(self.tok, prefix_text + suffixes[0])
        cut = full0.index(prefix_text) + len(prefix_text)
        pre_ids = self.encode(full0[:cut])
        t0 = time.time()
        cache = make_prompt_cache(self.model)
        self.mx.eval(self._logits(pre_ids, cache))
        prefill = time.time() - t0
        self.tokens += len(pre_ids)
        out, per = [], []
        for s in suffixes:
            ids = self.encode(chat(self.tok, prefix_text + s))
            t1 = time.time()
            if ids[:len(pre_ids)] == pre_ids:
                suf = ids[len(pre_ids):]
                out.append(self.yes_no(self._logits(suf, cache)))
                trim_prompt_cache(cache, len(suf))
                self.tokens += len(suf)
            else:
                out.append(self.yes_no(self._logits(ids)))
                self.tokens += len(ids)
            self.calls += 1
            per.append(time.time() - t1)
        return out, prefill, per

    def generate(self, prompt_text, max_tokens):
        from mlx_lm import generate
        prompt = chat(self.tok, prompt_text)
        self.calls += 1
        self.tokens += len(self.encode(prompt))
        return generate(self.model, self.tok, prompt, max_tokens=max_tokens, verbose=False).strip()

    def cost(self, t_work, **extra):
        return {"seconds": t_work, "load_seconds": self.load_seconds, "calls": self.calls, "tokens": self.tokens,
                "peak_gb_mlx": float(self.mx.get_peak_memory() / 1e9), "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                "start": self.start, "end": cost_stamp(), "reader": self.tag, **extra}


# ------------------------------------------------------------------------------------------- tasks
def meme_list(tier=None):
    memes = json.load(open(PRE / "memes_for_tracing.json"))["memes"]
    return [m for m in memes if tier is None or m["tier"] == tier]


def examples_of(m, which="main"):
    sets = ("explicit", "reworded") if which == "main" else ("far",)
    return [e for s in sets for e in m["examples"].get(s, [])]


def question(m, which="main"):
    ex = "\n".join(f"- {e}" for e in examples_of(m, which))
    return JUDGE_QUESTION.format(label=m["label"], definition=m["definition"], examples=ex)


def sheet(memes, which="main"):
    return SHEET_HEAD + "".join(SHEET_ITEM.format(id=m["id"], label=m["label"], definition=m["definition"],
                                                  examples="\n".join(f"- {e}" for e in examples_of(m, which))) for m in memes)


def story_text(sentences):
    return " ".join(sentences)


def _write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_judge(set_name, model, tier=None, on_summary=False, which="main"):
    S = A.Set(set_name)
    memes = meme_list(tier)
    rd = Reader(model)
    task = "sjudge" if on_summary else "judge"
    out_path = S.feat / f"llm_{model}_{task}.jsonl"
    done = {(r["work"], r["meme"]) for r in A.read_jsonl(out_path)} if out_path.exists() else set()
    summaries = {r["work"]: r["summary"] for r in (S.llm(model, "summary") or [])} if on_summary else {}
    S.feat.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prefill_total, per_all = 0.0, []
    for w in S.works:
        todo = [m for m in memes if (w["id"], m["id"]) not in done]
        if not todo:
            continue
        text = summaries[w["id"]] if on_summary else story_text(w["sentences"])
        res, prefill, per = rd.ask_many(JUDGE_STORY.format(text=text), [question(m, which) for m in todo])
        prefill_total += prefill
        per_all += per
        _write_rows(out_path, [{"work": w["id"], "meme": m["id"], "p_yes": p, "logit": lg} for m, (p, lg) in zip(todo, res)])
    json.dump(rd.cost(time.time() - t0, prefill_seconds=prefill_total, per_meme_seconds=float(np.mean(per_all)) if per_all else None,
                      n_memes=len(memes), n_works=len(S.works)), open(S.feat / f"cost_{task}_{model}.json", "w"))


def run_judge_shared(set_name, model, one_pass=False, tier=None, which="main"):
    """The cheapest reasonable LLM judge: the meme sheet is cached once for the whole set; each story is appended to
    it; then either one short question per meme (cache trimmed after each) or, with one_pass, one answer sheet read in
    a single forward pass (the logits after each '[id]:' marker, with the filler token ' ?' fed in its place)."""
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    S = A.Set(set_name)
    memes = meme_list(tier)
    rd = Reader(model)
    task = "judge-1pass" if one_pass else "judge-shared"
    out_path = S.feat / f"llm_{model}_{task}.jsonl"
    done = {r["work"] for r in A.read_jsonl(out_path)} if out_path.exists() else set()
    sh = sheet(memes, which)
    probe = chat(rd.tok, sh + "XXSTORYXX")
    head_ids = rd.encode(probe[:probe.index(sh) + len(sh)])
    t0 = time.time()
    cache = make_prompt_cache(rd.model)
    rd.mx.eval(rd._logits(head_ids, cache))
    sheet_seconds = time.time() - t0
    rd.tokens += len(head_ids)
    t1 = time.time()
    for w in S.works:
        if w["id"] in done:
            continue
        story = SHEET_STORY.format(text=story_text(w["sentences"]))
        rows = []
        if not one_pass:
            full0 = chat(rd.tok, sh + story + SHEET_QUESTION.format(id=memes[0]["id"]))
            mid_ids = rd.encode(full0[:full0.index(story) + len(story)])
            ok = mid_ids[:len(head_ids)] == head_ids
            st_ids = mid_ids[len(head_ids):]
            rd.mx.eval(rd._logits(st_ids, cache))
            rd.tokens += len(st_ids)
            for m in memes:
                ids = rd.encode(chat(rd.tok, sh + story + SHEET_QUESTION.format(id=m["id"])))
                if ok and ids[:len(mid_ids)] == mid_ids:
                    suf = ids[len(mid_ids):]
                    p, lg = rd.yes_no(rd._logits(suf, cache))
                    trim_prompt_cache(cache, len(suf))
                    rd.tokens += len(suf)
                else:
                    p, lg = rd.ask(sh + story + SHEET_QUESTION.format(id=m["id"]))
                rd.calls += 1
                rows.append({"work": w["id"], "meme": m["id"], "p_yes": p, "logit": lg})
            trim_prompt_cache(cache, len(st_ids))
        else:
            body = story + ONEPASS_HEAD
            marks = []
            for m in memes:
                body += ONEPASS_LINE.format(id=m["id"])
                marks.append(len(body))
                body += " ?\n"
            full = chat(rd.tok, sh + body)
            ids = rd.encode(full)
            base = full.index(sh) + len(sh)
            pos = [len(rd.encode(full[:base + k])) - 1 for k in marks]
            if ids[:len(head_ids)] == head_ids:
                L = rd._logits(ids[len(head_ids):], cache, all_positions=True)
                off = len(head_ids)
                for m, pp in zip(memes, pos):
                    p, lg = rd.yes_no(L[pp - off])
                    rows.append({"work": w["id"], "meme": m["id"], "p_yes": p, "logit": lg})
                trim_prompt_cache(cache, len(ids) - len(head_ids))
            rd.calls += 1
            rd.tokens += len(ids) - len(head_ids)
        _write_rows(out_path, rows)
    json.dump(rd.cost(time.time() - t1, sheet_seconds=sheet_seconds, n_memes=len(memes), n_works=len(S.works)),
              open(S.feat / f"cost_{task}_{model}.json", "w"))


def summarise(rd, sentences):
    chunks = [sentences[i:i + CHUNK_SENTENCES] for i in range(0, len(sentences), CHUNK_SENTENCES)]
    parts = [rd.generate(SUMMARY_CHUNK.format(text=" ".join(c)), 120) for c in chunks]
    while len(parts) > 6:                                      # long texts: merge in groups of 6 until one remains
        parts = [rd.generate(SUMMARY_MERGE.format(text="\n".join(parts[i:i + 6])), 260) for i in range(0, len(parts), 6)]
    if len(parts) == 1:
        return parts[0], parts
    return rd.generate(SUMMARY_MERGE.format(text="\n".join(parts)), 260), parts


def run_summaries(set_name, model):
    S = A.Set(set_name)
    rd = Reader(model)
    out_path = S.feat / f"llm_{model}_summary.jsonl"
    done = {r["work"] for r in A.read_jsonl(out_path)} if out_path.exists() else set()
    S.feat.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for w in S.works:
        if w["id"] in done:
            continue
        t1 = time.time()
        s, parts = summarise(rd, w["sentences"])
        _write_rows(out_path, [{"work": w["id"], "summary": s, "parts": parts, "seconds": time.time() - t1, "has_cjk": A.is_cjk(s)}])
    json.dump(rd.cost(time.time() - t0, n_works=len(S.works)), open(S.feat / f"cost_summarise_{model}.json", "w"))


def _select(S, how):
    if how in (None, "all"):
        return list(range(len(S.ids)))
    if how == "screen":
        return A._screen(S)
    if how == "comics":
        return A._comics(S)
    if how == "primary":
        return [i for i, w in enumerate(S.works) if w.get("primary")]
    if how == "drift":
        return [i for i, w in enumerate(S.works) if w.get("drift_sample")]
    raise ValueError(how)


def run_pairs(rows_spec, cands_spec, model, full=False):
    """LLM lineage reading. Summary pairs (recursive-summarisation arm): both works' summaries in one prompt. Full-text
    pairs: the candidate's full text is the cached prefix and every row's full text is appended (candidate first)."""
    rname, _, rhow = rows_spec.partition(":")
    cname, _, chow = cands_spec.partition(":")
    R, C = A.Set(rname), A.Set(cname)
    rd = Reader(model)
    task = "pairsfull" if full else "pairs"
    out_path = R.feat / f"llm_{model}_{task}.jsonl"
    done = {(r["a"], r["b"]) for r in A.read_jsonl(out_path)} if out_path.exists() else set()
    rows_i, cands_i = _select(R, rhow or None), _select(C, chow or None)
    t0 = time.time()
    if not full:
        rs = {r["work"]: r["summary"] for r in R.llm(model, "summary")}
        cs = {r["work"]: r["summary"] for r in C.llm(model, "summary")}
        for i in rows_i:
            res = []
            for j in cands_i:
                a, b = R.ids[i], C.ids[j]
                if (a, b) in done or a == b:
                    continue
                p, lg = rd.ask(PAIR.format(a=cs[b], b=rs[a]))
                res.append({"a": a, "b": b, "p_yes": p, "logit": lg})
            _write_rows(out_path, res)
    else:
        for j in cands_i:
            b = C.ids[j]
            todo = [i for i in rows_i if (R.ids[i], b) not in done and R.ids[i] != b]
            if not todo:
                continue
            prefix = PAIR_FULL_A.format(a=story_text(C.works[j]["sentences"]))
            res, _, _ = rd.ask_many(prefix, [PAIR_FULL_B.format(b=story_text(R.works[i]["sentences"])) for i in todo])
            _write_rows(out_path, [{"a": R.ids[i], "b": b, "p_yes": p, "logit": lg} for i, (p, lg) in zip(todo, res)])
    json.dump(rd.cost(time.time() - t0), open(R.feat / f"cost_{task}_{model}_{cname}.json", "w"))


def run_likeness(set_name, model, exemplar_set="t3_dev_masked"):
    """T3 LLM likeness with the same reference the other methods get: three DEV Batman works (the first three in a fixed
    sha256 order) given as the reader's own recursive summaries; the target is read in full (masked)."""
    import hashlib
    S, E = A.Set(set_name), A.Set(exemplar_set)
    es = {r["work"]: r["summary"] for r in E.llm(model, "summary")}
    bats = sorted([w["id"] for w in E.works if w.get("group") == "batman"], key=lambda t: hashlib.sha256((LIKENESS_SALT + t).encode()).hexdigest())[:3]
    head = LIKENESS_HEAD.format(e1=es[bats[0]], e2=es[bats[1]], e3=es[bats[2]])
    rd = Reader(model)
    t0 = time.time()
    out_path = S.feat / f"llm_{model}_likeness.jsonl"
    res, _, _ = rd.ask_many(head, [LIKENESS_Q.format(text=story_text(w["sentences"])) for w in S.works])
    _write_rows(out_path, [{"work": w["id"], "p_yes": p, "logit": lg, "exemplars": bats} for w, (p, lg) in zip(S.works, res)])
    json.dump(rd.cost(time.time() - t0), open(S.feat / f"cost_likeness_{model}.json", "w"))


def run_recognise(set_name, model):
    S = A.Set(set_name)
    rd = Reader(model)
    t0 = time.time()
    _write_rows(S.feat / f"llm_{model}_recognise.jsonl",
                [{"work": w["id"], "answer": rd.generate(RECOGNISE.format(text=story_text(w["sentences"])), 24)} for w in S.works])
    json.dump(rd.cost(time.time() - t0), open(S.feat / f"cost_recognise_{model}.json", "w"))


def parse_list(text, n=10, max_words=3):
    out = []
    for line in text.splitlines():
        t = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if t and len(t.split()) <= max_words and t.lower() not in [x.lower() for x in out]:
            out.append(t)
    return out[:n]


def run_keywords(source, expanded=False):
    """The analyst's keyword query per meme, from label, definition and the main examples. Plain: up to 10 terms.
    Expanded (evasion-aware, H1's rival): 30-40 mostly single-word terms, then one revision after seeing each term's
    hit counts on the examples and on the DEV reference-film sentences; WordNet expansion is added at F1."""
    memes = meme_list()
    out_dir = A.FEAT / "t2_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bg = A.Set("t2_background").sentences
    name = f"expanded-{source}" if expanded else source
    if source == "tfidf":
        raise SystemExit("the TF-IDF queries are computed by analysis.py dev-freeze")
    rd = Reader(source)
    kw = {}
    for m in memes:
        ex = examples_of(m)
        exs = "\n".join(f"- {e}" for e in ex)
        if not expanded:
            kw[m["id"]] = parse_list(rd.generate(KEYWORDS.format(label=m["label"], definition=m["definition"], examples=exs), 120))
            continue
        draft = parse_list(rd.generate(KEYWORDS_EXPANDED.format(label=m["label"], definition=m["definition"], examples=exs), 400), n=40)
        cache_ex, cache_bg = A.StemCache(ex), A.StemCache(bg)
        table = "\n".join(f"{t}: {sum(cache_ex.hits(i, [t]) > 0 for i in range(len(ex)))} examples, "
                          f"{sum(cache_bg.hits(i, [t]) > 0 for i in range(len(bg)))} ordinary" for t in draft)
        missed = "\n".join(f"- {e}" for i, e in enumerate(ex) if not cache_ex.hits(i, draft)) or "(none)"
        rev = rd.generate(KEYWORDS_EXPANDED.format(label=m["label"], definition=m["definition"], examples=exs) + "\n\n"
                          + KEYWORDS_REVISE.format(k=len(ex), n=len(bg), table=table, missed=missed), 400)
        kw[m["id"]] = parse_list(rev, n=40) or draft
    cost = rd.cost(time.time() - t0)
    json.dump(kw, open(out_dir / f"keywords_{name}.json", "w"), indent=1, ensure_ascii=False)
    json.dump(cost, open(out_dir / f"cost_keywords_{name}.json", "w"))


def run_zh_keywords(source, model="qwen3-8b"):
    out_dir = A.FEAT / "t2_examples"
    kw = json.load(open(out_dir / f"keywords_{source}.json"))
    rd = Reader(model)
    zh = {}
    for mid, terms in kw.items():
        lines = [l.strip() for l in rd.generate(ZH_KEYWORDS.format(terms="\n".join(terms)), 400).splitlines() if l.strip()]
        zh[mid] = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", l) for l in lines][:len(terms)]
    json.dump(zh, open(out_dir / f"keywords_{source}_zh.json", "w"), indent=1, ensure_ascii=False)


def run_mt(set_name, model="qwen3-8b"):
    S = A.Set(set_name)
    rd = Reader(model)
    t0 = time.time()
    with open(A.DATA / f"{set_name}_mt.jsonl", "w") as f:
        for w in S.works:
            f.write(json.dumps({**w, "sentences": [rd.generate(MT.format(text=s), 160) for s in w["sentences"]], "mt_from": set_name},
                               ensure_ascii=False) + "\n")
    json.dump(rd.cost(time.time() - t0), open(S.feat / f"cost_mt_{model}.json", "w"))


def run_nli(set_name, batch=64):
    """Zero-shot NLI baseline: entailment probability of 'This text describes the following: <label>. <definition>'
    for every sentence and meme (DeBERTa-v3 zero-shot, pinned revision); story score = top-3 mean (analysis.py)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    repo, rev = NLI_MODEL
    S = A.Set(set_name)
    memes = meme_list()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    mdl = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).to(dev).eval()
    load = time.time() - t0
    ent = next(i for i, l in mdl.config.id2label.items() if l.lower().startswith("entail"))
    out = np.zeros((len(S.sentences), len(memes)), np.float16)
    t1 = time.time()
    with torch.no_grad():
        for j, m in enumerate(memes):
            hyp = NLI_HYPOTHESIS.format(label=m["label"], definition=m["definition"])
            for i in range(0, len(S.sentences), batch):
                b = tok(S.sentences[i:i + batch], [hyp] * len(S.sentences[i:i + batch]), padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(dev)
                pr = torch.softmax(mdl(**b).logits, -1)[:, ent]
                out[i:i + batch, j] = pr.cpu().numpy()
    S.feat.mkdir(parents=True, exist_ok=True)
    np.save(S.feat / "nli_deberta-v3-base.npy", out)
    json.dump({"seconds": time.time() - t1, "load_seconds": load, "pairs": int(out.size), "memes": [m["id"] for m in memes],
               "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, **cost_stamp()},
              open(S.feat / "cost_nli.json", "w"))


# ------------------------------------------------------------------------------------------- embeddings
def _units_text(S, unit):
    if unit == "sentence":
        return S.sentences
    if unit == "doc":
        return [" ".join(w["sentences"]) for w in S.works]
    out = []
    for w in S.works:
        s = w["sentences"]
        out += [" ".join(s[max(0, k - 1):k + 2]) for k in range(len(s))]
    return out


def run_embed(set_name, enc, unit="sentence", batch=32):
    """L2-normalised embeddings of sentences, 3-sentence windows centred on each sentence, or whole works. e5 and MiniLM
    use mean pooling, Qwen3-Embedding last-token pooling (left padding, no instruction: both sides are documents),
    LaBSE its sentence-transformers pooling. Maximum length: 512 tokens (sentences, windows), 2,048 (Qwen3 documents)."""
    import torch
    repo, rev, prefix, pool = ENCODERS[enc]
    S = A.Set(set_name)
    texts = [prefix + t for t in _units_text(S, unit)]
    max_length = 2048 if (unit == "doc" and pool == "last") else 512
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    if pool == "st":
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(repo, revision=rev, device=dev)
        st.max_seq_length = max_length
        load = time.time() - t0
        t1 = time.time()
        E = st.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    else:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, padding_side="left" if pool == "last" else "right")
        dtype = torch.float16 if pool == "last" else torch.float32
        mdl = AutoModel.from_pretrained(repo, revision=rev, torch_dtype=dtype).to(dev).eval()
        load = time.time() - t0
        t1 = time.time()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
                h = mdl(**b).last_hidden_state
                if pool == "last":
                    e = h[:, -1]
                else:
                    msk = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                    e = (h * msk).sum(1) / msk.sum(1)
                out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        E = np.concatenate(out) if out else np.zeros((0, mdl.config.hidden_size))
    S.feat.mkdir(parents=True, exist_ok=True)
    suffix = {"sentence": "", "win3": "_win3", "doc": "_doc"}[unit]
    np.save(S.feat / f"emb_{enc}{suffix}.npy", E.astype(np.float16))
    sec = time.time() - t1
    meta = {"model": repo, "revision": rev, "prefix": prefix, "pooling": pool, "unit": unit, "seconds": sec, "load_seconds": load,
            "n": len(texts), "max_length": max_length, "device": dev, "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
            **cost_stamp()}
    json.dump(meta, open(S.feat / f"emb_{enc}{suffix}.json", "w"))
    json.dump({**meta, "bytes": int(E.astype(np.float16).nbytes)}, open(S.feat / f"cost_emb_{enc}{suffix}.json", "w"))


# ------------------------------------------------------------------------------------------- cross-family check reader
def run_check(task, input_file, output_file):
    """The cross-family check reader (Gemma 2 9B; Mistral 7B if Gemma cannot load) for the protocol checks:
    xread (Q4/Q5 second family: which listed elements a story contains), xannotate (A2 work-level presence),
    xretain (D3 retention of drift pairs), backtranslate (L2). input_file: one JSON object per line."""
    rd = Reader(CHECK_READER)
    rows = A.read_jsonl(input_file)
    out = []
    for r in rows:
        if task in ("xread", "xannotate"):
            items = "\n".join(f"{it['id']}: {it['label']}. {it['definition']}" for it in r["items"])
            text = rd.generate(XREAD.format(story=r["text"], items=items), 200)
            ids = {it["id"] for it in r["items"]}
            out.append({**{k: v for k, v in r.items() if k not in ("text", "items")}, "family": rd.tag,
                        "memes_found": sorted({t for t in re.findall(r"[a-z_]+", text) if t in ids})})
        elif task == "xretain":
            pairs = "\n".join(f"{k + 1}. ORIGINAL: {a}\n   REWRITE: {b}" for k, (a, b) in enumerate(zip(r["original"], r["rewrite"])))
            text = rd.generate(XRETAIN.format(pairs=pairs), 20 * len(r["original"]) + 40)
            out.append({"id": r["id"], "family": rd.tag, "verdicts": re.findall(r"\b(same|partly|different)\b", text.lower())})
        elif task == "backtranslate":
            out.append({"id": r["id"], "family": rd.tag, "sentences": [rd.generate(BACKTRANSLATE.format(text=s), 160) for s in r["sentences"]]})
    with open(output_file, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "judge-shared", "judge-1pass", "sjudge", "summarise", "pairs", "pairsfull", "likeness",
                                    "recognise", "keywords", "zh-keywords", "mt", "nli", "embed", "check"])
    ap.add_argument("--set")
    ap.add_argument("--rows")
    ap.add_argument("--cands")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--source")
    ap.add_argument("--tier")
    ap.add_argument("--examples", default="main", choices=["main", "far"])
    ap.add_argument("--encoder")
    ap.add_argument("--unit", default="sentence", choices=["sentence", "win3", "doc"])
    ap.add_argument("--expanded", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--input")
    ap.add_argument("--output")
    a = ap.parse_args()
    if a.cmd == "judge":
        run_judge(a.set, a.model, a.tier, which=a.examples)
    elif a.cmd in ("judge-shared", "judge-1pass"):
        run_judge_shared(a.set, a.model, one_pass=a.cmd == "judge-1pass", tier=a.tier, which=a.examples)
    elif a.cmd == "sjudge":
        run_judge(a.set, a.model, a.tier, on_summary=True, which=a.examples)
    elif a.cmd == "summarise":
        run_summaries(a.set, a.model)
    elif a.cmd in ("pairs", "pairsfull"):
        run_pairs(a.rows, a.cands, a.model, full=a.cmd == "pairsfull")
    elif a.cmd == "likeness":
        run_likeness(a.set, a.model)
    elif a.cmd == "recognise":
        run_recognise(a.set, a.model)
    elif a.cmd == "keywords":
        run_keywords(a.model, a.expanded)
    elif a.cmd == "zh-keywords":
        run_zh_keywords(a.source, a.model)
    elif a.cmd == "mt":
        run_mt(a.set, a.model)
    elif a.cmd == "nli":
        run_nli(a.set)
    elif a.cmd == "embed":
        run_embed(a.set, a.encoder, a.unit)
    elif a.cmd == "check":
        run_check(a.task, a.input, a.output)


if __name__ == "__main__":
    main()
