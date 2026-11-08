"""Name-free versions of a new corpus, by the Batman corpus's procedure (run from study/).

    ../.venv/bin/python v2/second/names.py selftest               the code step reproduces data/corpus_neutral.jsonl from data/anon
    ../.venv/bin/python v2/second/names.py items <corpus-dir>     prompt files for the rewriting agents
    ../.venv/bin/python v2/second/names.py collect <corpus-dir>   check the rewrites, write anon/ and corpus_neutral.jsonl, list what to redo
    ../.venv/bin/python v2/second/names.py redo <corpus-dir>      prompt files for the works that failed a check

<corpus-dir> holds corpus.jsonl (for example data/second/holmes). As for Batman, an AI rewrite is used only to FIND
the names: every span the rewriter changed becomes a code X1, X2, ... (the rule of study/make_neutral.py, copied here
as neutralise() because that script cannot be imported). The rewriting prompt is v2/prompts/name_remover.md, the
wording of preregistration/name_removal_protocol.md. Unlike make_neutral.py this script refuses a rewrite whose
sentence count differs, or that leaves more than two words it replaced elsewhere in the same work (or words on the
franchise's name list), or that changed more than 12% of the lower-case words.
"""
import difflib
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "v2/analysis")
from explain_agents import ANSWER, wrapped  # noqa: E402

PER_FILE = 8
CODE = re.compile(r"\bX\d+\b")
# capitalised words that are not names: months, days, nationalities and the like may stay (the checker's notes for the Batman corpus)
NOT_NAMES = set("""January February March April May June July August September October November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday
Christmas Easter Halloween Thanksgiving English British American French German Italian Spanish Russian Chinese Japanese Indian Irish Scottish Welsh Dutch Greek Turkish
Polish Swedish Swiss Austrian Hungarian Egyptian Mexican Canadian Australian African European Asian Arab Jewish Christian Catholic Protestant Muslim Hindu Buddhist
God Bible I Mr Mrs Miss Ms Dr Sir Lady Lord Professor Inspector Colonel Captain Major General Sergeant Lieutenant Father Mother Brother Sister King Queen Prince Princess
Duke Duchess Count Countess Baron Baroness Doctor Detective Constable Reverend Judge President Senator Governor Mayor Chief Commissioner Admiral Commander Emperor Empress
Victorian Edwardian Nazi Nazis Communist World War II TV FBI CIA US UK USA DNA CEO MI5 MI6 KGB OK""".split())


def key(words):
    return re.sub(r"[^\w\s-]", "", re.sub(r"'s\b|’s\b", "", " ".join(words))).strip().lower()


def neutralise(original, rewrite):
    """Sentences with every span the rewriter changed replaced by a code; the rule of study/make_neutral.py."""
    codes, out, n_spans = {}, [], 0
    for a, b in zip(original, rewrite):
        A, B = a.split(), b.split()
        toks = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=A, b=B, autojunk=False).get_opcodes():
            if op == "equal":
                toks += A[i1:i2]
            elif op == "delete":
                continue
            else:
                src = A[i1:i2] if i2 > i1 else B[j1:j2]
                k = key(src) or key(B[j1:j2])
                code = codes.setdefault(k, f"X{len(codes) + 1}")
                last = (A[i2 - 1] if i2 > i1 else B[j2 - 1])
                poss = "'s" if re.search(r"(’s|'s)\W*$", B[j2 - 1] if j2 > j1 else last) else ""
                tail = re.search(r"[.,;:!?)\"”]*$", B[j2 - 1] if j2 > j1 else last).group(0)
                lead = re.match(r"^[(\"“]*", B[j1] if j2 > j1 else A[i1]).group(0)
                toks.append(f"{lead}{code}{poss}{tail}")
                n_spans += 1
        out.append(re.sub(r"\s+", " ", " ".join(toks)).strip())
    return out, len(codes), n_spans


def selftest():
    orig = [json.loads(l) for l in open("data/corpus.jsonl")]
    want = [json.loads(l) for l in open("data/corpus_neutral.jsonl")]
    anon = {}
    for f in sorted(glob.glob("data/anon/batch_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            anon[r["id"]] = r["sentences"]
    same = sum(neutralise(w["sentences"], anon[w["id"]])[0] == v["sentences"] and neutralise(w["sentences"], anon[w["id"]])[1] == v["n_codes"] for w, v in zip(orig, want))
    print("the code step reproduces", same, "of", len(orig), "works of data/corpus_neutral.jsonl")
    assert same == len(orig)


def load(corpus_dir):
    return [json.loads(l) for l in open(Path(corpus_dir) / "corpus.jsonl")]


def folder(corpus_dir):
    d = Path("v2/cache/second") / Path(corpus_dir).name / "names"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def write_prompts(works, d, rnd, note=None):
    template = Path("v2/prompts/name_remover.md").read_text()
    files = []
    for b in range(0, len(works), PER_FILE):
        blocks = []
        for w in works[b:b + PER_FILE]:
            lines = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(w["sentences"]))
            extra = f"\n(Note for this summary: {note[w['id']]})" if note and w["id"] in note else ""
            blocks.append(f"### SUMMARY id {w['id']} ({len(w['sentences'])} sentences){extra}\n{lines}")
        name = f"r{rnd}_batch_{b // PER_FILE:03d}"
        text = template.replace("{items}", "\n\n".join(blocks)) + ANSWER.format(out=d / f"{name}.out.json", what="summary")
        (d / f"{name}.prompt.txt").write_text(wrapped(text))
        files.append(name)
    json.dump(files, open(d / f"r{rnd}_prompts.json", "w"))
    return files


def items(corpus_dir):
    works = load(corpus_dir)
    assert not any(CODE.search(s) for w in works for s in w["sentences"]), "an original text holds something that looks like a code"
    d = folder(corpus_dir)
    files = write_prompts(works, d, 1)
    print(len(works), "works,", sum(len(w["sentences"]) for w in works), "sentences ->", len(files), "prompt files in", d)


def answers(d):
    """Every answered rewrite, later rounds over earlier ones."""
    got = {}
    for f in sorted(d.glob("r*_batch_*.out.json")):
        try:
            data = json.load(open(f))
        except ValueError as e:
            print("unreadable answer file", f.name, str(e)[:80])
            continue
        if isinstance(data, dict):
            data = data.get("summaries") or data.get("items") or [data]
        for r in data:
            try:
                got[int(r["id"])] = [str(s) for s in r["rewrite"]]
            except (KeyError, ValueError, TypeError):
                print("unusable row in", f.name)
    return got


def leftovers(w, coded, names):
    """Words of a coded text that still look like names: capitalised, not first in the sentence, not a code, not a calendar or nationality word."""
    out = []
    for k, s in enumerate(coded):
        toks = s.split()
        for i, t in enumerate(toks):
            word = re.sub(r"^[^\w]+|[^\w]+$", "", t)
            word = re.sub(r"('s|’s)$", "", word)
            if not word or CODE.fullmatch(word):
                continue
            if names.search(word):
                out.append((k + 1, word))
            elif i > 0 and word[0].isupper() and word not in NOT_NAMES and not word.isupper() and not re.match(r"^[\"“(]", toks[i]) and not toks[i - 1].endswith((".", "!", "?", ":", '"', "”")):
                out.append((k + 1, word))
    return out


def collect(corpus_dir, name_list="holmes|sherlock|watson|moriarty|lestrade|mycroft|baskerville|adler|hudson|doyle"):
    works = load(corpus_dir)
    d = folder(corpus_dir)
    got = answers(d)
    names = re.compile(name_list, re.I)
    ok, redo, stats = {}, {}, []
    for w in works:
        r = got.get(w["id"])
        if r is None:
            redo[w["id"]] = "no rewrite was returned"
            continue
        if len(r) != len(w["sentences"]):
            redo[w["id"]] = f"the rewrite had {len(r)} sentences and the summary has {len(w['sentences'])}: give exactly one rewritten sentence for each numbered sentence"
            continue
        coded, n_codes, n_spans = neutralise(w["sentences"], r)
        # a word left in the text counts as a missed name only if it is on the franchise's name list or the rewriter itself
        # replaced the same word somewhere else in this work (titles, peoples and the like, which the protocol keeps, do not count)
        strip = lambda t: re.sub(r"('s|’s)$", "", re.sub(r"^[^\w]+|[^\w]+$", "", t))
        replaced = {strip(t) for a, b in zip(w["sentences"], coded) for t in set(a.split()) - set(b.split()) if strip(t)[:1].isupper()}
        left = [(k, x) for k, x in leftovers(w, coded, names) if names.search(x) or x in replaced]
        empty = [k + 1 for k, s in enumerate(coded) if not s.strip()]
        swallowed = sum(1 for a, b in zip(w["sentences"], coded) for t in set(a.split()) - set(b.split()) if t[:1].islower())
        n_tokens = sum(len(a.split()) for a in w["sentences"])
        stats.append({"id": w["id"], "n_codes": n_codes, "spans": n_spans, "left": left, "lower_case_words_changed": swallowed, "tokens": n_tokens,
                      "unchanged_sentences": sum(a == b for a, b in zip(w["sentences"], r))})
        if empty:
            redo[w["id"]] = f"sentences {empty} came back empty"
        elif len(left) > 2:
            redo[w["id"]] = "these names were still there: " + ", ".join(sorted({x for _, x in left})[:25])
        elif swallowed > 0.12 * n_tokens:
            redo[w["id"]] = "too many ordinary words were changed: change only the names and keep every other word exactly as it is"
        ok[w["id"]] = (r, coded, n_codes)
    out = Path(corpus_dir)
    (out / "anon").mkdir(exist_ok=True)
    for f in (out / "anon").glob("batch_*.jsonl"):
        f.unlink()
    done = [w for w in works if w["id"] in ok]
    for b in range(0, len(works), 10):
        with open(out / "anon" / f"batch_{b // 10:02d}.jsonl", "w") as f:
            for w in works[b:b + 10]:
                if w["id"] in ok:
                    f.write(json.dumps({"id": w["id"], "sentences": ok[w["id"]][0]}, ensure_ascii=False) + "\n")
    if len(done) == len(works):
        with open(out / "corpus_neutral.jsonl", "w") as f:
            for w in works:
                f.write(json.dumps({**w, "sentences": ok[w["id"]][1], "n_codes": ok[w["id"]][2]}, ensure_ascii=False) + "\n")
    json.dump({"redo": {str(k): v for k, v in redo.items()}, "stats": stats}, open(d / "collect.json", "w"), indent=1, ensure_ascii=False)
    tot_left = sum(len(s["left"]) for s in stats)
    print(f"{len(got)} rewrites for {len(works)} works; usable {len(ok)}; to redo {len(redo)}; capitalised or listed words left: {tot_left}; "
          f"lower-case words changed: {sum(s['lower_case_words_changed'] for s in stats)} of {sum(s['tokens'] for s in stats)} tokens; "
          f"corpus_neutral.jsonl {'written' if len(done) == len(works) else 'not written yet'}")
    return redo


def redo(corpus_dir):
    works = load(corpus_dir)
    d = folder(corpus_dir)
    todo = {int(k): v for k, v in json.load(open(d / "collect.json"))["redo"].items()}
    rnd = 1 + max(int(re.match(r"r(\d+)_", f.name).group(1)) for f in d.glob("r*_prompts.json"))
    files = write_prompts([w for w in works if w["id"] in todo], d, rnd, note={k: "an earlier rewrite was not accepted because " + v for k, v in todo.items()})
    print(len(todo), "works to redo ->", len(files), "prompt files, round", rnd)


if __name__ == "__main__":
    {"selftest": selftest, "items": items, "collect": collect, "redo": redo}[sys.argv[1]](*sys.argv[2:])
