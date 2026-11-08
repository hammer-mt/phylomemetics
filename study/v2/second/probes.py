"""The recognition probes for the two new corpora (v2/second/PLAN.md, Section 10): jobs for v2/gpu/ablate.py, and their scoring.

    ../.venv/bin/python v2/second/probes.py jobs                 -> v2/cache/second/jobs/{holmes_h1,holmes_h2,remakes_r1,remakes_r2}.json
    ../.venv/bin/python v2/second/probes.py score <model-tag>    -> results/v2/second/holmes_recognition_<tag>.json, remakes_recognition_<tag>.json

With no sets and no conditions ablate.py simply puts each text and question to the untouched model and keeps its greedy
reply and the log-probability that the reply starts with one of the item's answers. Every question must be in the plan
word for word. Scoring is by rule (does the reply contain the title?), with no judge.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

H1 = ("Every name in this plot summary has been replaced by a code such as X1. Which famous fictional hero is the story about? "
      "Reply with only the name the hero is best known by, not a code.")
H2 = ("Every name in this plot summary has been replaced by a code such as X1. It is a summary of a Sherlock Holmes story, or of a film, "
      "television episode or game about Sherlock Holmes. Which story by Arthur Conan Doyle does it tell or adapt? Reply with the story's title only.")
R1 = "Which film is this a plot summary of? Reply with the film's title only."
R2 = ("This is the plot summary of a film that is a remake of an earlier film. Which earlier film does it remake? "
      "Reply with the earlier film's title only.")
PLAN = Path("v2/second/PLAN.md")
JOBS = Path("v2/cache/second/jobs")
ACTS = Path("v2/acts_second")
RESULTS = Path("results/v2/second")


ARTICLES = {"the", "a", "an"}
NAMED = re.compile(r"holmes|sherlock|watson|moriarty|baker street", re.I)          # H1: the reply names the franchise


def words(s):
    """Lower case, accents and punctuation dropped, split into words, the words the, a, an left out wherever they stand."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", s).split() if w not in ARTICLES]


def title_words(title):
    """A page title as people say it: a bracketed suffix and a leading 'The Adventure of' dropped."""
    t = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    t = re.sub(r"^The Adventure of\s+", "", t)
    return words(t)


def distance(a, b):
    """Edit distance between two strings (insertions, deletions, substitutions)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def allowed(n):
    """Letters that may differ: none in a title under 5 letters, 1 up to 9, 2 up to 19, 3 from 20 (other spellings of one title)."""
    return 0 if n < 5 else 1 if n < 10 else 2 if n < 20 else 3


def names_title(reply, title):
    """Does the reply contain the title? The title's words must stand together as whole words in the reply, within a few letters'
    difference (allowed()); a title of fewer than four letters must be the whole reply."""
    r, t = words(reply), title_words(title)
    if not t or not r:
        return False
    want = " ".join(t)
    if len(want.replace(" ", "")) < 4:
        return " ".join(r) == want
    for n in {len(t), len(t) - 1, len(t) + 1} - {0}:                 # a spelling may join or split a word
        for i in range(len(r) - n + 1):
            if distance(" ".join(r[i:i + n]), want) <= allowed(len(want.replace(" ", ""))):
                return True
    return False


def jobs():
    plan = PLAN.read_text()
    for q in (H1, H2, R1, R2):
        assert q in plan, "every question must be in the plan, word for word: " + q[:50]
    JOBS.mkdir(parents=True, exist_ok=True)
    # ---- Holmes
    rows = [json.loads(l) for l in open("data/second/holmes/corpus.jsonl")]
    neutral = [json.loads(l) for l in open("data/second/holmes/corpus_neutral.jsonl")]
    spec = json.load(open("data/second/holmes/spec.json"))
    assert [r["id"] for r in rows] == [r["id"] for r in neutral] == list(range(len(rows)))
    text = lambda i: " ".join(neutral[i]["sentences"])
    h1 = [{"id": str(r["id"]), "group": r["group"], "text": text(r["id"]), "answers": spec["hero_names"]} for r in rows]
    lineage = spec["lineage"]
    h2 = [{"id": str(i), "group": rows[i]["group"], "text": text(i), "answers": spec["hero_names"]} for i in lineage]
    json.dump({"question": H1, "reply_tokens": 12, "items": h1}, open(JOBS / "holmes_h1.json", "w"), ensure_ascii=False)
    json.dump({"question": H2, "reply_tokens": 24, "items": h2}, open(JOBS / "holmes_h2.json", "w"), ensure_ascii=False)
    print("Holmes:", len(h1), "texts for H1,", len(h2), "for H2")
    # ---- remakes (the texts as the models read them: original names)
    pairs_path, films_path = Path("data/second/remakes/pairs.json"), Path("v2/cache/second/remakes/films.jsonl")
    if pairs_path.exists() and films_path.exists():
        P = json.load(open(pairs_path))["pairs"]
        films = {}
        for l in open(films_path):
            f = json.loads(l)
            if f["in_pair"]:
                films[f["page"]] = f
        r1 = [{"id": str(f["id"]), "page": p, "text": " ".join(f["sentences"]), "answers": [f["title"]]} for p, f in films.items()]
        seen, r2 = set(), []
        for x in P:
            if x["remake"] not in seen:
                seen.add(x["remake"])
                f = films[x["remake"]]
                srcs = [y["source"] for y in P if y["remake"] == x["remake"]]
                r2.append({"id": str(f["id"]), "page": x["remake"], "text": " ".join(f["sentences"]), "answers": [films[s]["title"] for s in srcs], "sources": srcs})
        json.dump({"question": R1, "reply_tokens": 24, "items": r1}, open(JOBS / "remakes_r1.json", "w"), ensure_ascii=False)
        json.dump({"question": R2, "reply_tokens": 24, "items": r2}, open(JOBS / "remakes_r2.json", "w"), ensure_ascii=False)
        print("remakes:", len(r1), "films for R1,", len(r2), "remakes for R2")


def score(tag):
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = ACTS / "probes" / tag
    # ---- Holmes
    if (d / "holmes_h1.json").exists():
        rows = [json.loads(l) for l in open("data/second/holmes/corpus.jsonl")]
        works = json.load(open("data/second/holmes/works.json"))["works"]
        spec = json.load(open("data/second/holmes/spec.json"))
        title = {r["id"]: r["title"] for r in rows}
        h1 = {int(x["id"]): x for x in json.load(open(d / "holmes_h1.json"))["items"]}
        out = {"model": tag, "question_h1": H1, "question_h2": H2, "h1": {}, "h2": {}}
        for i, x in h1.items():
            reply = x["untouched"]["reply"]
            out["h1"][str(i)] = {"named": bool(NAMED.search(reply)), "logp": x["untouched"]["logp"], "reply": reply}
        if (d / "holmes_h2.json").exists():
            sources = {**{int(a): s for a, s in spec["adaptations"].items()}, **{int(a): s for a, s in spec["adaptations_elements_only"].items()}}
            for x in json.load(open(d / "holmes_h2.json"))["items"]:
                i, reply = int(x["id"]), x["untouched"]["reply"]
                want = [i] if rows[i]["group"] == "story" else sources.get(i, [])
                out["h2"][str(i)] = {"source_named": any(names_title(reply, title[s]) for s in want), "has_source": bool(want), "reply": reply}
        fr, neg = spec["franchise"], spec["negatives"]
        era = spec["era"]
        # as in paper v2: how well the log-probability of the hero's name alone tells the franchise from the negatives (within era, like E1)
        per = []
        for i in fr:
            same = [j for j in neg if era[str(j)] == era[str(i)]]
            a, b = out["h1"][str(i)]["logp"], [out["h1"][str(j)]["logp"] for j in same]
            per.append(sum((a > x) + 0.5 * (a == x) for x in b) / len(b))
        guesses = {}
        if out["h2"]:
            for i in spec["baseline"]:                     # screen works with no stated source: nothing to recognise, so every title given is a guess
                reply = out["h2"][str(i)]["reply"]
                hit = [title[c] for c in spec["candidates"] if names_title(reply, title[c])]
                for t in hit or ["(no story's title)"]:
                    guesses[t] = guesses.get(t, 0) + 1
        out["summary"] = {"screen_works_named": sum(out["h1"][str(i)]["named"] for i in fr), "of": len(fr),
                          "detective_films_named_as_holmes": sum(out["h1"][str(i)]["named"] for i in neg), "of_negatives": len(neg),
                          "within_era_auc_of_the_log_probability_of_the_hero_name": sum(per) / len(per),
                          "stories_named": sum(out["h1"][str(i)]["named"] for i in spec["candidates"]),
                          "stories_whose_title_is_named": sum(out["h2"][str(i)]["source_named"] for i in spec["candidates"]) if out["h2"] else None,
                          "primary_adaptations_whose_source_is_named": sum(out["h2"][a]["source_named"] for a in spec["adaptations"]) if out["h2"] else None,
                          "titles_given_for_the_screen_works_with_no_stated_source": dict(sorted(guesses.items(), key=lambda kv: -kv[1]))}
        json.dump(out, open(RESULTS / f"holmes_recognition_{tag}.json", "w"), indent=1, ensure_ascii=False)
        print("Holmes", tag, out["summary"])
    # ---- remakes
    if (d / "remakes_r1.json").exists():
        r1 = json.load(open(d / "remakes_r1.json"))["items"]
        jobs1 = {x["id"]: x for x in json.load(open(JOBS / "remakes_r1.json"))["items"]}
        out = {"model": tag, "question_r1": R1, "question_r2": R2, "r1": {}, "r2": {}}
        for x in r1:
            j = jobs1[x["id"]]
            out["r1"][j["page"]] = {"identified": names_title(x["untouched"]["reply"], j["answers"][0]) or names_title(x["untouched"]["reply"], j["page"]),
                                    "logp": x["untouched"]["logp"], "reply": x["untouched"]["reply"]}
        if (d / "remakes_r2.json").exists():
            jobs2 = {x["id"]: x for x in json.load(open(JOBS / "remakes_r2.json"))["items"]}
            for x in json.load(open(d / "remakes_r2.json"))["items"]:
                j = jobs2[x["id"]]
                reply = x["untouched"]["reply"]
                out["r2"][j["page"]] = {"source_named": any(names_title(reply, t) for t in j["answers"] + j["sources"]), "logp": x["untouched"]["logp"], "reply": reply}
        P = json.load(open("data/second/remakes/pairs.json"))["pairs"]
        ident = lambda page: out["r1"].get(page, {}).get("identified", False)
        named = lambda page: out["r2"].get(page, {}).get("source_named", False)
        out["pairs"] = [{"remake": x["remake"], "source": x["source"], "recognised": bool(named(x["remake"]) or ident(x["remake"]) or ident(x["source"])),
                         "recognised_both_films": bool(named(x["remake"]) or (ident(x["remake"]) and ident(x["source"])))} for x in P]
        prim = [q for q, x in zip(out["pairs"], P) if x["block"] == "indian_cross_industry" and x["wording"] == "firm"]
        out["summary"] = {"films_identified": sum(v["identified"] for v in out["r1"].values()), "of": len(out["r1"]),
                          "remakes_whose_source_is_named": sum(v["source_named"] for v in out["r2"].values()), "of_remakes": len(out["r2"]),
                          "primary_pairs_recognised": sum(q["recognised"] for q in prim), "primary_pairs_recognised_both_films": sum(q["recognised_both_films"] for q in prim),
                          "of_primary_pairs": len(prim)}
        json.dump(out, open(RESULTS / f"remakes_recognition_{tag}.json", "w"), indent=1, ensure_ascii=False)
        print("remakes", tag, out["summary"])


if __name__ == "__main__":
    {"jobs": jobs, "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
