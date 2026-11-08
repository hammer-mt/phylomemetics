"""The remakes corpus, built from what each film's own Wikipedia article says (run from study/; PLAN: v2/second/PLAN.md).

    ../.venv/bin/python v2/second/build_remakes.py scan      every "remake of" statement in a lead -> v2/cache/second/remake_statements.jsonl
    ../.venv/bin/python v2/second/build_remakes.py items     prompt files for two readers of the statements not yet given out
    ../.venv/bin/python v2/second/build_remakes.py third     prompt files for a third reader of the statements the two read differently
    ../.venv/bin/python v2/second/build_remakes.py build     the readers' answers -> data/second/remakes/pairs.json; the texts the models read ->
                                                             v2/cache/second/remakes/films.jsonl and data/second/remakes/films_index.json; the families

The films are those of the local table of Wikipedia film plots (v2/cache/wiki_movie_plots.csv, 34,886 rows). For each
of its pages the lead section of the live article was fetched (wiki.py leads). A statement is a sentence of a lead
that says the film is a remake of another film; the other film is the first link in that sentence that leads to
another page of the table. Nothing here looks at a plot.
"""
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, "v2/second")
import plots  # noqa: E402
import wiki  # noqa: E402

OUT = Path("v2/cache/second")
LINK = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|([^\]]*))?\]\]")


def table():
    import pandas as pd
    df = pd.read_csv("v2/cache/wiki_movie_plots.csv")
    df["page"] = df["Wiki Page"].map(lambda u: urllib.parse.unquote(u.split("/wiki/")[-1]).replace("_", " "))
    return df


def norm(title):
    t = re.sub(r"\s+", " ", title.replace("_", " ")).strip()
    return t[:1].upper() + t[1:]


def sentences_with(wikitext, pattern):
    """Sentences of the lead (still as wikitext) that match the pattern."""
    lead = re.split(r"^==\s*[^=].*?==\s*$", wikitext, maxsplit=1, flags=re.M)[0]
    lead = re.sub(r"<ref[^>/]*/>", "", lead)
    lead = re.sub(r"<ref[^>]*>.*?</ref>", "", lead, flags=re.S)
    lead = re.sub(r"<!--.*?-->", "", lead, flags=re.S)
    # leave out the infobox and other templates at the top: keep running prose only
    prose = "\n".join(l for l in lead.split("\n") if l.strip() and not l.lstrip().startswith(("{{", "|", "}}", "[[File", "[[Image")))
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z'\[\"])", prose):
        if pattern.search(s):
            out.append(s.strip())
    return out


REMAKE = re.compile(r"\bremake\b|\bremade\b|\breboot\b", re.I)


def scan():
    df = table()
    pages = set(df["page"])
    by_norm = {norm(p): p for p in pages}
    rows, n_read = [], 0
    for page in dict.fromkeys(df["page"]):
        f = wiki.CACHE / "lead" / (wiki.safe(page) + ".json")
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        n_read += 1
        if r["missing"] or not r["wikitext"]:
            continue
        for s in sentences_with(r["wikitext"], REMAKE):
            links = [norm(m.group(1)) for m in LINK.finditer(s)]
            rows.append({"page": page, "live_title": r["title"], "revid": r["revid"], "sentence_wikitext": s[:900],
                         "sentence": re.sub(r"\s+", " ", plots.plain(s)).strip()[:700], "links": links,
                         "links_in_table": [by_norm[l] for l in links if l in by_norm and by_norm[l] != page]})
    with open(OUT / "remake_statements.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(n_read, "leads read;", len(rows), "sentences that say remake, remade or reboot;", sum(bool(r["links_in_table"]) for r in rows), "of them link to another film of the table")


# ---------------------------------------------------------------- two readers say what each statement states
REL_RULE = """You are helping to build a research corpus of film remakes from Wikipedia. Each item below gives the title of a film's Wikipedia page (PAGE), ONE sentence from the opening of that page, and the OTHER FILMS that sentence links to. For each other film, record what THE SENTENCE SAYS about PAGE and that film. You are not asked what you know about the films.

For each other film give:
- "relation":
    "page_remakes_other"   the sentence says PAGE is a remake of the other film;
    "other_remakes_page"   the sentence says the other film is a remake of PAGE (for example "it was remade in Telugu as ...");
    "none"                 anything else.
- "wording":
    "firm"    the sentence states the remake plainly as a fact: "is a remake of", "an official remake of", "was remade as", "was remade in Hindi as";
    "loose"   the sentence weakens or hedges it: "loosely", "partly", "unofficial", "uncredited", "inspired by", "based on the idea of", "reported to be", "said to be", "has been described as", "considered", "similar to", "reboot", "reimagining", "re-adaptation";
    "" when the relation is "none".

Rules.
- Use only what the sentence says. Do not use your own knowledge of the films, and do not guess from titles or years.
- If the sentence lists several films as remakes of PAGE, each of them is "other_remakes_page".
- If the sentence says PAGE is a remake of X, and that X is itself a remake of or based on Y, then X is "page_remakes_other" and Y is "none", unless the sentence also says PAGE is a remake of Y.
- "none" covers: a sequel, prequel or spin-off; a film that is only compared or mentioned; another remake of the same original (a sibling); a film based on the same novel, play or true story that the sentence does not call a remake; a film whose remake status the sentence denies; a dubbed version; a film shot at the same time in another language ("simultaneously", "bilingual").
- If you cannot tell which film the word "remake" or "remade" applies to, answer "none".
"""
PER_READER_FILE = 40


def statement_key(r):
    import hashlib
    return hashlib.sha1((r["page"] + "\n" + r["sentence_wikitext"]).encode()).hexdigest()[:16]


def items():
    """Prompt files for the two readers: v2/cache/second/remakes/statements/<reader>_<k>.prompt.txt. Statements already
    given out keep their file; new ones (the scan grows as leads arrive) go into new files. Each reader gets its own order."""
    import numpy as np
    d = (OUT / "remakes" / "statements").resolve()
    d.mkdir(parents=True, exist_ok=True)
    reg_path = d / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"given": {}, "files": []}
    rows = [json.loads(l) for l in open(OUT / "remake_statements.jsonl")]
    new = [r for r in rows if r["links_in_table"] and statement_key(r) not in reg["given"]]
    made = []
    for reader, seed in (("r1", 21), ("r2", 22)):
        order = np.random.default_rng(seed + len(reg["files"])).permutation(len(new))
        start = sum(1 for f in reg["files"] if f.startswith(reader + "_"))
        for b in range(0, len(new), PER_READER_FILE):
            name = f"{reader}_{start + b // PER_READER_FILE:03d}"
            blocks = []
            for i in order[b:b + PER_READER_FILE]:
                r = new[i]
                blocks.append(f"### ITEM {statement_key(r)}\nPAGE: {r['page']}\nSENTENCE: {r['sentence']}\nOTHER FILMS: {json.dumps(list(dict.fromkeys(r['links_in_table'])), ensure_ascii=False)}\n")
            out = d / f"{name}.out.json"
            (d / f"{name}.prompt.txt").write_text(REL_RULE + "\nThe items follow.\n\n" + "\n".join(blocks) +
                                                  f"\nWrite your answer with the Write tool to {out} as a JSON list with one object per item: "
                                                  '{"item": "<the code after ITEM>", "links": [{"film": "<title exactly as listed>", "relation": "...", "wording": "..."}]} '
                                                  "with one entry in \"links\" for every film listed under OTHER FILMS. Write nothing else to that file.\n")
            made.append(name)
    for r in new:
        reg["given"][statement_key(r)] = True
    reg["files"] += made
    reg_path.write_text(json.dumps(reg))
    print(len(new), "new statements ->", len(made), "prompt files:", made[:2], "...", made[-1:] if made else "")



def _answers():
    """{statement key: {reader: {film: (relation, wording)}}} from every answer file."""
    d = OUT / "remakes" / "statements"
    got = {}
    for f in sorted(d.glob("r*_*.out.json")):
        reader = f.name.split("_")[0]
        try:
            ans = json.loads(f.read_text())
        except Exception:
            print("  unreadable answer file:", f.name)
            continue
        for a in ans:
            links = {norm(str(x.get("film", ""))): (x.get("relation", "none"), x.get("wording", "") or "") for x in a.get("links", [])}
            got.setdefault(str(a.get("item")), {})[reader] = links
    return got


RELATIONS = ("page_remakes_other", "other_remakes_page", "none")


def _decide(key, film, got):
    """(relation, wording, readers who answered) for one linked film: both readers agreeing decide; where they differ a
    third reader's answer gives the majority; with no majority the relation is 'none'. Firm needs a majority saying firm."""
    votes = [got.get(key, {}).get(r, {}).get(norm(film)) for r in ("r1", "r2", "r3")]
    votes = [v for v in votes if v and v[0] in RELATIONS]
    if len(votes) < 2:
        return None, None, len(votes)
    rel = [v[0] for v in votes]
    top = max(set(rel), key=rel.count)
    if rel.count(top) < 2:
        return ("none", "", len(votes)) if len(votes) == 3 else (None, None, len(votes))
    if top == "none":
        return "none", "", len(votes)
    firm = sum(v[1] == "firm" for v in votes if v[0] == top)
    return top, ("firm" if firm >= 2 else "loose"), len(votes)


def third():
    """Statements where the two readers differ (relation, or firm against loose) go to a third reader."""
    d = (OUT / "remakes" / "statements").resolve()
    rows = {statement_key(r): r for r in map(json.loads, open(OUT / "remake_statements.jsonl")) if r["links_in_table"]}
    got = _answers()
    need = []
    for k, r in rows.items():
        a = got.get(k, {})
        if "r1" in a and "r2" in a and "r3" not in a:
            if any(a["r1"].get(norm(f)) != a["r2"].get(norm(f)) for f in r["links_in_table"]):
                need.append(r)
    start = len(list(d.glob("r3_*.prompt.txt")))
    made = []
    for b in range(0, len(need), PER_READER_FILE):
        name = f"r3_{start + b // PER_READER_FILE:03d}"
        blocks = [f"### ITEM {statement_key(r)}\nPAGE: {r['page']}\nSENTENCE: {r['sentence']}\nOTHER FILMS: {json.dumps(list(dict.fromkeys(r['links_in_table'])), ensure_ascii=False)}\n"
                  for r in need[b:b + PER_READER_FILE]]
        out = d / f"{name}.out.json"
        (d / f"{name}.prompt.txt").write_text(REL_RULE + "\nThe items follow.\n\n" + "\n".join(blocks) +
                                              f"\nWrite your answer with the Write tool to {out} as a JSON list with one object per item: "
                                              '{"item": "<the code after ITEM>", "links": [{"film": "<title exactly as listed>", "relation": "...", "wording": "..."}]} '
                                              "with one entry in \"links\" for every film listed under OTHER FILMS. Write nothing else to that file.\n")
        made.append(name)
    answered = sum(1 for k in rows if "r1" in got.get(k, {}) and "r2" in got.get(k, {}))
    print(answered, "of", len(rows), "statements read by both readers;", len(need), "read differently ->", made)


# ---------------------------------------------------------------- the texts: the film table's plots, cleaned by rule
INDIAN = {"Bollywood", "Tamil", "Telugu", "Malayalam", "Kannada", "Bengali", "Marathi", "Punjabi", "Assamese"}
MIN_SENTENCES, MAX_SENTENCES = 15, 40
CAST = re.compile(r"\s*\(\s*[A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,3}\s*\)")          # "(Rajinikanth)", "(K. R. Vijaya)": the actor after a character's name
NOTE = re.compile(r"\[\s*(?:\d+|citation needed|note \d+|[a-z])\s*\]")


def clean(plot):
    """A table plot as sentences: bracketed cast names and reference marks removed, v1's sentence splitter, sentences that
    speak of a remake dropped, at most the first 40 kept."""
    sys.path.insert(0, ".")
    from fetch_corpus import sentences
    t = NOTE.sub("", str(plot))
    t = CAST.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    out = [x for x in sentences(t) if not REMAKE.search(x)]
    return out[:MAX_SENTENCES]


def five_grams(sents):
    w = re.findall(r"[a-z0-9']+", " ".join(sents).lower())
    return {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}


def lead_record(page):
    f = wiki.CACHE / "lead" / (wiki.safe(page) + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def infobox_year(page):
    """The earliest year in the date fields of the first infobox of the page's live article (build_holmes.py's rule), or None."""
    import build_holmes as H
    r = lead_record(page)
    if not r or r["missing"] or not r["wikitext"]:
        return None
    _, f = H.first_infobox(r["wikitext"])
    ys = [int(y) for k in H.DATE_FIELDS if f.get(k) for y in H.YEAR.findall(re.sub(r"<ref.*?(</ref>|/>)", "", f[k], flags=re.S))]
    return min(ys) if ys else None


def one_row_per_page():
    """The table with one row for each page. The table lists some films once for every industry they were released or dubbed in
    (780 pages have several rows). The row used is the one whose year is nearest to the earliest year in the date fields of the
    live article's first infobox; a tie, or an article without such a year, goes to the first row. Its year, industry and plot
    are the film's."""
    df = table()
    keep, n_multi, n_moved = [], 0, 0
    for page, g in df.groupby("page", sort=False):
        i = g.index[0]
        if len(g) > 1:
            n_multi += 1
            y = infobox_year(page)
            if y is not None:
                i = (g["Release Year"] - y).abs().idxmin()          # idxmin: the first of equal distances
                n_moved += i != g.index[0]
        keep.append(i)
    out = df.loc[sorted(keep)].reset_index(drop=True)
    out.attrs.update(pages_with_several_rows=n_multi, pages_where_a_later_row_is_used=int(n_moved))
    return out


def one_film_one_text(df):
    """{page: the page whose text stands for it}. Two pages hold the same film when their cleaned texts (of at least 15 sentences)
    are identical; the page that comes first in the table is kept. Also returns every page's cleaned sentences."""
    canon, texts, first = {}, {}, {}
    for page, plot in zip(df["page"], df["Plot"]):
        texts[page] = clean(plot)
        canon[page] = first.setdefault(" ".join(texts[page]), page) if len(texts[page]) >= MIN_SENTENCES else page
    return canon, texts


COPY_SHARE = 0.10


def pairs():
    """data/second/remakes/pairs.json: every (remake, source) the leads state, with the stating sentences, by the rule in PLAN.md Section 3.
    Families are added by `families` once the films read are known."""
    df = one_row_per_page()
    canon, texts = one_film_one_text(df)
    df = df.set_index("page")
    rows = {statement_key(r): r for r in map(json.loads, open(OUT / "remake_statements.jsonl")) if r["links_in_table"]}
    got = _answers()
    stated, undecided, same_film = {}, 0, []
    for k, r in rows.items():
        for film in dict.fromkeys(r["links_in_table"]):
            rel, wording, n = _decide(k, film, got)
            if rel is None:
                undecided += 1
                continue
            if rel == "none":
                continue
            remake, source = (r["page"], film) if rel == "page_remakes_other" else (film, r["page"])
            ev = {"on_page": r["page"], "revid": r["revid"], "sentence": r["sentence"], "wording": wording, "readers": n}
            if canon[remake] == canon[source]:
                same_film.append({"remake": remake, "source": source, "stated": [ev]})
                continue
            stated.setdefault((canon[remake], canon[source]), []).append(dict(ev, pages=[remake, source]) if (remake, source) != (canon[remake], canon[source]) else ev)
    kept, dropped = [], [dict(x, dropped_because="the two pages hold the same text") for x in same_film]
    for (remake, source), ev in sorted(stated.items()):
        why = None
        yr, ys = int(df.loc[remake, "Release Year"]), int(df.loc[source, "Release Year"])
        if (source, remake) in stated:
            why = "stated in both directions"
        elif yr < ys:
            why = "the remake's year in the table is earlier than the source's"
        elif len(texts[remake]) < MIN_SENTENCES or len(texts[source]) < MIN_SENTENCES:
            why = "a plot under 15 sentences"
        row = {"remake": remake, "source": source, "wording": "firm" if any(e["wording"] == "firm" for e in ev) else "loose",
               "remake_year": yr, "source_year": ys, "remake_origin": df.loc[remake, "Origin/Ethnicity"], "source_origin": df.loc[source, "Origin/Ethnicity"],
               "remake_sentences": len(texts[remake]), "source_sentences": len(texts[source]), "stated": ev}
        if why:
            dropped.append(dict(row, dropped_because=why))
            continue
        ro, so = row["remake_origin"], row["source_origin"]
        row["block"] = "indian_cross_industry" if ro in INDIAN and so in INDIAN and ro != so else "same_industry" if ro == so else "other_cross_industry"
        g = five_grams(texts[remake])
        row["share_of_five_word_sequences_also_in_source"] = round(len(g & five_grams(texts[source])) / max(len(g), 1), 4)
        row["copied"] = row["share_of_five_word_sequences_also_in_source"] >= COPY_SHARE
        kept.append(row)
    out = Path("data/second/remakes")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"what": "Film remakes as stated in the lead of a film's own English Wikipedia article (PLAN.md, Section 3). Titles are pages of the film table.",
               "table": {"rows": int(len(table())), "pages": int(len(df)), **df.attrs}, "pairs": kept, "dropped": dropped},
              open(out / "pairs.json", "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    print(len(rows), "statements;", undecided, "links not yet decided;", len(stated) + len(same_film), "stated pairs;", len(kept), "kept;", len(dropped), "dropped",
          dict(Counter(d["dropped_because"] for d in dropped)))
    print("  by block and wording:", dict(Counter((r["block"], r["wording"]) for r in kept)), "| copied:", sum(r["copied"] for r in kept),
          "| primary set copied:", sum(r["copied"] for r in kept if r["block"] == "indian_cross_industry" and r["wording"] == "firm"))


def films():
    """Every film the models read for the remakes: each film of the table with at least 15 cleaned sentences whose industry is
    the industry of a source film of some kept pair, and every film of a kept pair; a text found under several pages is kept once.
    Texts go to v2/cache (they are the table's, rebuilt by this rule); the committed index holds each film's page, year, industry,
    sentence count and text hash."""
    import hashlib
    P = json.load(open("data/second/remakes/pairs.json"))["pairs"]
    df = one_row_per_page()
    canon, texts = one_film_one_text(df)
    industries = {r["source_origin"] for r in P}
    in_pair = {r["remake"] for r in P} | {r["source"] for r in P}
    rows = []
    for page, year, origin, title in zip(df["page"], df["Release Year"], df["Origin/Ethnicity"], df["Title"]):
        if canon[page] != page or len(texts[page]) < MIN_SENTENCES or (origin not in industries and page not in in_pair):
            continue
        rows.append({"id": len(rows), "page": page, "title": str(title).strip(), "year": int(year), "origin": origin, "in_pair": page in in_pair,
                     "family": None, "sentences": texts[page], "sha1": hashlib.sha1(" ".join(texts[page]).encode()).hexdigest()})
    assert in_pair <= {r["page"] for r in rows}
    cache = OUT / "remakes"
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "films.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(rows), "films;", sum(r["in_pair"] for r in rows), "in a pair; by industry:", dict(Counter(r["origin"] for r in rows).most_common()))


def families():
    """Who is whose relative. A family is a set of films joined by any of:
      a remake statement that links two films, whatever the readers made of it (so siblings and pairs dropped for a short plot count);
      two table pages that lead to the same live article;
      a film that holds at least 10% of the five-word sequences of a film of a kept pair (a copy of it under another title).
    A film is never a candidate source for a remake of its own family, and the tests treat a family as one cluster. Writes the
    family into pairs.json and films.jsonl, and the index data/second/remakes/films_index.json."""
    import hashlib
    from collections import Counter, defaultdict
    from sklearn.feature_extraction.text import CountVectorizer
    path = Path("data/second/remakes/pairs.json")
    D = json.load(open(path))
    P = D["pairs"]
    films = [json.loads(l) for l in open(OUT / "remakes" / "films.jsonl")]
    df = one_row_per_page()
    canon, _ = one_film_one_text(df)
    parent, why = {}, defaultdict(set)

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b, reason):
        if a != b:
            parent[find(a)] = find(b)
            why[a].add(reason); why[b].add(reason)

    for r in map(json.loads, open(OUT / "remake_statements.jsonl")):
        for f in r["links_in_table"]:
            join(canon[r["page"]], canon[f], "a remake statement links them")
    live = defaultdict(list)
    for page in df["page"]:
        rec = lead_record(page)
        if rec and not rec["missing"] and rec.get("pageid"):
            live[rec["pageid"]].append(canon[page])
    for pages in live.values():
        for q in pages[1:]:
            join(pages[0], q, "the same live article")
    in_pair = [f for f in films if f["in_pair"]]
    grams = lambda sents: [" ".join(w[i:i + 5]) for w in [re.findall(r"[a-z0-9']+", " ".join(sents).lower())] for i in range(len(w) - 4)]
    M = CountVectorizer(analyzer=grams, binary=True).fit_transform([f["sentences"] for f in films]).astype("float32").tocsr()
    A = M[[f["id"] for f in in_pair]]
    share = (A @ M.T).toarray() / A.sum(1).A
    copies = []
    for k, f in enumerate(in_pair):
        for j in (share[k] >= COPY_SHARE).nonzero()[0]:
            if j != f["id"]:
                join(f["page"], films[j]["page"], "it holds at least 10% of the other's five-word sequences")
                copies.append([f["page"], films[j]["page"], round(float(share[k, j]), 4)])
    fam = {}
    for r in P:
        r["family"] = fam.setdefault(find(r["remake"]), len(fam))
        assert find(r["source"]) == find(r["remake"])
    for f in films:
        f["family"] = fam.get(find(f["page"])) if f["page"] in parent else None
    pair_films = {f["page"] for f in in_pair}
    D["relatives_that_are_not_in_a_kept_pair"] = sorted([{"film": f["page"], "family": f["family"], "why": sorted(why[f["page"]])}
                                                         for f in films if f["family"] is not None and f["page"] not in pair_films], key=lambda x: (x["family"], x["film"]))
    D["families"] = {"n": len(fam), "rule": families.__doc__.split("Writes")[0].strip()}
    json.dump(D, open(path, "w"), indent=1, ensure_ascii=False)
    cache = OUT / "remakes"
    with open(cache / "films.jsonl", "w") as f:
        for r in films:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    index = [{k: r[k] for k in ("id", "page", "year", "origin", "in_pair", "family", "sha1")} | {"n_sentences": len(r["sentences"])} for r in films]
    json.dump({"what": "The films read for the remakes corpus, in the order of v2/cache/second/remakes/films.jsonl (rebuilt by build_remakes.py from the pinned film table).",
               "table": "vishnupriyavr/wiki-movie-plots-with-summaries @ a1feef07 (v2/build_reference.py)", "films": index},
              open("data/second/remakes/films_index.json", "w"), ensure_ascii=False)
    sizes = Counter(f["family"] for f in films if f["family"] is not None)
    print(len(fam), "families;", sum(sizes.values()), "films in a family;", len(D["relatives_that_are_not_in_a_kept_pair"]), "of them in no kept pair; largest families:", sizes.most_common(4))
    print("  copies found outside the kept pairs:", sum(1 for a, b, s in copies if not any({a, b} == {r["remake"], r["source"]} for r in P)))
    print("  pairs.json sha256", hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], "| films.jsonl sha256", hashlib.sha256(open(cache / "films.jsonl", "rb").read()).hexdigest()[:16])


def build():
    pairs()
    films()
    families()


if __name__ == "__main__":
    {"scan": scan, "items": items, "third": third, "build": build}[sys.argv[1]]()
