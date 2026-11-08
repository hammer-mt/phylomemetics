"""The Sherlock Holmes corpus, built by rule from live Wikipedia (run from study/; PLAN: v2/second/PLAN.md).

    ../.venv/bin/python v2/second/build_holmes.py lists        the pages to fetch -> v2/cache/second/holmes_lists.json
    ../.venv/bin/python v2/second/build_holmes.py candidates   every fetched page with a usable plot -> v2/cache/second/holmes_candidates.jsonl

The lists are made by rule, not from memory:
  canon      the 60 stories of the Conan Doyle canon (4 novels, 56 short stories)
  screen     every page in Category:Sherlock Holmes films and its sub-categories, Category:Sherlock (TV series)
             episodes, and Category:Video games based on Sherlock Holmes and its sub-categories
  ancestors  five earlier detective stories (Poe's three Dupin tales, The Moonstone, Monsieur Lecoq)
  negatives  every American or British film in the local film table whose genre says mystery, detective or whodunit,
             whose plot there names neither Holmes nor Watson, and whose page is in no Sherlock Holmes category
A page is usable if its article has a plot section (the names study/fetch_corpus.py accepts) of at least 8 sentences;
the first 40 sentences are kept, as for the Batman corpus.
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
CANON_FILE = Path("../preregistration/second_corpus/candidates/holmes_story_articles_checked.json")
ANCESTORS = ["The Murders in the Rue Morgue", "The Purloined Letter", "The Mystery of Marie Rogêt", "The Moonstone", "Monsieur Lecoq (novel)"]
SCREEN_CATEGORIES = [("Category:Sherlock Holmes films", True), ("Category:Sherlock (TV series) episodes", False), ("Category:Video games based on Sherlock Holmes", True)]
MIN_SENTENCES = 8


def category_pages(name, recursive, seen=None):
    seen = set() if seen is None else seen
    if name in seen:
        return []
    seen.add(name)
    c = wiki.category(name)
    out = list(c["pages"])
    if recursive:
        for s in c["subcategories"]:
            out += category_pages(s, True, seen)
    return out


def all_holmes_pages():
    """Every page in any category under Category:Sherlock Holmes that was walked: used only to keep them out of the negatives."""
    cats = json.load(open(OUT / "holmes_categories.json"))
    return {p for c in cats.values() for p in c["pages"]}


def lists():
    import pandas as pd
    canon = [s["resolved"] or s["asked"] for s in json.load(open(CANON_FILE)) if s.get("group") == "canon"]
    assert len(canon) == 60
    screen = []
    for name, rec in SCREEN_CATEGORIES:
        screen += category_pages(name, rec)
    screen = list(dict.fromkeys(screen))
    df = pd.read_csv("v2/cache/wiki_movie_plots.csv")
    df["page"] = df["Wiki Page"].map(lambda u: urllib.parse.unquote(u.split("/wiki/")[-1]).replace("_", " "))
    tagged = df["Origin/Ethnicity"].isin(["American", "British"]) & df["Genre"].fillna("").str.lower().str.contains(r"mystery|detective|whodunit|whodunnit")
    holmes = all_holmes_pages() | set(screen)
    neg = df[tagged & ~df["Plot"].str.contains(r"\bHolmes\b|\bWatson\b", regex=True) & ~df["page"].isin(holmes)].drop_duplicates("page")
    meta = {r.page: {"table_year": int(r._1), "table_genre": r.Genre, "table_origin": r._3} for r in neg.rename(columns={"Release Year": "_1", "Origin/Ethnicity": "_3"}).itertuples()}
    json.dump({"canon": canon, "screen": screen, "ancestors": ANCESTORS, "negatives": list(neg["page"]), "negative_meta": meta},
              open(OUT / "holmes_lists.json", "w"), indent=1, ensure_ascii=False)
    titles = list(dict.fromkeys(canon + screen + ANCESTORS + list(neg["page"])))
    (OUT / "holmes_titles.txt").write_text("\n".join(titles) + "\n")
    print({k: len(v) for k, v in (("canon", canon), ("screen", screen), ("ancestors", ANCESTORS), ("negatives", neg))}, "|", len(titles), "pages")


def first_year(text):
    m = re.search(r"\b(1[89]\d\d|20[012]\d)\b", text)
    return int(m.group(1)) if m else None


def candidates():
    L = json.load(open(OUT / "holmes_lists.json"))
    rows, seen, dropped = [], {}, []
    for group in ("canon", "screen", "ancestors", "negatives"):
        got = wiki.wikitexts(L[group])                      # all cached by now
        for t in L[group]:
            r = got[t]
            if r["missing"] or not r["wikitext"]:
                dropped.append((group, t, "no page"))
                continue
            if r["pageid"] in seen:                          # a title that leads to a page already taken (a film title that redirects to its story)
                dropped.append((group, t, f"same page as {seen[r['pageid']]}"))
                continue
            head, sents = plots.plot_sentences(r["wikitext"])
            if len(sents) < MIN_SENTENCES:
                dropped.append((group, t, f"plot section {head!r} has {len(sents)} sentences"))
                continue
            seen[r["pageid"]] = t
            lead = plots.lead(r["wikitext"])
            row = {"asked": t, "title": r["title"], "pageid": r["pageid"], "revid": r["revid"], "rev_timestamp": r["rev_timestamp"], "group": group,
                   "section": head, "n_sentences": len(sents), "sentences": sents, "lead": lead[:1500], "year_in_lead": first_year(lead)}
            if group == "negatives":
                row.update(L["negative_meta"][t])
            rows.append(row)
    with open(OUT / "holmes_candidates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dropped, open(OUT / "holmes_dropped.json", "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    print("usable:", dict(Counter(r["group"] for r in rows)), "| dropped:", dict(Counter(g for g, _, _ in dropped)))
    residue = [(r["title"], s) for r in rows for s in r["sentences"] if re.search(r"\{\{|\}\}|\[\[|\]\]|<ref|</|\|\s*\w+\s*=|''", s)]
    print("sentences with wiki markup left in:", len(residue))
    for t, s in residue[:15]:
        print("   ", t, "|", s[:160])




# ---------------------------------------------------------------- what each screen work says about its own source
TIER_RULE = """You are helping to build a research corpus about Sherlock Holmes on screen. For each article below, record what THE ARTICLE ITSELF SAYS the work is based on. You are not asked what you know about the work or about the stories.

For each article give:
- "kind": one of "film", "tv_episode", "tv_film", "video_game", "series" (an article about a whole film series, television series or game series, not one work), "other".
- "year": the year of first release or broadcast that the article gives (an integer), or null.
- "holmes_appears": true if the article presents Sherlock Holmes as a character in the work, else false.
- "sources": a list, possibly empty, of the Conan Doyle stories the article names as a source of the work. Each entry has:
    "story": the story's title, copied exactly from the list of the 60 canonical stories given below;
    "tier": "A", "B" or "C" by the rule below;
    "quote": the sentence or infobox entry of the article that says so, copied word for word (at most 40 words).
- "other_sources": a list of any source the article names that is not one of the 60 stories (a play, a later novel, another film), as short strings. Empty if none.

The tiers. Use the article's own wording.
- Tier A, a stated adaptation. The article says outright that the work is an adaptation of the story, is based on it, is a version of it, or dramatises it, with no loosening word. An infobox entry "based on" that names the story counts.
- Tier B, a stated loose adaptation. The article says the work is loosely, partly or freely based on the story, or is a modern-day or updated adaptation of it, or takes its plot from it while changing it, or is credited as an adaptation although the plot differs.
- Tier C, elements only. The article says the work is inspired by the story, borrows or uses elements, characters, a device, a line or its title from it, alludes or refers to it, or that critics see a resemblance.

Rules.
- Use only what the article says about the work's sources. Do not add a source from your own knowledge, and do not infer one from character names or from anything else.
- If the article names several stories, give each its own entry and its own tier.
- If the article says the screenplay is original, or names no canonical story as a source, "sources" is an empty list.
- A statement that the work features characters created by Conan Doyle is not a source statement.
- A story named only as the source of a DIFFERENT work (an earlier film, a sequel) is not a source of this one.
"""


def infobox_fields(wikitext):
    import mwparserfromhell
    out = {}
    for t in mwparserfromhell.parse(wikitext).filter_templates():
        if str(t.name).strip().lower().startswith("infobox"):
            for p in t.params:
                k = str(p.name).strip().lower()
                if k in ("based_on", "story", "writer", "screenplay", "director", "released", "first_aired", "airdate", "release", "series", "developer", "genre"):
                    v = re.sub(r"\s+", " ", plots.plain(str(p.value))).strip()
                    if v:
                        out[k] = v[:300]
            break
    return out


def article_without_plot(wikitext, max_words=1400):
    """The article as plain text with its plot section, cast lists and end matter left out."""
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", wikitext, flags=re.M)
    out = [("Lead", plots.plain(parts[0]).strip())]
    skip = re.compile(r"^(plot|synopsis|plot summary|story|summary|premise|storyline|plot and themes|narrative|overview|cast|characters|voice cast|references|external links|"
                      r"notes|see also|further reading|bibliography|footnotes|citations|sources|gameplay|reception|critical reception|box office|release|home media|accolades|awards|soundtrack|music)\b", re.I)
    for i in range(1, len(parts) - 1, 2):
        head = plots.plain(parts[i]).strip()
        if skip.match(head):
            continue
        body = re.sub(r"\n{2,}", "\n", plots.plain(parts[i + 1])).strip()
        if body:
            out.append((head, body))
    text, used = [], 0
    for head, body in out:
        words = body.split()
        if used >= max_words:
            break
        text.append(f"[{head}] " + " ".join(words[:max_words - used]))
        used += len(words)
    return "\n".join(text)


def tier_items(per_file=9):
    """Prompt files for the readers of the screen works' articles: v2/cache/second/tiers/<rater>_<k>.prompt.txt."""
    import numpy as np
    rows = [json.loads(l) for l in open(OUT / "holmes_candidates.jsonl")]
    canon = [r["title"] for r in rows if r["group"] == "canon"]
    screen = [r for r in rows if r["group"] == "screen"]
    d = (OUT / "tiers").resolve()
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for rater, seed in (("r1", 11), ("r2", 12)):
        order = np.random.default_rng(seed).permutation(len(screen))
        for b in range(0, len(screen), per_file):
            items = []
            for i in order[b:b + per_file]:
                r = screen[i]
                w = wiki.wikitexts([r["asked"]])[r["asked"]]["wikitext"]
                items.append(f"### ARTICLE id {r['pageid']}: {r['title']}\nInfobox: {json.dumps(infobox_fields(w), ensure_ascii=False)}\n{article_without_plot(w)}\n")
            name = f"{rater}_{b // per_file:02d}"
            out = d / f"{name}.out.json"
            text = (TIER_RULE + "\nThe 60 canonical stories (copy a title exactly as written here):\n" + "\n".join("- " + c for c in canon) +
                    "\n\nThe articles follow. The plot summaries have been left out on purpose.\n\n" + "\n".join(items) +
                    f"\nWrite your answer with the Write tool to {out} as a JSON list with one object per article, each with the keys "
                    '"id" (the number after "ARTICLE id"), "kind", "year", "holmes_appears", "sources", "other_sources". Write nothing else to that file.\n')
            (d / f"{name}.prompt.txt").write_text(text)
            files.append(name)
    json.dump(files, open(d / "prompts.json", "w"))
    print(len(screen), "screen works,", len(files), "prompt files ->", d, "| words in the longest:", max(len((d / f"{n}.prompt.txt").read_text().split()) for n in files))


def corpus():
    """data/second/holmes/corpus.jsonl: every usable page, ids in the order stories, screen works, ancestors, detective films.
    The ids are fixed here and never change; what each work is (kind, year, sources) is added by `links` in holmes_works.json."""
    rows = [json.loads(l) for l in open(OUT / "holmes_candidates.jsonl")]
    order = {"canon": 0, "screen": 1, "ancestors": 2, "negatives": 3}
    rows.sort(key=lambda r: order[r["group"]])                  # stable: keeps the list order inside a group
    group = {"canon": "story", "screen": "screen", "ancestors": "ancestor", "negatives": "detective_film"}
    out = Path("data/second/holmes")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": i, "title": r["title"], "label": r["title"], "year": r.get("table_year") or r["year_in_lead"], "medium": None, "era": None,
                                "group": group[r["group"]], "section": r["section"], "sentences": r["sentences"], "pageid": r["pageid"], "revid": r["revid"],
                                "rev_timestamp": r["rev_timestamp"]}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(rows), "works ->", out / "corpus.jsonl", dict(Counter(group[r["group"]] for r in rows)), "| sentences:", sum(len(r["sentences"]) for r in rows))


# ---------------------------------------------------------------- year, medium and era of every work, by rule from its own page
DATE_FIELDS = ("pub_date", "english_pub_date", "publication_date", "release_date", "published", "released", "first_aired", "airdate", "release")
YEAR = re.compile(r"\b(1[89]\d\d|20[012]\d)\b")
MEDIUM = {"infobox short story": "short_story", "infobox book": "novel", "infobox film": "film", "infobox television episode": "tv_episode",
          "infobox television": "tv", "infobox video game": "video_game"}


def first_infobox(wikitext):
    import mwparserfromhell
    for t in mwparserfromhell.parse(wikitext).filter_templates():
        if str(t.name).strip().lower().startswith("infobox"):
            name = re.sub(r"<!--.*?-->", "", str(t.name), flags=re.S).strip().lower()
            return name, {str(p.name).strip().lower(): str(p.value) for p in t.params}
    return None, {}


def era(year):
    return "to_1949" if year <= 1949 else "1950_1999" if year <= 1999 else "from_2000"


def meta():
    """Rewrites three fields of data/second/holmes/corpus.jsonl from each page's first infobox and nothing else:
      year    the earliest year written in the infobox's date fields (DATE_FIELDS), references removed
      medium  the kind of infobox (MEDIUM)
      era     to_1949, 1950_1999 or from_2000
    A detective film whose page is not about a film (its first infobox is a book's or a television programme's) is marked
    group "not_used": the table linked the film to a page whose plot section describes something else. Ids and sentences never change."""
    asked = {json.loads(l)["title"]: json.loads(l)["asked"] for l in open(OUT / "holmes_candidates.jsonl")}
    path = Path("data/second/holmes/corpus.jsonl")
    rows = [json.loads(l) for l in open(path)]
    changed, dropped = [], []
    for r in rows:
        name, f = first_infobox(wiki.wikitexts([asked[r["title"]]])[asked[r["title"]]]["wikitext"])
        ys = [int(y) for k in DATE_FIELDS if f.get(k) for y in YEAR.findall(re.sub(r"<ref.*?(</ref>|/>)", "", f[k], flags=re.S))]
        assert ys and name in MEDIUM, (r["title"], name)
        if min(ys) != r["year"]:
            changed.append([r["title"], r["year"], min(ys)])
        r["year"], r["medium"], r["era"], r["infobox"] = min(ys), MEDIUM[name], era(min(ys)), name
        if r["group"] in ("detective_film", "not_used"):
            r["group"] = "detective_film" if name == "infobox film" else "not_used"
            if r["group"] == "not_used":
                dropped.append([r["id"], r["title"], name])
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    json.dump({"what": "years taken from the infobox that differ from the first year in the lead or the film table's year; and the detective films not used",
               "year_changed": changed, "not_used": dropped}, open(OUT / "holmes_meta.json", "w"), indent=1, ensure_ascii=False)
    print(len(rows), "works |", dict(Counter(r["group"] for r in rows)), "|", dict(Counter((r["group"], r["medium"]) for r in rows)), "|",
          dict(Counter((r["group"], r["era"]) for r in rows)), "| years changed:", len(changed), "| not used:", dropped)


def pageviews():
    """data/second/holmes/pageviews.json: total English Wikipedia page views of each story, ancestor and screen work, July 2015 to
    September 2026 (the window of data/pageviews.json), from the Wikimedia page-view API. Titles are the live page titles."""
    import time
    import requests
    path = Path("data/second/holmes/pageviews.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    rows = [json.loads(l) for l in open("data/second/holmes/corpus.jsonl")]
    for r in rows:
        if r["group"] not in ("story", "ancestor", "screen") or str(r["id"]) in out:
            continue
        t = urllib.parse.quote(r["title"].replace(" ", "_"), safe="")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{t}/monthly/20150701/20260930"
        for attempt in range(8):
            q = requests.get(url, headers=wiki.HEADERS, timeout=30)
            if q.status_code == 429 or q.status_code >= 500:
                time.sleep(int(q.headers.get("retry-after", 0) or 0) or 5 * 2 ** attempt)
                continue
            break
        items = q.json().get("items", []) if q.ok else []
        out[str(r["id"])] = {"title": r["title"], "views": sum(i["views"] for i in items), "months": len(items)}
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        time.sleep(1.0)
    print("pageviews:", len(out), "works; without views:", [v["title"] for v in out.values() if not v["views"]])



# ---------------------------------------------------------------- the sets every Holmes analysis uses, as data
ICON = (r"\b(sherlock|holmes|watson|moriarty|lestrade|mycroft|adler|baker|deerstalker|pipes?|violins?|magnifying|hounds?|"
        r"deduc(?:e|es|ed|ing|tion|tions|tive)|detectives?|sleuths?|consulting|landlady|cocaine|disguised?|disguises|"
        r"scotland yard|inspectors?|doctor|hansom|moor|reichenbach)\b")
HERO_NAMES = ["Sherlock Holmes", "Holmes", "Sherlock"]


def reference_overlap(pages):
    """Which of these film-table pages are documents of v2's reference corpus: {page: [document number, 'bgplot' or 'plot']}."""
    out = {}
    want = set(pages)
    for l in open("v2/cache/reference.jsonl"):
        d = json.loads(l)
        if d["src"] not in ("bgplot", "plot"):
            break                                                # the film plots are the first 12,000 documents
        if d["id"] in want:
            out[d["id"]] = [d["i"], d["src"]]
    return out


def spec():
    """data/second/holmes/spec.json: every set of works the plan's tests use, made from corpus.jsonl and works.json by rule."""
    rows = [json.loads(l) for l in open("data/second/holmes/corpus.jsonl")]
    W = json.load(open("data/second/holmes/works.json"))["works"]
    story = {r["title"]: r["id"] for r in rows if r["group"] == "story"}
    ids = lambda g: [r["id"] for r in rows if r["group"] == g]
    by = lambda tiers: {int(i): sorted({story[x["story"]] for x in w["sources"] if x["tier"] in tiers}) for i, w in W.items()
                        if any(x["tier"] in tiers for x in w["sources"])}
    primary, straight = by("AB"), by("A")
    any_tier = by("ABC")
    loose_only = {a: s for a, s in any_tier.items() if a not in primary}
    screen = ids("screen")
    baseline = [a for a in screen if a not in any_tier]
    n_adapt = {c: sum(c in s for s in any_tier.values()) for c in ids("story")}
    asked = {json.loads(l)["title"]: json.loads(l)["asked"] for l in open(OUT / "holmes_candidates.jsonl")}
    films = {asked[r["title"]]: r["id"] for r in rows if r["group"] in ("screen", "detective_film")}
    films.update({r["title"]: r["id"] for r in rows if r["group"] in ("screen", "detective_film")})
    over = reference_overlap(films)
    overlap = sorted([{"id": films[p], "title": p, "document": d, "set": s} for p, (d, s) in over.items()], key=lambda x: x["document"])
    out = {
        "what": "Every set of works the tests of v2/second/PLAN.md use for the Sherlock Holmes corpus. Ids are rows of corpus.jsonl.",
        "franchise": screen, "franchise_where_holmes_appears": [int(i) for i, w in W.items() if w["holmes_appears"]],
        "negatives": ids("detective_film"), "ancestors": ids("ancestor"), "not_used": ids("not_used"),
        "era": {str(r["id"]): r["era"] for r in rows},
        "lineage": ids("story") + screen, "candidates": ids("story"),
        "adaptations": {str(a): s for a, s in primary.items()},
        "adaptations_straight": {str(a): s for a, s in straight.items()},
        "adaptations_elements_only": {str(a): s for a, s in loose_only.items()},
        "baseline": baseline,
        "early": ids("story"), "later": [a for a in screen if a not in primary],
        "strata": {str(r["id"]): r["medium"] for r in rows if r["group"] == "story"},
        "n_screen_works_naming_the_story": {str(c): n for c, n in n_adapt.items()},
        "icon_regex": ICON, "hero_names": HERO_NAMES,
        "in_reference_corpus": overlap,
        "background_documents_left_out": sorted(x["document"] for x in overlap if x["set"] == "bgplot"),
    }
    json.dump(out, open("data/second/holmes/spec.json", "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    srcs = Counter(c for s in primary.values() for c in s)
    print({k: len(v) for k, v in out.items() if isinstance(v, (list, dict))})
    print("primary adaptations", len(primary), "on", len(srcs), "stories; most adapted:", [(rows[c]["title"], n) for c, n in srcs.most_common(3)])
    print("in the reference corpus:", Counter(x["set"] for x in overlap), "| background documents left out:", out["background_documents_left_out"])
    icon = re.compile(ICON, re.I)
    neutral = Path("data/second/holmes/corpus_neutral.jsonl")
    if neutral.exists():
        nr = [json.loads(l) for l in open(neutral)]
        for g in ("screen", "detective_film"):
            sh = [sum(bool(icon.search(x)) for x in r["sentences"]) / len(r["sentences"]) for r, o in zip(nr, rows) if o["group"] == g]
            print(f"  share of name-free sentences the iconography list drops, {g}: mean {sum(sh) / len(sh):.2f}")


# ---------------------------------------------------------------- the readers' answers -> data/second/holmes/works.json
def _answers(rater):
    got = {}
    for f in sorted((OUT / "tiers").glob(f"{rater}_*.out.json")):
        for x in json.load(open(f)):
            got[int(x["id"])] = x
    return got


def _sources(x, canon):
    return {s["story"]: s["tier"] for s in x.get("sources", []) if s["story"] in canon and s["tier"] in "ABC"}


def links(third_items=False):
    """Merge the two readers. Where they differ on the kind of work, on which stories are named or on a tier, a third
    reader is asked and the majority decides; with no majority on a tier the weaker tier stands."""
    rows = [json.loads(l) for l in open("data/second/holmes/corpus.jsonl")]
    canon = [r["title"] for r in rows if r["group"] == "story"]
    screen = [r for r in rows if r["group"] == "screen"]
    r1, r2, r3 = _answers("r1"), _answers("r2"), _answers("r3")
    differ = [w for w in screen if _sources(r1[w["pageid"]], canon) != _sources(r2[w["pageid"]], canon) or r1[w["pageid"]]["kind"] != r2[w["pageid"]]["kind"]]
    if third_items:
        d = (OUT / "tiers").resolve()
        cand = {json.loads(l)["pageid"]: json.loads(l) for l in open(OUT / "holmes_candidates.jsonl")}
        items = []
        for w in differ:
            wt = wiki.wikitexts([cand[w["pageid"]]["asked"]])[cand[w["pageid"]]["asked"]]["wikitext"]
            items.append(f"### ARTICLE id {w['pageid']}: {w['title']}\nInfobox: {json.dumps(infobox_fields(wt), ensure_ascii=False)}\n{article_without_plot(wt)}\n")
        out = d / "r3_00.out.json"
        text = (TIER_RULE + "\nThe 60 canonical stories (copy a title exactly as written here):\n" + "\n".join("- " + c for c in canon) +
                "\n\nThe articles follow. The plot summaries have been left out on purpose.\n\n" + "\n".join(items) +
                f"\nWrite your answer with the Write tool to {out} as a JSON list with one object per article, each with the keys "
                '"id" (the number after "ARTICLE id"), "kind", "year", "holmes_appears", "sources", "other_sources". Write nothing else to that file.\n')
        (d / "r3_00.prompt.txt").write_text(text)
        print(len(differ), "works for a third reader ->", d / "r3_00.prompt.txt")
        return
    works, n3 = {}, 0
    for w in screen:
        a, b, c = r1[w["pageid"]], r2[w["pageid"]], r3.get(w["pageid"])
        votes = [x for x in (a, b, c) if x]
        kinds = [x["kind"] for x in votes]
        kind = max(set(kinds), key=kinds.count)
        srcs = [_sources(x, canon) for x in votes]
        need = 2
        final = {}
        for story in set().union(*srcs):
            tiers = [s[story] for s in srcs if story in s]
            if len(tiers) >= need:
                final[story] = max(set(tiers), key=lambda t: (tiers.count(t), t))          # the majority tier; a tie goes to the weaker (later) letter
        quotes = {s["story"]: s.get("quote", "") for x in votes for s in x.get("sources", []) if s["story"] in final}
        n3 += bool(c)
        years = [x["year"] for x in votes if x.get("year")]
        works[str(w["id"])] = {"title": w["title"], "kind": kind, "year": max(set(years), key=years.count) if years else None,
                               "holmes_appears": sum(bool(x["holmes_appears"]) for x in votes) * 2 > len(votes),
                               "sources": [{"story": s, "tier": t, "quote": quotes.get(s, "")} for s, t in sorted(final.items(), key=lambda kv: (kv[1], kv[0]))],
                               "other_sources": sorted({o for x in votes for o in x.get("other_sources", [])}), "readers": len(votes)}
    assert not [w for w in differ if w["pageid"] not in r3], "the third reader has not answered yet"
    json.dump({"what": "What each Holmes screen work's own Wikipedia article says it is based on, read by two readers who did not see the plots; "
                       "a third reader where they differed. Tiers: A stated adaptation, B stated loose adaptation, C elements only.",
               "works_read_by_a_third_reader": n3, "works": works}, open("data/second/holmes/works.json", "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    best = Counter(min((s["tier"] for s in v["sources"]), default="none") for v in works.values() if v["kind"] != "series")
    print(len(works), "screen works;", dict(Counter(v["kind"] for v in works.values())), "| best tier:", dict(best), "| read by a third reader:", n3)


if __name__ == "__main__":
    {"lists": lists, "candidates": candidates, "tier-items": tier_items, "corpus": corpus, "links": links,
     "third-items": lambda: links(third_items=True), "meta": meta, "pageviews": pageviews, "spec": spec}[sys.argv[1]]()
