"""The counts of the two new corpora that a methods section quotes, collected from the committed files.

    cd study && ../.venv/bin/python v2/second/corpus_counts.py      -> results/v2/second/corpus_counts.json

No count is typed in: each is computed here from a committed file under data/second/ or results/v2/second/, and is
written with a one-line description ("what") and where it comes from ("source"). Where two committed files hold the
same count, both are read and compared ("checks"; a disagreement is printed). Files that are not committed
(v2/cache/second/jobs, v2/acts_second) are read only as further checks, when they are on disk. A count that no
committed file holds is listed under "not_found" with what was looked for. No model is loaded and no test is run.
"""
import ast
import json
import re
import statistics
from collections import Counter
from pathlib import Path

H, R, OUT = Path("data/second/holmes"), Path("data/second/remakes"), Path("results/v2/second")
READERS = OUT / "readers"
TAGS = ["qwen3-32b", "qwen3-8b", "llama3.1-8b", "llama3.3-70b"]
NAMES = {"qwen3-32b": "Qwen3-32B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B", "llama3.3-70b": "Llama 3.3 70B"}
KIND = {"story": "stories", "screen": "screen works", "ancestor": "ancestors", "detective_film": "detective films", "not_used": "not used"}
MEDIUM = {"film": "films", "tv": "television films or programmes", "tv_episode": "episodes", "video_game": "video games"}
ERA = {"to_1949": "up to 1949", "1950_1999": "1950 to 1999", "from_2000": "from 2000"}

out = {"what": "Counts of the Sherlock Holmes corpus and of the remakes corpus (v2/second/PLAN.md, Sections 2, 3, 10 and 15), each computed from a "
               "committed file by v2/second/corpus_counts.py. value: the count; what: what it counts; source: the file and how it was counted.",
       "holmes": {}, "remakes": {}, "checks": [], "not_found": []}


def load(p):
    return json.load(open(p))


def add(corpus, key, value, what, source):
    out[corpus][key] = {"value": value, "what": what, "source": source}
    return value


def check(what, **values):
    """The same count from several places: recorded, and printed if they differ."""
    agree = len({json.dumps(v, sort_keys=True) for v in values.values()}) == 1
    out["checks"].append({"what": what, "values": values, "agree": agree})
    if not agree:
        print("DISAGREEMENT:", what, values)


def three(xs):
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs)}


def constant(path, name):
    """A constant assigned at the top level of a script, read without importing the script."""
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def default_of(path, function, argument):
    """The default value of one argument of a function defined at the top level of a script, read without importing the script."""
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            names = [a.arg for a in node.args.args]
            return ast.literal_eval(node.args.defaults[names.index(argument) - len(names)])
    raise KeyError(function)


def words_shown(folder, reader):
    """{article id: words of the article's text in that reader's prompts}. An article is printed as a line "### ARTICLE id <id>: <title>",
    an infobox line, then one line per section, "[<heading>] <text>" (build_holmes.article_without_plot); only the section text counts towards the limit."""
    out = {}
    for f in sorted(folder.glob(f"{reader}_*.prompt.txt")):
        blocks = re.split(r"^### ARTICLE id (\d+): .*$", f.read_text(), flags=re.M)
        for i in range(1, len(blocks), 2):
            out[blocks[i]] = sum(len(m.group(1).split()) for m in re.finditer(r"^\[[^\]]+\] (.*)$", blocks[i + 1], flags=re.M))
    return out


def answers(folder, reader):
    """Every item a reader answered, over that reader's answer files."""
    return [a for f in sorted(folder.glob(f"{reader}_*.out.json")) for a in load(f)]


# ====================================================================== Holmes
rows = [json.loads(l) for l in open(H / "corpus.jsonl")]
works, spec, views = load(H / "works.json"), load(H / "spec.json"), load(H / "pageviews.json")
W = works["works"]
src = "data/second/holmes/"
by_id = {r["id"]: r for r in rows}

add("holmes", "texts", len(rows), "texts in the corpus (the plot section of each work's own Wikipedia article)", src + "corpus.jsonl: rows")
add("holmes", "texts_by_kind", {KIND[k]: v for k, v in Counter(r["group"] for r in rows).items()},
    "texts by kind: the Conan Doyle stories, the screen works, the ancestors (read, used by no test), the detective films (the negatives of E1) "
    "and the pages kept in the file but not used", src + "corpus.jsonl: rows by `group`")
n_sent = [len(r["sentences"]) for r in rows]
add("holmes", "sentences", sum(n_sent), "sentences in all the texts together", src + "corpus.jsonl: sum of the lengths of `sentences`")
add("holmes", "sentences_per_text", three(n_sent), "sentences per text (PLAN Section 2 sets the fewest a text may have and the most that are kept)", src + "corpus.jsonl: length of `sentences`")
add("holmes", "name_free_texts", sum(1 for _ in open(H / "corpus_neutral.jsonl")), "name-free versions of the texts (every name replaced by a code)",
    src + "corpus_neutral.jsonl: rows")
screen = [r for r in rows if r["group"] == "screen"]
add("holmes", "screen_works_by_medium", {MEDIUM[k]: v for k, v in Counter(r["medium"] for r in screen).items()},
    "screen works by medium, from the kind of the page's first infobox", src + "corpus.jsonl: `medium` of the rows whose `group` is screen")
READER_KIND = {"film": "films", "tv_film": "television films", "series": "pages about a whole series", "tv_episode": "episodes", "video_game": "video games", "other": "other"}
add("holmes", "screen_works_by_the_readers_label", {READER_KIND.get(k, k): v for k, v in Counter(w["kind"] for w in W.values()).items()},
    "screen works by the label the readers of the articles gave (series: the page is about a whole series, not one work)", src + "works.json: works > * > kind")
add("holmes", "pages_about_a_whole_series", sum(w["kind"] == "series" for w in W.values()),
    "screen-work pages that, by the readers' label, are about a whole series; they stay in the corpus", src + "works.json: works whose kind is series")
strata = Counter(spec["strata"][str(i)] for i in spec["early"])
add("holmes", "novels", strata["novel"], "novels among the stories", src + "spec.json: strata, over the ids of `early`")
add("holmes", "short_stories", strata["short_story"], "short stories among the stories", src + "spec.json: strata, over the ids of `early`")

ad, straight, elements = spec["adaptations"], spec["adaptations_straight"], spec["adaptations_elements_only"]
stories_adapted = sorted({s for v in ad.values() for s in v})
add("holmes", "primary_adaptations", len(ad), "screen works whose article states at least one story at tier A (an adaptation) or B (a loose or partial adaptation)",
    src + "spec.json: adaptations (keys)")
add("holmes", "stories_with_a_primary_adaptation", len(stories_adapted), "stories that at least one primary adaptation names at tier A or B", src + "spec.json: adaptations (distinct values)")
add("holmes", "primary_adaptations_naming_more_than_one_story", sum(len(v) > 1 for v in ad.values()),
    "primary adaptations that name two or more stories at tier A or B", src + "spec.json: adaptations (values with more than one story)")
hound = [r["id"] for r in rows if r["group"] == "story" and r["title"] == "The Hound of the Baskervilles"]
assert len(hound) == 1
add("holmes", "primary_adaptations_of_the_hound_of_the_baskervilles", sum(hound[0] in v for v in ad.values()),
    "primary adaptations that are versions of The Hound of the Baskervilles",
    src + f"spec.json: adaptations whose stories include id {hound[0]} (the row of corpus.jsonl titled The Hound of the Baskervilles)")
add("holmes", "straight_adaptations", len(straight), "screen works with a tier A story (the article states it is an adaptation of the story)",
    src + "spec.json: adaptations_straight (keys)")
add("holmes", "straight_adaptations_of_the_hound_of_the_baskervilles", sum(hound[0] in v for v in straight.values()),
    "straight adaptations that are versions of The Hound of the Baskervilles", src + f"spec.json: adaptations_straight whose stories include id {hound[0]}")
add("holmes", "elements_only_works", len(elements), "screen works whose article states only that elements, characters or a title are taken (tier C and nothing stronger)",
    src + "spec.json: adaptations_elements_only (keys)")
add("holmes", "stories_named_by_the_elements_only_works", len({s for v in elements.values() for s in v}), "stories the elements-only works name at tier C",
    src + "spec.json: adaptations_elements_only (distinct values)")
add("holmes", "screen_works_with_no_stated_source", len(spec["baseline"]), "screen works whose article states no Conan Doyle source (the baseline of E2 and the later works of E3)",
    src + "spec.json: baseline")
add("holmes", "works_the_two_readers_disagreed_on", works["works_read_by_a_third_reader"],
    "screen works on which the two readers of the articles differed; a third reader read them and the majority decided", src + "works.json: works_read_by_a_third_reader")
add("holmes", "screen_works_in_which_holmes_appears", len(spec["franchise_where_holmes_appears"]),
    "screen works in which, by both readers, Sherlock Holmes appears as a character", src + "spec.json: franchise_where_holmes_appears")
add("holmes", "screen_works_by_era", {ERA[k]: v for k, v in Counter(spec["era"][str(i)] for i in spec["franchise"]).items()},
    "the screen works (the franchise of E1) by era", src + "spec.json: era, over the ids of `franchise`")
add("holmes", "detective_films_by_era", {ERA[k]: v for k, v in Counter(spec["era"][str(i)] for i in spec["negatives"]).items()},
    "the detective films (the negatives of E1) by era", src + "spec.json: era, over the ids of `negatives`")
add("holmes", "lineage_works", len(spec["lineage"]), "works of the lineage of E2: the stories and the screen works", src + "spec.json: lineage")
add("holmes", "later_works_widened", len(spec["later"]), "screen works that are not primary adaptations (the wider set of later works that E3 reports beside)", src + "spec.json: later")
add("holmes", "screen_works_that_are_films", len(spec["franchise_films_only"]), "screen works that are films (the franchise narrowed to its films, a robustness check of E1)",
    src + "spec.json: franchise_films_only")
ref = spec["in_reference_corpus"]
add("holmes", "texts_that_are_documents_of_v2_reference_corpus", len(ref),
    "films of the Holmes corpus (screen works and detective films) that are, in an older wording, among the film plots of v2's reference corpus",
    src + "spec.json: in_reference_corpus")
add("holmes", "of_which_background_plots", sum(x["set"] == "bgplot" for x in ref),
    "of those, the ones among its background plots (they are left out of the base rates)", src + "spec.json: in_reference_corpus whose set is bgplot")
add("holmes", "of_which_other_reference_plots", sum(x["set"] != "bgplot" for x in ref), "of those, the ones among its other film plots",
    src + "spec.json: in_reference_corpus whose set is not bgplot")
add("holmes", "reference_documents_by_kind_of_text", {KIND[k]: v for k, v in Counter(by_id[x["id"]]["group"] for x in ref).items()},
    "the same films by kind of text", src + "spec.json: in_reference_corpus > id, with `group` of corpus.jsonl")
add("holmes", "texts_shared_with_the_batman_study", len(spec["texts_shared_with_the_batman_study"]),
    "texts that are, sentence for sentence, texts of the Batman study (PLAN Section 4 names them)", src + "spec.json: texts_shared_with_the_batman_study")
add("holmes", "stories_with_page_views", sum(str(i) in views for i in spec["early"]), "stories with a page-view total (the popularity measure of E3)",
    src + "pageviews.json: keys that are ids of spec.json `early`")
add("holmes", "months_of_page_views", sorted({views[str(i)]["months"] for i in spec["early"]}), "months in the page-view window of each story",
    src + "pageviews.json: * > months, over the stories")

rec = {t: load(OUT / f"holmes_recognition_{t}.json") for t in TAGS if (OUT / f"holmes_recognition_{t}.json").exists()}
if rec:
    h1, h2 = {t: len(r["h1"]) for t, r in rec.items()}, {t: len(r["h2"]) for t, r in rec.items()}
    which = ", ".join(NAMES[t] for t in rec)
    add("holmes", "texts_asked_question_h1", sorted(set(h1.values()))[0] if len(set(h1.values())) == 1 else h1,
        "texts each model was asked question H1 about (which famous hero is the story about?)", f"results/v2/second/holmes_recognition_<tag>.json: entries of `h1` ({which})")
    add("holmes", "texts_asked_question_h2", sorted(set(h2.values()))[0] if len(set(h2.values())) == 1 else h2,
        "texts each model was asked question H2 about (which Conan Doyle story does it tell or adapt?): the lineage works",
        f"results/v2/second/holmes_recognition_<tag>.json: entries of `h2` ({which})")
    check("texts asked H1, by model, against the texts of the corpus", corpus=len(rows), **{t: v for t, v in h1.items()})
    check("texts asked H2, by model, against the lineage", lineage=len(spec["lineage"]), **{t: v for t, v in h2.items()})
else:
    out["not_found"].append({"count": "texts asked questions H1 and H2", "looked_for": "results/v2/second/holmes_recognition_<tag>.json (none is there)"})

# the same counts from a second committed file
tiers = {k: {s["tier"] for s in w["sources"]} for k, w in W.items()}
check("screen works", corpus=len(screen), works_json=len(W), spec_franchise=len(spec["franchise"]))
check("primary adaptations (a tier A or B story)", spec=len(ad), works_json=sum(bool(t & {"A", "B"}) for t in tiers.values()))
check("stories with a primary adaptation", spec=len(stories_adapted), works_json=len({s["story"] for w in W.values() for s in w["sources"] if s["tier"] in "AB"}))
check("straight adaptations (a tier A story)", spec=len(straight), works_json=sum("A" in t for t in tiers.values()))
check("elements-only works (tier C and nothing stronger)", spec=len(elements), works_json=sum(t == {"C"} for t in tiers.values()))
check("screen works with no stated source", spec=len(spec["baseline"]), works_json=sum(not t for t in tiers.values()))
check("works the two readers disagreed on", works_json_count=works["works_read_by_a_third_reader"], works_with_three_readers=sum(w["readers"] == 3 for w in W.values()),
      items_the_third_reader_answered=len(answers(READERS / "holmes_sources", "r3")))
check("screen works in which Holmes appears", spec=len(spec["franchise_where_holmes_appears"]), works_json=sum(bool(w["holmes_appears"]) for w in W.values()))
check("screen works by era", spec={k: v for k, v in Counter(spec["era"][str(i)] for i in spec["franchise"]).items()}, corpus=dict(Counter(r["era"] for r in screen)))
check("screen works that are films", spec=len(spec["franchise_films_only"]), corpus=sum(r["medium"] == "film" for r in screen))
check("background plots left out of the base rates", in_reference_corpus=sum(x["set"] == "bgplot" for x in ref), background_documents_left_out=len(spec["background_documents_left_out"]))
check("sets of spec.json against the kinds of corpus.jsonl", spec={"stories": len(spec["early"]), "detective films": len(spec["negatives"]), "ancestors": len(spec["ancestors"]),
                                                                  "not used": len(spec["not_used"])},
      corpus={k: Counter(r["group"] for r in rows)[g] for k, g in (("stories", "story"), ("detective films", "detective_film"), ("ancestors", "ancestor"), ("not used", "not_used"))})
if (OUT / "holmes_words.json").exists():
    hw = load(OUT / "holmes_words.json")
    add("holmes", "background_plots_of_the_base_rates", hw["n_background_plots"], "background film plots the base rates are counted over (v2's background plots less those left out)",
        "results/v2/second/holmes_words.json: n_background_plots")
    add("holmes", "share_of_name_free_sentences_matching_the_iconography_list", hw["counts"]["share_of_name_free_sentences_matching_the_iconography_list"],
        "share of name-free sentences that match the iconography list (they are dropped in one clause of E1), in screen works and in detective films",
        "results/v2/second/holmes_words.json: counts > share_of_name_free_sentences_matching_the_iconography_list")
    check("sentences in all", corpus=sum(n_sent), holmes_words=hw["counts"]["sentences"])
    check("sizes of the sets", spec={k: len(spec[k]) for k in ("franchise", "negatives", "lineage", "candidates", "adaptations", "baseline", "early")},
          holmes_words={k: hw["sets"][k] for k in ("franchise", "negatives", "lineage", "candidates", "adaptations", "baseline", "early")})
for q, n in (("holmes_h1", len(rows)), ("holmes_h2", len(spec["lineage"]))):
    f = Path("v2/cache/second/jobs") / f"{q}.json"
    if f.exists():
        check(f"texts in the question file {f} (not committed)", question_file=len(load(f)["items"]), expected=n)
LIMIT = default_of("v2/second/build_holmes.py", "article_without_plot", "max_words")     # the most words of an article the readers were shown
shown = {r: words_shown(READERS / "holmes_sources", r) for r in ("r1", "r2")}
add("holmes", "words_of_an_article_shown_to_the_readers_at_most", LIMIT, "the most words of a screen work's article that the readers were shown (plot, cast and end sections left out)",
    "v2/second/build_holmes.py: the default of max_words in article_without_plot")
add("holmes", "articles_cut_short_by_the_word_limit", sum(n >= LIMIT for n in shown["r1"].values()),
    "screen-work articles that were cut short for the readers: the text shown is exactly as long as the limit (a longer text is never shown)",
    "results/v2/second/readers/holmes_sources/r1_*.prompt.txt: articles whose section text has as many words as the limit")
check("articles the two readers were shown, against the screen works", corpus=len(screen), **{r: len(v) for r, v in shown.items()})
check("articles cut short by the word limit, in each reader's prompts", **{r: sum(n >= LIMIT for n in v.values()) for r, v in shown.items()})
check("no article is shown beyond the limit (longest text shown, against the limit)", longest_shown=max(max(v.values()) for v in shown.values()), limit=LIMIT)
late = [r for r in screen if r["era"] == "from_2000"]
add("holmes", "screen_works_from_2000_by_medium", {MEDIUM[k]: v for k, v in Counter(r["medium"] for r in late).items()},
    "the screen works from 2000 on by medium, from the kind of the page's first infobox (the era in which most screen works are not films)",
    src + "corpus.jsonl: `medium` of the screen works whose `era` is from_2000")
add("holmes", "screen_works_from_2000_that_are_not_films", sum(r["medium"] != "film" for r in late),
    "screen works from 2000 on that are not films: episodes, video games and television films or programmes", src + "corpus.jsonl: screen works whose `era` is from_2000 and whose `medium` is not film")

# ====================================================================== remakes
P, films, rr = load(R / "pairs.json"), load(R / "films_index.json")["films"], load(R / "rows.json")
pairs, dropped = P["pairs"], P["dropped"]
src = "data/second/remakes/"
film = {f["page"]: f for f in films}
INDIAN = constant("v2/second/build_remakes.py", "INDIAN")
SETS = {"primary": lambda p: p["wording"] == "firm" and p["block"] == "indian_cross_industry",
        "firm_other_cross_industry": lambda p: p["wording"] == "firm" and p["block"] == "other_cross_industry",
        "firm_same_industry": lambda p: p["wording"] == "firm" and p["block"] == "same_industry",
        "firm_all": lambda p: p["wording"] == "firm", "loose_all": lambda p: p["wording"] == "loose"}
prim = [p for p in pairs if SETS["primary"](p)]

for k, what in (("rows", "rows of the film table (the pinned copy of the Wikipedia Movie Plots table)"), ("pages", "Wikipedia pages of the film table"),
                ("pages_with_several_rows", "pages the table lists in several rows (one row per page is used)"),
                ("pages_where_a_later_row_is_used", "of those, the pages for which a row other than the first is used")):
    add("remakes", "film_table_" + k, P["table"][k], what, src + f"pairs.json: table > {k}")
st = [json.loads(l) for l in open(READERS / "remake_statements" / "statements.jsonl")]
read = [s for s in st if s["links_in_table"]]
rs = "results/v2/second/readers/remake_statements/"
add("remakes", "sentences_with_remake_remade_or_reboot", len(st), "sentences in the opening of a film's article that contain remake, remade or reboot", rs + "statements.jsonl: rows")
add("remakes", "statements_read", len(read), "statements: those sentences that link to at least one other page of the table; each was read by two readers",
    rs + "statements.jsonl: rows with a non-empty links_in_table")
third = {a["item"] for a in answers(READERS / "remake_statements", "r3")}
add("remakes", "statements_the_two_readers_read_differently", len(third), "statements the two readers read differently; a third reader read them",
    rs + "r3_*.out.json: distinct items")
add("remakes", "stated_pairs", len(pairs) + len(dropped), "pairs (remake, source) stated by at least one statement and contradicted by none", src + "pairs.json: pairs and dropped together")
add("remakes", "kept_pairs", len(pairs), "stated pairs that meet every condition of PLAN Section 3 (years in order, both plots long enough, two different texts)", src + "pairs.json: pairs")
add("remakes", "dropped_pairs", len(dropped), "stated pairs that fail a condition", src + "pairs.json: dropped")
add("remakes", "dropped_pairs_by_reason", dict(Counter(d["dropped_because"] for d in dropped)), "the dropped pairs by the condition they fail", src + "pairs.json: dropped > * > dropped_because")
add("remakes", "kept_pairs_by_block_and_wording", {f"{b} | {w}": v for (b, w), v in sorted(Counter((p["block"], p["wording"]) for p in pairs).items())},
    "kept pairs by block (the industries of the two films) and by the wording of the statement (firm or loose)", src + "pairs.json: pairs > * > block, wording")
add("remakes", "kept_pairs_by_wording", dict(Counter(p["wording"] for p in pairs)), "kept pairs by the wording of the statement", src + "pairs.json: pairs > * > wording")
COPY_SHARE = constant("v2/second/build_remakes.py", "COPY_SHARE")           # the builder's line for a copied summary
add("remakes", "copied_pairs", sum(p["copied"] for p in pairs), f"kept pairs flagged copied: at least {COPY_SHARE:.0%} of the five-word sequences of the remake's text are also in the source's text",
    src + "pairs.json: pairs > * > copied")
check("copied pairs: the flag, against the stored share of five-word sequences and the builder's threshold", flagged=sum(p["copied"] for p in pairs),
      share_at_or_above_the_threshold=sum(p["share_of_five_word_sequences_also_in_source"] >= COPY_SHARE for p in pairs))
LOW_COPY = constant("v2/second/remakes.py", "LOW_COPY")                     # the analysis's line for "almost no copied wording"
low = lambda p: p["share_of_five_word_sequences_also_in_source"] < LOW_COPY      # noqa: E731
for name, label in (("primary", "the primary set (firm Indian cross-industry pairs)"), ("firm_other_cross_industry", "the firm pairs of other cross-industry remakes"),
                    ("firm_same_industry", "the firm pairs within one industry"), ("firm_all", "all the firm pairs"), ("loose_all", "the loose pairs")):
    e = [p for p in pairs if SETS[name](p)]
    add("remakes", f"set_{name}", {"pairs": len(e), "copied": sum(p["copied"] for p in e), "not_copied": sum(not p["copied"] for p in e),
                                   "under_1pct_of_five_word_sequences_shared": sum(low(p) for p in e), "remakes": len({p["remake"] for p in e}),
                                   "sources": len({p["source"] for p in e}), "families": len({p["family"] for p in e})},
        f"{label}: pairs, how many are copied and not copied, how many share under {LOW_COPY:.0%} of five-word sequences, and their distinct remakes, sources and families",
        src + "pairs.json: pairs > * > wording, block, copied, share_of_five_word_sequences_also_in_source, remake, source, family")
add("remakes", "primary_pairs", len(prim), "pairs of the primary set: firm statements, both films from different Indian industries", src + "pairs.json: pairs with wording firm and block indian_cross_industry")
add("remakes", "primary_pairs_copied", sum(p["copied"] for p in prim), "primary pairs flagged copied (the positive control of the heredity test)", src + "pairs.json: primary pairs > copied")
add("remakes", "primary_pairs_not_copied", sum(not p["copied"] for p in prim), "primary pairs not flagged copied (the confirmatory heredity test is run on these)", src + "pairs.json: primary pairs > copied")
add("remakes", "primary_pairs_with_under_1pct_of_five_word_sequences_shared", sum(low(p) for p in prim), f"primary pairs whose remake shares under {LOW_COPY:.0%} of its five-word sequences with the source",
    src + f"pairs.json: primary pairs > share_of_five_word_sequences_also_in_source below {LOW_COPY}")
add("remakes", "primary_remakes", len({p["remake"] for p in prim}), "distinct remakes in the primary set", src + "pairs.json: primary pairs > remake")
add("remakes", "primary_sources", len({p["source"] for p in prim}), "distinct sources in the primary set", src + "pairs.json: primary pairs > source")
add("remakes", "primary_families", len({p["family"] for p in prim}), "families in the primary set (a family is one cluster in every test)", src + "pairs.json: primary pairs > family")
add("remakes", "primary_remakes_with_two_or_more_documented_sources", sum(v > 1 for v in Counter(p["remake"] for p in prim).values()),
    "remakes of the primary set with two or more documented sources (each pair is one entry)", src + "pairs.json: primary pairs > remake")
add("remakes", "families", P["families"]["n"], "families of films in all (films joined by a remake statement, a shared article, or copied text); every kept pair lies within one family",
    src + "pairs.json: families > n")
check("families: the stored number, against the families of the kept pairs and the families of the films read",
      stored=P["families"]["n"], of_the_kept_pairs=len({p["family"] for p in pairs}), of_the_films_read=len({f["family"] for f in films if f["family"] is not None}))
check("films read that have a family: the films of the kept pairs and the relatives in no kept pair", films_with_a_family=sum(f["family"] is not None for f in films),
      pair_films_and_relatives=len(rr["pair_films"]) + len(P["relatives_that_are_not_in_a_kept_pair"]))
add("remakes", "relatives_in_no_kept_pair", len(P["relatives_that_are_not_in_a_kept_pair"]), "films read that belong to the family of a kept pair but are in no kept pair themselves (never a candidate for their family's remakes)",
    src + "pairs.json: relatives_that_are_not_in_a_kept_pair")
add("remakes", "films_of_the_kept_pairs", len(rr["pair_films"]), "distinct films in the kept pairs", src + "rows.json: pair_films")
add("remakes", "remakes_of_the_kept_pairs", len(rr["rows"]), "distinct remakes in the kept pairs (the rows of the counts)", src + "rows.json: rows")
add("remakes", "sources_of_the_kept_pairs", len({p["source"] for p in pairs}), "distinct sources in the kept pairs", src + "pairs.json: pairs > * > source")
add("remakes", "films_read", len(films), "films read: every film of the table with a long enough cleaned plot from an industry that is the source's in some kept pair, and every film of a kept pair",
    src + "films_index.json: films")
add("remakes", "films_read_by_industry", dict(Counter(f["origin"] for f in films).most_common()), "films read by the table's industry label", src + "films_index.json: films > * > origin")
add("remakes", "films_read_of_the_indian_industries", sum(f["origin"] in INDIAN for f in films), "films read whose industry is one of the Indian industries",
    src + "films_index.json: films > * > origin, with the list INDIAN of v2/second/build_remakes.py")
add("remakes", "sentences_per_film", three([f["n_sentences"] for f in films]), "cleaned sentences per film read (PLAN Section 3 sets the fewest a film may have and the most that are kept)", src + "films_index.json: films > * > n_sentences")

ver = {t: load(OUT / f"remakes_verify_{t}.json") for t in TAGS if (OUT / f"remakes_verify_{t}.json").exists()}
if ver:
    n_read = {t: v["films_read_by_the_model"] for t, v in ver.items()}
    add("remakes", "films_read_by_each_model", {NAMES[t]: v for t, v in n_read.items()},
        "films each model read (Llama 3.3 70B read the films of the Indian industries and every film of a pair)", "results/v2/second/remakes_verify_<tag>.json: films_read_by_the_model")
    every = [t for t, v in n_read.items() if v == len(films)]
    add("remakes", "models_that_read_every_film", [NAMES[t] for t in every], "models that read every film", "results/v2/second/remakes_verify_<tag>.json: films_read_by_the_model equal to the films of films_index.json")
    add("remakes", "cells_recounted_per_model", sorted({v["cells_compared"] for v in ver.values()}),
        "counts of shared units (each remake against each film of a pair) recounted away from the GPU machine for each model; none differs",
        "results/v2/second/remakes_verify_<tag>.json: cells_compared")
    check("recounted cells that differ, by model (all must be 0)", **{t: v["shared_counts_that_differ"] + v["n_optional_counts_that_differ"] for t, v in ver.items()}, expected=0)
    if "llama3.3-70b" in n_read:
        check("films read by Llama 3.3 70B against the films of the Indian industries and of the pairs", verify_file=n_read["llama3.3-70b"],
              films_index=sum(f["origin"] in INDIAN or f["in_pair"] for f in films))
    for t in ver:
        m = Path("v2/acts_second/remakes") / t / "meta.json"
        if m.exists():
            check(f"films read by {NAMES[t]}: the committed recount against {m} (not committed)", verify_file=n_read[t], meta=load(m)["n_texts"])
else:
    out["not_found"].append({"count": "films read by each model", "looked_for": "results/v2/second/remakes_verify_<tag>.json (none is there)"})


def n_candidates(p, frame):
    """Candidates of one entry (PLAN Section 9): never later than the remake, never of its family; which films, by the frame. The source is not counted."""
    r, s = film[p["remake"]], film[p["source"]]
    ok = {"set": lambda f: f["origin"] in INDIAN and f["origin"] != r["origin"], "source_industry": lambda f: f["origin"] == s["origin"], "all": lambda f: True}[frame]
    return sum(1 for f in films if ok(f) and f["year"] <= r["year"] and f["family"] != p["family"])


frames = {"set": "films of the Indian industries other than the remake's own (the primary frame)", "source_industry": "films of the source's industry (the scouts' frame)",
          "all": "every film read"}
cand = {fr: three([n_candidates(p, fr) for p in prim]) for fr in frames}
for fr, label in frames.items():
    add("remakes", "candidates_per_primary_pair" + ("" if fr == "set" else "_" + fr), cand[fr],
        f"candidate films per primary pair, the source not counted: {label}, never later than the remake, never of its family",
        src + "films_index.json (origin, year, family) and pairs.json (primary pairs), counted here by the rule of PLAN Section 9")

rw = load(OUT / "remakes_words.json") if (OUT / "remakes_words.json").exists() else None
if rw:
    c = rw["counts"]
    pools = c["sets"]["primary"]["pools"]
    add("remakes", "pool_sizes_per_primary_pair_as_stored", {fr: {k: pools[fr][k] for k in ("min", "median", "max")} for fr in pools},
        "the same sizes as the analysis stores them: they count the source, so each is one more than the candidates (PLAN Section 16, entry 6)",
        "results/v2/second/remakes_words.json: counts > sets > primary > pools > <frame> > min, median, max")
    for fr in frames:
        check(f"candidates per primary pair, frame {fr}: counted here, against the stored pool sizes less the source",
              counted_here=cand[fr], stored_less_one={k: pools[fr][k] - 1 for k in ("min", "median", "max")})
    refc = c["pair_films_in_v2_reference_corpus"]
    add("remakes", "pair_films_that_are_documents_of_v2_reference_corpus", {"films of the pairs": refc["pair_films"], "reference documents": refc["are_reference_documents"],
                                                                             "of which background plots": refc["by_kind"]["bgplot"]},
        "films of the kept pairs that are, in the table's older wording, documents of v2's reference corpus", "results/v2/second/remakes_words.json: counts > pair_films_in_v2_reference_corpus")
    add("remakes", "primary_films_that_are_documents_of_v2_reference_corpus", {"films of the primary set": refc["primary_set_films"],
                                                                                "reference documents": refc["primary_set_films_that_are_reference_documents"],
                                                                                "of which background plots": refc["primary_by_kind"]["bgplot"]},
        "the same for the films of the primary set", "results/v2/second/remakes_words.json: counts > pair_films_in_v2_reference_corpus")
    add("remakes", "primary_pairs_whose_source_is_a_reference_document", refc["entries_whose_source_is_a_reference_document"]["primary"]["source_is_a_reference_document"],
        "primary pairs whose source is a document of v2's reference corpus",
        "results/v2/second/remakes_words.json: counts > pair_films_in_v2_reference_corpus > entries_whose_source_is_a_reference_document > primary")
    wc = rw["word_columns"]
    add("remakes", "word_columns_of_the_word_model", {"registered": wc["columns_with_the_cube"], "the designers'": wc["columns_of_the_designers_with_the_cube"],
                                                        "measures added by rule (each in two forms)": len(wc["added_measures"])},
        "columns of the registered word model: the designers' columns and those of the measures added by rule before the freeze",
        "results/v2/second/remakes_words.json: word_columns > columns_with_the_cube, columns_of_the_designers_with_the_cube, added_measures")
    add("remakes", "stand_ins", len(rw["word_columns"]["panel"]), "word-only stand-ins that go through the decision test in the units' place", "results/v2/second/remakes_words.json: word_columns > panel")
    sp = c["sets"]["primary"]
    check("primary set: pairs.json against remakes_words.json",
          pairs_json={k: out["remakes"]["set_primary"]["value"][k] for k in ("pairs", "copied", "not_copied", "under_1pct_of_five_word_sequences_shared", "remakes", "sources", "families")},
          remakes_words={"pairs": sp["entries"], "copied": sp["copied"], "not_copied": sp["not_copied"], "under_1pct_of_five_word_sequences_shared": sp["copy_share_under_1pct"],
                         "remakes": sp["remakes"], "sources": sp["sources"], "families": sp["families"]})
    check("kept pairs, their families, remakes, sources and films; films read; copied pairs",
          data_files={"pairs": len(pairs), "families": P["families"]["n"], "remakes": len(rr["rows"]), "sources": len({p["source"] for p in pairs}), "films": len(rr["pair_films"]),
                      "films_read": len(films), "copied": sum(p["copied"] for p in pairs), "indian": sum(f["origin"] in INDIAN for f in films)},
          remakes_words={"pairs": c["pairs"], "families": c["families_of_the_pairs"], "remakes": c["remakes"], "sources": c["sources"], "films": c["films_in_a_pair"],
                         "films_read": c["films_read"], "copied": c["copied_pairs"], "indian": c["films_read_of_the_indian_industries"]})
    check("kept pairs by block and wording", pairs_json=out["remakes"]["kept_pairs_by_block_and_wording"]["value"], remakes_words=dict(sorted(c["pairs_by_block_and_wording"].items())))
else:
    out["not_found"].append({"count": "films of the pairs that are documents of v2's reference corpus", "looked_for": "results/v2/second/remakes_words.json (it is not there)"})

rec = {t: load(OUT / f"remakes_recognition_{t}.json") for t in TAGS if (OUT / f"remakes_recognition_{t}.json").exists()}
if rec:
    r1, r2 = {t: len(r["r1"]) for t, r in rec.items()}, {t: len(r["r2"]) for t, r in rec.items()}
    which = ", ".join(NAMES[t] for t in rec)
    add("remakes", "films_asked_question_r1", sorted(set(r1.values()))[0] if len(set(r1.values())) == 1 else r1,
        "films each model was asked question R1 about (which film is this a plot summary of?): every film of a kept pair",
        f"results/v2/second/remakes_recognition_<tag>.json: entries of `r1` ({which})")
    add("remakes", "remakes_asked_question_r2", sorted(set(r2.values()))[0] if len(set(r2.values())) == 1 else r2,
        "remakes each model was asked question R2 about (which earlier film does it remake?): every remake of a kept pair",
        f"results/v2/second/remakes_recognition_<tag>.json: entries of `r2` ({which})")
    check("films asked R1, by model, against the films of the kept pairs", rows_json=len(rr["pair_films"]), **r1)
    check("remakes asked R2, by model, against the remakes of the kept pairs", rows_json=len(rr["rows"]), **r2)
else:
    out["not_found"].append({"count": "films asked R1 and remakes asked R2", "looked_for": "results/v2/second/remakes_recognition_<tag>.json (none is there)"})
for q, n in (("remakes_r1", len(rr["pair_films"])), ("remakes_r2", len(rr["rows"]))):
    f = Path("v2/cache/second/jobs") / f"{q}.json"
    if f.exists():
        check(f"films in the question file {f} (not committed)", question_file=len(load(f)["items"]), expected=n)

check("statements read: the statements file against each reader's answers", statements=len(read), **{r: len({a["item"] for a in answers(READERS / "remake_statements", r)}) for r in ("r1", "r2")})
check("kept pairs: films of the pairs in rows.json, in pairs.json and flagged in films_index.json", rows_json=len(rr["pair_films"]),
      pairs_json=len({p[k] for p in pairs for k in ("remake", "source")}), films_index=sum(bool(f["in_pair"]) for f in films))
check("relatives in no kept pair that are among the films read", listed=len(P["relatives_that_are_not_in_a_kept_pair"]),
      among_the_films_read=sum(x["film"] in film for x in P["relatives_that_are_not_in_a_kept_pair"]))
check("year and industry of each film of a pair: pairs.json against films_index.json (pairs that differ; must be 0)", differ=sum(
    p[f"{k}_year"] != film[p[k]]["year"] or p[f"{k}_origin"] != film[p[k]]["origin"] for p in pairs for k in ("remake", "source")), expected=0)
missing = {"count": "pages of the film table that have no article today (PLAN Section 15 gives the count)",
           "looked_for": "data/second/remakes/pairs.json (table), films_index.json and results/v2/second/readers/remake_statements/: no committed file stores it"}
log = OUT / "logs" / "second_leads2.log"                                    # the log of the fetch; the logs folder is not committed
if log.exists():
    m = re.findall(r"^(\d+) leads; (\d+) missing", log.read_text(), flags=re.M)
    if m:
        missing["seen_in_a_file_that_is_not_committed"] = {"file": str(log), "pages_fetched": int(m[-1][0]), "pages_with_no_article": int(m[-1][1])}
        check(f"pages of the film table: pairs.json against the pages fetched in {log} (not committed)", pairs_json=P["table"]["pages"], log=int(m[-1][0]))
out["not_found"].append(missing)
add("remakes", "statements_read_by_three_readers_that_state_a_pair", len({(s["on_page"], s["sentence"]) for p in pairs + dropped for s in p["stated"] if s["readers"] == 3}),
    "statements read by a third reader that ended up stating a pair (the other statements read differently state none)", src + "pairs.json: pairs and dropped > * > stated > * > readers equal to 3")

json.dump(out, open(OUT / "corpus_counts.json", "w"), indent=1, ensure_ascii=False)
bad = [c for c in out["checks"] if not c["agree"]]
print("wrote", OUT / "corpus_counts.json")
print("keys:", list(out))
for corpus in ("holmes", "remakes"):
    print(f"{corpus}: {len(out[corpus])} counts:", ", ".join(out[corpus]))
print(f"checks: {len(out['checks'])}, of which {len(bad)} disagree; not found: {len(out['not_found'])}")
