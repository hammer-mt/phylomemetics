"""The paper's tables for the two new corpora, written from the committed results so that no number is copied by hand.

    cd study && ../.venv/bin/python v2/second/paper_tables.py
        -> results/v2/second/paper_tables.md      the tables, each followed by its sources
        -> results/v2/second/paper_tables.json    the same cells, each with the file and key path it was read from

Reads results/v2/second/*.json. A model whose file is not there yet shows "not yet run"; run the script again when it
arrives. Number formats follow v2/analysis/paper_tables.py (AUC and rho to 2 decimals, shares to 3, p as that script
writes it, except that a very small p is written out, 0.00001, as the paper does). No model is loaded and no test is run:
the one number computed here is the noise-calibrated p of PLAN.md Section 16, entry 3, by the formula of noise_p() in
v2/second/remake_null_size.py (the formula is repeated here so that this script imports no analysis code).
A file that cannot be read yet (one that is still being written) is treated as not there, and said so on screen.
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
OUT = Path("results/v2/second")
NAMES = {"qwen3-32b": "Qwen3-32B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B", "llama3.3-70b": "Llama 3.3 70B"}
ORDER = ["qwen3-32b", "qwen3-8b", "llama3.1-8b", "llama3.3-70b"]
PRIMARY = ORDER[0]
COL = {t: NAMES[t] + (" (primary)" if t == PRIMARY else "") for t in ORDER}
MODELS = [COL[t] for t in ORDER]
NOT_YET, NA = "not yet run", "n/a"


class Missing(Exception):
    """The file a cell needs is not there (yet)."""


class Val:
    """One value read from a file, with where it was read."""
    def __init__(self, v, file, keys):
        self.v, self.file, self.keys = v, file, tuple(keys)


class File:
    def __init__(self, name):
        self.name, p, self.d = name, OUT / name, None
        if p.exists():
            try:
                self.d = json.load(open(p))
            except ValueError:                          # still being written: read it on the next run
                print(f"{name}: cannot be read yet, treated as not there")

    def __bool__(self):
        return self.d is not None

    def get(self, *keys):
        if self.d is None:
            raise Missing(self.name)
        x = self.d
        for k in keys:
            x = x[k]
        return Val(x, self.name, keys)

    def has(self, *keys):
        try:
            self.get(*keys)
            return True
        except (Missing, KeyError, TypeError, IndexError):
            return False


def cell(text, *vals):
    """A table cell: its text and every value it was written from."""
    return {"text": str(text), "from": [{"file": v.file, "key": list(v.keys), "value": v.v} for v in vals]}


def join(sep, *cells):
    return {"text": sep.join(c["text"] for c in cells), "from": [s for c in cells for s in c["from"]]}


def note(c, text):
    return dict(c, text=c["text"] + text)


def C(fn):
    """A cell from a function; what to show when its file or its key is not there."""
    try:
        return fn()
    except Missing:
        return cell(NOT_YET)
    except (KeyError, TypeError, IndexError, AttributeError):
        return cell(NA)


# ---- formats (v2/analysis/paper_tables.py: f, pv, n)
def fx(v, d=2):
    return cell(f"{v.v:.{d}f}", v)


ALPHA = 0.05                                        # every test is read at 0.05 (PLAN Section 13)


def p_text(p):
    if p >= 0.995:                                  # not written as 1.00 unless it is 1
        return "1" if p == 1 else f"{p:.3f}" if p < 0.9995 else "above 0.999"
    if p >= 0.001:
        text = f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
        d = len(text) - 2
        while float(text) == ALPHA and p != ALPHA and d < 12:      # a p just beside 0.05 is not printed as 0.05: one more digit until it shows its side
            d += 1
            text = f"{p:.{d}f}"
        return text
    return format(Decimal(f"{p:.2g}"), "f")         # 1e-05 is written 0.00001, as in the paper


def pv(v):
    return cell(p_text(v.v), v)


def num(v):
    return cell(f"{int(v.v):,}", v)


def of(k, n):
    return cell(f"{int(k.v):,} of {int(n.v):,}", k, n)


def pct_text(x, d=1):
    """A share as a percentage, halves rounded up (217 of 2,000 is 10.9%, not 10.8%)."""
    return str((Decimal(repr(x)) * 100).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)) + "%"


def pct(v, d=1):
    return cell(pct_text(v.v, d), v)


def interval(a, ci, d=2, lines=(), lower_lines=()):
    """A value with its interval. lines: levels the plan compares the value with (lower_lines: the interval's lower bound); where
    rounding would print a number as one of its levels although it is under it, the unrounded number is given too (0.6991 is
    printed 0.70 and is under 0.70)."""
    text = f"{a.v:.{d}f} ({ci.v[0]:.{d}f} to {ci.v[1]:.{d}f})"
    for x, levels, what in ((a.v, lines, ""), (ci.v[0], lower_lines, "lower bound ")):
        for line in levels:
            if x < line and f"{x:.{d}f}" == f"{line:.{d}f}":
                more = next(k for k in range(d + 1, d + 12) if f"{x:.{k}f}" != f"{line:.{k}f}")
                text += f"; {what}unrounded {x:.{more}f}, under {line:.{d}f}"
    return cell(text, a, ci)


def word(v):
    return cell(str(v.v).replace("_", " "), v)


def yes_no(v):
    return cell("yes" if v.v else "no", v)


def confined(v):
    """The plan's rule on recognition (Section 10) as holmes.py stores it. Its key is not_confined_to_works_the_model_names, so the stored
    word "holds" means NOT confined; the other stored words are printed as they are."""
    return cell("not confined: the works it does not name reach " + (LEVEL or "the level the rule asks") if v.v == "holds" else str(v.v).replace("_", " "), v)


def outcome(f):
    """The outcome of the decision test in a model's remakes file. The stored word "claim" is spelled out in the plan's wording, with the
    three qualifications the plan fixes for it (PLAN Section 9), each read from the file: the pairs with almost no copied wording, the
    embedding in the units' place, and the units with the embedding among the word columns."""
    v = f.get("decision", "outcome")
    if v.v != "claim":
        return cell(str(v.v).replace("_", " "), v)
    text, used = "the claim, in the plan's wording: information about the source beyond these word measures", [v]
    for keys, when, words in ((("decision", "sum_over_the_entries_with_a_copy_share_under_1pct", "above_zero"), False, "in pairs whose summaries share copied wording"),
                              (("embedding", "in_the_units_place", "passes"), True, "a general-purpose embedding does as well"),
                              (("embedding", "units_with_the_embedding_among_the_word_columns", "conditions_3_4_5_hold"), False, "not beyond a general-purpose embedding")):
        if f.has(*keys):
            used.append(f.get(*keys))
            if f.get(*keys).v is when:
                text += "; " + words
    return cell(text, *used)


# ---- tables
class Table:
    def __init__(self, key, title, cols, first="", intro=""):
        self.key, self.title, self.cols, self.first, self.intro, self.rows = key, title, list(cols), first, intro, []

    def row(self, label, cells):
        """cells: {column: cell}; a column left out is empty."""
        self.rows.append((label, {c: cells.get(c, cell("")) for c in self.cols}))

    def md(self):
        head = [self.first] + self.cols
        lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        lines += ["| " + " | ".join([label] + [cells[c]["text"].replace("|", "\\|") for c in self.cols]) + " |" for label, cells in self.rows]
        src = []
        for label, cells in self.rows:
            by_file = {}
            for c in self.cols:
                for s in cells[c]["from"]:
                    f = s["file"]
                    for t in ORDER if c in MODELS else []:          # a model column: one line for the four files; any other column names its file
                        f = f.replace(t, "<tag>")
                    path = " > ".join(str(k) for k in s["key"])
                    if path not in by_file.setdefault(f, []):
                        by_file[f].append(path)
            if by_file:
                src.append(f"- {label}: " + "; ".join(f"`{f}`: " + ", ".join(f"`{p}`" for p in ps) for f, ps in by_file.items()))
        return (f"## {self.key}. {self.title}\n\n" + (self.intro + "\n\n" if self.intro else "") + "\n".join(lines) +
                "\n\nSources (file: key path; `<tag>` is the model's tag in the file name):\n\n" + "\n".join(src))

    def json(self):
        return {"title": self.title, "columns": self.cols, "rows": [{"row": label, "cells": cells} for label, cells in self.rows]}


def per_model(files, fn):
    """{model column: cell} from one function of the model's file."""
    return {COL[t]: C(lambda f=files[t]: fn(f)) for t in ORDER}


# ====================================================================== the files
HM = {t: File(f"holmes_{t}.json") for t in ORDER}
HR = {t: File(f"holmes_recognition_{t}.json") for t in ORDER}
HF = {t: File(f"holmes_e1_followup_{t}.json") for t in ORDER}
HW = File("holmes_words.json")
RM = {t: File(f"remakes_{t}.json") for t in ORDER}
RR = {t: File(f"remakes_recognition_{t}.json") for t in ORDER}
RW = File("remakes_words.json")
RN = File("remakes_null_size.json")
CF = File("confirmatory.json")
TF, WP = "Words: TF-IDF", "Words: word presence"
tables = []


# ====================================================================== T1: Holmes by model
def plan_lines(clause):
    """The levels E1's clauses set for an AUC (0.80 for the whole works; 0.70 for the openings, which is also the level under which E1
    is falsified and the level asked of the works a model does not name) or for the interval's lower bound. Their numbers are spelled
    in the clause names of the results, e.g. auc_at_least_0.80."""
    import re
    for f, where in [(HM[t], ("E1",)) for t in ORDER] + [(HW, ("E1", "tfidf_centroid"))]:
        if f.has(*where, "verdict_within_era", "clauses"):
            return sorted({float(m.group(1)) for k in f.get(*where, "verdict_within_era", "clauses").v if clause in k for m in [re.search(r"(\d\.\d+)$", k)] if m})
    return []


LINES, LOWER_LINES = plan_lines("auc_at_least"), plan_lines("ci_lower_at_least")


def stored(default, *sources):
    """A level of the plan as the results store it: the first of (file, key path) that is there; `default` (words, no number) if none is."""
    for f, keys in sources:
        if f.has(*keys):
            return f.get(*keys).v
    return default


def not_named_level():
    """The AUC the plan's rule asks of the works a model does not name, as the rule is quoted in the Holmes results ("... AUC at least 0.70 ...")."""
    import re
    for t in ORDER:
        if HM[t].has("recognition", "not_confined_to_works_the_model_names", "E1_rule"):
            m = re.search(r"AUC at least (\d\.\d+)", HM[t].get("recognition", "not_confined_to_works_the_model_names", "E1_rule").v)
            if m:
                return m.group(1)
    return None


LEVEL = not_named_level()
NEED = stored(None, *[(HM[t], ("E2", "positive_control", "need")) for t in ORDER])
NEED_OF = stored(None, *[(HM[t], ("E2", "positive_control", "n")) for t in ORDER])


def e1_auc(f, *where):
    return interval(f.get(*where, "within_era", "auc"), f.get(*where, "within_era", "ci95"), lines=LINES, lower_lines=LOWER_LINES if where[-1] == "A_name_free" else ())


def e2_share(f, *where):
    return fx(f.get(*where, "share"), 3)


def e2_share_p(f, *where):
    return join("; p ", fx(f.get(*where, "share"), 3), pv(f.get(*where, "p")))


def named(f, side, *top):
    r = (*top, "E1", side)
    return join("; ", num(f.get(*r, "n")), interval(f.get(*r, "auc_within_era"), f.get(*r, "ci95"), lines=LINES))


def followup(fn):
    """A cell of the follow-up, for the models that have one (the plan fixes it for a met E1)."""
    return {COL[t]: C(lambda f=HF[t]: fn(f)) if HF[t] else cell("no follow-up" if HM[t] else NOT_YET) for t in ORDER}


def words_e1(key, *where):
    return C(lambda: e1_auc(HW, "E1", key, *where))


def rho_p(f, key):
    return join("; p ", fx(f.get("E3", key, "rho")), pv(f.get("E3", key, "p_one_sided")))


T = Table("T1", "Holmes: recognition, identity (E1), heredity (E2), selection (E3)", MODELS + [TF, WP], first="",
          intro="Name-free texts, strongest sentence at 30% of scale. AUC: the chance that a screen work scores above a detective film of its own era "
                "(0.5 is a coin toss). E1's verdict follows PLAN Section 8: met when all six clauses hold, falsified when the AUC is too low or its interval "
                "includes 0.5, not evaluable when the median canon is too small to judge. The two word columns put the same statistic on words in place of units: for E1 the TF-IDF centroid and the canon of words, "
                "for E2 the TF-IDF cosine and word presence.")
T.row("Recognition: screen works the model names as Sherlock Holmes", per_model(HR, lambda f: of(f.get("summary", "screen_works_named"), f.get("summary", "of"))))
T.row("Recognition: detective films it names as Sherlock Holmes", per_model(HR, lambda f: of(f.get("summary", "detective_films_named_as_holmes"), f.get("summary", "of_negatives"))))
T.row("Recognition alone: within-era AUC of the log-probability the model gives the hero's name",
      per_model(HR, lambda f: fx(f.get("summary", "within_era_auc_of_the_log_probability_of_the_hero_name"))))
T.row("E1 identity: within-era AUC (95% interval)", {**per_model(HM, lambda f: e1_auc(f, "E1", "A_name_free")), TF: words_e1("tfidf_centroid", "A_name_free"),
                                                     WP: words_e1("word_presence_canon_share", "A_name_free")})
T.row("E1: 95th percentile of that AUC when the labels are shuffled (the level clause 3 asks the AUC to exceed)", {
    **per_model(HM, lambda f: fx(f.get("E1", "A_name_free", "within_era", "null_label_shuffle", "p95"))),
    TF: C(lambda: fx(HW.get("E1", "tfidf_centroid", "A_name_free", "within_era", "null_label_shuffle", "p95"))),
    WP: C(lambda: fx(HW.get("E1", "word_presence_canon_share", "A_name_free", "within_era", "null_label_shuffle", "p95")))})
T.row("E1 on the openings only: within-era AUC (95% interval)", {**per_model(HM, lambda f: e1_auc(f, "E1", "openings")), TF: words_e1("tfidf_centroid", "openings"),
                                                                 WP: words_e1("word_presence_canon_share", "openings")})
T.row("E1 with the iconography sentences dropped: within-era AUC (95% interval)", {
    **per_model(HM, lambda f: e1_auc(f, "E1", "A_iconography_dropped")), TF: words_e1("tfidf_centroid", "A_iconography_dropped"),
    WP: words_e1("word_presence_canon_share", "A_iconography_dropped")})
T.row("E1 on the original texts, names in (the ceiling): within-era AUC (95% interval)", {
    **per_model(HM, lambda f: e1_auc(f, "E1", "ceiling_original")), TF: words_e1("tfidf_centroid", "ceiling_original"), WP: words_e1("word_presence_canon_share", "ceiling_original")})
T.row("E1: median size of the canon (units; words in the word column)", {
    **per_model(HM, lambda f: num(f.get("E1", "A_name_free", "canon_units_median"))), WP: C(lambda: num(HW.get("E1", "word_presence_canon_share", "A_name_free", "canon_units_median")))})
T.row("E1 verdict by the six clauses", {**per_model(HM, lambda f: word(f.get("E1", "verdict_within_era", "verdict"))),
                                        TF: C(lambda: word(HW.get("E1", "tfidf_centroid", "verdict_within_era", "verdict"))),
                                        WP: C(lambda: word(HW.get("E1", "word_presence_canon_share", "verdict_within_era", "verdict")))})
T.row("E1 for the screen works the model names: number; within-era AUC (95% interval)", per_model(HM, lambda f: named(f, "named", "recognition")))
T.row("E1 for the screen works it does not name: number; within-era AUC (95% interval)", per_model(HM, lambda f: named(f, "not_named", "recognition")))
T.row("E1: is the result confined to the works the model names? (the plan's rule: the works not named must reach " + (LEVEL or "a set level") + " with an interval that excludes 0.5)", per_model(HM, lambda f: confined(f.get("confirmatory", "E1_not_confined_to_works_the_model_names"))))
T.row("E1 follow-up, the token after each code masked too: within-era AUC (95% interval)", followup(
    lambda f: interval(f.get("E1_name_free", "token_after_each_code_masked_too", "within_era_auc"), f.get("E1_name_free", "token_after_each_code_masked_too", "within_era_ci95"), lines=LINES)))
T.row("E1 follow-up, the screen works it does not name: number; within-era AUC (95% interval)", followup(lambda f: named(f, "not_named", "recognition", "token_after_each_code_masked_too")))
T.row("E1 follow-up: is the result confined to the works the model names? (the same rule, in the follow-up's reading)", followup(lambda f: confined(f.get("recognition", "token_after_each_code_masked_too", "not_confined_to_works_the_model_names"))))
E2 = ("E2", "primary", "mean_over_sources")
T.row("E2 heredity: share of stories ranked above the documented source, averaged over source stories", {
    **per_model(HM, lambda f: e2_share(f, *E2)), TF: C(lambda: e2_share(HW, "E2", "tfidf_cosine", *E2[1:])), WP: C(lambda: e2_share(HW, "E2", "word_presence", *E2[1:]))})
T.row("E2: the same share under random relabelling (chance)", {
    **per_model(HM, lambda f: fx(f.get(*E2, "null_mean"), 3)), TF: C(lambda: fx(HW.get("E2", "tfidf_cosine", *E2[1:], "null_mean"), 3)),
    WP: C(lambda: fx(HW.get("E2", "word_presence", *E2[1:], "null_mean"), 3))})
T.row("E2: p", {**per_model(HM, lambda f: pv(f.get(*E2, "p"))), TF: C(lambda: pv(HW.get("E2", "tfidf_cosine", *E2[1:], "p"))), WP: C(lambda: pv(HW.get("E2", "word_presence", *E2[1:], "p")))})


def control(f, *where):
    return note(of(f.get(*where, "in_top_two"), f.get(*where, "n")), "") if f.get(*where, "passes").v else note(of(f.get(*where, "in_top_two"), f.get(*where, "n")), " (fails)")


T.row("E2 positive control: straight adaptations whose own story is in the top two" + (f" ({NEED} of {NEED_OF} needed)" if NEED else ""), {
    **per_model(HM, lambda f: control(f, "E2", "positive_control")), TF: C(lambda: control(HW, "E2", "tfidf_cosine", "positive_control")),
    WP: C(lambda: control(HW, "E2", "word_presence", "positive_control"))})
def control_by_story(f, *where):
    """The control apart for the most adapted story (the one with most straight adaptations) and for the other straight adaptations."""
    by, rows = f.get(*where, "by_story"), f.get(*where, "rows")
    most = max(by.v, key=lambda k: by.v[k]["adaptations"])
    a = [r for r in rows.v if int(most) in r["sources"]]
    b = [r for r in rows.v if int(most) not in r["sources"]]
    top = lambda rs: sum(r["rank"] <= 2 for r in rs)                                                  # noqa: E731
    return cell(f"{top(a)} of {len(a)}; {top(b)} of {len(b)}", Val(f"{len(rows.v)} rows: rank of the own story, by whether story {most} is among the sources", f.name, (*where, "rows")), by)


T.row("E2 positive control apart: the versions of The Hound of the Baskervilles; the other straight adaptations", {
    **per_model(HM, lambda f: control_by_story(f, "E2", "positive_control")), TF: C(lambda: control_by_story(HW, "E2", "tfidf_cosine", "positive_control")),
    WP: C(lambda: control_by_story(HW, "E2", "word_presence", "positive_control"))})
for label, key in (("E2 without The Hound of the Baskervilles, averaged over source stories: share; p", "without_the_most_adapted_source_2"),
                   ("E2 on the elements-only works (a secondary set), averaged over source stories: share; p", "elements_only_works_tier_C_sources")):
    T.row(label, {**per_model(HM, lambda f, k=key: e2_share_p(f, "E2", k, "mean_over_sources")), TF: C(lambda k=key: e2_share_p(HW, "E2", "tfidf_cosine", k, "mean_over_sources")),
                  WP: C(lambda k=key: e2_share_p(HW, "E2", "word_presence", k, "mean_over_sources"))})
T.row("E2 by v2's statistic, averaged over adaptations, not over source stories: share; p", {
    **per_model(HM, lambda f: e2_share_p(f, "E2", "primary", "mean_over_adaptations")), TF: C(lambda: e2_share_p(HW, "E2", "tfidf_cosine", "primary", "mean_over_adaptations")),
    WP: C(lambda: e2_share_p(HW, "E2", "word_presence", "primary", "mean_over_adaptations"))})
T.row("Residual test (reported, not confirmatory): p of the source-averaged share, the larger of its two forms (nine word measures; every word and code measure)", per_model(HM, lambda f: pv(f.get("confirmatory", "residual_p_larger_of_the_two_forms"))))
T.row("E3 selection, four covariates (as registered): rank correlation with page views; one-sided p", per_model(HM, lambda f: rho_p(f, "primary")))
T.row("E3 with v2's three covariates (reported beside it): rank correlation; one-sided p", per_model(HM, lambda f: rho_p(f, "v2_covariates_only")))
T.row("E3: permutations behind each p (four covariates; three covariates)", per_model(HM, lambda f: join("; ", num(f.get("E3", "primary", "n_perm")), num(f.get("E3", "v2_covariates_only", "n_perm")))))
tables.append(T)

# ---- T1b: what reaches E1 with no model
T = Table("T1b", "Holmes E1 with no model: what words, the placeholder codes and length alone reach", [
    "Within-era AUC (95% interval)", "95th percentile under shuffled labels", "Openings only", "Iconography sentences dropped", "Verdict by the six clauses"], first="Score",
    intro="Name-free texts, the same held-out procedure as the units' E1.")
for label, key in (("TF-IDF centroid", "tfidf_centroid"), ("Canon of words (share of the canon present)", "word_presence_canon_share"), ("Number of distinct codes alone", "distinct_codes_alone"),
                   ("Sentence count alone", "length_alone_sentences"), ("Word count alone", "length_alone_words")):
    T.row(label, {"Within-era AUC (95% interval)": words_e1(key, "A_name_free"),
                  "95th percentile under shuffled labels": C(lambda k=key: fx(HW.get("E1", k, "A_name_free", "within_era", "null_label_shuffle", "p95"))),
                  "Openings only": words_e1(key, "openings"), "Iconography sentences dropped": words_e1(key, "A_iconography_dropped"),
                  "Verdict by the six clauses": C(lambda k=key: word(HW.get("E1", k, "verdict_within_era", "verdict")))})
tables.append(T)

# ====================================================================== T2: remakes by model
PS, PALL, PSRC = "primary | set", "primary | all", "primary | source_industry"
BM, EMB = "Words: BM25", "Sentence embedding"
SET_LABEL = {"firm_other_cross_industry": "firm pairs, other cross-industry", "firm_same_industry": "firm pairs, same industry", "firm_all": "all firm pairs", "loose_all": "loose pairs"}


def words_heredity(measure):
    """Where the heredity of a word measure is stored: the primary model's file, or the words-only file while that is not there."""
    return (RM[PRIMARY], ("heredity", "by_words", measure, "hub_corrected")) if RM[PRIMARY] else (RW, ("analyses", PS, "heredity_by_words", measure, "hub_corrected"))


def her_columns(fn, part="not_copied"):
    """One heredity row: the models, the three word measures and the embedding. fn(file, key path of the block) -> cell."""
    cells = per_model(RM, lambda f: fn(f, ("heredity", part)))
    for col, m in ((TF, "tfidf_cosine"), (BM, "bm25"), (WP, "word_presence")):
        f, keys = words_heredity(m)
        cells[col] = C(lambda f=f, keys=keys: fn(f, (*keys, part)))
    cells[EMB] = C(lambda: fn(RM[PRIMARY], ("embedding", "in_the_units_place", "heredity", "hub_corrected", part)))
    return cells


def control_copied(f, keys):
    c = of(f.get(*keys, "source_strictly_above_every_candidate"), f.get(*keys, "entries"))
    return c if f.get(*keys, "passed").v else note(c, " (fails: " + p_share(f.get(*keys, "needed").v) + " needed)")


def p_share(x):
    return pct_text(x, 0)


T = Table("T2a", "Remakes: recognition and heredity", MODELS + [TF, BM, WP, EMB],
          intro="The primary set: firm pairs whose two films come from different Indian industries. Candidates: films of the Indian industries other than the remake's own, "
                "never later than the remake and never of its family. Each measure ranks the candidates by its hub-adjusted similarity to the remake. "
                "The word and embedding columns put the same statistic on a word measure, or on a general-purpose sentence embedding, in place of the units.")
T.row("Primary pairs the model recognises (it names the source, or identifies either film by title)",
      per_model(RR, lambda f: of(f.get("summary", "primary_pairs_recognised"), f.get("summary", "of_primary_pairs"))))
T.row("Primary pairs where it names the source or identifies both films",
      per_model(RR, lambda f: of(f.get("summary", "primary_pairs_recognised_both_films"), f.get("summary", "of_primary_pairs"))))
T.row("Heredity test, on the pairs that are not copied: number of pairs", her_columns(lambda f, k: num(f.get(*k, "entries"))))
T.row("Pairs not copied: mean share of candidate films ranked above the documented source, averaged over families (the test's statistic)", her_columns(lambda f, k: fx(f.get(*k, "S"), 3)))
T.row("Pairs not copied: the same share under random relabelling (chance)", her_columns(lambda f, k: fx(f.get(*k, "null_mean"), 3)))
T.row("Pairs not copied: p of the heredity test", her_columns(lambda f, k: pv(f.get(*k, "p"))))
T.row("Pairs not copied: documented source ranked first (strictly above every candidate)", her_columns(lambda f, k: of(f.get(*k, "source_strictly_above_every_candidate"), f.get(*k, "entries"))))
T.row("Pairs not copied: documented source among the first ten", her_columns(lambda f, k: of(f.get(*k, "source_in_the_top_10"), f.get(*k, "entries"))))
SHARE_NEEDED = stored(None, (RW, ("settings", "positive_control_share_needed")), *[(RM[t], ("settings", "positive_control_share_needed")) for t in ORDER])
T.row("Positive control, the copied pairs: documented source ranked first" + (f" ({pct_text(SHARE_NEEDED, 0)} needed)" if SHARE_NEEDED else ""), her_columns(control_copied, "positive_control_copied_entries"))
T.row("All primary pairs, copied ones included: mean share above the source; p", her_columns(lambda f, k: join("; p ", fx(f.get(*k, "S"), 3), pv(f.get(*k, "p"))), "all_entries"))
T.row("All primary pairs: documented source ranked first", her_columns(lambda f, k: of(f.get(*k, "source_strictly_above_every_candidate"), f.get(*k, "entries")), "all_entries"))
tables.append(T)


NOISE_COLUMNS = "39 columns (registered)"                          # the registered word model, in remakes_null_size.json


def noise_p(z):
    """The noise-calibrated p of a Z (PLAN Section 16, entry 3) and the count behind it: the share of the pure-noise runs
    whose Z is at or above z, the observed value counted as one run. The formula of noise_p() in remake_null_size.py."""
    zs = RN.get("by_columns", NOISE_COLUMNS, "Z_sorted").v
    at_or_above = sum(1 for x in zs if x >= z)
    return (1 + at_or_above) / (1 + len(zs)), at_or_above, len(zs)


def noise_p_cell(z):
    p, at_or_above, n = noise_p(z.v)
    floor = "; this p cannot be smaller" if at_or_above == 0 else ""
    return cell(f"{p_text(p)} ({at_or_above:,} of {n:,} noise runs reach it{floor})", z,
                Val(f"{n} values; (1 + runs at or above Z) / (1 + runs)", RN.name, ("by_columns", NOISE_COLUMNS, "Z_sorted")))


def z_p(f, *where):
    for i in range(1, len(where) + 1):                             # a block the run left out holds a sentence saying so
        if isinstance(f.get(*where[:i]).v, str):
            return cell(f.get(*where[:i]).v, f.get(*where[:i]))
    return join("; p ", fx(f.get(*where, "Z")), pv(f.get(*where, "p")))


def frame(f, label, fn):
    """A cell of another set or frame: the analysis, or why the model has none."""
    if f.has("analyses", label):
        return fn(f)
    s, fr = label.split(" | ")
    why = [x for x in f.get("analyses_left_out").v if x["set"] == s and x["frame"] == fr]
    if why:
        return cell("not run: " + why[0]["why"], Val(why[0]["why"], f.name, ("analyses_left_out",)))
    return cell(NA)


def panel(label):
    """Stand-ins below 0.05 for one set and frame, from the words-only file."""
    return of(RW.get("stand_in_panel_by_set_and_frame", label, "below_0.05"), RW.get("stand_in_panel_by_set_and_frame", label, "n"))


def conditions(f):
    c = f.get("decision", "conditions")
    held = [k for k, v in c.v.items() if v]
    fails = [k.split(" ")[0] for k, v in c.v.items() if not v]
    return cell(f"{len(held)} of {len(c.v)}" + (f" (not condition {', '.join(fails)})" if fails else ""), c)


def other_frame(f, label):
    return frame(f, label, lambda f: join("; ", join("", cell("Z "), fx(f.get("analyses", label, "slope", "Z"))), join("", cell("p "), pv(f.get("analyses", label, "slope", "p"))),
                                          note(fx(f.get("analyses", label, "slope", "effective_pairs_left_open"), 1), " pairs left open")))


def other_frame_standins(f, label):
    return frame(f, label, lambda f: join("; ", join("", cell("largest Z "), fx(f.get("analyses", label, "standins", "z_max"))), note(panel(label), " below 0.05"),
                                          word(RW.get("stand_in_panel_by_set_and_frame", label, "reading"))))


EU = "Sentence embedding in the units' place"
emb = ("embedding", "in_the_units_place")
T = Table("T2b", "Remakes: the decision test (do the units add to the word measures?)", MODELS + [EU],
          intro="Primary set in its primary frame, strongest sentence at 30% of scale. A word model gives every candidate a probability of being the source; "
                "the test asks whether the units still favour the documented source in the pairs the words leave open. Z: the units' advantage, summed over families, "
                "in standard units. A stand-in is a word-only measure put through the same test in the units' place.")
T.row("Pairs the word model leaves open (effective number)", {**per_model(RM, lambda f: fx(f.get("decision", "slope", "effective_pairs_left_open"), 1)),
                                                              EU: C(lambda: fx(RM[PRIMARY].get(*emb, "slope", "effective_pairs_left_open"), 1))})
T.row("Z", {**per_model(RM, lambda f: fx(f.get("decision", "slope", "Z"))), EU: C(lambda: fx(RM[PRIMARY].get(*emb, "slope", "Z")))})
T.row("p as registered (sign flips over families)", {**per_model(RM, lambda f: pv(f.get("decision", "slope", "p"))), EU: C(lambda: pv(RM[PRIMARY].get(*emb, "slope", "p")))})
T.row("p calibrated on pure noise: the share of pure-noise runs whose Z is at least as large, the observed value counted as one run",
      {**per_model(RM, lambda f: noise_p_cell(f.get("decision", "slope", "Z"))), EU: C(lambda: noise_p_cell(RM[PRIMARY].get(*emb, "slope", "Z")))})
T.row("Gain in held-out log-likelihood (the log-probability the model gives the documented sources, summed over the pairs) when the measure is added to the word model", {**per_model(RM, lambda f: fx(f.get("decision", "gain"), 1)),
                                                                                      EU: C(lambda: fx(RM[PRIMARY].get(*emb, "gain", "gain"), 1))})
T.row("Largest Z among the word-only stand-ins (the units' Z must be above it)", {**per_model(RM, lambda f: fx(f.get("decision", "standins_largest_Z"))),
                                                                                  EU: C(lambda: fx(RM[PRIMARY].get(*emb, "standins", "z_max")))})
FS = ("analyses", PS, "family_sums")
T.row("Families whose sum favours the documented source; families whose sum goes against it", per_model(RM, lambda f: join("; ", num(f.get("decision", "slope", "families_positive")),
                                                                                                                         num(f.get("decision", "slope", "families_negative")))))
T.row("How concentrated the result is: effective number of families carrying it (the square of the summed absolute family sums over the sum of their squares), of the families with a sum that is not zero", per_model(RM, lambda f: join(
    " of ", fx(f.get(*FS, "effective_number_of_families"), 1), num(f.get(*FS, "families_with_a_non_zero_sum")))))
T.row("Share of the squared family sums held by the five largest families", per_model(RM, lambda f: pct(f.get(*FS, "share_of_the_sum_of_squares_in_the_five_largest"), 0)))
PLANES_NEEDED = stored(None, (RW, ("settings", "planes_needed")), *[(RM[t], ("settings", "planes_needed")) for t in ORDER])
N_PLANES = len(stored([], (RW, ("settings", "planes")), *[(RM[t], ("settings", "planes")) for t in ORDER]))
NEEDED_TEXT = f"at least {PLANES_NEEDED} of the {N_PLANES}" if PLANES_NEEDED else "a set number"
T.row(f"Rule settings with p below 0.05 ({NEEDED_TEXT} are needed; table T5)", per_model(RM, lambda f: of(f.get("decision", "planes_below_0.05"), f.get("decision", "planes_available"))))
T.row("Pairs the model does not recognise: number; pairs left open; Z; p", per_model(RM, lambda f: join(
    "; ", num(f.get("decision", "recall_rule", "entries_not_recognised")), note(fx(f.get("decision", "recall_rule", "E_n"), 1), " left open"),
    join("", cell("Z "), fx(f.get("decision", "recall_rule", "Z_n"))), join("", cell("p "), pv(f.get("decision", "recall_rule", "p_n"))))))
T.row("Does it pass on its own numbers: p below 0.05, a positive gain, Z above every stand-in (conditions 3 to 5)?", {
    **per_model(RM, lambda f: yes_no(f.get("analyses", PS, "conditions_3_4_5_hold"))), EU: C(lambda: yes_no(RM[PRIMARY].get(*emb, "passes")))})
T.row("Conditions of the plan that hold (all seven are needed for the claim)", per_model(RM, conditions))
T.row("Outcome by the plan's seven conditions", per_model(RM, outcome))
T.row("Other frame, every film read as a candidate: Z; p; pairs left open", per_model(RM, lambda f: other_frame(f, PALL)))
T.row("Every film read: the stand-ins there", per_model(RM, lambda f: other_frame_standins(f, PALL)))
T.row("Other frame, films of the source's industry: Z; p; pairs left open", per_model(RM, lambda f: other_frame(f, PSRC)))
T.row("Films of the source's industry: the stand-ins there", per_model(RM, lambda f: other_frame_standins(f, PSRC)))
T.row("Without the units that are above their scale in any film: Z; p", per_model(RM, lambda f: z_p(f, "scale_robustness", "slope")))
T.row("With the sentence embedding added to the word columns: Z; p", per_model(RM, lambda f: z_p(f, "embedding", "units_with_the_embedding_among_the_word_columns", "slope")))
tables.append(T)

# ====================================================================== T3: the primary model's decision test by subset and setting
P = RM[PRIMARY]
A = ("analyses", PS)
T3C = ["Pairs", "Pairs the words leave open", "Z", "p", "Stand-ins below 0.05 on the same pairs or setting"]
CAL = ("primary_calibration",)


def slope_cells(f, keys, standins=None):
    """Pairs, open pairs, Z and p of one slope block; Z and p are not read where no pair is left open."""
    def z_or_none(fn, k):
        return fn(f.get(*keys, k)) if f.get(*keys, "effective_pairs_left_open").v >= 0.05 else cell("none: no pair is left open", f.get(*keys, "effective_pairs_left_open"))
    cells = {"Pairs": C(lambda: num(f.get(*keys, "entries"))), "Pairs the words leave open": C(lambda: fx(f.get(*keys, "effective_pairs_left_open"), 1)),
             "Z": C(lambda: z_or_none(fx, "Z")), "p": C(lambda: z_or_none(pv, "p"))}
    if standins:
        def panel_share():
            k, n, limit = RW.get(*standins, "below_0.05"), RW.get(*standins, "n"), RW.get("settings", "standin_share_limit")
            c = join("", of(k, n), cell("", limit))
            return note(c, f" (over the {pct_text(limit.v, 0)} allowed)") if k.v / n.v > limit.v else c
        cells[T3C[4]] = C(panel_share)
    return cells


def span(vals, d):
    lo, hi = min(vals, key=lambda v: v.v), max(vals, key=lambda v: v.v)
    fmt = (lambda x: f"{int(x):,}") if d is None else (lambda x: f"{x:.{d}f}")
    return cell(fmt(lo.v) if fmt(lo.v) == fmt(hi.v) else f"{fmt(lo.v)} to {fmt(hi.v)}", *vals)


def families_left_out():
    fam = P.get(*A, "subsets", "each_of_the_largest_families_left_out")
    g = lambda k: [P.get(*A, "subsets", "each_of_the_largest_families_left_out", f, k) for f in fam.v]      # noqa: E731
    ps = g("p")
    return {"Pairs": span(g("entries"), None), "Pairs the words leave open": span(g("effective_pairs_left_open"), 1), "Z": span(g("Z"), 2),
            "p": cell("at most " + p_text(max(v.v for v in ps)), *ps)}


by = (*CAL, "standins_by_subset_of_the_entries_descriptive")
T = Table("T3", f"Remakes, {NAMES[PRIMARY]}: the decision test by subset and setting", T3C, first="Pairs or setting",
          intro="The subsets are sums of the primary fit's contributions over some of the pairs; nothing is refitted. The settings refit the test with one thing changed. "
                "The last column counts the word-only stand-ins that reach p below 0.05 in the units' place on the same pairs or setting: where they are more than the plan allows, "
                "the row is not calibrated by words alone and makes no claim. p as registered (sign flips over families).")
for label, key, st in (("All primary pairs", "all_entries", (*by, "all_entries")), ("Pairs that are not copied", "not_copied", (*by, "not_copied")), ("Copied pairs", "copied", None),
                       ("Pairs with almost no copied wording (copy share under 1%)", "copy_share_under_1pct", (*by, "copy_share_under_1pct")),
                       ("Pairs the model recognises", "recognised", None), ("Pairs it does not recognise", "not_recognised", None),
                       ("Source is a document of v2's reference corpus", "source_is_a_document_of_v2_reference_corpus", (*by, "source_is_a_document_of_v2_reference_corpus")),
                       ("Source is not a document of v2's reference corpus", "source_is_not_a_document_of_v2_reference_corpus", (*by, "source_is_not_a_document_of_v2_reference_corpus")),
                       ("Only the pairs the words do not already settle (source probability under 0.99)",
                        "contributions_set_to_zero_where_the_held_out_source_probability_is_0.99_or_more", (*by, "held_out_source_probability_under_0.99"))):
    T.row(label, slope_cells(P, (*A, "subsets", key), st))
T.row("Each of the five families with the largest sums left out in turn (range)", {c: C(lambda c=c: families_left_out()[c]) for c in T3C[:4]})
B = (*A, "beside_the_decision")
T.row("Penalty 1 in place of 3", slope_cells(P, (*B, "lambda_1", "slope"), (*CAL, "other_penalties", "1", "standins", "summary")))
T.row("Penalty 10 in place of 3", slope_cells(P, (*B, "lambda_10", "slope"), (*CAL, "other_penalties", "10", "standins", "summary")))
T.row("No hub adjustment of the unit rate", slope_cells(P, (*B, "no_hub_correction", "slope"), (*CAL, "no_hub_correction", "summary")))
T.row("Cells the word model rules out left out of the sum", slope_cells(P, (*B, "slope_with_pruned_cells"), (*CAL, "pruning_check", "summary_pruned")))
tables.append(T)

# ====================================================================== T4: secondary sets, every model that has them
T4C = ["Candidates", "Pairs", "Pairs the words leave open", "Z", "p", "Gain in held-out log-likelihood", "Largest stand-in Z", "Stand-ins below 0.05", "Reading by words alone"]
T = Table("T4", "Remakes: secondary sets (the decision test)", T4C, first="Model and set",
          intro="Reported, never pooled into the primary set. Z and p as registered, strongest sentence at 30% of scale. Where many stand-ins reach p below 0.05 the word model "
                "does not span the words on that set, and the row makes no claim either way.")
T4B = ["Pairs not copied", "Mean share of candidates ranked above the source", "Chance", "p", "Source ranked first", "Copied pairs with the source ranked first"]
Tb = Table("T4b", "Remakes: secondary sets (heredity by units)", T4B, first="Model and set", intro="The heredity test of T2a on each secondary set with the candidates of T4, hub-adjusted unit rate, pairs that are not copied. "
                                                                                           "Source ranked first: strictly above every candidate.")
for t in ORDER:
    f = RM[t]
    if not f:
        T.row(NAMES[t], {T4C[0]: cell(NOT_YET)})
        Tb.row(NAMES[t], {T4B[0]: cell(NOT_YET)})
        continue
    for s, label in SET_LABEL.items():
        a, name = f"{s} | set", f"{NAMES[t]}: {label}"
        if not f.has("analyses", a):
            T.row(name, {T4C[0]: C(lambda f=f, a=a: frame(f, a, None))})
            Tb.row(name, {T4B[0]: C(lambda f=f, a=a: frame(f, a, None))})
            continue
        k = ("analyses", a)
        T.row(name, {"Candidates": C(lambda f=f, s=s: word(f.get("settings", "frame_set", s))), "Pairs": C(lambda f=f, k=k: num(f.get(*k, "entries"))),
                     "Pairs the words leave open": C(lambda f=f, k=k: fx(f.get(*k, "slope", "effective_pairs_left_open"), 1)), "Z": C(lambda f=f, k=k: fx(f.get(*k, "slope", "Z"))),
                     "p": C(lambda f=f, k=k: pv(f.get(*k, "slope", "p"))), "Gain in held-out log-likelihood": C(lambda f=f, k=k: fx(f.get(*k, "gain", "gain"), 1)),
                     "Largest stand-in Z": C(lambda f=f, k=k: fx(f.get(*k, "standins", "z_max"))), "Stand-ins below 0.05": C(lambda a=a: panel(a)),
                     "Reading by words alone": C(lambda a=a: word(RW.get("stand_in_panel_by_set_and_frame", a, "reading")))})
        h = (*k, "heredity", "hub_corrected")
        Tb.row(name, {"Pairs not copied": C(lambda f=f, h=h: num(f.get(*h, "not_copied", "entries"))),
                      "Mean share of candidates ranked above the source": C(lambda f=f, h=h: fx(f.get(*h, "not_copied", "S"), 3)),
                      "Chance": C(lambda f=f, h=h: fx(f.get(*h, "not_copied", "null_mean"), 3)), "p": C(lambda f=f, h=h: pv(f.get(*h, "not_copied", "p"))),
                      "Source ranked first": C(lambda f=f, h=h: of(f.get(*h, "not_copied", "source_strictly_above_every_candidate"), f.get(*h, "not_copied", "entries"))),
                      "Copied pairs with the source ranked first": C(lambda f=f, h=h: control_copied(f, (*h, "positive_control_copied_entries")))
                      if f.has(*h, "positive_control_copied_entries", "entries") else cell("no copied pair")})
tables += [T, Tb]

# ====================================================================== T5: the eight rule settings
SCORE = {"top1": "strongest sentence", "top3": "mean of the three strongest sentences"}
planes = next((src.get("settings", "planes").v for src in [RW, *RM.values()] if src), [])
T = Table("T5", "Remakes: the eight rule settings of the decision test", MODELS, first="A unit is present when",
          intro=f"Primary set, primary frame. Z; p as registered. The plan asks for p below 0.05 in {NEEDED_TEXT}.")
for pl in planes:
    score, cut = pl.split("_")
    T.row(f"{SCORE[score]} at {int(cut)}% of scale", per_model(RM, lambda f, pl=pl: z_p(f, *A, "planes", "rows", pl) if not f or f.has(*A, "planes", "rows", pl) else cell("not read at this setting")))
T.row("Settings with p below 0.05", per_model(RM, lambda f: of(f.get("decision", "planes_below_0.05"), f.get("decision", "planes_available"))))
tables.append(T)

# ====================================================================== T6: calibration with no model
V = "Result"
PC, WC, NS = (*CAL, "withheld_vocabulary_positive_control"), ("claim_conditions_that_words_decide",), ("by_columns", "39 columns (registered)")
T = Table("T6", "Calibration of the decision test with no model", [V], first="Check",
          intro="Everything here uses words or random numbers in the units' place, on the primary set in its primary frame. A test that worked as its p says would reject 5 in 100.")
sa = ("word_columns", "stand_ins_after")
T.row("Word-only stand-ins with p below 0.05", {V: C(lambda: of(RW.get(*sa, "below_0.05"), RW.get(*sa, "n")))})
T.row("Largest Z among the stand-ins; the measure", {V: C(lambda: join("; ", fx(RW.get(*sa, "z_max")), word(RW.get(*sa, "z_max_measure"))))})
rf = (*CAL, "random_families_of_stand_ins", "all")


def of_pct(k, n, share, d=1):
    return join(" ", of(k, n), cell(f"({pct_text(share.v, d)})", share))


T.row("Random word-only measures with p below 0.05 (they took no part in choosing the word columns)", {V: C(lambda: of_pct(RW.get(*rf, "below_0.05"), RW.get(*rf, "draws"), RW.get(*rf, "rate")))})
cp = (*CAL, "noisy_and_curved_copies", "all_copies_that_do_not_use_the_labels")
T.row("Noisy and curved copies of the word columns with p below 0.05", {V: C(lambda: of_pct(RW.get(*cp, "below_0.05"), RW.get(*cp, "draws"), RW.get(*cp, "rate_below_0.05")))})
T.row("Pure noise in the units' place: runs with p below 0.05", {V: C(lambda: of_pct(RN.get(*NS, "below_0.05"), RN.get("draws"), RN.get(*NS, "rate_0.05")))})
T.row("Pure noise: runs with p below 0.01", {V: C(lambda: of_pct(RN.get(*NS, "below_0.01"), RN.get("draws"), RN.get(*NS, "rate_0.01")))})
T.row("Pure noise: share of runs whose Z is above the stand-ins' largest Z (the error rate of the plan's condition 5)", {V: C(lambda: join(" of ", pct(RN.get(*NS, "share_Z_above_the_stand_ins_largest_Z"), 2), note(num(RN.get("draws")), " runs")))})
T.row("Pure noise: standard deviation of Z over the runs (the sign flips assume 1)", {V: C(lambda: fx(RN.get(*NS, "sd_Z")))})
T.row("The test's own positive control (word measures on a withheld half of the vocabulary): runs with p below 0.05", {V: C(lambda: note(
    of(RW.get(*PC, "below_0_05"), RW.get(*PC, "n")), f" ({len(RW.get(*PC, 'measures').v)} word measures in each of {len(RW.get(*PC, 'seeds').v)} random splits of the vocabulary)"))})
T.row("Positive control: range of Z", {V: C(lambda: join(" to ", fx(RW.get(*PC, "smallest_Z"), 1), fx(RW.get(*PC, "largest_Z"), 1)))})
sb = (*WC, "smallest_boost_detected_with_80pct_power")
T.row("Smallest planted effect found 80% of the time, in standard deviations of the measure among a remake's candidates: where the units' word-related part is one of the word columns",
      {V: C(lambda: fx(RW.get(*sb, "word twin plus noise")))})
T.row("The same where the rest of the unit measure is pure noise", {V: C(lambda: fx(RW.get(*sb, "pure noise")))})
tables.append(T)

# ====================================================================== T7: the confirmatory tests, when confirmatory.py has been run
TEST = {"holmes_E2": "Holmes E2, heredity", "holmes_E3": "Holmes E3, selection", "remakes_heredity": "Remakes, heredity", "remakes_decision": "Remakes, the decision test"}
HEAD = {"p": "p", "holm_within_corpus": "Holm-adjusted p within the corpus", "holm_over_all_four": "Holm-adjusted p over the four tests", "reading": "Reading at 0.05"}
NOISE_BLOCK = "with_the_noise_calibrated_p_of_the_decision_test"
NOISE_HEAD = "Noise-calibrated version: "
missing = []
if CF:
    tests = list(CF.get("p").v)

    def per_test(d):
        return isinstance(d, dict) and set(d) == set(tests)
    # every block that has one value per test, whatever confirmatory.py adds later: (key path, column heading)
    keyed = [((k,), HEAD.get(k, k.replace("_", " "))) for k, v in CF.d.items() if per_test(v)]
    for k, v in CF.d.items():                                                                       # and the same blocks one level down (the noise-calibrated versions)
        if isinstance(v, dict) and not per_test(v):
            keyed += [((k, k2), (NOISE_HEAD if k == NOISE_BLOCK else k.replace("_", " ") + ": ") + HEAD.get(k2, k2.replace("_", " "))) for k2, v2 in v.items() if per_test(v2)]
    cols = [c for _, c in keyed]
    VERDICT = HEAD["reading"] if HEAD["reading"] in cols else (cols[-1] if cols else None)           # a verdict is not a p: it goes under the reading
    T = Table("T7", f"Confirmatory tests ({NAMES.get(CF.get('model').v, CF.get('model').v)}, primary setting)", cols, first="Test",
              intro="One family per corpus; each test is read at 0.05. Holmes E1 has a verdict and no p. Holm's adjustment is within each corpus (Holmes: E2 and E3; "
                    "remakes: heredity and the decision test) and over the four together. The last columns repeat it with the decision test's noise-calibrated p "
                    "(table T2b) in place of its registered p (the columns headed noise-calibrated version); that version enters no rule of the plan.")

    def conf(keys, t):
        v = CF.get(*keys, t)
        return cell("no p") if v.v is None else pv(v) if isinstance(v.v, float) else word(v)
    if cols:
        T.row("Holmes E1, identity", {cols[0]: cell("no p: a verdict by six clauses"), VERDICT: C(lambda: word(CF.get("holmes", "E1", "verdict")))})
    for t in tests:
        T.row(TEST.get(t, t), {c: C(lambda keys=keys, t=t: conf(keys, t)) for keys, c in keyed})
    if cols:
        T.row("Remakes, the decision test: the outcome by the plan's seven conditions", {VERDICT: C(lambda: outcome(RM[CF.get("model").v]))})
    tables.append(T)
else:
    missing.append("T7 (confirmatory tests): results/v2/second/confirmatory.json is not there yet; it is made by v2/second/confirmatory.py")


# ====================================================================== the same number in two files
checks = []


def same(what, *vals, tol=1e-9):
    """Record whether values that should be one number are. vals: functions returning a Val; one whose file is missing is skipped."""
    got = []
    for fn in vals:
        try:
            got.append(fn())
        except (Missing, KeyError, TypeError, IndexError):
            pass
    if len(got) < 2:
        return
    v = [g.v for g in got]
    agree = all(abs(x - v[0]) <= tol for x in v) if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) else all(x == v[0] for x in v)
    checks.append({"what": what, "agree": bool(agree), "values": [{"file": g.file, "key": list(g.keys), "value": g.v} for g in got]})


for t in ORDER:
    h, r, n = HM[t], RM[t], NAMES[t]
    same(f"Holmes, {n}: screen works named, recognition file against analysis file", lambda: HR[t].get("summary", "screen_works_named"), lambda: h.get("recognition", "E1", "named", "n"))
    same(f"Holmes, {n}: E2 p, confirmatory block against the test", lambda: h.get("confirmatory", "E2_p"), lambda: h.get("E2", "primary", "mean_over_sources", "p"))
    same(f"Holmes, {n}: E3 p, confirmatory block against the test", lambda: h.get("confirmatory", "E3_p"), lambda: h.get("E3", "primary", "p_one_sided"))
    same(f"Holmes, {n}: E1 verdict, confirmatory block against the within-era verdict", lambda: h.get("confirmatory", "E1_verdict"), lambda: h.get("E1", "verdict_within_era", "verdict"))
    same(f"Holmes, {n}: residual p, the larger of the two forms against the two forms", lambda: h.get("confirmatory", "residual_p_larger_of_the_two_forms"),
         lambda: max((h.get("residual", k, "mean_over_sources", "p") for k in ("decision", "every_measure")), key=lambda v: v.v))
    same(f"Holmes, {n}: E1 within-era AUC, the follow-up's copy of the committed run against the committed run",
         lambda: HF[t].get("E1_name_free", "codes_masked_only_as_committed", "within_era_auc"), lambda: h.get("E1", "A_name_free", "within_era", "auc"))
    same(f"Remakes, {n}: primary pairs recognised, recognition file against analysis file", lambda: RR[t].get("summary", "primary_pairs_recognised"),
         lambda: r.get("recognition", "by_set", "primary", "recognised"), lambda: r.get("decision", "recall_rule", "entries_recognised"))
    same(f"Remakes, {n}: pairs the words leave open, analysis file against the words-only file", lambda: r.get("decision", "slope", "effective_pairs_left_open"),
         lambda: RW.get("claim_conditions_that_words_decide", "1 evaluable: the word model leaves at least 5 effective pairs open", "effective_pairs_left_open"))
    same(f"Remakes, {n}: the stand-ins' largest Z, analysis file against the words-only file and the pure-noise file", lambda: r.get("decision", "standins_largest_Z"),
         lambda: RW.get("claim_conditions_that_words_decide", "stand_ins_largest_Z_the_units_must_exceed", "Z"), lambda: RN.get("stand_ins_largest_Z"))
    same(f"Remakes, {n}: heredity of TF-IDF, analysis file against the words-only file", lambda: r.get("heredity", "by_words", "tfidf_cosine", "hub_corrected", "not_copied", "S"),
         lambda: RW.get("analyses", PS, "heredity_by_words", "tfidf_cosine", "hub_corrected", "not_copied", "S"))
    same(f"Remakes, {n}: the decision block against the primary analysis (Z)", lambda: r.get("decision", "slope", "Z"), lambda: r.get("analyses", PS, "slope", "Z"))
    if t != PRIMARY:
        same(f"Remakes: the embedding in the units' place, {n}'s file against {NAMES[PRIMARY]}'s (Z)", lambda: r.get(*emb, "slope", "Z"), lambda: RM[PRIMARY].get(*emb, "slope", "Z"))
same("Pure noise: runs below 0.05 over the draws against the stored rate", lambda: Val(RN.get(*NS, "below_0.05").v / RN.get("draws").v, RN.name, (*NS, "below_0.05 / draws")), lambda: RN.get(*NS, "rate_0.05"))
same("Stand-ins below 0.05 on the primary set: the panel against the word columns' record", lambda: RW.get("stand_in_panel_by_set_and_frame", PS, "below_0.05"), lambda: RW.get(*sa, "below_0.05"))
same("Confirmatory file: Holmes E2 p against the Holmes file", lambda: CF.get("p", "holmes_E2"), lambda: HM[PRIMARY].get("confirmatory", "E2_p"))
same("Confirmatory file: Holmes E3 p against the Holmes file", lambda: CF.get("p", "holmes_E3"), lambda: HM[PRIMARY].get("confirmatory", "E3_p"))
same("Confirmatory file: remakes heredity p against the remakes file", lambda: CF.get("p", "remakes_heredity"), lambda: RM[PRIMARY].get("heredity", "not_copied", "p"))
same("Confirmatory file: remakes decision p against the remakes file", lambda: CF.get("remakes", "decision", "p"), lambda: RM[PRIMARY].get("decision", "slope", "p"))
same("Confirmatory file: the noise-calibrated p of the decision test against the one computed here from the pure-noise runs",
     lambda: CF.get(NOISE_BLOCK, "noise_calibrated_p"), lambda: Val(noise_p(RM[PRIMARY].get("decision", "slope", "Z").v)[0], RN.name, ("by_columns", NOISE_COLUMNS, "Z_sorted")))
same("Confirmatory file: the outcome of the decision test against the remakes file", lambda: CF.get("remakes", "decision", "outcome"), lambda: RM[PRIMARY].get("decision", "outcome"))
same("Confirmatory file: the E1 verdict against the Holmes file", lambda: CF.get("holmes", "E1", "verdict"), lambda: HM[PRIMARY].get("E1", "verdict_within_era", "verdict"))


# ====================================================================== write
def floors():
    """The smallest p each sampled null can give, from the numbers of draws the results record."""
    draws = sorted({int(n) for n in (stored(None, *[(HM[t], ("e2_null", "n_draws")) for t in ORDER]), stored(None, (RW, ("settings", "sign_flips"))),
                                    stored(None, (RW, ("settings", "heredity_draws")))) if n}, reverse=True)
    if not draws:
        return "A sampled null cannot give a p smaller than 1 in its number of draws."
    return "A p of " + ", or of ".join(f"{p_text(1 / n)} from {n:,} random draws" for n in draws) + ", is the smallest that null can give."


def main():
    found = {"Holmes": [NAMES[t] for t in ORDER if HM[t]], "remakes": [NAMES[t] for t in ORDER if RM[t]]}
    waiting = {"Holmes": [NAMES[t] for t in ORDER if not HM[t]], "remakes": [NAMES[t] for t in ORDER if not RM[t]]}
    head = ("# Tables for the two new corpora\n\nWritten by `v2/second/paper_tables.py` from the committed results in `results/v2/second/`. Do not edit by hand.\n\n"
            f"Results found: Holmes for {', '.join(found['Holmes']) or 'no model'}; remakes for {', '.join(found['remakes']) or 'no model'}."
            + "".join(f" Not there yet ({c}): {', '.join(w)}." for c, w in waiting.items() if w) + " " + NAMES[PRIMARY] + " is the primary model: it decides every test; the others are replications."
            + "\n\n" + floors() +
              " The noise-calibrated p of the decision test cannot be smaller than 1 in (1 + the number of pure-noise runs)."
              "\n\nRounding: a number is rounded once, from the unrounded value in the file. "
              "A p under 0.001 is written out to two significant digits; a p that would round to 1.00 is written to three decimals; "
              "a p that would round to exactly 0.05 is given more digits. Where rounding would print an AUC as one of the plan's levels "
              f"({', '.join(f'{x:.2f}' for x in sorted(LINES, reverse=True))}; {', '.join(f'{x:.2f}' for x in LOWER_LINES)} for the lower bound of the interval) "
              "although it is under that level, the unrounded value is given beside it.")
    text = head + "\n\n" + "\n\n".join(t.md() for t in tables) + ("\n\n## Not made\n\n" + "\n".join("- " + m for m in missing) if missing else "") + "\n"
    (OUT / "paper_tables.md").write_text(text)
    json.dump({"what": "The cells of results/v2/second/paper_tables.md, each with its text and the values it was written from (file, key path, value). Made by v2/second/paper_tables.py.",
               "results_found": found, "results_not_there_yet": waiting, "tables_not_made": missing, "checks": checks, "tables": {t.key: t.json() for t in tables}},
              open(OUT / "paper_tables.json", "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT / "paper_tables.md", "and", OUT / "paper_tables.json", f"({len(tables)} tables: {', '.join(t.key for t in tables)})")
    print("results found:", found, "| not there yet:", {c: w for c, w in waiting.items() if w}, "| not made:", missing)
    differ = [c for c in checks if not c["agree"]]
    print(f"checks of one number in two files: {len(checks)}, of which {len(differ)} differ")
    for c in differ:
        print("  DIFFERS:", c["what"], [x["value"] for x in c["values"]])
    bad = [(t.key, label, c) for t in tables for label, cells in t.rows for c, v in cells.items() if v["text"] == NA]
    if bad:
        print(f"{len(bad)} cells are n/a (a key is missing):", bad[:12])


if __name__ == "__main__":
    main()
