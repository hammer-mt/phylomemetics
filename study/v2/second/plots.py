"""The plot section of a Wikipedia article as plain sentences, from its wikitext (run from study/).

The Batman corpus read the plain-text extract of each article, one request a page (study/fetch_corpus.py). The API now
allows this machine only a few requests a minute, so the second corpora read the wikitext, 50 pages a request, and
this module turns a plot section into plain text. The section names, the cut into sentences, the first 40 sentences
and the floor of 8 are fetch_corpus.py's own. `check` compares the result with the plain-text extract for every page
that has both in the cache.

    ../.venv/bin/python v2/second/plots.py check
"""
import json
import re
import sys
from pathlib import Path

import mwparserfromhell

sys.path.insert(0, ".")
import fetch_corpus as F  # noqa: E402  (SECTION_NAMES, sentences, MAX_SENTENCES)

KEEP_FIRST = {"lang", "nihongo", "em", "strong", "nowrap", "nobr", "small", "smallcaps", "sc", "angbr", "visible anchor", "vanchor", "isbn"}
SPACE = {"nbsp", "spaces", "space", "hair space", "thin space", "nbs", "sp"}
DASH = {"snd": " – ", "spnd": " – ", "spaced ndash": " – ", "spaced en dash": " – ", "ndash": "–", "en dash": "–", "mdash": "—", "em dash": "—",
        "emdash": "—", "spaced em dash": " — ", "'": "'", "'s": "'s", "\"": "\"", "!": "|", "sic": "", "=": "="}


def template_text(t):
    """What a template shows, for the few that carry words in running prose; nothing for the rest."""
    name = str(t.name).strip().lower()
    if name in DASH:
        return DASH[name]
    if name in SPACE:
        return " "
    if name == "lang" and len(t.params) >= 2:
        return str(t.params[1].value)
    if name.startswith("lang-") and t.params:
        return str(t.params[0].value)
    if name in KEEP_FIRST and t.params:
        return str(t.params[0].value)
    if name in ("convert", "cvt") and len(t.params) >= 2:
        return f"{t.params[0].value} {t.params[1].value}"
    if name in ("frac", "fraction") and t.params:
        return "/".join(str(p.value) for p in t.params)
    if name in ("abbr",) and t.params:
        return str(t.params[0].value)
    if name in ("quote", "blockquote", "quotation", "cquote") and t.params:
        return str(t.params[0].value)
    return ""


def plain(wikitext):
    """Wikitext prose as plain text: links by their label, references, files, tables and comments dropped."""
    s = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    s = re.sub(r"<ref[^>/]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<(gallery|timeline|imagemap)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"^\{\|.*?^\|\}", "", s, flags=re.S | re.M)
    code = mwparserfromhell.parse(s)
    for link in code.filter_wikilinks(recursive=True):
        if re.match(r"\s*(File|Image|Category|Media)\s*:", str(link.title), flags=re.I):
            try:
                code.remove(link)
            except ValueError:
                pass
    for _ in range(4):                                         # innermost templates first
        ts = [t for t in code.filter_templates(recursive=True) if "{{" not in str(t)[2:-2]]
        if not ts:
            break
        for t in ts:
            try:
                code.replace(t, template_text(t))
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.M)
    text = re.sub(r"\s*\((?:roughly |about |approximately )?equivalent to[^()]*\)", "", text)     # a sum in today's money, added by a template: not plot
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)                  # brackets a dropped template left empty
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return re.sub(r"[ \t]+", " ", text)


def plot_section(wikitext):
    """(section name, plain body) of the first level-2 section whose name the Batman fetcher accepts, or (None, None)."""
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", wikitext, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        head = plain(parts[i]).strip()
        if F.SECTION_NAMES.match(head):
            body = re.sub(r"^===+.*?===+\s*$", "", parts[i + 1], flags=re.M)
            return head, plain(body)
    return None, None


def lead(wikitext):
    """The lead as plain text (everything before the first level-2 heading)."""
    return plain(re.split(r"^==\s*[^=].*?==\s*$", wikitext, maxsplit=1, flags=re.M)[0]).strip()


def plot_sentences(wikitext):
    head, body = plot_section(wikitext)
    return head, (F.sentences(body) if body else [])


def check():
    cache = Path("v2/cache/wiki")
    n = same = 0
    worst = []
    for f in sorted((cache / "extract").glob("*.json")):
        e = json.loads(f.read_text())
        w = cache / "wikitext" / f.name
        if not w.exists() or e["missing"]:
            continue
        text = e["extract"]
        parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", text, flags=re.M)
        ref = None
        for i in range(1, len(parts) - 1, 2):
            if F.SECTION_NAMES.match(parts[i]):
                ref = F.sentences(re.sub(r"^===+.*?===+\s*$", "", parts[i + 1], flags=re.M))
                break
        head, mine = plot_sentences(json.loads(w.read_text())["wikitext"])
        if ref is None and not mine:
            continue
        n += 1
        norm = lambda ss: [re.sub(r"\W+", " ", x).strip().lower() for x in (ss or [])]
        a, b = norm(ref), norm(mine)
        if a == b:
            same += 1
        else:
            diff = [(x, y) for x, y in zip(a, b) if x != y]
            worst.append((e["title"], len(a), len(b), diff[:1]))
    print(n, "pages with both; identical sentence lists (ignoring punctuation and case):", same)
    for w in worst[:12]:
        print("  differs:", w[0], w[1], w[2], str(w[3])[:300])


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
