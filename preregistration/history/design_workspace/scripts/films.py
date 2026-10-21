"""Fetch every article in the Shakespeare film/TV category tree plus named extra candidates, and record
metadata needed to pick E5 adaptations: plot section and sentence count, which play(s) Wikipedia says it is
based on (direct category membership and lead-text mentions), medium, whether the plot keeps the play's
character names, and leakage flags (plot text that names Shakespeare or the play).

Writes scout/films_all.jsonl (one row per article, with raw plot section text; not analysed)."""
import json
import re
import sys

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, fetch_extract, plot_section, sentences, split_sections, MAX_SENTENCES
from plays import PLAYS

# candidates named in the design brief or known loose adaptations, in case the category tree misses them
EXTRA = ["West Side Story (1961 film)", "West Side Story (2021 film)", "The Lion King", "10 Things I Hate About You",
         "She's the Man", "O (film)", "Ran (film)", "Throne of Blood", "Forbidden Planet", "My Own Private Idaho",
         "Haider (film)", "Omkara (2006 film)", "Maqbool", "Scotland, PA", "Warm Bodies (film)", "A Thousand Acres (film)",
         "Get Over It (film)", "The Bad Sleep Well", "Men of Respect", "King of Texas", "Jubal (film)", "All Night Long (1962 film)",
         "Deliver Us from Eva", "McLintock!", "Kiss Me Kate (film)", "Strange Brew", "The Banquet (2006 film)", "Prince of the Himalayas",
         "Romeo Must Die", "Gnomeo & Juliet", "China Girl (1987 film)", "Goliyon Ki Raasleela Ram-Leela", "Qayamat Se Qayamat Tak",
         "Angoor (1982 film)", "My Kingdom (film)", "The King Is Alive", "House of Strangers", "Broken Lance", "Yellow Sky",
         "Tempest (1982 film)", "Kaliyattam", "Joe Macbeth", "Macbeth (2006 film)", "Veeram (2016 film)", "The Lion King II: Simba's Pride",
         "Were the World Mine", "A Midsummer Night's Rave", "Hamlet (2000 film)", "Romeo + Juliet", "Much Ado About Nothing (2012 film)",
         "Coriolanus (film)", "Titus (film)", "Richard III (1995 film)", "As You Like It (2006 film)", "Cymbeline (film)",
         "Love's Labour's Lost (2000 film)", "Measure for Measure (2019 film)", "Tromeo and Juliet", "Iago (film)", "Othello (2001 film)",
         "The Hungry", "Jogi (2005 film)", "Arshinagar", "Issaq", "Ishaqzaade", "Bhrantibilas", "Do Dooni Chaar (1968 film)",
         "Ek Duuje Ke Liye", "Rosencrantz & Guildenstern Are Dead (film)", "The King (2019 film)", "Chimes at Midnight",
         "Big Business (1988 film)", "Hamlet Goes Business", "Karmayogi (2012 film)", "Ophelia (film)", "Strange Magic (film)",
         "Private Romeo", "Romeo and Juliet (1968 film)", "The Tempest (2010 film)", "Sharpe's Peninsula"]

PLAY_TITLES = {lab: re.sub(r" \(play\)$", "", t) for t, lab, _, _ in PLAYS}
ALIASES = {"Henry IV Part 1": ["Henry IV", "Henriad"], "Henry IV Part 2": ["Henry IV", "Henriad"],
           "Henry VI Part 1": ["Henry VI"], "Henry VI Part 2": ["Henry VI"], "Henry VI Part 3": ["Henry VI"],
           "Henry V": ["Henry V", "Henriad"], "Midsummer Night's Dream": ["Midsummer"], "Pericles": ["Pericles"]}
CAT2PLAY = {}
for t, lab, _, _ in PLAYS:
    base = re.sub(r" \(play\)$", "", t)
    CAT2PLAY[base] = lab
CAT2PLAY["Henry IV"] = "Henry IV Part 1"   # category is 'Films based on Henry IV (play)': both parts, resolved below
CAT2PLAY["Henry VI"] = "Henry VI Part 1"
NAMES = {lab: names for _, lab, _, names in PLAYS}
SKIP = re.compile(r"^(List of|Songs?|Category|.*\(song\)$|.*\(soundtrack\)$|.*\(album\)$|.*\(musical\)$|.*\(play\)$|.*\(opera\)$|.*\(ballet\)$)")
BASED = re.compile(r"(based on|adaptation of|adapted from|adapts|inspired by|retelling of|reimagining|re-imagining|version of|loosely|modern[- ]day|updat|transpos|retells|take on|reworking|derived from)", re.I)


def direct_plays(cats):
    out = []
    for c in cats:
        m = re.match(r"Category:(?:\w+ )*(?:[Ff]ilms|[Tt]elevision shows|[Tt]elevision films|[Tt]elevision plays|[Ww]orks) based on (.+?)(?: \(play\))?$", c)
        if m and m.group(1) in CAT2PLAY:
            out.append(CAT2PLAY[m.group(1)])
    return sorted(set(out))


def lead_plays(lead):
    found = []
    for lab, base in PLAY_TITLES.items():
        keys = [base] + ALIASES.get(lab, [])
        for k in keys:
            if re.search(r"\b" + re.escape(k) + r"\b", lead):
                found.append(lab)
                break
    # 'Henry IV' would match Part 1/2 both; keep both
    return sorted(set(found))


def medium(lead):
    l = lead[:500].lower()
    for key, val in [("television film", "tv film"), ("tv film", "tv film"), ("television movie", "tv film"), ("made-for-television", "tv film"),
                     ("made-for-tv", "tv film"), ("miniseries", "tv series"), ("television series", "tv series"), ("tv series", "tv series"),
                     ("web series", "tv series"), ("television play", "tv play"), ("anime", "tv series?"), ("opera", "opera?"), ("ballet", "ballet?"),
                     ("musical", "film?"), ("film", "film"), ("movie", "film"), ("documentary", "documentary")]:
        if key in l:
            return val
    return "unknown"


def main():
    cm = json.load(open(SCOUT / "category_members.json"))["members"]
    titles = [t for t in cm if not SKIP.match(t)] + [t for t in EXTRA if t not in cm]
    rows = []
    for i, t in enumerate(titles):
        page = fetch_extract(t)
        if "missing" in page:
            rows.append(dict(query=t, title=None, missing=True))
            continue
        text = page.get("extract") or ""
        lead, secs = split_sections(text)
        lead = re.sub(r"\s+", " ", lead).strip()
        name, body = plot_section(page)
        sents = sentences(body) if body else []
        cats = [c["title"] for c in page.get("categories", [])]
        dp, lp = direct_plays(cats), lead_plays(lead)
        # lead sentence(s) that state the relationship
        ev = [s for s in sentences(lead) if BASED.search(s) and (lead_plays(s) or "Shakespeare" in s)][:2]
        plot_txt = " ".join(sents)
        name_hits = {}
        for p in sorted(set(dp) | set(lp)):
            name_hits[p] = [n for n in NAMES[p] if re.search(r"\b" + re.escape(n) + r"\b", plot_txt)]
        rows.append(dict(query=t, title=page.get("title"), in_category_tree=t in cm, tree_paths=cm.get(t, []),
                         direct_play_cats=dp, lead_plays=lp, medium=medium(lead), lead=lead[:1200], evidence=ev,
                         plot_section=name, n_sentences=len(sents), n_sentences_capped=min(len(sents), MAX_SENTENCES),
                         plot_names_from_play=name_hits,
                         plot_mentions_shakespeare=len(re.findall(r"Shakespeare", plot_txt)),
                         plot_mentions_play_title={p: len(re.findall(re.escape(PLAY_TITLES[p]), plot_txt)) for p in name_hits},
                         categories=cats, sentences=sents, raw_section=body))
        if i % 25 == 0:
            print(i, "/", len(titles), t, flush=True)
    with open(SCOUT / "films_all.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
