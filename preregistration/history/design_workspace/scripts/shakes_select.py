"""Select Shakespeare film adaptations for T1d from scout/films_all.jsonl. Metadata only: no similarity between
texts is computed. Writes scout/shakespeare_adaptations.jsonl (no texts; texts stay in films_all.jsonl) and
scout/shakespeare_summary.json.

Inclusion: a film or TV film (not a series, documentary, stage work or novel) with a plot section of >= 8
sentences and at least one documented play source: a direct 'Films based on <play>' category, a Wikidata
P144 'based on' link, or a lead sentence that says it is based on / adapted from / a retelling of the play.
Flags: keeps_names (>= 2 distinctive principal names of the play appear in the plot section), partial (1),
changes_names (0); mentions_source (plot names Shakespeare or the play title: meta films, staged-play plots);
in_dev (already in the DEV corpus)."""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT
from films import NAMES, PLAY_TITLES, BASED, lead_plays
from t3_common import DEV_TITLES

# principal names too common (or too generic) to count as evidence that an adaptation kept the play's names
COMMON = {"Henry", "John", "Richard", "Edward", "Philip", "Arthur", "Elizabeth", "Anne", "Katherine", "Margaret", "Paris",
          "Page", "Ford", "Hero", "Bottom", "Diana", "Julia", "Prince", "Countess", "York", "Exeter", "Warwick", "Lucius",
          "Oliver", "Valentine", "Malcolm", "Duncan", "More", "Lincoln", "Surrey", "Hal", "Quickly", "Shallow", "Pistol",
          "Toby", "Ariel", "Miranda", "Olivia", "Sebastian", "Jessica", "Portia", "Emilia", "Helena", "Marina", "Celia",
          "Joan", "Talbot", "Clifford", "Gloucester", "Buckingham", "Clarence", "Elizabeth", "Constance", "Hubert", "Flavius",
          "Antony", "Arthur", "Claudio", "Angelo", "Mariana", "Viola", "Dauphin", "Edgar", "Edmund", "Regan", "Ulysses",
          "Hector", "Achilles", "Caesar", "Brutus", "Cassius", "Casca", "Lear"}
EPONYMOUS = {"Romeo and Juliet", "Hamlet", "Macbeth", "Othello", "King Lear", "Antony and Cleopatra", "Titus Andronicus",
             "Julius Caesar", "Coriolanus", "Timon of Athens", "Troilus and Cressida", "Cymbeline", "Pericles", "Richard II",
             "Richard III", "Henry IV Part 1", "Henry IV Part 2", "Henry V", "Henry VI Part 1", "Henry VI Part 2", "Henry VI Part 3",
             "Henry VIII", "King John", "Edward III", "Sir Thomas More"}
FILMISH = {"film", "tv film", "film?", "short film", "unknown"}

import os


def shortdesc_of(r):
    f = SCOUT / "raw" / "pages" / (re.sub(r"[^\w]+", "_", r["query"]) + ".json")
    if f.exists():
        return json.load(open(f)).get("pageprops", {}).get("wikibase-shortdesc")
    return None


rows = {}
for l in open(SCOUT / "films_all.jsonl"):
    r = json.loads(l)
    if r.get("missing") or not r.get("title"):
        continue
    rows.setdefault(r["title"], r)

out = []
for t, r in rows.items():
    wd_p144 = sorted({x.split(":")[0] for x in r.get("wikidata_plays", []) if x.endswith(":P144")})
    wd_p941 = sorted({x.split(":")[0] for x in r.get("wikidata_plays", []) if x.endswith(":P941")})
    lead_ev = sorted({p for s in r.get("evidence", []) for p in lead_plays(s)})
    sources = defaultdict(list)
    for p in r.get("direct_play_cats", []):
        sources[p].append("category")
    for p in wd_p144:
        sources[p].append("wikidata_P144")
    for p in lead_ev:
        sources[p].append("lead_sentence")
    for p in wd_p941:
        sources[p].append("wikidata_P941_inspired_by")
    documented = {p: v for p, v in sources.items() if set(v) - {"wikidata_P941_inspired_by"}}
    med = r.get("medium", "unknown")
    lead_l = (r.get("lead") or "")[:400].lower()
    is_film = med in FILMISH and not re.search(r"\b(opera|ballet|television series|tv series|miniseries|documentary|episode|stage production|musical play)\b", lead_l[:250])
    # medium from Wikipedia's short description where there is one (e.g. '1948 film by Laurence Olivier', 'Musical')
    sd = shortdesc_of(r)
    if sd:
        sdl = sd.lower()
        is_film = bool(re.search(r"\b(film|movie|cartoon|animated short|short subject)\b", sdl)) and not re.search(
            r"\b(television series|tv series|web series|miniseries|episode|opera|ballet|novel|video game|documentary|stage musical|soundtrack|album|song)\b", sdl)
    elif re.search(r"\((musical|novel|play|opera|ballet|album|song|tv series|television series)\)", t, re.I):
        is_film = False
    if re.search(r"\(([\w-]+ )*film\)$", t):
        is_film = True
    is_novel = bool(sd and re.search(r"\b(novel|book)\b", sd.lower())) or "(novel)" in t
    plot = " ".join(r.get("sentences", []))
    names = {}
    for p in documented or sources:
        found = [n for n in NAMES[p] if re.search(r"\b" + re.escape(n) + r"\b", plot)]
        names[p] = dict(found=found, distinctive=[n for n in found if n not in COMMON], n_principal=len(NAMES[p]))
    best = max((len(v["distinctive"]) for v in names.values()), default=0)
    anyfound = max((len(v["found"]) for v in names.values()), default=0)
    flag = "keeps_names" if best >= 2 else ("partial" if best == 1 or anyfound >= 2 else "changes_names")
    # a play title counts as a mention only when it is not just the hero's name (Hamlet, Romeo and Juliet, Macbeth...)
    mentions = len(re.findall(r"Shakespeare", plot)) + sum(len(re.findall(re.escape(PLAY_TITLES[p]), plot)) for p in names
                                                          if p not in EPONYMOUS)
    out.append(dict(title=t, medium=med, is_film=is_film, n_sentences=r["n_sentences"], n_sentences_capped=r["n_sentences_capped"],
                    plot_section=r.get("plot_section"), documented_sources={p: sorted(set(v)) for p, v in documented.items()},
                    inspired_only=sorted(set(sources) - set(documented)), names=names, names_flag=flag,
                    mentions_source=mentions, in_dev=t in DEV_TITLES, shortdesc=sd, is_novel=is_novel, evidence=(r.get("evidence") or [None])[0],
                    remake_of_dev=bool(re.match(r"^The Lion King", t)) and t not in DEV_TITLES,
                    from_wikidata=r.get("source") == "wikidata"))

eligible = [o for o in out if o["is_film"] and o["n_sentences"] >= 8 and o["documented_sources"]]
primary = [o for o in eligible if not o["in_dev"] and o["mentions_source"] == 0 and len(o["documented_sources"]) == 1]
with open(SCOUT / "shakespeare_adaptations.jsonl", "w") as f:
    for o in sorted(out, key=lambda o: (not (o in eligible), o["title"])):
        o["eligible"] = o in eligible
        o["primary_candidate"] = o in primary
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

plays = [json.loads(l) for l in open(SCOUT / "plays.jsonl")]
by_play = defaultdict(lambda: Counter())
for o in eligible:
    for p in o["documented_sources"]:
        by_play[p]["eligible"] += 1
        if o in primary:
            by_play[p]["primary"] += 1
            by_play[p][o["names_flag"]] += 1
summ = dict(
    plays=dict(n=len(plays), with_plot_ge8=sum(p["n_sentences"] >= 8 for p in plays),
               sentences_capped40=sum(min(p["n_sentences"], 40) for p in plays), sentences_uncapped=sum(p["n_sentences"] for p in plays),
               missing=["Cardenio (lost)", "Love's Labour's Won (lost)"]),
    articles_considered=len(out), eligible=len(eligible), primary=len(primary),
    eligible_breakdown=dict(in_dev=sum(o["in_dev"] for o in eligible), mentions_source=sum(o["mentions_source"] > 0 for o in eligible),
                            multi_play=sum(len(o["documented_sources"]) > 1 for o in eligible)),
    primary_names_flags=Counter(o["names_flag"] for o in primary),
    primary_loose_changes_names=sum(o["names_flag"] == "changes_names" for o in primary),
    plays_with_ge2_loose=sorted(p for p, c in by_play.items() if c["changes_names"] >= 2),
    loose_sentences_capped40=sum(o["n_sentences_capped"] for o in primary if o["names_flag"] == "changes_names"),
    primary_sentences_capped40=sum(o["n_sentences_capped"] for o in primary),
    eligible_sentences_capped40=sum(o["n_sentences_capped"] for o in eligible),
    primary_evidence=Counter("+".join(sorted({e for v in o["documented_sources"].values() for e in v})) for o in primary),
    by_play={p: dict(c) for p, c in sorted(by_play.items(), key=lambda kv: -kv[1]["eligible"])},
    plays_with_primary_ge3=sorted(p for p, c in by_play.items() if c["primary"] >= 3),
    plays_with_changes_names_primary=sorted(p for p, c in by_play.items() if c["changes_names"] >= 1),
    in_dev_eligible=[o["title"] for o in eligible if o["in_dev"]],
    novel_adaptations_ge8=sorted(o["title"] for o in out if o["is_novel"] and o["n_sentences"] >= 8 and o["documented_sources"]),
)
json.dump(summ, open(SCOUT / "shakespeare_summary.json", "w"), indent=1, default=dict)
print(json.dumps(summ, indent=1, default=dict))
