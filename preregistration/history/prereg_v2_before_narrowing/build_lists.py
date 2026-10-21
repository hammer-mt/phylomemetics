"""Build the pre-registration's frozen work lists and inputs (prereg-v2). Metadata, documented relationships, text
hashes and agent-written inputs only: no similarity, score or outcome is computed. Deterministic.

  python prereg/build_lists.py memes       # memes_for_tracing.json: 34 memes (24 study memes + 10 neutral motifs);
                                           # merges the P1 example sets from prereg/agent/examples/ when present
  python prereg/build_lists.py relations   # merges the P1 relation typing (prereg/agent/relations/) into
                                           # documented_relatives.json
  python prereg/build_lists.py lists       # works_shakespeare.json, documented_relatives.json, culture_pool.json,
                                           # works_batman_extended.json, dev_units.json, planted_design.json
  python prereg/build_lists.py texts <stage>   # data assembly after F0 (PREREGISTRATION.md section 3):
        examples (P1, before F0) | pilot (P2, before F0) | dev | wikipedia | names | drift | zh | zhwiki | stories |
        long | t2n | codes

Every source this script reads is inside the project: prereg/sources/ holds gzipped copies of the scout caches
(Wikipedia/Wikidata metadata and plot sections fetched 2026-10-20 with fetch_corpus.py's section rule and sentence
splitter) and the three robustness-review files dev_units needs. Wikipedia text is CC BY-SA 4.0 (sources/LICENSE.md).
"""
import ast
import gzip
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRE = Path(__file__).resolve().parent
ROOT = PRE.parent
SOURCES = PRE / "sources"
AGENT = PRE / "agent"
DATA = PRE / "data"
CAP = 40                 # films, pool works and Batman works: first 40 plot sentences (fetch_corpus.py)
PLAY_CAP = 80            # Shakespeare plays: first 80 sentences after the commentary filter (only Troilus is cut)
SALT_VIG = "phylomemetics-T3-vigilante-v1|"
SALT_T2N = "phylomemetics-T2N-v2|"
SALT_DRIFT = "phylomemetics-T1b-drift-sample-v2|"
SALT_DESIGN = 20261020
N_VIG = 400
N_T2N_PER_STRATUM = 40
DRIFT_PER_PLAY = 6


def jl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def as_list(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)


def h(sentences):
    return hashlib.sha256("\n".join(sentences).encode("utf-8")).hexdigest()


def sha_order(items, salt):
    return sorted(items, key=lambda t: hashlib.sha256((salt + t).encode()).hexdigest())


def dump(obj, name):
    with open(PRE / name, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print("wrote", PRE / name)


DEV = jl(ROOT / "data" / "corpus.jsonl")
DEV_TITLES = {w["title"] for w in DEV} | {w["label"] for w in DEV}


# ----------------------------------------------------------------------------------------------- memes
# Ten neutral motifs: the first ten plantable items, in sha256('phylomemetics-neutral-motifs-v1|' + title) order, among
# the article pages of five Wikipedia categories (Literary motifs, Plot (narrative), Recurrent elements in fairy tales,
# Recurring elements in folklore, Narrative techniques), fetched 2026-10-21 (sources/neutral_motif_frame.json). A
# candidate is plantable when (P1) the article names an event, act or situation that a 1-3 sentence passage of a plot
# summary can show happening; (P2) it is not one of the 24 study memes or a near-synonym of one; (P3) it is not tied to
# one work, tradition or object; (P4) it can happen without the supernatural, so it fits every genre. The walk, with
# the reason for every rejection, is in sources/neutral_motif_walk.json. No neuron unit was looked at for these.
NEUTRAL_MEMES = [
    {"id": "closed_circle", "wiki": "Closed circle of suspects", "rank": 15, "label": "Closed circle of suspects",
     "definition": "A crime is committed among a small group cut off from the outside world, so the culprit must be one of them.",
     "keywords": ["suspect", "suspects", "suspicion", "culprit", "whodunit", "murderer", "cut off", "snowed in", "stranded",
                  "one of them", "locked room"],
     "decoy_hint": "a 'suspected' illness, a car stranded with no crime, a board game about suspects",
     "do_not_combine_with": []},
    {"id": "poisoned_garment", "wiki": "Poison dress", "rank": 17, "label": "Poisoned garment",
     "definition": "A character gives someone a garment or cloth that has been secretly poisoned, and wearing it harms or kills the wearer.",
     "keywords": ["poison", "poisoned", "venom", "toxic", "robe", "gown", "dress", "cloak", "shirt", "garment", "gift"],
     "decoy_hint": "a dress given as a gift that does not fit, poison ivy on a hike, a cloakroom ticket",
     "do_not_combine_with": ["antidote_race", "feigned_death_potion", "vat_of_chemicals"]},
    {"id": "rash_promise", "wiki": "Rash promise", "rank": 35, "label": "Rash promise",
     "definition": "A character makes a hasty, binding promise without foreseeing its cost, and is then forced to keep it at great loss.",
     "keywords": ["promise", "promised", "vow", "oath", "swear", "swore", "sworn", "pledge", "rash", "hasty", "gave his word",
                  "gave her word", "keep his word", "keep her word"],
     "decoy_hint": "a promising young athlete, a pledge drive on the radio, an oath of office read at a ceremony with no cost",
     "do_not_combine_with": ["revenge_vow", "deal_with_devil", "literal_loophole"]},
    {"id": "locked_out_lover", "wiki": "Paraklausithyron", "rank": 41, "label": "Locked-out lover",
     "definition": "A lover shut out of the beloved's house waits at the closed door through the night, pleading or singing to be let in.",
     "keywords": ["lover", "beloved", "door", "doorstep", "threshold", "locked out", "serenade", "plead", "lament",
                  "let him in", "let her in"],
     "decoy_hint": "a door locked because the keys are lost, a serenade at a wedding, a lover of fine wine",
     "do_not_combine_with": []},
    {"id": "self_fulfilling_prophecy", "wiki": "Self-fulfilling prophecy", "rank": 55, "label": "Self-fulfilling prophecy",
     "definition": "A prediction about what will happen leads people to act in ways that make it come true, often because they try to prevent it.",
     "keywords": ["prophecy", "prophesy", "prophet", "prediction", "predict", "foretell", "foretold", "fate", "destiny",
                  "oracle", "fortune-teller", "come true"],
     "decoy_hint": "a weather prediction that is simply right, a fortune-teller at a fair whose words change nothing",
     "do_not_combine_with": []},
    {"id": "love_triangle", "wiki": "Love triangle", "rank": 87, "label": "Love triangle",
     "definition": "Two characters love or compete for the same person, and the rivalry for that person drives the story.",
     "keywords": ["love triangle", "rival", "rivals", "rivalry", "jealous", "jealousy", "suitor", "suitors", "torn between",
                  "compete"],
     "decoy_hint": "business rivals with no romance, a child jealous of a sibling's new bicycle",
     "do_not_combine_with": []},
    {"id": "late_collision_meeting", "wiki": "Girl Late for School with Toast in Her Mouth", "rank": 93,
     "label": "Late-running collision meeting",
     "definition": "A character rushing because they are late, often eating on the run, collides with a stranger who turns out to matter later, typically a future love.",
     "keywords": ["late", "running late", "rush", "rushing", "hurry", "collide", "collision", "bump into", "bumps into",
                  "crash into", "toast", "breakfast"],
     "decoy_hint": "a late fee at the library, a toast at a wedding, a car collision in which nobody meets",
     "do_not_combine_with": []},
    {"id": "literal_loophole", "wiki": "Quibble (plot device)", "rank": 95, "label": "Literal loophole",
     "definition": "An agreement, oath or law is satisfied by its exact wording while its intent is defeated, through a clever literal reading.",
     "keywords": ["loophole", "technicality", "literally", "letter of the law", "exact words", "wording", "clause",
                  "fine print", "contract", "terms"],
     "decoy_hint": "a contract signed with no trick, reading the fine print on a cereal box, the terms of a lease read aloud",
     "do_not_combine_with": ["rash_promise", "deal_with_devil"]},
    {"id": "shoe_throwing", "wiki": "Shoe throwing", "rank": 125, "label": "Thrown shoe",
     "definition": "A character throws a shoe at someone, as an insult or protest, or after a departing couple for luck.",
     "keywords": ["shoe", "shoes", "boot", "boots", "sandal", "slipper", "footwear", "throw", "threw", "thrown", "hurl", "fling"],
     "decoy_hint": "shoe shopping, a boot sale, throwing a party",
     "do_not_combine_with": ["coin_flip"]},
    {"id": "exile", "wiki": "Exile", "rank": 130, "label": "Exile",
     "definition": "A character is banished from their home, country or community and forced to live away from it.",
     "keywords": ["exile", "exiled", "banish", "banished", "banishment", "expel", "expelled", "outcast", "cast out", "deport",
                  "deported", "homeland"],
     "decoy_hint": "being 'banished' from the kitchen for burning dinner, a novel titled Exile in a shop window",
     "do_not_combine_with": []},
]
EXAMPLE_SETS = ("explicit", "reworded", "far", "hard_negative")
K_EXAMPLES = 5


def build_memes():
    draft = json.load(open(PRE / "history" / "memes_for_tracing_draft_v1.json"))
    du = json.load(open(PRE / "dev_units.json")) if (PRE / "dev_units.json").exists() else {"t2_mapped": {}}
    memes = []
    for m in draft["memes"]:
        memes.append({"id": m["id"], "tier": m["tier"], "label": m["label"], "definition": m["definition"],
                      "keywords": m["keywords"], "decoy_hint": m["decoy_hint"],
                      "do_not_combine_with": m.get("do_not_combine_with", []), "dev_unit_qwen": m.get("dev_unit_qwen"),
                      "dev_class": (du["t2_mapped"].get(m["id"]) or {}).get("dev_class"),
                      "examples": {k: [] for k in EXAMPLE_SETS}, "source": "study meme (draft v1 label, definition and ban list)"})
    ids = {m["id"] for m in memes}
    for n in NEUTRAL_MEMES:
        assert n["id"] not in ids
        memes.append({"id": n["id"], "tier": "N", "label": n["label"], "definition": n["definition"], "keywords": n["keywords"],
                      "decoy_hint": n["decoy_hint"], "do_not_combine_with": n["do_not_combine_with"], "dev_unit_qwen": None,
                      "dev_class": None, "examples": {k: [] for k in EXAMPLE_SETS},
                      "source": f"neutral motif: Wikipedia '{n['wiki']}' (rank {n['rank']} in the seeded order)"})
    by = {m["id"]: m for m in memes}
    for m in memes:                       # make do_not_combine_with symmetric
        for o in m["do_not_combine_with"]:
            if m["id"] not in by[o]["do_not_combine_with"]:
                by[o]["do_not_combine_with"].append(m["id"])
    status = "not yet written (step P1)"
    ex_dir = AGENT / "examples"
    if (ex_dir / "examples_final.jsonl").exists():   # written by the P1 agents, screened (annotation_protocol.md A8)
        for r in jl(ex_dir / "examples_final.jsonl"):
            by[r["meme"]]["examples"][r["set"]] = r["sentences"]
        status = "written by P1 agents; see agent/examples/"
    doc = {"description": "The 34 named memes of T2 and T3: 24 study memes (Tier A orphan-vigilante family, Tier B Batman "
                          "story memes, Tier C general memes) and 10 neutral motifs (Tier N) drawn by seed from Wikipedia's "
                          "motif and plot-device categories before any unit was examined. Every method receives the same "
                          "inputs for a meme: its label, its definition and its example sentences. 'keywords' are the words "
                          "people usually use for the meme; they are the ban list for the banned, implicit, coded and control "
                          "versions of planted stories (with the derivational-family rule of analysis.ban_hits) and are never "
                          "given to any method. Example sets (written at step P1 by fresh agent sessions that see only the "
                          "label and definition, annotation_protocol.md A8): explicit (usual words), reworded (none of the "
                          "ban-list words), far (no crime city, colonial California, revolutionary France, billionaire "
                          "inventor or costumed avenger; screened against DEV Batman and cousin sentences), hard_negative "
                          "(the ban-list words in a sense that does not make the meme happen). dev_unit_qwen: the DEV unit "
                          "used for H3 (dev_units.json).",
           "tiers": {"A": "orphan-vigilante family layer (Mike Taylor's earlier book pp. 21-22)", "B": "Batman story memes",
                     "C": "general narrative memes", "N": "neutral motifs (seeded draw, sources/neutral_motif_frame.json)"},
           "k_examples": K_EXAMPLES, "example_sets": list(EXAMPLE_SETS), "examples_status": status, "memes": memes}
    dump(doc, "memes_for_tracing.json")
    return doc


# ----------------------------------------------------------------------------------------------- Shakespeare
COMMENTARY = re.compile(r"\b(Shakespeare\w*|themes?|interpret\w*|audiences?|critics?|critical|scholars?|scholarly|"
                        r"deus ex machina|the play's|the play does|is considered|are considered)\b", re.I)


def play_sentences(raw):
    """Plot sentences of a play: commentary sentences dropped by a fixed rule, then the first PLAY_CAP."""
    s = as_list(raw)
    kept = [x for x in s if not COMMENTARY.search(x)]
    return kept[:PLAY_CAP], len(s) - len(kept)


def evidence_tier(ev):
    kinds = set()
    for v in ev.values():
        kinds |= set(v if isinstance(v, list) else [v])
    if kinds & {"lead_sentence", "wikidata_P144"}:
        return "lead_or_p144"
    if "wikidata_P941_inspired_by" in kinds:
        return "p941"
    return "category_only"


def shakespeare():
    plays = jl(SOURCES / "plays.jsonl.gz")
    films_txt = {r["title"]: r for r in jl(SOURCES / "films_all.jsonl.gz") if r.get("title")}
    meta = jl(SOURCES / "shakespeare_adaptations.jsonl.gz")
    play_rows = []
    for p in plays:
        s, dropped = play_sentences(p["sentences"])
        play_rows.append({"id": p["label"], "title": p["title"], "section": p["plot_section"], "n_sentences": len(s),
                          "n_sentences_full": len(as_list(p["sentences"])), "n_commentary_dropped": dropped,
                          "principal_names": p["names"], "sha256": h(s)})
    film_rows = []
    for r in meta:
        if not r.get("eligible"):
            continue
        t = films_txt.get(r["title"], {})
        full = as_list(t.get("sentences", []))
        s = full[:CAP]
        srcs = sorted(r["documented_sources"])
        prim = bool(r.get("primary_candidate"))
        renamed = prim and r.get("names_flag") == "changes_names"
        tier = evidence_tier(r["documented_sources"])
        film_rows.append({"id": r["title"], "title": r["title"], "shortdesc": r.get("shortdesc"), "medium": r.get("medium"),
                          "sources": srcs, "evidence": {k: v for k, v in r["documented_sources"].items()}, "evidence_tier": tier,
                          "lead_evidence": r.get("evidence"), "names_flag": r.get("names_flag"), "names_found": r.get("names"),
                          "mentions_source": bool(r.get("mentions_source")), "in_dev": r.get("in_dev"),
                          "remake_of_dev": bool(r.get("remake_of_dev")), "primary": prim, "renamed": renamed,
                          "h5": bool(renamed and tier == "lead_or_p144" and not r.get("remake_of_dev")),
                          "n_sentences": len(s), "truncated": len(full) > CAP, "sha256": h(s) if s else None})
    known = {p["id"] for p in play_rows}
    for f in film_rows:
        assert all(s in known for s in f["sources"]), (f["id"], f["sources"])
    by_play = defaultdict(list)
    for f in film_rows:
        if f["h5"]:
            by_play[f["sources"][0]].append(f["id"])
    drift = set()
    for p, ids in by_play.items():
        drift |= set(sha_order(ids, SALT_DRIFT)[:DRIFT_PER_PLAY])
    for f in film_rows:
        f["drift_sample"] = f["id"] in drift
    prim = [f for f in film_rows if f["primary"]]
    ren = [f for f in prim if f["renamed"]]
    h5 = [f for f in prim if f["h5"]]
    c5 = Counter(f["sources"][0] for f in h5)
    out = {"description": "T1d fresh lineage: Shakespeare's plays (candidates) and their film adaptations (targets). A source is "
                          "documented by a 'Films based on <play>' category, a Wikidata P144 link or a lead sentence saying the "
                          "film is based on or adapted from the play; evidence_tier = lead_or_p144 when a lead sentence or P144 "
                          "documents it, category_only otherwise. Primary = one documented play, not in DEV, plot names neither "
                          "Shakespeare nor a non-eponymous play title. Renamed = primary and fewer than 2 of the play's "
                          "distinctive principal names appear in the plot (names_flag changes_names; this measures names only, "
                          "not setting or wording). H5 sample = renamed, evidence_tier lead_or_p144, not a remake of a DEV work. "
                          "drift_sample = at most 6 H5 films per play in sha256 order (T1b). Plays: commentary sentences "
                          "dropped by a fixed rule (build_lists.COMMENTARY), then the first 80 sentences.",
           "counts": {"plays": len(play_rows), "eligible_films": len(film_rows), "primary": len(prim), "renamed_primary": len(ren),
                      "h5": len(h5), "h5_plays": len(c5), "h5_by_play": dict(c5.most_common()),
                      "h5_single_film_plays": sorted(p for p, c in c5.items() if c == 1),
                      "renamed_category_only": sum(1 for f in ren if f["evidence_tier"] == "category_only"),
                      "drift_sample": len(drift), "drift_by_play": dict(Counter(f["sources"][0] for f in film_rows if f["drift_sample"]).most_common()),
                      "films_truncated_at_40": sum(1 for f in prim if f["truncated"]),
                      "plays_with_commentary_dropped": {p["id"]: p["n_commentary_dropped"] for p in play_rows if p["n_commentary_dropped"]},
                      "primary_by_play": dict(Counter(f["sources"][0] for f in prim).most_common())},
           "excluded_reference_film": "The Lion King (a documented Hamlet adaptation) is dropped from the reference films for T1d",
           "plays": play_rows, "films": film_rows}
    dump(out, "works_shakespeare.json")
    return out


# ----------------------------------------------------------------------------------------------- relatives and culture pool
CATEGORY_STRATA = [("superhero", {"superhero_film", "superhero_novel", "pulp_hero", "family_iron_man"}),
                   ("swashbuckler", {"swashbuckler_film", "outlaw_hero_film", "family_zorro", "family_pimpernel"}),
                   ("pulp_crime", {"pulp_story", "relative_monte_cristo", "vigilante_film"})]
OUT_OF_POOL_CATEGORY = {
    "The Bat (play)": "pulp_crime", "The Bat Whispers": "pulp_crime", "The Bat (1959 film)": "pulp_crime",
    "The Sign of the Four": "pulp_crime", "The Hound of the Baskervilles": "pulp_crime", "The Valley of Fear": "pulp_crime",
    "Dick Tracy (1990 film)": "pulp_crime", "Dick Tracy (1945 film)": "pulp_crime", "John Wick: Chapter 3 – Parabellum": "pulp_crime",
    "The Man Who Laughs (1928 film)": "random", "The Man Who Laughs": "random",
    '"Pimpernel" Smith': "swashbuckler", "Puss in Boots (2011 film)": "swashbuckler", "Puss in Boots: The Last Wish": "swashbuckler",
    "Watchmen": "superhero", "Birdman (film)": "superhero", "Jay and Silent Bob Strike Back": "superhero",
    "Scooby-Doo! Mask of the Blue Falcon": "superhero",
}
# genre rule for strata (primary): the work's own short description and first lead sentence, not the category crawl
GENRE_STRATA = [("superhero", re.compile(r"super-?\s?hero", re.I)),
                ("swashbuckler", re.compile(r"swashbuckl|pirate|musketeer|\bknights?\b|outlaw|highwayman|\bsword|cloak and dagger", re.I)),
                ("pulp_crime", re.compile(r"\bcrime|detective|myster|thriller|noir|gangster|vigilante|\bpulp|\bspy\b|heist|revenge", re.I))]
BOOK = [(1, "Zorro", re.compile(r"Zorro")), (1, "Scarlet Pimpernel", re.compile(r"Pimpernel|Eldorado \(novel\)")),
        (1, "Iron Man", re.compile(r"^Iron Man")), (2, "Spider-Man", re.compile(r"Spider-Man")),
        (2, "John Wick", re.compile(r"^John Wick")), (2, "Puss in Boots", re.compile(r"^Puss in Boots"))]
T3_RELATIONS = {"influence", "predecessor", "derivative", "sibling"}
# entities with a DEV cousin work (data/corpus.jsonl group 'cousin'): their TEST works are partly predictable from DEV
DEV_FRANCHISE = {"Zorro", "The Scarlet Pimpernel", "Iron Man", "Spider-Man", "The Shadow", "Sherlock Holmes", "Superman",
                 "The Bat (Rinehart and Hopwood play) / The Bat Whispers", "Puss in Boots (Shrek)", "Daredevil", "Count Dracula",
                 "The Count of Monte Cristo"}
# Batman lineage works by title that are not DC Batman stories (relabelled before F0, review round 1)
LINEAGE_RELABEL = {"The Wild World of Batwoman": ("relative", "Batwoman (Jerry Warren)"),
                   "Batman XXX": ("contested", "pornographic parody of Batman, not a DC work"),
                   "The People's Joker": ("contested", "parody of Batman, not a DC work"),
                   "Catwoman (film)": ("contested", "2004 Catwoman with no Batman connection")}
# rule (a): the pre-mask plot or lead names a Batman-family character (case-sensitive)
BATFAM = re.compile(r"\b(Batman|Bruce Wayne|Gotham|Batgirl|Batwoman|Nightwing|Harley Quinn|Catwoman|Joker|Two-Face|Riddler|"
                    r"Ra's al Ghul|Damian Wayne|Dick Grayson|Jason Todd|Tim Drake|Alfred Pennyworth|Commissioner Gordon|"
                    r"Barbara Gordon|Arkham Asylum|Batmobile|Batcave|Wayne Enterprises|Teen Titans)\b")
BATFAM_WHITELIST = re.compile(r"Kamen Rider")      # 'Joker' is a Kamen Rider character, not Batman's
# rule (b): every work of a T3-positive relative entity's franchise that is not a documented relative work
# (entity, title regex, first-lead-sentence regex, optional regex the title or lead must also match)
FRANCHISE_RULES = [
    ("Superman", re.compile(r"\bSuperman\b|Man of Steel|Supergirl"), None),
    ("Robin Hood (source of Robin's name)", re.compile(r"Robin Hood"), re.compile(r"Robin Hood")),
    ("Darkman", re.compile(r"Darkman"), None),
    ("Nite Owl (Watchmen)", re.compile(r"Watchmen"), None),
    ("The Phantom (Lee Falk)", None, re.compile(r"Lee Falk")),
    ("Dick Tracy", re.compile(r"Dick Tracy"), re.compile(r"Dick Tracy")),
    ("Iron Man", re.compile(r"Iron Man|Avengers"), None, re.compile(r"Iron Man|Marvel")),
    ("Sherlock Holmes", re.compile(r"Sherlock Holmes"), re.compile(r"Sherlock Holmes")),
    ("The Shadow", re.compile(r"^The Shadow( \(|$| Strikes| Returns)"), re.compile(r"Lamont Cranston")),
    ("The Bat (Rinehart and Hopwood play) / The Bat Whispers", re.compile(r"^The Bat\b|Circular Staircase"), None),
    ("Doc Savage", re.compile(r"Doc Savage"), re.compile(r"Doc Savage")),
    ("Moon Knight", re.compile(r"Moon Knight"), None),
    ("Bluntman", re.compile(r"Jay (and|&) Silent Bob"), None),
    ("The Man Who Laughs (source of the Joker's look)", re.compile(r"The Man Who Laughs"), None),
    ("Blue Falcon", re.compile(r"Blue Falcon"), None),
]
# rule (c): heroes named in a relative's evidence sentence as part of Batman's lineage
EVIDENCE_NAMED = [("Doctor Syn", re.compile(r"Doctor Syn|Dr\.? Syn"), re.compile(r"Doctor Syn|Dr\.? Syn")),
                  ("The Spider (pulp)", re.compile(r"^The Spider('s Web| Returns)"), re.compile(r"Richard Wentworth")),
                  ("Green Hornet", re.compile(r"Green Hornet"), re.compile(r"Green Hornet"))]


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text or "", maxsplit=1)[0]


def genre_stratum(shortdesc, lead, fallback):
    text = f"{shortdesc or ''} {first_sentence(lead)}"
    for s, rx in GENRE_STRATA:
        if rx.search(text):
            return s
    return fallback


def title_stem(t):
    """Franchise cluster key for pool negatives (entity-level T3 null): the title without its trailing parenthetical,
    subtitle, sequel number or leading article."""
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
    t = re.split(r"\s*(?::|–|—| - )\s*", t)[0]
    t = re.sub(r"\s+(\d+|[IVX]{1,4}|Part \w+|Chapter \w+|Returns|Strikes Back|Begins|Reloaded|Revolutions|Forever)$", "", t)
    t = re.sub(r"^(The|A|An|Return of the|Son of|Revenge of the|Bride of the|Curse of the)\s+", "", t, flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def relatives():
    rel = json.load(open(SOURCES / "relatives_documented_relatives.json"))
    coded = {}
    if (AGENT / "relations" / "relation_types_final.jsonl").exists():
        coded = {r["entity"]: r for r in jl(AGENT / "relations" / "relation_types_final.jsonl")}
    out = []
    for r in rel:
        rels = set(r["relation"])
        t3 = bool(rels & T3_RELATIONS) and r["strength"] in ("A", "B")
        works = []
        for w in r.get("works_found", []):
            if "title" not in w:
                continue
            works.append({"title": w["title"], "year": w.get("year"), "n_sentences": w.get("n_sentences"), "in_dev": w.get("in_dev"),
                          "in_t3_pool": w.get("in_t3_pool"),
                          "t3_eligible": bool((w.get("n_sentences") or 0) >= 8 and not w.get("in_dev"))})
        c = coded.get(r["name"], {})
        out.append({"entity": r["name"], "relation": sorted(rels), "strength": r["strength"], "wikidata": r.get("wikidata"),
                    "evidence": r.get("evidence", []), "works": works, "t3_positive": t3,
                    "dev_franchise": r["name"] in DEV_FRANCHISE,
                    "relation_type": c.get("relation_type"), "relation_type_codes": c.get("codes", []),
                    "t3_note": None if t3 else ("not a T3 positive: " + ("strength " + r["strength"] if r["strength"] not in ("A", "B")
                                                                        else "relation " + "/".join(sorted(rels))))})
    doc = {"description": "Documented Batman relatives for T3. Strength: A = creator statement or Wikidata link plus Wikipedia "
                          "text; B = a Wikipedia text claim; C = a comparison only. T3 positives = strength A or B with relation "
                          "influence, predecessor, derivative or sibling. relation_type (coded at step P1 by two blind coders, "
                          "annotation_protocol.md A6): plot_role (shares plot or role-level content), iconography_name (look, "
                          "costume or name only), parody (parody or pastiche), meta (casting or reference), transitive (related "
                          "through another relative). dev_franchise = the entity has a DEV cousin work; the others are "
                          "franchise-fresh (the confirmatory set for S15).",
           "relatives": out}
    dump(doc, "documented_relatives.json")
    return doc


def culture_pool(rel):
    elig = {r["title"]: r for r in jl(SOURCES / "t3_pool_eligible.jsonl.gz")}
    cand = {}
    for r in jl(SOURCES / "t3_candidates.jsonl.gz"):
        if r.get("title") and r["title"] in elig and r["title"] not in cand:
            cand[r["title"]] = r
    rand = [r for r in jl(SOURCES / "t3_random_films.jsonl.gz") if r.get("status") == "accepted"]
    relw = {r["title"]: r for r in jl(SOURCES / "relatives_works.jsonl.gz") if r.get("title")}
    rel_of = defaultdict(set)
    for r in rel["relatives"]:
        if r["t3_positive"]:
            for w in r["works"]:
                if w["t3_eligible"]:
                    rel_of[w["title"]].add(r["entity"])
    for t, r in elig.items():
        b = set(r["buckets"])
        if "family_zorro" in b:
            rel_of[t].add("Zorro")
        if "family_pimpernel" in b:
            rel_of[t].add("The Scarlet Pimpernel")
    rel_of["The Wild World of Batwoman"].add("Batwoman (Jerry Warren)")
    vig_only = sha_order([t for t, r in elig.items() if set(r["buckets"]) == {"vigilante_film"} and not r["batman_lineage"]], SALT_VIG)
    keep_vig = set(vig_only[:N_VIG])

    def book(t, buckets=()):
        b = set(buckets)
        if "family_zorro" in b:
            return 1, "Zorro"
        if "family_pimpernel" in b:
            return 1, "Scarlet Pimpernel"
        if "family_iron_man" in b:
            return 1, "Iron Man"
        for tier, ent, rx in BOOK:
            if rx.search(t):
                return tier, ent
        return None, None

    def contested_reasons(t, sentences, lead):
        text = " ".join(sentences) + " " + (lead or "")
        why = []
        f = sorted({m.group(1) for m in BATFAM.finditer(text)})
        if f and not BATFAM_WHITELIST.search(t):
            why.append("names Batman-family character: " + ", ".join(f))
        for rule in FRANCHISE_RULES:
            ent, trx, lrx = rule[:3]
            need = rule[3] if len(rule) > 3 else None
            if ((trx and trx.search(t)) or (lrx and lrx.search(first_sentence(lead)))) and (need is None or need.search(t + " " + (lead or ""))):
                why.append(f"franchise of relative entity {ent}")
        for ent, trx, lrx in EVIDENCE_NAMED:
            if (trx and trx.search(t)) or (lrx and lrx.search(first_sentence(lead))):
                why.append(f"named in a relative's evidence sentence: {ent}")
        return why

    works = []

    def add(t, source, kind, year, buckets, sents, lead, shortdesc, cat_stratum, extra):
        tier, ent_b = book(t, buckets)
        if t in LINEAGE_RELABEL:
            role, reason = LINEAGE_RELABEL[t]
            why = [reason] if role == "contested" else []
        elif extra.get("batman_lineage"):
            role, why = "batman_lineage", []
        elif t in rel_of:
            role, why = "relative", []
        elif tier:
            role, why = "book_family", []
        else:
            why = contested_reasons(t, sents, lead)
            role = "contested" if why else "pool"
        stratum = cat_stratum if source == "random_films" else genre_stratum(shortdesc, lead, cat_stratum)
        works.append({"title": t, "source": source, "kind": kind, "year": year, "buckets": sorted(buckets),
                      "stratum": stratum, "stratum_category": cat_stratum, "role": role, "contested_reasons": why,
                      "entity": sorted(rel_of[t])[0] if t in rel_of else ent_b, "relative_entities": sorted(rel_of[t]),
                      "book_tier": tier, "book_entity": ent_b, "neg_cluster": title_stem(t),
                      "mike_candidate": bool(re.search(r"Monte Cristo", t)), **{k: v for k, v in extra.items() if k != "batman_lineage"},
                      "n_sentences": len(sents), "sha256": h(sents)})

    for t, r in sorted(elig.items()):
        b = set(r["buckets"])
        if (b == {"vigilante_film"} and not r["batman_lineage"] and t not in keep_vig and t not in rel_of and book(t, b)[0] is None):
            continue
        c = cand[t]
        s = as_list(c["sentences"])[:CAP]
        cat = next((st for st, bs in CATEGORY_STRATA if b & bs), "pulp_crime")
        add(t, "category_pool", r["kind"], r.get("year"), b, s, c.get("lead"), c.get("shortdesc"), cat,
            {"batman_lineage": r["batman_lineage"], "wikibase_item": r.get("wikibase_item")})
    have = {w["title"] for w in works}
    for r in rand:
        t = r["title"]
        if t in have:
            continue
        s = as_list(r["sentences"])[:CAP]
        y = int(r["year"]) if str(r.get("year", "")).isdigit() else r.get("year")
        add(t, "random_films", "film", y, {"random_film"}, s, r.get("lead"), r.get("shortdesc"), "random",
            {"random_rank": int(r["rank"])})
        have.add(t)
    for t, r in sorted(relw.items()):
        if t in have or r.get("in_dev") or (r.get("n_sentences_capped") or 0) < 8 or t not in rel_of:
            continue
        s = as_list(r["sentences"])[:CAP]
        add(t, "relatives", r.get("medium"), r.get("year"), {"relative_outside_pool"}, s, r.get("lead"), r.get("shortdesc"),
            OUT_OF_POOL_CATEGORY.get(t, "random"), {})
        have.add(t)
    # T2-N extra works: 40 pool negatives from each of the random, superhero and pulp_crime strata, in seeded order
    for st in ("random", "superhero", "pulp_crime"):
        ids = sha_order([w["title"] for w in works if w["role"] == "pool" and w["stratum"] == st], SALT_T2N)[:N_T2N_PER_STRATUM]
        for w in works:
            if w["title"] in ids:
                w["t2n_extra"] = True
    for w in works:
        w["t2n"] = bool(w["role"] in ("batman_lineage", "relative", "book_family") or w.get("t2n_extra"))
    cl = Counter(w["neg_cluster"] for w in works if w["role"] == "pool")
    for w in works:
        w["neg_cluster_size"] = cl[w["neg_cluster"]] if w["role"] == "pool" else None
    roles = Counter(w["role"] for w in works)
    strata = Counter((w["stratum"], w["role"]) for w in works)
    doc = {"description": "T3 culture pool. Category pool (21 Wikipedia category roots, depth 3; eligible = film, TV film, serial, "
                          "novel or short story with >= 8 plot sentences, not in DEV) minus the vigilante-only works except a "
                          f"{N_VIG}-work hash subsample (sha256('{SALT_VIG}' + title)), plus 1,000 random films (sha256 order "
                          "over 169,476 'Category:YYYY films' members, 1915-2024) and the documented relatives outside the pool. "
                          "Roles: batman_lineage (positive control and T4), relative (T3 positives), book_family (Marketing "
                          "Memetics tiers that are not documented relatives), contested (excluded from the negatives and from "
                          "the superhero centroid, scored and reported separately: (a) the pre-mask plot or lead names a "
                          "Batman-family character, (b) a work of a T3-positive relative entity's franchise that is not a "
                          "documented relative work, (c) a hero named in a relative's evidence sentence, or a relabelled lineage "
                          "title), pool (negatives). stratum = genre rule on the work's own short description and first lead "
                          "sentence (random films stay 'random'); stratum_category = the category-crawl stratum (sensitivity). "
                          "neg_cluster = title-stem franchise cluster (entity-level null). mike_candidate = a Count of Monte "
                          "Cristo work (named by the author, not documented; reported descriptively). t2n = annotated for T2-N "
                          f"(lineage, relatives, book family, and {N_T2N_PER_STRATUM} seeded pool works from each of the random, "
                          "superhero and pulp_crime strata).",
           "counts": {"works": len(works), "sentences": int(sum(w["n_sentences"] for w in works)), "roles": dict(roles),
                      "strata_by_role": {f"{a}|{b}": c for (a, b), c in sorted(strata.items())},
                      "contested_by_rule": dict(Counter(r.split(":")[0].split(" entity")[0] for w in works for r in w["contested_reasons"])),
                      "book_tier1": int(sum(1 for w in works if w["book_tier"] == 1)),
                      "book_tier2": int(sum(1 for w in works if w["book_tier"] == 2)),
                      "t2n_works": int(sum(1 for w in works if w["t2n"])),
                      "pool_clusters_size_ge2": int(sum(1 for c in cl.values() if c >= 2))},
           "works": works}
    dump(doc, "culture_pool.json")
    return doc


# ----------------------------------------------------------------------------------------------- extended Batman lineage
FRESH = {"Batman: Year One (film)": ["Year One"], "Batman: The Dark Knight Returns (film)": ["Dark Knight Returns"],
         "Batman: The Killing Joke (film)": ["The Killing Joke"], "Batman: Hush (film)": ["Hush"],
         "Batman: Death in the Family": ["A Death in the Family"], "Batman: The Long Halloween (film)": ["The Long Halloween"],
         "Batman: Knightfall (film)": ["Knightfall"]}


def batman_extended():
    elig = {r["title"]: r for r in jl(SOURCES / "t3_pool_eligible.jsonl.gz")}
    cand = {r["title"]: r for r in jl(SOURCES / "t3_candidates.jsonl.gz") if r.get("title")}
    rows = []
    for t, r in sorted(elig.items()):
        if not r["batman_lineage"] or t in LINEAGE_RELABEL:
            continue
        c = cand[t]
        s = as_list(c["sentences"])[:CAP]
        lead = c.get("lead") or ""
        ev = next((x.strip() for x in re.split(r"(?<=[.!?])\s+", lead) if re.search(r"based on|adapt|explores alternate outcomes", x, re.I)), None)
        rows.append({"id": t, "title": t, "year": int(r["year"]) if str(r.get("year", "")).isdigit() else r.get("year"),
                     "kind": r.get("kind"), "n_sentences": len(s), "sha256": h(s),
                     "fresh_adaptation_of": FRESH.get(t), "lead_evidence": ev if t in FRESH else None})
    dev = [{"id": w["label"], "title": w["title"], "year": w["year"], "medium": w["medium"]} for w in DEV if w["group"] == "batman"]
    doc = {"description": "Extended Batman lineage. DEV: the 32 DEV Batman works. TEST: the Batman-lineage works in the T3 pool "
                          "(not in DEV, >= 8 plot sentences) minus four relabelled titles (The Wild World of Batwoman becomes a "
                          "relative; Batman XXX, The People's Joker and Catwoman (2004) are contested). Seven TEST works are "
                          "animated adaptations of DEV comics documented in their article's lead (T1e and extra H4 targets). T4 "
                          "uses all DEV and TEST works in date order, all re-run through the v2 name-removal protocol in one "
                          "pass. No TEST work is earlier than 1995.",
           "relabelled": {k: v[0] for k, v in LINEAGE_RELABEL.items()}, "dev": dev, "test": rows, "fresh_adaptations": FRESH,
           "test_min_year": min(r["year"] for r in rows if isinstance(r["year"], int))}
    dump(doc, "works_batman_extended.json")
    return doc


# ----------------------------------------------------------------------------------------------- DEV units
def dev_units(memes):
    labels = json.load(open(ROOT / "results" / "labels.json"))
    pm = {p["unit"]: p for p in json.load(open(SOURCES / "robust_agreement_analysis.json"))["per_meme"]}
    um = [int(u) for u in np.load(SOURCES / "robust_agreement_units_m.npz")["units_m"]]
    z = np.load(SOURCES / "robust_agreement_lexical_null_units.npz")
    wording = json.load(open(ROOT / "results" / "wording.json"))
    surv = {int(m["unit"]): bool(m["qwen"]["reworded"] > m["qwen"]["neutral_max"]) for m in wording["memes"]}
    rev = wording["revenge"]
    mapped = {}
    for m in memes:
        u = m.get("dev_unit_qwen")
        if u is None:
            continue
        if m["id"] == "revenge_vow":
            mapped[m["id"]] = {"qwen3-8b": rev["qwen3-8b"]["chosen"]["unit"], "llama3.1-8b": rev["llama3.1-8b"]["chosen"]["unit"],
                               "dev_class": "concept", "dev_rewording_survivor": False,
                               "source": "DEV revenge-unit selection (results/wording.json): each model's unit was chosen from "
                                         "50 candidates for winning DEV minimal pairs; Qwen's failed its DEV test (4 of 12)"}
            continue
        tw = pm.get(u)
        mapped[m["id"]] = {"qwen3-8b": int(u), "llama3.1-8b": int(tw["llama_twin"]) if tw else None,
                           "dev_label": labels.get(str(u), {}).get("label") or (tw or {}).get("label"),
                           "dev_class": (tw or {}).get("class_labels"), "dev_rewording_survivor": surv.get(int(u)),
                           "source": "DEV labelled Qwen unit; Llama twin = best-correlated Llama unit on DEV original texts"}
    concepts = []
    for p in json.load(open(SOURCES / "robust_agreement_analysis.json"))["per_meme"]:
        if p["class_labels"] != "concept":
            continue
        i = um.index(p["unit"])
        concepts.append({"qwen_unit": int(p["unit"]), "label": p["label"], "dev_rank": p["rank"],
                         "null_units": [int(x) for x in z["units_sel"][3 * i:3 * i + 3]],
                         "dev_twin_original": int(p["llama_twin"]), "dev_r_original": p["r_orig_m"], "dev_r_namefree_fresh": p["r_anon_fresh_m"]})
    rng = np.random.default_rng(SALT_DESIGN)
    out_lt, out_rand = {}, {}
    for model, n_units in (("qwen3-8b", 884736), ("llama3.1-8b", 917504)):
        fam = json.load(open(ROOT / "results" / model / "memes.json"))["families"]
        # best-of arm (T6): DEV meme families ranked by enrichment in Batman works over the reference films, excluding
        # the near-universal canon (present in more than 28 of 32 Batman works), strength as tie-break
        sc = [(f["n_batman"] - 32 * f["n_baseline"] / 69, f["strength"], int(f["unit"])) for f in fam if f["n_batman"] <= 28]
        out_lt[model] = [u for _, _, u in sorted(sc, key=lambda x: (-x[0], -x[1]))[:200]]
        out_rand[model] = sorted(int(x) for x in rng.choice(n_units, 400, replace=False))
    doc = {"description": "DEV units frozen for TEST. t2_mapped: DEV units for T2 memes that match a DEV-labelled meme (H3 uses "
                          "the dev_class 'concept' ones); unit index u < N is the positive pole of SwiGLU channel u (layer = u // "
                          "hidden), u >= N the negative pole of channel u - N (N = 442,368 for Qwen3-8B, 458,752 for Llama 3.1 "
                          "8B). dev_rewording_survivor: Qwen's DEV rewording passed the raw criterion (results/wording.json). "
                          "t5_concepts: the 27 DEV concept memes with their 3 lexically matched null units each (robustness "
                          "review, agreement s9). label_test_best_of: the 200 DEV meme families ranked by Batman enrichment "
                          "(T6 best-of arm); label_test_random: 400 random units per model (T6 random arm). Both lists go into "
                          "the per-sentence unit list at F1; T6 takes, in order, the first 40 present in 1-20% of T3 pool works.",
           "t2_mapped": mapped, "t5_concepts": concepts, "label_test_best_of": out_lt, "label_test_random": out_rand}
    dump(doc, "dev_units.json")
    return doc


# ----------------------------------------------------------------------------------------------- planted-story design
GENRES = ["corporate drama", "epic fantasy", "hard science fiction", "space opera", "contemporary romance", "western",
          "historical drama in Tang-dynasty China", "family saga on a farm", "cozy village mystery", "sports drama",
          "hospital drama", "seafaring adventure", "political thriller", "gothic horror", "school coming-of-age story",
          "post-apocalyptic survival", "Roman-era historical drama"]
WRITERS = ["W1", "W2", "W3", "W4"]
PLOTS_PER_MEME = 12
PER_STORY = 3
VERSIONS = ["explicit", "banned", "implicit", "coded", "control", "decoy"]


def planted_design(memes):
    ids = [m["id"] for m in memes]
    n_plots = len(ids) * PLOTS_PER_MEME // PER_STORY
    related = {frozenset((m["id"], r)) for m in memes for r in m.get("do_not_combine_with", [])}
    rng = np.random.default_rng(SALT_DESIGN)
    best = None
    for attempt in range(20000):
        bag = [m for m in ids for _ in range(PLOTS_PER_MEME)]
        rng.shuffle(bag)
        plots, ok, pair, pool = [], True, Counter(), list(bag)
        for p in range(n_plots):
            chosen = []
            for m in list(pool):
                if m in chosen or any(frozenset((m, c)) in related for c in chosen) or any(pair[frozenset((m, c))] >= 2 for c in chosen):
                    continue
                chosen.append(m)
                if len(chosen) == PER_STORY:
                    break
            if len(chosen) < PER_STORY:
                ok = False
                break
            for m in chosen:
                pool.remove(m)
            for a, b in itertools.combinations(chosen, 2):
                pair[frozenset((a, b))] += 1
            plots.append(sorted(chosen))
        if ok:
            best = plots
            break
    assert best is not None, "no valid design found"

    def spread(labels, per, limit, iters=400000):
        perm = list(rng.permutation([l for l in labels for _ in range(per)]))
        c = Counter((m, perm[p]) for p in range(n_plots) for m in best[p])
        def excess(cnt):
            return sum(max(0, v - limit) for v in cnt.values())
        e = excess(c)
        for _ in range(iters):
            if e == 0:
                return perm
            a, b = rng.integers(0, n_plots, 2)
            if perm[a] == perm[b]:
                continue
            for m in best[a]:
                c[(m, perm[a])] -= 1; c[(m, perm[b])] += 1
            for m in best[b]:
                c[(m, perm[b])] -= 1; c[(m, perm[a])] += 1
            e2 = excess(c)
            if e2 <= e:
                perm[a], perm[b] = perm[b], perm[a]
                e = e2
            else:
                for m in best[a]:
                    c[(m, perm[a])] += 1; c[(m, perm[b])] -= 1
                for m in best[b]:
                    c[(m, perm[b])] += 1; c[(m, perm[a])] -= 1
        raise AssertionError("no balanced assignment")
    assert n_plots % len(GENRES) == 0 and n_plots % len(WRITERS) == 0
    gl = spread(GENRES, n_plots // len(GENRES), 1)
    wl = spread(WRITERS, n_plots // len(WRITERS), 4)
    rows = [{"plot": f"P{p + 1:03d}", "genre": gl[p], "writer": wl[p], "memes": best[p]} for p in range(n_plots)]
    c = Counter(m for r in rows for m in r["memes"])
    assert set(c.values()) == {PLOTS_PER_MEME}
    by_genre = Counter((m, r["genre"]) for r in rows for m in r["memes"])
    by_writer = Counter((m, r["writer"]) for r in rows for m in r["memes"])
    doc = {"description": f"T2 planted-meme design: {n_plots} base plots, each planting 3 of the {len(ids)} memes; every meme in "
                          f"exactly {PLOTS_PER_MEME} plots; no pair of memes in more than 2 plots; memes marked "
                          "do_not_combine_with never share a plot. 17 genres unrelated to Batman (8 plots each) and 4 writers "
                          f"(W1-W4, independent agent sessions, {n_plots // 4} plots each), assigned so that no meme meets a genre "
                          "more than once or a writer in more than 4 plots. Each plot is written in 6 versions "
                          "(planted_story_protocol.md): explicit, banned, implicit, coded, control, decoy.",
           "seed": SALT_DESIGN, "genres": GENRES, "writers": WRITERS, "versions": VERSIONS,
           "max_plots_per_meme_and_genre": max(by_genre.values()), "max_plots_per_meme_and_writer": max(by_writer.values()),
           "pair_cooccurrence_max": max(Counter(frozenset(p) for r in rows for p in itertools.combinations(r["memes"], 2)).values()),
           "plots": rows}
    dump(doc, "planted_design.json")
    return doc


# ----------------------------------------------------------------------------------------------- data assembly (after F0)
def _write(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = json.load(open(DATA / "manifest.json")) if (DATA / "manifest.json").exists() else {}
    man[name] = {"n_works": len(rows), "n_sentences": sum(len(r["sentences"]) for r in rows),
                 "sha256": hashlib.sha256(open(DATA / f"{name}.jsonl", "rb").read()).hexdigest()}
    json.dump(man, open(DATA / "manifest.json", "w"), indent=1)
    print("wrote", name, len(rows))


def _check(rows, frozen):
    for r in rows:
        f = frozen.get(r["id"])
        assert f is None or f == h(r["sentences"]), f"text of {r['id']} differs from the frozen hash"


def _A():
    sys.path.insert(0, str(PRE))
    import analysis as A
    return A


CTX_FILM_SALT = "phylomemetics-ctx-paragraph-v2|"


def stage_dev():
    """DEV-derived sets: dev_original, dev_namefree, t2_background (each reference-film sentence as its own work),
    t2_reference (the 69 reference films, original texts), t3_dev_masked (DEV corpus, relation statements stripped,
    masked, franchise-scanned), t2_examples (one work per example sentence of every set, and the definition), and
    t2_examples_ctx (each positive example after a fixed 3-sentence DEV paragraph, recorded alone: the m1ctx variant)."""
    A = _A()
    neut = jl(ROOT / "data" / "corpus_neutral.jsonl")
    keep = ("title", "group", "medium", "year", "era")
    _write("dev_original", [{"id": w["label"], **{k: w.get(k) for k in keep}, "sentences": w["sentences"]} for w in DEV])
    _write("dev_namefree", [{"id": w["label"], **{k: w.get(k) for k in keep}, "sentences": w["sentences"]} for w in neut])
    _write("t2_background", [{"id": f"{w['label']}#{k}", "film": w["label"], "group": "baseline", "sentences": [s]}
                             for w in DEV if w["group"] == "baseline" for k, s in enumerate(w["sentences"])])
    _write("t2_reference", [{"id": w["label"], **{k: w.get(k) for k in keep}, "sentences": w["sentences"]} for w in DEV if w["group"] == "baseline"])
    _write("t3_dev_masked", [{"id": w["label"], **{k: w.get(k) for k in keep}, "sentences": A.t3_text(w["sentences"])[0]} for w in DEV])
    memes = json.load(open(PRE / "memes_for_tracing.json"))["memes"]
    rows = []
    for m in memes:
        rows.append({"id": f"{m['id']}#definition", "meme": m["id"], "set": "definition", "sentences": [m["definition"]]})
        for st in EXAMPLE_SETS:
            for k, e in enumerate(m["examples"][st]):
                rows.append({"id": f"{m['id']}#{st}{k}", "meme": m["id"], "set": st, "sentences": [e]})
    assert all(len(m["examples"][st]) for m in memes for st in EXAMPLE_SETS), "example sets missing: run step P1 first"
    _write("t2_examples", rows)
    films = sha_order([w["label"] for w in DEV if w["group"] == "baseline"], CTX_FILM_SALT)
    ctx = next(w for w in DEV if w["label"] == films[0])["sentences"][:3]
    _write("t2_examples_ctx", [{**r, "context_sentences": ctx} for r in rows])


def stage_examples():
    """Step P1 (before F0): screen the agent-written example sets (annotation_protocol.md A8) and write
    agent/examples/examples_final.jsonl, which 'build_lists.py memes' merges into memes_for_tracing.json.
    explicit: every sentence has >= 1 ban-list term; reworded: none (derivational rule); hard_negative: >= 2 and a
    blind reader found the meme absent; positives: a blind reader found the meme present; far: no forbidden setting
    word (analysis.FAR_FORBIDDEN) or franchise name, then the TF-IDF screen against DEV Batman and cousin sentences
    (analysis.far_screen_tfidf); 5 kept per set (the first 5 that pass, in the order written; a set with fewer than 3
    after the screen goes back to a fresh writer once)."""
    A = _A()
    memes = {m["id"]: m for m in json.load(open(PRE / "memes_for_tracing.json"))["memes"]}
    raw = jl(AGENT / "examples" / "raw.jsonl")
    reads = {(r["meme"], r["sentence"]): r["contains"] for r in jl(AGENT / "examples" / "reads.jsonl")} if (AGENT / "examples" / "reads.jsonl").exists() else {}
    bat = [s for w in DEV if w["group"] in ("batman", "cousin") for s in w["sentences"]]
    ref = [s for w in DEV if w["group"] == "baseline" for s in w["sentences"]]
    out, report = [], {}
    for r in raw:
        m = memes[r["meme"]]
        ok = []
        for s in r["sentences"]:
            hits = A.ban_hits(s, m["keywords"])
            rd = reads.get((r["meme"], s))
            if r["set"] == "explicit" and hits >= 1 and rd is not False:
                ok.append(s)
            elif r["set"] == "reworded" and hits == 0 and rd is not False:
                ok.append(s)
            elif r["set"] == "hard_negative" and hits >= 2 and rd is not True:
                ok.append(s)
            elif r["set"] == "far" and hits == 0 and not A.FAR_FORBIDDEN.search(s) and not A.FRANCHISE.search(s) and rd is not False:
                ok.append(s)
        if r["set"] == "far" and ok:
            keep, ex = A.far_screen_tfidf(ok, bat, ref)
            report[f"{r['meme']}|far_tfidf_excess"] = ex
            ok = [ok[i] for i in keep]
        report[f"{r['meme']}|{r['set']}"] = {"written": len(r["sentences"]), "kept": len(ok[:K_EXAMPLES])}
        out.append({"meme": r["meme"], "set": r["set"], "sentences": ok[:K_EXAMPLES]})
    (AGENT / "examples").mkdir(parents=True, exist_ok=True)
    with open(AGENT / "examples" / "examples_final.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(report, open(AGENT / "examples" / "screen_report.json", "w"), indent=1)
    print("wrote", AGENT / "examples" / "examples_final.jsonl")


def stage_wikipedia():
    """Wikipedia TEST texts, checked against the frozen hashes: t1d_plays_original, t1d_films_original, t3_pool (relation
    statements stripped, masked, franchise-scanned), t2n_works (the t2n works of t3_pool), t4_new_original."""
    A = _A()
    sh = json.load(open(PRE / "works_shakespeare.json"))
    plays = {p["title"]: p for p in jl(SOURCES / "plays.jsonl.gz")}
    pr = [{"id": p["id"], "title": p["title"], "sentences": play_sentences(plays[p["title"]]["sentences"])[0]} for p in sh["plays"]]
    _check(pr, {p["id"]: p["sha256"] for p in sh["plays"]})
    _write("t1d_plays_original", pr)
    ft = {r["title"]: r for r in jl(SOURCES / "films_all.jsonl.gz") if r.get("title")}
    keys = ("primary", "renamed", "h5", "drift_sample", "evidence_tier", "sources", "names_flag", "remake_of_dev")
    fr = [{"id": f["id"], **{k: f[k] for k in keys}, "sentences": as_list(ft[f["id"]]["sentences"])[:CAP]} for f in sh["films"]]
    _check(fr, {f["id"]: f["sha256"] for f in sh["films"]})
    _write("t1d_films_original", fr)
    pool = json.load(open(PRE / "culture_pool.json"))["works"]
    cand = {r["title"]: r for r in jl(SOURCES / "t3_candidates.jsonl.gz") if r.get("title")}
    rand = {r["title"]: r for r in jl(SOURCES / "t3_random_films.jsonl.gz") if r.get("status") == "accepted"}
    relw = {r["title"]: r for r in jl(SOURCES / "relatives_works.jsonl.gz") if r.get("title")}
    rows, audit = [], {}
    for w in pool:
        srcrow = {"category_pool": cand, "random_films": rand, "relatives": relw}[w["source"]][w["title"]]
        s = as_list(srcrow["sentences"])[:CAP]
        assert h(s) == w["sha256"], w["title"]
        masked, info = A.t3_text(s)
        audit[w["title"]] = info
        rows.append({"id": w["title"], "role": w["role"], "stratum": w["stratum"], "stratum_category": w["stratum_category"],
                     "t2n": w["t2n"], "sentences": masked})
    _write("t3_pool", rows)
    _write("t2n_works", [r for r in rows if r["t2n"]])
    json.dump(audit, open(DATA / "t3_text_audit.json", "w"), indent=0)
    ext = json.load(open(PRE / "works_batman_extended.json"))
    _write("t4_new_original", [{"id": w["id"], "year": w["year"], "group": "batman", "medium": "film",
                                "sentences": as_list(cand[w["id"]]["sentences"])[:CAP]} for w in ext["test"]])
    _write("t4_dev_original", [{"id": w["label"], "year": w["year"], "group": "batman", "medium": w["medium"], "sentences": w["sentences"]}
                               for w in DEV if w["group"] == "batman"])


def stage_names():
    """Name-free sets from the agent rewrites (name_removal_protocol.md) via analysis.neutralise: t1d_plays_namefree,
    t1d_films_namefree, t1e_fresh_namefree, and t4_batman (all DEV and TEST Batman works re-run in one v2 pass)."""
    A = _A()
    def apply(src_set, rewrite_file):
        rw = {r["id"]: r["rewrite"] for r in jl(AGENT / "name_removal" / rewrite_file)}
        out = []
        for w in jl(DATA / f"{src_set}.jsonl"):
            assert len(rw[w["id"]]) == len(w["sentences"]), w["id"]
            s, n = A.neutralise(w["sentences"], rw[w["id"]])
            out.append({**w, "sentences": s, "n_codes_replaced": n})
        return out
    _write("t1d_plays_namefree", apply("t1d_plays_original", "plays.jsonl"))
    _write("t1d_films_namefree", apply("t1d_films_original", "films.jsonl"))
    new = apply("t4_new_original", "batman_new.jsonl")
    fresh = set(json.load(open(PRE / "works_batman_extended.json"))["fresh_adaptations"])
    _write("t1e_fresh_namefree", [w for w in new if w["id"] in fresh])
    dev_v2 = apply("t4_dev_original", "batman_dev_v2.jsonl")
    _write("t4_batman", [dict(w, origin="dev") for w in dev_v2] + [dict(w, origin="test") for w in new])


def stage_drift():
    """t1b_drift<L> (20 DEV screen works), t1b_fresh_drift<L> (7 fresh works) and t1b_shk_drift<L> (the drift_sample
    Shakespeare films) for L in 25, 50, 75, 100 and the style-only control 'style', from drift_protocol.md outputs.
    Rewrites still failing after the repair round are kept and flagged."""
    meta = {w["id"]: w for w in jl(DATA / "dev_namefree.jsonl")}
    fresh = {w["id"]: w for w in jl(DATA / "t1e_fresh_namefree.jsonl")}
    films = [w for w in jl(DATA / "t1d_films_namefree.jsonl") if w["drift_sample"]]
    for L in ("25", "50", "75", "100", "style"):
        rows = {r["id"]: r for r in jl(AGENT / "drift" / f"level{L}.jsonl")}
        suffix = L if L != "style" else "_style"
        _write(f"t1b_drift{suffix}", [{**{k: meta[i].get(k) for k in ("group", "medium", "year", "title")}, "id": i,
                                       "sentences": rows[i]["sentences"], "drift_flags": rows[i].get("flags", [])}
                                      for i in meta if meta[i]["group"] == "batman" and meta[i]["medium"] != "comic"])
        _write(f"t1b_fresh_drift{suffix}", [{**{k: fresh[i].get(k) for k in ("group", "medium", "year")}, "id": i,
                                             "sentences": rows[i]["sentences"], "drift_flags": rows[i].get("flags", [])} for i in fresh])
        _write(f"t1b_shk_drift{suffix}", [{**{k: w[k] for k in ("primary", "renamed", "h5", "drift_sample", "sources", "names_flag")},
                                           "id": w["id"], "sentences": rows[w["id"]]["sentences"],
                                           "drift_flags": rows[w["id"]].get("flags", [])} for w in films])


def stage_zh():
    """t1c_zh (20 DEV screen works + 69 reference films), t1c_fresh_zh and t2_stories_zh (banned and control versions)
    from translation_protocol.md outputs (sentence-aligned)."""
    tr = {r["id"]: r for r in jl(AGENT / "translation" / "zh.jsonl")}
    dev = [w for w in jl(DATA / "dev_namefree.jsonl") if (w["group"] == "batman" and w["medium"] != "comic") or w["group"] == "baseline"]
    rows = []
    for w in dev:
        assert len(tr[w["id"]]["sentences"]) == len(w["sentences"]), w["id"]
        rows.append({**{k: w.get(k) for k in ("group", "medium", "year")}, "id": w["id"], "sentences": tr[w["id"]]["sentences"],
                     "translation_flags": tr[w["id"]].get("flags", [])})
    _write("t1c_zh", rows)
    _write("t1c_fresh_zh", [{**w, "sentences": tr[w["id"]]["sentences"], "translation_flags": tr[w["id"]].get("flags", [])}
                            for w in jl(DATA / "t1e_fresh_namefree.jsonl")])
    st = jl(DATA / "t2_stories.jsonl")
    _write("t2_stories_zh", [{**w, "sentences": tr[w["id"]]["sentences"], "translation_flags": tr[w["id"]].get("flags", [])}
                             for w in st if w["version"] in ("banned", "control")])


def stage_zhwiki():
    """T1c natural arm: the zh.wikipedia plot sections of DEV Batman screen works, fetched by translation_protocol.md
    step N1 (agent/zhwiki/plots.jsonl, fetched on one date and committed with F2-T1c), name-removed in Chinese
    (agent/zhwiki/rewrites.jsonl) through analysis.neutralise_cjk."""
    A = _A()
    raw = {r["id"]: r for r in jl(AGENT / "zhwiki" / "plots.jsonl")}
    rw = {r["id"]: r["rewrite"] for r in jl(AGENT / "zhwiki" / "rewrites.jsonl")}
    meta = {w["id"]: w for w in jl(DATA / "dev_namefree.jsonl")}
    rows = []
    for i, r in raw.items():
        if len(r["sentences"]) < 8 or i not in rw:
            continue
        s, n = A.neutralise_cjk(r["sentences"], rw[i])
        rows.append({**{k: meta[i].get(k) for k in ("group", "medium", "year")}, "id": i, "sentences": s, "n_codes_replaced": n,
                     "zh_title": r.get("zh_title"), "fetched": r.get("fetched")})
    _write("t1c_zhwiki", rows)


def stage_stories():
    """t2_stories from planted_story_protocol.md outputs and checks Q1-Q6. valid = script checks Q1-Q3 pass;
    exclude_memes = memes for which the story is left out (failed Q4 or Q5); dropped_memes = memes failing Q4 at the
    explicit or banned level in more than 3 of their 12 plots after repair."""
    A = _A()
    memes = {m["id"]: m for m in json.load(open(PRE / "memes_for_tracing.json"))["memes"]}
    design = {p["plot"]: p for p in json.load(open(PRE / "planted_design.json"))["plots"]}
    stories = jl(AGENT / "stories" / "stories.jsonl")
    reads = defaultdict(list)
    for r in jl(AGENT / "stories" / "blind_reads.jsonl"):          # Claude readers and the cross-family reader
        reads[(r["plot"], r["version"])].append((r.get("family", "claude"), set(r["memes_found"])))
    sal = defaultdict(list)
    for r in jl(AGENT / "stories" / "salience.jsonl") if (AGENT / "stories" / "salience.jsonl").exists() else []:
        sal[(r["plot"], r["version"], r["slot"])].append(r["rating"])
    by_plot = defaultdict(list)
    for s in stories:
        by_plot[s["plot"]].append(s)
    rows = []
    fail_q4 = Counter()
    for s in stories:
        p = design[s["plot"]]
        q = A.story_checks(s, p, memes, [o for o in by_plot[s["plot"]] if o["version"] != s["version"]])
        found = reads[(s["plot"], s["version"])]
        claude = [f for fam, f in found if fam == "claude"]
        excl = A.q4_q5_exclusions(s["version"], p["memes"], list(memes), claude, [f for fam, f in found if fam != "claude"])
        for m in excl & set(p["memes"]):
            if s["version"] in ("explicit", "banned"):
                fail_q4[m] += 1
        rows.append({"id": f"{s['plot']}-{s['version']}", "plot": s["plot"], "version": s["version"], "memes": p["memes"],
                     "genre": p["genre"], "writer": p["writer"], "valid": q["valid"], "checks": q, "exclude_memes": sorted(excl),
                     "slots": s.get("slots"), "salience": {m: sal.get((s["plot"], s["version"], m)) for m in p["memes"]},
                     "reads_claude": [sorted(f) for f in claude], "reads_other": [sorted(f) for fam, f in found if fam != "claude"],
                     "sentences": s["sentences"]})
    dropped = sorted(m for m, c in fail_q4.items() if c > 3)
    for r in rows:
        r["dropped_memes"] = dropped
    _write("t2_stories", rows)


def stage_long():
    """T2-long: 48 long texts, each the concatenation of 12 valid T2 stories from 12 different plots: 2 reworded versions
    (banned or implicit, chosen by seed) and 10 control versions. Memes present = the planted memes of the 2 reworded
    stories. Deterministic (seed 20261020)."""
    st = [w for w in jl(DATA / "t2_stories.jsonl") if w["valid"]]
    by = defaultdict(dict)
    for w in st:
        by[w["plot"]][w["version"]] = w
    plots = sorted(p for p, v in by.items() if {"banned", "implicit", "control"} <= set(v))
    rng = np.random.default_rng(SALT_DESIGN)
    rows = []
    for k in range(48):
        pick = [str(x) for x in rng.choice(plots, 12, replace=False)]
        chosen, present, unclear = [], set(), set()
        for j, p in enumerate(pick):
            ver = ("banned", "implicit")[int(rng.integers(0, 2))] if j < 2 else "control"
            w = by[p][ver]
            chosen.append(w)
            if j < 2:
                present |= set(w["memes"]) - set(w["exclude_memes"])
            unclear |= set(w["exclude_memes"])
        seq = [chosen[i] for i in rng.permutation(12)]
        rows.append({"id": f"L{k + 1:02d}", "stories": [w["id"] for w in seq], "sentences": [s for w in seq for s in w["sentences"]],
                     "memes_present": sorted(present), "exclude_memes": sorted(unclear - present)})
    _write("t2_long", rows)


def stage_t2n():
    """Adds the exhaustive annotations (annotation_protocol.md A2) to t2n_works."""
    ann = {r["id"]: r for r in jl(AGENT / "annotation" / "t2n_adjudicated.jsonl")}
    raw = defaultdict(list)
    for r in jl(AGENT / "annotation" / "t2n_raw.jsonl"):
        raw[r["id"]].append(sorted({m for ids in r["sentences"] for m in ids}))
    rows = [{**w, "memes_present": sorted(ann[w["id"]]["memes_present"]), "sentence_labels": ann[w["id"]]["sentence_labels"],
             "evidence_spans": ann[w["id"]].get("evidence_spans"), "annotators": raw[w["id"]][:2]} for w in jl(DATA / "t2n_works.jsonl")]
    _write("t2n_works", rows)


def stage_codes():
    """Code-collapsed copies (every X-code replaced by the one token 'X0') for the X-code sensitivity of H5 and T1c."""
    A = _A()
    for name in ("t1d_plays_namefree", "t1d_films_namefree", "t1c_zh", "t1c_fresh_zh", "dev_namefree", "t1e_fresh_namefree"):
        _write(f"{name}_cc", [{**w, "sentences": A.collapse_codes(w["sentences"])} for w in jl(DATA / f"{name}.jsonl")])


def stage_pilot():
    """Pilot material (step P2, excluded from TEST and never scored by a method): 6 pilot plots planting memes in genres
    outside the design, 2 DEV reference films for drift at levels 50 and 100, and 3 DEV reference films for translation.
    Writes prereg/agent/pilot/pilot_inputs.json."""
    memes = json.load(open(PRE / "memes_for_tracing.json"))["memes"]
    rng = np.random.default_rng(SALT_DESIGN + 1)
    ids = [m["id"] for m in memes]
    plots = [{"plot": f"PILOT{k + 1}", "genre": g, "writer": WRITERS[k % 4], "memes": sorted(rng.choice(ids, 3, replace=False).tolist())}
             for k, g in enumerate(["detective procedural", "nautical comedy", "court intrigue", "mountaineering drama",
                                    "boarding-house farce", "desert trek"])]
    films = sha_order([w["label"] for w in DEV if w["group"] == "baseline"], "phylomemetics-pilot-v2|")
    out = {"plots": plots, "drift": {"films": films[:2], "levels": [50, 100]}, "translation": {"films": films[2:5]}}
    (AGENT / "pilot").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AGENT / "pilot" / "pilot_inputs.json", "w"), indent=1)
    print("wrote", AGENT / "pilot" / "pilot_inputs.json")


def main(what):
    if what == "memes":
        build_memes()
        return
    memes = json.load(open(PRE / "memes_for_tracing.json"))["memes"]
    if what == "relations":
        relatives()
    elif what == "lists":
        shakespeare()
        rel = relatives()
        culture_pool(rel)
        batman_extended()
        dev_units(memes)
        planted_design(memes)
    elif what == "texts":
        stage = sys.argv[2]
        {"examples": stage_examples, "dev": stage_dev, "wikipedia": stage_wikipedia, "names": stage_names, "drift": stage_drift, "zh": stage_zh,
         "zhwiki": stage_zhwiki, "stories": stage_stories, "long": stage_long, "t2n": stage_t2n, "codes": stage_codes,
         "pilot": stage_pilot}[stage]()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lists")
