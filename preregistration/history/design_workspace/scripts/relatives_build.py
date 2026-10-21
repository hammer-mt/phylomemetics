"""Curated ground truth for T3: documented Batman influences, derivatives, siblings and the book's family, each with
its documentary evidence (Wikipedia article + verbatim sentence found by relatives_wp.py / manual check, Wikidata
property), a strength grade, and the candidate works (films, novels) that would carry it into the T3 pool.
Fetches each work's article and records plot-section size, DEV membership and T3-pool membership (metadata only).
Writes scout/relatives/documented_relatives.json and scout/relatives/works.jsonl (with raw plot sections)."""
import json
import re
import sys
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, fetch_extract, sentences
from t3_common import row_for

OUT = SCOUT / "relatives"
POOL = set(json.load(open(SCOUT / "t3" / "category_members.json"))["members"])

# relation: influence (Batman drew on it), predecessor (same type, earlier; named as predecessor), derivative (drew on Batman),
# sibling (shares a documented ancestor with Batman), book_tier1 / book_tier2 (Mike Taylor's earlier book pp. 21-22).
# strength: A = creator statement or Wikidata link plus Wikipedia text; B = Wikipedia text claim by a named source;
# C = comparison or review remark only. (article, regex) pairs locate the evidence sentence.
R = [
 dict(name="Zorro", relation=["influence", "book_tier1"], strength="A", wikidata="Batman P941 Zorro",
      evidence=[("Batman", r"Mark of Zorro \(1920\)"), ("Bob Kane", r"Zorro"), ("Zorro", r"Bob Kane has credited Zorro")],
      works=["The Curse of Capistrano", "The Mark of Zorro (1920 film)", "The Mark of Zorro (1940 film)", "The Mask of Zorro", "The Legend of Zorro", "Zorro (1975 Italian film)", "The Bold Caballero", "Zorro's Fighting Legion", "Zorro Rides Again"]),
 dict(name="The Bat (Rinehart and Hopwood play) / The Bat Whispers", relation=["influence"], strength="A", wikidata=None,
      evidence=[("Batman", r"Bat Whispers"), ("Bob Kane", r"Bat Whispers"), ("The Bat (play)", r"influenced the creation"), ("The Bat Whispers", r"inspiration for his character")],
      works=["The Bat (play)", "The Bat (1926 film)", "The Bat Whispers", "The Bat (1959 film)"]),
 dict(name="The Shadow", relation=["influence"], strength="A", wikidata="Batman P941 The Shadow",
      evidence=[("Batman", r"Partners of Peril"), ("Batman", r"Sherlock Holmes and the Shadow"), ("The Shadow", r"Partners of Peril"), ("The Shadow", r"biggest influence")],
      works=["The Shadow (1994 film)", "The Shadow Strikes", "International Crime (film)", "The Shadow (serial)", "The Shadow Returns", "Behind the Mask (1946 film)", "The Missing Lady", "The Living Shadow"]),
 dict(name="Sherlock Holmes", relation=["influence"], strength="A", wikidata="Batman P941 Sherlock Holmes",
      evidence=[("Batman", r"Sherlock Holmes and the Shadow"), ("Batman", r"master sleuth")],
      works=["A Study in Scarlet", "The Sign of the Four", "The Hound of the Baskervilles", "The Valley of Fear"]),
 dict(name="Doc Savage", relation=["influence"], strength="B", wikidata=None, evidence=[("Batman", r"Doc Savage")],
      works=["Doc Savage: The Man of Bronze", "The Man of Bronze"]),
 dict(name="Dick Tracy", relation=["influence"], strength="B", wikidata=None, evidence=[("Batman", r"Dick Tracy")],
      works=["Dick Tracy (1990 film)", "Dick Tracy (serial)", "Dick Tracy (1945 film)"]),
 dict(name="The Phantom (Lee Falk)", relation=["influence"], strength="B", wikidata=None,
      evidence=[("Bob Kane", r"The Phantom"), ("Bill Finger", r"The Phantom")],
      works=["The Phantom (1996 film)", "The Phantom (serial)"]),
 dict(name="The Black Bat (pulp)", relation=["influence"], strength="B", wikidata=None,
      evidence=[("Black Bat (pulp fiction character)", r"permanent influence on the Batman")], works=[]),
 dict(name="The Man Who Laughs (source of the Joker's look)", relation=["influence"], strength="A", wikidata=None,
      evidence=[("Bob Kane", r"Man Who Laughs")], works=["The Man Who Laughs (1928 film)", "The Man Who Laughs"]),
 dict(name="Robin Hood (source of Robin's name)", relation=["influence"], strength="B", wikidata=None,
      evidence=[("Bob Kane", r"Robin Hood")], works=["The Adventures of Robin Hood", "Robin Hood (1922 film)"]),
 dict(name="Superman", relation=["influence"], strength="A", wikidata="Batman P941 Superman",
      evidence=[("Detective Comics 27", r"inspiration to create a new Supe")], works=["Superman (1978 film)"]),
 dict(name="Count Dracula", relation=["influence"], strength="C", wikidata="Batman P941 Count Dracula", evidence=[], works=["Dracula"]),
 dict(name="The Scarlet Pimpernel", relation=["predecessor", "book_tier1"], strength="B", wikidata=None,
      evidence=[("Batman", r"predecessors in the Scarlet Pimpernel"), ("The Scarlet Pimpernel", r"Superman and Batman followed"), ("Zorro", r"direct literary predecessor")],
      works=["The Scarlet Pimpernel", "The Scarlet Pimpernel (1934 film)", "The Scarlet Pimpernel (1982 film)", "The Elusive Pimpernel", "Pimpernel Smith", "The Return of the Scarlet Pimpernel", "Eldorado (novel)", "The Elusive Pimpernel (novel)"]),
 dict(name="The Count of Monte Cristo", relation=["candidate_unverified"], strength="none", wikidata=None, evidence=[],
      works=["The Count of Monte Cristo", "The Count of Monte Cristo (2002 film)", "The Count of Monte Cristo (1934 film)", "The Count of Monte Cristo (2024 film)"],
      note="named in the design brief; no Wikipedia or Wikidata statement linking it to Batman was found in this scout"),
 dict(name="Iron Man", relation=["derivative", "book_tier1"], strength="B", wikidata=None,
      evidence=[("Iron Man", r"Layton was inspired by the vast collection of specialized outfits used by Batman"), ("Iron Man", r"Howard Hughes")],
      works=["Iron Man (2008 film)", "Iron Man 2", "Iron Man 3", "The Invincible Iron Man (film)", "Iron Man: Rise of Technovore"],
      note="Wikipedia documents a Batman influence on Iron Man's armours (Bob Layton) and Lee's Howard Hughes model; the book calls Iron Man Marvel's response to Batman"),
 dict(name="Moon Knight", relation=["derivative"], strength="A", wikidata="Moon Knight P941 Batman",
      evidence=[("Moon Knight", r"Marvel's version of Batman"), ("Moon Knight", r"pastiche of Batman")], works=["Moon Knight (miniseries)"]),
 dict(name="Nighthawk (Marvel)", relation=["derivative"], strength="A", wikidata="Nighthawk P941 Batman",
      evidence=[("Nighthawk (Marvel Comics)", r"Nighthawk based on Batman")], works=[]),
 dict(name="Nite Owl (Watchmen)", relation=["derivative"], strength="A", wikidata="Nite Owl P144 Batman",
      evidence=[("Nite Owl", r"parodies the appearance of the iconic popular superhero Batman")], works=["Watchmen (2009 film)", "Watchmen"]),
 dict(name="Midnighter", relation=["derivative"], strength="B", wikidata=None,
      evidence=[("The Authority (comics)", r"Batman pastiche"), ("Homosexuality in the Batman franchise", r"analogue of Batman")], works=[]),
 dict(name="Riggan Thomson / Birdman", relation=["derivative"], strength="A", wikidata="Riggan Thomson P144 Batman",
      evidence=[("Birdman (film)", r"references to Burton's Batman films")], works=["Birdman (film)"]),
 dict(name="Chase Me (film)", relation=["batman_lineage"], strength="A", wikidata="Chase Me P144 Batman", evidence=[], works=["Chase Me"],
      note="a 2003 Batman animated short: Batman lineage, not a derivative; keep as a positive control"),
 dict(name="Batwoman (Jerry Warren)", relation=["derivative"], strength="B", wikidata=None,
      evidence=[("The Wild World of Batwoman", r"make his own superhero bat-film")], works=["The Wild World of Batwoman"]),
 dict(name="Rat Pfink", relation=["derivative"], strength="B", wikidata=None,
      evidence=[("Rat Pfink a Boo Boo", r"Rat Pfink is a parody of Batman")], works=["Rat Pfink a Boo Boo"]),
 dict(name="Commander Courage", relation=["derivative"], strength="B", wikidata=None,
      evidence=[("Comic Book: The Movie", r"parody of Batman")], works=["Comic Book: The Movie"]),
 dict(name="Bluntman", relation=["derivative"], strength="B", wikidata=None,
      evidence=[("Jay & Silent Bob's Super Groovy Cartoon Movie!", r"parody of Batman")], works=["Jay & Silent Bob's Super Groovy Cartoon Movie!", "Jay and Silent Bob Strike Back"]),
 dict(name="Superhero Movie parody", relation=["derivative"], strength="C", wikidata=None,
      evidence=[("Batman Begins", r"parody of Batman Begins")], works=["Superhero Movie"], note="mainly a Spider-Man parody; the Batman Begins article lists it as a parody of that film"),
 dict(name="Blackwolf (Soon I Will Be Invincible)", relation=["derivative"], strength="B", wikidata=None,
      evidence=[("Soon I Will Be Invincible", r"Batman-like character")], works=["Soon I Will Be Invincible"]),
 dict(name="Blue Falcon", relation=["derivative"], strength="A", wikidata="Blue Falcon P941 Batman",
      evidence=[("Gary Owens", r"parody of Batman")], works=["Scooby-Doo! Mask of the Blue Falcon"]),
 dict(name="Darkwing Duck", relation=["derivative"], strength="A", wikidata="Darkwing P941 Batman",
      evidence=[("Darkwing Duck", r"reminiscent of pulp heroes")], works=[]),
 dict(name="Black Panther (Priest's run)", relation=["derivative_partial"], strength="C", wikidata=None,
      evidence=[("Black Panther (character)", r"initial concept was inspired by Batman")], works=[]),
 dict(name="Darkman", relation=["sibling"], strength="B", wikidata=None,
      evidence=[("Darkman", r"desire to adapt The Shadow"), ("Darkman", r"Comparison was also drawn to Batman")], works=["Darkman"]),
 dict(name="Puss in Boots (Shrek)", relation=["sibling", "book_tier2"], strength="B", wikidata=None,
      evidence=[("Puss in Boots (Shrek)", r"inspirations for Puss's characterization, such as Zorro")],
      works=["Puss in Boots (2011 film)", "Puss in Boots: The Last Wish", "Puss in Boots"]),
 dict(name="Spider-Man", relation=["book_tier2"], strength="book", wikidata=None, evidence=[], works=["Spider-Man (2002 film)", "Spider-Man 2", "The Amazing Spider-Man (film)", "Spider-Man: Homecoming"]),
 dict(name="Daredevil", relation=["book_tier2"], strength="book+C", wikidata=None,
      evidence=[("Daredevil (Marvel Comics character)", r"like Batman, Daredevil")], works=["Daredevil (film)"]),
 dict(name="John Wick", relation=["book_tier2"], strength="book", wikidata=None, evidence=[], works=["John Wick (film)", "John Wick: Chapter 2", "John Wick: Chapter 3 – Parabellum", "John Wick: Chapter 4"]),
 dict(name="Angel (Buffy the Vampire Slayer)", relation=["book_tier2"], strength="book", wikidata=None, evidence=[], works=[],
      note="a television character; no film or novel with a plot section"),
]


def find_evidence(article, pattern):
    page = fetch_extract(article, pause=0.15)
    if "missing" in page:
        return None
    for s in sentences(page.get("extract") or ""):
        if re.search(pattern, s):
            return s[:600]
    return None


out = []
works = []
for r in R:
    ev = [dict(article=a, sentence=find_evidence(a, p)) for a, p in r["evidence"]]
    r2 = {**r, "evidence": ev, "evidence_found": sum(1 for e in ev if e["sentence"]), "works_found": []}
    for t in r["works"]:
        w = row_for(t)
        if w.get("missing"):
            r2["works_found"].append(dict(query=t, missing=True))
            continue
        w["relative"] = r["name"]
        w["in_t3_pool"] = w["title"] in POOL
        works.append(w)
        r2["works_found"].append(dict(title=w["title"], medium=w["medium"], year=w["year"], n_sentences=w["n_sentences"],
                                      in_dev=w["in_dev"], in_t3_pool=w["in_t3_pool"]))
    out.append(r2)
    print(r["name"], "| evidence", r2["evidence_found"], "/", len(ev), "|", [(x.get("title") or x["query"], x.get("n_sentences"), "DEV" if x.get("in_dev") else "", "POOL" if x.get("in_t3_pool") else "") for x in r2["works_found"]], flush=True)
json.dump(out, open(OUT / "documented_relatives.json", "w"), indent=1, ensure_ascii=False)
with open(OUT / "works.jsonl", "w") as f:
    for w in works:
        f.write(json.dumps(w, ensure_ascii=False) + "\n")
