"""Ground truth scout: documented Batman influences and derivatives from English Wikipedia text.
(1) Sentences in the Batman, Bob Kane, Bill Finger and Detective Comics #27 articles that name an influence.
(2) Full-text search for derivative phrases ('inspired by Batman', 'pastiche of Batman', ...), with the sentence
that carries the claim in each hit article. Evidence sentences are saved verbatim in the scout file (Wikipedia,
CC BY-SA) for later human checking; nothing is analysed. Writes scout/relatives/wikipedia_evidence.json."""
import json
import re
import sys
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, api, fetch_extract, sentences

OUT = SCOUT / "relatives"
INFL = re.compile(r"\b(inspir\w*|influenc\w*|based (?:on|upon)|model(?:l)?ed (?:on|after)|borrow\w*|drew (?:on|from|upon)|derived|"
                  r"swipe\w*|lifted|cop(?:y|ied)|took .{0,30}from|reminiscent|resembl\w*|patterned)\b", re.I)
res = {"influence_sentences": {}, "search_hits": {}, "derivative_evidence": {}}
for t in ["Batman", "Bob Kane", "Bill Finger", "Detective Comics 27", "Batman (comic book)", "Bruce Wayne", "Alfred Pennyworth",
          "Zorro", "The Shadow", "The Scarlet Pimpernel", "Iron Man", "Moon Knight", "Nighthawk (Marvel Comics)", "Owlman (character)",
          "Midnighter", "Nite Owl", "Darkwing Duck", "The Black Bat", "Green Hornet", "The Phantom", "Darkman", "Doc Savage",
          "The Bat (play)", "The Bat Whispers", "The Mark of Zorro (1920 film)", "Batman in popular culture", "Parodies of Batman",
          "The Spider (pulp fiction)", "The Spirit", "Black Panther (character)", "Ozymandias (Watchmen)", "Batfink", "Blue Falcon",
          "Knight Watchman", "Mermaid Man", "Professor Night", "Rat-Man", "Hourman", "The Clock (comics)", "Big Daddy (Kick-Ass)",
          "The Tick", "Batman (Batman Beyond)", "Spawn (character)", "Rorschach (character)", "The Night Owl", "Daredevil (Marvel Comics character)",
          "Tony Stark (Marvel Cinematic Universe)", "Puss in Boots", "Count of Monte Cristo", "Sherlock Holmes", "Count Dracula",
          "The Gray Seal", "Jimmie Dale", "Nick Carter (literary character)", "Leonardo da Vinci's ornithopter"]:
    page = fetch_extract(t, pause=0.15)
    if "missing" in page:
        res["influence_sentences"][t] = "MISSING"
        continue
    ss = sentences(page.get("extract") or "")
    hits = [s for s in ss if "Batman" in s and INFL.search(s)] if t not in ("Batman", "Bob Kane", "Bill Finger", "Detective Comics 27", "Batman (comic book)", "Bruce Wayne") \
        else [s for s in ss if INFL.search(s)]
    res["influence_sentences"][page["title"]] = hits[:40]
    print(page["title"], len(hits), flush=True)

PHRASES = ["inspired by Batman", "based on Batman", "pastiche of Batman", "parody of Batman", "analogue of Batman", "homage to Batman",
           "Marvel's Batman", "answer to Batman", "modeled on Batman", "modelled on Batman", "Batman analogue", "Batman pastiche",
           "Batman expy", "version of Batman", "riff on Batman", "take on Batman", "Batman-like", "Batman knockoff", "Batman clone",
           "spoof of Batman", "Batman parody", "influenced by Batman", "similar to Batman"]
for ph in PHRASES:
    d = api({"action": "query", "list": "search", "srsearch": f"\"{ph}\"", "srnamespace": 0, "srlimit": 50})
    res["search_hits"][ph] = [(h["title"], re.sub(r"<[^>]+>", "", h["snippet"])) for h in d["query"]["search"]]
    print(ph, d["query"]["searchinfo"]["totalhits"], flush=True)
json.dump(res, open(OUT / "wikipedia_evidence.json", "w"), indent=1, ensure_ascii=False)
