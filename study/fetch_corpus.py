"""Fetch plot summaries from Wikipedia for the Batman lineage, its cousins, and a baseline story pool.

Writes data/corpus.jsonl: one record per work with its sentences.
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

# (wikipedia title, short label, year, medium, era)
BATMAN = [
    ("Batman (serial)", "1943 serial", 1943, "film", "Golden Age"),
    ("Batman and Robin (serial)", "1949 serial", 1949, "film", "Golden Age"),
    ("Batman (1966 film)", "Batman '66", 1966, "film", "Camp"),
    ("The Dark Knight Returns", "Dark Knight Returns", 1986, "comic", "Grim & gritty"),
    ("Batman: Year One", "Year One", 1987, "comic", "Grim & gritty"),
    ("Batman: The Killing Joke", "The Killing Joke", 1988, "comic", "Grim & gritty"),
    ("Batman: A Death in the Family", "A Death in the Family", 1988, "comic", "Grim & gritty"),
    ("Batman: Gotham by Gaslight", "Gotham by Gaslight", 1989, "comic", "Grim & gritty"),
    ("Arkham Asylum: A Serious House on Serious Earth", "Arkham Asylum (comic)", 1989, "comic", "Grim & gritty"),
    ("Batman (1989 film)", "Batman '89", 1989, "film", "Burton/Schumacher"),
    ("Batman Returns", "Batman Returns", 1992, "film", "Burton/Schumacher"),
    ("Batman: Mask of the Phantasm", "Mask of the Phantasm", 1993, "animation", "Animated"),
    ("Batman: Knightfall", "Knightfall", 1993, "comic", "Grim & gritty"),
    ("Batman Forever", "Batman Forever", 1995, "film", "Burton/Schumacher"),
    ("Batman: The Long Halloween", "The Long Halloween", 1996, "comic", "Grim & gritty"),
    ("Batman & Robin (film)", "Batman & Robin", 1997, "film", "Burton/Schumacher"),
    ("Batman Beyond: Return of the Joker", "Return of the Joker", 2000, "animation", "Animated"),
    ("Batman: Hush", "Hush", 2002, "comic", "Modern comics"),
    ("Batman Begins", "Batman Begins", 2005, "film", "Nolan"),
    ("The Dark Knight", "The Dark Knight", 2008, "film", "Nolan"),
    ("Batman R.I.P.", "Batman R.I.P.", 2008, "comic", "Modern comics"),
    ("Batman: Arkham Asylum", "Arkham Asylum (game)", 2009, "game", "Arkham games"),
    ("Batman: Under the Red Hood", "Under the Red Hood", 2010, "animation", "Animated"),
    ("Batman: Arkham City", "Arkham City", 2011, "game", "Arkham games"),
    ("The Dark Knight Rises", "Dark Knight Rises", 2012, "film", "Nolan"),
    ("Batman: Earth One", "Earth One", 2012, "comic", "Modern comics"),
    ("Batman: Arkham Knight", "Arkham Knight", 2015, "game", "Arkham games"),
    ("Batman v Superman: Dawn of Justice", "Batman v Superman", 2016, "film", "Snyder/Reeves"),
    ("The Lego Batman Movie", "Lego Batman", 2017, "film", "Parody"),
    ("Batman: White Knight", "White Knight", 2017, "comic", "Modern comics"),
    ("Batman Ninja", "Batman Ninja", 2018, "animation", "Animated"),
    ("Joker (2019 film)", "Joker (2019)", 2019, "film", "Snyder/Reeves"),
    ("The Batman (film)", "The Batman", 2022, "film", "Snyder/Reeves"),
]

# Ancestors and cousins: the "Billionaire Vigilante Orphans" slide plus Batman's documented influences.
COUSINS = [
    ("Puss in Boots", "Puss in Boots", 1697, "folktale", "Cousin"),
    ("The Count of Monte Cristo", "Count of Monte Cristo", 1844, "novel", "Cousin"),
    ("A Study in Scarlet", "Sherlock Holmes", 1887, "novel", "Cousin"),
    ("Dracula", "Dracula", 1897, "novel", "Cousin"),
    ("The Scarlet Pimpernel", "Scarlet Pimpernel", 1905, "novel", "Cousin"),
    ("The Mark of Zorro (1940 film)", "Zorro", 1940, "film", "Cousin"),
    ("The Bat (1926 film)", "The Bat (1926)", 1926, "film", "Cousin"),
    ("The Shadow (1994 film)", "The Shadow", 1994, "film", "Cousin"),
    ("Superman (1978 film)", "Superman", 1978, "film", "Cousin"),
    ("Spider-Man (2002 film)", "Spider-Man", 2002, "film", "Cousin"),
    ("Daredevil (film)", "Daredevil", 2003, "film", "Cousin"),
    ("Iron Man (2008 film)", "Iron Man", 2008, "film", "Cousin"),
    ("The Green Hornet (2011 film)", "Green Hornet", 2011, "film", "Cousin"),
    ("Moon Knight (miniseries)", "Moon Knight", 2022, "tv", "Cousin"),
]

# A broad pool of ordinary stories, no superheroes or masked vigilantes. Used only to set the
# "is this meme present?" threshold for each neuron.
BASELINE = [
    "Titanic (1997 film)", "Forrest Gump", "Toy Story", "Finding Nemo", "The Godfather", "Jaws (film)",
    "Casablanca (film)", "Pride & Prejudice (2005 film)", "Jurassic Park (film)", "The Lion King",
    "Back to the Future", "E.T. the Extra-Terrestrial", "The Shawshank Redemption", "Pulp Fiction",
    "Groundhog Day (film)", "Mean Girls", "The Notebook", "Up (2009 film)", "Frozen (2013 film)",
    "Gravity (2013 film)", "The Martian (film)", "Alien (film)", "Rocky", "Home Alone", "Ratatouille (film)",
    "Mamma Mia! (film)", "La La Land", "Gone Girl (film)", "The Silence of the Lambs (film)",
    "Fargo (1996 film)", "Chinatown (1974 film)", "Heat (1995 film)", "Die Hard", "Raiders of the Lost Ark",
    "The Wizard of Oz (1939 film)", "Singin' in the Rain", "Some Like It Hot", "The Sound of Music (film)",
    "Psycho (1960 film)", "Rear Window", "Vertigo (film)", "2001: A Space Odyssey (film)", "Star Wars (film)",
    "The Matrix", "Inception", "Ghostbusters", "Notting Hill (film)", "The Devil Wears Prada (film)",
    "Legally Blonde", "Mrs. Doubtfire", "Big (film)", "Shrek", "Spirited Away", "Parasite (2019 film)",
    "Knives Out", "Get Out", "Little Miss Sunshine", "The Social Network", "The Grand Budapest Hotel",
    "Mad Max: Fury Road", "Interstellar (film)", "Saving Private Ryan", "The Exorcist", "The Shining (film)",
    "Goodfellas", "Taxi Driver", "Moonlight (2016 film)", "Lost in Translation (film)", "Babe (film)",
    "The Princess Bride (film)",
]

SECTION_NAMES = re.compile(r"^(plot|synopsis|plot summary|story|summary|premise|storyline|plot and themes|narrative|overview)", re.I)
MAX_SENTENCES = 40
ABBREVS = ("Mr.", "Mrs.", "Dr.", "St.", "Jr.", "Sr.", "Ms.", "Lt.", "Sgt.", "vs.", "U.S.", "R.I.P.", "Gen.", "Capt.", "Det.")
SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z\"“‘(])")

HEADERS = {"User-Agent": "phylomemetics-pilot/0.1 (research project)"}


CACHE = Path("data/raw")


def fetch_extract(title):
    """Full plain-text article, cached on disk so reruns don't hit the API again."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (re.sub(r"[^\w]+", "_", title) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(8):
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, params={
            "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
            "titles": title, "redirects": 1, "format": "json", "maxlag": 5}, timeout=30)
        try:
            page = next(iter(r.json()["query"]["pages"].values()))
            path.write_text(json.dumps(page))
            time.sleep(1.5)
            return page
        except (ValueError, KeyError):
            wait = int(r.headers.get("retry-after", 0) or 0) or 5 * 2 ** attempt
            print(f"  rate limited on {title}, waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {title}")


def plot_section(title):
    page = fetch_extract(title)
    text = page.get("extract") or ""
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", text, flags=re.M)
    # parts = [lead, h1, body1, h2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        if SECTION_NAMES.match(parts[i]):
            body = re.sub(r"^===+.*?===+\s*$", "", parts[i + 1], flags=re.M)
            return page.get("title"), parts[i], body
    return page.get("title"), None, None


def sentences(body):
    body = re.sub(r"\s+", " ", body).strip()
    merged = []
    for piece in SPLIT.split(body):
        if merged and merged[-1].endswith(ABBREVS):
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    out = [s.strip() for s in merged if len(s.strip()) > 15]
    return out[:MAX_SENTENCES]


def main():
    out = Path("data/corpus.jsonl")
    out.parent.mkdir(exist_ok=True)
    rows = []
    jobs = [(t, l, y, m, e, "batman") for t, l, y, m, e in BATMAN]
    jobs += [(t, l, y, m, e, "cousin") for t, l, y, m, e in COUSINS]
    jobs += [(t, t.split(" (")[0], None, "film", "Baseline", "baseline") for t in BASELINE]
    for title, label, year, medium, era, group in jobs:
        resolved, section, body = plot_section(title)
        if not body:
            print(f"MISSING plot section: {title}", file=sys.stderr)
            continue
        sents = sentences(body)
        if len(sents) < 8:
            print(f"TOO SHORT ({len(sents)} sentences): {title}", file=sys.stderr)
            continue
        rows.append(dict(id=len(rows), title=resolved, label=label, year=year, medium=medium,
                         era=era, group=group, section=section, sentences=sents))
        print(f"{group:8s} {label:28s} {section:10s} {len(sents):3d} sentences")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} works")


if __name__ == "__main__":
    main()
