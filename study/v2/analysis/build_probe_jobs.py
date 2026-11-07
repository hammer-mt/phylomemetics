"""The recognition probe (PLAN.md, Section 7): does the model work out which story a name-free summary tells?

Writes two job files for v2/gpu/ablate.py, which with no sets simply puts each text and question to the untouched model:

  v2/cache/jobs/recognise_hero.json   question 1; kept: the greedy reply and the log-probability that it starts with
                                      each of the hero's names (Batman's for every work; a hero film's own hero too)
  v2/cache/jobs/recognise_title.json  question 2; kept: the greedy reply (24 tokens) and the log-probability of the
                                      work's own title

    cd study && ../.venv/bin/python v2/analysis/build_probe_jobs.py
"""
import json
from pathlib import Path

import common as C  # noqa: F401  (puts study/ on the path)

import experiments as X

Q_HERO = ("Every name in this plot summary has been replaced by a code such as X1. Which famous fictional hero is the story about? "
          "Reply with only the name the hero is best known by, not a code.")
Q_TITLE = ("Every name in this plot summary has been replaced by a code such as X1. Which film, comic, game or book is this a summary of? "
           "Reply with its title only.")
BATMAN = ["Batman", "The Batman", "Bruce Wayne"]
HEROES = {"Zorro": ["Zorro"], "The Shadow": ["The Shadow", "Lamont Cranston"], "Superman": ["Superman", "Clark Kent"],
          "Spider-Man": ["Spider-Man", "Peter Parker"], "Daredevil": ["Daredevil", "Matt Murdock"], "Iron Man": ["Iron Man", "Tony Stark"],
          "Green Hornet": ["The Green Hornet", "Britt Reid"]}          # The Bat (1926) is left out: its title character is the criminal


def text_of(work):
    return " ".join(work["sentences"])


def main():
    plan = Path("v2/PLAN.md").read_text()
    assert Q_HERO in plan and Q_TITLE in plan, "the questions must be the plan's, word for word"
    out = Path("v2/cache/jobs")
    out.mkdir(parents=True, exist_ok=True)
    hero, title = [], []
    for w, a in zip(X.works, X.neutral):
        own = HEROES.get(w["label"], [])
        hero.append({"id": w["label"], "group": w["group"], "text": text_of(a), "answers": BATMAN + own, "own_hero_names": own})
        title.append({"id": w["label"], "group": w["group"], "text": text_of(a), "answers": [w["title"]]})
    json.dump({"question": Q_HERO, "reply_tokens": 12, "items": hero}, open(out / "recognise_hero.json", "w"), ensure_ascii=False)
    json.dump({"question": Q_TITLE, "reply_tokens": 24, "items": title}, open(out / "recognise_title.json", "w"), ensure_ascii=False)
    print("wrote", len(hero), "items to", out / "recognise_hero.json", "and", out / "recognise_title.json")


if __name__ == "__main__":
    main()
