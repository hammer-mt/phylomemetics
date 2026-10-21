"""Crawl Wikipedia's 'Films based on works by William Shakespeare' category tree (plus TV-film categories).
Writes scout/category_members.json: article -> list of category paths."""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, category_members, api

ROOTS = ["Category:Films based on works by William Shakespeare",
         "Category:Television films based on works by William Shakespeare",
         "Category:Television shows based on works by William Shakespeare",
         "Category:Modern adaptations of works by William Shakespeare"]


def exists(cat):
    d = api({"action": "query", "titles": cat})
    return "-1" not in d["query"]["pages"]


def main():
    members = defaultdict(set)
    seen = set()
    stack = [(r, (r,), 0) for r in ROOTS if exists(r)]
    print("roots:", [s[0] for s in stack])
    while stack:
        cat, path, depth = stack.pop()
        if cat in seen:
            continue
        seen.add(cat)
        for m in category_members(cat):
            if m["ns"] == 14 and depth < 4:
                stack.append((m["title"], path + (m["title"],), depth + 1))
            elif m["ns"] == 0:
                members[m["title"]].add(" > ".join(c.replace("Category:", "") for c in path))
    print(len(seen), "categories,", len(members), "articles")
    json.dump({"categories": sorted(seen), "members": {k: sorted(v) for k, v in sorted(members.items())}},
              open(SCOUT / "category_members.json", "w"), indent=1)


if __name__ == "__main__":
    main()
