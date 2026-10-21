"""T3 pool scout, step 1: crawl Wikipedia category trees for superhero, vigilante, swashbuckler, masked-avenger
and pulp-hero films and novels, plus the named family members' film categories. Metadata only.
Writes scout/t3/category_members.json: {root: {...}, members: {title: [category paths]}}."""
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT, category_members

OUT = SCOUT / "t3"
# root -> (bucket, max depth)
ROOTS = {
    "Category:Superhero films": ("superhero_film", 3),
    "Category:Vigilante films": ("vigilante_film", 3),
    "Category:Swashbuckler films": ("swashbuckler_film", 3),
    "Category:Zorro films": ("family_zorro", 1),
    "Category:Films based on works by Johnston McCulley": ("family_zorro", 1),
    "Category:Scarlet Pimpernel films": ("family_pimpernel", 1),
    "Category:Films based on The Count of Monte Cristo": ("relative_monte_cristo", 1),
    "Category:Works based on The Count of Monte Cristo": ("relative_monte_cristo", 2),
    "Category:Iron Man films": ("family_iron_man", 2),
    "Category:Puss in Boots": ("family_puss", 2),
    "Category:The Green Hornet films": ("pulp_hero", 1),
    "Category:Green Hornet": ("pulp_hero", 2),
    "Category:The Phantom": ("pulp_hero", 2),
    "Category:Doc Savage": ("pulp_hero", 2),
    "Category:Spider (pulp fiction character)": ("pulp_hero", 2),
    "Category:Works originally published in The Shadow (magazine)": ("pulp_hero", 1),
    "Category:Pulp stories": ("pulp_story", 1),
    "Category:Robin Hood films": ("outlaw_hero_film", 2),
    "Category:Superhero novels": ("superhero_novel", 3),
    "Category:Superhero comedy films": ("superhero_film", 2),
    "Category:Superhero film serials": ("superhero_film", 1),
}
# subcategories that hold characters, music, media files or navigation rather than works
BAD_SUB = re.compile(r"(characters?|soundtracks?|songs?|music|navigational|navbox|templates?|images|posters|redirects|album|video games|"
                     r"actors|people|cast|episodes|comics|television|TV series|stubs|fan films|toys|lists)", re.I)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    members = defaultdict(set)
    buckets = defaultdict(set)
    seen = set()
    skipped = set()
    for root, (bucket, maxd) in ROOTS.items():
        stack = [(root, (root,), 0)]
        while stack:
            cat, path, depth = stack.pop()
            if (cat, bucket) in seen:
                continue
            seen.add((cat, bucket))
            for m in category_members(cat):
                if m["ns"] == 14:
                    if depth + 1 <= maxd and not BAD_SUB.search(m["title"].replace("Category:", "")):
                        stack.append((m["title"], path + (m["title"],), depth + 1))
                    elif depth + 1 <= maxd:
                        skipped.add(m["title"])
                elif m["ns"] == 0:
                    members[m["title"]].add(" > ".join(c.replace("Category:", "") for c in path))
                    buckets[m["title"]].add(bucket)
        print(root, "->", sum(1 for t in buckets if bucket in buckets[t]), "articles in bucket", bucket, flush=True)
    cats = sorted({c for c, _ in seen})
    json.dump({"roots": {k: list(v) for k, v in ROOTS.items()}, "categories": cats, "skipped_subcats": sorted(skipped),
               "members": {k: {"paths": sorted(v), "buckets": sorted(buckets[k])} for k, v in sorted(members.items())}},
              open(OUT / "category_members.json", "w"), indent=1)
    print(len(cats), "categories crawled;", len(members), "articles;", len(skipped), "subcats skipped")


if __name__ == "__main__":
    main()
