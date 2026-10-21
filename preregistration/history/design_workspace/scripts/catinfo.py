"""Print page/subcat counts for candidate Wikipedia categories (metadata only)."""
import sys
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import api

cats = sys.argv[1:]
for i in range(0, len(cats), 40):
    chunk = cats[i:i + 40]
    d = api({"action": "query", "prop": "categoryinfo", "titles": "|".join(chunk)})
    for p in d["query"]["pages"].values():
        ci = p.get("categoryinfo", {})
        print(f"{p['title']:70s} {'MISSING' if 'missing' in p else ''} pages={ci.get('pages')} subcats={ci.get('subcats')}")
