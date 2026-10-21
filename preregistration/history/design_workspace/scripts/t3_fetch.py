"""T3 pool scout, step 2: fetch every crawled candidate article and record metadata plus the raw plot section
(saved, not analysed). Writes scout/t3/candidates.jsonl."""
import json
import sys

sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import SCOUT
from t3_common import row_for, SKIP

OUT = SCOUT / "t3"
cm = json.load(open(OUT / "category_members.json"))["members"]
done = {}
path = OUT / "candidates.jsonl"
if path.exists():
    for l in open(path):
        r = json.loads(l)
        done[r["query"]] = r
titles = [t for t in cm if not SKIP.match(t)]
print(len(titles), "titles;", len(done), "already done", flush=True)
with open(path, "a") as f:
    for i, t in enumerate(titles):
        if t in done:
            continue
        r = row_for(t)
        r["buckets"] = cm[t]["buckets"]
        r["paths"] = cm[t]["paths"]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        if i % 100 == 0:
            print(i, "/", len(titles), t, flush=True)
print("done")
