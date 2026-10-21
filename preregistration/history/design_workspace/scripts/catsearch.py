"""Search category names (namespace 14) on Wikipedia. Metadata only."""
import sys
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from wiki import api
for q in sys.argv[1:]:
    d = api({"action": "query", "list": "search", "srsearch": f"intitle:{q}", "srnamespace": 14, "srlimit": 40})
    print("##", q, ":", [h["title"].replace("Category:", "") for h in d["query"]["search"]])
