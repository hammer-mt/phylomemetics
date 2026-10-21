"""Wikidata SPARQL helper (metadata only)."""
import json
import sys
import time
import requests

HEADERS = {"User-Agent": "phylomemetics-scout/0.1 (https://github.com/hammer-mt/phylomemetics; research project) python-requests",
           "Accept": "application/sparql-results+json"}


def sparql(q, tries=5):
    for a in range(tries):
        try:
            r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, headers=HEADERS, timeout=90)
            if r.status_code == 200:
                return r.json()["results"]["bindings"]
            print("sparql status", r.status_code, r.text[:200], file=sys.stderr)
        except requests.RequestException as e:
            print("sparql error", e, file=sys.stderr)
        time.sleep(5 * 2 ** a)
    raise RuntimeError("sparql failed")
