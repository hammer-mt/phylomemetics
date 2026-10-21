"""Fetch the lead image of each work's Wikipedia article (~500px thumbnail) into data/images/, with data/images.json.

The images are copyrighted posters, covers and box art: they are used only for analysis and are not redistributed.
PageImages skips non-free images, so infobox_images() reads the infobox's image field for the rest.
"""
import json
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "phylomemetics-images/0.1 (research project)"}
works = [json.loads(l) for l in open("data/works.jsonl")]


def get(url, **kw):
    for attempt in range(8):
        r = requests.get(url, headers=HEADERS, timeout=30, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("retry-after", 0) or 0) or 5 * 2 ** attempt
            print(f"  waiting {wait}s ({r.status_code})", flush=True)
            time.sleep(wait)
            continue
        return r
    raise RuntimeError(url)


def images():
    Path("data/images").mkdir(parents=True, exist_ok=True)
    meta_path = Path("data/images.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    titles = [w["title"] for w in works]
    url_for = {}
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        r = get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "pageimages", "piprop": "thumbnail|name", "pithumbsize": 500,
            "titles": "|".join(chunk), "format": "json", "redirects": 1})
        for p in r.json()["query"]["pages"].values():
            if "thumbnail" in p:
                url_for[p["title"]] = (p["thumbnail"]["source"], p.get("pageimage"))
        time.sleep(2)
    for w in works:
        if str(w["id"]) in meta:
            continue
        if w["title"] not in url_for:
            meta[str(w["id"])] = {"title": w["title"], "file": None}
            continue
        src, name = url_for[w["title"]]
        r = get(src)
        path = Path("data/images") / f"{w['id']}.jpg"
        path.write_bytes(r.content)
        meta[str(w["id"])] = {"title": w["title"], "file": str(path), "source": src, "commons_name": name}
        time.sleep(0.5)
    meta_path.write_text(json.dumps(meta, indent=1))
    have = sum(1 for v in meta.values() if v["file"])
    print("images:", have, "of", len(works), "; missing:", [v["title"] for v in meta.values() if not v["file"]])


def infobox_images():
    """Lead images skipped by PageImages (non-free posters and covers): read the infobox's image field instead."""
    import re
    meta_path = Path("data/images.json")
    meta = json.loads(meta_path.read_text())
    missing = [w for w in works if not meta.get(str(w["id"]), {}).get("file")]
    files = {}
    for i in range(0, len(missing), 25):
        chunk = missing[i:i + 25]
        r = get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "titles": "|".join(w["title"] for w in chunk), "format": "json", "formatversion": 2})
        for p in r.json()["query"]["pages"]:
            text = p.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
            m = re.search(r"\|\s*(?:image|cover|image_file|poster|cover_image)\s*=\s*(?:\[\[)?(?:File:|Image:)?([^\n|\]<{]+?\.(?:jpe?g|png|gif|webp|svg))", text, re.I)
            if m:
                files[p["title"]] = m.group(1).strip()
        time.sleep(2)
    names = list(files.values())
    urls = {}
    for i in range(0, len(names), 40):
        r = get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 500,
            "titles": "|".join("File:" + n for n in names[i:i + 40]), "format": "json", "formatversion": 2})
        q = r.json()["query"]
        norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
        for p in q["pages"]:
            if "imageinfo" in p:
                key = norm.get(p["title"], p["title"]).replace("File:", "", 1)
                urls[key] = p["imageinfo"][0].get("thumburl") or p["imageinfo"][0]["url"]
        time.sleep(2)
    for w in missing:
        name = files.get(w["title"])
        src = urls.get(name) if name else None
        if not src:
            src = next((u for k, u in urls.items() if name and k.replace("_", " ").lower() == name.replace("_", " ").lower()), None)
        if not src:
            meta[str(w["id"])] = {"title": w["title"], "file": None, "infobox_name": name}
            continue
        path = Path("data/images") / f"{w['id']}.jpg"
        if not path.exists():
            try:
                r = get(src.split("?")[0])
                if not r.ok:
                    raise RuntimeError(r.status_code)
                path.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print("  could not fetch", w["title"], e, flush=True)
                meta[str(w["id"])] = {"title": w["title"], "file": None, "infobox_name": name}
                continue
            time.sleep(3)
        meta[str(w["id"])] = {"title": w["title"], "file": str(path), "source": src, "infobox_name": name}
        meta_path.write_text(json.dumps(meta, indent=1))
    meta_path.write_text(json.dumps(meta, indent=1))
    have = sum(1 for v in meta.values() if v.get("file"))
    print("images:", have, "of", len(works), "; still missing:", [v["title"] for v in meta.values() if not v.get("file")])


if __name__ == "__main__":
    images()
    infobox_images()
