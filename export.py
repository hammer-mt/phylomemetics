"""Stage 3: merge discovery, labels and phylogeny results into report/data.js for the write-up page."""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from discover import load_units

ACTS = Path(os.environ.get("PHYLO_ACTS", "activations"))
RESULTS = Path("results")

# Era groups used for colour on the page (fewer than the fine-grained eras in the corpus)
ERA_GROUP = {
    "Golden Age": "Early (1943–66)", "Camp": "Early (1943–66)",
    "Grim & gritty": "Grim & gritty comics (1986–96)",
    "Burton/Schumacher": "Burton & Schumacher films",
    "Animated": "Animated films",
    "Modern comics": "Modern comics (2002–17)",
    "Nolan": "Nolan trilogy",
    "Arkham games": "Arkham games",
    "Snyder/Reeves": "Recent films (2016–22)", "Parody": "Recent films (2016–22)",
    "Cousin": "Cousins & ancestors",
}


def main():
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    sentences = [s for w in works for s in w["sentences"]]
    lineage = [w for w in works if w["group"] != "baseline"]
    lin_ids = [w["id"] for w in lineage]
    phylo = json.load(open(RESULTS / "phylo.json"))
    memes_q = json.load(open(RESULTS / "qwen3-8b" / "memes.json"))["families"]
    labels = json.load(open(RESULTS / "labels.json"))          # unit -> {label, note, kind}

    meta, pos, neg, ps, ns = load_units(ACTS / "qwen3-8b")
    n = pos.shape[1]
    sent_work = np.array(meta["sent_work"])

    llama = phylo["models"].get("llama3.1-8b", {})
    q17 = phylo["models"].get("qwen3-1.7b", {})
    fam_index = {f["unit"]: i for i, f in enumerate(memes_q)}
    n_bat = sum(w["group"] == "batman" for w in works)

    out_memes = []
    for key, lab in labels.items():
        u = int(key)
        f = memes_q[fam_index[u]]
        prof = (pos[:, u].astype(np.float32) * ps[u]) if u < n else (-neg[:, u - n].astype(np.float32) * ns[u - n])
        best = {}
        for wid in lin_ids:
            rows = np.where(sent_work == wid)[0]
            i = rows[np.argmax(prof[rows])]
            best[wid] = sentences[i]
        z = {wid: f["z"][wid] for wid in lin_ids}
        k = fam_index[u]
        entry = {
            "unit": u, "label": lab["label"], "note": lab.get("note", ""), "kind": lab.get("kind", "trope"),
            "addr": f"L{f['layer']}·{f['neuron']}{f['pole']}", "n_members": f["n_members"],
            "n_batman": f["n_batman"], "band": "canon" if f["n_batman"] >= 0.7 * n_bat else "era", "present": [w for w in f["present"] if w in set(lin_ids)],
            "z": z, "best_sentence": best,
            "top": [{"work": t["work"], "text": t["text"]} for t in f["top_sentences"][:4]],
        }
        for tag, res in (("llama", llama), ("qwen17", q17)):
            if "partners" in res:
                p = res["partners"][k]
                entry[tag] = {"r": p["r"], "addr": f"L{p['layer']}{p['pole']}", "presence_jaccard": p["presence_jaccard"],
                              "present": [w for w in p["present"] if w in set(lin_ids)],
                              "top": p["top_sentences"][:4]}
        out_memes.append(entry)

    # cross-model match by meme strength band (bands are by Qwen strength rank, independent of labelling)
    strength = np.array([f["strength"] for f in memes_q])
    order = np.argsort(-strength)
    bands = [(0, 33), (33, 100), (100, 500), (500, 2000), (2000, 6000), (6000, len(memes_q))]
    match = {}
    for tag in ("llama3.1-8b", "qwen3-1.7b"):
        pr = phylo["models"][tag]["partner_r"]
        r = np.array(pr["memes"])
        match[tag] = {"bands": [{"from": a + 1, "to": b, "median_r": round(float(np.median(r[order[a:b]])), 3),
                                 "share_over_07": round(float(np.mean(r[order[a:b]] > 0.7)), 3)} for a, b in bands],
                      "all_median": pr["memes_median"], "random_median": pr["random_units_median"],
                      "shuffled_median": pr["shuffled_median"],
                      "labelled_median": float(np.median([e[("llama" if tag.startswith("llama") else "qwen17")]["r"] for e in out_memes]))}

    # chance rate for "nearest neighbour is from the same era" among Batman works
    eras = [w["era"] for w in works if w["group"] == "batman"]
    nn_chance = float(np.mean([(eras.count(e) - 1) / (len(eras) - 1) for e in eras]))

    # cell-level agreement of the meme grid: Qwen meme present/absent vs its twin, over Batman works + cousins
    def grid_agreement(ks, tag):
        P = phylo["models"][tag]["partners"]
        a = np.array([w in set(memes_q[k]["present"]) for k in ks for w in lin_ids])
        b = np.array([w in set(P[k]["present"]) for k in ks for w in lin_ids])
        po = float((a == b).mean())
        pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        return {"agree": round(po, 3), "chance": round(pe, 3), "kappa": round((po - pe) / (1 - pe), 3)}

    named_k = [fam_index[int(u)] for u in labels]
    agreement = {tag: {"named": grid_agreement(named_k, tag), "strongest_33": grid_agreement(order[:33], tag),
                       "all": grid_agreement(range(len(memes_q)), tag)} for tag in ("llama3.1-8b", "qwen3-1.7b")}

    stats = {"tfidf": {k: v for k, v in phylo["tfidf"].items() if k != "newick"}, "nn_chance": nn_chance, "match": match,
             "agreement": agreement}
    trees = {"tfidf": phylo["tfidf"]["newick"]}
    for tag, res in phylo["models"].items():
        stats[tag] = {k: v for k, v in res.items() if k not in ("newick", "D", "partners", "partner_r")}
        trees[tag] = res["newick"]

    data = {
        "updated": datetime.now().strftime("%-d %b %Y, %H:%M"),
        "works": [{"id": w["id"], "label": w["label"], "title": w["title"], "year": w["year"], "era": w["era"],
                   "group_label": ERA_GROUP[w["era"]], "medium": w["medium"], "group": w["group"],
                   "n_sentences": len(w["sentences"])} for w in lineage],
        "n_baseline": sum(w["group"] == "baseline" for w in works),
        "baseline_titles": [w["label"] for w in works if w["group"] == "baseline"],
        "n_families_qwen": len(memes_q),
        "memes": out_memes, "stats": stats, "trees": trees,
        "n_sentences": len(sentences), "n_units_qwen": 2 * n,
        "wording": json.load(open(RESULTS / "wording.json")) if (RESULTS / "wording.json").exists() else None,
    }
    Path("report/data.js").write_text("window.PM_DATA = " + json.dumps(data) + ";\n")
    Path("report/data.json").write_text(json.dumps(data, indent=1))
    print("wrote report/data.js", len(out_memes), "memes", Path("report/data.js").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
