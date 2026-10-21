"""Build report/data2.js: canon, spread, experiments (Phase 2), images (Phase 3), lineage, book links."""
import json
from pathlib import Path

import numpy as np

works = [json.loads(l) for l in open("data/corpus.jsonl")]
W = {w["id"]: w for w in works}
canon = json.load(open("results/canon.json"))
E = json.load(open("results/experiments.json"))
fams = json.load(open("results/qwen3-8b/memes.json"))["families"]
fam_by_unit = {f["unit"]: f for f in fams}
lab_old = {int(k): v for k, v in json.load(open("results/labels.json")).items()}
lab_wf = {int(k): v for k, v in json.load(open("results/labels_wf.json")).items()}
ph = json.load(open("results/phylo.json"))
pv = json.load(open("data/pageviews.json"))


def label_for(u):
    if u in lab_old:
        return {**lab_old[u], "confidence": "high", "batman_specific": None}
    return lab_wf.get(u)


# ---- canon chips: labelled families, tier by Batman share, specific by cousin count
chips = []
for r in canon["memes"]:
    lab = label_for(r["unit"])
    if not lab or lab.get("kind") in ("mixed", "register") or lab.get("confidence") == "low":
        continue
    chips.append({"unit": r["unit"], "label": lab["label"], "kind": lab["kind"], "note": lab.get("note", ""),
                  "n_batman": r["n_batman"], "n_cousin": r["n_cousin"], "tier": r["tier"], "specific": r["specific"],
                  "first_year": r["first_year"]})
# keep one chip per label (highest prevalence)
seen, uniq = set(), []
for c in sorted(chips, key=lambda c: (-c["n_batman"], c["label"])):
    key = c["label"].lower()
    if key in seen:
        continue
    seen.add(key)
    uniq.append(c)

# ---- spread swimlane: named optional memes, years present
SWIM = ["Taking in an orphan as a ward", "The villain's lair", "A bomb", "The Riddler", "The antidote", "Harvey Dent",
        "A funeral", "Jason Todd", "A corrupt police force", "Fell into a vat of chemicals", "Fear toxin", "Two-Face's coin",
        "Lucius Fox", "Ra's al Ghul", "A mutating super-formula"]
by_label = {v["label"]: int(k) for k, v in json.load(open("results/labels.json")).items()}
swim = []
for name in SWIM:
    u = by_label.get(name)
    if u is None or u not in fam_by_unit:
        continue
    pres = set(fam_by_unit[u]["present"])
    swim.append({"label": name, "batman": sorted(w["id"] for w in works if w["group"] == "batman" and w["id"] in pres),
                 "cousins": sorted(w["id"] for w in works if w["group"] == "cousin" and w["id"] in pres)})

# ---- experiments summary
CORE_LABELS = {124334: "A secret identity kept with a disguise", 131948: "The origin wound: a costume to fight crime",
               327575: "Fighting crime in costume", 363862: "A self-imposed mission, by night", 558852: "Becoming a costumed hero",
               698739: "Partners and alter egos", 839470: "No clear theme"}


def slim_e1(e1):
    keep = {}
    for v in ("A_name_free", "B_canon_from_name_free", "ceiling_original", "A_iconography_dropped"):
        keep[v] = {k: e1[v][k] for k in ("hero_film_cousins", "held_out_films", "literary_cousins")}
    keep["per_cousin_beaten_by"] = e1["A_name_free"]["per_cousin_beaten_by"]
    keep["per_batman_vs_hero"] = e1["A_name_free"]["per_batman_vs_hero"]
    keep["null"] = e1["null_label_shuffle"]
    keep["length_only"] = e1["length_only_auc_vs_hero"]
    keep["core"] = {k: v for k, v in e1["core"].items() if not isinstance(v, list)}
    return keep


exps = {}
for tag in ("qwen3-8b", "llama3.1-8b"):
    r = E[tag]
    e2 = r["E2"]
    exps[tag] = {"E1": slim_e1(r["E1"]),
                 "E2": {"primary": e2["primary"], "sens": {k: {"stat": e2[k]["mean_share_beating_source"], "null": e2[k]["null_mean"], "p": e2[k]["p_one_sided"]}
                                                             for k in ("original_upper_bound", "raw_rate", "no_time_rule", "strong_subset", "without_killing_joke", "band_3_16", "band_4_16", "band_2_8")},
                        "sequels": e2["sequels_name_free"], "sequels_original_top2": e2["sequels_original"]["top2"]},
                 "E3": {k: {kk: vv for kk, vv in r["E3"][k].items()} for k in ("primary", "original_text", "with_adapted", "reverse_time_placebo", "baseline_film_placebo")}}
exps["core_labels"] = [{"unit": u, "label": l} for u, l in CORE_LABELS.items() if u in E["qwen3-8b"]["E1"]["core"]["core_units"]]

# ---- lineage: closest relatives
lineage = {}
lab = [w["label"] for w in ph["works"]]
grp = [w["group"] for w in ph["works"]]
for tag in ("qwen3-8b", "llama3.1-8b"):
    D = np.array(ph["models"][tag]["D"])
    bat = [i for i, g in enumerate(grp) if g == "batman"]
    lineage[tag] = {"closest_to_batman": sorted(({"label": lab[i], "year": ph["works"][i]["year"], "mean_distance": round(float(D[i, bat].mean()), 3)}
                                                 for i, g in enumerate(grp) if g == "cousin"), key=lambda x: x["mean_distance"]),
                    "nearest": {name: [lab[j] for j in np.argsort(D[lab.index(name)])[1:4]] for name in ("Zorro", "Iron Man", "Scarlet Pimpernel", "Puss in Boots")}}

data = {"canon": {"tiers": canon["tiers"], "n_core": canon["n_core"], "n_core_specific": canon["n_core_specific"], "chips": uniq,
                  "per_work": canon["per_work"], "per_cousin": canon["per_cousin"]},
        "swim": swim, "experiments": exps, "lineage": lineage,
        "pageviews": {str(k): v for k, v in pv.items()},
        "doc_url": "https://claude.ai/artifact/U2ifBBtEGYfSdia2FjeaZy"}
Path("report/data2.js").write_text("window.PM2 = " + json.dumps(data) + ";\n")
# the exploratory image experiment has its own page (report/images.html)
vpath = Path("results/vision_final.json")
if vpath.exists():
    Path("report/vision.js").write_text("window.PM_VISION = " + json.dumps(json.load(open(vpath))) + ";\n")
print("wrote report/data2.js", Path("report/data2.js").stat().st_size // 1024, "KB;", len(uniq), "canon chips;", len(swim), "swim rows")
