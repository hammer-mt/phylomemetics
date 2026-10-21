"""Phase 3: visual memes. Uses the activations saved by vision_extract.py.

1. Score: a work's score on a unit = mean over its 10 views. A unit is present when the score beats all but one
   of the ordinary-film posters (the same rule as for text).
2. Visual memes = units present in >= MIN_BAT of the Batman images. Contact sheets of each candidate's top
   images are written to data/vision_sheets/ for labelling (never published).
3. Twins: each CLIP visual meme's best-correlated DINOv2 unit over all 1,110 views; baselines = active CLIP units
   that are not memes, and views shuffled.
4. Probes: generated images (data/probes/) run through both models; each meme unit's response reported as a
   percentile of its responses to the 111 corpus images.

Writes results/vision.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.argv = sys.argv[:1] + ["probe"]
import vision_extract as vx  # noqa: E402

RNG = np.random.default_rng(3)
V = vx.ACTS
MIN_BAT = 6
works = [json.loads(l) for l in open("data/works.jsonl")]
W = {w["id"]: w for w in works}


def load(model):
    meta = json.load(open(V / model / "meta.json"))
    views = np.load(V / model / "views.npy").astype(np.float32)
    iw = np.array(meta["image_work"])
    S = views.reshape(len(iw), 10, -1).mean(1)
    return meta, views, iw, S


def main():
    out = {}
    mc, vc, iw, Sc = load("clip")
    md, vd, iw_d, Sd = load("dino")
    assert (iw == iw_d).all()
    grp = np.array([W[i]["group"] for i in iw])
    year = np.array([W[i]["year"] or 0 for i in iw])
    bat, ref, cou = np.where(grp == "batman")[0], np.where(grp == "baseline")[0], np.where(grp == "cousin")[0]
    out["n_images"] = {"batman": int(len(bat)), "films": int(len(ref)), "cousins": int(len(cou))}
    from scipy.stats import binom
    p0 = 2 / (len(ref) + 1)
    for name, S in (("clip", Sc), ("dino", Sd)):
        bar = np.partition(S[ref], -2, axis=0)[-2]
        P = S > bar
        nb = P[bat].sum(0)
        spread = bar - np.median(S[ref], axis=0) + 1e-6
        strength = ((S[bat] - bar) / spread).clip(min=0).mean(0)
        dist = {k: int((nb >= k).sum()) for k in (3, 6, 9, 12, 15, 20)}
        chance = {k: round(float(S.shape[1] * binom.sf(k - 1, len(bat), p0)), 2) for k in (3, 6, 9, 12, 15, 20)}
        print(name, "units:", S.shape[1], "present in >=k Batman images:", dist, "chance:", chance)
        cand = np.where(nb >= MIN_BAT)[0]
        cand = cand[np.lexsort((-strength[cand], -nb[cand]))]
        out[name] = {"n_units": int(S.shape[1]), "present_counts": dist, "chance": chance, "n_memes": int(len(cand)),
                     "memes": [{"unit": int(u), "n_batman": int(nb[u]), "n_cousin": int(P[cou, u].sum()), "strength": round(float(strength[u]), 3),
                                "present": [int(iw[i]) for i in np.where(P[:, u])[0]],
                                "top_works": [int(iw[i]) for i in np.argsort(-S[:, u])[:8]]} for u in cand[:40]],
                     "presence_by_year": None}
        # contact sheets for the top candidates (analysis only)
        sheet_dir = Path("data/vision_sheets") / name
        sheet_dir.mkdir(parents=True, exist_ok=True)
        imeta = json.load(open("data/images.json"))
        for m in out[name]["memes"][:24]:
            thumbs = []
            for wid in m["top_works"][:8]:
                im = Image.open(imeta[str(wid)]["file"]).convert("RGB")
                im.thumbnail((220, 220))
                tile = Image.new("RGB", (230, 250), "white")
                tile.paste(im, ((230 - im.width) // 2, 5))
                thumbs.append(tile)
            sheet = Image.new("RGB", (230 * 4, 250 * 2), "white")
            for k, t in enumerate(thumbs):
                sheet.paste(t, ((k % 4) * 230, (k // 4) * 250))
            sheet.save(sheet_dir / f"{m['unit']}.jpg", quality=80)
            m["sheet"] = str(sheet_dir / f"{m['unit']}.jpg")
            m["top_titles"] = [W[w]["label"] for w in m["top_works"]]
        out[name]["_P"] = P
        out[name]["_S"] = S

    # twins: CLIP memes -> DINOv2 units, over all views
    def zs(X):
        return (X - X.mean(0)) / (X.std(0) + 1e-6)
    Zd = zs(vd)
    n = len(vd)
    clip_units = [m["unit"] for m in out["clip"]["memes"]]
    active = np.where(out["clip"]["_P"].sum(0) >= 3)[0]
    rand_units = RNG.choice(np.setdiff1d(active, clip_units), size=min(len(clip_units), 40), replace=False)

    def best(units, shuffle=False):
        Zq = zs(vc[:, units])
        if shuffle:
            Zq = Zq[RNG.permutation(n)]
        R = Zq.T @ Zd / n
        return R.max(1), R.argmax(1)

    r_m, u_m = best(clip_units)
    r_r, _ = best(rand_units)
    r_s, _ = best(clip_units, shuffle=True)
    Pd = out["dino"]["_P"]
    Pc = out["clip"]["_P"]
    for k, m in enumerate(out["clip"]["memes"]):
        d = int(u_m[k])
        a, b = Pc[:, m["unit"]], Pd[:, d]
        m["dino_twin"] = {"unit": d, "r": round(float(r_m[k]), 3), "layer": int(d // md["info"]["hidden"]),
                          "presence_agree": round(float((a == b).mean()), 3),
                          "presence_jaccard": round(float((a & b).sum() / max((a | b).sum(), 1)), 3)}
    out["twins"] = {"memes_median": float(np.median(r_m)), "random_median": float(np.median(r_r)), "shuffled_median": float(np.median(r_s))}
    print("twins CLIP->DINOv2: memes", round(out["twins"]["memes_median"], 3), "random", round(out["twins"]["random_median"], 3), "shuffled", round(out["twins"]["shuffled_median"], 3))

    # probes
    probes = sorted(Path("data/probes").glob("*.jpg"))
    items = [(p.stem, str(p)) for p in probes]
    pc, _ = vx.run_clip(items)
    pdn, _ = vx.run_dino(items)
    pc = pc.astype(np.float32).reshape(len(items), 10, -1).mean(1)
    pdn = pdn.astype(np.float32).reshape(len(items), 10, -1).mean(1)
    out["probes"] = {}
    for name, PS, S in (("clip", pc, out["clip"]["_S"]), ("dino", pdn, out["dino"]["_S"])):
        units = [m["unit"] for m in out[name]["memes"][:24]]
        out["probes"][name] = {stem: {str(u): round(float((S[:, u] < PS[i, u]).mean() * 100), 1) for u in units} for i, (stem, _) in enumerate(items)}
    for name in ("clip", "dino"):
        del out[name]["_P"], out[name]["_S"]
    json.dump(out, open("results/vision.json", "w"), indent=1)
    print("wrote results/vision.json")


if __name__ == "__main__":
    main()
