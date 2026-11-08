"""The GPU side of the two new corpora (v2/second/PLAN.md, Sections 5, 6 and 10). Run from study/ with RUNPOD_STATE set
to a pod's state file on which `pip install transformers==5.22.2 accelerate hf_transfer` has been run.

    ../.venv/bin/python v2/second/gpu.py texts              the files a pod reads -> v2/cache/second/gpu/, hashes -> results/v2/second/gpu_inputs.json
    ../.venv/bin/python v2/second/gpu.py check <tag>        NOT new texts: presence_pass.py must reproduce v2's stored presence of the Batman works
    ../.venv/bin/python v2/second/gpu.py bgdrop <tag>       NOT new texts: the background plots left out of Holmes's base rates, by background_pass.py
    ../.venv/bin/python v2/second/gpu.py bgfull <tag>       only if bgdrop is not exact: the base rates recounted over the whole background
    ../.venv/bin/python v2/second/gpu.py holmes <tag> [primary]     the Holmes texts, original and name-free
    ../.venv/bin/python v2/second/gpu.py probes <tag> <job> [...]   recognition questions (holmes_h1 holmes_h2 remakes_r1 remakes_r2)
    ../.venv/bin/python v2/second/gpu.py remakes <tag> [primary] [small]    the remake films, and the co-presence counts
    ../.venv/bin/python v2/second/gpu.py embed              the sentence-embedding comparator on the remake films (any pod; not a study model)

Every step after `check` and `bgdrop` refuses to run unless the plan, the corpora and the code are committed and pushed
and the input files match the hashes in results/v2/second/gpu_inputs.json. Outputs land under v2/acts_second/ (never under
v2/acts/, which holds the Batman study's files). A step whose output is already here is not run again.
"""
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "v2/gpu")
sys.path.insert(0, "v2/second")
import runpod_ctl as rc  # noqa: E402

GPU = Path("v2/cache/second/gpu")
ACTS = Path("v2/acts_second")
RECORD = Path("results/v2/second/gpu_inputs.json")
OUT, DATA, CODE = "/workspace/out", "/workspace/data", "/workspace/code"
Q = "2>&1 | grep --line-buffered -v 'Loading weights\\|Fetching\\|Warning'"
SEED = 20261108
INDIAN = {"Bollywood", "Tamil", "Telugu", "Malayalam", "Kannada", "Bengali", "Marathi", "Punjabi", "Assamese"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def cut_length(sentences, max_s=20, max_w=400):
    """study_pass.cut_length: the opening the length control keeps."""
    words = k = 0
    for s in sentences[:max_s]:
        n = len(s.split())
        if k and words + n > max_w:
            break
        words, k = words + n, k + 1
    return k


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def texts():
    GPU.mkdir(parents=True, exist_ok=True)
    rec = {"what": "The files the GPU machines read for the two new corpora, by SHA-256. gpu.py refuses to run a step on a file that does not match.", "files": {}}
    # ---- Batman, for the check of the pipeline (not new texts)
    icon = re.compile(re.search(r'ICON = re\.compile\(r"(.*?)", re\.I\)', Path("v2/gpu/study_pass.py").read_text()).group(1), re.I)
    orig = [json.loads(l) for l in open("data/corpus.jsonl")]
    anon = [json.loads(l) for l in open("data/corpus_neutral.jsonl")]
    cuts = [cut_length(w["sentences"]) for w in orig]
    for name, works in (("batman_orig", orig), ("batman_anon", anon)):
        write(GPU / f"{name}.jsonl", [{"id": w["id"], "sentences": w["sentences"], "cut": c, "drop": [i for i, s in enumerate(w["sentences"]) if icon.search(s)]}
                                      for w, c in zip(works, cuts)])
    # ---- Holmes
    H = Path("data/second/holmes")
    spec = json.load(open(H / "spec.json"))
    icon = re.compile(spec["icon_regex"], re.I)
    orig = [json.loads(l) for l in open(H / "corpus.jsonl")]
    anon = [json.loads(l) for l in open(H / "corpus_neutral.jsonl")]
    assert [w["id"] for w in orig] == [w["id"] for w in anon] == list(range(len(orig)))
    assert all(len(a["sentences"]) == len(o["sentences"]) for a, o in zip(anon, orig))
    cuts = [cut_length(w["sentences"]) for w in orig]
    for name, works in (("holmes_orig", orig), ("holmes_anon", anon)):
        write(GPU / f"{name}.jsonl", [{"id": w["id"], "sentences": w["sentences"], "cut": c, "drop": [i for i, s in enumerate(w["sentences"]) if icon.search(s)]}
                                      for w, c in zip(works, cuts)])
    drop = set(spec["background_documents_left_out"])
    docs = []
    for l in open("v2/cache/reference.jsonl"):
        d = json.loads(l)
        if d["src"] != "bgplot":
            break
        if d["i"] in drop:
            docs.append(d)
    assert len(docs) == len(drop)
    write(GPU / "holmes_bgdrop.jsonl", docs)
    write(GPU / "background_plots.jsonl", [json.loads(l) for l in open("v2/cache/reference.jsonl") if '"src": "bgplot"' in l[:40]])
    # ---- remakes
    films_path, pairs_path = Path("v2/cache/second/remakes/films.jsonl"), Path("data/second/remakes/pairs.json")
    if films_path.exists() and pairs_path.exists():
        films = [json.loads(l) for l in open(films_path)]
        write(GPU / "remakes_films.jsonl", [{"id": f["id"], "sentences": f["sentences"]} for f in films])
        P = json.load(open(pairs_path))["pairs"]
        row = {f["page"]: f["id"] for f in films}
        remakes = sorted({row[p["remake"]] for p in P})
        rows = remakes                                    # the rows of the co-presence counts: every remake of a kept pair
        json.dump(rows, open(GPU / "remakes_rows.json", "w"))
        pair_films = sorted(f["id"] for f in films if f["in_pair"])
        json.dump(pair_films, open(GPU / "remakes_pair_films.json", "w"))
        Path("data/second/remakes").mkdir(parents=True, exist_ok=True)
        json.dump({"what": "Rows of the co-presence counts: every remake of a kept pair. pair_films: every film of a kept pair (their presence bits "
                           "are fetched so that the counts can be recomputed away from the GPU machine). Numbers are ids of films_index.json.",
                   "rows": rows, "pair_films": pair_films}, open("data/second/remakes/rows.json", "w"))
        print("remakes:", len(films), "films;", len(remakes), "remakes (rows);", len(pair_films), "films in a pair")
        # the smaller set the 70B model reads: the Indian industries' films and every film of a pair, in the same order
        indian = [f for f in films if f["origin"] in INDIAN or f["in_pair"]]
        write(GPU / "remakes_films_small.jsonl", [{"id": f["id"], "sentences": f["sentences"]} for f in indian])
        at = {f["id"]: k for k, f in enumerate(indian)}
        json.dump([at[i] for i in rows], open(GPU / "remakes_rows_small.json", "w"))
        json.dump([at[i] for i in pair_films], open(GPU / "remakes_pair_films_small.json", "w"))
    for f in sorted(GPU.glob("*.json*")):
        rec["files"][f.name] = sha(f)
    for f in sorted(Path("v2/cache/second/jobs").glob("*.json")):
        rec["files"]["jobs/" + f.name] = sha(f)
    models = ("qwen3-32b", "qwen3-8b", "llama3.1-8b", "llama3.3-70b")
    rec["scale_ref_sha256"] = {t: sha(f"v2/acts/{t}/scale_ref.npy") for t in models}
    rec["bg_counts_ref_scale_sha256"] = {t: sha(f"v2/acts/{t}/bg_counts_ref_scale.npy") for t in models}        # v2's stored counts, which set Holmes's base rates
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(RECORD, "w"), indent=1)
    print("wrote", RECORD, "with", len(rec["files"]), "files")


FROZEN_PATHS = ["v2/second", "v2/gpu", "v2/analysis", "v2/prompts", "experiments.py", "discover.py", "fetch_corpus.py", "data/second",
                "results/v2/second/gpu_inputs.json", "results/v2/second/holmes_words.json", "results/v2/second/holmes_e3_power.json",
                "results/v2/second/remakes_words.json", "../preregistration/second_corpus"]


def freeze_state():
    """(None, commit) when the plan says it is frozen and the plan, the code, the corpora and the words-only results it cites are the
    committed and pushed ones; else (the reason, commit)."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *FROZEN_PATHS)
    if dirty:
        return "uncommitted changes in the plan, the code, the corpora or the words-only results: commit them first\n" + dirty, head
    if head != git("rev-parse", "@{u}"):
        return "this commit is not pushed: push it first", head
    if "**Status: frozen" not in Path("v2/second/PLAN.md").read_text():
        return "the plan does not say it is frozen", head
    return None, head


def frozen():
    """The plan, the corpora and the code must be the committed and pushed ones."""
    why, head = freeze_state()
    if why:
        raise SystemExit(why)
    print("commit", head, "pushed; plan frozen", flush=True)
    return head


def recorded(*names):
    rec = json.load(open(RECORD))["files"]
    for n in names:
        path = Path("v2/cache/second/jobs") / n[5:] if n.startswith("jobs/") else GPU / n
        assert sha(path) == rec[n], f"{n} is not the file that was recorded"


def go(name, cmd):
    print(rc.run(name, f"set -o pipefail; cd {CODE} && {cmd}"), flush=True)
    code = rc.wait(name, poll=10, max_print=2500)
    if code != 0:
        raise SystemExit(f"{name} exited with {code}")


def sync():
    subprocess.run([sys.executable, "v2/gpu/remote.py", "sync", "-"], check=True)


def put_scales(tag, names=("scale_ref.npy",)):
    for n in names:
        rc.put(f"v2/acts/{tag}/{n}", f"out/{tag}/study/{n}")


def fetch(remote, local, gz=False):
    """Download one file; with gz it is compressed on the pod first and unpacked here (presence bits are mostly zeros)."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        go("gz_" + hashlib.sha1(remote.encode()).hexdigest()[:10], f"gzip -kf -1 /workspace/{remote} && ls -la /workspace/{remote}.gz | awk '{{print $5}}'")
        tmp = local.with_name(local.name + ".gz")
        rc.get(remote + ".gz", tmp, chunk=32 << 20)
        with gzip.open(tmp, "rb") as a, open(local, "wb") as b:
            shutil.copyfileobj(a, b, 1 << 24)
        tmp.unlink()
    else:
        rc.get(remote, local, chunk=32 << 20)


def planes_of(which, cut_icon):
    names = [f"{s}_{t}" for s in ("top1", "top3") for t in (20, 30, 40, 50)] if which == "all" else ["top1_30"]
    return names + (["cut_top1_30", "icon_top1_30"] if cut_icon else []) + ["over100"]


def check(tag):
    """presence_pass.py on the Batman works at v2's main scale must give exactly the presence v2 stored (uint8 share >= 60)."""
    sync()
    put_scales(tag, ("scale.npy",))
    res = {"model": tag, "what": "presence_pass.py against v2's stored presence of the 114 Batman-study works (strongest sentence at 20, 30, 40 and 50% of v2's main "
                                 "scale; the opening and the iconography-dropped text at 30%)", "planes": {}}
    for ver, flag in (("orig", ""), ("anon", "--mask-codes")):
        rc.put(GPU / f"batman_{ver}.jsonl", f"data/second/batman_{ver}.jsonl")
        go(f"check_{ver}_{tag}".replace(".", "_"), f"python3 presence_pass.py {tag} {DATA}/second/batman_{ver}.jsonl {OUT}/{tag}/second/batman_{ver} "
                                                    f"--scale {OUT}/{tag}/study/scale.npy --planes all {flag} {Q}")
        compare = [(f"top1_{t}", f"{ver}_main_top1", 2 * t) for t in (20, 30, 40, 50)] + [("cut_top1_30", f"{ver}_cut_main_top1", 60)]
        compare += [("icon_top1_30", "anon_icon_main_top1", 60)] if ver == "anon" else []
        for plane, stored, cut in compare:
            local = ACTS / "batman_check" / tag / ver / f"{plane}.bits.npy"
            fetch(f"out/{tag}/second/batman_{ver}/{plane}.bits.npy", local, gz=True)
            new = np.load(local, mmap_mode="r")
            old = np.load(f"v2/acts/{tag}/{stored}.u8.npy", mmap_mode="r")
            diff = total = 0
            for w in range(old.shape[0]):
                a = np.unpackbits(new[w])[: old.shape[1]].astype(bool)
                b = old[w] >= cut                               # v2's stored share, in steps of 0.005: 60 steps is 30% of scale
                diff += int((a != b).sum())
                total += int(b.sum())
            res["planes"][f"{ver}_{plane}"] = {"units_present_in_v2": total, "bits_that_differ": diff}
            print(f"  {ver} {plane}: {diff} bits differ of {total:,} present", flush=True)
            local.unlink()
        fetch(f"out/{tag}/second/batman_{ver}/meta.json", ACTS / "batman_check" / tag / ver / "meta.json")
    res["provenance"] = json.load(open(ACTS / "batman_check" / tag / "orig" / "meta.json"))["provenance"]
    res["exact"] = all(v["bits_that_differ"] == 0 for v in res["planes"].values())
    why, head = freeze_state()
    res["after_the_freeze"], res["commit"] = why is None, head          # the check is repeated on every machine once the plan is frozen
    res["checked_utc"] = __import__("time").strftime("%Y-%m-%dT%H:%MZ", __import__("time").gmtime())
    Path("results/v2/second").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(f"results/v2/second/pipeline_check_{tag}.json", "w"), indent=1)
    print("exact:", res["exact"])


def bgdrop(tag):
    """The 34 background plots that are older summaries of Holmes-corpus films, read by v2's own background_pass.py: their counts
    on the reference-only scale (to subtract), and their presence at v2's main scale (must equal what v2 stored)."""
    sync()
    put_scales(tag, ("scale.npy", "scale_ref.npy", "scale_ref5.npy", "scale_raw.npy"))
    rc.put(GPU / "holmes_bgdrop.jsonl", "data/second/holmes_bgdrop.jsonl")
    go(f"bgdrop_{tag}".replace(".", "_"), f"python3 background_pass.py {tag} {DATA}/second/holmes_bgdrop.jsonl {OUT} --name second_bgdrop {Q}")
    d = ACTS / "holmes" / tag / "bgdrop"
    for f in ("counts_ref_scale.npy", "counts.npy", "meta.json"):
        fetch(f"out/{tag}/second_bgdrop/{f}", d / f)
    fetch(f"out/{tag}/second_bgdrop/present30.bits.npy", d / "present30.bits.npy", gz=True)
    meta = json.load(open(d / "meta.json"))
    stored_meta = json.load(open(f"v2/acts/{tag}/bg_meta.json"))
    where = {doc: r for r, doc in enumerate(stored_meta["doc_index"])}
    stored = np.load(f"v2/acts/{tag}/bg_present30.bits.npy", mmap_mode="r")
    new = np.load(d / "present30.bits.npy", mmap_mode="r")
    diff = sum(int((np.unpackbits(new[r]) != np.unpackbits(stored[where[doc]])).sum()) for r, doc in enumerate(meta["doc_index"]))
    res = {"model": tag, "documents": meta["doc_index"], "bits_that_differ_from_v2s_stored_presence": diff, "exact": diff == 0, "provenance": meta["provenance"]}
    json.dump(res, open(f"results/v2/second/bgdrop_check_{tag}.json", "w"), indent=1)
    (d / "present30.bits.npy").unlink()
    print(res)


def bgfull(tag):
    """PLAN Section 6's fallback: if the 34 left-out plots' stored presence is not reproduced exactly on this machine, the base rates
    are recounted over all 2,965 background plots by v2's own background_pass.py. Use with: holmes.py run <tag> --recounted --stored <file>."""
    frozen()
    recorded("background_plots.jsonl")
    sync()
    put_scales(tag, ("scale.npy", "scale_ref.npy", "scale_ref5.npy", "scale_raw.npy"))
    spec = json.load(open("data/second/holmes/spec.json"))
    rc.put(GPU / "background_plots.jsonl", "data/second/background_plots.jsonl")
    out_docs = " ".join(str(i) for i in [843] + spec["background_documents_left_out"])
    go(f"bgfull_{tag}".replace(".", "_"), f"python3 background_pass.py {tag} {DATA}/second/background_plots.jsonl {OUT} --name second_bgfull --no-bits --exclude-docs {out_docs} {Q}")
    d = ACTS / "holmes" / tag / "bgfull"
    for f in ("counts_ref_scale.npy", "meta.json"):
        fetch(f"out/{tag}/second_bgfull/{f}", d / ("bg_counts_ref_scale.npy" if f.startswith("counts") else "bg_meta.json"))
    shutil.copy(f"v2/acts/{tag}/scale_ref.npy", d / "scale_ref.npy")
    print("  full recount fetched to", d, flush=True)


def holmes(tag, which="all"):
    frozen()
    recorded("holmes_orig.jsonl", "holmes_anon.jsonl")
    assert sha(f"v2/acts/{tag}/scale_ref.npy") == json.load(open(RECORD))["scale_ref_sha256"][tag]
    sync()
    put_scales(tag)
    for ver, flag in (("orig", ""), ("anon", "--mask-codes")):
        d = ACTS / "holmes" / tag / ver
        if (d / "meta.json").exists():
            continue
        rc.put(GPU / f"holmes_{ver}.jsonl", f"data/second/holmes_{ver}.jsonl")
        go(f"holmes_{ver}_{tag}".replace(".", "_"), f"python3 presence_pass.py {tag} {DATA}/second/holmes_{ver}.jsonl {OUT}/{tag}/second/holmes_{ver} "
                                                     f"--scale {OUT}/{tag}/study/scale_ref.npy --planes {which} {flag} {Q}")
        keep = planes_of(which, True) if ver == "anon" else ["top1_30", "cut_top1_30", "over100"]
        for p in keep:
            fetch(f"out/{tag}/second/holmes_{ver}/{p}.bits.npy", d / f"{p}.bits.npy", gz=True)
        fetch(f"out/{tag}/second/holmes_{ver}/meta.json", d / "meta.json")
        print(f"  {tag} {ver}: fetched {keep}", flush=True)


def probes(tag, jobs):
    frozen()
    recorded(*[f"jobs/{j}.json" for j in jobs])
    sync()
    for j in jobs:
        local = ACTS / "probes" / tag / f"{j}.json"
        if local.exists():
            continue
        rc.put(f"v2/cache/second/jobs/{j}.json", f"data/second/jobs/{j}.json")
        print(rc.run(f"ask_{j}_{tag}".replace(".", "_"), f"set -o pipefail; cd {CODE} && python3 ablate.py {tag} {DATA}/second/jobs/{j}.json {OUT}/{tag}/second/{j}.json {Q} | tail -2"))
        code = rc.wait(f"ask_{j}_{tag}".replace(".", "_"), poll=15, quiet=True)             # replies are not printed here: they are read from the file
        if code != 0:
            raise SystemExit(f"{j} exited with {code}")
        fetch(f"out/{tag}/second/{j}.json", local)
        print(f"  {tag} {j}: fetched", flush=True)


def remakes(tag, which="all", small=False):
    """small: the 70B model's set (the Indian industries' films and the pair films)."""
    frozen()
    films_file, rows_file, pair_file = (("remakes_films_small.jsonl", "remakes_rows_small.json", "remakes_pair_films_small.json") if small
                                        else ("remakes_films.jsonl", "remakes_rows.json", "remakes_pair_films.json"))
    recorded(films_file, rows_file, pair_file)
    assert sha(f"v2/acts/{tag}/scale_ref.npy") == json.load(open(RECORD))["scale_ref_sha256"][tag]
    d = ACTS / "remakes" / tag
    if (d / "meta.json").exists():
        return
    sync()
    put_scales(tag)
    rc.put(GPU / films_file, "data/second/remakes_films.jsonl")
    rc.put(GPU / rows_file, "data/second/remakes_rows.json")
    planes = [f"{s}_{t}" for s in ("top1", "top3") for t in (20, 30, 40, 50)] if which == "all" else ["top1_30"]
    go(f"remakes_{tag}".replace(".", "_"), f"python3 presence_pass.py {tag} {DATA}/second/remakes_films.jsonl {OUT}/{tag}/second/remakes "
                                            f"--scale {OUT}/{tag}/study/scale_ref.npy --planes {which} {Q}")
    go(f"copresence_{tag}".replace(".", "_"), f"python3 copresence.py {OUT}/{tag}/second/remakes {DATA}/second/remakes_rows.json {OUT}/{tag}/second/remakes_counts "
                                               f"--planes-list {' '.join(planes)}")
    # a second count at the primary setting without the units that are above their scale in any film read (PLAN Section 9)
    go(f"copresence_over_{tag}".replace(".", "_"), f"python3 copresence.py {OUT}/{tag}/second/remakes {DATA}/second/remakes_rows.json {OUT}/{tag}/second/remakes_counts "
                                                    f"--planes-list top1_30 --without over100")
    for p in planes + ["top1_30_without_over100"]:
        fetch(f"out/{tag}/second/remakes_counts/{p}.npz", d / "counts" / f"{p}.npz")
    # for an independent recount here: how many films each unit is in, and the presence bits of every film of a pair
    fetch(f"out/{tag}/second/remakes_counts/top1_30.unit_counts.npy", d / "top1_30.unit_counts.npy", gz=True)
    rc.put(GPU / pair_file, "data/second/remakes_pair_films.json")
    go(f"pairbits_{tag}".replace(".", "_"), "python3 -c \"import numpy as np, json; b = np.load('" + f"{OUT}/{tag}/second/remakes/top1_30.bits.npy" + "', mmap_mode='r'); "
       "r = json.load(open('" + f"{DATA}/second/remakes_pair_films.json" + "')); np.save('" + f"{OUT}/{tag}/second/remakes_counts/pair_films_top1_30.bits.npy" + "', np.ascontiguousarray(b[r]))\"")
    fetch(f"out/{tag}/second/remakes_counts/pair_films_top1_30.bits.npy", d / "pair_films_top1_30.bits.npy", gz=True)
    fetch(f"out/{tag}/second/remakes/meta.json", d / "meta.json")
    print(f"  {tag}: counts for {planes} fetched", flush=True)


def embed():
    """The embedding comparator of PLAN.md Section 9: one vector per film read, from v2/gpu/embed_pass.py."""
    frozen()
    recorded("remakes_films.jsonl")
    local = ACTS / "remakes" / "embedding.npy"
    if local.exists():
        return
    sync()
    rc.put(GPU / "remakes_films.jsonl", "data/second/remakes_films.jsonl")
    go("embed", f"python3 embed_pass.py {DATA}/second/remakes_films.jsonl {OUT}/second_embedding.npy {Q}")
    fetch("out/second_embedding.npy", local)
    fetch("out/second_embedding.npy.json", ACTS / "remakes" / "embedding.json")
    print("  embedding fetched", flush=True)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "texts":
        texts()
    elif cmd == "check":
        check(a[0])
    elif cmd == "bgdrop":
        bgdrop(a[0])
    elif cmd == "bgfull":
        bgfull(a[0])
    elif cmd == "holmes":
        holmes(a[0], "primary" if "primary" in a else "all")
    elif cmd == "probes":
        probes(a[0], a[1:])
    elif cmd == "remakes":
        remakes(a[0], "primary" if "primary" in a else "all", small="small" in a)
    elif cmd == "embed":
        embed()
    else:
        raise SystemExit(__doc__)
