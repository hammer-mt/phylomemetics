"""Run one step on the GPU machine and fetch what it wrote (run from study/).

    ../.venv/bin/python v2/gpu/remote.py <step> <model-tag> [args]        with RUNPOD_STATE pointing at the pod's state file

    sentences <tag> <units.npy> [prefix]     sentence-level activations of chosen units          -> v2/acts/<tag>/<prefix>_*
    top <tag> <units.npy>                    their strongest reference examples, whole documents  -> v2/acts/<tag>/top_examples.json
    score <tag> <texts.jsonl> <units.npy> <name>      any texts on chosen units                   -> v2/acts/<tag>/<name>.npz (+ tokens)
    ask <tag> <jobs.json> <name> [scale]     questions to the model, with or without units off    -> v2/acts/<tag>/<name>.json
    steer <tag> <units.npy> <name> <multiples...>     continuations with a unit held active       -> v2/acts/<tag>/<name>.json
    twins <tag> <queries.npz>                each query unit's twin in this model                 -> v2/acts/<tag>/twins.json
    background <tag> <units.npy>             per-plot shares of chosen units (--select)           -> v2/acts/<tag>/bg_selected.*
    sync - [skip ...]                        upload every GPU script that differs from the pod's copy (not the ones named)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

Q = "2>&1 | grep --line-buffered -v 'Loading weights\\|Fetching\\|Warning'"
OUT, DATA, CODE = "/workspace/out", "/workspace/data", "/workspace/code"


def up(local, sub="jobs"):
    remote = f"data/{sub}/{Path(local).name}"
    rc.put(local, remote)
    return "/workspace/" + remote


def go(name, cmd):
    tag = sys.argv[2]
    print(rc.run(name, f"set -o pipefail; mkdir -p {OUT}/{tag} && cd {CODE} && {cmd}"), flush=True)
    code = rc.wait(name, poll=10, max_print=3000)
    if code != 0:
        raise SystemExit(f"{name} exited with {code}")


def pull(tag, *files):
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "pull.py"), tag, "--only", *files], check=True)


def sync(skip=()):
    here = Path(__file__).resolve().parent
    files = [f for f in sorted(here.glob("*.py")) + sorted(here.glob("*.sh")) if f.name not in ("runpod_ctl.py", "pull.py", "remote.py", *skip)]
    have = rc.remote_sha256([f"code/{f.name}" for f in files])
    for f in files:
        if have.get(f"code/{f.name}") != rc.sha256_file(f):
            rc.put(f, f"code/{f.name}")
            print("uploaded", f.name, flush=True)


def main():
    step, tag, a = sys.argv[1], sys.argv[2], sys.argv[3:]
    if step == "sync":
        sync(a)
    elif step == "sentences":
        prefix = a[1] if len(a) > 1 else "sent"
        go(f"sent_{tag}_{prefix}".replace(".", "_"), f"python3 study_sentences.py {tag} {DATA}/study {OUT} {up(a[0])} {prefix} {Q}")
        pull(tag, f"--sentences={prefix}")
    elif step == "top":
        go(f"top_{tag}".replace(".", "_"), f"python3 top_examples.py {tag} {DATA}/reference.jsonl,{DATA}/supplement.jsonl {OUT} {up(a[0])} {OUT}/{tag}/top_examples.json --k 24 {Q}")
        pull(tag, "--file=top_examples.json")
    elif step == "score":
        name = a[2]
        go(f"score_{tag}_{name}".replace(".", "_"), f"python3 score_texts.py {tag} {up(a[0])} {up(a[1])} {OUT}/{tag}/{name}.npz {Q}")
        pull(tag, f"--file={name}.npz", f"--file={name}.tokens.json")
    elif step == "ask":
        name = a[1]
        scale = f"{OUT}/{tag}/study/scale.npy" if len(a) > 2 else ""
        go(f"ask_{tag}_{name}".replace(".", "_"), f"python3 ablate.py {tag} {up(a[0])} {OUT}/{tag}/{name}.json {scale} {Q}")
        pull(tag, f"--file={name}.json")
    elif step == "steer":
        name = a[1]
        go(f"steer_{tag}_{name}".replace(".", "_"), f"python3 steer.py {tag} {OUT} {up(a[0])} {OUT}/{tag}/{name}.json --multiples {' '.join(a[2:])} {Q}")
        pull(tag, f"--file={name}.json")
    elif step == "twins":
        go(f"twins_{tag}".replace(".", "_"), f"python3 twin_search.py {tag} {DATA}/study {OUT} {up(a[0])} {OUT}/{tag}/twins.json {Q}")
        pull(tag, "--file=twins.json")
    elif step == "background":
        go(f"bgsel_{tag}".replace(".", "_"), f"python3 background_pass.py {tag} {DATA}/reference.jsonl {OUT} --exclude-docs 843 --select {up(a[0])} --name background_sel {Q}")
        pull(tag, "--file=background_sel/selected.u8.npy", "--file=background_sel/selected_units.npy")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
