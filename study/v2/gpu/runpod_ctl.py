"""Drive a Runpod GPU pod from this machine: create it, push code, run jobs, pull results, terminate it.

All traffic is HTTPS: the Runpod REST API (v2) for the pod's lifecycle and spend, and the pod's own job runner
(pod_agent.py) through Runpod's HTTP proxy. The proxy cuts requests off after 100 seconds, so jobs are started in
the background and polled, and files move in chunks.

The Runpod API key is read from RUNPOD_API_KEY. (In a Claude Code cloud session the key is injected by the
session's network proxy instead, so the placeholder below is sent and replaced on the way out.)

State (pod id, URL and the job runner's token) is kept in STATE, which is not committed.

    python gpu/runpod_ctl.py gpus                       # GPU types with at least 80 GB, with prices
    python gpu/runpod_ctl.py create "NVIDIA A100-SXM4-80GB" --volume 300
    python gpu/runpod_ctl.py put gpu/neurons.py code/neurons.py
    python gpu/runpod_ctl.py run smoke "python code/smoke.py"
    python gpu/runpod_ctl.py wait smoke
    python gpu/runpod_ctl.py get out/x.npy results/x.npy
    python gpu/runpod_ctl.py spend
    python gpu/runpod_ctl.py terminate
"""
import argparse
import base64
import hashlib
import shlex
import json
import os
import secrets
import sys
import time
from pathlib import Path

import requests

API = "https://api.runpod.io/v2"
HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get("RUNPOD_STATE", HERE / ".state" / "pod.json"))
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CHUNK = 16 << 20


def _headers():
    return {"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', 'proxy-injected')}",
            "Content-Type": "application/json"}


def api(method, path, body=None, params=None, ok=(200, 201, 202, 204)):
    for attempt in range(5):
        r = requests.request(method, API + path, headers=_headers(), json=body, params=params, timeout=60)
        if r.status_code in ok:
            return r.json() if r.content else {}
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:600]}")
    raise RuntimeError(f"{method} {path} kept failing: {r.status_code} {r.text[:300]}")


def load_state():
    return json.loads(STATE.read_text())


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


# ------------------------------------------------------------------ lifecycle
def gpus(min_mem=80):
    rows = api("GET", "/catalog/gpus", params={"include": "AVAILABILITY", "product": "POD"})["gpus"]
    out = [g for g in rows if g["memory"] >= min_mem]
    return sorted(out, key=lambda g: (g["price"].get("secure") or 99))


def create(gpu_id, count=1, cloud="SECURE", volume=300, disk=40, max_hours=8, idle_min=40, name="phylomemetics-v2",
           stop_action="terminate", datacenters=None, min_cuda=None):
    """Create a pod whose only process is the job runner. It terminates itself when idle or past its deadline."""
    token = secrets.token_urlsafe(32)
    agent_b64 = base64.b64encode((HERE / "pod_agent.py").read_bytes()).decode()
    # PID 1: write the agent, keep it alive, and keep a last-resort deadline in plain bash in case the agent dies.
    boot = ("mkdir -p /workspace/logs && echo $POD_AGENT_B64 | base64 -d > /workspace/pod_agent.py && "
            "(sleep $(( ${POD_AGENT_MAX_SECONDS} + 900 )); curl -s -X DELETE -H \"Authorization: Bearer $RUNPOD_API_KEY\" "
            "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID) & "
            "while true; do python3 /workspace/pod_agent.py >> /workspace/logs/agent.log 2>&1; sleep 3; done")
    body = {"name": name, "image": IMAGE, "cloud": cloud, "gpu": {"id": gpu_id, "count": count},
            "disk": disk, "mounts": {"persistent": {"size": volume, "path": "/workspace"}},
            "ports": ["8000/http"], "entrypoint": ["bash", "-c"], "cmd": [boot],
            "env": {"POD_AGENT_TOKEN": token, "POD_AGENT_B64": agent_b64, "POD_AGENT_MAX_HOURS": str(max_hours),
                    "POD_AGENT_MAX_SECONDS": str(int(max_hours * 3600)), "POD_AGENT_IDLE_MIN": str(idle_min),
                    "POD_AGENT_STOP_ACTION": stop_action, "HF_HOME": "/workspace/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}}
    if datacenters:
        body["dataCenterIds"] = datacenters
    if min_cuda:
        body["gpu"]["minCudaVersion"] = min_cuda
    pod = api("POST", "/pods", body)
    state = {"id": pod["id"], "token": token, "url": f"https://{pod['id']}-8000.proxy.runpod.net", "gpu": gpu_id,
             "count": count, "cloud": cloud, "created": time.time(), "create_response": pod}
    save_state(state)
    return state


def pod_info():
    return api("GET", f"/pods/{load_state()['id']}")


def terminate():
    s = load_state()
    api("DELETE", f"/pods/{s['id']}", ok=(200, 202, 204, 404))
    s["terminated"] = time.time()
    save_state(s)
    return s


def action(name):
    """"stop" (keeps the volume, ends GPU billing) or "start" (boots the pod again with fresh timers)."""
    return api("POST", f"/pods/{load_state()['id']}/action", {"action": name})


def spend(days=30):
    """Total billed so far, from the billing API (lags real time by up to an hour)."""
    return api("GET", "/billing")["metadata"]["totals"]


# ------------------------------------------------------------------ the pod's job runner
def agent(method, path, params=None, data=None, timeout=90, raw=False):
    s = load_state()
    last = None
    for attempt in range(6):
        try:
            r = requests.request(method, s["url"] + path, params=params, data=data, timeout=timeout,
                                 headers={"X-Token": s["token"]})
            if r.status_code == 200:
                return r.content if raw else r.json()
            last = f"{r.status_code} {r.text[:200]}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"agent {method} {path} failed: {last}")


def wait_ready(timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return agent("GET", "/ping", timeout=20)
        except Exception as e:
            print(f"  not ready yet ({round(time.time() - t0)}s): {str(e)[:100]}", flush=True)
            time.sleep(15)
    raise TimeoutError("pod agent did not come up")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def remote_sha256(paths):
    """sha256 of files on the pod (paths relative to /workspace), computed there by a short job."""
    name = "sha_" + secrets.token_hex(4)
    listing = f"logs/{name}.txt"
    run(name, "cd /workspace && sha256sum " + " ".join(shlex.quote(p) for p in paths) + f" > {listing} 2>&1")
    wait(name, poll=3, quiet=True)
    text = agent("GET", "/file", params={"path": listing, "offset": 0, "length": 4 << 20}, raw=True).decode()
    out = {}
    for line in text.splitlines():
        digest, _, path = line.partition("  ")
        if len(digest) == 64:
            out[path.strip()] = digest
    return out


def put(local, remote, tries=3):
    """Upload a file and check its size and hash on the pod (a retried chunk would otherwise be appended twice)."""
    data = Path(local).read_bytes()
    want = hashlib.sha256(data).hexdigest()
    for _ in range(tries):
        for i in range(0, max(len(data), 1), CHUNK):
            r = agent("PUT", "/file", params={"path": remote, "append": "1" if i else "0"}, data=data[i:i + CHUNK])
        if r.get("size") == len(data) and remote_sha256([remote]).get(remote) == want:
            return len(data)
    raise RuntimeError(f"upload of {local} could not be verified")


def get(remote, local, chunk=CHUNK, sha=None):
    """Download a file. With `sha` (its sha256 on the pod) a local copy that already matches is kept, and the download is
    verified; without it the file is always fetched afresh. Nothing is ever resumed from an earlier call."""
    local = Path(local)
    if sha and local.exists() and sha256_file(local) == sha:
        return 0
    st = agent("GET", "/stat", params={"path": remote})
    if not st.get("exists"):
        raise FileNotFoundError(remote)
    size, have = st["size"], 0
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    with open(part, "wb") as f:
        while have < size:
            b = agent("GET", "/file", params={"path": remote, "offset": have, "length": chunk}, raw=True, timeout=95)
            if not b:
                raise RuntimeError(f"{remote}: the pod sent nothing at offset {have}")
            f.write(b)
            have += len(b)
    if sha and sha256_file(part) != sha:
        raise RuntimeError(f"{remote}: the downloaded file does not match the pod's hash (was it rewritten meanwhile?)")
    os.replace(part, local)
    return size


def run(name, cmd, env=None, cwd="/workspace"):
    return agent("POST", "/run", data=json.dumps({"name": name, "cmd": cmd, "env": env or {}, "cwd": cwd}))


def status(name=None):
    return agent("GET", "/status", params={"name": name} if name else None)


def log(name, offset=0, length=200000):
    return agent("GET", "/log", params={"name": name, "offset": offset, "length": length})


def wait(name, poll=20, timeout=6 * 3600, quiet=False, max_print=4000):
    """Follow a job's log until it exits; returns its exit code."""
    off, t0 = 0, time.time()
    while True:
        chunk = log(name, off)
        if chunk["text"] and not quiet:
            text = chunk["text"]
            print(text if len(text) <= max_print else text[:800] + "\n  ...\n" + text[-max_print + 800:], end="", flush=True)
        off = chunk["offset"] + len(chunk["text"].encode("utf-8", "replace"))
        st = status(name).get(name)
        if st and not st["running"] and off >= chunk["size"]:
            return st["exit"]
        if time.time() - t0 > timeout:
            raise TimeoutError(name)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    c = sub.add_parser("create")
    c.add_argument("gpu")
    c.add_argument("--count", type=int, default=1)
    c.add_argument("--cloud", default="SECURE")
    c.add_argument("--volume", type=int, default=300)
    c.add_argument("--max-hours", type=float, default=8)
    c.add_argument("--idle-min", type=float, default=40)
    for n in ("ping", "info", "terminate", "spend", "status"):
        sub.add_parser(n)
    p = sub.add_parser("put"); p.add_argument("local"); p.add_argument("remote")
    g = sub.add_parser("get"); g.add_argument("remote"); g.add_argument("local")
    r = sub.add_parser("run"); r.add_argument("name"); r.add_argument("command")
    w = sub.add_parser("wait"); w.add_argument("name")
    lg = sub.add_parser("log"); lg.add_argument("name"); lg.add_argument("--offset", type=int, default=-4000)
    a = ap.parse_args()
    if a.cmd == "gpus":
        for g in gpus():
            print(f"{g['memory']:4d} GB  secure ${g['price'].get('secure')}  community ${g['price'].get('community')}  {g['id']}  {g.get('availability', '')}")
    elif a.cmd == "create":
        s = create(a.gpu, a.count, a.cloud, a.volume, max_hours=a.max_hours, idle_min=a.idle_min)
        print("created", s["id"], s["url"])
        print(wait_ready())
    elif a.cmd == "ping":
        print(agent("GET", "/ping"))
    elif a.cmd == "info":
        print(json.dumps(pod_info(), indent=1))
    elif a.cmd == "terminate":
        print("terminated", terminate()["id"])
    elif a.cmd == "spend":
        print(json.dumps(spend(), indent=1))
    elif a.cmd == "status":
        print(status())
    elif a.cmd == "put":
        print(put(a.local, a.remote), "bytes")
    elif a.cmd == "get":
        print(get(a.remote, a.local), "bytes")
    elif a.cmd == "run":
        print(run(a.name, a.command))
    elif a.cmd == "wait":
        sys.exit(wait(a.name) or 0)
    elif a.cmd == "log":
        print(log(a.name, a.offset)["text"])


if __name__ == "__main__":
    main()
