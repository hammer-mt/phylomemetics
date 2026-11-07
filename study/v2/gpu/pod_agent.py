"""A tiny job runner that lives on the GPU pod.

The pod is reached only over Runpod's HTTPS proxy, which cuts every request off after 100 seconds, so nothing
here blocks: jobs run in the background and are polled, and files move in ranged chunks. Standard library only.

Every request must carry the X-Token header (the secret passed to the pod as POD_AGENT_TOKEN).

Cost guards, both enforced on the pod itself so they hold even if the controller disappears:
  * idle: no authenticated request and no running job for IDLE_MIN minutes -> the pod stops itself
  * deadline: after MAX_HOURS since start (pushed back by POST /extend) -> the pod stops itself
"Stops itself" means POST {"action": STOP_ACTION} to the Runpod API with the pod-scoped key Runpod injects.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ["POD_AGENT_TOKEN"]
ROOT = os.environ.get("POD_AGENT_ROOT", "/workspace")
IDLE_MIN = float(os.environ.get("POD_AGENT_IDLE_MIN", "40"))
MAX_HOURS = float(os.environ.get("POD_AGENT_MAX_HOURS", "8"))
STOP_ACTION = os.environ.get("POD_AGENT_STOP_ACTION", "terminate")
PORT = int(os.environ.get("POD_AGENT_PORT", "8000"))
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)

state = {"last_seen": time.time(), "deadline": time.time() + MAX_HOURS * 3600, "started": time.time()}
jobs = {}          # name -> Popen
lock = threading.Lock()


def running_jobs():
    return [n for n, p in jobs.items() if p.poll() is None]


def stop_pod(reason):
    print(f"[agent] stopping pod: {reason}", flush=True)
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        return
    if STOP_ACTION == "terminate":
        req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}", method="DELETE",
                                     headers={"Authorization": f"Bearer {key}"})
    else:
        req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}/action", method="POST",
                                     data=json.dumps({"action": "stop"}).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for _ in range(5):
        try:
            urllib.request.urlopen(req, timeout=30)
            return
        except Exception as e:      # keep trying: a pod that cannot stop itself keeps billing
            print("[agent] stop failed:", e, flush=True)
            time.sleep(20)


def watchdog():
    while True:
        time.sleep(30)
        now = time.time()
        if now > state["deadline"]:
            stop_pod("deadline reached")
        elif not running_jobs() and now - state["last_seen"] > IDLE_MIN * 60:
            stop_pod(f"idle for {IDLE_MIN} minutes")


def safe(path):
    full = os.path.abspath(os.path.join(ROOT, path.lstrip("/")))
    if not (full == ROOT or full.startswith(ROOT + os.sep)):
        raise ValueError("path outside root")
    return full


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_any(self, method):
        if self.headers.get("X-Token") != TOKEN:
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            return self.send(403, {"error": "forbidden"})
        state["last_seen"] = time.time()
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        try:
            out = getattr(self, f"{method}_{u.path.strip('/').replace('/', '_') or 'ping'}")(q, body)
        except AttributeError:
            return self.send(404, {"error": "no such endpoint"})
        except Exception as e:
            return self.send(500, {"error": repr(e)})
        if isinstance(out, (bytes, bytearray)):
            return self.send(200, out, "application/octet-stream")
        return self.send(200, out)

    def do_GET(self):
        self.handle_any("get")

    def do_POST(self):
        self.handle_any("post")

    def do_PUT(self):
        self.handle_any("put")

    # ---- endpoints
    def get_ping(self, q, body):
        return {"ok": True, "uptime_s": round(time.time() - state["started"]), "running": running_jobs(),
                "deadline_in_s": round(state["deadline"] - time.time()), "pod": os.environ.get("RUNPOD_POD_ID")}

    def post_extend(self, q, body):
        state["deadline"] = time.time() + float(q.get("hours", MAX_HOURS)) * 3600
        return {"deadline_in_s": round(state["deadline"] - time.time())}

    def post_run(self, q, body):
        d = json.loads(body)
        name = d["name"]
        with lock:
            if name in jobs and jobs[name].poll() is None:
                return {"error": "already running", "name": name}
            log = open(os.path.join(LOGS, name + ".log"), "ab")
            log.write(f"\n=== {time.strftime('%F %T')} $ {d['cmd']}\n".encode())
            log.flush()
            env = dict(os.environ, **d.get("env", {}))
            jobs[name] = subprocess.Popen(["bash", "-lc", d["cmd"]], stdout=log, stderr=subprocess.STDOUT,
                                          cwd=d.get("cwd", ROOT), env=env, start_new_session=True)
        return {"started": name, "pid": jobs[name].pid}

    def get_status(self, q, body):
        out = {}
        for n, p in jobs.items():
            if q.get("name") in (None, n):
                out[n] = {"running": p.poll() is None, "exit": p.poll()}
        return out

    def post_kill(self, q, body):
        p = jobs.get(q["name"])
        if p and p.poll() is None:
            os.killpg(p.pid, 15)
            return {"killed": q["name"]}
        return {"killed": None}

    def get_log(self, q, body):
        path = os.path.join(LOGS, q["name"] + ".log")
        size = os.path.getsize(path) if os.path.exists(path) else 0
        off = int(q.get("offset", 0))
        if off < 0:
            off = max(0, size + off)
        with open(path, "rb") as f:
            f.seek(off)
            data = f.read(int(q.get("length", 200000)))
        return {"offset": off, "size": size, "text": data.decode("utf-8", "replace")}

    def put_file(self, q, body):
        full = safe(q["path"])
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "ab" if q.get("append") == "1" else "wb") as f:
            f.write(body)
        return {"path": full, "size": os.path.getsize(full)}

    def get_file(self, q, body):
        full = safe(q["path"])
        with open(full, "rb") as f:
            f.seek(int(q.get("offset", 0)))
            return f.read(int(q.get("length", 8 << 20)))

    def get_stat(self, q, body):
        full = safe(q["path"])
        if not os.path.exists(full):
            return {"exists": False}
        if os.path.isdir(full):
            return {"exists": True, "dir": True,
                    "entries": sorted((e.name, e.stat().st_size, e.is_dir()) for e in os.scandir(full))}
        return {"exists": True, "dir": False, "size": os.path.getsize(full)}


if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[agent] listening on {PORT}; idle limit {IDLE_MIN} min, deadline {MAX_HOURS} h, stop action {STOP_ACTION}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
