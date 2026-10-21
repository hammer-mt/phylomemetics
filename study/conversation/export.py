"""Export the Claude Code session that produced this project into conversation/conversation.json and .js.

Keeps Mike's messages (including those sent mid-turn and his answers to Claude's questions), Claude's replies,
and one line per tool action. Leaves out raw tool output, Claude's private reasoning, injected system text and
skill instructions, and redacts emails, private links, secrets and local paths.

    python conversation/export.py ~/.claude/projects/<project>/<session-id>.jsonl      # from study/
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

OUT = Path(__file__).parent

SCRATCH = re.compile(r"/private/tmp/claude-\d+/[^\s\"'`)]*?/scratchpad")
REDACTIONS = [
    (re.compile(r"\b(?!noreply@anthropic\.com)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[email]"),
    (re.compile(r"\b(?:gho_|ghp_|sk-|AIza)[A-Za-z0-9_-]{16,}\b"), "[redacted]"),
    # Mike's book attachment: keep that he attached it, drop the local path and draft filename.
    (re.compile(r'@"[^"\n]*/Marketing Memetics[^"\n]*\.pdf"'), "[Attached: his book, Marketing Memetics (PDF)]"),
    # Names of Mike's private GitHub repos.
    (re.compile(r"`?marketingmemetics`? and `?meme_mapping`? repos"), "two related private repos"),
    (re.compile(r"(?:hammer-mt/|`)(?:marketingmemetics|meme_mapping)`?"), "a private repo"),
    (re.compile(r"/Users/michaeltaylor/\.claude/projects/[^\s\"'`)]*"), "<session files>"),
    (re.compile(r"/private/tmp/claude-\d+/[^\s\"'`)]*"), "<temporary file>"),
    (re.compile(r"/Users/michaeltaylor/Every-org/phylomemetics/"), ""),
    (re.compile(r"/Users/michaeltaylor/Every-org/phylomemetics\b"), "the project folder"),
    (re.compile(r"/Users/michaeltaylor"), "~"),
    (re.compile(r"`?~/Every-org/phylomemetics/?`?"), "the project folder"),
    (re.compile(r"\bmjt145\b"), "[username]"),
]


def redact(s):
    s = SCRATCH.sub("<scratchpad>", s or "")
    for pat, rep in REDACTIONS:
        s = pat.sub(rep, s)
    return s


def strip_injected(s):
    s = re.sub(r"<system-reminder>.*?</system-reminder>", "", s, flags=re.S)
    return s.strip()


def user_parts(text):
    """Split a user message into plain text, pasted blocks and slash commands."""
    parts = []
    cmd = re.search(r"<command-name>/?([^<]+)</command-name>", text)
    if cmd:
        args = re.search(r"<command-args>(.*?)</command-args>", text, flags=re.S)
        args = args.group(1).strip() if args else ""
        return [{"type": "command", "text": redact("/" + cmd.group(1).strip() + (" " + args if args else ""))}]
    pos = 0
    for m in re.finditer(r'<pasted_content id="[^"]*">\n?(.*?)\n?</pasted_content id="[^"]*">', text, flags=re.S):
        before = text[pos:m.start()].strip()
        if before:
            parts.append({"type": "text", "text": redact(before)})
        parts.append({"type": "pasted", "text": redact(m.group(1).strip())})
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        parts.append({"type": "text", "text": redact(rest)})
    return parts


def rel(path):
    return redact(str(path or "")).lstrip("/") or "a file"


def workflow_name(inp):
    script = inp.get("script") or ""
    m = re.search(r"description:\s*'([^']*)'", script) or re.search(r'description:\s*"([^"]*)"', script)
    if m:
        return m.group(1)
    if inp.get("scriptPath"):
        return Path(inp["scriptPath"]).stem.rsplit("-wf_", 1)[0].replace("-", " ") + " (re-run)"
    return inp.get("name") or "a workflow"


def action(name, inp):
    """One plain line for a tool call, or None to leave it out."""
    if name in ("ToolSearch",) or name.endswith("__guide"):
        return None
    if name == "Bash":
        return inp.get("description") or "Ran a shell command"
    if name == "Read":
        return f"Read {rel(inp.get('file_path'))}"
    if name == "Edit":
        return f"Edited {rel(inp.get('file_path'))}"
    if name == "Write":
        return f"Wrote {rel(inp.get('file_path'))}"
    if name == "Workflow":
        return f"Ran a multi-agent workflow: {workflow_name(inp)}"
    if name == "Agent":
        return f"Started an agent: {inp.get('description', '')}"
    if name == "Skill":
        return f"Loaded the {inp.get('skill', '')} skill"
    if name == "TaskStop":
        return "Stopped a background task"
    if name == "Monitor":
        return f"Watched a background job: {inp.get('description', '')}".rstrip(": ")
    if name == "WebSearch":
        return f"Searched the web: {inp.get('query', '')}"
    if name == "WebFetch":
        return f"Fetched {urlparse(inp.get('url', '')).netloc}"
    if name == "Artifact":
        act = inp.get("action") or "publish"
        if act == "publish":
            label = f" ({inp['label']})" if inp.get("label") else ""
            return f"Published {Path(inp.get('file_path', 'a page')).name} as an Artifact page{label}"
        return f"Artifact: {act}"
    if name.startswith("mcp__1a59c906"):
        verb = name.rsplit("__", 1)[1]
        return {"create": "Created the paper in Claude Docs", "batch": "Wrote to the paper in Claude Docs",
                "update": "Edited the paper in Claude Docs", "read": "Read the paper in Claude Docs",
                "export": "Exported the paper from Claude Docs"}.get(verb, f"Claude Docs: {verb}")
    if name.startswith("mcp__f5fd1d63"):
        return "Read a file from Google Drive"
    if "Claude_Browser" in name or "claude-in-chrome" in name:
        where = "Chrome" if "claude-in-chrome" in name else "the built-in browser"
        verb = name.rsplit("__", 1)[1].replace("_", " ")
        url = inp.get("url") if isinstance(inp, dict) else None
        host = urlparse(url).netloc if url and url.startswith("http") else ""
        return f"Used {where}: {verb}" + (f" ({host})" if host and "localhost" not in host else "")
    return name.split("__")[-1].replace("_", " ")


def main(path):
    rows = [json.loads(line) for line in open(path)]
    events, seen_tools, seen_text = [], set(), set()
    asked = {}

    def add_actions(item, t):
        if events and events[-1]["kind"] == "actions":
            events[-1]["items"].append(item)
        else:
            events.append({"kind": "actions", "time": t, "items": [item]})

    for d in rows:
        if d.get("isSidechain"):
            continue
        typ, t = d.get("type"), d.get("timestamp")
        if typ == "system" and d.get("subtype") == "compact_boundary":
            events.append({"kind": "divider", "time": t, "text": "The session ran out of context here. Claude continued from an automatic summary of everything above."})
            continue
        if typ == "attachment":
            a = d.get("attachment") or {}
            if a.get("type") == "queued_command" and (a.get("origin") or {}).get("kind") == "human":
                prompt = a.get("prompt") or ""
                if isinstance(prompt, list):
                    prompt = "\n\n".join(b.get("text", "") for b in prompt if isinstance(b, dict) and b.get("type") == "text")
                text = strip_injected(prompt)
                if text:
                    events.append({"kind": "user", "time": a.get("timestamp") or t, "midturn": True, "parts": user_parts(text)})
            continue
        if typ == "user":
            if d.get("isCompactSummary") or d.get("isMeta"):
                # isMeta rows are written by the harness (skill text, auto-loaded commands such as
                # /workflow-authoring in Ultracode mode, image-path notes), never typed by Mike.
                continue
            c = d["message"]["content"]
            if isinstance(c, str):
                if c.lstrip().startswith("<task-notification>"):
                    summ = re.search(r"<summary>(.*?)</summary>", c, flags=re.S)
                    status = re.search(r"<status>(.*?)</status>", c)
                    if summ and "workflow" in summ.group(1).lower():
                        events.append({"kind": "event", "time": t, "text": redact(summ.group(1).strip())})
                    continue
                if re.match(r"\s*<local-command-(?:stdout|stderr|caveat)>", c):
                    continue
                text = strip_injected(c)
                if text:
                    events.append({"kind": "user", "time": t, "midturn": False, "parts": user_parts(text)})
                continue
            parts = []
            for b in c:
                if b.get("type") == "text":
                    txt = b.get("text") or ""
                    if d.get("isMeta") or txt.startswith("Base directory for this skill") or txt.startswith("# Workflow authoring reference") or txt.startswith("[Image: source:"):
                        continue
                    txt = strip_injected(txt)
                    if txt:
                        parts.extend(user_parts(txt))
                elif b.get("type") == "image":
                    parts.append({"type": "image", "text": "Mike attached a screenshot of his Reddit post (its text is pasted above)."})
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in asked:
                    res = b.get("content")
                    res = res if isinstance(res, str) else json.dumps(res)
                    tur = d.get("toolUseResult")
                    if isinstance(tur, dict) and isinstance(tur.get("answers"), dict) and tur["answers"]:
                        answers = [str(v) for v in tur["answers"].values()]
                    else:
                        answers = re.findall(r'"[^"]*(?:"[^"=]*")*[^"]*"="([^"]*)"', res)
                    q = asked.pop(b["tool_use_id"])
                    events.append({"kind": "user", "time": t, "midturn": False, "answer_to": q,
                                   "parts": [{"type": "text", "text": "Answered: " + redact("; ".join(answers) or res)}]})
            if parts and any(p["type"] != "image" for p in parts) or (parts and len(parts) == 1):
                events.append({"kind": "user", "time": t, "midturn": False, "parts": parts})
            continue
        if typ == "assistant":
            if d["message"].get("model") == "<synthetic>" or d.get("isApiErrorMessage"):
                continue  # harness placeholders such as "No response requested.", not Claude's words
            for b in d["message"]["content"]:
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    # Key on the message id too, so a reply Claude legitimately repeats later is kept.
                    h = hashlib.sha1((str(d["message"].get("id")) + "\0" + b["text"]).encode()).hexdigest()
                    if h in seen_text:
                        continue
                    seen_text.add(h)
                    events.append({"kind": "claude", "time": t, "text": redact(b["text"].strip())})
                elif b.get("type") == "tool_use":
                    if b["id"] in seen_tools:
                        continue
                    seen_tools.add(b["id"])
                    if b["name"] == "AskUserQuestion":
                        qs = b["input"].get("questions", [])
                        text = "\n\n".join(
                            f"**{q['question']}**\n\n" + "\n".join(f"- {o['label']}: {o.get('description', '')}" for o in q.get("options", []))
                            for q in qs)
                        asked[b["id"]] = redact(qs[0]["question"]) if qs else ""
                        events.append({"kind": "question", "time": t, "text": redact(text)})
                        continue
                    line = action(b["name"], b.get("input") or {})
                    if line:
                        add_actions({"tool": b["name"].split("__")[-1], "text": redact(line)}, t)

    users = [e for e in events if e["kind"] == "user"]
    data = {
        "title": "Phylomemetics: the working session",
        "participants": ["Mike Taylor", "Claude Opus 5.5"],
        "started": events[0]["time"] if events else None,
        "ended": events[-1]["time"] if events else None,
        "counts": {"mike": len(users), "claude": sum(e["kind"] == "claude" for e in events),
                   "actions": sum(len(e["items"]) for e in events if e["kind"] == "actions")},
        "events": events,
    }
    (OUT / "conversation.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    (OUT / "conversation.js").write_text("window.CONVERSATION = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print("exported", len(events), "events:", data["counts"])


if __name__ == "__main__":
    main(sys.argv[1])
