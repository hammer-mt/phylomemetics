"""The sentences run on their own, in a fixed order: [(key, text)]. Keys are tuples written as strings."""
import ast, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def consts(path, names):
    """Read literal constants out of a script without importing it (its imports need MLX)."""
    out = {}
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


W = consts(STUDY / "wording_test.py", {"REWORDED", "NEUTRAL", "DEV", "TEST"})
W |= consts(STUDY / "robustness/nulls.py", {"HARM", "VOW"})


def items():
    out = []
    for u, t in W["REWORDED"].items():
        out.append((f"reworded|{u}", t))
    for i, t in enumerate(W["NEUTRAL"]):
        out.append((f"neutral|{i}", t))
    for nm in ("DEV", "TEST"):
        for i, (a, b) in enumerate(W[nm]):
            out += [(f"rev{nm}|{i}|r", a), (f"rev{nm}|{i}|c", b)]
    for nm in ("HARM", "VOW"):
        out += [(f"{nm}|{i}", t) for i, t in enumerate(W[nm])]
    sel, tst = (json.load(open(HERE / f"data/{n}.json")) for n in ("select", "test"))
    for m, d in sel.items():
        for kind in ("explicit", "reworded"):
            out += [(f"sel|{m}|{kind}|{i}", t) for i, t in enumerate(d[kind])]
    for m, d in tst.items():
        for i, p in enumerate(d["pairs"]):
            out += [(f"test|{m}|{i}|p", p["present"]), (f"test|{m}|{i}|a", p["absent"])]
    for w in map(json.loads, open(STUDY / "data/corpus.jsonl")):
        if w["group"] == "baseline":
            out += [(f"film|{w['id']}|{i}", t) for i, t in enumerate(w["sentences"])]
    return out


if __name__ == "__main__":
    print(len(items()))
