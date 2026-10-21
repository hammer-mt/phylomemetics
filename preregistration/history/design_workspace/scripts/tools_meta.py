"""Metadata only (no package or weight downloads): Hugging Face model info (file sizes, license, pinned sha) for
candidate sentence-embedding models, and PyPI metadata (latest version, Python and macOS arm64 wheel support) for
packages that are not installed. Writes scout/tools/meta.json."""
import json
import requests

H = {"User-Agent": "phylomemetics-scout/0.1 (research)"}
MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-small",
          "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large", "sentence-transformers/LaBSE", "BAAI/bge-m3"]
PKGS = ["sentence-transformers", "bertopic", "umap-learn", "hdbscan", "pynndescent", "numba", "gensim", "llvmlite", "jieba"]
out = {"models": {}, "packages": {}}
for m in MODELS:
    d = requests.get(f"https://huggingface.co/api/models/{m}", params={"blobs": "true"}, headers=H, timeout=30).json()
    sib = {s["rfilename"]: s.get("size") for s in d.get("siblings", [])}
    weights = {k: v for k, v in sib.items() if k in ("model.safetensors", "pytorch_model.bin")}
    out["models"][m] = dict(sha=d.get("sha"), license=(d.get("cardData") or {}).get("license"), downloads=d.get("downloads"),
                            created=d.get("createdAt"), last_modified=d.get("lastModified"),
                            weights_MB={k: round(v / 1e6) for k, v in weights.items() if v},
                            tokenizer_files=[k for k in sib if "tokenizer" in k or k.endswith(".model")][:6],
                            languages=len((d.get("cardData") or {}).get("language") or []))
    print(m, out["models"][m], flush=True)
for p in PKGS:
    d = requests.get(f"https://pypi.org/pypi/{p}/json", headers=H, timeout=30).json()
    info = d["info"]
    files = d["urls"]
    arm = [f["filename"] for f in files if "macosx" in f["filename"] and "arm64" in f["filename"]]
    out["packages"][p] = dict(version=info["version"], requires_python=info.get("requires_python"),
                              upload_time=files[0]["upload_time"] if files else None,
                              wheels=[f["filename"] for f in files][:8], arm64_wheels=arm[:4],
                              pure_python=any(f["filename"].endswith("py3-none-any.whl") for f in files),
                              requires_dist=(info.get("requires_dist") or [])[:12])
    print(p, out["packages"][p]["version"], out["packages"][p]["requires_python"], "pure" if out["packages"][p]["pure_python"] else arm[:2], flush=True)
json.dump(out, open("<scratchpad>/prereg/scout/tools/meta.json", "w"), indent=1)
