"""Check, without downloading any weights, that this machine's transformers/torch can build and run the
architectures of the candidate multilingual sentence-embedding models on CPU and MPS. config.json is read
into memory from the Hub API at a pinned revision; the model is randomly initialised; inputs are random ids.
Reports throughput so the cost of embedding ~6,000 sentences can be estimated."""
import json
import time

import requests
import torch
from transformers import AutoConfig, AutoModel

REPOS = {"intfloat/multilingual-e5-base": "d128750597153bb5987e10b1c3493a34e5a4502a",
         "intfloat/multilingual-e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
         "sentence-transformers/LaBSE": "836121a0533e5664b21c7aacc5d22951f2b8b25b"}
out = {}
for repo, rev in REPOS.items():
    cfg = requests.get(f"https://huggingface.co/{repo}/raw/{rev}/config.json", timeout=30).json()
    mt = cfg.pop("model_type")
    config = AutoConfig.for_model(mt, **{k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("architectures", "transformers_version")})
    model = AutoModel.from_config(config).eval()
    n_par = sum(p.numel() for p in model.parameters())
    res = {"model_type": mt, "params_M": round(n_par / 1e6), "hidden": config.hidden_size, "layers": config.num_hidden_layers}
    for dev in ["cpu", "mps"]:
        m = model.to(dev)
        ids = torch.randint(5, config.vocab_size - 1, (32, 64), device=dev)
        att = torch.ones_like(ids)
        with torch.no_grad():
            m(input_ids=ids, attention_mask=att)           # warm-up
            if dev == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            for _ in range(3):
                h = m(input_ids=ids, attention_mask=att).last_hidden_state
            if dev == "mps":
                torch.mps.synchronize()
            dt = (time.time() - t0) / 3
        res[f"{dev}_sent_per_s_at_64tok"] = round(32 / dt)
        res[f"{dev}_finite"] = bool(torch.isfinite(h).all().item())
    out[repo] = res
    print(repo, res, flush=True)
    del model
json.dump(out, open("<scratchpad>/prereg/scout/embed_arch_check.json", "w"), indent=1)
