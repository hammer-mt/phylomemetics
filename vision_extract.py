"""Phase 3: memes in images. Record neuron activations of two vision models on each work's lead image.

Models
  clip    OpenAI CLIP RN50x4 image encoder (open_clip): channels of the last two ResNet stages,
          strongest response anywhere in the image (spatial max). These are the "multimodal neurons"
          studied by Goh et al. (2021).
  dino    Meta DINOv2-base (transformers): each MLP neuron (after GELU) in all 12 layers, strongest
          response over image patches.

Each image is seen as 10 views (full image, four corner crops, centre crop, and their mirror images).
A work's score on a unit = the mean over its 10 views; the per-view activations are kept so the two
models can be matched neuron-to-neuron over 10x more samples.

Writes ACTS/<model>/views.npy [n_images*10, n_units] float16, ACTS/<model>/meta.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ACTS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ.get("PHYLO_VISION", "activations_vision"))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def views(img, size):
    """10 square views: full (padded), 4 corners, centre, and mirrors."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    full = Image.new("RGB", (max(w, h),) * 2, (0, 0, 0))
    full.paste(img, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
    crops = [full, img.crop((0, 0, s, s)), img.crop((w - s, 0, w, s)), img.crop((0, h - s, s, h)),
             img.crop((w - s, h - s, w, h)), img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))]
    crops = crops[:5] + [crops[5]]
    out = []
    for c in crops:
        c = c.resize((size, size), Image.BICUBIC)
        out += [c, c.transpose(Image.FLIP_LEFT_RIGHT)]
    return out[:10]


def load_images():
    meta = json.load(open("data/images.json"))
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    items = [(w["id"], meta[str(w["id"])]["file"]) for w in works if meta.get(str(w["id"]), {}).get("file")]
    return items


def run_clip(items):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("RN50x4", pretrained="openai")
    model = model.visual.eval().to(DEV)
    size = preprocess.transforms[0].size if isinstance(preprocess.transforms[0].size, int) else preprocess.transforms[0].size[0]
    norm = preprocess.transforms[-1]
    feats = {}

    def hook(name):
        def f(_, __, out):
            feats[name] = out.amax(dim=(2, 3))
        return f

    model.layer3.register_forward_hook(hook("layer3"))
    model.layer4.register_forward_hook(hook("layer4"))
    import torchvision.transforms.functional as TF
    rows = []
    with torch.no_grad():
        for wid, path in items:
            batch = torch.stack([norm(TF.to_tensor(v)) for v in views(Image.open(path), size)]).to(DEV)
            model(batch)
            rows.append(torch.cat([feats["layer3"], feats["layer4"]], dim=1).float().cpu().numpy())
            print(f"  clip {wid}", flush=True)
    names = {"layer3": int(feats["layer3"].shape[1]), "layer4": int(feats["layer4"].shape[1])}
    return np.concatenate(rows).astype(np.float16), names


def run_dino(items):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV)
    acts = [None] * len(model.encoder.layer)

    def hook(i):
        def f(_, __, out):
            acts[i] = out.amax(dim=1)          # max over tokens, after the MLP's activation function
        return f

    for i, layer in enumerate(model.encoder.layer):
        layer.mlp.activation_fn.register_forward_hook(hook(i))
    rows = []
    with torch.no_grad():
        for wid, path in items:
            vs = views(Image.open(path), 224)
            inputs = proc(images=vs, return_tensors="pt", do_resize=False, do_center_crop=False).to(DEV)
            model(**inputs)
            rows.append(torch.cat(acts, dim=1).float().cpu().numpy())
            print(f"  dino {wid}", flush=True)
    return np.concatenate(rows).astype(np.float16), {"layers": len(acts), "hidden": int(acts[0].shape[1])}


def main(which):
    items = load_images()
    out = ACTS / which
    out.mkdir(parents=True, exist_ok=True)
    arr, info = (run_clip if which == "clip" else run_dino)(items)
    np.save(out / "views.npy", arr)
    json.dump({"model": which, "info": info, "image_work": [wid for wid, _ in items], "n_views": 10},
              open(out / "meta.json", "w"))
    print(which, arr.shape, "saved to", out)


if __name__ == "__main__":
    main(sys.argv[1])
