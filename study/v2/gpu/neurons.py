"""Read every MLP neuron of a language model, the way Cammarata's "Neural networks are already clean" does.

A neuron is one channel of a SwiGLU feed-forward block: activation = act(gate) * value, the input to the
down-projection. Its two poles are separate units. The product can also be produced by two "echo" sign regimes
(gate < 0), which the essay shows look unrelated to the neuron's main job, so each activation is split:

    main  = activation where gate > 0     (+ x +  -> positive pole,   + x -  -> negative pole)
    echo  = activation where gate <= 0    (- x -  -> small positive,  - x +  -> small negative)

Units are indexed as in v1: unit = layer * width + neuron for the positive pole, plus n_neurons for the negative
pole. Everything here runs in PyTorch at the checkpoint's own precision (bfloat16), on one or more GPUs.
"""
import json
import os
import time

import torch

MODELS = {
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama3.3-70b": "unsloth/Llama-3.3-70B-Instruct",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",          # tests only
}


def load(tag, dtype=None, device_map=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = MODELS.get(tag, tag)
    tok = AutoTokenizer.from_pretrained(repo)
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True)
    model.eval()
    placed = getattr(model, "hf_device_map", None)
    if torch.cuda.is_available() and placed:
        off = {k: v for k, v in placed.items() if not isinstance(v, int)}
        assert not off, f"part of the model was not placed on a GPU (too little memory?): {off}"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return model, tok


def provenance(model):
    """What produced a pass's numbers: the checkpoint's commit, library versions and the GPU."""
    import transformers
    return {"repo": getattr(model.config, "_name_or_path", None), "commit": getattr(model.config, "_commit_hash", None),
            "dtype": str(model.dtype), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}


def lead_in(tok):
    """Token ids to put before a short text so that its first word is not at position 0. Position 0 is never scored
    (a first token's activations are an artefact of being first). Tokenizers that add a begin-of-text token need
    nothing; for the others (Qwen) the end-of-text token, which separates documents in training, stands in."""
    probe = tok("a", add_special_tokens=True)["input_ids"]
    return [] if len(probe) > 1 and probe[0] == tok.bos_token_id else [tok.eos_token_id]


def layers_of(model):
    return model.model.layers


def geometry(model):
    ls = layers_of(model)
    return len(ls), ls[0].mlp.gate_proj.weight.shape[0]


class Tap:
    """Patches every MLP so `fn(layer, gate, act)` sees the gate pre-activation and the neuron activation.

    If fn returns a tensor it replaces the activation (used for ablation and clamping). Call .remove() to undo.
    """

    def __init__(self, model, fn, layers=None):
        self.saved = []
        for i, layer in enumerate(layers_of(model)):
            if layers is not None and i not in layers:
                continue
            mlp = layer.mlp
            self.saved.append((mlp, mlp.forward))
            mlp.forward = self._make(mlp, i, fn)

    @staticmethod
    def _make(mlp, i, fn):
        def forward(x):
            g = mlp.gate_proj(x)
            h = mlp.act_fn(g) * mlp.up_proj(x)
            out = fn(i, g, h)
            if out is not None:
                h = out
            return mlp.down_proj(h)
        return forward

    def remove(self):
        for mlp, f in self.saved:
            mlp.forward = f
        self.saved = []


def split_regimes(g, h):
    """(main, echo): the activation where the gate is open, and where it is closed."""
    open_ = g > 0
    return torch.where(open_, h, torch.zeros_like(h)), torch.where(open_, torch.zeros_like(h), h)


def encode_document(tok, sentences):
    """Token ids for sentences joined by single spaces, and each sentence's [start, end) token span.

    Position 0 (a BOS token, or the first word when the tokenizer adds none) never belongs to a span: its
    activations are an artefact of being first, as in v1.
    """
    text, char_spans = "", []
    for s in sentences:
        if text:
            text += " "
        start = len(text)
        text += s
        char_spans.append((start, len(text)))
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = []
    for cs, ce in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if b > a and a < ce and b > cs and i > 0]
        if not toks:
            raise ValueError(f"sentence has no tokens of its own: {text[cs:ce]!r}")
        spans.append((toks[0], toks[-1] + 1))
    return ids, spans, text, offsets


@torch.no_grad()
def run_hidden(model, ids, attention_mask=None):
    """Forward pass through the transformer body only (no vocabulary projection)."""
    dev = model.model.embed_tokens.weight.device
    return model.model(input_ids=ids.to(dev), attention_mask=None if attention_mask is None else attention_mask.to(dev))


def now():
    return time.strftime("%H:%M:%S")


def save_json(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
