"""Shared helpers for the GPU scripts: load a Qwen model in bf16, find its decoder layers, and hook one MLP.

A neuron is one channel of the SwiGLU hidden layer, act = silu(gate) * up, the input to down_proj. Its sign regime is
sign(gate) x sign(up):

    open+  gate > 0, up > 0   act > 0   gate-open, positive pole
    open-  gate > 0, up < 0   act < 0   gate-open, negative pole
    echo+  gate < 0, up < 0   act > 0   the "double-negative echo": silu(gate) is a small negative number (>= -0.28)
    echo-  gate < 0, up > 0   act < 0

The scripts record gate, up and act separately so the regimes can be told apart afterwards.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REGIMES = ("open+", "open-", "echo+", "echo-")


def pick_device(name=None):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_name, device, n_layers=None):
    """Loads the model in bf16 and returns (tokenizer, model, decoder layers). With n_layers, keeps only the first
    n_layers decoder layers: recording layer L needs layers 0..L, which saves memory and time on a big model."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "right"  # the scripts drop position 0 as the first real token
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "text_config"):  # Qwen3.5 / 3.6 checkpoints are image-text models; use the language model only
        from transformers import AutoModelForImageTextToText as Auto
    else:
        Auto = AutoModelForCausalLM
    try:  # device_map loads the weights straight onto the GPU; it needs accelerate
        model = Auto.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device)
    except ValueError:
        model = Auto.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.eval()
    layers = decoder_layers(model)
    if n_layers is not None:
        del layers[n_layers:]
    return tok, model, layers


def decoder_layers(model):
    inner = model.model
    lm = getattr(inner, "language_model", inner)
    return lm.layers


def regime_masks(gate, up):
    gp, upos = gate > 0, up > 0
    return {"open+": gp & upos, "open-": gp & ~upos, "echo+": ~gp & ~upos, "echo-": ~gp & upos}


class MLPRecorder:
    """Keeps the last forward pass's gate, up and act (= down_proj input) for one layer's MLP."""

    def __init__(self, layer):
        mlp = layer.mlp
        if not hasattr(mlp, "gate_proj"):
            raise SystemExit("this layer's MLP has no gate_proj (a mixture-of-experts layer?); see SETUP.md, MoE models")
        self.gate = self.up = self.act = self.resid = None
        self.handles = [
            mlp.gate_proj.register_forward_hook(lambda m, i, o: setattr(self, "gate", o.detach())),
            mlp.up_proj.register_forward_hook(lambda m, i, o: setattr(self, "up", o.detach())),
            mlp.down_proj.register_forward_pre_hook(lambda m, i: setattr(self, "act", i[0].detach())),
            layer.register_forward_hook(lambda m, i, o: setattr(self, "resid", (o[0] if isinstance(o, tuple) else o).detach())),
        ]

    def remove(self):
        for h in self.handles:
            h.remove()


def steer_hook(layer, neuron, add):
    """Adds `add` to one neuron's activation at every position (prompt and generated). The same as adding
    add * W_down[:, neuron] to the MLP output, which is what the vLLM steering fork does at its mlp_out hook."""
    def pre(m, i):
        x = i[0].clone()
        x[..., neuron] += add
        return (x,)
    return layer.mlp.down_proj.register_forward_pre_hook(pre)
