#!/usr/bin/env bash
# Builds the GPU environment in gpu/.venv: Python 3.12 with uv, the vLLM steering fork that neuron-explorer uses
# (RhizoNymph/vllm, which brings a matching CUDA PyTorch), Hugging Face transformers for the hooks, and the
# linear-attention kernels Qwen3.5/3.6 need. Mirrors neuron-explorer's sweep/setup_pod.sh, with data under
# $PHYLO_DATA instead of /workspace. Run from the repository root, on Linux aarch64 (DGX OS, or Ubuntu inside WSL):
#     bash gpu/setup_env.sh
# Then start every shell with: source ~/phylo-env.sh && source gpu/.venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."

PHYLO_DATA="${PHYLO_DATA:-$HOME/phylo-data}"
# The fork's head on 2026-08-21 (merge of upstream 2026-08-19), and the upstream commit it last merged, whose
# precompiled wheel (aarch64 and x86_64, CUDA 13) supplies the compiled kernels.
FORK_COMMIT="${FORK_COMMIT:-4506f935bba37b28a41746cd8f5897f82e46588d}"
BASE_COMMIT="${BASE_COMMIT:-755492e37d7d7201d93f0effa200acac56601f38}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"   # if auto picks a CPU or cu12x build, rerun with TORCH_BACKEND=cu130

[ "$(uname -m)" = aarch64 ] || echo "note: $(uname -m), not aarch64; the steps are the same on x86_64 rented GPUs"

cat > "$HOME/phylo-env.sh" <<EOF
export PHYLO_DATA=$PHYLO_DATA
export HF_HOME=$PHYLO_DATA/hf UV_CACHE_DIR=$PHYLO_DATA/.uv-cache
export PATH=\$HOME/.local/bin:\$PATH
EOF
grep -q phylo-env.sh ~/.bashrc || echo "source ~/phylo-env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source "$HOME/phylo-env.sh"
mkdir -p "$PHYLO_DATA"/{hf,src,dense,acts,runs,logs}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v tmux >/dev/null || echo "tmux missing: sudo apt-get install -y tmux git build-essential"

SRC="$PHYLO_DATA/src/vllm-steer"
[ -d "$SRC" ] || git clone --filter=blob:none https://github.com/RhizoNymph/vllm "$SRC"
git -C "$SRC" fetch -q origin "$FORK_COMMIT" && git -C "$SRC" checkout -q "$FORK_COMMIT"

[ -d gpu/.venv ] || uv venv --python 3.12 gpu/.venv
# shellcheck disable=SC1091
source gpu/.venv/bin/activate
(cd "$SRC" && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$BASE_COMMIT" \
    uv pip install -e . --torch-backend="$TORCH_BACKEND")
uv pip install transformers accelerate safetensors datasets numpy
uv pip install --no-deps flash-linear-attention fla-core einops

python - <<'EOF'
import torch, transformers, vllm
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__)
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("device", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability(),
      "| kernels for", torch.cuda.get_arch_list())
try:
    import fla.ops.gated_delta_rule  # noqa: F401
    print("fla ok")
except Exception as e:  # only Qwen3.5 / 3.6 need it
    print("fla failed (only Qwen3.5/3.6 need it):", e)
EOF
echo "done. Next: python gpu/check_gpu.py"
