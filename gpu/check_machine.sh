#!/usr/bin/env bash
# What is this machine? Run it on the new box (in DGX OS, or inside WSL on a Windows RTX Spark) before installing
# anything, and paste the output into gpu/MACHINE.md. It only reads; it changes nothing.
#     bash gpu/check_machine.sh | tee machine.txt
set -u
hr() { printf '\n== %s\n' "$1"; }

hr "CPU architecture (expect aarch64 on DGX Spark and RTX Spark)"
uname -m
hr "OS"
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -r
grep -qi microsoft /proc/version 2>/dev/null && echo "running inside WSL (Windows host: this is an RTX Spark PC, not a DGX Spark)"
[ -f /etc/dgx-release ] && cat /etc/dgx-release
hr "CPU"
nproc
lscpu 2>/dev/null | grep -E 'Model name|Vendor ID|^CPU\(s\)' | sort -u
hr "Memory as the OS sees it (unified: the GPU draws on the same pool)"
free -h
hr "GPU and driver"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,power.limit --format=csv 2>/dev/null
else
  echo "nvidia-smi not found (in WSL it lives in /usr/lib/wsl/lib)"; ls /usr/lib/wsl/lib 2>/dev/null | head
fi
hr "CUDA toolkit"
(command -v nvcc && nvcc --version) || ls -d /usr/local/cuda* 2>/dev/null || echo "no nvcc (fine: the PyTorch and vLLM wheels bring their own CUDA runtime)"
hr "Disk (the model weights and activation shards need room: plan 300 GB free)"
df -h "$HOME" /
hr "Python and tools"
for c in python3 uv git tmux docker; do printf '%-8s ' "$c"; (command -v "$c" >/dev/null && "$c" --version 2>&1 | head -1) || echo missing; done
hr "Network"
hostname; hostname -I 2>/dev/null || ip -brief addr 2>/dev/null
command -v tailscale >/dev/null && tailscale ip 2>/dev/null
