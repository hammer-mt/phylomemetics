# Setting up the NVIDIA "Spark" for full-precision neuron work

A guide for the first day with the new machine: work out what it is, install a CUDA stack that runs Qwen models in bf16 with hooks on every MLP neuron and with neuron steering, and run a smoke test on Qwen3-32B. It follows the method of Nick Cammarata's essay *Neural Networks Are Already Clean* ([Goodfire](https://already-clean.goodfire.com/)) and the tooling of [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). Written on 27 October 2026, before the machine arrived; anything that depends on the real machine is marked **check**.

The scripts beside this file:

| Script | What it does |
| --- | --- |
| [`check_machine.sh`](check_machine.sh) | Prints the specs: architecture, OS, GPU, driver, memory, disk. Read-only. |
| [`setup_env.sh`](setup_env.sh) | Builds `gpu/.venv`: uv, Python 3.12, the vLLM steering fork with its CUDA PyTorch, transformers, the linear-attention kernels. |
| [`check_gpu.py`](check_gpu.py) | Checks PyTorch sees the GPU; measures bf16 matmul TFLOPS and memory bandwidth. |
| [`smoke_record.py`](smoke_record.py) | Records gate, up and act for one layer over a few hundred sentences of the study's corpus; checks the recording; picks neurons to steer. |
| [`smoke_steer.py`](smoke_steer.py) | Steers one neuron with Hugging Face hooks, against a random-direction control. |
| [`smoke_vllm_steer.py`](smoke_vllm_steer.py) | Repeats the steering in the vLLM fork and compares with Hugging Face. |
| [`record_topk.py`](record_topk.py) | The corpus scan at scale: top-K firings of every neuron in every sign regime, resumable; also the throughput benchmark. |
| [`common.py`](common.py) | Model loading and the MLP hooks the scripts share. |

## 1. What is an "RTX Spark"?

There are two NVIDIA machines it could be, built on the same chip family, and they need different setups.

| | **RTX Spark PC** | **DGX Spark** |
| --- | --- | --- |
| What it is | Windows 11 on Arm laptops and small desktops, on sale since 16 October 2026 ([NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-rtx-spark-pcs-available-today-a-new-beginning-for-windows-pcs)) | A small Linux box for AI developers, sold since late 2025 ([NVIDIA docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)) |
| Chip | N1X superchip: Grace CPU (20 cores, or 18 in the smaller version) + Blackwell RTX GPU (6,144 or 5,120 CUDA cores) ([NVIDIA](https://www.nvidia.com/en-us/products/rtx-spark/), [Wikipedia](https://en.wikipedia.org/wiki/Nvidia_RTX_Spark)) | GB10 Grace Blackwell: 20 Arm cores + Blackwell GPU with 6,144 CUDA cores |
| Memory | Up to 128 GB LPDDR5X unified (64 GB max on the 18-core version); soldered, so fixed at purchase | 128 GB LPDDR5X unified, 273 GB/s; a 64 GB version followed ([NVIDIA playbooks](https://build.nvidia.com/spark)) |
| Bandwidth | ~300 GB/s ([Wikipedia](https://en.wikipedia.org/wiki/Nvidia_RTX_Spark)); one review measured 309 GB/s reads against 234 GB/s on a DGX Spark ([MindStudio](https://www.mindstudio.ai/blog/msi-rtx-spark-laptop-review)) | 273 GB/s |
| Power | 45–80 W chip in laptops | ~140 W chip, 240 W supply |
| OS | Windows 11 on Arm; Linux tools through WSL. CUDA on Windows on Arm is a developer preview (CUDA 13.4) ([XDA](https://www.xda-developers.com/things-nobody-tells-you-before-buying-an-rtx-spark-laptop/)) | DGX OS (Ubuntu-based Linux) with the NVIDIA driver and CUDA preinstalled |
| Models named | ASUS ProArt P14/P16, Dell XPS 16, HP OmniBook Ultra 16, Lenovo Yoga Pro 9n, Surface Laptop Ultra, MSI Prestige N16; compact desktops from November ([NVIDIA developer blog](https://developer.nvidia.com/blog/code-test-and-deploy-on-one-machine-with-nvidia-rtx-spark/)) | NVIDIA DGX Spark, plus OEM versions (ASUS Ascent GX10, Dell Pro Max GB10, and others) |

**Assumption.** Because the name you were given is "RTX Spark", and RTX Spark PCs went on sale eleven days ago, the most likely machine is an **RTX Spark PC running Windows on Arm**, with 64 or 128 GB. People also say "RTX Spark" when they mean the DGX Spark, so the guide covers both. On the GPU side they are the same: an Arm (aarch64) CPU, a Blackwell GPU and one pool of unified memory, on which the Linux steps are identical. They differ in the OS layer: on a DGX Spark you work in Linux directly; on an RTX Spark PC you work inside WSL2 (Ubuntu on Windows), which adds a memory limit, a networking layer and a developer-preview CUDA.

### Checklist when it arrives (10 minutes)

1. **What boots?** A Windows desktop means an RTX Spark PC; an Ubuntu "DGX OS" login means a DGX Spark.
2. **On Windows**, open Settings → System → About and write down the model, the processor and the installed RAM. Then Task Manager → Performance → GPU: write down "Dedicated GPU memory" and "Shared GPU memory". Run `nvidia-smi` in PowerShell for the driver and CUDA versions.
3. **In Linux** (DGX OS, or WSL once step 2a below is done), run:

   ```bash
   bash gpu/check_machine.sh | tee machine.txt
   ```

   It prints `uname -m` (expect `aarch64`), the OS, whether it is in WSL, `nproc`, `free -h`, `nvidia-smi` (with compute capability; GB10 reports 12.1), the CUDA toolkit, free disk and network addresses. On a DGX Spark, `nvidia-smi` shows "Memory-Usage: Not Supported"; that is expected for an integrated GPU ([DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)). Use `free -h` and `check_gpu.py` instead.
4. Commit the output as `gpu/MACHINE.md`, so the estimates below can be redone with real numbers.

What the answers decide:

| Memory | What runs at full precision (bf16) |
| --- | --- |
| 128 GB | Qwen3.5-4B, Qwen3-8B, Qwen3-32B, Qwen3.6-35B-A3B. |
| 64 GB | Qwen3.5-4B and Qwen3-8B. Qwen3-32B only for recording the lower ~40 layers (recording layer L needs layers 0..L; see section 2). No 32B steering in bf16. |
| 32 GB | Qwen3.5-4B and Qwen3-8B, little else. |

## 2. Which models fit, and how long the work takes

Qwen's checkpoints are stored in bf16, so bf16 is full precision here: nothing is rounded beyond what the published weights already are. Parameter counts and shapes are from each model's `config.json` and safetensors index on Hugging Face.

| Model | Params | bf16 weights | MLP neurons | Fits in 128 GB? | Fits in 64 GB? |
| --- | --- | --- | --- | --- | --- |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (the other session's model) | 4.66B (with the vision encoder) | 9.3 GB | 32 × 9,216 = 294,912 | yes | yes |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (the paper's model, at 4 bits) | 8.19B | 16.4 GB | 36 × 12,288 = 442,368 | yes | yes |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) (Cammarata's audits) | 32.8B | 65.5 GB | 64 × 25,600 = 1,638,400 | yes, ~40 GB to spare | no (recording layers 0..~40 only) |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (mixture of experts) | 36.0B, 3B active | 71.9 GB | 40 × 256 experts × 512 = 5.24M routed, plus 40 × 512 shared | yes | no |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | 2.8T, 104B active | ~1.5 TB as shipped (MXFP4); 5.6 TB in bf16 | 896 experts, 16 routed per token | **no** | no |

Notes on the two hard cases:

- **Qwen3.6-35B-A3B** fits in bf16 on a 128 GB machine, but it is a different kind of job. Each token goes through 8 of 256 experts plus one shared expert, so an expert neuron sees about 3% of the tokens, and its "neuron" is defined only when its expert is routed. Transformers may run the experts as fused tensors with no per-expert `gate_proj` module to hook, and vLLM's fused MoE kernels never materialise the hidden layer (the fork has a routed-experts capturer, [`routed_experts_capturer.py`](https://github.com/RhizoNymph/vllm/tree/feat/integration/vllm/model_executor/layers/fused_moe), not yet checked). `common.py` stops with a message on a layer without `gate_proj`. Treat it as a second project after the dense models work.
- **Kimi K3** cannot run on this machine at any precision. Its released MXFP4 checkpoint alone is about 1.5 TB ([model card](https://huggingface.co/moonshotai/Kimi-K3), [Northflank](https://northflank.com/blog/what-is-kimi-k3-self-hosting)); 128 GB is a twelfth of that, and stacking Sparks (NVIDIA supports pairing them) does not get near it. What it would take: in MXFP4, one 8×B300 node (8 × 288 GB = 2.3 TB) or two 8×B200 or 8×H200 nodes; H100/H200 have no native FP4, so on them it becomes an FP8 job of ~2.8 TB across three or more nodes; in bf16, 5.6 TB, three or more 8×B300 nodes. All of these are rented clusters with multi-node expert parallelism, and the hooks have to work inside that. Cammarata's Kimi K3 result is an attribution graph, not a full corpus scan. Not a first-month goal.

### Throughput: what the memory bandwidth does and does not limit

Two different speeds matter, and the Spark is good at one and slow at the other.

- **Recording (the corpus scan) is compute-bound.** It pushes large batches of tokens through the model once, without generating, so the speed is set by bf16 matrix throughput, not bandwidth. A community benchmark on a DGX Spark measured about 1,066 tokens/s of bf16 prompt processing for the dense 31B Gemma at 2,048-token prompts ([NVIDIA forums](https://forums.developer.nvidia.com/t/gemma-4-day-1-inference-on-nvidia-dgx-spark-preliminary-benchmarks/365503)), which works out to about 66 TFLOPS of useful work. An RTX Spark laptop measured about the same matrix throughput as a DGX Spark (101 against 97 in one review's GPU matmul test, [MindStudio](https://www.mindstudio.ai/blog/msi-rtx-spark-laptop-review)), but laptops run at 45–80 W and may hold it less well under days of load.
- **Generation (steering rollouts) is bandwidth-bound for one sequence.** Each new token reads all the weights once: 65.5 GB for Qwen3-32B over ~273–300 GB/s caps one stream at about 4 tokens/s (the same benchmark measured 3.7 tokens/s decode for the 31B Gemma). Batching many rollouts together, which vLLM does, moves it back towards compute-bound; a few hundred tokens/s in total for 32B at batch 64+ is a reasonable expectation, to be measured.

### Time estimates for recording every MLP neuron

**These are estimates, not measurements.** They assume Hugging Face with hooks on every layer, 1,024-token windows, 60–90% of the ~66 TFLOPS above, and 20% overhead for the top-K bookkeeping. `record_topk.py --tokens 2000000` measures the real rate in about an hour and prints the same table; redo this table from it.

| Model | Rate (est.) | 100M tokens | 1B tokens | 10B tokens |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B | 4,000–7,000 tok/s | 4–7 h | 1.7–2.9 days | 17–29 days |
| Qwen3-8B | 2,500–3,500 tok/s | 8–11 h | 3.3–4.6 days | 33–46 days |
| Qwen3-32B | 600–900 tok/s | 1.3–1.9 days | 13–19 days | 4–6 months |
| Qwen3.6-35B-A3B | 1,500–5,000 tok/s (very uncertain: MoE runs inefficiently in HF) | 6–19 h | 2.3–8 days | 23–77 days |

For comparison, on rented GPUs (also estimates): an H100 SXM has ~990 TFLOPS of dense bf16, so at ~40% use one H100 records Qwen3-32B at ~5,000 tokens/s and eight at ~40,000. That is 1B tokens in ~7 hours and 10B in ~3 days (~560 GPU-hours; at $2–4 per H100-hour, roughly $1,100–2,200 for 10B tokens). A B200 is roughly twice an H100 per GPU.

How much corpus does the method need? The essay's Qwen3-32B neuron pages are drawn from a **100M-token corpus** ("100 million tokens through one Qwen3-32B neuron"; I did not find a 10B figure in the essay). 100M tokens of Qwen3-32B is a day or two on the Spark, which is the right first target. 1B is two to three weeks of continuous running, possible but better rented; 10B on 32B belongs on rented H100s/B200s. The small models reach 1B tokens on the Spark in days.

Steering sweeps are slower per neuron than recording. neuron-explorer's main design is 10 rollouts per neuron of up to 256 tokens. For one layer of Qwen3-32B (25,600 neurons) that is ~50M generated tokens, about 2–6 days per layer at 100–300 tokens/s (estimate). So steer chosen neurons, or random samples of a few hundred per layer as neuron-explorer's `run_depth.sh` does, not whole layers. For Qwen3.5-4B a whole layer is hours.

## 3. Setup, step by step

### 3a. RTX Spark PC (Windows): get to a Linux shell with the GPU

Skip to 3b on a DGX Spark.

1. **Windows updates and the NVIDIA driver.** Install all Windows updates and the newest RTX Spark driver (NVIDIA App or the maker's support page). NVIDIA warns the driver installer can leave the screen blank for about two minutes ([LeCompute](https://lecompute.fr/en/silicon/rtx-spark-windows-pile-llm/)). Then, in PowerShell, `nvidia-smi` should show the GPU and a CUDA version (one reviewer's driver reported CUDA 13.5).
2. **WSL with Ubuntu.** NVIDIA's RTX Spark playbook installs WSL with Ubuntu, Python and uv through Microsoft's Windows Developer Config ([playbook](https://build.nvidia.com/playbooks/setting-up-your-dev-environment)); in an Administrator Windows Terminal:

   ```powershell
   irm https://aka.ms/devconfig/standard/setup.ps1 | iex
   ```

   (Or just `wsl --install -d Ubuntu-24.04`.) Do **not** install a Linux NVIDIA driver inside WSL: WSL uses the Windows driver, and puts `nvidia-smi` and `libcuda` in `/usr/lib/wsl/lib`.
3. **Give WSL the memory.** WSL takes half of Windows' memory by default ([XDA](https://www.xda-developers.com/things-nobody-tells-you-before-buying-an-rtx-spark-laptop/)). Create `C:\Users\<you>\.wslconfig`:

   ```ini
   [wsl2]
   memory=100GB          # on a 128 GB machine; leaves ~28 GB for Windows. On 64 GB try 48GB.
   swap=16GB
   networkingMode=mirrored
   vmIdleTimeout=-1      # keep the VM up while nobody is logged into a WSL shell
   ```

   then `wsl --shutdown` and reopen Ubuntu. **Check:** how WSL's `memory=` limit and Windows' GPU memory split interact on unified memory is not documented anywhere I found. One reviewer saw Windows offer CUDA up to 55.4 GB on a 64 GB machine and changed a "GPU memory split" setting that needs a reboot ([XDA](https://www.xda-developers.com/things-nobody-tells-you-before-buying-an-rtx-spark-laptop/)); another saw CUDA reach 109.7 GiB on a 128 GB machine with 91 GiB reserved for the GPU ([daily.dev summary of a Linux port](https://daily.dev/posts/putting-linux-on-my-rtx-spark-laptop-because-i-don-t-have-a-microsoft-account-alokbhvko)). After step 3c, run `python gpu/check_gpu.py` in WSL and read "memory CUDA can see": it needs to be above ~80 GiB for Qwen3-32B in bf16. If it is not, look for the GPU memory allocation setting (in the NVIDIA App or the firmware setup), raise it, reboot, and check again.
4. **Known preview problems** ([LeCompute](https://lecompute.fr/en/silicon/rtx-spark-windows-pile-llm/)): NVIDIA warns that some PyTorch workloads can hit a GPU timeout during the CUDA developer preview, which can hang or restart the machine. Windows resets the GPU when a single kernel runs longer than its timeout (TDR, 2 seconds by default). Keep batches moderate (`record_topk.py --batch 8` launches kernels of milliseconds). If timeouts happen anyway, Microsoft documents the `TdrDelay` registry value that lengthens it ([Microsoft](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys)); that is a system setting to change deliberately, not by default.
5. **Keep it awake.** Settings → System → Power: never sleep when plugged in; Control Panel → Power Options → lid close does nothing when plugged in. A multi-day recording run on a laptop also wants it on a hard surface with the charger in.
6. **Keep data inside WSL.** Work in the Linux file system (`~/...`), never under `/mnt/c`: the Windows drive is reached over a slow bridge, and model loading from it is many times slower.

A native Linux install exists only as an unofficial port (Arch Linux ARM on an ASUS ProArt P16); NVIDIA has not announced Linux support for the laptop chips. Not for tomorrow.

### 3b. DGX Spark: first boot

Follow the first-boot wizard; DGX OS ships with the driver and CUDA 13. Then `sudo apt update && sudo apt full-upgrade` and reboot. NVIDIA's playbooks at [build.nvidia.com/spark](https://build.nvidia.com/spark) cover remote access ("connect-to-your-spark", which uses the NVIDIA Sync app on the Mac), vLLM, and pairing two Sparks. When memory looks full after a big model is unloaded, NVIDIA's documented workaround is to drop the page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` ([known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).

### 3c. The Python environment (both machines, in Linux)

```bash
sudo apt-get install -y git tmux build-essential
git clone https://github.com/hammer-mt/phylomemetics.git ~/phylomemetics
cd ~/phylomemetics
bash gpu/check_machine.sh | tee machine.txt
bash gpu/setup_env.sh
source ~/phylo-env.sh && source gpu/.venv/bin/activate
python gpu/check_gpu.py
```

What `setup_env.sh` does, and why:

- **Data under one folder.** `PHYLO_DATA` (default `~/phylo-data`) holds the Hugging Face cache, the source checkouts and all outputs; `~/phylo-env.sh` exports it and `HF_HOME`.
- **uv and Python 3.12** in `gpu/.venv`, separate from the MLX environment at the repository root (MLX does not exist on Linux).
- **The vLLM steering fork first, PyTorch with it.** It installs [RhizoNymph/vllm](https://github.com/RhizoNymph/vllm) ("vLLM with activation steering, capture, and patching") exactly as neuron-explorer's [`sweep/setup_pod.sh`](https://github.com/lyramakesmusic/neuron-explorer/blob/main/sweep/setup_pod.sh) does: the fork's additions are Python and Triton, so `VLLM_USE_PRECOMPILED=1` takes the compiled kernels from the upstream wheel at the commit the fork last merged (`755492e3`). That commit has an **aarch64** wheel built for CUDA 13 at [wheels.vllm.ai](https://wheels.vllm.ai/755492e37d7d7201d93f0effa200acac56601f38/cu130/vllm/), so nothing needs compiling on the Spark. `--torch-backend=auto` lets uv pick the matching CUDA PyTorch. The script pins the fork to its head commit of 21 August 2026 (`4506f935`), the newest on its main branch when this was written; neuron-explorer clones the head, so if the other session finds a newer commit matters, set `FORK_COMMIT`.
- **Why the CUDA 13 (cu130) builds matter on this chip.** The GPU is Blackwell with compute capability 12.1 (sm_121). Older cu128 PyTorch wheels ship a `ptxas` that does not know sm_121; cu130 builds work, sometimes printing a harmless warning that sm_121 is not in the official list ([community guides](https://github.com/natolambert/dgx-spark-setup), [NVIDIA forums](https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862)). If `check_gpu.py` warns that the build has no kernels for the device, or `torch.version.cuda` starts with 12, rerun with `TORCH_BACKEND=cu130 bash gpu/setup_env.sh`.
- **transformers, accelerate, datasets, safetensors**, and the linear-attention kernels (`flash-linear-attention`, `fla-core`) that Qwen3.5/3.6 need; without them transformers falls back to a pure-PyTorch path that is correct but very slow.

**Fallback if the wheels fail:** NVIDIA's NGC containers are its supported route on DGX Spark ("the latest features and performance improvements are available through NVIDIA NGC containers"), and one guide calls `nvcr.io/nvidia/pytorch:25.10-py3` or newer the cleanest PyTorch path ([martimramos guide](https://github.com/martimramos/dgx-spark-ml-guide)). `docker run --gpus all -it --rm -v ~/phylomemetics:/w -v ~/phylo-data:/data nvcr.io/nvidia/pytorch:25.10-py3`, then install transformers and the fork inside it. Docker inside WSL on an RTX Spark is unverified.

### 3d. How the hooks follow Cammarata's method

Qwen's MLP is SwiGLU: `act = silu(gate_proj(x)) * up_proj(x)`, then `down_proj(act)`. A neuron is one channel of `act`. The scripts hook three places per layer, as neuron-explorer's `neuron_stats.py` does: the output of `gate_proj` (gate), the output of `up_proj` (up, the "value"), and the input of `down_proj` (act, the product). Recording gate and up separately lets every firing be put in its sign regime:

| Regime | gate | up | act | What it is |
| --- | --- | --- | --- | --- |
| open+ | > 0 | > 0 | > 0 | gate open, positive pole |
| open− | > 0 | < 0 | < 0 | gate open, negative pole |
| echo+ | < 0 | < 0 | > 0 | the double-negative echo: silu of a negative gate is a small negative number (≥ −0.28), so a large negative up gives a modest positive act |
| echo− | < 0 | > 0 | < 0 | the echo's negative counterpart |

This is the split the study's audit used (`study/cammarata-audit.md`): there, gate-closed echo firings filled much of the space below each unit's top 8 and dominated the scores that set the paper's presence threshold. neuron-explorer notes the echo holds the whole top of the positive list for about 4% of neurons. Recording act alone would mix the regimes; recording all three costs three times the memory for the dense smoke captures, and nothing extra for the top-K scan, which reads gate and up as they pass.

Other choices that follow the essay and neuron-explorer's notes:

- **Poles are separate units.** Positive and negative activations are separate objects (`L32 N1234+`, `L32 N1234-`); the top-K scan keeps each regime's top firings.
- **Raw activations, not percentiles,** stored as values; rarity can be shown later as "fires once every N pages".
- **Long context.** Windows of 1,024 tokens of packed documents; the essay notes that long-context neurons look polysemantic in short windows.
- **The first token is dropped,** because position 0 is an attention sink with outsized activations (neuron-explorer and the study's audits both do this).
- **Read around the peak, including before it.** Many neurons fire on the token before the thing; the scan stores positions so the full context can be read back.
- **Steering is part of the evidence,** as in the essay (turn the neuron up and read what the model writes). The HF hook adds a value to one channel of act; the vLLM fork adds `value × W_down[:, n]` to the MLP output, which is the same thing. `smoke_steer.py` checks the two agree.

### 3e. Storage layout

```
~/phylo-data/                         (PHYLO_DATA; inside WSL's ext4 disk on an RTX Spark, never /mnt/c)
  hf/                                 Hugging Face cache: model weights (Qwen3-32B 66 GB, Qwen3.6-35B-A3B 72 GB, 8B 16 GB, 4B 9 GB)
  src/vllm-steer/                     the steering fork (editable install)
  src/neuron-explorer/                optional: neuron-explorer, for its sweep scripts
  dense/<model>/smoke_L<l>.*          dense gate/up/act captures of small probe sets (safetensors, bf16) + json
  acts/<model>/<run>/                 corpus scans from record_topk.py
      state.json                      progress; --resume continues from the last checkpoint
      tokens/tokens-<batch>.npy       uint32 token ids of every window (4 GB per 1B tokens)
      L<layer>.npz                    per regime: top-K values and token positions [K, neurons], firing counts
  runs/                               steering rollouts (jsonl/json)
  logs/                               nohup / tmux logs
```

Sizes, so the disk can be checked (`df -h ~`): a Qwen3-32B scan with K = 64 keeps 1.64M neurons × 4 regimes × 64 × 12 bytes ≈ 5 GB of top-K, plus 4 GB of token ids per 1B tokens. Dense captures are what get big: one layer of Qwen3-32B is 25,600 × 3 × 2 bytes = 150 KB per token, so 10,000 tokens is 1.5 GB and all 64 layers would be 98 GB; keep dense captures to small probe sets (minimal pairs, blind maxima, near misses), and use the top-K scan for the corpus. Plan 300 GB free. Commit only small summaries to the repository's `study/results/`-style folders, never the shards.

### 3f. Remote access from the Mac

You will mostly drive the box from the Mac: SSH in, run long jobs in tmux, and run Claude Code on the box so it can read the outputs where they are.

**DGX Spark.** On the same network, `ssh <user>@<hostname>.local` works from the first boot, or use NVIDIA Sync on the Mac ([playbook](https://build.nvidia.com/spark)). Add your Mac's key: `ssh-copy-id <user>@<hostname>.local`.

**RTX Spark PC (WSL).** The SSH server runs inside WSL, so the Mac lands directly in Linux:

```bash
# inside Ubuntu (WSL)
sudo apt-get install -y openssh-server
sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config      # 22 may be taken by Windows' own OpenSSH
sudo systemctl enable --now ssh
```

```powershell
# in an Administrator PowerShell on Windows: let inbound connections reach WSL in mirrored mode
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```

The firewall command and mirrored mode are from Microsoft's [WSL networking docs](https://learn.microsoft.com/en-us/windows/wsl/networking); the narrower alternative there, `New-NetFirewallHyperVRule ... -LocalPorts 2222`, opens only the SSH port. Then from the Mac: `ssh -p 2222 <linux-user>@<windows-hostname>.local` (or the LAN IP from `check_machine.sh`). WSL starts when someone logs into Windows and opens it; `vmIdleTimeout=-1` keeps it up after that. **Check** that it survives a reboot without a login; if not, a Task Scheduler task at startup that runs `wsl.exe -d Ubuntu-24.04 --exec /bin/true` starts it.

**Away from home** on either machine: install [Tailscale](https://tailscale.com/download) (on Windows for an RTX Spark PC, since mirrored mode shares Windows' network; inside DGX OS for a DGX Spark) and on the Mac, then SSH to the Tailscale name. No ports opened to the internet.

**A Mac SSH config entry** (`~/.ssh/config`) saves typing:

```
Host spark
    HostName <hostname>.local        # or the Tailscale name
    User <linux-user>
    Port 2222                        # 22 on a DGX Spark
    ServerAliveInterval 30
```

**Claude Code on the box.** Install it in Linux (`curl -fsSL https://claude.ai/install.sh | bash`), log in once, and run it inside tmux over SSH (`ssh spark`, `tmux new -s claude`, `cd ~/phylomemetics && claude`), so a long session survives the Mac sleeping. The Claude desktop app can also open sessions on an SSH host, which keeps the app's interface while the commands run on the Spark. VS Code's Remote-SSH works too. Long jobs go in their own tmux window with a log: `nohup python gpu/record_topk.py ... > ~/phylo-data/logs/scan.log 2>&1 &`.

## 4. First smoke test: Qwen3-32B in bf16

Needs the 128 GB machine. On 64 GB, run the same steps with `--model Qwen/Qwen3-8B --layer 18`. Each step prints what it checked.

**Step 0: download the weights** (66 GB; the rest waits on it):

```bash
source ~/phylo-env.sh && source gpu/.venv/bin/activate
hf download Qwen/Qwen3-32B
```

**Step 1: record one layer** (layer 32 of 64, over 300 of the study's sentences, ~10,000 tokens):

```bash
python gpu/smoke_record.py --model Qwen/Qwen3-32B --layer 32 --n 300
```

It loads only layers 0–32 (34 GB), records gate, up and act for all 25,600 neurons at every token, and checks:

- `act` equals `silu(gate) * up` to bf16 rounding (error ≲ 0.004 of the batch max), which shows the three hooks saw the same tokens;
- a sentence read alone gives the same activations as inside a padded batch (≲ 0.05);
- the share of firings in each regime, and the share of neurons whose positive maximum here is an echo firing (high on so few tokens, because many neurons never open their gate; it falls with more text);
- the residual-norm quantiles at the layer, which set the steering doses.

It then prints the top firings of the five strongest neurons, with the peak token in `[[ ]]` and the work it came from, and a list of steering candidates: neurons whose top 8 gate-open firings mostly peak on one content word. Read them: on Batman summaries, expect some lexical neurons for names and frequent words, which is what the study's audit found in Qwen3-8B.

**Step 2: steer a candidate in transformers:**

```bash
python gpu/smoke_steer.py --model Qwen/Qwen3-32B --layer 32                 # first candidate
python gpu/smoke_steer.py --model Qwen/Qwen3-32B --layer 32 --neuron <n> --doses=-0.7,0.5,0.8,1.5,3
```

It generates three prompts greedily at each dose (in neuron-explorer's relative units: a dose of 0.8 pushes the residual stream by 0.8 times its median norm) and pushes a random direction just as hard as a control. It checks that a zero dose reproduces the unsteered text exactly, that adding to the neuron's activation and adding its `W_down` column give the same tokens, and whether the neuron's word appears more often under positive doses than under the baseline and the control.

**What to expect.** Run on the Mac with Qwen3.5-4B (layer 12, the neuron whose top firings peak on "Batman"), the first two checks passed, and doses up to +0.8 changed the wording but did not make the model mention Batman. That is a normal result: a neuron that *reads* a word need not *write* it, and the essay and neuron-explorer both find the steerable neurons by steering many of them. If a candidate does nothing, try larger doses (neuron-explorer's free-form prompt uses +3) and the other candidates. The test passes when the checks pass and at least one candidate moves the text towards its concept more than the random control does.

**Step 3: the same steering in vLLM** (a separate process; HF and vLLM each need a full 66 GB copy, so they cannot both be loaded):

```bash
python gpu/smoke_vllm_steer.py --run ~/phylo-data/runs/smoke_steer_L32_N<n>.json
```

Greedy HF and vLLM outputs part after some tokens even unsteered, because their bf16 kernels round differently. The fork steers the same neuron the same way if the steered pairs agree about as long as the unsteered pair. Lower `--gpu-mem` (default 0.8 of the unified memory) if vLLM fails to start because Windows or the page cache holds memory.

**Step 4: measure the real recording speed** (about an hour; it streams FineWeb-Edu and UltraChat, as neuron-explorer does):

```bash
python gpu/record_topk.py --model Qwen/Qwen3-32B --tokens 2000000 --run bench-q3-32b
```

It prints tokens/s and the time for 100M, 1B and 10B tokens. Put those in `gpu/MACHINE.md` and in section 2. A first real scan, matching the essay's 100M tokens:

```bash
nohup python gpu/record_topk.py --model Qwen/Qwen3-32B --tokens 100000000 --run q3-32b-100M --k 64 \
    > ~/phylo-data/logs/q3-32b-100M.log 2>&1 &
# after an interruption, the same command with --resume
```

Two things are not done yet and come after the smoke test: a reader that turns `L<layer>.npz` positions back into highlighted text (the format is set out in `record_topk.py`), and porting the study's `cammarata_mapping.py` (minimal pairs, blind explanations, blind maxima, near misses) from MLX to these hooks.

### Running neuron-explorer's sweep here

The environment matches neuron-explorer's pod setup, so its sweep runs on the Spark for Qwen3.5-4B without a pod: clone it to `~/phylo-data/src/neuron-explorer` and run `sweep/neuron_stats.py`, `sweep/check_equivalence.py` and `sweep/steer_sweep.py` with `gpu/.venv` active, as in its README. Its `setup_pod.sh` writes to `/workspace`; skip that script, since `setup_env.sh` has done the same work.

## 5. Open questions that depend on the actual machine

1. **Which machine and how much memory?** RTX Spark PC or DGX Spark; 128, 64 or 32 GB. 64 GB rules out Qwen3-32B and Qwen3.6-35B-A3B in bf16 (section 1 table).
2. **On an RTX Spark PC: how much memory can CUDA use inside WSL?** It depends on WSL's `memory=` limit, Windows' GPU memory split and the driver; none of it is documented for these machines. Qwen3-32B in bf16 needs more than ~80 GiB. `check_gpu.py` answers it.
3. **Does CUDA in WSL on Windows on Arm work with the PyTorch and vLLM aarch64 wheels,** and which CUDA version does the driver report? Reviewers ran vLLM inside WSL on RTX Spark laptops, but CUDA on Windows on Arm is a developer preview (13.4). If WSL CUDA fails, the fallbacks are the NGC container, or renting.
4. **Compute capability.** GB10 is 12.1; the RTX Spark chip is probably the same but not documented. It decides whether the cu130 wheels have kernels for it (`check_gpu.py` warns if not).
5. **GPU timeouts.** Whether the preview's PyTorch timeouts appear on long scans, and whether `TdrDelay` needs changing.
6. **Sustained speed.** Real tokens/s for recording and for batched steering, and whether a laptop holds it over days (`record_topk.py` benchmark, then a 100M-token run). The time estimates in section 2 should be replaced by these.
7. **Disk.** Free space after the four models (~165 GB of weights) and whether an external SSD is needed for 1B+ token scans.
8. **Remote access.** Whether WSL starts and sshd answers after a reboot without anyone logging in.
9. **Qwen3.6-35B-A3B hooks.** Whether transformers exposes per-expert `gate_proj`/`up_proj` modules for this model and whether the fork's routed-experts capturer gives the expert hidden layer. Independent of the machine, but it decides whether the MoE model is a hooks job or a kernel job.

## Sources

- Cammarata, *Neural Networks Are Already Clean*, Goodfire: https://already-clean.goodfire.com/
- neuron-explorer (`sweep/setup_pod.sh`, `sweep/neuron_stats.py`, `sweep/steer_sweep.py`, `sweep/check_equivalence.py`, `NOTES.md`): https://github.com/lyramakesmusic/neuron-explorer
- RhizoNymph/vllm steering fork and its docs: https://github.com/RhizoNymph/vllm (branch `feat/integration`); precompiled wheels: https://wheels.vllm.ai/755492e37d7d7201d93f0effa200acac56601f38/
- NVIDIA RTX Spark product page: https://www.nvidia.com/en-us/products/rtx-spark/
- NVIDIA newsroom, RTX Spark PCs available (16 Oct 2026): https://nvidianews.nvidia.com/news/nvidia-rtx-spark-pcs-available-today-a-new-beginning-for-windows-pcs
- NVIDIA developer blog, RTX Spark development: https://developer.nvidia.com/blog/code-test-and-deploy-on-one-machine-with-nvidia-rtx-spark/
- NVIDIA RTX Spark dev-environment playbook: https://build.nvidia.com/playbooks/setting-up-your-dev-environment
- Wikipedia, Nvidia RTX Spark: https://en.wikipedia.org/wiki/Nvidia_RTX_Spark
- XDA, five things about RTX Spark laptops: https://www.xda-developers.com/things-nobody-tells-you-before-buying-an-rtx-spark-laptop/
- MindStudio, MSI RTX Spark laptop review: https://www.mindstudio.ai/blog/msi-rtx-spark-laptop-review
- LeCompute, RTX Spark on Windows: CUDA 13.4, PyTorch, vLLM and WSL: https://lecompute.fr/en/silicon/rtx-spark-windows-pile-llm/
- Linux on an RTX Spark laptop (summary): https://daily.dev/posts/putting-linux-on-my-rtx-spark-laptop-because-i-don-t-have-a-microsoft-account-alokbhvko
- NVIDIA DGX Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html and known issues: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
- NVIDIA DGX Spark playbooks: https://build.nvidia.com/spark
- DGX Spark bf16 benchmark (Gemma 31B prefill and decode): https://forums.developer.nvidia.com/t/gemma-4-day-1-inference-on-nvidia-dgx-spark-preliminary-benchmarks/365503
- vLLM on DGX Spark, community thread: https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862
- DGX Spark setup guides: https://github.com/natolambert/dgx-spark-setup, https://github.com/martimramos/dgx-spark-ml-guide
- Microsoft, WSL networking (mirrored mode, Hyper-V firewall): https://learn.microsoft.com/en-us/windows/wsl/networking
- Microsoft, TDR registry keys: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- Model configs: https://huggingface.co/Qwen/Qwen3-32B, https://huggingface.co/Qwen/Qwen3-8B, https://huggingface.co/Qwen/Qwen3.5-4B, https://huggingface.co/Qwen/Qwen3.6-35B-A3B, https://huggingface.co/moonshotai/Kimi-K3
