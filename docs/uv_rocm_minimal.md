# Minimal UV Setup for AMD 7900 XTX (ROCm)

## Overview

Use `uv` to manage the environment, but install PyTorch from ROCm index instead of CUDA.

---

## Step 1: System Setup (One-time)

```bash
# Install ROCm on the AMD server
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.2-602_all.deb
sudo dpkg -i amdgpu-install_6.0.2-602_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -a -G render,video $USER
# Reboot
sudo reboot

# Verify ROCm
rocminfo | grep gfx1100  # Should show gfx1100 for 7900 XTX
```

---

## Step 2: Create UV Environment with ROCm PyTorch

```bash
# Navigate to project
cd /path/to/SpeechGR

# Create venv with uv (Python 3.10 recommended for ROCm)
uv venv --python 3.10

# Activate
source .venv/bin/activate

# Install PyTorch with ROCm support (CRITICAL: use ROCm index)
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/rocm6.0

# Install rest of dependencies (excluding torch)
uv pip install \
  accelerate \
  datasets \
  evaluate \
  "hydra-core>=1.3" \
  ipykernel \
  joblib \
  librosa \
  numpy \
  pandas \
  rouge-score \
  sentencepiece \
  soundfile \
  tqdm \
  "transformers==4.44.2" \
  wandb \
  einops \
  torchcodec \
  rank-bm25 \
  sentence-transformers \
  "lightning>=2.6.1"

# Install project in editable mode
uv pip install -e .
```

---

## Step 3: Required Environment Variable

Add to `~/.bashrc` or set before every run:

```bash
# Critical for 7900 XTX recognition
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Or create a wrapper script `run_rocm.sh`:

```bash
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# Run with uv
uv run "$@"
```

---

## Step 4: Verify Installation

```bash
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print('GPU computation: OK')
"
```

Expected output:
```
PyTorch: 2.1.1+rocm6.0
CUDA/ROCm available: True
Device: AMD Radeon RX 7900 XTX
GPU computation: OK
```

---

## Step 5: Run Training

```bash
# Set env var and run
export HSA_OVERRIDE_GFX_VERSION=11.0.0
uv run python scripts/phase0_prep/train_rqvae.py \
  --config-path configs/training \
  --config-name rqvae_sliding_window \
  data.manifest_path=/path/to/manifest.txt
```

Or use the wrapper:
```bash
./run_rocm.sh python scripts/phase0_prep/train_rqvae.py ...
```

---

## Minimal Changes Summary

| Current (CUDA) | ROCm |
|----------------|------|
| `uv venv` | Same |
| `uv pip install -e .` | Install PyTorch separately first, then rest |
| No env var | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| `torch` from default | `torch` from `https://download.pytorch.org/whl/rocm6.0` |

---

## Alternative: Modify pyproject.toml

Create `pyproject.rocm.toml`:

```toml
[project]
name = "speechgr"
version = "0.1.0"
description = "Speech-grounded generative retrieval experiments"
readme = "README.md"
requires-python = "~=3.10.0"
dependencies = [
    "torch==2.1.1+rocm6.0",  # Specify ROCm version
    "torchvision==0.16.1+rocm6.0",
    "torchaudio==2.1.1+rocm6.0",
    # ... rest of deps
]

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm6.0"
```

Then:
```bash
uv pip install -e . --index pytorch-rocm
```

---

## Quick Reference

### Full Setup Commands

```bash
# 1. System ROCm (one-time)
sudo amdgpu-install --usecase=rocm

# 2. UV environment
cd SpeechGR
uv venv --python 3.10
source .venv/bin/activate

# 3. Install ROCm PyTorch
uv pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/rocm6.0

# 4. Install other deps
uv pip install -r requirements.txt  # Without torch

# 5. Env var
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 6. Run
uv run python scripts/phase0_prep/train_rqvae.py ...
```

### Check GPU

```bash
rocm-smi  # GPU usage
rocminfo | grep gfx  # GPU model
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No GPU detected` | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| `HIP error` | `export ROCM_USE_FLASH_ATTN_V2_SCAN=0` |
| OOM | Reduce `batch_size` in config (try 16) |
| Slow | Normal - ROCm ~70-80% of CUDA speed |
| SSL model error | Some ops may need `PYTORCH_ROCM_ARCH=gfx1100` |

---

## Summary

**Only 3 things change:**
1. Install ROCm PyTorch separately: `uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0`
2. Set `export HSA_OVERRIDE_GFX_VERSION=11.0.0` before running
3. Use Python 3.10 (better ROCm support than 3.12)

Everything else (`uv run`, `uv pip install`, project structure) stays the same!
