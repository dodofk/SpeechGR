# SpeechGR

## S2GSR-UnitY: Generative Retrieval

This project implements the **S2GSR-UnitY** architecture for zero-resource speech-to-ID retrieval. It uses a dual-decoder model (Semantic Bridge + Retrieval Head) to map audio queries directly to document IDs using a Deep RQ-VAE neural index.

### Quick Start (Smoke Test)
Verify your environment and the model pipeline with dummy data:
```bash
uv run python scripts/test/smoke_test_unity.py
```

### S2GSR-UnitY: Master Workflow

The system operates in three distinct phases to ensure stability and scalability.

#### Phase 0: Neural Indexing (Offline)
Before training the main model, you must generate the document IDs and semantic summaries:
1.  **Train RQ-VAE**: Follow the "RQ-VAE Training" section below to train a document-level quantizer.
2.  **Train K-Means**: Train a K-Means model for semantic token clustering:
    ```bash
    uv run python scripts/phase0_prep/generate_indices.py \
        --audio_root /path/to/librispeech \
        --output_dir outputs/indices/v1 \
        --train_kmeans --kmeans_k 500
    ```
3.  **Generate Manifests**: Use the indexing scripts to process your corpus and save `id_map.json` and `semantic_map.json`:
    ```bash
    uv run python scripts/phase0_prep/generate_indices.py \
        --audio_root /path/to/librispeech \
        --output_dir outputs/indices/v1 \
        --rqvae_checkpoint /path/to/rqvae.pt \
        --kmeans_model outputs/indices/v1/kmeans_model.pkl
    ```
    
#### Phase 1: Dual-Task Pre-training
Train the model on LibriSpeech to both **Memorize** (Indexing) and **Retrieve** (random crops):
```bash
uv run python scripts/phase1_train/train_unity.py \
    --id_map outputs/indices/v1/id_map.json \
    --semantic_map outputs/indices/v1/semantic_map.json \
    --audio_root /path/to/librispeech \
    --batch_size 16 \
    --lambda_ret 2.0
```

#### Phase 2: Fine-tuning
Adapt the model to real questions (e.g., SLUE-SQA5) while maintaining the index stability using the Replay Buffer (managed automatically by the dataset).

### Management Strategy: Decoupled Manifests
To manage the complexity of multi-stage experiments:
- **Immutable Indices**: Once an `id_map.json` is generated for a corpus version, treat it as a fixed dataset.
- **Text-like Pipeline**: The main training script interacts only with the JSON manifests and raw audio. This decouples the "retrieval target" from the "acoustic model," allowing you to swap WavLM versions or RQ-VAE depths simply by pointing to a different JSON file.
- **Versioning**: Store manifests in `outputs/indices/{version_name}/` to track which index corresponds to which experiment run.

## Environment Setup (uv)
SpeechGR uses [uv](https://github.com/astral-sh/uv) to manage Python 3.12 environments and dependencies defined in `pyproject.toml`.

```bash
# Install uv if you do not already have it (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# For Homebrew users:
# brew install uv

# From the repository root
uv sync                # create the virtualenv and install all dependencies
uv run python -m pip --version  # sanity check inside the env
```

To run commands inside the environment:

```bash
uv run python -m speechgr.cli.train task=retrieval
uv run python -m pytest
```

Regenerate the lockfile whenever dependencies change:

```bash
uv lock --upgrade-package transformers
uv sync
```

### Notebooks
Any `.ipynb` notebooks checked into the repo (e.g., `explore.ipynb`) can be launched inside the uv environment:

```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

This ensures the kernel sees the same dependencies as the CLI runners.

## AMD GPU Setup (ROCm)

For AMD GPUs (e.g., Radeon 7900 XTX), you need ROCm instead of CUDA. The setup is nearly identical, with two key differences:

### 1. Install ROCm System Dependencies

```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.2-602_all.deb
sudo dpkg -i amdgpu-install_6.0.2-602_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -a -G render,video $USER
sudo reboot
```

Verify installation:
```bash
rocminfo | grep gfx1100  # Should show gfx1100 for 7900 XTX
rocm-smi                 # GPU status
```

### 2. Create UV Environment with ROCm PyTorch

```bash
# Use Python 3.10 (better ROCm support)
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch with ROCm support (CRITICAL)
uv pip install torch==2.1.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0

# Install remaining dependencies
uv pip install -e .
```

### 3. Required Environment Variable

**Critical:** Set this before every run or add to `~/.bashrc`:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

You can also create a helper script `run_rocm.sh`:
```bash
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
uv run "$@"
```

Then use: `./run_rocm.sh python scripts/phase0_prep/train_rqvae.py ...`

### Verification

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
uv run python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
# Should output: AMD Radeon RX 7900 XTX
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `No GPU detected` | Ensure `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set |
| `HIP error` | `export ROCM_USE_FLASH_ATTN_V2_SCAN=0` |
| Out of memory | Reduce `batch_size` in config (try 16 instead of 32) |
| Slow performance | Normal - ROCm is ~70-80% of CUDA speed |

### SLUE SQA5 Prep
Generate CSV manifests and encoder caches for the SLUE SQA5 splits with:

```bash
UV_PYTHON=3.12 uv run python -m speechgr.cli.prepare_slue output_root=outputs/slue_wavtok
```

Override encoder parameters (e.g., switch to HuBERT) via `encoder.question.params.*` and `encoder.document.params.*`.

## Fairseq Requirement
HuBERT-based precomputation depends on the `fairseq` toolkit. Install it manually before running the HuBERT utilities:

```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install -e .
```

Then point `fairseq_root` in the Hydra configs (e.g., `configs/precompute_hubert.yaml`) to the location of this checkout so the HuBERT encoder can import `examples/hubert` utilities.

## RQ-VAE Training and Usage
SpeechGR supports training RQ-VAE (Residual Quantization Variational Autoencoder) on top of SSL features (e.g., HuBERT, WavLM) and using it as a discrete encoder.

### 1. Download Dataset (LibriSpeech)
For Phase 0 training, we recommend starting with LibriSpeech-100:
```bash
mkdir -p data/librispeech && cd data/librispeech
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvf train-clean-100.tar.gz
# (Optional) Download dev-clean for validation
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xvf dev-clean.tar.gz
cd ../..
```

### 2. Prepare Manifests
The RQ-VAE training script requires a `.txt` manifest of absolute audio paths:
```bash
# Training manifest
uv run python scripts/phase0_prep/create_manifest.py \
    --audio_root data/librispeech/LibriSpeech/train-clean-100 \
    --output_path data/train_manifest.txt

# Validation manifest
uv run python scripts/phase0_prep/create_manifest.py \
    --audio_root data/librispeech/LibriSpeech/dev-clean \
    --output_path data/val_manifest.txt
```

### 3. Training RQ-VAE
Train the model using the Hydra config:
```bash
uv run python scripts/phase0_prep/train_rqvae.py \
    --config-name training/rqvae \
    data.manifest_path=data/train_manifest.txt \
    data.val_manifest=data/val_manifest.txt
```

### 4. Recommended RQ-VAE Experiment Plan (S2GSR-Unity)
If current retrieval quality is unstable, use the following staged plan before spending budget on full phase-1 Unity training.

#### Step A: Lock a Reproducible Baseline
1. Train with the same config/seed recipe 2-3 times:
```bash
uv run python scripts/phase0_prep/train_rqvae.py \
    --config-name training/rqvae_sliding_window \
    data.manifest_path=data/train_manifest.txt \
    data.val_manifest=data/val_manifest.txt \
    training.epochs=50
```
2. Track and archive at least:
- `codebook/avg_utilization`
- `codebook/avg_perplexity`
- `recon/snr_db`
- `train/loss_total`
3. Keep this as the control run (`B0`).

#### Step B: Aggregation Ablation First (Cheap, High Impact)
Run retrieval-code ablations before architecture-heavy changes.

Current built-in options in `SlidingWindowDocumentRQVAE.encode()`:
- `aggregate_for_retrieval=mean`
- `aggregate_for_retrieval=first`
- `aggregate_for_retrieval=all`

Suggested order:
1. `B0`: `mean` (control)
2. `B1`: `first`
3. `B2`: `all` (for multi-vector experiments)

Keep all other hyperparameters fixed while comparing these modes.

#### Step C: Add Retrieval-Aligned Objective (Optional but Recommended)
If reconstruction looks good but retrieval is weak, add a small retrieval-aligned auxiliary loss in phase-0 RQ-VAE training (for example, contrastive regularization on document representations).

Guideline:
- Start with low weight (e.g., `0.05` to `0.1`)
- Ramp up after warmup
- Ensure reconstruction metrics do not collapse

#### Step D: Promotion Gates (Fail Fast)
Only promote a phase-0 run to phase-1 Unity when all gates pass:
- `codebook/avg_utilization >= 0.35`
- `codebook/avg_perplexity` remains stable and non-collapsed
- `recon/snr_db >= 10` (or your team baseline)
- Retrieval proxy improves over `B0` (e.g., `Recall@10` improvement)

If a run fails these gates, stop early and adjust aggregation/loss setup before full Unity training.

#### Step E: Budget Rule for Phase-1 Unity
Do **not** run full `scripts/phase1_train/train_unity.py` on every phase-0 variant.
Only train phase-1 for the top 1-2 phase-0 candidates that pass Step D.

This saves substantial compute and avoids tuning phase-1 on weak document codes.

### Using RQ-VAE in Experiments
To use the trained RQ-VAE as an encoder in SpeechGR experiments, configure the `rqvae` encoder in your experiment config:

```yaml
encoder:
  name: rqvae
  ssl_model_name: "facebook/hubert-base-ls960"  # Must match training
  rqvae_checkpoint: "/path/to/rqvae_final.pt"
  rqvae_config:
    latent_dim: 768       # Match SSL feature dim
    codebook_size: 1024   # Match training
    num_codebooks: 4      # Match training
```
