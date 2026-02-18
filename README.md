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

### Training RQ-VAE
Train an RQ-VAE model using the provided script. You need a manifest file containing paths to audio files (one per line).

```bash
# Train on HuBERT-base features
python scripts/train_rqvae.py \
    ssl.model_name="facebook/hubert-base-ls960" \
    rqvae.codebook_size=1024 \
    rqvae.num_codebooks=4 \
    data.manifest_path=/path/to/train.manifest \
    data.val_manifest=/path/to/val.manifest  # Optional validation
```

Key arguments:
*   `ssl.model_name`: HuggingFace model ID for the upstream SSL model (default: `facebook/hubert-base-ls960`).
*   `rqvae.codebook_size`: Number of codes per codebook.
*   `rqvae.num_codebooks`: Number of residual quantization levels.
*   `training.save_steps`: Frequency of checkpoint saving.

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
