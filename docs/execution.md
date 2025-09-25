# Running SpeechGR Workflows

SpeechGR now uses Hydra for configuration and uv for environment management. The commands below assume Python 3.12 and that you have already installed uv (`pip install uv` or prebuilt binaries).

## 1. Bootstrap the Environment

```bash
uv sync
```

- Installs all runtime and optional developer dependencies defined in `pyproject.toml`.
- Creates a `.venv` folder managed by uv (no conda or manual pip required).

## 2. Launch Generative Retrieval (`speechgr.cli.train`)

```bash
uv run python -m speechgr.cli.train task=retrieval
```

Key overrides:
- Change dataset location: `uv run python -m speechgr.cli.train task=retrieval data.dataset_path=/custom/slue_sqa5`
- Load a specific checkpoint: `uv run python -m speechgr.cli.train task=retrieval model.model_path=ckpts/audio-t5.pt`
- Short smoke test: `uv run python -m speechgr.cli.train task=retrieval training.training_args.max_steps=200`
- Previous shell-script configuration is preserved as a Hydra preset: `uv run python -m speechgr.cli.train experiment=retrieval/slue_sqa5`
- Transcript-only baseline (text modality): `uv run python -m speechgr.cli.train experiment=retrieval/slue_sqa5_text`

Artifacts (`eval_results.json`, `eval_raw.json`) remain in the repository root. WandB logging uses the project defined in `configs/logging/wandb.yaml`.

## 3. Ranking Refinement (`speechgr.cli.train`)

```bash
uv run python -m speechgr.cli.train task=ranking
```

Useful overrides:
- Validation-only inference: `ranking.do_inference=true`
- Debug subset (first 100 samples): `ranking.do_debug=true`
- Alternate codebook path: `data.code_path=/home/ricky/dodofk/dataset/slue_sqa_code_c512`
- Full SLUE ranking recipe: `uv run python -m speechgr.cli.train experiment=ranking/slue_sqa5`

## 4. Q-Former Training (`speechgr.cli.train`)

```bash
uv run python -m speechgr.cli.train task=qformer
```

Toggle Whisper features by setting `model.use_whisper_features=true`. When enabled, the script will stream audio via `SlueSQA5WhisperDataset`; otherwise it falls back to discrete units configured in `data.code_path`.
Use the legacy SLUE profile via `experiment=qformer/slue_sqa5`.

## 5. Query Generation

```bash
uv run python -m speechgr.cli.train task=qg
```

Apply the prepared experiment preset instead with `uv run python -m speechgr.cli.train experiment=qg/slue_sqa5`.

## 6. T5 Pretraining

```bash
uv run python -m speechgr.cli.train task=t5_pretrain
```

The full SLUE preset lives at `experiment=t5_pretrain/slue_codes`.

## 7. Feature Extraction Utilities

```bash
uv run python scripts/save_ssl_feature.py --output_dir data/ssl_demo --limit 50
```

Adjust `--output_dir` and dataset switches as needed. The script reads Hydra configs only indirectly; pass CLI flags for quick experiments.

To precompute Whisper encoder features via the new modality encoder abstraction:

```bash
uv run python -m speechgr.cli.precompute dataset.split=train output_dir=precomputed/whisper/train
```

- Outputs a single cache file per split (e.g., `train_whisper.pt`) so training can reuse features without recomputation.
- HuBERT k-means units: `uv run python -m speechgr.cli.precompute --config-name precompute_hubert encoder.params.sample_id_field=question_id encoder.params.audio_field=question_audio output_dir=precomputed/hubert/question`
- SLUE preparations (CSV manifests + encoder caches):
  ```bash
  uv run python -m speechgr.cli.prepare_slue output_root=outputs/slue_wavtok
  ```
  Override encoder parameters via `encoder.question.params.*` or `encoder.document.params.*` when switching between discrete unit generators.

## Notes on Paths

- Default dataset paths still point to `/home/ricky/dodofk/...` so existing jobs stay reproducible. Update the relevant `configs/data/*.yaml` entries once the shared storage location is finalized.
- Hydra output directories are pinned to the current working directory to preserve legacy artifact layouts.
- Ready-made presets for legacy training flows live under `configs/experiment/`; run them via `uv run python -m speechgr.cli.train experiment=<group/name>`.
