# Repository Guidelines

## Project Structure & Module Organization
SpeechGR is packaged under `speechgr/`. Entry points in `speechgr/cli/` cover generative retrieval, ranking, question generation, Q-former tuning, and T5 pretraining. `train.py` selects a task via Hydra, while `precompute.py` caches encoder features. Retrieval is strictly generative—legacy dense-index runners (e.g., `run.py`) are retired in favor of sequence-to-sequence decoding. Datasets derive from `speechgr/data/datasets.py`, share collators in `speechgr/data/collators.py`, and rely on modality encoders grouped in `speechgr/encoders/`. Experiments and defaults live in `configs/` (`configs/task/`, `configs/experiment/`, `configs/precompute.yaml`). Utility scripts sit in `scripts/`; historic runs stay under `baseline_exp/`. External helper libraries (fairseq HuBERT utilities, archived configs) are mirrored under `inventory/` for quick reference without polluting the package namespace.

## Environment, Build & Run Commands
Target Python 3.12 and manage dependencies with uv: `uv sync` (install/lock), `uv run python -m speechgr.cli.train task=retrieval` (generative retrieval smoke), `uv run python -m speechgr.cli.precompute dataset.split=train` (feature cache), `UV_PYTHON=3.12 uv run python -m speechgr.cli.prepare_slue output_root=outputs/slue_wavtok` (SLUE CSV + precompute scaffolding), and `uv run python -m pytest` (unit tests). Use Hydra overrides inline, e.g. `uv run python -m speechgr.cli.train task=retrieval data.dataset_path=/path/to/slue`. Regenerate `uv.lock` after dependency bumps so CI matches local state.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, snake_case for functions, CamelCase for classes, and explicit type hints for public APIs. Keep data classes in `speechgr/data/` and encoders in modality subpackages (`encoders/whisper`, `encoders/discrete`, etc.). Format code with `uv run python -m black path/to/file` and `uv run python -m isort path/to/file`. Record new runtime dependencies in `pyproject.toml` and mirror notable additions in this guideline for quick reference.

## Testing Guidelines
Tests live beside code under `tests/` or the relevant package module. Write scenario-focused cases that cover dataset/encoder boundaries and Hydra configs. Name tests `test_<behavior>.py` and ensure fixtures mock external storage. Continuous smoke tests run via `uv run python -m pytest`; add task-specific checks such as `training.training_args.max_steps=20` for quicker CI validation.

## Commit & Pull Request Guidelines
Keep commit messages imperative and under 50 chars (e.g., `align generative retriever config`). PRs must summarize experiment goals, list reproducible uv commands, link Hydra configs or overrides, surface WandB run IDs, and call out new datasets or checkpoints. When refactoring, describe impacts on modality encoders and dataset schemas so downstream agents can adapt.

## Dependency Snapshot
Baseline runtime stack: torch, torchaudio, transformers ≥4.44, hydra-core ≥1.3, accelerate, datasets, sentencepiece, wandb, numpy, pandas, librosa, soundfile, evaluate, rouge-score, joblib, tqdm. Dev extras: black, isort, pytest. Manage everything through uv; avoid ad-hoc `pip install` to keep environments reproducible.
