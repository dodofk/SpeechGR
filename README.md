# SpeechGR

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
