# SpeechGR

End-to-end speech generative retrieval on SLUE-SQA5 with HuBERT layer-22 K=500
units. This repo is set up so a new server can prepare the unit dataset, optionally
pretrain the unit T5 backbone, then train the GR model.

For more detailed handoff notes and debugging commands, see
[docs/TRAINING_HANDOFF.md](docs/TRAINING_HANDOFF.md).

## Quickstart: launch training

### 0) Set up Python

```bash
git clone git@github.com:dodofk/SpeechGR.git
cd SpeechGR

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Download or preprocess the unit dataset

If you just want to use the prepared packed SLUE-SQA5 unit dataset, run:

```bash
python3 scripts/download_unit_dataset.py
python3 scripts/smoke_test_dataloaders.py
```

If you are starting from newly extracted or legacy per-utterance unit files,
pack them first and then point training commands at that packed directory:

```bash
python3 scripts/pack_unit_codes.py \
  --input-dir /path/to/slue_sqa_code_l22_c500_legacy \
  --output-dir /data/slue_sqa_code_l22_c500

python3 scripts/smoke_test_dataloaders.py \
  --code-dir /data/slue_sqa_code_l22_c500
```

### 2) Optional: unit-T5 pretraining

Skip this step if you already have a unit-pretrained checkpoint. Use this when
you want to pretrain the audio/unit T5 backbone on a server.

```bash
export WANDB_API_KEY="..."
export HF_TOKEN="..."

# Optional: mirror latest/ and best/ checkpoints to a public HF model repo.
# Leave HF_CHECKPOINT_REPO_ID unset to disable Hub checkpoint mirroring.
export HF_CHECKPOINT_REPO_ID="your-hf-user/speechgr-unit-t5-live"
export HF_CHECKPOINT_PRIVATE="False"
export HF_CHECKPOINT_MODE="model"

# Optional shared cache location on a cluster/server.
export HF_HOME="/data/hf_cache"

bash run_t5_pt.sh
```

Useful switches:

```bash
# Debug run without periodic local checkpoints or final save.
SAVE_CHECKPOINTS=False SAVE_FINAL_MODEL=False bash run_t5_pt.sh
```

### 3) Train SpeechGR / DSI retrieval

Run GR training with the default unit dataset and the configured unit-T5
checkpoint path:

```bash
export WANDB_API_KEY="..."
export HF_TOKEN="..."

# Optional: mirror latest/ and best/ GR checkpoints to a public HF model repo.
export HF_CHECKPOINT_REPO_ID="your-hf-user/speechgr-gr-live"
export HF_CHECKPOINT_PRIVATE="False"
export HF_CHECKPOINT_MODE="model"
export HF_HOME="/data/hf_cache"

# If your unit-pretrained checkpoint is somewhere else, override it here.
export MODEL_PATH="/path/to/audio-t5-pt/checkpoint-or-best"

bash run.sh
```

Useful switches:

```bash
# Disable periodic local checkpoints and Hub mirroring for quick debugging.
SAVE_CHECKPOINTS=False bash run.sh
```

## Optional QG experiment

For pseudo-query / UnitQG experiments, use the same environment template and run:

```bash
bash run_qg.sh
```

`run_qg.sh` also supports `MODEL_PATH=/path/to/unit-t5-checkpoint` and the same
`HF_CHECKPOINT_*`, `SAVE_CHECKPOINTS`, and `SAVE_FINAL_MODEL` switches.
