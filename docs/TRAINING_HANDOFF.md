# SpeechGR Training Handoff

This is the short launch guide for running the HuBERT layer-22 K=500 unit
training path on a server.

## Assumptions

- Repo is checked out on the training server.
- Python dependencies are installed from `requirements.txt` or `pyproject.toml`.
- GR and QG need `DATASET_PATH` set to the SLUE-SQA5 CSV metadata directory.
  The required files under that directory include:

```text
train.csv
validation.csv
test.csv
verified_test.csv
slue_sqa5_corpus.csv
slue_sqa5_pq10_llama32_3b_clean.csv
```

`run_t5_pt.sh` generates `ckpts/token_lookups/flan-t5-base-c500-l22-token-lookup.txt`
if it does not already exist. This lookup maps HuBERT unit ids `0..499` to
actual T5 vocabulary ids, avoids T5 special/sentinel ids, and is stored in
unit-T5 checkpoint configs so GR/QG can continue from the checkpoint with the
same unit vocabulary. Unit-T5 pretraining does not require `DATASET_PATH`.

The discrete unit dataset is downloaded from Hugging Face by default:

```text
hf://datasets/dodofk/slue-sqa-code-l22-c500
```

## First-Time Setup

Use one of these environment setups.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Recommended project-specific cache for servers:

```bash
export HF_HOME=/data/$USER/hf_cache/speechgr
mkdir -p "$HF_HOME"
```

## Verify Before Training

Prefetch the packed unit dataset:

```bash
python3 scripts/download_unit_dataset.py
```

Run a tiny trainer smoke. This validates the DSI and QG Trainer paths without
starting a real experiment:

```bash
python3 scripts/smoke_train_units.py \
  --mode both \
  --max-steps 2 \
  --batch-size 1 \
  --max-length 64 \
  --label-max-length 32 \
  --truncate-offset 8
```

If units are already materialized locally:

```bash
python3 scripts/smoke_train_units.py \
  --mode both \
  --code-dir /data/slue_sqa_code_l22_c500 \
  --max-steps 2 \
  --batch-size 1 \
  --max-length 64 \
  --label-max-length 32 \
  --truncate-offset 8
```

## Launch QG

Default QG training:

```bash
export DATASET_PATH=/path/to/slue_sqa5_metadata
bash run_qg.sh
```

QG initialized from a unit-pretrained checkpoint:

```bash
MODEL_PATH=ckpts/audio-t5-pt-flant5-base-c500-l22/checkpoint-219000 \
bash run_qg.sh
```

QG with public Hugging Face latest/best monitoring:

```bash
HF_TOKEN=... \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-live \
bash run_qg.sh
```

Joyboy/data-server QG from the 50k unit-pretrained checkpoint, with local
checkpoint saving plus Hugging Face latest/best mirroring:

```bash
cd /home/ricky/SpeechGR
source .venv/bin/activate

export DATASET_PATH=/home/ricky/SpeechGR/data/slue_sqa5_metadata
export CODE_DIR=hf://datasets/dodofk/slue-sqa-code-l22-c500
export MODEL_NAME_OR_PATH=google/flan-t5-base
export MODEL_PATH=ckpts/audio-t5-pt-flant5-base-c500-l22/checkpoint-50000

export HF_HOME=/storage/ricky/speechgr/hf_cache
mkdir -p "$HF_HOME"

export WANDB_API_KEY=...
export HF_TOKEN=...
export HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-unitpt50k-live
export HF_CHECKPOINT_PRIVATE=False
export HF_CHECKPOINT_MODE=model

bash run_qg.sh
```

Local QG checkpoints are saved under `ckpts/flan-t5-QG/`, and the final model is
saved to `ckpts/flan-t5-querygen/`. Leave `HF_CHECKPOINT_REPO_ID` unset if you
only want local server checkpoints.

## QG Augmentation Then GR

Build a separate augmented training set from the best QG checkpoint:

```bash
cd /home/ricky/SpeechGR
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/storage/ricky/speechgr/hf_cache
export PYTHON_BIN=.venv/bin/python
export DATASET_PATH=data/slue_sqa5_metadata
export CODE_DIR=hf://datasets/dodofk/slue-sqa-code-l22-c500
export QG_MODEL_PATH=ckpts/flan-t5-QG-unitpt50k-b12/checkpoint-10000
export OUTPUT_DATASET_PATH=data/slue_sqa5_qg_aug_unitpt50k_ckpt10000
export OUTPUT_CODE_DIR=data/slue_sqa5_qg_aug_unitpt50k_ckpt10000_codes

bash run_qg_augment.sh
```

For multi-query augmentation, enable sampling and set how many outputs to keep
per document chunk:

```bash
export DO_SAMPLE=True
export NUM_RETURN_SEQUENCES=3
export TOP_P=0.95
export TEMPERATURE=1.0

bash run_qg_augment.sh
```

Current corpus chunking produces `47,343` chunks from `15,883` documents
(`2.98` chunks per document on average). Therefore:

```text
NUM_RETURN_SEQUENCES=1 -> 47,343 pseudo queries, 2.98/doc average
NUM_RETURN_SEQUENCES=3 -> 142,029 pseudo queries, 8.94/doc average
NUM_RETURN_SEQUENCES=5 -> 236,715 pseudo queries, 14.90/doc average
```

Use `3` as the first serious QG-augmentation run; use `5` only if disk/time are
acceptable and generated samples look diverse.

This keeps the original metadata and packed unit store untouched. The augmented
`train.csv` adds generated `qg_*` question ids, and the augmented `train.npz`
stores their generated raw unit-code sequences.

Train GR against the augmented data:

```bash
cd /home/ricky/SpeechGR
export CUDA_VISIBLE_DEVICES=1
export PYTHON_BIN=.venv/bin/python
bash run_gr_qg_augmented.sh
```

The Hub repo keeps only:

```text
latest/
best/
```

Old Hub `checkpoint-*` folders are pruned by default.

## Launch DSI

DSI training from the current configured pretrained checkpoint:

```bash
export DATASET_PATH=/path/to/slue_sqa5_metadata
bash run.sh
```

DSI with public Hugging Face latest/best monitoring:

```bash
HF_TOKEN=... \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-dsi-live \
bash run.sh
```

## Launch Unit T5 Pretraining

This pretrains directly on the packed SLUE-SQA5 unit dataset by default:

```bash
bash run_t5_pt.sh
```

If you already materialized the packed unit dataset locally:

```bash
CODE_DIR=/data/slue_sqa_code_l22_c500 bash run_t5_pt.sh
```

With public Hugging Face latest/best monitoring:

```bash
HF_TOKEN=... \
HF_CHECKPOINT_REPO_ID=dodofk/unit_t5_live \
bash run_t5_pt.sh
```

Smoke-test Hub latest/best upload before a long run:

```bash
python3 scripts/smoke_train_units.py \
  --mode hub \
  --hf-checkpoint-repo-id "$HF_CHECKPOINT_REPO_ID" \
  --max-steps 2
```

## Debug Runs Without Saving

Disable periodic local checkpoints and Hub mirroring:

```bash
SAVE_CHECKPOINTS=False bash run_qg.sh
SAVE_CHECKPOINTS=False bash run.sh
SAVE_CHECKPOINTS=False bash run_t5_pt.sh
```

For QG/pretraining, also skip final model saving:

```bash
SAVE_CHECKPOINTS=False SAVE_FINAL_MODEL=False bash run_qg.sh
SAVE_CHECKPOINTS=False SAVE_FINAL_MODEL=False bash run_t5_pt.sh
```

## Common Switches

Use a different base model or local initialized model for QG:

```bash
MODEL_NAME_OR_PATH=google/flan-t5-base bash run_qg.sh
MODEL_PATH=/path/to/checkpoint bash run_qg.sh
```

Keep the Hub mirror private:

```bash
HF_CHECKPOINT_PRIVATE=True \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-live \
bash run_qg.sh
```

Upload full Trainer checkpoints to Hub instead of model-only snapshots:

```bash
HF_CHECKPOINT_MODE=trainer \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-live \
bash run_qg.sh
```

Use `trainer` mode only when remote resume from Hub matters. It includes
optimizer/scheduler state and is much larger.

## Current Metrics

- QG best checkpoint: `unit_f1`
- DSI best checkpoint: `Hits@20`
- Unit T5 pretraining best checkpoint: `eval_loss`

## Disk Budget

Recommended free space:

- one run with a few checkpoints: `15-20GB`
- comfortable single QG or DSI run: `25GB`
- QG + DSI + pretraining outputs together: `40-50GB`

The packed unit dataset is about `0.5GB`; Trainer checkpoints dominate disk use.
