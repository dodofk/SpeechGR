# SpeechGR

## Training Quickstart

For server handoff and copy-paste launch commands, start with
[docs/TRAINING_HANDOFF.md](docs/TRAINING_HANDOFF.md).

## Packed SLUE-SQA5 Unit Dataset

The SLUE-SQA5 HuBERT layer-22 K=500 unit files are stored on Hugging Face as
packed NumPy archives:

```text
hf://datasets/dodofk/slue-sqa-code-l22-c500
```

The training scripts use this path by default. On a new machine, installing
`requirements.txt` is enough for `data.py`, `qg.py`, and `run.py` to download
the files through the Hugging Face cache on first use.

For a remote server or cluster job, prefetch and verify the dataset first:

```bash
python3 scripts/download_unit_dataset.py
```

Quick smoke test:

```bash
python3 scripts/download_unit_dataset.py --splits verified_test --local-dir /tmp/slue_sqa_code_l22_c500_smoke
```

Use a shared cache if multiple runs should reuse the same download:

```bash
HF_HOME=/path/to/hf_cache python3 scripts/download_unit_dataset.py
```

You can also materialize the files into a normal directory and pass that path
to existing commands:

```bash
python3 scripts/download_unit_dataset.py --local-dir /data/slue_sqa_code_l22_c500
python3 run.py --code_path /data/slue_sqa_code_l22_c500 ...
```

Validate the training dataloaders without starting a training run:

```bash
python3 scripts/smoke_test_dataloaders.py
```

If the packed units are already materialized locally, validate against those
real unit sequences:

```bash
python3 scripts/smoke_test_dataloaders.py --code-dir /data/slue_sqa_code_l22_c500
```

Run a tiny CPU training smoke with the same packed-unit DSI and QG training
paths:

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

## Optional Hub Checkpoint Mirror

Long server runs can mirror the newest validation snapshot and the current best
snapshot to a Hugging Face model repo. This is off by default. Enable it by
setting `HF_CHECKPOINT_REPO_ID` before a run script:

```bash
HF_TOKEN=... \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-live \
bash run_qg.sh
```

By default, this uploads model-only snapshots to `latest/` after every
validation and to `best/` when the configured validation metric improves. It
also prunes stale Hub `checkpoint-*` paths so the public repo does not grow one
checkpoint per validation. Disable pruning only if you intentionally want to
keep old Hub checkpoints:

```bash
HF_CHECKPOINT_PRUNE_OLD=False \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-live \
bash run_qg.sh
```

Use full Trainer checkpoints only when you really need optimizer/scheduler state:

```bash
HF_CHECKPOINT_MODE=trainer \
HF_CHECKPOINT_REPO_ID=dodofk/speechgr-qg-live \
bash run_qg.sh
```

`trainer` mode is much larger because it includes optimizer state. For normal
monitoring, keep the default `model` mode.

Disable periodic local checkpoints and Hub mirroring for a debug run:

```bash
SAVE_CHECKPOINTS=False bash run_qg.sh
```

For QG/pretraining scripts, disable the final `save_model()` too:

```bash
SAVE_CHECKPOINTS=False SAVE_FINAL_MODEL=False bash run_qg.sh
```
