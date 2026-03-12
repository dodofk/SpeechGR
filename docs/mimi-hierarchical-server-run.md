# Mimi Hierarchical Server Run

This note captures the minimal server-side workflow for the current Mimi-first hierarchical retrieval pipeline. It uses explicit Hydra config overrides instead of environment variables so the commands are easy to inspect and modify.

## 1. Prepare CSV + Mimi precompute

The Mimi prepare config uses the safer multi-pass path:

- `streaming: false`
- `decode_audio: false`
- config default device is `cpu`

This means `prepare_slue` uses the non-streaming Hugging Face dataset path while still keeping audio decoding disabled. This is important because the current prepare pipeline scans the dataset multiple times: once to write split CSVs and again to precompute question and corpus caches. Streaming in this multi-pass setup can multiply network I/O and make the run slower rather than faster.

On the CUDA server, you should explicitly override the encoder device to `cuda`:

```bash
uv run python -m speechgr.cli.prepare_slue \
  --config-name slue_sqa5_mimi \
  decode_audio=false \
  encoder.params.device=cuda
```

Expected outputs:

- `outputs/slue_sqa5_mimi/csv/`
- `outputs/slue_sqa5_mimi/precomputed/train/train_mimi.pt`
- `outputs/slue_sqa5_mimi/precomputed/validation/validation_mimi.pt`
- `outputs/slue_sqa5_mimi/precomputed/test/test_mimi.pt`
- `outputs/slue_sqa5_mimi/precomputed/verified_test/verified_test_mimi.pt`
- `outputs/slue_sqa5_mimi/precomputed/corpus/corpus_mimi.pt`

## 2. Build hierarchical DocIDs from Mimi corpus cache

```bash
uv run python scripts/build_mimi_hierarchical_docids.py \
  --corpus-cache outputs/slue_sqa5_mimi/precomputed/corpus/corpus_mimi.pt \
  --output-dir outputs/slue_sqa5_mimi/docids \
  --vocab-size 2048 \
  --projection-dim 256 \
  --target-cluster-size 64 \
  --leaf1-size 128 \
  --leaf2-size 64
```

Expected outputs:

- `outputs/slue_sqa5_mimi/docids/doc_ids.json`
- `outputs/slue_sqa5_mimi/docids/embeddings.npy`
- `outputs/slue_sqa5_mimi/docids/embedding_diagnostics.json`
- `outputs/slue_sqa5_mimi/docids/docid_map.json`
- `outputs/slue_sqa5_mimi/docids/cluster_members.json`
- `outputs/slue_sqa5_mimi/docids/valid_paths.json`
- `outputs/slue_sqa5_mimi/docids/docid_diagnostics.json`

## 3. Train hierarchical retrieval baseline

```bash
uv run python -m speechgr.cli.train \
  experiment=retrieval/slue_sqa5_mimi_hierarchical \
  data.dataset_path=outputs/slue_sqa5_mimi/csv \
  data.precompute_root=outputs/slue_sqa5_mimi/precomputed \
  data.docid_map_path=outputs/slue_sqa5_mimi/docids/docid_map.json \
  training.training_args.output_dir=models/slue_sqa5-mimi-hierarchical-flan-t5-base \
  training.training_args.run_name=slue_sqa5-mimi-hierarchical-flan-t5-base
```

Recommended short smoke run before a long CUDA job:

```bash
uv run python -m speechgr.cli.train \
  experiment=retrieval/slue_sqa5_mimi_hierarchical \
  data.dataset_path=outputs/slue_sqa5_mimi/csv \
  data.precompute_root=outputs/slue_sqa5_mimi/precomputed \
  data.docid_map_path=outputs/slue_sqa5_mimi/docids/docid_map.json \
  training.training_args.max_steps=50 \
  training.training_args.eval_steps=25 \
  training.training_args.save_steps=25
```

## 4. Pre-flight checks on the CUDA server

Before launching a long run, confirm:

- `outputs/slue_sqa5_mimi/precomputed/corpus/corpus_mimi.pt` exists
- `outputs/slue_sqa5_mimi/docids/docid_map.json` exists
- `outputs/slue_sqa5_mimi/docids/docid_diagnostics.json` has an acceptable collision rate
- `outputs/slue_sqa5_mimi/stats.json` looks reasonable under Mimi `semantic_only`
- a 50-step smoke training run completes cleanly

## Notes

- Current Mimi path uses `semantic_only` as the intended retrieval representation.
- Current hierarchical DocID format is `1 cluster + 2 leaf` tokens.
- The first full run should be treated as a baseline for diagnostics, not yet as the final DocID method.
- If corpus size or collisions demand it, increase leaf depth before changing the retrieval backbone.
