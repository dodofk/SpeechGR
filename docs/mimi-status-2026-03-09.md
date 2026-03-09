# Mimi Path Status Report — 2026-03-09

## Scope
Focused only on Mimi path validation and progress status.

## What was requested
- Run a Mimi precompute smoke run for **5 queries + 5 passages**.
- Confirm current development status for Mimi track.
- Record status for next-stage execution.

## Run results

### 1) Real SLUE mini precompute (train[:5]) — **passed (after fixes)**
Final command path:
- loaded SLUE `train[:5]`
- cast audio columns with `Audio(decode=False)`
- precomputed with real `kyutai/mimi` on CPU for both question and passage

Artifacts:
- `outputs/smoke/mimi_precompute_5q_5p_real/question/train5_mimi.pt`
- `outputs/smoke/mimi_precompute_5q_5p_real/passage/passage5_mimi.pt`
- `outputs/smoke/mimi_precompute_5q_5p_real/summary.json`

Observed stats (5 samples):
- question mean token length: `2112`
- passage mean token length: `16000`

### 2) Fixes applied to unblock Mimi setup

#### A. TorchCodec/FFmpeg decode blocker bypassed safely for precompute path
- `MimiEncoder` now supports raw dataset audio payloads with:
  - `{"array", "sampling_rate"}`
  - `{"bytes"}`
  - `{"path"}`
- This allows SLUE precompute with `datasets.Audio(decode=False)` and avoids the failing torchcodec decode path.

#### B. Mimi model compatibility fixed
- Upgraded `transformers` from `4.44.2` to `5.3.0` (now includes `MimiModel`).
- Added `trust_remote_code=True` in Mimi HF loading to improve compatibility.
- Verified direct Mimi encode on synthetic audio now works.

### 3) Mimi retrieval smoke pipeline — **passed**
Command run:
```bash
uv run python scripts/test/smoke_test_mimi_retrieval.py \
  --output-root outputs/smoke/mimi_retrieval_status \
  --epochs 1
```

Result:
- succeeded end-to-end in CPU smoke mode
- summary generated at:
  - `outputs/smoke/mimi_retrieval_status/smoke_summary.json`

## Current Mimi stage status vs plan

### Stage 1 (Mimi integration)
- Code integration: **mostly done**
- Real precompute on dataset: **smoke-verified on real SLUE subset (5q/5p)**

### Stage 2 (hierarchical DocID builder)
- Builder scaffold + artifacts: **done**
- embedding-quality diagnostics: **not done**

## Immediate blockers to unblock next stage
1. Run full-split precompute (`train/validation/test/corpus`) and collect token stats at scale.
2. Confirm runtime stability/memory footprint for longer corpus precompute runs.

## Recommended next step (aligned with progress plan)
Now that blockers are unblocked, run:
1. Mimi precompute for train/validation/test/corpus
2. collect token-length stats for real data
3. run Stage-2 diagnostics (collision + cluster purity)
4. move to Stage-3 baseline (real query -> hierarchical DocID)

## Artifacts generated today
- `docs/mimi-status-2026-03-09.md` (this report)
- `outputs/smoke/mimi_retrieval_status/smoke_summary.json`
- `outputs/smoke/mimi_precompute_5q_5p_real/summary.json`
