# Status Note — 2025-09-26

- **Environment**: Current run confirmed wavtokenizer retrieval config points at `outputs/slue_wavtok/{csv,precomputed}` with automatic atomic document ids.
- **Smoke Test**: `scripts/run_retrieval_smoke_cpu.sh` runs a 1-step CPU-only check (atomic IDs enabled).
- **Full Train Script**: `scripts/run_retrieval_full_train.sh` launches full training with WandB project `speechgr` (set `WANDB_ENTITY` as needed).
- **Outstanding**: Need to re-run smoke/full scripts after restart; `uv run` emits `pyproject.toml` warnings but functional.
- **Pending License**: `git status` blocked until Xcode license accepted (`sudo xcodebuild -license`).

_Generated on 2025-09-26 17:54 CST._
