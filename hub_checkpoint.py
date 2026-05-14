"""
Opt-in Hugging Face Hub checkpoint mirroring for long remote training runs.

The default mode uploads model-only snapshots to `latest/` at every validation
and to `best/` when the configured validation metric improves. This keeps Hub
uploads small enough for frequent monitoring while local Trainer checkpoints
keep optimizer/scheduler state for resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

STALE_HUB_CHECKPOINT_PATTERNS = [
    "checkpoint-*/*",
    "checkpoint-*",
    "pytorch_model.bin",
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state*.pth",
    "scaler.pt",
]


@dataclass
class HubCheckpointArguments:
    hf_checkpoint_repo_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional Hugging Face model repo id for latest/best checkpoint mirroring."
        },
    )
    hf_checkpoint_private: bool = field(
        default=False,
        metadata={"help": "Create the Hub model repo as private when it does not exist."},
    )
    hf_checkpoint_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hub branch/revision to upload to."},
    )
    hf_checkpoint_latest_path: str = field(
        default="latest",
        metadata={"help": "Path in the Hub repo for the latest validation snapshot."},
    )
    hf_checkpoint_best_path: str = field(
        default="best",
        metadata={"help": "Path in the Hub repo for the best validation snapshot."},
    )
    hf_checkpoint_mode: str = field(
        default="model",
        metadata={
            "help": "Upload mode: 'model' for model-only snapshots, 'trainer' for full local checkpoint folders."
        },
    )
    hf_checkpoint_fail_on_error: bool = field(
        default=False,
        metadata={"help": "Raise Hub upload errors instead of warning and continuing."},
    )
    hf_checkpoint_prune_old: bool = field(
        default=True,
        metadata={
            "help": "Delete stale Hub checkpoint-* folders/root checkpoint files so the repo only keeps latest/ and best/ snapshots."
        },
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class LatestBestHubCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        hub_args: HubCheckpointArguments,
        tokenizer=None,
    ):
        if hub_args.hf_checkpoint_mode not in {"model", "trainer"}:
            raise ValueError(
                "hf_checkpoint_mode must be 'model' or 'trainer', "
                f"got {hub_args.hf_checkpoint_mode!r}"
            )
        self.hub_args = hub_args
        self.tokenizer = tokenizer
        self.best_metric = None
        self.last_eval_is_best = False
        self._repo_ready = False

    @property
    def enabled(self) -> bool:
        return bool(self.hub_args.hf_checkpoint_repo_id)

    def _handle_error(self, message: str, exc: Exception) -> None:
        if self.hub_args.hf_checkpoint_fail_on_error:
            raise exc
        logger.warning("%s: %s", message, exc)

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi()

    def _ensure_repo(self) -> None:
        if self._repo_ready:
            return
        self._api().create_repo(
            repo_id=self.hub_args.hf_checkpoint_repo_id,
            repo_type="model",
            private=self.hub_args.hf_checkpoint_private,
            exist_ok=True,
        )
        self._repo_ready = True

    def _metric_name(self, args) -> Optional[str]:
        return args.metric_for_best_model

    def _metric_value(self, args, metrics) -> Optional[float]:
        if not metrics:
            return None
        metric_name = self._metric_name(args)
        if metric_name is None:
            return None

        candidates = [metric_name]
        if not metric_name.startswith("eval_"):
            candidates.insert(0, f"eval_{metric_name}")
        for key in candidates:
            if key in metrics:
                return float(metrics[key])
        return None

    def _is_better(self, args, value: Optional[float]) -> bool:
        if value is None:
            return False
        if self.best_metric is None:
            return True
        greater = args.greater_is_better
        if greater is None:
            metric_name = self._metric_name(args) or ""
            greater = not metric_name.endswith("loss")
        return value > self.best_metric if greater else value < self.best_metric

    def _metadata(self, args, state, metrics, tag: str) -> dict:
        return {
            "tag": tag,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "best_metric": self.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "metrics": _jsonable(metrics or {}),
        }

    def _delete_patterns(self, path_in_repo: str) -> list[str]:
        patterns = [f"{path_in_repo}/*"]
        if self.hub_args.hf_checkpoint_prune_old:
            patterns.extend(STALE_HUB_CHECKPOINT_PATTERNS)
        return patterns

    def _upload_folder(self, folder: Path, path_in_repo: str, args, state, metrics, tag: str) -> None:
        self._ensure_repo()
        commit_message = f"Update {tag} checkpoint at step {state.global_step}"
        self._api().upload_folder(
            repo_id=self.hub_args.hf_checkpoint_repo_id,
            repo_type="model",
            revision=self.hub_args.hf_checkpoint_revision,
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            delete_patterns=self._delete_patterns(path_in_repo),
            commit_message=commit_message,
            commit_description=json.dumps(
                self._metadata(args, state, metrics, tag),
                indent=2,
                sort_keys=True,
            ),
        )
        logger.info(
            "Uploaded %s checkpoint to https://huggingface.co/%s/tree/%s/%s",
            tag,
            self.hub_args.hf_checkpoint_repo_id,
            self.hub_args.hf_checkpoint_revision or "main",
            path_in_repo,
        )

    def _push_model_snapshot(self, args, state, metrics, model, path_in_repo: str, tag: str) -> None:
        with tempfile.TemporaryDirectory(prefix="speechgr_hf_checkpoint_") as tmp_dir:
            folder = Path(tmp_dir)
            model.save_pretrained(folder, safe_serialization=args.save_safetensors)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(folder)
            with open(folder / "speechgr_checkpoint_info.json", "w") as f:
                json.dump(
                    self._metadata(args, state, metrics, tag),
                    f,
                    indent=2,
                    sort_keys=True,
                )
            self._upload_folder(folder, path_in_repo, args, state, metrics, tag)

    def _push_trainer_checkpoint(
        self,
        args,
        state,
        metrics,
        path_in_repo: str,
        tag: str,
    ) -> None:
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            logger.warning("Skip Hub %s upload; missing %s", tag, checkpoint_dir)
            return
        self._upload_folder(checkpoint_dir, path_in_repo, args, state, metrics, tag)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not self.enabled or args.process_index != 0:
            return
        value = self._metric_value(args, metrics)
        self.last_eval_is_best = self._is_better(args, value)
        if self.last_eval_is_best:
            self.best_metric = value

        if self.hub_args.hf_checkpoint_mode != "model":
            return
        if model is None:
            logger.warning("Skip Hub model checkpoint upload; callback did not receive model")
            return
        try:
            self._push_model_snapshot(
                args,
                state,
                metrics,
                model,
                self.hub_args.hf_checkpoint_latest_path,
                "latest",
            )
            if self.last_eval_is_best:
                self._push_model_snapshot(
                    args,
                    state,
                    metrics,
                    model,
                    self.hub_args.hf_checkpoint_best_path,
                    "best",
                )
        except Exception as exc:
            self._handle_error("Hub model checkpoint upload failed", exc)

    def on_save(self, args, state, control, **kwargs):
        if (
            not self.enabled
            or args.process_index != 0
            or self.hub_args.hf_checkpoint_mode != "trainer"
        ):
            return
        try:
            self._push_trainer_checkpoint(
                args,
                state,
                None,
                self.hub_args.hf_checkpoint_latest_path,
                "latest",
            )
            if self.last_eval_is_best:
                self._push_trainer_checkpoint(
                    args,
                    state,
                    None,
                    self.hub_args.hf_checkpoint_best_path,
                    "best",
                )
        except Exception as exc:
            self._handle_error("Hub trainer checkpoint upload failed", exc)


def build_hub_checkpoint_callbacks(
    hub_args: HubCheckpointArguments,
    tokenizer=None,
) -> list[TrainerCallback]:
    if not hub_args.hf_checkpoint_repo_id:
        return []
    return [LatestBestHubCheckpointCallback(hub_args, tokenizer=tokenizer)]
