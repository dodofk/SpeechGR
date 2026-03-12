"""Unified training entrypoint for SpeechGR tasks."""

from __future__ import annotations

from typing import Callable, Dict

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from speechgr.cli import retrieval, ranking, qformer, qg, t5_pretrain


_TASK_DISPATCH: Dict[str, Callable[[DictConfig], None]] = {
    "retrieval": retrieval.run,
    "ranking": ranking.run,
    "qformer": qformer.run,
    "qg": qg.run,
    "t5_pretrain": t5_pretrain.run,
}


def _merge_task_config(cfg: DictConfig) -> tuple[str, DictConfig]:
    task_value = cfg.get("task")
    if not isinstance(task_value, DictConfig):
        return str(task_value), cfg

    task_name = task_value.get("name") or task_value.get("task")
    base_dict = OmegaConf.to_container(cfg, resolve=False)
    task_dict = OmegaConf.to_container(task_value, resolve=False)

    # Apply task defaults first, then let root-level config (including experiment
    # presets and CLI overrides) win. The previous merge order let the nested task
    # block overwrite experiment-provided root keys such as `data.*`.
    merged_base = dict(base_dict)
    merged_base.pop("task", None)
    merged_cfg = OmegaConf.merge(
        OmegaConf.create(task_dict),
        OmegaConf.create(merged_base),
    )
    return str(task_name), merged_cfg


@hydra_main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    task_name, merged_cfg = _merge_task_config(cfg)
    if task_name not in _TASK_DISPATCH:
        available = ", ".join(sorted(_TASK_DISPATCH))
        raise ValueError(f"Unsupported task '{task_name}'. Available: {available}")

    set_seed(cfg.get("seed", 42))

    runner = _TASK_DISPATCH[task_name]
    runner(merged_cfg)


if __name__ == "__main__":
    main()
