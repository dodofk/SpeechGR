"""Unified training entrypoint for SpeechGR tasks."""

from __future__ import annotations

from typing import Callable, Dict

from hydra import main as hydra_main
from omegaconf import DictConfig
from transformers import set_seed

from speechgr.cli import retrieval, ranking, qformer, qg, t5_pretrain


_TASK_DISPATCH: Dict[str, Callable[[DictConfig], None]] = {
    "retrieval": retrieval.run,
    "ranking": ranking.run,
    "qformer": qformer.run,
    "qg": qg.run,
    "t5_pretrain": t5_pretrain.run,
}


@hydra_main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    task_value = cfg.get("task")
    if isinstance(task_value, DictConfig):
        task_name = task_value.get("name")
    else:
        task_name = task_value
    if task_name not in _TASK_DISPATCH:
        available = ", ".join(sorted(_TASK_DISPATCH))
        raise ValueError(f"Unsupported task '{task_name}'. Available: {available}")

    set_seed(cfg.get("seed", 42))

    runner = _TASK_DISPATCH[task_name]
    runner(cfg)


if __name__ == "__main__":
    main()
