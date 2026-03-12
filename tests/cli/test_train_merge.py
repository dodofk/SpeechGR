from omegaconf import OmegaConf

from speechgr.cli.train import _merge_task_config


def test_merge_task_config_preserves_root_level_experiment_overrides():
    cfg = OmegaConf.create(
        {
            "task": {
                "task": "retrieval",
                "data": {
                    "dataset_path": "outputs/slue_wavtok/csv",
                    "precompute_root": "outputs/slue_wavtok/precomputed",
                },
                "run": {"max_length": 512},
            },
            "data": {
                "dataset_path": "outputs/slue_sqa5_mimi/csv",
                "precompute_root": "outputs/slue_sqa5_mimi/precomputed",
                "docid_map_path": "outputs/slue_sqa5_mimi/docids/docid_map.json",
            },
        }
    )

    task_name, merged = _merge_task_config(cfg)

    assert task_name == "retrieval"
    assert merged.data.dataset_path == "outputs/slue_sqa5_mimi/csv"
    assert merged.data.precompute_root == "outputs/slue_sqa5_mimi/precomputed"
    assert (
        merged.data.docid_map_path
        == "outputs/slue_sqa5_mimi/docids/docid_map.json"
    )
