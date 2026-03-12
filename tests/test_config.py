from speechgr.config import build_training_arguments


def test_build_training_arguments_translates_evaluation_strategy(tmp_path):
    args = build_training_arguments(
        {
            "output_dir": str(tmp_path / "out"),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "evaluation_strategy": "steps",
            "save_strategy": "no",
            "report_to": [],
            "use_cpu": True,
        }
    )

    assert str(args.eval_strategy).lower().endswith("steps")
