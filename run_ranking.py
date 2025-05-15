"""
This file is used to train DSI with ranking loss
"""
from data import (
    IndexingCollator,
    SlueSQA5DatasetV2,
)

from utils import RestrictDecodeVocab

from transformers import (
    T5ForConditionalGeneration,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)

from trainer import DSIRankingTrainer, NegLambdaScheduleCallback, RankingLossCallback
import numpy as np
import torch
import wandb
from torch.utils.data import Subset
from dataclasses import dataclass, field
from typing import Optional, List
import json
from tqdm import tqdm

set_seed(42)


@dataclass
class RunArguments:
    model_name: str = field(default="google/flan-t5-base")
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=512)
    id_max_length: Optional[int] = field(default=128)
    run_notes: str = field(default="")
    code_path: str = field(default="/home/ricky/dodofk/dataset/slue_sqa_code_c512")
    dataset_path: str = field(default="/home/ricky/dodofk/dataset/slue_sqa5/")
    run_notes: str = field(default="")
    special_token: Optional[int] = field(default=32000)
    discrete_code_num: Optional[int] = field(default=500)
    code_path: Optional[str] = field(
        default="/home/ricky/dodofk/dataset/slue_sqa_code_c512"
    )
    lookup_file_name: Optional[str] = field(default=None)
    do_inference: Optional[bool] = field(default=False)
    do_debug: Optional[bool] = field(default=False)


def make_compute_metrics(tokenizer, valid_ids):
    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        hit_at_20 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)
            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            hits_at_20 = np.where(np.array(filtered_rank_list)[:20] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
            if len(hits_at_20) != 0:
                hit_at_20 += 1

        metrics = {
            "Hits@1": hit_at_1 / len(eval_preds.predictions),
            "Hits@10": hit_at_10 / len(eval_preds.predictions),
            "Hits@20": hit_at_20 / len(eval_preds.predictions),
        }

        # Save the metrics to a JSON file
        with open("eval_results.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Also log the metrics directly to WandB dashboard.
        wandb.log(metrics)

        return metrics

    return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI", name=training_args.run_name, notes=run_args.run_notes)

    tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir="cache")
    fast_tokenizer = AutoTokenizer.from_pretrained(
        run_args.model_name, cache_dir="cache"
    )
    # only support flan-t5 (or other t5 base model)
    if run_args.model_path:
        model = T5ForConditionalGeneration.from_pretrained(
            run_args.model_path, cache_dir="cache"
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            run_args.model_name, cache_dir="cache"
        )

    train_dataset = SlueSQA5DatasetV2(
        split="train",
        max_length=run_args.max_length,
        model_name_or_path=run_args.model_name,
        code_path=run_args.code_path,
        dataset_path=run_args.dataset_path,
        special_token=run_args.special_token,
        discrete_code_num=run_args.discrete_code_num,
        lookup_file_name=run_args.lookup_file_name,
    )

    valid_dataset = SlueSQA5DatasetV2(
        split="validation",
        max_length=run_args.max_length,
        model_name_or_path=run_args.model_name,
        code_path=run_args.code_path,
        dataset_path=run_args.dataset_path,
        special_token=run_args.special_token,
        discrete_code_num=run_args.discrete_code_num,
        lookup_file_name=run_args.lookup_file_name,
    )

    test_dataset = SlueSQA5DatasetV2(
        split="test",
        max_length=run_args.max_length,
        model_name_or_path=run_args.model_name,
        code_path=run_args.code_path,
        dataset_path=run_args.dataset_path,
        special_token=run_args.special_token,
        discrete_code_num=run_args.discrete_code_num,
        lookup_file_name=run_args.lookup_file_name,
    )
    
    if run_args.do_inference:
        # change max_steps to 1
        eval_dataset = test_dataset
    else:
        eval_dataset = valid_dataset

    restrict_decode_vocab = RestrictDecodeVocab(
        valid_ids=train_dataset.valid_ids, tokenizer=tokenizer
    )
    compute_metrics = make_compute_metrics(fast_tokenizer, train_dataset.valid_ids)
    
    if run_args.do_debug:
        train_dataset = Subset(train_dataset, range(100)) # keep it small for deb ug
        eval_dataset = Subset(eval_dataset, range(100))

    trainer = DSIRankingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding="longest",
        ),
        compute_metrics=compute_metrics,
        id_max_length=run_args.id_max_length,
        restrict_decode_vocab=restrict_decode_vocab,
    )
    
    neg_lambda_callback = NegLambdaScheduleCallback(
        inbatch_schedule = [(0, 0.0), (5, 0.1)],   # start in-batch at epoch 1
    )
    neg_lambda_callback.trainer = trainer
    
    ranking_loss_callback = RankingLossCallback()
    
    trainer.add_callback(neg_lambda_callback)
    trainer.add_callback(ranking_loss_callback)
    
    trainer.train()


if __name__ == "__main__":
    main()
