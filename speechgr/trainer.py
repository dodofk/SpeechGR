from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from transformers import TrainerCallback
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import wandb

class DSITrainer(Trainer):
    def __init__(
        self,
        restrict_decode_vocab,
        id_max_length: int = 128,
        train_continuous_embedding: bool = False,
        use_whisper_features: bool = False,
        **kwds,
    ):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.train_continuous_embedding = train_continuous_embedding
        self.use_whisper_features = use_whisper_features

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.train_continuous_embedding:
            if self.use_whisper_features:
                # Use input_features for Whisper features
                loss = model(
                    input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                ).loss
            else:
                # Use inputs_embeds for other continuous features
                loss = model(
                    inputs_embeds=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                ).loss
            if return_outputs:
                return loss, [None, None]  # fake outputs
            return loss
        else:
            loss = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            ).loss
            if return_outputs:
                return loss, [None, None]  # fake outputs
            return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs["labels"] = inputs["labels"].to(self.args.device)
        
        with torch.no_grad():
            # Handle different types of features based on mode
            if self.train_continuous_embedding and "input_features" in inputs:
                if self.use_whisper_features:
                    # For Whisper features: source from input_features, use inputs_embeds for generation
                    feature_input = inputs["input_features"].to(self.args.device)
                    
                    if self.restrict_decode_vocab is not None:
                        batch_beams = model.generate(
                            inputs_embeds=feature_input,
                            attention_mask=inputs["attention_mask"].to(self.args.device),
                            max_length=20,
                            num_beams=20,
                            prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                            num_return_sequences=20,
                            early_stopping=True,
                        )
                    else:
                        batch_beams = model.generate(
                            inputs_embeds=feature_input,
                            attention_mask=inputs["attention_mask"].to(self.args.device),
                            max_length=20,
                            num_beams=20,
                            num_return_sequences=20,
                            early_stopping=True,
                        )
                else:
                    # For other continuous features: source from input_features, use inputs_embeds for generation
                    feature_input = inputs["input_features"].to(self.args.device)
                    
                    if self.restrict_decode_vocab is not None:
                        batch_beams = model.generate(
                            inputs_embeds=feature_input,
                            attention_mask=inputs["attention_mask"].to(self.args.device),
                            max_length=20,
                            num_beams=20,
                            prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                            num_return_sequences=20,
                            early_stopping=True,
                        )
                    else:
                        batch_beams = model.generate(
                            inputs_embeds=feature_input,
                            attention_mask=inputs["attention_mask"].to(self.args.device),
                            max_length=20,
                            num_beams=20,
                            num_return_sequences=20,
                            early_stopping=True,
                        )
            else:
                # Original implementation for input_ids (discrete tokens)
                if self.restrict_decode_vocab is not None:
                    batch_beams = model.generate(
                        inputs["input_ids"].to(self.args.device),
                        max_length=20,
                        num_beams=20,
                        prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                        num_return_sequences=20,
                        early_stopping=True,
                    )
                else:
                    batch_beams = model.generate(
                        inputs["input_ids"].to(self.args.device),
                        max_length=20,
                        num_beams=20,
                        num_return_sequences=20,
                        early_stopping=True,
                    )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(
                    batch_beams, self.id_max_length
                )

            inputs["labels"] = self._pad_tensors_to_max_len(
                inputs["labels"], self.id_max_length
            )
            
            # Calculate reshape dimension based on input type
            if self.train_continuous_embedding and "input_features" in inputs:
                batch_size = inputs["input_features"].shape[0]
            else:
                batch_size = inputs["input_ids"].shape[0]
                
            batch_beams = batch_beams.reshape(batch_size, 20, -1)

        return (None, batch_beams, inputs["labels"])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        ).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        outputs = self.model.generate(
            input_ids=inputs[0]["input_ids"].to(self.args.device),
            attention_mask=inputs[0]["attention_mask"].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences,
        )
        labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(
            self.num_return_sequences
        )

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (
            None,
            outputs.reshape(
                inputs[0]["input_ids"].shape[0], self.num_return_sequences, -1
            ),
            labels.reshape(
                inputs[0]["input_ids"].shape[0], self.num_return_sequences, -1
            ),
        )

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        max_length: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(
            test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )


class SelfNegMiningCallback(TrainerCallback):
    def __init__(self, beam_k: int = 4, mine_every_epoch: int = 2):
        self.beam_k = beam_k
        self.every = mine_every_epoch
        self.trainer = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.trainer is None:
            raise Exception("Please Manually link the trainer")
        print("Debugging kwargs: ", kwargs.keys())
        model = kwargs["model"]
        train_dataloader = kwargs["train_dataloader"]
        tokenizer = kwargs["tokenizer"]

        if self.trainer.lambda_hard == 0 or state.epoch % self.every:
            return  # skip mining

        model.eval()
        new_cache = {}

        device = args.device
        for batch in train_dataloader:
            # crude query id: hash of input tokens
            qids = batch["query_doc_id"]
            batch.pop("query_doc_id")

            # convert qids to list
            if torch.is_tensor(qids):
                qids = qids.tolist()
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                beams = model.generate(
                    input_ids=batch["input_ids"],
                    num_beams=self.beam_k + 1,
                    num_return_sequences=self.beam_k + 1,
                    max_length=128,
                    early_stopping=True,
                    prefix_allowed_tokens_fn=self.trainer.restrict_decode_vocab,
                )

            # [B*(k+1),L]
            gold = batch["labels"].clone()
            gold[gold == -100] = tokenizer.pad_token_id

            beams = beams.view(len(qids), self.beam_k + 1, -1)  # drop top-1

            picked = []

            for i in range(len(qids)):
                k_list = []
                for beam in beams[i]:
                    if not torch.equal(gold[i], beam):
                        # beam = F.pad(beam, (0, self.trainer.id_max_length - beam.shape[-1]), value=tokenizer.pad_token_id)
                        k_list.append(beam)
                    if len(k_list) == self.beam_k:
                        break
                if len(k_list) < self.beam_k:
                    raise Exception("Unexpected behavior for not enough hard negatives")
                picked.append(k_list)

            for qid, neg_pack in zip(qids, picked):
                # neg_pack: List contains beam_k tensors of shape [Ldoc]
                new_cache[qid] = torch.stack(neg_pack, dim=0).to(
                    "cpu"
                )  # save cache to cpu for memory efficiency
        self.trainer.hardneg_cache = new_cache

        model.train()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs.pop("query_doc_id")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class NegLambdaScheduleCallback(TrainerCallback):
    """
    This callback is used to schedule the lambda values for the in-batch and self-mined hard negatives.

    The start_epoch should manually sorted, the callback does not check for it.

    Args
    ----
    inbatch_schedule : list[tuple[int, float]]
        (start_epoch, lambda_value)   pairs, sorted by start_epoch.
        Example:  [(0, 0.0), (2, 1.0)]
    hard_schedule    : list[tuple[int, float]]
        Same format but for lambda_hard.
    """

    def __init__(
        self,
        inbatch_schedule: List[Tuple[int, float]] = None,
        hard_schedule: List[Tuple[int, float]] = None,
    ):
        # default: start both at 0.0, never change
        self.inb_sched = inbatch_schedule or [(0, 0.0)]
        self.hard_sched = hard_schedule or [(0, 0.0)]
        self.trainer = None

    # ---------------------------------------------------------------
    def _value_for_epoch(self, epoch, schedule):
        """
        Return the last lambda value whose start_epoch ≤ current epoch.
        """
        val = schedule[0][1]  # fallback
        for start_e, v in schedule:
            if epoch >= start_e:
                val = v
            else:
                break
        return val

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.trainer is None:
            raise Exception("Please Manually link the trainer")

        λ_inb = self._value_for_epoch(state.epoch, self.inb_sched)
        λ_hard = self._value_for_epoch(state.epoch, self.hard_sched)

        # update trainer attributes
        self.trainer.lambda_inbatch = λ_inb
        self.trainer.lambda_hard = λ_hard

        # optional: log to console / WandB
        if args.local_rank in (-1, 0):  # main process only
            print(
                f"[NegLambdaSchedule] epoch {state.epoch:.0f}  "
                f"λ_inbatch = {λ_inb:.3f} , λ_hard = {λ_hard:.3f}"
            )


class RankingLossCallback(TrainerCallback):
    def on_log(self, args, state, control, **kw):
        if state.is_world_process_zero:
            d = {
                "loss_ce": state.loss_ce_step.item(),
                "loss_inb": (
                    state.loss_inbatch_step.item() if state.loss_inbatch_step else 0.0
                ),
                "loss_hard": (
                    state.loss_hard_step.item() if state.loss_hard_step else 0.0
                ),
            }
            # print(
            #     f"[step {state.global_step}]",
            #     " | ".join(f"{k}:{v:6.3f}" for k, v in d.items()),
            # )
            wandb.log(d)
            # wandb.log(d, step=state.global_step)   # optional


class DSIRankingTrainer(DSITrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hardneg_cache = {}
        self.lambda_inbatch: float = 0  # control the loss portion, change by callback
        self.lambda_hard: float = 0

    def _ripor_logprob(self, dec_hidden_state, labels):
        labels = labels.clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        decoder_embeds = self.model.decoder.embed_tokens(labels)
        margin = (dec_hidden_state * decoder_embeds).sum(-1).sum(-1)
        return margin

    def compute_loss(self, model, inputs, return_outputs=True):
        """
        total_loss = CE
                + λ_inbatch · margin(in-batch)        (if λ_inbatch ≠ 0)
                + λ_hard    · margin(self-negatives)  (if λ_hard    ≠ 0)
        """

        # --------- constants / shortcuts ---------
        device = inputs["input_ids"].device if "input_ids" in inputs else inputs["input_features"].device
        pad_id = self.tokenizer.pad_token_id

        qids = inputs["query_doc_id"]
        inputs.pop("query_doc_id")

        if torch.is_tensor(qids):
            qids = qids.tolist()

        # --------- forward pass (gives CE) ---------
        if self.train_continuous_embedding:
            if self.use_whisper_features:
                # Use input_features for Whisper features
                out = model(
                    input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    output_hidden_states=True,
                )
            else:
                # Use inputs_embeds for other continuous features
                out = model(
                    inputs_embeds=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    output_hidden_states=True,
                )
        else:
            # Use input_ids for discrete tokens
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                output_hidden_states=True,
            )
            
        ce_loss = out.loss
        # enc_hid = out.encoder_last_hidden_state  # [B,Btok,d]
        # logits = out.logits  # [B,L,V]
        labels = inputs["labels"]  # [B,L]
        B, _ = labels.shape

        # --------- positive scores ---------
        # mask = labels != -100
        lbl = labels.clone()
        lbl[lbl == -100] = pad_id
        # log_p = torch.log_softmax(logits, dim=-1)
        # # s_pos = (log_p.gather(-1, lbl.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)  # [B]

        # print("debug out available: ", out.decoder_hidden_states[-1].shape)
        dec_hidden_state = out.decoder_hidden_states[-1]
        # dec_embs = model.decoder.embed_tokens(lbl)
        # print("dec_embs shape: ", dec_embs.shape)
        # s_pos = (dec_hidden_state * dec_embs).sum(-1).sum(-1)
        s_pos = self._ripor_logprob(dec_hidden_state, labels)

        total_loss = ce_loss
        rank_losses = {}

        # ================================================================
        # 1) IN-BATCH NEGATIVES  (only if λ_inbatch ≠ 0)
        # ================================================================
        if self.lambda_inbatch != 0.0:
            docs_pad = lbl  # [B, Ldoc]
            inb_ids, inb_enc = [], []
            for i in range(B):
                inb_ids.append(torch.cat([docs_pad[:i], docs_pad[i + 1 :]], 0))
                inb_enc.append(
                    dec_hidden_state[i].expand(B - 1, -1, -1)
                )  # expand to [B-1, Btok, d]
            inb_ids = torch.cat(inb_ids, 0)  # [B*(B-1), Ldoc]
            inb_enc = torch.cat(inb_enc, 0)  # [B*(B-1), Btok, d]

            with torch.enable_grad():
                # s_inb = model.sequence_logprob(inb_enc, inb_ids)  # [B*(B-1)]
                s_inb = self._ripor_logprob(inb_enc, inb_ids)

            s_inb = s_inb.view(B, B - 1)

            inb_loss = F.margin_ranking_loss(
                s_pos.unsqueeze(1)
                .expand_as(s_inb)
                .reshape(-1),  # [B] -> [B, 1] -> [B, B-1] -> [B*(B-1)]
                s_inb.reshape(-1),  # [B*(B-1)]
                torch.ones_like(s_inb).reshape(-1),  # [B*(B-1)]
                margin=1.0,
            )
            total_loss = total_loss + self.lambda_inbatch * inb_loss
            rank_losses["inbatch"] = inb_loss.detach()  # detach to use in logging

        # ================================================================
        # 2) SELF-MINED HARD NEGATIVES (only if λ_hard ≠ 0)
        # ================================================================
        if self.lambda_hard != 0.0:
            raise NotImplementedError("Self-mined hard negatives are not done")
            hard_ids, hard_enc = [], []
            for qid, enc in zip(qids, dec_hidden_state):
                for neg in self.hardneg_cache.get(qid, []):
                    hard_ids.append(neg.to(device))
                    hard_enc.append(enc)
            print("debug hard ids: ", hard_ids)
            if hard_ids:  # avoid zero-length tensor
                hard_ids = torch.stack(hard_ids)
                hard_enc = torch.stack(hard_enc)

                with torch.enable_grad():
                    s_hard = model.ripor_logprob(hard_enc, hard_ids)  # [B*Kh]
                Kh = s_hard.numel() // B
                s_hard = s_hard.view(B, Kh)

                hard_loss = F.margin_ranking_loss(
                    s_pos.unsqueeze(1).expand_as(s_hard).reshape(-1),
                    s_hard.reshape(-1),
                    torch.ones_like(s_hard).reshape(-1),
                    margin=1.0,
                )
                total_loss = total_loss + self.lambda_hard * hard_loss
                print("Debugging hard loss: ", hard_loss)
                rank_losses["selfneg"] = hard_loss.detach()

        self.state.loss_ce_step = ce_loss.detach()
        self.state.loss_inbatch_step = (
            rank_losses["inbatch"].detach() if "inbatch" in rank_losses else 0.0
        )
        self.state.loss_hard_step = (
            rank_losses["selfneg"].detach() if "selfneg" in rank_losses else 0.0
        )

        return total_loss


if __name__ == "__main__":
    from speechgr.data import (
        SlueSQA5DatasetV2,
        IndexingCollator,
        IndexingCollatorWithMetadata,
    )
    from speechgr.model import LatentQueryT5
    from transformers import AutoTokenizer
    from torch.utils.data import Subset
    from transformers import TrainingArguments
    from speechgr.utils import RestrictDecodeVocab

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    train_dataset = SlueSQA5DatasetV2(
        split="train",
        max_length=512,
        dataset_path="outputs/slue_wavtok/csv",
        code_path="outputs/slue_wavtok/precomputed",
    )
    valid_dataset = SlueSQA5DatasetV2(
        split="validation",
        max_length=512,
        dataset_path="outputs/slue_wavtok/csv",
        code_path="outputs/slue_wavtok/precomputed",
        include_corpus=False,
    )
    collator = IndexingCollator(tokenizer, padding="longest")
    metadata_collator = IndexingCollatorWithMetadata(tokenizer, padding="longest")
    train_subset = Subset(train_dataset, range(128))
    valid_subset = Subset(valid_dataset, range(128))
    model = LatentQueryT5()

    args = TrainingArguments(
        output_dir="./debug_tmp_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        logging_steps=1,
        save_steps=0,
        report_to=[],
    )

    metadata_dataloader = DataLoader(
        train_subset, batch_size=4, collate_fn=metadata_collator
    )

    trainer = DSIRankingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_subset,
        eval_dataset=valid_subset,
        data_collator=collator,
        # compute_metrics=make_compute_metrics(tokenizer, train_dataset.valid_ids),
        id_max_length=128,
        restrict_decode_vocab=RestrictDecodeVocab(
            valid_ids=train_dataset.valid_ids, tokenizer=tokenizer
        ),
    )

    neg_lambda_callback = NegLambdaScheduleCallback(
        inbatch_schedule=[(0, 0.0), (1, 1.0)],  # start in-batch at epoch 1
        hard_schedule=[(0, 0.0), (2, 1.0)],  # hard negatives from epoch 2
    )

    neg_lambda_callback.trainer = trainer

    neg_mining_callback = SelfNegMiningCallback(
        beam_k=4,
        mine_every_epoch=1,
    )
    neg_mining_callback.trainer = trainer

    trainer.add_callback(neg_lambda_callback)
    trainer.add_callback(neg_mining_callback)

    trainer.train()
    # Testing
