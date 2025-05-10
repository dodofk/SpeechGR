import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    # Seq2SeqLMOutput,
)
from transformers.modeling_outputs import Seq2SeqLMOutput


class ContinousEmbT5(T5ForConditionalGeneration):
    def __init__(
        self, config, ssl_feat_dim: int = 1024, downsample_factor: int = 2, **kwargs
    ):
        super().__init__(config, **kwargs)

        # For T5-small, embedding dim is typically 512
        # for T5-base, 768, etc. So let's read from the existing embedding matrix
        hidden_dim = self.get_input_embeddings().weight.size(-1)

        self.linear_adapter = nn.Linear(
            ssl_feat_dim * downsample_factor,  # e.g. 1024*2=2048 if you chunk frames
            hidden_dim,
        )
        nn.init.xavier_uniform_(self.linear_adapter.weight)
        nn.init.constant_(self.linear_adapter.bias, 0.0)

    def forward(self, input_embeds=None, attention_mask=None, labels=None, **kwargs):
        """
        Args:
          input_embeds: (batch_size, seq_len, ssl_feat_dim * downsample_factor)
            The user supplies their own embeddings from SSL or a downsampling step.
          attention_mask: (batch_size, seq_len)
          labels: (batch_size, tgt_seq_len) for seq2seq training
          kwargs: anything else T5’s forward might expect (like decoder_input_ids, etc.).
        """
        # 1) Project the custom input embeddings into T5’s hidden dimension
        if input_embeds is not None:
            input_embeds = self.linear_adapter(input_embeds)

        # 2) Pass them to T5
        return super().forward(
            # We pass input_ids=None to ensure T5 doesn't do its own embedding lookup
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


# ---------- latent‑query module ----------
class LatentQueryCompressor(nn.Module):
    """
    Compresses the input sequence into a set of latent queries to shorten the sequence length
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_latents: int = 8,
        n_self_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        nn.init.normal_(self.latents, 0, std=d_model**-0.5)
        self.cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.self_blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True)
            for _ in range(n_self_layers)
        )

        # init

    def forward(self, x):  # x: [B,L,d]
        B = x.size(0)
        z = self.latents.expand(B, -1, -1)  # [B,Btok,d]

        z_ln = nn.functional.layer_norm(z, z.shape[1:])
        z2, _ = self.cross(z_ln, x, x)
        for blk in self.self_blocks:
            z2 = blk(z2)
        z = z + z2
        return z  # [B,Btok,d]


# ---------- custom encoder wrapper ----------
class CompressedT5Encoder(nn.Module):
    """
    Replaces the stock encoder. get_encoder() will return this module,
    so `model.generate` works out‑of‑the‑box.
    """

    def __init__(self, base_encoder, shared, compressor: LatentQueryCompressor):
        super().__init__()
        self.base_encoder = base_encoder
        self.shared = shared  # embedding matrix
        self.compress = compressor

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)

        # 1) compress long seq => B latents
        latents = self.compress(inputs_embeds)

        # 2) run original T5 encoder **on the latents only**
        attn_mask = torch.ones(
            input_ids.size(0),
            self.compress.n_latents,
            device=latents.device,
            dtype=torch.long,
        )

        return self.base_encoder(
            inputs_embeds=latents, attention_mask=attn_mask, **kwargs
        )

    @property
    def main_input_name(self):
        return "input_ids"


# ---------- full model ----------
class LatentQueryT5(T5ForConditionalGeneration):
    """
    Drop‑in replacement for T5ForConditionalGeneration with a latent compressor.
    * generate(), save_pretrained(), HF Trainer all work unchanged. *
    """

    def __init__(
        self,
        base_model_name: str = "google/flan-t5-base",
        n_latents: int = 8,
        n_self_layers: int = 1,
    ):
        # 1) load vanilla T5
        super().__init__(T5Config.from_pretrained(base_model_name))
        # self.from_pretrained(base_model_name)

        state_dict = T5ForConditionalGeneration.from_pretrained(
            base_model_name
        ).state_dict()
        self.load_state_dict(state_dict, strict=True)

        # 2) build compressor + wrap encoder
        d_model = self.config.d_model
        n_heads = self.config.num_heads

        compressor = LatentQueryCompressor(
            d_model=d_model,
            n_heads=n_heads,
            n_latents=n_latents,
            n_self_layers=n_self_layers,
        )
        self.encoder = CompressedT5Encoder(
            self.encoder, self.shared, compressor
        )  # replace in‑place

        self.encoder.shared = self.shared
        self.tie_weights
        
    # optional: expose enc_hidden so Trainer can access it
    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.pop("attention_mask", None)
        outputs = super().forward(input_ids=input_ids, labels=labels, output_hidden_states=True, **kwargs)
        # outputs is Seq2SeqLMOutput; we add encoder hidden state for ranking later
        return outputs
        # return {
        #     "loss": outputs.loss,
        #     "logits": outputs.logits,
        #     "enc_hidden": outputs.encoder_last_hidden_state,
        # }
        
    def ripor_logprob(
        self, dec_hidden: torch.Tensor, doc_tokens: torch.Tensor,
    ):
        labels = doc_tokens.clone()
        labels[labels == -100] = self.config.pad_token_id
        decoder_embeds = self.decoder.embed_tokens(labels)
        print("margin times shape: ", (dec_hidden * decoder_embeds).shape, (dec_hidden * decoder_embeds))
        margin = (dec_hidden * decoder_embeds).sum(-1).sum(-1)
        return margin

    def sequence_logprob(
        self, enc_hidden: torch.Tensor, doc_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log P(doc_tokens | query) for each item in a batch.

        Args
        ----
        enc_hidden :  [B*,  Btok, d]     – encoder memory (from latent compressor)
        doc_tokens :  [B*,  L_doc]       – full DocID strings
                    (can contain -100 for padding)

        Returns
        -------
        logp_seq  :  [B*]                – summed log-probability per sequence
        """
        # 1.  prepend decoder-start token (<pad> for T5) to build the context
        start_tok = self.config.decoder_start_token_id
        start_col = torch.full(
            (doc_tokens.size(0), 1),
            start_tok,
            dtype=doc_tokens.dtype,
            device=doc_tokens.device,
        )
        dec_in = torch.cat(
            [start_col, doc_tokens[:, :-1]], dim=1
        )  # same length as target
        dec_in[dec_in == -100] = self.config.pad_token_id
        
        # 2.  replace -100 paddings in labels with <pad> so gather() is safe
        labels = doc_tokens.clone()
        pad_mask = labels == -100
        labels[pad_mask] = start_tok

        # 3.  run the decoder once
        dec_out = self.decoder(
            input_ids=dec_in.contiguous(),
            encoder_hidden_states=enc_hidden.contiguous(),
            use_cache=False,
        )
        sequence_output = dec_out["last_hidden_state"]
        decoder_embeds = self.decoder.embed_tokens(dec_in)
        
        seq_len = sequence_output.shape[1]
        n_dim = sequence_output.shape[2]
        
        margin = (sequence_output * decoder_embeds).sum(-1).sum(-1)
        return margin
        
        # if self.config.tie_word_embeddings:
        #     sequence_output = sequence_output * (self.model_dim ** -0.5)
        
        # logits = self.lm_head(sequence_output)
            
        # log_p = F.log_softmax(logits, dim=-1)  # [B*, L_doc, V]
        # # 4.  pick the log-prob assigned to each target token
        # tok_lp = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B*, L_doc]
        # tok_lp[pad_mask] = 0.0  # ignore padding
        # # 5.  sum over all positions → sequence log-prob
        # return tok_lp.sum(dim=-1)


if __name__ == "__main__":    
    from data import SlueSQA5DatasetV2, IndexingCollator
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import os
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataset = SlueSQA5DatasetV2(split="train", max_length=512, discrete_code_num=500)
    collator = IndexingCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)

    # Testing

    batch = next(iter(dataloader))
    batch = {k: v.to("cuda") for k, v in batch.items()}
    batch.pop("query_doc_id")
    model = LatentQueryT5().to("cuda")
    outputs = model(**batch)
    
    # testing sequence_logprob
    # enc_hidden = outputs["encoder_last_hidden_state"]
    dec_hidden = outputs["decoder_hidden_states"][-1]
    doc_tokens = batch["labels"]

    logp_seq = model.ripor_logprob(dec_hidden, doc_tokens)
    # logp_seq = model.sequence_logprob(dec_hidden, doc_tokens)
    print("logp_seq.shape: ", logp_seq.shape, logp_seq)

    # testing predict
    # input_ids = batch["input_ids"]
    # outputs = model.generate(input_ids)
    # print("generate outputs: ", outputs)
