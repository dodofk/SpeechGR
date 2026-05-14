import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    # Seq2SeqLMOutput,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
import math


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
          kwargs: anything else T5's forward might expect (like decoder_input_ids, etc.).
        """
        # 1) Project the custom input embeddings into T5's hidden dimension
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
        outputs = super().forward(
            input_ids=input_ids, labels=labels, output_hidden_states=True, **kwargs
        )
        # outputs is Seq2SeqLMOutput; we add encoder hidden state for ranking later
        return outputs

    def ripor_logprob(
        self,
        dec_hidden: torch.Tensor,
        doc_tokens: torch.Tensor,
    ):
        labels = doc_tokens.clone()
        labels[labels == -100] = self.config.pad_token_id
        decoder_embeds = self.decoder.embed_tokens(labels)
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


class QFormerBlock(nn.Module):
    """Cross-Attn → Self-Attn → FFN with pre-norm residuals."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        # layer norms
        self.ln_q1 = nn.LayerNorm(d_model)
        self.ln_q2 = nn.LayerNorm(d_model)
        self.ln_q3 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # q: (B, M, D) learnable queries
        # x: (B, L, D) window frames (no grad through x if upstream encoder frozen)
        q = q + self.cross_attn(self.ln_q1(q), x, x, need_weights=False)[0]
        q = (
            q
            + self.self_attn(
                self.ln_q2(q), self.ln_q2(q), self.ln_q2(q), need_weights=False
            )[0]
        )
        q = q + self.ffn(self.ln_q3(q))
        return q


class WindowQFormer(nn.Module):
    """
    Args:
        d_model: feature dim (must match speech encoder output, we use discrete unit, so dim same as t5 text embedding dim)
        n_heads: multi-head attn heads
        n_queries: queries per window (N)
        depth: number of Q-Former blocks
        win_size_f: window length **in frames** (L)
        win_stride_f: hop size in frames
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_queries: int = 1,
        depth: int = 2,
        win_size_f: int = 17,
        win_stride_f: int = 17,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.win_size_f = win_size_f
        self.win_stride_f = win_stride_f

        # shared learnable queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)

        # stack of Q-Former blocks
        self.blocks = nn.ModuleList(
            [QFormerBlock(d_model, n_heads, dropout) for _ in range(depth)]
        )
        self.ln_out = nn.LayerNorm(d_model)

    # --------------------------------------------------------------
    def _framed_windows(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Slice (B, T, D) into overlapping windows → list[(B, L, D)]."""
        B, T, D = x.shape
        windows = []
        for start in range(0, T, self.win_stride_f):
            end = start + self.win_size_f
            if end > T:
                pad_len = end - T
                pad = x.new_zeros(B, pad_len, D)
                window = torch.cat([x[:, start:T, :], pad], dim=1)
            else:
                window = x[:, start:end, :]
            windows.append(window)
            if end >= T:
                break
        return windows

    # --------------------------------------------------------------
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats : (B, T_frames, D) – speech encoder output
        Returns compressed tokens: (B, W*n_queries, D)
        """
        summaries = []
        for window in self._framed_windows(feats):
            q = self.queries.expand(window.size(0), -1, -1)  # (B, M, D)
            for blk in self.blocks:
                q = blk(q, window)
            summaries.append(self.ln_out(q))
        return torch.cat(summaries, dim=1)  # (B, Windows*M, D)


class QFormerEncoderWrapper(nn.Module):
    """Replaces vanilla T5 encoder with window-level Q-Former → T5 encoder."""

    def __init__(
        self,
        base_encoder: nn.Module,
        shared_emb: nn.Embedding,
        compressor: WindowQFormer,
        freeze_base_encoder: bool = True,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.shared = shared_emb
        self.compressor = compressor

        if freeze_base_encoder:
            for p in self.base_encoder.parameters():
                p.requires_grad = False

    @property
    def main_input_name(self):
        # keeps HF Trainer happy
        return "input_ids"

    # ----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,  # (B,T,D) speech feats
        attention_mask: torch.Tensor = None,  # ignored; recomputed
        **kwargs,
    ):
        """
        1) compress long feat seq → (B, S', D) S' is the equals to seq_len / win_stride_f
        2) run (frozen) T5 encoder on compressed sequence
        """
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)

        latent = self.compressor(inputs_embeds)

        return self.base_encoder(
            inputs_embeds=latent, attention_mask=attention_mask, **kwargs
        )


class QFormerT5(T5ForConditionalGeneration):
    """
    Usage
    -----
    >>> model = QFormerT5(
    ...     "google/flan-t5-base",
    ...     d_model_front=1024,
    ...     win_size_f=17, win_stride_f=17,
    ...     n_queries=1, depth=2,
    ...     use_whisper_features=True,  # Enable Whisper feature input
    ... )
    >>> feats = whisper_features  # continuous Whisper features
    >>> seq = model.generate(input_features=feats)  # Use input_features for Whisper
    """

    def __init__(
        self,
        base_name: str = "google/flan-t5-base",
        *,
        d_model_front: int,
        win_size_f: int = 17,
        win_stride_f: int = 17,
        n_queries: int = 1,
        depth: int = 2,
        freeze_t5_encoder: bool = True,
        use_whisper_features: bool = False,  # New parameter to control input type
    ):
        # 1) load vanilla weights
        super().__init__(T5Config.from_pretrained(base_name))
        self.load_state_dict(
            T5ForConditionalGeneration.from_pretrained(base_name).state_dict(),
            strict=True,
        )
        self.win_size_f = win_size_f
        self.win_stride_f = win_stride_f
        self.n_queries = n_queries
        self.use_whisper_features = use_whisper_features

        # 2) build Q-Former compressor (d_model_front may ≠ T5 d_model)
        if d_model_front != self.config.d_model:
            self.front_proj = nn.Linear(d_model_front, self.config.d_model)
        else:
            self.front_proj = nn.Identity()

        compressor = WindowQFormer(
            d_model=self.config.d_model,
            n_heads=self.config.num_heads,
            n_queries=n_queries,
            depth=depth,
            win_size_f=win_size_f,
            win_stride_f=win_stride_f,
        )

        # 3) wrap / replace encoder
        self.encoder = QFormerEncoderWrapper(
            self.encoder,
            self.shared,
            compressor,
            freeze_base_encoder=freeze_t5_encoder,
        )
        self.encoder.shared = self.shared  # keep tie_weights happy

    def _build_window_mask(
        self,
        orig_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress a frame-level attention mask (B, Tf) → (B, W*n_q)

        A window is 'valid' (mask =1) iff at least one of its `win` frames is 1.
        The flag is then repeated `n_q` times because every window yields n_q
        query tokens.

        orig_mask : 1 = real frame, 0 = padding
        """
        _, Tf = orig_mask.shape
        out_chunks = []
        for start in range(0, Tf, self.win_stride_f):
            end = start + self.win_size_f
            # clamp end so we don't step beyond `Tf`
            seg = orig_mask[:, start : min(end, Tf)]
            has_real = (seg.sum(dim=1, keepdim=True) > 0).long()  # (B,1)
            out_chunks.append(has_real.repeat(1, self.n_queries))  # (B,n_q)
            if end >= Tf:
                break
        return torch.cat(out_chunks, dim=1).to(orig_mask.device)

    def _build_full_window_mask(self, orig_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate a full window mask for the input sequence. Where the size is same after the sliding window.
        """
        B, TF = orig_mask.shape
        n_win = max(1, math.ceil((TF - self.win_size_f) / self.win_stride_f) + 1)
        S = n_win * self.n_queries

        return torch.ones(B, S, device=orig_mask.device, dtype=torch.long)
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids, model_kwargs, model_input_name, generation_config, 
    ):
        # HuggingFace will call this before running the encoder.
        # Handle both discrete units and Whisper features using inputs_embeds
        frame_mask = model_kwargs.pop("attention_mask", None)
        
        # For generation, always use inputs_embeds (HuggingFace standard)
        features = model_kwargs.pop("inputs_embeds", None)
        if features is not None:
            features = self.front_proj(features)

        # Build the windowed mask
        if frame_mask is not None:
            window_mask = self._build_window_mask(frame_mask)
        else:
            # Create a full mask if no attention mask provided
            if features is not None:
                dummy_mask = torch.ones(features.shape[:2], device=features.device)
                window_mask = self._build_window_mask(dummy_mask)
            else:
                window_mask = None

        # Now actually run the encoder once and stash `encoder_outputs`
        # When using inputs_embeds, input_ids should be None
        encoder_outputs = self.encoder(
            input_ids=None,  # Set to None when using inputs_embeds
            inputs_embeds=features,
            attention_mask=window_mask,
        )

        return {
            "encoder_outputs": encoder_outputs, 
            "attention_mask": window_mask
        }

    # ----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,  # For discrete units
        input_features: torch.Tensor = None,  # For Whisper features  
        attention_mask=None,
        **kwargs,
    ):
        # Determine which type of input to use
        if self.use_whisper_features and input_features is not None:
            # Use Whisper features
            proj_features = self.front_proj(input_features)
            input_embeds_to_use = proj_features
        elif inputs_embeds is not None:
            # Use discrete unit embeddings
            proj_features = self.front_proj(inputs_embeds)
            input_embeds_to_use = proj_features
        else:
            # Use input_ids with shared embedding lookup
            input_embeds_to_use = None

        # Handle attention mask and encoder outputs
        if kwargs.get("encoder_outputs", None) is not None:
            new_mask = attention_mask # if we have encoder output which means our attention mask already has window mask
        elif attention_mask is not None:
            new_mask = self._build_window_mask(attention_mask)
        else:
            if input_embeds_to_use is not None:
                # Create dummy mask based on input sequence length
                dummy_mask = torch.ones(input_embeds_to_use.shape[:2], device=input_embeds_to_use.device)
                new_mask = self._build_window_mask(dummy_mask)
            else:
                print("Info: Potential Error no attention mask or encoder output")
                new_mask = self._build_full_window_mask(attention_mask)

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=input_embeds_to_use,
            attention_mask=new_mask,
            **kwargs,
        )


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
    model = QFormerT5(
        base_name="google/flan-t5-base",
        d_model_front=768,
        win_size_f=17,
        win_stride_f=17,
        n_queries=1,
        depth=2,
        use_whisper_features=False,  # Set to False for discrete units
    ).to("cuda")
    # testing sequence_logprob
    outputs = model(**batch)
    print("Outputs: ", outputs)
    # testing predict
    # input_ids = batch["input_ids"]
    # outputs = model.generate(input_ids)
    # print("generate outputs: ", outputs)
