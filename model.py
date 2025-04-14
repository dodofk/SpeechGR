import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class ContinousEmbT5(T5ForConditionalGeneration):
    def __init__(
        self,
        config,
        ssl_feat_dim: int = 1024,
        downsample_factor: int = 2,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        # For T5-small, embedding dim is typically 512
        # for T5-base, 768, etc. So let's read from the existing embedding matrix
        hidden_dim = self.get_input_embeddings().weight.size(-1)

        self.linear_adapter = nn.Linear(
            ssl_feat_dim * downsample_factor,  # e.g. 1024*2=2048 if you chunk frames
            hidden_dim
        )
        nn.init.xavier_uniform_(self.linear_adapter.weight)
        nn.init.constant_(self.linear_adapter.bias, 0.0)

    def forward(
        self,
        input_embeds=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
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
            **kwargs
        )