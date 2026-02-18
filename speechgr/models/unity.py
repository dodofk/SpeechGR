import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from speechgr.models.ssl_wrapper import SSLModelWrapper

class UnitySpeechModel(nn.Module):
    """
    S2GSR-UnitY Model Architecture.
    Dual-decoder setup: Semantic Bridge (Decoder 1) + Retrieval Head (Decoder 2).
    """
    def __init__(
        self,
        ssl_model_name: str = "microsoft/wavlm-large",
        ssl_layer: int = 24,
        semantic_vocab_size: int = 5000,
        retrieval_vocab_size: int = 256,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ):
        super().__init__()
        
        # 1. Encoder (WavLM + Downsampling)
        self.ssl_model = SSLModelWrapper(ssl_model_name, layer=ssl_layer)
        self.downsample = nn.Conv1d(
            self.ssl_model.feature_dim, 
            d_model, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 2. Decoder 1 (Semantic Bridge)
        self.semantic_embed = nn.Embedding(semantic_vocab_size, d_model)
        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.semantic_decoder = nn.TransformerDecoder(decoder_layer1, num_layers)
        self.semantic_head = nn.Linear(d_model, semantic_vocab_size)
        
        # 3. Decoder 2 (Retrieval Head)
        self.retrieval_embed = nn.Embedding(retrieval_vocab_size, d_model)
        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.retrieval_decoder = nn.TransformerDecoder(decoder_layer2, num_layers)
        self.retrieval_head = nn.Linear(d_model, retrieval_vocab_size)
        
        self.d_model = d_model
        self.semantic_vocab_size = semantic_vocab_size
        self.retrieval_vocab_size = retrieval_vocab_size

    def forward(
        self, 
        input_values: torch.Tensor, 
        semantic_labels: Optional[torch.Tensor] = None, 
        retrieval_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for training.
        Args:
            input_values: (B, T)
            semantic_labels: (B, L_sem)
            retrieval_labels: (B, L_ret)
            attention_mask: (B, T)
        """
        # 1. Encoder
        h_enc = self.ssl_model(input_values, attention_mask=attention_mask) # [B, T_raw, D_ssl]
        h_enc = h_enc.transpose(1, 2)
        h_enc = self.downsample(h_enc).transpose(1, 2) # [B, T_enc, d_model]
        h_enc = h_enc + self.pos_encoder[:, :h_enc.size(1), :]
        
        # 2. Decoder 1 (Semantic)
        semantic_logits = None
        h_dec1 = None
        if semantic_labels is not None:
            # Standard Seq2Seq teacher forcing
            # labels typically contain <BOS> ... <EOS>
            # We use labels[:, :-1] as input, labels[:, 1:] as target
            sem_in = self.semantic_embed(semantic_labels[:, :-1])
            sem_in = sem_in + self.pos_encoder[:, :sem_in.size(1), :]
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sem_in.size(1)).to(sem_in.device)
            
            h_dec1 = self.semantic_decoder(
                tgt=sem_in,
                memory=h_enc,
                tgt_mask=tgt_mask
            )
            semantic_logits = self.semantic_head(h_dec1)
            
        # 3. Decoder 2 (Retrieval)
        retrieval_logits = None
        if retrieval_labels is not None:
            ret_in = self.retrieval_embed(retrieval_labels[:, :-1])
            ret_in = ret_in + self.pos_encoder[:, :ret_in.size(1), :]
            
            # UnitY Fusion: Memory is concat of H_enc and H_dec1
            if h_dec1 is not None:
                memory_fused = torch.cat([h_enc, h_dec1], dim=1)
            else:
                memory_fused = h_enc
            
            tgt_mask_ret = nn.Transformer.generate_square_subsequent_mask(ret_in.size(1)).to(ret_in.device)
            
            h_dec2 = self.retrieval_decoder(
                tgt=ret_in,
                memory=memory_fused,
                tgt_mask=tgt_mask_ret
            )
            retrieval_logits = self.retrieval_head(h_dec2)
            
        return semantic_logits, retrieval_logits

    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_sem_len: int = 50,
        max_ret_len: int = 8,
        sem_start_token: int = 0,
        ret_start_token: int = 0,
        num_beams: int = 1,
        logits_processor = None
    ):
        """
        Inference: Generate Semantic tokens then Retrieval codes.
        Supports Beam Search for Retrieval codes.
        """
        device = input_values.device
        B = input_values.size(0)
        
        # 1. Encode
        h_enc = self.ssl_model(input_values, attention_mask=attention_mask)
        h_enc = h_enc.transpose(1, 2)
        h_enc = self.downsample(h_enc).transpose(1, 2)
        h_enc = h_enc + self.pos_encoder[:, :h_enc.size(1), :]
        
        # 2. Generate Semantic (Greedy for now - Bridge only)
        sem_tokens = torch.full((B, 1), sem_start_token, device=device, dtype=torch.long)
        h_dec1 = None
        
        for _ in range(max_sem_len):
            tgt_emb = self.semantic_embed(sem_tokens) + self.pos_encoder[:, :sem_tokens.size(1), :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sem_tokens.size(1)).to(device)
            
            h_dec1 = self.semantic_decoder(tgt=tgt_emb, memory=h_enc, tgt_mask=tgt_mask)
            logits = self.semantic_head(h_dec1[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            sem_tokens = torch.cat([sem_tokens, next_token], dim=1)
            
            # Simple early stopping if all generated 0 (EOS)
            if (next_token == 0).all():
                break
        
        # 3. Generate Retrieval (Beam Search or Greedy)
        memory_fused = torch.cat([h_enc, h_dec1], dim=1)
        
        if num_beams == 1:
            # Greedy Search
            ret_tokens = torch.full((B, 1), ret_start_token, device=device, dtype=torch.long)
            for i in range(max_ret_len):
                tgt_emb = self.retrieval_embed(ret_tokens) + self.pos_encoder[:, :ret_tokens.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(ret_tokens.size(1)).to(device)
                
                h_dec2 = self.retrieval_decoder(tgt=tgt_emb, memory=memory_fused, tgt_mask=tgt_mask)
                logits = self.retrieval_head(h_dec2[:, -1:, :])
                
                if logits_processor is not None:
                    logits = logits_processor(ret_tokens, logits.squeeze(1)).unsqueeze(1)
                    
                next_token = torch.argmax(logits, dim=-1)
                ret_tokens = torch.cat([ret_tokens, next_token], dim=1)
            return sem_tokens, ret_tokens
        
        else:
            # Custom Beam Search
            from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor
            
            # Expand memory for beams: [B * num_beams, MemLen, D]
            memory_fused_expanded = memory_fused.repeat_interleave(num_beams, dim=0)
            
            # Initial input: [B * num_beams, 1]
            input_ids = torch.full((B * num_beams, 1), ret_start_token, device=device, dtype=torch.long)
            
            beam_scorer = BeamSearchScorer(
                batch_size=B,
                num_beams=num_beams,
                device=device,
                length_penalty=1.0, # Neutral
                do_early_stopping=False, # We want exactly 8 codes usually
            )
            
            logits_processor_list = LogitsProcessorList()
            if logits_processor is not None:
                logits_processor_list.append(logits_processor)
            
            # Tracking finished sequences
            # Since we are not using HF model.generate, we manually loop
            
            beam_scores = torch.zeros((B, num_beams), dtype=torch.float, device=device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((B * num_beams,))

            cur_len = 1
            while cur_len <= max_ret_len:
                # Forward Pass
                tgt_emb = self.retrieval_embed(input_ids) + self.pos_encoder[:, :input_ids.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(1)).to(device)
                
                h_dec2 = self.retrieval_decoder(tgt=tgt_emb, memory=memory_fused_expanded, tgt_mask=tgt_mask)
                next_token_logits = self.retrieval_head(h_dec2[:, -1, :]) # [B*Beams, V]
                
                # Logits Processing (Constraints)
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                next_token_scores = logits_processor_list(input_ids, next_token_scores)
                
                # Beam Step
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                
                # Reshape for beam selection: [B, Beams * V]
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(B, num_beams * vocab_size)
                
                # Top-K
                next_scores, next_tokens = torch.topk(next_token_scores, num_beams, dim=1, largest=True, sorted=True)
                
                # Indices reconstruction
                next_batch_idx = torch.div(next_tokens, vocab_size, rounding_mode='floor')
                next_token_indices = next_tokens % vocab_size
                
                # Update scores
                beam_scores = next_scores.view(-1)
                
                # Reorder inputs based on selected beams
                # We need to map [B, Beams] back to [B*Beams]
                # next_batch_idx contains beam index (0..num_beams-1) relative to batch
                
                # Global beam indices
                beam_offset = torch.arange(0, B * num_beams, step=num_beams, device=device, dtype=torch.long).unsqueeze(1)
                global_beam_idx = beam_offset + next_batch_idx
                global_beam_idx = global_beam_idx.view(-1)
                
                input_ids = input_ids[global_beam_idx]
                input_ids = torch.cat([input_ids, next_token_indices.view(-1, 1)], dim=-1)
                
                cur_len += 1
                
            # Return top beam for each batch
            # input_ids is [B*Beams, L]
            # beam_scores is [B*Beams] (already sorted per batch group)
            
            # Reshape
            final_hyps = input_ids.view(B, num_beams, -1)
            # return all beams so evaluation can calculate Recall@K
            return sem_tokens, final_hyps
