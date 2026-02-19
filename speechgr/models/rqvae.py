import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization Layer using Exponential Moving Average (EMA) updates.
    Includes 'Dead Code Revival' logic.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # EMA Buffers (Not learnable parameters)
        self.register_buffer('embedding', torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer('num_updates', torch.tensor([0], dtype=torch.long))
        
        # Initialization
        self.embedding.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.ema_w.data.copy_(self.embedding)
        self.ema_cluster_size.fill_(1)

    def forward(self, inputs):
        # inputs: [B, ..., D]
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # 1. Calculate Distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))
            
        # 2. Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 3. EMA Update (Training Only)
        if self.training:
            self.num_updates += 1
            # Update cluster size count
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            # Update embedding sum
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Update actual embeddings with laplace smoothing for denominator
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            
            # 4. Dead Code Revival
            # Wait for EMA buffers to warm up before reviving
            if self.num_updates > 100:
                dead_codes = cluster_size < 0.01
                if dead_codes.any():
                    num_dead = dead_codes.sum()
                    indices = torch.randperm(flat_input.size(0))[:num_dead]
                    if indices.size(0) > 0:
                        selected_vectors = flat_input[indices.to(inputs.device)]
                        if selected_vectors.size(0) < num_dead:
                            selected_vectors = selected_vectors.repeat(int(num_dead/selected_vectors.size(0)) + 1, 1)[:num_dead]
                        
                        self.embedding.data[dead_codes] = selected_vectors.detach()
                        self.ema_w.data[dead_codes] = self.embedding.data[dead_codes]
                        self.ema_cluster_size.data[dead_codes] = 1.0

        # 5. Quantize & Loss
        quantized = torch.matmul(encodings, self.embedding).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(input_shape[:-1])

class ResidualVectorQuantizerEMA(nn.Module):
    """
    Residual Vector Quantizer (RVQ) consisting of cascading EMA VQ layers.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int, commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, ..., D]
        quantized_out = 0
        residual = x
        all_losses = 0
        all_indices = []

        for layer in self.layers:
            quantized, loss, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_losses = all_losses + loss
            all_indices.append(indices)

        # distinct codes per layer: [B, ..., num_layers]
        all_indices = torch.stack(all_indices, dim=-1)

        return quantized_out, all_losses, all_indices

class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling for sequence aggregation.
    Formula: v = sum(softmax(W * x_t) * x_t)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        # mask: [B, T] (1 for valid, 0 for padding)
        scores = self.attention(x).squeeze(-1) # [B, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # [B, T, 1]
        pooled = torch.sum(x * weights, dim=1) # [B, D]
        return pooled

class DocumentRQVAE(nn.Module):
    """
    Robust Transformer-based RQ-VAE for Document Indexing.
    Forces full sequence reconstruction from a single bottleneck code.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        codebook_size: int,
        num_codebooks: int,
        max_seq_len: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        commitment_cost: float = 0.25,
        decay: float = 0.99
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # 1. Input Projection & Normalization
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.feature_norm = nn.LayerNorm(latent_dim)
        
        # 2. Transformer Encoder (Contextualizes features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4, 
            batch_first=True, dropout=dropout
        )
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Pooling
        self.pooling = AttentiveStatisticsPooling(latent_dim)
        
        # 4. Residual VQ (EMA Version)
        self.rvq = ResidualVectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            num_layers=num_codebooks,
            commitment_cost=commitment_cost,
            decay=decay
        )
        
        # 5. Decoder: Learnable Queries for Sequence Reconstruction
        self.positional_queries = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4, 
            batch_first=True, dropout=dropout
        )
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_head = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, input_dim]
        # mask: [B, T] (1 for valid, 0 for pad)
        B, T, _ = x.shape
        
        # --- Encode ---
        x_emb = self.input_norm(self.input_proj(x))
        
        # Create padding mask for Transformer (True = ignore)
        key_padding_mask = (mask == 0) if mask is not None else None
        
        z_seq = self.encoder_transformer(x_emb, src_key_padding_mask=key_padding_mask)
        z_seq = self.feature_norm(z_seq)
        
        # --- Pool & Quantize ---
        z_pooled = self.pooling(z_seq, mask).unsqueeze(1) # [B, 1, D]
        z_q, vq_loss, codes = self.rvq(z_pooled) # codes: [B, 1, num_layers]
        
        # --- Decode ---
        # 1. Prepare Queries: Slice learnable queries to match current batch sequence length
        queries = self.positional_queries[:, :T, :].expand(B, -1, -1)
        
        # 2. Cross-Attend: Queries attend to z_q (memory)
        # memory is [B, 1, D]
        recon_features = self.decoder_transformer(tgt=queries, memory=z_q)
        
        x_recon = self.output_head(recon_features)

        # --- Loss ---
        if mask is not None:
            # Broadcast mask to match [B, T, D]
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            recon_loss = F.mse_loss(x_recon * mask_expanded, x * mask_expanded, reduction='sum')
            # Normalize by valid frames * D
            recon_loss = recon_loss / (mask.sum() * self.input_dim)
        else:
            recon_loss = F.mse_loss(x_recon, x)
            
        return x_recon, recon_loss + vq_loss, codes.squeeze(1)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the document-level discrete codes."""
        self.eval() # Ensure no EMA updates during inference
        with torch.no_grad():
            x_emb = self.input_norm(self.input_proj(x))
            key_padding_mask = (mask == 0) if mask is not None else None
            z_seq = self.encoder_transformer(x_emb, src_key_padding_mask=key_padding_mask)
            z_seq = self.feature_norm(z_seq)
            z_pooled = self.pooling(z_seq, mask).unsqueeze(1)
            _, _, codes = self.rvq(z_pooled)
        return codes.squeeze(1) # [B, num_codebooks]

# Legacy RQVAE kept for potential sequence-level tasks or backwards compatibility
# but updated to use EMA for stability
class RQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, num_codebooks, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.rvq = ResidualVectorQuantizerEMA(codebook_size, latent_dim, num_codebooks, commitment_cost, decay)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, codes = self.rvq(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss + vq_loss, codes
    def encode(self, x):
        z = self.encoder(x)
        _, _, codes = self.rvq(z)
        return codes
