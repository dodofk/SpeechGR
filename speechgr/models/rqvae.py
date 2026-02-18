import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantization Layer.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs: [B, T, D]
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(inputs.shape[0], -1)

class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) consisting of cascading VQ layers.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int, commitment_cost: float = 0.25):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
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

        # distinct codes per layer: [B, T, num_layers]
        all_indices = torch.stack(all_indices, dim=-1)

        return quantized_out, all_losses, all_indices

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection

class RQVAE(nn.Module):
    """
    RQ-VAE Model.
    Projects input features to a latent space, quantizes them using RVQ, and projects back.
    Includes ResBlocks for better capacity.
    """
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        codebook_size: int, 
        num_codebooks: int, 
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            ResBlock(latent_dim),
            ResBlock(latent_dim),
        )
        
        self.rvq = ResidualVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            num_layers=num_codebooks,
            commitment_cost=commitment_cost
        )
        
        self.decoder = nn.Sequential(
            ResBlock(latent_dim),
            ResBlock(latent_dim),
            nn.Linear(latent_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, input_dim]
        
        # 1. Encode
        z = self.encoder(x)
        
        # 2. Quantize
        z_q, vq_loss, codes = self.rvq(z)
        
        # 3. Decode
        x_recon = self.decoder(z_q)
        
        # 4. Recon Loss
        recon_loss = F.mse_loss(x_recon, x)
        
        total_loss = recon_loss + vq_loss
        
        return x_recon, total_loss, codes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the discrete codes for the input."""
        z = self.encoder(x)
        _, _, codes = self.rvq(z)
        return codes

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
    RQ-VAE for Document-level indexing.
    Pools variable length features into a single vector before quantization.
    Includes ResBlocks for better capacity.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        codebook_size: int,
        num_codebooks: int,
        commitment_cost: float = 0.25,
        pooling_hidden_dim: int = 128
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            ResBlock(latent_dim),
            ResBlock(latent_dim),
        )
        
        self.pooling = AttentiveStatisticsPooling(latent_dim, pooling_hidden_dim)
        
        self.rvq = ResidualVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            num_layers=num_codebooks,
            commitment_cost=commitment_cost
        )
        
        self.decoder = nn.Sequential(
            ResBlock(latent_dim),
            ResBlock(latent_dim),
            nn.Linear(latent_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, input_dim]
        
        # 1. Encode
        z = self.encoder(x)
        
        # 2. Pool
        z_pooled = self.pooling(z, mask).unsqueeze(1) # [B, 1, latent_dim]
        
        # 3. Quantize
        z_q, vq_loss, codes = self.rvq(z_pooled)
        
        # 4. Decode
        x_recon = self.decoder(z_q.squeeze(1)) # [B, input_dim]
        
        # 5. Target for reconstruction: pooled original features
        with torch.no_grad():
            x_target = x.mean(dim=1) if mask is None else (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        recon_loss = F.mse_loss(x_recon, x_target)
        total_loss = recon_loss + vq_loss
        
        return x_recon, total_loss, codes.squeeze(1) # [B, num_codebooks]

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the document-level discrete codes."""
        z = self.encoder(x)
        z_pooled = self.pooling(z, mask).unsqueeze(1)
        _, _, codes = self.rvq(z_pooled)
        return codes.squeeze(1) # [B, num_codebooks]
