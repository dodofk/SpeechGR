# Sliding Window RQ-VAE Implementation Proposal

## Executive Summary

Replace global pooling with sliding window pooling to improve codebook utilization and preserve local temporal structure.

## Current vs Proposed

### Current (Global Pooling)
```
Input: [B, T=500, D=1024]
        ↓
  Global Pooling (AttentiveStatisticsPooling)
        ↓
Output: [B, 1, D=1024] → VQ → [B, 8] codes

Compression: 500:1
Problem: Too aggressive, 6.6% utilization
```

### Proposed (Sliding Window)
```
Input: [B, T=500, D=1024]
        ↓
  Sliding Window Pooling (window=25, stride=12)
        ↓
Windows: 500/12 ≈ 42 windows
        ↓
Per-Window VQ: [B, 42, D=1024] → [B, 42, 8] codes

Compression: ~12:1
Expected: 40-60% utilization
```

## Architecture Changes

### 1. New Pooling Module: `SlidingWindowStatsPooling`

```python
class SlidingWindowStatsPooling(nn.Module):
    """
    Sliding window pooling with overlap.

    Args:
        input_dim: Input feature dimension
        window_size: Frames per window (~0.5s = 25 frames)
        stride: Step between windows (overlap = window_size - stride)
        hidden_dim: Hidden dimension for attention

    Input: [B, T, D]
    Output: [B, num_windows, D]
    """
    def __init__(self, input_dim: int, window_size: int = 25,
                 stride: int = 12, hidden_dim: int = 128):
        self.window_size = window_size
        self.stride = stride
        self.pool = AttentiveStatisticsPooling(input_dim, hidden_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, T, D = x.shape

        # Extract sliding windows
        windows = []
        window_masks = []

        for start in range(0, T - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = x[:, start:end]  # [B, window_size, D]

            # Create mask for this window
            if mask is not None:
                window_mask = mask[:, start:end]
            else:
                window_mask = None

            # Pool this window
            pooled = self.pool(window, window_mask)  # [B, D]
            windows.append(pooled)

        # Stack: [B, num_windows, D]
        return torch.stack(windows, dim=1)
```

### 2. Modified DocumentRQVAE

```python
class DocumentRQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        codebook_size: int,
        num_codebooks: int,
        window_size: int = 25,      # NEW
        window_stride: int = 12,    # NEW
        max_seq_len: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        commitment_cost: float = 0.25,
        decay: float = 0.9
    ):
        # ... existing init ...

        # Replace single pooling with sliding window
        self.pooling = SlidingWindowStatsPooling(
            latent_dim,
            window_size=window_size,
            stride=window_stride
        )

        # VQ now processes multiple windows
        self.rvq = ResidualVectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            num_layers=num_codebooks,
            commitment_cost=commitment_cost,
            decay=decay
        )

        # Decoder now attends to multiple window codes
        # Use cross-attention where queries are positional
        # and memory is all window codes

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        B, T, _ = x.shape

        # Encode
        x_emb = self.input_norm(self.input_proj(x))
        z_seq = self.encoder_transformer(x_emb, src_key_padding_mask=...)
        z_seq = self.feature_norm(z_seq)

        # Pool to windows: [B, num_windows, D]
        z_windows = self.pooling(z_seq, mask)

        # VQ each window: [B, num_windows, D], [B, num_windows, num_codebooks]
        z_q_windows, vq_loss, codes = self.rvq(z_windows)

        # Decode: each position attends to all window codes
        # Use decoder with z_q_windows as memory
        queries = self.positional_queries[:, :T, :].expand(B, -1, -1)

        # Cross-attention: queries [B, T, D] attend to memory [B, num_windows, D]
        # Need custom attention or reshape
        recon_features = self.decoder_with_multi_memory(
            queries, z_q_windows
        )

        x_recon = self.output_head(recon_features)
        return x_recon, recon_loss + vq_loss, codes
```

### 3. Decoder with Multi-Window Memory

```python
class MultiWindowDecoder(nn.Module):
    """
    Decoder that attends to multiple window codes.
    """
    def __init__(self, latent_dim: int, num_layers: int = 4, nhead: int = 8):
        super().__init__()
        # Self-attention over positions
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=nhead,
                dim_feedforward=latent_dim*4, batch_first=True
            )
            for _ in range(num_layers // 2)
        ])

        # Cross-attention to window codes
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=nhead,
                batch_first=True
            )
            for _ in range(num_layers // 2)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Linear(latent_dim * 4, latent_dim)
            )
            for _ in range(num_layers // 2)
        ])

    def forward(self, queries: Tensor, memory: Tensor):
        """
        Args:
            queries: [B, T, D] - positional queries
            memory: [B, num_windows, D] - window codes
        Returns:
            [B, T, D] - reconstructed features
        """
        x = queries

        # Alternate self-attention and cross-attention
        for self_attn, cross_attn, ffn in zip(
            self.self_attn_layers, self.cross_attn_layers, self.ffn_layers
        ):
            # Self-attention over sequence
            x = self_attn(x)

            # Cross-attention to window codes
            attn_out, _ = cross_attn(x, memory, memory)
            x = x + attn_out

            # FFN
            x = x + ffn(x)

        return x
```

## Configuration

```yaml
# configs/training/rqvae_sliding_window.yaml

defaults:
  - rqvae  # Inherit from base

rqvae:
  pooling_type: "sliding_window"  # "global" or "sliding_window"
  window_size: 25      # ~0.5s at 50fps
  window_stride: 12    # 50% overlap

  # Alternative presets
  # window_size: 50, stride: 25  # 1s windows, 50% overlap
  # window_size: 10, stride: 5   # 0.2s windows, 50% overlap
```

## Expected Metrics

| Metric | Global Pooling | Sliding Window (w=25, s=12) |
|--------|---------------|----------------------------|
| Codes per 10s | 8 | ~336 (42 windows × 8) |
| Compression | 500:1 | ~12:1 |
| Expected Utilization | 6.6% | 40-60% |
| Reconstruction SNR | Poor | Better |
| Training Stability | Low | Higher |

## Implementation Plan

### Phase 1: Core Implementation (1-2 hours)
1. Create `SlidingWindowStatsPooling` class
2. Modify `DocumentRQVAE` to support both pooling types
3. Implement `MultiWindowDecoder`

### Phase 2: Integration (30 min)
1. Update training script to handle multiple codes
2. Update monitoring for per-window metrics
3. Add configuration options

### Phase 3: Testing (1 hour)
1. Test forward/backward pass
2. Verify code shapes
3. Run short training to check utilization

## Open Questions

1. **Window size**: Start with 25 frames (~0.5s) or 50 frames (~1s)?
2. **Overlap**: 50% overlap (stride=window/2) or no overlap?
3. **Code aggregation for retrieval**:
   - Option A: Use all window codes (336 codes per doc)
   - Option B: Mean pool quantized windows (8 codes per doc)
   - Option C: Attention-based aggregation

## Recommendation

Start with:
- `window_size=25` (~0.5s)
- `stride=12` (~50% overlap)
- Option B for retrieval (mean pool to 8 codes)

This gives:
- Rich representation during training (42 windows)
- Manageable for retrieval (8 codes after pooling)
- Good balance of detail and compression

## Next Steps

Ready to implement? I can:
1. Create the `SlidingWindowStatsPooling` class
2. Update `DocumentRQVAE` with multi-window support
3. Update training script
4. Push to git

Estimated time: 2-3 hours
