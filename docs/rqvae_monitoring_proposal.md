# RQ-VAE Training Monitoring Proposal

## Executive Summary

This proposal outlines a comprehensive monitoring strategy for the DocumentRQVAE training pipeline to detect and diagnose potential issues early. Based on code review, we identified several areas requiring monitoring: codebook utilization, EMA stability, reconstruction quality, and training dynamics.

---

## 1. Codebook Health Monitoring

### 1.1 Core Metrics

| Metric | Definition | Target Range | Alert Threshold |
|--------|------------|--------------|-----------------|
| **Utilization Rate** | % of codes used in past N batches | 60-90% | < 50% or > 95% |
| **Perplexity** | exp(-Σ p_i log p_i) | Close to num_embeddings | < num_embeddings/10 |
| **Dead Code Count** | Codes with cluster_size < threshold | < 20% | > 30% |
| **Usage Entropy** | Evenness of code usage | High (close to max) | Sudden drops |

### 1.2 Implementation

```python
class CodebookMonitor:
    def __init__(self, num_embeddings, num_layers):
        self.history = defaultdict(list)
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers

    def update(self, all_indices: torch.Tensor):
        """
        Args:
            all_indices: [B, num_layers] tensor of code indices per layer
        """
        for layer_idx in range(self.num_layers):
            layer_indices = all_indices[:, layer_idx]

            # Usage counts
            usage_counts = torch.bincount(
                layer_indices.flatten(),
                minlength=self.num_embeddings
            )

            # Utilization rate
            active_codes = (usage_counts > 0).sum().item()
            utilization = active_codes / self.num_embeddings

            # Perplexity (measure of code distribution entropy)
            probs = usage_counts.float() / usage_counts.sum()
            probs = probs[probs > 0]  # Remove zeros for log
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs))).item()

            # Dead codes (usage below threshold)
            dead_codes = (usage_counts < 5).sum().item()

            self.history[f'layer_{layer_idx}_utilization'].append(utilization)
            self.history[f'layer_{layer_idx}_perplexity'].append(perplexity)
            self.history[f'layer_{layer_idx}_dead_codes'].append(dead_codes)

    def get_summary(self):
        """Returns current state and trend analysis."""
        summary = {}
        for layer_idx in range(self.num_layers):
            util_history = self.history[f'layer_{layer_idx}_utilization'][-100:]
            summary[f'layer_{layer_idx}'] = {
                'current_utilization': util_history[-1] if util_history else 0,
                'utilization_trend': np.polyfit(range(len(util_history)), util_history, 1)[0]
                                    if len(util_history) > 10 else 0,
                'avg_perplexity': np.mean(self.history[f'layer_{layer_idx}_perplexity'][-100:]),
                'dead_code_ratio': self.history[f'layer_{layer_idx}_dead_codes'][-1] / self.num_embeddings
            }
        return summary
```

### 1.3 Dashboard Metrics to Log

```python
# Log to wandb every N steps
wandb.log({
    # Per-layer metrics
    f"codebook/layer_{i}_utilization": utilization,
    f"codebook/layer_{i}_perplexity": perplexity,
    f"codebook/layer_{i}_dead_codes": dead_count,

    # Aggregated metrics
    "codebook/avg_utilization": avg_util,
    "codebook/avg_perplexity": avg_perplex,
    "codebook/worst_layer_utilization": min_util,

    # Layer-to-layer analysis
    "codebook/layer_usage_correlation": correlation_matrix,
})
```

---

## 2. EMA Stability Monitoring

### 2.1 Metrics to Track

| Metric | Purpose | Alert Condition |
|--------|---------|-----------------|
| **EMA Update Magnitude** | Track embedding movement | Sudden spikes |
| **Cluster Size Ratio** | Monitor EMA cluster sizes dropping | cluster_size < 0.1 |
| **Embedding Variance** | Check for collapsing embeddings | Variance < 1e-6 |
| **Revival Rate** | Track dead code revival frequency | > 10 revivals/step |

### 2.2 Implementation

```python
class EMAMonitor:
    def __init__(self, vq_layers):
        self.vq_layers = vq_layers
        self.prev_embeddings = [None] * len(vq_layers)

    def update(self):
        logs = {}
        for i, vq in enumerate(self.vq_layers):
            # Embedding movement
            if self.prev_embeddings[i] is not None:
                movement = torch.norm(
                    vq.embedding - self.prev_embeddings[i],
                    dim=1
                ).mean().item()
                logs[f"ema/layer_{i}_embedding_movement"] = movement

            # Store for next comparison
            self.prev_embeddings[i] = vq.embedding.clone().detach()

            # Cluster size statistics
            cluster_sizes = vq.ema_cluster_size.cpu().numpy()
            logs[f"ema/layer_{i}_cluster_size_mean"] = cluster_sizes.mean()
            logs[f"ema/layer_{i}_cluster_size_min"] = cluster_sizes.min()
            logs[f"ema/layer_{i}_cluster_size_std"] = cluster_sizes.std()

            # Check for very small clusters (potential dead codes)
            tiny_clusters = (cluster_sizes < 0.01).sum()
            logs[f"ema/layer_{i}_tiny_clusters"] = tiny_clusters

        return logs
```

---

## 3. Reconstruction Quality Monitoring

### 3.1 Metrics Beyond MSE Loss

| Metric | Description | When to Alert |
|--------|-------------|---------------|
| **SNR** | Signal-to-Noise Ratio | < 10 dB |
| **Feature Correlation** | Cosine similarity between input/output | < 0.8 |
| **Temporal Consistency** | Frame-to-frame smoothness | Sudden drops |
| **Spectral Distance** | Frequency domain comparison | > threshold |

### 3.2 Implementation

```python
class ReconstructionMonitor:
    def compute_metrics(self, original, reconstructed, mask=None):
        """
        Args:
            original: [B, T, D] input features
            reconstructed: [B, T, D] output features
            mask: [B, T] valid frame mask
        """
        metrics = {}

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            original = original * mask_expanded
            reconstructed = reconstructed * mask_expanded
            normalizer = mask.sum() * original.size(-1)
        else:
            normalizer = original.numel()

        # MSE (already computed in model)
        mse = F.mse_loss(reconstructed, original, reduction='sum') / normalizer
        metrics['recon/mse'] = mse.item()

        # SNR calculation
        signal_power = (original ** 2).sum() / normalizer
        noise_power = ((original - reconstructed) ** 2).sum() / normalizer
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        metrics['recon/snr_db'] = snr.item()

        # Feature correlation (per-frame cosine similarity)
        cos_sim = F.cosine_similarity(
            original.flatten(0, 1),
            reconstructed.flatten(0, 1),
            dim=-1
        )
        if mask is not None:
            cos_sim = cos_sim * mask.flatten()
            cos_sim = cos_sim.sum() / mask.sum()
        else:
            cos_sim = cos_sim.mean()
        metrics['recon/cosine_similarity'] = cos_sim.item()

        # Temporal consistency (L2 distance between consecutive frames)
        if original.size(1) > 1:
            orig_diff = torch.norm(original[:, 1:] - original[:, :-1], dim=-1)
            recon_diff = torch.norm(reconstructed[:, 1:] - reconstructed[:, :-1], dim=-1)
            temp_consistency = F.cosine_similarity(orig_diff.flatten(0, 1),
                                                   recon_diff.flatten(0, 1), dim=0)
            metrics['recon/temporal_consistency'] = temp_consistency.item()

        return metrics
```

### 3.3 Periodic Full Evaluation

```python
def run_full_eval(model, val_loader, ssl_model, device):
    """Run comprehensive evaluation every N epochs."""
    model.eval()
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            recon, loss, codes = model(features, mask)

            # Collect per-sample metrics
            metrics = reconstruction_monitor.compute_metrics(features, recon, mask)
            for k, v in metrics.items():
                all_metrics[k].append(v)

            # Store codes for codebook analysis
            all_codes.append(codes.cpu())

    # Compute statistics
    summary = {
        k: {'mean': np.mean(v), 'std': np.std(v), 'min': np.min(v), 'max': np.max(v)}
        for k, v in all_metrics.items()
    }

    return summary
```

---

## 4. Training Dynamics Monitoring

### 4.1 Gradient and Optimization Health

```python
class TrainingMonitor:
    def log_step(self, model, optimizer, loss_components, step):
        logs = {}

        # Loss component ratios
        total = loss_components['total']
        logs['loss_ratio/recon'] = loss_components['recon'] / total
        logs['loss_ratio/vq'] = loss_components['vq'] / total

        # Gradient norms per component
        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f'grad_norm/{name}'] = param.grad.norm().item()

        # Global gradient norm (already clipped, but log pre-clip)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        logs['grad_norm/global'] = total_norm.item()

        # Learning rate
        logs['train/lr'] = optimizer.param_groups[0]['lr']

        # Check for issues
        if total_norm > 10:
            logs['alerts/gradient_spike'] = True
        if torch.isnan(loss_components['total']):
            logs['alerts/nan_loss'] = True

        return logs
```

### 4.2 Loss Decomposition Tracking

Track how VQ loss vs Reconstruction loss evolves:

```python
# Expected healthy pattern:
# - recon_loss: decreases steadily (target: < 0.1)
# - vq_loss: stabilizes early, small fluctuations (target: < 0.5)
# - ratio vq/recon: should be < 5 (if higher, VQ is dominating)

wandb.log({
    "loss_decomposition/recon_vs_vq_ratio": vq_loss / (recon_loss + 1e-8),
    "loss_decomposition/total_trend": smoothed_total_loss,
})
```

---

## 5. Automated Alert System

### 5.1 Alert Conditions

```python
ALERT_RULES = {
    # Critical - Stop training
    'nan_loss': lambda logs: torch.isnan(logs['train/loss_total']),
    'zero_gradients': lambda logs: logs.get('grad_norm/global', 1) < 1e-8,
    'codebook_collapse': lambda logs: logs.get('codebook/avg_utilization', 1) < 0.1,

    # Warning - Log and continue
    'low_utilization': lambda logs: logs.get('codebook/avg_utilization', 1) < 0.5,
    'high_vq_ratio': lambda logs: logs.get('loss_decomposition/recon_vs_vq_ratio', 0) > 10,
    'dead_code_spike': lambda logs: logs.get('codebook/dead_code_revival_rate', 0) > 5,

    # Info - Just log
    'ema_warmup_complete': lambda logs: logs.get('ema/num_updates', 0) == 100,
}
```

### 5.2 Alert Handler

```python
class AlertManager:
    def __init__(self, config):
        self.config = config
        self.alert_history = []

    def check(self, logs, step):
        alerts = []
        for rule_name, check_fn in ALERT_RULES.items():
            if check_fn(logs):
                alerts.append({
                    'step': step,
                    'rule': rule_name,
                    'severity': self.get_severity(rule_name),
                    'message': self.format_message(rule_name, logs)
                })

        # Send notifications (wandb, slack, etc.)
        for alert in alerts:
            self.send_alert(alert)

        return alerts
```

---

## 6. Implementation Roadmap

### Phase 1: Basic Monitoring (Week 1)
- [ ] Add `CodebookMonitor` to training loop
- [ ] Log per-layer utilization to wandb
- [ ] Add SNR metric to reconstruction
- [ ] Set up basic alerting (NaN, zero grad)

### Phase 2: Advanced Diagnostics (Week 2)
- [ ] Implement `EMAMonitor`
- [ ] Add gradient norm tracking per layer
- [ ] Create evaluation script with full metrics
- [ ] Set up periodic checkpoint evaluation

### Phase 3: Visualization & Analysis (Week 3)
- [ ] Codebook usage heatmaps
- [ ] Reconstruction quality plots
- [ ] Loss decomposition visualizations
- [ ] Trend analysis dashboard

---

## 7. Expected Normal Ranges

Based on the configuration (`latent_dim=1024`, `codebook_size=256`, `num_codebooks=8`):

| Metric | Healthy Range | Concerning | Critical |
|--------|---------------|------------|----------|
| Layer Utilization | 60-90% | 40-60% or 90-95% | < 40% or > 95% |
| Perplexity | 100-256 | 50-100 | < 50 |
| SNR | > 15 dB | 10-15 dB | < 10 dB |
| VQ/Recon Ratio | 0.1-2.0 | 2-5 | > 10 |
| Gradient Norm | 0.1-5.0 | 5-10 | > 10 or < 1e-6 |
| Dead Code % | < 20% | 20-40% | > 40% |

---

## 8. Quick Reference: Debugging Guide

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low utilization (< 50%) | Dead code threshold too aggressive, or high EMA decay | Lower `decay` to 0.9, or increase batch size |
| High VQ loss, low recon loss | Commitment cost too high | Lower `commitment_cost` to 0.1 |
| Recon loss not decreasing | Decoder capacity insufficient | Add more decoder layers or increase `latent_dim` |
| Codebook collapse (all same code) | Learning rate too high | Reduce LR, or increase warmup steps |
| Exploding gradients | Gradient clipping not working | Check clip value, or use gradient accumulation |
| Perplexity << codebook_size | Mode collapse in VQ | Reduce EMA decay, or increase epsilon |

---

## 9. Code Integration Points

### Modify `train_rqvae.py`:

```python
# Initialize monitors
monitor = TrainingMonitor()
codebook_monitor = CodebookMonitor(
    num_embeddings=cfg.rqvae.codebook_size,
    num_layers=cfg.rqvae.num_codebooks
)
ema_monitor = EMAMonitor(rqvae.rvq.layers)
recon_monitor = ReconstructionMonitor()
alert_manager = AlertManager(config)

# In training loop:
for step, batch in enumerate(dataloader):
    # ... forward pass ...

    # Update monitors
    codebook_monitor.update(codes)
    ema_logs = ema_monitor.update()
    recon_logs = recon_monitor.compute_metrics(features, recon, mask)

    # Aggregate logs
    logs = {
        **train_logs,
        **ema_logs,
        **recon_logs,
        **codebook_monitor.get_summary()
    }

    # Check alerts
    alerts = alert_manager.check(logs, step)

    # Log to wandb
    wandb.log(logs, step=step)
```

---

## Appendix: Visualization Examples

### Codebook Usage Heatmap
```python
# Visualize which codes are being used across layers
usage_matrix = compute_usage_matrix(all_codes)  # [num_layers, codebook_size]
plt.imshow(usage_matrix, aspect='auto', cmap='hot')
plt.colorbar(label='Usage Count')
plt.xlabel('Code Index')
plt.ylabel('Layer Index')
plt.title('Codebook Usage Heatmap')
```

### Training Curves
- Total loss with VQ and Recon components
- Per-layer utilization over time
- SNR improvement over epochs
- Gradient norm tracking

---

*Proposal Version: 1.0*
*Last Updated: 2026-02-19*
