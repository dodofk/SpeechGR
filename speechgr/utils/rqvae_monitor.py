"""
RQ-VAE Training Monitoring System

Comprehensive monitoring for RQ-VAE training including:
- Codebook health (utilization, perplexity, dead codes)
- EMA stability (embedding movement, cluster sizes)
- Reconstruction quality (SNR, cosine similarity, temporal consistency)
- Training dynamics (gradients, loss decomposition)
- Automated alerting
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert data structure."""
    step: int
    rule: str
    severity: str  # 'critical', 'warning', 'info'
    message: str
    metrics: Dict[str, Any]


class CodebookMonitor:
    """
    Monitor codebook utilization and health across RQ-VAE layers.

    Tracks:
    - Utilization rate (% of codes used)
    - Perplexity (distribution entropy)
    - Dead code count
    - Usage patterns over time
    """

    def __init__(self, num_embeddings: int, num_layers: int, history_size: int = 1000):
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.history_size = history_size
        self.history = defaultdict(lambda: defaultdict(list))
        self._window_size = 100  # For trend analysis

    def _get_layer_indices(self, all_indices: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Return indices for a specific codebook layer across all batch dimensions."""

        if all_indices.ndim < 2:
            raise ValueError(
                f"Expected code indices with at least 2 dims, got {tuple(all_indices.shape)}"
            )

        # Common layouts:
        # - [B, L]
        # - [B, W, L] (sliding-window codes)
        if all_indices.shape[-1] == self.num_layers:
            return all_indices[..., layer_idx].reshape(-1)

        # Backward compatibility for [B, L] where L might be second dim.
        if all_indices.ndim == 2 and all_indices.shape[1] == self.num_layers:
            return all_indices[:, layer_idx].reshape(-1)

        raise ValueError(
            "Unexpected code tensor layout for CodebookMonitor: "
            f"shape={tuple(all_indices.shape)}, num_layers={self.num_layers}"
        )

    def update(self, all_indices: torch.Tensor) -> Dict[str, float]:
        """
        Update monitor with new batch of codes.

        Args:
            all_indices: code indices with shape [B, num_layers] or
                [B, num_windows, num_layers]

        Returns:
            Dictionary of metrics for this step
        """
        metrics = {}

        for layer_idx in range(self.num_layers):
            layer_indices = self._get_layer_indices(all_indices, layer_idx).to(torch.long)

            # Usage counts per code
            usage_counts = torch.bincount(
                layer_indices,
                minlength=self.num_embeddings
            ).float()

            # 1. Utilization rate
            active_codes = (usage_counts > 0).sum().item()
            utilization = active_codes / self.num_embeddings

            # 2. Perplexity (measure of code distribution entropy)
            # H = -sum(p * log(p)), Perplexity = exp(H)
            probs = usage_counts / usage_counts.sum()
            probs = probs[probs > 0]  # Remove zeros for log
            if len(probs) > 0:
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                perplexity = torch.exp(entropy).item()
            else:
                perplexity = 1.0  # All mass on single code

            # 3. Dead/inactive code signals
            # "Dead" here means no usage in current batch/window set.
            dead_codes = (usage_counts == 0).sum().item()
            dead_code_ratio = dead_codes / self.num_embeddings

            # Keep a low-count signal for debugging imbalance without over-penalizing.
            low_count_codes = (usage_counts < 5).sum().item()
            low_count_ratio = low_count_codes / self.num_embeddings

            # 4. Usage concentration (Gini coefficient - measure of inequality)
            sorted_counts = torch.sort(usage_counts, descending=True)[0]
            cumsum = torch.cumsum(sorted_counts, dim=0)
            n = len(usage_counts)
            gini = (n + 1 - 2 * cumsum.sum() / cumsum[-1]).item() / n if cumsum[-1] > 0 else 0.0

            # Store in history
            layer_key = f"layer_{layer_idx}"
            self.history[layer_key]["utilization"].append(utilization)
            self.history[layer_key]["perplexity"].append(perplexity)
            self.history[layer_key]["dead_code_ratio"].append(dead_code_ratio)
            self.history[layer_key]["low_count_ratio"].append(low_count_ratio)
            self.history[layer_key]["gini"].append(gini)
            self.history[layer_key]["active_codes"].append(active_codes)

            # Trim history if too long
            for key in self.history[layer_key]:
                if len(self.history[layer_key][key]) > self.history_size:
                    self.history[layer_key][key] = self.history[layer_key][key][-self.history_size:]

            # Add to metrics
            metrics[f"codebook/{layer_key}_utilization"] = utilization
            metrics[f"codebook/{layer_key}_perplexity"] = perplexity
            metrics[f"codebook/{layer_key}_dead_code_ratio"] = dead_code_ratio
            metrics[f"codebook/{layer_key}_low_count_ratio"] = low_count_ratio
            metrics[f"codebook/{layer_key}_gini"] = gini
            metrics[f"codebook/{layer_key}_active_codes"] = active_codes

        # Compute aggregate metrics
        all_util = [metrics[f"codebook/layer_{i}_utilization"] for i in range(self.num_layers)]
        all_perp = [metrics[f"codebook/layer_{i}_perplexity"] for i in range(self.num_layers)]

        metrics["codebook/avg_utilization"] = np.mean(all_util)
        metrics["codebook/min_utilization"] = np.min(all_util)
        metrics["codebook/max_utilization"] = np.max(all_util)
        metrics["codebook/utilization_std"] = np.std(all_util)
        metrics["codebook/avg_perplexity"] = np.mean(all_perp)
        metrics["codebook/min_perplexity"] = np.min(all_perp)

        return metrics

    def get_trend(self, layer_idx: int, metric: str, window: Optional[int] = None) -> float:
        """Get trend (slope) of a metric over recent history."""
        window = window or self._window_size
        layer_key = f"layer_{layer_idx}"
        values = self.history[layer_key][metric][-window:]

        if len(values) < 10:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of codebook health."""
        summary = {
            "per_layer": {},
            "aggregate": {},
            "trends": {}
        }

        for layer_idx in range(self.num_layers):
            layer_key = f"layer_{layer_idx}"
            hist = self.history[layer_key]

            if not hist["utilization"]:
                continue

            # Current values
            summary["per_layer"][layer_key] = {
                "utilization": hist["utilization"][-1],
                "perplexity": hist["perplexity"][-1],
                "dead_code_ratio": hist["dead_code_ratio"][-1],
                "active_codes": hist["active_codes"][-1]
            }

            # Trends
            summary["trends"][layer_key] = {
                "utilization_trend": self.get_trend(layer_idx, "utilization"),
                "perplexity_trend": self.get_trend(layer_idx, "perplexity")
            }

        # Aggregate stats
        if self.history["layer_0"]["utilization"]:
            recent_utils = [
                self.history[f"layer_{i}"]["utilization"][-1]
                for i in range(self.num_layers)
            ]
            summary["aggregate"]["avg_utilization"] = np.mean(recent_utils)
            summary["aggregate"]["worst_layer"] = int(np.argmin(recent_utils))
            summary["aggregate"]["best_layer"] = int(np.argmax(recent_utils))

        return summary


class EMAMonitor:
    """
    Monitor EMA (Exponential Moving Average) stability in VQ layers.

    Tracks:
    - Embedding movement between steps
    - Cluster size distribution
    - Tiny cluster detection (potential dead codes)
    """

    def __init__(self, vq_layers: List[torch.nn.Module]):
        self.vq_layers = vq_layers
        self.num_layers = len(vq_layers)
        self.prev_embeddings: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.prev_cluster_sizes: List[Optional[torch.Tensor]] = [None] * self.num_layers

    def update(self) -> Dict[str, float]:
        """Update EMA metrics and return logs."""
        logs = {}

        for i, vq in enumerate(self.vq_layers):
            layer_prefix = f"ema/layer_{i}"

            # 1. Embedding movement (L2 distance from previous)
            if self.prev_embeddings[i] is not None:
                movement = torch.norm(
                    vq.embedding - self.prev_embeddings[i],
                    dim=1
                )
                logs[f"{layer_prefix}_embedding_movement_mean"] = movement.mean().item()
                logs[f"{layer_prefix}_embedding_movement_max"] = movement.max().item()
                logs[f"{layer_prefix}_embedding_movement_std"] = movement.std().item()

            # Store current for next comparison
            self.prev_embeddings[i] = vq.embedding.clone().detach()

            # 2. Cluster size statistics
            cluster_sizes = vq.ema_cluster_size.clone().detach()
            logs[f"{layer_prefix}_cluster_size_mean"] = cluster_sizes.mean().item()
            logs[f"{layer_prefix}_cluster_size_min"] = cluster_sizes.min().item()
            logs[f"{layer_prefix}_cluster_size_max"] = cluster_sizes.max().item()
            logs[f"{layer_prefix}_cluster_size_std"] = cluster_sizes.std().item()

            # 3. Check for very small clusters (potential dead codes)
            tiny_threshold = 0.01
            tiny_clusters = (cluster_sizes < tiny_threshold).sum().item()
            logs[f"{layer_prefix}_tiny_clusters"] = tiny_clusters
            logs[f"{layer_prefix}_tiny_cluster_ratio"] = tiny_clusters / len(cluster_sizes)

            # 4. Cluster size movement
            if self.prev_cluster_sizes[i] is not None:
                size_change = torch.abs(cluster_sizes - self.prev_cluster_sizes[i])
                logs[f"{layer_prefix}_cluster_change_mean"] = size_change.mean().item()
                logs[f"{layer_prefix}_cluster_change_max"] = size_change.max().item()

            self.prev_cluster_sizes[i] = cluster_sizes

            # 5. Embedding statistics
            logs[f"{layer_prefix}_embedding_norm_mean"] = torch.norm(vq.embedding, dim=1).mean().item()
            logs[f"{layer_prefix}_embedding_variance"] = vq.embedding.var(dim=0).mean().item()

        # Aggregate metrics
        all_util = [logs.get(f"ema/layer_{i}_cluster_size_mean", 0) for i in range(self.num_layers)]
        if all_util:
            logs["ema/avg_cluster_size_mean"] = np.mean(all_util)
            logs["ema/min_cluster_size_mean"] = np.min(all_util)

        return logs


class ReconstructionMonitor:
    """
    Monitor reconstruction quality beyond simple MSE loss.

    Tracks:
    - SNR (Signal-to-Noise Ratio)
    - Cosine similarity between input and output
    - Temporal consistency
    """

    def __init__(self, normalize_like_training: bool = False):
        # Match target normalization used in RQ-VAE training loss so SNR/MSE are comparable.
        self.normalize_like_training = normalize_like_training

    @staticmethod
    def _normalize_target_like_training(
        original: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
            target_mean = (original * mask_expanded).sum(dim=1, keepdim=True) / denom
            target_std = torch.sqrt(
                ((original - target_mean) ** 2 * mask_expanded).sum(dim=1, keepdim=True)
                / (denom * original.size(-1) + 1e-6)
            )
        else:
            target_mean = original.mean(dim=1, keepdim=True)
            target_std = torch.sqrt(
                ((original - target_mean) ** 2).mean(dim=1, keepdim=True) + 1e-6
            )

        return (original - target_mean) / (target_std + 1e-6)

    def compute_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.

        Args:
            original: [B, T, D] input features
            reconstructed: [B, T, D] output features
            mask: [B, T] valid frame mask (1=valid, 0=padding)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        target = (
            self._normalize_target_like_training(original, mask)
            if self.normalize_like_training
            else original
        )

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            # Normalize by actual valid elements
            normalizer = mask.sum() * original.size(-1)
            valid = True
        else:
            mask_expanded = torch.ones_like(original[..., :1])
            normalizer = original.numel()
            valid = False

        # 1. MSE (more precise than model's version)
        diff = (reconstructed - target) * mask_expanded
        mse = (diff ** 2).sum() / (normalizer + 1e-10)
        metrics["recon/mse"] = mse.item()
        metrics["recon/rmse"] = np.sqrt(mse.item())

        # 2. SNR calculation (in dB)
        signal_power = ((target * mask_expanded) ** 2).sum() / (normalizer + 1e-10)
        noise_power = mse
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * torch.log10(snr_linear + 1e-10)
        metrics["recon/snr_db"] = snr_db.item()
        metrics["recon/snr_linear"] = snr_linear.item()

        # 3. Feature correlation (per-frame cosine similarity)
        orig_flat = target.reshape(-1, target.size(-1))
        recon_flat = reconstructed.reshape(-1, target.size(-1))

        cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=-1)

        if valid:
            mask_flat = mask.flatten()
            cos_sim_masked = cos_sim * mask_flat
            cosine_similarity = cos_sim_masked.sum() / (mask_flat.sum() + 1e-10)
        else:
            cosine_similarity = cos_sim.mean()

        metrics["recon/cosine_similarity_mean"] = cosine_similarity.item()
        metrics["recon/cosine_similarity_min"] = cos_sim.min().item()

        # 4. L1 distance (more robust to outliers)
        l1_dist = (torch.abs(diff)).sum() / (normalizer + 1e-10)
        metrics["recon/l1_distance"] = l1_dist.item()

        # 5. Max absolute error
        if valid:
            max_error = (torch.abs(diff)).max().item()
        else:
            max_error = (torch.abs(diff)).max().item()
        metrics["recon/max_abs_error"] = max_error

        # 6. Temporal consistency (for sequential data)
        if original.size(1) > 1:
            # Compute frame-to-frame differences
            orig_diff = target[:, 1:] - target[:, :-1]
            recon_diff = reconstructed[:, 1:] - reconstructed[:, :-1]

            if valid:
                # Apply mask to differences
                mask_diff = mask[:, 1:] * mask[:, :-1]
                mask_diff_expanded = mask_diff.unsqueeze(-1)

                orig_diff_masked = orig_diff * mask_diff_expanded
                recon_diff_masked = recon_diff * mask_diff_expanded

                # Cosine similarity of differences
                orig_diff_flat = orig_diff_masked.reshape(-1, target.size(-1))
                recon_diff_flat = recon_diff_masked.reshape(-1, target.size(-1))
                mask_diff_flat = mask_diff.flatten()

                diff_cos_sim = F.cosine_similarity(
                    orig_diff_flat + 1e-8,
                    recon_diff_flat + 1e-8,
                    dim=-1
                )
                diff_cos_sim_masked = diff_cos_sim * mask_diff_flat
                temp_consistency = diff_cos_sim_masked.sum() / (mask_diff_flat.sum() + 1e-10)
            else:
                orig_diff_flat = orig_diff.reshape(-1, target.size(-1))
                recon_diff_flat = recon_diff.reshape(-1, target.size(-1))
                temp_consistency = F.cosine_similarity(
                    orig_diff_flat + 1e-8,
                    recon_diff_flat + 1e-8,
                    dim=-1
                ).mean()

            metrics["recon/temporal_consistency"] = temp_consistency.item()

        # 7. Relative error (normalized by signal magnitude)
        signal_magnitude = torch.norm(target * mask_expanded, dim=-1).mean()
        relative_error = torch.norm(diff, dim=-1).mean() / (signal_magnitude + 1e-10)
        metrics["recon/relative_error"] = relative_error.item()

        return metrics


class TrainingMonitor:
    """
    Monitor training dynamics including gradients and loss components.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.loss_history = defaultdict(list)
        self.max_history = 1000

    def log_step(
        self,
        loss_components: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step: int
    ) -> Dict[str, float]:
        """Log training step metrics."""
        logs = {}

        # 1. Loss components
        total = loss_components.get("total", torch.tensor(0.0)).item()
        recon = loss_components.get("recon", torch.tensor(0.0)).item()
        vq = loss_components.get("vq", torch.tensor(0.0)).item()

        logs["train/loss_total"] = total
        logs["train/loss_recon"] = recon
        logs["train/loss_vq"] = vq

        # 2. Loss ratios
        if total > 1e-8:
            logs["loss_ratio/recon"] = recon / total
            logs["loss_ratio/vq"] = vq / total
            logs["loss_ratio/vq_to_recon"] = vq / (recon + 1e-10)

        # 3. Store in history for trend analysis
        self.loss_history["total"].append(total)
        self.loss_history["recon"].append(recon)
        self.loss_history["vq"].append(vq)

        for key in self.loss_history:
            if len(self.loss_history[key]) > self.max_history:
                self.loss_history[key] = self.loss_history[key][-self.max_history:]

        # 4. Compute moving averages
        window = min(100, len(self.loss_history["total"]))
        if window > 0:
            logs["train/loss_total_ma"] = np.mean(self.loss_history["total"][-window:])
            logs["train/loss_recon_ma"] = np.mean(self.loss_history["recon"][-window:])

        # 5. Learning rate
        logs["train/lr"] = optimizer.param_groups[0]["lr"]

        return logs

    def log_gradients(self) -> Dict[str, float]:
        """Log gradient statistics."""
        logs = {}

        # Global gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        logs["grad_norm/global"] = total_norm

        # Per-layer gradient norms (for transformer layers)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Only log important layers to avoid clutter
                if any(key in name for key in ["encoder", "decoder", "rvq"]):
                    logs[f"grad_norm/{name}"] = grad_norm

        return logs

    def get_loss_trend(self, component: str = "total", window: int = 100) -> float:
        """Get loss trend (negative is good - decreasing loss)."""
        values = self.loss_history[component][-window:]
        if len(values) < 10:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class AlertManager:
    """
    Automated alert system for detecting training issues.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Tuple[Callable, str, str, int]]] = None,
        cooldown_steps: int = 100,
        num_embeddings: int = 256,
        codebook_collapse_threshold: float = 0.1,
        low_utilization_threshold: float = 0.5,
        low_perplexity_ratio: float = 0.2,
        poor_reconstruction_snr_db: float = 5.0,
    ):
        self.cooldown_steps = cooldown_steps
        self.num_embeddings = num_embeddings
        self.codebook_collapse_threshold = codebook_collapse_threshold
        self.low_utilization_threshold = low_utilization_threshold
        self.low_perplexity_threshold = max(10.0, low_perplexity_ratio * num_embeddings)
        self.poor_reconstruction_snr_db = poor_reconstruction_snr_db

        if rules is None:
            # Default alert rules: (condition_fn, severity, message_template, min_step)
            self.rules = {
                # Critical - stop training candidates.
                "nan_loss": (
                    lambda logs: np.isnan(logs.get("train/loss_total", 0)),
                    "critical",
                    "NaN loss detected at step {step}!",
                    0,
                ),
                "inf_loss": (
                    lambda logs: np.isinf(logs.get("train/loss_total", 0)),
                    "critical",
                    "Inf loss detected at step {step}!",
                    0,
                ),
                "zero_gradients": (
                    lambda logs: logs.get("grad_norm/global", 1.0) < 1e-8,
                    "critical",
                    "Zero gradients detected at step {step}!",
                    100,
                ),
                "codebook_collapse": (
                    lambda logs: logs.get("codebook/avg_utilization", 1.0)
                    < self.codebook_collapse_threshold,
                    "critical",
                    "Codebook collapsed! Avg utilization {value:.2%} at step {step}",
                    500,
                ),
                # Warnings.
                "low_utilization": (
                    lambda logs: logs.get("codebook/avg_utilization", 1.0)
                    < self.low_utilization_threshold,
                    "warning",
                    "Low codebook utilization: {value:.2%} at step {step}",
                    100,
                ),
                "high_vq_ratio": (
                    lambda logs: logs.get("loss_ratio/vq_to_recon", 0) > 10,
                    "warning",
                    "VQ loss dominating: VQ/Recon = {value:.2f} at step {step}",
                    0,
                ),
                "poor_reconstruction": (
                    lambda logs: logs.get("recon/snr_db", 100)
                    < self.poor_reconstruction_snr_db,
                    "warning",
                    "Poor reconstruction quality: SNR = {value:.2f} dB at step {step}",
                    50,
                ),
                "gradient_spike": (
                    lambda logs: logs.get("grad_norm/global", 0) > 100,
                    "warning",
                    "Gradient spike detected: norm = {value:.2f} at step {step}",
                    0,
                ),
                "low_perplexity": (
                    lambda logs: logs.get("codebook/avg_perplexity", self.num_embeddings)
                    < self.low_perplexity_threshold,
                    "warning",
                    "Low codebook perplexity: {value:.1f} at step {step} (possible mode collapse)",
                    100,
                ),
                # Info only.
                "ema_warmup_complete": (
                    lambda logs: logs.get("ema/num_updates", 0) == 100,
                    "info",
                    "EMA warmup complete at step {step}",
                    0,
                ),
            }
        else:
            self.rules = rules

        self.alert_history: List[Alert] = []
        self.last_alert_step: Dict[str, int] = {}
        self.triggered_rules: set = set()

    def check(self, logs: Dict[str, float], step: int) -> List[Alert]:
        """
        Check all alert rules against current logs.

        Args:
            logs: Current metrics dictionary
            step: Current training step

        Returns:
            List of triggered alerts
        """
        alerts = []

        for rule_name, (check_fn, severity, message_template, min_step) in self.rules.items():
            # Skip if before minimum step (warmup period)
            if step < min_step:
                continue

            # Check cooldown (don't spam same alert)
            if step - self.last_alert_step.get(rule_name, -self.cooldown_steps) < self.cooldown_steps:
                continue

            # Evaluate condition
            try:
                if check_fn(logs):
                    # Extract relevant value for message
                    if rule_name in {"low_utilization", "codebook_collapse"}:
                        value = logs.get("codebook/avg_utilization", 0)
                    elif rule_name == "high_vq_ratio":
                        value = logs.get("loss_ratio/vq_to_recon", 0)
                    elif rule_name == "poor_reconstruction":
                        value = logs.get("recon/snr_db", 0)
                    elif rule_name in {"gradient_spike", "zero_gradients"}:
                        value = logs.get("grad_norm/global", 0)
                    elif rule_name == "low_perplexity":
                        value = logs.get("codebook/avg_perplexity", 0)
                    else:
                        value = 0

                    message = message_template.format(step=step, value=value)

                    alert = Alert(
                        step=step,
                        rule=rule_name,
                        severity=severity,
                        message=message,
                        metrics=dict(logs)
                    )
                    alerts.append(alert)

                    self.alert_history.append(alert)
                    self.last_alert_step[rule_name] = step
                    self.triggered_rules.add(rule_name)

                    # Log immediately
                    if severity == "critical":
                        logger.error(f"[CRITICAL] {message}")
                    elif severity == "warning":
                        logger.warning(f"[WARNING] {message}")
                    else:
                        logger.info(f"[INFO] {message}")

            except Exception as e:
                logger.warning(f"Error checking alert rule {rule_name}: {e}")

        return alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts triggered."""
        summary = {
            "total_alerts": len(self.alert_history),
            "critical_count": sum(1 for a in self.alert_history if a.severity == "critical"),
            "warning_count": sum(1 for a in self.alert_history if a.severity == "warning"),
            "info_count": sum(1 for a in self.alert_history if a.severity == "info"),
            "triggered_rules": list(self.triggered_rules),
            "recent_alerts": [
                {
                    "step": a.step,
                    "rule": a.rule,
                    "severity": a.severity,
                    "message": a.message
                }
                for a in self.alert_history[-10:]
            ]
        }
        return summary

    def should_stop_training(self) -> bool:
        """Check if any critical alerts should stop training."""
        return any(a.severity == "critical" for a in self.alert_history[-10:])


class RQVAEMonitor:
    """
    Main monitoring coordinator that combines all monitors.

    Usage:
        monitor = RQVAEMonitor(model, num_embeddings=256, num_layers=8)

        for step, batch in enumerate(dataloader):
            # Training step
            recon, loss, codes = model(features, mask)

            # Update monitors
            logs = monitor.update(
                model=model,
                codes=codes,
                original=features,
                reconstructed=recon,
                mask=mask,
                loss_components={"total": loss, "recon": recon_loss, "vq": vq_loss},
                optimizer=optimizer,
                step=step
            )

            # Check alerts
            alerts = monitor.check_alerts(logs, step)
            if any(a.severity == "critical" for a in alerts):
                break

            # Log to wandb
            wandb.log(logs, step=step)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_embeddings: int,
        num_layers: int,
        enable_codebook: bool = True,
        enable_ema: bool = True,
        enable_recon: bool = True,
        enable_training: bool = True,
        enable_alerts: bool = True,
        alert_cooldown: int = 100,
        low_utilization_threshold: float = 0.5,
        codebook_collapse_threshold: float = 0.1,
        low_perplexity_ratio: float = 0.2,
        poor_reconstruction_snr_db: float = 5.0,
        normalize_recon_to_training_target: bool = True,
    ):
        self.model = model
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers

        # Initialize sub-monitors
        self.codebook_monitor = CodebookMonitor(num_embeddings, num_layers) if enable_codebook else None
        self.ema_monitor = None
        if enable_ema:
            rvq_layers = None
            if hasattr(model, "rvq") and hasattr(model.rvq, "layers"):
                rvq_layers = model.rvq.layers
            elif hasattr(model, "rqvae") and hasattr(model.rqvae, "rvq") and hasattr(model.rqvae.rvq, "layers"):
                rvq_layers = model.rqvae.rvq.layers
            elif hasattr(model, "rqvae") and hasattr(model.rqvae, "layers"):
                rvq_layers = model.rqvae.layers

            if rvq_layers is not None:
                self.ema_monitor = EMAMonitor(rvq_layers)
            else:
                logger.warning(
                    "EMA monitoring requested but no RVQ layers found on model %s",
                    type(model).__name__,
                )
        self.recon_monitor = (
            ReconstructionMonitor(
                normalize_like_training=normalize_recon_to_training_target
            )
            if enable_recon
            else None
        )
        self.training_monitor = TrainingMonitor(model) if enable_training else None
        self.alert_manager = (
            AlertManager(
                cooldown_steps=alert_cooldown,
                num_embeddings=num_embeddings,
                codebook_collapse_threshold=codebook_collapse_threshold,
                low_utilization_threshold=low_utilization_threshold,
                low_perplexity_ratio=low_perplexity_ratio,
                poor_reconstruction_snr_db=poor_reconstruction_snr_db,
            )
            if enable_alerts
            else None
        )

        self.step_count = 0

    def update(
        self,
        codes: torch.Tensor,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: Optional[torch.Tensor],
        loss_components: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step: int
    ) -> Dict[str, float]:
        """Update all monitors and return combined logs."""
        all_logs = {}

        # Codebook monitoring
        if self.codebook_monitor is not None:
            codebook_logs = self.codebook_monitor.update(codes)
            all_logs.update(codebook_logs)

        # EMA monitoring
        if self.ema_monitor is not None:
            ema_logs = self.ema_monitor.update()
            all_logs.update(ema_logs)

        # Reconstruction monitoring
        if self.recon_monitor is not None:
            recon_logs = self.recon_monitor.compute_metrics(original, reconstructed, mask)
            all_logs.update(recon_logs)

        # Training monitoring
        if self.training_monitor is not None:
            training_logs = self.training_monitor.log_step(loss_components, optimizer, step)
            all_logs.update(training_logs)
            grad_logs = self.training_monitor.log_gradients()
            all_logs.update(grad_logs)

        self.step_count = step
        return all_logs

    def check_alerts(self, logs: Dict[str, float], step: int) -> List[Alert]:
        """Check for alerts."""
        if self.alert_manager is not None:
            return self.alert_manager.check(logs, step)
        return []

    def should_stop(self) -> bool:
        """Check if training should stop due to critical alerts."""
        if self.alert_manager is not None:
            return self.alert_manager.should_stop_training()
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all monitoring."""
        summary = {
            "step": self.step_count,
            "codebook": self.codebook_monitor.get_summary() if self.codebook_monitor else None,
            "alerts": self.alert_manager.get_alert_summary() if self.alert_manager else None,
        }

        if self.training_monitor:
            summary["loss_trend"] = {
                "total": self.training_monitor.get_loss_trend("total"),
                "recon": self.training_monitor.get_loss_trend("recon"),
                "vq": self.training_monitor.get_loss_trend("vq")
            }

        return summary

    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print(f"RQ-VAE Monitoring Summary (Step {summary['step']})")
        print("="*60)

        # Codebook summary
        if summary.get("codebook"):
            cb = summary["codebook"]
            print("\n📊 Codebook Health:")
            if "aggregate" in cb:
                agg = cb["aggregate"]
                print(f"  Avg Utilization: {agg.get('avg_utilization', 0):.2%}")
                print(f"  Worst Layer: {agg.get('worst_layer', 'N/A')}")

            for layer_key, layer_data in cb.get("per_layer", {}).items():
                util = layer_data.get("utilization", 0)
                perp = layer_data.get("perplexity", 0)
                dead = layer_data.get("dead_code_ratio", 0)
                status = "✓" if util > 0.5 else "⚠"
                print(f"  {status} {layer_key}: util={util:.1%}, perp={perp:.1f}, dead={dead:.1%}")

        # Loss trends
        if summary.get("loss_trend"):
            print("\n📉 Loss Trends (slope over last 100 steps):")
            for name, trend in summary["loss_trend"].items():
                direction = "↓" if trend < 0 else "↑" if trend > 0 else "→"
                print(f"  {direction} {name}: {trend:.6f}")

        # Alerts
        if summary.get("alerts"):
            alerts = summary["alerts"]
            print(f"\n🚨 Alerts: {alerts['total_alerts']} total")
            print(f"  Critical: {alerts['critical_count']}, Warning: {alerts['warning_count']}, Info: {alerts['info_count']}")

            if alerts["recent_alerts"]:
                print("\n  Recent Alerts:")
                for alert in alerts["recent_alerts"][-3:]:
                    icon = "🔴" if alert["severity"] == "critical" else "🟡" if alert["severity"] == "warning" else "🔵"
                    print(f"    {icon} [{alert['step']}] {alert['rule']}: {alert['message'][:60]}...")

        print("="*60 + "\n")
