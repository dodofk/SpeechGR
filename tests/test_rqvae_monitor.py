"""Tests for RQ-VAE monitoring system."""

import pytest
import torch
import numpy as np
from speechgr.utils.rqvae_monitor import (
    CodebookMonitor,
    EMAMonitor,
    ReconstructionMonitor,
    TrainingMonitor,
    AlertManager,
    RQVAEMonitor,
)
from speechgr.models.rqvae import VectorQuantizerEMA, ResidualVectorQuantizerEMA


class TestCodebookMonitor:
    def test_initialization(self):
        monitor = CodebookMonitor(num_embeddings=256, num_layers=8)
        assert monitor.num_embeddings == 256
        assert monitor.num_layers == 8
        assert len(monitor.history) == 0

    def test_update_single_batch(self):
        monitor = CodebookMonitor(num_embeddings=256, num_layers=8)

        # Simulate batch of codes [B, num_layers]
        batch_size = 32
        codes = torch.randint(0, 256, (batch_size, 8))

        logs = monitor.update(codes)

        # Check that all expected metrics are present
        assert "codebook/avg_utilization" in logs
        assert "codebook/avg_perplexity" in logs
        assert "codebook/min_utilization" in logs

        # Check per-layer metrics
        for i in range(8):
            assert f"codebook/layer_{i}_utilization" in logs
            assert f"codebook/layer_{i}_perplexity" in logs

    def test_utilization_range(self):
        monitor = CodebookMonitor(num_embeddings=256, num_layers=8)

        # Test with diverse codes
        codes = torch.randint(0, 256, (100, 8))
        logs = monitor.update(codes)

        # Utilization should be between 0 and 1
        assert 0 <= logs["codebook/avg_utilization"] <= 1

    def test_perplexity_range(self):
        monitor = CodebookMonitor(num_embeddings=256, num_layers=8)

        # Test with diverse codes
        codes = torch.randint(0, 256, (100, 8))
        logs = monitor.update(codes)

        # Perplexity should be between 1 and num_embeddings
        assert 1 <= logs["codebook/avg_perplexity"] <= 256

    def test_get_summary(self):
        monitor = CodebookMonitor(num_embeddings=256, num_layers=8)

        # Add some data
        for _ in range(10):
            codes = torch.randint(0, 256, (32, 8))
            monitor.update(codes)

        summary = monitor.get_summary()

        assert "per_layer" in summary
        assert "aggregate" in summary
        assert "trends" in summary


class TestEMAMonitor:
    def test_initialization(self):
        # Create mock VQ layers
        vq_layers = [
            VectorQuantizerEMA(num_embeddings=256, embedding_dim=128)
            for _ in range(4)
        ]
        monitor = EMAMonitor(vq_layers)

        assert len(monitor.vq_layers) == 4
        assert monitor.num_layers == 4

    def test_update_returns_logs(self):
        vq_layers = [
            VectorQuantizerEMA(num_embeddings=256, embedding_dim=128)
            for _ in range(4)
        ]
        monitor = EMAMonitor(vq_layers)

        logs = monitor.update()

        # Check that metrics are returned
        assert "ema/avg_cluster_size_mean" in logs
        for i in range(4):
            assert f"ema/layer_{i}_cluster_size_mean" in logs
            assert f"ema/layer_{i}_tiny_clusters" in logs


class TestReconstructionMonitor:
    def test_compute_metrics(self):
        monitor = ReconstructionMonitor()

        B, T, D = 8, 50, 128
        original = torch.randn(B, T, D)
        reconstructed = original + torch.randn(B, T, D) * 0.1  # Slightly noisy

        metrics = monitor.compute_metrics(original, reconstructed)

        # Check expected metrics
        assert "recon/mse" in metrics
        assert "recon/snr_db" in metrics
        assert "recon/cosine_similarity_mean" in metrics
        assert "recon/rmse" in metrics

    def test_snr_calculation(self):
        monitor = ReconstructionMonitor()

        B, T, D = 8, 50, 128
        original = torch.randn(B, T, D)

        # Perfect reconstruction should have infinite SNR
        perfect_recon = original.clone()
        metrics_perfect = monitor.compute_metrics(original, perfect_recon)
        assert metrics_perfect["recon/snr_db"] > 50  # Very high SNR

        # Noisy reconstruction should have lower SNR
        noisy_recon = original + torch.randn(B, T, D) * 0.5
        metrics_noisy = monitor.compute_metrics(original, noisy_recon)
        assert metrics_noisy["recon/snr_db"] < metrics_perfect["recon/snr_db"]

    def test_metrics_with_mask(self):
        monitor = ReconstructionMonitor()

        B, T, D = 8, 50, 128
        original = torch.randn(B, T, D)
        reconstructed = original + torch.randn(B, T, D) * 0.1
        mask = torch.ones(B, T)
        mask[:, 25:] = 0  # Mask out second half

        metrics = monitor.compute_metrics(original, reconstructed, mask)

        assert "recon/mse" in metrics
        assert metrics["recon/mse"] >= 0


class TestTrainingMonitor:
    def test_log_step(self):
        # Create simple model
        model = torch.nn.Linear(10, 10)
        monitor = TrainingMonitor(model)

        # Mock loss components
        loss_components = {
            "total": torch.tensor(1.5),
            "recon": torch.tensor(1.0),
            "vq": torch.tensor(0.5)
        }

        optimizer = torch.optim.Adam(model.parameters())
        logs = monitor.log_step(loss_components, optimizer, step=0)

        assert "train/loss_total" in logs
        assert "train/loss_recon" in logs
        assert "train/loss_vq" in logs
        assert "train/lr" in logs

    def test_loss_history(self):
        model = torch.nn.Linear(10, 10)
        monitor = TrainingMonitor(model)

        optimizer = torch.optim.Adam(model.parameters())

        # Log multiple steps
        for i in range(10):
            loss_components = {
                "total": torch.tensor(float(i)),
                "recon": torch.tensor(float(i) * 0.7),
                "vq": torch.tensor(float(i) * 0.3)
            }
            monitor.log_step(loss_components, optimizer, step=i)

        # Check history was recorded
        assert len(monitor.loss_history["total"]) == 10

    def test_loss_trend(self):
        model = torch.nn.Linear(10, 10)
        monitor = TrainingMonitor(model)
        optimizer = torch.optim.Adam(model.parameters())

        # Log decreasing loss
        for i in range(20):
            loss_components = {
                "total": torch.tensor(10.0 - i * 0.5),
                "recon": torch.tensor(7.0 - i * 0.35),
                "vq": torch.tensor(3.0 - i * 0.15)
            }
            monitor.log_step(loss_components, optimizer, step=i)

        trend = monitor.get_loss_trend("total")
        assert trend < 0  # Loss should be decreasing


class TestAlertManager:
    def test_check_no_alerts(self):
        manager = AlertManager()
        logs = {"train/loss_total": 1.0, "codebook/avg_utilization": 0.7}

        alerts = manager.check(logs, step=0)
        assert len(alerts) == 0

    def test_nan_loss_alert(self):
        manager = AlertManager()
        logs = {"train/loss_total": float('nan'), "codebook/avg_utilization": 0.7}

        alerts = manager.check(logs, step=0)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].rule == "nan_loss"

    def test_low_utilization_alert(self):
        manager = AlertManager()
        logs = {"train/loss_total": 1.0, "codebook/avg_utilization": 0.3}

        # Rule has a warmup gate at step 100.
        assert manager.check(logs, step=99) == []
        alerts = manager.check(logs, step=100)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].rule == "low_utilization"

    def test_cooldown(self):
        manager = AlertManager(cooldown_steps=10)

        # First alert
        logs = {"train/loss_total": 1.0, "codebook/avg_utilization": 0.3}
        alerts1 = manager.check(logs, step=100)
        assert len(alerts1) == 1

        # Same alert within cooldown - should be suppressed
        alerts2 = manager.check(logs, step=105)
        assert len(alerts2) == 0

        # Same alert after cooldown - should trigger again
        alerts3 = manager.check(logs, step=115)
        assert len(alerts3) == 1

    def test_should_stop_training(self):
        manager = AlertManager()

        # No critical alerts initially
        assert not manager.should_stop_training()

        # Add a critical alert
        logs = {"train/loss_total": float('nan')}
        manager.check(logs, step=0)

        assert manager.should_stop_training()


class TestRQVAEMonitor:
    def test_ema_monitor_initializes_from_rvq(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rvq = ResidualVectorQuantizerEMA(
                    num_embeddings=8,
                    embedding_dim=4,
                    num_layers=2,
                )

        model = DummyModel()
        monitor = RQVAEMonitor(
            model=model,
            num_embeddings=8,
            num_layers=2,
            enable_ema=True,
            enable_codebook=False,
            enable_recon=False,
            enable_training=False,
            enable_alerts=False,
        )

        assert monitor.ema_monitor is not None

    def test_initialization(self):
        # Create simple model
        model = torch.nn.Linear(128, 128)

        monitor = RQVAEMonitor(
            model=model,
            num_embeddings=256,
            num_layers=8
        )

        assert monitor.num_embeddings == 256
        assert monitor.num_layers == 8

    def test_update_all_enabled(self):
        model = torch.nn.Linear(128, 128)
        monitor = RQVAEMonitor(
            model=model,
            num_embeddings=256,
            num_layers=8,
            enable_codebook=True,
            enable_ema=False,  # Skip EMA (no vq layers)
            enable_recon=True,
            enable_training=True,
            enable_alerts=True
        )

        B, T, D = 8, 50, 128
        codes = torch.randint(0, 256, (B, 8))
        original = torch.randn(B, T, D)
        reconstructed = torch.randn(B, T, D)
        mask = torch.ones(B, T)

        loss_components = {
            "total": torch.tensor(1.5),
            "recon": torch.tensor(1.0),
            "vq": torch.tensor(0.5)
        }

        optimizer = torch.optim.Adam(model.parameters())

        logs = monitor.update(
            codes=codes,
            original=original,
            reconstructed=reconstructed,
            mask=mask,
            loss_components=loss_components,
            optimizer=optimizer,
            step=0
        )

        # Check that various metrics are present
        assert "train/loss_total" in logs
        assert "codebook/avg_utilization" in logs
        assert "recon/mse" in logs

    def test_get_summary(self):
        model = torch.nn.Linear(128, 128)
        monitor = RQVAEMonitor(
            model=model,
            num_embeddings=256,
            num_layers=8,
            enable_ema=False
        )

        # Add some data
        B, T, D = 8, 50, 128
        for i in range(10):
            codes = torch.randint(0, 256, (B, 8))
            original = torch.randn(B, T, D)
            reconstructed = original + torch.randn(B, T, D) * 0.1
            mask = torch.ones(B, T)

            loss_components = {
                "total": torch.tensor(1.5 - i * 0.1),
                "recon": torch.tensor(1.0 - i * 0.05),
                "vq": torch.tensor(0.5)
            }

            optimizer = torch.optim.Adam(model.parameters())

            monitor.update(
                codes=codes,
                original=original,
                reconstructed=reconstructed,
                mask=mask,
                loss_components=loss_components,
                optimizer=optimizer,
                step=i
            )

        summary = monitor.get_summary()

        assert "step" in summary
        assert summary["step"] == 9
        assert "codebook" in summary
        assert "loss_trend" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
