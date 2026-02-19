"""
Diagnostic script for RQ-VAE training issues.
Run this to check codebook health and identify problems.
"""

import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from speechgr.models.rqvae import DocumentRQVAE
from speechgr.models.ssl_wrapper import SSLModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_codebook(checkpoint_path, model_config, output_dir="diagnostics"):
    """Analyze codebook health from a checkpoint."""
    Path(output_dir).mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model = DocumentRQVAE(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    results = {
        "codebook_stats": [],
        "embedding_stats": [],
        "problems": []
    }

    print("\n" + "="*70)
    print("RQ-VAE CODEBOOK DIAGNOSTICS")
    print("="*70)

    for i, vq_layer in enumerate(model.rvq.layers):
        print(f"\n📊 Layer {i}:")
        print("-" * 40)

        # Get codebook info
        embeddings = vq_layer.embedding.data.cpu().numpy()
        cluster_sizes = vq_layer.ema_cluster_size.data.cpu().numpy()
        num_updates = vq_layer.num_updates.item()

        # 1. Check embedding statistics
        embed_norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Embedding norms: mean={embed_norms.mean():.4f}, std={embed_norms.std():.4f}")
        print(f"  Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

        # 2. Check cluster sizes
        active_clusters = (cluster_sizes > 0.01).sum()
        dead_clusters = (cluster_sizes < 0.01).sum()

        print(f"  Cluster sizes: mean={cluster_sizes.mean():.4f}, min={cluster_sizes.min():.4f}")
        print(f"  Active clusters: {active_clusters}/{len(cluster_sizes)} ({active_clusters/len(cluster_sizes)*100:.1f}%)")
        print(f"  Dead clusters: {dead_clusters} ({dead_clusters/len(cluster_sizes)*100:.1f}%)")
        print(f"  Num EMA updates: {num_updates}")

        # 3. Check for problems
        if dead_clusters > len(cluster_sizes) * 0.5:
            problem = f"Layer {i}: Too many dead clusters ({dead_clusters})"
            results["problems"].append(problem)
            print(f"  ⚠️  WARNING: {problem}")

        if embed_norms.mean() < 0.01:
            problem = f"Layer {i}: Embeddings collapsed to near zero"
            results["problems"].append(problem)
            print(f"  ⚠️  WARNING: {problem}")

        if cluster_sizes.std() < 0.001 and num_updates > 100:
            problem = f"Layer {i}: Cluster sizes not updating (std={cluster_sizes.std():.6f})"
            results["problems"].append(problem)
            print(f"  ⚠️  WARNING: {problem}")

        results["codebook_stats"].append({
            "layer": i,
            "active_clusters": active_clusters,
            "dead_clusters": dead_clusters,
            "embedding_norm_mean": embed_norms.mean(),
            "embedding_norm_std": embed_norms.std(),
            "cluster_size_mean": cluster_sizes.mean(),
            "cluster_size_std": cluster_sizes.std(),
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results["problems"]:
        print(f"\n🚨 Found {len(results['problems'])} problems:")
        for p in results["problems"]:
            print(f"  - {p}")
    else:
        print("\n✅ No obvious problems detected")

    # Generate plots
    plot_codebook_stats(results["codebook_stats"], output_dir)

    return results


def plot_codebook_stats(stats, output_dir):
    """Plot codebook statistics."""
    layers = [s["layer"] for s in stats]
    active = [s["active_clusters"] for s in stats]
    dead = [s["dead_clusters"] for s in stats]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Active clusters
    axes[0, 0].bar(layers, active)
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Active Clusters")
    axes[0, 0].set_title("Active Clusters per Layer")
    axes[0, 0].axhline(y=len(stats) * 256 * 0.5, color='r', linestyle='--', label="50% threshold")
    axes[0, 0].legend()

    # Dead clusters
    axes[0, 1].bar(layers, dead, color='red')
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Dead Clusters")
    axes[0, 1].set_title("Dead Clusters per Layer")

    # Embedding norms
    norms = [s["embedding_norm_mean"] for s in stats]
    axes[1, 0].bar(layers, norms)
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Mean Embedding Norm")
    axes[1, 0].set_title("Embedding Norms per Layer")
    axes[1, 0].axhline(y=0.01, color='r', linestyle='--', label="Collapse threshold")
    axes[1, 0].legend()

    # Cluster size distribution
    cluster_means = [s["cluster_size_mean"] for s in stats]
    cluster_stds = [s["cluster_size_std"] for s in stats]
    axes[1, 1].errorbar(layers, cluster_means, yerr=cluster_stds, fmt='o-')
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Cluster Size (mean ± std)")
    axes[1, 1].set_title("Cluster Size Distribution")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/codebook_diagnostics.png", dpi=150)
    print(f"\n📊 Plots saved to {output_dir}/codebook_diagnostics.png")


def test_forward_pass(checkpoint_path, model_config, ssl_model_name="microsoft/wavlm-large", ssl_layer=24):
    """Test forward pass and check gradients."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Testing forward pass...")

    # Load models
    ssl_model = SSLModelWrapper(model_name=ssl_model_name, layer=ssl_layer, freeze=True).to(device)
    model = DocumentRQVAE(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.train()  # Set to train for gradient checking

    # Create dummy input
    dummy_audio = torch.randn(2, 16000).to(device)  # 2 samples, 1 second at 16kHz

    with torch.no_grad():
        features = ssl_model(dummy_audio)

    print(f"\n📊 Forward Pass Test:")
    print(f"  SSL features shape: {features.shape}")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Feature mean: {features.mean():.3f}, std: {features.std():.3f}")

    # Test RQVAE forward
    mask = torch.ones(features.size(0), features.size(1)).to(device)
    recon, loss, codes = model(features, mask=mask)

    print(f"\n  RQVAE output:")
    print(f"    Reconstruction shape: {recon.shape}")
    print(f"    Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
    print(f"    Total loss: {loss.item():.4f}")
    print(f"    Codes shape: {codes.shape}")
    print(f"    Code range: [{codes.min()}, {codes.max()}]")

    # Check if gradients flow
    loss.backward()

    has_grad = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_grad[name] = grad_norm

    print(f"\n  Gradient check:")
    print(f"    Parameters with gradients: {len(has_grad)}/{sum(1 for _ in model.parameters())}")

    if has_grad:
        avg_grad = sum(has_grad.values()) / len(has_grad)
        max_grad = max(has_grad.values(), key=lambda x: abs(x))
        print(f"    Average grad norm: {avg_grad:.6f}")
        print(f"    Max grad norm: {max_grad:.6f}")

        if avg_grad < 1e-8:
            print("    ⚠️  WARNING: Very small gradients!")
        elif avg_grad > 10:
            print("    ⚠️  WARNING: Very large gradients!")
        else:
            print("    ✅ Gradients look healthy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose RQ-VAE training issues")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="diagnostics", help="Output directory for plots")
    parser.add_argument("--test-forward", action="store_true", help="Test forward pass")

    args = parser.parse_args()

    # Model config matching your training
    model_config = {
        "input_dim": 1024,  # WavLM-large layer 24
        "latent_dim": 1024,
        "codebook_size": 256,
        "num_codebooks": 8,
        "commitment_cost": 0.25,
        "decay": 0.99,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
    }

    # Analyze codebook
    results = analyze_codebook(args.checkpoint, model_config, args.output_dir)

    # Test forward pass if requested
    if args.test_forward:
        test_forward_pass(args.checkpoint, model_config)

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if not results["problems"]:
        print("\n✅ Model looks healthy. The issue may be:")
        print("  - Learning rate too high/low")
        print("  - Insufficient training time")
        print("  - Data quality issues")
    else:
        print("\n🔧 Based on the problems found:")

        has_dead_codes = any("dead" in p.lower() for p in results["problems"])
        has_collapse = any("collapsed" in p.lower() for p in results["problems"])

        if has_dead_codes:
            print("  1. Too many dead codes:")
            print("     → Lower EMA decay: decay=0.9 (instead of 0.99)")
            print("     → Increase batch size")
            print("     → Lower dead code threshold: cluster_size < 0.001")

        if has_collapse:
            print("  2. Embeddings collapsed:")
            print("     → Reduce learning rate: lr=1e-5")
            print("     → Increase warmup steps")
            print("     → Add gradient clipping: max_norm=0.5")

    print("\n" + "="*70)
