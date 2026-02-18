import torch
import torch.nn as nn
from speechgr.models.rqvae import DocumentRQVAE

def test_document_rqvae():
    print("Testing Robust DocumentRQVAE...")
    device = torch.device("cpu")
    
    B, T, D = 4, 50, 1024
    latent_dim = 256
    codebook_size = 8
    num_codebooks = 2
    
    model = DocumentRQVAE(
        input_dim=D,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        num_encoder_layers=2,
        num_decoder_layers=2,
        decay=0.9
    ).to(device)
    
    # 1. Test Forward without mask
    print("- Testing forward without mask...")
    x = torch.randn(B, T, D).to(device)
    recon, loss, codes = model(x)
    
    print(f"  Recon shape: {recon.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Codes shape: {codes.shape}")
    
    assert recon.shape == (B, T, D)
    assert codes.shape == (B, num_codebooks)
    print("  OK")
    
    # 2. Test Forward with mask
    print("- Testing forward with mask...")
    mask = torch.ones(B, T).to(device)
    mask[0, 40:] = 0 # First sample is shorter
    
    recon, loss, codes = model(x, mask=mask)
    assert recon.shape == (B, T, D)
    print("  OK")
    
    # 3. Test EMA/Revival Logic (Simplified)
    print("- Testing EMA cluster size updates...")
    # Access first layer cluster size
    model.train()
    cluster_size_init = model.rvq.layers[0].ema_cluster_size.clone()
    print(f"  Initial avg cluster size: {cluster_size_init.mean().item():.4f}")
    
    # Force revival off for test by setting num_updates high
    model.rvq.layers[0].num_updates = torch.tensor([101])
    
    # Run a few steps
    for _ in range(50):
        _, _, _ = model(x)
    
    cluster_size_end = model.rvq.layers[0].ema_cluster_size
    print(f"  End raw cluster sizes: {cluster_size_end.tolist()}")
    
    # Check if ANY value changed from 1.0
    assert not torch.allclose(cluster_size_end, cluster_size_init, atol=1e-5)
    print("  OK")

    print("\nRobust DocumentRQVAE Test Passed!")

if __name__ == "__main__":
    test_document_rqvae()
