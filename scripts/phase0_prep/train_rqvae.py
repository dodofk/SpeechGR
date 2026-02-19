import logging
import os
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import soundfile as sf
import numpy as np
import wandb

from speechgr.models.ssl_wrapper import SSLModelWrapper
from speechgr.models.rqvae import DocumentRQVAE

logger = logging.getLogger(__name__)

class AudioManifestDataset(Dataset):
    """
    Dataset to load audio and return mask information.
    """
    def __init__(self, manifest_path, max_length=160000):
        with open(manifest_path, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        path = self.lines[idx]
        try:
            wav, sr = sf.read(path)
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            
            orig_len = len(wav)
            # Pad or trim
            if orig_len > self.max_length:
                wav = wav[:self.max_length]
                actual_len = self.max_length
            else:
                actual_len = orig_len
                wav = np.pad(wav, (0, self.max_length - orig_len))
                
            return torch.tensor(wav, dtype=torch.float32), torch.tensor(actual_len, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return torch.zeros(self.max_length, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

def create_mask(lengths, max_len, device):
    """Create a binary mask: 1 for valid, 0 for padding."""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask.float()

@hydra.main(version_base=None, config_path="../../configs")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logging.mode
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Models
    logger.info("Initializing SSL Model...")
    ssl_model = SSLModelWrapper(
        model_name=cfg.ssl.model_name,
        layer=cfg.ssl.layer,
        freeze=True
    ).to(device)
    
    logger.info("Initializing Robust DocumentRQVAE...")
    # Map Hydra config to DocumentRQVAE params
    rqvae = DocumentRQVAE(
        input_dim=ssl_model.feature_dim,
        latent_dim=cfg.rqvae.latent_dim,
        codebook_size=cfg.rqvae.codebook_size,
        num_codebooks=cfg.rqvae.num_codebooks,
        commitment_cost=cfg.rqvae.commitment_cost,
        decay=getattr(cfg.rqvae, "decay", 0.99),
        num_encoder_layers=getattr(cfg.rqvae, "num_encoder_layers", 4),
        num_decoder_layers=getattr(cfg.rqvae, "num_decoder_layers", 4)
    ).to(device)
    
    # 2. Setup Data
    num_workers = getattr(cfg.data, "num_workers", 4)
    dataset = AudioManifestDataset(cfg.data.manifest_path, max_length=cfg.data.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=num_workers)
    
    val_dataloader = None
    if getattr(cfg.data, "val_manifest", None):
        val_dataset = AudioManifestDataset(cfg.data.val_manifest, max_length=cfg.data.max_length)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers)
    
    # 3. Setup Optimizer & Scheduler
    optimizer = optim.AdamW(rqvae.parameters(), lr=cfg.training.lr, weight_decay=0.01)
    
    # Calculate total steps for warmup
    num_training_steps = cfg.training.epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # 4. Training Loop
    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    step = 0
    for epoch in range(cfg.training.epochs):
        rqvae.train()
        pbar = tqdm(dataloader)
        for audio, lengths in pbar:
            audio = audio.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            # Extract SSL features
            with torch.no_grad():
                features = ssl_model(audio) # [B, T_feat, D]
                feat_lengths = torch.clamp(lengths // 320, max=features.size(1))
                mask = create_mask(feat_lengths, features.size(1), device)
            
            # Forward RQ-VAE with mask
            recon, total_loss, _ = rqvae(features, mask=mask)
            
            # Extract individual losses for logging
            with torch.no_grad():
                # Re-calculate normalized target for logging consistency
                target_mean = (features * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)
                target_std = torch.sqrt(
                    ((features - target_mean)**2 * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / 
                    (mask.sum(dim=1, keepdim=True).unsqueeze(-1) * features.size(-1) + 1e-6)
                )
                features_norm = (features - target_mean) / (target_std + 1e-6)
                
                mask_expanded = mask.unsqueeze(-1).expand_as(features)
                rl = F.mse_loss(recon * mask_expanded, features_norm * mask_expanded, reduction='sum')
                rl = rl / (mask.sum() * features.size(-1))
                vl = total_loss - rl
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(rqvae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            wandb.log({
                "train/loss_total": total_loss.item(), 
                "train/loss_recon": rl.item(),
                "train/loss_vq": vl.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch
            }, step=step)
            pbar.set_description(f"Epoch {epoch} | Loss: {total_loss.item():.4f}")
            
            if step % cfg.training.save_steps == 0:
                save_path = os.path.join(os.getcwd(), f"rqvae_checkpoint_{step}.pt")
                torch.save(rqvae.state_dict(), save_path)
                logger.info(f"Saved checkpoint to {save_path}")

        # Evaluation Loop
        if val_dataloader:
            rqvae.eval()
            total_val_loss = 0
            total_val_recon = 0
            val_steps = 0
            with torch.no_grad():
                for audio, lengths in val_dataloader:
                    audio = audio.to(device)
                    lengths = lengths.to(device)
                    try:
                        features = ssl_model(audio)
                        feat_lengths = torch.clamp(lengths // 320, max=features.size(1))
                        mask = create_mask(feat_lengths, features.size(1), device)
                        recon, loss, _ = rqvae(features, mask=mask)
                        
                        target_mean = (features * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)
                        target_std = torch.sqrt(
                            ((features - target_mean)**2 * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / 
                            (mask.sum(dim=1, keepdim=True).unsqueeze(-1) * features.size(-1) + 1e-6)
                        )
                        features_norm = (features - target_mean) / (target_std + 1e-6)

                        mask_expanded = mask.unsqueeze(-1).expand_as(features)
                        rl = F.mse_loss(recon * mask_expanded, features_norm * mask_expanded, reduction='sum')
                        rl = rl / (mask.sum() * features.size(-1))
                        
                        total_val_loss += loss.item()
                        total_val_recon += rl.item()
                        val_steps += 1
                    except Exception as e:
                        logger.error(f"Error during validation: {e}")
            
            if val_steps > 0:
                avg_val_loss = total_val_loss / val_steps
                avg_val_recon = total_val_recon / val_steps
                wandb.log({
                    "val/loss_total": avg_val_loss,
                    "val/loss_recon": avg_val_recon,
                    "val/loss_vq": avg_val_loss - avg_val_recon
                }, step=step)
                logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")
            rqvae.train()

    # Final Save
    final_path = os.path.join(os.getcwd(), "rqvae_final.pt")
    torch.save(rqvae.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    main()
