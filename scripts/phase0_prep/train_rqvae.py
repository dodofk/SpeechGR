import logging
import os
import hydra
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import soundfile as sf
import numpy as np
import wandb

from speechgr.models.ssl_wrapper import SSLModelWrapper
from speechgr.models.rqvae import RQVAE

logger = logging.getLogger(__name__)

class AudioManifestDataset(Dataset):
    """
    Simple dataset to load audio from a manifest file.
    Manifest file should contain paths to audio files, one per line.
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
            # Load audio (assuming 16k mono for simplicity, real impl might need resampling)
            wav, sr = sf.read(path)
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1) # mix to mono
            
            # Pad or trim
            if len(wav) > self.max_length:
                wav = wav[:self.max_length]
            else:
                wav = np.pad(wav, (0, self.max_length - len(wav)))
                
            return torch.tensor(wav, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return torch.zeros(self.max_length, dtype=torch.float32)

@hydra.main(version_base=None, config_path="../../configs")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize wandb
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
    
    logger.info("Initializing RQ-VAE...")
    rqvae = RQVAE(
        input_dim=ssl_model.feature_dim,
        latent_dim=cfg.rqvae.latent_dim,
        codebook_size=cfg.rqvae.codebook_size,
        num_codebooks=cfg.rqvae.num_codebooks,
        commitment_cost=cfg.rqvae.commitment_cost
    ).to(device)
    
    # 2. Setup Data
    num_workers = getattr(cfg.data, "num_workers", 4)
    dataset = AudioManifestDataset(cfg.data.manifest_path, max_length=cfg.data.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=num_workers)
    
    val_dataloader = None
    if getattr(cfg.data, "val_manifest", None):
        val_dataset = AudioManifestDataset(cfg.data.val_manifest, max_length=cfg.data.max_length)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers)
    
    # 3. Setup Optimizer
    optimizer = optim.Adam(rqvae.parameters(), lr=cfg.training.lr)
    
    # 4. Training Loop
    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    step = 0
    for epoch in range(cfg.training.epochs):
        rqvae.train()
        pbar = tqdm(dataloader)
        for audio in pbar:
            audio = audio.to(device)
            
            optimizer.zero_grad()
            
            # Extract SSL features
            with torch.no_grad():
                features = ssl_model(audio) # [B, T, D]
            
            # Forward RQ-VAE
            recon, loss, _ = rqvae(features)
            
            loss.backward()
            optimizer.step()
            
            step += 1
            wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=step)
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
            if step % cfg.training.save_steps == 0:
                save_path = os.path.join(os.getcwd(), f"rqvae_checkpoint_{step}.pt")
                torch.save(rqvae.state_dict(), save_path)
                logger.info(f"Saved checkpoint to {save_path}")

        # Evaluation Loop
        if val_dataloader:
            rqvae.eval()
            total_val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for audio in val_dataloader:
                    audio = audio.to(device)
                    try:
                        features = ssl_model(audio)
                        _, loss, _ = rqvae(features)
                        total_val_loss += loss.item()
                        val_steps += 1
                    except Exception as e:
                        logger.error(f"Error during validation: {e}")
            
            if val_steps > 0:
                avg_val_loss = total_val_loss / val_steps
                wandb.log({"val/loss": avg_val_loss}, step=step)
                logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")
            rqvae.train()

    # Final Save
    final_path = os.path.join(os.getcwd(), "rqvae_final.pt")
    torch.save(rqvae.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    main()
