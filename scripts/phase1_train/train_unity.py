import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from speechgr.models.unity import UnitySpeechModel
from speechgr.data.dual_task import DualTaskDataset, DualTaskCollator

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="../../configs", config_name="train_unity", version_base="1.3")
def main(cfg: DictConfig):
    # 1. Setup Fabric
    logger = WandbLogger(
        project=cfg.logging.project,
        name=cfg.logging.name,
        entity=cfg.logging.entity,
        mode=cfg.logging.mode
    )
    fabric = Fabric(
        accelerator=cfg.env.accelerator,
        devices=cfg.env.devices,
        precision=cfg.env.precision,
        strategy=cfg.env.strategy,
        loggers=logger
    )
    fabric.launch()
    fabric.seed_everything(cfg.training.seed)
    
    if fabric.global_rank == 0:
        fabric.print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        
    # 2. Dataset & Dataloader
    dataset = DualTaskDataset(
        id_map_path=cfg.data.id_map,
        semantic_map_path=cfg.data.semantic_map,
        audio_root=cfg.data.audio_root,
        indexing_prob=cfg.data.indexing_prob,
        crop_duration=cfg.data.crop_duration,
        sample_rate=cfg.data.sample_rate
    )
    
    collator = DualTaskCollator(
        sem_pad_id=-100,
        ret_pad_id=-100,
        sem_bos_id=0,
        ret_bos_id=0
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    dataloader = fabric.setup_dataloaders(dataloader)
    
    # 3. Model
    model = UnitySpeechModel(
        ssl_model_name=cfg.model.ssl_model_name,
        ssl_layer=cfg.model.ssl_layer,
        semantic_vocab_size=cfg.model.semantic_vocab_size,
        retrieval_vocab_size=cfg.model.retrieval_vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    model, optimizer = fabric.setup(model, optimizer)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 4. Training Loop
    fabric.print(f"Starting training for {cfg.training.epochs} epochs...")
    global_step = 0
    
    # Get output directory from Hydra
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except:
        output_dir = os.getcwd()
    
    for epoch in range(cfg.training.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=fabric.global_rank != 0)
        
        for batch in pbar:
            # batch is already on device thanks to Fabric setup_dataloaders
            
            sem_logits, ret_logits = model(
                input_values=batch["input_values"],
                semantic_labels=batch["semantic_labels"],
                retrieval_labels=batch["retrieval_labels"],
                attention_mask=batch["attention_mask"]
            )
            
            # Loss for Semantic Bridge
            # logits: [B, L-1, V], targets: [B, L-1] (shifted by BOS)
            sem_loss = criterion(
                sem_logits.reshape(-1, sem_logits.size(-1)),
                batch["semantic_labels"][:, 1:].reshape(-1)
            )
            
            # Loss for Retrieval Head
            ret_loss = criterion(
                ret_logits.reshape(-1, ret_logits.size(-1)),
                batch["retrieval_labels"][:, 1:].reshape(-1)
            )
            
            total_loss = sem_loss + cfg.training.lambda_ret * ret_loss
            
            optimizer.zero_grad()
            fabric.backward(total_loss)
            
            if cfg.training.grad_clip > 0:
                fabric.clip_gradients(model, optimizer, max_norm=cfg.training.grad_clip)
                
            optimizer.step()
            
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                fabric.log_dict({
                    "loss/total": total_loss.item(),
                    "loss/sem": sem_loss.item(),
                    "loss/ret": ret_loss.item(),
                    "step": global_step,
                    "epoch": epoch
                })
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            
            # Checkpointing
            if global_step % cfg.training.save_steps == 0:
                save_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
                fabric.save(save_path, {
                    "model": model, 
                    "optimizer": optimizer, 
                    "step": global_step,
                    "epoch": epoch,
                    "config": OmegaConf.to_container(cfg)
                })
                fabric.print(f"Saved checkpoint to {save_path}")

        # End of Epoch Save
        save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
        fabric.save(save_path, {
            "model": model,
            "optimizer": optimizer,
            "step": global_step,
            "epoch": epoch
        })

if __name__ == "__main__":
    main()
