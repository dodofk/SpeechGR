import torch
import json
import argparse
import os
from typing import Dict

from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
import joblib
from sklearn.cluster import MiniBatchKMeans

from speechgr.models.ssl_wrapper import SSLModelWrapper
from speechgr.models.rqvae import DocumentRQVAE, SlidingWindowDocumentRQVAE

def load_kmeans(path):
    return joblib.load(path)

def train_kmeans(features, n_clusters=500, batch_size=10000):
    print(f"Training K-Means with K={n_clusters} on {features.shape[0]} frames...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42, n_init=3)
    kmeans.fit(features)
    return kmeans

def get_audio_paths(root):
    return list(Path(root).rglob("*.wav")) + list(Path(root).rglob("*.flac"))


def _unwrap_state_dict(raw_checkpoint):
    """Support both plain state_dict and wrapped checkpoints."""
    if not isinstance(raw_checkpoint, dict):
        return raw_checkpoint

    for key in ("state_dict", "model_state_dict", "model"):
        if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
            return raw_checkpoint[key]
    return raw_checkpoint


def _infer_pooling_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    if any(k.startswith("pooling.pool.") or k.startswith("decoder.self_attn_layers") for k in state_dict):
        return "sliding_window"
    return "global"


def _build_rqvae(args, input_dim: int, pooling_type: str):
    common_kwargs = {
        "input_dim": input_dim,
        "latent_dim": args.rqvae_latent_dim,
        "codebook_size": args.rqvae_codebook_size,
        "num_codebooks": args.rqvae_num_codebooks,
        "commitment_cost": args.rqvae_commitment_cost,
        "decay": args.rqvae_decay,
        "num_encoder_layers": args.rqvae_num_encoder_layers,
        "num_decoder_layers": args.rqvae_num_decoder_layers,
    }

    if pooling_type == "sliding_window":
        return SlidingWindowDocumentRQVAE(
            **common_kwargs,
            window_size=args.rqvae_window_size,
            window_stride=args.rqvae_window_stride,
            pooling_hidden_dim=args.rqvae_pooling_hidden_dim,
            aggregate_for_retrieval=args.rqvae_aggregate_for_retrieval,
        )

    return DocumentRQVAE(**common_kwargs)


def _compact_window_codes(codes: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Convert [B, num_windows, num_codebooks] to [B, num_codebooks].

    This keeps ID lengths bounded for trie-based retrieval.
    """
    if mode == "vote":
        return torch.mode(codes, dim=1).values
    if mode == "first":
        return codes[:, 0]
    raise ValueError(f"Unsupported compaction mode: {mode}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load SSL Model
    print(f"Loading SSL Model: {args.ssl_model}...")
    ssl_model = SSLModelWrapper(args.ssl_model, layer=args.ssl_layer).to(device)
    ssl_model.eval()
    
    # 2. Load RQ-VAE (for Document IDs)
    rqvae = None
    if args.rqvae_checkpoint and os.path.exists(args.rqvae_checkpoint):
        print(f"Loading RQ-VAE from {args.rqvae_checkpoint}...")
        raw_checkpoint = torch.load(args.rqvae_checkpoint, map_location=device)
        state_dict = _unwrap_state_dict(raw_checkpoint)

        pooling_type = args.rqvae_pooling_type
        if pooling_type == "auto":
            pooling_type = _infer_pooling_type_from_state_dict(state_dict)
        print(f"Detected/selected RQ-VAE pooling type: {pooling_type}")

        rqvae = _build_rqvae(args, ssl_model.feature_dim, pooling_type).to(device)
        load_result = rqvae.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys:
            print(f"Warning: missing keys when loading RQ-VAE: {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"Warning: unexpected keys when loading RQ-VAE: {len(load_result.unexpected_keys)}")
        rqvae.eval()
    else:
        print("Warning: No RQ-VAE checkpoint provided. Skipping ID generation.")

    # 3. K-Means (Train or Load)
    kmeans = None
    if args.train_kmeans:
        # Pass 1: Collect features for training
        print("Collecting features for K-Means training...")
        audio_paths = get_audio_paths(args.audio_root)
        # Downsample for training speed (e.g. 10% of files)
        train_paths = audio_paths[:min(len(audio_paths), 2000)] 
        
        all_feats = []
        with torch.no_grad():
            for p in tqdm(train_paths):
                try:
                    audio, _ = librosa.load(p, sr=16000)
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
                    feats = ssl_model(audio_tensor).squeeze(0).cpu().numpy()
                    # Randomly sample frames from each file to avoid memory issues
                    if feats.shape[0] > 100:
                        indices = np.random.choice(feats.shape[0], 100, replace=False)
                        feats = feats[indices]
                    all_feats.append(feats)
                except Exception as e:
                    print(f"Error processing {p}: {e}")
        
        train_data = np.concatenate(all_feats, axis=0)
        kmeans = train_kmeans(train_data, n_clusters=args.kmeans_k)
        
        os.makedirs(args.output_dir, exist_ok=True)
        joblib.dump(kmeans, os.path.join(args.output_dir, "kmeans_model.pkl"))
        print(f"K-Means model saved to {args.output_dir}/kmeans_model.pkl")
    
    elif args.kmeans_model and os.path.exists(args.kmeans_model):
        print(f"Loading K-Means from {args.kmeans_model}...")
        kmeans = load_kmeans(args.kmeans_model)
    else:
        print("Warning: No K-Means model provided. Skipping Semantic Token generation.")

    # 4. Process Audio (Inference)
    if not args.train_kmeans: # Only generate if not just training
        audio_paths = get_audio_paths(args.audio_root)
        print(f"Found {len(audio_paths)} audio files for indexing.")
        
        id_map = {}
        semantic_map = {}
        warned_window_compaction = False
        
        with torch.no_grad():
            for p in tqdm(audio_paths):
                try:
                    audio, _ = librosa.load(p, sr=16000)
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
                    
                    # Extract Features
                    feats = ssl_model(audio_tensor) # [1, T, D]
                    
                    # Generate Document ID
                    if rqvae:
                        codes = rqvae.encode(feats) # [1, 8]

                        # Guardrail: avoid oversized variable-length IDs from window-level output.
                        if codes.dim() == 3:
                            if not warned_window_compaction:
                                print(
                                    "Warning: encode() returned window-level codes; "
                                    f"compacting with mode={args.rqvae_compact_all_mode}"
                                )
                                warned_window_compaction = True
                            codes = _compact_window_codes(codes, args.rqvae_compact_all_mode)

                        if codes.dim() != 2:
                            raise ValueError(
                                "Expected document codes with shape [B, num_codebooks], "
                                f"got {tuple(codes.shape)}"
                            )

                        id_map[str(p)] = codes.squeeze(0).cpu().tolist()
                    
                    # Generate Semantic Tokens
                    if kmeans:
                        flat_feats = feats.squeeze(0).cpu().numpy()
                        clusters = kmeans.predict(flat_feats)
                        # Apply run-length encoding (deduplication) as simple "subword modeling"
                        # [0, 0, 1, 1, 1, 2] -> [0, 1, 2]
                        dedup_clusters = [clusters[0]]
                        for c in clusters[1:]:
                            if c != dedup_clusters[-1]:
                                dedup_clusters.append(c)
                        semantic_map[str(p)] = [int(c) for c in dedup_clusters]
                except Exception as e:
                    print(f"Error indexing {p}: {e}")
                
        # 5. Save Maps
        os.makedirs(args.output_dir, exist_ok=True)
        if id_map:
            with open(os.path.join(args.output_dir, "id_map.json"), "w") as f:
                json.dump(id_map, f)
        if semantic_map:
            with open(os.path.join(args.output_dir, "semantic_map.json"), "w") as f:
                json.dump(semantic_map, f)
        
        print(f"Indices saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ssl_model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--ssl_layer", type=int, default=24)
    
    # RQ-VAE
    parser.add_argument("--rqvae_checkpoint", type=str, default=None)
    parser.add_argument(
        "--rqvae_pooling_type",
        type=str,
        default="auto",
        choices=["auto", "global", "sliding_window"],
        help="Model family used to build the RQ-VAE before loading checkpoint",
    )
    parser.add_argument("--rqvae_latent_dim", type=int, default=1024)
    parser.add_argument("--rqvae_codebook_size", type=int, default=256)
    parser.add_argument("--rqvae_num_codebooks", type=int, default=8)
    parser.add_argument("--rqvae_num_encoder_layers", type=int, default=4)
    parser.add_argument("--rqvae_num_decoder_layers", type=int, default=4)
    parser.add_argument("--rqvae_commitment_cost", type=float, default=0.25)
    parser.add_argument("--rqvae_decay", type=float, default=0.99)
    parser.add_argument("--rqvae_window_size", type=int, default=25)
    parser.add_argument("--rqvae_window_stride", type=int, default=12)
    parser.add_argument("--rqvae_pooling_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--rqvae_aggregate_for_retrieval",
        type=str,
        default="vote",
        choices=["vote", "mean", "first", "all"],
        help="Sliding-window aggregation mode used by SlidingWindowDocumentRQVAE.encode()",
    )
    parser.add_argument(
        "--rqvae_compact_all_mode",
        type=str,
        default="vote",
        choices=["vote", "first"],
        help="Fallback compaction if encode() returns all window codes",
    )
    
    # K-Means
    parser.add_argument("--kmeans_model", type=str, default=None)
    parser.add_argument("--train_kmeans", action="store_true", help="Train K-Means on the dataset")
    parser.add_argument("--kmeans_k", type=int, default=500)
    
    args = parser.parse_args()
    main(args)
