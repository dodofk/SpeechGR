import torch
import json
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
import joblib
from sklearn.cluster import MiniBatchKMeans

from speechgr.models.ssl_wrapper import SSLModelWrapper
from speechgr.models.rqvae import DocumentRQVAE

def load_kmeans(path):
    return joblib.load(path)

def train_kmeans(features, n_clusters=500, batch_size=10000):
    print(f"Training K-Means with K={n_clusters} on {features.shape[0]} frames...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42, n_init=3)
    kmeans.fit(features)
    return kmeans

def get_audio_paths(root):
    return list(Path(root).rglob("*.wav")) + list(Path(root).rglob("*.flac"))

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
        rqvae = DocumentRQVAE(
            input_dim=ssl_model.feature_dim,
            latent_dim=args.rqvae_latent_dim,
            codebook_size=args.rqvae_codebook_size,
            num_codebooks=args.rqvae_num_codebooks
        ).to(device)
        rqvae.load_state_dict(torch.load(args.rqvae_checkpoint, map_location=device))
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
    parser.add_argument("--rqvae_latent_dim", type=int, default=1024)
    parser.add_argument("--rqvae_codebook_size", type=int, default=256)
    parser.add_argument("--rqvae_num_codebooks", type=int, default=8)
    
    # K-Means
    parser.add_argument("--kmeans_model", type=str, default=None)
    parser.add_argument("--train_kmeans", action="store_true", help="Train K-Means on the dataset")
    parser.add_argument("--kmeans_k", type=int, default=500)
    
    args = parser.parse_args()
    main(args)
