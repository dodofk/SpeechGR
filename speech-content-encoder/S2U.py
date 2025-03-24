"""
The file is script to preprocess the slue sqa5 data, which use hubert as checkpoint for kmeans. 
"""

import numpy as np
import joblib
import torch
import torchaudio
from tqdm import tqdm
import os
import torch
from datasets import load_dataset
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import fairseq


@dataclass
class Config:
    """Configuration for the SLUE data processing script."""
    km_path: str
    output_dir: str
    ckpt_path: Optional[str] = None
    layer: int = 22
    sample_rate: int = 16000
    chunk_length: int = 250000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ApplyKmeans(object):
    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if self.return_diff:
                return min_dist.indices.cpu().numpy(), min_dist.values.cpu().numpy()
            else:
                return min_dist.indices.cpu().numpy()
        else:
            dist = np.sqrt(
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            if self.return_diff:
                return np.argmin(dist, axis=1), np.min(dist, axis=1)
            else:
                return np.argmin(dist, axis=1)


def reader(fname, sample_rate):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != sample_rate:
        wav = torchaudio.transforms.Resample(ori_sr, sample_rate)(wav)
    silence_samples = int(0.005 * sample_rate)
    # Create a tensor filled with zeros for the silent gap
    silence = torch.zeros((1, silence_samples))
    # Concatenate the silence tensor with the original waveform
    new_waveform = torch.cat((silence, wav), dim=1)
    # return wav.squeeze()
    return new_waveform.squeeze()


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            _,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk

    def get_feats(self, waveform):
        # Assume waveform is a numpy array
        with torch.no_grad():
            x = torch.from_numpy(waveform).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    layer=self.layer,
                )
                feat.append(feat_chunk["x"])
        return torch.cat(feat, 1).squeeze(0)


def process_split(split, dataset, config, extractor, apply_kmeans):
    print(f"Processing split: {split}")
    files = dataset[split]

    for _, data in enumerate(
        tqdm(files, desc="transforming passage to discrete code")
    ):
        q_wavs = data["question_audio"]["array"]
        q_id = data["question_id"]
        q_text = data["normalized_question_text"]
        
        p_wavs = data["document_audio"]["array"]
        p_id = data["document_id"]
        p_text = data["normalized_document_text"]

        # check if question code is already generated
        if not os.path.exists(
            os.path.join(config.output_dir, f"{split}_code", f"{q_id}.code")
        ):
            if torch.cuda.is_available() and config.device == "cuda":
                wavs = torch.FloatTensor(q_wavs)
                wavs = wavs.cuda()

            if len(wavs) > config.chunk_length + 1250:
                chunks = list(torch.split(wavs, config.chunk_length, dim=-1))
                # if last chunk too small, concat back
                if len(chunks[-1]) < 500:
                    last_chunk = chunks[-1]
                    chunks = chunks[0:-1]
                    chunks[-1] = torch.cat([chunks[-1], last_chunk])
                code = list()
                for chunk in chunks:
                    feature = extractor([chunk])
                    code.append(torch.tensor(apply_kmeans(feature['hidden_state_22'].squeeze(dim=0).cuda())))
                code = torch.cat(code)
            else:
                feature = extractor([wavs])
                code = apply_kmeans(feature['hidden_state_22'].squeeze().cuda())
                code = torch.tensor(code)

            merged_code, counts = torch.unique_consecutive(code, return_counts=True)

            np.savetxt(
                os.path.join(config.output_dir, f"{split}_code", f"{q_id}.code"),
                merged_code.long(),
                fmt="%i",
            )
            np.savetxt(
                os.path.join(config.output_dir, f"{split}_code", f"{q_id}.cnt"),
                counts.long(),
                fmt="%i",
            )
            # Save transcription to a text file
            with open(
                os.path.join(config.output_dir, f"{split}_code", f"{q_id}.trans.txt"), "w"
            ) as trans_file:
                trans_file.write(q_text)
                
        # check if document code is already generated
        if not os.path.exists(
            os.path.join(config.output_dir, f"document_code", f"{p_id}.code")
        ):
            if torch.cuda.is_available() and config.device == "cuda":
                wavs = torch.FloatTensor(p_wavs)
                wavs = wavs.cuda()

            if len(wavs) > config.chunk_length + 1250:
                chunks = list(torch.split(wavs, config.chunk_length, dim=-1))
                # if last chunk too small, concat back
                if len(chunks[-1]) < 500:
                    last_chunk = chunks[-1]
                    chunks = chunks[0:-1]
                    chunks[-1] = torch.cat([chunks[-1], last_chunk])
                code = list()
                for chunk in chunks:
                    feature = extractor([chunk])
                    code.append(torch.tensor(apply_kmeans(feature['hidden_state_22'].squeeze(dim=0).cuda())))
                code = torch.cat(code)
            else:
                feature = extractor([wavs])
                code = apply_kmeans(feature['hidden_state_22'].squeeze().cuda())
                code = torch.tensor(code)

            merged_code, counts = torch.unique_consecutive(code, return_counts=True)

            np.savetxt(
                os.path.join(config.output_dir, f"document_code", f"{p_id}.code"),
                merged_code.long(),
                fmt="%i",
            )
            np.savetxt(
                os.path.join(config.output_dir, f"document_code", f"{p_id}.cnt"),
                counts.long(),
                fmt="%i",
            )
            # Save transcription to a text file
            with open(
                os.path.join(config.output_dir, f"document_code", f"{p_id}.trans.txt"), "w"
            ) as trans_file:
                trans_file.write(p_text)


def setup_directories(config):
    """Create necessary directories."""
    splits = ["train", "validation", "test", "verified_test"]
    
    for split in splits:
        os.makedirs(os.path.join(config.output_dir, f"{split}_code"), exist_ok=True)
    
    os.makedirs(os.path.join(config.output_dir, f"document_code"), exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process SLUE SQA5 data")
    
    parser.add_argument("--km_path", type=str, required=True,
                        help="Path to the kmeans model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store the output")
    parser.add_argument("--ckpt_path", type=str,
                        help="Path to checkpoint for HubertFeatureReader (if used)")
    parser.add_argument("--layer", type=int, default=22,
                        help="Layer to extract features from")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Audio sample rate")
    parser.add_argument("--chunk_length", type=int, default=250000,
                        help="Length of chunks for processing long audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (cuda/cpu)")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create config
    config = Config(
        km_path=args.km_path,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt_path,
        layer=args.layer,
        sample_rate=args.sample_rate,
        chunk_length=args.chunk_length,
        device=args.device
    )
    
    # Create output directories
    setup_directories(config)
    
    # Load dataset
    dataset = load_dataset("asapp/slue-phase-2", "sqa5")
    
    # Initialize extractor
    extractor = torch.hub.load("s3prl/s3prl", "hubert_large_ll60k")
    extractor.eval()

    # Move extractor to the correct device
    if torch.cuda.is_available() and config.device == "cuda":
        extractor = extractor.cuda()
    else:
        extractor = extractor.cpu()

    # Initialize kmeans model
    apply_kmeans = ApplyKmeans(config.km_path)

    # Process each split
    splits = ["train", "validation", "test", "verified_test"]
    for split in splits:
        process_split(split, dataset, config, extractor, apply_kmeans)


if __name__ == "__main__":
    main() 