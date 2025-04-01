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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import fairseq
import glob
import json

from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration for data processing.
    Extensible to other datasets by customizing the data-loading and iteration logic.
    """

    km_path: str
    output_dir: str
    layer: int = 22
    sample_rate: int = 16000
    chunk_length: int = 250000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    task: str = "slue_sqa5"

    def __post_init__(self):
        assert self.task in [
            "slue_sqa5",
            "librispeech",
        ], "task must be either slue_sqa5 or librispeech"


class ApplyKmeans:
    """
    Wrapper for applying k-means clustering to extracted representations.
    """

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
        """
        Apply k-means to a tensor or numpy array.
        """
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


def extract_discrete_code(
    wavs: torch.Tensor, config: Config, extractor, apply_kmeans: ApplyKmeans
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the discrete code from a given waveform using the provided extractor
    and k-means model, in a chunked fashion if needed.
    Returns:
        - merged_code: The chunked and consecutive-unique code sequence.
        - counts: The consecutive run lengths for each code.
    """
    # Move waveform to device if needed
    if torch.cuda.is_available() and config.device == "cuda":
        wavs = wavs.cuda()

    # If audio is large, process in chunks
    if len(wavs) > config.chunk_length + 1250:
        chunks = list(torch.split(wavs, config.chunk_length, dim=-1))
        # if last chunk is too small, concat it back
        if len(chunks[-1]) < 500:
            last_chunk = chunks[-1]
            chunks = chunks[:-1]
            chunks[-1] = torch.cat([chunks[-1], last_chunk])
        code = []
        for chunk in chunks:
            feature = extractor([chunk])
            chunk_code = apply_kmeans(feature["hidden_state_22"].squeeze(dim=0).cuda())
            code.append(torch.tensor(chunk_code))
        code = torch.cat(code)
    else:
        feature = extractor([wavs])
        code = apply_kmeans(feature["hidden_state_22"].squeeze().cuda())
        code = torch.tensor(code)

    merged_code, counts = torch.unique_consecutive(code, return_counts=True)
    return merged_code, counts


def process_sqa_split(
    split: str, dataset, config: Config, extractor, apply_kmeans: ApplyKmeans
):
    """
    Process a specific split (train, validation, etc.) of the SQA dataset.
    Extracts discrete codes for question and document audio.
    """
    print(f"Processing split: {split}")
    files = dataset[split]

    for _, data in enumerate(tqdm(files, desc=f"Extracting codes for {split} split")):
        q_wavs = data["question_audio"]["array"]
        q_id = data["question_id"]
        q_text = data["normalized_question_text"]

        p_wavs = data["document_audio"]["array"]
        p_id = data["document_id"]
        p_text = data["normalized_document_text"]

        q_code_path = os.path.join(config.output_dir, f"{split}_code", f"{q_id}.code")
        p_code_path = os.path.join(config.output_dir, "document_code", f"{p_id}.code")

        # Extract question codes if not already done
        if not os.path.exists(q_code_path):
            q_tensor = torch.FloatTensor(q_wavs)
            merged_code, counts = extract_discrete_code(
                q_tensor, config, extractor, apply_kmeans
            )
            # Save code
            np.savetxt(q_code_path, merged_code.long(), fmt="%i")
            np.savetxt(
                os.path.join(config.output_dir, f"{split}_code", f"{q_id}.cnt"),
                counts.long(),
                fmt="%i",
            )
            # Save transcription
            with open(
                os.path.join(config.output_dir, f"{split}_code", f"{q_id}.trans.txt"),
                "w",
            ) as trans_file:
                trans_file.write(q_text)

        # Extract document codes if not already done
        if not os.path.exists(p_code_path):
            p_tensor = torch.FloatTensor(p_wavs)
            merged_code, counts = extract_discrete_code(
                p_tensor, config, extractor, apply_kmeans
            )
            # Save code
            np.savetxt(p_code_path, merged_code.long(), fmt="%i")
            np.savetxt(
                os.path.join(config.output_dir, "document_code", f"{p_id}.cnt"),
                counts.long(),
                fmt="%i",
            )
            # Save transcription
            with open(
                os.path.join(config.output_dir, "document_code", f"{p_id}.trans.txt"),
                "w",
            ) as trans_file:
                trans_file.write(p_text)


def process_sqa_dataset(dataset, config: Config, extractor, apply_kmeans: ApplyKmeans):
    """
    High-level function to process all splits of the SQA dataset.
    Calls process_sqa_split for each relevant split.
    """
    splits = ["train", "validation", "test", "verified_test"]

    setup_odqa_directories(config, splits)

    for split in splits:
        process_sqa_split(split, dataset, config, extractor, apply_kmeans)


def setup_odqa_directories(config: Config, splits: list[str]):
    """
    Create necessary directories for storing discrete codes and related metadata for open domain question answering dataset.
    """
    for split in splits:
        os.makedirs(os.path.join(config.output_dir, f"{split}_code"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "document_code"), exist_ok=True)


def process_librispeech(config: Config, extractor, apply_kmeans: ApplyKmeans):
    """
    Process the LibriSpeech dataset.
    """
    # Manually change the file_dir_path to the path of the LibriSpeech dataset you want to process
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # file_dir_path = "/home/ricky/dodofk/dataset/LibriSpeech/train-clean-100"
    file_dir_path = "/home/ricky/dodofk/dataset/medium"

    # Get all flac files in the directory
    flac_files = glob.glob(os.path.join(file_dir_path, "**/*.flac"), recursive=True)

    fname_list = []

    logger.info(f"Processing {len(flac_files)} files for LibriLight 6k dataset")

    for flac_file in tqdm(flac_files, desc="Processing LibriSpeech dataset"):
        flac_fname = os.path.basename(flac_file)
        wav, sr = torchaudio.load(flac_file)
        if sr != config.sample_rate:
            wav = torchaudio.transforms.Resample(sr, config.sample_rate)(wav)
        wav = wav.squeeze()
        code = extract_discrete_code(wav, config, extractor, apply_kmeans)
        np.savetxt(
            os.path.join(config.output_dir, f"{flac_fname}.code"), code, fmt="%i"
        )
        fname_list.append(flac_fname)

    with open(os.path.join(config.output_dir, "fname_list.json"), "w") as f:
        json.dump(fname_list, f)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process SLUE SQA5 data")
    parser.add_argument(
        "--km_path", type=str, required=True, help="Path to the kmeans model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to store the output"
    )
    parser.add_argument(
        "--layer", type=int, default=22, help="Layer to extract features from"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=250000,
        help="Length of chunks for processing long audio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (cuda/cpu)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="slue_sqa5",
        help="Task to process (slue_sqa5/librispeech)",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Create config
    config = Config(
        km_path=args.km_path,
        output_dir=args.output_dir,
        layer=args.layer,
        sample_rate=args.sample_rate,
        chunk_length=args.chunk_length,
        device=args.device,
        task=args.task,
    )

    # Load dataset (example: SLUE SQA5)
    dataset = load_dataset("asapp/slue-phase-2", "sqa5")

    # Example: load an extractor (S3PRL HuBERT). If using a local or fairseq model, adjust accordingly.
    extractor = torch.hub.load("s3prl/s3prl", "hubert_large_ll60k")
    extractor.eval()
    if config.device == "cuda" and torch.cuda.is_available():
        extractor = extractor.cuda()
    else:
        extractor = extractor.cpu()

    # Initialize kmeans
    apply_kmeans = ApplyKmeans(config.km_path)

    if config.task == "slue_sqa5":
        # Process SQA dataset
        logger.info("Processing SLUE SQA5 dataset")
        process_sqa_dataset(dataset, config, extractor, apply_kmeans)
    elif config.task == "librispeech":
        logger.info("Processing LibriSpeech dataset")
        process_librispeech(config, extractor, apply_kmeans)


if __name__ == "__main__":
    main()
