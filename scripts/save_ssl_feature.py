"""
This script is used for pre-compute SSL model feature for SLUE SQA continuous GR.

Please manually change the output dir for your path
"""

import argparse
import logging
import os
from typing import List, Dict, Optional, Any, Tuple

import h5py
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    PreTrainedModel,
    FeatureExtractionMixin,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SSL features for SLUE SQA 5.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ricky/dodofk/dataset/slue_sqa5_wavlm_large",
        help="Output directory for saving features",
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/wavlm-large", help="Model name or path"
    )
    parser.add_argument(
        "--use_vad",
        action="store_true",
        default=True,
        help="Whether to use VAD to remove silent parts",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
        help="Whether to use CUDA for computation",
    )
    return parser.parse_args()


def load_vad_model(use_cuda: bool = False) -> Tuple[torch.nn.Module, Any]:
    """Load the VAD model from Silero."""
    logger.info("Loading VAD model")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )

    (get_speech_timestamps, _, _, _, _) = utils

    if torch.cuda.is_available() and use_cuda:
        vad_model = vad_model.to("cuda")

    return vad_model, get_speech_timestamps


def collect_chunks(
    corpus_wav: torch.Tensor, speech_timestamps: List[Dict[str, int]]
) -> List[torch.Tensor]:
    """
    Collect chunks of audio based on VAD timestamps.
    
    Args: 
        corpus_wav: torch.Tensor, the waveform audio tensor
        speech_timestamps: List[Dict[str, int]], the speech timestamps

    Returns:
        chunks: List[torch.Tensor], the chunks of audio with speech
    """
    chunks = []
    for timestamp in speech_timestamps:
        start, end = timestamp["start"], timestamp["end"]
        chunks.append(corpus_wav[start:end])
    return chunks


def apply_vad(
    audio: np.ndarray,
    vad_model: torch.nn.Module,
    get_speech_timestamps: Any,
    use_cuda: bool = False,
) -> np.ndarray:
    """Apply VAD to remove silent parts from audio."""
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio)

    if torch.cuda.is_available() and use_cuda:
        audio = audio.to("cuda")

    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        threshold=0.5,
        min_speech_duration_ms=300,
        sampling_rate=16000,
    )

    chunks = collect_chunks(audio, speech_timestamps)
    if chunks:
        processed_audio = torch.cat(chunks)
    else:
        # If no speech detected, return the original audio
        processed_audio = audio

    # convert back to numpy
    return processed_audio.cpu().numpy()


def extract_features(
    audio: np.ndarray,
    feature_extractor: FeatureExtractionMixin,
    model: PreTrainedModel,
    use_cuda: bool = False,
) -> np.ndarray:
    """Extract features from audio using the SSL model."""
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    if torch.cuda.is_available() and use_cuda:
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state.cpu().numpy()
    return last_hidden_states


def process_corpus(
    ds: DatasetDict,
    splits: List[str],
    output_dir: str,
    feature_extractor: FeatureExtractionMixin,
    model: PreTrainedModel,
    do_vad: bool = True,
    use_cuda: bool = False,
) -> None:
    """Process and save corpus features."""
    logger.info("Processing corpus features")
    CORPUS_LEN = 15883  # pre-processed corpus length
    corpus_set = set()

    corpus_h5_file = h5py.File(f"{output_dir}/slue_sqa5_corpus.h5", "w")
    str_dtype = h5py.string_dtype(encoding="utf-8")
    vlen_float_dtype = h5py.vlen_dtype(np.float32)

    corpus_id_dataset = corpus_h5_file.create_dataset(
        "ids",
        shape=(CORPUS_LEN,),
        dtype=str_dtype,
    )
    corpus_feat_dataset = corpus_h5_file.create_dataset(
        "features",
        shape=(CORPUS_LEN,),
        dtype=vlen_float_dtype,
    )

    # Load VAD model if needed
    if do_vad:
        vad_model, get_speech_timestamps = load_vad_model(use_cuda)

    cur_idx = 0
    for split in splits:
        for data in tqdm(ds[split], desc=f"Processing corpus in {split}"):
            if data["document_id"] in corpus_set:
                continue

            corpus_set.add(data["document_id"])
            corpus_id_dataset[cur_idx] = data["document_id"]

            corpus_wav = data["document_audio"]["array"]

            # Apply VAD if needed
            if do_vad:
                corpus_wav = apply_vad(
                    corpus_wav, vad_model, get_speech_timestamps, use_cuda
                )

            # Extract features
            features = extract_features(corpus_wav, feature_extractor, model, use_cuda)

            # Save to h5 file
            corpus_feat_dataset[cur_idx] = features
            cur_idx += 1

            if cur_idx == CORPUS_LEN:
                break
        if cur_idx == CORPUS_LEN:
            break

    corpus_h5_file.close()
    logger.info(f"Successfully processed {cur_idx} corpus items")


def process_splits(
    ds: DatasetDict,
    splits: List[str],
    output_dir: str,
    feature_extractor: FeatureExtractionMixin,
    model: PreTrainedModel,
    do_vad: bool = True,
    use_cuda: bool = False,
) -> None:
    """Process and save features for each split."""
    # Load VAD model if needed
    if do_vad:
        vad_model, get_speech_timestamps = load_vad_model(use_cuda)

    str_dtype = h5py.string_dtype(encoding="utf-8")
    vlen_float_dtype = h5py.vlen_dtype(np.float32)

    for split in splits:
        logger.info(f"Processing {split} split")
        split_h5_file = h5py.File(f"{output_dir}/{split}.h5", "w")
        split_id_dataset = split_h5_file.create_dataset(
            "ids",
            shape=(len(ds[split]),),
            dtype=str_dtype,
        )
        split_label_dataset = split_h5_file.create_dataset(
            "labels",
            shape=(len(ds[split]),),
            dtype=str_dtype,
        )
        split_feat_dataset = split_h5_file.create_dataset(
            "features",
            shape=(len(ds[split]),),
            dtype=vlen_float_dtype,
        )

        for i, data in enumerate(tqdm(ds[split], desc=f"Processing {split} split")):
            split_id_dataset[i] = data["question_id"]
            split_label_dataset[i] = data["document_id"]

            corpus_wav = data["question_audio"]["array"]

            # Apply VAD if needed
            if do_vad:
                corpus_wav = apply_vad(
                    corpus_wav, vad_model, get_speech_timestamps, use_cuda
                )

            # Extract features
            features = extract_features(corpus_wav, feature_extractor, model, use_cuda)

            # Save to h5 file
            split_feat_dataset[i] = features

        split_h5_file.close()
        logger.info(f"Successfully processed {split} split with {len(ds[split])} items")


def main() -> None:
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading dataset from asapp/slue-phase-2")
    ds = load_dataset("asapp/slue-phase-2", "sqa5")

    logger.info(f"Loading model {args.model}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    if torch.cuda.is_available() and args.use_cuda:
        model = model.to("cuda")
        logger.info("Using CUDA for model inference")

    splits = ["train", "validation", "test", "verified_test"]

    # Process corpus
    process_corpus(
        ds,
        splits,
        args.output_dir,
        feature_extractor,
        model,
        args.use_vad,
        args.use_cuda,
    )

    # Process splits
    process_splits(
        ds,
        splits,
        args.output_dir,
        feature_extractor,
        model,
        args.use_vad,
        args.use_cuda,
    )

    logger.info("All processing completed successfully")


if __name__ == "__main__":
    main()
