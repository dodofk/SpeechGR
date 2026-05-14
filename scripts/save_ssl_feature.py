"""
This script is used for pre-compute SSL model feature for SLUE SQA continuous GR.

Please manually change the output dir for your path
"""
import argparse
import logging
import os
import pickle
from typing import List, Dict, Optional, Any, Tuple

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
        default="/home/ricky/dodofk/dataset/slue_sqa5_wavlm_large_pkl",
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
        vad_model.eval()

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
    with torch.no_grad():
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
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

    # Get the last hidden states and ensure it's 2D (sequence_length, hidden_size)
    last_hidden_states = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    return last_hidden_states.cpu().numpy()


def process_corpus(
    ds: DatasetDict,
    splits: List[str],
    output_dir: str,
    feature_extractor: FeatureExtractionMixin,
    model: PreTrainedModel,
    do_vad: bool = True,
    use_cuda: bool = False,
) -> None:
    """Process and save corpus features as pickle."""
    logger.info("Processing corpus features")
    output_file = f"{output_dir}/slue_sqa5_corpus.pkl"
    
    # Check if file exists and load existing data
    corpus_data = []
    processed_ids = set()
    
    if os.path.exists(output_file):
        logger.info(f"Found existing corpus file at {output_file}, loading existing data")
        try:
            with open(output_file, 'rb') as f:
                corpus_data = pickle.load(f)
                processed_ids = {item["doc_id"] for item in corpus_data}
                logger.info(f"Found {len(processed_ids)} existing processed documents")
        except Exception as e:
            logger.warning(f"Error reading existing data: {str(e)}. Starting fresh.")
            if os.path.exists(output_file):
                backup_file = f"{output_file}.bak"
                logger.info(f"Creating backup of corrupted file at {backup_file}")
                os.rename(output_file, backup_file)
                corpus_data = []
    
    # Load VAD model if needed
    if do_vad:
        vad_model, get_speech_timestamps = load_vad_model(use_cuda)
    
    # Process data
    for split in splits:
        for data in tqdm(ds[split], desc=f"Processing corpus in {split}"):
            if data["document_id"] in processed_ids:
                continue
                
            try:
                processed_ids.add(data["document_id"])
                corpus_wav = data["document_audio"]["array"]

                # Apply VAD if needed
                if do_vad:
                    corpus_wav = apply_vad(
                        corpus_wav, vad_model, get_speech_timestamps, use_cuda
                    )

                # Extract features
                features = extract_features(corpus_wav, feature_extractor, model, use_cuda)
                
                # Add to corpus data
                corpus_data.append({
                    "doc_id": data["document_id"], 
                    "feature": features
                })
                
                # Save periodically to avoid data loss
                if len(corpus_data) % 1000 == 0:
                    with open(output_file, 'wb') as f:
                        pickle.dump(corpus_data, f)
                        
            except Exception as e:
                logger.error(f"Error processing document {data['document_id']}: {str(e)}")
                continue
    
    # Save final data
    with open(output_file, 'wb') as f:
        pickle.dump(corpus_data, f)
    
    logger.info(f"Successfully processed {len(corpus_data)} corpus items")


def process_splits(
    ds: DatasetDict,
    splits: List[str],
    output_dir: str,
    feature_extractor: FeatureExtractionMixin,
    model: PreTrainedModel,
    do_vad: bool = True,
    use_cuda: bool = False,
) -> None:
    """Process and save features for each split as pickle."""
    # Load VAD model if needed
    if do_vad:
        vad_model, get_speech_timestamps = load_vad_model(use_cuda)

    for split in splits:
        logger.info(f"Processing {split} split")
        output_file = f"{output_dir}/{split}.pkl"
        
        # Check if file exists and load existing data
        split_data = []
        processed_ids = set()
        
        if os.path.exists(output_file):
            logger.info(f"Found existing {split} file at {output_file}, loading existing data")
            try:
                with open(output_file, 'rb') as f:
                    split_data = pickle.load(f)
                    processed_ids = {item["q_id"] for item in split_data}
                    logger.info(f"Found {len(processed_ids)} existing processed questions in {split}")
            except Exception as e:
                logger.warning(f"Error reading existing data: {str(e)}. Starting fresh.")
                if os.path.exists(output_file):
                    backup_file = f"{output_file}.bak"
                    logger.info(f"Creating backup of corrupted file at {backup_file}")
                    os.rename(output_file, backup_file)
                    split_data = []
        
        # Process data
        for data in tqdm(ds[split], desc=f"Processing {split} split"):
            if data["question_id"] in processed_ids:
                continue
                
            try:
                corpus_wav = data["question_audio"]["array"]

                # Apply VAD if needed
                if do_vad:
                    corpus_wav = apply_vad(
                        corpus_wav, vad_model, get_speech_timestamps, use_cuda
                    )

                # Extract features
                features = extract_features(corpus_wav, feature_extractor, model, use_cuda)
                
                # Add to split data
                split_data.append({
                    "q_id": data["question_id"],
                    "doc_id": data["document_id"],
                    "feature": features
                })
                
                # Save periodically to avoid data loss
                if len(split_data) % 1000 == 0:
                    with open(output_file, 'wb') as f:
                        pickle.dump(split_data, f)
                        
            except Exception as e:
                logger.error(f"Error processing question {data['question_id']}: {str(e)}")
                continue
        
        # Save final data
        with open(output_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        logger.info(f"Successfully processed {split} split with {len(split_data)} items")


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

    # Save all data in a single consolidated dictionary
    logger.info("Consolidating data into a single file")
    all_data = {"corpus": [], "train": [], "validation": [], "test": [], "verified_test": []}
    
    # Load corpus data
    corpus_path = f"{args.output_dir}/slue_sqa5_corpus.pkl"
    if os.path.exists(corpus_path):
        with open(corpus_path, 'rb') as f:
            all_data["corpus"] = pickle.load(f)
    
    # Load split data
    for split in splits:
        split_path = f"{args.output_dir}/{split}.pkl"
        if os.path.exists(split_path):
            with open(split_path, 'rb') as f:
                all_data[split] = pickle.load(f)
    
    # Save consolidated data
    consolidated_path = f"{args.output_dir}/slue_sqa5_all_features.pkl"
    with open(consolidated_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    logger.info(f"All processing completed successfully. Consolidated data saved to {consolidated_path}")


if __name__ == "__main__":
    main()

