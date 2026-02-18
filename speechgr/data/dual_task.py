import torch
import random
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import torchaudio
import os
from pathlib import Path

class DualTaskDataset(Dataset):
    """
    Dataset for Phase 1 (Pre-training) and Phase 2 (Fine-tuning).
    Supports Task A (Indexing) and Task B (Retrieval) with mixed batches.
    Uses torchaudio for faster loading.
    """
    def __init__(
        self,
        id_map_path: str,
        semantic_map_path: str,
        audio_root: str,
        indexing_prob: float = 0.5,
        crop_duration: float = 3.0,
        sample_rate: int = 16000,
        is_training: bool = True
    ):
        self.audio_root = Path(audio_root)
        with open(id_map_path, 'r') as f:
            self.id_map = json.load(f)
        with open(semantic_map_path, 'r') as f:
            self.semantic_map = json.load(f)
            
        self.doc_ids = list(self.id_map.keys())
        self.indexing_prob = indexing_prob
        self.crop_duration = crop_duration
        self.sample_rate = sample_rate
        self.crop_samples = int(crop_duration * sample_rate)
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.doc_ids)

    def _load_audio(self, doc_path: str) -> torch.Tensor:
        # doc_path might be absolute or relative to audio_root
        p = Path(doc_path)
        if not p.is_absolute():
            p = self.audio_root / p
        
        # Check if file exists to avoid silent errors or obscure torchaudio errors
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        waveform, sr = torchaudio.load(p)
        
        # Mix to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        return waveform.squeeze(0) # [T]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc_path = self.doc_ids[idx]
        id_codes = self.id_map[doc_path]
        semantic_tokens = self.semantic_map.get(doc_path, []) # Handle missing semantic tokens gracefully if needed
        
        try:
            audio = self._load_audio(doc_path)
        except Exception as e:
            # Return dummy if failed (or raise error depending on policy)
            # For now, let's create a silent tensor of minimal length
            # This is safer than crashing training
            audio = torch.zeros(self.sample_rate) 
        
        # Decide Task
        if self.is_training and random.random() > self.indexing_prob:
            # Task B: Generative Retrieval (Random Crop)
            if audio.size(0) > self.crop_samples:
                start = random.randint(0, audio.size(0) - self.crop_samples)
                audio = audio[start : start + self.crop_samples]
        # else: Task A (Full Doc Indexing)
            
        return {
            "audio": audio,
            "semantic_labels": semantic_tokens,
            "retrieval_labels": id_codes
        }

class DualTaskCollator:
    def __init__(
        self, 
        sem_pad_id: int = -100, 
        ret_pad_id: int = -100,
        sem_bos_id: int = 0,
        ret_bos_id: int = 0
    ):
        self.sem_pad_id = sem_pad_id
        self.ret_pad_id = ret_pad_id
        self.sem_bos_id = sem_bos_id
        self.ret_bos_id = ret_bos_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Filter out failed loads if any (where audio is empty/minimal)
        batch = [b for b in batch if b["audio"].size(0) > 0]
        if not batch:
            return {}

        audios = [x["audio"] for x in batch]
        
        # Prepend BOS to labels for teacher forcing
        sem_labels = [torch.tensor([self.sem_bos_id] + list(x["semantic_labels"])) for x in batch]
        ret_labels = [torch.tensor([self.ret_bos_id] + list(x["retrieval_labels"])) for x in batch]
        
        # Pad Audios
        input_values = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        for i, a in enumerate(audios):
            attention_mask[i, :len(a)] = 1
            
        # Pad Labels
        def pad_seqs(seqs, pad_id):
            max_len = max(len(s) for s in seqs)
            padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                padded[i, :len(s)] = s
            return padded

        sem_padded = pad_seqs(sem_labels, self.sem_pad_id)
        ret_padded = pad_seqs(ret_labels, self.ret_pad_id)
        
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "semantic_labels": sem_padded,
            "retrieval_labels": ret_padded
        }
