import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
from typing import List, Dict
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class WhisperASR:
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Whisper model: {model_name} on {self.device}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Default transcription config
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")

    def transcribe_batch(self, audios: List[np.ndarray], sampling_rate: int = 16000) -> List[str]:
        """
        Transcribe a batch of audio arrays.
        """
        inputs = self.processor(
            audios, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features, 
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=448
            )
            
        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcriptions

    def transcribe_dataset(self, dataset, audio_column: str = "audio", text_column: str = "text", batch_size: int = 4) -> Dict[str, str]:
        """
        Transcribe all items in a dataset. Expects dataset items to have 'document_id' or similar if used for corpus.
        """
        results = {}
        for i in tqdm(range(0, len(dataset), batch_size), desc="Transcribing"):
            batch = dataset[i : i + batch_size]
            audios = [x["array"] for x in batch[audio_column]]
            sr = batch[audio_column][0]["sampling_rate"]
            
            transcripts = self.transcribe_batch(audios, sampling_rate=sr)
            
            # Use IDs if available, else index
            for j, transcript in enumerate(transcripts):
                item = batch[j] if isinstance(batch, list) else {k: v[j] for k, v in batch.items()}
                idx = item.get("document_id", item.get("question_id", str(i + j)))
                results[idx] = transcript
                
        return results
