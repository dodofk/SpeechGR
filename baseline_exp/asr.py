import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_metric
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model and processor from Hugging Face

model_name_or_path = "openai/whisper-large-v3"

processor = WhisperProcessor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

ds = load_dataset("asapp/slue-phase-2", "sqa5")

splits = ["train", "validation", "test", "verified_test"]

corpus_set = set()
corpus_id, corpus_gt, corpus_asr = [], []

for split in splits:
    print(f"Processing {split} split corpus part")
    for data in ds[split]:
        if data["document_id"] in corpus_set:
            continue
        corpus_set.add(data["document_id"])
        
        wav, sr = data["document_audio"]["array"], data["document_audio"]["sampling_rate"]
        
        inputs = processor(
            audio=wav,
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        outputs = model.generate(**inputs, max_new_tokens=448)
        
        corpus_asr.append(processor.decode(outputs[0], skip_special_tokens=True))
        corpus_gt.append(data["normalized_document_text"])
        corpus_id.append(data["document_id"])
        
# Calculate wer for corpus
wer_metric = load_metric("wer")
wer = wer_metric.compute(predictions=corpus_asr, references=corpus_gt)


# test set
test_ds = ds["test"]
query_id, query_gt, query_asr = [], []

for data in test_ds:
    query_id.append(data["question_id"])
    query_gt.append(data["raw_question_text"])
    query_asr.append(data["asr_transcription"])
    
# Calculate wer for query
wer_metric = load_metric("wer")
wer = wer_metric.compute(predictions=query_asr, references=query_gt)

print(f"Query WER: {wer}")

# save the results
with open(f"asr_results_{model_name_or_path}.json", "w") as f:
    json.dump(
        {
            "corpus_id": corpus_id,
            "corpus_gt": corpus_gt,
            "corpus_asr": corpus_asr,
            "query_id": query_id,
            "query_gt": query_gt,
            "query_asr": query_asr
        }, f)

print(f"Corpus WER: {wer}")





# print(len(corpus))
