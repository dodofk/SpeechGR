import torch
import numpy as np
import json
import os
from speechgr.models.unity import UnitySpeechModel
from speechgr.data.dual_task import DualTaskDataset, DualTaskCollator
from torch.utils.data import DataLoader

def create_dummy_data():
    """Create dummy maps and audio for smoke testing."""
    os.makedirs("tests/dummy_audio", exist_ok=True)
    
    # Create 2 dummy audio files
    audio1 = np.random.randn(16000 * 5).astype(np.float32) # 5 seconds
    audio2 = np.random.randn(16000 * 4).astype(np.float32) # 4 seconds
    
    import soundfile as sf
    sf.write("tests/dummy_audio/doc1.wav", audio1, 16000)
    sf.write("tests/dummy_audio/doc2.wav", audio2, 16000)
    
    id_map = {
        "tests/dummy_audio/doc1.wav": [10, 20, 30, 40, 50, 60, 70, 80],
        "tests/dummy_audio/doc2.wav": [11, 21, 31, 41, 51, 61, 71, 81]
    }
    
    semantic_map = {
        "tests/dummy_audio/doc1.wav": [100, 101, 102],
        "tests/dummy_audio/doc2.wav": [200, 201]
    }
    
    with open("tests/dummy_id_map.json", "w") as f:
        json.dump(id_map, f)
    with open("tests/dummy_semantic_map.json", "w") as f:
        json.dump(semantic_map, f)

def smoke_test():
    print("--- Starting Smoke Test ---")
    create_dummy_data()
    
    device = torch.device("cpu") # Use CPU for smoke test
    
    # 1. Test Dataset and Collator
    print("Testing Dataset and Collator...")
    dataset = DualTaskDataset(
        id_map_path="tests/dummy_id_map.json",
        semantic_map_path="tests/dummy_semantic_map.json",
        audio_root=".",
        indexing_prob=0.5,
        crop_duration=1.0
    )
    
    collator = DualTaskCollator()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Audio shape: {batch['input_values'].shape}")
    print(f"Semantic labels shape: {batch['semantic_labels'].shape}")
    print(f"Retrieval labels shape: {batch['retrieval_labels'].shape}")
    
    # 2. Test Model Forward Pass
    print("\nTesting Model Forward Pass...")
    model = UnitySpeechModel(
        ssl_model_name="hf-internal-testing/tiny-random-wavlm", # Use tiny model for speed
        ssl_layer=2,
        semantic_vocab_size=5000,
        retrieval_vocab_size=256,
        d_model=32, # Small dim
        num_heads=4,
        num_layers=2
    ).to(device)
    
    sem_logits, ret_logits = model(
        input_values=batch["input_values"],
        semantic_labels=batch["semantic_labels"],
        retrieval_labels=batch["retrieval_labels"],
        attention_mask=batch["attention_mask"]
    )
    
    print(f"Semantic logits shape: {sem_logits.shape}")
    print(f"Retrieval logits shape: {ret_logits.shape}")
    
    # 3. Test Backward Pass
    print("\nTesting Backward Pass...")
    loss = sem_logits.sum() + ret_logits.sum()
    loss.backward()
    print("Backward pass successful.")
    
    # 4. Test Generation
    print("\nTesting Generation...")
    sem_gen, ret_gen = model.generate(
        input_values=batch["input_values"][:1],
        max_sem_len=5,
        max_ret_len=8
    )
    print(f"Generated Semantic shape: {sem_gen.shape}")
    print(f"Generated Retrieval shape: {ret_gen.shape}")
    
    print("\n--- Smoke Test Passed Successfully! ---")

if __name__ == "__main__":
    smoke_test()
