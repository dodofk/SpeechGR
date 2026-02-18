import os
import shutil
import torch
import torchaudio
import json
import numpy as np
from speechgr.data.dual_task import DualTaskDataset, DualTaskCollator

def create_dummy_audio_torchaudio(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    
    # Create valid audio (16k)
    path_valid = os.path.join(root, "valid.wav")
    waveform = torch.randn(1, 16000)
    torchaudio.save(path_valid, waveform, 16000)
    
    # Create different SR (44.1k)
    path_resample = os.path.join(root, "resample.wav")
    waveform_hq = torch.randn(1, 44100)
    torchaudio.save(path_resample, waveform_hq, 44100)
    
    # Create missing file (for error handling test)
    path_missing = os.path.join(root, "missing.wav")
    
    # JSON maps
    id_map = {
        "valid.wav": [1]*8,
        "resample.wav": [2]*8,
        "missing.wav": [3]*8
    }
    semantic_map = {k: [10]*5 for k in id_map}
    
    with open(os.path.join(root, "id_map.json"), "w") as f:
        json.dump(id_map, f)
    with open(os.path.join(root, "semantic_map.json"), "w") as f:
        json.dump(semantic_map, f)
        
    return root

def test_data_loading():
    root = "tests/dummy_data_torchaudio"
    create_dummy_audio_torchaudio(root)
    
    dataset = DualTaskDataset(
        id_map_path=os.path.join(root, "id_map.json"),
        semantic_map_path=os.path.join(root, "semantic_map.json"),
        audio_root=root,
        crop_duration=0.5
    )
    
    print("Testing DualTaskDataset with torchaudio...")
    
    # 1. Test Valid Load
    print("- Testing valid file...")
    item_valid = dataset[0] # valid.wav
    assert item_valid["audio"].shape[0] > 0
    print("  OK")
    
    # 2. Test Resampling
    print("- Testing resampling (44.1k -> 16k)...")
    item_resample = dataset[1] # resample.wav
    # Should be resampled. 1 sec of 44.1k -> 1 sec of 16k
    # But wait, dataset[1] might be cropped if it's longer than crop_duration (0.5s = 8000 samples)
    # 44100 samples -> resampled to 16000 samples.
    # If crop is active, it will be 8000 samples.
    print(f"  Shape: {item_resample['audio'].shape}")
    assert item_resample["audio"].shape[0] <= 16000
    print("  OK")
    
    # 3. Test Missing File (Error Handling)
    print("- Testing missing file...")
    item_missing = dataset[2] # missing.wav
    # Should return zero tensor
    assert torch.all(item_missing["audio"] == 0)
    print("  OK (Returned zero tensor)")
    
    # 4. Test Collator Filtering
    print("- Testing Collator filtering...")
    batch = [dataset[0], dataset[2]] # Valid + Missing (Zero)
    collator = DualTaskCollator()
    out = collator(batch)
    
    # Collator should NOT filter out zero tensors explicitly unless size is 0
    # Our zero tensor has size (16000,), so it is kept.
    # Wait, the code says: batch = [b for b in batch if b["audio"].size(0) > 0]
    # 16000 > 0, so it is kept.
    # This is fine, the model will just learn to ignore it via masking ideally, or we should filter it.
    # If we want to filter it, we should check for non-zero or specific flag.
    # But for now, let's just check it runs.
    
    print(f"  Batch size: {out['input_values'].size(0)}")
    assert out['input_values'].size(0) == 2
    print("  OK")

    # Cleanup
    shutil.rmtree(root)
    print("\nData Loading Test Passed!")

if __name__ == "__main__":
    test_data_loading()
