import os
import shutil
import subprocess
import numpy as np
import json
import soundfile as sf
import sys

def create_dummy_data(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    
    # Create dummy audio
    audio_paths = []
    for i in range(5):
        path = os.path.join(root, f"doc_{i}.wav")
        # 1 second of random noise
        audio = np.random.uniform(-1, 1, 16000)
        sf.write(path, audio, 16000)
        audio_paths.append(path)
        
    # Create maps
    # Paths in id_map should be relative to audio_root
    id_map = {f"doc_{i}.wav": [1]*8 for i in range(5)}
    semantic_map = {f"doc_{i}.wav": [10]*5 for i in range(5)}
    
    with open(os.path.join(root, "id_map.json"), "w") as f:
        json.dump(id_map, f)
    with open(os.path.join(root, "semantic_map.json"), "w") as f:
        json.dump(semantic_map, f)
        
    return root

def test_train_unity_hydra():
    root = "tests/dummy_train_hydra"
    create_dummy_data(root)
    
    id_map_path = os.path.abspath(os.path.join(root, "id_map.json"))
    semantic_map_path = os.path.abspath(os.path.join(root, "semantic_map.json"))
    audio_root = os.path.abspath(root)
    
    # Command to run train_unity.py with hydra overrides
    cmd = [
        "uv", "run", "python", "scripts/phase1_train/train_unity.py",
        f"data.id_map={id_map_path}",
        f"data.semantic_map={semantic_map_path}",
        f"data.audio_root={audio_root}",
        "model.ssl_model_name=hf-internal-testing/tiny-random-wavlm",
        "model.ssl_layer=2",
        "model.d_model=32",
        "model.num_heads=4",
        "model.num_layers=2",
        "training.epochs=1",
        "training.save_steps=2",
        "data.batch_size=2",
        "data.num_workers=0",
        "logging.mode=disabled",
        "hydra.run.dir=tests/output_train_hydra"
    ]
    
    print("Running train_unity.py with Hydra and Fabric...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running script:")
        print(result.stderr)
        sys.exit(1)
        
    print("Output tail:")
    print("\n".join(result.stdout.splitlines()[-20:]))
    
    # Check if checkpoint saved
    checkpoint_path = "tests/output_train_hydra/checkpoint_epoch_0.pt"
    if os.path.exists(checkpoint_path):
        print(f"SUCCESS: Checkpoint saved at {checkpoint_path}")
    else:
        print(f"FAILURE: Checkpoint not found at {checkpoint_path}")
        # list directory to debug
        print("Directory content:")
        subprocess.run(["ls", "-R", "tests/output_train_hydra"])
        sys.exit(1)

    # Cleanup
    shutil.rmtree(root)
    shutil.rmtree("tests/output_train_hydra")

if __name__ == "__main__":
    test_train_unity_hydra()
