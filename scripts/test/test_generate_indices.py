import os
import shutil
import subprocess
import numpy as np
import soundfile as sf
import sys

def create_dummy_audio_dir(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    
    # Create a few dummy audio files
    print(f"Creating dummy audio in {root}...")
    for i in range(5):
        # 1 second of random noise
        audio = np.random.uniform(-1, 1, 16000)
        sf.write(os.path.join(root, f"test_{i}.wav"), audio, 16000)

def test_generate_indices():
    audio_root = "tests/dummy_audio_indices"
    output_dir = "tests/output_indices"
    
    create_dummy_audio_dir(audio_root)
    
    # Command to run generate_indices.py with --train_kmeans
    # Using tiny model for speed
    cmd = [
        "uv", "run", "python", "scripts/phase0_prep/generate_indices.py",
        "--audio_root", audio_root,
        "--output_dir", output_dir,
        "--ssl_model", "hf-internal-testing/tiny-random-wavlm",
        "--ssl_layer", "2",
        "--train_kmeans",
        "--kmeans_k", "10"
    ]
    
    print("Running generate_indices.py (K-Means Train Mode)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running script:")
        print(result.stderr)
        sys.exit(1)
        
    print("Output:")
    print(result.stdout)
    
    # Check if model saved
    model_path = os.path.join(output_dir, "kmeans_model.pkl")
    if os.path.exists(model_path):
        print(f"SUCCESS: K-Means model saved at {model_path}")
    else:
        print(f"FAILURE: K-Means model not found at {model_path}")
        sys.exit(1)

    # Cleanup
    shutil.rmtree(audio_root)
    shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_generate_indices()
