import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create a manifest file for audio datasets.")
    parser.add_argument("--audio_root", type=str, required=True, help="Path to the audio dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the manifest.txt")
    parser.add_argument("--extension", type=str, default="flac", help="Audio extension (flac, wav)")
    
    args = parser.parse_args()
    
    audio_files = list(Path(args.audio_root).rglob(f"*.{args.extension}"))
    print(f"Found {len(audio_files)} files. Writing to {args.output_path}...")
    
    with open(args.output_path, "w") as f:
        for p in audio_files:
            f.write(f"{os.path.abspath(p)}
")
            
    print("Done.")

if __name__ == "__main__":
    main()
