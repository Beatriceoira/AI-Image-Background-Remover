#!/usr/bin/env python3
"""
Download the U²-Net model weights for background removal.
"""
import os
import hashlib
import requests
from tqdm import tqdm

# Model configuration
MODEL_NAME = "u2net"
MODEL_URL = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
EXPECTED_MD5 = "aa29ec80d58966e8e0f8e0e9a8e4c6f5"  # Example MD5, please verify

def download_model():
    """Download the model weights if not already present."""
    # Define the model directory and path
    model_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models", "weights")
    model_path = os.path.join(model_dir, f"{MODEL_NAME}.pth")

    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file already exists
    if os.path.exists(model_path):
        print(f"Model weights already exist at {model_path}")
        # Optionally, verify the checksum
        if EXPECTED_MD5:
            print("Verifying checksum...")
            md5_hash = hashlib.md5()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            if md5_hash.hexdigest() == EXPECTED_MD5:
                print("Checksum verified.")
                return
            else:
                print("Checksum mismatch. Redownloading...")

    # Download the model
    print(f"Downloading {MODEL_NAME} weights from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(model_path, 'wb') as file, tqdm(
            desc=MODEL_NAME,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    # Verify the checksum after download
    if EXPECTED_MD5:
        print("Verifying checksum...")
        md5_hash = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        if md5_hash.hexdigest() == EXPECTED_MD5:
            print("Download successful and checksum verified.")
        else:
            print("Warning: Downloaded file checksum does not match expected value.")
            print("Please check the model file manually.")
    else:
        print("Download completed. No checksum verification performed.")

if __name__ == "__main__":
    download_model()