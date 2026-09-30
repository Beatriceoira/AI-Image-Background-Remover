#!/usr/bin/env python3
"""
Benchmark script for MatteForge background removal.
Measures preprocessing, inference, matting, and encoding times.

Usage:
    python scripts/benchmark.py [--image PATH] [--iterations N] [--device DEVICE]

Examples:
    python scripts/benchmark.py
    python scripts/benchmark.py --image test.jpg --iterations 5
    python scripts/benchmark.py --device cpu
"""
import argparse
import io
import os
import sys
import time
import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.inference import InferenceService
from backend.app.services.preprocessing import PreprocessingService
from backend.app.services.segmentation import SegmentationService
from backend.app.services.matting import MattingService
from backend.app.services.postprocessing import PostprocessingService
from backend.app.core.config import Config


def create_test_image(size=(512, 512)):
    """Create a synthetic test image with gradients."""
    img = Image.new("RGB", size)
    pixels = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(255 * x / size[0])
            g = int(255 * y / size[1])
            b = int(128 * (1 + np.sin(x / 20) * np.cos(y / 20)))
            pixels[x, y] = (r % 256, g % 256, b % 256)
    return np.array(img)


def benchmark_preprocess(service, image, iterations=3):
    """Benchmark preprocessing."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = service.preprocess(image)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return min(times)


def benchmark_inference(inference, tensor, iterations=3):
    """Benchmark inference."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.inference_mode():
            _ = inference.predict(tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return min(times)


def benchmark_segmentation(seg_service, image, tensor, iterations=3):
    """Benchmark full segmentation."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = seg_service.segment(image, tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return min(times)


def benchmark_matting(mat_service, image, mask, iterations=3):
    """Benchmark matting."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = mat_service.matte(image, mask, background_mode="transparent")
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return min(times)


def benchmark_encoding(image_array, format="PNG", iterations=3):
    """Benchmark image encoding."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        buf = io.BytesIO()
        pil_img = Image.fromarray(image_array, mode="RGBA")
        pil_img.save(buf, format=format)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return min(times)


def print_section(title):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MatteForge background removal")
    parser.add_argument("--image", type=str, help="Path to test image (default: synthetic)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations (default: 5)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device to use (default: auto)")
    parser.add_argument("--output-format", type=str, default="PNG", choices=["PNG", "JPEG", "WEBP"],
                        help="Output format for encoding benchmark (default: PNG)")
    args = parser.parse_args()

    print_section("MatteForge Benchmark")

    # Load or create test image
    if args.image and os.path.exists(args.image):
        print(f"\nLoading test image: {args.image}")
        img = Image.open(args.image)
        img = img.convert("RGB")
        image = np.array(img)
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")
    else:
        print("\nCreating synthetic test image (512x512)...")
        image = create_test_image((512, 512))
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")

    # Setup
    print("\nSetting up services...")
    inference = InferenceService()
    preprocess_service = PreprocessingService()
    seg_service = SegmentationService()
    mat_service = MattingService()
    post_service = PostprocessingService()

    # Load model
    print("Loading model...")
    try:
        inference.load_model()
    except FileNotFoundError:
        print("ERROR: Model weights not found. Run: python scripts/download_model.py")
        sys.exit(1)

    device = inference.device
    print(f"  Device: {device}")
    print(f"  Model params: {sum(p.numel() for p in inference.model.model.parameters()):,}")

    # Preprocess
    print("\nRunning preprocessing...")
    tensor = preprocess_service.preprocess(image)
    print(f"  Input tensor: {tuple(tensor.shape)}")

    # Warmup
    print("\nWarming up model...")
    _ = inference.predict(tensor)

    # Benchmark preprocessing
    print("\nBenchmarking preprocessing...")
    preprocess_time = benchmark_preprocess(preprocess_service, image, args.iterations)
    print(f"  Time: {preprocess_time:.2f} ms")

    # Benchmark segmentation (includes inference + resize)
    print("\nBenchmarking segmentation...")
    seg_time = benchmark_segmentation(seg_service, image, tensor, args.iterations)
    print(f"  Time: {seg_time:.2f} ms")

    # Break down inference time
    print("\nBenchmarking inference only...")
    inference_time = benchmark_inference(inference, tensor, args.iterations)
    print(f"  Inference time: {inference_time:.2f} ms")

    # Benchmark matting
    print("\nBenchmarking matting...")
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) // 2
    mat_time = benchmark_matting(mat_service, image, mask, args.iterations)
    print(f"  Time: {mat_time:.2f} ms")

    # Benchmark encoding
    print("\nBenchmarking encoding...")
    rgba = np.dstack([image, mask]).astype(np.uint8)
    enc_time = benchmark_encoding(rgba, args.output_format, args.iterations)
    print(f"  Time: {enc_time:.2f} ms")

    # Total time
    total_time = preprocess_time + seg_time + mat_time + enc_time

    # Memory usage (if available)
    memory_info = ""
    if inference.device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        memory_info = f"  GPU memory allocated: {mem_allocated:.1f} MB"
        memory_info += f"\n  GPU memory reserved: {mem_reserved:.1f} MB"

    # Summary
    print_section("Results Summary")
    print(f"""
  Image:       {image.shape[1]}x{image.shape[0]}
  Device:      {device}
  Model:       U²-Net ({sum(p.numel() for p in inference.model.model.parameters()):,} params)

  Preprocess:  {preprocess_time:>8.2f} ms
  Inference:   {inference_time:>8.2f} ms
  Segmentation:{seg_time:>8.2f} ms
  Matting:     {mat_time:>8.2f} ms
  Encoding:    {enc_time:>8.2f} ms
  ──────────────────────────
  Total:       {total_time:>8.2f} ms
  FPS:         {1000/total_time:>8.2f}
{memory_info}
""")

    # Cleanup
    inference.unload()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
