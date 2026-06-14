#!/usr/bin/env python3
"""Pre-download the ONNX embedder model to HF_HOME cache.

Called during Docker build so the container never needs internet access.
The embedder is instantiated once, runs a dummy encode to populate the
HuggingFace cache, then exits.  At runtime, ``get_embedder()`` loads
from cache with zero network activity.

Usage:
    python scripts/download_embedder.py
    # → ONNX embedder ready, dim=384
"""

from embed_anything import Dtype, EmbeddingModel, ONNXModel, TextEmbedConfig, WhichModel


def main():
    import os, shutil, subprocess

    hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    print(f"HF_HOME={hf_home}")
    print("Downloading ONNX embedder...", flush=True)

    model = EmbeddingModel.from_pretrained_onnx(
        WhichModel.Bert,
        model_name=ONNXModel.AllMiniLML6V2Q,
        dtype=Dtype.Q4F16,
    )
    config = TextEmbedConfig(chunk_size=256, batch_size=32)
    dummy = model.embed_query(["preload"], config=config)
    dim = len(dummy[0].embedding)

    # Find and copy cached files to a stable path we can COPY into the image
    cache_dirs = [
        "/root/.cache/huggingface",
        "/home/appuser/.cache/huggingface",
        hf_home,
    ]
    for src in cache_dirs:
        if os.path.exists(src) and os.listdir(src):
            dst = "/embedder-cache"
            print(f"Copying HF cache from {src} to {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            files = subprocess.run(["find", dst, "-type", "f"], capture_output=True, text=True, timeout=10)
            print(f"Copied {len(files.stdout.strip().split(chr(10)))} files to {dst}")
            break
    else:
        print("WARNING: No HF cache directory found!")

    print(f"ONNX embedder ready, dim={dim}")


if __name__ == "__main__":
    main()
