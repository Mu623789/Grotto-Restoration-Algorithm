# Cave Restoration Algorithm (Stone Grotto Image Restoration)

This repository provides a practical, reproducible pipeline for restoring stone grotto mural/statue images captured in real-world conditions, where images often suffer from **low resolution**, **weathering degradation**, **noise/compression artifacts**, and **missing fine details**.

It focuses on:
- **Super-Resolution (SR)** for real captured grotto images  
- **Texture/detail enhancement** (high-frequency patterns & structural edges)
- **Memory-friendly tile inference** for very large images (adjustable tile / overlap)

The current implementation includes two SR routes: **VARSR** (VQ-based autoregressive SR) and **SwinIR** (Transformer-based SR). Choose based on your data characteristics and deployment needs.

---

## Key Features

- **Real-world degradation friendly**: robust to blur/noise/compression/illumination issues
- **Detail-preserving**: improves fine textures, cracks, and edge continuity
- **Engineering-ready**: supports large-image tile inference to reduce VRAM usage
- **semi-supervised training**: mix **5% labeled + 95% unlabeled** with pseudo labels

---

## Method Overview

### 1) VARSR Route (VQ + Autoregressive Generation)
- Encode images into discrete tokens with VQ-VAE
- Autoregressively generate high-resolution token sequences
- Pros: strong texture synthesis ability  

### 2) SwinIR Route (Transformer SR)
- Image restoration model based on Swin Transformer
- Strong baseline for Real-World SR (e.g., BSRGAN-like degradations)
- Pros: fast inference, easier deployment, mature tile inference

## Environment

Recommended: Linux + NVIDIA CUDA. (GPU memory >= 48GiB)

PyTorch  2.8.0

Python  3.12(ubuntu22.04)

CUDA  12.8

### Weights
Download `VQVAE.pth` from **Releases** and put it under `checkpoints/VQVAE.pth`.

Download `VARSR.pth` from **Releases** and put it under `checkpoints/VARSR.pth`.
