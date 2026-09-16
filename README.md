# Neural Style Transfer (AdaIN)

An end-to-end deep learning web application for real-time **Arbitrary Neural Style Transfer** powered by **Adaptive Instance Normalization (AdaIN)**, PyTorch, and Flask. The system combines the semantic content of one image with the artistic color palette, texture, and brushstrokes of another at adjustable style blending strengths.
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge)](https://neural-style-transfer-mfpl.onrender.com)

# 🎨 Neural Style Transfer

[🚀 Live Demo](https://neural-style-transfer-mfpl.onrender.com)
---

## Overview

Traditional neural style transfer methods (such as Gatys et al.) treat style transfer as an optimization problem, requiring hundreds of gradient descent iterations taking minutes per image. 

This application uses the **AdaIN (Adaptive Instance Normalization)** feedforward architecture (Huang & Belongie):
- **VGG-19 Encoder**: A fixed, pretrained convolutional network extracts multi-scale feature maps from content and style inputs up to layer `relu4_1`.
- **AdaIN Layer**: Normalizes the channel-wise mean and variance of content features to align with those of the style features in latent space without learnable parameters.
- **Trained Decoder**: A symmetrical convolutional decoder trained to invert the AdaIN-modified feature representation back into a coherent, high-resolution RGB image.
- **Flask Web Interface**: A responsive web frontend with an interactive cyberpunk-themed UI, live alpha strength slider (0.0 to 1.0), real-time previews, preset examples, and a lightweight `/health` monitoring endpoint.

---

## Architecture

```
Content Image (I_c) ───┐
                       ├──> [ VGG-19 Encoder ] ──> Content Features (f_c) ─┐
Style Image (I_s)   ───┘    (Pretrained)        ──> Style Features (f_s)   ─┼─> [ AdaIN Layer ]
                                                                            │          │
                                                                            │     Normalized
                                                                            │      Features (t)
                                                                            │          │
                                                                            └── [ Alpha Blending ]
                                                                                       │
                                                                               t' = α*t + (1-α)*f_c
                                                                                       │
                                                                                [ CNN Decoder ]
                                                                                       │
                                                                                Stylized Image
```

Mathematical formulation of AdaIN:
$$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$

where $x$ is the content feature map and $y$ is the style feature map.

---

## Features

- **Content & Style Image Upload**: Supports `.jpg`, `.jpeg`, `.png`, and `.webp` formats with PIL integrity verification.
- **Adjustable Style Strength (Alpha Slider)**: Control the interpolation between content geometry ($\alpha = 0.0$) and full artistic abstraction ($\alpha = 1.0$).
- **Pretrained Deep Learning Models**: Includes `vgg_normalised.pth` (VGG encoder) and `decoder_final.pth` (trained decoder checkpoint) ready for instant inference.
- **Hardware Acceleration with CPU Fallback**: Automatically leverages NVIDIA CUDA GPUs when available and seamlessly falls back to CPU execution.
- **Interactive Quick-Select Presets**: One-click example selection using curated content and style pairs.
- **Responsive Dark Web UI**: Features glassmorphic cards, dynamic particle neural canvas, inference progress spinner, and one-click image download.
- **Production Ready**: Configured for Docker, Gunicorn WSGI server, and Hugging Face Spaces deployment with a dedicated `/health` check.

---

## Local Setup (Windows)

### 1. Prerequisites
- **Python**: 3.10 or 3.11 (Python 3.11.9 recommended)
- **Git**: Installed and configured on your path

### 2. Clone or Navigate to Project
```powershell
cd NST_Code
```

### 3. Create & Activate Virtual Environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 4. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 5. Verify Environment & Hardware
```powershell
python check_torch.py
```

### 6. Run the Application
```powershell
python app.py
```
Open your browser at:
```
http://127.0.0.1:7860
```

---

## GitHub Setup

Follow these steps to initialize and push this project to your GitHub repository:

```bash
# 1. Initialize Git repository (if not already done)
git init

# 2. Stage all project files (ignoring caches, virtual environments, and temporary files)
git add .

# 3. Create initial commit
git commit -m "Initial Neural Style Transfer project"

# 4. Set default branch to main
git branch -M main

# 5. Connect to your GitHub repository
# Replace YOUR_GITHUB_USERNAME and YOUR_REPOSITORY_NAME with your actual details:
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git

# 6. Push to GitHub
git push -u origin main
```

---

## Hugging Face Spaces Deployment

Deploy this application directly to Hugging Face Spaces using the official Docker SDK:

### Step 1: Create a New Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Navigate to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
3. Set your Space details:
   - **Space Name**: e.g., `neural-style-transfer`
   - **License**: `mit` or `apache-2.0`
   - **Select the Space SDK**: Choose **Docker** -> **Blank**.
   - **Space Hardware**: Choose **CPU basic (free)** or any GPU tier.

### Step 2: Push Repository to Hugging Face
You can deploy either by syncing directly with GitHub or by pushing via Git:

```bash
# Add Hugging Face Space as a remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME

# Push to Hugging Face
git push --force space main
```

### Step 3: Automatic Build & Launch
- Hugging Face will detect the `Dockerfile`, install dependencies, copy model checkpoints, and launch Gunicorn on port `7860`.
- The live application will be available at:
  `https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME`

---

## Limitations & Notes

- **CPU Inference Speed**: On standard free-tier CPU instances (2 vCPUs), feedforward inference typically takes between 2 to 5 seconds per 512px image. On GPU-enabled environments with CUDA, inference executes in under 100 milliseconds.
- **Ephemeral Storage**: Uploaded files and generated images stored in `static/uploads/` are temporary and will be cleared when the container restarts.
