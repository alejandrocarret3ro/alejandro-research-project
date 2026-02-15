# Alejandro Research Project

A framework for solving inverse problems in video using learned denoiser priors. The core idea is to train a blind video denoiser and then leverage the prior it embeds to guide iterative reconstruction for arbitrary inverse problems, given a known measurement (forward) operator.

> **Status**: The denoiser module is implemented and under active training. The inverse problem framework is in development.

## Project Structure

```
alejandro-research-project/
├── denoiser/                          # Blind video denoiser (UNet)
│   ├── models/
│   │   └── unet_denoiser.py           #   UNet architecture (BlindVideoDenoiserUNet)
│   ├── data/
│   │   └── dataloader.py              #   BlindDenoiseDataset for DAVIS frames
│   ├── training/
│   │   └── unet_denoiser_training.py  #   Training loop, data splits, logging
│   ├── inference/
│   │   └── model_testing.py           #   ModelTester, PSNR/SSIM metrics, visualization
│   └── notebooks/
│       ├── Unet_denoiserv1.ipynb      #   Training notebook (Google Colab)
│       └── denoiser_testingv1.ipynb   #   Testing/evaluation notebook
│
├── framework/                         # Inverse problem solver (coming soon)
│   └── __init__.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Denoiser

### Overview

A bias-free UNet-based blind video denoiser trained to remove Gaussian noise at extreme levels (σ = 5–250) from video frames. The model takes triplets of consecutive frames (previous, current, next) as temporal context and outputs a denoised center frame.

### Architecture

- **Bias-free UNet** with 3 or 4 downsampling stages and skip connections
- **Input**: 3 consecutive RGB frames concatenated → (B, 9, H, W)
- **Output**: Denoised center frame → (B, 3, H, W)
- No bias terms in convolutions (essential for blind denoising)
- Mixed-precision (FP16) training with `torch.compile` support

### Training

Trained on the [DAVIS 2017](https://davischallenge.org/davis2017/code.html) dataset (video object segmentation frames used as clean ground truth). The dataset is not included in this repository.

```python
from denoiser.models import BlindVideoDenoiserUNet
from denoiser.training import create_train_val_split, create_data_loaders, train

# Create datasets (splits by video, not by frame)
train_dataset, val_dataset = create_train_val_split(
    '/path/to/DAVIS/JPEGImages',
    val_split=0.2,
    resize_to=(256, 256),
    use_fp16=True
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_dataset, val_dataset,
    batch_size=16, num_workers=2
)

# Train
model = BlindVideoDenoiserUNet(in_channels=9, out_channels=3, base_channels=64, num_stages=3)
train(model, train_loader, val_loader,
      num_epochs=100, initial_lr=1e-3,
      device='cuda', use_amp=True,
      loss_type='combined', loss_alpha=0.7)
```

### Inference

```python
from denoiser.models import BlindVideoDenoiserUNet
from denoiser.inference import ModelTester
import torch

# Load trained model
model = BlindVideoDenoiserUNet()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Denoise a video folder
tester = ModelTester(model, device='cuda')
denoised_frames, names, resolution = tester.denoise_video_folder(
    'path/to/video/frames/', resize_to=(512, 512)
)
```

### Training Configuration

| Parameter        | Value                    |
|------------------|--------------------------|
| Noise range (σ)  | 5 – 250                  |
| Loss             | Combined L1 + L2 (α=0.7) |
| Optimizer        | Adam (weight decay 1e-5) |
| LR schedule      | Cosine annealing → 1e-6  |
| Mixed precision  | FP16 (AMP)               |
| Early stopping   | Patience = 15 epochs     |

---

## Framework (Coming Soon)

The inverse problem framework will use the trained denoiser as a learned prior to solve general inverse problems of the form:

**y = A(x) + noise**

where **A** is a known linear measurement operator, **y** is the observed (degraded) video, and **x** is the clean video to be recovered.

### Planned Components

- **`inverse_problem_framework.py`** — Iterative solver using the denoiser prior (e.g., PnP, RED, or diffusion-based approach)
- **`linear_operators.py`** — Measurement operators (inpainting masks, blur kernels, downsampling, compressed sensing, etc.)
- **`benchmark.py`** — Evaluation pipeline across inverse problem types and noise levels

---

## Installation

```bash
git clone https://github.com/<your-username>/alejandro-research-project.git
cd alejandro-research-project
pip install -r requirements.txt
```

## License

MIT
