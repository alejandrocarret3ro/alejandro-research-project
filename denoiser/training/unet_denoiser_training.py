import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import torchvision.transforms as transforms


class TemporalDenoiseDataset(Dataset):
    """
    Efficient PyTorch Dataset that returns temporal triplets (prev, curr, next).
    
    Key optimizations vs previous version:
    - Resizes at PIL level (before tensor conversion) to avoid full-res tensor ops
    - Adds noise directly to the resized image (less memory, less compute)
    - Returns float16 tensors to reduce CPU->GPU transfer time
    - No dependency on external BlindDenoiseDataset — self-contained
    """
    def __init__(self, frame_data, noise_std_range=(5, 250), resize_to=(256, 256), use_fp16=True):
        """
        Args:
            frame_data: List of dicts with 'path', 'video', 'frame' keys
            noise_std_range: (min_std, max_std) for random Gaussian noise (0-255 scale)
            resize_to: Target resolution (height, width)
            use_fp16: If True, return float16 tensors (faster transfer, less memory)
        """
        self.frame_data = frame_data
        self.noise_std_range = noise_std_range
        self.resize_to = resize_to  # (H, W)
        self.use_fp16 = use_fp16
        self.target_size = (resize_to[1], resize_to[0])  # PIL uses (W, H)

        # Group frames by video for temporal triplet construction
        self.videos = {}
        for idx, frame_info in enumerate(self.frame_data):
            video = frame_info['video']
            if video not in self.videos:
                self.videos[video] = []
            self.videos[video].append(idx)

        # Pre-compute O(1) lookup: idx -> (video_name, position_in_video)
        self.idx_to_video = {}
        self.idx_to_pos = {}
        for video, indices in self.videos.items():
            for pos, idx in enumerate(indices):
                self.idx_to_video[idx] = video
                self.idx_to_pos[idx] = pos

    def _load_and_resize(self, idx):
        """Load a frame from disk and resize it. Returns numpy float32 array (H, W, 3) in 0-255."""
        path = self.frame_data[idx]['path']
        img = Image.open(path).convert('RGB')
        img = img.resize(self.target_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    def _add_noise(self, image_np):
        """Add random Gaussian noise. Input/output in 0-255 float32."""
        noise_std = np.random.uniform(*self.noise_std_range)
        noise = np.random.normal(0, noise_std, image_np.shape).astype(np.float32)
        noisy = np.clip(image_np + noise, 0, 255)
        return noisy, noise_std

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        """
        Returns:
            noisy_triplet: (9, H, W) tensor — 3 noisy RGB frames concatenated
            clean_curr: (3, H, W) tensor — clean center frame
            noise_std: float — noise level used for center frame
        """
        video = self.idx_to_video[idx]
        pos = self.idx_to_pos[idx]
        video_frames = self.videos[video]

        # Temporal neighbors with boundary handling
        prev_idx = video_frames[max(0, pos - 1)]
        next_idx = video_frames[min(len(video_frames) - 1, pos + 1)]

        # Load and resize at PIL level (avoids full-res tensor operations)
        prev_np = self._load_and_resize(prev_idx)
        curr_np = self._load_and_resize(idx)
        next_np = self._load_and_resize(next_idx)

        # Add independent noise to each frame
        noisy_prev, _ = self._add_noise(prev_np)
        noisy_curr, noise_std = self._add_noise(curr_np)
        noisy_next, _ = self._add_noise(next_np)

        # Convert to tensors: (H, W, 3) float32 [0-255] -> (9, H, W) float [0-1]
        noisy_triplet = np.concatenate([noisy_prev, noisy_curr, noisy_next], axis=2)  # (H, W, 9)
        noisy_triplet = torch.from_numpy(noisy_triplet).permute(2, 0, 1) / 255.0  # (9, H, W)

        clean_curr_tensor = torch.from_numpy(curr_np).permute(2, 0, 1) / 255.0  # (3, H, W)

        # Convert to float16 for faster CPU->GPU transfer
        if self.use_fp16:
            noisy_triplet = noisy_triplet.half()
            clean_curr_tensor = clean_curr_tensor.half()

        return noisy_triplet, clean_curr_tensor, noise_std


class TrainingLogger:
    """Log training metrics and save checkpoints."""
    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time_seconds": []
        }
        self.start_time = datetime.now()

    def log_epoch(self, train_loss, val_loss, lr, epoch_time=None):
        self.metrics["train_loss"].append(float(train_loss))
        self.metrics["val_loss"].append(float(val_loss))
        self.metrics["learning_rate"].append(float(lr))
        if epoch_time is not None:
            self.metrics["epoch_time_seconds"].append(float(epoch_time))

        epoch = len(self.metrics["train_loss"])
        time_str = f" | Time: {epoch_time:.1f}s" if epoch_time else ""
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}{time_str}")

    def save_metrics(self):
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        epochs = range(1, len(self.metrics["train_loss"]) + 1)
        ax1.plot(epochs, self.metrics["train_loss"], label="Train Loss", marker='o')
        ax1.plot(epochs, self.metrics["val_loss"], label="Val Loss", marker='s')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(epochs, self.metrics["learning_rate"], label="Learning Rate", marker='^')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.log_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Metrics saved to {self.log_dir}")


class CombinedL1L2Loss(nn.Module):
    """Combined L1 and L2 loss for sharp, natural denoising."""
    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: Weight for L1 loss (0-1). Higher alpha = more L1.
                   0.5 gives equal weight to L1 and L2
                   0.7 gives 70% L1, 30% L2 (sharper)
        """
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * l2


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for noisy_triplets, clean_frames, noise_stds in train_loader:
        # Data arrives as fp16 from dataset, autocast handles the rest
        noisy_triplets = noisy_triplets.to(device, non_blocking=True)
        clean_frames = clean_frames.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                # autocast handles fp16 input -> fp16 forward -> fp32 loss
                denoised = model(noisy_triplets)
                loss = criterion(denoised, clean_frames)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Without AMP, cast fp16 data to fp32 for the model
            noisy_triplets = noisy_triplets.float()
            clean_frames = clean_frames.float()
            denoised = model(noisy_triplets)
            loss = criterion(denoised, clean_frames)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, criterion, device, use_amp=False):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for noisy_triplets, clean_frames, noise_stds in val_loader:
            noisy_triplets = noisy_triplets.to(device, non_blocking=True)
            clean_frames = clean_frames.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    denoised = model(noisy_triplets)
                    loss = criterion(denoised, clean_frames)
            else:
                noisy_triplets = noisy_triplets.float()
                clean_frames = clean_frames.float()
                denoised = model(noisy_triplets)
                loss = criterion(denoised, clean_frames)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    initial_lr=1e-3,
    device="cuda",
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    loss_type="l2",
    loss_alpha=0.5,
    use_amp=True,
    save_every_n_epochs=10,
    resume_from=None,
    use_torch_compile=True
):
    """
    Main training loop for blind video denoiser.

    Args:
        model: BlindVideoDenoiserUNet model
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        num_epochs: Number of training epochs
        initial_lr: Initial learning rate
        device: Device to train on ("cuda" or "cpu")
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        loss_type: Loss function type ("l2", "l1", or "combined")
        loss_alpha: Alpha parameter for combined loss (only used if loss_type="combined")
        use_amp: Whether to use automatic mixed precision (recommended for GPU)
        save_every_n_epochs: Save periodic checkpoints every N epochs
        resume_from: Path to checkpoint to resume training from (optional)
        use_torch_compile: Whether to use torch.compile for extra speed (PyTorch 2.0+)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    logger = TrainingLogger(log_dir)

    # Loss function
    if loss_type == "l2":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "combined":
        criterion = CombinedL1L2Loss(alpha=loss_alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'l1', 'l2', or 'combined'")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = GradScaler() if (use_amp and device == "cuda") else None

    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_loss = float('inf')
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    model = model.to(device)

    # torch.compile for fused kernels (PyTorch 2.0+, ~10-20% speedup)
    compiled_model = model
    if use_torch_compile and device == "cuda":
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"torch.compile not available, using eager mode: {e}")
            compiled_model = model

    patience = 15
    patience_counter = 0

    amp_str = "ON" if scaler is not None else "OFF"
    print(f"\nTraining config: AMP={amp_str}, Loss={loss_type}, LR={initial_lr}")
    print(f"Epochs: {start_epoch}-{num_epochs}, Patience: {patience}\n")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(compiled_model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(compiled_model, val_loader, criterion, device, use_amp=(scaler is not None))

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(train_loss, val_loss, current_lr, epoch_time)

        # Build checkpoint dict (always save the original model, not compiled)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        # Save periodic checkpoint every N epochs
        if epoch % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            patience_counter = 0
            print(f"  → Best model saved! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        scheduler.step()

    logger.save_metrics()
    logger.plot_metrics()

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")

    return model, logger


def create_train_val_split(davis_root_dir, val_split=0.2, seed=42,
                           noise_std_range=(5, 250), resize_to=(256, 256), use_fp16=True):
    """
    Create train/val datasets from a single DAVIS folder.
    Returns TemporalDenoiseDataset instances ready for DataLoader.

    Args:
        davis_root_dir: Path to DAVIS JPEGImages folder
        val_split: Fraction of videos to use for validation (0.2 = 20%)
        seed: Random seed for reproducibility
        noise_std_range: (min_std, max_std) for Gaussian noise
        resize_to: Target resolution (height, width)
        use_fp16: Return float16 tensors for faster data transfer

    Returns:
        train_dataset, val_dataset (TemporalDenoiseDataset instances)
    """
    np.random.seed(seed)

    # Scan directory structure
    frame_data = []
    video_list = []

    for video_name in sorted(os.listdir(davis_root_dir)):
        video_path = os.path.join(davis_root_dir, video_name)

        if not os.path.isdir(video_path):
            continue

        frames = sorted([
            f for f in os.listdir(video_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        if frames:
            video_list.append({
                'name': video_name,
                'start_idx': len(frame_data),
                'end_idx': len(frame_data) + len(frames)
            })

            for frame_name in frames:
                frame_data.append({
                    'path': os.path.join(video_path, frame_name),
                    'video': video_name,
                    'frame': frame_name
                })

    # Split videos (not frames) into train/val
    num_videos = len(video_list)
    num_val_videos = max(1, int(num_videos * val_split))

    val_video_indices = np.random.choice(
        num_videos, size=num_val_videos, replace=False
    )
    val_video_names = {video_list[i]['name'] for i in val_video_indices}

    # Partition frame_data by split
    train_frames = []
    val_frames = []
    for frame_info in frame_data:
        if frame_info['video'] in val_video_names:
            val_frames.append(frame_info)
        else:
            train_frames.append(frame_info)

    print(f"Total videos: {num_videos}")
    print(f"Train videos: {num_videos - num_val_videos} ({len(train_frames)} frames)")
    print(f"Val videos: {num_val_videos} ({len(val_frames)} frames)")
    print(f"Resolution: {resize_to[1]}x{resize_to[0]}, FP16: {use_fp16}")

    # Create datasets directly — no intermediate BlindDenoiseDataset needed
    train_dataset = TemporalDenoiseDataset(
        train_frames, noise_std_range=noise_std_range,
        resize_to=resize_to, use_fp16=use_fp16
    )
    val_dataset = TemporalDenoiseDataset(
        val_frames, noise_std_range=noise_std_range,
        resize_to=resize_to, use_fp16=use_fp16
    )

    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size=8, num_workers=2):
    """
    Create optimized PyTorch DataLoaders.

    Args:
        train_dataset: TemporalDenoiseDataset for training
        val_dataset: TemporalDenoiseDataset for validation
        batch_size: Batch size
        num_workers: Number of parallel data loading workers

    Returns:
        train_loader, val_loader (PyTorch DataLoader instances)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"\nDataLoader config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: True")
    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == "__main__":
    print("=== Optimized Blind Video Denoiser Training ===")
    print("")
    print("Step 0: Copy data to local SSD (run once per Colab session)")
    print("  # First time: zip on Drive for fast future copies")
    print("  # !cd '/content/drive/MyDrive/ResearchProject' && zip -r DAVISDataset.zip DAVISDataset")
    print("  !cp '/content/drive/MyDrive/ResearchProject/DAVISDataset.zip' /content/")
    print("  !unzip -q /content/DAVISDataset.zip -d /content/")
    print("")
    print("Step 1: Create datasets")
    print("  train_dataset, val_dataset = create_train_val_split(")
    print("      '/content/DAVISDataset', val_split=0.2, seed=42,")
    print("      resize_to=(256, 256), use_fp16=True")
    print("  )")
    print("")
    print("Step 2: Create DataLoaders")
    print("  train_loader, val_loader = create_data_loaders(")
    print("      train_dataset, val_dataset, batch_size=16, num_workers=2")
    print("  )")
    print("")
    print("Step 3: Train")
    print("  from unet_denoiser import BlindVideoDenoiserUNet")
    print("  model = BlindVideoDenoiserUNet(in_channels=9, out_channels=3, base_channels=64, num_stages=3)")
    print("  train(model, train_loader, val_loader, num_epochs=100, initial_lr=1e-3,")
    print("        device='cuda', use_amp=True, use_torch_compile=True,")
    print("        loss_type='combined', loss_alpha=0.7)")