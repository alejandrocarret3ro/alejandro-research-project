import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class ModelTester:
    """Test blind video denoiser on video sequences."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def denoise_frame(self, prev_frame, curr_frame, next_frame):
        """
        Denoise a single frame using temporal context.
        Maintains color accuracy through minimal conversions.
        
        Args:
            prev_frame: PIL Image or torch tensor (H, W, 3)
            curr_frame: PIL Image or torch tensor (H, W, 3)
            next_frame: PIL Image or torch tensor (H, W, 3)
        
        Returns:
            denoised: numpy array (H, W, 3) in range [0, 1] as float32
        """
        # Convert to tensors if PIL images (direct conversion to preserve precision)
        if isinstance(prev_frame, Image.Image):
            prev_np = np.array(prev_frame, dtype=np.float32) / 255.0
            prev_tensor = torch.from_numpy(prev_np).permute(2, 0, 1)
        else:
            prev_tensor = prev_frame
        
        if isinstance(curr_frame, Image.Image):
            curr_np = np.array(curr_frame, dtype=np.float32) / 255.0
            curr_tensor = torch.from_numpy(curr_np).permute(2, 0, 1)
        else:
            curr_tensor = curr_frame
        
        if isinstance(next_frame, Image.Image):
            next_np = np.array(next_frame, dtype=np.float32) / 255.0
            next_tensor = torch.from_numpy(next_np).permute(2, 0, 1)
        else:
            next_tensor = next_frame
        
        # Concatenate frames: (9, H, W)
        triplet = torch.cat([prev_tensor, curr_tensor, next_tensor], dim=0)
        triplet = triplet.unsqueeze(0).to(self.device)  # (1, 9, H, W)
        
        # Inference
        with torch.no_grad():
            denoised = self.model(triplet)
        
        # Convert back to numpy (0-1 range) - clamp to valid range
        denoised = denoised.squeeze(0).cpu() # (3, H, W)
        denoised = torch.clamp(denoised, 0.0, 1.0) # clamping to restrict the changes in pixels and avoid artifacts
        denoised = denoised.permute(1, 2, 0).numpy().astype(np.float32) # (H, W, 3)
        
        return denoised
    
    def denoise_video_folder(self, video_folder_path, resize_to=(512, 512)):
        """
        Denoise all frames in a video folder.
        
        Args:
            video_folder_path: Path to folder containing sequential frames
            resize_to: Target resolution (H, W) for processing. Must be even dimensions.
        
        Returns:
            denoised_frames: List of denoised frames (numpy arrays, 0-1 range, original resolution)
            frame_names: List of original frame names
            original_resolution: Original (H, W) before resizing
        """
        from torchvision.transforms import Resize
        
        # Load frames
        frame_files = sorted([
            f for f in os.listdir(video_folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        frames = []
        for frame_file in frame_files:
            img = Image.open(os.path.join(video_folder_path, frame_file)).convert('RGB')
            frames.append(img)
        
        original_resolution = frames[0].size[::-1]  # (H, W)
        
        print(f"Loaded {len(frames)} frames from {video_folder_path}")
        print(f"Original resolution: {original_resolution[1]}x{original_resolution[0]}")
        print(f"Processing resolution: {resize_to[1]}x{resize_to[0]}")
        
        # Resize frames for processing
        resize_transform = Resize(resize_to)
        frames_resized = []
        for frame in frames:
            frame_resized = frame.resize((resize_to[1], resize_to[0]), Image.BILINEAR)
            frames_resized.append(frame_resized)
        
        denoised_frames_resized = []
        
        for i in range(len(frames_resized)):
            # Get temporal triplet (with boundary handling)
            prev_idx = max(0, i - 1)
            curr_idx = i
            next_idx = min(len(frames_resized) - 1, i + 1)
            
            denoised = self.denoise_frame(frames_resized[prev_idx], frames_resized[curr_idx], frames_resized[next_idx])
            denoised_frames_resized.append(denoised)
            
            if (i + 1) % 10 == 0:
                print(f"  Denoised {i + 1}/{len(frames_resized)} frames")
        
        # Resize denoised frames back to original resolution (HIGH QUALITY)
        denoised_frames = []
        for denoised in denoised_frames_resized:
            # Convert float32 [0, 1] directly without uint8 intermediate
            denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8))
            # Use high-quality resampling filter
            denoised_pil = denoised_pil.resize(
                (original_resolution[1], original_resolution[0]), 
                Image.Resampling.LANCZOS
            )
            # Convert back to float32 [0, 1] preserving precision
            denoised_np = np.array(denoised_pil, dtype=np.float32) / 255.0
            denoised_frames.append(denoised_np)
        
        return denoised_frames, frame_files, original_resolution
    
    def compute_metrics(self, noisy_frames, clean_frames, denoised_frames, noise_stds=None):
        """
        Compute PSNR and SSIM metrics, optionally normalized by noise level.
        
        Args:
            noisy_frames: List of noisy frames (0-1 range, numpy arrays)
            clean_frames: List of clean frames (0-1 range, numpy arrays)
            denoised_frames: List of denoised frames (0-1 range, numpy arrays)
            noise_stds: List of noise standard deviations (in 0-1 range, e.g., 50/255).
                       If None, assumes constant noise across all frames.
        
        Returns:
            metrics: Dict with PSNR, SSIM values (raw and noise-normalized)
        """
        noisy_psnr_list = []
        denoised_psnr_list = []
        noisy_ssim_list = []
        denoised_ssim_list = []
        
        # Noise-normalized metrics
        noisy_psnr_normalized_list = []
        denoised_psnr_normalized_list = []
        noisy_ssim_normalized_list = []
        denoised_ssim_normalized_list = []
        
        for i, (noisy, clean, denoised) in enumerate(zip(noisy_frames, clean_frames, denoised_frames)):
            # Convert to 0-255 for metric computation
            clean_255 = (clean * 255).astype(np.uint8)
            noisy_255 = (noisy * 255).astype(np.uint8)
            denoised_255 = (denoised * 255).astype(np.uint8)
            
            # PSNR (in dB)
            noisy_psnr = psnr(clean_255, noisy_255, data_range=255)
            denoised_psnr = psnr(clean_255, denoised_255, data_range=255)
            noisy_psnr_list.append(noisy_psnr)
            denoised_psnr_list.append(denoised_psnr)
            
            # SSIM
            noisy_ssim = ssim(clean_255, noisy_255, data_range=255, channel_axis=2)
            denoised_ssim = ssim(clean_255, denoised_255, data_range=255, channel_axis=2)
            noisy_ssim_list.append(noisy_ssim)
            denoised_ssim_list.append(denoised_ssim)
            
            # Noise-normalized metrics
            if noise_stds is not None:
                # Get noise std for this frame (in 0-255 range)
                if isinstance(noise_stds, (list, tuple)):
                    noise_std_255 = noise_stds[i]
                else:
                    noise_std_255 = noise_stds  # Single value for all
                
                # Normalize by noise level
                # Higher noise_std = easier to denoise = lower normalized error
                noisy_psnr_norm = noisy_psnr - 10 * np.log10((noise_std_255 / 255.0) ** 2)
                denoised_psnr_norm = denoised_psnr - 10 * np.log10((noise_std_255 / 255.0) ** 2)
                
                # SSIM normalization: relative improvement over noise
                # Measures how much we improved the structural similarity
                noisy_ssim_norm = (noisy_ssim + 1) / (1 + np.exp(-noise_std_255 / 50))  # Sigmoid-based
                denoised_ssim_norm = (denoised_ssim + 1) / (1 + np.exp(-noise_std_255 / 50))
                
                noisy_psnr_normalized_list.append(noisy_psnr_norm)
                denoised_psnr_normalized_list.append(denoised_psnr_norm)
                noisy_ssim_normalized_list.append(noisy_ssim_norm)
                denoised_ssim_normalized_list.append(denoised_ssim_norm)
        
        metrics = {
            # Raw metrics
            'noisy_psnr_mean': np.mean(noisy_psnr_list),
            'denoised_psnr_mean': np.mean(denoised_psnr_list),
            'psnr_improvement': np.mean(denoised_psnr_list) - np.mean(noisy_psnr_list),
            'noisy_ssim_mean': np.mean(noisy_ssim_list),
            'denoised_ssim_mean': np.mean(denoised_ssim_list),
            'ssim_improvement': np.mean(denoised_ssim_list) - np.mean(noisy_ssim_list),
        }
        
        # Add noise-normalized metrics if available
        if noise_stds is not None and len(denoised_psnr_normalized_list) > 0:
            metrics.update({
                'noisy_psnr_normalized_mean': np.mean(noisy_psnr_normalized_list),
                'denoised_psnr_normalized_mean': np.mean(denoised_psnr_normalized_list),
                'psnr_normalized_improvement': np.mean(denoised_psnr_normalized_list) - np.mean(noisy_psnr_normalized_list),
                'noisy_ssim_normalized_mean': np.mean(noisy_ssim_normalized_list),
                'denoised_ssim_normalized_mean': np.mean(denoised_ssim_normalized_list),
                'ssim_normalized_improvement': np.mean(denoised_ssim_normalized_list) - np.mean(noisy_ssim_normalized_list),
            })
        
        return metrics
    
    def visualize_results(self, clean_frame, noisy_frame, denoised_frame, save_path=None):
        """
        Visualize comparison between noisy, denoised, and clean frames.
        
        Args:
            clean_frame: Clean frame (0-1 range, numpy array)
            noisy_frame: Noisy frame (0-1 range, numpy array)
            denoised_frame: Denoised frame (0-1 range, numpy array)
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Noisy
        axes[0].imshow(noisy_frame)
        axes[0].set_title('Noisy')
        axes[0].axis('off')
        
        # Denoised
        axes[1].imshow(denoised_frame)
        axes[1].set_title('Denoised')
        axes[1].axis('off')
        
        # Clean
        axes[2].imshow(clean_frame)
        axes[2].set_title('Clean (Ground Truth)')
        axes[2].axis('off')
        
        # Difference (denoised vs clean)
        diff = np.abs(denoised_frame.astype(np.float32) - clean_frame.astype(np.float32))
        axes[3].imshow(diff, cmap='hot')
        axes[3].set_title('Error Map')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


    def save_video(self, frames, save_path, fps=24):
        """
        Save a list of frames as an MP4 video.

        Args:
            frames: List of numpy arrays (H, W, 3) in [0, 1] float32 range
            save_path: Output path (e.g., 'output.mp4')
            fps: Frames per second
        """
        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        for frame in frames:
            frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"Video saved: {save_path} ({len(frames)} frames, {fps} fps, {w}x{h})")

    def save_comparison_video(self, clean_frames, noisy_frames, denoised_frames,
                              save_path, fps=24, label_height=40):
        """
        Save a side-by-side comparison video: Noisy | Denoised | Clean.
        Each panel is labelled at the top.

        Args:
            clean_frames: List of clean numpy arrays (H, W, 3) in [0, 1]
            noisy_frames: List of noisy numpy arrays (H, W, 3) in [0, 1]
            denoised_frames: List of denoised numpy arrays (H, W, 3) in [0, 1]
            save_path: Output MP4 path
            fps: Frames per second
            label_height: Height in pixels for the text label bar
        """
        import cv2

        h, w = clean_frames[0].shape[:2]
        canvas_w = w * 3
        canvas_h = h + label_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (canvas_w, canvas_h))

        labels = ["Noisy", "Denoised", "Clean"]

        for noisy, denoised, clean in zip(noisy_frames, denoised_frames, clean_frames):
            # Create canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Add label bar (dark background)
            canvas[:label_height, :, :] = 30  # dark gray

            for i, (label, frame) in enumerate(zip(labels, [noisy, denoised, clean])):
                # Place frame
                frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                x_start = i * w
                canvas[label_height:, x_start:x_start + w, :] = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

                # Add label text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = x_start + (w - text_size[0]) // 2
                text_y = label_height - 12
                cv2.putText(canvas, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(canvas)

        writer.release()
        print(f"Comparison video saved: {save_path}")
        print(f"  Layout: Noisy | Denoised | Clean ({canvas_w}x{canvas_h}, {len(clean_frames)} frames)")

    def denoise_and_save_videos(self, video_folder_path, output_dir, noise_std=50,
                                 resize_to=(512, 512), fps=24, seed=42):
        """
        Full pipeline: load video, add noise, denoise, and save all videos.

        Args:
            video_folder_path: Path to DAVIS video folder with sequential frames
            output_dir: Directory to save output videos
            noise_std: Gaussian noise standard deviation (0-255 scale)
            resize_to: Processing resolution for the denoiser
            fps: Output video frame rate
            seed: Random seed for reproducible noise

        Returns:
            metrics: Dict with PSNR/SSIM results
        """
        video_name = os.path.basename(video_folder_path)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing video: {video_name} (noise σ={noise_std})")
        print(f"{'='*60}")

        # 1. Load clean frames
        frame_files = sorted([
            f for f in os.listdir(video_folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        clean_frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(video_folder_path, fname)).convert('RGB')
            clean_frames.append(np.array(img, dtype=np.float32) / 255.0)

        print(f"Loaded {len(clean_frames)} clean frames")

        # 2. Add noise (seeded for reproducibility)
        rng = np.random.RandomState(seed)
        noisy_frames = []
        for frame in clean_frames:
            noise = rng.normal(0, noise_std / 255.0, frame.shape).astype(np.float32)
            noisy_frames.append(np.clip(frame + noise, 0, 1))

        # 3. Denoise frame by frame with temporal context
        print("Denoising...")
        denoised_frames = []
        for i in range(len(noisy_frames)):
            prev_idx = max(0, i - 1)
            next_idx = min(len(noisy_frames) - 1, i + 1)

            # Resize for processing
            prev = self._resize_frame(noisy_frames[prev_idx], resize_to)
            curr = self._resize_frame(noisy_frames[i], resize_to)
            nxt = self._resize_frame(noisy_frames[next_idx], resize_to)

            denoised = self.denoise_frame(
                Image.fromarray((prev * 255).astype(np.uint8)),
                Image.fromarray((curr * 255).astype(np.uint8)),
                Image.fromarray((nxt * 255).astype(np.uint8))
            )

            # Resize back to original resolution
            orig_h, orig_w = clean_frames[0].shape[:2]
            denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8))
            denoised_pil = denoised_pil.resize((orig_w, orig_h), Image.LANCZOS)
            denoised_frames.append(np.array(denoised_pil, dtype=np.float32) / 255.0)

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(noisy_frames)} frames done")

        # 4. Compute metrics
        metrics = self.compute_metrics(noisy_frames, clean_frames, denoised_frames)
        print(f"\nPSNR: {metrics['noisy_psnr_mean']:.2f} → {metrics['denoised_psnr_mean']:.2f} dB "
              f"(+{metrics['psnr_improvement']:.2f})")
        print(f"SSIM: {metrics['noisy_ssim_mean']:.4f} → {metrics['denoised_ssim_mean']:.4f} "
              f"(+{metrics['ssim_improvement']:.4f})")

        # 5. Save videos
        prefix = f"{video_name}_sigma{noise_std}"

        self.save_video(
            noisy_frames,
            os.path.join(output_dir, f"{prefix}_noisy.mp4"), fps=fps
        )
        self.save_video(
            denoised_frames,
            os.path.join(output_dir, f"{prefix}_denoised.mp4"), fps=fps
        )
        self.save_video(
            clean_frames,
            os.path.join(output_dir, f"{prefix}_clean.mp4"), fps=fps
        )
        self.save_comparison_video(
            clean_frames, noisy_frames, denoised_frames,
            os.path.join(output_dir, f"{prefix}_comparison.mp4"), fps=fps
        )

        print(f"\nAll videos saved to {output_dir}")
        return metrics

    def _resize_frame(self, frame_np, size):
        """Resize (H, W, 3) float32 [0,1] numpy array to (size[0], size[1])."""
        pil = Image.fromarray((frame_np * 255).astype(np.uint8))
        pil = pil.resize((size[1], size[0]), Image.BILINEAR)
        return np.array(pil, dtype=np.float32) / 255.0


def test_on_custom_frames(model_path, frame_paths, device='cuda'):
    """
    Test model on custom frames.
    
    Args:
        model_path: Path to best_model.pt checkpoint
        frame_paths: List of 3 frame paths [prev, curr, next]
        device: 'cuda' or 'cpu'
    
    Returns:
        denoised_frame: Denoised center frame (0-1 range)
    """
    from unet_denoiser import BlindVideoDenoiserUNet
    
    # Load model
    model = BlindVideoDenoiserUNet(in_channels=9, out_channels=3, base_channels=64, num_stages=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    tester = ModelTester(model, device=device)
    
    # Load frames
    frames = [Image.open(p).convert('RGB') for p in frame_paths]
    
    # Denoise
    denoised = tester.denoise_frame(frames[0], frames[1], frames[2])
    
    return denoised


if __name__ == "__main__":
    print("Use this module to test your trained denoiser model")
    print("")
    print("Example 1: Test on a video folder")
    print("from model_testing import ModelTester")
    print("from unet_denoiser import BlindVideoDenoiserUNet")
    print("import torch")
    print("")
    print("model = BlindVideoDenoiserUNet()")
    print("checkpoint = torch.load('best_model.pt')")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("tester = ModelTester(model, device='cuda')")
    print("denoised_frames, names = tester.denoise_video_folder('path/to/video')")
    print("")
    print("Example 2: Test on custom frames")
    print("denoised = test_on_custom_frames('best_model.pt', ['frame1.png', 'frame2.png', 'frame3.png'])")