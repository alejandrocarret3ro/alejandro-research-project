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