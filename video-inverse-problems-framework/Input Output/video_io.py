
"""
Video I/O utilities for loading, saving, and preprocessing video data.
"""

import torch
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Optional, Tuple, List
import imageio


class VideoLoader:
    """Load videos from various formats."""
    
    @staticmethod
    def load_video(video_path: str, max_frames: Optional[int] = None,
                   resize: Optional[Tuple[int, int]] = None,
                   device: str = 'cuda') -> torch.Tensor:
        """
        Load video file and convert to tensor.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None = all)
            resize: Resize to (height, width) if specified
            device: 'cuda' or 'cpu'
        
        Returns:
            video: Tensor of shape (T, C, H, W) in range [0, 1]
        """
        video_reader = imageio.get_reader(video_path)
        
        frames = []
        for i, frame in enumerate(video_reader):
            if max_frames is not None and i >= max_frames:
                break
            
            # Convert to numpy array
            frame_np = np.array(frame, dtype=np.float32) / 255.0
            
            # Resize if needed
            if resize is not None:
                frame_np = cv2.resize(frame_np, (resize[1], resize[0]),
                                     interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame_np)
        
        # Stack frames: (T, H, W, C)
        video_np = np.stack(frames, axis=0)
        
        # Convert to tensor: (T, C, H, W)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(device)
        
        print(f"Loaded video: {video_path}")
        print(f"  Shape: {video.shape} (T, C, H, W)")
        print(f"  Range: [{video.min():.3f}, {video.max():.3f}]")
        
        return video
    
    @staticmethod
    def load_frame_sequence(folder_path: str, max_frames: Optional[int] = None,
                           resize: Optional[Tuple[int, int]] = None,
                           device: str = 'cuda') -> torch.Tensor:
        """
        Load sequence of frames from folder.
        
        Args:
            folder_path: Path to folder containing sequential frames
            max_frames: Maximum number of frames to load
            resize: Resize to (height, width) if specified
            device: 'cuda' or 'cpu'
        
        Returns:
            video: Tensor of shape (T, C, H, W) in range [0, 1]
        """
        frame_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if max_frames is not None:
            frame_files = frame_files[:max_frames]
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            
            if resize is not None:
                frame = cv2.resize(frame, (resize[1], resize[0]),
                                 interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
        
        # Stack: (T, H, W, C) -> (T, C, H, W)
        video_np = np.stack(frames, axis=0)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(device)
        
        print(f"Loaded {len(frame_files)} frames from: {folder_path}")
        print(f"  Shape: {video.shape} (T, C, H, W)")
        
        return video


class VideoSaver:
    """Save videos and frame sequences."""
    
    @staticmethod
    def save_video(video: torch.Tensor, output_path: str, fps: int = 30) -> None:
        """
        Save tensor to video file.
        
        Args:
            video: Tensor of shape (T, C, H, W) in range [0, 1]
            output_path: Path to save video
            fps: Frames per second
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to numpy (T, H, W, C) in 0-255 range
        video_np = video.permute(0, 2, 3, 1).cpu().numpy()
        video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        
        # Write video
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()
        
        print(f"Saved video: {output_path}")
    
    @staticmethod
    def save_frame_sequence(video: torch.Tensor, output_dir: str,
                           prefix: str = 'frame') -> None:
        """
        Save video as sequence of PNG frames.
        
        Args:
            video: Tensor of shape (T, C, H, W) in range [0, 1]
            output_dir: Directory to save frames
            prefix: Frame filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy (T, H, W, C) in 0-255 range
        video_np = video.permute(0, 2, 3, 1).cpu().numpy()
        video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        
        for t, frame in enumerate(video_np):
            # Convert RGB to BGR for cv2.imwrite
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(output_dir, f'{prefix}_{t:05d}.png')
            cv2.imwrite(frame_path, frame_bgr)
        
        print(f"Saved {len(video_np)} frames to: {output_dir}")


class VideoProcessor:
    """Process and transform videos."""
    
    @staticmethod
    def add_gaussian_noise(video: torch.Tensor, noise_std: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to video.
        
        Args:
            video: Tensor (T, C, H, W) in range [0, 1]
            noise_std: Noise standard deviation in 0-1 range
        
        Returns:
            noisy_video: Tensor (T, C, H, W) in range [0, 1]
            noise: Noise tensor (T, C, H, W)
        """
        noise = torch.randn_like(video) * noise_std
        noisy_video = torch.clamp(video + noise, 0, 1)
        
        return noisy_video, noise
    
    @staticmethod
    def center_crop(video: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
        """
        Center crop video frames.
        
        Args:
            video: Tensor (T, C, H, W)
            crop_size: (height, width)
        
        Returns:
            cropped_video: Tensor (T, C, H', W')
        """
        T, C, H, W = video.shape
        h, w = crop_size
        
        y_start = (H - h) // 2
        x_start = (W - w) // 2
        
        return video[:, :, y_start:y_start+h, x_start:x_start+w]
    
    @staticmethod
    def resize_video(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize video to target size.
        
        Args:
            video: Tensor (T, C, H, W)
            size: (height, width)
        
        Returns:
            resized_video: Tensor (T, C, H', W')
        """
        import torch.nn.functional as F
        
        T, C, H, W = video.shape
        h, w = size
        
        # Reshape for interpolation: (T*C, 1, H, W)
        video_reshaped = video.view(T * C, 1, H, W)
        
        # Resize
        resized = F.interpolate(video_reshaped, size=(h, w), mode='bilinear', align_corners=False)
        
        # Reshape back
        resized = resized.view(T, C, h, w)
        
        return resized
    
    @staticmethod
    def normalize_video(video: torch.Tensor, mean: Optional[List[float]] = None,
                       std: Optional[List[float]] = None) -> torch.Tensor:
        """
        Normalize video to zero mean and unit variance.
        
        Args:
            video: Tensor (T, C, H, W) in range [0, 1]
            mean: Mean per channel (if None, computed from video)
            std: Std per channel (if None, computed from video)
        
        Returns:
            normalized_video: Tensor (T, C, H, W)
        """
        if mean is None:
            mean = video.mean(dim=(0, 2, 3), keepdim=True)
        else:
            mean = torch.tensor(mean).view(1, -1, 1, 1).to(video.device)
        
        if std is None:
            std = video.std(dim=(0, 2, 3), keepdim=True)
        else:
            std = torch.tensor(std).view(1, -1, 1, 1).to(video.device)
        
        return (video - mean) / (std + 1e-8)
    
    @staticmethod
    def rgb_to_y(video: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB video to grayscale (Y channel).
        Standard: Y = 0.299*R + 0.587*G + 0.114*B
        
        Args:
            video: Tensor (T, C, H, W) where C=3 (RGB)
        
        Returns:
            gray_video: Tensor (T, 1, H, W)
        """
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(video.device)
        return (video * weights).sum(dim=1, keepdim=True)