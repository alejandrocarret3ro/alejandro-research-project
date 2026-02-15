import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class BlindDenoiseDataset(Dataset):
    """
    Dataset for training a blind denoiser on video frames.
    Adds variable Gaussian noise to clean frames from DAVIS dataset.
    """
    def __init__(self, root_dir, noise_std_range=(5, 250), transform=None, seed=None):
        """
        Args:
            root_dir: Path to DAVIS JPEGImages folder
            noise_std_range: Tuple of (min_std, max_std) for random noise levels
            transform: Optional torchvision transforms
            seed: Random seed for reproducibility (optional)
        """
        self.root_dir = root_dir
        self.noise_std_range = noise_std_range
        self.transform = transform
        
        if seed is not None:
            np.random.seed(seed)
        
        # Load all frame paths with video info
        self.frame_data = self._load_frame_paths()
        
    def _load_frame_paths(self):
        """Load all frame paths and organize by video."""
        frame_data = []
        
        for video_name in sorted(os.listdir(self.root_dir)):
            video_path = os.path.join(self.root_dir, video_name)
            
            if not os.path.isdir(video_path):
                continue
                
            frames = sorted([
                f for f in os.listdir(video_path) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            for frame_name in frames:
                frame_data.append({
                    'path': os.path.join(video_path, frame_name),
                    'video': video_name,
                    'frame': frame_name
                })
        
        return frame_data
    
    def _add_noise(self, image_np):
        """
        Add random Gaussian noise to image.
        Noise std is randomly sampled for each frame (blind training).
        """
        # Sample random noise level for this frame
        noise_std = np.random.uniform(*self.noise_std_range)
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, image_np.shape)
        
        # Add noise and clip to valid range
        noisy_image = np.clip(image_np + noise, 0, 255)
        
        return noisy_image.astype(np.uint8), noise_std
    
    def __len__(self):
        return len(self.frame_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            noisy_frame: Tensor of noisy frame
            clean_frame: Tensor of clean frame (ground truth)
            noise_std: The noise level used (for analysis, not training input)
        """
        frame_info = self.frame_data[idx]
        
        # Load clean frame
        clean_image = Image.open(frame_info['path']).convert('RGB')
        clean_np = np.array(clean_image, dtype=np.float32)
        
        # Add random noise
        noisy_np, noise_std = self._add_noise(clean_np)
        noisy_image = Image.fromarray(noisy_np)
        
        # Apply transforms
        if self.transform:
            clean_frame = self.transform(clean_image)
            noisy_frame = self.transform(noisy_image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            to_tensor = transforms.ToTensor()
            clean_frame = to_tensor(clean_image)
            noisy_frame = to_tensor(noisy_image)
        
        return noisy_frame, clean_frame, noise_std
