"""
Denoiser wrapper classes for different video denoising models.

These wrappers adapt various denoiser architectures to the common VideoDenoiser
interface required by the Kadkhodaie & Simoncelli inverse problem solver.

All wrappers return the DENOISED IMAGE (clean estimate), not the noise residual.
The solver handles the conversion internally.
"""

import torch
import torch.nn.functional as F
from inverse_problem_framework import VideoDenoiser


class BlindVideoDenoiserWrapper(VideoDenoiser):
    """
    Wrapper for the BlindVideoDenoiserUNet.

    Takes 3 concatenated frames (B, 9, H, W) → denoised center frame (B, 3, H, W).
    Handles temporal sliding window and spatial padding.
    """

    def __init__(self, model, device: str = 'cuda', pad_to: int = 8):
        """
        Args:
            model: BlindVideoDenoiserUNet instance
            device: 'cuda' or 'cpu'
            pad_to: Pad spatial dims to be divisible by this.
                    3-stage UNet: pad_to=8, 4-stage: pad_to=16
        """
        self.model = model.to(device)
        self.device = device
        self.pad_to = pad_to
        self.model.eval()

    def _pad(self, frame: torch.Tensor):
        """Pad frame so H, W are divisible by pad_to."""
        C, H, W = frame.shape
        ph = (self.pad_to - H % self.pad_to) % self.pad_to
        pw = (self.pad_to - W % self.pad_to) % self.pad_to
        if ph > 0 or pw > 0:
            frame = F.pad(frame, (0, pw, 0, ph), mode='reflect')
        return frame, (H, W)

    def _unpad(self, frame: torch.Tensor, orig_size: tuple):
        """Remove padding to restore original size."""
        H, W = orig_size
        return frame[:, :H, :W]

    def denoise(self, noisy_video: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """
        Denoise entire video frame by frame with temporal context.
        Args: noisy_video: (T, C, H, W) in [0, 1]
        Returns: denoised_video: (T, C, H, W) in [0, 1]
        """
        T, C, H, W = noisy_video.shape
        noisy_video = noisy_video.to(self.device)
        frames = []

        with torch.no_grad():
            for t in range(T):
                p, c, n = max(0, t - 1), t, min(T - 1, t + 1)
                d = self.denoise_frame(
                    noisy_video[p], noisy_video[c], noisy_video[n], noise_std
                )
                frames.append(d)

        return torch.clamp(torch.stack(frames), 0, 1)

    def denoise_frame(self, prev_frame, curr_frame, next_frame, noise_std=0.0):
        """Denoise single frame with temporal context."""
        pf, _ = self._pad(prev_frame)
        cf, orig = self._pad(curr_frame)
        nf, _ = self._pad(next_frame)

        triplet = torch.cat([pf, cf, nf], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(triplet)

        return torch.clamp(self._unpad(out.squeeze(0), orig), 0, 1)


class FastDVDNetWrapper(VideoDenoiser):
    """Wrapper for FastDVDNet. Converts noise_std from 0-1 to 0-255 scale."""

    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def denoise(self, noisy_video: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        T, C, H, W = noisy_video.shape
        noisy_video = noisy_video.to(self.device)
        video_batch = noisy_video.unsqueeze(0)

        noise_std_255 = noise_std * 255.0 if noise_std <= 1.0 else noise_std
        noise_map = torch.full((1, 1, H, W), noise_std_255 / 255.0, device=self.device)

        with torch.no_grad():
            denoised = self.model(video_batch, noise_map)

        return torch.clamp(denoised.squeeze(0), 0, 1)

    def denoise_frame(self, prev_frame, curr_frame, next_frame, noise_std=0.0):
        video = torch.stack([prev_frame, curr_frame, next_frame], dim=0)
        denoised = self.denoise(video, noise_std)
        return denoised[1]


class VRTWrapper(VideoDenoiser):
    """Wrapper for VRT (Video Restoration Transformer)."""

    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def denoise(self, noisy_video: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        noisy_video = noisy_video.to(self.device)
        video_batch = noisy_video.unsqueeze(0)

        with torch.no_grad():
            denoised = self.model(video_batch)

        return torch.clamp(denoised.squeeze(0), 0, 1)

    def denoise_frame(self, prev_frame, curr_frame, next_frame, noise_std=0.0):
        video = torch.stack([prev_frame, curr_frame, next_frame], dim=0).unsqueeze(0)
        with torch.no_grad():
            denoised = self.model(video)
        return torch.clamp(denoised[0, 1], 0, 1)