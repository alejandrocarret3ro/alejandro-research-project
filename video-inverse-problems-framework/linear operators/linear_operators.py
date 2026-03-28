"""
Linear operators for video inverse problems.

Currently focused on demosaicing (Bayer pattern).
Demosaicing is structurally identical to inpainting: the forward operator is a
per-channel binary mask determined by the Bayer color filter array (CFA).

For the Kadkhodaie & Simoncelli algorithm, the key property is that
project(x) = A^T(A(x)) is an exact projection. This holds perfectly for
demosaicing since the mask is diagonal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from inverse_problem_framework import LinearOperator


class DemosaicingOperator(LinearOperator):
    """
    Demosaicing (Bayer CFA) operator.

    Forward model: each pixel location observes exactly ONE color channel
    according to the Bayer pattern. The other two channels are zeroed out.

    For an RGGB pattern on a 4x4 block:
        R G R G ...      Channel 0 (R): 1 0 1 0 / 0 0 0 0 / ...
        G B G B ...      Channel 1 (G): 0 1 0 1 / 1 0 1 0 / ...
        R G R G ...      Channel 2 (B): 0 0 0 0 / 0 1 0 1 / ...
        G B G B ...

    The mask has shape (1, 3, H, W) — one binary mask per channel.
    Each spatial location has exactly one channel set to 1.

    This is mathematically identical to inpainting: y = mask * x,
    so project(x) = mask * x is an exact projection, and the
    standard KadkhodaieSolver works directly.
    """

    def __init__(self, H: int, W: int, pattern: str = 'RGGB', device: str = 'cuda'):
        """
        Args:
            H: Frame height (must be even)
            W: Frame width (must be even)
            pattern: Bayer pattern string. One of 'RGGB', 'BGGR', 'GRBG', 'GBRG'.
            device: 'cuda' or 'cpu'
        """
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for Bayer pattern"
        self.device = device
        self.H = H
        self.W = W
        self.pattern = pattern

        # Build the (1, 3, H, W) Bayer mask
        self.mask = self._build_bayer_mask(H, W, pattern).to(device)

    def _build_bayer_mask(self, H: int, W: int, pattern: str) -> torch.Tensor:
        """
        Build a Bayer CFA mask of shape (1, 3, H, W).
        Each spatial position (i, j) has exactly one channel == 1.
        """
        mask = torch.zeros(1, 3, H, W)

        # Map pattern string to 2x2 channel indices
        color_map = {'R': 0, 'G': 1, 'B': 2}
        tl = color_map[pattern[0]]  # top-left
        tr = color_map[pattern[1]]  # top-right
        bl = color_map[pattern[2]]  # bottom-left
        br = color_map[pattern[3]]  # bottom-right

        # Fill the mask by tiling the 2x2 pattern
        mask[0, tl, 0::2, 0::2] = 1.0  # top-left positions
        mask[0, tr, 0::2, 1::2] = 1.0  # top-right positions
        mask[0, bl, 1::2, 0::2] = 1.0  # bottom-left positions
        mask[0, br, 1::2, 1::2] = 1.0  # bottom-right positions

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Bayer CFA: keep only the observed channel at each pixel.
        Args: x: (T, 3, H, W) or (3, H, W) full RGB video/frame
        Returns: y: same shape, with unobserved channels zeroed out
        """
        return x * self.mask

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint is the same as forward for a diagonal binary mask."""
        return y * self.mask

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto CFA-observed subspace."""
        return x * self.mask

    def null_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto the unobserved (missing) subspace."""
        return x * (1.0 - self.mask)

    def mosaic_to_rgb_nearest(self, mosaiced: torch.Tensor) -> torch.Tensor:
        """
        Simple nearest-neighbor demosaicing for baseline comparison.
        NOT used by the solver — just for creating a naive baseline.

        Args: mosaiced: (T, 3, H, W) CFA-masked image
        Returns: demosaiced: (T, 3, H, W) naive interpolation
        """
        result = mosaiced.clone()
        kernel_size = 3
        pad = kernel_size // 2

        for c in range(3):
            channel_mask = self.mask[0, c]  # (H, W)
            channel_data = mosaiced[:, c]   # (T, H, W)

            observed = channel_data * channel_mask.unsqueeze(0)
            observed_4d = observed.unsqueeze(1)
            mask_4d = channel_mask.unsqueeze(0).unsqueeze(0).expand(
                observed_4d.shape[0], -1, -1, -1
            )

            sum_vals = F.avg_pool2d(
                F.pad(observed_4d, (pad, pad, pad, pad), mode='reflect'),
                kernel_size, stride=1
            ) * (kernel_size ** 2)
            sum_mask = F.avg_pool2d(
                F.pad(mask_4d, (pad, pad, pad, pad), mode='reflect'),
                kernel_size, stride=1
            ) * (kernel_size ** 2)

            interpolated = sum_vals / (sum_mask + 1e-8)
            interpolated = interpolated.squeeze(1)

            result[:, c] = (channel_data * channel_mask.unsqueeze(0) +
                            interpolated * (1 - channel_mask.unsqueeze(0)))

        return result


def create_bayer_mosaic(clean_video: torch.Tensor, H: int, W: int,
                        pattern: str = 'RGGB', device: str = 'cuda'):
    """
    Convenience function: create a mosaiced video from a clean video.

    Returns:
        operator: DemosaicingOperator instance
        mosaiced: (T, 3, H, W) — the CFA-sampled observation
        naive_demosaiced: (T, 3, H, W) — simple baseline for comparison
    """
    operator = DemosaicingOperator(H, W, pattern=pattern, device=device)
    mosaiced = operator.forward(clean_video)
    naive_demosaiced = operator.mosaic_to_rgb_nearest(mosaiced)
    return operator, mosaiced, naive_demosaiced