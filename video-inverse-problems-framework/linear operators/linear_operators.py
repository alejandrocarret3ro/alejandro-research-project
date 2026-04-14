"""
Linear operators for video inverse problems.

Supported operators:
- DemosaicingOperator: Bayer CFA pattern (exact projection)
- SuperResolutionOperator: Downsampling by integer factor (approximate projection)
- DeblurringOperator: Convolution with known blur kernel (NOT a projection —
  requires PnPSolver instead of KadkhodaieSolver)
- InpaintingOperator: Binary mask zeroing out a rectangular region (exact projection)
"""

import torch
import torch.nn.functional as F
import numpy as np
from inverse_problem_framework import LinearOperator


# ============================================================
# Demosaicing Operator
# ============================================================

class DemosaicingOperator(LinearOperator):
    """
    Demosaicing (Bayer CFA) operator.
    Forward model: each pixel sees exactly ONE color channel (Bayer mask).
    """

    def __init__(self, H: int, W: int, pattern: str = 'RGGB', device: str = 'cuda'):
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for Bayer pattern"
        self.device = device
        self.H = H
        self.W = W
        self.pattern = pattern
        self.mask = self._build_bayer_mask(H, W, pattern).to(device)

    def _build_bayer_mask(self, H, W, pattern):
        mask = torch.zeros(1, 3, H, W)
        color_map = {'R': 0, 'G': 1, 'B': 2}
        tl, tr = color_map[pattern[0]], color_map[pattern[1]]
        bl, br = color_map[pattern[2]], color_map[pattern[3]]
        mask[0, tl, 0::2, 0::2] = 1.0
        mask[0, tr, 0::2, 1::2] = 1.0
        mask[0, bl, 1::2, 0::2] = 1.0
        mask[0, br, 1::2, 1::2] = 1.0
        return mask

    def forward(self, x): return x * self.mask
    def adjoint(self, y): return y * self.mask
    def project(self, x): return x * self.mask
    def null_project(self, x): return x * (1.0 - self.mask)

    def mosaic_to_rgb_nearest(self, mosaiced):
        result = mosaiced.clone()
        kernel_size, pad = 3, 1
        for c in range(3):
            cm, cd = self.mask[0, c], mosaiced[:, c]
            obs = (cd * cm.unsqueeze(0)).unsqueeze(1)
            m4 = cm.unsqueeze(0).unsqueeze(0).expand(obs.shape[0], -1, -1, -1)
            sv = F.avg_pool2d(F.pad(obs, (pad,pad,pad,pad), mode='reflect'), kernel_size, stride=1) * 9
            sm = F.avg_pool2d(F.pad(m4, (pad,pad,pad,pad), mode='reflect'), kernel_size, stride=1) * 9
            interp = (sv / (sm + 1e-8)).squeeze(1)
            result[:, c] = cd * cm.unsqueeze(0) + interp * (1 - cm.unsqueeze(0))
        return result


def create_bayer_mosaic(clean_video, H, W, pattern='RGGB', device='cuda'):
    op = DemosaicingOperator(H, W, pattern=pattern, device=device)
    mos = op.forward(clean_video)
    naive = op.mosaic_to_rgb_nearest(mos)
    return op, mos, naive


# ============================================================
# Super-Resolution Operator
# ============================================================

class SuperResolutionOperator(LinearOperator):
    """
    Super-resolution: y = downsample(x) via average pooling.
    Adjoint: nearest-neighbor upsample.
    project(x) = upsample(downsample(x)) — approximate projection.
    """

    def __init__(self, scale_factor: int = 2, device: str = 'cuda'):
        self.device = device
        self.scale_factor = scale_factor

    def forward(self, x):
        if x.dim() == 3:
            return F.avg_pool2d(x.unsqueeze(0), self.scale_factor).squeeze(0)
        return F.avg_pool2d(x, self.scale_factor)

    def adjoint(self, y):
        if y.dim() == 3:
            return F.interpolate(y.unsqueeze(0), scale_factor=self.scale_factor,
                                 mode='nearest').squeeze(0)
        return F.interpolate(y, scale_factor=self.scale_factor, mode='nearest')

    def project(self, x):
        return self.adjoint(self.forward(x))

    def null_project(self, x):
        return x - self.project(x)


def create_sr_degradation(clean_video, scale_factor=2, device='cuda'):
    op = SuperResolutionOperator(scale_factor=scale_factor, device=device)
    lr = op.forward(clean_video)
    lr_up = op.adjoint(lr)
    if lr.dim() == 4:
        bi_up = F.interpolate(lr, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    else:
        bi_up = F.interpolate(lr.unsqueeze(0), scale_factor=scale_factor,
                              mode='bilinear', align_corners=False).squeeze(0)
    return op, lr, lr_up, bi_up


# ============================================================
# Deblurring Operator
# ============================================================

class DeblurringOperator(LinearOperator):
    """
    Deblurring operator: y = blur(x) = conv(x, kernel).
    Forward: convolution with a known blur kernel.
    Adjoint: correlation = convolution with flipped kernel.
    NOT a projection — use PnPSolver.
    """

    def __init__(self, kernel: torch.Tensor, device: str = 'cuda'):
        self.device = device
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.kernel = (kernel / (kernel.sum() + 1e-8)).to(device)
        self.padding = (self.kernel.shape[2] // 2, self.kernel.shape[3] // 2)

    def forward(self, x):
        if x.dim() == 3:
            C, H, W = x.shape
            return F.conv2d(x.reshape(C, 1, H, W), self.kernel,
                            padding=self.padding).reshape(C, H, W)
        T, C, H, W = x.shape
        return F.conv2d(x.reshape(T * C, 1, H, W), self.kernel,
                        padding=self.padding).reshape(T, C, H, W)

    def adjoint(self, y):
        k_flip = torch.flip(self.kernel, [2, 3])
        if y.dim() == 3:
            C, H, W = y.shape
            return F.conv2d(y.reshape(C, 1, H, W), k_flip,
                            padding=self.padding).reshape(C, H, W)
        T, C, H, W = y.shape
        return F.conv2d(y.reshape(T * C, 1, H, W), k_flip,
                        padding=self.padding).reshape(T, C, H, W)


def create_gaussian_blur_kernel(kernel_size, sigma=1.0):
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


def create_motion_blur_kernel(kernel_size, angle=0.0):
    kernel = torch.zeros(kernel_size, kernel_size)
    center = kernel_size // 2
    for i in range(kernel_size):
        offset = i - center
        xi = int(center + offset * np.cos(np.radians(angle)))
        yi = int(center + offset * np.sin(np.radians(angle)))
        if 0 <= xi < kernel_size and 0 <= yi < kernel_size:
            kernel[yi, xi] = 1.0
    return (kernel / (kernel.sum() + 1e-8)).unsqueeze(0).unsqueeze(0)


def create_blur_degradation(clean_video, kernel, device='cuda'):
    operator = DeblurringOperator(kernel, device=device)
    blurred = operator.forward(clean_video)
    naive_deblurred = torch.clamp(operator.adjoint(blurred), 0, 1)
    return operator, blurred, naive_deblurred


# ============================================================
# Inpainting Operator
# ============================================================

class InpaintingOperator(LinearOperator):
    """
    Inpainting operator: y = mask * x.

    A binary mask where 1 = observed (kept) and 0 = missing (to be filled in).
    The missing region is typically a centered rectangle, as in Kadkhodaie (2021).

    The mask has shape (1, 1, H, W) — same mask applied to all 3 RGB channels.
    This is an exact projection: project(x) = mask * x, applying it twice
    gives the same result. The KadkhodaieSolver works directly.

    The algorithm keeps the observed pixels fixed and uses the denoiser's
    implicit prior to fill in the missing rectangle with natural-looking content.
    """

    def __init__(self, mask: torch.Tensor, device: str = 'cuda'):
        """
        Args:
            mask: Binary mask. Shape (H, W), (1, 1, H, W), or (1, C, H, W).
                  1 = observed, 0 = missing.
            device: 'cuda' or 'cpu'
        """
        self.device = device
        if mask.dim() == 2:
            self.mask = mask.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        elif mask.dim() == 3:
            self.mask = mask.unsqueeze(0).to(device)
        else:
            self.mask = mask.to(device)

    def forward(self, x):
        """Apply mask: zero out missing pixels."""
        return x * self.mask

    def adjoint(self, y):
        """Adjoint is the same for a diagonal binary operator."""
        return y * self.mask

    def project(self, x):
        """Project onto observed pixels."""
        return x * self.mask

    def null_project(self, x):
        """Project onto missing pixels."""
        return x * (1.0 - self.mask)


def create_center_mask(H: int, W: int, hole_h: int = 128, hole_w: int = 128,
                        device: str = 'cuda') -> torch.Tensor:
    """
    Create a mask with a centered rectangular hole.

    Args:
        H, W: Frame dimensions
        hole_h, hole_w: Size of the missing rectangle
        device: 'cuda' or 'cpu'

    Returns:
        mask: (1, 1, H, W) float tensor. 1 = observed, 0 = missing.
    """
    mask = torch.ones(1, 1, H, W)
    y1 = (H - hole_h) // 2
    x1 = (W - hole_w) // 2
    mask[:, :, y1:y1 + hole_h, x1:x1 + hole_w] = 0
    return mask.to(device)


def create_inpainting_degradation(clean_video: torch.Tensor, H: int, W: int,
                                    hole_h: int = 128, hole_w: int = 128,
                                    device: str = 'cuda'):
    """
    Convenience function: create an inpainting degradation with a center hole.

    Args:
        clean_video: (T, C, H, W) in [0, 1]
        H, W: Frame dimensions
        hole_h, hole_w: Size of missing rectangle
        device: 'cuda' or 'cpu'

    Returns:
        operator: InpaintingOperator instance
        masked_video: (T, C, H, W) — observation with hole zeroed out
        mask: (1, 1, H, W) — the binary mask
    """
    mask = create_center_mask(H, W, hole_h, hole_w, device=device)
    operator = InpaintingOperator(mask, device=device)
    masked_video = operator.forward(clean_video)
    return operator, masked_video, mask