"""
Inverse problem framework for videos based on:
"Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
by Kadkhodaie & Simoncelli (2021).

CRITICAL NOTE ON MODEL OUTPUT CONVENTION:
The paper's model(y) returns the NOISE RESIDUAL (estimated noise).
Your UNet's forward() returns the DENOISED IMAGE (noisy_center - residual).
Therefore inside the solver we compute: f_y = y - denoise(y) to convert
from denoised image back to noise residual before applying the update rule.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import time


class LinearOperator(ABC):
    """
    Abstract base class for linear measurement operators.

    For the algorithm, the key property is:
    - project(x) = adjoint(forward(x)) is an exact projection
    - null_project(x) = x - project(x) is the complement
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator (degradation): y = A(x)."""
        pass

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint operator: x_approx = A^T(y)."""
        pass

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto measurement subspace: A^T(A(x))."""
        return self.adjoint(self.forward(x))

    def null_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto null space: (I - A^T A)(x)."""
        return x - self.project(x)


class VideoDenoiser(ABC):
    """
    Abstract base class for video denoisers.

    Requirements for the Kadkhodaie & Simoncelli algorithm:
    1. Trained to remove additive white Gaussian noise (MSE objective)
    2. "Blind" (universal) — works across a range of noise levels
    3. Returns the DENOISED IMAGE (not the noise residual)
    """

    @abstractmethod
    def denoise(self, noisy_video: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """
        Denoise a video.
        Args:
            noisy_video: (T, C, H, W) in [0, 1] range
            noise_std: noise level hint (blind denoisers may ignore this)
        Returns:
            denoised_video: (T, C, H, W) — the CLEAN image estimate
        """
        pass

    @abstractmethod
    def denoise_frame(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor,
                      next_frame: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """Denoise single frame with temporal context."""
        pass


class KadkhodaieSolver:
    """
    Implements the iterative algorithm from Kadkhodaie & Simoncelli (2021),
    adapted for video inverse problems.

    CRITICAL: The paper's model returns the noise residual.
    Our denoiser returns the clean image. So we convert:
        f_y = y - denoise(y)   (noise residual)
    before using the paper's update rule:
        d = f_y - project(f_y) + project(y) - Mx_c

    This ensures:
    - In the null space: d = y - denoise(y) → pushes y toward the denoiser prior
    - In the measured space: d = y - measurements → enforces data consistency
    """

    def __init__(
        self,
        operator: LinearOperator,
        denoiser: VideoDenoiser,
        device: str = 'cuda'
    ):
        self.operator = operator
        self.denoiser = denoiser
        self.device = device

    def solve(
        self,
        y_measured: torch.Tensor,
        sigma_0: float = 1.0,
        sigma_L: float = 0.01,
        h0: float = 0.01,
        beta: float = 0.01,
        max_iterations: int = 2000,
        verbose: bool = True,
        log_freq: int = 50,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solve inverse problem using denoiser-guided iteration.

        Args:
            y_measured: Degraded video (T, C, H, W) — the observations.
            sigma_0: Initial noise level (high, e.g. 1.0).
            sigma_L: Stopping threshold (low, e.g. 0.01).
            h0: Initial step size.
            beta: Noise injection control (0,1]. Lower = more noise injection.
            max_iterations: Safety limit on iterations.
            verbose: Print progress.
            log_freq: How often to log.

        Returns:
            reconstructed: Reconstructed clean video (T, C, H, W)
            metrics: Dict with convergence info
        """
        device = self.device
        x_c = y_measured.to(device)

        # Map measurements back to signal space
        Mx_c = self.operator.adjoint(x_c)
        T, C, H, W = Mx_c.shape
        N = T * C * H * W

        # Initialize: 0.5 (gray) in null space + measurements in observed space + noise
        e = torch.ones_like(Mx_c)
        y = self.operator.null_project(e) * 0.5 + Mx_c
        y = y + torch.randn_like(y) * sigma_0

        metrics = {'sigma': [], 'iteration': []}
        sigma = torch.tensor(sigma_0)
        t_iter = 1
        start = time.time()

        while sigma.item() > sigma_L and t_iter <= max_iterations:
            # Adaptive step size (harmonic schedule)
            h = h0 * t_iter / (1 + h0 * (t_iter - 1))

            with torch.no_grad():
                # Denoise current estimate (returns CLEAN IMAGE)
                denoised = self.denoiser.denoise(y, noise_std=sigma.item())

                # CRITICAL: convert to noise residual (paper's convention)
                f_y = y - denoised

                # Update direction (paper's equation)
                d = (f_y
                     - self.operator.project(f_y)
                     + self.operator.project(y)
                     - Mx_c)

                # Estimate current noise level
                sigma = torch.norm(d) / np.sqrt(N)

                # Noise injection amount
                inner = (1 - beta * h) ** 2 - (1 - h) ** 2
                gamma = sigma.item() * np.sqrt(max(inner, 0))

                # Update
                y = y - h * d + gamma * torch.randn_like(y)

            metrics['sigma'].append(sigma.item())
            metrics['iteration'].append(t_iter)

            if verbose and t_iter % log_freq == 0:
                print(f"  Iter {t_iter:4d} | sigma={sigma.item():.6f} | "
                      f"h={h:.4f} | gamma={gamma:.6f}")

            t_iter += 1

        total_time = time.time() - start
        if verbose:
            print(f"  Converged in {t_iter-1} iters ({total_time:.1f}s), "
                  f"final sigma={sigma.item():.6f}")

        # Final denoising
        with torch.no_grad():
            reconstructed = self.denoiser.denoise(y, noise_std=sigma.item())
        reconstructed = torch.clamp(reconstructed, 0, 1)

        metrics['total_iterations'] = t_iter - 1
        metrics['total_time'] = total_time
        metrics['final_sigma'] = sigma.item()

        return reconstructed, metrics