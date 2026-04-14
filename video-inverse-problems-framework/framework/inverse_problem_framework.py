"""
Inverse problem framework for videos based on:
"Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
by Kadkhodaie & Simoncelli (2021).

Two solvers:
- KadkhodaieSolver: For operators where project(x) is an exact/approximate projection
  (demosaicing, super-resolution). Uses the paper's direct update rule.
- PnPSolver: For operators where the adjoint doesn't form a clean projection
  (deblurring). Uses Plug-and-Play alternating between denoiser and data consistency,
  with sigma annealed from high to low (inspired by the same paper's philosophy).

CRITICAL NOTE ON MODEL OUTPUT CONVENTION:
The paper's model(y) returns the NOISE RESIDUAL (estimated noise).
Your UNet's forward() returns the DENOISED IMAGE (noisy_center - residual).
The KadkhodaieSolver handles this conversion internally.
The PnPSolver uses the denoised image directly (no conversion needed).
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import time


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        pass

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(x))

    def null_project(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.project(x)


class VideoDenoiser(ABC):
    @abstractmethod
    def denoise(self, noisy_video: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        pass

    @abstractmethod
    def denoise_frame(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor,
                      next_frame: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        pass


class KadkhodaieSolver:
    """
    Kadkhodaie & Simoncelli solver for operators with clean projections
    (demosaicing, super-resolution, inpainting).

    Uses the paper's direct update rule with noise residual conversion.
    """

    def __init__(self, operator, denoiser, device='cuda'):
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

        device = self.device
        x_c = y_measured.to(device)
        Mx_c = self.operator.adjoint(x_c)
        T, C, H, W = Mx_c.shape
        N = T * C * H * W

        e = torch.ones_like(Mx_c)
        y = self.operator.null_project(e) * 0.5 + Mx_c
        y = y + torch.randn_like(y) * sigma_0

        metrics = {'sigma': [], 'iteration': []}
        sigma = torch.tensor(sigma_0)
        t_iter = 1
        start = time.time()

        while sigma.item() > sigma_L and t_iter <= max_iterations:
            h = h0 * t_iter / (1 + h0 * (t_iter - 1))

            with torch.no_grad():
                denoised = self.denoiser.denoise(y, noise_std=sigma.item())
                f_y = y - denoised
                d = (f_y
                     - self.operator.project(f_y)
                     + self.operator.project(y)
                     - Mx_c)
                sigma = torch.norm(d) / np.sqrt(N)
                inner = (1 - beta * h) ** 2 - (1 - h) ** 2
                gamma = sigma.item() * np.sqrt(max(inner, 0))
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

        with torch.no_grad():
            reconstructed = self.denoiser.denoise(y, noise_std=sigma.item())
        reconstructed = torch.clamp(reconstructed, 0, 1)

        metrics['total_iterations'] = t_iter - 1
        metrics['total_time'] = total_time
        metrics['final_sigma'] = sigma.item()
        return reconstructed, metrics


class PnPSolver:
    """
    Plug-and-Play solver with annealed sigma for deblurring.

    Deblurring can't use the Kadkhodaie update rule directly because
    blur isn't a clean projection. Instead, we alternate:

    1. Denoiser step: x_den = D_sigma(x)
       Trust the denoiser prior at the current noise level.

    2. Data consistency step: gradient descent on ||A(x) - y||^2 + rho*||x - x_den||^2
       Push x to be consistent with the blurred observation while staying
       close to the denoiser's estimate.

    Sigma is annealed geometrically from sigma_0 to sigma_L, so early iterations
    make big structural changes (high sigma = denoiser sees heavy noise = acts boldly)
    and later iterations refine fine detail (low sigma = denoiser is conservative).

    This is mathematically equivalent to a half-quadratic splitting approach
    and shares the same philosophy as the Kadkhodaie paper: use the denoiser's
    implicit prior at decreasing noise levels to guide reconstruction.
    """

    def __init__(self, operator, denoiser, device='cuda'):
        self.operator = operator
        self.denoiser = denoiser
        self.device = device

    def solve(
        self,
        y_measured: torch.Tensor,
        sigma_0: float = 0.2,
        sigma_L: float = 0.005,
        num_iterations: int = 200,
        rho: float = 0.5,
        data_steps: int = 8,
        data_lr: float = 0.1,
        verbose: bool = True,
        log_freq: int = 20,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solve deblurring with PnP alternating minimization.

        Args:
            y_measured: (T, C, H, W) blurred observation
            sigma_0: Initial denoiser sigma (high → bold corrections)
            sigma_L: Final denoiser sigma (low → fine refinement)
            num_iterations: Number of outer (denoise + data) iterations
            rho: Balance between data fidelity and denoiser prior.
                 Higher rho = trust denoiser more. Lower = trust data more.
            data_steps: Gradient descent steps for data consistency per iteration
            data_lr: Learning rate for data consistency gradient descent
            verbose: Print progress
            log_freq: How often to print

        Returns:
            reconstructed: (T, C, H, W) deblurred video
            metrics: convergence info
        """
        device = self.device
        y_meas = y_measured.to(device)

        # Initialize with the blurred observation (identity initialization)
        x = y_meas.clone()

        # Geometric sigma schedule from sigma_0 to sigma_L
        sigmas = np.geomspace(sigma_0, sigma_L, num_iterations)

        metrics = {'sigma': [], 'data_loss': [], 'iteration': []}
        start = time.time()

        for i, sigma in enumerate(sigmas):
            with torch.no_grad():
                # Step 1: Denoiser — get a clean estimate at current noise level
                x_den = self.denoiser.denoise(x, noise_std=sigma)

                # Step 2: Data consistency — gradient descent on
                # ||A(z) - y||^2 + rho * ||z - x_den||^2
                z = x.clone()
                lr = data_lr / (1.0 + rho)

                for _ in range(data_steps):
                    # Gradient of ||A(z) - y||^2 w.r.t. z
                    residual = self.operator.forward(z) - y_meas
                    grad_data = self.operator.adjoint(residual)

                    # Gradient of rho * ||z - x_den||^2 w.r.t. z
                    grad_prior = rho * (z - x_den)

                    # Update
                    z = z - lr * (grad_data + grad_prior)

                x = z

            # Track metrics
            with torch.no_grad():
                data_loss = torch.mean((self.operator.forward(x) - y_meas) ** 2).item()

            metrics['sigma'].append(sigma)
            metrics['data_loss'].append(data_loss)
            metrics['iteration'].append(i)

            if verbose and (i + 1) % log_freq == 0:
                print(f"  Iter {i+1:4d}/{num_iterations} | sigma={sigma:.5f} | "
                      f"data_loss={data_loss:.6f}")

        x = torch.clamp(x, 0, 1)

        total_time = time.time() - start
        if verbose:
            print(f"  Done in {total_time:.1f}s ({num_iterations} iterations)")

        metrics['total_time'] = total_time
        metrics['total_iterations'] = num_iterations
        return x, metrics