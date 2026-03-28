"""
Benchmark suite for evaluating video demosaicing.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class MetricsCalculator:
    """Calculate quality metrics for video restoration."""

    @staticmethod
    def psnr(target: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
        target = np.clip(target, 0, data_range)
        pred = np.clip(pred, 0, data_range)
        return psnr(target, pred, data_range=data_range)

    @staticmethod
    def ssim_score(target: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
        target = np.clip(target, 0, data_range)
        pred = np.clip(pred, 0, data_range)
        if target.ndim == 3:
            return ssim(target, pred, data_range=data_range, channel_axis=2)
        else:
            return ssim(target, pred, data_range=data_range)


class DemosaicingBenchmark:
    """Benchmark suite for video demosaicing solvers."""

    def __init__(self, results_dir: str = './benchmark_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_per_frame(
        self,
        clean_video: torch.Tensor,
        restored_video: torch.Tensor,
        baseline_video: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate demosaicing quality per-frame, then average.
        All inputs: (T, C, H, W) in [0, 1].
        """
        clean = clean_video.cpu().numpy().astype(np.float32)
        restored = restored_video.cpu().numpy().astype(np.float32)
        baseline = baseline_video.cpu().numpy().astype(np.float32)

        psnr_res, ssim_res = [], []
        psnr_base, ssim_base = [], []

        for t in range(clean.shape[0]):
            c = np.transpose(clean[t], (1, 2, 0))
            r = np.transpose(restored[t], (1, 2, 0))
            b = np.transpose(baseline[t], (1, 2, 0))

            psnr_res.append(MetricsCalculator.psnr(c, r))
            ssim_res.append(MetricsCalculator.ssim_score(c, r))
            psnr_base.append(MetricsCalculator.psnr(c, b))
            ssim_base.append(MetricsCalculator.ssim_score(c, b))

        return {
            'psnr_restored': np.mean(psnr_res),
            'psnr_restored_std': np.std(psnr_res),
            'ssim_restored': np.mean(ssim_res),
            'ssim_restored_std': np.std(ssim_res),
            'psnr_baseline': np.mean(psnr_base),
            'ssim_baseline': np.mean(ssim_base),
            'psnr_improvement': np.mean(psnr_res) - np.mean(psnr_base),
            'ssim_improvement': np.mean(ssim_res) - np.mean(ssim_base),
        }

    def visualize_demosaicing(
        self,
        clean: torch.Tensor,
        mosaiced: torch.Tensor,
        baseline: torch.Tensor,
        restored: torch.Tensor,
        frame_idx: int = 0,
        title: str = 'Demosaicing Result',
        save_path: Optional[str] = None,
    ):
        """Visualize a single frame: clean, mosaic, baseline, restored."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        imgs = [
            (clean[frame_idx], 'Clean (Ground Truth)'),
            (mosaiced[frame_idx], 'Mosaiced (CFA)'),
            (baseline[frame_idx], 'Naive Interpolation'),
            (restored[frame_idx], 'Solver Output'),
        ]

        for ax, (img, label) in zip(axes, imgs):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(label, fontweight='bold', fontsize=11)
            ax.axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def save_results(self, results_df: pd.DataFrame, name: str = 'demosaicing'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f'{name}_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        return csv_path