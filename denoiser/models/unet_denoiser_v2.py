import torch
import torch.nn as nn


class ConvBlockNoBias(nn.Module):
    """Convolutional block without bias for blind denoising."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DownsampleBlock(nn.Module):
    """Downsampling block: conv -> relu -> maxpool."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlockNoBias(in_channels, out_channels)
        self.conv2 = ConvBlockNoBias(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpsampleBlock(nn.Module):
    """Upsampling block: upsample -> concat skip -> conv."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlockNoBias(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlockNoBias(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BlindVideoDenoiserUNet(nn.Module):
    """
    Bias-free UNet denoiser for blind video denoising with temporal fusion.

    Architecture (inspired by FastDVDNet's two-stage approach):
    1. Shared spatial encoder: Processes each of the 5 input frames independently
       through 2 downsampling stages with shared weights. This extracts per-frame
       spatial features without temporal mixing, keeping parameter count low.
    2. Temporal fusion: Concatenates all 5 frames' features and fuses them with
       a convolution layer. This is done at 1/4 resolution (after 2 downsamples),
       making the fusion computationally cheap.
    3. Deep encoder: 3 more downsampling stages on the fused features for a total
       of 5 stages of spatial hierarchy.
    4. Decoder: Standard UNet decoder with skip connections back to the fused
       encoder features. Residual learning predicts noise to subtract from the
       noisy center frame.

    Args:
        num_input_frames: Number of input frames (default 5)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel width (32 gives ~4.7M params)
    """
    def __init__(self, num_input_frames=5, out_channels=3, base_channels=32):
        super().__init__()
        self.num_input_frames = num_input_frames
        b = base_channels

        # --- Shared spatial encoder (applied to each frame independently) ---
        # Input: (B, 3, H, W) per frame → output: (B, b*2, H/4, W/4) per frame
        self.spatial_down1 = DownsampleBlock(3, b)         # → (b, H/2, W/2)
        self.spatial_down2 = DownsampleBlock(b, b * 2)     # → (b*2, H/4, W/4)

        # --- Temporal fusion ---
        # Concatenate all frames: (B, num_frames * b*2, H/4, W/4) → (B, b*4, H/4, W/4)
        fusion_in = num_input_frames * b * 2
        self.fusion = nn.Sequential(
            ConvBlockNoBias(fusion_in, b * 4),
            ConvBlockNoBias(b * 4, b * 4),
        )

        # --- Deep encoder (after fusion, 3 more stages) ---
        self.down3 = DownsampleBlock(b * 4, b * 4)   # → (b*4, H/8, W/8)
        self.down4 = DownsampleBlock(b * 4, b * 4)   # → (b*4, H/16, W/16)
        self.down5 = DownsampleBlock(b * 4, b * 8)   # → (b*8, H/32, W/32)

        # --- Bottleneck ---
        self.bottle_conv1 = ConvBlockNoBias(b * 8, b * 8)
        self.bottle_conv2 = ConvBlockNoBias(b * 8, b * 8)

        # --- Decoder ---
        self.up5 = UpsampleBlock(b * 8, b * 8, b * 4)   # + skip5 → (b*4, H/16)
        self.up4 = UpsampleBlock(b * 4, b * 4, b * 4)   # + skip4 → (b*4, H/8)
        self.up3 = UpsampleBlock(b * 4, b * 4, b * 2)   # + skip3 → (b*2, H/4)
        self.up2 = UpsampleBlock(b * 2, b * 2, b)       # + skip2 → (b, H/2)
        self.up1 = UpsampleBlock(b, b, b)                # + skip1 → (b, H)

        # --- Output (no bias, no activation for residual prediction) ---
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, num_input_frames * 3, H, W) — concatenated RGB frames.
                For 5 frames: (B, 15, H, W). Center frame is at channels 6:9.

        Returns:
            Denoised center frame: (B, 3, H, W)
        """
        B = x.shape[0]
        center_idx = self.num_input_frames // 2  # = 2 for 5 frames

        # Extract noisy center frame for residual learning
        noisy_center = x[:, center_idx * 3:(center_idx + 1) * 3, :, :]

        # --- Shared spatial encoder: process each frame independently ---
        # Split input into individual frames: list of (B, 3, H, W)
        frames = [x[:, i * 3:(i + 1) * 3, :, :] for i in range(self.num_input_frames)]

        # Encode each frame with shared weights
        frame_features = []
        skip1_center = None
        skip2_center = None

        for i, frame in enumerate(frames):
            f1, s1 = self.spatial_down1(frame)   # (B, b, H/2, W/2)
            f2, s2 = self.spatial_down2(f1)      # (B, b*2, H/4, W/4)
            frame_features.append(f2)

            # Keep skip connections from center frame only
            if i == center_idx:
                skip1_center = s1  # (B, b, H, W)
                skip2_center = s2  # (B, b*2, H/2, W/2)

        # --- Temporal fusion ---
        # Concatenate all frame features along channel dim
        fused = torch.cat(frame_features, dim=1)  # (B, num_frames * b*2, H/4, W/4)
        fused = self.fusion(fused)                 # (B, b*4, H/4, W/4)

        # --- Deep encoder ---
        x3, skip3 = self.down3(fused)    # (B, b*4, H/8, W/8)
        x4, skip4 = self.down4(x3)      # (B, b*4, H/16, W/16)
        x5, skip5 = self.down5(x4)      # (B, b*8, H/32, W/32)

        # --- Bottleneck ---
        x = self.bottle_conv1(x5)
        x = self.bottle_conv2(x)

        # --- Decoder ---
        x = self.up5(x, skip5)          # (B, b*4, H/16, W/16)
        x = self.up4(x, skip4)          # (B, b*4, H/8, W/8)
        x = self.up3(x, skip3)          # (B, b*2, H/4, W/4)
        x = self.up2(x, skip2_center)   # (B, b, H/2, W/2)
        x = self.up1(x, skip1_center)   # (B, b, H, W)

        # --- Output: predicted noise residual ---
        residual = self.out_conv(x)

        # Subtract predicted noise from noisy center frame
        return noisy_center - residual


# Example usage
if __name__ == "__main__":
    model = BlindVideoDenoiserUNet(num_input_frames=5, base_channels=32)
    print("5-frame Temporal Fusion UNet (5 stages, base=32):")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass — 5 frames = 15 input channels
    x = torch.randn(2, 15, 256, 256)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify input must be divisible by 32 (5 downsampling stages)
    x_small = torch.randn(1, 15, 128, 128)
    y_small = model(x_small)
    print(f"\nSmall input:  {x_small.shape} → {y_small.shape}")

    # Print architecture summary
    print(f"\nArchitecture:")
    print(f"  Spatial encoder: shared across {model.num_input_frames} frames")
    print(f"  Temporal fusion at 1/4 resolution")
    print(f"  5 total downsampling stages")
    print(f"  Residual learning (predicts noise)")