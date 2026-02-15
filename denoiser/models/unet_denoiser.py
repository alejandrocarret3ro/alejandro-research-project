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
    """Upsampling block: upsample -> conv."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenation will add skip_channels to in_channels
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
    Bias-free UNet denoiser for blind video denoising.
    
    Takes 3 consecutive frames (shape: B x 9 x H x W) and outputs 1 denoised frame (B x 3 x H x W).
    Features:
    - No bias in convolutional layers (essential for blind denoising)
    - 3-4 downsampling stages
    - Skip connections for preserving detail
    - Suitable for diffusion-based inverse problem solving
    
    Args:
        in_channels: Number of input channels (9 for 3 RGB frames concatenated)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Number of channels in the first layer
        num_stages: Number of downsampling stages (3 or 4)
    """
    def __init__(self, in_channels=9, out_channels=3, base_channels=64, num_stages=3):
        super().__init__()
        assert num_stages in [3, 4], "num_stages must be 3 or 4"
        self.num_stages = num_stages
        
        # Encoder (downsampling path)
        self.down1 = DownsampleBlock(in_channels, base_channels)
        self.down2 = DownsampleBlock(base_channels, base_channels * 2)
        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4)
        
        if num_stages == 4:
            self.down4 = DownsampleBlock(base_channels * 4, base_channels * 8)
            # Bottleneck
            self.bottle_conv1 = ConvBlockNoBias(base_channels * 8, base_channels * 16)
            self.bottle_conv2 = ConvBlockNoBias(base_channels * 16, base_channels * 8)
        else:
            # Bottleneck (3 stages)
            self.bottle_conv1 = ConvBlockNoBias(base_channels * 4, base_channels * 8)
            self.bottle_conv2 = ConvBlockNoBias(base_channels * 8, base_channels * 4)
        
        # Decoder (upsampling path)
        if num_stages == 4:
            self.up4 = UpsampleBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        
        self.up3 = UpsampleBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up2 = UpsampleBlock(base_channels * 2, base_channels * 2, base_channels)
        self.up1 = UpsampleBlock(base_channels, base_channels, base_channels)
        
        # Final output layer (no bias, no activation for denoising residuals)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 9, H, W) containing 3 concatenated RGB frames
        
        Returns:
            output: Denoised frame of shape (B, 3, H, W)
        """
        # Encoder
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        if self.num_stages == 4:
            x4, skip4 = self.down4(x3)
            # Bottleneck
            x = self.bottle_conv1(x4)
            x = self.bottle_conv2(x)
            # Decoder
            x = self.up4(x, skip4)
        else:
            # Bottleneck
            x = self.bottle_conv1(x3)
            x = self.bottle_conv2(x)
        
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        # Output (no activation for blind denoising)
        x = self.out_conv(x)
        
        return x


# Example usage
if __name__ == "__main__":
    # Test with 3 stages
    model_3stage = BlindVideoDenoiserUNet(in_channels=9, out_channels=3, base_channels=64, num_stages=3)
    print("3-stage UNet:")
    print(model_3stage)
    print(f"Total parameters: {sum(p.numel() for p in model_3stage.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 9, 256, 256)  # B=2, 3 frames of RGB
    y = model_3stage(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}\n")
    
    # Test with 4 stages
    model_4stage = BlindVideoDenoiserUNet(in_channels=9, out_channels=3, base_channels=64, num_stages=4)
    print("4-stage UNet:")
    print(f"Total parameters: {sum(p.numel() for p in model_4stage.parameters()):,}")
    
    y = model_4stage(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")