"""
Navigators IDR - Temporal Convolutional Network (TCN)
Lightweight TCN for velocity estimation from windowed IMU data.

Architecture:
    - Dilated causal convolutions with exponentially increasing dilation
    - Residual connections (skip connections) for gradient flow
    - Designed for edge deployment (~500K parameters)

Input:  (batch, window_size, 6) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
Output: (batch, 2) - [v_north, v_east] predicted velocity
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CausalConv1d(nn.Module):
    """
    Causal convolution: output at time t only depends on inputs at time ≤ t.
    Achieved by left-padding the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        out = self.conv(x)
        # Remove the extra padding on the right (causal = no future leakage)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Single TCN residual block with two causal convolutions.

    Structure:
        x → CausalConv → BatchNorm → ReLU → Dropout →
            CausalConv → BatchNorm → ReLU → Dropout → (+x) → out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution for residual connection if channel dims differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNVelocityEstimator(nn.Module):
    """
    Temporal Convolutional Network for vehicle velocity estimation.

    Processes windowed IMU data (6 channels × window_size timesteps)
    to predict 2D velocity [v_north, v_east].

    Designed to be lightweight for edge deployment on smartphones.
    """

    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 2,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 7,
        dropout: float = 0.2,
        use_skip_connections: bool = True,
    ):
        """
        Args:
            input_channels: Number of input features (6 for IMU).
            output_dim: Output dimension (2 for [v_north, v_east]).
            num_channels: List of channel sizes for each TCN block.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            use_skip_connections: Whether to sum skip connections from all blocks.
        """
        super().__init__()

        if num_channels is None:
            num_channels = [64, 128, 256, 256, 512]

        self.use_skip = use_skip_connections
        self.num_blocks = len(num_channels)

        # Build TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i, out_ch in enumerate(num_channels):
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        # Skip connection projections (project each block output to same dim)
        if use_skip_connections:
            self.skip_projections = nn.ModuleList([
                nn.Conv1d(ch, num_channels[-1], 1) for ch in num_channels
            ])

        # Global average pooling + regression head
        final_channels = num_channels[-1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool across time
            nn.Flatten(),
            nn.Linear(final_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, window_size, input_channels) IMU window data.
               Note: Input is (batch, seq_len, channels) - we transpose internally.

        Returns:
            (batch, output_dim) predicted velocity.
        """
        # Transpose to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)

        if self.use_skip:
            skip_sum = 0
            out = x
            for block, skip_proj in zip(self.blocks, self.skip_projections):
                out = block(out)
                skip_sum = skip_sum + skip_proj(out)
            out = skip_sum
        else:
            out = x
            for block in self.blocks:
                out = block(out)

        return self.head(out)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_receptive_field(self) -> int:
        """
        Compute the effective receptive field of the TCN.
        RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
        """
        kernel_size = int(self.blocks[0].conv1.conv.kernel_size[0])  # type: ignore
        dilations = [2 ** i for i in range(self.num_blocks)]
        return 1 + 2 * (kernel_size - 1) * sum(dilations)
