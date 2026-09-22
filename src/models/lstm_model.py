"""
Navigators IDR - LSTM Velocity Estimator
Bidirectional LSTM with attention for velocity estimation from IMU sequences.

Compared to TCN:
    + Better for very long sequences (captures long-range dependencies)
    + Attention mechanism highlights relevant timesteps
    - Heavier compute (not ideal for real-time edge deployment)
    - Sequential nature limits parallelism

Input:  (batch, window_size, 6) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
Output: (batch, 2) - [v_north, v_east] predicted velocity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class TemporalAttention(nn.Module):
    """
    Scaled dot-product attention over time steps.
    Learns which parts of the IMU sequence are most informative
    for velocity estimation (e.g., acceleration phases vs. cruising).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size) LSTM hidden states.

        Returns:
            Tuple of:
                - context: (batch, hidden_size) weighted sum of hidden states
                - weights: (batch, seq_len) attention weights
        """

        scores = self.attention_weights(lstm_output).squeeze(-1)
        weights = F.softmax(scores, dim=1)


        context = torch.bmm(
            weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)

        return context, weights


class LSTMVelocityEstimator(nn.Module):
    """
    Bidirectional LSTM with temporal attention for velocity estimation.

    Architecture:
        Input → LayerNorm → Bi-LSTM (3 layers) → Attention → FC → Output

    The bidirectional LSTM processes the window in both directions,
    and the attention mechanism selectively focuses on the most
    informative timesteps for velocity prediction.
    """

    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 2,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        """
        Args:
            input_channels: Input feature dimension (6 for IMU).
            output_dim: Output dimension (2 for velocity).
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
            use_attention: Whether to use temporal attention.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1


        self.input_norm = nn.LayerNorm(input_channels)


        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )


        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions


        if use_attention:
            self.attention = TemporalAttention(lstm_output_size)


        self.head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for m in [self.input_proj, self.head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (batch, window_size, input_channels) IMU window data.
            return_attention: If True, also return attention weights.

        Returns:
            (batch, output_dim) predicted velocity.
            Optionally also (batch, window_size) attention weights.
        """
        batch_size = x.size(0)


        x = self.input_norm(x)


        x = self.input_proj(x)


        lstm_out, (h_n, c_n) = self.lstm(x)



        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:

            if self.bidirectional:

                h_forward = h_n[-2]
                h_backward = h_n[-1]
                context = torch.cat([h_forward, h_backward], dim=1)
            else:
                context = h_n[-1]
            attn_weights = None


        output = self.head(context)

        if return_attention and attn_weights is not None:
            return output, attn_weights
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
