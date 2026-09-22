"""Tests for TCN and LSTM velocity estimation models."""

import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.tcn_model import TCNVelocityEstimator
from models.lstm_model import LSTMVelocityEstimator


class TestTCNModel:
    def test_forward_pass_shape(self):
        model = TCNVelocityEstimator(input_channels=6, output_dim=2)
        x = torch.randn(4, 200, 6)
        out = model(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_forward_pass_various_window_sizes(self):
        model = TCNVelocityEstimator(input_channels=6, output_dim=2)
        for window_size in [50, 100, 200, 500]:
            x = torch.randn(2, window_size, 6)
            out = model(x)
            assert out.shape == (2, 2)

    def test_single_sample_inference(self):
        model = TCNVelocityEstimator()
        model.eval()
        x = torch.randn(1, 200, 6)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all()

    def test_parameter_count(self):
        model = TCNVelocityEstimator(
            num_channels=[64, 64, 128, 128],
            kernel_size=7,
        )
        params = model.count_parameters()
        assert params > 0
        assert params < 2_000_000, "Model should be lightweight (<2M params)"
        print(f"TCN parameters: {params:,}")

    def test_receptive_field(self):
        model = TCNVelocityEstimator(
            num_channels=[64, 64, 128, 128],
            kernel_size=7,
        )
        rf = model.get_receptive_field()
        assert rf > 0
        print(f"TCN receptive field: {rf} samples")

    def test_gradient_flow(self):
        model = TCNVelocityEstimator()
        x = torch.randn(4, 200, 6)
        y = torch.randn(4, 2)

        out = model(x)
        loss = torch.nn.MSELoss()(out, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient for {name}"

    def test_no_skip_connections(self):
        model = TCNVelocityEstimator(use_skip_connections=False)
        x = torch.randn(2, 200, 6)
        out = model(x)
        assert out.shape == (2, 2)


class TestLSTMModel:
    def test_forward_pass_shape(self):
        model = LSTMVelocityEstimator(input_channels=6, output_dim=2)
        x = torch.randn(4, 200, 6)
        out = model(x)
        assert out.shape == (4, 2)

    def test_attention_weights(self):
        model = LSTMVelocityEstimator(use_attention=True)
        model.eval()
        x = torch.randn(2, 100, 6)
        with torch.no_grad():
            out, attn = model(x, return_attention=True)
        assert out.shape == (2, 2)
        assert attn.shape == (2, 100)

        assert torch.allclose(attn.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_no_attention(self):
        model = LSTMVelocityEstimator(use_attention=False)
        x = torch.randn(2, 200, 6)
        out = model(x)
        assert out.shape == (2, 2)

    def test_unidirectional(self):
        model = LSTMVelocityEstimator(bidirectional=False)
        x = torch.randn(2, 200, 6)
        out = model(x)
        assert out.shape == (2, 2)

    def test_parameter_count(self):
        model = LSTMVelocityEstimator()
        params = model.count_parameters()
        assert params > 0
        print(f"LSTM parameters: {params:,}")

    def test_gradient_flow(self):
        model = LSTMVelocityEstimator()
        x = torch.randn(4, 100, 6)
        y = torch.randn(4, 2)

        out = model(x)
        loss = torch.nn.MSELoss()(out, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelInterchangeability:
    """Verify TCN and LSTM have the same interface."""

    @pytest.mark.parametrize("ModelClass", [TCNVelocityEstimator, LSTMVelocityEstimator])
    def test_same_io_signature(self, ModelClass):
        model = ModelClass(input_channels=6, output_dim=2)
        x = torch.randn(2, 200, 6)
        out = model(x)
        assert out.shape == (2, 2)

    @pytest.mark.parametrize("ModelClass", [TCNVelocityEstimator, LSTMVelocityEstimator])
    def test_has_count_parameters(self, ModelClass):
        model = ModelClass()
        assert hasattr(model, 'count_parameters')
        assert model.count_parameters() > 0
