"""Tests for MambaEncoder and its tokenizer.

Covers:
  - Tokenizer: shapes, padding, stride, all aggregation methods
  - tokenize_sequence convenience wrapper
  - MambaEncoder: forward shapes, pooling modes, parameter count, gradient flow
  - MambaEncoder.forward_sequence intermediate shapes
  - create_default_encoder factory
  - Edge-cases: short sequences, single-token, large batch
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.mamba_encoder import MambaEncoder, Tokenizer, tokenize_sequence, create_default_encoder


# ==========================================================================
# Tokenizer
# ==========================================================================

class TestTokenizer:
    # ---- "values" method (default) ----------------------------------------

    def test_values_shape_exact_fit(self):
        """Sequence length divisible by token_size → no padding needed.

        Use F=2 so that squeeze(2) does not collapse the feature dimension.
        With F=1 the tokenizer squeezes that dim and returns a 3-D tensor;
        the 4-D form is only produced when F > 1.
        """
        tk = Tokenizer(token_size=16, method="values")
        x = torch.randn(3, 2, 64)         # (B, F=2, T=64)
        out = tk(x)                        # swaps to (B, T, F) first
        # 64 timesteps / 16 token_size = 4 tokens
        # unfold → (3, 4, 2, 16); squeeze(2) no-ops since F=2 → 4-D
        assert out.ndim == 4
        B, n_tok, F, tok_len = out.shape
        assert B == 3
        assert n_tok == 4
        assert F == 2
        assert tok_len == 16

    def test_values_shape_with_padding(self):
        """Sequence not divisible → last token is zero-padded."""
        tk = Tokenizer(token_size=16, method="values")
        x = torch.randn(2, 1, 70)         # 70 → ceil((70-16)/16)+1 = 5 tokens
        out = tk(x)
        assert out.shape[1] == 5

    def test_pad_false_raises_on_short_sequence(self):
        tk = Tokenizer(token_size=32, pad=False)
        x = torch.randn(1, 1, 20)
        with pytest.raises(ValueError, match="pad=False"):
            tk(x)

    @pytest.mark.parametrize("method", ["mean", "max", "first"])
    def test_aggregated_methods_shape(self, method):
        """mean/max/first aggregate over the F dimension after unfold.

        With input (B=4, F=2, T=32) and token_size=8:
          swapaxes → (4, 32, 2)
          unfold(dim=1, size=8, step=8) → (4, 4, 2, 8)  [F=2: squeeze is no-op]
          .mean/.max/.first on dim=2 (the F dim) → (4, 4, 8)
        So the returned shape is (B, n_tokens, token_size).
        """
        tk = Tokenizer(token_size=8, method=method)
        x = torch.randn(4, 2, 32)         # 32/8 = 4 tokens, F=2
        out = tk(x)
        assert out.ndim == 3
        # mean is over the F dimension (not window); last dim = token_size
        assert out.shape == (4, 4, 8)

    def test_stride_overlapping(self):
        """stride < token_size → overlapping windows."""
        tk = Tokenizer(token_size=8, stride=4, method="values")
        x = torch.randn(1, 1, 20)
        out = tk(x)
        # (20 - 8) / 4 = 3, so 4 windows
        assert out.shape[1] == 4

    def test_invalid_token_size(self):
        with pytest.raises(ValueError):
            Tokenizer(token_size=0)

    def test_invalid_stride(self):
        with pytest.raises(ValueError):
            Tokenizer(token_size=8, stride=-1)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            Tokenizer(token_size=8, method="unknown")


class TestTokenizeSequenceWrapper:
    def test_matches_tokenizer_class(self):
        x = torch.randn(2, 1, 64)
        tk_out = Tokenizer(token_size=16, method="values")(x)
        fn_out = tokenize_sequence(x, token_size=16, method="values")
        assert torch.allclose(tk_out, fn_out)

    def test_default_no_overlap(self):
        x = torch.randn(1, 1, 32)
        out = tokenize_sequence(x, token_size=8)
        assert out.shape[1] == 4   # 32 / 8 = 4 tokens


# ==========================================================================
# MambaEncoder – forward shapes
# ==========================================================================

class TestMambaEncoderShapes:
    @pytest.fixture
    def encoder(self):
        return MambaEncoder(
            input_dim=16,
            model_dim=32,
            depth=2,
            state_dim=4,
            expand_factor=1.5,
            embedding_dim=64,
            pooling="mean",
        )

    def test_output_is_embedding_dim(self, encoder):
        """Encoder must return (batch, embedding_dim)."""
        x = torch.randn(4, 1, 96)        # (B, F, T)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_batch_size_1(self, encoder):
        x = torch.randn(1, 1, 96)
        assert encoder(x).shape == (1, 64)

    def test_large_batch(self, encoder):
        x = torch.randn(32, 1, 96)
        assert encoder(x).shape == (32, 64)

    def test_multi_feature_input(self):
        encoder = MambaEncoder(input_dim=16, model_dim=32, depth=1, embedding_dim=32, expand_factor=1.5)
        x = torch.randn(2, 3, 96)        # 3 features
        assert encoder(x).shape == (2, 32)

    def test_output_finite(self, encoder):
        x = torch.randn(2, 1, 96)
        out = encoder(x)
        assert torch.isfinite(out).all()


class TestMambaEncoderPooling:
    @pytest.mark.parametrize("pooling", ["mean", "last", "cls"])
    def test_pooling_modes(self, pooling):
        encoder = MambaEncoder(input_dim=16, model_dim=32, depth=1, embedding_dim=24, pooling=pooling, expand_factor=1.5)
        x = torch.randn(2, 1, 64)
        out = encoder(x)
        assert out.shape == (2, 24)


class TestMambaEncoderForwardSequence:
    def test_forward_sequence_shape(self):
        encoder = MambaEncoder(input_dim=16, model_dim=32, depth=2, embedding_dim=32, expand_factor=1.5)
        x = torch.randn(3, 1, 80)
        hidden = encoder.forward_sequence(x)
        # hidden: (B, n_tokens, model_dim)
        assert hidden.ndim == 3
        assert hidden.shape[0] == 3
        assert hidden.shape[2] == 32     # model_dim

    def test_forward_sequence_finite(self):
        encoder = MambaEncoder(input_dim=16, model_dim=32, depth=1, embedding_dim=32, expand_factor=1.5)
        x = torch.randn(2, 1, 48)
        hidden = encoder.forward_sequence(x)
        assert torch.isfinite(hidden).all()


class TestMambaEncoderGradients:
    def test_backward_does_not_raise(self):
        encoder = MambaEncoder(input_dim=8, model_dim=16, depth=1, embedding_dim=16, expand_factor=1.5)
        x = torch.randn(2, 1, 48)
        encoder(x).sum().backward()

    def test_trainable_params_receive_gradients(self):
        encoder = MambaEncoder(input_dim=8, model_dim=16, depth=1, embedding_dim=8, expand_factor=1.5)
        encoder(torch.randn(2, 1, 48)).sum().backward()
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for: {name}"


class TestMambaEncoderParameterCount:
    def test_count_parameters_positive(self):
        encoder = MambaEncoder(input_dim=16, model_dim=32, depth=2, embedding_dim=32, expand_factor=1.5)
        assert encoder.count_parameters() > 0

    def test_count_trainable_only(self):
        encoder = MambaEncoder(input_dim=16, model_dim=32, depth=2, embedding_dim=32, expand_factor=1.5)
        # Freeze all params and check count returns 0
        for p in encoder.parameters():
            p.requires_grad = False
        assert encoder.count_parameters(trainable_only=True) == 0

    def test_depth_increases_params(self):
        small = MambaEncoder(input_dim=8, model_dim=16, depth=1, embedding_dim=8, expand_factor=1.5)
        large = MambaEncoder(input_dim=8, model_dim=16, depth=4, embedding_dim=8, expand_factor=1.5)
        assert large.count_parameters() > small.count_parameters()


class TestMambaEncoderValidation:
    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="depth"):
            MambaEncoder(depth=0)

    def test_invalid_input_dim(self):
        with pytest.raises(ValueError, match="input_dim"):
            MambaEncoder(input_dim=0)


class TestCreateDefaultEncoder:
    def test_factory_returns_encoder(self):
        enc = create_default_encoder()
        assert isinstance(enc, MambaEncoder)

    def test_factory_overrides_embedding_dim(self):
        enc = create_default_encoder(embedding_dim=48)
        assert enc.embedding_dim == 48
