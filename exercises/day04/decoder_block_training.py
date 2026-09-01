"""Day 4: a tiny trainable decoder-only Transformer forward/backward pass.

Run with:
    uv run python exercises/day04/decoder_block_training.py

The script requires CUDA on purpose: Day 4 verifies the project GPU environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocabulary_size: int = 32
    hidden_size: int = 64
    num_heads: int = 4
    intermediate_size: int = 128
    num_layers: int = 2
    max_sequence_length: int = 16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def validate(self) -> None:
        assert self.hidden_size % self.num_heads == 0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        mean_square = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(mean_square + self.eps).to(hidden.dtype)
        return normalized * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _split_heads(self, hidden: Tensor) -> Tensor:
        batch, sequence, hidden_size = hidden.shape
        assert hidden_size == self.num_heads * self.head_dim
        return hidden.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, sequence, hidden_size = hidden.shape
        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(hidden))
        value = self._split_heads(self.v_proj(hidden))

        scores = query @ key.transpose(-2, -1) / sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(sequence, sequence, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = weights @ value

        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, hidden_size)
        return self.o_proj(merged), weights


class SwiGLU(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(gated)


class DecoderBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.feed_forward = SwiGLU(config)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        attention_update, weights = self.attention(self.attention_norm(hidden))
        hidden = hidden + attention_update
        hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
        return hidden, weights


class TinyDecoderLM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocabulary_size, bias=False
        )

    def forward(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        batch, sequence = input_ids.shape
        assert sequence <= self.config.max_sequence_length
        positions = torch.arange(sequence, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        all_attention_weights: list[Tensor] = []
        for block in self.blocks:
            hidden, weights = block(hidden)
            all_attention_weights.append(weights)

        logits = self.lm_head(self.final_norm(hidden))
        assert logits.shape == (batch, sequence, self.config.vocabulary_size)
        return logits, all_attention_weights


def shifted_language_model_batch(tokens: Tensor) -> tuple[Tensor, Tensor]:
    assert tokens.ndim == 2 and tokens.size(1) >= 2
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def assert_causality(model: TinyDecoderLM, input_ids: Tensor) -> None:
    model.eval()
    changed = input_ids.clone()
    changed[:, -1] = (changed[:, -1] + 7) % model.config.vocabulary_size
    with torch.no_grad():
        original_logits, _ = model(input_ids)
        changed_logits, _ = model(changed)
    torch.testing.assert_close(
        original_logits[:, :-1], changed_logits[:, :-1], rtol=0.0, atol=0.0
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Day 4 requires CUDA; run nvidia-smi and verify the cu130 torch build."
        )

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    device = torch.device("cuda")
    config = Config()
    model = TinyDecoderLM(config).to(device)

    # Each row is one complete token sequence.  Inputs and labels are shifted views.
    tokens = torch.tensor(
        [
            [1, 5, 9, 4, 3, 2],
            [1, 8, 6, 7, 3, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    input_ids, labels = shifted_language_model_batch(tokens)
    logits, attention_weights = model(input_ids)

    batch, sequence = input_ids.shape
    loss = F.cross_entropy(
        logits.reshape(batch * sequence, config.vocabulary_size),
        labels.reshape(batch * sequence),
    )
    loss.backward()
    torch.cuda.synchronize()

    assert input_ids.shape == labels.shape == (2, 5)
    assert logits.shape == (2, 5, config.vocabulary_size)
    assert len(attention_weights) == config.num_layers
    assert attention_weights[0].shape == (2, config.num_heads, 5, 5)
    assert torch.isfinite(loss)

    upper_triangle = torch.triu(
        attention_weights[0], diagonal=1
    )
    assert torch.count_nonzero(upper_triangle).item() == 0

    required_gradients = {
        "token_embedding": model.token_embedding.weight.grad,
        "q_proj": model.blocks[0].attention.q_proj.weight.grad,
        "gate_proj": model.blocks[0].feed_forward.gate_proj.weight.grad,
        "lm_head": model.lm_head.weight.grad,
    }
    assert all(gradient is not None for gradient in required_gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in required_gradients.values())
    assert all(gradient.norm().item() > 0 for gradient in required_gradients.values())

    assert_causality(model, input_ids)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    allocated_mib = torch.cuda.memory_allocated() / 1024**2
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2

    print("Environment")
    print(f"  torch:                 {torch.__version__}")
    print(f"  compiled CUDA:         {torch.version.cuda}")
    print(f"  device:                {torch.cuda.get_device_name(0)}")
    print("\nShapes")
    print(f"  tokens:                {tuple(tokens.shape)}")
    print(f"  input_ids / labels:    {tuple(input_ids.shape)}")
    print(f"  logits:                {tuple(logits.shape)}")
    print(f"  attention weights:     {tuple(attention_weights[0].shape)}")
    print("\nTraining evidence")
    print(f"  loss:                  {loss.item():.6f}")
    for name, gradient in required_gradients.items():
        assert gradient is not None
        print(f"  {name:20s} grad norm: {gradient.norm().item():.6f}")
    print(f"  parameter count:       {parameter_count:,}")
    print(f"  CUDA allocated:        {allocated_mib:.2f} MiB")
    print(f"  CUDA peak allocated:   {peak_allocated_mib:.2f} MiB")
    print("\nBehavior checks")
    print("  future attention weights are zero:                    PASS")
    print("  changing final input leaves all earlier logits equal: PASS")
    print("  forward, cross-entropy, and backward on CUDA:          PASS")


if __name__ == "__main__":
    main()
