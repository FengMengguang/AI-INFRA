from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionConfig:
    name: str
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_query_heads

    @property
    def queries_per_kv_head(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    def validate(self) -> None:
        assert self.hidden_size % self.num_query_heads == 0
        assert self.num_query_heads % self.num_kv_heads == 0


KVCache = tuple[Tensor, Tensor]


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_query_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    @staticmethod
    def _split_heads(projected: Tensor, num_heads: int) -> Tensor:
        batch, sequence, projected_size = projected.shape
        head_dim = projected_size // num_heads
        return projected.view(batch, sequence, num_heads, head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: Tensor,
        past_key_value: KVCache | None = None,
    ) -> tuple[Tensor, KVCache, Tensor, Tensor, Tensor]:
        batch, query_length, hidden_size = hidden.shape
        query = self._split_heads(
            self.q_proj(hidden), self.config.num_query_heads
        )
        new_key = self._split_heads(self.k_proj(hidden), self.config.num_kv_heads)
        new_value = self._split_heads(
            self.v_proj(hidden), self.config.num_kv_heads
        )

        if past_key_value is None:
            past_length = 0
            compact_key = new_key
            compact_value = new_value
        else:
            past_key, past_value = past_key_value
            past_length = past_key.shape[2]
            compact_key = torch.cat((past_key, new_key), dim=2)
            compact_value = torch.cat((past_value, new_value), dim=2)

        kv_head_for_query = torch.arange(
            self.config.num_query_heads, device=hidden.device
        ) // self.config.queries_per_kv_head
        expanded_key = compact_key[:, kv_head_for_query, :, :]
        expanded_value = compact_value[:, kv_head_for_query, :, :]

        scores = query @ expanded_key.transpose(-2, -1) / sqrt(
            self.config.head_dim
        )
        query_positions = past_length + torch.arange(
            query_length, device=hidden.device
        )
        key_positions = torch.arange(compact_key.shape[2], device=hidden.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attended = weights @ expanded_value
        merged = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, query_length, hidden_size)
        )
        output = self.o_proj(merged)
        return output, (compact_key, compact_value), weights, query, expanded_key


def cache_bytes(
    *,
    num_layers: int,
    batch_size: int,
    sequence_length: int,
    num_kv_heads: int,
    head_dim: int,
    element_size: int,
) -> int:
    return (
        2
        * num_layers
        * batch_size
        * sequence_length
        * num_kv_heads
        * head_dim
        * element_size
    )


@torch.no_grad()
def run_variant(
    config: AttentionConfig,
    hidden: Tensor,
    decode_hidden: Tensor,
) -> dict[str, object]:
    torch.manual_seed(1010 + config.num_kv_heads)
    if hidden.device.type == "cuda":
        torch.cuda.manual_seed_all(1010 + config.num_kv_heads)
    attention = GroupedQueryAttention(config).to(hidden.device).eval()

    prefill_output, prefill_cache, prefill_weights, prefill_query, expanded_key = (
        attention(hidden)
    )
    old_key, old_value = prefill_cache
    decode_output, decode_cache, decode_weights, decode_query, _ = attention(
        decode_hidden, prefill_cache
    )
    new_key, new_value = decode_cache

    batch, prefill_length, _ = hidden.shape
    expected_prefill_kv_shape = (
        batch,
        config.num_kv_heads,
        prefill_length,
        config.head_dim,
    )
    assert prefill_query.shape == (
        batch,
        config.num_query_heads,
        prefill_length,
        config.head_dim,
    )
    assert old_key.shape == old_value.shape == expected_prefill_kv_shape
    assert expanded_key.shape == (
        batch,
        config.num_query_heads,
        prefill_length,
        config.head_dim,
    )
    assert decode_query.shape == (
        batch,
        config.num_query_heads,
        1,
        config.head_dim,
    )
    assert new_key.shape == new_value.shape == (
        batch,
        config.num_kv_heads,
        prefill_length + 1,
        config.head_dim,
    )
    assert prefill_output.shape == hidden.shape
    assert decode_output.shape == decode_hidden.shape
    assert prefill_weights.shape == (
        batch,
        config.num_query_heads,
        prefill_length,
        prefill_length,
    )
    assert decode_weights.shape == (
        batch,
        config.num_query_heads,
        1,
        prefill_length + 1,
    )
    assert torch.count_nonzero(torch.triu(prefill_weights, diagonal=1)).item() == 0
    torch.testing.assert_close(
        new_key[:, :, :prefill_length, :], old_key, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        new_value[:, :, :prefill_length, :], old_value, rtol=0.0, atol=0.0
    )

    group_size = config.queries_per_kv_head
    for query_head in range(config.num_query_heads):
        source_kv_head = query_head // group_size
        torch.testing.assert_close(
            expanded_key[:, query_head],
            old_key[:, source_kv_head],
            rtol=0.0,
            atol=0.0,
        )

    projection_parameters = sum(
        parameter.numel() for parameter in attention.parameters()
    )
    theoretical_cache_bytes = cache_bytes(
        num_layers=2,
        batch_size=2,
        sequence_length=16,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        element_size=hidden.element_size(),
    )
    actual_one_layer_cache_bytes = (
        new_key.numel() + new_value.numel()
    ) * new_key.element_size()

    return {
        "name": config.name,
        "query_shape": tuple(prefill_query.shape),
        "compact_kv_shape": tuple(old_key.shape),
        "group_size": group_size,
        "projection_parameters": projection_parameters,
        "actual_one_layer_cache_bytes": actual_one_layer_cache_bytes,
        "theoretical_cache_bytes": theoretical_cache_bytes,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(10)
    hidden = torch.randn(2, 5, 64, device=device)
    decode_hidden = torch.randn(2, 1, 64, device=device)

    configs = [
        AttentionConfig("MHA", hidden_size=64, num_query_heads=8, num_kv_heads=8),
        AttentionConfig("GQA", hidden_size=64, num_query_heads=8, num_kv_heads=2),
        AttentionConfig("MQA", hidden_size=64, num_query_heads=8, num_kv_heads=1),
    ]
    results = [run_variant(config, hidden, decode_hidden) for config in configs]

    mha_cache_bytes = int(results[0]["theoretical_cache_bytes"])
    assert int(results[1]["theoretical_cache_bytes"]) * 4 == mha_cache_bytes
    assert int(results[2]["theoretical_cache_bytes"]) * 8 == mha_cache_bytes
    assert len({result["query_shape"] for result in results}) == 1

    print(f"device: {device}")
    print("fixed cache accounting: layers=2 batch=2 sequence=16 dtype=float32")
    for result in results:
        cache_size = int(result["theoretical_cache_bytes"])
        print(f"\n{result['name']}")
        print(f"  prefill Q shape:              {result['query_shape']}")
        print(f"  compact prefill K/V shape:    {result['compact_kv_shape']}")
        print(f"  query heads per KV head:      {result['group_size']}")
        print(f"  Q/K/V/O projection params:    {result['projection_parameters']}")
        print(
            "  one-layer cache after decode: "
            f"{int(result['actual_one_layer_cache_bytes'])} bytes"
        )
        print(f"  two-layer cache at S=16:      {cache_size} bytes")
        print(f"  capacity relative to MHA:     {cache_size / mha_cache_bytes:.3f}x")

    print("\nquery-head shape invariant:      PASS")
    print("compact KV-head mapping:         PASS")
    print("causal prefill weights:          PASS")
    print("old cache preservation:          PASS")
    print("cache capacity ratios:           PASS")


if __name__ == "__main__":
    main()
