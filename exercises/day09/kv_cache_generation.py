from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from typing import TypeAlias

import torch
from torch import Tensor
from torch.nn import functional as F

from exercises.day04.decoder_block_training import CausalSelfAttention, TinyDecoderLM
from exercises.day08.naive_autoregressive_generation import (
    make_model,
    naive_greedy_generate,
)


LayerKVCache: TypeAlias = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class CachedStep:
    step: int
    phase: str
    input_length: int
    cache_length: int
    attention_score_elements: int
    maximum_logit_difference: float
    next_token_id: int


def cached_attention(
    attention: CausalSelfAttention,
    hidden: Tensor,
    past_key_value: LayerKVCache | None,
) -> tuple[Tensor, LayerKVCache, Tensor]:
    batch, query_length, hidden_size = hidden.shape
    query = attention._split_heads(attention.q_proj(hidden))
    new_key = attention._split_heads(attention.k_proj(hidden))
    new_value = attention._split_heads(attention.v_proj(hidden))

    if past_key_value is None:
        past_length = 0
        key = new_key
        value = new_value
    else:
        past_key, past_value = past_key_value
        assert past_key.shape == past_value.shape
        assert past_key.shape[:2] == (batch, attention.num_heads)
        assert past_key.shape[3] == attention.head_dim
        past_length = past_key.shape[2]
        key = torch.cat((past_key, new_key), dim=2)
        value = torch.cat((past_value, new_value), dim=2)

    total_key_length = key.shape[2]
    scores = query @ key.transpose(-2, -1) / sqrt(attention.head_dim)

    query_positions = past_length + torch.arange(
        query_length, device=hidden.device
    )
    key_positions = torch.arange(total_key_length, device=hidden.device)
    causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    attended = weights @ value
    merged = (
        attended.transpose(1, 2)
        .contiguous()
        .view(batch, query_length, hidden_size)
    )
    return attention.o_proj(merged), (key, value), weights


@torch.no_grad()
def cached_forward(
    model: TinyDecoderLM,
    input_ids: Tensor,
    past_key_values: list[LayerKVCache] | None,
) -> tuple[Tensor, list[LayerKVCache], list[Tensor]]:
    batch, query_length = input_ids.shape
    if past_key_values is None:
        past_length = 0
    else:
        if len(past_key_values) != model.config.num_layers:
            raise ValueError("cache must contain one K/V pair per decoder layer")
        layer_lengths = {key.shape[2] for key, _ in past_key_values}
        if len(layer_lengths) != 1:
            raise ValueError("all decoder-layer caches must have the same length")
        past_length = next(iter(layer_lengths))

    if past_length + query_length > model.config.max_sequence_length:
        raise ValueError("cached sequence exceeds the model context length")

    positions = torch.arange(
        past_length,
        past_length + query_length,
        device=input_ids.device,
    )
    hidden = model.token_embedding(input_ids) + model.position_embedding(positions)

    new_past_key_values: list[LayerKVCache] = []
    all_attention_weights: list[Tensor] = []
    for layer_index, block in enumerate(model.blocks):
        layer_past = (
            None if past_key_values is None else past_key_values[layer_index]
        )
        attention_update, layer_cache, weights = cached_attention(
            block.attention,
            block.attention_norm(hidden),
            layer_past,
        )
        hidden = hidden + attention_update
        hidden = hidden + block.feed_forward(block.ffn_norm(hidden))
        new_past_key_values.append(layer_cache)
        all_attention_weights.append(weights)

    logits = model.lm_head(model.final_norm(hidden))
    assert logits.shape == (
        batch,
        query_length,
        model.config.vocabulary_size,
    )
    return logits, new_past_key_values, all_attention_weights


@torch.no_grad()
def cached_greedy_generate(
    model: TinyDecoderLM,
    prompt_ids: Tensor,
    maximum_new_tokens: int,
    *,
    compare_with_no_cache: bool = False,
    record_trace: bool = True,
) -> tuple[Tensor, list[LayerKVCache], list[CachedStep]]:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("this teaching example expects prompt shape [1, sequence]")
    if maximum_new_tokens < 1:
        raise ValueError("maximum_new_tokens must be positive")
    if prompt_ids.shape[1] + maximum_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt plus generated tokens exceeds the model context length")

    generated_ids = prompt_ids.clone()
    model_input = prompt_ids
    past_key_values: list[LayerKVCache] | None = None
    trace: list[CachedStep] = []

    for step in range(maximum_new_tokens):
        previous_cache = past_key_values
        cached_logits, past_key_values, attention_weights = cached_forward(
            model, model_input, past_key_values
        )
        cached_last_logits = cached_logits[:, -1, :]

        maximum_difference = 0.0
        if compare_with_no_cache:
            full_logits, _ = model(generated_ids)
            full_last_logits = full_logits[:, -1, :]
            maximum_difference = (
                cached_last_logits - full_last_logits
            ).abs().max().item()
            torch.testing.assert_close(
                cached_last_logits,
                full_last_logits,
                rtol=1e-5,
                atol=1e-5,
            )

        next_token = cached_last_logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)

        if record_trace:
            cache_length = past_key_values[0][0].shape[2]
            for layer_index, (key, value) in enumerate(past_key_values):
                expected_shape = (
                    prompt_ids.shape[0],
                    model.config.num_heads,
                    cache_length,
                    model.config.head_dim,
                )
                assert key.shape == value.shape == expected_shape
                if previous_cache is not None:
                    old_key, old_value = previous_cache[layer_index]
                    torch.testing.assert_close(
                        key[:, :, : old_key.shape[2], :],
                        old_key,
                        rtol=0.0,
                        atol=0.0,
                    )
                    torch.testing.assert_close(
                        value[:, :, : old_value.shape[2], :],
                        old_value,
                        rtol=0.0,
                        atol=0.0,
                    )

            trace.append(
                CachedStep(
                    step=step,
                    phase="prefill" if step == 0 else "decode",
                    input_length=model_input.shape[1],
                    cache_length=cache_length,
                    attention_score_elements=sum(
                        weights.numel() for weights in attention_weights
                    ),
                    maximum_logit_difference=maximum_difference,
                    next_token_id=next_token.item(),
                )
            )
        model_input = next_token

    assert past_key_values is not None
    return generated_ids, past_key_values, trace


def benchmark(
    generation_function,
    model: TinyDecoderLM,
    prompt_ids: Tensor,
    maximum_new_tokens: int,
    warmup_steps: int = 5,
    measured_steps: int = 20,
) -> tuple[float, int | None]:
    for _ in range(warmup_steps):
        generation_function(model, prompt_ids, maximum_new_tokens)

    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize(prompt_ids.device)
        torch.cuda.reset_peak_memory_stats(prompt_ids.device)

    start = time.perf_counter()
    for _ in range(measured_steps):
        generation_function(model, prompt_ids, maximum_new_tokens)
    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize(prompt_ids.device)
    elapsed = time.perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated(prompt_ids.device)
        if prompt_ids.device.type == "cuda"
        else None
    )
    return elapsed * 1_000 / measured_steps, peak_allocated


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(device)
    prompt_ids = torch.tensor([[1, 5, 9, 4, 3]], device=device)
    maximum_new_tokens = 8

    no_cache_ids, _ = naive_greedy_generate(
        model, prompt_ids, maximum_new_tokens, record_trace=False
    )
    cached_ids, final_cache, trace = cached_greedy_generate(
        model,
        prompt_ids,
        maximum_new_tokens,
        compare_with_no_cache=True,
    )
    torch.testing.assert_close(cached_ids, no_cache_ids, rtol=0.0, atol=0.0)

    prompt_length = prompt_ids.shape[1]
    cached_token_positions = sum(item.input_length for item in trace)
    expected_cached_positions = prompt_length + maximum_new_tokens - 1
    assert cached_token_positions == expected_cached_positions

    cached_score_elements = sum(item.attention_score_elements for item in trace)
    expected_cached_score_elements = model.config.num_layers * model.config.num_heads * (
        prompt_length**2
        + sum(prompt_length + step for step in range(1, maximum_new_tokens))
    )
    assert cached_score_elements == expected_cached_score_elements

    cache_elements = sum(
        key.numel() + value.numel() for key, value in final_cache
    )
    cache_bytes = cache_elements * final_cache[0][0].element_size()
    expected_cache_bytes = (
        2
        * model.config.num_layers
        * prompt_ids.shape[0]
        * (prompt_length + maximum_new_tokens - 1)
        * model.config.num_heads
        * model.config.head_dim
        * final_cache[0][0].element_size()
    )
    assert cache_bytes == expected_cache_bytes

    no_cache_ms, no_cache_peak = benchmark(
        lambda current_model, current_prompt, count: naive_greedy_generate(
            current_model, current_prompt, count, record_trace=False
        ),
        model,
        prompt_ids,
        maximum_new_tokens,
    )
    cached_ms, cached_peak = benchmark(
        lambda current_model, current_prompt, count: cached_greedy_generate(
            current_model, current_prompt, count, record_trace=False
        ),
        model,
        prompt_ids,
        maximum_new_tokens,
    )

    maximum_logit_difference = max(
        item.maximum_logit_difference for item in trace
    )
    print(f"device:                         {device}")
    print(f"generated token IDs:            {cached_ids[0].tolist()}")
    print("cache trace:")
    for item in trace:
        print(
            f"  step={item.step} phase={item.phase:<7} "
            f"input_length={item.input_length:2d} "
            f"cache_length={item.cache_length:2d} "
            f"next_token={item.next_token_id:2d}"
        )
    print(f"final layer K shape:            {tuple(final_cache[0][0].shape)}")
    print(f"final layer V shape:            {tuple(final_cache[0][1].shape)}")
    print(f"cached token positions:         {cached_token_positions}")
    print(f"cached attention score elements:{cached_score_elements:8d}")
    print(f"final cache size:               {cache_bytes / 1024**2:.5f} MiB")
    print(f"maximum logit difference:       {maximum_logit_difference:.8g}")
    print(f"warmup steps per method:        5")
    print(f"measured steps per method:      20")
    print(f"no-cache average time:          {no_cache_ms:.3f} ms")
    print(f"KV-cache average time:          {cached_ms:.3f} ms")
    print(f"time ratio cache/no-cache:      {cached_ms / no_cache_ms:.3f}x")
    if no_cache_peak is not None and cached_peak is not None:
        print(f"no-cache CUDA peak allocated:   {no_cache_peak / 1024**2:.2f} MiB")
        print(f"KV-cache CUDA peak allocated:   {cached_peak / 1024**2:.2f} MiB")
    print("cache/no-cache token IDs:       EXACT MATCH")
    print("cache/no-cache logits:          CLOSE")
    print("old cache preservation:         PASS")
    print("cache shape/byte accounting:    PASS")


if __name__ == "__main__":
    main()
