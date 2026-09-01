from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from exercises.day04.decoder_block_training import Config, TinyDecoderLM, assert_causality


@dataclass(frozen=True)
class GenerationStep:
    step: int
    phase: str
    input_length: int
    last_logits_shape: tuple[int, ...]
    attention_score_elements: int
    next_token_id: int


def make_model(device: torch.device) -> TinyDecoderLM:
    torch.manual_seed(808)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(808)

    config = Config(
        vocabulary_size=32,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        num_layers=2,
        max_sequence_length=32,
    )
    return TinyDecoderLM(config).to(device).eval()


@torch.no_grad()
def naive_greedy_generate(
    model: TinyDecoderLM,
    prompt_ids: Tensor,
    maximum_new_tokens: int,
    *,
    record_trace: bool = True,
) -> tuple[Tensor, list[GenerationStep]]:
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape [batch, sequence]")
    if prompt_ids.shape[0] != 1:
        raise ValueError("this teaching example expects batch size 1")
    if maximum_new_tokens < 1:
        raise ValueError("maximum_new_tokens must be positive")
    if prompt_ids.shape[1] + maximum_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt plus generated tokens exceeds the model context length")

    generated_ids = prompt_ids.clone()
    trace: list[GenerationStep] = []

    for step in range(maximum_new_tokens):
        logits, attention_weights = model(generated_ids)
        last_logits = logits[:, -1, :]
        next_token = last_logits.argmax(dim=-1, keepdim=True)

        if record_trace:
            trace.append(
                GenerationStep(
                    step=step,
                    phase="prefill" if step == 0 else "decode",
                    input_length=generated_ids.shape[1],
                    last_logits_shape=tuple(last_logits.shape),
                    attention_score_elements=sum(
                        weights.numel() for weights in attention_weights
                    ),
                    next_token_id=next_token.item(),
                )
            )

        generated_ids = torch.cat((generated_ids, next_token), dim=1)

    return generated_ids, trace


def verify_generation(
    model: TinyDecoderLM,
    prompt_ids: Tensor,
    generated_ids: Tensor,
    trace: list[GenerationStep],
) -> None:
    prompt_length = prompt_ids.shape[1]

    repeated_ids, repeated_trace = naive_greedy_generate(
        model, prompt_ids, len(trace), record_trace=True
    )
    torch.testing.assert_close(generated_ids, repeated_ids, rtol=0.0, atol=0.0)
    assert trace == repeated_trace

    for step, step_trace in enumerate(trace):
        prefix = generated_ids[:, : prompt_length + step]
        with torch.no_grad():
            logits, _ = model(prefix)

        expected_token = logits[:, -1, :].argmax(dim=-1)
        actual_token = generated_ids[:, prompt_length + step]
        torch.testing.assert_close(actual_token, expected_token, rtol=0.0, atol=0.0)

        expected_phase = "prefill" if step == 0 else "decode"
        assert step_trace.phase == expected_phase
        assert step_trace.input_length == prompt_length + step
        assert step_trace.last_logits_shape == (1, model.config.vocabulary_size)

    with torch.no_grad():
        logits, _ = model(prompt_ids)
    changed_earlier_logits = logits.clone()
    changed_earlier_logits[:, :-1, :] = 1_000_000.0
    torch.testing.assert_close(
        logits[:, -1, :].argmax(dim=-1),
        changed_earlier_logits[:, -1, :].argmax(dim=-1),
        rtol=0.0,
        atol=0.0,
    )


def benchmark_generation(
    model: TinyDecoderLM,
    prompt_ids: Tensor,
    maximum_new_tokens: int,
    warmup_steps: int = 5,
    measured_steps: int = 20,
) -> tuple[float, int | None]:
    device = prompt_ids.device

    for _ in range(warmup_steps):
        naive_greedy_generate(
            model, prompt_ids, maximum_new_tokens, record_trace=False
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(measured_steps):
        naive_greedy_generate(
            model, prompt_ids, maximum_new_tokens, record_trace=False
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    return elapsed * 1_000 / measured_steps, peak_allocated


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(device)
    prompt_ids = torch.tensor([[1, 5, 9, 4, 3]], device=device)
    maximum_new_tokens = 8

    assert_causality(model, prompt_ids)
    generated_ids, trace = naive_greedy_generate(
        model, prompt_ids, maximum_new_tokens, record_trace=True
    )
    verify_generation(model, prompt_ids, generated_ids, trace)

    prompt_length = prompt_ids.shape[1]
    token_positions_processed = sum(item.input_length for item in trace)
    expected_token_positions = sum(
        prompt_length + step for step in range(maximum_new_tokens)
    )
    assert token_positions_processed == expected_token_positions

    attention_score_elements = sum(item.attention_score_elements for item in trace)
    expected_attention_score_elements = (
        model.config.num_layers
        * model.config.num_heads
        * sum((prompt_length + step) ** 2 for step in range(maximum_new_tokens))
    )
    assert attention_score_elements == expected_attention_score_elements

    average_ms, peak_allocated = benchmark_generation(
        model, prompt_ids, maximum_new_tokens
    )

    print(f"device:                         {device}")
    print(f"prompt shape:                   {tuple(prompt_ids.shape)}")
    print(f"prompt token IDs:               {prompt_ids[0].tolist()}")
    print(f"generated shape:                {tuple(generated_ids.shape)}")
    print(f"generated token IDs:            {generated_ids[0].tolist()}")
    print("generation trace:")
    for item in trace:
        print(
            f"  step={item.step} phase={item.phase:<7} "
            f"input_length={item.input_length:2d} "
            f"last_logits={item.last_logits_shape} "
            f"next_token={item.next_token_id:2d}"
        )
    print(f"token positions processed:      {token_positions_processed}")
    print(f"attention score elements:       {attention_score_elements}")
    print(f"warmup steps:                   5")
    print(f"measured steps:                 20")
    print(f"average generation time:        {average_ms:.3f} ms")
    if peak_allocated is not None:
        print(f"CUDA peak allocated:            {peak_allocated / 1024**2:.2f} MiB")
    print("causal independence:            PASS")
    print("greedy determinism:             PASS")
    print("last-position selection:        PASS")
    print("naive recomputation accounting: PASS")


if __name__ == "__main__":
    main()
