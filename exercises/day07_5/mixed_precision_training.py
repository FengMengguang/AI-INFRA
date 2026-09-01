"""Day 7.5: compare FP32 training with CUDA FP16 automatic mixed precision.

Run from the project root:
    uv run python exercises/day07_5/mixed_precision_training.py
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from exercises.day04.decoder_block_training import (  # noqa: E402
    Config,
    TinyDecoderLM,
    shifted_language_model_batch,
)


@dataclass(frozen=True)
class TrainingResult:
    name: str
    parameter_dtype: torch.dtype
    logits_dtype: torch.dtype
    loss_dtype: torch.dtype
    gradient_dtype: torch.dtype
    optimizer_state_dtype: torch.dtype
    initial_loss: float
    final_loss: float
    initial_scale: float
    final_scale: float
    average_step_ms: float
    peak_allocated_mib: float


def make_config() -> Config:
    return Config(
        vocabulary_size=512,
        hidden_size=128,
        num_heads=8,
        intermediate_size=384,
        num_layers=2,
        max_sequence_length=128,
    )


def make_batch(config: Config, batch_size: int) -> Tensor:
    generator = torch.Generator().manual_seed(755)
    return torch.randint(
        0,
        config.vocabulary_size,
        (batch_size, config.max_sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    )


def clone_state_dict(model: TinyDecoderLM) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def forward_loss(
    model: TinyDecoderLM,
    input_ids: Tensor,
    labels: Tensor,
    *,
    use_amp: bool,
) -> tuple[Tensor, Tensor]:
    with torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=use_amp
    ):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocabulary_size), labels.reshape(-1)
        )
    return logits, loss


@torch.no_grad()
def evaluate_loss(
    model: TinyDecoderLM,
    input_ids: Tensor,
    labels: Tensor,
    *,
    use_amp: bool,
) -> tuple[float, torch.dtype, torch.dtype]:
    model.eval()
    logits, loss = forward_loss(
        model, input_ids, labels, use_amp=use_amp
    )
    return loss.item(), logits.dtype, loss.dtype


def train_one_step(
    model: TinyDecoderLM,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler,
    input_ids: Tensor,
    labels: Tensor,
    *,
    use_amp: bool,
) -> torch.dtype:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = forward_loss(model, input_ids, labels, use_amp=use_amp)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    first_parameter = next(model.parameters())
    assert first_parameter.grad is not None
    assert torch.isfinite(first_parameter.grad).all()
    gradient_dtype = first_parameter.grad.dtype

    scaler.step(optimizer)
    scaler.update()
    return gradient_dtype


def run_training_mode(
    name: str,
    initial_state: dict[str, Tensor],
    config: Config,
    complete_sequences: Tensor,
    device: torch.device,
    *,
    use_amp: bool,
    warmup_steps: int,
    measured_steps: int,
) -> TrainingResult:
    model = TinyDecoderLM(config).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    input_ids, labels = shifted_language_model_batch(
        complete_sequences.to(device)
    )

    initial_loss, logits_dtype, loss_dtype = evaluate_loss(
        model, input_ids, labels, use_amp=use_amp
    )
    initial_scale = scaler.get_scale()

    gradient_dtype = torch.float32
    for _ in range(warmup_steps):
        gradient_dtype = train_one_step(
            model,
            optimizer,
            scaler,
            input_ids,
            labels,
            use_amp=use_amp,
        )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(measured_steps):
        gradient_dtype = train_one_step(
            model,
            optimizer,
            scaler,
            input_ids,
            labels,
            use_amp=use_amp,
        )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    final_loss, final_logits_dtype, final_loss_dtype = evaluate_loss(
        model, input_ids, labels, use_amp=use_amp
    )
    assert final_loss < initial_loss
    assert final_logits_dtype == logits_dtype
    assert final_loss_dtype == loss_dtype

    first_parameter = next(model.parameters())
    optimizer_state = optimizer.state[first_parameter]
    optimizer_state_dtype = optimizer_state["exp_avg"].dtype
    assert optimizer_state["exp_avg_sq"].dtype == optimizer_state_dtype

    result = TrainingResult(
        name=name,
        parameter_dtype=first_parameter.dtype,
        logits_dtype=logits_dtype,
        loss_dtype=loss_dtype,
        gradient_dtype=gradient_dtype,
        optimizer_state_dtype=optimizer_state_dtype,
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_scale=initial_scale,
        final_scale=scaler.get_scale(),
        average_step_ms=elapsed_seconds / measured_steps * 1000,
        peak_allocated_mib=peak_allocated_mib,
    )

    del model, optimizer, scaler, input_ids, labels
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_result(result: TrainingResult) -> None:
    print(f"\n{result.name}")
    print(f"  parameter dtype:          {result.parameter_dtype}")
    print(f"  logits dtype:             {result.logits_dtype}")
    print(f"  loss dtype:               {result.loss_dtype}")
    print(f"  gradient dtype:           {result.gradient_dtype}")
    print(f"  optimizer state dtype:    {result.optimizer_state_dtype}")
    print(f"  initial loss:             {result.initial_loss:.6f}")
    print(f"  final loss:               {result.final_loss:.6f}")
    print(f"  initial/final scale:      {result.initial_scale:.1f} / {result.final_scale:.1f}")
    print(f"  average measured step:    {result.average_step_ms:.3f} ms")
    print(f"  CUDA peak allocated:      {result.peak_allocated_mib:.2f} MiB")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 7.5 requires the project's CUDA PyTorch environment.")

    torch.manual_seed(750)
    device = torch.device("cuda")
    config = make_config()
    batch_size = 16
    warmup_steps = 5
    measured_steps = 20
    complete_sequences = make_batch(config, batch_size)

    initial_model = TinyDecoderLM(config).to(device)
    initial_state = clone_state_dict(initial_model)
    del initial_model
    torch.cuda.empty_cache()

    fp32_result = run_training_mode(
        "FP32 baseline",
        initial_state,
        config,
        complete_sequences,
        device,
        use_amp=False,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    fp16_result = run_training_mode(
        "FP16 AMP",
        initial_state,
        config,
        complete_sequences,
        device,
        use_amp=True,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )

    assert fp32_result.parameter_dtype == torch.float32
    assert fp16_result.parameter_dtype == torch.float32
    assert fp32_result.logits_dtype == torch.float32
    assert fp16_result.logits_dtype == torch.float16
    assert fp16_result.loss_dtype == torch.float32
    assert fp16_result.gradient_dtype == torch.float32
    assert fp16_result.optimizer_state_dtype == torch.float32

    native_bf16_supported = torch.cuda.is_bf16_supported(
        including_emulation=False
    )

    print("Environment and benchmark configuration")
    print(f"  torch:                    {torch.__version__}")
    print(f"  compiled CUDA:            {torch.version.cuda}")
    print(f"  device:                   {torch.cuda.get_device_name(device)}")
    print(f"  compute capability:       {torch.cuda.get_device_capability(device)}")
    print(f"  native BF16 supported:    {native_bf16_supported}")
    print(f"  batch / input length:     {batch_size} / {config.max_sequence_length}")
    print(f"  hidden / layers:          {config.hidden_size} / {config.num_layers}")
    print(f"  warmup / measured steps:  {warmup_steps} / {measured_steps}")
    print_result(fp32_result)
    print_result(fp16_result)

    speed_ratio = fp32_result.average_step_ms / fp16_result.average_step_ms
    memory_saved = fp32_result.peak_allocated_mib - fp16_result.peak_allocated_mib
    print("\nObserved comparison")
    print(f"  FP32 time / FP16 time:    {speed_ratio:.3f}x")
    print(f"  peak allocation saved:    {memory_saved:.2f} MiB")
    if speed_ratio > 1.0:
        print("  timing result:            FP16 AMP was faster in this run")
    else:
        print("  timing result:            FP16 AMP was not faster in this run")
    if not native_bf16_supported:
        print("  BF16 benchmark:           SKIPPED (no native device support)")

    print("\nBehavior checks")
    print("  both FP32 and FP16 losses decreased:       PASS")
    print("  autocast produced FP16 logits:             PASS")
    print("  cross-entropy stayed FP32 under autocast:  PASS")
    print("  parameters/gradients/AdamW state are FP32: PASS")
    print("  synchronized timing and peak memory read:  PASS")


if __name__ == "__main__":
    main()

