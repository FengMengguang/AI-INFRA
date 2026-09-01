from __future__ import annotations

import gc
import time
import warnings
from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def naive_causal_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    sequence = query.shape[-2]
    scores = query @ key.transpose(-2, -1) / sqrt(query.shape[-1])
    causal_mask = torch.triu(
        torch.ones(sequence, sequence, device=query.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


def verify_sdpa_math(device: torch.device) -> float:
    torch.manual_seed(111)
    base_query = torch.randn(2, 4, 32, 16, device=device)
    base_key = torch.randn(2, 4, 32, 16, device=device)
    base_value = torch.randn(2, 4, 32, 16, device=device)

    naive_inputs = [tensor.clone().requires_grad_(True) for tensor in (
        base_query,
        base_key,
        base_value,
    )]
    sdpa_inputs = [tensor.clone().requires_grad_(True) for tensor in (
        base_query,
        base_key,
        base_value,
    )]

    naive_output = naive_causal_attention(*naive_inputs)
    with sdpa_kernel(SDPBackend.MATH):
        sdpa_output = F.scaled_dot_product_attention(
            *sdpa_inputs, is_causal=True
        )

    torch.testing.assert_close(naive_output, sdpa_output, rtol=1e-5, atol=1e-5)
    naive_output.square().mean().backward()
    sdpa_output.square().mean().backward()
    for naive_tensor, sdpa_tensor in zip(naive_inputs, sdpa_inputs, strict=True):
        torch.testing.assert_close(
            naive_tensor.grad,
            sdpa_tensor.grad,
            rtol=2e-5,
            atol=2e-6,
        )
    return (naive_output - sdpa_output).abs().max().item()


def probe_sdpa_backends(device: torch.device) -> dict[str, str]:
    torch.manual_seed(112)
    query = torch.randn(2, 8, 128, 32, device=device, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    backends = {
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
        "MATH": SDPBackend.MATH,
    }
    results: dict[str, str] = {}
    for name, backend in backends.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel(backend):
                    output = F.scaled_dot_product_attention(
                        query, key, value, is_causal=True
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            assert output.shape == query.shape and torch.isfinite(output).all()
            results[name] = "AVAILABLE"
        except RuntimeError as error:
            first_line = str(error).splitlines()[0]
            results[name] = f"UNAVAILABLE ({first_line})"
    return results


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.down(F.gelu(self.up(hidden)))


class CheckpointStack(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            FeedForwardBlock(hidden_size, intermediate_size)
            for _ in range(num_layers)
        )

    def forward(self, hidden: Tensor, use_checkpoint: bool) -> Tensor:
        for block in self.blocks:
            if use_checkpoint:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return hidden


@dataclass
class SavedTensorStats:
    count: int = 0
    logical_bytes: int = 0

    def pack(self, tensor: Tensor) -> Tensor:
        self.count += 1
        self.logical_bytes += tensor.numel() * tensor.element_size()
        return tensor

    @staticmethod
    def unpack(tensor: Tensor) -> Tensor:
        return tensor


@dataclass(frozen=True)
class TrainingResult:
    output: Tensor
    input_gradient: Tensor
    parameter_gradients: list[Tensor]
    saved_tensor_count: int
    saved_tensor_bytes: int
    peak_delta_bytes: int | None


def training_step(
    model: CheckpointStack,
    base_input: Tensor,
    use_checkpoint: bool,
) -> TrainingResult:
    model.zero_grad(set_to_none=True)
    hidden = base_input.detach().clone().requires_grad_(True)
    if hidden.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(hidden.device)
        baseline_allocated = torch.cuda.memory_allocated(hidden.device)
    else:
        baseline_allocated = 0

    stats = SavedTensorStats()
    with torch.autograd.graph.saved_tensors_hooks(stats.pack, stats.unpack):
        output = model(hidden, use_checkpoint=use_checkpoint)
        loss = output.square().mean()
    loss.backward()
    if hidden.device.type == "cuda":
        torch.cuda.synchronize(hidden.device)
        peak_delta = torch.cuda.max_memory_allocated(hidden.device) - baseline_allocated
    else:
        peak_delta = None

    return TrainingResult(
        output=output.detach().cpu(),
        input_gradient=hidden.grad.detach().cpu(),
        parameter_gradients=[
            parameter.grad.detach().cpu().clone() for parameter in model.parameters()
        ],
        saved_tensor_count=stats.count,
        saved_tensor_bytes=stats.logical_bytes,
        peak_delta_bytes=peak_delta,
    )


def verify_checkpointing(
    model: CheckpointStack,
    base_input: Tensor,
) -> tuple[TrainingResult, TrainingResult, float]:
    normal = training_step(model, base_input, use_checkpoint=False)
    checkpointed = training_step(model, base_input, use_checkpoint=True)

    torch.testing.assert_close(normal.output, checkpointed.output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        normal.input_gradient,
        checkpointed.input_gradient,
        rtol=1e-5,
        atol=1e-6,
    )
    maximum_gradient_difference = 0.0
    for normal_gradient, checkpoint_gradient in zip(
        normal.parameter_gradients,
        checkpointed.parameter_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            normal_gradient,
            checkpoint_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
        maximum_gradient_difference = max(
            maximum_gradient_difference,
            (normal_gradient - checkpoint_gradient).abs().max().item(),
        )
    assert checkpointed.saved_tensor_bytes < normal.saved_tensor_bytes
    return normal, checkpointed, maximum_gradient_difference


def benchmark_training(
    model: CheckpointStack,
    base_input: Tensor,
    use_checkpoint: bool,
    warmup_steps: int = 3,
    measured_steps: int = 10,
) -> float:
    def one_step() -> None:
        model.zero_grad(set_to_none=True)
        hidden = base_input.detach().clone().requires_grad_(True)
        output = model(hidden, use_checkpoint=use_checkpoint)
        output.square().mean().backward()

    for _ in range(warmup_steps):
        one_step()
    if base_input.device.type == "cuda":
        torch.cuda.synchronize(base_input.device)
    start = time.perf_counter()
    for _ in range(measured_steps):
        one_step()
    if base_input.device.type == "cuda":
        torch.cuda.synchronize(base_input.device)
    return (time.perf_counter() - start) * 1_000 / measured_steps


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum_attention_difference = verify_sdpa_math(device)
    backend_results = probe_sdpa_backends(device)

    torch.manual_seed(113)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(113)
    model = CheckpointStack().to(device)
    base_input = torch.randn(4, 128, 256, device=device)
    normal, checkpointed, maximum_gradient_difference = verify_checkpointing(
        model, base_input
    )

    normal_ms = benchmark_training(model, base_input, use_checkpoint=False)
    checkpoint_ms = benchmark_training(model, base_input, use_checkpoint=True)

    score_elements = 2 * 8 * 128 * 128
    score_bytes_fp16 = score_elements * 2
    print(f"device:                         {device}")
    print(f"torch:                          {torch.__version__}")
    print(f"explicit score shape:           (2, 8, 128, 128)")
    print(f"explicit score elements:        {score_elements}")
    print(f"explicit score size FP16:       {score_bytes_fp16 / 1024**2:.3f} MiB")
    print(f"naive/SDPA max output diff:     {maximum_attention_difference:.8g}")
    print("forced SDPA backend probe:")
    for name, result in backend_results.items():
        print(f"  {name:<22}{result}")

    print("\nactivation checkpointing")
    print(f"  input shape:                  {tuple(base_input.shape)}")
    print(f"  layers:                       {len(model.blocks)}")
    print(f"  normal saved tensors:         {normal.saved_tensor_count}")
    print(
        f"  normal saved logical bytes:   "
        f"{normal.saved_tensor_bytes / 1024**2:.3f} MiB"
    )
    print(f"  checkpoint saved tensors:     {checkpointed.saved_tensor_count}")
    print(
        f"  checkpoint saved logical bytes:"
        f"{checkpointed.saved_tensor_bytes / 1024**2:8.3f} MiB"
    )
    if normal.peak_delta_bytes is not None and checkpointed.peak_delta_bytes is not None:
        print(
            f"  normal CUDA peak delta:        "
            f"{normal.peak_delta_bytes / 1024**2:.3f} MiB"
        )
        print(
            f"  checkpoint CUDA peak delta:    "
            f"{checkpointed.peak_delta_bytes / 1024**2:.3f} MiB"
        )
    print(f"  normal average train step:    {normal_ms:.3f} ms")
    print(f"  checkpoint average train step:{checkpoint_ms:8.3f} ms")
    print(f"  time ratio checkpoint/normal: {checkpoint_ms / normal_ms:.3f}x")
    print(f"  maximum gradient difference:  {maximum_gradient_difference:.8g}")
    print("naive/SDPA output and gradients: CLOSE")
    print("checkpoint output:               EXACT MATCH")
    print("checkpoint gradients:            CLOSE")
    print("saved-tensor reduction:          PASS")

    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
