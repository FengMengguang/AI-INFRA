"""Day 12: reconcile a theoretical training-memory ledger with CUDA evidence.

Run from the project root:
    uv run python -m exercises.day12.training_memory_ledger
"""

from __future__ import annotations

import gc
import sys
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


MIB = 1024**2


@dataclass(frozen=True)
class MemorySnapshot:
    name: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


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


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def model_parameter_bytes(model: TinyDecoderLM) -> int:
    return sum(tensor_bytes(parameter) for parameter in model.parameters())


def model_gradient_bytes(model: TinyDecoderLM) -> int:
    return sum(
        tensor_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def optimizer_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        total += sum(
            tensor_bytes(value) for value in state.values() if isinstance(value, Tensor)
        )
    return total


def adam_moment_bytes(optimizer: torch.optim.AdamW) -> int:
    return sum(
        tensor_bytes(state[name])
        for state in optimizer.state.values()
        for name in ("exp_avg", "exp_avg_sq")
    )


def snapshot(name: str, device: torch.device) -> MemorySnapshot:
    torch.cuda.synchronize(device)
    return MemorySnapshot(
        name=name,
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def make_config() -> Config:
    return Config(
        vocabulary_size=2048,
        hidden_size=256,
        num_heads=8,
        intermediate_size=768,
        num_layers=4,
        max_sequence_length=128,
    )


def print_mib(label: str, byte_count: int) -> None:
    print(f"  {label:<31}{byte_count / MIB:9.3f} MiB")


def print_snapshot(item: MemorySnapshot, baseline: MemorySnapshot) -> None:
    print(f"\n{item.name}")
    print_mib("allocated delta:", item.allocated_bytes - baseline.allocated_bytes)
    print_mib("reserved delta:", item.reserved_bytes - baseline.reserved_bytes)
    print_mib(
        "peak allocated delta:",
        item.peak_allocated_bytes - baseline.allocated_bytes,
    )
    print_mib(
        "peak reserved delta:",
        item.peak_reserved_bytes - baseline.reserved_bytes,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 12 requires the project's CUDA PyTorch environment.")

    torch.manual_seed(1200)
    torch.cuda.manual_seed_all(1200)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = snapshot("baseline", device)

    config = make_config()
    model = TinyDecoderLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = model_parameter_bytes(model)
    parameter_tensor_count = sum(1 for _ in model.parameters())
    after_model = snapshot("after model creation", device)

    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert parameter_bytes == parameter_count * torch.float32.itemsize
    assert after_model.allocated_bytes - baseline.allocated_bytes >= parameter_bytes
    assert optimizer_tensor_bytes(optimizer) == 0

    generator = torch.Generator().manual_seed(1201)
    tokens = torch.randint(
        0,
        config.vocabulary_size,
        (8, config.max_sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    input_ids, labels = shifted_language_model_batch(tokens)

    torch.cuda.reset_peak_memory_stats(device)
    before_forward = snapshot("before forward", device)
    saved = SavedTensorStats()
    with torch.autograd.graph.saved_tensors_hooks(saved.pack, saved.unpack):
        logits, attention_weights = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocabulary_size), labels.reshape(-1)
        )
    loss.backward()
    after_backward = snapshot("after backward", device)

    gradient_bytes = model_gradient_bytes(model)
    assert logits.shape == (8, 128, config.vocabulary_size)
    assert len(attention_weights) == config.num_layers
    assert gradient_bytes == parameter_bytes
    assert optimizer_tensor_bytes(optimizer) == 0
    assert torch.isfinite(loss)

    optimizer.step()
    after_optimizer_step = snapshot("after first AdamW step", device)
    optimizer_bytes = optimizer_tensor_bytes(optimizer)
    moment_bytes = adam_moment_bytes(optimizer)
    step_metadata_bytes = optimizer_bytes - moment_bytes
    expected_moment_bytes = parameter_count * 2 * torch.float32.itemsize
    expected_step_metadata_bytes = parameter_tensor_count * torch.float32.itemsize
    assert moment_bytes == expected_moment_bytes
    assert step_metadata_bytes == expected_step_metadata_bytes

    del logits, attention_weights, loss
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    after_zero_grad = snapshot("after zero_grad(set_to_none=True)", device)
    assert model_gradient_bytes(model) == 0

    persistent_with_grad_bytes = parameter_bytes + gradient_bytes + optimizer_bytes
    persistent_without_grad_bytes = parameter_bytes + optimizer_bytes
    transient_peak_gap = (
        after_backward.peak_allocated_bytes - after_backward.allocated_bytes
    )

    print("Environment")
    print(f"  torch:                         {torch.__version__}")
    print(f"  compiled CUDA:                 {torch.version.cuda}")
    print(f"  device:                        {torch.cuda.get_device_name(0)}")
    print(f"  compute capability:            {torch.cuda.get_device_capability(0)}")
    print("\nConfiguration")
    print(f"  parameters:                    {parameter_count:,}")
    print(f"  parameter tensors:             {parameter_tensor_count}")
    print("  dtype:                         torch.float32")
    print("  batch / sequence:              8 / 128")
    print("  layers / hidden / FFN:         4 / 256 / 768")
    print("\nExact tensor ledger")
    print_mib("parameters:", parameter_bytes)
    print_mib("gradients after backward:", gradient_bytes)
    print_mib("Adam exp_avg + exp_avg_sq:", moment_bytes)
    print_mib("Adam step tensor metadata:", step_metadata_bytes)
    print_mib("optimizer tensor total:", optimizer_bytes)
    print_mib("persistent total with grads:", persistent_with_grad_bytes)
    print_mib("persistent total after zero_grad:", persistent_without_grad_bytes)
    print("\nInitial-forward saved-tensor evidence")
    print(f"  saved tensor count:             {saved.count}")
    print_mib("saved logical payload:", saved.logical_bytes)
    print_mib("composite transient peak gap:", transient_peak_gap)

    print_snapshot(after_model, baseline)
    print_snapshot(before_forward, baseline)
    print_snapshot(after_backward, baseline)
    print_snapshot(after_optimizer_step, baseline)
    print_snapshot(after_zero_grad, baseline)

    allocated_after_zero_grad = (
        after_zero_grad.allocated_bytes - baseline.allocated_bytes
    )
    unclassified_allocated = allocated_after_zero_grad - persistent_without_grad_bytes
    print("\nReconciliation after zero_grad")
    print_mib("allocator allocated delta:", allocated_after_zero_grad)
    print_mib("exact persistent tensor ledger:", persistent_without_grad_bytes)
    print_mib("unclassified live allocation:", unclassified_allocated)
    print("\nBehavior checks")
    print("  FP32 parameter bytes equal parameter_count * 4:      PASS")
    print("  FP32 gradient bytes equal parameter bytes:           PASS")
    print("  Adam moments equal parameter_count * 8:              PASS")
    print("  optimizer state is empty before first step:          PASS")
    print("  zero_grad(set_to_none=True) removes gradient tensors: PASS")
    print("  allocator counters are reported separately:          PASS")


if __name__ == "__main__":
    main()
