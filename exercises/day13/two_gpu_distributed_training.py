"""Day 13: validate two-GPU NCCL collectives, DDP, and FSDP2.

Static validation on any machine:
    python -m exercises.day13.two_gpu_distributed_training --validate-only

Real experiment on one host with exactly two visible NVIDIA GPUs:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
        -m exercises.day13.two_gpu_distributed_training \
        --output /tmp/day13_two_gpu.json

The output path must not already exist. The experiment never writes checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from exercises.day04.decoder_block_training import (  # noqa: E402
    Config,
    TinyDecoderLM,
    shifted_language_model_batch,
)


EXPECTED_WORLD_SIZE = 2
MODEL_SEED = 1300
DATA_SEED = 1301
MIB = 1024**2


@dataclass(frozen=True)
class ModeMetrics:
    mode: str
    rank: int
    local_loss: float
    local_parameter_bytes: int
    local_gradient_bytes: int
    local_optimizer_bytes: int
    baseline_allocated_bytes: int
    after_step_allocated_bytes: int
    peak_allocated_bytes: int
    peak_delta_bytes: int
    average_step_ms: float
    tokens_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON result path. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check local APIs and print the real-run contract without starting workers.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        parser.error("warmup-steps must be >= 0 and measured-steps must be > 0")
    if not args.validate_only and args.output is None:
        parser.error("--output is required for a real two-GPU run")
    return args


def make_config() -> Config:
    return Config(
        vocabulary_size=2048,
        hidden_size=256,
        num_heads=8,
        intermediate_size=768,
        num_layers=4,
        max_sequence_length=128,
    )


def validate_only() -> None:
    config = make_config()
    config.validate()
    assert callable(fully_shard)
    assert dist.is_available()
    print("Static contract: PASS")
    print(f"torch: {torch.__version__}")
    print(f"distributed available: {dist.is_available()}")
    print(f"NCCL compiled/available: {dist.is_nccl_available()}")
    print(f"CUDA visible now: {torch.cuda.is_available()}")
    print(f"visible GPU count now: {torch.cuda.device_count()}")
    print("real run requires: one host, world_size=2, two visible CUDA GPUs, NCCL")
    print("real run writes: one new user-selected JSON file; no checkpoints")


def require_two_gpu_environment() -> tuple[int, int, int, torch.device]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch with python -m torch.distributed.run; "
            f"missing environment variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != EXPECTED_WORLD_SIZE or local_world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            "Day 13 requires exactly two ranks on one host: "
            f"world_size={world_size}, local_world_size={local_world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this worker")
    if torch.cuda.device_count() < EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"two visible GPUs are required; found {torch.cuda.device_count()}"
        )
    if not dist.is_nccl_available():
        raise RuntimeError("this PyTorch build does not provide the NCCL backend")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the visible GPU range"
        )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def validate_output_contract(output: Path, rank: int) -> None:
    error: str | None = None
    if rank == 0:
        if output.exists():
            error = f"output already exists and will not be overwritten: {output}"
        elif not output.parent.exists():
            error = f"output parent directory does not exist: {output.parent}"
        elif not output.parent.is_dir():
            error = f"output parent is not a directory: {output.parent}"
    payload = [error]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(payload[0])


def tensor_local_bytes(value: Tensor) -> int:
    local = value.to_local() if hasattr(value, "to_local") else value
    return local.numel() * local.element_size()


def parameter_local_bytes(model: nn.Module) -> int:
    return sum(tensor_local_bytes(parameter) for parameter in model.parameters())


def gradient_local_bytes(model: nn.Module) -> int:
    return sum(
        tensor_local_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def optimizer_local_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        tensor_local_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    )


def make_global_tokens(config: Config) -> Tensor:
    generator = torch.Generator().manual_seed(DATA_SEED)
    return torch.randint(
        0,
        config.vocabulary_size,
        (8, config.max_sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    )


def local_batch(
    global_tokens: Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    assert global_tokens.size(0) % world_size == 0
    rows_per_rank = global_tokens.size(0) // world_size
    start = rank * rows_per_rank
    local_tokens = global_tokens[start : start + rows_per_rank].to(device)
    return shifted_language_model_batch(local_tokens)


def model_loss(model: nn.Module, input_ids: Tensor, labels: Tensor) -> Tensor:
    logits, _ = model(input_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )


def collect_full_parameter_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        full = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        state[name] = full.detach().cpu().clone()
    return state


def compare_parameter_states(
    reference: dict[str, Tensor], candidate: dict[str, Tensor]
) -> float:
    assert reference.keys() == candidate.keys()
    maximum_difference = 0.0
    for name in reference:
        difference = (reference[name] - candidate[name]).abs().max().item()
        maximum_difference = max(maximum_difference, difference)
        torch.testing.assert_close(
            reference[name], candidate[name], rtol=1e-5, atol=2e-6
        )
    return maximum_difference


def run_one_update(
    mode: str,
    model: nn.Module,
    state_model: nn.Module,
    optimizer: torch.optim.AdamW,
    input_ids: Tensor,
    labels: Tensor,
    rank: int,
    device: torch.device,
) -> tuple[ModeMetrics, dict[str, Tensor]]:
    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    loss = model_loss(model, input_ids, labels)
    loss.backward()
    torch.cuda.synchronize(device)
    local_gradient_bytes = gradient_local_bytes(model)
    optimizer.step()
    torch.cuda.synchronize(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    after_step_allocated = torch.cuda.memory_allocated(device)
    state = collect_full_parameter_state(state_model)
    metrics = ModeMetrics(
        mode=mode,
        rank=rank,
        local_loss=loss.item(),
        local_parameter_bytes=parameter_local_bytes(model),
        local_gradient_bytes=local_gradient_bytes,
        local_optimizer_bytes=optimizer_local_bytes(optimizer),
        baseline_allocated_bytes=baseline_allocated,
        after_step_allocated_bytes=after_step_allocated,
        peak_allocated_bytes=peak_allocated,
        peak_delta_bytes=peak_allocated - baseline_allocated,
        average_step_ms=0.0,
        tokens_per_second=0.0,
    )
    return metrics, state


def benchmark_steps(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    input_ids: Tensor,
    labels: Tensor,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
) -> tuple[float, float]:
    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model, input_ids, labels)
        loss.backward()
        optimizer.step()

    for _ in range(warmup_steps):
        one_step()
    dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(measured_steps):
        one_step()
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - start
    average_step_ms = elapsed * 1000 / measured_steps
    global_tokens_per_step = input_ids.numel() * dist.get_world_size()
    tokens_per_second = global_tokens_per_step / (elapsed / measured_steps)
    return average_step_ms, tokens_per_second


def with_benchmark(
    metrics: ModeMetrics, average_step_ms: float, tokens_per_second: float
) -> ModeMetrics:
    values = asdict(metrics)
    values["average_step_ms"] = average_step_ms
    values["tokens_per_second"] = tokens_per_second
    return ModeMetrics(**values)


def make_ddp_model(
    config: Config, device: torch.device, local_rank: int
) -> tuple[TinyDecoderLM, DistributedDataParallel]:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    base_model = TinyDecoderLM(config).to(device)
    ddp_model = DistributedDataParallel(
        base_model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    return base_model, ddp_model


def make_fsdp2_model(config: Config, world_size: int) -> TinyDecoderLM:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = TinyDecoderLM(config)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    for block in model.blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    return model


def verify_collectives(rank: int, device: torch.device) -> dict[str, list[float]]:
    all_reduce_value = torch.tensor([1.0 + 4 * rank, 3.0 + 4 * rank], device=device)
    dist.all_reduce(all_reduce_value, op=dist.ReduceOp.SUM)
    all_reduce_value /= EXPECTED_WORLD_SIZE
    torch.testing.assert_close(
        all_reduce_value, torch.tensor([3.0, 5.0], device=device)
    )

    reduce_scatter_input = torch.tensor(
        [1.0, 2.0, 3.0, 4.0] if rank == 0 else [10.0, 20.0, 30.0, 40.0],
        device=device,
    )
    reduce_scatter_output = torch.empty(2, device=device)
    dist.reduce_scatter_tensor(reduce_scatter_output, reduce_scatter_input)
    expected_rs = (
        torch.tensor([11.0, 22.0], device=device)
        if rank == 0
        else torch.tensor([33.0, 44.0], device=device)
    )
    torch.testing.assert_close(reduce_scatter_output, expected_rs)

    local_shard = torch.tensor([1.0 + 2 * rank, 2.0 + 2 * rank], device=device)
    gathered = torch.empty(4, device=device)
    dist.all_gather_into_tensor(gathered, local_shard)
    torch.testing.assert_close(
        gathered, torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    )

    all_to_all_input = torch.tensor(
        [0.0, 1.0, 2.0, 3.0] if rank == 0 else [10.0, 11.0, 12.0, 13.0],
        device=device,
    )
    all_to_all_output = torch.empty(4, device=device)
    dist.all_to_all_single(all_to_all_output, all_to_all_input)
    expected_a2a = (
        torch.tensor([0.0, 1.0, 10.0, 11.0], device=device)
        if rank == 0
        else torch.tensor([2.0, 3.0, 12.0, 13.0], device=device)
    )
    torch.testing.assert_close(all_to_all_output, expected_a2a)

    return {
        "all_reduce_mean": all_reduce_value.cpu().tolist(),
        "reduce_scatter_sum_shard": reduce_scatter_output.cpu().tolist(),
        "all_gather": gathered.cpu().tolist(),
        "all_to_all": all_to_all_output.cpu().tolist(),
    }


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"UNAVAILABLE: {type(error).__name__}: {error}"


def gather_objects(local: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


def write_result_atomically(output: Path, payload: dict[str, Any]) -> None:
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary output unexpectedly exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def distributed_main(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = require_two_gpu_environment()
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        assert args.output is not None
        validate_output_contract(args.output, rank)
        collectives = verify_collectives(rank, device)
        config = make_config()
        global_tokens = make_global_tokens(config)
        input_ids, labels = local_batch(global_tokens, rank, world_size, device)

        ddp_base, ddp_model = make_ddp_model(config, device, local_rank)
        ddp_optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
        ddp_metrics, ddp_state = run_one_update(
            "ddp",
            ddp_model,
            ddp_base,
            ddp_optimizer,
            input_ids,
            labels,
            rank,
            device,
        )
        ddp_ms, ddp_tokens_per_second = benchmark_steps(
            ddp_model,
            ddp_optimizer,
            input_ids,
            labels,
            device,
            args.warmup_steps,
            args.measured_steps,
        )
        ddp_metrics = with_benchmark(
            ddp_metrics, ddp_ms, ddp_tokens_per_second
        )
        del ddp_model, ddp_base, ddp_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

        fsdp_model = make_fsdp2_model(config, world_size)
        fsdp_optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)
        fsdp_metrics, fsdp_state = run_one_update(
            "fsdp2",
            fsdp_model,
            fsdp_model,
            fsdp_optimizer,
            input_ids,
            labels,
            rank,
            device,
        )
        maximum_parameter_difference = compare_parameter_states(
            ddp_state, fsdp_state
        )
        fsdp_ms, fsdp_tokens_per_second = benchmark_steps(
            fsdp_model,
            fsdp_optimizer,
            input_ids,
            labels,
            device,
            args.warmup_steps,
            args.measured_steps,
        )
        fsdp_metrics = with_benchmark(
            fsdp_metrics, fsdp_ms, fsdp_tokens_per_second
        )

        environment = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        all_collectives = gather_objects(collectives)
        all_ddp_metrics = gather_objects(asdict(ddp_metrics))
        all_fsdp_metrics = gather_objects(asdict(fsdp_metrics))
        all_environments = gather_objects(environment)
        all_differences = gather_objects(maximum_parameter_difference)

        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": "completed",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "torch_version": torch.__version__,
                "compiled_cuda_version": torch.version.cuda,
                "nccl_version": torch.cuda.nccl.version(),
                "world_size": world_size,
                "configuration": {
                    **asdict(config),
                    "global_batch_size": global_tokens.size(0),
                    "per_rank_batch_size": input_ids.size(0),
                    "sequence_length": input_ids.size(1),
                    "dtype": "torch.float32",
                    "warmup_steps": args.warmup_steps,
                    "measured_steps": args.measured_steps,
                },
                "environments": all_environments,
                "nvidia_smi_topology": command_output(["nvidia-smi", "topo", "-m"]),
                "collectives_by_rank": all_collectives,
                "ddp_by_rank": all_ddp_metrics,
                "fsdp2_by_rank": all_fsdp_metrics,
                "maximum_ddp_fsdp2_parameter_difference_by_rank": all_differences,
                "checks": {
                    "all_reduce": "PASS",
                    "reduce_scatter": "PASS",
                    "all_gather": "PASS",
                    "all_to_all": "PASS",
                    "ddp_fsdp2_parameter_update": "PASS",
                },
            }
            write_result_atomically(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(f"result written: {args.output}")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return
    distributed_main(args)


if __name__ == "__main__":
    main()
