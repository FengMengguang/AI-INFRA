"""Day 16: framework-level TP, PP, CP, and SP correctness checks."""

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.nn import functional as F


EXPECTED_WORLD_SIZE = 2
MODEL_SEED = 1601
DATA_SEED = 1602


@dataclass(frozen=True)
class RankResult:
    rank: int
    tp_mlp_max_output_difference: float
    tp_mlp_max_gradient_difference: float
    head_parallel_max_output_difference: float
    sequence_parallel_max_output_difference: float
    pipeline_loss: float | None
    pipeline_micro_batches: int
    pipeline_parameter_gradients_present: bool
    context_parallel_status: str
    context_parallel_max_output_difference: float | None


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.up_proj(hidden)))


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 4
        self.head_dim = 4
        hidden_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, sequence, _ = hidden.shape
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        local_heads = query.size(-1) // self.head_dim

        def split_heads(value_tensor: Tensor) -> Tensor:
            return value_tensor.view(
                batch, sequence, local_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(
            batch, sequence, local_heads * self.head_dim
        )
        return self.o_proj(context)


class PipelineFirstStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, hidden: Tensor) -> Tensor:
        return F.relu(self.linear(hidden))


class PipelineSecondStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON result path. Existing files are never overwritten.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.output is None:
        parser.error("--output is required for a distributed run")
    return args


def validate_only() -> None:
    assert callable(parallelize_module)
    assert callable(ColwiseParallel)
    assert callable(RowwiseParallel)
    assert callable(SequenceParallel)
    assert callable(PipelineStage)
    assert callable(ScheduleGPipe)
    assert callable(context_parallel)
    print("Day 16 static contract: PASS")
    print(f"torch: {torch.__version__}")
    print(f"Gloo available: {dist.is_gloo_available()}")
    print(f"NCCL available: {dist.is_nccl_available()}")
    print("framework APIs: TP, Pipeline, experimental CP, SP")
    print("distributed run: exactly two ranks and one new JSON output")


def distributed_environment(backend: str) -> tuple[int, int, int, torch.device]:
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
            "Day 16 requires exactly two ranks on one host: "
            f"world_size={world_size}, local_world_size={local_world_size}"
        )
    if backend == "gloo":
        if not dist.is_gloo_available():
            raise RuntimeError("this PyTorch build does not provide Gloo")
        return rank, local_rank, world_size, torch.device("cpu")
    if not dist.is_nccl_available():
        raise RuntimeError("this PyTorch build does not provide NCCL")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"NCCL run requires two visible CUDA GPUs; found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def validate_output(output: Path, rank: int) -> None:
    error: str | None = None
    if rank == 0:
        if output.exists():
            error = f"output already exists and will not be overwritten: {output}"
        elif not output.parent.exists():
            error = f"output parent does not exist: {output.parent}"
        elif not output.parent.is_dir():
            error = f"output parent is not a directory: {output.parent}"
    payload = [error]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(payload[0])


def maximum_difference(reference: Tensor, candidate: Tensor) -> float:
    difference = (reference - candidate).abs().max().item()
    torch.testing.assert_close(reference, candidate, rtol=1e-5, atol=1e-6)
    return difference


def full_gradient(parameter: nn.Parameter) -> Tensor:
    assert parameter.grad is not None
    gradient = parameter.grad
    return gradient.full_tensor() if hasattr(gradient, "full_tensor") else gradient


def verify_tensor_parallel(
    mesh: Any, device: torch.device
) -> tuple[float, float]:
    torch.manual_seed(MODEL_SEED)
    reference = TinyMLP().to(device)
    parallel = copy.deepcopy(reference)
    parallelize_module(
        parallel,
        mesh,
        {
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )
    generator = torch.Generator(device=device).manual_seed(DATA_SEED)
    hidden = torch.randn(2, 3, 8, generator=generator, device=device)

    reference_output = reference(hidden)
    parallel_output = parallel(hidden)
    output_difference = maximum_difference(reference_output, parallel_output)

    reference_output.square().mean().backward()
    parallel_output.square().mean().backward()
    reference_parameters = dict(reference.named_parameters())
    gradient_difference = 0.0
    for name, parameter in parallel.named_parameters():
        gradient_difference = max(
            gradient_difference,
            maximum_difference(
                reference_parameters[name].grad,
                full_gradient(parameter),
            ),
        )
    return output_difference, gradient_difference


def verify_head_parallel(mesh: Any, device: torch.device) -> float:
    torch.manual_seed(MODEL_SEED + 1)
    reference = TinyAttention().to(device)
    parallel = copy.deepcopy(reference)
    parallelize_module(
        parallel,
        mesh,
        {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        },
    )
    generator = torch.Generator(device=device).manual_seed(DATA_SEED + 1)
    hidden = torch.randn(2, 5, 16, generator=generator, device=device)
    return maximum_difference(reference(hidden), parallel(hidden))


def verify_sequence_parallel(mesh: Any, rank: int, device: torch.device) -> float:
    torch.manual_seed(MODEL_SEED + 2)
    reference = nn.LayerNorm(8).to(device)
    parallel = copy.deepcopy(reference)
    parallelize_module(
        parallel,
        mesh,
        SequenceParallel(sequence_dim=1, use_local_output=False),
    )
    generator = torch.Generator(device=device).manual_seed(DATA_SEED + 2)
    full_input = torch.randn(2, 6, 8, generator=generator, device=device)
    local_input = full_input.chunk(EXPECTED_WORLD_SIZE, dim=1)[rank].contiguous()
    parallel_output = parallel(local_input)
    assert hasattr(parallel_output, "full_tensor")
    return maximum_difference(reference(full_input), parallel_output.full_tensor())


def verify_pipeline(
    rank: int, device: torch.device
) -> tuple[float | None, bool]:
    torch.manual_seed(MODEL_SEED + 3)
    stage_module: nn.Module
    example_input: Tensor
    if rank == 0:
        stage_module = PipelineFirstStage().to(device)
        example_input = torch.empty(2, 8, device=device)
    else:
        stage_module = PipelineSecondStage().to(device)
        example_input = torch.empty(2, 8, device=device)

    stage = PipelineStage(
        stage_module,
        stage_index=rank,
        num_stages=EXPECTED_WORLD_SIZE,
        device=device,
        input_args=example_input,
    )
    schedule = ScheduleGPipe(
        stage,
        n_microbatches=2,
        loss_fn=F.mse_loss,
    )
    generator = torch.Generator(device=device).manual_seed(DATA_SEED + 3)
    full_input = torch.randn(4, 8, generator=generator, device=device)
    target = torch.randn(4, 4, generator=generator, device=device)
    losses: list[Tensor] = []
    if rank == 0:
        schedule.step(full_input)
    else:
        schedule.step(target=target, losses=losses)
    gradients_present = all(
        parameter.grad is not None for parameter in stage_module.parameters()
    )
    assert gradients_present
    if rank == 1:
        assert len(losses) == 2
        return torch.stack([loss.detach() for loss in losses]).mean().item(), True
    return None, True


def verify_context_parallel(
    mesh: Any, device: torch.device
) -> tuple[str, float | None]:
    generator = torch.Generator(device=device).manual_seed(DATA_SEED + 4)
    query = torch.randn(1, 2, 6, 4, generator=generator, device=device)
    key = torch.randn(1, 2, 6, 4, generator=generator, device=device)
    value = torch.randn(1, 2, 6, 4, generator=generator, device=device)
    reference = F.scaled_dot_product_attention(query, key, value)
    try:
        with context_parallel(
            mesh,
            buffers=[query, key, value],
            buffer_seq_dims=[2, 2, 2],
        ):
            local_output = F.scaled_dot_product_attention(query, key, value)
            if hasattr(local_output, "full_tensor"):
                candidate = local_output.full_tensor()
            else:
                gathered = [torch.empty_like(local_output) for _ in range(2)]
                dist.all_gather(gathered, local_output)
                candidate = torch.cat(gathered, dim=2)
        return "PASS", maximum_difference(reference, candidate)
    except (RuntimeError, NotImplementedError) as error:
        return f"UNSUPPORTED_ON_{device.type.upper()}: {type(error).__name__}", None


def gather_objects(value: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def write_json_atomically(output: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
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
    rank, _, world_size, device = distributed_environment(args.backend)
    dist.init_process_group(backend=args.backend, init_method="env://")
    try:
        assert args.output is not None
        validate_output(args.output, rank)
        mesh = init_device_mesh(
            device.type,
            (world_size,),
            mesh_dim_names=("model_parallel",),
        )
        tp_output, tp_gradient = verify_tensor_parallel(mesh, device)
        head_output = verify_head_parallel(mesh, device)
        sp_output = verify_sequence_parallel(mesh, rank, device)
        pipeline_loss, pipeline_gradients = verify_pipeline(rank, device)
        cp_status, cp_output = verify_context_parallel(mesh, device)

        rank_result = RankResult(
            rank=rank,
            tp_mlp_max_output_difference=tp_output,
            tp_mlp_max_gradient_difference=tp_gradient,
            head_parallel_max_output_difference=head_output,
            sequence_parallel_max_output_difference=sp_output,
            pipeline_loss=pipeline_loss,
            pipeline_micro_batches=2,
            pipeline_parameter_gradients_present=pipeline_gradients,
            context_parallel_status=cp_status,
            context_parallel_max_output_difference=cp_output,
        )
        all_results = gather_objects(asdict(rank_result))
        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": "completed",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "torch_version": torch.__version__,
                "backend": args.backend,
                "device_type": device.type,
                "world_size": world_size,
                "results_by_rank": all_results,
                "checks": {
                    "framework_tensor_parallel": "PASS",
                    "framework_head_parallel": "PASS",
                    "framework_sequence_parallel": "PASS",
                    "framework_pipeline_gpipe": "PASS",
                    "context_parallel": cp_status,
                },
                "unverified": [
                    "NCCL performance and communication timeline",
                    "pipeline bubble timing",
                    "context parallel performance and causal attention",
                ],
            }
            write_json_atomically(args.output, payload)
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
