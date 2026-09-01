"""Day 15: validate FSDP2 sharding, resharding, and checkpoint restore.

Local two-process correctness run:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
      -m exercises.day15.fsdp_zero_sharding \
      --backend gloo --output /tmp/day15_gloo.json

Real two-GPU run:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
      -m exercises.day15.fsdp_zero_sharding \
      --backend nccl --output /tmp/day15_nccl.json
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn import functional as F

from exercises.day04.decoder_block_training import Config, TinyDecoderLM


EXPECTED_WORLD_SIZE = 2
MODEL_SEED = 1501
DATA_SEED = 1502
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class RankMetrics:
    rank: int
    local_loss: float
    logical_parameter_elements: int
    local_parameter_elements: int
    local_gradient_elements: int
    local_optimizer_tensor_elements: int
    parameter_dtypes: list[str]
    gradient_dtypes: list[str]
    optimizer_tensor_dtypes: list[str]
    checkpoint_files: int
    checkpoint_bytes: int
    restored_max_parameter_difference: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--mixed-precision", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument(
        "--keep-parameters-after-forward",
        action="store_true",
        help="Set reshard_after_forward=False for capacity/communication comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON result path. Existing files are never overwritten.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.output is None:
        parser.error("--output is required for a distributed run")
    if args.backend == "gloo" and args.mixed_precision != "fp32":
        parser.error("the local Gloo correctness path intentionally uses fp32")
    return args


def make_config() -> Config:
    return Config(
        vocabulary_size=128,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        num_layers=2,
        max_sequence_length=16,
    )


def validate_only() -> None:
    config = make_config()
    config.validate()
    assert callable(fully_shard)
    assert callable(get_state_dict)
    assert callable(set_state_dict)
    assert callable(dcp.save)
    assert callable(dcp.load)
    print("Day 15 static contract: PASS")
    print(f"torch: {torch.__version__}")
    print(f"Gloo available: {dist.is_gloo_available()}")
    print(f"NCCL available: {dist.is_nccl_available()}")
    print("local correctness run: exactly two Gloo/CPU ranks, fp32")
    print("target run: exactly two NCCL/CUDA ranks, fp32 or bf16")
    print("checkpoint: temporary sharded files are restored, measured, then removed")
    print("persistent output: one new rank-0 JSON file")


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
            "Day 15 requires exactly two ranks on one host: "
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


def validate_mixed_precision(mode: str, device: torch.device) -> torch.dtype | None:
    if mode == "fp32":
        return None
    if device.type != "cuda":
        raise RuntimeError("bf16 mixed precision requires the NCCL/CUDA path")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support native bf16")
    return torch.bfloat16


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


def global_tokens(config: Config) -> Tensor:
    generator = torch.Generator().manual_seed(DATA_SEED)
    return torch.randint(
        0,
        config.vocabulary_size,
        (8, config.max_sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    )


def local_batch(
    tokens: Tensor, rank: int, world_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    assert tokens.size(0) % world_size == 0
    rows = tokens.size(0) // world_size
    local = tokens[rank * rows : (rank + 1) * rows].to(device)
    return local[:, :-1], local[:, 1:]


def make_fsdp_model(
    config: Config,
    device: torch.device,
    world_size: int,
    mixed_dtype: torch.dtype | None,
    reshard_after_forward: bool,
) -> TinyDecoderLM:
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_SEED)
    model = TinyDecoderLM(config).to(device)
    mesh = init_device_mesh(
        device.type,
        (world_size,),
        mesh_dim_names=("fsdp",),
    )
    policy = MixedPrecisionPolicy(
        param_dtype=mixed_dtype,
        reduce_dtype=mixed_dtype,
        output_dtype=torch.float32 if mixed_dtype is not None else None,
    )
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=policy,
    )
    return model


def local_tensor(value: Tensor) -> Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def tensor_elements(values: list[Tensor]) -> int:
    return sum(local_tensor(value).numel() for value in values)


def tensor_dtypes(values: list[Tensor]) -> list[str]:
    return sorted({str(local_tensor(value).dtype) for value in values})


def optimizer_tensors(optimizer: torch.optim.Optimizer) -> list[Tensor]:
    return [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    ]


def full_parameter_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        full = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        state[name] = full.detach().cpu().clone()
    return state


def maximum_state_difference(
    reference: dict[str, Tensor], candidate: dict[str, Tensor]
) -> float:
    assert reference.keys() == candidate.keys()
    maximum = 0.0
    for name in reference:
        maximum = max(
            maximum,
            (reference[name] - candidate[name]).abs().max().item(),
        )
        torch.testing.assert_close(reference[name], candidate[name], rtol=0, atol=0)
    return maximum


def create_shared_checkpoint_directory(output: Path, rank: int) -> Path:
    checkpoint_path: str | None = None
    if rank == 0:
        checkpoint_path = tempfile.mkdtemp(
            prefix=f".{output.stem}.checkpoint.",
            dir=output.parent,
        )
    payload = [checkpoint_path]
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    return Path(payload[0])


def checkpoint_round_trip(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
) -> tuple[int, int, float]:
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=options,
    )
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=checkpoint_path,
    )
    dist.barrier()

    reference = full_parameter_state(model)
    with torch.no_grad():
        for parameter in model.parameters():
            local_tensor(parameter).add_(1.0)

    load_model_state, load_optimizer_state = get_state_dict(
        model,
        optimizer,
        options=options,
    )
    state: dict[str, Any] = {
        "model": load_model_state,
        "optimizer": load_optimizer_state,
    }
    dcp.load(state, checkpoint_id=checkpoint_path)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    restored_difference = maximum_state_difference(
        reference,
        full_parameter_state(model),
    )

    files = [path for path in checkpoint_path.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files), restored_difference


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
    rank, local_rank, world_size, device = distributed_environment(args.backend)
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, init_method="env://")
    checkpoint_path: Path | None = None
    try:
        assert args.output is not None
        validate_output(args.output, rank)
        mixed_dtype = validate_mixed_precision(args.mixed_precision, device)
        config = make_config()
        reshard_after_forward = not args.keep_parameters_after_forward
        model = make_fsdp_model(
            config,
            device,
            world_size,
            mixed_dtype,
            reshard_after_forward,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        input_ids, labels = local_batch(
            global_tokens(config), rank, world_size, device
        )

        logical_parameter_elements = sum(
            parameter.numel() for parameter in TinyDecoderLM(config).parameters()
        )
        local_parameter_elements = tensor_elements(list(model.parameters()))
        assert local_parameter_elements * world_size == logical_parameter_elements

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)
        assert logits.shape == (
            input_ids.size(0),
            input_ids.size(1),
            config.vocabulary_size,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        local_gradient_elements = tensor_elements(gradients)
        optimizer.step()
        optimizer_values = optimizer_tensors(optimizer)

        checkpoint_path = create_shared_checkpoint_directory(args.output, rank)
        checkpoint_files, checkpoint_bytes, restored_difference = (
            checkpoint_round_trip(model, optimizer, checkpoint_path)
        )
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_path)
        dist.barrier()
        assert not checkpoint_path.exists()

        metrics = RankMetrics(
            rank=rank,
            local_loss=loss.item(),
            logical_parameter_elements=logical_parameter_elements,
            local_parameter_elements=local_parameter_elements,
            local_gradient_elements=local_gradient_elements,
            local_optimizer_tensor_elements=tensor_elements(optimizer_values),
            parameter_dtypes=tensor_dtypes(list(model.parameters())),
            gradient_dtypes=tensor_dtypes(gradients),
            optimizer_tensor_dtypes=tensor_dtypes(optimizer_values),
            checkpoint_files=checkpoint_files,
            checkpoint_bytes=checkpoint_bytes,
            restored_max_parameter_difference=restored_difference,
        )
        all_metrics = gather_objects(asdict(metrics))

        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": "completed",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "torch_version": torch.__version__,
                "backend": args.backend,
                "device_type": device.type,
                "world_size": world_size,
                "mixed_precision": args.mixed_precision,
                "reshard_after_forward": reshard_after_forward,
                "configuration": {
                    **asdict(config),
                    "global_batch_size": 8,
                    "per_rank_batch_size": input_ids.size(0),
                    "learning_rate": LEARNING_RATE,
                },
                "results_by_rank": all_metrics,
                "checks": {
                    "parameters_sharded_evenly": "PASS",
                    "forward_backward_step": "PASS",
                    "sharded_checkpoint_restore": "PASS",
                    "checkpoint_temporary_directory_removed": "PASS",
                },
                "unverified": [
                    "checkpoint restore with a different world size",
                    "performance and allocator memory require the NCCL/CUDA path",
                ],
            }
            write_json_atomically(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(f"result written: {args.output}")
        dist.barrier()
        checkpoint_path = None
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        if rank == 0 and checkpoint_path is not None and checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
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
