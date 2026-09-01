"""Day 14: inspect manual gradient averaging and DDP bucket synchronization.

Local correctness run:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
      -m exercises.day14.ddp_nccl_execution \
      --backend gloo --output /tmp/day14_gloo.json

Real two-GPU run:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
      -m exercises.day14.ddp_nccl_execution \
      --backend nccl --output /tmp/day14_nccl.json
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from exercises.day04.decoder_block_training import Config, TinyDecoderLM


EXPECTED_WORLD_SIZE = 2
MODEL_SEED = 1401
DATA_SEED = 1402
LEARNING_RATE = 1e-3


@dataclass
class CommunicationState:
    calls: int = 0
    elements: int = 0


@dataclass(frozen=True)
class ModeResult:
    mode: str
    rank: int
    local_loss: float
    communication_calls: int
    communication_elements: int
    calls_after_first_micro_batch: int | None
    parameters_synchronized: bool


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
    assert dist.is_available()
    assert callable(DistributedDataParallel)
    print("Day 14 static contract: PASS")
    print(f"torch: {torch.__version__}")
    print(f"Gloo available: {dist.is_gloo_available()}")
    print(f"NCCL available: {dist.is_nccl_available()}")
    print("real distributed run requires exactly two ranks")


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
            "Day 14 requires exactly two ranks on one host: "
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


def global_tokens(config: Config) -> Tensor:
    generator = torch.Generator().manual_seed(DATA_SEED)
    return torch.randint(
        0,
        config.vocabulary_size,
        (8, config.max_sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    )


def local_language_model_batch(
    tokens: Tensor, rank: int, world_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    assert tokens.size(0) % world_size == 0
    rows = tokens.size(0) // world_size
    local = tokens[rank * rows : (rank + 1) * rows].to(device)
    return local[:, :-1], local[:, 1:]


def new_model(config: Config, device: torch.device) -> TinyDecoderLM:
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_SEED)
    return TinyDecoderLM(config).to(device)


def loss_for(model: nn.Module, input_ids: Tensor, labels: Tensor) -> Tensor:
    logits, _ = model(input_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )


def clone_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


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
        torch.testing.assert_close(
            reference[name], candidate[name], rtol=1e-5, atol=2e-6
        )
    return maximum


def assert_parameters_synchronized(model: nn.Module) -> None:
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)


def manual_all_reduce_update(
    config: Config,
    input_ids: Tensor,
    labels: Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[ModeResult, dict[str, Tensor]]:
    model = new_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_for(model, input_ids, labels)
    loss.backward()

    communication_calls = 0
    communication_elements = 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)
        communication_calls += 1
        communication_elements += parameter.grad.numel()

    optimizer.step()
    assert_parameters_synchronized(model)
    result = ModeResult(
        mode="manual_all_reduce",
        rank=rank,
        local_loss=loss.item(),
        communication_calls=communication_calls,
        communication_elements=communication_elements,
        calls_after_first_micro_batch=None,
        parameters_synchronized=True,
    )
    return result, clone_parameters(model)


def average_bucket_hook(
    state: CommunicationState, bucket: dist.GradBucket
) -> torch.futures.Future[Tensor]:
    state.calls += 1
    state.elements += bucket.buffer().numel()
    work = dist.all_reduce(bucket.buffer(), op=dist.ReduceOp.SUM, async_op=True)

    def divide(future: torch.futures.Future[list[Tensor]]) -> Tensor:
        value = future.value()[0]
        value.div_(dist.get_world_size())
        return value

    return work.get_future().then(divide)


def ddp_update(
    mode: str,
    config: Config,
    input_ids: Tensor,
    labels: Tensor,
    rank: int,
    local_rank: int,
    device: torch.device,
    use_no_sync: bool,
) -> tuple[ModeResult, dict[str, Tensor]]:
    base_model = new_model(config, device)
    ddp_kwargs: dict[str, Any] = {
        "forward_sync_buffers": False,
        "gradient_as_bucket_view": True,
        "bucket_cap_mb": 0.01,
    }
    if device.type == "cuda":
        ddp_kwargs.update(
            device_ids=[local_rank],
            output_device=local_rank,
        )
    model = DistributedDataParallel(base_model, **ddp_kwargs)
    communication = CommunicationState()
    model.register_comm_hook(communication, average_bucket_hook)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)

    calls_after_first_micro_batch: int | None = None
    if use_no_sync:
        assert input_ids.size(0) % 2 == 0
        input_micro_batches = input_ids.chunk(2)
        label_micro_batches = labels.chunk(2)
        losses: list[Tensor] = []
        for index, (input_micro, label_micro) in enumerate(
            zip(input_micro_batches, label_micro_batches, strict=True)
        ):
            context = model.no_sync() if index == 0 else nullcontext()
            with context:
                micro_loss = loss_for(model, input_micro, label_micro)
                (micro_loss / 2).backward()
                losses.append(micro_loss.detach())
            if index == 0:
                calls_after_first_micro_batch = communication.calls
                assert calls_after_first_micro_batch == 0
        local_loss = torch.stack(losses).mean().item()
    else:
        loss = loss_for(model, input_ids, labels)
        loss.backward()
        local_loss = loss.item()

    assert communication.calls > 0
    optimizer.step()
    assert_parameters_synchronized(base_model)
    result = ModeResult(
        mode=mode,
        rank=rank,
        local_loss=local_loss,
        communication_calls=communication.calls,
        communication_elements=communication.elements,
        calls_after_first_micro_batch=calls_after_first_micro_batch,
        parameters_synchronized=True,
    )
    return result, clone_parameters(base_model)


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
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def distributed_main(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = distributed_environment(args.backend)
    dist.init_process_group(backend=args.backend, init_method="env://")
    try:
        assert args.output is not None
        validate_output(args.output, rank)
        config = make_config()
        input_ids, labels = local_language_model_batch(
            global_tokens(config), rank, world_size, device
        )

        manual_result, manual_state = manual_all_reduce_update(
            config, input_ids, labels, rank, world_size, device
        )
        ddp_result, ddp_state = ddp_update(
            "ddp",
            config,
            input_ids,
            labels,
            rank,
            local_rank,
            device,
            use_no_sync=False,
        )
        no_sync_result, no_sync_state = ddp_update(
            "ddp_no_sync_accumulation",
            config,
            input_ids,
            labels,
            rank,
            local_rank,
            device,
            use_no_sync=True,
        )

        ddp_difference = maximum_state_difference(manual_state, ddp_state)
        no_sync_difference = maximum_state_difference(manual_state, no_sync_state)
        results = gather_objects(
            {
                "manual": asdict(manual_result),
                "ddp": asdict(ddp_result),
                "ddp_no_sync": asdict(no_sync_result),
                "manual_vs_ddp_max_parameter_difference": ddp_difference,
                "manual_vs_no_sync_max_parameter_difference": no_sync_difference,
            }
        )

        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": "completed",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "torch_version": torch.__version__,
                "backend": args.backend,
                "world_size": world_size,
                "device_type": device.type,
                "configuration": {
                    **asdict(config),
                    "global_batch_size": 8,
                    "per_rank_batch_size": input_ids.size(0),
                    "micro_batches_for_no_sync": 2,
                    "learning_rate": LEARNING_RATE,
                },
                "results_by_rank": results,
                "checks": {
                    "manual_parameters_synchronized": "PASS",
                    "ddp_parameters_synchronized": "PASS",
                    "no_sync_parameters_synchronized": "PASS",
                    "manual_vs_ddp_update": "PASS",
                    "manual_vs_no_sync_update": "PASS",
                    "no_sync_first_micro_batch_communication_calls": 0,
                },
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
