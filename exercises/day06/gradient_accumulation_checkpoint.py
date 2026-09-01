"""Day 6: verify gradient accumulation and resumable checkpoints on CUDA.

Run from the project root:
    uv run python exercises/day06/gradient_accumulation_checkpoint.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

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
from exercises.day05.overfit_training_loop import (  # noqa: E402
    make_dataset,
    sample_batch,
)


REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "completed_steps",
    "sampling_generator_state",
    "torch_rng_state",
    "cuda_rng_state_all",
}


def make_config() -> Config:
    return Config(
        vocabulary_size=32,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        num_layers=1,
        max_sequence_length=16,
    )


def make_model_and_optimizer(
    config: Config, device: torch.device
) -> tuple[TinyDecoderLM, torch.optim.AdamW]:
    model = TinyDecoderLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.01
    )
    return model, optimizer


def loss_for_batch(
    model: TinyDecoderLM, input_ids: Tensor, labels: Tensor
) -> Tensor:
    logits, _ = model(input_ids)
    return F.cross_entropy(
        logits.reshape(-1, model.config.vocabulary_size), labels.reshape(-1)
    )


@torch.no_grad()
def evaluate_loss(
    model: TinyDecoderLM, dataset: Tensor, device: torch.device
) -> float:
    model.eval()
    input_ids, labels = shifted_language_model_batch(dataset.to(device))
    return loss_for_batch(model, input_ids, labels).item()


def clone_state_dict(model: TinyDecoderLM) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state_dicts_close(
    left: dict[str, Tensor], right: dict[str, Tensor], *, atol: float
) -> float:
    assert left.keys() == right.keys()
    maximum_difference = 0.0
    for name in left:
        difference = (left[name] - right[name]).abs().max().item()
        maximum_difference = max(maximum_difference, difference)
        torch.testing.assert_close(left[name], right[name], rtol=0.0, atol=atol)
    return maximum_difference


def verify_accumulation_equivalence(
    config: Config, dataset: Tensor, device: torch.device
) -> float:
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    full_model, full_optimizer = make_model_and_optimizer(config, device)

    accumulated_model, accumulated_optimizer = make_model_and_optimizer(config, device)
    accumulated_model.load_state_dict(full_model.state_dict())

    full_input_ids, full_labels = shifted_language_model_batch(dataset.to(device))
    full_optimizer.zero_grad(set_to_none=True)
    full_loss = loss_for_batch(full_model, full_input_ids, full_labels)
    full_loss.backward()
    full_optimizer.step()

    accumulation_steps = 2
    accumulated_optimizer.zero_grad(set_to_none=True)
    for micro_batch in dataset.chunk(accumulation_steps):
        input_ids, labels = shifted_language_model_batch(micro_batch.to(device))
        micro_loss = loss_for_batch(accumulated_model, input_ids, labels)
        (micro_loss / accumulation_steps).backward()
    accumulated_optimizer.step()

    return assert_state_dicts_close(
        full_model.state_dict(), accumulated_model.state_dict(), atol=2e-7
    )


def train_steps(
    model: TinyDecoderLM,
    optimizer: torch.optim.AdamW,
    dataset: Tensor,
    generator: torch.Generator,
    device: torch.device,
    *,
    start_step: int,
    end_step: int,
    accumulation_steps: int,
    micro_batch_size: int,
) -> None:
    model.train()
    for _step in range(start_step, end_step):
        optimizer.zero_grad(set_to_none=True)
        for _micro_step in range(accumulation_steps):
            input_ids, labels = sample_batch(
                dataset,
                batch_size=micro_batch_size,
                generator=generator,
                device=device,
            )
            loss = loss_for_batch(model, input_ids, labels)
            (loss / accumulation_steps).backward()
        optimizer.step()


def save_checkpoint(
    path: Path,
    model: TinyDecoderLM,
    optimizer: torch.optim.AdamW,
    *,
    completed_steps: int,
    sampling_generator: torch.Generator,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": completed_steps,
        "sampling_generator_state": sampling_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(
    path: Path,
    model: TinyDecoderLM,
    optimizer: torch.optim.AdamW,
    sampling_generator: torch.Generator,
    device: torch.device,
) -> int:
    checkpoint: dict[str, Any] = torch.load(
        path, map_location=device, weights_only=True
    )
    missing_keys = REQUIRED_CHECKPOINT_KEYS - checkpoint.keys()
    if missing_keys:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing_keys)}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    sampling_generator.set_state(checkpoint["sampling_generator_state"].cpu())
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_states = [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
    torch.cuda.set_rng_state_all(cuda_rng_states)
    return int(checkpoint["completed_steps"])


def verify_checkpoint_resume(
    config: Config, dataset: Tensor, device: torch.device, checkpoint_path: Path
) -> tuple[float, float, float, int]:
    total_steps = 40
    interruption_step = 17
    accumulation_steps = 2
    micro_batch_size = 2

    torch.manual_seed(211)
    torch.cuda.manual_seed_all(211)
    initial_model, _ = make_model_and_optimizer(config, device)
    initial_state = clone_state_dict(initial_model)

    continuous_model, continuous_optimizer = make_model_and_optimizer(config, device)
    continuous_model.load_state_dict(initial_state)
    continuous_generator = torch.Generator().manual_seed(307)
    train_steps(
        continuous_model,
        continuous_optimizer,
        dataset,
        continuous_generator,
        device,
        start_step=0,
        end_step=total_steps,
        accumulation_steps=accumulation_steps,
        micro_batch_size=micro_batch_size,
    )

    interrupted_model, interrupted_optimizer = make_model_and_optimizer(config, device)
    interrupted_model.load_state_dict(initial_state)
    interrupted_generator = torch.Generator().manual_seed(307)
    train_steps(
        interrupted_model,
        interrupted_optimizer,
        dataset,
        interrupted_generator,
        device,
        start_step=0,
        end_step=interruption_step,
        accumulation_steps=accumulation_steps,
        micro_batch_size=micro_batch_size,
    )
    save_checkpoint(
        checkpoint_path,
        interrupted_model,
        interrupted_optimizer,
        completed_steps=interruption_step,
        sampling_generator=interrupted_generator,
    )

    resumed_model, resumed_optimizer = make_model_and_optimizer(config, device)
    resumed_generator = torch.Generator()
    completed_steps = load_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_generator,
        device,
    )
    assert completed_steps == interruption_step
    train_steps(
        resumed_model,
        resumed_optimizer,
        dataset,
        resumed_generator,
        device,
        start_step=completed_steps,
        end_step=total_steps,
        accumulation_steps=accumulation_steps,
        micro_batch_size=micro_batch_size,
    )

    maximum_difference = assert_state_dicts_close(
        continuous_model.state_dict(), resumed_model.state_dict(), atol=0.0
    )
    continuous_loss = evaluate_loss(continuous_model, dataset, device)
    resumed_loss = evaluate_loss(resumed_model, dataset, device)
    assert continuous_loss == resumed_loss
    return (
        continuous_loss,
        resumed_loss,
        maximum_difference,
        checkpoint_path.stat().st_size,
    )


def verify_invalid_checkpoint(
    path: Path,
    config: Config,
    device: torch.device,
) -> None:
    torch.save({"model_state_dict": {}}, path)
    model, optimizer = make_model_and_optimizer(config, device)
    generator = torch.Generator()
    try:
        load_checkpoint(path, model, optimizer, generator, device)
    except ValueError as error:
        assert "missing keys" in str(error)
    else:
        raise AssertionError("invalid checkpoint should have been rejected")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 6 requires the project's CUDA PyTorch environment.")

    device = torch.device("cuda")
    dataset = make_dataset()
    config = make_config()

    accumulation_difference = verify_accumulation_equivalence(
        config, dataset, device
    )

    with tempfile.TemporaryDirectory(prefix="ai-infra-day06-") as directory:
        directory_path = Path(directory)
        checkpoint_path = directory_path / "training_checkpoint.pt"
        invalid_checkpoint_path = directory_path / "invalid_checkpoint.pt"
        (
            continuous_loss,
            resumed_loss,
            resume_difference,
            checkpoint_size_bytes,
        ) = verify_checkpoint_resume(config, dataset, device, checkpoint_path)
        verify_invalid_checkpoint(invalid_checkpoint_path, config, device)

    torch.cuda.synchronize()
    print("Environment")
    print(f"  torch:                         {torch.__version__}")
    print(f"  compiled CUDA:                 {torch.version.cuda}")
    print(f"  device:                        {torch.cuda.get_device_name(0)}")
    print("\nGradient accumulation")
    print("  large batch size:              8")
    print("  micro-batch size:              4")
    print("  accumulation steps:            2")
    print(f"  maximum parameter difference:  {accumulation_difference:.9g}")
    print("\nCheckpoint resume")
    print("  total optimizer steps:         40")
    print("  interruption after step:       17")
    print(f"  continuous final loss:         {continuous_loss:.6f}")
    print(f"  resumed final loss:            {resumed_loss:.6f}")
    print(f"  maximum parameter difference:  {resume_difference:.9g}")
    print(f"  checkpoint size:               {checkpoint_size_bytes / 1024:.2f} KiB")
    print("\nBehavior checks")
    print("  accumulated update matches large batch:       PASS")
    print("  checkpoint restores exact training progress:  PASS")
    print("  continuous and resumed parameters match:      PASS")
    print("  incomplete checkpoint is rejected:            PASS")
    print("  temporary checkpoint files were cleaned:      PASS")


if __name__ == "__main__":
    main()
