"""Day 5: overfit a tiny token dataset with a complete CUDA training loop.

Run from the project root:
    uv run python exercises/day05/overfit_training_loop.py
"""

from __future__ import annotations

import sys
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


TRAINING_SEQUENCES = (
    (1, 9, 17, 25, 29, 0),
    (2, 10, 18, 26, 30, 0),
    (3, 11, 19, 27, 29, 0),
    (4, 12, 20, 28, 30, 0),
    (5, 13, 21, 25, 29, 0),
    (6, 14, 22, 26, 30, 0),
    (7, 15, 23, 27, 29, 0),
    (8, 16, 24, 28, 30, 0),
)


def make_dataset() -> Tensor:
    dataset = torch.tensor(TRAINING_SEQUENCES, dtype=torch.long)
    assert dataset.ndim == 2
    assert dataset.min().item() >= 0
    return dataset


def sample_batch(
    dataset: Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    indices = torch.randint(
        low=0,
        high=dataset.size(0),
        size=(batch_size,),
        generator=generator,
    )
    complete_sequences = dataset[indices].to(device)
    return shifted_language_model_batch(complete_sequences)


@torch.no_grad()
def evaluate(
    model: TinyDecoderLM,
    dataset: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    input_ids, labels = shifted_language_model_batch(dataset.to(device))
    logits, _ = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocabulary_size), labels.reshape(-1)
    )
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    return loss.item(), accuracy.item()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 5 requires the project's CUDA PyTorch environment.")

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    sampling_generator = torch.Generator().manual_seed(23)
    device = torch.device("cuda")

    dataset = make_dataset()
    config = Config(
        vocabulary_size=32,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        num_layers=2,
        max_sequence_length=16,
    )
    model = TinyDecoderLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.01
    )

    initial_loss, initial_accuracy = evaluate(model, dataset, device)
    steps = 300
    batch_size = 4
    progress: list[tuple[int, float]] = []

    model.train()
    for step in range(1, steps + 1):
        input_ids, labels = sample_batch(
            dataset,
            batch_size=batch_size,
            generator=sampling_generator,
            device=device,
        )
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocabulary_size), labels.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            progress.append((step, loss.item()))

    torch.cuda.synchronize()
    final_loss, final_accuracy = evaluate(model, dataset, device)

    assert final_loss < initial_loss * 0.1
    assert final_loss < 0.05
    assert final_accuracy == 1.0

    first_parameter = next(model.parameters())
    optimizer_state = optimizer.state[first_parameter]
    assert "step" in optimizer_state
    assert "exp_avg" in optimizer_state
    assert "exp_avg_sq" in optimizer_state
    assert optimizer_state["exp_avg"].shape == first_parameter.shape
    assert optimizer_state["exp_avg_sq"].shape == first_parameter.shape

    print("Environment")
    print(f"  torch:                 {torch.__version__}")
    print(f"  compiled CUDA:         {torch.version.cuda}")
    print(f"  device:                {torch.cuda.get_device_name(0)}")
    print("\nDataset and loop")
    print(f"  dataset shape:         {tuple(dataset.shape)}")
    print(f"  training predictions:  {dataset.size(0) * (dataset.size(1) - 1)}")
    print(f"  batch size:            {batch_size}")
    print(f"  optimizer steps:       {steps}")
    print("\nSampled-batch training loss")
    for step, recorded_loss in progress:
        print(f"  step {step:3d}:              {recorded_loss:.6f}")
    print("\nFull-dataset evaluation")
    print(f"  initial loss:          {initial_loss:.6f}")
    print(f"  final loss:            {final_loss:.6f}")
    print(f"  initial accuracy:      {initial_accuracy:.2%}")
    print(f"  final accuracy:        {final_accuracy:.2%}")
    print("\nOptimizer state for one parameter")
    print(f"  parameter shape:       {tuple(first_parameter.shape)}")
    print(f"  exp_avg shape:         {tuple(optimizer_state['exp_avg'].shape)}")
    print(f"  exp_avg_sq shape:      {tuple(optimizer_state['exp_avg_sq'].shape)}")
    print("\nBehavior checks")
    print("  full-dataset loss fell by more than 90%: PASS")
    print("  final full-dataset loss is below 0.05:    PASS")
    print("  teacher-forced token accuracy is 100%:    PASS")
    print("  AdamW first/second-moment states exist:   PASS")


if __name__ == "__main__":
    main()
