"""Day 7: run the phase-one end-to-end acceptance checks on CUDA.

Run from the project root:
    uv run python exercises/day07/phase_one_acceptance.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from exercises.day04.decoder_block_training import (  # noqa: E402
    TinyDecoderLM,
    assert_causality,
    shifted_language_model_batch,
)
from exercises.day05.overfit_training_loop import make_dataset  # noqa: E402
from exercises.day06.gradient_accumulation_checkpoint import (  # noqa: E402
    make_config,
    verify_accumulation_equivalence,
    verify_checkpoint_resume,
    verify_invalid_checkpoint,
)


def verify_one_training_step(
    model: TinyDecoderLM,
    dataset: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, tuple[int, ...]]:
    input_ids, labels = shifted_language_model_batch(dataset[:4].to(device))
    assert input_ids.shape == (4, 5)
    assert labels.shape == (4, 5)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.01
    )
    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(input_ids)
    assert logits.shape == (4, 5, model.config.vocabulary_size)
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocabulary_size), labels.reshape(-1)
    )
    loss.backward()

    assert first_parameter.grad is not None
    assert first_parameter.grad.shape == first_parameter.shape
    assert torch.isfinite(first_parameter.grad).all()
    gradient_norm = first_parameter.grad.norm().item()

    optimizer.step()
    parameter_change = (first_parameter.detach() - before).abs().max().item()
    assert parameter_change > 0.0
    return loss.item(), gradient_norm, tuple(logits.shape)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 7 requires the project's CUDA PyTorch environment.")

    torch.manual_seed(701)
    device = torch.device("cuda")
    dataset = make_dataset()
    config = make_config()
    model = TinyDecoderLM(config).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    assert_causality(model, dataset[:2].to(device))
    loss, gradient_norm, logits_shape = verify_one_training_step(
        model, dataset, device
    )
    accumulation_difference = verify_accumulation_equivalence(
        config, dataset, device
    )

    with tempfile.TemporaryDirectory(prefix="ai-infra-day07-") as directory:
        directory_path = Path(directory)
        checkpoint_path = directory_path / "phase_one_checkpoint.pt"
        invalid_path = directory_path / "invalid_checkpoint.pt"
        (
            continuous_loss,
            resumed_loss,
            resume_difference,
            checkpoint_size_bytes,
        ) = verify_checkpoint_resume(config, dataset, device, checkpoint_path)
        verify_invalid_checkpoint(invalid_path, config, device)

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    print("Environment")
    print(f"  torch:                         {torch.__version__}")
    print(f"  compiled CUDA:                 {torch.version.cuda}")
    print(f"  device:                        {torch.cuda.get_device_name(device)}")
    print("\nEnd-to-end training step")
    print(f"  dataset shape:                 {tuple(dataset.shape)}")
    print(f"  logits shape:                  {logits_shape}")
    print(f"  cross-entropy loss:            {loss:.6f}")
    print(f"  first-parameter grad norm:     {gradient_norm:.6f}")
    print("\nAccumulation and recovery")
    print(f"  accumulation max difference:   {accumulation_difference:.9g}")
    print(f"  continuous final loss:         {continuous_loss:.6f}")
    print(f"  resumed final loss:            {resumed_loss:.6f}")
    print(f"  resume max difference:         {resume_difference:.9g}")
    print(f"  temporary checkpoint size:     {checkpoint_size_bytes / 1024:.2f} KiB")
    print(f"  CUDA peak allocated:           {peak_allocated / 1024**2:.2f} MiB")
    print("\nPhase-one behavior checks")
    print("  shifted-label shapes are correct:          PASS")
    print("  causal independence is preserved:          PASS")
    print("  forward/loss/backward/AdamW step works:    PASS")
    print("  accumulated and large-batch updates match: PASS")
    print("  interrupted training resumes exactly:      PASS")
    print("  invalid checkpoint is rejected:            PASS")
    print("  temporary checkpoint files were cleaned:  PASS")


if __name__ == "__main__":
    main()

