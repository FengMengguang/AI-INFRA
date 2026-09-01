"""Day 1: verify Transformer tensor shapes, parameter counts, and memory.

This script intentionally uses only the Python standard library.
"""

from dataclasses import dataclass
from math import prod


DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


@dataclass(frozen=True)
class Config:
    batch_size: int = 2
    sequence_length: int = 128
    vocabulary_size: int = 8_000
    hidden_size: int = 384
    query_heads: int = 6
    kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1_024
    layers: int = 6

    def validate(self) -> None:
        assert self.hidden_size == self.query_heads * self.head_dim
        assert self.query_heads % self.kv_heads == 0


def numel(shape: tuple[int, ...]) -> int:
    return prod(shape)


def memory_bytes(shape: tuple[int, ...], dtype: str) -> int:
    return numel(shape) * DTYPE_BYTES[dtype]


def mib(byte_count: int) -> float:
    return byte_count / 1024**2


def main() -> None:
    cfg = Config()
    cfg.validate()

    b = cfg.batch_size
    s = cfg.sequence_length
    h = cfg.hidden_size
    nq = cfg.query_heads
    nkv = cfg.kv_heads
    d = cfg.head_dim
    i = cfg.intermediate_size
    v = cfg.vocabulary_size

    shapes = {
        "input_ids": (b, s),
        "hidden": (b, s, h),
        "query": (b, nq, s, d),
        "key": (b, nkv, s, d),
        "value": (b, nkv, s, d),
        "attention_scores": (b, nq, s, s),
        "ffn_gate_or_up": (b, s, i),
        "logits": (b, s, v),
    }

    print("Tensor shapes and theoretical memory")
    for name, shape in shapes.items():
        fp32 = mib(memory_bytes(shape, "fp32"))
        fp16 = mib(memory_bytes(shape, "fp16"))
        print(
            f"{name:20s} shape={str(shape):18s} "
            f"numel={numel(shape):>10,d} fp32={fp32:>8.3f} MiB "
            f"fp16={fp16:>8.3f} MiB"
        )

    embedding = v * h
    attention = h * (nq * d) + 2 * h * (nkv * d) + h * h
    ffn = 3 * h * i
    norms_per_block = 2 * h
    block = attention + ffn + norms_per_block
    final_norm = h
    tied_total = embedding + cfg.layers * block + final_norm
    untied_total = tied_total + h * v

    print("\nParameter counts (biases omitted)")
    print(f"embedding:          {embedding:>12,d}")
    print(f"attention/block:    {attention:>12,d}")
    print(f"ffn/block:          {ffn:>12,d}")
    print(f"whole block:        {block:>12,d}")
    print(f"model, tied head:   {tied_total:>12,d}")
    print(f"model, untied head: {untied_total:>12,d}")

    short_scores = (b, nq, s, s)
    long_s = 512
    long_scores = (b, nq, long_s, long_s)
    print("\nSequence-length pressure")
    print(f"S={s}:   score elements={numel(short_scores):,}")
    print(f"S={long_s}: score elements={numel(long_scores):,}")
    print(f"growth={numel(long_scores) / numel(short_scores):.0f}x")


if __name__ == "__main__":
    main()
