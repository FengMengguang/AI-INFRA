"""Day 2: verify the data flow from text to logits without third-party packages.

This is a shape-and-semantics exercise, not a real Transformer implementation.
"""

from __future__ import annotations


VOCAB = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "我": 3,
    "喜欢": 4,
    "学习": 5,
    "AI": 6,
    "你": 7,
}

PIECES = ("喜欢", "学习", "AI", "我", "你")
HIDDEN_SIZE = 4
NUM_BLOCKS = 2


def shape_2d(values: list[list[object]]) -> tuple[int, int]:
    assert values, "expected at least one row"
    width = len(values[0])
    assert all(len(row) == width for row in values)
    return len(values), width


def shape_3d(values: list[list[list[float]]]) -> tuple[int, int, int]:
    batch, sequence = shape_2d(values)
    hidden = len(values[0][0])
    assert all(len(vector) == hidden for row in values for vector in row)
    return batch, sequence, hidden


def tokenize(text: str) -> list[str]:
    """Greedy longest-match tokenizer for the tiny teaching vocabulary."""
    tokens: list[str] = []
    cursor = 0
    while cursor < len(text):
        piece = next((p for p in PIECES if text.startswith(p, cursor)), None)
        if piece is None:
            unknown = text[cursor]
            byte_values = " ".join(f"{value:02X}" for value in unknown.encode("utf-8"))
            raise ValueError(
                f"tiny vocabulary cannot encode {unknown!r}; UTF-8 fallback bytes would be {byte_values}"
            )
        tokens.append(piece)
        cursor += len(piece)
    return tokens


def encode(text: str) -> list[int]:
    return [VOCAB["<BOS>"], *(VOCAB[token] for token in tokenize(text)), VOCAB["<EOS>"]]


def pad(sequences: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    max_length = max(map(len, sequences))
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for sequence in sequences:
        pad_count = max_length - len(sequence)
        input_ids.append(sequence + [VOCAB["<PAD>"]] * pad_count)
        attention_mask.append([1] * len(sequence) + [0] * pad_count)
    return input_ids, attention_mask


def make_embedding_table(vocab_size: int, hidden_size: int) -> list[list[float]]:
    # Deterministic values make the exercise reproducible; real weights are learned.
    return [
        [(token_id + 1) * 0.1 + dimension * 0.01 for dimension in range(hidden_size)]
        for token_id in range(vocab_size)
    ]


def embedding_lookup(
    input_ids: list[list[int]], table: list[list[float]]
) -> list[list[list[float]]]:
    return [[table[token_id][:] for token_id in row] for row in input_ids]


def add_absolute_positions(
    hidden: list[list[list[float]]],
) -> list[list[list[float]]]:
    return [
        [
            [value + position * 0.001 for value in vector]
            for position, vector in enumerate(row)
        ]
        for row in hidden
    ]


def placeholder_block(hidden: list[list[list[float]]]) -> list[list[list[float]]]:
    # Preserve shape while changing values. Day 3 replaces this with real operations.
    return [
        [[value + 0.0001 for value in vector] for vector in row]
        for row in hidden
    ]


def placeholder_lm_head(
    hidden: list[list[list[float]]], vocab_size: int
) -> list[list[list[float]]]:
    # Produce deterministic logits with the correct logical shape.
    return [
        [
            [sum(vector) * (token_id + 1) / vocab_size for token_id in range(vocab_size)]
            for vector in row
        ]
        for row in hidden
    ]


def main() -> None:
    texts = ["我喜欢学习AI", "你"]

    print("UTF-8 bytes:")
    for text in texts:
        print(f"  {text!r} -> {text.encode('utf-8').hex(' ')}")

    encoded = [encode(text) for text in texts]
    input_ids, attention_mask = pad(encoded)

    assert shape_2d(input_ids) == (2, 6)
    assert shape_2d(attention_mask) == (2, 6)
    assert input_ids == [[1, 3, 4, 5, 6, 2], [1, 7, 2, 0, 0, 0]]
    assert attention_mask == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]

    embedding_table = make_embedding_table(len(VOCAB), HIDDEN_SIZE)
    hidden = embedding_lookup(input_ids, embedding_table)
    assert shape_2d(embedding_table) == (8, 4)
    assert shape_3d(hidden) == (2, 6, 4)

    positioned_hidden = add_absolute_positions(hidden)
    assert shape_3d(positioned_hidden) == (2, 6, 4)
    assert positioned_hidden[0][0] != positioned_hidden[0][1]

    block_hidden = positioned_hidden
    for _ in range(NUM_BLOCKS):
        block_hidden = placeholder_block(block_hidden)
        assert shape_3d(block_hidden) == (2, 6, 4)

    logits = placeholder_lm_head(block_hidden, len(VOCAB))
    assert shape_3d(logits) == (2, 6, 8)

    last_valid_positions = [sum(mask) - 1 for mask in attention_mask]
    next_token_logits = [
        logits[batch_index][position]
        for batch_index, position in enumerate(last_valid_positions)
    ]
    assert shape_2d(next_token_logits) == (2, 8)

    print(f"input_ids:             {input_ids}")
    print(f"attention_mask:        {attention_mask}")
    print(f"input_ids shape:       {shape_2d(input_ids)}")
    print(f"embedding table shape: {shape_2d(embedding_table)}")
    print(f"hidden shape:          {shape_3d(hidden)}")
    print(f"after position shape:  {shape_3d(positioned_hidden)}")
    print(f"after blocks shape:    {shape_3d(block_hidden)}")
    print(f"logits shape:          {shape_3d(logits)}")
    print(f"last valid positions:  {last_valid_positions}")
    print(f"next logits shape:     {shape_2d(next_token_logits)}")


if __name__ == "__main__":
    main()
