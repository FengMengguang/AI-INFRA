"""Day 3: implement and verify a tiny causal multi-head attention sublayer.

This script uses only the Python standard library.  It is intentionally small
and explicit: the goal is to expose the data flow, not to provide a fast model.
"""

from __future__ import annotations

from math import exp, sqrt


Matrix = list[list[float]]
Heads = list[Matrix]


EPS = 1e-6


def shape_2d(values: Matrix) -> tuple[int, int]:
    assert values, "expected at least one row"
    width = len(values[0])
    assert width > 0
    assert all(len(row) == width for row in values)
    return len(values), width


def transpose(values: Matrix) -> Matrix:
    rows, columns = shape_2d(values)
    return [[values[row][column] for row in range(rows)] for column in range(columns)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, shared = shape_2d(left)
    right_rows, right_columns = shape_2d(right)
    assert shared == right_rows
    return [
        [
            sum(left[row][k] * right[k][column] for k in range(shared))
            for column in range(right_columns)
        ]
        for row in range(left_rows)
    ]


def add(left: Matrix, right: Matrix) -> Matrix:
    assert shape_2d(left) == shape_2d(right)
    return [
        [left_value + right_value for left_value, right_value in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def rms_norm(hidden: Matrix, weight: list[float], eps: float = EPS) -> Matrix:
    _, hidden_size = shape_2d(hidden)
    assert len(weight) == hidden_size
    normalized: Matrix = []
    for vector in hidden:
        rms = sqrt(sum(value * value for value in vector) / hidden_size + eps)
        normalized.append(
            [value / rms * scale for value, scale in zip(vector, weight)]
        )
    return normalized


def split_heads(hidden: Matrix, num_heads: int) -> Heads:
    sequence_length, hidden_size = shape_2d(hidden)
    assert hidden_size % num_heads == 0
    head_dim = hidden_size // num_heads
    return [
        [
            hidden[position][head * head_dim : (head + 1) * head_dim]
            for position in range(sequence_length)
        ]
        for head in range(num_heads)
    ]


def merge_heads(heads: Heads) -> Matrix:
    assert heads
    sequence_length, head_dim = shape_2d(heads[0])
    assert all(shape_2d(head) == (sequence_length, head_dim) for head in heads)
    return [
        [value for head in heads for value in head[position]]
        for position in range(sequence_length)
    ]


def softmax(row: list[float]) -> list[float]:
    maximum = max(row)
    exponentials = [0.0 if value == float("-inf") else exp(value - maximum) for value in row]
    denominator = sum(exponentials)
    assert denominator > 0.0
    return [value / denominator for value in exponentials]


def causal_attention(
    query: Matrix,
    key: Matrix,
    value: Matrix,
    *,
    use_causal_mask: bool,
) -> tuple[Matrix, Matrix, Matrix]:
    sequence_length, head_dim = shape_2d(query)
    assert shape_2d(key) == (sequence_length, head_dim)
    assert shape_2d(value) == (sequence_length, head_dim)

    raw_scores = matmul(query, transpose(key))
    scaled_scores = [
        [score / sqrt(head_dim) for score in row] for row in raw_scores
    ]

    masked_scores: Matrix = []
    for query_position, row in enumerate(scaled_scores):
        masked_scores.append(
            [
                float("-inf")
                if use_causal_mask and key_position > query_position
                else score
                for key_position, score in enumerate(row)
            ]
        )

    attention_weights = [softmax(row) for row in masked_scores]
    output = matmul(attention_weights, value)
    return output, masked_scores, attention_weights


def multi_head_attention(
    hidden: Matrix,
    *,
    num_heads: int,
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    use_causal_mask: bool,
) -> tuple[Matrix, list[Matrix], list[Matrix]]:
    query = matmul(hidden, w_q)
    key = matmul(hidden, w_k)
    value = matmul(hidden, w_v)

    query_heads = split_heads(query, num_heads)
    key_heads = split_heads(key, num_heads)
    value_heads = split_heads(value, num_heads)

    head_outputs: Heads = []
    all_scores: list[Matrix] = []
    all_weights: list[Matrix] = []
    for query_head, key_head, value_head in zip(
        query_heads, key_heads, value_heads
    ):
        head_output, scores, weights = causal_attention(
            query_head,
            key_head,
            value_head,
            use_causal_mask=use_causal_mask,
        )
        head_outputs.append(head_output)
        all_scores.append(scores)
        all_weights.append(weights)

    concatenated = merge_heads(head_outputs)
    return matmul(concatenated, w_o), all_scores, all_weights


def causal_attention_sublayer(
    hidden: Matrix,
    *,
    num_heads: int,
    projection: Matrix,
    norm_weight: list[float],
    use_causal_mask: bool,
) -> tuple[Matrix, list[Matrix], list[Matrix]]:
    normalized = rms_norm(hidden, norm_weight)
    attention_output, scores, weights = multi_head_attention(
        normalized,
        num_heads=num_heads,
        w_q=projection,
        w_k=projection,
        w_v=projection,
        w_o=projection,
        use_causal_mask=use_causal_mask,
    )
    return add(hidden, attention_output), scores, weights


def rounded(values: Matrix) -> Matrix:
    return [[round(value, 4) for value in row] for row in values]


def close(left: list[float], right: list[float], tolerance: float = 1e-9) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def main() -> None:
    sequence_length = 4
    hidden_size = 4
    num_heads = 2
    head_dim = hidden_size // num_heads

    hidden: Matrix = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.5, 1.0, 1.5, 0.0],
    ]
    identity: Matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    norm_weight = [1.0] * hidden_size

    output, scores, weights = causal_attention_sublayer(
        hidden,
        num_heads=num_heads,
        projection=identity,
        norm_weight=norm_weight,
        use_causal_mask=True,
    )

    assert shape_2d(output) == (sequence_length, hidden_size)
    assert len(scores) == num_heads
    assert len(weights) == num_heads
    assert all(shape_2d(head) == (sequence_length, sequence_length) for head in scores)
    assert all(shape_2d(head) == (sequence_length, sequence_length) for head in weights)
    assert all(
        weights[head][query_position][key_position] == 0.0
        for head in range(num_heads)
        for query_position in range(sequence_length)
        for key_position in range(query_position + 1, sequence_length)
    )

    changed_future = [row[:] for row in hidden]
    changed_future[-1] = [100.0, -100.0, 50.0, -50.0]
    causal_changed, _, _ = causal_attention_sublayer(
        changed_future,
        num_heads=num_heads,
        projection=identity,
        norm_weight=norm_weight,
        use_causal_mask=True,
    )
    assert all(close(output[position], causal_changed[position]) for position in range(3))

    unmasked_original, _, _ = causal_attention_sublayer(
        hidden,
        num_heads=num_heads,
        projection=identity,
        norm_weight=norm_weight,
        use_causal_mask=False,
    )
    unmasked_changed, _, _ = causal_attention_sublayer(
        changed_future,
        num_heads=num_heads,
        projection=identity,
        norm_weight=norm_weight,
        use_causal_mask=False,
    )
    assert any(
        not close(unmasked_original[position], unmasked_changed[position])
        for position in range(3)
    )

    print("Configuration")
    print(f"  S={sequence_length}, H={hidden_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"  hidden shape:           {shape_2d(hidden)}")
    print(f"  one-head scores shape:  {shape_2d(scores[0])}")
    print(f"  sublayer output shape:  {shape_2d(output)}")
    print("\nHead 0 masked scores (-inf is the causal mask)")
    print(rounded(scores[0]))
    print("\nHead 0 attention weights")
    print(rounded(weights[0]))
    print("\nResidual sublayer output")
    print(rounded(output))
    print("\nCausality checks")
    print("  future-token change leaves earlier causal outputs unchanged: PASS")
    print("  future-token change affects earlier unmasked outputs:          PASS")


if __name__ == "__main__":
    main()
