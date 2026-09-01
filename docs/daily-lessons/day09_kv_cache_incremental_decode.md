# Day 9：KV Cache 与增量 Decode

## 1. 今日核心问题

Day 8 的无缓存生成每产生一个 token，都会重新计算整个前缀。今天要解决：如何保存每层
历史 token 的 Key 和 Value，使后续 decode 每步只输入一个新 token，同时保证它与完整
前缀计算得到相同的 logits 和 greedy token。

今天的最终验收不是“程序能生成”，而是：

```text
同一模型参数 + 同一 prompt + greedy decoding
→ cache/no-cache token IDs 完全相同
→ 每一步最后位置 logits 在 FP32 容差内一致
```

## 2. 前置知识与术语

- **KV Cache**：按 Decoder 层保存历史 token 的 Key 和 Value 张量。
- **Prefill**：一次处理完整 prompt，并为每层建立初始 K/V。
- **Incremental decode（增量解码）**：每一步只处理刚生成的一个 token，并把它的新 K/V
  追加到各层缓存。
- **Past K/V**：进入当前 decode step 前已经保存的历史 K/V。
- **Query length**：本次 forward 新输入的 token 数。增量 decode 中通常为 1。
- **Cache length**：当前缓存已经覆盖的 token 数。
- **State owner（状态所有者）**：负责持有、传入、更新和释放缓存的组件。本实验中是生成
  循环，而不是模型参数或 optimizer。

KV Cache 不缓存 Query。历史 Query 已经完成了读取任务；未来的新 token 只需要自己的
新 Query 去读取历史 K/V。

## 3. 从直觉到机制

可以把 self-attention 想成一次资料查询：

- Query 是当前 token 提出的问题；
- Key 是每个历史 token 的索引；
- Value 是索引命中后提供的内容。

过去 token 的 K/V 在参数不变时不会改变。无缓存实现却在每一步重新生成它们。KV Cache
把这些历史结果保存下来，让新 token 只生成自己的 Q/K/V：

```text
Prefill：prompt[P]
         → 每层生成 Q/K/V
         → 保存 K/V，cache length = P
         → 最后位置 logits 预测 token 1

Decode： token 1[1]
         → 每层只生成一个新 Q/K/V
         → 新 Q 读取 past K/V + new K/V
         → 追加 new K/V，cache length = P + 1
         → logits 预测 token 2
```

下一步输入的是刚生成的 token，而不是再次输入整个前缀。

### 3.1 为什么每层都有独立缓存

第 0 层和第 1 层接收的 hidden state 不同，而且各自的 `k_proj`、`v_proj` 参数不同。
因此即使它们描述同一批 token，各层 K/V 数值也不同，不能共享一份缓存。

本实验缓存结构是：

```text
past_key_values = [
    (layer_0_key, layer_0_value),
    (layer_1_key, layer_1_value),
]
```

缓存不是 `state_dict()` 中的模型参数。它属于一次生成请求的临时运行状态；请求结束、
取消或模型切换后，应由调用方释放。

### 3.2 Prefill 如何写缓存

prompt 长度为 5 时，每层先计算：

```text
Q, K, V: [B, heads, 5, head_dim]
```

Q 用于本轮 Attention，K/V 则作为第一版缓存返回。prefill 同时处理多个 Query，所以仍
需要 causal mask：prompt 的较早位置不能读取后面的 prompt token。

### 3.3 Decode 如何读取并追加缓存

假设 past K/V 长度为 5，新输入只有一个 token：

```text
new Q/K/V:      [B, heads, 1, head_dim]
past K/V:       [B, heads, 5, head_dim]
combined K/V:   [B, heads, 6, head_dim]
scores:         [B, heads, 1, 6]
```

新 Query 位于绝对位置 5，可以读取位置 0 到 5 的全部 K/V，没有它之后的未来 Key。
因此单 token decode 不需要遮掉历史位置。实验仍使用绝对位置比较构造通用 causal mask，
此时 mask 全为 `False`。

### 3.4 位置编号为什么必须带偏移

无缓存 forward 每次输入整个前缀，位置自然是 `0...S-1`。增量 decode 只输入一个 token，
若仍把它标成位置 0，位置 embedding 就与无缓存路径不同。

因此本轮位置从 `past_length` 开始：

```python
positions = torch.arange(past_length, past_length + query_length)
```

第一次 decode 的 `past_length=5`，新 token 的位置就是 5。对于 RoPE 模型，同样需要把
正确的位置偏移用于旋转 Q/K，只是具体实现不同。

## 4. 极小手算例子

设 `B=1`、`heads=2`、`head_dim=2`，prompt 长度为 3。

prefill 后某一层的缓存 shape：

```text
K: [1,2,3,2]
V: [1,2,3,2]
```

第一次 decode 输入一个 token：

```text
new Q/K/V: [1,2,1,2]
```

追加以后：

```text
combined K/V: [1,2,4,2]
scores:       [1,2,1,4]
output:       [1,2,1,2]
```

K 和 V 各有 `1 × 2 × 4 × 2 = 16` 个元素，因此这一层缓存共有 32 个元素。若 dtype 是
FP32，每个元素 4 bytes，这一层缓存占 `32 × 4 = 128 bytes`。

## 5. 正式模型与实验配置

Day 9 复用 Day 4/8 的同一个 `TinyDecoderLM` 和同一组参数：

```text
词表大小 V                 32
隐藏维度 H                 64
Attention heads            4
每个 head 维度             16
Decoder layers             2
最大序列长度               32
batch size B               1
prompt 长度 P              5
生成 token 数 N            8
参数与 cache dtype         FP32
生成策略                    greedy argmax
warmup                      每种方法 5 次
measured                    每种方法 20 次
```

对照实验的唯一主要变量是是否复用 K/V。模型参数、prompt、生成长度、dtype 和生成策略
保持一致。

## 6. 完整数据流与 Shape/Dtype

### 6.1 Prefill：step 0

```text
prompt IDs                 [1,5]       int64
hidden                     [1,5,64]    float32
每层 Q/K/V                [1,4,5,16]  float32
每层 scores               [1,4,5,5]   float32
每层返回 K cache          [1,4,5,16]  float32
每层返回 V cache          [1,4,5,16]  float32
logits                     [1,5,32]    float32
next token                 [1,1]       int64
```

### 6.2 第一次增量 Decode：step 1

```text
新 token ID                [1,1]       int64
位置编号                    [1]         int64，数值为 5
hidden                     [1,1,64]    float32
每层 new Q/K/V            [1,4,1,16]  float32
每层 combined K/V         [1,4,6,16]  float32
每层 scores               [1,4,1,6]   float32
logits                     [1,1,32]    float32
next token                 [1,1]       int64
```

后续 decode 的输入长度始终为 1，缓存长度依次变成 7、8、9、10、11、12。

### 6.3 为什么最终生成长度是 13，缓存长度却是 12

初始 prompt 有 5 个 token，随后生成 8 个，所以返回序列长度是 13。但第 8 个新 token
刚刚由长度 12 的状态预测出来；因为生成任务此时结束，它还没有再次进入模型，因而缓存
没有它的 K/V。若继续预测第 9 个 token，先输入它，缓存才会增长到 13。

## 7. 参数、内存与计算成本

### 7.1 KV Cache 容量

标准 MHA 中，逻辑缓存字节数为：

$$
2 \times L \times B \times S \times N_h \times D_h \times \text{dtype bytes}
$$

其中：

- `2` 表示 K 和 V；
- $L$ 是 Decoder 层数；
- $B$ 是 batch 或序列数；
- $S$ 是已缓存长度；
- $N_h$ 是 KV head 数，本实验 MHA 中等于 Query head 数；
- $D_h$ 是每个 head 的维度。

本实验最终缓存长度为 12：

```text
2 × 2 layers × 1 batch × 12 tokens × 4 heads × 16 dims × 4 bytes
= 12,288 bytes
= 0.01172 MiB
```

这是 Tensor 逻辑数据量，不包括 allocator、Tensor 元数据、临时拼接和碎片。

### 7.2 处理的 token 位置

生成 $N$ 个 token 时，本实验的 cached path 处理：

$$
P + (N-1)
$$

因为 prefill 处理 $P$ 个 prompt token，后续只有前 $N-1$ 个生成 token 需要再次进入模型。
本实验为 `5 + 7 = 12`，无缓存路径则处理 68 个位置。

### 7.3 Attention score 元素

prefill 仍构造 $P \times P$ 的 score；第 $t$ 个 decode Query 只产生一行、读取当前全部
K/V。本实验两层四 head 的逻辑元素数为：

```text
2 × 4 × (5² + 6 + 7 + 8 + 9 + 10 + 11 + 12)
= 704
```

无缓存基线是 4,960 个，逻辑 score 元素减少约 7.05 倍。

### 7.4 为什么当前实验没有加速

当前实测 cache/no-cache 时间比为 `1.114x`，即缓存版本反而慢约 11.4%。这不与计算量
下降矛盾，原因包括：

- 模型和序列太小，GPU 矩阵并行度很低；
- 每个 token、每一层都有 Python 调度和许多小 kernel；
- 教学实现用 `torch.cat` 追加 K/V，每步会分配并复制缓存；
- 较大的完整矩阵有时比许多极小矩阵更容易利用 GPU；
- 没有 fused attention、预分配缓存或 CUDA Graph。

因此 KV Cache 减少重复数学工作是已验证事实；“当前实现一定更快”不是事实。长上下文、
大模型和生产 kernel 下通常更有机会兑现收益，但仍应实测。

## 8. 最小代码验证

实验文件：

```text
exercises/day09/kv_cache_generation.py
```

从项目根目录运行：

```bash
uv run python -m exercises.day09.kv_cache_generation
```

代码没有创建第二套模型参数，而是直接调用 Day 4 模型各层的 Norm、Q/K/V/O 投影、FFN
与 LM Head。它验证：

- 每层都有一个 K/V 对，shape 随 cache length 增长；
- 追加缓存后，旧 K/V 前缀逐元素完全保留；
- 正确的位置 offset 被用于 token 和 position embedding 相加；
- 每一步 cached logits 与完整前缀最后 logits 在容差内一致；
- greedy token IDs 全序列完全相同；
- 缓存元素数、字节数和 Attention score 元素数符合手算。

### 8.1 当前机器实际输出

2026-08-30 在 NVIDIA GeForce RTX 2060、PyTorch 2.13.0+cu130 上得到：

```text
generated token IDs:            [1, 5, 9, 4, 3, 17, 3, 29, 12, 27, 22, 13, 8]
prefill input/cache length:      5 / 5
decode input lengths:            1, 1, 1, 1, 1, 1, 1
decode cache lengths:            6, 7, 8, 9, 10, 11, 12
final layer K/V shape:           [1,4,12,16]
cached token positions:          12
cached attention score elements: 704
final cache size:                0.01172 MiB
maximum logit difference:        2.3841858e-07
no-cache average time:           5.394 ms
KV-cache average time:           6.009 ms
time ratio cache/no-cache:       1.114x
no-cache CUDA peak allocated:    8.52 MiB
KV-cache CUDA peak allocated:    8.51 MiB
cache/no-cache token IDs:        EXACT MATCH
cache/no-cache logits:           CLOSE
old cache preservation:          PASS
cache shape/byte accounting:     PASS
```

计时前后使用 `torch.cuda.synchronize()`，每种路径 warmup 5 次、测量 20 次。peak allocated
是 PyTorch allocator 统计，并不等于 `nvidia-smi` 进程占用或缓存逻辑字节数。

## 9. 常见误解与边界

### 9.1 “KV Cache 保存整个 hidden state”

本实验保存的是每层 Attention 的 K/V，不是所有 hidden state，也不保存历史 Q、logits、
Attention weights 或 FFN 中间激活。

### 9.2 “缓存是模型参数的一部分”

缓存由当前请求的 token、模型参数和位置共同产生。它不参与训练更新，也不应写进普通
模型 `state_dict()`。不同请求需要独立缓存。

### 9.3 “有缓存就完全不需要 causal mask”

prefill 同时处理多个 prompt token，仍需要 causal mask。单 token decode 的新 Query
前方只有合法历史位置，所以不需要完整三角遮罩；若一次追加多个 token，新增 token 之间
仍需保持因果关系。

### 9.4 “cache/no-cache logits 必须逐 bit 相同”

两条路径使用不同矩阵 shape，底层 kernel 和浮点累加顺序可能不同。因此应在合理浮点
容差内比较 logits。本实验最大差异约 `2.38e-07`。如果最大和第二大 logit 极接近，这种
微小差异仍可能让 argmax 翻转，所以 token 一致性也必须单独验证。

### 9.5 “KV Cache 越长越省显存”

它省的是重复计算，不是缓存容量。缓存会随 batch、层数、序列长度、KV heads、head dim
和 dtype 线性增长。高并发长上下文服务中，它可能成为主要显存压力。

### 9.6 当前尚未验证

- MQA/GQA 减少 KV heads 后的容量与数据流，安排在 Day 10；
- 预分配或 block-based cache，当前 `torch.cat` 会重复分配；
- padding、不同长度 batch、beam search 中的 cache 重排；
- EOS、请求取消以及缓存释放生命周期；
- Paged KV Cache、prefix caching 和 continuous batching；
- 真实大模型与长上下文上的加速和带宽瓶颈。

## 10. 手算练习

### 练习 1

某模型有 4 层、8 个 KV heads、head dim 为 16，batch 为 2，缓存长度为 10，dtype 为
FP16。KV Cache 逻辑字节数是多少？

答案：

```text
2(K/V) × 4 × 2 × 10 × 8 × 16 × 2 bytes = 40,960 bytes
```

### 练习 2

prompt 长度为 4，要生成 5 个 token。cached path 一共处理多少 token 位置？为什么不是
9？

答案：`4 + (5-1) = 8`。第 5 个生成 token 是最后一次 forward 的输出，任务结束后没有
再把它作为输入处理；如果还要预测下一个 token，才会处理它。

### 练习 3

past K shape 是 `[2,4,7,16]`，new K shape 是 `[2,4,1,16]`。沿哪个维度拼接，输出
shape 是什么？

答案：沿序列/缓存长度维 `dim=2` 拼接，得到 `[2,4,8,16]`。不能沿 head 或 head dim
拼接，因为 token 数增长，head 结构不变。

## 11. 面试口述

### 问题 1：KV Cache 保存什么，为什么不保存 Query？

30 秒回答：每层保存历史 token 的 K 和 V。未来 token 的新 Query 需要读取它们，而历史
Query 已经完成了当时的读取，不会再被未来 Query 使用，因此没有必要缓存历史 Query。

### 问题 2：Prefill 和 cached decode 的 shape 有何区别？

30 秒回答：prefill 的 Q/K/V 都覆盖整个 prompt，例如 `[B,h,P,d]`，score 是
`[B,h,P,P]`。单 token decode 的新 Q/K/V 是 `[B,h,1,d]`，新 Q 读取拼接后的全部 K/V，
score 是 `[B,h,1,S]`，随后缓存长度增加 1。

### 问题 3：为什么 KV Cache 减少计算却可能没有加速？

2 分钟回答要点：它消除了旧 token 的 Q/K/V、Block 和方形 Attention 重算，但真实时间
还受 kernel launch、矩阵大小、GPU 利用率、缓存追加复制、内存带宽和实现融合影响。小模型
短序列下，单 token 小 kernel 与 Python 开销可能超过省下的计算；生产实现会用预分配或
分页缓存、融合 kernel 和批处理改善，但最终仍要在目标配置上测量。

## 12. 当日验收

- [ ] 能画出每层 `past K/V + new K/V` 的读写过程。
- [ ] 能解释为什么缓存 K/V 而不缓存历史 Q。
- [ ] 能从 `past_length` 推导新 token 的位置编号。
- [ ] 能手算给定配置的 KV Cache 字节数。
- [ ] 能解释最终生成长度 13、缓存长度 12 的原因。
- [ ] 能运行实验并解释 logits 近似一致而 token IDs 完全一致。
- [ ] 能解释为什么当前缓存版本减少逻辑工作却没有加速。

下一步 Day 10：比较 MHA、MQA 和 GQA 中 Query heads 与 KV heads 的关系，手算不同结构
的 KV Cache 容量，并验证减少 KV heads 如何改变缓存 shape，而不是同比减少 Query heads。
