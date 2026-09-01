# Day 10：MHA、MQA、GQA 与 KV Cache 容量

## 1. 今日核心问题

Day 9 的标准 Multi-Head Attention 为每个 Query head 保存一组独立 K/V。序列变长、并发
增加后，KV Cache 会占用大量显存并产生读取带宽压力。今天要回答：MQA 和 GQA 如何通过
减少 KV heads 降低这部分成本，为什么 Query heads 通常保持不变，以及怎样从模型配置
手算缓存容量。

今天的验收是：给出层数、batch、序列长度、Query heads、KV heads、head dimension 和
dtype，能够写出 Q/K/V shape，并算出 KV Cache 的元素数与字节数。

## 2. 前置知识与术语

- **MHA（Multi-Head Attention）**：多头注意力，每个 Query head 都有自己的 K/V head。
- **MQA（Multi-Query Attention）**：多查询注意力，多个 Query heads 共享唯一一组 K/V。
- **GQA（Grouped-Query Attention）**：分组查询注意力，把 Query heads 分组，每组共享一组
  K/V，是 MHA 与 MQA 之间的折中。
- **Query heads**：提出不同查询、形成多组 Attention 输出的 head。
- **KV heads**：产生并缓存 Key/Value 的 head 数。
- **Head dimension**：每个 head 的特征维度，记作 $D_h$。
- **Group size**：每个 KV head 服务多少个 Query heads。

设 Query heads 为 $N_q$、KV heads 为 $N_{kv}$，则：

$$
\text{group size}=\frac{N_q}{N_{kv}}
$$

通常要求 $N_q$ 能被 $N_{kv}$ 整除。

## 3. 从直觉到机制

假设有 8 位提问者，也就是 8 个 Query heads。

### 3.1 MHA

每位提问者都有自己的一套资料索引和内容：

```text
Q0 → K0/V0
Q1 → K1/V1
...
Q7 → K7/V7
```

因此 `Query heads = 8`、`KV heads = 8`，每个 KV head 只服务一个 Query head。

### 3.2 MQA

8 位提问者保留各自的问题，但共享同一套资料：

```text
Q0 ─┐
Q1  │
... ├→ K0/V0
Q7 ─┘
```

因此 `Query heads = 8`、`KV heads = 1`。缓存只有 MHA 的八分之一。

### 3.3 GQA

把 8 位提问者分成两组，每组四人共享资料：

```text
Q0,Q1,Q2,Q3 → K0/V0
Q4,Q5,Q6,Q7 → K1/V1
```

因此 `Query heads = 8`、`KV heads = 2`，缓存是 MHA 的四分之一。GQA 在保留多组 K/V
表达能力和降低缓存成本之间提供中间选择。

### 3.4 为什么不同比例减少 Query heads

KV Cache 只保存历史 K/V，不保存历史 Query。减少 KV heads 会直接减少需要长期保存和
每步读取的状态；减少 Query heads 并不会同比缩小 KV Cache，除非同时改变 KV heads。

Query heads 还承担多组查询和多组 Attention 输出的表达能力。MQA/GQA 的设计目标是让
这些查询继续从不同角度读取上下文，只共享被读取的 K/V。如果把 Query heads 也从 8
降到 1，就变成单头 Attention，改变的不只是缓存成本，而是整个 Attention 表示结构。

这不代表 Query heads 越多一定越好，也不代表共享 K/V 完全没有质量代价。具体 head 数
是模型架构与训练的选择，需要通过模型质量和目标硬件实测，不能只凭容量公式决定。

### 3.5 多头结果为什么还要合并并经过 `Wo`

每个 Query head 得到 `[B,S,D_h]` 的结果。把所有 head 拼接后得到 `[B,S,H]`，只是把各
head 的特征排列在一起，还没有学习如何跨 head 组合信息。

输出投影 `Wo[H,H]` 有两个作用：

1. 学习不同 head 特征之间的重新组合，而不是让各 head 永远互相隔离；
2. 输出与 residual stream 相同的隐藏维度 $H$，从而可以与残差相加。

MQA/GQA 减少的是 K/V heads，通常仍保留相同数量的 Query heads，所以拼接后的宽度和
`Wo` shape 通常不变。

## 4. 极小手算例子

设：

```text
B = 1
S = 3
H = 8
Query heads = 4
head dim = H / Query heads = 2
```

### MHA：4 个 KV heads

```text
Q: [1,4,3,2]
K: [1,4,3,2]
V: [1,4,3,2]
group size = 4 / 4 = 1
```

K/V 缓存元素数是：

```text
2(K/V) × 1 × 3 × 4 × 2 = 48
```

### GQA：2 个 KV heads

```text
Q: [1,4,3,2]
K: [1,2,3,2]
V: [1,2,3,2]
group size = 4 / 2 = 2
```

缓存元素数是 24，为 MHA 的二分之一。

### MQA：1 个 KV head

```text
Q: [1,4,3,2]
K: [1,1,3,2]
V: [1,1,3,2]
group size = 4 / 1 = 4
```

缓存元素数是 12，为 MHA 的四分之一。三种结构的 Q shape 始终不变。

## 5. 正式实验配置

本日实验固定以下条件，只改变 KV heads：

```text
hidden size H               64
Query heads Nq              8
head dimension Dh           8
prefill batch B             2
prefill sequence S          5
decode query length         1
cache 容量核算层数 L        2
cache 容量核算长度          16
dtype                       FP32，4 bytes
```

三个变量配置：

```text
MHA: KV heads = 8，group size = 1
GQA: KV heads = 2，group size = 4
MQA: KV heads = 1，group size = 8
```

这里的 GQA 配置只是一个清楚的教学示例。真实模型可能采用其他 Query/KV head 比例。

## 6. 完整数据流与 Shape/Dtype

输入 hidden state 为 `X[B,S,H] = [2,5,64]`，dtype 为 FP32。

### 6.1 Query 投影始终不变

三种结构都保留 8 个 Query heads：

```text
X                     [2,5,64]
X @ Wq                [2,5,64]
split Query heads     [2,8,5,8]
```

`Wq` 的输出宽度仍是 `8 × 8 = 64`。

### 6.2 K/V 投影随 KV heads 缩小

```text
MHA K/V projection output: [2,5,64] → split [2,8,5,8]
GQA K/V projection output: [2,5,16] → split [2,2,5,8]
MQA K/V projection output: [2,5, 8] → split [2,1,5,8]
```

缓存保持这个紧凑 shape，不需要物理复制成 8 个 KV heads。

### 6.3 Query head 如何找到对应 KV head

对于 Query head 编号 $i$，它读取的 KV head 编号是：

$$
\text{kv head index}=\left\lfloor\frac{i}{\text{group size}}\right\rfloor
$$

GQA 的 group size 为 4，因此 Query heads 0–3 读取 KV head 0，Query heads 4–7 读取
KV head 1。

实验为了用普通矩阵乘法清楚展示映射，临时把紧凑 K/V 按该索引展开到 Query head 数：

```text
compact K             [2,2,5,8]
temporary expanded K  [2,8,5,8]
Q @ Kᵀ scores         [2,8,5,5]
```

这个临时展开是教学实现的计算方式，不代表生产 GQA kernel 必须物化复制张量。缓存仍是
紧凑的 `[2,2,5,8]`。

### 6.4 Decode 后缓存增长

输入一个新 token 后，Query length 是 1，缓存长度从 5 变为 6：

```text
MHA Q: [2,8,1,8]  compact K/V: [2,8,6,8]
GQA Q: [2,8,1,8]  compact K/V: [2,2,6,8]
MQA Q: [2,8,1,8]  compact K/V: [2,1,6,8]
```

三种结构的 Attention score 都有 8 个 Query heads，逻辑 shape 是 `[2,8,1,6]`。

## 7. 参数、内存与计算成本

### 7.1 KV Cache 公式

KV Cache 字节数为：

$$
2 \times L \times B \times S \times N_{kv} \times D_h
\times \text{dtype bytes}
$$

这里必须使用 KV heads 数 $N_{kv}$，不能错误地一律使用 Query heads 数 $N_q$。

本实验按 `L=2, B=2, S=16, Dh=8, FP32=4 bytes` 核算：

```text
MHA: 2 × 2 × 2 × 16 × 8 × 8 × 4 = 32,768 bytes
GQA: 2 × 2 × 2 × 16 × 2 × 8 × 4 =  8,192 bytes
MQA: 2 × 2 × 2 × 16 × 1 × 8 × 4 =  4,096 bytes
```

容量比例为 `1 : 1/4 : 1/8`。

### 7.2 Q/K/V/O 投影参数

不含 bias 时：

```text
Wq: H × (Nq × Dh)
Wk: H × (Nkv × Dh)
Wv: H × (Nkv × Dh)
Wo: H × H
```

固定 `H=64, Nq=8, Dh=8` 后，本实验总投影参数为：

```text
MHA: 16,384
GQA: 10,240
MQA:  9,216
```

Q 和 O 参数没有减少；变化来自 K/V projection。

### 7.3 哪些成本下降，哪些不同比例下降

- KV Cache 容量和 decode 时的 K/V 读取量按 KV heads 数下降；
- K/V projection 参数和计算量下降；
- Query projection 不变，因为 Query heads 不变；
- Attention 仍要为每个 Query head 计算 score 和聚合，因此这部分不会简单按 KV heads
  比例全部消失；
- `Wo` 通常不变，因为合并后的 Query-head 输出宽度仍是 $H$。

因此“GQA 的 KV heads 减少四倍”不等于“整个 Attention FLOPs、参数和时间都减少四倍”。

## 8. 最小代码验证

实验文件：

```text
exercises/day10/mha_mqa_gqa_cache.py
```

运行：

```bash
uv run python -m exercises.day10.mha_mqa_gqa_cache
```

代码断言：

- 三种结构的 Query shape 完全相同；
- compact K/V shape 只由 KV heads 改变；
- 每个 Query head 精确读取所属组的 KV head；
- prefill 的未来 Attention weights 为 0；
- decode 追加后旧缓存逐元素不变；
- GQA/MQA 容量分别是当前 MHA 的四分之一和八分之一。

### 8.1 当前机器实际输出

2026-08-30 在 NVIDIA GeForce RTX 2060、PyTorch 2.13.0+cu130 上运行：

```text
                        MHA             GQA             MQA
prefill Q shape         [2,8,5,8]       [2,8,5,8]       [2,8,5,8]
compact K/V shape       [2,8,5,8]       [2,2,5,8]       [2,1,5,8]
Queries per KV head     1               4               8
projection params       16,384          10,240           9,216
one-layer cache at S=6  6,144 bytes     1,536 bytes        768 bytes
two-layer cache at S=16 32,768 bytes    8,192 bytes      4,096 bytes
relative capacity       1.000x          0.250x           0.125x
```

所有 Query shape、映射、causal mask、旧缓存保留和容量比例断言均通过。

本实验没有报告三种结构的速度。教学实现为计算方便物化了 expanded K/V，不能代表生产
GQA/MQA kernel 的带宽和显存行为，用它做性能排名会混淆紧凑缓存与临时张量成本。

## 9. 常见误解与边界

### 9.1 “MQA 只有一个 Attention head”

不对。MQA 可以保留多个 Query heads，只是它们共享一组 K/V。它不是单头 Attention。

### 9.2 “GQA 会把 compact K/V 永久复制到每个 Query head”

逻辑上每个 Query head 都读取对应 K/V，但高效 kernel 可以直接按分组关系读取紧凑 K/V。
本实验的 expanded Tensor 是清晰实现矩阵运算的方法，不是缓存物理 layout 的规定。

### 9.3 “减少 KV heads 会同比减少全部 Attention 成本”

它直接降低 K/V projection、缓存容量和缓存读取量，但 Query projection、每个 Query head
的 score/Value 聚合以及输出投影仍然存在。

### 9.4 “可以把训练好的 MHA 任意改成 MQA，输出不变”

结构和参数 shape 已经变化，不能直接删掉 K/V heads 并期待行为不变。模型需要从该结构
训练，或使用有明确转换、微调和质量验证的方法。今天没有验证质量差异。

### 9.5 当前尚未验证

- 真实模型采用的 Query/KV head 配置和转换方式；
- MHA、GQA、MQA 对模型质量的影响；
- 支持 compact GQA 的生产 kernel 性能和显存；
- Tensor Parallel 下 KV heads 如何切分或复制；
- KV Cache quantization；
- 不同模型中的 MLA 等其他 KV 压缩机制。

## 10. 手算练习

### 练习 1

模型有 24 层、32 个 Query heads、8 个 KV heads、head dim 为 128。一个 KV head 服务
多少个 Query heads？这是 MHA、MQA 还是 GQA？

答案：`32 / 8 = 4`，每个 KV head 服务 4 个 Query heads，这是 GQA。

### 练习 2

沿用练习 1，batch 为 2、缓存长度为 1,024、dtype 为 FP16。KV Cache 占多少 bytes 和
MiB？

答案：

```text
2 × 24 × 2 × 1024 × 8 × 128 × 2 = 201,326,592 bytes
201,326,592 / 1024² = 192 MiB
```

### 练习 3

若把练习 2 的结构改成 MHA，保持 32 个 Query/KV heads，缓存是多少？

答案：KV heads 从 8 增加到 32，是原来的 4 倍，即 `768 MiB`。Query heads 没变，变化
来自 KV heads。

## 11. 面试口述

### 问题 1：MHA、MQA 和 GQA 的区别是什么？

30 秒回答：MHA 通常让每个 Query head 有独立 K/V；MQA 保留多个 Query heads，但共享
唯一 K/V head；GQA 把 Query heads 分组，每组共享一组 K/V。它们主要在 K/V 表达能力、
缓存容量和读取带宽之间权衡。

### 问题 2：为什么 GQA 降低 K/V 成本，却不同比例减少 Query heads？

30 秒回答：历史缓存只保存 K/V，所以减少 KV heads 能直接压缩持久状态。多个 Query
heads 仍负责从不同角度查询这些共享 K/V，保留 Attention 的多组输出。若 Query heads
也同比减少，就进一步改变了表示宽度和模型结构，不再只是压缩缓存。

### 问题 3：如何手算 KV Cache？

2 分钟回答：先确认层数、请求或 batch 数、实际缓存长度、KV heads、head dim 和 dtype
bytes；然后计算 `2 × L × B × S × Nkv × Dh × bytes`，2 表示 K 和 V。必须使用 KV heads
而不是 Query heads。结果是逻辑 Tensor 数据量，实际服务还会有 block、padding、碎片、
allocator 和并行切分等开销。

## 12. 当日验收

- [ ] 能画出 8 Query heads 在 MHA/GQA/MQA 中的 KV 映射。
- [ ] 能解释为什么 MQA 不是单头 Attention。
- [ ] 能写出三种结构的 Q/K/V shape。
- [ ] 能解释 compact cache 和临时逻辑展开的区别。
- [ ] 能手算练习中的 192 MiB 与 768 MiB。
- [ ] 能说明 K/V heads 减少后哪些成本下降、哪些保持不变。
- [ ] 能运行实验并解释 `1、1/4、1/8` 的容量比例。

下一步 Day 11：学习 FlashAttention、activation checkpointing 和 IO-aware 思维，重点
区分 Attention 数学结果是否改变、哪些中间张量不再写回显存，以及用重计算换激活显存。
