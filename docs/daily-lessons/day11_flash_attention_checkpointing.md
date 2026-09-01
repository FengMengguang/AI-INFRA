# Day 11：FlashAttention、Activation Checkpointing 与 IO-Aware 思维

## 1. 今日核心问题

前面的课程主要从 FLOPs 和 Tensor shape 理解成本，但 GPU 性能还取决于数据在不同存储
层级之间移动多少次。今天回答两个问题：

1. FlashAttention 如何在不改变 Attention 数学目标的前提下，减少巨大中间矩阵对显存的
   读写？
2. Activation Checkpointing 如何少保存 forward 激活，并在 backward 时重新计算，以
   额外计算换取训练显存？

今天的核心验收是能够区分：

```text
数学发生变化
实现方式发生变化
保存状态发生变化
计算与显存之间发生交换
```

## 2. 前置知识与术语

- **IO**：这里主要指 GPU 不同存储层级之间的数据移动，不只是磁盘或网络输入输出。
- **HBM / global memory**：GPU 的大容量显存，容量较大，但访问成本高于片上存储。
- **SRAM / shared memory / registers**：更靠近计算单元的片上存储，容量小但访问更快。
- **Tiling（分块）**：把大矩阵拆成可以放入片上存储的小块逐块计算。
- **FlashAttention**：一种 IO-aware 的精确 Attention 算法，通过分块和在线 Softmax 减少
  HBM 读写，避免把完整 score/概率矩阵长期物化到 HBM。
- **SDPA（Scaled Dot-Product Attention）**：PyTorch 的缩放点积注意力接口，可以根据
  硬件、dtype 和 shape 分派到 math、memory-efficient、Flash 或其他 backend。
- **Activation（激活）**：forward 过程中产生、backward 计算梯度时可能需要的中间 Tensor。
- **Activation Checkpointing**：只保留部分边界激活，backward 时重新执行部分 forward。
- **Saved tensors**：autograd 为 backward 保存或引用的 Tensor。

Activation Checkpointing 与训练状态 checkpoint 不是同一个东西：

```text
训练 checkpoint：保存到磁盘，用于中断恢复
activation checkpoint：留在一次训练 step 内，用重计算降低激活显存
```

它也不是推理 KV Cache。KV Cache 保存历史 K/V 以避免重复计算；activation checkpointing
反而主动少保存部分 forward 中间量，允许 backward 时重复计算。

## 3. 从直觉到机制

### 3.1 为什么只看 FLOPs 不够

两个实现可以执行等价的数学运算，但其中一个不断把大中间 Tensor 写入 HBM、随后再读回，
另一个让数据尽量停留在片上存储。即使 FLOPs 接近，第二个实现仍可能更快、更省峰值显存。

这就是 IO-aware 思维：不仅问“算了多少”，还要问：

```text
数据从哪里读入？
中间结果写到哪里？
同一数据被读写几次？
是否必须物化完整中间 Tensor？
```

### 3.2 朴素 Attention 的中间矩阵

标准 Attention 是：

$$
O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{D_h}}+M\right)V
$$

$M$ 是 causal mask 或其他 mask。朴素实现常按以下阶段执行：

```text
Q @ Kᵀ
→ 写出完整 scores[B,h,S,S]
→ 读取 scores 做 mask 和 Softmax
→ 写出 probabilities[B,h,S,S]
→ 读取 probabilities，与 V 相乘
```

序列维出现两次，因此 scores 和 probabilities 按 $S^2$ 增长。

### 3.3 FlashAttention 改变什么

FlashAttention 把 Q、K、V 分块加载到片上存储，对一个 Q block 逐块遍历 K/V block，并
维护当前 Softmax 所需的统计量和输出累积。完整的 `[S,S]` 矩阵不需要长期写回 HBM。

它改变的是：

- 运算分块和执行顺序；
- 中间 Tensor 的物化方式；
- HBM 与片上存储之间的读写量；
- backward 中保存与重计算的实现策略。

它不改变的是：

- Q、K、V 的语义；
- scaled dot-product Attention 的数学目标；
- causal mask 的因果约束；
- 最终输出的逻辑 shape。

不同 kernel 的浮点累加顺序可能不同，因此不要求逐 bit 相同，而应在合理容差内比较。

### 3.4 在线 Softmax 为什么允许分块

普通 Softmax 看起来需要一次看到一整行 score。在线 Softmax 会为已经处理的块维护：

- 当前最大值 $m$，用于数值稳定；
- 当前指数和 $l$，用于最终归一化；
- 按相同缩放规则累计的 Value 加权和。

新 score block 到来时，如果出现更大的最大值，就按照新旧最大值差重新缩放已有累计量，
再加入当前块。这样可以逐块处理，而不必保存整行 score。

### 3.5 Activation Checkpointing 改变什么

普通训练会保存每个 Block 内 backward 所需的中间激活：

```text
x0 → Block 1 内部激活 → x1
x1 → Block 2 内部激活 → x2
x2 → Block 3 内部激活 → x3
```

checkpointing 可以只保留各段边界。backward 到某段时，从边界输入重新执行该段 forward，
恢复所需中间激活，然后计算梯度。

它不改变模型参数、层结构、loss 或理论梯度，主要改变：

- forward 阶段保存哪些激活；
- backward 阶段是否重算 forward；
- 峰值激活显存；
- 一次训练 step 的执行时间。

## 4. 极小手算例子

### 4.1 Attention score 容量

设 `B=1`、`heads=2`、`S=4`、dtype 为 FP16。显式 score shape 是：

```text
[1,2,4,4]
```

元素数与字节数：

```text
1 × 2 × 4 × 4 = 32 elements
32 × 2 bytes = 64 bytes
```

若序列长度从 4 增加到 8，score 元素变为：

```text
1 × 2 × 8 × 8 = 128 elements
```

序列长度增加 2 倍，score 容量增加 4 倍。

### 4.2 分块 Attention

把长度 4 按每块 2 个 token 分成两个 Q blocks 和两个 K/V blocks：

```text
Q block 0 依次读取 K/V block 0、1
Q block 1 依次读取 K/V block 0、1
```

数学上仍覆盖相同的 `4 × 4` token 对；变化在于每次只处理 `2 × 2` 小块，并在线累计
Softmax 与输出，不必让完整概率矩阵长期存在于 HBM。

### 4.3 Activation Checkpointing

假设三个 Block 各有 10 MiB 必须保存的内部激活。普通方式逻辑上需要保存约 30 MiB。
若以每个 Block 为一个 checkpoint 段，只保存每段 2 MiB 的输入边界，则保存量约 6 MiB，
但 backward 会重新运行三个 Block 的部分 forward。

这只是机制示例。真实显存还包含参数、梯度、optimizer state、临时 workspace 和 allocator
行为，不能直接用这个数字预测 CUDA peak。

## 5. 正式实验配置

### 5.1 Attention/SDPA 对照

```text
正确性配置：       B=2, heads=4, S=32, head_dim=16, FP32
backend 探测配置： B=2, heads=8, S=128, head_dim=32, FP16
mask：             causal
对照：             显式 scores/Softmax 与 PyTorch SDPA
```

FP32 小配置用于严格比较输出和 Q/K/V 梯度。FP16 配置用于逐个强制探测当前 GPU 的 SDPA
backend；某个 backend 不可用时记录为不支持，不进行静默 fallback。

### 5.2 Checkpointing 对照

```text
输入 shape            [4,128,256]
Block 数              6
FFN intermediate      1024
参数 dtype            FP32
checkpoint API        use_reentrant=False
warmup                3 steps
measured              10 steps
```

两个路径使用同一个模型、相同参数、相同输入和相同 loss，唯一主要变量是是否对每个 Block
启用 activation checkpointing。

## 6. 完整数据流与 Shape/Dtype

### 6.1 显式 Attention

```text
Q/K/V                    [2,8,128,32]  FP16
K transpose              [2,8,32,128]  FP16
scores                    [2,8,128,128] FP16
causal probabilities      [2,8,128,128] FP16
output                    [2,8,128,32]  FP16
```

实验中的显式 score 有 262,144 个元素，FP16 逻辑容量为 0.500 MiB。probabilities 若同样
物化，还会有另一份相同 shape 的中间量。

### 6.2 SDPA 接口

```python
F.scaled_dot_product_attention(query, key, value, is_causal=True)
```

输入和输出逻辑 shape 与显式 Attention 相同，但内部是否物化完整 scores，取决于实际选择
的 backend。调用 SDPA 接口不等于已经使用 FlashAttention。

### 6.3 Checkpointing forward/backward

普通路径：

```text
input [4,128,256]
→ six FFN blocks
→ output [4,128,256]
→ loss scalar
→ backward 使用 forward 保存的内部激活
```

checkpoint 路径：

```text
input [4,128,256]
→ six checkpointed FFN blocks
→ 主要保存段边界
→ output [4,128,256]
→ loss scalar
→ backward 到每段时重新 forward，再计算梯度
```

两条路径输出、输入梯度和参数梯度的逻辑 shape 完全相同。

## 7. 参数、内存与计算成本

### 7.1 Attention 的平方中间量

显式 score 元素数为：

$$
B \times N_h \times S_q \times S_k
$$

self-attention 中通常 $S_q=S_k=S$，所以为 $B N_h S^2$。FlashAttention 不会把
Attention 的主要数学计算复杂度从平方变成线性；它主要降低 HBM IO 和完整中间矩阵的
物化成本。

因此以下说法要区分：

```text
减少显式 S×S 中间 Tensor：是
减少 HBM 读写：是其核心目标
把精确 Attention 的所有计算变成 O(S)：不是
彻底不计算 QKᵀ 元素：不是
```

### 7.2 Checkpointing 的交换关系

普通训练倾向于：

```text
更多 saved activations
更少 backward 重计算
```

checkpointing 倾向于：

```text
更少 saved activations
更多 backward 重计算
```

模型参数、参数梯度和 optimizer state 不会因为 activation checkpointing 自动消失。
Day 12 会把这些类别放入完整显存账本。

### 7.3 Saved logical bytes 与 CUDA peak 的区别

`saved_tensors_hooks` 统计的是 autograd 保存或引用 Tensor 的逻辑 payload，用于观察保存
策略变化。它不等于新增物理显存，因为某些 saved tensors 可能只是引用已经存在的参数。

CUDA peak delta 是 PyTorch allocator 在该测量区间观察到的峰值增量，也不等于
`nvidia-smi` 进程占用。两种证据要分别报告，不能混为同一个指标。

## 8. 最小代码验证

实验文件：

```text
exercises/day11/flash_attention_checkpointing.py
```

运行：

```bash
uv run python -m exercises.day11.flash_attention_checkpointing
```

实验验证：

- 显式 causal Attention 与强制 math SDPA 的输出和 Q/K/V 梯度一致；
- 每个 SDPA backend 单独强制运行，不允许静默换到其他 backend；
- checkpoint 与普通路径输出完全相同，输入和参数梯度在容差内一致；
- checkpoint 保存的逻辑 Tensor payload 更少；
- 同时记录 CUDA peak delta 和训练 step 时间。

### 8.1 当前机器实际输出

2026-08-30 在 PyTorch 2.13.0+cu130、NVIDIA GeForce RTX 2060 上得到：

```text
explicit score shape:             [2,8,128,128]
explicit score elements:          262,144
explicit score size FP16:         0.500 MiB
naive/SDPA max output difference: 0

FLASH_ATTENTION backend:          UNAVAILABLE
EFFICIENT_ATTENTION backend:      AVAILABLE
CUDNN_ATTENTION backend:          UNAVAILABLE
MATH backend:                     AVAILABLE

normal saved tensors:             31
normal saved logical bytes:       39.500 MiB
checkpoint saved tensors:         7
checkpoint saved logical bytes:   3.500 MiB
normal CUDA peak delta:           30.502 MiB
checkpoint CUDA peak delta:       18.026 MiB
normal average train step:        3.034 ms
checkpoint average train step:    3.892 ms
checkpoint/normal time ratio:     1.283x
maximum gradient difference:      0
```

当前机器不能强制运行 PyTorch 的 `FLASH_ATTENTION` backend，但可以运行
`EFFICIENT_ATTENTION`。因此本日验证了 SDPA 数学一致性和 memory-efficient backend 可用，
没有声称已经在 RTX 2060 上实测 FlashAttention kernel。

checkpoint 将当前配置的 CUDA peak delta 减少约 12.476 MiB，但训练步慢约 28.3%，符合
“以重计算换显存”的机制。该结果只适用于当前短实验，不能外推其他模型或 GPU。

2026-08-30 在生成 Day 12 时重复回归两次，正确性、saved logical bytes 和 peak
delta 保持一致，但 checkpoint/normal 时间比分别为 `1.833x` 和 `1.700x`。这证明
10 个 measured steps 的短 benchmark 存在明显波动；`1.283x` 保留为首次观测，不是稳定性能常数。

## 9. 常见误解与边界

### 9.1 “调用 SDPA 就一定用了 FlashAttention”

不对。SDPA 是统一接口，实际 backend 取决于 GPU、dtype、shape、mask、软件构建和配置。
本机强制 Flash backend 明确失败，但 efficient 和 math backend 可用。

### 9.2 “FlashAttention 是近似 Attention”

FlashAttention 的目标是精确计算标准 Attention，改变的是分块、在线 Softmax 和 IO 路径，
不是把 Attention 定义替换成近似线性 Attention。浮点执行顺序不同仍可能产生微小误差。

### 9.3 “FlashAttention 把复杂度从平方降到线性”

它避免完整 $S^2$ 中间矩阵写回 HBM，并降低 IO；标准 Attention 的 token 对计算仍然与
序列长度平方相关。不要把空间/IO 优化写成数学计算复杂度被彻底改变。

### 9.4 “Activation Checkpointing 会保存模型到磁盘”

不会。它只影响当前 forward/backward 内激活的保存与重计算。训练中断恢复仍需要 Day 6
学习的磁盘 checkpoint。

### 9.5 “Checkpointing 一定按相同比例降低总显存”

它主要影响可 checkpoint 区域的激活。参数、梯度、optimizer state、不可重算状态和临时
workspace 仍存在，因此 saved activations 降十倍不代表总进程显存也降十倍。

### 9.6 当前尚未验证

- 支持 Flash backend 的 Ampere 或更新 GPU 上的真实 kernel；
- FlashAttention 不同版本、不同 mask 和 GQA 的 backend 支持边界；
- dropout、随机算子与 checkpoint RNG 状态处理；
- 含 BatchNorm 或外部副作用模块的重计算安全性；
- 不同 checkpoint 分段粒度的时间/显存最优点；
- 大模型长序列上的 profiler 时间线与 HBM 流量。

## 10. 手算练习

### 练习 1

`B=4`、`heads=16`、`S=2,048`，显式 Attention score 使用 FP16。逻辑容量是多少？

答案：

```text
4 × 16 × 2048 × 2048 × 2 bytes
= 536,870,912 bytes
= 512 MiB
```

这还只是一份 score，不含 probabilities、Q/K/V、输出和其他临时量。

### 练习 2

序列长度从 1,024 增加到 4,096，其他配置不变，显式 score 容量增加多少倍？

答案：长度增加 4 倍，因为 score 按 $S^2$ 增长，所以容量增加 16 倍。

### 练习 3

普通路径保存 40 MiB 激活，checkpoint 路径保存 8 MiB，但模型参数、梯度和 optimizer state
共占 120 MiB。忽略其他开销，两条路径总量分别是多少？checkpoint 降低多少？

答案：普通路径为 160 MiB，checkpoint 路径为 128 MiB，总量只降低 32 MiB，也就是 20%，
而不是因为激活下降 5 倍就让总显存下降 5 倍。

## 11. 面试口述

### 问题 1：FlashAttention 改变了 Attention 数学吗？

30 秒回答：没有改变标准 scaled dot-product Attention 的数学目标。它通过 tiling、在线
Softmax 和片上累计避免完整 score/概率矩阵频繁写入 HBM，主要优化 IO 和中间激活。
由于浮点累加顺序不同，数值可能在容差内略有差异。

### 问题 2：SDPA 与 FlashAttention 是什么关系？

30 秒回答：SDPA 是 PyTorch 的统一 Attention 接口，FlashAttention 是它可能选择的一个
backend。实际选择取决于 GPU、dtype、shape 和 mask。调用 SDPA 不能单独证明 Flash
kernel 已运行，应强制 backend 或通过 profiler 核验。

### 问题 3：Activation Checkpointing 为什么省显存又变慢？

2 分钟回答：普通 forward 会保存许多 backward 所需激活；checkpointing 只保留分段边界，
backward 到某段时重跑 forward 恢复中间量，因此减少激活驻留，但增加计算和 kernel 调度。
它不减少参数、梯度或 optimizer state，也不同于磁盘训练 checkpoint。实际收益取决于激活
占总显存的比例、分段粒度、模型结构和硬件。

## 12. 当日验收

- [ ] 能解释 FLOPs 与 HBM IO 为什么是不同成本。
- [ ] 能画出朴素 Attention 对 `[S,S]` 中间矩阵的读写。
- [ ] 能说明 FlashAttention 改变什么、保留什么。
- [ ] 能区分 SDPA 接口与实际 Flash backend。
- [ ] 能手算长度 2,048 时的 512 MiB score。
- [ ] 能画出 checkpoint forward 保存边界、backward 重计算的流程。
- [ ] 能解释 saved logical bytes 与 CUDA peak delta 的区别。
- [ ] 能解释当前实验为何省显存但变慢，以及短 benchmark 时间比为何会波动。

下一步 Day 12：建立参数、梯度、optimizer state、activation、临时 workspace 和 allocator
统计口径的完整显存账本，并把理论值与 PyTorch 实测逐项对照。
