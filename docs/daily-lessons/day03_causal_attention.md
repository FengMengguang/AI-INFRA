# Day 3：从 Q/K/V 到可运行的多头因果注意力

## 1. 今日核心问题

今天只解决一个完整问题：

> Decoder-only Transformer 中的一个 Attention 子层，如何把 `[B,S,H]` 变成 Q/K/V，拆成多个 head，在 softmax 前加入 causal mask，读取 Value，再通过 RMSNorm 和 residual 保持稳定的信息流？

完整主线是：

```text
hidden X[B,S,H]
→ RMSNorm
→ Q/K/V 线性投影
→ 拆分 heads
→ QKᵀ / √d
→ 加入 causal mask
→ softmax
→ attention weights @ V
→ 合并 heads
→ output projection
→ residual add
→ output[B,S,H]
```

今天完成后，应能回答：

1. Q、K、V 分别表示什么，又分别来自哪里？
2. causal mask 精确加在哪一步，为什么必须在 softmax 之前？
3. 多个 head 是如何拆分和合并的？
4. RMSNorm 改变了什么，不改变什么？
5. residual 为什么要求 Attention 输出回到 `[B,S,H]`？
6. 如何用“改变未来 token”的实验直接验证因果性？

今天不实现 SwiGLU、完整 Decoder Block、LM Head 和 loss。Day 4 会将今天的 Attention 子层与 FFN 组合成完整 Block。

## 2. 昨日回忆

不查 Day 2，先回答：

1. token ID、Embedding 权重和 hidden activation 有什么区别？
2. `input_ids[B,S]` 经过 Embedding 后为什么变成 `[B,S,H]`？
3. padding mask 和 causal mask 分别阻止什么？
4. Decoder-only 训练为什么能同时产生所有位置的 logits？

如果无法把文本一直追踪到 `[B,S,H]`，先复习 Day 2，再继续今天的矩阵运算。

## 3. 前置知识与配置

### 3.1 正式教学配置

沿用前两天的模型主配置：

```text
B = 2       batch size
S = 8       sequence length
H = 384     hidden size
Nq = 6      Query heads
Nkv = 2     Key/Value heads
D = 64      head dimension
```

因为：

```text
H = Nq × D = 6 × 64 = 384
```

`Nq=6, Nkv=2` 实际对应 GQA（Grouped-Query Attention）。今天的最小代码先使用 `Nq=Nkv=2`，也就是标准 MHA，避免同时引入 KV head 复制。Day 10 再正式对比 MHA、MQA 和 GQA。

### 3.2 术语

- **Self-Attention**：Q、K、V 来自同一条 hidden sequence。
- **Head**：在较小特征子空间中独立执行一次 Attention 的分支。
- **Attention Scores**：`QKᵀ/√D` 得到的未归一化匹配分数。
- **Attention Weights**：Scores 加 mask 后经过 softmax 得到的读取比例。
- **Causal Mask**：使位置 `i` 无法读取 `j>i` 位置的上三角遮罩。
- **RMSNorm**：使用向量的均方根缩放数值尺度，再乘可学习缩放参数。
- **Residual connection**：将子层输入直接加回子层输出。
- **Pre-Norm**：先归一化，再进入 Attention 或 FFN。

## 4. Attention 到底在做什么

### 4.1 三种角色

对当前 token 而言：

```text
Q（Query）：我现在想查找什么？
K（Key）：每个历史位置可以用什么特征被匹配？
V（Value）：匹配到该位置后，实际读取什么内容？
```

Q 和 K 决定读取路径，V 承载被加权汇总的内容。Q/K 不是“无用的中间量”；如果 Key 变化，即使 Value 不变，读取比例也会变。

### 4.2 Q/K/V 是可学习投影

输入：

```text
X[B,S,H]
```

每个 token 的 hidden vector 会经过三组不同权重：

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

在标准 MHA 中：

```text
Wq[H,H]
Wk[H,H]
Wv[H,H]

Q,K,V: [B,S,H]
```

三组投影权重在训练中学习。它们不是 token 自带的固定属性。

## 5. 多头是如何拆分的

从：

```text
Q[B,S,H]
```

拆成：

```text
Q[B,S,N,D]
```

为了让 head 维更方便进行矩阵乘法，常转置为：

```text
Q[B,N,S,D]
K[B,N,S,D]
V[B,N,S,D]
```

单个 head 内：

```text
Q_head[S,D] @ K_headᵀ[D,S] → Scores[S,S]
Weights[S,S] @ V_head[S,D] → O_head[S,D]
```

所有 head 输出合并：

```text
[B,N,S,D]
→ transpose
[B,S,N,D]
→ reshape
[B,S,H]
```

然后经过输出投影：

```text
O = Concat(heads) @ Wo
Wo[H,H]
O[B,S,H]
```

`Wo` 允许不同 head 的信息在回到 residual stream 前再次混合。

## 6. 一个可手算的 Causal Attention

暂时只看一个 head，有 3 个 token，`D=2`：

```text
Q = [[1,0],
     [0,1],
     [1,1]]

K = [[1,0],
     [0,1],
     [1,1]]
```

先计算：

```text
QKᵀ = [[1,0,1],
       [0,1,1],
       [1,1,2]]
```

再缩放：

```text
Scores = QKᵀ / √2
       ≈ [[0.707,0,0.707],
          [0,0.707,0.707],
          [0.707,0.707,1.414]]
```

### 6.1 causal mask 在哪一步

Mask 加在缩放后的 Scores 上，且必须位于 softmax 之前：

```text
QKᵀ
→ 除以 √D
→ 加 causal mask       ← 精确位置
→ softmax
→ 乘 V
```

Mask 矩阵：

```text
M = [[0, -inf, -inf],
     [0,    0, -inf],
     [0,    0,    0]]
```

相加后：

```text
Masked Scores
≈ [[0.707, -inf,  -inf],
   [0,      0.707, -inf],
   [0.707,  0.707, 1.414]]
```

因为 `exp(-inf)=0`，softmax 后上三角权重为 0：

```text
Weights ≈ [[1.00, 0.00, 0.00],
           [0.33, 0.67, 0.00],
           [0.25, 0.25, 0.50]]
```

位置 0 无法读位置 1、2 的 Value；位置 1 无法读位置 2 的 Value。

### 6.2 为什么不能 softmax 后才 mask

如果未遮罩分数是：

```text
[1,2,3]
```

直接 softmax 得到：

```text
[0.09,0.24,0.67]
```

即使后来把第三项置 0，前两项仍然只和为 `0.33`，因为未来位置已经参与 softmax 分母。

正确方法是：

```text
[1,2,3] + [0,0,-inf]
→ [1,2,-inf]
→ softmax
→ [0.27,0.73,0]
```

实际代码常用 dtype 可表示的极小有限值而不是真正的 `-inf`，但目标相同：使被遮罩位置的 softmax 权重为 0。

## 7. Causal Mask 建立的是信息边界

对序列：

```text
[BOS, 我, 喜欢, 学习]
```

可见性为：

```text
BOS  → BOS
我   → BOS, 我
喜欢 → BOS, 我, 喜欢
学习 → BOS, 我, 喜欢, 学习
```

训练时，完整目标序列同时存在，Mask 防止前面位置偷看右侧标签。

Prefill 时，整个 Prompt 同时存在，Mask 使每个位置的隐藏状态和各层 K/V 仍保持训练时的因果结构。

单 token Decode 时，KV Cache 中只有过去和当前位置，未来 K/V 尚不存在，因此通常不需要显式构造上三角 Mask。

## 8. RMSNorm 解决什么

### 8.1 数值尺度问题

残差流经过多层变换时，各 token 向量的数值尺度可能不稳定。RMSNorm 对每个 token 的 H 维向量独立计算：

```text
rms(x) = sqrt(mean(x²) + eps)
```

然后：

```text
RMSNorm(x) = x / rms(x) * g
```

`g[H]` 是可学习的逐维缩放参数。

例如：

```text
x = [3,4]
mean(x²) = (9+16)/2 = 12.5
rms(x) ≈ 3.536

x/rms(x) ≈ [0.849,1.131]
```

RMSNorm：

- 不改变 `[B,S,H]` shape；
- 不混合不同 token；
- 通常不减去均值，这是它与 LayerNorm 的一个重要差异；
- 它调整的是数值尺度，不是直接让所有 token 变得相同。

### 8.2 Pre-Norm 数据流

今天使用现代 Decoder-only 常见的 Pre-Norm 抽象：

```text
X
├─────────────────────┐
↓                     │
RMSNorm              residual path
↓                     │
Causal MHA               │
↓                     │
Attention output          │
└───────── add ◄──────────┘
↓
Y
```

公式：

```text
Y = X + Attention(RMSNorm(X))
```

## 9. Residual 为什么重要

Attention 子层不是完全替换 X，而是学习一个需要加到 X 上的更新量：

```text
output = X + attention_update
```

这条直接路径使原信息可以跨越子层，也为训练时的梯度传播提供更直接的路径。

要做逐元素相加，两边 shape 必须一致：

```text
X:                [B,S,H]
attention_update: [B,S,H]
output:           [B,S,H]
```

这也是多个 head 最后必须合并并回到 H 维的直接原因之一。

## 10. 完整 Shape 数据流

使用正式教学配置：

```text
X                           [B,S,H]       = [2,8,384]
RMSNorm(X)                  [B,S,H]       = [2,8,384]
Q                           [B,S,Nq,D]    = [2,8,6,64]
K,V                         [B,S,Nkv,D]   = [2,8,2,64]
Q transpose                 [B,Nq,S,D]    = [2,6,8,64]
K,V transpose               [B,Nkv,S,D]   = [2,2,8,64]
```

因为正式配置是 GQA，每个 KV head 服务：

```text
Nq / Nkv = 6 / 2 = 3 个 Query heads
```

逻辑对齐后：

```text
Scores                     [B,Nq,S,S] = [2,6,8,8]
Causal Mask broadcast      [1,1,S,S]  = [1,1,8,8]
Attention Weights          [B,Nq,S,S] = [2,6,8,8]
Head outputs               [B,Nq,S,D] = [2,6,8,64]
Merged                     [B,S,H]    = [2,8,384]
Output projection          [B,S,H]    = [2,8,384]
Residual output            [B,S,H]    = [2,8,384]
```

Mask 在 batch 和 head 维上可广播，逻辑上不需要为每个 batch、每个 head 复制一份实体矩阵。

## 11. 参数、激活与计算成本

### 11.1 参数量

标准 MHA 忽略 bias：

```text
Wq: H×H
Wk: H×H
Wv: H×H
Wo: H×H
```

总参数：

```text
4H²
```

`H=384` 时：

```text
4 × 384² = 589,824
```

但 GQA 的 K/V 输出维度是 `Nkv×D`，因此：

```text
Wq: H × (Nq×D)
Wk: H × (Nkv×D)
Wv: H × (Nkv×D)
Wo: (Nq×D) × H
```

对 `H=384,Nq=6,Nkv=2,D=64`：

```text
Wq = 384×384
Wk = 384×128
Wv = 384×128
Wo = 384×384
总计 = 393,216 参数
```

### 11.2 主要逻辑激活

```text
Q                  O(B·S·H)
K,V                O(B·S·Nkv·D)
Scores/Weights     O(B·Nq·S²)
Output             O(B·S·H)
```

序列长度加倍时，hidden、Q/K/V 线性增长，而朴素全注意力的 Scores 二次增长。

但 Scores 和 Weights 是数学上的逻辑形状，FlashAttention 等 IO-aware 实现可以分块计算，不把完整 `[S,S]` 中间结果长期写回 HBM。

## 12. 最小代码验证

当前环境没有安装 PyTorch，今天的脚本继续只使用 Python 标准库：

```bash
python3 exercises/day03/causal_attention_block.py
```

它实际实现：

- 二维矩阵乘法与转置；
- RMSNorm；
- Q/K/V 线性投影；
- 2-head 拆分和合并；
- scaled dot-product attention；
- softmax 前的 causal mask；
- output projection 和 residual add。

关键验证不只是“程序能跑”，而是：

1. 每个 head 的 Scores 为 `[S,S]`；
2. 所有未来位置的 Attention Weight 精确为 0；
3. 修改最后一个 token 后，前三个位置的 causal output 不变；
4. 关闭 mask 后，同样的未来 token 修改会影响前面位置；
5. 残差子层输出仍为 `[S,H]`。

脚本使用单 batch 和单位投影以便观察数值。它是真实的 Attention 数学路径，但不是可训练框架，也不用于性能测试。

## 13. 常见误解与边界

- Q、K、V 是由 hidden states 通过不同可学习权重投影得到的，不是 token 的固定字段。
- V 承载最终汇总内容，但 Q/K 决定读取哪里以及读取多少。
- causal mask 不会删除未来 token，而是使它们对当前 Query 的权重变成 0。
- Mask 必须在 softmax 前作用；否则未来分数已经污染归一化分母。
- 数学上的完整 `[S,S]` 不代表高效 kernel 必须物化整张矩阵。
- 多个 head 不保证自动学成人类可命名的固定分工。
- RMSNorm 和 softmax 不是同一类归一化：前者缩放 hidden vector，后者把一行 Scores 变成和为 1 的权重。
- residual 是逐元素相加，不是在 feature 维拼接。
- Pre-Norm 和 Post-Norm 的顺序不同；今天只实现 Pre-Norm。
- 训练阶段需要保留反向传播所需的激活；本脚本只做 forward，不能代表训练显存。
- 今天的单位投影是为了使证据可解释；真正模型的 Wq/Wk/Wv/Wo 会经训练学习。

## 14. 手算练习

### 练习 1：单头 Scores 与 Mask

给定：

```text
Q = [[1,0],[0,1],[1,1]]
K = [[1,0],[0,1],[1,1]]
D = 2
```

1. 手算 `QKᵀ`。
2. 除以 `√2` 后的 Scores 是什么？
3. 写出 3×3 causal mask。
4. 写出每行 softmax 后哪些位置必须为 0。

### 练习 2：多头 Shape

给定：

```text
B=2, S=8, H=384, N=6, D=64
```

写出：

1. X、Wq 和 Q 的 shape。
2. Q 拆 head 前后的 shape。
3. 单 head 和所有 head 的 Scores shape。
4. head outputs 合并后的 shape。
5. 为什么 residual 加法可以成立？

### 练习 3：RMSNorm

对：

```text
x=[3,4], g=[1,1], eps 忽略
```

1. 计算 `mean(x²)` 和 `rms(x)`。
2. 计算 RMSNorm 输出。
3. 输出的 shape 是什么？
4. 它与 LayerNorm 在“是否减均值”上有什么区别？

### 练习 4：因果性压力测试

原序列：

```text
[A,B,C,D]
```

只把 D 改成 X：

```text
[A,B,C,X]
```

回答：

1. 在 causal self-attention 中，哪些位置的输出必须保持不变？
2. 哪个位置允许变化？
3. 取消 mask 后，前面位置为什么可能变化？
4. 为什么这比“看到一个上三角矩阵”更能证明因果性？

### 练习 5：成本

1. `Scores[2,6,8,8]` 有多少元素？
2. 将 S 从 8 增加到 16，Scores 元素增长多少倍？
3. `H=384` 的标准 MHA 投影参数量是多少？
4. 为什么 GQA 降低 K/V 成本，却不同比例减少 Query heads？

## 15. 面试口述

### 15.1 30 秒目标

Attention 先将 hidden states 分别投影成 Q、K、V；QKᵀ 表示位置间匹配分数，经缩放和 causal mask 后做 softmax，再加权汇总 V。多个 head 独立计算后合并回 H 维，经输出投影并加回 residual。Mask 在 softmax 前将未来位置变成零权重。

### 15.2 两分钟目标

回答需包含：

1. Q/K/V 的信息角色和投影来源；
2. `[B,S,H] → [B,N,S,D]` 的 head 拆分；
3. Scores、Mask、Softmax、Value 汇总的精确顺序；
4. 训练、Prefill 和 Decode 时 causal 约束的表现；
5. RMSNorm、residual 和 Pre-Norm 的数据流；
6. Scores 的 `S²` 成本及逻辑 shape 与物理实现的边界。

### 15.3 三道口述题

1. causal mask 为什么必须在 softmax 之前加？
2. 多头 Attention 为什么最后还要合并并经过 `Wo`？
3. 如何用一个输入对照实验证前面位置没有读取未来？

## 16. 当日验收

请独立完成：

1. 不看讲义，画出 `X → RMSNorm → Q/K/V → Scores → Mask → Softmax → V → Wo → Residual` 的数据流。
2. 完成练习 1、2 和 4。
3. 在纸上写出 4×4 causal mask，解释每个 `-inf` 阻止了什么。
4. 手算一行 masked softmax，并解释为什么未来权重为 0。
5. 运行最小脚本，解释三项 shape 和两项 causality check。
6. 说明 RMSNorm 和 residual 各自改变什么、保留什么。
7. 用两分钟回答“一个 Decoder-only Attention 子层如何工作”。
8. 记录今天最不确定的一个问题。

只有同时满足以下条件，Day 3 才算通过：

- 能写对 Q/K/V、Scores、Weights 和输出的 shape；
- 能说明 causal mask 在 softmax 前加入的原因；
- 能用改变未来 token 的对照试验验证因果性；
- 不把 RMSNorm 和 softmax 混为一类操作；
- 能画出 Pre-Norm residual 数据流；
- 能区分数学上的 `[S,S]` 与高效 kernel 的物理实现；
- 能说出至少一个失效条件或实现边界。
