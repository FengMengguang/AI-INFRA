# Day 1：从数字、向量和矩阵理解 Transformer 的 shape 与内存

> 今日核心问题：一批 token 进入 Transformer 后，为什么会出现 `[B,S,H]`、多个 Attention heads 和 `[B,N,S,S]` 等张量？这些数字分别表示什么，又会占用多少内存？

## 1. 今日完成标准

今天不要求背诵公式。完成标准是能够用自己的话解释：

1. 标量、向量、矩阵和张量分别是什么。
2. `X[B,S,H]` 中每一维对应什么现实对象。
3. 矩阵乘法为什么能把每个 token 从一种维度变成另一种维度。
4. 一个 Attention head 是什么，为什么一个模型会有多个 heads。
5. `reshape` 和 `transpose` 做了什么、没有做什么。
6. Query、Key、Value 和 Attention Scores 的 shape 如何一步步产生。
7. 如何根据 shape 和 dtype 计算理论内存。

## 2. 前置知识与术语

### 2.1 从一个数字到张量

#### 标量

一个数字就是标量，例如：

```text
3.14
```

它的 shape 可以写成 `[]`，因为没有需要继续索引的维度。

#### 向量

一排数字是向量，例如：

```text
[0.2, -0.7, 1.1, 0.5]
```

它有 4 个元素，shape 是 `[4]`。

在语言模型中，一个 token 不会只用一个数字表示，而是用一排数字表示。比如一个极小模型可能用 4 个数字表示 token“苹果”，这就是一个 4 维 token 向量。

#### 矩阵

多排等长向量组成矩阵：

```text
[
  [0.2, -0.7, 1.1, 0.5],
  [0.8,  0.1, 0.3, 0.9],
  [0.4, -0.2, 0.6, 0.7]
]
```

这里有 3 行，每行 4 个数，所以 shape 是 `[3,4]`。

如果三行分别对应三个 token，那么第一维表示 token 数，第二维表示每个 token 的特征数。

#### 张量

张量是对标量、向量、矩阵和更高维数组的统一称呼。两条句子组成一个 batch 后，可以得到三维张量：

```text
[句子数量, 每句 token 数, 每个 token 的特征数]
```

这就是语言模型中常见的 `[B,S,H]`。

### 2.2 `X[B,S,H]` 到底是什么意思

`X` 是当前层接收到的 token 表示。三个字母含义是：

- `B`：Batch size，一次同时处理多少条序列。
- `S`：Sequence length，每条序列有多少个 token 位置。
- `H`：Hidden size，每个 token 用多少个数字表示。

假设：

```text
B = 2
S = 3
H = 4
```

那么 `X[2,3,4]` 表示：

```text
2 条句子
× 每条句子 3 个 token
× 每个 token 4 个数字
```

总元素数是：

```text
2 × 3 × 4 = 24
```

取出 `X[0,1]`，得到第 1 条句子的第 2 个 token，它是一个 shape 为 `[4]` 的向量。

### 2.3 如何阅读矩阵乘法 `@`

`@` 表示矩阵乘法。先记住 shape 规则：

```text
[m,k] @ [k,n] → [m,n]
```

中间的 `k` 必须相等，它在结果 shape 中消失；外侧的 `m` 和 `n` 被保留。

例如：

```text
[3,4] @ [4,2] → [3,2]
```

自然语言含义是：

- 输入有 3 个对象；
- 每个对象原来有 4 个特征；
- 权重矩阵把每个对象变成 2 个新特征；
- 对象数量 3 不变，特征维度从 4 变成 2。

对于三维输入：

```text
X[B,S,H] @ W[H,O] → Y[B,S,O]
```

权重 `W` 被应用到每条句子的每个 token，只变换最后一维。Batch 和 token 位置不会因此混在一起。

### 2.4 `reshape` 与 `transpose`

#### reshape

`reshape` 改变我们如何解释同一组元素，不负责学习，也不产生新的数值。

例如：

```text
[a,b,c,d]
```

可以 reshape 成：

```text
[
  [a,b],
  [c,d]
]
```

元素仍是 4 个，只是 shape 从 `[4]` 变成 `[2,2]`。

#### transpose

`transpose` 交换维度顺序。例如 `[S,N,D]` 变成 `[N,S,D]`。它改变索引方式；底层实现可能只改变 stride，也可能因后续算子要求而产生连续拷贝。

## 3. Attention 要解决什么问题

Embedding 后，每个 token 已经有自己的向量，但还没有充分结合上下文。

例如：

```text
小明 把 苹果 放在 桌上，他 随后 吃了 它
```

模型处理“它”时，需要根据上下文判断“它”更可能指“苹果”。Attention 让每个 token 根据其他允许访问的 token 更新自己的表示。

可以把它理解为三个角色：

- Query：当前位置“正在寻找什么”。
- Key：每个位置“可以用什么特征被匹配”。
- Value：匹配后真正被读取和汇总的信息。

Query、Key、Value 都是数值向量，不是人类语言中的问题、关键词和答案。它们由模型通过不同的可训练权重从同一个输入 `X` 计算得到。

## 4. 什么是一个 Attention head

一个 head 是一套独立的 Query/Key/Value 表示子空间和 Attention 计算。

如果 hidden size 是 4，我们可以把 4 个特征拆成 2 个 heads，每个 head 2 个特征：

```text
H = 4
Nq = 2
D = 2
H = Nq × D = 2 × 2
```

其中：

- `Nq`：Query head 的数量。
- `D`：每个 head 的维度，称为 head dimension。

两个 heads 不是两个模型，也不是两个 GPU 核心。它们是同一层内部的两个并行表示空间。

直觉上，不同 heads 可以学习不同关系，例如局部搭配、远距离依赖或指代关系。但这只是帮助理解的可能性，不能默认某个训练后的 head 一定具有清晰、固定、可命名的职责。

## 5. 极小例子：3 个 token、2 个 heads

先不使用正式模型的 384 维配置。使用可手算的极小配置：

```text
B = 1    一条句子
S = 3    三个 token
H = 4    每个 token 四维
Nq = 2   两个 Query heads
Nkv = 2  两个 KV heads，暂时使用普通 MHA
D = 2    每个 head 两维
```

输入 `X` 的 shape 是：

```text
X[B,S,H] = X[1,3,4]
```

可以想象它包含：

```text
token 0 → [x00,x01,x02,x03]
token 1 → [x10,x11,x12,x13]
token 2 → [x20,x21,x22,x23]
```

### 5.1 从 X 产生 Query

Query 权重需要把每个 token 的 4 维输入变成两个 heads、每个 head 2 维的 Query：

```text
输出维度 = Nq × D = 2 × 2 = 4
Wq shape = [H,Nq×D] = [4,4]
```

矩阵乘法：

```text
X[1,3,4] @ Wq[4,4]
→ Q_flat[1,3,4]
```

逐个 token 看，就是：

```text
一个 token [4] @ Wq[4,4] → Query [4]
```

`Q_flat` 最后一维的 4 个数实际上包含两个 heads：

```text
[q0,q1,q2,q3]
→ [[q0,q1],[q2,q3]]
```

因此 reshape：

```text
Q_flat[B,S,Nq×D] = [1,3,4]
→ Q[B,S,Nq,D]    = [1,3,2,2]
```

然后交换 token 维和 head 维，让同一个 head 的全部 token 排在一起：

```text
[B,S,Nq,D]
→ transpose
→ [B,Nq,S,D]

[1,3,2,2] → [1,2,3,2]
```

现在 `[1,2,3,2]` 可以读成：

```text
1 条句子
× 2 个 Query heads
× 每个 head 有 3 个 token 位置
× 每个位置的 Query 是 2 维
```

这就是下面公式的完整含义：

```text
X[B,S,H] @ Wq[H,Nq×D]
→ Q_flat[B,S,Nq×D]
→ reshape [B,S,Nq,D]
→ transpose [B,Nq,S,D]
```

### 5.2 Key 和 Value

普通 MHA 中 `Nkv = Nq = 2`：

```text
Wk[H,Nkv×D] = [4,4]
Wv[H,Nkv×D] = [4,4]

K = [B,Nkv,S,D] = [1,2,3,2]
V = [B,Nkv,S,D] = [1,2,3,2]
```

`Wq`、`Wk`、`Wv` 是三份不同参数，因此 Q、K、V 即使来自同一个 X，数值也不同。

### 5.3 一个 head 如何产生 Scores

只看第一个 head：

```text
Q_head[S,D] = [3,2]
K_head[S,D] = [3,2]
```

想让每个 Query token 与每个 Key token 都做一次点积，需要转置 K：

```text
Q_head[3,2] @ K_headᵀ[2,3]
→ Scores_head[3,3]
```

`Scores_head[i,j]` 表示第 `i` 个 Query token 与第 `j` 个 Key token 的匹配分数。

一个 `3×3` Scores 可以按行理解：

```text
              被查看的 Key 位置
              token0 token1 token2
Query token0    s00    s01    s02
Query token1    s10    s11    s12
Query token2    s20    s21    s22
```

每一行回答：“当前位置应该以多大程度关注各个位置？”

加入 batch 和两个 heads：

```text
Scores shape = [B,Nq,S,S] = [1,2,3,3]
```

最后两个 `S` 分别代表 Query 位置和 Key 位置。这就是标准 Attention 中随序列长度二次增长的 `S×S`。

### 5.4 Causal mask、softmax 和 Value

Decoder-only 语言模型不能在预测当前位置时偷看未来 token，因此先加入 causal mask。

三 token 情况中，允许关系是：

```text
token0 只能看 token0
token1 可以看 token0、token1
token2 可以看 token0、token1、token2
```

然后对每一行做 softmax，使允许位置的权重变成非负数并且总和为 1。

假设 token2 的权重是：

```text
[0.6,0.3,0.1]
```

它的新信息就是：

```text
0.6 × Value(token0)
+ 0.3 × Value(token1)
+ 0.1 × Value(token2)
```

Shape 计算：

```text
Probabilities_head[S,S] @ V_head[S,D]
→ Context_head[S,D]

[3,3] @ [3,2] → [3,2]
```

### 5.5 合并 heads

两个 heads 分别产生 `[S,D] = [3,2]`，组合后：

```text
Context[B,Nq,S,D] = [1,2,3,2]
→ transpose [B,S,Nq,D] = [1,3,2,2]
→ reshape [B,S,Nq×D] = [1,3,4]
→ [B,S,H] = [1,3,4]
```

每个 token 的两个 2 维 head 结果重新拼成 4 维。输出恢复 `[B,S,H]` 后，才能经过输出投影并与 residual 分支相加。

## 6. 正式教学模型配置

理解极小例子后，再看实际准备实现的配置：

```text
B   = 2       一次处理 2 条序列
S   = 128     每条序列 128 个 token
V   = 8000    词表有 8000 种 token
H   = 384     每个 token 的 hidden state 有 384 个数
Nq  = 6       6 个 Query heads
Nkv = 2       2 个 KV heads，采用 GQA
D   = 64      每个 head 64 维，6×64=384
I   = 1024    FFN 中间维度
L   = 6       6 个 Transformer Blocks
```

### 6.1 “6 个 Query heads”是什么意思

每个 token 的 384 维 Query 输出被组织成：

```text
head 0：64 个数
head 1：64 个数
head 2：64 个数
head 3：64 个数
head 4：64 个数
head 5：64 个数
总计：6 × 64 = 384
```

这不是简单地把原输入向量机械切开。先由可训练矩阵 `Wq[384,384]` 生成新的 384 个 Query 数值，再把输出组织成 6 组。

完整变化为：

```text
X[2,128,384] @ Wq[384,384]
→ Q_flat[2,128,384]
→ reshape [2,128,6,64]
→ transpose [2,6,128,64]
```

最终可以读成：2 条序列，每条序列有 6 个 Query heads，每个 head 包含 128 个 token，每个 token 的 Query 是 64 维。

### 6.2 为什么只有 2 个 KV heads

本例采用 Grouped-Query Attention（GQA）。6 个 Query heads 分组共享 2 个 KV heads：

```text
Query heads 0、1、2 → 共享 KV head 0
Query heads 3、4、5 → 共享 KV head 1
```

所以：

```text
Q = [2,6,128,64]
K = [2,2,128,64]
V = [2,2,128,64]
```

每 3 个 Query heads 共享一组 K/V。Query 仍然各自不同，只是查看共同的 K/V。

与 6 个 KV heads 的普通 MHA 相比，GQA 在这里把 K/V 参数和未来推理时的 KV Cache 降到约三分之一。它是否带来等比例速度提升取决于实现、硬件和负载，不能仅从 shape 推断。

### 6.3 Attention Scores

每个 Query head 都需要为 128 个 Query 位置与 128 个 Key 位置计算匹配：

```text
Scores = [B,Nq,S,S]
       = [2,6,128,128]
```

元素数：

```text
2 × 6 × 128 × 128 = 196,608
```

逻辑上这是完整 Scores 的 shape。FlashAttention 等实现不一定把整个矩阵长期写入高带宽内存，因此逻辑 shape 不应直接等同于实际峰值显存。

## 7. Embedding、FFN 和 LM Head

### 7.1 Embedding

词表有 8000 个 token，每个 token 对应一个 384 维向量：

```text
W_embed[V,H] = [8000,384]
```

若某位置 token ID 是 57，Embedding lookup 就取出第 57 行：

```text
token ID 57 → W_embed[57] → [384]
```

整个 batch：

```text
input_ids[B,S] = [2,128]
→ hidden X[B,S,H] = [2,128,384]
```

Embedding 参数量：

```text
8000 × 384 = 3,072,000
```

### 7.2 SwiGLU FFN

Attention 负责 token 之间交换信息。FFN 则对每个 token 独立进行通道变换。

SwiGLU 简化数据流：

```text
gate = X @ W_gate
up   = X @ W_up
middle = SiLU(gate) × up
out = middle @ W_down
```

Shape：

```text
X:       [B,S,H] = [2,128,384]
gate/up: [B,S,I] = [2,128,1024]
out:     [B,S,H] = [2,128,384]
```

三个权重矩阵：

```text
W_gate[H,I] = [384,1024]
W_up[H,I]   = [384,1024]
W_down[I,H] = [1024,384]
```

参数量：

```text
3 × H × I = 3 × 384 × 1024 = 1,179,648
```

FFN 不混合不同 token，因为每个 token 都独立使用同一组权重。Token 间信息交换发生在 Attention。

### 7.3 LM Head

最后需要把每个 token 的 384 维 hidden state 转成对 8000 个词表项的分数：

```text
hidden[B,S,H] @ W_lm[H,V]
→ logits[B,S,V]

[2,128,384] @ [384,8000]
→ [2,128,8000]
```

每个 token 位置都会得到 8000 个 logits。Logit 是未经 softmax 的分数。

如果 LM Head 与 Embedding 共享权重，可以减少一份 `[8000,384]` 参数，但 logits 的 `[B,S,V]` shape 和相关计算不会消失。

## 8. dtype 与理论内存

常用存储大小：

```text
FP32  = 4 bytes/element
FP16  = 2 bytes/element
BF16  = 2 bytes/element
INT8  = 1 byte/element
INT4  ≈ 0.5 byte/element，实际还需要 scale、分组和打包元数据
```

理论内存公式：

```text
元素数 = shape 各维度相乘
理论字节数 = 元素数 × 每元素字节数
```

例如 hidden：

```text
shape = [2,128,384]
元素数 = 2 × 128 × 384 = 98,304
FP32 = 98,304 × 4 = 393,216 bytes = 384 KiB
FP16 = 98,304 × 2 = 196,608 bytes = 192 KiB
```

Attention Scores：

```text
[2,6,128,128]
元素数 = 196,608
FP32 = 768 KiB
```

Logits：

```text
[2,128,8000]
元素数 = 2,048,000
FP32 ≈ 7.81 MiB
```

即使模型较小，词表维度很大的 logits 也可能成为明显的临时张量。

## 9. 参数量与训练内存不是一回事

本教学模型在共享 Embedding/LM Head 时约 12.5M 参数。FP32 权重本体约 50 MB（十进制）。

训练时还需要：

- 参数梯度；
- Adam 一阶状态；
- Adam 二阶状态；
- forward 激活；
- backward 临时张量；
- allocator 预留；
- Python、PyTorch 和 Metal runtime；
- 操作系统与其他程序占用的统一内存。

因此：

```text
参数量 × 权重 dtype
```

只能估算权重本体，不能代表完整训练峰值内存。

## 10. 序列长度为什么危险

Hidden states 包含一个序列维度：

```text
[B,S,H]
```

把 `S` 从 128 增加到 512，即增加 4 倍，hidden 元素数增加 4 倍。

Attention Scores 包含两个序列维度：

```text
[B,Nq,S,S]
```

因此：

```text
128 × 128 → 512 × 512
增长倍数 = 4 × 4 = 16
```

这就是标准 Attention 在序列长度上的二次增长来源。完整模型的实际内存和运行时间还受 kernel、重计算、缓存及是否物化 Scores 影响。

## 11. 最小代码验证

当前脚本不依赖 PyTorch，只验证 shape、参数量和内存算术：

```bash
python3 exercises/day01/tensor_shape_memory.py
```

脚本应输出：

- hidden、Q/K/V、Scores、FFN 中间张量和 logits 的 shape；
- FP32/FP16 理论内存；
- tied/untied LM Head 参数量；
- `S=128` 到 `S=512` 时 Scores 增长 16 倍。

它只能验证算术，不能证明真实训练峰值，也不能验证 MPS kernel 性能。

## 12. 常见误解与边界

- 6 个 heads 不是 6 个模型，也不是 6 个硬件核心。
- `reshape` 不生成新特征，只重新解释维度。
- `transpose` 改变维度顺序，不是训练参数。
- Query 不是自然语言问题，而是数值向量。
- 不同 heads 可能学习不同关系，但不能默认每个 head 有固定可解释职责。
- GQA 共享 K/V，不意味着 Query heads 相同。
- shape 描述逻辑结构，不一定表示中间张量会完整写入内存。
- 参数文件大小不是训练内存。
- 理论 FLOPs、理论内存与实测性能是三类不同证据。

## 13. 手算练习

先独立计算，再运行脚本核对。

### 练习 1：极小 Attention

给定：

```text
B=1, S=3, H=4, Nq=2, Nkv=2, D=2
```

回答：

1. `X`、`Wq` 和 `Q_flat` 的 shape。
2. `Q_flat` reshape 和 transpose 后每一步的 shape。
3. 单个 head 的 Q/K shape。
4. 单个 head 的 Scores shape。
5. 加入两个 heads 后完整 Scores shape。

### 练习 2：正式 Attention

给定正式配置：

1. Q、K、V 的 shape 分别是什么？
2. 为什么 Q 有 6 个 heads，而 K/V 只有 2 个？
3. 每个 KV head 被几个 Query heads 共享？
4. Scores 有多少元素，FP32 理论占多少 KiB？

### 练习 3：Embedding 与 FFN

1. Embedding table 的 shape 和参数量。
2. gate/up 中间张量的 shape。
3. 三个 FFN 权重矩阵的 shape。
4. FFN 总参数量。
5. 为什么 FFN 不负责 token 间的信息交换？

### 练习 4：LM Head

1. LM Head 权重和 logits 的 shape。
2. Logits 的 FP16 理论大小。
3. Weight tying 减少什么，不减少什么？

### 练习 5：长度压力

把 `S` 从 128 增加到 512：

1. Hidden 元素数增加多少倍？
2. Scores 元素数增加多少倍？
3. 原因是什么？

## 14. 面试口述

### 30 秒目标

能够简短解释：Attention head 是同一层中的一个表示子空间；模型把每个 token 的 hidden state 投影为多组 Q/K/V，每组独立计算位置匹配，再把结果拼回 hidden size。

### 2 分钟目标

需要包含：

1. `[B,S,H]` 的含义。
2. `Wq` 如何产生 `Nq×D` 个输出。
3. reshape/transpose 为什么需要。
4. Scores 为什么是 `[B,Nq,S,S]`。
5. GQA 如何减少 K/V heads。
6. 这些 shape 对参数和内存的影响。

## 15. 当日验收

请提交以下内容：

1. 用自然语言解释“6 个 Query heads”。
2. 完整写出正式配置从 `X` 到 `Q` 的四步 shape 变化，并解释每一步。
3. 完成练习 2 和练习 5 的手算。
4. 回答：为什么 12.5M 参数模型训练时的内存明显大于 FP32 权重本体？
5. 写出今天最不确定的一个概念。

只有能解释“为什么”，而不只是写出 shape，Day 1 才算通过。
