# Day 4：从 Attention 子层到可反向传播的 Decoder LM

## 1. 今日核心问题

今天要把前三天的零散部件接成第一条真实训练路径：

> token IDs 如何经过 Embedding、完整 Decoder Blocks、LM Head 变成 logits；真实 token 如何右移成 labels；cross-entropy 如何产生标量 loss；Autograd 又如何把梯度传回 Embedding、Attention、SwiGLU 和 LM Head？

今日数据流：

```text
complete tokens[B,S+1]
├→ input_ids = tokens[:,:-1]   [B,S]
└→ labels    = tokens[:,1:]    [B,S]

input_ids
→ Token Embedding + Position Embedding
→ Decoder Block × L
     ├→ RMSNorm → Causal MHA → residual
     └→ RMSNorm → SwiGLU    → residual
→ Final RMSNorm
→ LM Head
→ logits[B,S,V]
→ cross-entropy(logits, labels)
→ scalar loss
→ backward
→ parameter gradients
```

今天完成后，应能回答：

1. 一个完整 Decoder Block 为什么有两个 Norm 和两条 residual？
2. SwiGLU 的 gate、up 和 down 三个投影分别做什么？
3. LM Head 为什么把 H 维映射成 V 维？
4. next-token labels 如何从原 token sequence 构造？
5. 为什么 cross-entropy 计算时直接传 logits，而不在模型中先做 softmax？
6. `loss.backward()` 之后，哪些参数应该拥有梯度？

今天只做一次 forward/backward，不执行 optimizer step、不训练到收敛。Day 5 再加入数据集、batch 采样、optimizer 和完整训练循环。

## 2. 昨日回忆

不查 Day 3，先回答：

1. Q/K/V 分别决定什么？
2. causal mask 加在 Scores 的哪个阶段？
3. `[B,S,H]` 如何拆成 `[B,N,S,D]`？
4. RMSNorm 改变什么，保留什么？
5. 为什么 Attention 输出必须合并 heads 并回到 H 维？

## 3. 今日配置与环境

### 3.1 正式教学配置

```text
B = 2
S = 8
V = 8000
H = 384
Nq = 6
Nkv = 2
D = 64
I = 1024
L = 6
```

`I` 是 intermediate size，表示 FFN 中间特征宽度。

### 3.2 最小实验配置

为了让 RTX 2060 上的正确性实验快速、稳定，代码使用：

```text
B = 2
complete sequence length = 6
training input length = 5
V = 32
H = 64
N = 4
D = 16
I = 128
L = 2
```

这只是缩小尺寸，不改变 Decoder LM 机制。

### 3.3 已核验环境

```text
Python 3.13.15
torch 2.13.0+cu130
PyTorch compiled CUDA 13.0
GPU: NVIDIA GeForce RTX 2060
```

`uv.lock` 是当前依赖解析的真相源。`pyproject.toml` 当前允许 `torch>=2.13.0`，所以未来主动升级锁文件时可能变更 torch 版本；今日结论只对上述实测环境负责。

## 4. 完整 Decoder Block

### 4.1 Pre-Norm 结构

今天的 Block 是：

```text
X
├── residual path 1 ───────────┐
↓                              │
RMSNorm                        │
↓                              │
Causal Multi-Head Attention   │
↓                              │
attention update                 │
└───────── add ◄─────────────────┘
↓
X_attn
├── residual path 2 ─────────┐
↓                              │
RMSNorm                        │
↓                              │
SwiGLU FFN                     │
↓                              │
ffn update                       │
└───────── add ◄─────────────────┘
↓
Y
```

公式：

```text
X_attn = X + Attention(RMSNorm_attn(X))
Y      = X_attn + FFN(RMSNorm_ffn(X_attn))
```

两个 RMSNorm 通常不共享参数，因为 Attention 和 FFN 是两个不同子层，各自学习自己的逐维缩放 `gamma[H]`。

### 4.2 Attention 与 FFN 的分工

- Attention 在 token 之间传递信息，它混合 sequence 维。
- FFN 对每个 token 独立处理，它不直接读取其他 token，主要混合 feature 维。

对同一个 batch 中的两个 token，FFN 使用同一组权重，但分别处理各自的 `[H]` 向量。

## 5. SwiGLU FFN

SwiGLU 由三个投影组成：

```text
gate = X @ W_gate
up   = X @ W_up
hidden = SiLU(gate) ⊙ up
output = hidden @ W_down
```

公式：

```text
SwiGLU(X) = (SiLU(XW_gate) ⊙ XW_up) W_down
```

其中：

```text
W_gate[H,I]
W_up[H,I]
W_down[I,H]
```

Shape 变化：

```text
X                       [B,S,H]
gate projection         [B,S,I]
up projection           [B,S,I]
SiLU(gate) ⊙ up       [B,S,I]
down projection         [B,S,H]
```

### 5.1 gate 是什么

`SiLU(gate)` 为每个中间特征产生一个与输入相关的缩放值，再与 `up` 分支逐元素相乘。因此这不是一个固定的人工开关，而是模型学到的输入依赖调制。

SiLU 定义为：

```text
SiLU(x) = x × sigmoid(x)
```

### 5.2 为什么还要 down projection

gate/up 将 H 维扩展到 I 维，以提供更宽的非线性特征空间。但 residual stream 宽度是 H，所以必须通过 `W_down[I,H]` 映射回 H，才能执行：

```text
X_attn + ffn_update
```

### 5.3 参数量

忽略 bias：

```text
SwiGLU parameters = H×I + H×I + I×H = 3HI
```

正式配置 `H=384,I=1024`：

```text
3 × 384 × 1024 = 1,179,648
```

这比今天正式 GQA 的 Attention 投影参数 `393,216` 更多。因此不能把 Transformer Block 的主要参数全部等同于 Attention。

## 6. 从完整 token sequence 构造 labels

假设完整序列是：

```text
tokens = [BOS, 我, 喜欢, 学习, EOS]
```

next-token training 构造：

```text
input_ids = [BOS, 我,   喜欢, 学习]
labels    = [我,   喜欢, 学习, EOS]
```

对齐关系：

```text
input BOS  位置的 logits 对应 label “我”
input 我    位置的 logits 对应 label “喜欢”
input 喜欢  位置的 logits 对应 label “学习”
input 学习  位置的 logits 对应 label EOS
```

所以代码中使用：

```python
input_ids = tokens[:, :-1]
labels = tokens[:, 1:]
```

labels 来自训练数据中的真实后续 token，不是由模型自己猜出来的监督信号。

## 7. LM Head：从 H 维回到词表

经过 L 个 Decoder Blocks 后：

```text
hidden[B,S,H]
```

最终 RMSNorm 不改变 shape，LM Head 使用：

```text
W_lm[H,V]
```

计算：

```text
logits = hidden @ W_lm
```

得到：

```text
logits[B,S,V]
```

每个 token 位置都有 V 个未归一化候选分数。训练时会使用所有有效位置；逐 token 生成时通常只使用最后一个有效位置的 logits。

### 7.1 Weight Tying

有些模型共享 Token Embedding 和 LM Head 权重：

```text
W_lm = W_embeddingᵀ
```

这可减少约 `V×H` 参数。今天的代码故意不共享，以便独立检查 Embedding 和 LM Head 的梯度。这是教学实现选择，不是所有真实模型的统一做法。

## 8. Cross-Entropy 如何连接 logits 和 labels

某个位置的 logits：

```text
z[V]
```

真实 label 是一个 token ID：

```text
y ∈ [0,V)
```

该位置的负对数似然：

```text
loss = -log(softmax(z)[y])
```

等价地：

```text
loss = logsumexp(z) - z[y]
```

后一形式不需要先物化一个可能接近 0 的概率再取对数，数值更稳定。因此 PyTorch `F.cross_entropy` 接收的是原始 logits，不是手动 softmax 后的概率。

代码先把 batch 和 sequence 位置展平：

```text
logits[B,S,V] → logits[B×S,V]
labels[B,S]   → labels[B×S]
```

然后对 `B×S` 个 next-token prediction 计算平均 loss。

## 9. Backward 究竟产生什么

Forward 产生标量：

```text
loss shape = []
```

调用：

```python
loss.backward()
```

Autograd 沿计算图使用链式法则，为参与 loss 的可训练参数累积：

```text
parameter.grad
```

今天至少检查：

```text
Token Embedding gradient
Attention Q projection gradient
SwiGLU gate projection gradient
LM Head gradient
```

梯度存在只能证明参数连接到 loss，不等于模型已经学会任务。Day 5 需要通过多次 optimizer step 和极小数据 overfit 验证真正的学习闭环。

## 10. 完整 Shape 账本

使用正式配置：

```text
complete tokens                [B,S+1]
input_ids                      [B,S]
labels                         [B,S]
embedding output               [B,S,H]
```

每个 Block：

```text
attention norm                 [B,S,H]
Q                              [B,Nq,S,D]
K,V                            [B,Nkv,S,D]
attention scores/weights       [B,Nq,S,S]
attention update               [B,S,H]
first residual output          [B,S,H]
ffn norm                       [B,S,H]
gate/up                        [B,S,I]
gated hidden                   [B,S,I]
ffn update                     [B,S,H]
second residual output         [B,S,H]
```

输出：

```text
final norm                     [B,S,H]
logits                         [B,S,V]
flattened logits               [B×S,V]
flattened labels               [B×S]
loss                           []
```

## 11. 参数与激活边界

每个正式 Block 的主要参数：

```text
GQA projections  = 393,216
SwiGLU           = 1,179,648
two RMSNorm      = 768
total/block      = 1,573,632
```

一个常见误区是把参数记忆、前向激活和训练总显存混成同一个数。今天的 CUDA 峰值还包含：

- 参数张量；
- forward activations；
- backward 需要的保留值；
- parameter gradients；
- PyTorch/CUDA allocator 已申请的 block；
- CUDA context 和库开销中未必都计入 `memory_allocated`的部分。

今天没有 optimizer，所以不包含 AdamW 的一阶和二阶状态。

## 12. 最小 GPU 代码验证

运行：

```bash
uv run python exercises/day04/decoder_block_training.py
```

程序会直接要求 CUDA。如果 `torch.cuda.is_available()` 为 False，它会明确失败，不会静默回退到 CPU。

它验证：

1. 模型、tokens、logits 和 loss 真正位于 CUDA 路径；
2. shifted inputs/labels 对齐；
3. 两个完整 Decoder Blocks 的 forward；
4. causal mask 右上三角权重全为 0；
5. 只改最后输入 token，之前所有位置 logits 完全不变；
6. cross-entropy loss 是有限标量；
7. Embedding、Q projection、SwiGLU gate 和 LM Head 都获得有限非零梯度；
8. 当前 CUDA allocated 和 peak allocated 内存。

## 13. 常见误解与边界

- Decoder Block 不只有 Attention；SwiGLU FFN 通常占有大量参数和计算。
- Attention 混合 token 间信息；FFN 对各 token 独立使用同一组权重处理特征。
- SwiGLU 的 gate 是连续数值调制，不是必然只取 0/1 的离散开关。
- `labels` 是整数 token IDs，不是 one-hot 概率向量。
- LM Head 输出的是 logits，不是概率。
- `F.cross_entropy` 内部稳定地组合 log-softmax 和 NLL；不要在前面手动 softmax。
- 训练时每个有效位置都可以提供 next-token loss，不是只训练最后一个位置。
- `loss.backward()` 只计算并累积梯度，不会更新参数；参数更新需要 optimizer step。
- 一次 backward 成功不证明 loss 会下降，也不证明模型能生成有意义文本。
- 今天的代码使用标准 MHA，正式教学配置是 GQA；GQA 的 KV head 共享将在 Day 10 单独实现。
- 今天使用可读的显式 Scores/Mask/Softmax，不代表生产实现应长期物化完整 `[S,S]` 矩阵。
- CUDA `memory_allocated` 不等于 `nvidia-smi` 显示的进程总显存。
- NumPy 未安装会在当前 torch import 中产生警告，但不影响本程序的纯 PyTorch CUDA 路径；本次未因此新增依赖。

## 14. 手算与理解练习

### 练习 1：完整 Block Shape

给定：

```text
B=2,S=8,H=384,Nq=6,Nkv=2,D=64,I=1024
```

写出：

1. Attention 前后 shape。
2. gate/up/down 三个分支的权重与 activation shape。
3. 两次 residual 加法两边的 shape。
4. 为什么 FFN 不改变 sequence length？

### 练习 2：Shifted Labels

给定：

```text
tokens = [BOS,A,B,C,EOS]
```

1. 写出 input_ids。
2. 写出 labels。
3. input 位置 B 的 logits 监督目标是什么？
4. 为什么 causal mask 使这些位置可以并行训练？

### 练习 3：Cross-Entropy

给定一个三词词表：

```text
logits = [2,1,0]
label = 0
```

1. 计算 softmax 概率。
2. 计算真实类别的 `-log(p)`。
3. 如果 label 改成 2，loss 为什么会增大？
4. 为什么代码应直接把 logits 传给 cross-entropy？

### 练习 4：梯度路径

从 loss 开始，反向写出至少一条到 Token Embedding 的路径。然后回答：

1. `loss.backward()` 是否直接修改权重？
2. 为什么一个从未在 batch 中出现的 token embedding row 梯度可能为 0？
3. residual 为反向传播提供了什么路径？

### 练习 5：参数量

1. `H=384,I=1024` 的 SwiGLU 有多少无 bias 参数？
2. 一个 Block 的两个 RMSNorm 有多少参数？
3. 未 tying 的 `W_embedding[V,H]` 和 `W_lm[H,V]` 各有多少参数？
4. Weight tying 减少什么，不减少什么？

## 15. 面试口述

### 15.1 30 秒目标

完整 Decoder Block 通常包含两个 Pre-Norm 子层：先做 RMSNorm、causal self-attention 和 residual，再做 RMSNorm、SwiGLU FFN 和 residual。多层 Block 后，Final Norm 和 LM Head 把 `[B,S,H]` 投影成 `[B,S,V]` logits。训练标签是原 token sequence 右移一位，cross-entropy 比较每个位置的 logits 与真实下一 token，backward 把梯度传回全部相关参数。

### 15.2 两分钟目标

需要说清：

1. Attention 与 FFN 分别混合哪个维度的信息；
2. SwiGLU 三个投影的数据流和 shape；
3. 两条 Pre-Norm residual 的作用；
4. shifted labels 的监督真值来源；
5. LM Head、logits 和 cross-entropy 的接口；
6. backward 产生梯度但不更新参数的边界。

### 15.3 三道口述题

1. 为什么 Transformer Block 既需要 Attention 又需要 FFN？
2. Causal LM 的 labels 如何构造，为什么每个位置都可以贡献 loss？
3. `loss.backward()` 和 `optimizer.step()` 分别做什么？

## 16. 当日验收

请独立完成：

1. 不看讲义画出完整 Decoder Block 的两条 Pre-Norm residual。
2. 为正式配置写出 Attention、SwiGLU 和 LM Head 的 shape。
3. 完成 shifted-labels 练习，逐位置说出输入与监督目标。
4. 说明为什么 cross-entropy 接受 logits 而不是手动 softmax 结果。
5. 运行 Day 4 脚本，核对 GPU、shape、loss、梯度与两项因果性检查。
6. 解释为什么有梯度不等于已经学会任务。
7. 用两分钟口述从 token IDs 到 scalar loss 和 parameter gradients 的完整路径。
8. 记录今天最不确定的一个问题。

只有同时满足以下条件，Day 4 才算通过：

- 能画出 Attention 和 SwiGLU 两个子层的数据流；
- 能正确构造 input_ids 和 labels；
- 不在 `F.cross_entropy` 前手动做 softmax；
- 能解释 scalar loss 如何连到至少四类参数梯度；
- 能区分 backward 与 optimizer step；
- 能用修改未来 token 的对照试验检查因果性；
- 程序在 CUDA 上完成 forward 和 backward；
- 能说出至少一个今日实验尚未验证的边界。
