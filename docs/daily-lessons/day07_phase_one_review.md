# Day 7：第一阶段复盘与完整训练链路验收

## 1. 今日核心问题

Day 7 不再增加一个孤立算子，而是回答：

> 能否从 token IDs 开始，完整解释并验证 Decoder-only Transformer 的 forward、loss、backward、AdamW、梯度累积和 checkpoint resume？

今天的三个产物是：

1. 项目根目录第一版 README，让新读者知道项目是什么、如何运行、验证了什么；
2. 一个第一阶段综合验收程序，复用 Day 4–6 的实现；
3. 一组覆盖 Transformer 基础和训练闭环的面试口述题。

完整链路：

```text
token sequences [B,S+1]
    ↓ shifted labels
input_ids [B,S] + labels [B,S]
    ↓ Embedding + position
hidden [B,S,H]
    ↓ N 个 Decoder Blocks
hidden [B,S,H]
    ↓ final RMSNorm + LM Head
logits [B,S,V]
    ↓ Cross-Entropy
scalar loss
    ↓ backward
parameter.grad
    ↓ AdamW / accumulation boundary
更新参数和 optimizer state
    ↓ checkpoint
可恢复训练状态
```

---

## 2. 前置知识与术语

### 2.1 Phase

**Phase** 的英文核心含义是“阶段”。这里的第一阶段指 Day 1–7：从 Tensor 基础到完整训练循环。

### 2.2 Acceptance

**Acceptance** 的英文核心含义是“验收”。验收不是再次阅读代码，而是用可观察证据判断能力是否真正成立。

```text
代码能导入     ≠ forward 正确
命令退出为 0   ≠ 数学行为正确
loss 能计算    ≠ 参数得到更新
权重能加载     ≠ 训练能够精确续跑
```

### 2.3 Baseline 与边界

**Baseline** 是用于比较的基准路径。Day 7 使用：

- 普通大 batch update 作为梯度累积 baseline；
- 不发生中断的连续训练作为 checkpoint resume baseline。

边界表示实验没有证明的范围。主动写出边界不是降低完成度，而是防止把局部结果错误推广成通用结论。

---

## 3. 从直觉到机制

### 3.1 Day 1：先学会看 Tensor，而不是只看模块名

任何 Transformer 数据流都应先追踪 shape：

```text
token IDs      [B,S]
hidden states  [B,S,H]
Q/K/V          [B,N,S,D]
scores         [B,N,S,S]
logits         [B,S,V]
```

符号含义：

- (B)：一次 forward 中的 sequence 数量；
- (S)：每条 sequence 的 token 数量；
- (H)：每个 token 的隐藏向量维度；
- (N)：Query head 数量；
- (D=H/N)：每个 head 的维度；
- (V)：词表大小。

### 3.2 Day 2：模型看到的是 token IDs，不是字符串

```text
文本
→ tokenizer
→ token IDs
→ token embedding
→ 加入位置信息
→ Transformer Blocks
→ logits
```

当前项目使用人工 token IDs，没有实现真实 tokenizer。因此已经验证的是模型 Tensor 数据流，不是文本切词质量。

### 3.3 Day 3：Attention 混合 token，FFN 混合 hidden features

Self-Attention 的核心问题是：当前位置应该从哪些历史 token 读取信息？

```text
Q：当前位置想查什么
K：每个历史位置提供什么检索标签
V：匹配后真正传递什么信息
```

Causal mask 在 Softmax 之前作用于 attention scores，把未来位置设为负无穷，使其 Softmax 权重为 0。

FFN 则对每个 token 位置独立地进行非线性 feature transformation。Attention 和 FFN 分别承担 sequence 维与 hidden-feature 维的信息变换。

### 3.4 Day 4：从隐藏状态到可求导 loss

Decoder Block：

```text
x
├─ RMSNorm → Causal Attention → residual add
└─ RMSNorm → SwiGLU FFN       → residual add
```

多个 Block 后：

```text
hidden [B,S,H]
→ final RMSNorm
→ LM Head
→ logits [B,S,V]
```

Cross-Entropy 读取每个位置对正确 token 给出的概率，形成 scalar loss；`backward()` 再沿计算图得到每个参数的梯度。

### 3.5 Day 5：梯度不等于参数更新

一个标准训练 step：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(input_ids)
loss = cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

- `zero_grad`：清除上一轮 `.grad`；
- forward：用当前参数计算 logits；
- loss：衡量预测分布与标签的差异；
- backward：计算梯度；
- AdamW step：读取梯度和历史 state，更新参数。

### 3.6 Day 6：一个 optimizer step 可以跨多个 micro-steps

如果一个大 batch 放不进显存：

```text
zero_grad
→ micro-batch 1 forward/backward
→ micro-batch 2 forward/backward
→ optimizer.step
```

对于等大 micro-batches 且 loss 为 mean：

$$
L_{effective}=\frac{1}{A}\sum_{a=1}^{A}L_a
$$

其中 $A$ 是 accumulation steps。因此每个 micro loss 应除以 $A$。

Checkpoint 则把 model、optimizer、completed steps 和随机/采样状态从内存写入持久化文件，使新进程能够重建训练状态。

---

## 4. 极小手算例子

配置：

```text
B = 2 sequences
S = 3 input tokens
H = 4 hidden features
V = 8 vocabulary entries
```

完整 token 数据长度为 4：

```text
tokens.shape = [2,4]
```

右移以后：

```text
input_ids.shape = [2,3]
labels.shape    = [2,3]
```

模型输出：

```text
logits.shape = [2,3,8]
```

一共存在：

$$
B\times S=2\times3=6
$$

个 next-token prediction positions。

如果使用两个 micro-batch，各自平均梯度为：

$$
g_1=2,\qquad g_2=4
$$

则 effective batch 的平均梯度为：

$$
g=\frac{2+4}{2}=3
$$

AdamW 使用这个最终梯度更新一阶矩、二阶矩和参数，而不是在每个 micro-step 更新一次。

---

## 5. 正式模型配置

Day 7 综合验收复用 Day 6 的小模型配置：

```text
Vocabulary V       = 32
Hidden H           = 32
Heads N            = 4
Head dimension D   = 8
FFN intermediate I = 64
Decoder layers L   = 1
Input length S     = 5
```

使用较小配置是为了快速验证行为，不是为了模拟真实大语言模型规模。

程序验证：

- Dataset `[8,6]` 右移为输入与标签；
- 取 4 条序列时 logits 为 `[4,5,32]`；
- future-token change 不影响更早 causal outputs；
- Cross-Entropy 能 backward，参数梯度有限且 shape 正确；
- AdamW step 后参数实际变化；
- 梯度累积与大 batch update 接近；
- checkpoint resume 与连续训练完全一致。

---

## 6. 完整数据流与 Shape

### 6.1 训练数据

```text
dataset              [8,6]
selected sequences   [4,6]
input_ids             [4,5]
labels                [4,5]
```

切片不改变 token 数值，只选择不同位置：

```python
input_ids = tokens[:, :-1]
labels = tokens[:, 1:]
```

### 6.2 模型内部

```text
input_ids             [4,5]
embedding              [4,5,32]
Q/K/V before heads     [4,5,32]
Q/K/V after heads      [4,4,5,8]
attention scores       [4,4,5,5]
attention output       [4,5,32]
FFN intermediate       [4,5,64]
block output           [4,5,32]
logits                 [4,5,32]
```

这里最后一个 32 在 hidden 和 logits 中含义不同：

- hidden 的 32 是 hidden features；
- logits 的 32 是 vocabulary entries。

数值碰巧相同不代表语义相同。

### 6.3 Loss 与梯度

Cross-Entropy 逻辑上读取：

```text
logits [4,5,32] → 20 个长度为 32 的分数向量
labels [4,5]    → 20 个正确 token IDs
```

最终 loss 是 scalar，也就是 shape `[]` 的零维 Tensor。

对参数：

```text
parameter.shape == gradient.shape
```

例如 Embedding 权重为 `[32,32]`，它的 `.grad`、AdamW `exp_avg` 和 `exp_avg_sq` 也都是 `[32,32]`。

### 6.4 状态的所有权

```text
model.state_dict()      → 参数与注册 buffer
parameter.grad          → 当前 accumulation window 的梯度
optimizer.state_dict()  → AdamW moments、step、parameter groups
generator state         → 随机采样位置
completed_steps         → 训练进度
checkpoint file         → 重启后的持久化真相源
```

这些状态不能因为都与训练相关，就混成一个概念。

---

## 7. 参数、内存与计算成本

训练显存至少需要区分：

$$
M_{training}
\approx
M_{parameters}
+M_{gradients}
+M_{optimizer}
+M_{activations}
+M_{temporary}
$$

- 参数量由模型结构决定；
- gradient 与参数逐元素对应；
- AdamW 主要维护 `exp_avg` 与 `exp_avg_sq`；
- activations 随 $B,S,H,L$ 增长；
- 普通 Attention 的 score 逻辑规模随 \(S^2\) 增长；
- 临时 workspace 和 CUDA allocator 会使峰值不同于简单理论和。

梯度累积主要减少单次保存的 batch activation，不减少参数、梯度和 optimizer state。

Day 7 使用：

```python
torch.cuda.reset_peak_memory_stats()
...
torch.cuda.max_memory_allocated()
```

记录综合实验期间 PyTorch Tensor allocation 的峰值。它是框架观测值，不包含所有 CUDA context 和驱动占用，也不能直接外推真实大模型。

---

## 8. 最小代码验证

运行：

```bash
uv run python exercises/day07/phase_one_acceptance.py
```

程序复用已有模块而不是复制实现：

- Day 4：模型、shifted labels 和 causal check；
- Day 5：固定 Dataset；
- Day 6：梯度累积和 checkpoint resume 验证。

验收信号：

```text
shifted-label shapes are correct
causal independence is preserved
forward/loss/backward/AdamW step works
accumulated and large-batch updates match
interrupted training resumes exactly
invalid checkpoint is rejected
temporary checkpoint files were cleaned
```

Day 7 没有保存长期 checkpoint；测试文件位于独立临时目录，程序结束后自动清理。

本机 RTX 2060、PyTorch 2.13.0+cu130 实测：

```text
logits shape                         (4, 5, 32)
single-step cross-entropy loss       3.526596
first-parameter gradient norm        0.093326
accumulation maximum difference      1.8812716e-07
continuous/resumed final loss        0.540906 / 0.540906
resume maximum parameter difference  0
temporary checkpoint size            179.14 KiB
CUDA peak allocated                  17.11 MiB
```

这些数值是当前环境的框架观测结果；PASS 所依赖的是 shape、有限梯度、参数确实
更新、数值容差和恢复一致性，不要求其他硬件复现完全相同的耗时或显存数字。

---

## 9. 常见误解与边界

### 9.1 Tiny overfit 不代表模型具有语言能力

它证明训练闭环能够学习人工确定性映射，不证明泛化或真实生成质量。

### 9.2 Teacher-forced accuracy 不等于自回归生成准确率

teacher forcing 的每个位置都拿到真实历史前缀；生成时则读取模型自己此前生成的 token，错误可能累积。

### 9.3 相同 BOS 对应不同首 token 不是错误数据

真实语言本来具有多种可能后续。模型应学习条件概率分布，不一定能在所有相同前缀位置达到 100% argmax accuracy。

### 9.4 Seed 不等于跨环境绝对确定

相同 seed 只控制随机序列起点。硬件、PyTorch/CUDA 版本、kernel、数据顺序或随机调用次数变化都可能影响结果。

### 9.5 Checkpoint 能加载不等于精确续跑

必须与连续 baseline 比较最终参数、loss 和训练进度，才能验证恢复语义。

### 9.6 混合精度已由 Day 7.5 补充

Day 7.5 已完成 FP32 与 FP16 AMP 的实际 dtype、loss、时间和 peak allocated
对照，并验证 Autocast 与 GradScaler 的训练路径。RTX 2060 的原生 BF16 支持检查
为 false，因此 BF16 benchmark 被明确跳过；这仍是当前硬件下未执行的实验，而不
是已经验证的能力。

详见 [Day 7.5：混合精度训练、Autocast 与 GradScaler](day07_5_mixed_precision_training.md)。

### 9.7 本阶段尚未覆盖多 GPU

DDP、All-Reduce 和 `no_sync()` 已经进行了概念讨论，但当前只有单卡实测，不能把概念理解写成多卡验证结果。

---

## 10. 手算练习

### 练习 1：Shape 链路

给定：

```text
B=3, S=128, H=384, N=6, V=8000
```

回答 Embedding、拆头后 Q/K/V、Attention scores、Block output 和 logits 的 shape。

### 练习 2：训练状态

将以下状态分到 model、gradient、optimizer、sampler/RNG 或 checkpoint：

1. `exp_avg`；
2. Embedding weight；
3. 当前 `.grad`；
4. completed optimizer steps；
5. 下一次 batch 的随机序列位置。

### 练习 3：梯度累积

每卡 micro-batch size 为 4，accumulation steps 为 8，world size 为 2：

1. 全局 effective batch size 是多少？
2. 每个 optimizer step 有多少次本地 forward/backward 调用？
3. 使用 DDP 时，为什么前 7 个 micro-steps 通常使用 `no_sync()`？

### 练习 4：证据边界

解释以下各自能证明什么、不能证明什么：

1. causal unit test PASS；
2. 一次 backward 成功；
3. tiny dataset loss 下降；
4. checkpoint 能加载；
5. resume 与 continuous 参数完全一致。

---

## 11. 面试口述

### 11.1 30 秒版本

Decoder-only Transformer 将 token IDs 映射为 hidden states，通过带 causal mask 的 Self-Attention 混合历史 token 信息，再通过 FFN 逐位置变换特征，最终 LM Head 产生词表 logits。训练使用 shifted labels 和 Cross-Entropy，backward 得到梯度，AdamW 根据一阶、二阶矩更新参数。显存不足时可以累积 micro-batch 梯度；精确续训还需要保存 optimizer、step 和随机采样状态。

### 11.2 两分钟版本

完整口述必须覆盖：

1. `[B,S] → [B,S,H] → [B,N,S,D] → [B,S,V]`；
2. Q/K/V、causal mask、Softmax 和 value aggregation；
3. RMSNorm、residual、SwiGLU FFN 的分工；
4. shifted inputs/labels 和 Cross-Entropy；
5. `.grad`、AdamW state 和 parameter update；
6. micro-batch 与 effective batch；
7. model-only save 与完整 checkpoint 的区别；
8. 已验证事实与尚未验证边界。

### 11.3 模拟面试题

1. 从一批 token IDs 开始，完整讲解 Decoder LM 如何产生 loss。
2. causal mask 在什么计算阶段加入？为什么训练和 prefill 都需要？
3. Attention 和 FFN 分别混合哪个维度的信息？
4. AdamW 为什么需要 `exp_avg` 和 `exp_avg_sq`？
5. loss 下降但 accuracy 不变是否矛盾？
6. 梯度为什么默认累积，什么时候应该清零？
7. 梯度累积何时严格等价于大 batch，何时不等价？
8. 为什么只保存模型权重不能精确续训？
9. 训练峰值显存由哪些部分组成？
10. 当前项目结果为什么不能直接推广到真实 LLM？

---

## 12. 当日验收

1. 从空白画出 token IDs 到 AdamW step 的完整数据流。
2. 不看资料写出五个关键 Tensor shape。
3. 解释 Attention、FFN、RMSNorm 和 residual 的不同职责。
4. 解释 logits、probability、Cross-Entropy 和 accuracy 的关系。
5. 区分 parameter、gradient、optimizer state 和 checkpoint。
6. 解释梯度累积的数学条件和性能代价。
7. 运行 Day 7 综合验收并核对所有 PASS 与数值结果。
8. 阅读项目根 README，确认命令、已验证能力和边界与实际一致。
9. 完成一次 2 分钟口述和至少 5 道模拟面试题。
10. 明确记录下一阶段入口：Day 8 的朴素生成、prefill、decode 与 KV Cache。

只有在能够解释实验为什么通过、也能说明它没有证明什么时，第一阶段才算完成。
