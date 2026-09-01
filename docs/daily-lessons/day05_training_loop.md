# Day 5：数据集、AdamW 与可收敛的训练循环

## 1. 今日核心问题

Day 4 已经证明一次 forward/backward 可以产生梯度。但模型只有反复执行“采样数据—计算 loss—更新参数”，才能真正学习。

今天只解决：

> 固定 token sequences 如何变成可采样的训练数据，AdamW 如何利用梯度维护优化状态并更新参数，一个最小训练循环又如何用完整数据集评估来证明模型确实学会了极小数据？

数据流：

```text
fixed token dataset[N,S+1]
→ sample batch[B,S+1]
→ shifted input_ids[B,S] + labels[B,S]
→ model forward
→ logits[B,S,V]
→ cross-entropy scalar loss
→ zero_grad
→ backward
→ AdamW step
→ repeat
→ full-dataset evaluation
```

今天完成后应能回答：

1. Dataset、batch 和单个 training example 的边界是什么？
2. 为什么每次需要 `zero_grad()`？
3. `backward()` 和 `optimizer.step()` 分别修改什么？
4. AdamW 的一阶、二阶状态分别记录什么？
5. 为什么训练 loss 和完整数据集 evaluation loss 不能混为同一个事实？
6. 为什么“极小数据 overfit”是正确性检查，但不是泛化能力证据？

今天不实现 checkpoint/resume、mixed precision、gradient accumulation 或训练数据文件管线，它们是 Day 6 的主线。

## 2. 昨日回忆

1. `tokens[:, :-1]` 和 `tokens[:, 1:]` 分别产生什么？
2. `F.cross_entropy` 为什么接收 logits，不接收手动 softmax 结果？
3. `loss.backward()` 产生了什么？
4. 为什么一次 backward 不证明模型已经学会？
5. SwiGLU 中 gate/up/down 三条数据路径的 shape 是什么？

## 3. Dataset、Example 与 Batch

### 3.1 一条完整序列

今天每条数据都是已经 tokenized 的完整序列：

```text
[序列标识, token_1, token_2, token_3, token_4, EOS]
```

这里故意让 8 条序列使用不同的“序列标识”。如果它们都以同一个 `BOS`
开始，却要求这个完全相同的前缀预测多个不同的下一个 token，那么训练标签本身
就互相冲突，模型不可能在该位置达到 100% 准确率。本练习要验证训练管线能否
精确记住确定性映射，因此先排除这种不可约的数据歧义。

Shape：

```text
one example: [S+1]
```

通过右移构造：

```text
input_ids: [序列标识, token_1, token_2, token_3, token_4]
labels:    [token_1, token_2, token_3, token_4, EOS]
```

一条长度 6 的完整序列提供 5 个 next-token predictions。

### 3.2 Dataset

代码中固定 8 条序列：

```text
dataset shape = [8,6]
```

所以完整数据集有：

```text
8 × 5 = 40 个 next-token 监督位置
```

这个 Dataset 不是自然语言语料，只是用来验证训练机制的确定性极小数据。

### 3.3 Batch 采样

每个 optimizer step 不使用全部 8 条序列，而是随机采样 4 条：

```text
sampled complete batch [4,6]
input_ids             [4,5]
labels                [4,5]
```

当前代码是有放回采样：同一条 sequence 在一个 batch 中可能出现多次，某条也可能没有出现。这是教学采样器的明确行为，不是工业 DataLoader 的唯一正确策略。

### 3.4 CPU 数据与 GPU Batch

固定 dataset 保存在 CPU，只把当前 batch 搬到 GPU：

```text
dataset[CPU]
→ index selected rows
→ batch.to(cuda)
→ model
```

在大规模训练中，数据通常来自磁盘、多进程 DataLoader、pinned memory 和预取队列。今天的实现只核对最小边界，不代表生产数据管线。

## 4. 一个 Training Step

标准顺序：

```python
logits, _ = model(input_ids)
loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

### 4.1 Forward

Forward 读取当前参数，产生 logits 和 loss。它会建立 Autograd 计算图，但尚未计算参数梯度。

### 4.2 zero_grad

PyTorch 默认将多次 backward 的梯度累加到 `parameter.grad`。如果每个独立 optimizer step 前不清理：

```text
step 1 grad = g1
step 2 grad = g1 + g2
step 3 grad = g1 + g2 + g3
```

今天每个 batch 就是一个独立 step，所以每轮都清理旧梯度。

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 将 `.grad` 设为 `None`，而不是将已有梯度 Tensor 逐元素填 0，通常可减少写内存工作。它也使“本轮没有产生梯度”和“本轮梯度数值恰好为 0”更容易区分。

### 4.3 Backward

```python
loss.backward()
```

它沿计算图计算并累积梯度：

```text
parameter.grad = ∂loss / ∂parameter
```

它不修改 `parameter.data`。

### 4.4 Optimizer Step

```python
optimizer.step()
```

它读取 `.grad` 和 optimizer state，真正更新参数。下一次 forward 因此使用不同权重。

## 5. 从 SGD 到 AdamW

### 5.1 SGD 的最小理解

最基本的梯度下降：

```text
θ_t = θ_(t-1) - learning_rate × gradient_t
```

同一个学习率直接作用于所有参数。

### 5.2 Adam 的一阶状态

Adam 维护梯度的指数移动平均：

```text
m_t = beta1 × m_(t-1) + (1-beta1) × g_t
```

`m_t` 可以理解为平滑后的近期梯度方向。PyTorch optimizer state 中常记为：

```text
exp_avg
```

### 5.3 Adam 的二阶状态

Adam 同时维护梯度平方的移动平均：

```text
v_t = beta2 × v_(t-1) + (1-beta2) × g_t²
```

PyTorch 中常记为：

```text
exp_avg_sq
```

它反映各参数位置的近期梯度尺度。Adam 用它对更新量做逐参数自适应缩放。

### 5.4 AdamW 的 Weight Decay

AdamW 将 weight decay 作为与梯度更新解耦的参数收缩：

```text
θ ← θ - learning_rate × weight_decay × θ
```

再结合 Adam 的自适应梯度更新。这与把 L2 正则项简单混入 Adam 梯度不是完全相同的更新机制。

今天配置：

```text
learning_rate = 3e-3
weight_decay = 0.01
```

这是为了让极小模型快速 overfit 的教学配置，不是对大模型训练的通用推荐。

## 6. Optimizer State 为什么占显存

对每个可训练参数，AdamW 通常需要：

```text
parameter
gradient
exp_avg
exp_avg_sq
```

如果它们都用 FP32，仅这四项逻辑上就是：

```text
4 bytes + 4 bytes + 4 bytes + 4 bytes = 16 bytes/parameter
```

混合精度训练还可能存在 FP32 master weights，而 activation memory 又是另一类成本。所以“模型 FP16 权重体积”不能直接当成训练总显存。

今天代码会直接检查一个参数对应的：

```text
step
exp_avg
exp_avg_sq
```

存在且状态 Tensor shape 与参数相同。

## 7. 为什么要极小数据 Overfit

对 8 条固定序列，小模型应该有足够容量记住它们。如果训练闭环正确，应观察到：

```text
full-dataset loss 显著下降
teacher-forced token accuracy 达到 100%
```

如果连极小数据都无法 overfit，应优先怀疑：

- labels 错位；
- causal mask 错误；
- 参数没有加入 optimizer；
- 遗忘 backward 或 step；
- 梯度在错误时机被清空；
- 学习率不合适；
- 某些应训参数被 detach 或冻结；
- 评估使用了不同数据或错误口径。

但 overfit 成功只证明：

> 对这个极小数据，当前模型、loss、梯度与 optimizer 可以形成可学习闭环。

它不证明：

- 对未见数据能泛化；
- 真实语言模型语义正确；
- 训练配置适合大规模；
- 没有数据泄漏；
- 训练性能高效。

## 8. Training Loss 与 Evaluation Loss

循环中打印的 training loss 来自当前随机 batch：

```text
sampled batch loss
```

不同 step 采样的 sequences 不同，所以 loss 可能短期波动，不要要求每个 step 严格单调下降。

验收使用完整固定数据集：

```text
same 8 sequences before training
vs.
same 8 sequences after training
```

同时固定：

- 模型为 `eval()`；
- 不记录梯度；
- 数据和口径不变；
- loss 对全部 40 个位置求平均。

这才能把 initial/final loss 作为可比较证据。

## 9. Teacher-Forced Token Accuracy

评估时：

```python
predictions = logits.argmax(dim=-1)
accuracy = (predictions == labels).float().mean()
```

这里每个位置都接收真实历史前缀，所以称为 teacher forcing。

例如评估“我喜欢学习”时，预测“学习”的位置输入中使用的是真实“我喜欢”，而不是模型自己前一步猜测的 token。

所以 100% teacher-forced accuracy 不等于自回归生成也必然 100%。自回归生成会把模型自己的输出作为下一步输入，早期错误可能累积。生成和 KV Cache 将在 Day 8～9 实现。

## 10. 循环状态与真相源

运行中有三类核心状态：

```text
model parameters
optimizer state
current step
```

它们当前只存在进程内存/GPU 显存中。程序退出后，训练结果消失，下次运行从新的随机初始化开始。

这是 Day 5 的明确边界，也是 Day 6 需要 checkpoint 的原因：

```text
RAM/VRAM runtime state
→ serialize checkpoint
→ durable disk state
→ restart
→ restore model/optimizer/step
```

## 11. 最小 GPU 实验

运行：

```bash
uv run python exercises/day05/overfit_training_loop.py
```

程序直接复用 Day 4 的 `Config`、`TinyDecoderLM` 和 shifted-label 函数，不复制第二套模型实现。

它会验证：

1. 8 条固定 token sequences 可稳定构造；
2. 每步随机采样 4 条 sequence；
3. CUDA forward、cross-entropy、backward 和 AdamW step 重复 300 次；
4. 训练前后对同一完整 Dataset 求 loss 和 accuracy；
5. final loss 小于 initial loss 的 10%；
6. final loss 小于 0.05；
7. teacher-forced token accuracy 达到 100%；
8. AdamW `exp_avg` 和 `exp_avg_sq` 状态存在且 shape 正确。

## 12. 常见误解与边界

- Dataset 是全部数据，batch 是某一步使用的子集，example 是一条样本。
- `zero_grad` 清理梯度，不清理 optimizer state，也不重置模型权重。
- `backward` 计算梯度，`step` 更新权重，两者不能互相替代。
- AdamW 的状态属于 optimizer，不存在于 model `state_dict` 里。
- 单个随机 batch loss 受采样影响，不能独立作为完整训练结论。
- 当前 accuracy 是 teacher-forced token accuracy，不是自回归生成准确率。
- Overfit 小数据证明机制可学习，不证明泛化、性能或真实语言能力。
- 当前训练状态只在内存中，中断后不能续跑。
- 训练数据是人工 token IDs，没有 tokenizer、文件解码、shuffle epoch、padding 或 ignore index。
- 固定 seed 改善可复现性，但 GPU 上的所有 kernel 在所有环境中都 bitwise deterministic 不是本次已证事实。

## 13. 手算与理解练习

### 练习 1：数据边界

给定 100 条完整 sequences，每条长度 17，batch size 为 8：

1. Dataset shape 是什么？
2. shifted input/labels shape 是什么？
3. 一个 batch 提供多少 next-token positions？
4. 如果有 padding，哪些位置不应计入 loss？

### 练习 2：梯度累积

1. 不调用 `zero_grad` 连续 backward 两次，`.grad` 变成什么？
2. `set_to_none=True` 与把 grad Tensor 填 0 在表达上有什么区别？
3. 为什么 Day 6 的 gradient accumulation 反而会故意跨 micro-batches 保留梯度？

### 练习 3：AdamW State

某参数 Tensor shape 为 `[384,384]`：

1. gradient shape 是什么？
2. `exp_avg` 和 `exp_avg_sq` shape 是什么？
3. 若都是 FP32，参数、梯度和两份 state 逻辑上占多少 MiB？
4. 为什么这还不是训练峰值显存？

### 练习 4：验收证据

1. 为什么不应将 step 1 和 step 300 的随机 batch loss 直接当成唯一证据？
2. 为什么 full-dataset evaluation 需要使用同一数据和口径？
3. 100% teacher-forced accuracy 不能证明什么？
4. 如果 loss 下降但 accuracy 未达 100%，这两个信号是否矛盾？

## 14. 面试口述

### 14.1 30 秒目标

训练循环每步从 Dataset 采样 batch，将完整 token sequence 右移构造 input_ids 和 labels，Forward 产生 logits，Cross-Entropy 产生 loss，zero_grad 清理上轮梯度，backward 计算新梯度，AdamW 根据梯度一阶/二阶状态和 weight decay 更新参数。极小数据 overfit 是训练闭环的正确性测试，不是泛化证据。

### 14.2 两分钟目标

需要说清：

1. Dataset、example、batch 和 token position 的层级；
2. CPU dataset 到 GPU batch 的边界；
3. forward、loss、zero_grad、backward、step 的严格顺序；
4. AdamW 的 `exp_avg`、`exp_avg_sq` 和 weight decay；
5. sampled training loss 与 full-dataset evaluation loss 的区别；
6. tiny overfit 能证明和不能证明什么。

### 14.3 三道口述题

1. 一个标准 Transformer training step 的完整数据流是什么？
2. AdamW 为什么比权重本体多需要两份主要状态？
3. 为什么极小数据 overfit 是必要的正确性 Gate？

## 15. 当日验收

1. 画出 Dataset 到 AdamW step 的完整数据流。
2. 不看讲义写出一个 training step 的五个主要阶段。
3. 说明 `zero_grad(set_to_none=True)` 的作用和边界。
4. 说明 AdamW 两份主要 state 的含义与 shape。
5. 运行 Day 5 脚本，核对 initial/final full-dataset loss、accuracy 和 optimizer state。
6. 解释为什么训练 batch loss 可以波动，而验收仍然成立。
7. 明确说出 tiny overfit 不是泛化证据。
8. 记录今天最不确定的一个问题。

只有同时满足：

- 能区分 Dataset、batch 和 example；
- 能区分 `.grad`、optimizer state 和 parameter；
- 能说明 backward 不会自动更新权重；
- 能解释梯度为什么默认累积；
- 能用同一完整数据集证明 loss 显著下降；
- teacher-forced token accuracy 达到 100%；
- AdamW 一阶/二阶状态已直接观测；
- 能说出至少一个尚未验证的训练边界；

Day 5 才算通过。
