# Day 6：梯度累积与可恢复 Checkpoint

## 1. 今日核心问题

今天解决两个连续的问题：

1. 一个理想 batch 放不进 GPU 时，如何拆成多个 micro-batch，同时保持一次参数更新的数学含义？
2. 训练在中途退出后，如何从文件恢复模型、AdamW、训练步数和采样位置，而不是重新开始？

完整数据流是：

```text
多个 micro-batches
    ↓ forward / loss / backward，暂不清 grad
累积得到一个 effective batch 的平均梯度
    ↓ optimizer.step()
模型参数和 AdamW state 更新
    ↓ 保存 checkpoint
进程退出
    ↓ 新建 model / optimizer / generator
加载 checkpoint
    ↓ 从 completed_steps 继续训练
```

今天的代码直接复用 Day 4 的 Decoder LM 和 Day 5 的 Dataset，不复制模型实现。

---

## 2. 前置知识与术语

### 2.1 Micro-batch

**Micro** 的英文核心含义是“微小的”。Micro-batch 是一次真正送进模型、参与一次 forward/backward 的小批数据。

如果一次只能装下 2 条 sequence：

```text
micro_batch_size = 2
```

那么每次 forward 的输入第一维 (B) 就是 2。

### 2.2 Accumulation steps

**Accumulation** 表示“累积”。`accumulation_steps` 表示多少次 micro-batch backward 以后，才执行一次参数更新。

例如：

```text
micro_batch_size   = 2
accumulation_steps = 4
```

等效 batch size 为：

\[
B_{effective}=B_{micro}\times N_{accumulation}=2\times4=8
\]

这里：

- (B_{effective})：一次 optimizer update 汇总的样本数量；
- (B_{micro})：一次 forward/backward 使用的样本数量；
- (N_{accumulation})：累积次数。

### 2.3 Checkpoint

**Checkpoint** 的英文原意是“检查点”。训练中指某一时刻可用于恢复运行的持久化快照。

Checkpoint 不等于只有模型权重。完整续跑至少需要回答：

- 模型现在是什么参数？
- AdamW 的 `exp_avg`、`exp_avg_sq` 和 step 是什么？
- 已经完成多少 optimizer steps？
- 下一批数据应该从哪里采样？
- PyTorch CPU/CUDA 随机状态是什么？

---

## 3. 从直觉到机制

### 3.1 为什么需要梯度累积

假设希望用 8 条 sequence 计算一次平均梯度，但 GPU 一次只能放下 4 条。

可以拆成：

```text
micro-batch 1：4 条 → backward → grad 暂存
micro-batch 2：4 条 → backward → grad 继续相加
                                  ↓
                         optimizer.step()
```

参数在两个 micro-batch 之间没有更新，所以两个梯度都针对同一组参数计算。

### 3.2 为什么 loss 要除以累积次数

PyTorch 的 Cross-Entropy 默认对一个 micro-batch 中的 token positions 求平均。

设两个等大的 micro-batch 平均 loss 分别为 (L_1,L_2)，理想大 batch 的平均 loss 是：

\[
L_{large}=\frac{L_1+L_2}{2}
\]

所以梯度为：

\[
\nabla L_{large}=\frac{\nabla L_1+\nabla L_2}{2}
\]

代码必须写成：

```python
(micro_loss / accumulation_steps).backward()
```

如果不除以 2，最终 `.grad` 是两份平均梯度之和，尺度约为目标值的 2 倍。

### 3.3 `zero_grad()` 和 `step()` 放在哪里

正确边界：

```python
optimizer.zero_grad(set_to_none=True)

for micro_batch in micro_batches:
    loss = compute_loss(micro_batch)
    (loss / accumulation_steps).backward()

optimizer.step()
```

也就是：

- 一个 accumulation window 开始前清一次梯度；
- window 内不清梯度；
- window 结束后更新一次参数。

如果在每个 micro-batch 前调用 `zero_grad()`，前面 micro-batch 的梯度会被删除，累积失效。

如果每个 micro-batch 都调用 `step()`，那就是多个小 batch update，不再等价于一个大 batch update。

### 3.4 等价成立的边界

“梯度累积等价于大 batch”不是无条件事实。本日实验满足：

- micro-batches 大小相同；
- token position 数量相同；
- loss 使用相同的 mean reduction；
- 累积期间参数不更新；
- 模型没有 dropout；
- 没有依赖 batch 统计的 BatchNorm；
- 样本顺序和数值计算路径可比较。

如果最后一个 micro-batch 更小，简单除以固定 accumulation steps 可能不能得到严格的按 token 平均。此时应按有效 token 数量加权。

---

## 4. 极小手算例子

假设模型只有一个参数 θ，两个等大 micro-batch 产生：

\[
g_1=\frac{\partial L_1}{\partial\theta}=2
\]

\[
g_2=\frac{\partial L_2}{\partial\theta}=4
\]

理想大 batch 的平均梯度是：

\[
g_{large}=\frac{2+4}{2}=3
\]

梯度累积时，每份 loss 先除以 2：

```text
第一次 backward：grad = 2 / 2 = 1
第二次 backward：grad = 1 + 4 / 2 = 3
```

最终 `.grad=3`，与大 batch 一致。

如果忘记除以 2：

```text
grad = 2 + 4 = 6
```

更新量变成预期的两倍。

---

## 5. 正式模型配置

Day 6 使用：

```text
Vocabulary size V = 32
Hidden size H     = 32
Attention heads   = 4
Head dimension D  = 8
FFN size I        = 64
Decoder layers L  = 1
Sequence length   = 6（shift 后为 5）
```

梯度累积等价性实验：

```text
large batch size      = 8
micro-batch size      = 4
accumulation steps    = 2
effective batch size  = 8
```

断点续跑实验：

```text
total optimizer steps = 40
interruption step     = 17
micro-batch size      = 2
accumulation steps    = 2
```

这里的 step 始终表示 `optimizer.step()` 次数，不表示 forward 次数。

一次 optimizer step 有 2 次 forward/backward，因此 40 个 optimizer steps 对应 80 个 micro-steps。

---

## 6. 完整数据流与 Shape

### 6.1 一个 micro-batch

完整 sequence Tensor：

\[
tokens[B_{micro},6]
\]

其中本实验 (B_{micro}=2)。右移构造：

\[
input\_ids[2,5]=tokens[:, :-1]
\]

\[
labels[2,5]=tokens[:, 1:]
\]

模型输出：

\[
logits[2,5,32]
\]

Cross-Entropy 将逻辑位置展平：

```text
logits：[2, 5, 32] → [10, 32]
labels：[2, 5]     → [10]
```

每个 micro-batch 提供 10 个 next-token predictions。

### 6.2 一个 accumulation window

```text
micro-step 1
  input [2,5] → logits [2,5,32] → scalar loss → backward
                                                   ↓
                                             parameter.grad
                                                   +
micro-step 2                                      ↓
  input [2,5] → logits [2,5,32] → scalar loss → backward
                                                   ↓
                                          optimizer.step()
```

在第二次 backward 完成前，参数 shape 不变，`.grad` 也与参数同 shape；变化的是 `.grad` 中累积的数值。

### 6.3 Checkpoint 的数据边界

本日 checkpoint schema：

```text
model_state_dict
optimizer_state_dict
completed_steps
sampling_generator_state
torch_rng_state
cuda_rng_state_all
```

所有权和恢复用途：

- `model_state_dict`：模型参数及注册 buffer；
- `optimizer_state_dict`：AdamW state 与 parameter groups；
- `completed_steps`：下一次训练从哪一个 optimizer step 开始；
- `sampling_generator_state`：下一次应该采到哪个 batch；本日采样器属于 CPU，加载时该状态必须回到 CPU；
- `torch_rng_state`：PyTorch CPU 随机序列位置；
- `cuda_rng_state_all`：各 CUDA 设备随机序列位置；PyTorch 的恢复接口同样接收 CPU ByteTensor 状态描述。

Checkpoint 文件是进程重启后的持久化真相源；Python 变量只是运行时状态。

### 6.4 恢复顺序

```text
重新创建相同 Config 的 model
重新创建相同类型的 AdamW
创建 sampling generator
        ↓
加载 checkpoint 文件并验证字段
        ↓
model.load_state_dict(...)
optimizer.load_state_dict(...)
generator.set_state(...)
恢复 CPU/CUDA RNG state
        ↓
从 completed_steps 继续
```

必须先创建 optimizer，再加载其 state，因为 optimizer state 中的参数编号需要映射回当前 optimizer 管理的参数。

---

## 7. 参数、内存与计算成本

### 7.1 梯度累积降低什么

它降低一次 forward/backward 需要同时保存的 activation 数量。

粗略看：

\[
activation\ memory\propto B_{micro}\times S\times H\times L
\]

把 batch 8 拆成两个 batch 4，单次 activation 峰值通常会下降。

### 7.2 梯度累积不降低什么

它通常不会降低：

- 模型参数显存；
- 参数梯度显存；
- AdamW 的 `exp_avg`；
- AdamW 的 `exp_avg_sq`。

这些 Tensor 的 shape 由模型参数决定，不由 micro-batch size 决定。

### 7.3 时间代价

对于相同的 effective batch，拆分前后处理的 sequence 总数相同，每条 sequence
也仍然只参与一次 forward 和一次 backward，因此理论上的主要样本计算量基本相同；
optimizer update 也都只有一次。变化的是执行组织方式：一个大 batch 的一次
forward/backward 调用，被拆成多次 micro-batch forward/backward 调用。这通常会
增加 Python 调度和 CUDA kernel launch 边界，并可能因单次矩阵规模变小而降低 GPU
利用率，所以实际运行时间仍可能增加。

因此梯度累积主要是**显存容量与吞吐量之间的权衡**，不是免费扩大 batch。

### 7.4 Checkpoint 大小

训练 checkpoint 通常比纯模型权重大，因为还包含 AdamW 两份主要状态。若参数和 optimizer state 都是 FP32，仅参数、`exp_avg`、`exp_avg_sq` 就接近三份参数规模；实际还可能包括其他状态和序列化开销。

本日程序直接读取临时 checkpoint 的实际字节数。这是框架序列化后的文件大小，不等同于训练峰值显存。

---

## 8. 最小代码验证

运行：

```bash
uv run python exercises/day06/gradient_accumulation_checkpoint.py
```

程序执行四项行为验证。

### 8.1 大 batch 与累积更新对比

两个模型从完全相同的参数开始：

- 模型 A：8 条数据一次 forward/backward；
- 模型 B：两次 4 条数据，每份 loss 除以 2 后累积；
- 两者各执行一次 AdamW step；
- 比较所有参数。

### 8.2 连续训练 baseline

使用固定采样 generator 连续执行 40 个 optimizer steps，得到最终参数和 loss。

### 8.3 中断后续跑

另一条训练路径使用相同初始参数与采样状态：

```text
训练 17 steps
    ↓
保存 checkpoint
    ↓
新建 model、optimizer、generator
    ↓
加载 checkpoint
    ↓
继续到 40 steps
```

最终逐参数比较连续路径和恢复路径。

### 8.4 非法文件验证

程序还会构造一个缺少必要字段的 checkpoint，确认加载函数明确拒绝它，而不是带着不完整状态继续训练。

实验 checkpoint 使用独立临时目录，程序退出后自动清理，不覆盖项目已有数据。

---

## 9. 常见误解与边界

### 9.1 Effective batch 变大，不等于一次显存中真的有这么多样本

GPU 同时处理的仍是 micro-batch。Effective batch 描述的是一次参数更新汇总了多少样本的梯度。

### 9.2 `optimizer.step()` 的次数没有增加

累积 4 次 backward，只执行一次 step。因此 scheduler 如果按 optimizer step 更新，也通常应在 accumulation boundary 更新，而不是每个 micro-step 更新。

### 9.3 `model.state_dict()` 不是完整训练 checkpoint

只保存模型可以用于推理，也可以重新开始优化；但不能精确恢复 AdamW 的动量历史和训练进度。

### 9.4 恢复 optimizer 仍不一定足够

真实训练还可能需要保存：

- learning-rate scheduler；
- AMP GradScaler；
- epoch、dataloader sampler 状态；
- 分布式 rank/shard 信息；
- tokenizer/config 版本；
- 数据集版本与代码版本。

本日没有这些组件，所以不虚构对应状态。

### 9.5 `torch.save` 不是任意不可信文件的安全解析器

本日使用 `weights_only=True` 限制加载范围，但 checkpoint 仍应来自可信训练流程，并验证 schema。不要随意加载互联网来源的未知文件。

### 9.6 原子替换解决的是“半写文件”风险，不解决所有并发问题

程序先写 `.tmp`，成功后使用 `os.replace` 替换目标文件。这减少进程在写入中途退出而留下半个正式 checkpoint 的风险。

它没有实现多进程写锁、远程对象存储事务或 checkpoint 版本保留。本日单进程实验不需要扩展到这些能力。

### 9.7 完全一致依赖可复现边界

本日模型无 dropout，采样 generator 与 PyTorch RNG 均被恢复，并在同一硬件/软件环境中比较，所以能够验证精确续跑。

不同 GPU、不同 PyTorch/CUDA、非确定性 kernel 或数据 pipeline 外部变化时，不应默认 bitwise identical。

---

## 10. 手算练习

### 练习 1：有效 batch

给定：

```text
micro_batch_size = 3
accumulation_steps = 8
sequence_length after shift = 127
```

回答：

1. effective batch size 是多少？
2. 一个 micro-step 有多少 token prediction positions？
3. 一个 optimizer step 汇总多少 positions？
4. 每个 micro-batch loss 都是 mean 时，应如何缩放？

### 练习 2：边界位置

判断下列调用应位于 micro-step 内还是 accumulation window 边界：

1. forward；
2. loss 计算；
3. backward；
4. `zero_grad`；
5. `optimizer.step`。

### 练习 3：Checkpoint

如果只保存：

```python
torch.save(model.state_dict(), path)
```

回答：

1. 模型参数能否恢复？
2. AdamW 的一阶/二阶矩能否恢复？
3. 下一次随机 batch 能否保证与未中断训练一致？
4. 已完成 step 能否从该文件直接确认？

### 练习 4：不等大的 micro-batches

两个 micro-batches 分别包含 10 和 5 个有效 token positions，各自 loss 都是 mean。为什么简单计算：

\[
\frac{L_1}{2}+\frac{L_2}{2}
\]

不等价于对全部 15 个 positions 求平均？正确权重应该分别是多少？

---

## 11. 面试口述

### 11.1 30 秒版本

梯度累积把一个大 batch 拆成多个 micro-batch，在一个 accumulation window 开始前清梯度，每个 micro-batch 的缩放 loss 分别 backward，最后只执行一次 optimizer step。它降低 activation 峰值，但不减少参数、梯度和 AdamW state。可恢复 checkpoint 除模型外还要保存 optimizer、训练进度和随机/采样状态，否则只能恢复权重，不能精确续跑。

### 11.2 两分钟版本

需要说清：

1. micro-batch size、accumulation steps 和 effective batch size 的关系；
2. 为什么 mean loss 要除以 accumulation steps；
3. `zero_grad`、`backward`、`step` 的正确边界；
4. 梯度累积降低 activation 峰值但增加执行次数；
5. 模型权重与完整训练状态的区别；
6. checkpoint 中 model、optimizer、step 和 RNG/sampler state 的用途；
7. 连续训练与中断恢复对比为什么是续跑正确性的强证据。

### 11.3 三道口述题

1. 梯度累积为什么能近似或等价模拟大 batch？哪些条件会破坏严格等价？
2. 为什么只保存 `model.state_dict()` 不能叫精确续训？
3. 梯度累积节省哪些显存，又不节省哪些显存？

---

## 12. 当日验收

1. 不看讲义画出一个 accumulation window。
2. 能手算两个 micro-batch 的平均梯度。
3. 能解释为什么 loss 要除以 accumulation steps。
4. 能正确放置 `zero_grad`、`backward` 和 `optimizer.step`。
5. 能列出本日 checkpoint 的六个字段及其所有者。
6. 能解释 model state 与 optimizer state 的区别。
7. 运行 Day 6 程序并看到四项 PASS。
8. 能说出严格大 batch 等价和 bitwise resume 的至少一个失效条件。
9. 记录一个尚未确认的问题，例如分布式 checkpoint 如何分片。

满足以上条件后，Day 6 才算通过。
