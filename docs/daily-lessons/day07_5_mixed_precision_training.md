# Day 7.5：混合精度训练、Autocast 与 GradScaler

## 1. 今日核心问题

今天回答：

> 为什么训练不能简单地把所有 Tensor 从 FP32 改成 FP16？PyTorch AMP 如何让不同算子使用不同精度，GradScaler 又如何保护很小的梯度？

核心数据流：

```text
FP32 模型参数
    ↓ autocast 根据算子选择计算 dtype
FP16 矩阵乘法 + FP32 敏感算子
    ↓ 得到 FP32 scalar loss
GradScaler 放大 loss
    ↓ backward 得到放大的 FP32 gradient buffer
unscale + 检查 inf/NaN
    ↓ 没有溢出才执行 AdamW step
更新 FP32 参数和 FP32 optimizer state
```

今天只比较 FP32 baseline 与 CUDA FP16 AMP。BF16 必须先通过当前 GPU 的原生支持检查；不支持时明确跳过。

---

## 2. 前置知识与术语

### 2.1 Precision

**Precision** 的英文核心含义是“精度”。浮点格式需要同时表达：

- 正负号；
- 数值的大致数量级；
- 这个数量级内的有效数字。

### 2.2 FP32、FP16 与 BF16

**FP** 是 floating point，表示浮点数。

```text
FP32：32 bit，范围和有效精度都较高
FP16：16 bit，指数范围明显小于 FP32，尾数也更短
BF16：16 bit，指数位与 FP32 相同，范围接近 FP32，但尾数更短
```

FP16 与 BF16 都占 2 bytes，但数值能力不同，不能因为字节数相同就把它们视为同一种格式。

### 2.3 AMP

**AMP** 全称是 Automatic Mixed Precision，中文是“自动混合精度”。

“混合”意味着不同操作可以使用不同 dtype，而不是把整个模型全部永久转换为 FP16。

### 2.4 Autocast

**Autocast** 可以理解为“自动类型转换”。在它的上下文中，PyTorch 根据算子的数值稳定性和性能策略选择 dtype。

例如 CUDA AMP 中：

- Linear、matrix multiplication 等通常适合 FP16；
- Softmax、Cross-Entropy、归约等敏感操作通常保持或提升到 FP32。

具体算子策略属于 PyTorch 实现契约，应以当前版本的官方 AMP operator reference 为准。

### 2.5 GradScaler

**Grad** 是 gradient，梯度；**Scaler** 是缩放器。

GradScaler 动态维护一个 scale，用于先放大 loss，从而把很小的梯度搬到 FP16 更容易表达的范围内。

---

## 3. 从直觉到机制

### 3.1 为什么希望使用低精度

一个 FP32 元素占 4 bytes，FP16/BF16 元素占 2 bytes。低精度可能带来：

- 更少的 activation 存储；
- 更少的显存读写字节；
- 在支持的 Tensor Core 和 kernel 上获得更高计算吞吐；
- 同样显存下容纳更大 batch 或模型。

但“理论字节减半”不等于“训练总显存减半”，也不保证运行时间减半。

### 3.2 为什么不能全部直接改成 FP16

FP16 的数值范围和精度有限，可能发生：

- overflow：数值太大，变成 `inf`；
- underflow：非零小数太小，变成 0；
- rounding：两个相近数值被舍入成相同表示；
- reduction error：大量数值求和时误差积累。

训练同时包含大激活、小梯度、指数、除法、归约和概率计算，因此不同算子的稳定需求不同。

### 3.3 Autocast 不会永久改变参数 dtype

典型 AMP 代码：

```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    logits, _ = model(input_ids)
    loss = cross_entropy(logits, labels)
```

模型参数仍可以是 FP32。Autocast 在算子执行边界选择输入或计算 dtype。

本日实验直接断言：

```text
parameter dtype       = FP32
logits dtype          = FP16
cross-entropy dtype   = FP32
```

这就是“混合精度”的直接证据。

### 3.4 Loss scaling 为什么有效

假设原始 loss 是 $L$，scale 是 $s$。GradScaler 计算：

$$
L_{scaled}=sL
$$

根据求导的线性关系：

$$
\frac{\partial L_{scaled}}{\partial\theta}
=s\frac{\partial L}{\partial\theta}
$$

也就是所有梯度都被放大 $s$ 倍。一个很小的梯度例如：

```text
原始梯度：0.00000001
scale：   65536
放大后：  0.00065536
```

放大后的数值更不容易在低精度计算中消失。

在 optimizer step 前必须再除回去：

$$
g_{original}=g_{scaled}/s
$$

因此 loss scaling 的目标不是改变最终数学梯度，而是保护中间表达。

### 3.5 动态 scale

GradScaler 会检查梯度是否包含 `inf` 或 `NaN`：

```text
梯度有限
→ 执行 optimizer.step
→ 连续稳定一段时间后可以增大 scale

梯度出现 inf/NaN
→ 跳过本次 optimizer.step
→ 减小 scale
```

所以 scale 是运行时状态，不是永远固定的常数。

---

## 4. 极小手算例子

假设参数 $θ=2$，原始 loss 对参数的梯度是：

```text
g = 0.00001
```

选择：

```text
scale = 1024
```

缩放后的 backward 梯度：

```text
scaled gradient
= 0.00001 × 1024
= 0.01024
```

在 step 前 unscale：

```text
unscaled gradient
= 0.01024 / 1024
= 0.00001
```

假设学习率为 0.1，忽略 AdamW moments：

```text
新参数
= 2 - 0.1 × 0.00001
= 1.999999
```

最终更新仍然使用原始梯度。Scale 只保护 backward 中间过程。

---

## 5. 正式实验配置

FP32 与 FP16 AMP 使用完全相同的：

```text
Vocabulary size       512
Hidden size           128
Attention heads       8
FFN intermediate      384
Decoder layers        2
Batch size            16
Input sequence length 128
Warmup steps          5
Measured steps        20
Optimizer             AdamW
Learning rate         1e-3
```

两条路径从同一份 FP32 初始参数开始，并使用同一个固定 token batch。

唯一主要变量是：

```text
FP32 baseline：autocast disabled，GradScaler disabled
FP16 AMP：     autocast FP16，GradScaler enabled
```

这不是收敛 benchmark。25 个 step 只用于检查两种训练路径是否能稳定降低同一 batch 的 loss，并测量当前小配置的时间和显存。

---

## 6. 完整数据流与 Shape、Dtype

### 6.1 输入

完整 token sequences：

```text
tokens      [16,129]  int64
input_ids   [16,128]  int64
labels      [16,128]  int64
```

token IDs 必须是整数索引，不会因为 AMP 变成 FP16。

### 6.2 模型参数

```text
Embedding weight       FP32
Linear weights         FP32
RMSNorm gamma          FP32
AdamW exp_avg           FP32
AdamW exp_avg_sq        FP32
```

本实验没有调用 `model.half()`。

### 6.3 Autocast forward

逻辑 shape 不因 dtype 改变：

```text
hidden                 [16,128,128]
Q/K/V                  [16,8,128,16]
attention scores       [16,8,128,128]
logits                 [16,128,512]
```

AMP 改变的是部分操作的计算和输出 dtype，不改变 Tensor 的逻辑 shape。

本日当前 PyTorch 策略下，Linear/Matmul 可产生 FP16 结果，而 Cross-Entropy 在 autocast 区域中使用 FP32 策略。因此预期：

```text
FP16 AMP logits        FP16
FP16 AMP scalar loss   FP32
```

### 6.4 Backward 与 GradScaler

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()
```

顺序含义：

1. `scale(loss)`：构造放大的 loss；
2. `backward()`：计算放大的梯度；
3. `unscale_()`：恢复原始梯度尺度，便于检查或裁剪；
4. `step()`：梯度有限时才调用 optimizer step；
5. `update()`：根据本轮结果更新 scale。

本日调用 `unscale_()` 是为了直接检查恢复尺度后的梯度 dtype 和有限性。若不需要检查或裁剪，`scaler.step()` 会在内部处理 unscale。

---

## 7. 参数、内存与计算成本

### 7.1 哪些内存可能下降

AMP 主要可能减少：

- 低精度算子的输出 activation；
- 为 backward 保存的部分低精度中间 Tensor；
- 某些临时计算结果；
- 相关显存带宽流量。

### 7.2 哪些内存没有减半

本实验仍为 FP32 的包括：

- 模型参数；
- 参数 `.grad`；
- AdamW `exp_avg`；
- AdamW `exp_avg_sq`；
- Cross-Entropy 等敏感算子的部分输出。

所以：

```text
FP16 元素占 2 bytes
```

不能推出：

```text
整个训练峰值显存一定下降 50%
```

### 7.3 为什么不一定更快

低精度真实加速依赖：

- GPU 是否有对应低精度 Tensor Core；
- 矩阵维度是否适合高效 kernel；
- 工作量是否足以让 GPU 饱和；
- autocast 转换和 GradScaler 开销占比；
- 是否受计算、显存带宽、CPU 调度或 kernel launch 限制。

本日必须保留“FP16 没有更快”的可能结果。一次小模型 benchmark 不能代表大型 Transformer。

### 7.4 正确计时

CUDA 异步执行，因此计时边界是：

```python
torch.cuda.synchronize()
start = time.perf_counter()

# measured training steps

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

并先执行 warmup，减少首次 kernel 初始化和 allocator 建立对结果的干扰。

### 7.5 显存口径

```python
torch.cuda.reset_peak_memory_stats()
...
torch.cuda.max_memory_allocated()
```

得到的是 PyTorch Tensor allocation 峰值，不是 `nvidia-smi` 所显示的整个进程显存，也不包含全部 CUDA context/driver 成本。

---

## 8. 最小代码验证

运行：

```bash
uv run python exercises/day07_5/mixed_precision_training.py
```

程序会输出：

- PyTorch、CUDA、GPU 和 compute capability；
- native BF16 support；
- FP32 与 FP16 的参数、logits、loss、gradient、optimizer state dtype；
- initial/final loss；
- GradScaler initial/final scale；
- warmup 后平均 step 时间；
- CUDA peak allocated；
- FP16 相对 FP32 的时间比和显存差。

行为验收不是要求 FP16 一定更快，而是要求：

```text
两种模式 loss 都下降
autocast 确实产生 FP16 logits
Cross-Entropy loss 保持 FP32
参数、unscaled gradient、AdamW state 保持 FP32
计时与显存读取使用正确边界
```

本机 RTX 2060、PyTorch 2.13.0+cu130 实测：

```text
compute capability                 (7, 5)
native BF16 support                False（跳过 BF16 benchmark）

FP32 initial / final loss          6.422430 / 3.965009
FP16 initial / final loss          6.422430 / 3.965055

FP32 average measured step         7.473 ms
FP16 average measured step         6.990 ms
FP32 time / FP16 time              1.069x

FP32 CUDA peak allocated           105.67 MiB
FP16 CUDA peak allocated           95.11 MiB
observed peak allocation saved     10.56 MiB

FP16 GradScaler initial/final      65536 / 65536
```

这次短实验中 FP16 AMP 略快且峰值 allocation 较低，但收益幅度有限。计时来自一次
短运行，会受到 GPU 时钟、温度和系统负载影响；它只能证明当前配置的观测结果，
不能直接外推更大模型或其他 GPU。

---

## 9. 常见误解与边界

### 9.1 AMP 不等于 `model.half()`

`model.half()` 会把模型浮点参数和 buffer 转为 FP16；AMP 则在 FP32 参数基础上，根据算子策略选择计算 dtype。两者不能混为一谈。

### 9.2 Autocast 只应包围 forward 和 loss

推荐：

```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    logits = model(input_ids)
    loss = loss_fn(logits, labels)

scaler.scale(loss).backward()
```

Backward 不需要放进 autocast 上下文；它会使用 forward 为各操作记录的 dtype 路径。

### 9.3 GradScaler 不保证永远增大 scale

发现溢出时 scale 会减小，并跳过参数更新。Scale 甚至可能低于 1；不能把默认初始值理解为永久值。

### 9.4 BF16 通常不需要 FP16 式 loss scaling，但仍需实测

BF16 的指数范围接近 FP32，underflow 风险通常低于 FP16，所以常见 BF16 训练不使用 GradScaler。但硬件支持、算子策略和训练稳定性仍需要在目标环境验证。

### 9.5 BF16 API 可用不等于硬件原生加速

本日读取：

```python
torch.cuda.is_bf16_supported(including_emulation=False)
```

只有当前设备报告原生支持时，才有理由进行对应 BF16 硬件 benchmark。不把软件模拟当作原生能力。

### 9.6 梯度累积时不能每个 micro-step 更新 scaler

一个 accumulation window 内，所有 micro-batch 必须使用同一个 scale。应在最后一个 micro-step 后统一 `step()` 和 `update()`，否则累积梯度可能混合不同尺度。

### 9.7 AMP checkpoint 还需要 scaler state

精确续跑时应额外保存：

```python
"scaler_state_dict": scaler.state_dict()
```

恢复时调用：

```python
scaler.load_state_dict(checkpoint["scaler_state_dict"])
```

本日 benchmark 不执行中断恢复；Day 6 的 checkpoint schema 因此没有被静默修改。

### 9.8 当前实验的事实边界

- 使用人工随机 token IDs；
- 使用单卡 RTX 2060；
- 没有 DDP、`torch.compile` 或 activation checkpointing；
- 时间只来自一次短 benchmark，存在运行波动；
- peak allocated 只对应当前模型和 batch；
- loss 下降只证明该固定 batch 上训练过程有效，不证明泛化。

---

## 10. 手算练习

### 练习 1：理论字节数

某 activation shape 为 `[16,128,384]`：

1. 元素数量是多少？
2. FP32 理论大小是多少 MiB？
3. FP16 理论大小是多少 MiB？
4. 为什么整个训练峰值不能直接按这个比例减半？

### 练习 2：Loss scaling

给定：

```text
原始梯度 = 0.000002
scale = 65536
```

1. scaled gradient 是多少？
2. unscale 后是多少？
3. 如果 scaled gradient overflow，GradScaler 应怎样处理 optimizer step 和 scale？

### 练习 3：Dtype 所有权

在本日 FP16 AMP 中，判断以下预期 dtype：

1. token IDs；
2. Linear parameter；
3. logits；
4. Cross-Entropy loss；
5. unscaled `.grad`；
6. AdamW `exp_avg`。

### 练习 4：梯度累积

设 accumulation steps 为 4：

1. `scaler.scale(loss / 4).backward()` 应调用几次？
2. `scaler.step(optimizer)` 应调用几次？
3. `scaler.update()` 应调用几次？
4. 为什么一个 window 内不能改变 scale？

---

## 11. 面试口述

### 11.1 30 秒版本

AMP 不会简单地把所有训练状态改成 FP16，而是由 autocast 让 Linear、Matmul 等适合的算子使用低精度，让 Cross-Entropy 等敏感算子保持 FP32。FP16 梯度可能 underflow，所以 GradScaler 先放大 loss，backward 后再 unscale，并在梯度有限时执行 optimizer step。参数、梯度和 AdamW state 在常见 AMP 路径中仍可保持 FP32，因此显存不会简单减半，速度收益也依赖硬件和 workload。

### 11.2 两分钟版本

需要说清：

1. FP32、FP16、BF16 的范围与有效精度差异；
2. AMP、autocast 和 GradScaler 各自解决什么；
3. 为什么矩阵乘法适合低精度，Cross-Entropy 等操作倾向 FP32；
4. loss scaling 如何保持最终数学梯度不变；
5. overflow 时为何跳过 optimizer step；
6. 参数、activation、gradient 和 optimizer state 的 dtype 不必相同；
7. 为什么 AMP 不保证显存减半或训练加速；
8. 梯度累积和 checkpoint 如何额外管理 scaler。

### 11.3 三道口述题

1. AMP 与直接调用 `model.half()` 有什么根本区别？
2. GradScaler 为什么只对 FP16 训练特别重要，它如何动态调整 scale？
3. 如何设计一个可信的 FP32/FP16 时间与显存对照实验？

---

## 12. 当日验收

1. 能解释 FP16 与 BF16 的范围/精度取舍。
2. 能画出 autocast、scaled backward、unscale、step、update 数据流。
3. 能说明 loss scaling 为什么不改变最终目标梯度。
4. 能列出本实验五类 Tensor 的实际 dtype。
5. 能解释为什么 AMP 不等于全部 FP16。
6. 运行 Day 7.5 程序并记录 FP32/FP16 loss、时间与 peak allocated。
7. 根据实际输出判断 FP16 是否更快，而不是预设结论。
8. 根据设备检查决定是否运行 BF16 benchmark。
9. 能说明 AMP 与梯度累积、checkpoint 的组合边界。
10. 记录一个未确认问题，例如更大模型下的收益是否改变。

只有能够解释 dtype 为什么不同，并能区分理论收益与当前硬件实测结果，Day 7.5 才算通过。

## 官方参考

- [PyTorch Automatic Mixed Precision package](https://docs.pytorch.org/docs/2.13/amp.html)
- [PyTorch Automatic Mixed Precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [PyTorch BF16 support check](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html)
