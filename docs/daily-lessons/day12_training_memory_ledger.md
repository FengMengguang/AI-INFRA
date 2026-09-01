# Day 12：训练显存账本与 PyTorch Allocator 口径

## 1. 今日核心问题

Day 11 观察到 Activation Checkpointing 降低了 saved logical bytes 和 CUDA peak
allocated，但两个数字不相等。这说明“模型参数量 × dtype 字节数”不是完整的
训练显存答案。

今天要回答：一次训练 step 中，参数、梯度、optimizer state、activation、临时
Tensor/workspace 和 allocator reserve 分别是什么，它们何时出现、由谁持有，以及怎样
把可精确求和的理论账本与 PyTorch 运行时证据对齐。

今日验收：

```text
给定参数量、dtype 和 optimizer
→ 手算静态训练状态
→ 画出 forward/backward/step/zero_grad 生命周期
→ 分开解释 logical bytes、allocated、reserved、peak 和 nvidia-smi
→ 不把未分类差额冒充成某一种确定 Tensor
```

## 2. 前置知识与术语

- **Parameter（参数）**：模型长期保存、由 optimizer 更新的 Tensor。
- **Gradient（梯度）**：`parameter.grad`，表示 loss 对参数的导数。
- **Optimizer state**：optimizer 为每个参数维护的状态。AdamW 的主体是一阶动量
  `exp_avg` 和二阶动量 `exp_avg_sq`。
- **FP32 master weight**：某些低精度训练实现额外保留的 FP32 主参数。它不是
  所有 autocast 训练都必然存在的第二份参数。
- **Activation（激活）**：forward 产生、backward 可能需要的中间 Tensor。
- **Temporary Tensor / workspace**：算子在某个短时间段为计算、排序、归约或 kernel
  实现使用的额外空间。不同 backend 可能不同。
- **Allocated**：PyTorch CUDA allocator 当前分配给活 Tensor 的字节。
- **Reserved**：PyTorch allocator 从 CUDA 取得并保留在缓存池中的字节，包含
  allocated 和当前可复用空闲 block。
- **Peak**：一个测量窗口中出现过的最大值，不是窗口结束时的当前值。
- **Logical bytes**：按 `numel × element_size` 求和的逻辑 payload。它可能包含
  view 或对已有 storage 的引用，不自动等于新增物理显存。
- **GB / GiB**：`1 GB = 10^9 bytes`，`1 GiB = 2^30 bytes`。`7B parameters` 中的 `B`
  表示 billion，不是 byte。

## 3. 从直觉到机制

### 3.1 显存像一本分类账本

可以把训练想成一家仓库。参数是长期库存，梯度是本次盘点结果，AdamW moments
是历史记录，activation 是为后续 backward 留下的中间材料，workspace 是操作期间
临时借用的工作台。

allocator 又像仓库租赁方：Tensor 释放后，PyTorch 可能保留已申请的显存 block，
下一次直接复用。所以“当前活 Tensor 变少”不代表 reserved 必须立即下降。

### 3.2 一次 step 中的生命周期

```text
创建模型
→ 参数存在，梯度不存在，Adam state 通常还未懒初始化

forward
→ 产生 logits、saved activations 和临时 Tensor

backward
→ 逐层消费 saved activations，生成 parameter.grad

第一次 optimizer.step()
→ AdamW 为每个参数创建 exp_avg、exp_avg_sq 和 step 状态

zero_grad(set_to_none=True)
→ parameter.grad 引用变为 None，梯度 Tensor 可释放
→ 参数和 Adam state 仍然存在
→ allocator reserved 可能不下降
```

### 3.3 AdamW state 不一定在创建 optimizer 时分配

PyTorch AdamW 通常在第一次 `step()` 时根据真正获得梯度的参数懒初始化
moments。因此：

```text
optimizer = AdamW(model.parameters())
```

命令执行完不能单独证明 Adam state 已占用 `8 bytes/parameter`。必须检查
`optimizer.state` 或在第一次 step 前后对照。

### 3.4 为什么 peak 不能被一个静态公式完全命中

某一时刻的活集合不只取决于“有哪些类别”，还取决于它们的生命期是否重叠。例如
backward 某层正在计算梯度时，后面层的梯度已生成，前面层的 activation 可能还没被
消费，当前 kernel 还可能需要 temporary Tensor。

因此实验中定义：

```text
composite transient peak gap
= peak allocated - backward 结束后的 allocated
```

它只表示测量窗口中曾经多出的复合瞬时用量，不能单独命名为 activation 或
workspace。

## 4. 极小手算例子

设模型只有 1,000 个参数，使用 FP32 参数、FP32 梯度和标准 AdamW moments。

```text
参数：      1,000 × 4 bytes = 4,000 bytes
梯度：      1,000 × 4 bytes = 4,000 bytes
Adam exp_avg:    1,000 × 4 bytes = 4,000 bytes
Adam exp_avg_sq: 1,000 × 4 bytes = 4,000 bytes
---------------------------------------------------
静态主体：                     16,000 bytes
```

也就是每参数 16 bytes。这还没有加 activation、临时量、allocator reserve 和 CUDA
context。AdamW 每个参数 Tensor 还可能有一个很小的 `step` Tensor，它按“参数 Tensor
个数”而不是“标量参数个数”增长。

如果将参数改为 FP16，但仍保留独立 FP32 master weight，且梯度为 FP16：

```text
FP16 parameter:     2 bytes/parameter
FP16 gradient:      2 bytes/parameter
FP32 master weight: 4 bytes/parameter
FP32 Adam m/v:      8 bytes/parameter
---------------------------------------
合计：               16 bytes/parameter
```

但如果模型参数本身长期保持 FP32，autocast 只在算子内使用低精度，就不能再额外
加一份同样大的 master weight。账本必须先核对真实参数所有权。

## 5. 正式实验配置

Day 12 复用 Day 4 的 `TinyDecoderLM`，不实现第二套 Transformer。配置为：

```text
vocabulary size                 2,048
hidden size H                     256
Attention heads                    8
head dimension D_h                32
FFN intermediate size            768
Decoder layers                     4
batch B                            8
sequence S                       128
parameter/gradient dtype        FP32
optimizer                       AdamW
```

`input_ids[B,S] = [8,128]` 中，`B` 是同时训练的序列数，`S` 是每条序列的 token
数。进入模型后 hidden state 为 `X[B,S,H] = [8,128,256]`，`H` 是每个 token
的隐藏特征维度。

实验不启用混合精度和 Activation Checkpointing，以免在第一份完整账本中同时改变
dtype 和 activation 保存策略。这是 FP32 baseline，不是生产 LLM 配置。

## 6. 完整数据流与 Shape/Dtype

### 6.1 创建模型

```text
4,491,520 个标量参数
× 4 bytes/FP32
= 17,966,080 bytes
= 17.134 MiB
```

模型创建后，参数 shape 由 Embedding、Attention projection、SwiGLU projection 和 Norm
共同决定。它们全部是 FP32。此时 `.grad is None`，AdamW state 也为空。

### 6.2 Forward

```text
input_ids[8,128] int64
→ embedding hidden[8,128,256] FP32
→ 4 层 Decoder Block
→ logits[8,128,2048] FP32
→ Cross-Entropy loss[] FP32
```

Attention 中显式产生的 weights shape 为 `[8,8,128,128]`。Day 4 模型会返回每层
weights，因此它们的生命期比一个只返回 logits 的生产模型更长。这会影响峰值和
forward 结束后的 allocated，不能外推为真实 LLM 的固定比例。

### 6.3 Backward

```text
loss.backward()
→ 从 LM Head 向 Embedding 反向传播
→ 逐段使用并释放 autograd 保存的中间状态
→ 为每个可训练参数生成 FP32 .grad
```

当所有参数都有同 shape、同 dtype 梯度时，梯度总容量与参数容量相同：

```text
gradient bytes = 17.134 MiB
```

### 6.4 第一次 AdamW step

AdamW 为每个参数创建两个同 shape 的 FP32 moments：

```text
exp_avg + exp_avg_sq
= 4,491,520 × (4 + 4) bytes
= 34.268 MiB
```

实验中有 40 个参数 Tensor，每个还有一个 FP32 scalar `step`，共 160 bytes。
它是真实状态，但在 MiB 保留三位小数时显示为 `0.000 MiB`。

### 6.5 zero_grad

`optimizer.zero_grad(set_to_none=True)` 将 `.grad` 变为 `None`，本实验中精确梯度账本
从 `17.134 MiB` 变为 0。参数和 AdamW state 仍然存在。

## 7. 参数、内存与计算成本

### 7.1 通用静态公式

设标量参数量为 $P$，参数、梯度、master weight、Adam 一阶和二阶动量的每元素
字节数分别是 $b_p$、$b_g$、$b_m$、$b_1$和 $b_2$。如果这些状态都存在，静态主体为：

$$
M_{static}=P(b_p+b_g+b_m+b_1+b_2)
$$

这里 $M_{static}$ 是字节数。某个实现不存在独立 master weight 时，$b_m$ 对应的整项
应删除，不是自动填 4。

### 7.2 当前 FP32 baseline

```text
parameters           17.134 MiB
gradients            17.134 MiB
Adam moments         34.268 MiB
Adam step metadata      160 bytes
--------------------------------
persistent with grad 68.535 MiB
```

`zero_grad(set_to_none=True)` 后：

```text
parameters + optimizer tensors = 51.402 MiB
```

### 7.3 Activation 不能只用参数量计算

activation 与 batch、sequence、hidden size、层数、算子、dtype 和保存策略有关。例如一份
hidden state `X[B,S,H]` 的逻辑容量是：

$$
M_X=B\times S\times H\times b_a
$$

$b_a$ 是 activation 每元素字节数。但真实 Block 还有 Q/K/V、Attention 中间量、Norm
中间量和 FFN 扩展激活，且 fused kernel 可能不物化某些逻辑 Tensor。不能用一份
hidden state 代表整层 activation。

### 7.4 四个常见显存数字的关系

```text
logical saved payload
→ autograd 保存/引用 Tensor 的 numel × element_size 之和

allocated
→ PyTorch allocator 当前分配给活 Tensor 的存储

reserved
→ PyTorch 已从 CUDA 取得并缓存的 block，通常 >= allocated

nvidia-smi process memory
→ 更接近进程在 driver 层占用，还可包含 CUDA context、library workspace
   和非 PyTorch allocator 分配，不应与 allocated 强制相等
```

## 8. 最小代码验证

实验文件：

```text
exercises/day12/training_memory_ledger.py
```

运行：

```bash
uv run python -m exercises.day12.training_memory_ledger
```

实验做了三种不同强度的核对：

1. 直接遍历参数、`.grad` 和 optimizer state Tensor，精确求和字节数；
2. 用 `saved_tensors_hooks` 记录初始 forward 为 backward 保存/引用的逻辑 payload；
3. 在生命周期关键点读取 `memory_allocated`、`memory_reserved` 和它们的 peak。

### 8.1 当前机器实际输出

2026-08-30 在 PyTorch 2.13.0+cu130、NVIDIA GeForce RTX 2060、compute capability 7.5 上
得到：

```text
parameters:                         17.134 MiB
gradients after backward:           17.134 MiB
Adam exp_avg + exp_avg_sq:          34.268 MiB
optimizer tensor total:             34.268 MiB + 160 bytes
persistent total with grads:        68.535 MiB
persistent after zero_grad:         51.402 MiB

initial-forward saved tensors:             154
initial-forward saved logical payload: 175.158 MiB
composite transient peak gap:           91.965 MiB

after model creation:
  allocated delta:                    17.134 MiB
  reserved delta:                     34.000 MiB

after backward:
  allocated delta:                    76.417 MiB
  peak allocated delta:              168.382 MiB
  reserved delta:                    180.000 MiB

after first AdamW step:
  allocated delta:                   111.685 MiB
  reserved delta:                    188.000 MiB

after zero_grad(set_to_none=True):
  allocated delta:                    69.550 MiB
  reserved delta:                    188.000 MiB

reconciliation after zero_grad:
  exact persistent tensor ledger:     51.402 MiB
  unclassified live allocation:       18.149 MiB
```

精确参数、梯度和 Adam moments 都与理论字节数一致。但 peak allocated 是静态主体的
数倍，说明 forward/backward 激活和瞬时量不能忽略。

`zero_grad(set_to_none=True)` 后 reserved 仍为 `188 MiB`，这是 caching allocator 保留 block
以便复用，不是梯度仍然存在的证据。实验直接检查了所有 `.grad is None`。

`18.149 MiB` 差额只能写成“未分类活分配”。本次没有用 memory snapshot 逐个 storage
归因，因此不能把它全部写成 workspace、activation 或 allocator 碎片；碎片通常更直接体现在
reserved 与 allocated 的差额中。

## 9. 常见误解与边界

### 9.1 “7B 模型就是 7 GB”

不对。`7B` 是 `7 × 10^9 parameters`。FP16/BF16 权重容量是：

```text
7 × 10^9 parameters × 2 bytes/parameter
= 14 × 10^9 bytes
= 14 GB
≈ 13.04 GiB
```

### 9.2 “AdamW 固定是 14 bytes/parameter”

不是统一事实。结果取决于参数、梯度、master weight 和 optimizer state 的真实
dtype 与是否独立存在。当前 FP32 baseline 的参数+梯度+Adam moments 为 16 bytes/parameter。

### 9.3 “混合精度必然有两份长期参数”

不对。PyTorch 常见 autocast 路径可以让模型参数本身保持 FP32，只让特定算子使用
FP16/BF16 计算。是否存在独立 FP32 master weight 要查真实 optimizer/framework 实现。

### 9.4 “reserved 高就是内存泄漏”

不对。reserved 可能是 caching allocator 为复用保留的 block。判断泄漏需要检查活 Tensor
引用和跨 step 持续增长，不能只看某一次 reserved 大于 allocated。

### 9.5 “saved logical bytes 就是 activation 新增显存”

不对。hooks 统计的 Tensor 可能是 view 或对参数等已有 storage 的引用。该数字适合比较
保存策略，不应与物理新增 allocated 做一对一等同。

### 9.6 当前尚未验证

- FP16/BF16 不同参数所有权与独立 FP32 master weight 的框架对照；
- SGD、8-bit optimizer、Adafactor 等其他 optimizer 的状态账本；
- Activation Checkpointing 和高效 Attention 同时启用后的完整账本；
- `torch.cuda.memory_snapshot()` 或 profiler 对未分类活分配的 storage 级归因；
- CUDA context 和非 PyTorch allocator 分配在 `nvidia-smi` 口径中的精确拆分；
- 真实大模型、长序列、多 GPU 与通信 buffer。

## 10. 手算练习

### 练习 1

一个 1B 参数模型使用 FP32 参数、FP32 梯度和两个 FP32 Adam moments。忽略 metadata 和
activation，静态主体是多少？

```text
1 × 10^9 parameters × (4 + 4 + 4 + 4) bytes/parameter
= 16 × 10^9 bytes
= 16 GB
≈ 14.90 GiB
```

### 练习 2

一份 BF16 hidden state shape 为 `[B,S,H] = [4,2048,4096]`，逻辑容量是多少？

```text
4 × 2048 × 4096 × 2 bytes
= 67,108,864 bytes
= 64 MiB
```

这只是一份 hidden state，不是一层 Transformer 的全部 activation。

### 练习 3

某时刻 `allocated = 10 GiB`、`reserved = 14 GiB`、`nvidia-smi = 15.5 GiB`。哪部分可以直接
写成“空闲可复用缓存”？

答案：`reserved - allocated = 4 GiB` 是 PyTorch allocator 池中当前未分配给活 Tensor
的保留容量。`nvidia-smi - reserved = 1.5 GiB` 不能直接命名为 CUDA context，因为还可能
有 library 或其他 allocator 的分配。

## 11. 面试口述

### 问题 1：如何估算全参数训练显存？

30 秒回答：先核对参数、梯度、master weight 和 optimizer state 是否独立存在及各自
dtype，用参数量乘每项字节数得到静态主体；再根据 batch、sequence、hidden size、层数和
kernel 估算 activation 与临时量。最后必须用 allocated/reserved/peak 实测，不把理论值
当成运行时硬上限。

### 问题 2：allocated、reserved 和 peak 有什么区别？

30 秒回答：allocated 是 allocator 当前分配给活 Tensor 的字节；reserved 是 PyTorch
已向 CUDA 申请并保留的缓存 block，包含 allocated 和空闲可复用部分；peak 是测量
窗口中出现过的最大值，不会因为当前 Tensor 释放就自动回落。

### 问题 3：为什么精确 Tensor 账本仍然和 peak allocated 不相等？

2 分钟回答：精确账本只对已识别的参数、梯度和 optimizer Tensor 求和。训练还有 forward
saved activations、backward 中间量、算子 workspace、模型返回并仍被引用的 Tensor，而且它们
的生命期会在某些时刻重叠。saved logical bytes 还可能包含 view 或已有 storage 引用。
因此应同时报告精确状态、allocator 快照和峰值，未归因差额保留为未知，不凭名字猜测。

## 12. 当日验收

- [ ] 能画出 parameter、gradient、Adam state、activation 的生命周期。
- [ ] 能解释 AdamW state 为何在第一次 step 后才出现。
- [ ] 能对 1B 参数 FP32+AdamW baseline 手算 `16 GB`静态主体。
- [ ] 能正确区分 GB、GiB 和模型名称中的 billion。
- [ ] 能区分 logical bytes、allocated、reserved、peak 和 `nvidia-smi`。
- [ ] 能解释 `zero_grad(set_to_none=True)` 为什么释放梯度却不要求 reserved 下降。
- [ ] 能解释当前实验的 `18.149 MiB` 差额为何不能擅自命名。
- [ ] 能根据真实参数所有权判断是否应计入独立 FP32 master weight。

下一步 Day 13：学习 DDP、FSDP/ZeRO、TP、PP、CP、EP 和 NCCL collectives，区分它们
切分什么、复制什么、需要什么通信。
