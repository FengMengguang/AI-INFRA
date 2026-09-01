# Day 14：DDP、Gradient Bucket 与 Collective 执行链

> 状态：已生成，部分验证。双进程 Gloo 正确性实验已在本机通过；真实双 GPU NCCL
> 时间线、通信性能与拓扑影响仍待目标环境验证。

## 1. 今日核心问题

Day 13 回答了“分布式训练可以沿哪些轴切分”。Day 14 不再横向罗列策略，而是沿
一次 DDP 训练 step 追踪执行细节：启动器怎样创建进程，各 rank 怎样得到数据，
backward 何时触发 gradient bucket 通信，以及为什么通信完成后每个 rank 能独立
执行相同的 optimizer step。

今天需要回答：

1. `torchrun` 创建的进程、rank、local rank 和 GPU 是什么关系？
2. 手工逐参数 All-Reduce 与 DDP bucket All-Reduce 为什么能得到相同更新？
3. backward 从后向前计算梯度时，通信怎样与尚未完成的计算重叠？
4. `gradient_as_bucket_view=True` 和 `no_sync()` 分别改变了什么？
5. All-Reduce 与 Reduce-Scatter 的输出所有权有什么根本区别？

本日路线图覆盖清单：

- 路线图原计划：rank、process group、gradient bucket、All-Reduce、
  Reduce-Scatter、通信计算重叠、单机多进程正确性实验；
- Day 13 衔接：DDP 的参数/梯度/optimizer state 复制关系，以及 collective 总览；
- 本日完成：上述全部基础机制和 Gloo 双进程正确性闭环；
- 直接依赖：Day 4 的 `TinyDecoderLM`，Day 13 的分布式术语；
- 延期边界：真实两卡 NCCL 时间线、带宽、step latency 与拓扑实验。

## 2. 前置知识与术语

### 2.1 启动器、进程与 rank

`torchrun` 是 PyTorch 的分布式启动入口。在单机双卡配置中，它不是让一个 Python
进程同时“变成两个 rank”，而是启动两个独立 Python 进程，并为每个进程设置环境
变量。

- `WORLD_SIZE=2`：整个作业有两个进程；
- `RANK=0/1`：进程在全局作业中的编号；
- `LOCAL_WORLD_SIZE=2`：当前机器上有两个进程；
- `LOCAL_RANK=0/1`：进程在当前机器上的编号，通常用来绑定本地 GPU；
- `process group`：一组能够参与 collective 的进程及其通信上下文；
- `backend`：collective 的实际通信后端，本日使用 CPU 上的 Gloo 做正确性实测，
  目标 GPU 环境使用 NCCL。

两行常见 CUDA 代码的分工是：

```python
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
```

第一行改变当前进程的默认 CUDA device；第二行创建一个明确指向该 GPU 的设备对象，
供 `.to(device)` 等调用使用。启动器提供 `LOCAL_RANK`，但脚本仍须显式完成绑定。

### 2.2 local gradient、global gradient 与 reduction

rank 0 和 rank 1 从相同参数出发，但读取不同 local batch，因此先得到不同的 local
gradient。`reduction` 的意思是把多个输入按元素用某种运算合并。本日采用求和，再
除以 world size 得到等 local batch 情况下的平均梯度。

设两个 rank 的 local gradients 为 $g_0$ 和 $g_1$，则同步后梯度为：

$$
g = \frac{g_0 + g_1}{2}
$$

每个 rank 都使用同一个 $g$ 和同一份 optimizer state 更新相同初始参数，因此更新后
的参数副本仍一致。

### 2.3 gradient bucket、view 与通信 hook

逐个小 Tensor 发起 collective 会产生大量启动开销。DDP reducer 会把多个参数的
梯度组织进较大的连续 bucket；当一个 bucket 所需的梯度都 ready 时，便可启动该
bucket 的通信。

`view` 是共享已有 storage 的逻辑窗口，不是再复制一份数值。启用
`gradient_as_bucket_view=True` 后，`parameter.grad` 可以成为 bucket 中对应区域的
view。这样通信直接原地修改 bucket，也就修改了这些 `.grad` 所看到的数据，减少
bucket 与独立 gradient buffer 之间的复制和一份潜在存储。

通信 hook 是 DDP 在 bucket ready 后调用的扩展点。本日 hook 只做三件事：记录调用
次数、对 bucket 执行异步 All-Reduce、把求和结果除以 world size。它是为了观察
机制，不是生产性能优化。

### 2.4 Future 与异步完成

异步 collective 发起后立即返回一个 work/Future，而不是要求 Python 当场等待所有
rank。Future 表示“结果稍后完成”。DDP 只有在对应 Future 完成后，才把这个 bucket
视为已经同步。这提供了通信与其他 backward 计算重叠的基础。

## 3. 从直觉到机制

### 3.1 一次双 rank 训练 step 的所有权

每个 rank 拥有：

- 一份完整模型参数；
- 自己的 local batch；
- 本地 forward activations；
- backward 逐步产生的 local gradients；
- 同步完成后的一份完整平均 gradients；
- 一份完整 AdamW optimizer state。

DDP 切分的是 global batch，不切参数、最终梯度或 optimizer state。训练前后没有同时
保留“本地完整梯度”和“汇总完整梯度”两套必需副本；通常是同一片 gradient/bucket
存储从本地值被原地改写为规约后的值。

### 3.2 启动与初始化

执行链为：

```text
torchrun
→ 创建两个 Python 进程并设置 rank 环境变量
→ 每个进程读取 RANK / LOCAL_RANK / WORLD_SIZE
→ CUDA 路径把进程绑定到 cuda:LOCAL_RANK
→ init_process_group 建立 collective 通信上下文
→ 两个进程从相同初始参数构造 DDP
```

`rank 0` 不是计算主节点，也不是唯一执行 optimizer 的进程。本实验仅让 rank 0 汇总
和写 JSON，以避免多个进程竞争同一结果文件。

### 3.3 数据沿 batch 维切分

全局 token Tensor shape 是 `[8,17]`。其中 8 是 global batch 中的样本数，17 是
用于构造 16 个输入 token 和 16 个右移标签的位置数。

两个 rank 各取连续 4 行：

```text
rank 0: rows 0–3 → input_ids[4,16], labels[4,16]
rank 1: rows 4–7 → input_ids[4,16], labels[4,16]
```

这里两个 local batch 等大，因此“对两个 local mean gradients 再平均”等价于对
global batch 的全部样本求平均。local batch 不等大时不能直接除以 rank 数。

### 3.4 手工 All-Reduce baseline

最透明的 baseline 是：

```text
local forward
→ local loss
→ local backward 得到每个 parameter.grad
→ 对每个 parameter.grad 分别 All-Reduce SUM
→ 每个结果除以 2
→ 每个 rank 独立 optimizer.step()
```

本日模型共有 22 个参数 Tensor，所以手工实现发起 22 次 collective。这里的 22 是
Tensor 对象数量，不是 29,344 个标量参数分别通信。所有 collective 合计处理
29,344 个 gradient 元素。

### 3.5 DDP bucket 路径

DDP 在模型参数上注册 autograd hooks。backward 从 loss 向前追踪依赖，较靠近输出
的参数梯度通常更早 ready。每次某个参数梯度 ready，reducer 就把对应 bucket 的
ready 状态向前推进。

```text
某些参数梯度 ready
→ 它们所属 bucket 尚不完整：继续 backward
→ bucket 所需梯度全部 ready
→ 通信 hook 发起异步 All-Reduce
→ autograd 继续计算其他梯度
→ Future 完成，bucket 原地成为平均梯度
→ 所有 bucket 完成
→ backward() 返回
→ optimizer.step()
```

关键点是：梯度计算仍从后层向前层推进；参数更新并不是“算完后层梯度就立刻更新
后层”。DDP 要等该 step 的 backward 和必要同步完成后，才统一进入 optimizer
step。否则提前改变参数会破坏仍在进行的 backward 所依赖的参数版本。

### 3.6 为什么可以通信计算重叠

如果输出侧 bucket 已 ready，而输入侧层仍在计算梯度，输出侧 bucket 的异步通信
可以与输入侧 backward 并行。能够重叠多少取决于：

- bucket 怎样划分以及梯度 ready 顺序；
- 网络、PCIe/NVLink 和 GPU kernel 的并发能力；
- 模型是否足够大，通信是否足够长；
- 实现是否因依赖或同步点提前等待。

“使用 DDP”不自动保证完全重叠。本机 Gloo 实验验证数值和调用次数，不提供 GPU
时间线证据。

### 3.7 `no_sync()` 的梯度累积

如果一个 rank 的 local batch 又被切成两个 micro-batches，默认每次 backward 都会
触发 DDP 通信。`no_sync()` 可以让前一个 micro-batch 只把 local gradient 累积到
`.grad`，最后一个 micro-batch 再触发同步：

```text
micro-batch 0: forward → backward inside no_sync → 0 次 bucket 通信
micro-batch 1: forward → normal backward → bucket 通信累计后的 local gradient
optimizer.step()
```

两个等大 micro-batches 的 loss 都除以 2，才能让累积 gradient 与完整 local batch
的 mean-loss gradient 一致。`no_sync()` 只延后 DDP 通信，不替代 loss scaling，
也不会自动执行 optimizer step。

### 3.8 All-Reduce 与 Reduce-Scatter

All-Reduce 可概念性拆成：

```text
Reduce-Scatter：对应元素先规约，每个 rank 只得到结果的一段
All-Gather：再收集所有结果段，让每个 rank 得到完整结果
```

因此：

- DDP 默认需要每个 rank 拥有完整同步梯度，使用 All-Reduce；
- ZeRO Stage 2/FSDP 类路径希望规约后每 rank 只拥有 gradient shard，可使用
  Reduce-Scatter；
- Reduce-Scatter 后不是“更新后的梯度只在一个 rank”，而是完整逻辑梯度被分片，
  每个 rank 各拥有不同的一段。

Day 14 的代码只实际调用 All-Reduce。Reduce-Scatter 在这里完成语义和数据流说明，
其参数/梯度分片生命周期在 Day 15 实作。

### 3.9 完成、失败与清理

分布式失败会影响整个作业，不能只考虑 rank 0 的正常路径。本实验的边界是：

- 缺少启动器环境变量时，在建立 process group 前明确失败；
- world size 不是 2 时明确失败；
- NCCL 路径看不到两张 GPU 时明确失败；
- 输出已存在时，由 rank 0 判定并广播错误，所有 ranks 一致退出；
- JSON 由 rank 0 临时写入后原子替换；
- 无论正常或异常，已初始化的 process group 都在 `finally` 中销毁。

## 4. 极小手算例子

### 4.1 两个 rank 的平均梯度

设一个参数向量有两个元素：

```text
rank 0 local gradient: [2, 6]
rank 1 local gradient: [4, 2]
```

All-Reduce SUM 后两个 ranks 都看到 `[6,8]`；除以 2 后都看到 `[3,4]`。若学习率为
0.1 且暂时忽略 AdamW 的 moments、weight decay，参数 `[10,20]` 更新为：

```text
[10,20] - 0.1 × [3,4] = [9.7,19.6]
```

### 4.2 一个 bucket view

设 bucket storage 是 `[1,2,3,4,5]`：

```text
parameter_a.grad → bucket[0:2]
parameter_b.grad → bucket[2:5]
```

如果 All-Reduce 原地把 bucket 改成 `[10,20,30,40,50]`，两个 `.grad` view 立即分别
看到 `[10,20]` 和 `[30,40,50]`，不需要再从 bucket 拷贝回两个独立 Tensor。

### 4.3 不等 local batch 的边界

rank 0 有 3 个样本、mean gradient 为 2；rank 1 有 1 个样本、mean gradient 为 6。
直接平均 local means 得到 4，但 global sample mean 是：

$$
g = \frac{3 \times 2 + 1 \times 6}{3 + 1} = 3
$$

所以“求和后除以 world size”依赖各 rank 对等权重。当最后一个 batch 不齐时，需要
sampler/drop-last 或按样本数加权等明确策略。

## 5. 正式实验配置

本日实验复用 Day 4 的两层 `TinyDecoderLM`：

```text
vocabulary_size       128
hidden_size            32
num_heads                4
intermediate_size       64
num_layers                2
sequence_length          16
global_batch_size         8
per_rank_batch_size       4
world_size                2
optimizer             AdamW
learning_rate          1e-3
bucket_cap_mb           0.01
```

同一份初始权重、同一份 global token 数据依次运行三条路径：

1. 手工逐参数 All-Reduce；
2. DDP bucket All-Reduce；
3. DDP + 两个 micro-batches + 首次 `no_sync()`。

Gloo/CPU 是当前机器的正确性验证后端。NCCL/CUDA 使用相同实验契约，但需要单机两张
可见 CUDA GPU；它不是 Gloo 结果的自动性能外推。

## 6. 完整数据流与 Shape/Dtype

### 6.1 输入与输出

- global tokens：`[8,17]`，`torch.int64`；
- 每 rank `input_ids`：`[4,16]`，`torch.int64`；
- 每 rank `labels`：`[4,16]`，`torch.int64`；
- hidden states：`[4,16,32]`，FP32；
- logits：`[4,16,128]`，FP32；
- local loss：scalar，FP32；
- 参数和 gradients：共 29,344 个元素，FP32。

`B=4` 表示一个 rank 的 local batch 样本数，`S=16` 表示每个样本的 token 位置数，
`H=32` 表示每个 token 的隐藏特征数，`V=128` 表示词表大小。

### 6.2 backward 与 bucket

每个参数的 `.grad` 逻辑 shape 与参数相同。reducer 可以把多个不同 shape 的 gradients
展平组织到一维 bucket storage；这不改变每个 `.grad` 对外呈现的逻辑 shape。

本次小模型首次 iteration 被组织为一个 bucket，因此 DDP hook 每 rank 调用 1 次，
处理 29,344 个元素。手工 baseline 每 rank 调用 22 次，处理的元素总数仍为 29,344。
这只是当前模型和 PyTorch 版本的一次观测，不能推广为“DDP 永远只有一个 bucket”。

### 6.3 optimizer step 后

三条路径都从相同参数开始。同步完成后，每个 rank 的完整参数必须逐元素一致；手工
baseline 与 DDP 参数更新也必须在浮点容差内一致。JSON 只由 rank 0 汇总写出，但
里面保留每个 rank 的 local loss、通信调用次数和参数一致性结果。

## 7. 参数、内存与通信成本

### 7.1 DDP 静态状态

若参数、gradient 和 AdamW 两个 FP32 moments 各按参数规模 $P$ 计，忽略 scalar
step、allocator padding 和临时 buffer，每 rank 的主体仍约为：

```text
parameters       4P bytes
gradients        4P bytes
AdamW moments    8P bytes
total           16P bytes per rank
```

DDP 增加 GPU 数不会把这部分按 world size 分片。bucket 可能引入额外 buffer；启用
gradient-as-bucket-view 可减少 gradient 与 bucket 重复存储，但实际峰值仍需 profiler
或 allocator 实测，不能只按逻辑字节断言。

### 7.2 All-Reduce 通信量

对大小为 $M$ bytes 的大消息，ring All-Reduce 的每 rank 理想发送/接收量常写为：

$$
2 \times \frac{N-1}{N} \times M
$$

$N$ 是 rank 数。这个式子描述 ring 算法的数据量主体，不等于实际延迟；小消息启动
开销、拓扑、协议、链路竞争和实现选择都会改变时间。

### 7.3 bucket 大小的权衡

- bucket 太小：更早 ready，潜在重叠更好，但 collective 次数和启动开销增加；
- bucket 太大：collective 次数减少，但必须等更多 gradients ready，可能缩短重叠窗口；
- 最佳值依赖模型结构、网络和硬件，不存在脱离实验的固定答案。

`bucket_cap_mb` 是期望上限，不应被理解为每个 bucket 必然严格等大；首次 iteration、
参数顺序和框架内部重建策略都可能影响实际 bucket 组织。

### 7.4 Gloo 与 NCCL 的事实边界

Gloo 是 PyTorch distributed 可使用的通信后端，本日用它在 CPU 上完成双进程数值
验证。NCCL 是面向 NVIDIA GPU collective 的通信库。Gloo 正确性通过不能证明 NCCL
性能，也不能观察 PCIe/NVLink 拓扑对 GPU 通信的影响。

## 8. 最小代码验证

实验入口：

```text
exercises/day14/ddp_nccl_execution.py
```

### 8.1 静态契约

```bash
.venv/bin/python -m exercises.day14.ddp_nccl_execution --validate-only
```

当前环境输出确认 PyTorch `2.13.0+cu130`、Gloo 和 NCCL 接口可用。接口可用不表示
当前环境具备两张可见 GPU。

### 8.2 本机双进程 Gloo 正确性

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day14.ddp_nccl_execution \
  --backend gloo --output /tmp/day14_gloo.json
```

2026-08-31 当前实测：

- world size 为 2，device type 为 CPU；
- 手工路径每 rank 发起 22 次通信，DDP 路径每 rank 的 hook 调用 1 次；
- 两者均处理 29,344 个 gradient 元素；
- 手工路径与 DDP 更新最大参数差异为 `0.0`；
- 手工路径与 `no_sync()` 累积更新最大参数差异为
  `3.129243850708008e-07`；
- `no_sync()` 第一个 micro-batch 后通信调用次数为 `0`；
- 三条路径的 rank 0/rank 1 参数同步断言全部通过。

这些数字只证明当前小模型的正确性和观察 hook 行为，不是性能 benchmark。

### 8.3 真实双 GPU NCCL 契约

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day14.ddp_nccl_execution \
  --backend nccl --output /tmp/day14_nccl.json
```

目标环境：单机两张可见 NVIDIA GPU。成功标准与 Gloo 数值检查相同，并需额外使用
profiler/Nsight Systems 记录 backward kernel、NCCL collective 与重叠时间线。运行前
还应记录 GPU 型号、compute capability、PCIe/NVLink 拓扑、driver、PyTorch、CUDA
和 NCCL 版本。

### 8.4 已验证失败路径

- `--nproc-per-node=1` 被明确拒绝，错误包含实际 world/local world size；
- 结果文件已经存在时两个 ranks 一致失败，原文件不被覆盖；
- 结果只写入用户指定的新 JSON，不产生 checkpoint。

## 9. 常见误解与边界

### 9.1 “rank 0 汇总梯度，再发给其他 ranks”

不是 DDP All-Reduce 的语义。collective 中所有 ranks 都参与规约，结束后所有 ranks
都得到结果。实现可以采用 ring/tree 等算法，不要求把 rank 0 当中央梯度服务器。

### 9.2 “一个 bucket ready 就立即更新对应参数”

不对。bucket ready 可以提前启动通信，但 optimizer step 通常在整个 backward 和
同步完成后执行。通信的提前与参数更新的提前是两回事。

### 9.3 “gradient bucket view 是切分梯度所有权”

不对。view 描述存储别名关系，不表示不同 rank 各自拥有不同 shard。DDP 最终仍让
每个 rank 得到完整同步 gradient。分片所有权是 Reduce-Scatter/FSDP/ZeRO 的问题。

### 9.4 “`no_sync()` 后最后一次只同步最后一个 micro-batch”

不对。前面 micro-batches 的梯度已累积在 `.grad` 中，最后一次正常 backward 触发
对累计值的 bucket 同步。前提是没有在中间 `zero_grad()`，并且 loss scaling 正确。

### 9.5 “bucket 越小重叠越好，所以一定越快”

不成立。更小 bucket 可能更早通信，也会增加 collective 启动次数。必须在真实模型、
互联和 batch/sequence 配置上测 step latency。

### 9.6 “Gloo 两进程通过就等于 NCCL 双卡通过”

不成立。Gloo 已验证 Python 进程生命周期、collective 数值和 DDP 状态一致性；GPU
device binding、NCCL kernel、链路拓扑、显存和通信计算重叠仍是未验证边界。

## 10. 手算练习

### 练习 1：All-Reduce

三个 ranks 的某个 gradient 分别为 `[1,2]`、`[4,5]`、`[7,8]`。求 SUM All-Reduce
结果和平均结果，并说明哪些 ranks 能看到它。

答案：SUM 为 `[12,15]`，平均为 `[4,5]`，All-Reduce 完成后三个 ranks 都能看到。

### 练习 2：bucket view

一个 8 元素 bucket 中，参数 A、B、C 的 gradient 分别占 `[0:2]`、`[2:5]`、
`[5:8]`。如果 A 的 `.grad.add_(10)`，bucket 哪些元素变化？

答案：若 A 的 `.grad` 是 bucket view，只有 bucket 的前两个元素原地加 10，不产生
一份独立的 A gradient copy。

### 练习 3：`no_sync()` scaling

一个 local batch 等分为 4 个 micro-batches，每个 micro loss 默认都是各自样本 mean。
要复现完整 local batch mean gradient，每次 backward 前应怎样缩放？何时允许同步？

答案：每个 micro loss 除以 4；前三次放在 `no_sync()` 中，最后一次正常 backward
触发累计 gradient 同步，然后只执行一次 optimizer step。

### 练习 4：选择 collective

1. 每个 rank 最终需要完整平均 gradient：All-Reduce。
2. 规约后每 rank 只保留不同 gradient shard：Reduce-Scatter。
3. 已有 shards，要让每 rank 恢复完整 Tensor：All-Gather。

## 11. 面试口述

### 问题 1：DDP 的一个 step 怎样执行？

30 秒回答：启动器为每张 GPU 创建一个进程，进程用 local rank 绑定设备并加入 process
group。各 rank 从相同参数出发，对不同 local batch 做 forward/backward。梯度 ready
后被 DDP reducer 组织进 buckets 并 All-Reduce；同步后的完整平均 gradient 在每 rank
上相同，因此各 rank 独立执行 optimizer step 仍得到相同参数。

### 问题 2：gradient bucket 为什么既影响性能又影响内存？

2 分钟回答：bucket 合并小 gradient，减少 collective 启动开销；bucket ready 后可异步
通信，与其余 backward 重叠。太大可能晚启动，太小可能通信次数过多。开启
gradient-as-bucket-view 后，parameter.grad 可以直接引用 bucket 区域，减少拷贝和
重复 buffer，但不改变 DDP 的完整梯度所有权。最佳 bucket 大小必须实测。

### 问题 3：All-Reduce 与 Reduce-Scatter 怎样区分？

30 秒回答：两者都规约对应元素。All-Reduce 结束后每个 rank 得到完整结果，适合
DDP 的复制梯度；Reduce-Scatter 结束后每个 rank 只得到不同结果 shard，适合
ZeRO/FSDP 的分片梯度所有权。All-Reduce 可概念性看作 Reduce-Scatter 再 All-Gather。

### 问题 4：怎样让 DDP 梯度累积不在每个 micro-batch 通信？

30 秒回答：前面的 micro-batches 在 `model.no_sync()` 中 backward，只累计 local
gradients；最后一个 micro-batch 正常 backward，触发累计 gradient 的 bucket 同步。
同时要按累积步数或实际样本权重缩放 loss，中间不能清空 gradient，最后只 step 一次。

## 12. 当日验收

### 12.1 基础机制

- [ ] 能画出 torchrun 创建进程、设置 ranks、绑定 GPU 和初始化 process group 的链路。
- [ ] 能解释 local gradient 怎样原地变成所有 ranks 相同的平均 gradient。
- [ ] 能解释 backward 顺序、bucket ready、异步 collective 和 optimizer step 的时序。
- [ ] 能区分 bucket view 与 gradient shard。
- [ ] 能区分 All-Reduce 与 Reduce-Scatter 的输出所有权。
- [ ] 能说明 `no_sync()` 的通信时机与 loss scaling 责任。

### 12.2 当前环境证据

- [x] 静态契约通过，Gloo 与 NCCL 接口可导入。
- [x] Gloo 双进程完成手工 All-Reduce、DDP 和 `no_sync()` 三条路径。
- [x] 手工与 DDP 参数更新最大差异为 `0.0`。
- [x] `no_sync()` 首个 micro-batch 通信次数为 0，最终更新在容差内等价。
- [x] 每条路径的两个 ranks 参数逐元素一致。
- [x] 错误 world size 与已有输出文件均被明确拒绝。
- [x] Day 13 静态契约回归通过。

### 12.3 目标双 GPU 环境待验证

- [ ] 两个 ranks 分别绑定两张 GPU，NCCL process group 初始化成功。
- [ ] NCCL 路径通过与 Gloo 相同的参数一致性和更新等价检查。
- [ ] profiler 时间线显示哪些 bucket communication 与哪些 backward kernels 重叠。
- [ ] 记录 GPU、compute capability、拓扑、软件版本和 NCCL 版本。
- [ ] 对比 bucket 配置的 step latency；不预设 GPU 数增加后线性加速。
- [ ] 将目标环境结果回写本讲义，不用 Gloo 数字替代 NCCL 性能证据。

尚未确认的问题：当前小模型首次 backward 只形成一个 bucket，无法展示多个 bucket
依次 ready 的真实重叠形态。该机制需要在更大模型和真实双 GPU profiler 时间线中验证。
