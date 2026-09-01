# Day 15：FSDP、ZeRO 与训练状态分片生命周期

> 状态：已生成，部分验证。两进程 Gloo/CPU 上的 FSDP2 参数、梯度、AdamW state
> 分片和 distributed checkpoint 恢复已经实测；真实双 GPU NCCL 显存、吞吐、BF16
> mixed precision 与不同 world size 恢复仍待目标环境验证。

## 1. 今日核心问题

Day 14 的 DDP 沿 batch 维切数据，但每个 rank 仍保留完整参数、完整同步梯度和完整
optimizer state。模型越大，这些重复状态越容易让单卡先 OOM。Day 15 追问：怎样把
训练状态也分散给不同 ranks，同时又让每一层仍能完成数学上等价的 forward、backward
和 optimizer step？

今天需要回答：

1. ZeRO Stage 1、2、3 分别切分什么，为什么不是“一个 rank 保存一阶动量，另一个
   rank 保存二阶动量”？
2. 参数已经分片后，一层 Linear 怎样获得计算需要的完整参数？
3. forward 后为什么可能再次丢弃完整参数，backward 前又为什么要重新 All-Gather？
4. backward 梯度怎样通过 Reduce-Scatter 变成每个 rank 的 local shard？
5. FSDP 与 ZeRO-3 高层生命周期相似，接口、包装边界和状态表示在哪里不同？
6. mixed precision 与 sharding 是两个什么维度？
7. 分布式 checkpoint 为什么不能简单保存 rank 0 当前看到的参数对象？

路线图覆盖清单：

- 路线图原计划：参数、梯度、optimizer state 分片；All-Gather、Reduce-Scatter；
  sharding strategy、reshard、mixed precision；distributed checkpoint 与 world-size
  边界；
- Day 14 衔接：每-rank 所有权、All-Reduce/Reduce-Scatter 语义和 process group；
- 本日完成：全部基础机制、FSDP2 两进程正确性与同 world-size checkpoint round trip；
- 直接依赖：Day 4 `TinyDecoderLM`、Day 12 显存账本、Day 13 分布式总览、Day 14
  collective 执行链；
- 明确延期：真实双 GPU allocator peak/吞吐/BF16，以及改变 world size 的 restore。

## 2. 前置知识与术语

### 2.1 训练状态不是一个整体

以 FP32 参数和 AdamW 为例，一个标量参数通常关联：

```text
parameter                 1 个 FP32
gradient                  1 个 FP32
first moment m            1 个 FP32
second moment v           1 个 FP32
```

此外，AdamW 常为每个“参数 Tensor”保存一个 scalar `step`。它不是每个标量参数一个
step。Day 15 的分片对象必须区分：逻辑模型参数、每-rank local shard、临时完整参数、
gradient shard、optimizer state shard 和 checkpoint 表示。

### 2.2 ZeRO

ZeRO 是 Zero Redundancy Optimizer，目标是消除数据并行 ranks 之间不必要的训练状态
重复。它由 DeepSpeed 生态推广，但这里先学习算法所有权，不把某一框架的具体配置
当作唯一实现。

- Stage 1：只分 optimizer state；
- Stage 2：再分 gradients；
- Stage 3：再分 parameters。

“分”通常按参数元素、flat buffer 或 bucket 的区间分片，而不是按状态种类分工。两个
ranks 下，某个参数区间归 rank 0，则该区间对应的 parameter/gradient/m/v 也尽量由
rank 0 管理；rank 1 管理另一段。

### 2.3 FSDP 与 FSDP2

FSDP 是 Fully Sharded Data Parallel，PyTorch 提供的全分片数据并行能力。FSDP2
使用 per-parameter DTensor 表示分片状态，并以 `fully_shard(module)` 组合式接口在
模块边界安装计算前后的通信与状态转换。

本日代码使用当前 PyTorch `2.13.0+cu130` 暴露的 FSDP2 接口。接口和内部实现可能随
版本变化，因此讲义区分稳定机制与当前代码事实。

### 2.4 DTensor、DeviceMesh 与 placement

- `DeviceMesh`：描述一组设备组成的逻辑网格；本日是长度为 2 的一维 `fsdp` mesh；
- `DTensor`：同时描述逻辑全局 shape、每-rank local Tensor 和 placement；
- `Shard(0)`：沿逻辑 Tensor 第 0 维分片；
- `Replicate()`：每个 rank 都保留完整副本。

逻辑 shape 不等于本地物理存储。例如逻辑参数可以是 `[128,32]`，两个 ranks 各自
只存其中约一半的 local shard。

### 2.5 All-Gather、Reduce-Scatter 与 reshard

- All-Gather：收集各 rank shards，使参与计算的 rank 获得逻辑完整 Tensor；
- Reduce-Scatter：对各 rank 的对应梯度贡献求和/规约，然后把结果分片返回；
- unshard：从持久 shard 进入临时完整参数状态；
- reshard：计算结束后丢弃临时完整参数，回到持久 shard 状态。

## 3. 从直觉到机制

### 3.1 从 DDP 到 ZeRO Stage 1

DDP 中两个 ranks 都有：

```text
rank 0: full P + full G + full Adam(m,v)
rank 1: full P + full G + full Adam(m,v)
```

Stage 1 先把 optimizer state 按参数区间分片：

```text
rank 0: full P + full G + Adam(m,v) shard 0
rank 1: full P + full G + Adam(m,v) shard 1
```

不是：

```text
rank 0 只保存所有参数的一阶动量 m
rank 1 只保存所有参数的二阶动量 v
```

那种切法会让更新同一个参数时必须跨 rank 取回另一种 moment，破坏状态局部性。按
参数区间切分时，一个 rank 同时拥有其负责区间的 m、v 和 step，能够本地更新这段
参数，再把更新结果同步给需要完整参数副本的 ranks。

### 3.2 ZeRO Stage 2：梯度也分片

Stage 2 的持久所有权变为：

```text
rank 0: full P + G shard 0 + Adam shard 0
rank 1: full P + G shard 1 + Adam shard 1
```

backward 仍会从输出层向输入层计算 local gradient contributions。并不是先把整个
完整梯度永久保存到每个 rank 再切掉，也不是算完最后一层就立即更新。工程实现可按
bucket 工作：一个 bucket 的 local contributions ready 后做 Reduce-Scatter，结果中
只有 shard 0 留在 rank 0、shard 1 留在 rank 1。整个 backward 完成后，各 rank 使用
自己拥有的 gradient/optimizer shards 更新对应参数区间。

### 3.3 ZeRO Stage 3：参数也分片

Stage 3 的静态所有权进一步变为：

```text
rank 0: P shard 0 + G shard 0 + Adam shard 0
rank 1: P shard 1 + G shard 1 + Adam shard 1
```

但 Linear/Attention 计算不能凭一半权重完成原来的普通算子，所以进入模块计算前需要
临时恢复其完整参数：

```text
persistent parameter shards
→ All-Gather
→ temporary full parameters
→ module forward
→ 可选 reshard，丢弃 temporary full parameters
```

这说明“参数已分片”描述的是持久状态所有权，不表示计算全过程中从不出现完整参数。
峰值显存还包含当前模块的临时 full parameters、activations、collective buffers 和
allocator 状态。

### 3.4 FSDP2 的逐模块 forward

本日对每个 Decoder Block 调用 `fully_shard()`，再对根模型调用一次。假设顺序为
Block 0、Block 1：

```text
Block 0 pre-forward:
  All-Gather Block 0 parameter shards
  → Block 0 full parameters ready

Block 0 forward:
  input[4,16,32] → output[4,16,32]

Block 0 post-forward:
  reshard_after_forward=True 时释放 full parameters

Block 1 重复同样过程
```

按模块分片让峰值更接近“持久 shards + 当前活跃模块 full parameters”，而不是同时
物化整个模型的 full parameters。具体 prefetch、buffer 复用和释放时机属于框架实现，
不能只根据概念图断言真实 allocator 峰值。

### 3.5 backward 为什么可能再次 All-Gather

如果 forward 后已经 reshard，backward 计算参数梯度或输入梯度时又需要该模块参数。
因此反向顺序中会再次 unshard：

```text
Block 1 backward pre-hook
→ All-Gather Block 1 parameters
→ backward compute
→ Reduce-Scatter gradient contributions
→ parameter/gradient 回到 shards

Block 0 再重复
```

`reshard_after_forward=True` 降低 forward 后的参数驻留显存，却可能增加 backward 前
重新 All-Gather 的通信。设置为 `False` 可以保留完整参数供 backward 使用，减少一次
重新收集，但会提高参数驻留显存。这是容量和通信的明确交换。

### 3.6 Reduce-Scatter 后保存的是什么

两个 ranks 对某个逻辑 gradient 都产生 local contribution：

```text
rank 0 local contribution: [g00,g01,g02,g03]
rank 1 local contribution: [g10,g11,g12,g13]
```

Reduce-Scatter SUM 后：

```text
rank 0: [g00+g10, g01+g11]
rank 1: [g02+g12, g03+g13]
```

如果训练语义需要平均，还要在框架定义的位置应用 world-size scaling。最终不是“只有
一个 GPU 保存更新后的完整梯度”，而是所有 ranks 各保存不同的规约结果 shard。

### 3.7 optimizer step 的局部性

参数、gradient、m、v 按相同逻辑区间对齐后，rank 0 可以本地更新 shard 0：

```text
P0, G0, m0, v0 → updated P0
```

rank 1 同时更新 shard 1：

```text
P1, G1, m1, v1 → updated P1
```

下一个模块计算需要完整参数时，再 All-Gather 更新后的 shards。这样避免每 rank 永久
保存所有 optimizer state。

### 3.8 FSDP 与 ZeRO-3 哪里相同、哪里不同

高层相同点：

- 都把 parameters、gradients、optimizer state 分片；
- 都在计算前收集参数；
- 都在 backward 后规约并分发梯度；
- 都用通信换取每-rank 静态显存下降。

不可直接画等号的部分：

- FSDP 是 PyTorch 原生模块/DTensor 集成，ZeRO 是算法家族，常由 DeepSpeed 配置；
- 参数分组、flat buffer、prefetch、offload、checkpoint API 和通信调度不同；
- FSDP2 的 `fully_shard()` 边界直接决定逐模块生命周期；
- 不同 stage/strategy 的命名和默认值不完全一一对应；
- 性能、峰值和 checkpoint 格式必须按具体版本实测。

因此准确表述是“FSDP full sharding 与 ZeRO Stage 3 的整体状态所有权和 collective
生命周期高度相似”，不是“两个库内部实现完全一样”。

### 3.9 mixed precision 与 sharding 是正交维度

sharding 回答：谁持有哪些元素。mixed precision 回答：持有或计算时使用什么 dtype。

FSDP2 `MixedPrecisionPolicy` 可以分别指定：

- `param_dtype`：forward/backward 计算时参数使用的 dtype；
- `reduce_dtype`：梯度 collective 使用的 dtype；
- `output_dtype`：模块输出 dtype；
- `cast_forward_inputs`：是否转换 forward 输入。

这不自动意味着持久参数和 AdamW state 永久变成 BF16。本日目标 NCCL 实验保留 FP32
optimizer states，BF16 仅在设备原生支持时启用。必须从实际 Tensor 和 profiler 读取
dtype，不能根据“开启 mixed precision”四个字推断所有状态。

### 3.10 distributed checkpoint

FSDP2 的运行状态是分片的。分布式 checkpoint 让各 rank 保存自己负责的 shards，并
生成描述逻辑 Tensor、placements 和文件布局的 metadata。恢复流程是：

```text
模型/optimizer 先按目标分布式布局初始化
→ 构造 state-dict 接收结构
→ 所有 ranks 参与 dcp.load
→ set_state_dict 写回模型和 optimizer
→ 用完整逻辑状态或后续训练行为验证
```

本日实测同一个 world size=2 的保存与恢复。框架能够支持某些 resharding 场景，不等于
任意模型、optimizer、版本和 world-size 变化都自动兼容；改变 world size 必须另做
真实恢复实验。

## 4. 极小手算例子

### 4.1 两 rank 的 Stage 1/2/3 所有权

设逻辑参数只有 8 个 FP32 元素：

```text
P = [p0,p1,p2,p3,p4,p5,p6,p7]
```

两个 ranks 均匀切分：

```text
shard 0 = [p0,p1,p2,p3]
shard 1 = [p4,p5,p6,p7]
```

各 stage 的主体元素数：

```text
DDP per rank:    P=8, G=8, m=8, v=8 → 32
Stage 1:         P=8, G=8, m=4, v=4 → 24
Stage 2:         P=8, G=4, m=4, v=4 → 20
Stage 3/FSDP:    P=4, G=4, m=4, v=4 → 16
```

乘以 4 bytes 后分别是 128、96、80、64 bytes。这里忽略 step scalar、padding、
activations、临时完整参数和通信 buffer。

### 4.2 All-Gather 参数

```text
rank 0 owns: [1,2]
rank 1 owns: [3,4]
```

All-Gather 后两个 ranks 的当前计算上下文都能看到：

```text
[1,2,3,4]
```

计算结束并 reshard 后，rank 0 重新只保留 `[1,2]`，rank 1 只保留 `[3,4]`。

### 4.3 Reduce-Scatter 梯度

```text
rank 0 contribution: [1,2,3,4]
rank 1 contribution: [10,20,30,40]
```

先规约得到 `[11,22,33,44]`，再分片：

```text
rank 0 gradient shard: [11,22]
rank 1 gradient shard: [33,44]
```

## 5. 正式实验配置

本日复用 Day 4 的 `TinyDecoderLM`：

```text
vocabulary_size          128
hidden_size               32
num_heads                  4
intermediate_size          64
num_layers                  2
sequence_length            16
global_batch_size           8
per_rank_batch_size         4
world_size                  2
optimizer               AdamW
learning_rate            1e-3
```

模型逻辑参数为 29,344 个元素。每个 Decoder Block 和根模型都调用 `fully_shard()`，
默认 `reshard_after_forward=True`。Gloo/CPU 路径固定 FP32；NCCL/CUDA 路径可以选择
FP32 或设备支持时的 BF16 mixed precision。

实验状态机：

```text
validate environment/output
→ init process group
→ construct sharded model/optimizer
→ forward/backward/step
→ save temporary sharded checkpoint
→ mutate local parameter shards
→ load and restore checkpoint
→ compare complete logical parameters
→ measure checkpoint files
→ remove temporary checkpoint directory
→ rank 0 atomically writes one JSON
→ destroy process group
```

## 6. 完整数据流与 Shape/Dtype

### 6.1 输入与输出

- global tokens：`[8,17]`，`torch.int64`；
- 每 rank `input_ids`：`[4,16]`，`torch.int64`；
- 每 rank `labels`：`[4,16]`，`torch.int64`；
- hidden states：逻辑 `[4,16,32]`；
- logits：`[4,16,128]`，FP32 输出；
- loss：每 rank 一个 FP32 scalar。

`B=4` 是每 rank 的样本数，`S=16` 是 token 位置数，`H=32` 是隐藏维度，`V=128`
是词表大小。activation 沿 data-parallel ranks 随 local batch 自然不同，不像参数那样
通过 All-Gather 恢复完整 global-batch activation。

### 6.2 参数 DTensor

逻辑模型参数元素数：

```text
29,344
```

两 rank 均匀分片后本机实测每 rank local 参数元素数：

```text
14,672
```

`parameter.numel()`、逻辑 shape、`parameter.to_local().numel()` 代表不同口径。账本必须
显式使用 local Tensor 才能回答实际每-rank shard 元素数。

### 6.3 backward 和 optimizer state

本机一次 FP32 step 后每 rank 实测：

```text
local gradient elements:          14,672
local optimizer Tensor elements:  29,366
```

AdamW 的两个 moment shards 共 `2 × 14,672 = 29,344` 个元素；额外 22 个元素来自
22 个参数 Tensor 各自的 FP32 scalar `step`。这验证 optimizer state 是按参数区间分片，
同时保留每参数 Tensor 的小型 metadata/state，而不是理想公式中严格只有 `2P/N`。

### 6.4 mixed precision 数据流

FP32 路径实测 parameter、gradient 和 optimizer Tensor 都是 `torch.float32`。BF16
目标路径的预期契约是：计算参数/reduction 使用 BF16，模块输出转回 FP32；持久参数和
optimizer state 的实际 dtype 必须从 JSON 与 profiler 重新核验，本机没有填入预期数值。

## 7. 参数、内存与通信成本

### 7.1 理想静态账本

设参数元素数为 $P$、world size 为 $N$，FP32 每元素 4 bytes，忽略 step scalar 和
padding：

```text
DDP:       (P + P + 2P) × 4 = 16P bytes per rank
ZeRO-1:    (P + P + 2P/N) × 4
ZeRO-2:    (P + P/N + 2P/N) × 4
ZeRO-3:    (P/N + P/N + 2P/N) × 4 = 16P/N bytes
```

这是静态训练状态主体，不是 CUDA peak allocated，也不含 activations、临时参数和
allocator reserved。

### 7.2 Stage 3/FSDP 的临时峰值

一层计算前需要当前模块的完整参数，因此峰值至少可能包含：

```text
persistent model shards
+ current module full parameters
+ gradient/reduction buffers
+ activations
+ optimizer shards
+ allocator fragmentation/reserved gap
```

所以 FSDP 的峰值不会严格等于 DDP 峰值除以 world size。模块 wrap 粒度越粗，一次
All-Gather 的完整参数越大；粒度越细，collective 次数和调度开销越多。

### 7.3 reshard 的成本

`reshard_after_forward=True`：

- forward 后释放完整参数，降低驻留显存；
- backward 前可能再次 All-Gather，增加通信。

`False`：

- 保留完整参数，可能减少重新 All-Gather；
- 提高 forward 与 backward 之间的显存占用。

它不是纯性能开关，也不是纯显存开关，而是通信、生命周期和容量的共同选择。

### 7.4 mixed precision 的成本

降低 parameter compute/reduce dtype 可以减少临时完整参数和 collective payload，并
可能利用 Tensor Cores；同时带来舍入误差、设备支持、loss stability 和额外 cast。
真实收益必须与 FP32 baseline 在相同模型、batch、warmup 和 measured steps 下比较。

## 8. 最小代码验证

实验入口：

```text
exercises/day15/fsdp_zero_sharding.py
```

### 8.1 静态契约

```bash
.venv/bin/python -m exercises.day15.fsdp_zero_sharding --validate-only
```

当前确认 PyTorch `2.13.0+cu130` 可导入 FSDP2、Gloo、NCCL、distributed checkpoint
以及 state-dict API。NCCL 接口存在不等于当前机器有两张可见 GPU。

### 8.2 本机两进程 Gloo/CPU

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day15.fsdp_zero_sharding \
  --backend gloo --output /tmp/day15_gloo.json
```

2026-08-31 当前实测：

- 两 ranks 的 local parameter/gradient elements 均为 `14,672`；
- logical parameter elements 为 `29,344`；
- 每 rank AdamW Tensor state 为 `29,366` 个元素；
- checkpoint 产生 3 个临时文件，本次总字节为 `655,306` bytes；
- 保存后将 local parameter shards 全部加 1，再恢复；
- 恢复后的完整逻辑参数最大差异为 `0.0`；
- 临时 checkpoint 目录删除后才写入 cleanup PASS；
- rank 0 写出一个新 JSON，不产生持久 checkpoint。

checkpoint bytes 包含 metadata、序列化和对齐开销，只是当前版本/配置的一次观测，
不能作为参数逻辑字节的固定倍数。

### 8.3 真实双 GPU NCCL FP32

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day15.fsdp_zero_sharding \
  --backend nccl --output /tmp/day15_nccl_fp32.json
```

### 8.4 真实双 GPU NCCL BF16

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day15.fsdp_zero_sharding \
  --backend nccl --mixed-precision bf16 \
  --output /tmp/day15_nccl_bf16.json
```

设备必须原生支持 BF16，否则脚本明确失败而不是悄悄 fallback。

### 8.5 reshard 对照

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day15.fsdp_zero_sharding \
  --backend nccl --keep-parameters-after-forward \
  --output /tmp/day15_nccl_no_reshard.json
```

当前脚本验证正确性和状态账本；目标机器上还需为 FP32/BF16、reshard true/false 增加
相同 warmup/measured steps 的 allocator peak、step latency 和 profiler collective
时间线。没有这些证据前不能宣称某设置更快或更省峰值。

### 8.6 已验证失败路径

- world size 不是 2 时，在 process group 初始化前明确拒绝；
- 输出文件已存在时，所有 ranks 收到相同错误且不覆盖；
- Gloo 路径请求 BF16 时参数校验拒绝；
- temporary checkpoint 在成功恢复后清理；异常路径也在 `finally` 中尝试清理；
- process group 在成功或失败路径最终销毁。

## 9. 常见误解与边界

### 9.1 “Stage 1 是 rank 0 保存 m、rank 1 保存 v”

不对。通常沿参数区间切分，每个 owner 同时保存该区间的 m、v 和 step，保证更新局部性。

### 9.2 “Stage 2 从来不产生完整 local gradient contribution”

不能这样绝对描述。autograd 会逐步产生梯度贡献，框架按 bucket/参数生命周期尽早
Reduce-Scatter；持久状态是 shard，不等于任何瞬间都绝不出现临时完整 buffer。

### 9.3 “Stage 3 每个 rank 永远只看到 1/N 参数”

不对。持久参数是 shards，模块计算前会临时 All-Gather 所需完整参数。

### 9.4 “reshard=True 一定更快”

不对。它通常降低参数驻留显存，但可能增加 backward 前的 All-Gather。更快还是更慢
取决于显存压力、模型 wrap、通信带宽和重叠。

### 9.5 “FSDP 就是 DeepSpeed ZeRO-3 的另一个名字”

不准确。两者高层 full-sharding 生命周期相似，但属于不同框架/API，内部表示、调度、
offload、checkpoint 和版本行为不同。

### 9.6 “开启 BF16 后所有训练状态都是 2 bytes”

不对。计算参数、reduction、输出、持久参数和 optimizer state 可以使用不同 dtype。
必须逐类读取实际 Tensor。

### 9.7 “checkpoint 保存成功就能任意 world size 恢复”

不对。必须用目标 world size、相同模型语义、兼容版本和 optimizer 配置真实 load，
再验证参数和续训结果。本日只验证 2→2。

## 10. 手算练习

### 练习 1：各 ZeRO stage 账本

逻辑参数有 1,000 个 FP32 元素，AdamW 有 m/v，world size=4。忽略 step 和其他开销，
分别计算 DDP、Stage 1、Stage 2、Stage 3 每 rank 主体元素数。

答案：

```text
DDP:       1000 + 1000 + 2000 = 4000
Stage 1:   1000 + 1000 +  500 = 2500
Stage 2:   1000 +  250 +  500 = 1750
Stage 3:    250 +  250 +  500 = 1000
```

### 练习 2：生命周期排序

将以下动作按 FSDP 模块 backward 顺序排列：参数 All-Gather、gradient
Reduce-Scatter、backward compute、参数 reshard。

答案：参数 All-Gather → backward compute → gradient Reduce-Scatter → 参数 reshard。
具体实现可能重叠或融合部分动作，但依赖关系不能颠倒。

### 练习 3：optimizer state 解释

本日每 rank 有 14,672 个参数 shard 元素，为什么 AdamW Tensor state 是 29,366 而
不是 29,344？

答案：两个 moments 是 `2 × 14,672 = 29,344`，再加 22 个参数 Tensor 各自的 scalar
step，共 29,366。

### 练习 4：选择 reshard

若训练因 forward 后到 backward 前的 full parameters 导致 OOM，优先测试哪个方向？

答案：优先测试 `reshard_after_forward=True`，同时记录额外 All-Gather 对 step latency
的影响；不能只看是否 OOM 就断言整体更优。

## 11. 面试口述

### 问题 1：ZeRO Stage 1 到 3 怎样演进？

30 秒回答：DDP 在每 rank 复制参数、梯度和 optimizer state。Stage 1 先按参数区间
分 optimizer state，Stage 2 再让规约后的 gradients 只保留 shards，Stage 3 继续把
parameters 分片。Stage 越高静态冗余越少，但参数计算前需要 All-Gather，backward
梯度需要 Reduce-Scatter，通信、临时峰值和 checkpoint 都更复杂。

### 问题 2：FSDP2 一个模块怎样执行？

2 分钟回答：持久状态中每 rank 只拥有参数 DTensor 的 local shard。模块 pre-forward
通过 All-Gather 临时恢复完整参数，完成 forward 后根据 reshard policy 选择释放或保留。
若已释放，backward 前再次 All-Gather；backward 计算各 rank 对 local batch 的梯度
贡献，再 Reduce-Scatter 为各 rank 的 gradient shard。optimizer 用同区间的 gradient、
moments 更新 parameter shard。下次计算再收集更新后的 shards。

### 问题 3：为什么 FSDP 省显存却可能更慢？

30 秒回答：它减少每-rank 持久参数、梯度和 optimizer state，但引入逐模块参数
All-Gather、梯度 Reduce-Scatter、临时 full parameters 和调度开销。小模型、慢互联、
过细 wrap 或频繁 reshard 时，通信开销可能超过节省带来的收益。

### 问题 4：分布式 checkpoint 要注意什么？

2 分钟回答：运行状态是分片的，checkpoint 需要所有 ranks 按统一 metadata 保存逻辑
模型和 optimizer shards，不能只保存 rank 0 的 local Tensor。恢复时先建立目标模型
布局，再 load 并 set state。必须验证模型版本、optimizer、dtype、world size 和续训
行为；“文件存在”或 save 返回成功都不足以证明可恢复。

## 12. 当日验收

### 12.1 基础机制

- [ ] 能按参数区间画出 ZeRO Stage 1、2、3 两-rank 所有权。
- [ ] 能解释为什么 m/v 一起按参数 shard，而不是按 moment 种类分给不同 ranks。
- [ ] 能画出 FSDP2 module forward/backward 的 All-Gather/Reduce-Scatter 生命周期。
- [ ] 能解释持久 parameter shard 与临时 full parameter 的区别。
- [ ] 能说明 `reshard_after_forward` 的显存/通信权衡。
- [ ] 能区分 sharding 与 mixed precision 两个维度。
- [ ] 能说明 distributed checkpoint 的 state-dict 和 world-size 边界。

### 12.2 当前环境证据

- [x] FSDP2、Gloo/NCCL 和 distributed checkpoint 静态契约通过。
- [x] 两进程 Gloo/CPU 完成 FSDP2 forward、backward 和 AdamW step。
- [x] 29,344 个逻辑参数均匀分为每 rank 14,672 个 local elements。
- [x] gradients 同样为每 rank 14,672 个 local elements。
- [x] AdamW 每 rank state 为 29,344 个 moment shard 元素加 22 个 scalar steps。
- [x] 临时分片 checkpoint 保存、参数破坏、恢复后最大差异 `0.0`。
- [x] checkpoint 临时目录清理后才记录 PASS。
- [x] 错误 world size 与重复输出明确失败。
- [x] Day 14 静态契约回归通过。

### 12.3 目标双 GPU 环境待验证

- [ ] NCCL FP32 路径通过同一组分片和 checkpoint 断言。
- [ ] 原生支持 BF16 的 GPU 上完成 mixed-precision 路径并记录各类 dtype。
- [ ] 对 DDP、FSDP reshard true/false 记录 allocator baseline/peak/reserved。
- [ ] 使用相同 warmup/measured steps 比较 step latency 和 tokens/s。
- [ ] profiler 确认参数 All-Gather、梯度 Reduce-Scatter 及其计算重叠。
- [ ] 记录 GPU、compute capability、driver、CUDA、NCCL 和 PCIe/NVLink topology。
- [ ] 使用不同 world size 完成 checkpoint restore 和后续一步训练一致性验证。

尚未确认的问题：当前极小模型在真实双 GPU 上很可能由 collective 启动和框架调度
主导，FSDP 不一定比 DDP 更快，甚至未必体现显著 allocator peak 优势。必须保留该
无收益结果，而不是为了得到“分片一定更好”的结论扩大或筛选数据。
