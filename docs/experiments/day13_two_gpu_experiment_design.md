# Day 13 两卡分布式训练实验设计

## 1. 状态与完成边界

本文是 Day 13 的实验协议，不是已完成的实验报告。当前本机只有一张 NVIDIA
GeForce RTX 2060，不能产生真实两卡 NCCL、DDP 和 FSDP2 证据。

Day 13 只能在以下条件全部成立后标记完成：

```text
同一主机有两张对当前进程可见的 NVIDIA GPU
`python -m torch.distributed.run` 启动两个 rank
rank 0 和 rank 1 分别绑定 cuda:0 和 cuda:1
NCCL process group 初始化成功
collective 的精确结果断言通过
DDP 与 FSDP2 一次参数更新在容差内一致
每 rank 显存、状态字节、step time 和 tokens/s 写入结构化 JSON
GPU 型号、compute capability、PyTorch/CUDA/NCCL 版本和拓扑被记录
```

命令退出码为 0 只是证据之一。结构化 JSON 中的逐 rank 状态和断言结果才是实验
事实源。

## 2. 为什么不用 CPU/Gloo 代替

CPU/Gloo 可以验证 AllReduce 等 collective 的数学语义，但不能验证：

- NCCL kernel 和 CUDA stream 行为；
- 每张 GPU 的 allocated/reserved/peak；
- PCIe/NVLink 拓扑对通信的影响；
- FSDP2 AllGather/ReduceScatter 与 GPU 计算的生命期重叠；
- 真实多进程 CUDA 错误、超时和资源清理。

因此本机只运行 `--validate-only`，不生成伪多卡数字。

## 3. 核心对照协议

实验使用 Day 4 的 `TinyDecoderLM`，不复制第二套 Transformer。DDP 和 FSDP2 共享：

```text
模型配置
参数初始化 seed
global token batch
global batch 切分规则
loss 定义
AdamW 配置
warmup/measured steps
```

全局 batch 为 8，两个 rank 各取 4 条序列。两个本地 loss 都使用 mean reduction，样本数
相等，因此 DDP/FSDP2 对梯度的平均对应同一个 global batch 目标。

执行顺序：

```text
1. 验证 AllReduce、ReduceScatter、AllGather、All-to-All 精确结果
2. 固定 seed 创建 DDP 模型，运行一次更新
3. 保存 DDP 更新后的 CPU 参数副本，运行短 benchmark
4. 释放 DDP 模型和 optimizer
5. 用相同 seed 创建 FSDP2 模型，运行一次更新
6. 所有 rank 参与 DTensor full_tensor() 聚合，逐参数与 DDP 对照
7. 运行 FSDP2 短 benchmark，汇总每 rank 证据
8. rank 0 原子写入唯一 JSON，所有 rank barrier 后销毁进程组
```

## 4. Collective 验收向量

### 4.1 AllReduce

```text
rank 0: [1,3]
rank 1: [5,7]
SUM / world_size
每个 rank 期望: [3,5]
```

对应 DDP 中多 rank 梯度规约与平均的直觉。

### 4.2 ReduceScatter

```text
rank 0 input: [1,2,3,4]
rank 1 input: [10,20,30,40]
先 SUM:       [11,22,33,44]
rank 0 output:   [11,22]
rank 1 output:   [33,44]
```

对应 FSDP2 backward 将完整梯度规约后只保留本 rank 分片的直觉。

### 4.3 AllGather

```text
rank 0 shard: [1,2]
rank 1 shard: [3,4]
每个 rank 期望: [1,2,3,4]
```

对应 FSDP2 在计算某组模块前临时聚合完整参数。

### 4.4 All-to-All

```text
rank 0 input: [0,1 | 2,3]
rank 1 input: [10,11 | 12,13]
rank 0 output: [0,1,10,11]
rank 1 output: [2,3,12,13]
```

对应 Expert Parallel 中 token 按目标专家所在 rank 重新路由的直觉。

## 5. DDP 与 FSDP2 状态所有权

### 5.1 DDP

```text
每个 rank:
完整模型参数
完整参数梯度
完整 AdamW state
不同本地 batch

backward:
本地梯度 → AllReduce/平均 → 各 rank 梯度一致
```

DDP 适合模型及完整训练状态能装入单卡的数据并行。

### 5.2 FSDP2

```text
稳态:
每 rank 保留 DTensor 参数分片
梯度和 optimizer state 也按数据并行 mesh 分片

某组模块计算前:
AllGather 参数分片 → 临时完整参数

backward 后:
完整梯度 → ReduceScatter → 本 rank 梯度分片
```

实验按 PyTorch FSDP2 官方建议自底向上对每个 Decoder Block 调用 `fully_shard`，
最后再对 root model 调用。每次 `fully_shard` 定义一个通信分组，避免只在 root 上形成
一个超大阻塞 AllGather/ReduceScatter 组。

## 6. 四张两卡数据流验收图

### 6.1 DDP

```text
rank 0 / GPU 0                     rank 1 / GPU 1
完整参数 W                      完整参数 W
local batch X0                    local batch X1
      │ local forward/backward           │ local forward/backward
      ▼                                  ▼
local gradient G0  ── AllReduce ── local gradient G1
      │                                  │
      ▼                                  ▼
average gradient G                average gradient G
optimizer.step                    optimizer.step
更新后完整 W'                    更新后完整 W'
```

### 6.2 FSDP2 / ZeRO-3 类直觉

```text
rank 0 / GPU 0                     rank 1 / GPU 1
参数分片 W0                      参数分片 W1
       \                              /
        \------ AllGather W --------/
                临时完整参数
                       │ forward/backward
                       ▼
                 完整梯度逻辑量
        /------ ReduceScatter ------\
       /                              \
梯度分片 G0                      梯度分片 G1
optimizer shard 0                  optimizer shard 1
更新 W0'                         更新 W1'
```

FSDP2 与 DeepSpeed ZeRO-3 都在数据并行 rank 间分片参数、梯度和 optimizer state，但
具体 API、参数表示、checkpoint 和调度实现不同，不应写成完全相同的框架。

### 6.3 Tensor Parallel

```text
一层权重 W[out,in]

rank 0 / GPU 0                     rank 1 / GPU 1
W0[out/2,in]                       W1[out/2,in]
X replicated                      X replicated
Y0 = X @ W0^T                     Y1 = X @ W1^T
        \                            /
         \--- AllGather/Concat ----/
                 Y[out]
```

真实 Transformer TP 通常组合 column-parallel 和 row-parallel Linear，不是每层固定只做一次
AllGather。本图只表达“单层矩阵沿输出维切分”的最小抽象。

### 6.4 Expert Parallel

```text
rank 0 / GPU 0                     rank 1 / GPU 1
Expert 0, Expert 1                Expert 2, Expert 3
local tokens                      local tokens
       \                            /
        \-- All-to-All routing ----/
    token 到达目标 expert 所在 GPU
       /                            \
本地 experts 计算                   本地 experts 计算
       \                            /
        \-- All-to-All return -----/
         token 回到原序列位置
```

## 7. PP 与 CP 的位置

Pipeline Parallel 沿层切分：

```text
GPU 0: layers 0..N/2-1
GPU 1: layers N/2..N-1
micro-batches 在 stage 间传递 activation/gradient
主要问题：pipeline bubble、调度、stage 负载和 activation 传输
```

Context Parallel 沿 sequence/context 维切分：

```text
GPU 0: X[:, first sequence shard, :]
GPU 1: X[:, second sequence shard, :]
长上下文 attention 需要交换 K/V 或部分 attention 结果
```

不同框架对 CP/SP 的名称和精确算法不完全一致。Day 13 讲清切分轴和通信目的，不将
一个特定框架实现写成通用定义。

## 8. 结果 JSON 契约

rank 0 写入的 JSON 至少包含：

```text
schema_version
status
created_at_utc
torch/CUDA/NCCL version
world_size
GPU name/compute capability/total memory per rank
nvidia-smi topo -m
model/global batch/per-rank batch/sequence/dtype/warmup/measured steps
collective exact outputs per rank
DDP metrics per rank
FSDP2 metrics per rank
DDP/FSDP2 maximum parameter difference per rank
named PASS checks
```

每个 mode/rank 的 metrics 包含：

```text
local parameter bytes
local gradient bytes
local optimizer bytes
baseline allocated
after-step allocated
peak allocated
peak delta
average step time
global tokens/s
```

本实验不写 model checkpoint。JSON 路径必须由用户显式传入，父目录必须已存在，目标
文件必须不存在。重复运行需要使用新文件名，避免静默覆盖证据。

## 9. 运行方式

在本机只做静态验证：

```bash
python -m exercises.day13.two_gpu_distributed_training --validate-only
```

在租用两卡主机上，先确认：

```bash
nvidia-smi
nvidia-smi topo -m
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

再使用项目环境执行：

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m exercises.day13.two_gpu_distributed_training \
  --output /tmp/day13_two_gpu.json
```

`/tmp/day13_two_gpu.json` 是示例。正式运行前必须确认路径不存在；脚本会拒绝覆盖。

## 10. 失败、中断与清理

- GPU 数量、world size、local world size 或 NCCL 不符合时，在训练前快速失败。
- 任何一个 collective 结果不符合预期时抛出错误，不继续写成功 JSON。
- 任意 rank 失败时，PyTorch 分布式 launcher 应使整个作业以非零状态结束。
- 正常或异常路径都在 `finally` 中调用 `destroy_process_group()`。
- Ctrl-C 取消属于前台 PyTorch 分布式 launcher 作业，本脚本不启动脱离终端的后台子作业。
- JSON 先写临时文件再原子替换；异常时删除自身临时文件，不留下半写成功产物。

## 11. 尚未验证与后续阶段

在真实两卡运行前，以下全部保持为未验证：

- NCCL collective 在租用 GPU 上的正确性与时间；
- PyTorch 2.13 FSDP2 当前脚本的真实运行兼容性；
- DDP/FSDP2 参数更新一致性；
- 每 rank 参数、梯度、optimizer state 分片字节；
- 小模型下 FSDP2 是否因通信和 runtime overhead 不省显存或更慢；
- PCIe/NVLink 拓扑对结果的影响；
- TP、PP、CP、EP 框架级实现；
- DeepSpeed ZeRO 框架实操。

Day 13 完成最小两卡正确性后，Day 22–24 再放大模型与 batch，运行更长测量窗口并加入
profiler，区分机制正确性、小模型 overhead 和真正的规模收益。
