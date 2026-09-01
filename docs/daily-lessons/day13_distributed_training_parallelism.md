# Day 13：从单卡训练到分布式并行

> 当前状态：讲义与两卡实验代码已生成，本机静态契约已验证；真实两卡 NCCL、
> DDP 与 FSDP2 仍待租用 GPU 验证。因此本日状态是“已生成，部分验证”，不是
> “已生成并实测”。

## 1. 今日核心问题

前 12 天的训练都可以抽象成一张 GPU 上的闭环：

    batch
    → 单卡完整模型
    → forward
    → loss
    → backward
    → 完整梯度
    → optimizer.step
    → 更新完整参数

Day 12 又把单卡显存拆成参数、梯度、optimizer state、activation、临时量与
allocator reserve。现在考虑两个逐渐出现的问题：

1. 模型能装进一张卡，但单卡处理数据太慢，怎样增加训练吞吐？
2. 模型或训练状态根本装不进一张卡，怎样把它切到多张卡？

这两个问题不能混为一谈。第一种问题首先引出 Data Parallel；第二种问题才继续引出
训练状态分片、Tensor Parallel、Pipeline Parallel、Context Parallel 和 Expert
Parallel。

今天沿下面的因果链学习：

    单卡训练
    → 多进程与多 GPU
    → 沿 batch 切分
    → 手工数据并行
    → DDP
    → DDP 的状态冗余
    → ZeRO/FSDP
    → 单层、深度、上下文或专家仍然过大
    → TP、PP、CP、EP
    → 多维混合并行

最终目标不是背六个缩写，而是面对一种资源瓶颈时，能够回答：

- 被切分的轴是什么？
- 哪些状态被复制，哪些状态被分片？
- 每张 GPU 得到什么输入？
- 为了恢复逻辑上的完整计算，需要什么通信？
- 它解决的是吞吐问题、容量问题，还是两者的一部分？

## 2. 前置知识与术语

### 2.1 从一张 GPU 到多个进程

在 PyTorch 常见的单机多卡训练中，通常采用“一进程一卡”：

    process 0 ↔ cuda:0
    process 1 ↔ cuda:1

每个进程有自己的 Python 解释器、模型对象、optimizer 对象和 CUDA context。
多卡训练不是一个 Python 对象自动跨越所有 GPU；各进程需要通过分布式通信协调。

### 2.2 Rank、world size 与 local rank

- Rank：进程在整个分布式作业中的全局编号。
- World size：整个作业中的进程总数。
- Local rank：进程在当前机器上的本地编号，通常用于选择本机 GPU。
- Node：一台机器。单机两卡只有一个 node，多机训练则包含多个 nodes。
- Process group：一组共同参加 collective 的 ranks。

本日实验固定为：

    world size = 2
    rank 0, local rank 0 → cuda:0
    rank 1, local rank 1 → cuda:1

### 2.3 Backend 与 NCCL

Backend 是分布式通信的具体实现后端。本日使用 NCCL。NCCL 的英文全称是
NVIDIA Collective Communications Library，即 NVIDIA 集合通信库，面向 NVIDIA
GPU 提供 AllReduce、AllGather、ReduceScatter、All-to-All 等 collective。

CPU/Gloo 可以帮助检查某些数学语义，但不能替代 NCCL、CUDA stream、GPU allocator、
PCIe/NVLink 拓扑和多进程 CUDA 生命周期的真实验证。

### 2.4 Collective

Collective 是一组 ranks 共同参加的通信操作。它不是 rank 0 单方面发起、其他 rank
可以忽略的普通函数调用。同一 process group 中，各 ranks 必须以匹配顺序进入相应
collective，否则可能等待、超时或整体失败。

本日不会先孤立背诵所有 collective，而是在每种切分第一次需要恢复信息时引入它。

### 2.5 并行策略名称

- DP：Data Parallel，数据并行。
- DDP：DistributedDataParallel，PyTorch 的分布式数据并行封装。
- ZeRO：Zero Redundancy Optimizer，逐级消除 data-parallel 状态冗余的策略家族。
- FSDP：Fully Sharded Data Parallel，PyTorch 的全分片数据并行方案。
- TP：Tensor Parallel，张量并行。
- PP：Pipeline Parallel，流水线并行。
- CP：Context Parallel，上下文并行。
- EP：Expert Parallel，专家并行。

这些名称不是互斥的整套系统。真实大模型可能同时使用多个并行维度。

## 3. 从直觉到机制

### 3.1 起点：单卡训练到底拥有什么

在单卡 FP32 AdamW 训练中，同一张 GPU 通常持有：

    完整参数
    完整梯度
    完整 AdamW moments
    当前 batch 的 activation
    当前算子的 temporary tensor/workspace

它完成一次独立的 forward、backward 和 optimizer update，不需要跨设备通信。

如果模型和训练状态能装入单卡，但希望单位时间处理更多样本，最自然的想法是：
让第二张 GPU 处理另一部分 batch。

### 3.2 第一步切分：沿 batch 维切数据

设 global batch 有 8 条序列：

    X_global[8,S]

其中 S 是每条序列的 token 数。两张卡均匀切分后：

    rank 0 得到 X_0[4,S]
    rank 1 得到 X_1[4,S]

这一步只切数据，没有切模型。每个 rank 必须先从相同初始参数 W 出发：

    rank 0: W + X_0 → loss_0 → gradient G_0
    rank 1: W + X_1 → loss_1 → gradient G_1

两个 local gradients 不相同，因为两个 ranks 看到了不同样本。如果各自直接调用
optimizer.step，参数会立即分叉：

    rank 0 更新为 W_0'
    rank 1 更新为 W_1'
    W_0' 通常不等于 W_1'

这还不是正确的数据并行。关键缺口是：怎样让两个 local gradients 对应同一个
global batch 的梯度？

### 3.3 手工数据并行：先理解梯度平均

若两个 local batch 大小相同，且每个 local loss 都对本地样本取 mean，则 global
batch 的平均梯度可以写成：

$$
G = \frac{G_0 + G_1}{2}
$$

两个 ranks 都必须使用同一个 G 更新参数：

    rank 0: W - learning_rate × G
    rank 1: W - learning_rate × G

只要初始参数、optimizer state 和更新顺序一致，更新后的完整参数仍然一致。

这里第一次需要 AllReduce：

    rank 0 local: [1,3]
    rank 1 local: [5,7]
    elementwise SUM: [6,10]
    divide by world size 2: [3,5]
    rank 0 result: [3,5]
    rank 1 result: [3,5]

AllReduce 的语义是：

    所有 ranks 提供输入
    → 对对应元素做 reduction
    → 所有 ranks 都得到 reduction 结果

它正好满足“聚合各 rank 梯度，并让每个 rank 得到相同结果”的需要。

必须注意：如果每个 rank 的样本数不同、loss reduction 不同，或者最后一个 batch
不均匀，简单除以 world size 不一定等于逐样本 global mean。此时要根据真实样本数
构造加权目标。

### 3.4 DDP：把手工数据并行变成可靠执行路径

手工为每个 parameter 调用 AllReduce 不够高效，也容易漏掉参数。在常见的一进程
一卡用法中，调用方先在每个 rank 创建并放置一份模型，然后用 PyTorch DDP
包装它。DDP 可在初始化时同步参数与 buffers，并在 backward 中通过 autograd
hooks 按 buckets 组织梯度同步。DDP 本身不会自动切分输入；数据切分仍由调用方或
DistributedSampler 之类的数据管道负责。[PyTorch DDP 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

两卡 DDP 的完整主线是：

    rank 0 / GPU 0                     rank 1 / GPU 1
    完整参数 W                          完整参数 W
    local batch X_0                    local batch X_1
          │ forward                          │ forward
          ▼                                  ▼
       loss_0                             loss_1
          │ backward                         │ backward
          ▼                                  ▼
    local gradients G_0 ── AllReduce ── local gradients G_1
          │                                  │
          ▼                                  ▼
    synchronized G                     synchronized G
          │ optimizer.step                   │ optimizer.step
          ▼                                  ▼
    完整参数 W'                         完整参数 W'

DDP 的核心切分与复制关系：

    被切分：global batch
    被复制：完整参数、完整梯度、完整 optimizer state
    主要通信对象：梯度
    主要目的：增加数据并行吞吐

DDP 还可以把某些梯度通信与尚未结束的 backward 计算重叠。是否真正重叠、重叠多少，
取决于模型、bucket、执行顺序和硬件，不能只根据 DDP 名称推断。

### 3.5 DDP 解决了吞吐，却没有解决单卡容量

假设模型有 P 个参数，FP32 参数、FP32 梯度和 AdamW 两个 FP32 moments 都独立存在。
忽略 activation、临时量与小型 scalar state，每个 DDP rank 的静态主体仍是：

    parameter       4P bytes
    gradient        4P bytes
    exp_avg         4P bytes
    exp_avg_sq      4P bytes
    total          16P bytes per rank

两张卡的集群总容量虽然增加，但每张卡仍需要放下这 16P bytes。两卡 DDP 不会自动
把每卡静态训练状态减半。

这暴露了 Data Parallel 的第二层问题：既然各 ranks 上的训练状态最终保持一致，
为什么每个 rank 都要长期保存完整副本？

### 3.6 从冗余出发理解 ZeRO Stage 1、2、3

ZeRO 的推导不是突然引入另一种模型并行，而是在 data-parallel group 内逐步分片
原本重复的训练状态。DeepSpeed 官方定义中，Stage 1、2、3 依次分片 optimizer
states、gradients 和 parameters。[DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)

#### 3.6.1 ZeRO Stage 1：先分 optimizer state

    每 rank 完整保存：
    - parameters
    - gradients

    在 ranks 间分片：
    - optimizer state

AdamW moments 往往占静态主体的重要部分，因此先消除 optimizer state 的重复。
但参数和梯度仍然完整复制。

#### 3.6.2 ZeRO Stage 2：再分 gradients

    每 rank 完整保存：
    - parameters

    在 ranks 间分片：
    - gradients
    - optimizer state

这时 backward 后不需要每个 rank 都长期保存完整梯度。这里自然需要
ReduceScatter：

    rank 0 input: [1,2,3,4]
    rank 1 input: [10,20,30,40]
    elementwise SUM: [11,22,33,44]
    rank 0 keeps: [11,22]
    rank 1 keeps: [33,44]

ReduceScatter 的语义是：

    先对各 rank 输入做 reduction
    → 再把结果切成 shards
    → 每个 rank 只得到自己的结果 shard

它同时完成“规约”和“只留下局部分片”。

#### 3.6.3 ZeRO Stage 3：最后连 parameters 也分片

    在 ranks 间分片：
    - parameters
    - gradients
    - optimizer state

现在每个 rank 的稳态只保存部分参数。但 forward 计算某个 Linear 或 Decoder Block
时，需要该模块逻辑上的完整参数。于是自然需要 AllGather：

    rank 0 parameter shard: [1,2]
    rank 1 parameter shard: [3,4]
    gathered logical parameter: [1,2,3,4]

AllGather 的语义是：

    每个 rank 提供自己的 shard
    → 所有 shards 按约定聚合
    → 参与计算的 rank 得到逻辑完整结果

因此 Stage 3 的关键不只是“参数除以卡数”，而是参数分片的生命周期：

    稳态参数 shard
    → 计算某个模块前 AllGather
    → 临时使用逻辑完整参数
    → 计算后释放或重新分片
    → backward 后 ReduceScatter gradient
    → 本地 optimizer state shard 更新本地 parameter shard

### 3.7 FSDP2：PyTorch 中的全分片执行

FSDP2 使用 fully_shard 表达这种全分片训练。当前实验中的参数在分片后由 DTensor
表示，并在 data-parallel device mesh 上拥有明确的 placement。[PyTorch FSDP2
fully_shard 官方文档](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)

本日代码不是只在 root model 上形成一个巨大通信组，而是：

    先 fully_shard Decoder Block 0
    再 fully_shard Decoder Block 1
    ...
    最后 fully_shard root model

这种自底向上的分组让参数 AllGather 与梯度 ReduceScatter 围绕模块边界发生。
它为通信与计算重叠提供了可能，但实际是否重叠以及峰值如何变化仍必须实测。

两卡 FSDP2 的最小图：

    rank 0 / GPU 0                     rank 1 / GPU 1
    parameter shard W_0                parameter shard W_1
             └────── AllGather W ──────┘
                    当前模块完整参数
                           │
                    forward/backward
                           │
             ┌──── ReduceScatter G ────┐
    gradient shard G_0                 gradient shard G_1
    optimizer shard O_0                optimizer shard O_1
             │                                  │
             ▼                                  ▼
    updated W_0'                        updated W_1'

FSDP2 与 DeepSpeed ZeRO-3 有相似的高层分片目标，但 API、参数表示、通信调度、
checkpoint 和状态恢复边界不同，不能把二者写成完全相同的实现。

### 3.8 为什么全分片以后还需要其他并行

FSDP/ZeRO 主要消除 data-parallel ranks 之间的训练状态冗余。但它不保证所有问题都
消失。例如：

- 单个巨型矩阵在计算时仍然太大；
- 模型层数太多，临时聚合与 activation 仍超过单卡；
- sequence 太长，单卡 attention activation 无法承受；
- MoE 的专家总权重太大，且不同 token 应由不同专家处理。

这些瓶颈分别引出 TP、PP、CP 和 EP。它们不是按“高级程度”排列，而是针对不同
结构轴。

### 3.9 TP：当单层 Tensor 需要跨卡计算

考虑 Linear：

    Y[B,S,O] = X[B,S,I] @ W[O,I]^T

B 是 batch，S 是 sequence length，I 是输入维度，O 是输出维度。

若沿 O 维把权重分成两块：

    rank 0 owns W_0[O/2,I]
    rank 1 owns W_1[O/2,I]

两卡分别计算：

    Y_0[B,S,O/2] = X @ W_0^T
    Y_1[B,S,O/2] = X @ W_1^T

如果后续需要完整 Y，则沿最后一维拼接：

    Y = concat(Y_0, Y_1, dim=-1)

两卡 TP 最小图：

    rank 0 / GPU 0                     rank 1 / GPU 1
    W_0[O/2,I]                         W_1[O/2,I]
    X replicated                      X replicated
          │                                  │
          ▼                                  ▼
    Y_0[B,S,O/2]                      Y_1[B,S,O/2]
             └──── AllGather/Concat ────┘
                         Y[B,S,O]

真实 Transformer TP 会组合 column-parallel 与 row-parallel Linear，尽量让一个算子的
输出分片直接成为下一个算子的输入分片，减少不必要的聚合。上图只是沿输出维切分的
最小例子，不是所有 TP 层固定只做一次 AllGather。

TP 解决的是单层权重与计算的容量问题，通信频率通常高，因此更依赖节点内高速互联。

### 3.10 PP：当模型深度适合按 stages 切分

若模型有 N 层，可以按深度切成两个 stages：

    GPU 0 owns layers 0 .. N/2-1
    GPU 1 owns layers N/2 .. N-1

数据流变为：

    micro-batch
    → stage 0 forward
    → send activation
    → stage 1 forward
    → loss
    → stage 1 backward
    → send activation gradient
    → stage 0 backward

如果一次只处理一个 batch，GPU 0 完成 forward 后可能等待 GPU 1。为了提高利用率，
通常把 global batch 再切成多个 micro-batches 并采用流水调度。

PP 的主要问题不再只是 collective，而是：

- stage 间点对点 activation/gradient 传输；
- pipeline bubble；
- 不同 stages 计算量或显存不均；
- micro-batch 数量与调度策略；
- tied weights 或跨 stage 状态。

PP 解决沿深度方向的容量问题，但不会自动解决单层矩阵太大或长上下文问题。

### 3.11 CP：当 sequence/context 维太长

输入 hidden states 为：

    X[B,S,H]

B 是 batch，S 是 sequence length，H 是 hidden size。两卡 CP 可以沿 S 切分：

    rank 0 owns X[:, 0:S/2, :]
    rank 1 owns X[:, S/2:S, :]

这样每张卡只长期持有部分 token positions。但 self-attention 的每个 query 可能需要
看到来自其他 sequence shards 的 key/value，因此必须交换 K/V、部分 attention 结果，
或使用具体框架定义的 ring/collective 算法。

CP 的目的主要是扩展长上下文训练。它和 TP 都会切 activation，但切分轴和通信原因
不同：

    TP：切 hidden/head/matrix 相关维度
    CP：切 sequence positions

不同框架对 Context Parallel、Sequence Parallel 的命名和实现并不完全一致。讲解某个
具体系统时必须回到该系统的官方定义。

### 3.12 EP：当 MoE experts 分布在不同 GPU

假设有四个 experts：

    rank 0 owns Expert 0, Expert 1
    rank 1 owns Expert 2, Expert 3

每个 rank 的本地 token 经 router 得到目标 expert。若 token 的目标 expert 在另一张
GPU，就必须重新分发 token：

    local tokens
    → router selects expert ids
    → All-to-All dispatch
    → token reaches target expert rank
    → local expert computation
    → All-to-All combine/return
    → token returns to sequence order

这里自然需要 All-to-All。极小例子：

    rank 0 input chunks: [0,1 | 2,3]
    rank 1 input chunks: [10,11 | 12,13]
    rank 0 receives: [0,1,10,11]
    rank 1 receives: [2,3,12,13]

All-to-All 与 AllGather 不同：每个 source rank 会向不同 destination rank 发送不同
数据块，而不是让所有 ranks 都得到同一个完整结果。

EP 的主要风险是路由不均与通信：

- 热门 expert 可能产生负载不平衡；
- 跨卡 token 越多，All-to-All 压力越大；
- 稀疏激活减少每 token 计算，不代表部署时只保存被激活 expert 的权重。

### 3.13 从单一并行到多维混合并行

当模型规模继续扩大时，一种切分往往不够。可以把 GPU ranks 组织成多维 mesh：

    DP/FSDP dimension × TP dimension × PP dimension × CP dimension × EP dimension

例如 8 张卡可以设计为：

    2-way DP × 2-way TP × 2-stage PP = 8 ranks

这里不能把每种策略的 world size 都独立理解为 8。每个并行维度只在自己的 process
group 内通信：

    DP group：同步或分片训练状态
    TP group：完成层内张量计算
    PP group：在 stages 间传 activation/gradient

并行维度越多，通信组、checkpoint、故障恢复、data sampler 和性能调优越复杂。
因此工程选择应从最小充分方案开始：

1. 模型能装单卡，只想扩吞吐：先 DDP。
2. 训练状态单卡放不下：先考虑 FSDP/ZeRO。
3. 单层矩阵仍放不下或需要层内扩展：加入 TP。
4. 模型深度适合分 stages：考虑 PP。
5. 长上下文 activation 是主要瓶颈：考虑 CP。
6. 模型是 MoE 且 experts 需要分布：考虑 EP。

这不是绝对优先级。最终方案还取决于 GPU 数量、节点边界、互联拓扑、模型结构和
框架能力。

## 4. 极小手算例子

### 4.1 两卡数据并行的 batch 与梯度

global batch 为 8，每个 rank 取得 4 条：

    rank 0 sample ids: [0,1,2,3]
    rank 1 sample ids: [4,5,6,7]

假设某一个 scalar parameter 的 local gradients 为：

    G_0 = 2
    G_1 = 6

两个 local batches 等大且各自 loss 取 mean：

    G = (2 + 6) / 2 = 4

两个 ranks 都必须用 4 更新参数，而不是分别用 2 和 6。

### 4.2 DDP 为什么不降低每卡静态训练状态

一个 1B 参数模型使用 FP32 参数、FP32 梯度和两个 FP32 Adam moments：

    1 × 10^9 × 16 bytes
    = 16 × 10^9 bytes
    = 16 GB
    ≈ 14.90 GiB per rank

两卡 DDP 的每卡主体仍是约 14.90 GiB；集群总主体约为 29.80 GiB。

### 4.3 理想全分片的容量直觉

如果上述四类状态在两个 ranks 间理想均匀分片：

    16 GB / 2 = 8 GB ≈ 7.45 GiB per rank

这只是静态主体，不是 peak allocated。它不含 activation、AllGather 临时参数、
communication buffer、workspace、allocator reserve 和 metadata。

### 4.4 四种 collective 的信息变化

    AllReduce
    每 rank 输入完整同 shape tensor
    → 每 rank 得到相同 reduction 结果

    ReduceScatter
    每 rank 输入可切分 tensor
    → reduction 后每 rank 只得到一个 shard

    AllGather
    每 rank 输入一个 shard
    → 每 rank 得到聚合后的逻辑完整 tensor

    All-to-All
    每 rank 为各 destination 准备不同 chunk
    → 每 rank 收到所有 sources 发给自己的 chunks

## 5. 正式模型与实验配置

实验复用 Day 4 的 TinyDecoderLM，不复制第二套 Transformer：

    vocabulary size     2048
    hidden size         256
    query heads         8
    FFN hidden size     768
    decoder layers      4
    sequence length     128
    global batch        8
    per-rank batch      4
    world size          2
    dtype               FP32
    optimizer           AdamW
    backend             NCCL

### 5.1 为什么实验只实作 DDP 与 FSDP2

Day 13 的知识范围包含 DP、FSDP/ZeRO、TP、PP、CP、EP，但一次实验必须保持主要变量
清楚。当前两卡实验只回答：

1. NCCL collectives 的数学语义是否与手算一致？
2. 相同初始参数与 global batch 下，DDP 和 FSDP2 一次更新是否一致？
3. DDP 与 FSDP2 的每 rank 状态字节、peak allocated、step time 和 tokens/s 有何差异？

TP、PP、CP、EP 在本日完成数据流与手算，不伪装成已经实作。它们需要各自独立的
baseline、正确性断言和目标硬件实验。

### 5.2 对照实验保持不变的条件

DDP 与 FSDP2 共享模型配置、参数初始化 seed、global token batch、batch 切分规则、
cross-entropy loss、AdamW 配置和测量步数。

一次只改变主要变量：参数、梯度和 optimizer state 的所有权及相应通信方式。

## 6. 完整数据流与 Shape/Dtype

### 6.1 Global batch 的构造与切分

    global input_ids: [8,128], int64
    global labels:    [8,128], int64

    rank 0 input_ids: [4,128], int64
    rank 0 labels:    [4,128], int64
    rank 1 input_ids: [4,128], int64
    rank 1 labels:    [4,128], int64

[B,S] 中 B 是序列条数，S 是每条序列的 token 数。两个 ranks 读取互不重叠的 rows。

### 6.2 DDP 的一步更新

    input_ids [4,128], int64
    → embedding/decoder
    → logits [4,128,2048], FP32
    → cross-entropy
    → local scalar loss, FP32
    → backward
    → gradients，shape 分别等于对应 parameters
    → DDP gradient synchronization
    → optimizer.step
    → 完整更新后 parameters

实验收集更新后的完整 CPU parameter copies，用来与 FSDP2 恢复出的完整逻辑参数
逐项比较。CPU copies 只服务于正确性验证，不计入 GPU 训练状态账本。

### 6.3 FSDP2 的一步更新

    parameter DTensor shards
    → AllGather 当前 Decoder Block 所需 parameters
    → local forward/backward
    → ReduceScatter gradients
    → local gradient shards
    → optimizer.step
    → updated local parameter shards

为了和 DDP 比较，所有 ranks 共同参与 DTensor full_tensor 操作，得到参数的完整逻辑值。
这个验证动作本身会临时聚合参数，不代表 FSDP2 稳态长期保存完整参数。

### 6.4 结果汇总

各 ranks 分别产生：

    local loss
    local parameter bytes
    local gradient bytes
    local optimizer bytes
    baseline allocated
    after-step allocated
    peak allocated
    peak delta
    average step time
    global tokens/s

rank 0 聚合这些结构化对象，并只在全部断言通过后原子写入一个 JSON。

## 7. 参数、内存、计算与通信成本

### 7.1 三种不同的扩展目标

- Throughput scaling：单位时间处理更多 tokens，DDP 常用于此目标。
- Capacity scaling：让单卡放不下的状态或计算跨卡，FSDP、TP、PP、CP、EP 分别
  处理不同容量维度。
- Step latency：一次 step 是否更短。增加通信后不保证随 GPU 数线性下降。

### 7.2 DDP 的通信量直觉

DDP backward 需要同步梯度。真实链路时间还取决于 collective 算法、PCIe/NVLink、
跨节点网络、bucket size、通信计算重叠、rank 间速度差异和 dtype。因此不能只用
参数字节除以理论带宽预测实际 step time。

### 7.3 FSDP2 的显存与通信交换

理想均匀分片可以把静态主体近似缩小为原来的 1/N，但 FSDP2 额外引入：

- 参数 AllGather；
- 梯度 ReduceScatter；
- prefetch、reshard 与模块级调度；
- communication buffers；
- 临时完整参数；
- DTensor/runtime metadata。

更细粒度分组可能降低同一时刻临时完整参数规模，也可能增加小通信和调度开销。
更粗粒度分组可能提高单次通信效率，也可能造成更高临时峰值。最优点依赖模型与硬件。

### 7.4 TP、PP、CP、EP 的主要成本

    TP：单层 tensor 分片；成本是层内高频 collective
    PP：layers 分 stages；成本是传输与 pipeline bubble
    CP：sequence activation 分片；成本是 attention 所需信息交换
    EP：experts 分片；成本是 token All-to-All 与负载不均

## 8. 最小代码验证

实验代码：

    exercises/day13/two_gpu_distributed_training.py

详细实验协议：

    docs/experiments/day13_two_gpu_experiment_design.md

### 8.1 当前环境已验证

    .venv/bin/python -m exercises.day13.two_gpu_distributed_training --validate-only

2026-08-31 当前环境实际结果：

    Static contract: PASS
    torch: 2.13.0+cu130
    distributed available: True
    NCCL compiled/available: True
    CUDA visible now: False
    visible GPU count now: 0
    real run requires: one host, world_size=2, two visible CUDA GPUs, NCCL
    real run writes: one new user-selected JSON file; no checkpoints

已核验事实：当前 PyTorch build 暴露 distributed 和 NCCL 能力，静态契约通过。

尚未核验：隔离环境当前看不到 CUDA GPU，因此没有真实 NCCL/DDP/FSDP2 数字。

### 8.2 目标双卡环境

要求是单机、两张可见 NVIDIA GPU、world size 2、NCCL backend，并让每个 rank 绑定
一张不同 GPU。

先记录：

    nvidia-smi
    nvidia-smi topo -m
    .venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

确认输出路径不存在，再运行：

    .venv/bin/python -m torch.distributed.run \
      --standalone \
      --nproc-per-node=2 \
      -m exercises.day13.two_gpu_distributed_training \
      --output /tmp/day13_two_gpu.json

使用 python -m torch.distributed.run，是因为当前项目虚拟环境中的 torchrun 入口脚本
曾保留旧项目路径。长期修复应按审批规则用 uv 重建 relocatable 环境，不能手工批量
替换 shebang。

### 8.3 正确性验收顺序

    1. 验证 world size、local world size、GPU 数量和 NCCL
    2. 精确断言四种 collectives
    3. 固定 seed 运行 DDP 一次更新
    4. 释放 DDP GPU 对象
    5. 相同 seed 运行 FSDP2 一次更新
    6. 聚合 FSDP2 完整逻辑参数并逐参数对照 DDP
    7. 运行短 benchmark
    8. 聚合每 rank 证据
    9. rank 0 原子写入 JSON

命令退出码 0 不是充分证据。必须检查 collective exact outputs、参数对照、per-rank
metrics、GPU/software identity、topology 和 named PASS checks。

### 8.4 输出、副作用与失败路径

- 输出路径由用户显式指定，父目录必须存在。
- 目标 JSON 已存在时拒绝覆盖。
- 本实验不写 model checkpoint。
- 任一断言失败时不写成功 JSON。
- 临时 JSON 先写入同目录，再原子替换最终路径。
- 正常和异常路径都尝试销毁 process group。
- Ctrl-C 取消前台 launcher，不创建后台训练作业。

## 9. 常见误解与成立边界

### 9.1 “多卡显存可以直接相加，所以模型一定能运行”

只有状态真正被对应策略分片时，集群总显存才承载同一模型的不同部分。DDP 复制完整
模型，单卡仍必须容纳完整训练状态主体。

### 9.2 “DDP 等于把一个 batch 复制到多张卡”

DDP 通常把 global batch 切成不同 local batches。重复处理相同样本不会获得预期的
数据吞吐收益。

### 9.3 “DDP 同步的是 loss”

DDP 的核心是 backward 中的 gradient synchronization。各 rank 可以有不同 local loss。
只对 loss 数值求平均不会自动让参数梯度一致。

### 9.4 “AllReduce 一定等全部 backward 完成才开始”

DDP 可以按 gradient buckets 在 backward 过程中逐步发起通信。但是否有效重叠需要
真实 profiler 证据。

### 9.5 “FSDP2 只是把所有 Tensor 永久除以 GPU 数”

参数在稳态分片，但模块计算需要临时聚合。显存峰值由分片状态、临时完整参数、
activation、communication buffer 和 workspace 的生命周期共同决定。

### 9.6 “ZeRO Stage 越高就必然越快”

更高 stage 消除更多冗余，也增加通信与状态管理复杂度。它首先改善容量，不保证速度。

### 9.7 “TP、PP、CP、EP 是 DDP 的替代品”

它们切不同轴，可以与 DP/FSDP 组合。真实系统常同时使用多种并行，而不是六选一。

### 9.8 “CPU/Gloo 通过等于两卡 NCCL 通过”

CPU/Gloo 不能证明 CUDA、NCCL、GPU topology、allocator 峰值和 FSDP2 CUDA 生命周期。

### 9.9 当前未验证边界

- 真实双卡 NCCL collectives；
- 当前 PyTorch 2.13 FSDP2 的双卡兼容性；
- DDP/FSDP2 参数更新的实际最大差异；
- 每 rank 状态字节、peak allocated、step time 和 tokens/s；
- PCIe/NVLink 拓扑影响；
- TP、PP、CP、EP 的框架级正确性与性能；
- DeepSpeed ZeRO 的代码与 checkpoint 实作；
- 多机 rendezvous、网络故障和 elastic recovery；
- distributed sampler、分布式 checkpoint 与精确恢复。

## 10. 手算练习

### 练习 1：从 local batch 到 global gradient

两个等大的 local batches，其 mean gradients 为 3 和 7：

    global mean gradient = (3 + 7) / 2 = 5

如果 rank 0 有 2 个样本、rank 1 有 6 个样本：

    weighted gradient
    = (2 × 3 + 6 × 7) / 8
    = 6

不能简单平均两个 local means。

### 练习 2：识别 ZeRO stage

    parameters: replicated
    gradients: sharded
    optimizer state: sharded

答案：ZeRO Stage 2 的高层状态所有权。

    parameters: sharded
    gradients: sharded
    optimizer state: sharded

答案：ZeRO Stage 3 类全分片目标。

### 练习 3：选择切分轴

- 模型能装单卡，只希望更高吞吐：DDP。
- AdamW state 导致容量不足：FSDP/ZeRO 是直接候选。
- 单个 Linear 权重无法在单卡计算：TP。
- 模型层数很多，适合切成 stages：PP。
- 长 sequence activation 是主要瓶颈：CP。
- MoE experts 总权重很大：EP。

这些只是第一候选，正式决策还要检查 GPU topology、框架支持和实测。

### 练习 4：判断 collective

1. 每个 rank 都得到梯度规约结果：AllReduce。
2. 规约后每个 rank 只保留梯度分片：ReduceScatter。
3. 从参数 shards 恢复逻辑完整参数：AllGather。
4. 不同 tokens 被发送到不同 expert ranks：All-to-All。

## 11. 面试口述

### 问题 1：怎样从单卡训练推导出 DDP？

30 秒回答：单卡能装模型但吞吐不足时，先沿 batch 维把数据分给多张 GPU。各 rank
从相同参数出发，对不同 local batch 得到不同 local gradients。为了让参数副本不
分叉，必须聚合梯度，再让所有 ranks 用相同梯度更新。DDP 把模型复制、梯度 bucket
和同步过程系统化。

### 问题 2：DDP 为什么不能解决模型单卡放不下？

30 秒回答：DDP 只切数据，默认在每 rank 复制完整参数、梯度和 optimizer state。
增加卡数可以扩大 global batch 和吞吐，却不会按卡数降低每卡静态训练状态。减少
这些冗余需要 ZeRO/FSDP；若其他结构轴仍太大，还要 TP、PP、CP 或 EP。

### 问题 3：ZeRO Stage 1 到 Stage 3 发生了什么？

2 分钟回答：Data Parallel 的训练状态在 ranks 间重复。Stage 1 先分 optimizer state；
Stage 2 再分 gradients，可以用 ReduceScatter 理解规约后只留下本 rank shard；
Stage 3 继续分 parameters。参数分片后，模块计算前要 AllGather 所需参数，backward
后再 ReduceScatter gradients。更高 stage 降低静态冗余，却增加通信、临时参数生命
周期、checkpoint 和调度复杂度，所以省显存不等于一定更快。

### 问题 4：TP、PP、CP、EP 怎样区分？

2 分钟回答：先看切分轴。TP 切单层 tensor 维度，PP 切模型 layers/stages，CP 切
sequence positions，EP 切 MoE experts 并用 All-to-All 路由 token。它们可以与
DP/FSDP 组合。选择时要同时说明状态所有权、缺失信息、通信方式和硬件拓扑。

## 12. 当日验收

### 12.1 基础知识与机制

- [ ] 能从单卡闭环推导出沿 batch 切分的数据并行。
- [ ] 能解释不聚合 gradients 为什么会让参数副本分叉。
- [ ] 能解释 rank、world size、local rank、process group 和 NCCL。
- [ ] 能区分吞吐扩展、容量扩展和 step latency。
- [ ] 能解释 DDP 切什么、复制什么、同步什么。
- [ ] 能按状态所有权说出 ZeRO Stage 1、2、3。
- [ ] 能解释 FSDP2 的 AllGather/ReduceScatter 生命周期。
- [ ] 能从具体瓶颈选择 TP、PP、CP 或 EP。

### 12.2 画图与手算

- [ ] 能画出 DDP、FSDP/ZeRO-3、TP、EP 四张两卡图。
- [ ] 能手算等 batch 与不等 batch 的 global gradient。
- [ ] 能手算 DDP 与理想全分片的静态状态主体。
- [ ] 能区分 AllReduce、ReduceScatter、AllGather、All-to-All。

### 12.3 当前环境验证

- [x] Day 13 实验脚本静态契约通过。
- [x] 当前 PyTorch build 可导入 distributed 与 NCCL 接口。
- [x] 明确记录当前隔离环境 CUDA 不可见，没有伪造两卡结果。

### 12.4 真实两卡验证（待完成）

- [ ] 两张 CUDA GPU 分别绑定两个 ranks，NCCL process group 初始化成功。
- [ ] 四种 collective 精确结果断言通过。
- [ ] DDP 与 FSDP2 一次参数更新在定义容差内一致。
- [ ] JSON 记录每 rank 状态字节、allocator peak 和 benchmark。
- [ ] JSON 记录 GPU、compute capability、PyTorch/CUDA/NCCL 与 topology。
- [ ] 解释 FSDP2 是否兑现显存收益，以及是否因小模型 overhead 更慢。

双卡验证完成后，把 JSON 实测数字回写本讲义，并把 Day 13 状态更新为
“已生成并实测”。在此之前，TP、PP、CP、EP 只标记为基础机制覆盖，不标记为
框架级实作完成。
