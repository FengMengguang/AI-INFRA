# Day 16：TP、PP、CP 与 SP 的模型并行布局

> 状态：已生成，部分验证。PyTorch 2.13.0+cu130 的双进程 CPU/Gloo 正确性实验已通过；双 GPU/NCCL 通信时间线、显存和性能仍待目标环境验证。

## 1. 今日核心问题

DDP 切 global batch，但每张 GPU 仍保存完整模型。当模型本身放不下，或单个算子的计算量太大时，需要进一步回答：

- Tensor Parallel（TP）怎样切一个 Linear 或 Attention？
- Pipeline Parallel（PP）怎样把连续层分给不同设备？
- Sequence Parallel（SP）和 Context Parallel（CP）为什么都切 sequence，却不是同一件事？
- 为什么 TP 特别依赖 NVLink/NVSwitch 等高带宽互连？

本日使用 PyTorch 框架接口验证布局，不用手写 shard 冒充框架实现。

## 2. 前置知识与术语

- **rank**：一个分布式进程的编号；本日固定两个 rank。
- **DeviceMesh**：把 ranks 组织成有名称的设备网格。
- **DTensor**：同时记录全局逻辑 shape、本地 shard 和 placement 的分布式张量。
- **collective**：一组 ranks 共同参与的通信，如 All-Reduce、All-Gather、Reduce-Scatter。
- **TP**：在一个层或张量内部切分计算。
- **PP**：按连续层切成 stage，stage 间传 activation。
- **SP**：让部分逐 token 算子持有 sequence shard。
- **CP**：分片上下文 token，同时用通信保持全局 Attention 语义。

本文采用 `nn.Linear` 的约定。权重 `W[O,I]` 中，`O` 是输出特征数，`I` 是输入特征数：

$$
Y = XW^T
$$

## 3. 从直觉到机制

### 3.1 Column-wise：切输出特征

把 `W[O,I]` 沿第 0 维切开。两个 rank 各保存一半输出行，但都读取完整输入：

```text
rank 0: W_0[O/2,I] -> Y_0[...,O/2]
rank 1: W_1[O/2,I] -> Y_1[...,O/2]
```

拼接两个 shard 才是完整输出；若下一层能直接消费 shard，就不必立即 All-Gather。

### 3.2 Row-wise：切输入特征

把权重沿第 1 维切开，同时切输入最后一维：

$$
Y = X_0W_0^T + X_1W_1^T
$$

每个 rank 先得到对完整输出的局部贡献，再通过 All-Reduce 求和得到复制输出。若目标仍是 shard，也可通过相应的 Reduce-Scatter 做布局转换。

### 3.3 MLP 为什么常用 Column → Row

```text
复制 X[...,H]
-> up_proj column-wise
-> 每 rank [...,F/2]
-> 本地 SiLU
-> down_proj row-wise
-> All-Reduce 局部贡献
-> 复制 Y[...,H]
```

中间宽激活不需要先拼回完整 `F`，避免一次不必要的物化和通信。

### 3.4 Attention 的 head split

若有 4 个 heads、两个 ranks，每个 rank 可计算 2 个 heads。Q/K/V projection 使用 column-wise，使输出 shard 对应 head shard；各 rank 独立完成本地 heads 的 Attention；`o_proj` 使用 row-wise 汇总各 head 的贡献。

这不是“一个 rank 保存 Q，另一个保存 K”。每个 rank 都需要其本地 heads 对应的 Q、K、V。

### 3.5 Pipeline Parallel：切连续层

```text
forward:  stage 0 -> activation -> stage 1 -> loss
backward: stage 0 <- activation gradient <- stage 1
```

把 batch 切成多个 **micro-batches**，不同 stage 可同时处理不同 micro-batch。GPipe 先推进所有 forward，再执行 backward；1F1B 在预热后交替执行一次 forward、一次 backward，通常降低 activation 驻留量。

若有 `p` 个 stage、`m` 个 micro-batches，只看理想化单向流水的槽位，空泡比例常写成：

$$
\frac{p-1}{m+p-1}
$$

它只说明增加 micro-batch 可以摊薄填充/排空，不是完整训练延迟公式。真实结果还受前后向不均衡、通信和调度影响。

### 3.6 Sequence Parallel：保留 sequence shard

LayerNorm、RMSNorm、Dropout 等逐 token 算子通常不需要看到其他 token。`X[B,S,H]` 沿 `S` 切开后，各 rank 可对 `X_r[B,S/2,H]` 独立计算，同时参数仍复制。

本日 PyTorch `SequenceParallel` 只包裹 `LayerNorm`。它不会自动让普通 self-attention 在缺少其他 token 时获得全局上下文。

### 3.7 Context Parallel：分片上下文但保持全局 Attention

CP 也沿 sequence 切输入，但每个 query 仍需要访问全局所需的 K/V，所以框架必须通信 K/V 或中间结果。本日使用实验性 `context_parallel()`：它临时分片指定 buffer，并改写兼容的 SDPA 执行路径。

不同框架可能把某些 sequence-sharding 技术统称为 SP。本文的 SP/CP 特指上述 PyTorch API 语义，不能仅凭同名跨框架等同。

### 3.8 TP 为什么对拓扑敏感

TP collective 位于许多层内部，频率高，消息规模通常随 `B*S*H*dtype_bytes` 增长。慢链路会让每层反复等待通信，所以通常把同一 TP group 放在 NVLink/NVSwitch 域内。跨节点更常优先考虑 PP 或 DP，但这只是工程启发，不是绝对规则。

## 4. 极小手算例子

设 `X[1,2,4]`，Linear 权重 `W[6,4]`，两个 ranks。

```text
Column-wise:
每 rank 权重 [3,4]
每 rank 输入 [1,2,4]，复制
每 rank 输出 [1,2,3]
完整逻辑输出 [1,2,6]

接 down_proj W[4,6] 的 Row-wise:
每 rank 权重 [4,3]
每 rank 输入 [1,2,3]
每 rank 局部贡献 [1,2,4]
All-Reduce 后 [1,2,4]
```

若 Attention 有 4 heads、每 head 维度 2，则隐藏维度是 8；两个 ranks 各计算 2 heads，本地 Q/K/V 为 `[B,2,S,2]`。

## 5. 正式实验配置

- world size：2；
- 本机正确性：CPU/Gloo、FP32；目标性能：双 GPU/NCCL；
- TP MLP：`8 -> 16 -> 8`；
- head split：4 heads、head dimension 4；
- SP：`LayerNorm(8)`，sequence length 6；
- PP：两个 stage、两个 micro-batches；
- CP：SDPA，sequence length 6。

## 6. 完整数据流与 Shape/Dtype

### 6.1 TP MLP 与 Head Parallel

TP 输入 `hidden[2,3,8]` 为 FP32，分别表示 batch 2、sequence 3、hidden size 8。`up_proj` 后每 rank 本地输出 `[2,3,8]`，全局逻辑输出 `[2,3,16]`；`down_proj` 汇总后输出 `[2,3,8]`。

Attention 输入 `[2,5,16]`；全局 4 heads，每 rank 2 heads。本地 Q/K/V 为 `[2,2,5,4]`，本地 context 合并为 `[2,5,8]`，`o_proj` 后复制输出 `[2,5,16]`。

### 6.2 SP、CP 与 PP

SP 完整输入 `[2,6,8]` 沿 sequence 切成每 rank `[2,3,8]`。CP 的 Q/K/V 全局逻辑 shape 为 `[1,2,6,4]`，sequence shard 位于第 2 维。逻辑全局 shape 不代表每 rank 都物化完整 storage。

PP 完整输入 `[4,8]` 切成两个 `[2,8]` micro-batches。stage 0 输出 `[2,8]` 并发送；stage 1 输出 `[2,4]`，计算 MSE loss；activation gradient 在 backward 中逆向返回。

## 7. 参数、内存与计算成本

- TP 减少目标层的单 rank 参数和矩阵乘法份额，但增加层内 collective。
- PP 减少单 rank 层数，但要传 stage 边界 activation，并存在 bubble。
- SP 减少可分片逐 token activation；参数是否分片取决于与 TP/FSDP 的组合。
- CP 分摊长上下文 Attention，但必须为全局语义支付通信。

“参数恰好减半”只对可整除且纳入 plan 的张量成立。embedding、norm、bias、未切模块和通信 buffer 可能仍复制。

## 8. 最小代码验证

```bash
uv run python -m exercises.day16.model_parallelism --validate-only

uv run python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day16.model_parallelism \
  --backend gloo \
  --output /tmp/day16_gloo.json
```

2026-09-01 实测：TP MLP 输出最大差异 `5.960464477539063e-08`，完整梯度最大差异 `3.958120942115784e-09`；head split 输出最大差异 `6.705522537231445e-08`；SP 与 CP 输出差异均为 `0.0`；两个 PP stage 都获得参数梯度，末级平均 loss 为 `0.886401355266571`。

目标双 GPU 命令：

```bash
uv run python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m exercises.day16.model_parallelism \
  --backend nccl \
  --output /tmp/day16_nccl.json
```

目标成功标准是全部正确性检查 PASS；性能实验还需 profiler 区分计算、collective 和空泡时间。当前程序不是 benchmark，不能据其运行时间声称加速。

## 9. 常见误解与边界

1. Column/Row 描述权重的切分维，不是内存中的视觉方向。
2. `ColwiseParallel` 后可保留 DTensor shard，不必立即得到普通完整 Tensor。
3. head parallel 不是把 Q/K/V 三种投影分给三张卡。
4. micro-batch 不等于 gradient accumulation；二者可以组合。
5. SP 不等于 CP：本地 LayerNorm 不需要其他 token，global self-attention 需要。
6. CPU/Gloo PASS 证明当前数值语义，不证明 NCCL kernel、GPU 显存、拓扑或性能。
7. PyTorch CP API 是 experimental，未来接口和支持范围可能变化。

## 10. 手算练习

隐藏维度 `H=12`、MLP 中间维度 `F=24`、TP size 3：

1. `up_proj[24,12]` column-wise 后每 rank 权重和输出最后一维是多少？
2. `down_proj[12,24]` row-wise 后每 rank 输入和权重是什么 shape？
3. 6 个 attention heads 如何分配？

答案：每 rank `up_proj[8,12]`，本地输出最后一维 8；每 rank `down_proj[12,8]`，输入最后一维 8；每 rank 2 heads。

## 11. 面试口述

### 三道口述题

1. 为什么 MLP 常采用 column-wise 后接 row-wise？
2. GPipe 的 micro-batch 如何减少 pipeline bubble？
3. PyTorch 本日语境下 SP 和 CP 的差别是什么？

### 30 秒版本

TP 切层内张量：column-wise 产生输出 shard，row-wise 汇总局部贡献，MLP 和 Attention 可让中间 shard 连续流动。PP 把连续层分成 stage，用 micro-batch 形成流水线。SP 让逐 token 算子保留 sequence shard，CP 则为分片上下文补足 Attention 的全局通信。TP collective 频繁，因此尤其依赖高带宽互连。

### 2 分钟版本要求

补充 `Y=XW^T` 的两种切法、Q/K/V 与 o_proj 的 head split、PP 正反向通信、bubble 公式边界，以及 CPU/Gloo 正确性与 NCCL/GPU 性能证据的区别。

## 12. 当日验收

- [x] 画出 column-wise → activation → row-wise 数据流。
- [x] 写出两卡 TP MLP 和 head split 的本地/global shape。
- [x] 用 PyTorch framework API 完成 TP、SP、CP、PP 双进程正确性验证。
- [x] 结果原子写入且拒绝覆盖；错误 world size 被拒绝。
- [x] Day 15 静态契约回归通过。
- [ ] 在真实双 GPU 上验证 NCCL 通信时间线、显存和 pipeline bubble。
- [ ] 验证 causal CP 和不同 micro-batch 数的性能拐点。

尚未确认：当前小配置的通信量远小于真实 LLM，不能回答租用设备的 NVLink/PCIe 拓扑下何时获得净加速。

下一步是 Day 17：Expert Parallel（EP）与 DP/TP/PP/CP 的组合并行，建立多维 DeviceMesh 和通信组边界。
