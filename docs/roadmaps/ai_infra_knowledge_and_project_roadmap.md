# AI Infra 知识架构、学习路线与项目规划

> 版本：2026-08-31  
> 定位：从 Transformer 基础出发，逐步建立分布式训练、现代 LLM 架构、推理引擎、性能观测和集群部署能力。  
> 简历项目：项目一 `MemScope`；项目二 `Mini-SGLang Memory-Aware Serving`。  
> 当前状态：Day 1–12 已完成并实测；Day 13–16 讲义已生成。Day 14–16 的 Gloo
> 双进程正确性已实测；Day 13–16 的真实双 GPU 行为尚未验证。

## 1. 总目标与事实边界

### 1.1 最终能力

完成本路线后，应能够：

1. 从文本、token、Transformer、loss、backward 到 optimizer update，解释完整训练数据流。
2. 从在线请求、prefill、KV Cache、decode、sampling 到流式返回，解释完整推理生命周期。
3. 估算并实测参数、梯度、优化器状态、激活、KV Cache、缓存池和运行时开销。
4. 说明 DDP、FSDP/ZeRO、TP、PP、CP、SP、EP 分别切分、复制和通信什么。
5. 理解 GQA、MLA、MoE、MTP、长上下文、量化和投机解码如何改变训练与推理系统。
6. 阅读 Nano-vLLM 和 Mini-SGLang 的请求、调度、缓存、执行和释放路径。
7. 使用 profiler、运行时指标和结构化实验定位 compute、memory、communication 与 scheduling 瓶颈。
8. 在 Kubernetes 中解释模型服务的部署、GPU 调度、扩缩容、滚动升级和故障恢复。
9. 完成两个具有明确问题、真实实验、失败边界和可复现产物的简历项目。

### 1.2 不追求的目标

- 不从零重写工业级训练框架、vLLM 或完整 SGLang。
- 不将阅读模型报告的数量视为学习进度。
- 不在单卡结果上声称多节点、多 GPU 或生产集群结论。
- 不同时实现所有前沿技术；每个阶段只选择能闭环验证的最小范围。
- 不为了项目名称而重复成熟开源项目已经完成的通用能力。

### 1.3 唯一主结构：模型生命周期

```text
数据与模型前向
→ 训练状态与反向传播
→ 分布式训练和通信
→ 现代模型架构
→ 推理引擎与 KV Cache
→ 解码、调度和 Serving 优化
→ 显存观测与系统优化
→ Kubernetes 与生产控制面
```

MoE、MLA、投机解码等技术不作为孤立名词学习，而要分别回答：数据如何流动、状态由谁持有、显存如何变化、需要什么通信、何时有效、何时失效。

## 2. 已完成基础：Day 1–13

### 2.1 Day 1–7：Transformer 与训练闭环

- Day 1：Tensor shape、参数量与理论显存。
- Day 2：文本/token IDs 到 logits。
- Day 3：Causal Attention、RMSNorm、residual。
- Day 4：Decoder Block、SwiGLU、Cross-Entropy、backward。
- Day 5：Dataset、AdamW、tiny-data overfit。
- Day 6：梯度累积、checkpoint、精确 resume。
- Day 7：第一阶段综合验收。
- Day 7.5：FP32/FP16 AMP、Autocast、GradScaler。

### 2.2 Day 8–12：推理基础与显存账本

- Day 8：朴素自回归生成、prefill、无缓存 decode。
- Day 9：各层 KV Cache、增量 decode、cache/no-cache 一致性。
- Day 10：MHA/MQA/GQA、Query/KV heads、KV Cache 容量。
- Day 11：FlashAttention、Activation Checkpointing、IO-aware 思维。
- Day 12：参数、梯度、AdamW state、activation 与 allocator 显存账本。

### 2.3 Day 13：分布式并行总览

Day 13 讲义已经生成，覆盖 DDP、FSDP/ZeRO、TP、PP、CP、EP 和 NCCL collectives。

当前完成边界：

- 概念、数据流和单机可验证部分已经进入讲义；
- 真实 DDP/FSDP 多卡显存、吞吐和通信行为仍待双卡环境验证；
- 后续课程不能把“讲义已生成”写成“多卡性能已经验证”。

## 3. 阶段一：分布式训练与 CS336 Systems

### 3.1 学习目标

从 Day 13 的横向总览进入每种并行策略的执行细节，并建立性能分析能力。

### 3.2 内容安排

#### Day 14：DDP 与 NCCL

- rank、process group、gradient bucket。
- All-Reduce 与 Reduce-Scatter 的数据流。
- 梯度同步和计算重叠。
- 单机多进程正确性实验。

验收：解释每个 rank 拥有什么，何时发生通信；有双卡时补充时间线证据。

#### Day 15：FSDP 与 ZeRO

- 参数、梯度和优化器状态的切分。
- All-Gather、Reduce-Scatter。
- sharding strategy、reshard 和 mixed precision。
- distributed checkpoint 与 world-size 边界。

验收：理论账本和实测显存分别记录；解释“省显存但可能更慢”。

#### Day 16：TP、PP、CP 与 SP

- 行并行、列并行、head 切分。
- Pipeline stage、micro-batch 和 bubble。
- Context/Sequence Parallel 的术语差异。
- 高带宽互联对 TP 的重要性。

验收：画出两卡张量布局和主要 collective，不把不同框架中的同名策略混为一谈。

#### Day 17：EP 与组合并行

- Expert Parallel 只先学习通信和布局基础。
- DP×TP×PP×CP×EP 的 rank group。
- 并行策略选择与硬件拓扑。
- 通信量、显存和负载不均衡。

验收：为 Dense 与 MoE 各设计一套并行配置，并说明未知的实测边界。

#### Day 18：CS336 Systems 精选

对应 Stanford CS336 Systems 中与当前主线直接相关的内容：

- profiling 与 benchmark 方法；
- memory-efficient training；
- Triton 基础；
- FlashAttention-2 机制与最小实现；
- distributed training。

不要求顺序照搬全部作业；优先复用本项目已有模型和实验，避免重复造第二套课程代码。

### 3.3 阶段验收

- 能为 DDP、FSDP、TP、PP、CP/SP、EP 写出状态所有权和通信图。
- 能区分理论通信量、profiler 观测和端到端性能。
- 至少保留一个并行或 kernel 优化没有加速的反例。
- 多卡不可用时，明确列出未验证项，不用单卡模拟代替真实结论。

## 4. 阶段二：现代 LLM 架构

本阶段按“系统问题”组织，而不是按模型排行榜组织。

### 4.1 Attention 与长序列状态

按以下顺序学习：

1. MHA、MQA、GQA 复盘；
2. MLA（Multi-head Latent Attention）；
3. Sliding Window 与 Local/Global Attention；
4. Linear/Hybrid Attention；
5. RoPE scaling、YaRN 与长上下文；
6. Context Parallelism 与长 prefill。

每种机制必须说明：

- Q/K/V 或替代状态如何产生；
- 训练与 decode 保存什么；
- 每 token 状态量如何增长；
- 对 FlashAttention、KV Cache layout 和并行策略有什么影响；
- 框架是否已有可用 kernel，还是只有数学描述。

### 4.2 MoE 架构与系统

MoE 是独立专题，覆盖：

- Dense FFN 与 MoE FFN；
- Router、routing logits 与 Top-K；
- Routed Expert 与 Shared Expert；
- token dispatch、expert compute、token combine；
- capacity、dropping、load imbalance；
- auxiliary loss 与 auxiliary-loss-free balancing；
- Expert Parallelism 与 All-to-All；
- 专家权重、路由结果、dispatch/combine buffer 和激活显存；
- prefill 与 decode 阶段的 MoE Serving 特征。

代表架构选择 DeepSeek、Qwen、Llama、Kimi 等官方公开模型，但只把官方资料明确披露的内容写成模型事实。

### 4.3 Multi-Token Prediction

- 普通 next-token prediction 与多未来 token 监督。
- 辅助预测头的输入、标签、loss 和训练成本。
- 推理时辅助头是否保留必须依据具体实现核对。
- MTP 与 speculative decoding 有关联，但不能默认等同。

### 4.4 低精度与量化

- FP16、BF16、FP8、INT8、INT4。
- Weight-only、Weight-Activation、KV Cache quantization。
- per-tensor、per-channel、per-token、per-block scaling。
- PTQ、QAT、GPTQ、AWQ、SmoothQuant。
- dequantization、kernel 支持、质量、显存和吞吐的联合验证。

### 4.5 模型架构雷达

建立持续更新的模型卡片，而不随每次发布重排课程。每个模型统一记录：

```text
模型类型与发布时间
→ Dense / MoE / Hybrid
→ Attention 与位置编码
→ 总参数 / 激活参数
→ KV Cache 或替代状态
→ 精度与量化格式
→ 训练并行与推理引擎支持
→ 权重、代码、数据、报告和 license 的开放边界
→ 可用于 MemScope / Serving 的实验问题
```

第一批候选：Qwen3、DeepSeek-V3、Llama 4、Kimi K2，以及一个 Linear/Hybrid Attention 模型。具体“最新”版本在写专题时重新从官方入口核验。

### 4.6 阶段验收

- 完成 MLA、MoE、MTP 的最小机制实验。
- 对 Dense/GQA/MLA/MoE 分别建立参数、状态、计算和通信账本。
- 不要求在本地加载超大权重；允许用缩小配置验证机制。
- 不把缩小复现的结果推广为原模型性能。

## 5. 阶段三：MemScope v0.1 与 v0.2

### 5.1 项目定位

`MemScope` 是面向 LLM 训练和推理生命周期的 GPU 显存归因与优化工具。它不替代 PyTorch Profiler、Memory Snapshot、NVML 或 Nsight，而是统一它们能确认的证据，并诚实保留无法精确归因的差值。

### 5.2 能力契约

用户给定模型、运行阶段和实验配置后，能够：

- 查看理论显存账本；
- 查看 PyTorch allocator 的 allocated、reserved 和 peak；
- 查看参数、梯度、优化器状态与激活分类；
- 对照 NVML 进程级总显存；
- 导出 JSON 和 HTML 报告；
- 比较两个实验配置的显存差异；
- 将不能确认的内存显示为 `unattributed`，而不是强行归类。

### 5.3 两种运行模式

#### Deep Profile Mode

- 用于短实验；
- 允许采集完整 trace、memory snapshot 和 stack；
- 能提供更细归因，但开销较高。

#### Production Monitor Mode

- 用于较长训练或服务；
- 只采集低频、低开销指标；
- 不承诺拥有 Deep Profile 的精细归因。

必须测量观测工具自身的时间和显存扰动，避免“为了观测而改变被观测系统”。

### 5.4 v0.1：单卡 PyTorch 训练

覆盖：

- 参数、梯度、optimizer state；
- activation 与临时 tensor；
- allocator allocated/reserved/peak；
- NVML 进程总占用；
- 理论值、框架观测与系统观测的差异。

实验矩阵：

- FP32/FP16 AMP；
- activation checkpointing；
- gradient accumulation；
- batch size 与 sequence length；
- MHA/GQA/MQA。

### 5.5 v0.2：分布式与现代架构

分布式适配：

- DDP/FSDP per-rank 显存；
- collective buffer 与 checkpoint 峰值；
- world-size、rank 和设备拓扑记录。

现代架构适配：

- MLA 与普通 KV Cache；
- MoE 全部专家权重与激活专家计算；
- router 输出与 expert token load；
- dispatch/combine buffer；
- MTP 辅助头；
- KV Cache quantization。

NCCL、CUDA Graph、第三方 kernel workspace 等非 PyTorch allocator 分配可能无法精确分类；第一版只报告系统总量与未归因差值。

### 5.6 核心验收

1. 同一配置重复运行，分类结果和峰值趋势可解释。
2. 各分类之和与 allocator/NVML 口径的关系写清楚。
3. AMP、checkpointing、GQA 等已知变量产生方向正确的变化。
4. profiler 开关本身的扰动得到测量。
5. OOM 前后的峰值、失败原因和报告产物完整。
6. 至少保留一个理论节省没有转化为端到端收益的案例。

## 6. 阶段四：Nano-vLLM 与推理引擎基础

### 6.1 学习顺序

先完整阅读 Nano-vLLM，再进入 Mini-SGLang。Nano-vLLM 用于建立一个可读的推理引擎心智模型，不作为简历项目名称。

```text
Request
→ Scheduler
→ Prefill
→ KV Cache allocation
→ Decode
→ Sampling
→ Completion / Cancellation
→ Cache release
```

### 6.2 必须追踪的数据流

- 请求在哪里创建、排队和结束；
- scheduler 每轮读取什么状态；
- token budget 如何计算；
- KV block/page 如何分配、引用和释放；
- prefix cache 如何命中；
- prefill 与 decode 如何组成 batch；
- CUDA Graph 与 eager 路径的状态边界；
- 请求取消、抢占和失败是否会遗留缓存。

### 6.3 核心 Serving 技术

按顺序学习：

1. static batching 与 continuous batching；
2. Paged KV Cache；
3. prefix caching；
4. chunked prefill；
5. preemption 与 recomputation；
6. CUDA Graph 与 `torch.compile`；
7. KV Cache quantization；
8. prefill/decode disaggregation。

### 6.4 指标与负载

- TTFT、TPOT、ITL、E2E latency；
- requests/s、input/output tokens/s；
- goodput；
- P50/P95/P99；
- accepted、rejected、failed、timed-out、cancelled；
- KV Cache occupancy、prefix hit 和 preemption；
- 峰值显存和 OOM。

实验必须覆盖短/长输入、短/长输出、低/高并发、共享/随机前缀和稳定/突发到达。

## 7. 阶段五：投机解码与前沿 Serving

### 7.1 Draft–Verify 基础

```text
Draft 生成 k 个候选 token
→ Target 一次前向验证多个位置
→ 接受合法前缀
→ 拒绝位置修正采样
→ 提交、截断或回滚 KV Cache
→ 进入下一轮
```

学习时必须分别说明 greedy 和 sampling，并核对“无损”保证成立的算法与数值边界。

### 7.2 技术族

建议顺序：

1. 独立小模型 Draft；
2. Prompt lookup / n-gram speculation；
3. Self-speculative decoding；
4. Medusa；
5. EAGLE 系列；
6. MTP-based speculation；
7. Tree-based verification。

统一比较：候选来源、候选长度、验证方式、接受率、额外权重、临时状态、KV Cache 提交/回滚和 batching 兼容性。

### 7.3 实验指标

- acceptance rate；
- accepted tokens per verification；
- draft/verify 时间占比；
- TTFT、TPOT、throughput、P99；
- 额外模型与 KV Cache 显存；
- 每输出 token 的成本；
- 不启用 speculative decoding 的 baseline。

### 7.4 反向压力测试

必须覆盖至少三类不加速场景：

- Draft 成本过高或接受率低；
- 大 batch 下 Target 已充分利用 GPU；
- 额外显存降低可容纳并发；
- 不同请求接受长度造成调度碎片；
- KV Cache 回滚和验证开销抵消收益。

### 7.5 其他前沿 Serving 专题

- MLA inference；
- MoE inference 与 Expert Parallel Serving；
- FP8/INT8 权重和 KV Cache；
- Multi-LoRA serving；
- P/D disaggregation；
- CPU/NVMe offload。

这些专题先达到 L1/L2；只有与两个项目直接相关的部分进入 L3 工程实现。

## 8. 阶段六：Mini-SGLang 与简历项目二

### 8.1 项目名称与定位

项目暂定名：`Mini-SGLang Memory-Aware Serving`。

它不是重新实现 Mini-SGLang，而是在保留其现有能力的基础上，增加一个经过代码审计确认的显存压力感知扩展。

### 8.2 开工前代码审计

在确定最终 patch 前，必须沿数据流核对 Mini-SGLang 的：

- request lifecycle；
- scheduler 与 token budget；
- Radix Cache；
- chunked prefill；
- overlap scheduling；
- CUDA Graph；
- request cancellation；
- cache release、eviction 和 preemption；
- metrics 与 benchmark。

只有确认现有实现缺口后，才锁定扩展点。不能根据模块名推断系统没有某项能力。

### 8.3 第一版核心问题

推荐第一版只解决：

> 如何使用 KV Cache 和整体显存压力信号，改善请求准入或调度，从而减少 OOM/抢占，并在吞吐、TTFT、TPOT 与 P99 之间取得可解释的权衡？

MemScope 负责产生可观测信号；Mini-SGLang 扩展负责消费信号并作出调度决定。

### 8.4 候选路线

代码审计后从以下路线中只选择一条：

1. cache-pressure-aware admission；
2. outstanding-token budget；
3. memory-aware batching；
4. cancellation 后的缓存清理强化；
5. proactive overload protection；
6. long/short request isolation。

### 8.5 第二阶段扩展

第一版稳定后，可选择一个扩展：

- 投机解码感知调度：考虑 Draft/Target 显存、接受率和验证成本；
- MoE Serving 观测：分析 expert load 与通信，但真实优化需要多 GPU；
- P/D 分离下的 KV 压力和资源配比。

不同时实现三项。

### 8.6 对照实验

- 原始 Mini-SGLang baseline；
- 新策略；
- 相同模型、请求集、随机种子和硬件；
- 短/长请求、稳定/突发负载、低/高显存压力；
- throughput、goodput、TTFT、TPOT、P99、OOM、preemption、rejection；
- 策略自身 CPU 和 GPU 开销。

### 8.7 行为验收

1. 每个请求都有唯一终态。
2. 取消、失败和拒绝不会泄漏 KV block。
3. 显存压力信号有明确来源和更新时间。
4. 高压力下不以无限排队伪造低失败率。
5. 同时报告延迟、吞吐、拒绝率和 goodput。
6. 新策略无收益或退化的负载必须保留。
7. 所有结论能回到 request-level events 和显存记录。

## 9. 阶段七：Kubernetes 与生产控制面

Kubernetes 放在理解本地推理生命周期之后，避免只会写 YAML 而不了解 worker 内部状态。

### 9.1 容器与 Kubernetes 基础

- image、container、volume 和 network；
- Pod、Deployment、Service；
- ConfigMap、Secret；
- requests/limits；
- readiness、liveness、startup probe；
- rolling update 与 rollback。

### 9.2 GPU 与模型服务

- NVIDIA Container Toolkit；
- Device Plugin 与 GPU Operator；
- node label、affinity 和 topology；
- GPU 资源发现和调度边界；
- 模型加载、warmup、readiness；
- 请求 draining 与优雅退出；
- metrics、logs、traces。

### 9.3 控制面与扩展

- 多副本路由与 autoscaling；
- prefix-aware routing；
- 模型版本与滚动升级；
- worker 故障恢复；
- AIBrix、KServe、llm-d 等项目的职责边界。

### 9.4 本地验证边界

本地单 GPU Kubernetes 可以验证部署生命周期、配置、健康检查和升级，但不能证明多节点 GPU 调度、RDMA/NCCL 性能或生产扩缩容效果。

## 10. 两个简历项目的关系

```text
现代模型与训练/推理实验
        ↓
MemScope
产生显存分类、KV Cache 占用和压力信号
        ↓
Mini-SGLang Memory-Aware Serving
使用这些信号进行 admission / scheduling
        ↓
Kubernetes
验证服务生命周期与控制面集成
```

### 10.1 项目一：MemScope

突出能力：

- GPU memory observability；
- 理论账本与多来源实测对齐；
- 训练/推理跨生命周期归因；
- 架构和优化手段的可重复对照；
- 对不可归因内存保持诚实边界。

### 10.2 项目二：Mini-SGLang Memory-Aware Serving

突出能力：

- 阅读并扩展成熟推理引擎；
- 请求、scheduler 和 cache 生命周期；
- 显存压力驱动的系统决策；
- overload、cancellation 和 failure handling；
- 吞吐、延迟、goodput 和稳定性的联合评估。

### 10.3 启动门槛

- MemScope v0.1：Day 14–18 后启动。
- MemScope v0.2：完成现代架构专题后启动。
- Nano-vLLM：MemScope v0.1 已能输出稳定报告后开始。
- Mini-SGLang 代码审计：完成 Nano-vLLM 生命周期追踪后开始。
- 简历项目二实现：审计确认真实缺口并冻结验收矩阵后开始。
- Kubernetes：本地 Serving 的启动、取消、失败和清理路径已经理解后开始。

## 11. 推荐时间安排

原路线的“一个月”不足以同时完成知识主干和两个工程项目，但也不要求学习四个月后
才能求职。时间安排拆成：

```text
8 周求职核心路线
+
8 周可选滚动增强
=
最长 16 周的完整路线
```

Day 1–13 已经完成或进入待验证状态，不重新计入下面从当前开始计算的 8 周。进度仍以
验收为准；某周未通过核心验收时，应缩小实验规模或把工程增强后置，不能删除知识主题。

### 11.1 前 8 周：求职核心路线

#### 第 1 周：分布式与 CS336 Systems

- 完成 Day 14–18；
- DDP/NCCL、FSDP/ZeRO、TP、PP、CP/SP、EP；
- CS336 Systems 的 profiling、Triton、FlashAttention、memory-efficient training 和 distributed training 精选；
- 有双卡时补真实通信实验；没有时保留明确待验证项。

完成等级：所有并行策略达到 L1；DDP/FSDP 达到 L2；直接影响后续项目的 profiling 达到 L2。

#### 第 2 周：现代 LLM 架构

- MLA、Sliding Window、Local/Global 与 Linear/Hybrid Attention；
- RoPE scaling、YaRN、长上下文与 Context Parallelism；
- MoE、Router、Shared/Routed Expert、EP 与 All-to-All；
- MTP；
- FP8、INT8/INT4、权重和 KV Cache 量化；
- 完成第一批模型架构雷达卡片。

完成等级：所有主题达到 L1；MLA、MoE、MTP 各完成一个最小 L2 实验。量化和长上下文的完整性能实验可进入增强阶段。

#### 第 3–4 周：MemScope v0.1

- 理论账本、PyTorch allocator、Memory Snapshot 与 NVML 对齐；
- 参数、梯度、optimizer state、activation 和 unattributed memory；
- Deep Profile Mode 与 Production Monitor Mode；
- AMP、activation checkpointing、gradient accumulation、sequence length 和 MHA/GQA/MQA 对照；
- JSON/HTML 报告、测试、失败记录和项目说明。

核心交付：可运行的单卡训练显存归因版本。DDP/FSDP、MLA、MoE、MTP 和 Serving adapter 进入增强阶段，不阻塞第一版简历材料。

#### 第 5 周：Nano-vLLM

- 追踪 Request、Scheduler、Prefill、Decode、Sampling 和 Completion；
- Paged KV Cache、prefix cache、continuous batching；
- cancellation、preemption 和 cache release；
- CUDA Graph 基础和结构化 Serving benchmark。

完成等级：请求和缓存生命周期达到 L3；P/D 分离和复杂量化路径先达到 L1。

#### 第 6 周：投机解码与 Mini-SGLang 审计

- 实现最小 Draft–Verify；
- 理解独立 Draft、prompt lookup、self-speculative、Medusa、EAGLE、MTP speculation 和 tree verification 的统一框架；
- 实测接受率、draft/verify 开销、显存和不加速场景；
- 审计 Mini-SGLang 的 scheduler、Radix Cache、chunked prefill、overlap scheduling、取消和释放路径；
- 依据代码证据冻结一个项目扩展点。

完成等级：Draft–Verify 达到 L2；至少一种现代方案完成官方实现分析；Mini-SGLang 关键生命周期达到 L3。

#### 第 7–8 周：Mini-SGLang MVP 与 Kubernetes 基础

- 实现一个显存压力感知 admission 或 scheduling 扩展；
- 与原始 Mini-SGLang baseline 完成正确性、负载和失败路径对照；
- 输出 throughput、goodput、TTFT、TPOT、P99、OOM、preemption 和 rejection；
- 完成 Docker/Kubernetes 基础部署；
- 验证 GPU 资源声明、health probes、请求 draining、滚动更新和清理；
- 整理两个项目的 README、架构图、实验报告、失败案例和面试口述。

核心交付：MemScope v0.1 和 Mini-SGLang Memory-Aware Serving MVP。Kubernetes 在核心路线只要求本地生命周期达到 L2，不要求生产集群性能。

### 11.2 第 9–16 周：可选滚动增强

这部分不阻塞第一轮投递，可根据面试反馈、硬件条件和项目完成度调整顺序。

#### 第 9–10 周：MemScope v0.2

- DDP/FSDP per-rank 显存；
- collective buffer 和 checkpoint 峰值；
- MLA、MoE、MTP 与 KV Cache quantization adapter；
- Nano-vLLM/Mini-SGLang Serving 生命周期归因。

#### 第 11 周：投机解码增强

- 深入 EAGLE、MTP-based 或 tree-based 方案中的一种；
- 投机解码与 continuous batching、KV Cache 提交/回滚的组合实验；
- 开启/关闭 speculative decoding 的自适应判定探索。

#### 第 12 周：MoE Serving

- expert placement、Expert Parallel 和 All-to-All；
- prefill/decode expert load；
- 热门专家、负载倾斜和尾延迟；
- 有多 GPU 时完成真实验证，否则只保留机制实验与模拟边界。

#### 第 13 周：高级 Serving

- P/D disaggregation；
- KV 传输和资源配比；
- Multi-LoRA serving；
- 多模态 Serving 的视觉 token、动态 batching 与 prefill/KV 成本；
- CPU/NVMe offload；
- MLA inference 与更完整的量化实验。

#### 第 14–15 周：Kubernetes 与控制面增强

- GPU Operator/Device Plugin；
- 多副本路由、autoscaling 和模型版本升级；
- prefix-aware routing 和故障恢复；
- AIBrix、KServe、llm-d 的职责与扩展点；
- 多节点和生产性能仍需要相应硬件环境验证。

#### 第 16 周：项目强化与上游贡献

- 扩大负载矩阵和跨硬件验证；
- 修复项目验收缺口；
- 整理可复现脚本、演示和简历数字；
- 在改动边界清晰时尝试 Mini-SGLang、MemScope 所依赖工具或文档的上游 PR。

### 11.3 知识完整性检查

压缩到 8 周改变的是实现深度和工程化时间，不删除知识主题：

- 分布式训练与 CS336 Systems：核心第 1 周，真实多卡增强在第 9–10/12 周；
- MLA、MoE、MTP、长上下文、量化：核心第 2 周，工程适配在第 9–13 周；
- MemScope：核心第 3–4 周完成 v0.1，第 9–10 周完成 v0.2；
- Nano-vLLM 与基础 Serving：核心第 5 周，高级 Serving 在第 13 周；
- 投机解码：核心第 6 周完成 L2，第 11 周深入一种现代方案；
- Mini-SGLang：核心第 6 周审计、第 7–8 周完成 MVP，第 16 周强化或上游贡献；
- Kubernetes：核心第 7–8 周完成本地生命周期，第 14–15 周扩展控制面；
- MoE Serving、P/D 分离、Multi-LoRA、多模态 Serving：保留在增强阶段，其中多模态 Serving 作为滚动专题，不挤占两个核心项目。

## 12. 知识掌握等级

为了控制范围，每项技术标记为：

- **L1 原理**：能解释数据流、状态、公式和成本。
- **L2 实现**：能写最小版本并验证正确性。
- **L3 工程**：能在真实框架中定位实现、做 benchmark 并解释边界。

目标等级：

- DDP/FSDP、KV Cache、Paged Cache、MoE、MLA、投机解码：L2，直接关联项目的部分达到 L3。
- Nano-vLLM、Mini-SGLang scheduler/cache lifecycle：L3。
- 量化、长上下文、Chunked Prefill、P/D 分离：至少 L2。
- 多模态、Linear Attention、Multi-LoRA：先达到 L1/L2。
- Kubernetes 本地生命周期：L2；多节点生产行为保持为未验证。

## 13. 每个专题的固定学习模板

每个知识点都按以下顺序完成：

1. 它解决什么问题；
2. 极小具体例子；
3. 输入、输出、shape 和 dtype；
4. 状态由谁创建、读取、更新和释放；
5. 正式机制或公式；
6. 参数、FLOPs、显存、带宽和通信成本；
7. 最小正确性实现；
8. baseline 与单一变量实验；
9. 失败模式和成立边界；
10. 30 秒与 2 分钟面试口述。

每个专题至少产出：一张数据流/状态图、一个可运行实验、一份实测记录、一个反例、三道口述题和一项未验证边界。

## 14. 实验与证据规则

### 14.1 正确性先于性能

- 先验证 shape、dtype、输出一致性和状态释放，再测速度。
- 命令退出为 0 不等于目标成立。
- cache 存在不等于 cache 命中。
- 模型文件变小不等于推理更快。
- 理论通信减少不等于多卡扩展效率提高。

### 14.2 性能实验最小记录

```text
硬件与拓扑
驱动、CUDA、PyTorch 和框架版本
模型与 tokenizer revision
dtype、batch、sequence length
请求输入/输出长度与到达模式
warmup、测量窗口和同步方法
显存统计口径
失败、拒绝、超时和取消数量
```

### 14.3 反向压力测试

所有优化都必须检查：

- 是否以尾延迟换吞吐；
- 是否以计算换显存；
- 是否以额外权重或状态换 decode 速度；
- 是否只在特定 batch、长度、硬件或接受率下有效；
- 是否因通信、调度、反量化或 profiler 开销抵消收益；
- 系统越有效时，观测或学习信号是否反而变稀疏。

## 15. 云 GPU 与外部项目使用策略

### 15.1 本地 RTX 2060

适合：

- 小模型正确性；
- 单卡显存账本；
- Nano-vLLM/Mini-SGLang 代码阅读和小型实验；
- Kubernetes 服务生命周期。

不适合证明：

- 原生 BF16/FP8 性能；
- 大模型长上下文吞吐；
- TP/EP/NCCL 多卡效率；
- 生产集群容量。

### 15.2 云 GPU

- 双卡优先验证 DDP/FSDP 和通信时间线；
- TP/EP 优先选择互联拓扑明确的同机实例；
- 大显存单卡用于长上下文、量化和 Serving 压力实验；
- 所有租用、模型下载和依赖安装在执行前单独确认。

### 15.3 开源项目边界

- Nano-vLLM：教学型推理引擎，用于完整理解生命周期。
- Mini-SGLang：简历项目的上游基础，先审计后扩展。
- vLLM/SGLang：生产能力和实验参照，不从零复刻。
- TorchTitan/Megatron-LM/Picotron：分布式训练参照。
- AIBrix/KServe/llm-d：Kubernetes 与 Serving 控制面参照。

如需克隆仓库，统一放入 `research_artifacts/repos/`，并在下载前说明大小、落盘位置和更新策略。

## 16. 面试能力树

### 16.1 模型与训练

- Decoder-only 数据流、loss 和 backward。
- MHA/GQA/MLA 的状态与 KV Cache 差异。
- MoE 路由、专家负载和通信。
- MTP 的训练信号与推理边界。
- AMP、checkpointing、DDP/FSDP/TP/PP/CP/EP。

### 16.2 推理与 Serving

- prefill/decode、Paged KV Cache、Radix/Prefix Cache。
- continuous batching、chunked prefill、preemption。
- speculative decoding 的正确性和收益条件。
- TTFT、TPOT、P99、throughput、goodput。
- 取消、超时、OOM、过载和资源释放。

### 16.3 性能与显存

- 参数、梯度、优化器、激活和 KV Cache 账本。
- allocated、reserved、NVML 和 unattributed memory。
- compute/memory/communication/scheduling-bound。
- profiler 如何扰动被测系统。
- 为什么优化可能只省容量、不加速。

### 16.4 生产系统

- GPU 服务容量规划与 overload protection。
- 单副本、TP 与多副本的权衡。
- Kubernetes GPU 资源、健康检查、升级和故障恢复。
- 模型引擎与集群控制面的职责边界。

## 17. 当前工具与官方参考入口

### 17.1 工具路线

- PyTorch 与 PyTorch Profiler：模型、训练和框架内存/算子观测。
- NVML/`nvidia-smi`：GPU 与进程级总量，不提供语义分类。
- Nsight Systems/Compute：CUDA、kernel 和通信时间线。
- NCCL：collective 通信。
- CS336 Systems：系统课程与实验参照。
- Nano-vLLM、Mini-SGLang、vLLM、SGLang：推理引擎学习与验证。
- Docker、Kubernetes：部署和控制面。

### 17.2 主要入口

- [Stanford CS336 Spring 2025](https://stanford-cs336.github.io/spring2025/)
- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch CUDA Memory](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [NVIDIA NCCL Collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [PyTorch TorchTitan](https://github.com/pytorch/torchtitan)
- [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Picotron](https://github.com/huggingface/picotron)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM Speculators](https://docs.vllm.ai/projects/speculators/en/latest/)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [Qwen3](https://github.com/QwenLM/Qwen3)
- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2)
- [AIBrix](https://github.com/vllm-project/aibrix)
- [KServe](https://github.com/kserve/kserve)
- [llm-d](https://llm-d.ai/)

这些入口会更新。涉及“最新模型”“当前支持算法”或版本能力时，必须在专题开工前重新核验官方资料。

## 18. 下一步执行

当前下一步是：

1. 生成 Day 17：EP 与组合并行；
2. 完成两卡张量布局和主要 collective 的可验证实验契约；
3. 有真实双卡环境时，补 Day 13–16 的 NCCL 多卡性能与时间线证据；
4. Day 14–18 完成后启动 MemScope v0.1 的详细需求定型。

两个简历项目的正式实现都必须在各自启动门槛满足后，再建立独立规格、行为验收矩阵和实验协议；本路线图只定义方向与边界。
