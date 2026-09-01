# AI Infra 七层核心知识体系

> 版本：2026-08-07  
> 定位：从计算基础到专题扩展的 AI Infra 核心知识正文。  
> 主结构：模型生命周期，而不是技术发布时间或名词分类。  
> 配套路线路径：`../roadmaps/ai_infra_knowledge_and_project_roadmap.md`

## 1. 如何理解这七层

AI Infra 不是一组孤立的 GPU、分布式和推理框架名词。它研究的是：模型的数据和状态如何产生、转换、存储、传输、恢复和失效，以及如何在硬件约束下稳定、经济地完成这些过程。

七层知识按照以下生命周期排列：

```text
计算与系统基础
→ Transformer 前向数据流
→ 完整训练生命周期
→ 完整推理生命周期
→ 性能优化机制
→ 生产服务与可靠性
→ 多模态、长上下文、MoE 等专题扩展
```

学习任何技术时都要回答六个问题：

1. 它在生命周期中的哪个位置？
2. 输入、输出、shape 和 dtype 是什么？
3. 写入了哪些状态，后续由谁读取，何时释放？
4. 消耗的是计算、显存容量、内存带宽还是通信带宽？
5. 它改善什么指标，又可能损害什么指标？
6. 如何用最小实验验证结论？

如果只能说出技术名称和直观作用，却不能回答这些问题，就还没有真正掌握该技术。

## 2. 第一层：计算与系统基础

这一层回答：模型的计算实际发生在哪里，一个张量为什么会占用显存，一段代码为什么可能很慢。

### 2.1 张量、shape 与存储

#### 2.1.1 张量是什么

张量是按某种 shape 和 dtype 解释的一块数据。大模型中常见的维度包括：

- `B`：batch size；
- `S`：sequence length；
- `H`：hidden size；
- `V`：vocabulary size；
- `Nq`：Query head 数；
- `Nkv`：Key/Value head 数；
- `D`：head dimension，通常满足 `H = Nq × D`。

例如 `X[B,S,H]` 表示一个 batch 中每个 token 的隐藏表示。它占用的理论字节数是：

```text
B × S × H × dtype_bytes
```

这只是张量本体，不包含 allocator reserve、计算图、临时 workspace 或其他副本。

#### 2.1.2 stride 与 contiguous

shape 描述逻辑维度，stride 描述沿每一维移动一个位置需要跨过多少存储元素。`transpose` 可能只改变 shape/stride 而不复制数据；某些 kernel 要求连续布局，这时框架可能产生实际拷贝。

因此，两个 shape 相同的张量不一定具有相同的物理布局，也不一定具有相同的执行效率。

#### 2.1.3 broadcast 与隐式扩展

broadcast 允许不同 shape 的张量按规则参与运算。逻辑扩展不一定立即复制数据，但后续算子是否物化中间结果取决于实现。分析显存时不能只看源代码表面上有没有显式 `repeat`。

### 2.2 矩阵乘法与算术强度

线性层可写成：

```text
X[B,S,H] @ W[H,O] → Y[B,S,O]
```

参数量约为 `H × O`，输出元素数为 `B × S × O`。计算量和内存传输量是不同概念：

- FLOPs 描述执行了多少浮点运算；
- bytes moved 描述数据在 HBM、cache 和计算单元之间移动多少字节；
- arithmetic intensity 描述每移动一个字节完成多少计算。

大矩阵乘法通常更容易利用 GPU；大量小算子即使总 FLOPs 不高，也可能被 kernel launch 和内存访问限制。

### 2.3 CPU、GPU 与 CUDA 执行模型

#### 2.3.1 基本职责

- CPU 负责程序控制、数据准备、网络和通用任务；
- GPU 负责高度并行的张量计算；
- CUDA runtime/driver 负责设备管理、内存、kernel launch 和同步；
- 深度学习框架把高层算子映射为一个或多个 GPU kernel。

#### 2.3.2 GPU 层级

本阶段需要建立以下直觉：

- thread 是最小执行线程；
- warp 是 NVIDIA GPU 的基本调度组；
- thread block 在一个 SM 上协作；
- SM 包含执行单元、寄存器和 shared memory；
- Tensor Core 加速特定低精度矩阵运算；
- HBM 容量大于片上存储，但访问代价更高。

入门阶段不要求立即手写复杂 CUDA kernel，但要理解为什么 tiling、shared memory 和数据复用可以减少 HBM 访问。

#### 2.3.3 stream 与异步

GPU kernel launch 通常对 CPU 是异步的。错误的计时方式可能只测到 launch 时间，没有测到 kernel 完成时间。做性能实验时需要使用正确同步、CUDA event 或 profiler。

### 2.4 四类常见瓶颈

#### Compute-bound

计算单元接近饱和，主要受 FLOPs 限制。减少计算量、使用更快数据类型或更高算力硬件可能有效。

#### Memory-bound

大量时间用于搬运权重、激活或 KV Cache。减少字节数、提高 cache reuse、融合算子或量化可能有效。

#### Communication-bound

多 GPU 在等待 AllReduce、AllGather、ReduceScatter、All-to-All 或点对点传输。需要检查数据量、拓扑、带宽和计算通信重叠。

#### Scheduling-bound

GPU 没有持续获得足够工作。原因可能是 dataloader、CPU launch、请求排队、batch 太小、同步点或调度策略。

不能只根据 GPU utilization 一个数字判断属于哪种瓶颈。

### 2.5 数值表示

需要区分：

- 参数存储 dtype；
- forward 计算 dtype；
- 累加 dtype；
- gradient dtype；
- optimizer state dtype；
- 通信 dtype。

常见格式包括 FP32、TF32、FP16、BF16、FP8、INT8 和 INT4。低位宽可能：

- 减少存储容量；
- 减少 HBM 读取字节；
- 使用更高吞吐的硬件单元；
- 同时引入舍入误差、溢出、校准和反量化开销。

模型文件变小不能单独证明训练或推理更快。

### 2.6 第一层掌握标准

应该能够：

1. 手算常见张量 shape 和理论字节数。
2. 区分参数量、FLOPs、激活和实际峰值显存。
3. 解释 compute、memory、communication、scheduling 四类瓶颈。
4. 用 PyTorch Profiler 观察算子时间和显存。
5. 说明异步执行为什么会导致错误计时。

## 3. 第二层：Transformer 完整前向数据流

这一层回答：一段文本如何经过模型变成下一个 token 的概率。

### 3.1 从文本到 token IDs

```text
原始文本
→ tokenizer
→ token IDs [B,S]
→ embedding lookup
→ hidden states [B,S,H]
```

Tokenizer 把文本映射到离散 token ID。token ID 只是词表索引，本身没有连续语义距离；embedding table 把它映射为连续向量。

需要理解：

- vocabulary 和特殊 token；
- BOS、EOS、PAD；
- attention mask、causal mask 和 loss mask；
- 文档边界；
- tokenizer 版本变化会改变整个训练数据表示。

### 3.2 原始 Encoder–Decoder Transformer

#### Encoder

Encoder 对输入执行双向 self-attention。每个有效位置可以关注全部输入位置。它输出一组上下文化表示，常用于理解、编码或作为 Decoder 的条件。

#### Decoder

Decoder 包含：

- masked self-attention：只能看当前位置及之前位置；
- cross-attention：Query 来自 Decoder，Key/Value 来自 Encoder；
- FFN：对每个 token 位置独立进行通道变换。

训练时，目标序列的各位置可以借助 causal mask 并行计算；生成时，下一个 token 依赖已经生成的 token，因此存在串行依赖。

### 3.3 三种结构

#### Encoder-only

以 BERT 类模型为代表，通常使用双向注意力，适合理解、分类和表示学习。

#### Encoder–Decoder

以原始 Transformer、T5 类模型为代表，适合输入条件到输出序列的生成。

#### Decoder-only

GPT、Llama、Qwen 等语言模型的主干通常属于 causal Decoder-only。它通过预测下一个 token 统一文本生成任务。

具体模型可能增加 MoE、GQA、线性注意力或视觉模块，不能仅凭“Decoder-only”推断完整架构。

### 3.4 现代 Decoder Block

常见 Pre-Norm 数据流：

```text
x
├─ RMSNorm → Attention → Output Projection ─┐
└──────────────────────────────────────────── + → h

h
├─ RMSNorm → SwiGLU FFN ────────────────────┐
└──────────────────────────────────────────── + → y
```

残差连接为信息和梯度提供直接路径；Norm 稳定数值；Attention 在 token 之间传递信息；FFN 在每个 token 的通道维度上变换。

### 3.5 Attention 的数据流

```text
Q = XWq
K = XWk
V = XWv
Scores = QKᵀ / sqrt(D)
Probabilities = softmax(Scores + Mask)
Context = Probabilities × V
```

典型 shape 可以写成：

```text
Q: [B,Nq,S,D]
K: [B,Nkv,S,D]
V: [B,Nkv,S,D]
Attention scores: [B,Nq,S,S]
```

GQA/MQA 中 K/V 会按共享关系供多个 Query heads 使用。具体实现可能避免物理复制，不能用逻辑 broadcast 直接推断真实显存。

### 3.6 Causal mask

Causal mask 在 softmax 前阻止位置 `i` 访问未来位置 `j > i`。正确性验证不能只检查 mask 的三角形外观，还要验证：改变未来 token 时，过去位置的 logits 不发生变化。

### 3.7 位置编码与 RoPE

Attention 本身不知道 token 顺序。位置机制让模型获得顺序或相对位置信息。

RoPE 通常对 Q/K 的维度对子执行旋转，使注意力分数携带相对位置关系。需要区分：

- 原始训练长度；
- 配置声明的最大长度；
- 位置缩放后的可输入长度；
- 模型在长上下文下的实际检索和推理能力。

“接口允许更长输入”不等于“模型在更长输入上仍然有效”。

### 3.8 FFN、SwiGLU 与 MoE 前置理解

普通 FFN 对所有 token 使用同一组参数。SwiGLU 使用门控分支增强非线性表达。

MoE 通常把 Dense FFN 替换为多个专家并路由 token。它改变的是参数激活与通信方式，不改变 Attention 是 token 间交互模块这一基本分工。

### 3.9 从 hidden states 到 token

```text
final hidden [B,S,H]
→ final norm
→ LM Head [H,V]
→ logits [B,S,V]
→ sampling/argmax
→ next token ID
→ detokenizer
```

Softmax 常在需要概率或采样时计算；训练中的 cross-entropy 实现可能融合 log-softmax 和 loss，避免显式保存完整 probability。

### 3.10 第二层掌握标准

应该能够：

1. 从文本开始画出到 logits 的完整数据流。
2. 写出 Q/K/V、scores、context 的 shape。
3. 解释 Attention 与 FFN 的不同职责。
4. 区分 Encoder、Decoder 和 Decoder-only。
5. 实现并测试一个 causal Decoder Block。

## 4. 第三层：完整训练生命周期

这一层回答：模型如何从数据得到监督信号，梯度和训练状态如何产生并更新。

### 4.1 数据管线

```text
原始语料
→ 来源与许可检查
→ 解析
→ 质量过滤
→ 去重
→ tokenizer
→ token stream
→ sequence cutting/packing
→ train/validation split
→ shuffle
→ batch
```

训练数据不仅是文本文件，还包括 tokenizer、过滤规则、抽样权重、文档边界、packing 策略、随机种子和版本。

### 4.2 数据质量与利用率

需要区分：

- 原始文本字节；
- tokenizer 后 token 数；
- 有效训练 token；
- padding token；
- 因截断丢弃的 token；
- 重复 token；
- train/validation 污染。

训练吞吐通常应以 tokens/s 表达。samples/s 在样本长度变化时可能误导。

### 4.3 Causal LM labels

对于 token 序列：

```text
t0, t1, t2, t3
```

监督关系是：

```text
输入位置：t0  t1  t2
目标位置：t1  t2  t3
```

实现可能移动 logits，也可能移动 labels，或者由模型内部完成。必须查看实际数据流，不能只根据变量名判断。

### 4.4 Forward 与 loss

Forward 使用当前参数生成 logits。Cross-entropy 衡量目标 token 在预测分布中的负对数概率。

需要理解：

- reduction 是 sum 还是 mean；
- padding、文档边界和特殊位置是否计入 loss；
- gradient accumulation 时 loss 是否正确缩放；
- distributed training 中 loss 指标如何聚合。

### 4.5 Autograd 与 backward

Forward 构建计算图并保存 backward 所需信息。Backward 从 loss 出发，通过链式法则得到参数梯度。

梯度不是更新后的参数。optimizer 在之后读取梯度和自己的状态，决定参数更新。

### 4.6 Optimizer

AdamW 通常维护一阶和二阶状态。训练显存因此不能只计算权重。

一个 step 中常见顺序为：

```text
forward
→ loss scaling/normalization
→ backward
→ gradient synchronization
→ unscale（如需要）
→ gradient clipping
→ optimizer step
→ scheduler step
→ zero gradients
```

具体顺序应以实现为准。

### 4.7 梯度累积

梯度累积通过多个 micro-batch 后再更新参数，扩大有效 batch：

```text
effective batch
= micro batch × accumulation steps × data-parallel world size
```

它降低单次激活显存压力，但增加一次参数更新所需的 forward/backward 次数。通信是否每个 micro-step 发生取决于实现。

### 4.8 混合精度训练

混合精度不会简单地把所有状态都变成低精度。参数、计算、梯度规约、optimizer state 和 master weights 可能使用不同 dtype。

FP16 容易因范围有限发生 underflow/overflow，可能需要 loss scaling；BF16 指数范围更接近 FP32，通常更稳定，但尾数精度更低。

### 4.9 Activation checkpointing

普通 backward 需要 forward 激活。Activation checkpointing 只保留部分边界状态，在 backward 中重新执行部分 forward：

```text
更少 activation memory
↔ 更多 recomputation
```

它可能允许更大 batch 并提高最终吞吐，也可能在 batch 不变时直接降低吞吐。必须实测。

### 4.10 训练显存账本

至少拆分：

- parameters；
- gradients；
- optimizer states；
- activations；
- temporary tensors/workspace；
- allocator reserved memory；
- CUDA context；
- communication buffers。

理论账本与框架报告的 allocated、reserved、peak 指标含义不同。

### 4.11 数据并行 DDP

每个 data-parallel rank 处理不同数据，通常持有完整模型副本。Backward 后通过 AllReduce 或等价规约同步梯度。

优点是简单、扩展 batch；局限是单卡仍需容纳模型、梯度和 optimizer 状态。

### 4.12 ZeRO 与 FSDP

ZeRO/FSDP 在 data-parallel ranks 间切分训练状态。不同策略可能切分 optimizer state、gradient 和 parameter。

节省显存的代价包括：

- 参数 AllGather；
- gradient ReduceScatter；
- 更复杂的 checkpoint；
- 通信与计算重叠要求；
- 小模型下额外开销可能超过收益。

### 4.13 Tensor Parallel

TP 把单层矩阵或 attention heads 切分到多个 GPU。它解决单层或模型不能放入单卡的问题，但层内通信频繁，依赖高带宽互联。

分析 TP 必须说明：

- 哪个矩阵沿哪一维切分；
- 每个 rank 持有什么；
- 何时 AllReduce/AllGather；
- 输出如何恢复完整语义。

### 4.14 Pipeline Parallel

PP 把不同层放到不同 stage，并通过 micro-batch 填充流水线。需要理解：

- pipeline bubble；
- stage balance；
- activation transfer；
- forward/backward scheduling；
- checkpoint 如何跨 stage 保存。

### 4.15 Context/Sequence Parallel

CP/SP 沿序列或相关激活维度切分，常用于长上下文和降低激活复制。不同框架的名称与数据布局不完全一致，必须依据具体实现说明。

### 4.16 Expert Parallel

EP 把不同 MoE 专家放到不同 rank。token 根据 router 结果发送到专家，常使用 All-to-All。

它存在关键权衡：

- 每个 token 只激活少量专家，计算稀疏；
- 所有专家权重仍需在集群中存放；
- 路由不均可能产生热点；
- All-to-All 可能成为瓶颈。

### 4.17 Checkpoint 与恢复

完整 checkpoint 不应只有模型权重，还可能包括：

- optimizer；
- scheduler；
- gradient scaler；
- global step 和 consumed tokens；
- RNG states；
- data cursor；
- parallelism configuration；
- format version 和完成标志。

写入成功不能只看目标目录存在。必须避免半写 checkpoint 被误识别为可恢复状态。

### 4.18 第三层掌握标准

应该能够：

1. 画出数据到 optimizer update 的完整路径。
2. 打印 input、label 和有效 loss 位置。
3. 分解训练显存。
4. 解释 DDP、FSDP、TP、PP、CP、EP 分别切分什么。
5. 保存并恢复完整训练状态。

## 5. 第四层：完整推理生命周期

这一层回答：线上请求如何变成逐 token 输出，延迟和显存花在哪里。

### 5.1 服务入口

```text
HTTP/API request
→ authentication/validation
→ queue
→ tokenizer
→ scheduler
→ model executor
→ sampler
→ detokenizer
→ streaming response
```

用户感知延迟包含排队、CPU、网络和 GPU 时间，不能把单次 model forward 当作完整服务延迟。

### 5.2 Prefill

Prefill 一次处理 prompt 的多个 token，建立各层的初始 KV Cache，并产生首个输出位置的 logits。

它通常具有较大的矩阵运算和较高并行度，因此很多配置下更偏 compute-bound。但模型、长度、batch、attention 实现和硬件都会改变结论。

### 5.3 Decode

Decode 每个序列每步通常新增一个 token。它读取模型权重和历史 KV Cache，只为新 token 计算新的层状态。

小 batch decode 经常更偏 memory-bound，因为每步需要读取大量权重而计算规模有限。batch 增大后瓶颈可能改变。

### 5.4 KV Cache

标准 attention 的 KV Cache 粗略容量为：

```text
active sequences
× sequence length
× layers
× num_kv_heads
× head_dim
× 2
× dtype_bytes
```

实际系统还需要考虑：

- TP 等并行切分；
- block size；
- padding；
- allocator；
- fragmentation；
- prefix sharing；
- sliding window 或特殊 attention；
- beam search 的分支。

### 5.5 MHA、MQA 与 GQA

- MHA：每个 Query head 通常有对应 K/V head；
- MQA：多个 Query heads 共享一组 K/V；
- GQA：Query heads 分组共享 K/V。

减少 KV heads 能降低 KV Cache 和读取带宽，但模型质量、kernel 和训练方式也需要考虑。

### 5.6 Sampling

常见策略包括：

- greedy；
- temperature；
- top-k；
- top-p；
- beam search。

生成终止还涉及 EOS、stop strings、最大 token、客户端取消和服务 deadline。

### 5.7 流式返回

流式返回允许客户端尽早看到 token，改善感知体验，但没有消除自回归串行依赖。服务还需要处理：

- token buffering；
- 网络 backpressure；
- 客户端断开；
- 取消是否真正停止 GPU 工作；
- 已产生但未发出的 token 如何处理。

### 5.8 核心指标

- TTFT：到首 token 延迟；
- TPOT：首 token 后平均每 token 时间；
- ITL：token 间延迟；
- E2E：完整请求时间；
- throughput：requests/s 或 tokens/s；
- goodput：满足 SLO 的有效吞吐；
- P50/P95/P99：分位延迟；
- error/reject/timeout rate。

报告 tokens/s 必须说明输入、输出还是总 token。报告延迟必须说明是否包含排队和网络。

### 5.9 第四层掌握标准

应该能够：

1. 从请求画到流式响应。
2. 区分 prefill 和 decode 的数据流与瓶颈。
3. 手算 KV Cache 的近似容量。
4. 解释 TTFT、TPOT、吞吐和尾延迟的关系。
5. 设计短/长输入、短/长输出和不同并发的实验矩阵。

## 6. 第五层：性能优化机制

这一层不再按技术名称堆叠，而是按它改变的系统资源分类。

### 6.1 Attention IO 优化：FlashAttention

标准实现可能把大规模 attention score/probability 中间结果写入 HBM。FlashAttention 使用 tiling、片上存储和重计算，减少 HBM 往返，同时保持精确 attention 结果。

它主要改变实现的数据移动方式，不改变模型的 attention 数学定义。需要区分：

- FlashAttention；
- 稀疏 attention；
- 线性/近似 attention；
- PagedAttention/KV 分页管理。

这些名称相似，但解决的问题不同。

### 6.2 Kernel fusion

把多个相邻操作融合可以减少：

- kernel launch；
- 中间张量；
- HBM 写回和读取。

融合也可能增加编译、动态 shape 和维护复杂度。不能假设更多 fusion 总是更好。

### 6.3 CUDA Graph

CUDA Graph 记录并重放相对稳定的执行图，减少 CPU launch overhead。它适合 shape 和内存地址相对稳定的路径；动态 batch、动态 shape 和条件分支会增加使用难度。

### 6.4 Paged KV Cache

在线请求长度和生命周期不同，连续大块分配会产生碎片和预留浪费。Paged KV Cache 把 KV 状态分成 block，由逻辑 block 映射到物理 block。

它改善的是缓存分配和管理，不减少单个有效 KV 元素的数学需求。block metadata、内部碎片和调度仍有成本。

### 6.5 Continuous batching

传统 static batch 要等待整批请求完成。Continuous batching 允许完成的序列退出，新序列进入，提高 GPU 利用率。

它需要调度器持续管理：

- 新请求 admission；
- token budget；
- KV block；
- prefill/decode 混合；
- fairness；
- 取消和超时。

### 6.6 Chunked prefill

长 prompt 的完整 prefill 可能长时间阻塞 decode。Chunked prefill 把 prompt 分块调度，让系统在 prefill 和 decode 之间交错。

它可能改善 decode 延迟和公平性，也可能增加调度开销或延长某些长请求 TTFT。

### 6.7 Prefix caching

多个请求具有完全相同 token 前缀时，可以复用对应 KV Cache，避免重复 prefill。

必须区分：

- cache 功能已开启；
- cache 中存在 block；
- 当前请求实际命中；
- 命中后节省了多少 prefill；
- cache 占用了多少容量并触发何种 eviction。

系统越积极保留 prefix，可能越挤压普通 KV Cache；高命中收益和缓存容量压力需要联合评估。

### 6.8 量化

量化可以作用于：

- weights；
- activations；
- KV Cache；
- 通信张量。

常见区分：

- PTQ 与 QAT；
- symmetric 与 asymmetric；
- per-tensor、per-channel、per-group；
- static 与 dynamic；
- weight-only 与 weight-activation。

GPTQ、AWQ、SmoothQuant 等方法解决的量化误差和对象不同。评估量化需要同时测质量、峰值显存、TTFT、TPOT、吞吐和目标硬件 kernel 支持。

### 6.9 Speculative decoding

Draft 模型或 draft head 先提出多个候选 token，target 模型批量验证，从而减少串行 target steps。

收益取决于：

- draft 成本；
- 接受率；
- target 验证效率；
- batch；
- sampling 设置；
- 任务难度。

接受率低时，额外 draft 工作可能抵消收益。

### 6.10 Prefill/Decode 分离

Prefill 与 decode 的资源特征不同，可以部署到不同 worker pool。分离后新增问题包括：

- KV Cache 如何传输；
- 两类 worker 如何配比；
- 网络带宽和延迟；
- 请求状态由谁持有；
- worker failure 如何恢复；
- 何时分离收益超过传输成本。

### 6.11 并行推理

#### Tensor Parallel

让一个模型跨多 GPU，但增加层内通信。适合模型放不入单卡或单请求延迟需要多卡计算的情况。

#### Pipeline Parallel

按层切分，在线 decode 的 micro-batching 和 pipeline bubble 需要特别处理。

#### 多副本

完整模型复制到不同 GPU，可提高独立请求吞吐并减少跨卡通信，但每个副本都占完整权重显存。

选择 TP 还是多副本取决于模型容量、互联、请求负载和延迟目标。

### 6.12 第五层掌握标准

应该能够对每项优化回答：

1. 原瓶颈是什么？
2. 数据流或内存访问发生了什么变化？
3. 节省计算、容量、带宽、通信还是调度开销？
4. 在什么 workload 下有效？
5. 代价和失效条件是什么？
6. 用哪些指标验证？

## 7. 第六层：生产服务与可靠性

这一层回答：如何把能运行的模型变成具有明确状态、SLO 和故障边界的服务。

### 7.1 服务组件

典型系统可能包含：

- API gateway；
- authentication、quota 和 rate limit；
- request validator；
- tokenizer pool；
- admission controller；
- scheduler/router；
- model workers；
- distributed executor；
- KV Cache manager；
- sampler/detokenizer；
- streaming layer；
- metrics、logs、traces；
- model registry 和 deployment controller。

### 7.2 请求状态机

```text
created
→ admitted | rejected
→ queued
→ dispatched
→ prefilling
→ decoding
→ completed | failed | timed_out | cancelled
```

每个请求必须有唯一终态。错误率为零可能只是失败请求没有被记录，因此汇总指标必须能由 request-level 事件重建。

### 7.3 进程权与状态权

负责启动、停止和检查 worker 的组件管理进程；负责请求状态迁移的组件管理状态。两者通过明确事件通信，避免进程组件直接修改业务状态或状态机凭间接信号猜测进程存活。

### 7.4 Admission control

系统过载时，无限接收请求会让队列和尾延迟失控。Admission control 可以依据：

- queue length；
- active sequences；
- outstanding tokens；
- KV Cache 预算；
- predicted completion time；
- SLO class。

拒绝请求会改善已接收请求的延迟，却降低接受率。需要使用 goodput 而不是单看吞吐或延迟。

### 7.5 调度公平性

短请求优先可以降低平均延迟，但可能让长请求饥饿；长 prefill 优先可能阻塞 decode；prefix affinity 可能提高缓存命中但制造热点。

调度策略必须同时观察：

- mean 和 tail latency；
- per-class SLO；
- starvation；
- cache hit；
- worker imbalance；
- goodput。

### 7.6 取消、超时与 backpressure

客户端断开不一定自动停止 GPU 计算。完整取消链路需要：

- 网络层识别断开；
- 请求状态变为 cancelled；
- scheduler 停止后续 token；
- 释放 KV Cache；
- 处理已排队或正在执行的 batch；
- 保证不会重复返回终态。

Streaming 客户端消费慢时还需要 backpressure，避免输出 buffer 无限增长。

### 7.7 模型生命周期

生产系统需要处理：

- 权重下载和校验；
- 模型加载；
- kernel 编译和 warmup；
- readiness；
- 流量切换；
- 多版本共存；
- 灰度发布；
- rollback；
- worker crash；
- cache invalidation。

API 返回成功不等于模型已准备好接收真实流量。

### 7.8 可观测性

#### Metrics

适合持续聚合：QPS、tokens/s、TTFT、TPOT、P99、queue、KV usage、error rate、GPU metrics。

#### Logs

适合记录离散事件、配置、错误和状态变化，但不应成为唯一状态源。

#### Traces

适合把网关、排队、tokenization、prefill、decode、网络等阶段串起来。

#### Profiles

用于解释 CPU、GPU kernel、memory 和 communication 的性能原因，不适合长期全量开启。

### 7.9 容量规划

需要输入：

- 模型权重和 dtype；
- GPU 容量、带宽和互联；
- 输入/输出长度分布；
- 到达率和 burst；
- SLO；
- KV Cache 预算；
- 并行与副本配置；
- 故障冗余。

输出不应只是“需要几张 GPU”，还应包含：

- 稳定负载区间；
- 过载转折点；
- 目标 P99 下的最大 goodput；
- 扩副本和扩大 TP 的权衡；
- 预留容量和失败策略。

### 7.10 第六层掌握标准

应该能够：

1. 设计具有明确请求状态机的服务。
2. 解释 admission、routing、scheduling 和 KV 管理的边界。
3. 处理取消、超时、worker failure 和模型升级。
4. 从 metrics、traces 和 profiles 定位问题。
5. 根据真实 workload 做容量规划。

## 8. 第七层：专题扩展

专题必须建立在前六层之上。它们不是新名词列表，而是对输入类型、模型结构、状态形式或系统负载的扩展。

### 8.1 多模态

#### 8.1.1 图片输入数据流

```text
JPEG/PNG bytes
→ decode
→ resize/normalize
→ patchify
→ vision encoder
→ projector/merger
→ visual tokens
→ 与 text tokens 组成统一序列
→ language model
→ output tokens
```

模型不能直接“理解 JPEG 文件”。图片先被解码为像素张量，再通过视觉编码器变成可供语言主干使用的表示。

#### 8.1.2 Infra 影响

- 动态分辨率导致 visual token 数变化；
- vision encoder 和 LLM 的计算形态不同；
- visual tokens 增加 prefill 和 KV Cache；
- 不同分辨率造成 batching/padding 浪费；
- 多轮复用图片时可以考虑视觉特征或 prefix/KV 缓存；
- 视频还增加时间维度和更大的 token 数。

具体模型使用 projector、cross-attention 还是统一 token，必须依据论文和官方代码确认。

### 8.2 长上下文

长上下文同时影响：

- 位置表示；
- attention 计算；
- KV Cache；
- prefill latency；
- scheduler；
- prefix cache；
- 评测方式。

需要区分四个层次：

1. 接口允许输入多长；
2. 显存和 kernel 能否执行；
3. 模型能否从长文本中检索信息；
4. 模型能否跨长距离正确推理。

### 8.3 MoE

MoE 的典型数据流：

```text
token hidden states
→ router scores
→ top-k experts
→ dispatch/all-to-all
→ expert FFN
→ combine
→ output hidden states
```

需要理解：

- total parameters 与 activated parameters；
- shared/routed experts；
- load balancing loss；
- token dropping/capacity；
- expert parallel；
- All-to-All；
- 热点专家和低并发利用率。

稀疏激活降低每 token 计算，不意味着只加载激活参数量的权重。

### 8.4 LoRA 与 QLoRA

LoRA 在基础权重旁加入低秩增量，只训练较少参数。QLoRA 通常把基座权重量化后训练 LoRA adapter，降低微调显存。

需要区分：

- 可训练参数减少；
- optimizer state 减少；
- 基础权重存储；
- forward 计算；
- adapter merge；
- 多 adapter serving。

PEFT 主要降低适配成本，不自动降低基础模型的推理成本。

### 8.5 RAG

RAG 位于模型外部：

```text
query
→ embedding/search
→ retrieve
→ rank/filter
→ prompt construction
→ LLM prefill/decode
→ cited response
```

Infra 关注点包括索引更新、检索延迟、上下文 token、prefix reuse、租户隔离和数据版本。检索成功可能增加 prompt 长度，因此事实性收益与 prefill 成本需要联合评估。

### 8.6 SSM、线性注意力与混合架构

这类架构试图改变标准 attention 随序列增长的计算或状态形式。分析时必须追踪：

- 训练是否能并行；
- 推理保存什么 recurrent state；
- state 是否固定大小；
- 是否仍包含全 attention 层；
- kernel 是否成熟；
- 精确检索和 in-context learning 是否受影响。

“理论线性复杂度”不自动等于真实硬件更快。

### 8.7 Reasoning 与推理时计算

Reasoning system 可能组合：

- 更长生成；
- 多候选采样；
- verifier/reranker；
- search；
- tool execution；
- self-consistency。

它提高任务成功率时，通常也增加 token、模型调用、状态和尾延迟。Infra 需要管理预算、并发、取消、结果聚合和失败分支。

### 8.8 Agent Infra

Agent 位于 LLM 之外的编排层，常见生命周期为：

```text
task
→ model decision
→ tool call
→ external result
→ state update
→ next decision
→ completed/failed/cancelled
```

关键不是让模型记住进度，而是把任务状态、工具结果、重试和终态保存到可靠真相源。模型负责生成和判断，确定性调度器负责状态跃迁、预算和收敛。

### 8.9 第七层掌握标准

应该能够：

1. 把视觉输入追踪到 visual tokens 和 LLM 输出。
2. 分析长上下文对位置、attention、KV 和调度的联合压力。
3. 解释 MoE 的路由、专家存储和 All-to-All。
4. 区分 LoRA 的训练收益与推理成本。
5. 把 RAG、reasoning 和 Agent 放到模型外部系统生命周期中分析。

## 9. 七层之间的依赖关系

```text
第一层：张量、GPU、数值、瓶颈
   ↓ 为所有计算和性能判断提供语言
第二层：Transformer 前向
   ↓ 定义模型计算与状态
第三层：训练生命周期
   ↓ 产生可部署的模型权重
第四层：推理生命周期
   ↓ 把权重转成在线 token 输出
第五层：性能优化
   ↓ 改变计算、内存、通信和调度成本
第六层：生产可靠性
   ↓ 把优化后的执行路径变成稳定服务
第七层：专题扩展
   ↳ 改变输入、结构、状态或系统负载
```

不能跳过前层直接学习后层。例如：

- 不理解 attention shape，就很难真正理解 FlashAttention；
- 不理解 KV Cache 生命周期，就很难理解 PagedAttention；
- 不理解 optimizer state，就很难理解 FSDP/ZeRO；
- 不理解请求状态机，就很难设计 prefill/decode 分离；
- 不理解 All-to-All，就很难判断 MoE 部署瓶颈。

## 10. 统一的掌握验证模板

每个主题最终都应产生以下证据：

### 10.1 理论证据

- 一张数据流图；
- 输入/输出 shape；
- 状态与生命周期；
- 复杂度或容量估算；
- 收益与代价。

### 10.2 代码证据

- 最小实现或明确的关键执行路径；
- 单元测试；
- 正向和异常路径；
- 可重复配置。

### 10.3 实验数据

- baseline；
- 单变量对照；
- 环境和版本；
- warmup 和测量口径；
- 原始结构化结果；
- 失败实验。

### 10.4 口述能力

可以依次回答：

1. 它是什么；
2. 为什么需要；
3. 内部数据如何流动；
4. 它优化了什么资源；
5. 什么情况下不会有效；
6. 自己如何验证过。

## 11. 与其他文档的关系

- 本文：七层核心知识正文。
- `transformer_technical_updates.md`：Transformer 技术演进索引，不作为学习顺序。
- `../roadmaps/ai_infra_knowledge_and_project_roadmap.md`：一个月学习计划、三个学习项目和两个简历项目。
- `../interview-notes/小鹏.txt`：真实面试问题与回答复盘。

后续扩写时，新增知识必须先判断属于七层中的哪一层，再插入相应小节；不要在正文末尾不断增加与主结构并列的技术名词章节。
