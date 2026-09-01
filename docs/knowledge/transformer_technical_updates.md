# Transformer 技术更新脉络：理解能力与推理效率

> 更新时间：2026-05-18  
> 范围说明：本文按“提升模型理解能力”和“提升模型推理/推演效率”两条主线总结 Transformer 自 2017 年以来的重要技术更新。由于相关研究极多，本文优先覆盖对现代大模型架构、训练、长上下文、服务系统和工程落地影响最大的技术。

## 1. 总览

Transformer 的演进大致可以理解为两种力量的共同推动：

1. **提升理解能力**：让模型更好地表达上下文关系、位置关系、世界知识、任务意图、长程依赖、多模态信息和复杂推理过程。
2. **提升推理效率**：让模型在训练、部署、长上下文处理和自回归生成时更快、更省显存、更高吞吐、更容易服务化。

早期 Transformer 的核心创新是**自注意力机制与多头注意力机制**。之后的发展不只是“把模型变大”，还包括位置编码、预训练目标、稀疏专家、检索增强、长上下文、对齐训练、低秩适配、量化、FlashAttention、KV Cache 优化、推测解码、PagedAttention 以及 Mamba/SSM 等替代或混合架构。

## 2. 提升模型理解能力的重大更新

### 2.1 自注意力与多头注意力：Transformer 的基础能力

**代表技术：Self-Attention、Scaled Dot-Product Attention、Multi-Head Attention**

2017 年的 Transformer 用注意力机制替代循环神经网络和卷积结构，使模型能够在一个序列内部直接建立任意 token 之间的关系。多头注意力将注意力空间拆分成多个子空间，让不同 head 学习不同类型的关系，例如句法结构、实体指代、局部搭配、长距离依赖等。

重要意义：

- 让模型摆脱 RNN 的顺序计算限制，训练可并行化。
- 让每个 token 能直接访问上下文中的其他 token。
- 多头机制增强了关系建模的多样性，是后续 BERT、GPT、T5、ViT 和现代 LLM 的共同基础。

局限：

- 标准全注意力的时间和显存复杂度是 `O(n^2)`。
- 注意力本身不携带顺序信息，必须额外引入位置编码。

### 2.2 位置编码：从绝对位置到长上下文外推

**代表技术：Sinusoidal/learned positional embedding、Relative Position Encoding、Transformer-XL、RoPE、ALiBi、XPos、YaRN、LongRoPE**

Transformer 不像 RNN 天然知道 token 顺序，因此位置编码成为理解能力的重要组成部分。

主要阶段：

- **绝对位置编码**：原始 Transformer 使用正弦位置编码，也可用可学习位置 embedding。优点是简单，缺点是对超出训练长度的上下文外推较弱。
- **相对位置编码**：让模型关注 token 间的相对距离，而不是固定绝对坐标，更适合语言中的相对结构。
- **Transformer-XL**：引入片段级递归和相对位置编码，改善长文本建模。
- **RoPE（Rotary Position Embedding）**：通过旋转变换把位置信息融入 query/key，被 LLaMA、Qwen、DeepSeek 等大量现代 LLM 采用。
- **ALiBi**：给注意力分数加入与距离相关的线性偏置，使模型能“短训练、长测试”，提升长度外推能力。
- **YaRN / LongRoPE**：在 RoPE 基础上扩展上下文窗口，用更低训练成本支持 32K、128K、百万级甚至更长上下文。

重要意义：

- 位置编码从“让模型知道顺序”发展成“让模型能够稳定处理超长上下文”。
- 现代长上下文模型的能力很大程度来自 RoPE 缩放、位置插值和长上下文微调策略。

### 2.3 预训练范式：从任务模型到通用基础模型

**代表技术：GPT、BERT、T5、denoising objective、causal LM、masked LM、seq2seq pretraining**

Transformer 真正成为基础模型，关键不只是结构，而是大规模预训练范式。

主要路线：

- **GPT 系列：自回归语言建模**  
  用“预测下一个 token”的目标训练 decoder-only Transformer，天然适合生成、对话和工具调用。

- **BERT：双向编码器预训练**  
  通过 masked language modeling 学习双向上下文表示，在分类、抽取、检索、问答等理解任务上带来巨大提升。

- **T5 / BART 等 encoder-decoder 范式**  
  把多种 NLP 任务统一成 text-to-text，有利于迁移学习和多任务学习。

重要意义：

- 模型从“为某个任务训练”转向“先学习通用语言表示，再适配任务”。
- 大规模无监督/自监督数据成为模型理解能力的核心来源。

### 2.4 规模化与数据策略：Scaling Law、Chinchilla 与数据质量

**代表技术：Scaling Laws、Chinchilla compute-optimal training、数据清洗、去重、合成数据、课程学习**

大模型能力的提升离不开规模化：参数规模、训练 token 数、计算量和数据质量共同决定模型能力。

关键变化：

- **Scaling Laws** 说明 loss 与模型大小、数据量、计算量之间存在可预测关系。
- **Chinchilla** 重新强调在固定计算预算下，很多模型并不是参数不够，而是训练 token 不够。
- 后续模型更加重视高质量数据、代码数据、数学数据、多语言数据、去重、过滤、合成数据和数据配比。

重要意义：

- “更大模型”转向“参数、数据、计算的更优配比”。
- 数据质量直接影响模型知识密度、推理能力、代码能力和安全性。

### 2.5 前馈层与归一化改进：让深层 Transformer 更稳定

**代表技术：Pre-LN、RMSNorm、GeLU、SwiGLU/GeGLU、残差缩放、深层初始化策略**

除了注意力层，Transformer 的 MLP/FFN 和归一化方式也经历了多次重要更新。

常见改进：

- **Pre-LN**：把 LayerNorm 放到子层之前，使深层模型训练更稳定。
- **RMSNorm**：去掉均值中心化，计算更轻，现代 LLM 中非常常见。
- **SwiGLU / GeGLU**：用门控前馈网络替代普通 FFN，提高表达能力和训练效率。
- **残差与初始化策略**：帮助非常深的 Transformer 避免梯度不稳定。

重要意义：

- 这些技术不如注意力机制显眼，但对千亿参数级模型训练稳定性非常关键。
- 现代 LLM 的“默认配方”通常是 Pre-Norm + RMSNorm + SwiGLU + RoPE。

### 2.6 稀疏专家模型：用更多参数提升能力，但不同比例增加计算

**代表技术：MoE、GShard、Switch Transformer、GLaM、Mixtral、DeepSeekMoE**

Mixture of Experts（MoE）把 FFN 层替换成多个专家网络，每个 token 只路由到少数专家。这样模型总参数可以很大，但每个 token 激活的参数较少。

重要意义：

- 在相近推理计算量下提高模型容量。
- 对多领域、多语言、多任务模型尤其有用。
- Mixtral、DeepSeek-V2/V3 等模型证明了稀疏专家在开源和工业模型中的实用价值。

挑战：

- 路由负载均衡复杂。
- 分布式训练通信成本高。
- 小 batch 或低并发推理时，MoE 的硬件利用率可能不理想。

### 2.7 检索增强与外部记忆：把知识从参数中部分移出

**代表技术：RAG、REALM、RETRO、Atlas、长上下文检索、向量数据库**

RAG 将参数模型与外部检索系统结合：模型生成答案前先检索相关文档，把文档片段放入上下文。

重要意义：

- 提升事实性和可追溯性。
- 降低模型完全依赖参数记忆的压力。
- 便于接入企业私有知识库和实时知识。

局限：

- 检索质量决定上限。
- 上下文拼接、排序、去重、引用和冲突处理会影响最终答案。
- RAG 不等于“自动正确”，仍需要评测和防幻觉设计。

### 2.8 指令微调、RLHF、偏好优化：让能力更可用

**代表技术：Instruction Tuning、FLAN、InstructGPT、RLHF、RLAIF、DPO、ORPO、KTO**

基础预训练模型只是学会“续写文本”，不一定会按用户意图完成任务。指令微调和偏好优化让模型更擅长遵循指令、对话、拒绝不安全请求、解释步骤和保持格式。

主要路线：

- **SFT（监督微调）**：用高质量指令数据教模型如何回答。
- **RLHF**：用人类偏好训练奖励模型，再优化语言模型。
- **DPO 等直接偏好优化**：绕过显式奖励模型，直接用偏好对优化策略。

重要意义：

- 让“会预测文本”的模型变成“可交互助手”。
- 对实际产品体验的提升非常大。

### 2.9 推理能力增强：Chain-of-Thought 与过程监督

**代表技术：CoT、Self-Consistency、Tree of Thoughts、ReAct、Toolformer、过程监督、验证器/奖励模型**

复杂问题往往需要中间步骤。Chain-of-Thought（CoT）通过显式或隐式中间推理提升数学、逻辑、代码和规划任务表现。

进一步发展：

- **Self-Consistency**：采样多条推理路径再投票。
- **ReAct**：把推理和外部动作结合，例如搜索、调用工具、执行代码。
- **过程监督**：不只评价最终答案，也评价中间推理步骤。
- **验证器与重排序**：生成多个候选答案，用 verifier 选择更可靠结果。

重要意义：

- 推理能力不仅来自模型结构，也来自训练目标、数据格式、解码策略和验证机制。
- 现代 reasoning model 往往把“生成、搜索、验证、反思”组合成系统能力。

### 2.10 多模态 Transformer：从语言理解扩展到视觉、音频和行动

**代表技术：ViT、CLIP、Flamingo、BLIP、LLaVA、Qwen-VL、Gemini 类多模态架构**

Transformer 的注意力机制不局限于文本。图像可以切成 patch，音频可以切成帧，视频可以切成时空 token，机器人状态也可以序列化。

主要方向：

- **ViT**：把图像 patch 当作 token。
- **CLIP**：通过图文对比学习对齐视觉和语言表示。
- **视觉语言模型**：用 cross-attention、投影器或统一 token 空间连接图像编码器和 LLM。
- **原生多模态模型**：把文本、图像、音频、视频统一到更通用的序列建模框架中。

重要意义：

- Transformer 从 NLP 模型演变为通用感知与生成架构。
- 多模态对“理解能力”的定义从文本语义扩展到跨模态对齐、视觉推理和世界状态理解。

## 3. 提升推理/推演效率的重大更新

### 3.1 并行训练与分布式扩展

**代表技术：数据并行、张量并行、流水线并行、ZeRO、Megatron-LM、DeepSpeed、FSDP**

Transformer 的自注意力天然比 RNN 更适合并行训练，但大模型仍需要分布式系统支持。

关键技术：

- **数据并行**：不同 GPU 处理不同 batch，再同步梯度。
- **张量并行**：把单层矩阵运算拆到多个 GPU。
- **流水线并行**：把不同层放到不同 GPU。
- **ZeRO/FSDP**：切分优化器状态、梯度和参数，显著降低单卡显存压力。
- **激活检查点**：用重新计算换显存。

重要意义：

- 让百亿、千亿、万亿参数训练成为可能。
- 训练效率直接决定模型能力上限。

### 3.2 混合精度与低精度训练/推理

**代表技术：FP16、BF16、FP8、INT8、INT4、Tensor Core、loss scaling**

低精度计算是 Transformer 工程化的关键。

主要阶段：

- **FP16/BF16**：大幅降低显存和提升吞吐，BF16 更稳定。
- **FP8**：在 Hopper 等新硬件上进一步提升训练/推理吞吐。
- **INT8/INT4 推理**：用量化降低权重和激活的存储与带宽成本。

重要意义：

- 对大模型而言，瓶颈常常不是纯 FLOPs，而是显存容量和内存带宽。
- 低精度让更大模型能部署在更少 GPU 或边缘设备上。

### 3.3 高效注意力：从稀疏注意力到 FlashAttention

**代表技术：Sparse Transformer、Longformer、BigBird、Reformer、Linformer、Performer、FlashAttention、FlashAttention-2、FlashAttention-3**

标准注意力需要显式构造 `n x n` 注意力矩阵，长上下文时非常昂贵。

主要路线：

- **稀疏注意力**：只关注局部窗口、全局 token 或固定稀疏模式，降低复杂度。
- **低秩/核方法/哈希方法**：用近似方式减少注意力计算，如 Linformer、Performer、Reformer。
- **FlashAttention**：不改变数学结果，而是通过分块、tiling、重计算和 IO-aware kernel 避免把完整注意力矩阵写入 HBM。
- **FlashAttention-2**：改进并行划分和 GPU 利用率。
- **FlashAttention-3**：面向 Hopper GPU，引入异步和低精度优化，进一步提高吞吐。

重要意义：

- FlashAttention 的关键不是“近似注意力”，而是“精确注意力的更优 GPU 实现”。
- 它已经成为现代 LLM 训练和长上下文推理的基础设施级优化。

### 3.4 KV Cache：自回归推理的核心加速

**代表技术：KV Cache、PagedAttention、vLLM、continuous batching、prefix caching**

Decoder-only LLM 生成第 `t` 个 token 时，不需要重复计算前面所有 token 的 key/value。KV Cache 保存历史 key/value，使每步只计算新 token 的表示。

重要意义：

- 把自回归生成从“每步重算全部上下文”变成“增量生成”。
- 对长上下文和多轮对话至关重要。

新问题：

- KV Cache 会随上下文长度和 batch 增长，占用大量显存。
- 服务端请求长度不同，显存碎片和调度浪费严重。

解决方向：

- **PagedAttention/vLLM**：借鉴操作系统分页，把 KV Cache 切成块管理，减少显存浪费并支持更灵活的共享。
- **Continuous batching**：动态把不同请求合并，提升吞吐。
- **Prefix caching**：多个请求共享相同 prompt 前缀时复用 KV。

### 3.5 MQA、GQA 与 MLA：减少 KV Cache 成本

**代表技术：Multi-Query Attention、Grouped-Query Attention、Multi-Head Latent Attention**

标准多头注意力中，每个 query head 都有自己的 key/value head，导致 KV Cache 很大。

主要改进：

- **MQA（Multi-Query Attention）**：多个 query head 共享一组 key/value，大幅降低 KV Cache。
- **GQA（Grouped-Query Attention）**：在 MHA 和 MQA 之间折中，多个 query head 分组共享 key/value，质量和效率更平衡。
- **MLA（Multi-Head Latent Attention）**：DeepSeek-V2 引入的低秩 latent 压缩思路，把 KV 表示压缩到更小 latent 空间，进一步降低长上下文推理成本。

重要意义：

- 对推理部署非常关键，尤其是长上下文、高并发场景。
- 现代 LLM 很少再使用最朴素的 MHA 作为唯一选择，GQA/MLA 等结构越来越常见。

### 3.6 量化：用更少 bit 部署大模型

**代表技术：LLM.int8、SmoothQuant、GPTQ、AWQ、QLoRA、NF4、KV Cache quantization**

量化把 FP16/BF16 权重或激活转换成 INT8、INT4 等低 bit 表示。

主要路线：

- **权重量化**：减少模型权重显存，常见于本地部署。
- **激活量化**：进一步减少计算和显存，但更难保持精度。
- **SmoothQuant**：平滑权重和激活的量化难度，使 W8A8 更可行。
- **GPTQ/AWQ**：面向权重量化，尽量保持生成质量。
- **QLoRA**：用 4-bit 量化基座模型加 LoRA 微调，大幅降低微调门槛。
- **KV Cache 量化**：降低长上下文推理时的缓存显存。

重要意义：

- 量化是消费级 GPU、本地 LLM 和低成本云推理的核心技术。
- 对超长上下文，KV Cache 量化与 GQA/MLA 同样重要。

### 3.7 参数高效微调：低成本获得领域能力

**代表技术：Adapter、Prefix Tuning、Prompt Tuning、LoRA、QLoRA、DoRA、S-LoRA**

完整微调大模型成本高且部署复杂。参数高效微调只训练少量新增参数或低秩矩阵。

重要意义：

- 降低领域适配成本。
- 支持一个基础模型挂载多个任务 adapter。
- S-LoRA 等服务方案进一步支持多 adapter 并发推理。

注意：

- PEFT 主要降低训练/适配成本，不一定总是降低单次推理成本。
- 多 adapter 服务时需要额外的调度和内存管理。

### 3.8 推测解码与多 token 解码

**代表技术：Speculative Decoding、Speculative Sampling、Medusa、EAGLE、Lookahead Decoding、self-speculative decoding**

自回归生成一次通常只产出一个 token，速度受串行依赖限制。推测解码用小模型或轻量 draft head 先预测多个 token，再由大模型一次性验证。

重要意义：

- 在不改变最终分布或尽量保持质量的前提下提升解码速度。
- 对低 batch、交互式聊天、代码补全等延迟敏感场景尤其有价值。

常见变体：

- **小模型 draft + 大模型 verify**。
- **同模型浅层 draft**，减少额外模型。
- **多头 draft**，一次提出多个未来 token。

局限：

- 加速比取决于 draft 命中率。
- 任务越难、采样越发散，验证失败越多，加速越低。

### 3.9 稀疏化、剪枝与蒸馏

**代表技术：知识蒸馏、结构化剪枝、非结构化稀疏、SparseGPT、Wanda、DistilBERT、TinyBERT**

这类方法通过移除冗余参数、让小模型模仿大模型、或利用稀疏矩阵加速推理。

重要意义：

- 适合边缘部署、小模型压缩、低成本场景。
- 蒸馏可把大模型能力转移到更小模型上。

现实限制：

- 非结构化稀疏不一定能在通用硬件上获得实际加速。
- 剪枝后需要恢复训练或校准，否则质量下降明显。

### 3.10 长序列替代架构与混合架构

**代表技术：Linear Attention、RWKV、S4、Hyena、Mamba、Mamba-2、Jamba、Transformer-SSM hybrid**

由于标准注意力在长序列上有 `O(n^2)` 成本，研究者提出了多种替代架构。

主要方向：

- **线性注意力**：把注意力近似成线性复杂度。
- **状态空间模型（SSM）**：用状态递推建模长程依赖，Mamba 通过选择性状态空间和硬件友好实现获得关注。
- **混合架构**：保留部分注意力层处理精确检索和 in-context learning，用 SSM 层提升长序列效率。

重要意义：

- SSM/Mamba 类模型在长序列、低延迟生成和线性复杂度方面有潜力。
- 目前 Transformer 仍是主流，但混合架构可能成为长上下文模型的重要方向。

### 3.11 服务系统优化：从单模型推理到高并发平台

**代表技术：Triton/TensorRT-LLM、vLLM、SGLang、TGI、CUDA Graph、kernel fusion、请求调度、分块预填充**

大模型部署不是单次 forward 的问题，而是系统问题。

关键优化：

- **prefill/decode 分离**：长 prompt 的预填充和逐 token decode 性质不同，需要不同调度。
- **continuous batching**：请求动态进出 batch。
- **chunked prefill**：避免长 prompt 阻塞 decode。
- **CUDA Graph**：减少 kernel launch overhead。
- **kernel fusion**：合并算子，减少内存读写。
- **张量并行推理**：多 GPU 承载大模型。

重要意义：

- 工业推理性能常由调度、显存管理和 kernel 实现共同决定。
- 同一个模型在不同 serving engine 上吞吐和延迟可能差异巨大。

## 4. 时间线速览

| 时间 | 技术/论文/系统 | 主要贡献 | 主要影响方向 |
|---|---|---|---|
| 2017 | Transformer | 自注意力、多头注意力、并行序列建模 | 理解能力 + 训练效率 |
| 2018 | GPT / BERT | 自回归生成与双向预训练 | 通用语言理解 |
| 2019 | Transformer-XL / Sparse Transformer / Megatron-LM | 长上下文记忆、稀疏注意力、大模型并行 | 理解 + 效率 |
| 2020 | T5 / Longformer / BigBird / RAG / Scaling Laws | text-to-text、长文档、检索增强、规模规律 | 理解能力 |
| 2021 | Switch Transformer / RoPE / LoRA / ZeRO/FSDP 普及 | MoE、旋转位置编码、低秩适配 | 理解 + 适配效率 |
| 2022 | Chinchilla / InstructGPT / CoT / FlashAttention / SmoothQuant | 计算最优训练、RLHF、推理链、IO-aware attention、量化 | 理解 + 推理效率 |
| 2023 | GPTQ/AWQ/QLoRA / GQA / vLLM PagedAttention / Speculative Decoding / YaRN | 低 bit 部署、KV 降本、高吞吐服务、快速解码、长上下文扩展 | 推理效率 |
| 2024 | FlashAttention-3 / Mamba-2 / LongRoPE / MLA / DeepSeekMoE | Hopper 优化、SSM 混合、百万级上下文、KV 压缩、经济 MoE | 效率 + 长上下文 |
| 2025-2026 | 更强推理模型、长上下文系统、混合架构、服务端编排 | 生成-搜索-验证结合，推理时计算扩展 | 理解 + 系统效率 |

## 5. 两条主线的关系

很多技术同时提升理解能力和效率。例如：

- **RoPE/ALiBi/YaRN/LongRoPE**：表面是位置编码，实际同时提升长上下文理解和长上下文可部署性。
- **MoE**：用稀疏激活提高参数容量，同时控制每 token 计算量。
- **FlashAttention**：不改变模型能力，但让更长上下文、更大 batch 和更大模型训练成为可能，间接提升能力。
- **GQA/MLA**：主要是推理效率技术，但降低 KV Cache 后，模型更容易支持长上下文。
- **RAG**：不改变 Transformer 内部结构，却显著提升事实理解、时效性和企业知识接入能力。
- **推测解码**：不提升模型智力，但让高质量大模型更容易用于实时交互。

## 6. 重要技术优先级

如果按对现代 LLM 的影响排序，可以优先理解以下技术：

1. **Self-Attention / Multi-Head Attention**：Transformer 的根。
2. **大规模预训练 + Scaling Laws + 数据质量**：现代基础模型能力的来源。
3. **位置编码与长上下文技术：RoPE、ALiBi、YaRN、LongRoPE**。
4. **高效注意力：FlashAttention 系列**。
5. **KV Cache、GQA/MQA/MLA、PagedAttention/vLLM**。
6. **Instruction Tuning、RLHF/DPO、CoT/reasoning 数据**。
7. **MoE：Switch、Mixtral、DeepSeekMoE 等稀疏专家路线**。
8. **量化：SmoothQuant、GPTQ、AWQ、QLoRA、KV Cache quantization**。
9. **推测解码、多 token 解码、服务调度系统**。
10. **Mamba/SSM/混合架构：长序列效率的潜在下一阶段**。

## 7. 参考资料

- Vaswani et al., *Attention Is All You Need*, 2017: https://arxiv.org/abs/1706.03762
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018: https://arxiv.org/abs/1810.04805
- Child et al., *Generating Long Sequences with Sparse Transformers*, 2019: https://arxiv.org/abs/1904.10509
- Brown et al., *Language Models are Few-Shot Learners*, 2020: https://arxiv.org/abs/2005.14165
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020: https://arxiv.org/abs/2001.08361
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020: https://arxiv.org/abs/2005.11401
- Beltagy et al., *Longformer: The Long-Document Transformer*, 2020: https://arxiv.org/abs/2004.05150
- Zaheer et al., *Big Bird: Transformers for Longer Sequences*, 2020: https://arxiv.org/abs/2007.14062
- Fedus et al., *Switch Transformers*, 2021: https://arxiv.org/abs/2101.03961
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021: https://arxiv.org/abs/2104.09864
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021: https://arxiv.org/abs/2106.09685
- Press et al., *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, 2021: https://arxiv.org/abs/2108.12409
- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022: https://arxiv.org/abs/2203.15556
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022: https://arxiv.org/abs/2203.02155
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, 2022: https://arxiv.org/abs/2201.11903
- Dao et al., *FlashAttention*, 2022: https://arxiv.org/abs/2205.14135
- Xiao et al., *SmoothQuant*, 2022: https://arxiv.org/abs/2211.10438
- Dettmers et al., *QLoRA*, 2023: https://arxiv.org/abs/2305.14314
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023: https://arxiv.org/abs/2305.13245
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023: https://arxiv.org/abs/2309.06180
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2023: https://arxiv.org/abs/2211.17192
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023: https://arxiv.org/abs/2302.01318
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, 2023: https://arxiv.org/abs/2309.00071
- Dao, *FlashAttention-2*, 2023/2024: https://arxiv.org/abs/2307.08691
- Liu et al., *LongRoPE*, 2024: https://arxiv.org/abs/2402.13753
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023: https://arxiv.org/abs/2312.00752
- Dao and Gu, *Transformers are SSMs*, 2024: https://proceedings.mlr.press/v235/dao24a.html
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024: https://arxiv.org/abs/2405.04434
- Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, 2024: https://arxiv.org/abs/2407.08608
