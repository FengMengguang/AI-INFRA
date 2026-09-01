# 项目级指令说明

本文件是 `AI Infta` 项目的增量规则文档，只补充当前项目特有约束，不覆盖全局规则。

在当前 Ubuntu 主机执行本项目任务时，必须同时阅读并遵守：

- `/home/kadajhin/.codex/AGENTS.md`
- 本文件 `AGENTS.md`

当前主机没有 `/home/kadajhin/.claude/CLAUDE.md`。若以后创建该文件，也必须一并
阅读并遵守。跨设备迁移时，应以目标主机实际存在的全局规则路径为准，不能把本机
路径静默套用到其他操作系统。

如果本文件与全局规则存在明显冲突，必须先向用户说明冲突点并等待确认。轻微差异
应解释为本文件对当前项目的更具体约束。

## 1. 项目目标与读者

本项目通过可运行的 PyTorch 实验，建立 Transformer 训练、推理和 AI Infra 的系统
理解。每日内容必须把以下层次连起来：

```text
机制直觉
→ Tensor 数据流与 shape
→ 数学表达
→ 最小代码
→ GPU/系统实测
→ 事实边界
→ 手算与面试口述
```

默认读者了解 Python 和少量 PyTorch 语法，但不默认掌握线性代数记号、Attention、
GPU、训练状态、推理缓存或分布式训练。

README 面向 GitHub 读者和学习者；本文件面向执行项目任务的 Agent。不得用本文件
取代根 `README.md`，也不得把面向 Agent 的规则堆入 README。

## 2. 当前环境与事实边界

当前已核验环境：

```text
操作系统              Ubuntu
项目根目录            /home/kadajhin/桌面/New-Project/AI-Infta
Python                3.13
环境管理              uv
PyTorch               2.13.0+cu130
PyTorch compiled CUDA 13.0
GPU                   NVIDIA GeForce RTX 2060
Compute capability    7.5
Native BF16 support   false
```

这些值可能变化。任何依赖当前版本、硬件、文件或运行结果的结论，都必须在回答或
修改前重新检查实际来源，不能只复用本文件或旧日志。

不得把当前单卡、小模型、人工 token 数据上的结论直接推广到真实 LLM、多卡环境或
其他硬件。理论估算、框架统计、profiler 观测和工程推断必须明确区分。

## 3. 项目真相源与优先级

不同文件分别拥有不同事实，禁止选择一个文件代替全部来源。

### 3.1 学习主题与日期安排

主路线图：

```text
docs/roadmaps/ai_infra_knowledge_and_project_roadmap.md
```

它决定每个 Day 原计划学习什么、阶段如何衔接。生成新一天前必须读取对应日期的
完整段落，不能只依赖前一天的预告或对话记忆。

### 3.2 讲义格式与学习闭环

讲义规则和索引：

```text
docs/daily-lessons/README.md
```

它决定讲义结构、知识点讲解顺序、公式格式、实验标准和每日完成定义。

### 3.3 已实现行为

实际代码和当前运行输出是行为事实源：

```text
exercises/day*/
pyproject.toml
uv.lock
.python-version
```

文档声称的 shape、dtype、loss、accuracy、时间、显存、设备支持或恢复一致性必须
来自实际运行，不得根据代码外观或历史记忆补写。

### 3.4 GitHub 项目入口

根 `README.md` 负责说明项目用途、环境准备、学习路径、运行命令、已验证能力和
当前边界。新增课程或验证结果影响这些内容时，应同步更新 README，但不要将详细
Agent 工作流复制进去。

## 4. 每日讲义生成前的强制核对

每次生成新的一天或补充日（例如 Day 7.5）前，必须先完成以下核对。

### 4.1 同时读取四类输入

1. 总路线图中对应 Day 的完整条目；
2. 前一天讲义结尾的下一步、练习和未确认问题；
3. 前一天及共享代码实际暴露的接口；
4. 用户在对话中新增或修改的学习需求。

不得只读取其中一类。

### 4.2 建立覆盖清单

开工前在工作说明中明确：

```text
路线图原计划：A、B、C、D
前一天衔接：B、C
本日准备完成：A、B、C、D
直接依赖：Day X / 共享模块 Y
验收证据：测试或实际输出 Z
```

如果内容过多，必须先向用户说明并给出拆分方案，例如：

```text
本日完成：A、B、C
明确延期：D
延期位置：Day X.5
延期原因：需要独立 baseline 和实测
```

禁止为了控制篇幅静默省略路线图内容。

### 4.3 发现冲突时的处理

- 路线图与前一天预告不一致：以路线图为计划基准，明确说明预告缺失或偏差。
- 路线图与用户最新要求不一致：用户最新明确要求优先，同时记录路线变化。
- 已有文档宣称完成但代码没有证据：标记为未验证，不得沿用完成状态。
- 某主题已经提前完成：核对证据后说明，避免机械重复。

## 5. 每日固定产物

每个 Day 至少包含：

```text
docs/daily-lessons/dayXX_<topic>.md
exercises/dayXX/<minimal_experiment>.py
```

并同步：

- `docs/daily-lessons/README.md` 的讲义索引；
- 根 `README.md` 中受影响的学习路径、运行命令、已验证能力或边界。

### 5.1 讲义产出与实验验证是两个独立状态

`docs/daily-lessons/README.md` 是“已生成正式讲义”的索引，不是“已在当前
本机完成全部实验”的索引。只要当日基础知识、机制讲解、手算、实验方案、代码
入口、验收标准和事实边界已经形成正式 Day 产物，就必须同步创建讲义和索引。

不得因为当前本机缺少多卡、特定 GPU 架构、更大显存、集群、真实服务流量或
其他外部条件，而推迟讲义和索引。此时应将实验拆为可在当前环境完成的
静态检查、正确性子集或 dry run，以及需在目标环境执行的真实验。

讲义和索引必须明确使用以下状态之一：

```text
已生成，待验证    讲义与实验方案已形成，关键实验尚未在目标环境运行
已生成，部分验证  只有部分路径或部分环境已实测，其余边界明确列出
已生成并实测      当日定义的关键实验和直接依赖回归均已获得证据
```

索引中出现某个 Day 只证明正式讲义已经存在，不自动证明其所有实验结论成立。
课程总状态、讲义顶部、当日验收和尚未验证边界必须保持一致。

每份讲义内部至少提供：

- 一份手算练习；
- 一个最小可运行程序；
- 一份实验证据或待执行实验契约：可运行时写入当前环境的实际结果；当前环境
  不具备关键条件时，必须写明目标环境、执行命令、输出契约、成功标准、失败路径
  与待验证边界；
- 三道口述题；
- 一项尚未确认的问题或未验证边界。

补充日可以使用 `day07_5` 这类稳定目录名和 `Day 7.5` 显示名。不得覆盖已经存在的
Day，也不得悄悄重排后续编号。

## 6. 讲义内容的强制顺序

首次出现的知识点按以下顺序讲解：

1. 它解决什么问题；
2. 生活化或可视化直觉；
3. 极小具体例子；
4. 术语、缩写和符号；
5. 逐步数据流；
6. 逐步 shape 和 dtype；
7. 正式公式；
8. 对应代码；
9. AI Infra 成本；
10. 常见误解和成立边界；
11. 最小验证；
12. 30 秒与 2 分钟口述。

每份 Day 文档使用以下主结构：

1. 今日核心问题
2. 前置知识与术语
3. 从直觉到机制
4. 极小手算例子
5. 正式模型配置或实验配置
6. 完整数据流与 Shape/Dtype
7. 参数、内存与计算成本
8. 最小代码验证
9. 常见误解与边界
10. 手算练习
11. 面试口述
12. 当日验收

主题允许小幅调整标题，但不能缺失讲解、数据流、代码、实验、边界和验收。

## 7. Markdown 与数学公式

项目在 VS Code 中使用 Markdown+Math 阅读讲义，统一采用其默认 `dollars` 定界符。

行内公式：

```markdown
$g_1 = 2$
```

块公式：

```markdown
$$
g = \frac{g_1 + g_2}{2}
$$
```

强制规则：

- 不使用 `\(...\)` 或 `\[...\]`；
- 数学环境内下标写 `g_1`，不写 `g\_1`；
- 块公式前后保留空行；
- 简单等式优先用普通文本或 `text` 代码块；
- 不使用 `\qquad` 等只为排版、却降低源码可读性的命令；
- 复杂公式之后必须紧跟自然语言解释；
- 写 `X[B,S,H]` 时，解释每一维的现实含义；
- 不能把 broadcast/view 等逻辑 shape 写成必然物化的物理存储。

## 8. 代码实现规则

- 代码首先服务于验证理论，不以代码量体现深度。
- 优先复用前一天的模型和公共函数，不复制第二套实现。
- 关键输入、输出、shape、dtype 和状态必须使用断言验证。
- 正确性实验先于性能实验。
- 新增实验必须有 baseline 和单一主要变量。
- 不为了得到漂亮结果删掉“没有加速”或“不支持”的案例。
- 不支持的能力必须区分：明确失败、跳过、软件模拟、CPU fallback 和性能退化。
- 训练脚本必须区分 parameter、`.grad`、optimizer state、activation 和 checkpoint。
- 有状态功能必须明确状态所有者、保存字段、恢复顺序和失败边界。

## 9. 实验与测量规则

### 9.1 正确性证据

命令退出码为 0 不是充分证据。还必须检查与目标直接对应的：

- shape 和 dtype；
- loss/accuracy；
- 参数是否变化；
- causal independence；
- 大 batch/梯度累积参数差异；
- continuous/resume 一致性；
- checkpoint 字段和临时文件清理；
- 设备能力检查。

### 9.2 性能实验

必须记录：

```text
硬件与 compute capability
PyTorch 与 CUDA 版本
模型配置
dtype
batch size
sequence length
warmup steps
measured steps
同步方法
显存统计口径
```

CUDA 计时前后使用 `torch.cuda.synchronize()`，并先 warmup。显存必须说明是理论值、
PyTorch allocated/reserved/peak，还是 `nvidia-smi` 进程占用。

一次短 benchmark 只能写成“当前配置的一次观测”，不能写成普遍性能结论。

### 9.3 实验结果写回

只有实际运行通过后，才能把数值写入讲义和 README。环境变化或重新运行结果变化时，
以新证据为准，并说明差异。

“尚未产生实测数值”只限制实验结论和完成状态，不限制创建当日基础讲义、
实验代码、运行手册和讲义索引。待执行部分必须用占位状态和验收契约表达，
不得预填性能、显存、正确性或设备支持数字。

## 10. 回归验证策略

默认使用“当天验证 + 直接依赖链回归”，不机械地每次从 Day 1 全部运行。

- 每次必须运行当前环境可执行的当天新增实验或静态契约检查；若关键路径
  需要当前不具备的目标硬件或外部环境，则保留为待验证状态，不阻塞讲义产出；
- 复用 Day X 的模块时，回归最接近且覆盖该共享路径的已有实验；
- 修改共享模型或公共函数时，回归所有直接受影响的 Day；
- 只新增独立讲义时，不运行无关 GPU 训练；
- 每个阶段结束时运行一次综合验收；
- 不能运行的路径必须明确报告原因和残余风险。

示例：Day 8 若复用 Day 4 模型并新增生成逻辑，应验证 Day 8，并回归覆盖 Day 4
模型行为的最近综合验收；不需要无条件重跑 Day 1–3。

## 11. 安装、GPU 和文件副作用

- 本项目使用 `uv` 和项目 `.venv`，不得把 Python 包安装到系统 site-packages。
- 安装前说明解释器、虚拟环境、缓存位置、将修改的清单/lockfile，并等待用户同意。
- 未经同意不得改变 PyTorch CUDA index、重建 `.venv` 或安装系统 CUDA Toolkit。
- PyTorch wheel 自带的 CUDA 用户态运行库与系统 NVIDIA driver 必须分开描述。
- GPU 实验需要访问真实 NVIDIA 设备；沙箱不可见时按权限流程申请，不得伪造结果。
- 实验 checkpoint、缓存和临时产物优先放入独立临时目录并自动清理。
- 不覆盖用户数据，不删除无关文件，不提交或推送 GitHub，除非用户明确要求。
- `.DS_Store`、`__pycache__` 等生成元数据不得作为正式项目产物提交。

## 12. 当前课程状态

已生成并实测：

```text
Day 1    Tensor shape、参数量与理论显存
Day 2    文本/token IDs 到 logits
Day 3    Causal Attention、RMSNorm、residual
Day 4    Decoder Block、SwiGLU、Cross-Entropy、backward
Day 5    Dataset、AdamW、tiny-data overfit
Day 6    梯度累积、checkpoint、精确 resume
Day 7    第一阶段综合验收
Day 7.5  FP32/FP16 AMP、Autocast、GradScaler
Day 8    朴素自回归生成、prefill、无缓存 decode
Day 9    各层 KV Cache、增量 decode、cache/no-cache 一致性
Day 10   MHA/MQA/GQA、Query/KV heads、KV Cache 容量
Day 11   FlashAttention、Activation Checkpointing、IO-aware 思维
Day 12   参数/梯度/AdamW state、activation、allocator 显存账本
```

已生成，部分验证（待目标环境关键验证）：

```text
Day 13   DDP、FSDP/ZeRO、TP、PP、CP、EP 和 NCCL collectives
         本机静态契约已通过；真实两卡 NCCL/DDP/FSDP2 待验证
Day 14   DDP、gradient bucket、All-Reduce/Reduce-Scatter 与通信计算重叠
         本机 Gloo 双进程正确性已实测；真实双 GPU NCCL 时间线与性能待验证
Day 15   FSDP/ZeRO、状态分片、reshard、mixed precision、distributed checkpoint
         本机 Gloo 双进程分片和同 world-size 恢复已实测；NCCL 双 GPU 待验证
Day 16   TP、PP、CP、SP 与模型并行布局
         本机 Gloo 双进程正确性已实测；真实双 GPU NCCL 时间线与性能待验证
```

原路线图把混合精度列入 Day 6，但 Day 6 初版遗漏；现已通过 Day 7.5 补齐。这个
历史用于提醒后续 Agent 必须执行覆盖清单，不代表以后可以随意改变编号。

下一个讲义产出为 Day 17；Day 13–16 的真实两卡验证作为独立待办，不阻塞后续
基础知识讲义按路线图继续生成。完成租用 GPU 实验后，再回写对应实测数字和状态。

当前尚未验证的主要边界：

- 真实 tokenizer 和文本数据；
- validation/generalization；
- 真实大模型和长上下文上的 KV Cache 加速；
- 预分配、分页缓存与请求级缓存生命周期；
- MHA/GQA/MQA 的真实质量差异与生产 kernel 性能；
- RTX 2060 原生 BF16（设备检查为不支持，实验已跳过）；
- 支持 Flash backend 的 Ampere 或更新 GPU 上的真实 FlashAttention kernel；
- dropout、随机算子与 checkpoint RNG 状态处理，以及不同 checkpoint 分段粒度的最优点；
- 混合精度不同参数所有权、8-bit optimizer 与未分类活分配的 storage 级归因；
- DDP/FSDP/TP 等多 GPU 行为；
- 生产级 checkpoint、scheduler 和分布式 sampler；
- 大模型 profiler、吞吐和显存结论。

## 13. 每日产出与完成报告

生成新一天或补完其实验验证后必须报告：

1. 新增或修改的文件；
2. 当天覆盖了路线图中的哪些条目；
3. 哪些条目被明确延期及其目标日期；
4. 实际运行的命令和关键结果；
5. 回归了哪条直接依赖链；
6. 没有安装什么、没有修改什么；
7. 尚未验证的边界。

报告必须明确区分“讲义已生成”“当前环境已验证”和“目标环境待验证”。

不得只说“内容已生成”或“测试通过”。
