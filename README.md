# AI Infra Learning

这是一个以可运行实验学习 Transformer 训练、推理与 AI Infra 的项目。目前已完成
Day 1–12；Day 13–16 的分布式讲义和实验已经生成。Day 14–16 的 Gloo 双进程
正确性已实测，Day 13–16 的真实双 GPU NCCL 行为仍待验证。后续将进入组合并行、
CS336 Systems、现代 LLM 架构与两个简历项目主线。

项目不以堆叠框架代码为目标，而是要求每个机制都能够回答：输入是什么、Tensor
shape 如何变化、状态由谁维护、显存和计算成本来自哪里，以及怎样用实验直接验证。

## 当前环境

本项目使用：

- Python 3.13；
- `uv` 管理项目环境；
- PyTorch 2.13.0+cu130；
- CUDA 13.0 PyTorch 用户态运行库；
- NVIDIA GeForce RTX 2060。

以上软件与 GPU 信息来自本项目 Day 5–7 的实际运行输出，不代表其他机器上的默认环境。

## 环境准备

项目已经包含 `.python-version`、`pyproject.toml` 和 `uv.lock`。在项目根目录同步环境：

```bash
uv sync
```

验证 PyTorch 与 GPU：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

## 第一阶段学习路径

完整讲义入口见 [`docs/daily-lessons/README.md`](docs/daily-lessons/README.md)。

1. Day 1：Tensor shape、参数量与理论显存。
2. Day 2：文本、token IDs、Embedding、隐藏状态与 logits。
3. Day 3：多头因果 Attention、RMSNorm 与 residual。
4. Day 4：SwiGLU、Decoder Block、LM Head、Cross-Entropy 与 backward。
5. Day 5：Dataset、shifted labels、AdamW 和 tiny-data overfit。
6. Day 6：梯度累积、训练 checkpoint 与精确续跑。
7. Day 7：全链路复盘、项目入口和第一阶段综合验收。
8. Day 7.5：FP32 与 FP16 AMP 的 dtype、稳定性、时间和显存对照。
9. Day 8：朴素自回归生成、prefill、decode 与无 KV Cache 重复计算。
10. Day 9：逐层 KV Cache、增量 decode 与 cache/no-cache 一致性。
11. Day 10：MHA、MQA、GQA 的 Query/KV head 映射与缓存容量。
12. Day 11：FlashAttention、Activation Checkpointing 与 IO-aware 思维。
13. Day 12：参数、梯度、AdamW state、activation 与 allocator 显存账本。
14. Day 13：从单卡训练逐步推导 DDP、FSDP/ZeRO、TP、PP、CP、EP 与 NCCL collectives。
15. Day 14：DDP 进程生命周期、gradient bucket、All-Reduce、Reduce-Scatter 与通信计算重叠。
16. Day 15：FSDP/ZeRO 状态分片、All-Gather/Reduce-Scatter、reshard、mixed precision 与分布式 checkpoint。
17. Day 16：TP 的 row/column/head 切分、PP micro-batch/bubble，以及 CP/SP 的语义边界。

## 运行实验

```bash
python3 exercises/day01/tensor_shape_memory.py
uv run python exercises/day02/text_to_logits.py
uv run python exercises/day03/causal_attention_block.py
uv run python exercises/day04/decoder_block_training.py
uv run python exercises/day05/overfit_training_loop.py
uv run python exercises/day06/gradient_accumulation_checkpoint.py
uv run python exercises/day07/phase_one_acceptance.py
uv run python exercises/day07_5/mixed_precision_training.py
uv run python -m exercises.day08.naive_autoregressive_generation
uv run python -m exercises.day09.kv_cache_generation
uv run python -m exercises.day10.mha_mqa_gqa_cache
uv run python -m exercises.day11.flash_attention_checkpointing
uv run python -m exercises.day12.training_memory_ledger
uv run python -m exercises.day13.two_gpu_distributed_training --validate-only
uv run python -m exercises.day14.ddp_nccl_execution --validate-only
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 -m exercises.day14.ddp_nccl_execution --backend gloo --output /tmp/day14_gloo.json
uv run python -m exercises.day15.fsdp_zero_sharding --validate-only
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 -m exercises.day15.fsdp_zero_sharding --backend gloo --output /tmp/day15_gloo.json
uv run python -m exercises.day16.model_parallelism --validate-only
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 -m exercises.day16.model_parallelism --backend gloo --output /tmp/day16_gloo.json
```

Day 4–12 的 GPU 实测需要 CUDA。每个脚本都有直接断言，命令退出成功只是第一层证据，还应
核对其输出中的 shape、loss、accuracy、参数差异和行为检查。

## 第一阶段已经验证的能力

- 因果 mask 阻止未来 token 改变更早位置输出；
- Decoder LM 能完成 forward、Cross-Entropy 和 backward；
- AdamW 能让固定小数据的完整数据集 loss 从 `3.608013` 降到 `0.002022`，
  teacher-forced token accuracy 达到 100%；
- 两个 micro-batch 的累积更新与对应大 batch 更新在 FP32 浮点容差内一致；
- 训练在第 17 个 optimizer step 保存后，可以由新模型和新 optimizer 恢复到第
  40 步，并与连续训练得到完全相同的参数和 loss；
- 不完整 checkpoint 会被拒绝，实验临时文件会自动清理。
- FP16 AMP 实测产生 FP16 logits 与 FP32 Cross-Entropy loss，同时保持 FP32 参数、
  gradient 和 AdamW state；在当前短实验中相对 FP32 为 `1.069x` 时间比，并减少
  `10.56 MiB` peak allocated；RTX 2060 不支持原生 BF16，因此 BF16 对照被跳过。
- 朴素 greedy generation 能把 prompt 经 prefill 和重复的完整前缀 decode 生成 8 个
  token；本次共处理 68 个 token 位置、构造 4,960 个 Attention score 元素，并通过
  causal independence、确定性和最后位置 logits 选择检查。
- KV Cache 版本只处理 12 个 token 位置、构造 704 个 Attention score 元素；各层最终
  K/V shape 为 `[1,4,12,16]`，cache/no-cache token IDs 完全相同，logits 最大差异为
  `2.3841858e-07`。当前极小模型中缓存版本时间比为 `1.114x`，因此没有宣称加速。
- 固定 8 个 Query heads 时，MHA/GQA/MQA 的 KV heads 为 8/2/1，Query shape 保持
  `[2,8,5,8]`；在两层、batch 2、长度 16、FP32 配置下，缓存容量分别为 32,768、
  8,192、4,096 bytes，即 MHA 的 `1、1/4、1/8`。
- 当前 RTX 2060 上，显式 causal Attention 与强制 math SDPA 的输出及 Q/K/V
  梯度一致；`EFFICIENT_ATTENTION` 可用，`FLASH_ATTENTION` 不可用，因此未声称实测
  FlashAttention kernel。Activation Checkpointing 将 saved logical bytes 从 `39.500 MiB`
  降至 `3.500 MiB`，CUDA peak delta 从 `30.502 MiB` 降至 `18.026 MiB`，训练步
  首次时间比为 `1.283x`，Day 12 前两次回归为 `1.833x` 和 `1.700x`，说明短
  benchmark 有明显波动；输出和梯度校验始终通过。
- FP32+AdamW 显存账本实测 4,491,520 个参数占 `17.134 MiB`，梯度同为
  `17.134 MiB`，两个 Adam moments 占 `34.268 MiB`；训练窗口 peak allocated
  为 `168.382 MiB`。`zero_grad(set_to_none=True)` 后梯度账本归零，但 reserved
  仍为 `188 MiB`，验证了静态 Tensor 账本、瞬时峰值与 allocator 缓存是不同口径。
- Day 14 在 CPU/Gloo 上启动两个本地进程：手工逐参数 All-Reduce 与 DDP bucket
  路径的参数更新最大差异为 `0.0`；`no_sync()` 首个 micro-batch 通信次数为 0，
  最终更新与手工 baseline 最大差异为 `3.129243850708008e-07`。这只证明当前小模型
  的分布式正确性，不代表 NCCL/GPU 性能。
- Day 15 在 CPU/Gloo 上运行 FSDP2：29,344 个逻辑参数和 gradients 各分为每 rank
  14,672 个 local elements；AdamW 每 rank 保存 29,344 个 moment shard 元素和
  22 个 scalar steps。临时 distributed checkpoint 在主动破坏参数后恢复，完整参数
  最大差异为 `0.0`，并在清理临时目录后才记录 PASS。
- Day 16 在 CPU/Gloo 上用 PyTorch framework APIs 运行双进程：TP MLP 输出/梯度最大
  差异为 `5.960464477539063e-08`/`3.958120942115784e-09`，head split 输出最大差异
  为 `6.705522537231445e-08`，SP 与实验性 CP 输出差异为 `0.0`；GPipe 两个 stage
  均产生参数梯度。这证明当前 FP32 小配置的布局与数值语义，不代表 GPU/NCCL 性能。

这些结论只覆盖当前小模型、人工 token 数据、单张 RTX 2060 和当前软件版本。

## 当前边界

本阶段尚未验证：

- tokenizer 与真实文本数据管线；
- validation split、泛化能力和真实语言生成质量；
- 真实大模型、长上下文和优化 kernel 下的 KV Cache 加速；
- 预分配或分页缓存，以及请求取消时的缓存释放；
- MHA/GQA/MQA 的真实模型质量与生产 kernel 性能差异；
- 支持 Flash backend 的 Ampere 或更新 GPU 上的真实 FlashAttention kernel；
- dropout、随机算子与 checkpoint RNG 状态处理，以及不同 checkpoint 分段粒度的最优点；
- 混合精度不同参数所有权、8-bit optimizer 与未分类活分配的 storage 级归因；
- DDP/FSDP/TP/PP/CP 的真实双 GPU NCCL 时间线、显存和性能，FSDP BF16 与不同
  world-size checkpoint 恢复，以及 causal CP 和 pipeline bubble 测量；
- 学习率 scheduler、分布式 sampler 和生产级 checkpoint 管理；
- 性能 profiler、吞吐量和大模型显存结论。

Day 7.5 已补充 FP16 AMP 对照；BF16 是否实测取决于当前 GPU 的原生支持检查，
不支持时会明确跳过。

## 目录结构

```text
docs/
  daily-lessons/   每日讲义与验收题
  roadmaps/        总体学习路线
  knowledge/       专题知识材料
exercises/
  day01 ... day16  与每日讲义对应的最小实验
pyproject.toml      Python 项目与依赖声明
uv.lock             可复现依赖锁文件
```

## 下一阶段

下一步是 Day 17：EP 与 DP/TP/PP/CP 的组合并行和通信组边界。
Day 13–16 的真实双 GPU 验证作为独立待办保留，不阻塞基础讲义继续推进。
完整长期顺序为：分布式与 CS336 Systems → 现代 LLM 架构 → MemScope → Nano-vLLM
→ 投机解码 → Mini-SGLang Memory-Aware Serving → Kubernetes。
