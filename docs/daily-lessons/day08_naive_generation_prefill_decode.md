# Day 8：朴素自回归生成、Prefill 与 Decode

## 1. 今日核心问题

训练 Decoder LM 时，我们把一整段 token 并行送入模型；真正生成文本时，模型却要
一次只决定一个新 token。今天要回答：一段 prompt 如何变成连续生成的 token，
prefill 和 decode 分别指什么，以及没有 KV Cache 时为什么会产生大量重复计算。

今天完成朴素生成基线。KV Cache 的状态结构、增量 Attention 和 cache/no-cache
输出一致性留到 Day 9，因为必须先有一个正确、可测量的无缓存基线。

## 2. 前置知识与术语

- **Prompt**：用户已经提供给模型的 token 序列。
- **Prefix（前缀）**：某次预测时模型已经能看到的全部 token。它开始时等于 prompt，
  后来还包含模型已经生成的 token。
- **Autoregressive generation（自回归生成）**：把上一次生成的 token 接回输入，再预测
  下一次，循环进行。
- **Prefill（提示词填充阶段）**：第一次把完整 prompt 送过所有 Decoder 层，建立 prompt
  的隐藏状态；有 KV Cache 时还会填充各层的 K/V。
- **Decode（逐 token 解码阶段）**：prefill 之后，每轮根据现有前缀生成一个新 token。
- **Logits**：LM Head 输出的未归一化分数；词表中的每个 token 都有一个分数。
- **Greedy decoding（贪心解码）**：每次选择 logit 最大的 token。
- **Sampling（采样）**：把 logits 转成概率分布，再按概率随机抽取 token。

这里的 decode 指“生成阶段”，不是 Encoder-Decoder 架构中特指的 Decoder 模块。

## 3. 从直觉到机制

假设 prompt 是：

```text
[我, 喜欢, 学习]
```

第一次 forward 同时计算三个位置。模型会输出三个位置的 logits，但要继续这段文字，
只使用最后位置“学习”之后的 logits。假设贪心选择“模型”，新前缀变成：

```text
[我, 喜欢, 学习, 模型]
```

第二轮再预测“模型”后面是什么。若选择“原理”，前缀继续增长。生成循环是：

```text
完整 prompt
→ forward
→ 取最后位置 logits
→ 选择一个 token
→ 拼到前缀末尾
→ 再次 forward
```

第一轮完整 prompt 的计算称为 prefill。后续轮次逻辑上称为 decode，但今天的朴素实现
没有 KV Cache，所以每次 decode 仍把整个前缀重新计算一遍，并非只计算新 token。

### 3.1 为什么只取最后位置 logits

输入长度为 $S$ 时，模型输出 `logits[B,S,V]`：

- `B`：batch 中有多少条序列；
- `S`：当前前缀有多少个位置；
- `V`：词表中有多少个候选 token。

`logits[:, i, :]` 表示根据位置 `0...i` 的信息预测位置 `i+1`。前缀中较早位置对应的
预测已经成为过去；继续生成只需要 `logits[:, -1, :]`，其 shape 是 `[B,V]`。

这不等于模型只计算了最后一个 Query。今天复用的完整 forward 会为全部 $S$ 个位置
重新计算 Q、K、V 和 Attention；只是最后选择 token 时丢弃了前面位置的 logits。

### 3.2 Causal Mask 在 Prefill 中仍然必要

prefill 并行处理 prompt 的全部位置。位置 0 不能读取位置 1 及其后面的 token，否则其
隐藏状态会混入未来信息，与训练时的因果关系不一致。因此完整 prompt 的 self-attention
仍使用 causal mask。

只计算一个新 Query 的增量 decode 不存在“这个 Query 后面的未来 K/V”，通常不需要再
构造完整三角 mask。但这是 Day 9 引入 KV Cache 后的实现，不是今天朴素 forward 的行为。

### 3.3 Greedy 与 Sampling

贪心选择可写为：

$$
y_{t+1} = \operatorname*{argmax}_{v} z_{t,v}
$$

$z_{t,v}$ 是当前最后位置对词表 token $v$ 的 logit。`argmax` 每次选择最大值，所以同一
模型和同一 prompt 会得到确定结果。

采样通常先用温度 $T$ 调整 logits：

$$
p_v = \operatorname{softmax}\left(\frac{z_v}{T}\right)
$$

$T$ 较小会让分布更尖锐，$T$ 较大使候选更接近。采样还可能配合 top-k 或 top-p。
Day 9 的 cache/no-cache 一致性将使用 greedy，避免随机数掩盖实现差异。

## 4. 极小手算例子

prompt 长度 $P=3$，要生成 $N=4$ 个 token。无缓存生成的四次 forward 长度为：

```text
第 0 步 prefill：3
第 1 步 decode： 4
第 2 步 decode： 5
第 3 步 decode： 6
```

一共处理的 token 位置数：

```text
3 + 4 + 5 + 6 = 18
```

注意，生成第 4 个 token 后没有再把它送入模型，因为任务已经结束。理想 KV Cache
实现只需在 prefill 处理 3 个 prompt token，后面三次各处理一个新 token：

```text
3 + 1 + 1 + 1 = 6
```

Attention 的差距更明显。无缓存每步的 score 矩阵都是方阵，单个 head 共构造：

```text
3² + 4² + 5² + 6² = 86 个 score 元素
```

有缓存时，prefill 仍是 `3 × 3`；后续新 Query 分别读取长度 4、5、6 的 K/V：

```text
3² + 4 + 5 + 6 = 24 个 score 元素
```

这是逻辑元素计数，不等同于真实 FLOPs、运行时间或物理显存分配。融合算子可能不会
物化完整 score 矩阵，kernel launch、带宽和小矩阵利用率也会影响时间。

## 5. 正式模型与实验配置

Day 8 直接复用 Day 4 的 `TinyDecoderLM`，没有复制第二套 Transformer：

```text
词表大小 V                 32
隐藏维度 H                 64
Attention heads            4
FFN 中间维度               128
Decoder layers             2
最大序列长度               32
batch size B               1
prompt 长度 P              5
生成 token 数 N            8
参数 dtype                 FP32
选择策略                    greedy argmax
warmup                      5 次完整生成
measured                    20 次完整生成
```

模型使用固定随机种子初始化，但没有经过语言训练，因此输出的 token ID 只用于验证生成
机制，不能解释成有意义的文字。

## 6. 完整数据流与 Shape/Dtype

初始输入是 `prompt_ids[B,P] = [1,5]`，dtype 为整数 `torch.int64`。

### 6.1 Prefill

```text
prompt_ids                 [1,5]       int64
token + position embedding[1,5,64]    float32
每层 Q/K/V                [1,4,5,16]  float32
Attention scores          [1,4,5,5]   float32
最终 hidden state         [1,5,64]    float32
LM Head logits            [1,5,32]    float32
最后位置 logits           [1,32]      float32
argmax next_token         [1,1]       int64
```

`[1,4,5,16]` 中四维分别是 batch、head 数、序列位置和每个 head 的维度。
`argmax` 把词表维消去，得到一个 token ID。

### 6.2 无缓存 Decode

第一个 token 拼接后，输入从 `[1,5]` 变为 `[1,6]`。下一轮完整 forward 是：

```text
grown prefix               [1,6]       int64
每层 Q/K/V                 [1,4,6,16]  float32
Attention scores          [1,4,6,6]   float32
LM Head logits            [1,6,32]    float32
最后位置 logits           [1,32]      float32
```

随后长度依次增长到 7、8、9、10、11、12。每一轮都重新计算旧 token 的中间结果。

### 6.3 位置与上下文限制

Day 4 模型使用绝对位置 embedding。每次完整重算时，前缀仍从位置 0 开始，因此位置编号
正确。`prompt 长度 + 生成数` 不能超过配置的最大序列长度 32；实验在生成前主动检查。

## 7. 参数、内存与计算成本

无缓存生成 $N$ 个 token 时，完整 forward 处理的 token 位置数为：

$$
\sum_{t=0}^{N-1}(P+t)=NP+\frac{N(N-1)}{2}
$$

本实验 $P=5$、$N=8$，所以处理 68 个 token 位置，而最终只新增 8 个 token。

对于标准 Attention，单层单 head 构造的 score 元素数为：

$$
\sum_{t=0}^{N-1}(P+t)^2
$$

乘以 2 层和 4 个 head 后，本实验实际返回的 Attention weight 张量总计 4,960 个元素。

无缓存并不意味着峰值显存等于所有轮次激活之和。推理在 `no_grad()` 下运行，上一轮
中间张量在下一轮前可以释放；peak 更接近最长那一轮加模型参数和框架开销。KV Cache
会减少重复计算，但会引入随层数和已缓存序列长度增长的持久 K/V 状态。

## 8. 最小代码验证

实验文件：

```text
exercises/day08/naive_autoregressive_generation.py
```

从项目根目录运行：

```bash
uv run python -m exercises.day08.naive_autoregressive_generation
```

核心循环只有四步：完整 forward、取最后 logits、argmax、拼接 token。脚本进一步断言：

- 相同模型和 prompt 运行两次，greedy 结果逐 token 完全相同；
- 每个生成 token 都等于对应前缀最后位置 logits 的 argmax；
- 任意修改较早位置 logits，不会改变本轮选择；
- 输入长度和 Attention score 元素数符合无缓存公式；
- Day 4 的 causal independence 仍成立。

### 8.1 当前机器的实际输出

2026-08-30 在 NVIDIA GeForce RTX 2060、PyTorch 2.13.0+cu130 上得到：

```text
prompt token IDs:               [1, 5, 9, 4, 3]
generated token IDs:            [1, 5, 9, 4, 3, 17, 3, 29, 12, 27, 22, 13, 8]
各轮输入长度:                   5, 6, 7, 8, 9, 10, 11, 12
token positions processed:      68
attention score elements:       4960
warmup steps:                   5
measured steps:                 20
average generation time:        5.316 ms
CUDA peak allocated:            8.51 MiB
causal independence:            PASS
greedy determinism:             PASS
last-position selection:        PASS
naive recomputation accounting: PASS
```

计时包含 8 轮 Python 生成循环、完整模型 forward、argmax 和拼接，并在测量前后执行
`torch.cuda.synchronize()`。它只是当前小模型配置的一次观测，不能外推真实 LLM 吞吐。

## 9. 常见误解与边界

### 9.1 “只用最后 logits，所以只需计算最后 Query”

这是优化后的目标，不是普通完整 forward 自动做到的行为。今天模型收到整个前缀，内部
仍为所有位置生成 Q/K/V；只有最后的 token 选择使用了最后一行 logits。

### 9.2 “Prefill 只计算最后一个 token”

不对。prefill 计算完整 prompt 的所有层状态。有 KV Cache 时，目的之一正是建立所有
prompt token 在每一层的 K/V，供后续新 Query 读取。

### 9.3 “Decode 一定只输入一个 token”

decode 是逻辑阶段。是否只输入一个 token 取决于接口是否保存并读取 KV Cache。无缓存
基线的 decode 每轮输入完整增长前缀。

### 9.4 “Greedy 输出最好”

greedy 只保证当前一步选择最高分，不保证整条序列全局概率最高，也可能产生重复内容。
采样可增加多样性，但结果受温度、过滤策略和随机数状态影响。

### 9.5 当前尚未验证

- KV Cache 的各层 shape、状态所有者和增量位置；
- cache/no-cache 输出是否逐 token 一致；
- 真实 tokenizer、EOS、padding 和 batch 内不同序列长度；
- beam search、top-k、top-p 和重复惩罚；
- 生产推理引擎的 fused kernel、continuous batching 与 paged KV Cache。

## 10. 手算练习

### 练习 1

prompt 长度为 4，要生成 5 个 token。写出无缓存每轮 forward 的输入长度，并计算总共
处理多少个 token 位置。

答案：输入长度是 `4, 5, 6, 7, 8`，总数是 `30`。

### 练习 2

模型输出 logits shape 为 `[2,7,100]`。继续生成一个 token 时，应该选哪一片 logits？
选取后的 shape 是什么？

答案：使用 `logits[:, -1, :]`，shape 是 `[2,100]`；对词表维做 argmax 后得到 `[2]`，
若要与输入拼接，通常再保持或恢复为 `[2,1]`。

### 练习 3

prompt 长度为 2，要生成 3 个 token。模型有 2 层、2 个 head。无缓存实现一共返回多少
个 Attention score 元素？

答案：每个层和 head 是 `2² + 3² + 4² = 29`，总数为 `29 × 2 × 2 = 116`。

## 11. 面试口述

### 问题 1：Prefill 和 Decode 有什么区别？

30 秒回答：prefill 首次并行处理完整 prompt，计算其隐藏状态，并在缓存实现中建立各层
K/V。decode 在此后逐 token 生成。没有 KV Cache 时，decode 虽然逻辑上逐 token，实际
仍会反复计算完整前缀。

### 问题 2：为什么生成时只取最后位置 logits？

30 秒回答：位置 $i$ 的 logits 用前缀 `0...i` 预测下一个位置。当前前缀的早期预测已经
成为过去，要追加一个 token，只需最后位置对整个词表的分数。但完整 forward 可能仍计算
所有位置，不能把“只使用最后 logits”和“只计算最后 Query”混为一谈。

### 问题 3：无缓存生成为什么慢？

2 分钟回答要点：每生成一个 token，前缀长度增加一；Q/K/V 投影和 Block 计算反复覆盖
旧位置；Attention score 每轮又形成增长的方阵；总 token 处理量是
$NP+N(N-1)/2$，score 成本则按平方长度累加。KV Cache 的核心是保存旧 token 各层 K/V，
让 decode 只投影新 token，并让单个新 Query 读取历史 K/V，但代价是持久显存和状态管理。

## 12. 当日验收

- [ ] 能画出 `prompt → prefill → 最后 logits → token → decode` 循环。
- [ ] 能解释为什么只使用最后 logits 不代表只计算最后 Query。
- [ ] 能手算无缓存生成处理的 token 位置数和 score 元素数。
- [ ] 能区分 greedy 与 sampling 的确定性。
- [ ] 能运行 Day 8 实验并解释 68 和 4,960 从哪里来。
- [ ] 能说出 KV Cache 减少什么计算，又新增什么状态。

下一步 Day 9：为每个 Decoder 层显式保存 K/V，只计算新 token 的 Query/K/V，并用
greedy generation 验证 cache 和 no-cache 的 token IDs 与 logits 在浮点容差内一致。
