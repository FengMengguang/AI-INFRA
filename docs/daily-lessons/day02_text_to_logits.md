# Day 2：从原始文本到 Logits——Tokenizer、Embedding、位置与 Transformer 架构

## 1. 今日核心问题

今天只解决一个完整问题：

> 一段人类可读的文本，如何一步步变成语言模型能够计算的张量，并最终得到下一个 token 的 logits？

完整主线是：

```text
原始文本
→ Unicode 字符串
→ Tokenizer
→ token IDs
→ 分块、padding 和 mask
→ Token Embedding
→ 加入位置信息
→ 多层 Transformer Blocks
→ 最终归一化
→ LM Head
→ logits
```

今天完成后，应能回答：

1. tokenizer 为什么不能简单地“一词一个 token”？
2. Byte-level tokenizer 遇到陌生字符为什么通常不需要 `[UNK]`？
3. token ID 为什么不是词义大小，Embedding 又做了什么？
4. 为什么只有 Token Embedding 还不够，模型必须知道位置？
5. Encoder、Encoder–Decoder 和 Decoder-only 的输入输出分别是什么？
6. 一条文本从字符串到 `[B,S,V]` logits 的 shape 如何变化？

今天暂不实现 Attention 内部计算。Day 3 会实现 MHA、causal mask、RMSNorm 和 residual。今天先建立它们在完整数据流中的位置。

## 2. 昨日回忆

在不查看 Day 1 的情况下回答：

1. `X[B,S,H]` 的三维分别是什么？
2. 为什么 Attention Scores 是 `[B,Nq,S,S]`？
3. `S` 从 128 增加到 512 时，hidden 和 Scores 分别增长多少倍？
4. 权重与激活有什么区别？

如果不能解释原因，先复习 Day 1 对应部分，再继续今天内容。

## 3. 前置知识与术语

### 3.1 今日正式配置

沿用 Day 1 的教学模型：

```text
B = 2       一个 batch 有 2 条独立序列
S = 8       今天用较短长度方便画图；正式项目可改为 128
V = 8000    tokenizer 词表大小
H = 384     hidden size
L = 6       Transformer Block 数量
```

今天缩短 `S` 不会改变机制，只是让例子更容易观察。

### 3.2 术语

- **Corpus（语料库）**：用于训练 tokenizer 或模型的大量文本。
- **Tokenizer（分词器）**：把文本编码成 token IDs，也负责把 token IDs 解码回文本。
- **Token**：tokenizer 选定的基本文本单位，可能是单词、子词、字符、字节或它们的组合。
- **Vocabulary（词表）**：token 与整数 ID 之间的固定映射。
- **Token ID**：某个 token 在词表中的整数编号。
- **Special token（特殊 token）**：承担控制含义的 token，例如 BOS、EOS、PAD。
- **Embedding**：把离散整数 ID 查表转换成连续向量。
- **Position information（位置信息）**：让模型区分相同 token 出现在不同位置。
- **Logit**：softmax 前的未归一化词表分数。

## 4. 为什么模型不能直接读取字符串

### 4.1 要解决的问题

神经网络的核心算子是矩阵乘法，它只能处理数值张量，不能直接把：

```text
"我喜欢学习AI"
```

送入矩阵乘法。

所以首先要建立：

```text
文本片段 ↔ 整数 ID
```

但直接为每个完整句子分配ID不可行，因为句子组合近乎无限。直接“一词一个ID”也会遇到新词、拼写变化、多语言、emoji、代码和生僻字符。

### 4.2 Token不是天然存在的语言单位

同一段文本可以有多种切分：

```text
unbelievable

方案一：[unbelievable]
方案二：[un] [believ] [able]
方案三：[u] [n] [b] ...
方案四：UTF-8 bytes
```

中文也不固定“一字一个token”：

```text
人工智能

可能是：[人工] [智能]
也可能是：[人工智能]
也可能是：[人] [工] [智] [能]
```

切分结果由 tokenizer 的算法、训练语料和词表决定，不由中文或英文天然决定。

## 5. Unicode、UTF-8与Byte-level回退

### 5.1 三个层次

```text
字符“你”
→ Unicode码点：U+4F60
→ UTF-8字节：E4 BD A0
```

Unicode 为字符规定编号；UTF-8 规定如何把码点编码成 1～4 个 byte。一个 byte 有 8 bits，因此有：

```text
2^8 = 256
```

种基础取值，即 `00` 到 `FF`。

### 5.2 “退回bytes”是什么意思

Byte-level tokenizer 的基础字母表能覆盖256种byte。训练时，它会把高频相邻byte逐步合并成更大的token。

例如常见字符“你”的UTF-8表示是：

```text
E4 BD A0
```

如果训练后词表包含整个组合，它可能编码成一个token；如果不包含，就可以退回：

```text
[byte_E4] [byte_BD] [byte_A0]
```

对于没见过的字符 `𠮷`：

```text
𠮷 → UTF-8：F0 A0 AE B7
   → [byte_F0] [byte_A0] [byte_AE] [byte_B7]
```

因此原始信息仍能恢复。它和 `[UNK]` 的区别是：

```text
[UNK]：多个未知字符可能映射成同一个ID，原始信息丢失
bytes：不同字符保留不同byte序列，原则上可以无损解码
```

但“能够编码”不等于“模型理解”。模型若从未在训练中见过某个字符，即使 tokenizer 能编码，模型也未必知道其含义。

### 5.3 原始文件编码发生在Tokenizer之前

数据文件可能使用 UTF-8、GBK 等编码。正确流程是：

```text
文件bytes
→ 按真实文件编码解码
→ Unicode字符串
→ tokenizer
```

如果将GBK文件错误地按UTF-8解码，可能在进入tokenizer前就报错或产生乱码。Byte-level tokenizer不能自动修复已经被错误解码的数据。

## 6. Tokenizer如何训练

### 6.1 它要解决什么权衡

tokenizer 希望同时做到：

- 任意文本都能表示；
- 常见内容使用较少token；
- 词表不要大到Embedding参数失控；
- 编码和解码稳定可复现。

tokenizer训练通常不是神经网络的反向传播，而是对语料做频率统计和离散合并或概率建模。

### 6.2 BPE极小例子

BPE 全称是 Byte Pair Encoding。它反复寻找高频相邻单元并合并。

假设语料为：

```text
low
lower
lowest
```

开始时先使用较小单位：

```text
l o w
l o w e r
l o w e s t
```

如果 `l+o` 高频，合并为 `lo`：

```text
lo w
lo w e r
lo w e s t
```

如果 `lo+w` 仍然高频，再合并：

```text
low
low e r
low e s t
```

反复进行，直到达到目标词表大小或合并次数。

训练产物至少需要保存：

```text
token ↔ ID 的词表
合并规则或分词模型
文本规范化规则
特殊token定义
```

这些内容必须和模型一起版本化。更换tokenizer会改变token IDs，原模型的Embedding行将不再对应原来的文本单位。

### 6.3 词表大小的权衡

词表增大可能：

- 让常见文本使用更少token；
- 缩短实际序列，降低部分序列计算成本；
- 但增大Embedding和LM Head参数；
- 增大每个位置产生的logits维度和softmax成本；
- 让低频token的训练样本更稀疏。

所以词表越大并不必然越好。

## 7. 从Token到Token ID

### 7.1 极小词表

为方便手算，先用一个人工词表：

```text
0  <PAD>
1  <BOS>
2  <EOS>
3  我
4  喜欢
5  学习
6  AI
7  你
```

文本：

```text
我喜欢学习AI
```

假设被编码为：

```text
[<BOS>, 我, 喜欢, 学习, AI, <EOS>]
```

对应：

```text
[1,3,4,5,6,2]
```

ID只是行号。`6`比`3`大不代表“AI”的语义大于“我”，ID之间没有天然距离或大小关系。

### 7.2 特殊Token

常见特殊token包括：

```text
<BOS>：序列开始
<EOS>：序列或文档结束
<PAD>：对齐batch长度
<UNK>：无法表示的未知内容；byte-level方案通常尽量避免它
```

聊天模型还可能定义 system、user、assistant 边界token。其具体名称和格式由模型与tokenizer共同规定，不能在不同模型间随意替换。

### 7.3 文档边界不能静默丢失

如果把两个文档直接拼接：

```text
文档A最后一句 + 文档B第一句
```

模型可能被训练成文档A后自然接文档B。常见做法是在边界插入EOS等特殊token：

```text
文档A <EOS> 文档B <EOS>
```

是否允许跨文档Attention是独立的数据策略，需要结合packing和attention mask明确设计。

## 8. Batch、Padding与Mask

### 8.1 为什么需要Padding

两条序列可能长度不同：

```text
序列A：[1,3,4,5,6,2]       长度6
序列B：[1,7,2]             长度3
```

为了组成规则的二维张量，可把短序列补到当前batch的最大长度：

```text
input_ids =
[
  [1,3,4,5,6,2],
  [1,7,2,0,0,0]
]

shape = [B,S] = [2,6]
```

这里的 `S=6` 是该batch的padding长度，不一定是模型最大上下文长度。

### 8.2 Attention mask

模型不应把PAD当作正常上下文，因此需要标记有效位置：

```text
attention_mask =
[
  [1,1,1,1,1,1],
  [1,1,1,0,0,0]
]
```

这里：

```text
1：有效token
0：padding位置
```

具体框架可能用布尔值、0/1或加性负无穷mask，语义相同但表示形式不一定相同。

### 8.3 Padding mask与Causal mask不是一回事

```text
Padding mask：不看补齐出来的PAD
Causal mask：Decoder-only中不允许当前位置看未来token
```

它们可以同时存在。Day 3 会展开causal mask矩阵。

## 9. Token Embedding：从离散ID到连续向量

### 9.1 为什么ID不能直接参与语义计算

如果直接使用ID：

```text
我=3，喜欢=4，学习=5
```

模型可能错误地把编号大小当作数值关系。ID只是索引，不携带可学习语义。

所以建立Embedding table：

```text
W_embed[V,H]
```

正式配置中：

```text
W_embed[8000,384]
```

每一行对应一个token的384维可训练向量。

### 9.2 查表不是与One-hot显式相乘

如果输入：

```text
input_ids[2,6]
```

对每个ID取Embedding矩阵对应行：

```text
input_ids[B,S]
→ embedding lookup
→ X_token[B,S,H]

[2,6]
→ [2,6,384]
```

数学上可以等价理解为one-hot向量乘Embedding矩阵，但工程实现不会物化巨大的one-hot矩阵，而是直接索引行。

### 9.3 Embedding如何学到含义

Embedding初始通常是随机数。训练中，预测误差通过反向传播更新本batch出现过的相关Embedding行。

它不是由tokenizer提前写入“词义”，而是在语言模型训练目标下逐渐形成有用表示。

## 10. 为什么模型还需要位置信息

### 10.1 要解决的问题

如果只使用Token Embedding：

```text
“我喜欢你”
“你喜欢我”
```

两者包含同一组token。Attention本身需要额外机制区分token出现的先后和相对位置。

### 10.2 绝对位置Embedding

一种直观方法是准备位置表：

```text
W_pos[max_seq_len,H]
```

第0个位置取第0行，第1个位置取第1行：

```text
X_input = TokenEmbedding(input_ids) + PositionEmbedding(position_ids)
```

shape：

```text
X_token：[B,S,H]
X_pos：  [1,S,H] 或 [B,S,H]
相加后：[B,S,H]
```

这里可能使用broadcast：同一套位置向量被batch中的多条序列共享，不一定真的复制成 `[B,S,H]`。

### 10.3 RoPE预告

较多Decoder-only模型使用RoPE（Rotary Position Embedding，旋转位置编码）。它通常不是把一个位置向量直接加到hidden state，而是在Attention内部根据位置旋转Q和K的部分维度。

今天只记住边界：

```text
绝对位置Embedding：通常加到输入表示上
RoPE：通常作用在每层Attention的Q和K上
```

两者都提供位置信息，但实现位置和数学机制不同。不能把所有模型都画成 `Token Embedding + Position Embedding`；画具体模型时要按其真实架构标注。

## 11. 三类Transformer架构

### 11.1 Encoder-only

Encoder 的目标通常是理解完整输入。每个位置通常可以看见输入中的左右两侧token：

```text
输入文本
→ Tokenizer
→ Encoder Blocks
→ 每个位置的上下文化表示
→ 分类、抽取或Embedding等任务头
```

典型用途包括文本分类、序列标注和表示学习。这里“可以双向看”是典型Encoder self-attention机制；具体任务仍可能增加其他mask。

输入输出shape通常保持：

```text
[B,S,H] → 多层Encoder → [B,S,H]
```

它不必通过LM Head执行自回归下一个token生成。

### 11.2 Encoder–Decoder

Encoder先读取源序列，Decoder再根据源序列和已经生成的目标前缀产生输出。

以翻译为例：

```text
英文源文本
→ Encoder
→ encoder memory
                 ↘
已生成的中文前缀 → Decoder → 下一个中文token
```

Decoder中有两种Attention：

1. **Masked self-attention**：目标token只能看已经出现的目标前缀。
2. **Cross-attention**：Decoder的Query读取Encoder输出形成的Key/Value。

shape示意：

```text
源序列：    [B,S_src,H]
Encoder输出：[B,S_src,H]

目标序列：  [B,S_tgt,H]
Decoder输出：[B,S_tgt,H]
logits：    [B,S_tgt,V]
```

`S_src` 和 `S_tgt` 不需要相等。

### 11.3 Decoder-only

Decoder-only模型把提示词和待生成内容组织到同一条token序列中：

```text
前缀token
→ causal Transformer Blocks
→ 每个位置预测它的下一个token
```

每个位置只能看到自己及之前的位置，不能看到未来。训练时虽然所有位置可以并行计算，但每个位置都受causal mask约束。

典型shape：

```text
input_ids[B,S]
→ hidden[B,S,H]
→ logits[B,S,V]
```

当生成下一个token时，通常取最后一个有效输入位置的logits：

```text
next_token_logits = logits[:, last_valid_position, :]
shape = [B,V]
```

然后再经过temperature、top-k/top-p和选择策略得到下一个token ID。

### 11.4 不要只用“有没有Decoder”判断架构

原始Transformer的Decoder包含cross-attention；现代所谓Decoder-only模型通常没有Encoder，因此也没有读取Encoder输出的cross-attention。

所以：

```text
Encoder–Decoder中的Decoder
≠
Decoder-only模型中的Block完全相同
```

二者都使用causal self-attention，但是否包含cross-attention不同。

## 12. Decoder-only完整数据流与Shape

使用正式配置：

```text
B=2, S=8, V=8000, H=384, L=6
```

### 12.1 文本到IDs

```text
两条原始文本
→ tokenizer编码、截断或分块、padding
→ input_ids[2,8]
→ attention_mask[2,8]
```

### 12.2 IDs到输入hidden states

```text
input_ids[2,8]
→ W_embed[8000,384]查表
→ X_token[2,8,384]
```

加入位置信息：

```text
绝对位置方案：X_token + X_pos → X[2,8,384]
RoPE方案：先保留X，进入每层Attention后作用于Q/K
```

### 12.3 经过Transformer Blocks

每个Block通过residual保持外部shape：

```text
Block 0：[2,8,384] → [2,8,384]
Block 1：[2,8,384] → [2,8,384]
...
Block 5：[2,8,384] → [2,8,384]
```

shape相同不表示数值不变。每一层都会根据上下文更新表示。

### 12.4 最终归一化与LM Head

```text
hidden[2,8,384]
→ final norm
→ normalized_hidden[2,8,384]

[2,8,384] @ W_lm[384,8000]
→ logits[2,8,8000]
```

每个位置都有8000个词表分数。

### 12.5 为什么训练时每个位置都有Logits

假设token序列是：

```text
[BOS, 我, 喜欢, AI, EOS]
```

训练关系是：

```text
看到BOS          → 预测“我”
看到BOS,我       → 预测“喜欢”
看到BOS,我,喜欢  → 预测“AI”
看到...AI        → 预测EOS
```

所以输入和标签通常错开一位：

```text
inputs： [BOS, 我,   喜欢, AI]
labels： [我,  喜欢, AI,   EOS]
```

训练可以并行产生所有输入位置的logits，因为真实token都已知；causal mask确保每个位置没有偷看右侧标签。具体loss将在Day 4和Day 5实现。

## 13. 参数、内存与计算成本

### 13.1 词表影响Embedding和LM Head

Embedding参数量：

```text
V × H = 8000 × 384 = 3,072,000
```

如果LM Head不共享权重，还需要同样数量的参数。

词表增大到原来的2倍，在 `H` 不变时：

```text
Embedding参数约2倍
独立LM Head参数约2倍
每个位置logits元素数约2倍
```

### 13.2 Logits可能比Hidden大很多

本日配置：

```text
hidden：[2,8,384]
元素数 = 6,144

logits：[2,8,8000]
元素数 = 128,000
```

logits元素数约为hidden的：

```text
8000 / 384 ≈ 20.83倍
```

训练实现通常会结合loss kernel、分片词表或避免长期保留不必要张量，但逻辑输出仍是 `[B,S,V]`。

### 13.3 Tokenizer效率会影响Infra成本

如果同一段文本：

```text
Tokenizer A → 1000 tokens
Tokenizer B → 1400 tokens
```

那么B不仅意味着更多token计费，还会影响：

- 训练样本的有效文本密度；
- Attention中的序列长度成本；
- 推理prefill计算；
- KV Cache占用；
- 服务可容纳的并发量。

但不能只追求token越少越好，因为过大词表也会增加Embedding、LM Head和softmax成本，并可能造成低频token学习不足。

### 13.4 Padding也会浪费计算

如果真实长度分别是：

```text
[8,2]
```

却统一padding到8，第二条序列有6个PAD位置。即使mask保证语义正确，某些实现仍可能对padding位置进行部分计算。

常见工程策略包括按长度分桶、动态padding和packing。是否真正节省计算取决于kernel和执行方式，不能仅根据mask存在就认定零开销。

## 14. 最小代码验证

今天的脚本只使用Python标准库，不安装新依赖：

```bash
python3 exercises/day02/text_to_logits.py
```

它验证：

- Unicode文本到UTF-8 bytes；
- 人工词表编码、BOS/EOS和padding；
- `input_ids` 与 `attention_mask` 的shape；
- Embedding lookup后的shape；
- 绝对位置向量相加不改变shape；
- 经过多个占位Block后仍保持 `[B,S,H]`；
- LM Head产生 `[B,S,V]`；
- 最后有效位置如何由attention mask确定。

脚本中的Block和LM Head只做shape与少量确定性数值验证，不是真实Transformer实现，也不能用于性能测试。Day 3开始逐步替换为真实算子。

## 15. 常见误解与边界

- token不等于英文单词，也不等于中文字符。
- Unicode码点不是UTF-8 byte；UTF-8的一个字符使用1～4 bytes。
- Byte-level能编码陌生字符，不代表模型理解陌生字符。
- token ID只是词表索引，不具有自然数大小关系。
- Embedding是可训练权重；查表结果是随输入变化的激活。
- padding长度、训练block size、模型最大上下文长度不是同一个概念。
- padding mask和causal mask解决不同问题。
- 每个batch样本是独立上下文，Attention不会跨batch连接。
- 位置Embedding和RoPE都提供位置信息，但作用位置不同。
- Encoder输出 `[B,S,H]` 不代表它一定执行下一个token生成。
- Encoder–Decoder中的Decoder通常有cross-attention；Decoder-only Block通常没有Encoder cross-attention。
- logits是softmax之前的分数，不是概率。
- 逻辑上的 `[B,S,V]` 不保证实现一定长期物化或保存整个张量。

## 16. 手算练习

### 练习1：Tokenizer与特殊Token

使用极小词表：

```text
0=<PAD>, 1=<BOS>, 2=<EOS>, 3=我, 4=喜欢, 5=学习, 6=AI, 7=你
```

回答：

1. `我喜欢AI` 加入BOS/EOS后的token IDs是什么？
2. `你` 加入BOS/EOS后的token IDs是什么？
3. 两条序列padding到长度5后的 `input_ids` 和 `attention_mask` 是什么？
4. 为什么不能从ID大小判断token语义？

### 练习2：Embedding与位置

给定：

```text
B=2, S=5, V=8, H=4
```

回答：

1. `input_ids`、Embedding table和查表结果的shape。
2. 绝对位置Embedding table若支持最长16个位置，其shape是什么？
3. 取前5个位置并与token embedding相加后shape是什么？
4. 为什么相同token出现在位置1和位置4时，输入表示可以不同？

### 练习3：三类架构

分别回答：

1. Encoder-only中一个token通常能否查看它右侧的token？
2. Encoder–Decoder的cross-attention中，Q来自哪里，K/V来自哪里？
3. Decoder-only为什么需要causal mask？
4. Encoder–Decoder中的Decoder和Decoder-only Block的主要结构差异是什么？

### 练习4：从文本到Logits

正式配置：

```text
B=2, S=8, V=8000, H=384, L=6
```

写出以下shape：

1. `input_ids` 与 `attention_mask`。
2. Embedding lookup结果。
3. 经过6层Block后的hidden states。
4. LM Head权重和logits。
5. 取每条序列最后有效位置的next-token logits后的shape。

### 练习5：参数与内存

1. `W_embed[8000,384]` 有多少参数？FP16权重本体多大？
2. `logits[2,8,8000]` 有多少元素？FP32理论大小是多少？
3. 若词表从8000扩大到16000，Embedding参数量和logits元素数如何变化？
4. Weight tying减少什么，不减少什么？

## 17. 面试口述

### 17.1 30秒目标

能够说明：Tokenizer把Unicode文本切成token并映射为ID；Embedding把ID查表变成连续hidden vectors；位置信息让模型区分顺序；Decoder-only Transformer在causal约束下更新每个位置，最后LM Head将 `[B,S,H]` 投影为 `[B,S,V]` logits。

### 17.2 两分钟目标

回答中需要包含：

1. 文件解码、Unicode、UTF-8 bytes和token之间的边界；
2. tokenizer训练与语言模型训练不是同一件事；
3. token IDs、Embedding权重和hidden activations的区别；
4. padding mask与causal mask的区别；
5. Encoder、Encoder–Decoder和Decoder-only的数据流；
6. 词表大小与序列长度对AI Infra成本的影响。

### 17.3 三道口述题

1. 为什么Byte-level tokenizer原则上能避免未知字符，但仍不能保证模型理解它？
2. Encoder–Decoder与Decoder-only在Attention数据来源上有什么区别？
3. 从一条文本到next-token logits，完整的数据流是什么？

## 18. 当日验收

请独立提交：

1. 不看讲义，画出“UTF-8文本 → token IDs → `[B,S,H]` → `[B,S,V]`”完整数据流。
2. 完成练习1、练习3和练习4。
3. 用自然语言解释token ID、Embedding权重、hidden activation三者的区别。
4. 解释为什么两条不同长度序列可以组成一个batch，以及mask分别防止什么。
5. 说明三类Transformer架构的输入、Attention可见范围和典型输出。
6. 运行最小脚本，解释每一个shape为什么合理。
7. 写出今天最不确定的一个概念。

只有同时满足以下条件，Day 2才算通过：

- 不把token等同于单词或汉字；
- 不把token ID当作语义数值；
- 能区分权重与激活；
- 能区分padding mask与causal mask；
- 能从文本一路写到logits；
- 能说出至少一个tokenizer或padding带来的Infra成本。
