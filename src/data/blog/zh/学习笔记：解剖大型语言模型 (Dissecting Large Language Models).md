---
author: 芙芙
pubDatetime: 2026-08-16T00:00:00+08:00
title: 学习笔记：解剖大型语言模型 (Dissecting Large Language Models)
featured: false
draft: false
tags:
  - LLM
  - Transformer
  - Self-Attention
  - 可解释性
  - 学习笔记
description: 整理自生成式 AI 公开课的学习笔记：从 Token ID 到 Logits 的宏观计算流讲起，拆解隐藏表征分析与 Logit Lens、Self-Attention 机制、FFN 记忆网络与位置编码，并用 Colab 实作解剖 LLaMA-3.2 与 Gemma-2。
---

- **课程来源**：[【生成式人工智慧與機器學習導論 2025】第 3 講：解剖大型語言模型](https://www.google.com/search?q=https://youtu.be/8iFvM7WUUs8)
- **课程主页**：[GenAI-ML 2025 课程主页](https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall.php)
- **配套代码**：[Google Colab 实作代码](https://colab.research.google.com/drive/1uU9aW020lhaqk236E_my4ObiCzzc0eKn?usp=sharing)

## 核心导览

1. **宏观计算流：从 Token ID 到 Logits & Softmax 概率分布**
2. **隐藏表征分析与可解释性：Representation Engineering & Logit Lens**
3. **微观层级解密：Self-Attention 机制与多头注意力（Multi-Head Attention）**
4. **前馈全连接网络（FFN）：Key-Value 记忆网络与神经元本质**
5. **Colab 实作解剖：LLaMA-3.2 与 Gemma-2 的层级与注意力观察**

## 1. 宏观计算流：从 Token ID 到 Logits

语言模型本质上是一个高维复杂函数 $y = f(x)$，将输入序列映射为词表上的概率分布。整体前向推理流程分为以下四个阶段：

```
Input Tokens (IDs) ──► Embedding Table ──► [Layer 1] ──► ... ──► [Layer L] ──► LM Head (Unembedding) ──► Logits ──► Softmax (Temperature) ──► Next Token Probabilities
```

### 1.1 Token Embedding 查表

- 输入序列被分词为 Token ID 向量 $[t_1, t_2, \dots, t_N]$。
- 通过一个可学习的权重矩阵 $\mathbf{W}_{emb} \in \mathbb{R}^{V \times D}$（$V$ 为词表大小，$D$ 为隐藏层维度），将每个离散整数 ID 转换为连续密集的初始向量表示 $\mathbf{e}_i$。

### 1.2 层级表征演化（Contextualized Embedding）

- 向量序列逐层流经 $L$ 个 Transformer Block：

  $$\mathbf{h}_i^{(l)} = \text{Layer}_l(\mathbf{h}_i^{(l-1)})$$

- 从第 0 层的静态词嵌入（Static Token Embedding），经过多层注意力融合上下文信息后，演化为富含语境语义的**上下文表征（Contextualized / Hidden Representation）**。

### 1.3 LM Head（Unembedding）与首尾呼应设计

- 取最后一层序列末位的表征向量 $\mathbf{h}_N^{(L)} \in \mathbb{R}^{D}$。

- 乘上输出投影矩阵 $\mathbf{W}_{head} \in \mathbb{R}^{V \times D}$ 计算未归一化得分（Logits $\mathbf{z} \in \mathbb{R}^{V}$）：

  $$\mathbf{z} = \mathbf{W}_{head} \mathbf{h}_N^{(L)}$$

- **权重绑定（Weight Tying / 首尾呼应）**：在现代主流模型（如 LLaMA、Gemma）中，$\mathbf{W}_{head}$ 往往与输入的 $\mathbf{W}_{emb}$ 共享参数或转置复用。此时 Logit 的计算本质上就是**最终隐藏状态与词表中每个 Token Embedding 计算点积相似度（Dot Product）**，相似度越高的 Token 越容易被接在后面！

### 1.4 Softmax 与温度调节（Temperature）

$$\mathbf{P}(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}$$

- **Temperature $T$ 调控**：$T > 1$ 使概率分布平缓，增加冷门 Token 的采样几率（“创意模式”）；$T < 1$ 使得极高概率的 Token 更加突出，输出更确定保守。

## 2. 隐藏表征分析与可解释性技术

如何探知高维连续向量中到底蕴含了什么信息？课程介绍了数种经典的内部表征观测与干预方法：

### 2.1 表征工程（Representation / Activation Engineering）

- **语义方向与激活操纵**：高维表征空间中存在具备特定语义的概念方向（如“拒绝”、“奉承”、“中英翻译”等）。

- **提取拒绝向量（Refusal Vector）**：

  $$\mathbf{v}_{refusal} = \frac{1}{\vert{}D_{harm}\vert{}}\sum_{x \in D_{harm}} \mathbf{h}^{(l)}(x) - \frac{1}{\vert{}D_{safe}\vert{}}\sum_{x \in D_{safe}} \mathbf{h}^{(l)}(x)$$

- **激活转向（Steering）**：在推理时强行注入该向量 $\mathbf{h}^{(l)} \leftarrow \mathbf{h}^{(l)} + \alpha \mathbf{v}_{refusal}$ 可强制模型拒绝安全请求；减去该向量 $\mathbf{h}^{(l)} \leftarrow \mathbf{h}^{(l)} - \alpha \mathbf{v}_{refusal}$ 则可实施“越狱”使模型回答违规指令。

### 2.2 逻辑透镜（Logit Lens）：偷窥模型的中间心智

- 将中间第 $l$ 层的隐藏状态 $\mathbf{h}^{(l)}$ 强行直接输入最顶层的 $\mathbf{W}_{head}$ 做 Unembedding，观察模型在每一层预备输出的候选词。
- **现象**：当要求多语言翻译（如法译中）时，模型在前中部层往往先在内心用英语概念（如 `flower` $\rightarrow$ `weather forecast`）进行语义对齐，在最后几层才转换为目标语言（中文字符“花”、“预报”）。

### 2.3 Patchscopes（特征补丁探索）

- 将待测句子在某层的特定特征向量提取出来，覆盖/替换到探针 Prompt（如 _“请简单介绍 X”_）的对应位置中，由模型后续自回归解码出人类可读的完整自然语言描述。

## 3. 微观层级解密：Self-Attention 机制

每个 Transformer Layer 由 **Causal Self-Attention** 与 **Feed-Forward Network (FFN)** 两大核心子模块构成：

### 3.1 自注意力计算三部曲

对第 $i$ 个位置的表征向量 $\mathbf{x}_i$：

1. **线性映射生成 Query、Key、Value**：

   $$\mathbf{q}_i = \mathbf{W}_Q \mathbf{x}_i, \quad \mathbf{k}_i = \mathbf{W}_K \mathbf{x}_i, \quad \mathbf{v}_i = \mathbf{W}_V \mathbf{x}_i$$

2. **计算注意力权重（因果掩码 Causal Masking）**：

   $$\alpha_{i, j} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} \quad (j \le i), \quad \mathbf{a}_i = \text{Softmax}(\boldsymbol{\alpha}_{i, 1:i})$$

3. **加权求和与残差连接（Residual Connection）**：

   $$\mathbf{o}_i = \sum_{j=1}^i a_{i, j} \mathbf{v}_j, \quad \mathbf{x}_i^{(mid)} = \mathbf{x}_i + \mathbf{W}_O \mathbf{o}_i$$

### 3.2 多头注意力（Multi-Head Attention）与分组查询注意力（GQA）

- **Multi-Head**：不同注意力头关注不同维度的语法语义（如一个头关注修饰形容词，另一个头关注代词指代或数量修饰）。
- **GQA / MQA**：为节省 KV 缓存显存开销，现代开源大模型（如 LLaMA-3、Gemma）常令多个 Query 头共享少量的 Key/Value 头。

### 3.3 位置编码（Positional Encoding & RoPE）

为了赋予无序注意力机制对序列先后次序与距离衰减的感知能力，现代模型广泛采用旋转位置编码（Rotary Position Embedding, RoPE），将相对位置信息直接编码入 Query 与 Key 的复数旋转变换中。

## 4. 前馈全连接网络（FFN）：Key-Value 记忆网络

过完 Self-Attention 后，表征向量会独立经过两层全连接升维与降维：

$$\mathbf{y} = \text{FFN}(\mathbf{x}) = \mathbf{W}_{down} \cdot \sigma(\mathbf{W}_{up} \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

- **神经元与矩阵本质**：所谓神经元（Neuron）即矩阵单行与输入向量的内积加偏置后过非线性激活函数（如 SwiGLU / GeLU / ReLU）。
- **Key-Value 存储器视角**：前馈网络的第一层权重可视为模式检测键（Keys），第二层权重可视为事实知识值（Values）。FFN 承担了模型中海量事实知识记忆库的角色。

## 5. Colab 实作解剖：参数量、层级与注意力热力图

配合 Google Colab 实作代码，课程详细探查了真实开源模型：

### 5.1 模型参数结构对比

- **LLaMA-3.2-3B**：包含 28 个 Transformer 层，隐藏层维度为 3072，FFN 升维至 8192，词表容量达 128,256。
- **Gemma-2-4B**：包含 34 个 Transformer 层，词表容量高达 262,208，隐藏维度为 2560。

### 5.2 上下文表征的语义距离实验

- 在第 0 层（静态词表），所有句子中的 `Apple` 向量完全相同（余弦相似度为 1）。
- 随着层数加深，指代“水果”的 `Apple` 与指代“科技公司”的 `Apple` 之间的余弦相似度迅速下降，模型通过注意力机制逐层将多义词解析至不同的语义流形区域。

### 5.3 注意力汇聚点（Attention Sink）现象

可视化生成的因果注意力热力图表明：

- 大量注意力头呈现下三角结构（由于因果自回归掩码无法查看未来 Token）。
- 许多注意力头在缺乏明确指代目标时，会将注意力权重默认集中分配给**句首的起始 Token（BOS / Start Token）**，形成“注意力汇聚点（Attention Sink）”。
