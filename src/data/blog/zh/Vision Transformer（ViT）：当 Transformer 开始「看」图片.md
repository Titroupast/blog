---
author: 芙芙
pubDatetime: 2026-08-31
title: Vision Transformer（ViT）：当 Transformer 开始「看」图片
featured: false
draft: false
tags:
  - Vision Transformer
  - ViT
  - Transformer
  - 注意力机制
  - 学习笔记
category: 深度学习
description: 整理自 B 站《15分钟认识ViT》等视频与知乎资料：从「图片能不能变成 Token」出发，串讲 Patch、Patch Embedding、CLS、位置编码与 Transformer Encoder 的完整流程，对比 CNN 的归纳偏置，并给出 DeiT、Swin 等改进路线。
---

> 本文主要整理自 B 站《15分钟认识ViT！【视觉Transformer】》等视频资料和知乎文章，串讲 Vision Transformer（ViT）的完整流程：图片如何切成 Patch 变成 Token、CLS 与位置编码的作用、Transformer Encoder 如何处理图像，以及 ViT 与 CNN 在归纳偏置上的根本区别。
>
> 一句话概括 ViT 的核心思想：

```text
Image → Patches → Tokens → Transformer
```

## 一、从 CNN 到 ViT：Transformer 开始「看」图片

### 1.1 图片也能变成 Token 吗

Transformer 最早是为自然语言处理设计的：一句话被拆成一系列 Token（比如 I love machine learning 变成 $[x_1,x_2,x_3,x_4]$），交给 Transformer 后用 Self-Attention 建模不同 Token 之间的关系。于是一个非常自然的问题出现了：**既然 Transformer 可以处理 Token 序列，那图片能不能也变成 Token 序列？**

答案就是 **Vision Transformer（ViT）**。2020 年 Google Research 在论文 *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*（ICLR 2021）中提出 ViT，做的事情异常简单：**把图片切成一小块一小块的 Patch，把每个 Patch 当成 NLP 中的一个 Token，然后直接送进 Transformer**。真正令人意外的地方在于：这么简单的方法，在足够大的数据集上，居然能够超过当时非常成熟的 CNN。

### 1.2 CNN 的归纳偏置

在 ViT 出现以前，计算机视觉基本是 CNN 的天下（AlexNet、VGG、ResNet、DenseNet、EfficientNet），视觉网络的基本结构一直离不开卷积。CNN 有一个非常符合图像特点的假设：**距离比较近的像素之间通常关系更密切**，因此卷积核只观察图像中的一个局部区域。例如一个 $3\times3$ 卷积只处理当前位置附近的像素；随着 CNN 一层一层堆叠（$3\times3\rightarrow5\times5\rightarrow7\times7\rightarrow\cdots$），感受野逐渐扩大，特征从 局部 → 局部组合 → 高级语义 → 全局信息 逐级抽象。比如识别一只猫时，CNN 可能首先学到边缘、纹理、猫耳朵、猫脸，最后才组合出「猫」。

这种设计非常适合图像，但也意味着 CNN 自带很强的 **Inductive Bias（归纳偏置）**：Locality（局部性）、Translation Equivariance（平移等变性）、Parameter Sharing（参数共享）——这些假设相当于提前告诉模型「图片应该按照这种方式理解」。而 Transformer 的思路完全不一样：它基本没有这种视觉先验，更接近「我先不假设谁和谁关系比较近，让模型自己学习所有 Token 之间的关系」。这也是 ViT 最核心的特点之一。

## 二、Patch：图片切成 16×16 的 Word

### 2.1 一个像素一个 Token 不现实

Transformer 接收的是 Token 序列，但图片是 $H\times W\times C$（比如 $224\times224\times3$）。一种最简单的想法是一个像素就是一个 Token，那么一张 $224\times224$ 的图片就有 $224\times224=50176$ 个 Token。问题马上出现了：Self-Attention 的复杂度与序列长度 $N$ 的平方有关，即 $O(N^2)$；当 $N=50176$ 时，Attention Matrix 就是 $50176\times50176$，大约包含 $2.5\times10^9$ 个元素，显然非常离谱。

于是 ViT 使用了一个非常简单但非常关键的办法：**不要把 Pixel 当 Token，而是把一小块图片当成一个 Token**——这就是 Patch。

### 2.2 Patch：196 个 Token

假设输入是 $224\times224\times3$，选择 Patch Size 为 $16\times16$，那么图片会被切成：

$$
\frac{224}{16}\times\frac{224}{16}=14\times14=196
$$

个 Patch。于是 $50176$ 个 Pixel Token 变成 $196$ 个 Patch Token，序列长度一下缩短了 256 倍。这也是论文标题「An Image is Worth 16×16 Words」的由来：16×16 个像素可以看成视觉世界中的一个 Word。当然，这个标题更多是形象表达，并不是说 Patch 真正等价于自然语言中的单词。

### 2.3 Patch Embedding：线性投影与卷积实现

一个 $16\times16\times3$ 的 Patch 本质上包含 $16\times16\times3=768$ 个数，先 Flatten 成 $x_p^i\in\mathbb R^{768}$。但 Transformer 通常希望所有 Token 都处于统一的 Embedding Space，因此还要经过一个 Linear Projection：$x_p^iE$，其中 $E\in\mathbb R^{(P^2C)\times D}$，把 Patch 映射成 $D$ 维向量。整个过程就是 Patch Embedding：

$$
Image\rightarrow Patches\rightarrow Flatten\rightarrow Linear\ Projection\rightarrow Patch\ Tokens
$$

原论文正式写为：

$$
\mathbf z_0=[\mathbf x_{class};\mathbf x_p^1E;\mathbf x_p^2E;\cdots;\mathbf x_p^NE]+E_{pos}
$$

其中 $x_p^i$ 是第 $i$ 个 Patch、$E$ 是 Patch Projection、$D$ 是 Embedding Dimension、$E_{pos}$ 是位置编码。

这里有一个很有意思的地方：Patch Embedding 其实可以直接用卷积实现。假设 Patch Size 为 16，用 Conv2d(kernel_size=16, stride=16, out_channels=768)，输入 $224\times224\times3$ 经过卷积后得到 $14\times14\times768$，Flatten 空间维度就是 $196\times768$——正好是 196 个 Token、每个 768 维。所以很多 PyTorch 实现里的 ViT Patch Embedding 本质就是一个大步长卷积。但这不意味着 ViT 又变成 CNN 了：这里的卷积主要承担 Patch 划分和线性投影功能，而不是像传统 CNN 那样逐层进行局部卷积特征提取。

## 三、让 Transformer 看懂图片：CLS、位置编码与 Encoder

### 3.1 CLS Token：汇总全图信息

现在得到了 196 个 Patch Token，但任务是图片分类（Image → Class），应该用哪个 Token 做分类？ViT 借鉴了 BERT 的设计，在 Patch Token 前加入一个特殊 Token CLS，于是 196 个 Token 变成 197 个：$[CLS],Patch_1,\cdots,Patch_{196}$。经过 Transformer 之后，CLS Token 会通过 Attention 和其他 Patch 交换信息，最终取 $z_L^0$ 作为整张图片的表示，再接 MLP/Linear 得到 Class。

可以把 CLS 粗略理解成「负责汇总全图信息的特殊 Token」。不过这只是帮助理解的说法：CLS 并不是主动把所有 Patch 做一次简单平均，而是经过多层 Self-Attention 不断与其他 Token 交互，最终学出一个适合分类的全局表示。

### 3.2 位置编码：Transformer 不知道 Patch 在哪里

假设有两个 Patch：猫的眼睛和猫的耳朵，Transformer 本身看到的只是 $Token_A$ 和 $Token_B$，并不知道 A 在左上角、B 在右上角；甚至对 Transformer 来说 $A,B,C,D$ 和 $D,C,B,A$ 并没有天然的空间含义。但图像显然非常依赖位置，所以 ViT 给每一个 Patch 加上 Position Embedding：

$$
Token_i=PatchEmbedding_i+PositionEmbedding_i
$$

值得注意的是，原始 ViT 甚至没有使用复杂的二维位置编码，而是用了 **Learnable 1D Position Embedding**：图片原本是 $14\times14$ 的二维网格，但 ViT 把它 Flatten 成 196 之后，直接学习 $197\times D$ 的位置 Embedding。论文实验发现专门设计二维位置编码并没有带来明显优势——即使只告诉 ViT Patch 的序号，模型自己仍然能逐渐学出图像中的二维空间结构。论文对位置 Embedding 可视化后发现：距离近的 Patch 位置编码更相似，同一行、同一列的 Patch 会表现出相关性。换句话说，**ViT 自己把二维结构学出来了**。

### 3.3 图像变成 Token 序列

把前面的步骤连起来：图像先切 Patch，然后 Patch Embedding，加入 CLS，加入 Position Embedding，得到 $z_0=[x_{cls};x_1;\cdots;x_N]+E_{pos}$，图片就变成了一个普通的 Token Sequence，接下来交给 Transformer Encoder 处理。这正是 ViT 最「暴力」的地方：它没有设计一个复杂的视觉 Transformer，而是**想办法把 Image 变成 Sequence，然后尽可能原封不动地使用 Transformer**。

### 3.4 Transformer Encoder：Pre-Norm、MSA 与 MLP

ViT 使用的是 Transformer Encoder，一个 Block 可以写成：

$$
z_l'=MSA(LN(z_{l-1}))+z_{l-1},\qquad z_l=MLP(LN(z_l'))+z_l'
$$

即 LayerNorm → Multi-Head Self-Attention → Residual → LayerNorm → MLP → Residual。ViT 使用的是 Pre-Norm（LayerNorm 在 Attention 之前），MLP 有两层全连接，中间用 GELU 激活。

其中 Self-Attention 是真正重要的部分：每个 Token 都生成 $Q=XW_Q$、$K=XW_K$、$V=XW_V$，然后：

$$
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$QK^T$ 实际计算的是**一个 Patch 和所有其他 Patch 到底有多相关**。比如「猫的左眼」Patch 可以对右眼、耳朵、身体、背景沙发分别计算 Attention 权重，最后从与它关系最大的 Patch 中聚合信息。这带来一个与 CNN 最大的区别：CNN 的局部性是人为规定的，一个像素想知道很远的位置发生了什么，需要经过很多层不断扩感受野；而 ViT 的 Self-Attention 从第一层开始，**理论上每一个 Patch 都能直接观察所有 Patch**——左上角的猫脑袋可以直接和右下角的猫尾巴产生 Attention。原论文对不同层的 Attention Distance 做了可视化，确实发现有些 Attention Head 从非常浅的层就已经在聚合跨越整幅图像的信息。

Multi-Head Attention 则把特征拆成多个子空间并行计算：$head_i=Attention(Q_i,K_i,V_i)$，$MSA(X)=Concat(head_1,\cdots,head_h)W_O$。不同 Head 可以在不同 Representation Subspace 中建立关系（比如某个 Head 关注形状、另一个关注颜色、另一个关注前景与背景），让模型有能力同时建模多种关系。Attention 解决的是 Token 之间怎么交换信息，而 MLP 更偏向每一个 Token 自己的特征怎么变换——所以 Transformer Block 是 Attention + MLP 不断交替，缺一不可。
## 四、ViT 的配置、数据与瓶颈

### 4.1 ViT-B/16 是什么意思

ViT 有一种类似 BERT 的命名方式：ViT-Base、ViT-Large、ViT-Huge，原论文的配置为：

| Model | Layers | Hidden Size | MLP Size | Heads | Params |
|---|---:|---:|---:|---:|---:|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

而 ViT-B/16 表示：B = Base，16 = Patch Size 16×16。所以 ViT-B/16 的配置是 Patch Size 16、Layers 12、Hidden Size 768、Heads 12、参数量约 86M；输入 $224\times224$ 时 Patch 数量为 $14\times14=196$，加 CLS 共 197 个 Token。整个流程就是：224×224×3 图像 → 16×16 Patch → 196 Patches → Patch Embedding → 196×768 → 加 CLS → 197×768 → 加位置编码 → 12 层 Transformer Encoder → 取 CLS → Linear → 分类。

### 4.2 Patch Size 的权衡

Patch Size 直接影响 Token 数量：$224\times224$ 输入下，Patch Size 32 时 Token 数 $7\times7=49$，Patch Size 16 时 $14\times14=196$，Patch Size 8 时 $28\times28=784$。Patch 越小，Token 数量越多、图像保留的细节越多，但 Attention 是 $O(N^2)$，计算量会迅速增加（$49^2=2401$、$196^2=38416$、$784^2=614656$）。所以**更小的 Patch 通常意味着更细粒度的视觉表示，但也意味着昂贵得多的 Attention**——这也是视觉 Transformer 后续很多研究一直试图解决的问题。

### 4.3 数据饥渴：ViT 最大的问题

看到这里可能觉得：Transformer 能全局建模，不是全面碾压 CNN？并不是。原始 ViT 有一个非常大的问题：**Data Hungry（非常吃数据）**。原因是 CNN 有很强的先验（比如「附近的像素关系比较重要」），无论给 CNN 什么图片这个假设一直存在；而 ViT 基本是在说「你别告诉我什么是局部、什么是二维空间，我自己学」，自由度更高的代价就是模型需要大量数据才能自己学习这些规律。

原论文的实验非常典型：在 ImageNet 规模的数据上训练时，ViT 并没有天然碾压 CNN，甚至在 ImageNet-1K 规模时明显落后于强 CNN；到了 ImageNet-21K（约 1400 万张图片）两者逐渐接近；到了 JFT-300M 这种数亿图像规模，ViT 的优势才真正体现出来。所以原论文得到一个非常重要的结论：**大规模数据可以弥补较弱的视觉归纳偏置**。更直白地说：CNN 是「老师提前告诉你很多解题技巧」，而 ViT 是「老师什么也不告诉你，但是给你刷几个亿的题」——题刷得足够多之后，ViT 可以自己把这些规律学出来。

### 4.4 大数据下的优势与 Scaling

归纳偏置并不是绝对的好事情。强归纳偏置意味着提前限制了模型应该如何理解数据：数据少的时候非常有帮助，因为搜索空间小很多；但如果数据特别多，强先验反而可能限制模型。Transformer 的假设更弱、模型自由度更高，随着数据量不断增长（$Data\uparrow$），它能够学习出更加复杂的规律，这也是后来 Transformer Scaling 能够不断发展的一个重要原因。

Google 后续甚至将 ViT 扩展到了 22B 参数，并观察到随着模型规模扩大，Frozen Representation、OOD Performance 等指标仍然能够继续改善。这一点和今天 LLM 的 Scaling 思想已经非常接近：

$$
Data\uparrow + Model\uparrow + Compute\uparrow \Rightarrow Performance\uparrow
$$

### 4.5 高分辨率瓶颈：O(N²) 的代价

Self-Attention 的经典问题是 $O(N^2)$，而 $N=HW/P^2$。假设图片尺寸扩大一倍（$224\times224\rightarrow448\times448$），Patch Size 不变时 Token 数量从 196 变成 784（4 倍），Attention Matrix 从 $196^2$ 变成 $784^2$，计算量变成大约 16 倍。因此原始 ViT 的 **Global Attention 在高分辨率视觉任务中会非常昂贵**——Google Research 后续也将这一点列为视觉 Transformer 的核心计算挑战，这也是为什么 Swin Transformer 后来会非常重要。

### 4.6 没有天然的多尺度结构

传统 CNN 通常有天然的特征金字塔：224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7，随着网络加深空间尺寸下降、Channel 上升，不同层可以学习不同尺度的物体，这对 Object Detection、Semantic Segmentation、Instance Segmentation 非常重要。但原始 ViT 从头到尾基本保持 196 个 Token，没有 CNN 那种天然的 Feature Pyramid，因此它特别适合 Image Classification，但作为通用视觉 Backbone 时就存在一定问题——这也直接推动了 Swin Transformer 等层级式视觉 Transformer 的发展。

## 五、两个重要改进：DeiT 与 Swin

### 5.1 DeiT：没有几亿图片怎么办

ViT 最大的问题之一是需要巨量预训练数据。Facebook AI 后来提出 **DeiT（Data-efficient Image Transformer）**，论文标题是 *Training data-efficient image transformers & distillation through attention*。DeiT 证明：通过更合理的训练策略和知识蒸馏，即使只使用 ImageNet-1K，也可以把 Vision Transformer 训练得很好，而不一定需要 JFT-300M 这种普通研究者根本拿不到的数据。

DeiT 还设计了一个非常有意思的 **Distillation Token**：除了 CLS，再加入一个 DIST Token（序列变成 $[CLS],[DIST],Patch_1,Patch_2,\cdots$），其中 CLS 学习真实标签，DIST 学习 Teacher 模型给出的知识，利用 Attention 让蒸馏信息也作为一个 Token 在 Transformer 内部传播。DeiT 的实验说明，一个没有卷积的 Transformer，在 ImageNet 上通过良好的训练和蒸馏，同样可以获得非常有竞争力的结果。可以理解为：ViT 回答「Transformer 能不能看图片？——可以，但是很吃数据」；DeiT 回答「那没有几亿张图片怎么办？——改训练策略 + Knowledge Distillation」。

### 5.2 Swin Transformer：Global Attention 太贵怎么办

接下来另一个大问题：Global Attention 是 $O(N^2)$，而且原始 ViT 缺少 CNN 那样的 Hierarchical Representation。于是微软提出 **Swin Transformer**（*Hierarchical Vision Transformer using Shifted Windows*），核心思路可以概括成两个词：**Window** 和 **Shift**。

首先不要让每个 Patch 都 Attention 整张图片，而是把图像划分成 Window，只在 Window 内做 Self-Attention，计算量大幅降低。但这样又会出现 Window 之间无法交流的问题，于是下一层把 Window 平移（Shifted Window），原本属于不同 Window 的 Patch 就会进入同一个新 Window，逐渐实现跨区域信息交换。

更重要的是，Swin Transformer 又重新引入了类似 CNN 的 **Hierarchical Architecture**，可以生成 $\frac14\rightarrow\frac18\rightarrow\frac1{16}\rightarrow\frac1{32}$ 不同尺度的 Feature Map，使它能够非常自然地用于 Classification、Detection、Segmentation 等任务。Swin 的作者指出，这种设计让 Attention 对图像大小具有近似线性的计算复杂度，同时获得多尺度表示，因此更适合作为通用视觉 Backbone。

## 六、ViT 与 CNN 的关系

### 6.1 一张表看懂区别

| CNN | ViT |
|---|---|
| Pixel / Feature Map | Patch Token |
| Convolution | Self-Attention |
| 强 Locality | Global Interaction |
| 强视觉归纳偏置 | 较弱视觉归纳偏置 |
| 数据效率较高 | 原始 ViT 非常吃数据 |
| 天然 Hierarchical | 原始 ViT 层级结构较弱 |
| 感受野逐层扩大 | 第一层即可全局 Attention |
| $O(HW)$ 类型局部操作 | Global Attention 存在 $O(N^2)$ |

但需要注意：**现在的 Vision Transformer 和 CNN 已经很难简单二分**。很多现代视觉 Transformer 又重新加入了 Local Window、Hierarchy、Convolution Stem、Multi-scale Feature、Locality Bias；与此同时，现代 CNN 也大量吸收 Attention、Transformer Training Recipe、LayerNorm、大规模预训练。所以现在的发展趋势已经不是 CNN VS Transformer，而更像是：**哪些视觉归纳偏置应该保留，哪些应该交给模型自己学习？**

### 6.2 ViT 真的是「没有卷积」吗

从 Backbone 的核心结构来说，原始 ViT 可以完全不依赖传统卷积网络进行特征提取。但 Patch Embedding 本质上又可以写成 Conv2D(kernel=P, stride=P)，所以代码里经常能看到：

```python
self.proj = nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size
)
```

这时候不能简单地看到 Conv2d 就说「ViT 其实也是 CNN」：这里 Conv2d 实现的数学操作等价于 Patch → Flatten → Linear Projection，真正决定 ViT 特性的仍然是后面的 Transformer Encoder。

### 6.3 ViT 的真正意义：统一 Token

如果单独看 ViT 的每一个组成部分：Transformer 不是 ViT 发明的（来自 2017 年的 *Attention Is All You Need*），CLS Token 是 BERT 已有的思想，Position Embedding Transformer 本身就有，Image Patch 也不是极其复杂的新数学结构。ViT 真正重要的贡献反而非常「朴素」：**它证明了一个高度通用、几乎没有视觉特化设计的 Transformer，在足够数据和规模下，可以直接成为非常强大的视觉模型**。

这件事情的重要性远远超过某一个新模块，因为它改变了一个重要的思想：过去是 NLP 用 Transformer、CV 用 CNN、Speech 用专门模型；而 Transformer 展现出的趋势逐渐变成 Text、Image、Audio、Video 都先变成 Token，然后使用统一架构建模。今天很多多模态大模型（CLIP、LLaVA、Flamingo、Gemini 等）背后都能看到这种统一 Token Representation 思想的延续。
## 七、回顾与总结

### 7.1 再来看论文标题：An Image is Worth 16×16 Words

现在再来看论文标题就好理解了。对于 NLP，流程是 Sentence → Word/Token → Embedding → Transformer；ViT 做的事情是 Image → Patch → Embedding → Transformer。一张 $224\times224\times3$ 的图片切成 $16\times16$ Patch，得到 $14\times14=196$ 个 Patch，每个 Patch 投影成 768 维（$196\times768$），加入 CLS 变成 $197\times768$，加入位置编码后经过 12 层 Transformer Encoder，最后取 CLS 接 Linear 输出 Image Class。如果只记 ViT 一个公式，甚至没必要记复杂公式，记住：

```text
Image → Patches → Tokens → Transformer
```

就已经理解 ViT 最核心的思想了。

### 7.2 总结：六步流程与八个特点

ViT 的整体流程其实非常简单：

1. **Patchify**：把 $H\times W\times C$ 的图片切成 $N=HW/P^2$ 个 Patch；
2. **Patch Embedding**：每个 $P\times P\times C$ 的 Patch Flatten 后经过 Linear Projection（$P^2C\rightarrow D$）得到 Patch Token；
3. **加入 CLS**：得到 $N+1$ 个 Token；
4. **加入 Position Embedding**：让 Transformer 获得 Patch 的空间位置信息；
5. **Transformer Encoder**：不断执行 MSA + MLP + Residual + LayerNorm，让不同 Patch 之间交换信息；
6. **分类**：取最后的 CLS Token $z_L^0$，经过 Linear Head 得到 Class。

ViT 最值得记住的几个特点：

1. 将图像 Patch 化，把视觉问题变成序列建模问题；
2. 使用几乎标准的 Transformer Encoder 处理图像；
3. Self-Attention 可以从浅层直接建立全局 Patch 关系；
4. 相比 CNN，ViT 的视觉归纳偏置更弱；
5. 弱归纳偏置导致原始 ViT 更吃数据，但大规模数据下扩展能力很强；
6. Global Attention 的 $O(N^2)$ 是高分辨率视觉的重要瓶颈；
7. DeiT 主要解决 ViT 的数据效率问题；
8. Swin Transformer 通过 Window + Shift + Hierarchy 改善计算效率和多尺度建模。

从历史角度来看，ViT 的意义可能并不只是「Transformer 也能做图像分类」，更重要的是它进一步证明了：**模型未必需要为每种数据模态设计完全不同的架构，只要能够把输入转换为合适的 Token，Transformer 就可能成为一种跨模态的通用计算框架**。这也是为什么在 ViT 之后，Transformer 很快从 NLP 扩展到了图像、视频、语音以及多模态领域。

### 7.3 进一步阅读：现代视觉 Transformer 学习路线

如果理解了 ViT，接下来的学习路线非常自然：

```text
ViT → DeiT → Swin Transformer → MAE → DINO/DINOv2 → CLIP
```

每个模型解决的核心问题：

- **ViT**（*An Image is Worth 16×16 Words*，ICLR 2021）：解决「Transformer 怎么看图片」——把图像切成 Patch 当 Token，直接用标准 Transformer 编码。[arXiv](https://arxiv.org/abs/2010.11929)
- **DeiT**（*Training data-efficient image transformers & distillation through attention*，ICML 2021）：解决「ViT 没有几亿训练数据怎么办」——通过知识蒸馏与更合理的训练策略，只用 ImageNet-1K 就能训好 ViT。[arXiv](https://arxiv.org/abs/2012.12877)
- **Swin Transformer**（*Hierarchical Vision Transformer using Shifted Windows*，ICCV 2021）：解决「Global Attention 太贵、且 ViT 缺多尺度」——Window Attention + Shifted Window 降低计算量，并重新引入层级结构，适合 Detection / Segmentation 等通用任务。[arXiv](https://arxiv.org/abs/2103.14030)
- **MAE**（*Masked Autoencoders Are Scalable Vision Learners*，CVPR 2022）：解决「视觉自监督预训练怎么做」——随机遮住大部分 Patch、让模型重建像素，从图像本身学到可扩展的表征。[arXiv](https://arxiv.org/abs/2111.06377)
- **DINO**（*Emerging Properties in Self-Supervised Vision Transformers*，ICCV 2021）：解决「没有标签怎么让 ViT 学到语义特征」——自蒸馏训练，学出的特征在分割、检测等任务上非常有用。[arXiv](https://arxiv.org/abs/2104.14294)
- **DINOv2**（*DINOv2: Learning Robust Visual Features without Supervision*）：把 DINO 的自监督蒸馏做到大规模数据上，产出可直接用于下游任务的通用视觉特征。[arXiv](https://arxiv.org/abs/2304.07193)
- **CLIP**（*Learning Transferable Visual Models From Natural Language Supervision*，ICML 2021）：解决「图文对齐与 Zero-shot 迁移」——图文对比学习，让视觉特征与文本特征对齐，分类时不需要微调。[arXiv](https://arxiv.org/abs/2103.00020)

而 MAE、DINO、CLIP 等工作进一步回答了「视觉 Transformer 应该如何进行大规模自监督 / 多模态预训练」。从这里开始，基本就进入现代视觉基础模型的主线了。

## 参考资料

- Dosovitskiy et al., _An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale_, ICLR 2021：https://arxiv.org/abs/2010.11929
- Vaswani et al., _Attention Is All You Need_, NeurIPS 2017：https://arxiv.org/abs/1706.03762
- Touvron et al., _Training data-efficient image transformers & distillation through attention_, ICML 2021：https://arxiv.org/abs/2012.12877
- Liu et al., _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows_, ICCV 2021：https://arxiv.org/abs/2103.14030
- 知乎：ViT (Vision Transformer)原理及代码解析：https://zhuanlan.zhihu.com/p/427388113
- 知乎：再读 VIT，还有多少细节是你不知道的：https://zhuanlan.zhihu.com/p/657666107
- Bilibili：15分钟认识ViT！【视觉Transformer】：https://www.bilibili.com/video/BV1gnWdzSEzY/
- Bilibili：ViT 论文精读：https://www.bilibili.com/video/BV15P4y137jb/
