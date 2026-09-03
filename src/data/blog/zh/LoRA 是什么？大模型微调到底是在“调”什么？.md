---
author: 芙芙
pubDatetime: 2026-09-03
title: LoRA 是什么？大模型微调到底是在“调”什么？
featured: false
draft: false
tags:
  - LoRA
  - 微调
  - PEFT
  - 大模型
  - 学习笔记
category: 深度学习
description: 整理自《什么是 LoRA 大模型微调是怎么回事》视频：从「微调到底在改什么」出发，讲清 ΔW 的低秩假设、BA 分解、A/B 初始化与 α/r 缩放，再理清 LoRA 与 Adapter、SFT、QLoRA 的关系及局限。
---

> 本文主要整理自隔壁的程序员老王《什么是 LoRA 大模型微调是怎么回事》([Bilibili][1])，并补充了 LoRA+ 等内容，围绕一个问题展开：**微调一个几十亿参数的大模型，到底在“调”什么？** 答案会一路指向 LoRA 的核心——冻结原模型，只学习一个低秩的参数更新量。
>
> 一句话核心：$W'=W_0+\frac{\alpha}{r}BA$——LoRA 把微调产生的参数变化 $\Delta W$ 分解成两个小矩阵 $A$、$B$ 的乘积来学习。

## 一、为什么要微调：从通识教育到岗位培训

### 1.1 Fine-tuning 在干什么

现在的大语言模型已经非常强了：写代码、翻译、总结文章、回答常识问题、做一定程度的数学推理。那为什么还需要 Fine-tuning（微调）？原因很简单：一个通用大模型虽然“什么都知道一点”，但它不一定熟悉某个公司的业务规范、会按固定格式回答、擅长某个垂直领域、拥有某种固定说话风格、能稳定完成某一种专门任务。

可以把预训练模型理解成**一个接受了通识教育的人**，而 Fine-tuning 更像**入职之后进行的岗位培训**：通用大模型 + 金融数据 → 金融领域模型；通用大模型 + 大量客服对话 → 客服模型。问题是：大模型有几十亿、几百亿甚至更多参数，微调时难道要把所有参数全部重新训练一遍吗？当然可以，这叫 Full Fine-tuning（全参数微调），但它非常贵。于是出现了一类方法 **PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）**，而 LoRA 就是其中最著名的方法之一。

### 1.2 微调本质上还是梯度下降

假设已经有一个训练好的模型 $f(x;\theta)$，其中 $\theta$ 表示全部参数（$W_1,W_2,W_3,\cdots$），预训练阶段已经得到 $\theta_{pretrained}$。现在有新的数据集 $D_{target}$，我们希望让模型适应这个新任务，于是继续计算 Loss 并更新：

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

也就是说，**微调本质上并没有什么神秘的地方，依然是梯度下降**——和训练普通神经网络一样执行 Forward → Loss → Backward → Gradient → Update Parameters，区别只是我们不是从随机参数开始，而是从一个已经训练好的大模型开始。所以 Pretraining → Fine-tuning 可以理解成：在已有知识基础上继续学习。

### 1.3 Full Fine-tuning 的问题：贵在显存和存储

全参数微调最直接，但有两个问题。第一是显存：假设模型有 7B（$7\times10^9$）个参数，用 FP16/BF16 每个参数大约需要 2 Bytes，仅模型权重就约 14GB；但训练时还要存 Gradient、Optimizer State、Activation 和中间结果（比如 Adam 要维护 $m_t$ 和 $v_t$），真正占用的显存远大于单纯加载模型。

第二是存储：假设我们用同一个通用模型针对医疗、法律、金融、客服、代码分别微调，如果每个任务都保存完整模型（Base 7B + Medical 7B + Law 7B + Finance 7B + ……），存储成本很高。但仔细想想，这些模型的大部分知识其实都是一样的，不同的只是 $\Delta W$——那为什么要把整个 $W'$ 都保存下来，而不是只保存 $\Delta W$？这就是理解 LoRA 的第一步。

## 二、微调在改 ΔW，而 ΔW 可能是低秩的

### 2.1 微调真正改变的是 ΔW

回到 $W'=W_0+\Delta W$：$W_0$ 是预训练模型，Fine-tuning 真正需要学习的是 $\Delta W$。换句话说，**我们真正关心的不是重新学习整个 $W$，而是学习“这个任务需要在原模型基础上修改多少”**。但如果 $W\in\mathbb R^{4096\times4096}$，$\Delta W$ 同样是 $4096\times4096$，一个矩阵就有 $4096^2=1677$ 万参数，还是很多。

于是 LoRA 提出最关键的假设：**模型微调产生的参数变化 $\Delta W$，可能并不需要一个完整的高秩矩阵来表示**——也就是说 $\Delta W$ 可能具有 Low Rank（低秩）结构。

### 2.2 什么是“秩”

先直观理解 Rank。假设 $W=\begin{bmatrix}1&2\\2&4\end{bmatrix}$，第二行其实是第一行的 2 倍，两行不是独立的信息，所以它的 Rank 只有 1。也就是说，虽然矩阵看起来有很多数字，但真正独立的信息量没有那么大。对于一个更大的矩阵 $W\in\mathbb R^{d\times k}$，如果它的 Rank $r\ll\min(d,k)$，就称它具有 Low-rank Structure——这意味着**看起来很大的矩阵，实际上可以由两个很小的矩阵表示**。

### 2.3 LoRA 的核心：把 ΔW 分解成 BA

假设 $\Delta W\in\mathbb R^{d\times k}$，LoRA 不直接学习完整的 $\Delta W$，而是写成：

$$
\Delta W=BA
$$

其中 $A\in\mathbb R^{r\times k}$、$B\in\mathbb R^{d\times r}$，且 $r\ll d,k$。例如 $d=k=4096$ 时，原本 $\Delta W$ 有 $4096\times4096=1677$ 万参数；如果 LoRA Rank $r=8$，那么 $A$ 是 $8\times4096$、$B$ 是 $4096\times8$，总参数 $8\times4096+4096\times8=65536$——比原来少了大约 256 倍。这就是 LoRA 的核心：原本学习 $\Delta W$，改成学习 $BA$；$W'=W_0+\Delta W$ 变成 $W'=W_0+BA$，训练时 $W_0$ 冻结，只训练 $A$ 和 $B$。这就是 Low-Rank Adaptation（低秩适配）。

### 2.4 LoRA 的前向传播与直观理解

原始 Linear 是 $y=W_0x$，加入 LoRA 后变成：

$$
y=(W_0+\Delta W)x=W_0x+BAx
$$

可以画成两条路径：上面的 $W_0$ 是原始预训练模型、不更新；下面的 $A\to B$ 是 LoRA Branch、会被训练。最终输出 = Original Output + LoRA Adjustment。

一个特别直观的理解是：把大模型想象成一个已经掌握大量知识的人，原始参数 $W_0$ 代表“这个人原本的大脑”；现在我们希望他学会公司的客服话术——Full Fine-tuning 相当于把整个大脑重新训练，而 LoRA 更像是**原来的知识全部保留，只增加一个很小的“岗位知识插件”**。因此 Base Model + LoRA Adapter A = 客服模型，换一个 Adapter 就变成医疗模型、代码模型——**一个 Base Model + 多个 LoRA** 就可以对应很多不同任务，这也是 LoRA 在实际应用中非常方便的一点。

### 2.5 为什么 ΔW 可以是低秩的

因为矩阵乘积 $BA$ 的 Rank 最大不会超过 $r$，即 $rank(BA)\le r$——如果 $r=8$，那么无论原始 $4096\times4096$ 矩阵有多大，LoRA 更新矩阵的 Rank 最大只有 8。那为什么微调的 $\Delta W$ 可以是低秩的？这是 LoRA 最核心的假设之一：大模型预训练之后已经学到了非常丰富的知识（什么是猫、什么是法律、什么是 Python、什么是金融、什么是语言结构），微调通常不是把整个模型推倒重学，而只是在原本能力基础上对某些行为做有限调整（比如“请用法律文书格式回答”“请模仿某种客服风格”），需要改变的有效方向远少于模型完整参数空间的维度。LoRA 原论文称，大模型在适应具体任务时权重更新存在较低的 intrinsic rank，因此可以通过低秩分解有效表示这种更新——实验中仅训练很少一部分参数，就获得了与全参数 Fine-tuning 相当甚至更好的效果。
## 三、LoRA 的关键设计

### 3.1 Rank r 与 Alpha 缩放

LoRA 最重要的超参数之一是 LoRA Rank $r$（常见 $r=4,8,16,32,64$）。Rank 越小，可训练参数越少——更省显存、更快、LoRA 文件更小，但表示能力也更弱；Rank 越大，Capacity 越高但同时参数也越多。所以 $r$ 实际上控制的是：**允许模型对原模型进行多复杂的修改**。

实际 LoRA 中通常还带一个缩放系数，写成：

$$
\Delta W=\frac{\alpha}{r}BA
$$

于是 $y=W_0x+\frac{\alpha}{r}BAx$。这里的 $\alpha$（LoRA Alpha）控制 LoRA Branch 对原模型影响的强度，$\frac{\alpha}{r}$ 可以简单理解成 LoRA 更新的 Scale——例如 $r=8$、$\alpha=16$ 时 $\alpha/r=2$，LoRA 的输出会乘 2 再加到原模型上。

### 3.2 A/B 的初始化：一个随机、一个为零

LoRA 有一个非常巧妙的设计：通常让一个矩阵随机初始化、另一个矩阵初始化为 0，比如 $A$ 随机初始化、$B=0$。这样训练开始时 $BA=0$、$\Delta W=0$、$W'=W_0$——**刚开始加入 LoRA 时，模型行为完全等于原始模型**，然后随着训练 $B$ 逐渐变化，$BA$ 开始产生有效更新，训练会比较稳定。

为什么 A 和 B 不能都初始化为 0？因为 $\Delta W=BA$ 对 Loss 求梯度时会涉及 $\partial(BA)/\partial A$（包含 $B$）和 $\partial(BA)/\partial B$（包含 $A$）：如果 $B=0$，那么对应方向梯度可能为零；$A=0$ 也会导致另一个方向无法获得有效梯度。两个矩阵全部初始化为 0，训练一开始就无法打破这种零状态。所以一般让 A Random、B Zero（或等价初始化），保证 $BA=0$ 初始模型不变的同时梯度能正常传播。

### 3.3 LoRA 插在哪里：Attention 与 MLP

Transformer 中存在很多大矩阵：Self-Attention 的 $Q=XW_Q$、$K=XW_K$、$V=XW_V$、$O=HW_O$，以及 MLP 的 $W_{up}$、$W_{down}$ 等。LoRA 可以加到这些 Linear Layer 上，例如 $W_Q$ 变成 $W_Q+B_QA_Q$、$W_V$ 变成 $W_V+B_VA_V$。原始 LoRA 工作重点讨论了在 Transformer 的 Attention Projection 上应用低秩更新（如 $W_Q$ 和 $W_V$），现代工具链中也经常把 LoRA 扩展到更多 Attention 和 MLP Linear Layer。所以配置 LoRA 时经常能看到 target_modules 填 ["q_proj", "v_proj"]，或更完整的 ["q_proj", "k_proj", "v_proj", "o_proj"]，甚至包含 "gate_proj"、"up_proj"、"down_proj"。

## 四、LoRA 的实战认知

### 4.1 到底训练多少参数，显存为什么降低

假设原模型 7B（70 亿参数），LoRA 可能只训练几百万到几十 M 参数，可训练比例可能只有 0.x% 甚至更低——LoRA 原论文在 GPT-3 175B 的实验报告，相比用 Adam 的全参数微调，可将可训练参数量降低最高约 $10^4$ 倍，同时显著减少 GPU 显存需求。但**LoRA 并没有让原模型变小**：原模型还是 7B，推理仍要加载 Base Model，只是需要训练的参数变少了。

显存下降的原因：Full Fine-tuning 要为所有参数保存 Weight、Gradient、Optimizer States；而 LoRA 中原始 $W_0$ 被冻结，不需要为所有原始参数保存完整训练状态，只需对 $A,B$ 保存 Gradient、Adam m、Adam v。不过要注意：**LoRA 并不意味着训练显存只剩百分之零点几**——仍然要加载 Base Model、保存 Activations、执行 Forward/Backward、保存部分中间结果，因此参数减少比例和最终显存减少比例并不是同一个概念。

### 4.2 推理：分开加载或 Merge

训练完成后 $W'=W_0+\frac{\alpha}{r}BA$，推理有两种方式。方法一是 Base + LoRA 分开加载（Base Model + LoRA Adapter → Inference），优点是可以快速切换不同 LoRA（Base + Medical LoRA、Base + Law LoRA……很方便）；方法二是 Merge：直接计算 $W_{merged}=W_0+\frac{\alpha}{r}BA$，把 $W_0$ 替换成 $W_{merged}$，推理时就不需要额外计算 LoRA Branch。LoRA 原论文的一个重要特点就是：**LoRA 权重可以和 Base 权重合并，因此理论上不会像传统 Adapter 那样额外增加推理延迟**。

### 4.3 LoRA 与 Adapter、SFT、QLoRA 的区别

- **LoRA vs Adapter**：在 LoRA 之前已经有 Adapter——在 Transformer 中额外插入一个小神经网络（Down Projection → Activation → Up Projection），训练时 Base Model 冻结、Adapter 训练。思想和 LoRA 很像（都不更新整个大模型），但区别是 Adapter **增加新的网络层**，LoRA **给原来的权重矩阵增加一个低秩更新**；LoRA 可以 Merge（$W_0+BA$），推理阶段直接恢复成普通 Linear Layer。
- **LoRA vs SFT**：不是一回事。SFT（Supervised Fine-Tuning）描述的是**使用监督数据进行微调**（instruction-response 数据训练模型）；LoRA 描述的是**参数怎么更新**。两者是不同维度的概念，因此 SFT + LoRA（用监督数据做 SFT、参数更新用 LoRA）和 SFT + Full Fine-tuning 都是成立的组合。
- **LoRA vs QLoRA**：LoRA 已经省显存，但 Base Model 本身仍要加载到显存（比如 65B 模型 FP16 权重就非常大）。QLoRA 的核心可以粗略理解成 Quantized Base Model + LoRA——4-bit Base + BF16/FP16 可训练的 LoRA，进一步降低 Base Model 的显存占用。所以 QLoRA 并不是完全替代 LoRA 的新东西，而是**在 LoRA 基础上进一步通过量化降低 Base Model 的显存占用**。
## 五、LoRA 的边界

### 5.1 一个常见误区：LoRA 不是模型压缩

严格来说不是。LoRA 的目标是 Parameter-Efficient Fine-Tuning（参数高效微调），不是 Model Compression（模型压缩）。7B Base + 20MB LoRA 并不意味着最终模型只有 20MB，而是 14GB Base Model + 20MB LoRA。LoRA 文件之所以小，是因为它只保存“相对于 Base Model 的变化”——所以 LoRA 文件离开对应 Base Model 通常不能单独运行。

一个更形象的例子：假设一本书有 1000 页，现在要发布第二版，但真正改动的只有 20 页。Full Fine-tuning 相当于把新的 1000 页整本重新保存一份；LoRA 相当于原来的 1000 页不动，只保存一份“修改说明”（第 37 页第 4 行改成……第 89 页增加……第 732 页删除……）。Base Model + LoRA 就像原版书 + 修改补丁，最终得到新的模型行为。

### 5.2 为什么适合个人小团队，而且不只用于 LLM

如果每个任务都 Full Fine-tuning，需要大显存、大量 GPU、很长的训练时间和完整的模型存储，普通用户很难承担。而 LoRA 只训练 $A,B$：显存压力显著降低、Checkpoint 非常小、同一个 Base Model 可以挂很多 LoRA、更适合快速实验和多任务定制——这也是 LoRA 迅速成为 LLM、Diffusion Model 等领域主流 PEFT 方法的重要原因。

LoRA 不只用于 LLM。Stable Diffusion 就是现在很多人最熟悉的 LoRA 应用之一：训练某个角色 LoRA、某种画风 LoRA、某种服装 LoRA，Base（Stable Diffusion）保持不变，只学习 LoRA。因此一个几 GB 的 Base Model 可以配几十 MB 大小的角色 LoRA——思想与 LLM 完全一样：$Base\ Knowledge + Small\ Task\ \Delta$。

### 5.3 LoRA 的局限

LoRA 并不是万能的：

1. **Rank 太小**：如果目标任务和 Base Model 差异特别大，$\Delta W$ 可能需要更复杂的变化，$r$ 太小可能无法表示足够复杂的更新；
2. **Rank 太大**：$r$ 不断增加时 LoRA 参数量 $r(d+k)$ 也不断增加，逐渐失去 Parameter-Efficient 的优势；
3. **不一定永远等于 Full Fine-tuning**：LoRA 在多个任务上结果很强，但在任务变化非常大、需要大幅修改模型内部表示时，Full Fine-tuning 仍可能有更高的能力上限；
4. **数据质量仍然非常重要**：LoRA 只解决“怎么更新参数”，不会自动解决“训练数据质量差”——错误、冲突、重复、格式混乱的数据，LoRA 同样会学进去。

## 六、LoRA 家族

### 6.1 为什么 LoRA 之后还有很多改进

LoRA 很成功，但不是终点，后面出现了很多变体：**QLoRA** 解决 Base Model 显存还是太高；**AdaLoRA** 尝试不同层动态分配不同 Rank；**DoRA** 进一步把权重更新中的 Direction 和 Magnitude 分离；**LoRA+** 研究 $A$、$B$ 使用相同 Learning Rate 是否合理，并提出让两者使用不同的学习率（后续研究发现这种简单修改在一些设置下可以加速训练并提高效果）。所以现在 LoRA 已经发展成一个很大的 PEFT 家族。

### 6.2 一张图理解整个关系

```text
                    Fine-tuning
                         │
            ┌────────────┴────────────┐
            │                         │
      Full Fine-tuning               PEFT
            │                         │
     所有参数都更新          ┌────────┼────────┐
                             │        │        │
                          Adapter  Prompt   LoRA
                                            │
                                  ┌─────────┼─────────┐
                                  │         │         │
                                QLoRA    AdaLoRA    DoRA
```

而如果按照训练目标区分，SFT、DPO、RLHF……和 Full FT、LoRA、QLoRA……其实是**两个维度**——SFT + LoRA、DPO + LoRA 都是完全成立的组合。

## 七、回顾与总结

### 7.1 重新理解 LoRA

LoRA 最开始看起来公式很多，但核心思想非常简单。原本 $W_0$ 已经训练得很好，新任务只需要一个 $\Delta W$，所以 $W'=W_0+\Delta W$；LoRA 进一步假设 $\Delta W$ 不需要完整高维参数空间，可以写成 $\Delta W=BA$：

$$
W'=W_0+BA
$$

训练时 $W_0$ Frozen，只学习 $A,B$——最终把一个 $d\times k$ 的大矩阵更新压缩成 $d\times r$ 和 $r\times k$ 两个小矩阵（$r\ll d,k$），参数量从 $dk$ 降低成 $r(d+k)$。这就是 LoRA 能够大幅减少可训练参数的数学原因。

### 7.2 总结

如果只记一句话：**LoRA 就是冻结原模型，只学习一个低秩的参数更新量**。完整一点：$W'=W_0+\frac{\alpha}{r}BA$，其中 $W_0$ 是预训练模型参数（冻结）、$A,B$ 是 LoRA 参数（训练）、$r$ 是 Low Rank、$\alpha$ 是 LoRA Scaling。它的核心优势是：可训练参数极少、显著降低微调显存、Checkpoint 很小、一个 Base Model 可配多个 LoRA、LoRA 可以 Merge 回原始权重、Merge 后通常不额外增加推理延迟。

因此 LoRA 真正解决的问题并不是“怎么让模型变得更小”，而是：

$$
\boxed{\text{怎么以更低成本修改一个已经很大的模型？}}
$$

如果把 Base Model 看作已经学会大量通用知识的大脑，那么 LoRA 就像**一个非常小的技能补丁**：Pretrained LLM（冻结）→ Base Model $W_0$ + Trainable LoRA $A\times B$ → Task-specific Model。这也是为什么在今天的大模型时代，LoRA 已经成为最重要的 Parameter-Efficient Fine-tuning 方法之一。

### 7.3 推荐后续学习

理解 LoRA 后，可以沿着 Fine-tuning → PEFT → LoRA → QLoRA → SFT → DPO/RLHF 这条路线继续。其中一定要把两个维度分清楚：

- **LoRA / QLoRA：怎么更新参数**；
- **SFT / DPO / RLHF：用什么训练目标让模型学什么**。

这两个概念一旦分清，大模型“训练、微调、对齐”这一整套体系就会容易理解很多。

## 参考资料

- 隔壁的程序员老王：《什么是 LoRA 大模型微调是怎么回事》：https://www.bilibili.com/video/BV1PvwYzxE9D/
- Hu et al., _LoRA: Low-Rank Adaptation of Large Language Models_, 2021：https://arxiv.org/abs/2106.09685
- Hayou et al., _LoRA+: Efficient Low Rank Adaptation of Large Models_, 2024：https://arxiv.org/abs/2402.12354

[1]: https://www.bilibili.com/video/BV1PvwYzxE9D/ "什么是LoRA 大模型微调是怎么回事_哔哩哔哩_bilibili"
