---
author: 芙芙
pubDatetime: 2026-08-31
title: 持续学习 Continual Learning：如何让模型学习新知识，却不忘记过去？
featured: false
draft: false
tags:
  - 持续学习
  - Continual Learning
  - Catastrophic Forgetting
  - EWC
  - 学习笔记
category: 深度学习
description: 系统整理持续学习（Continual Learning）主线：从灾难性遗忘出发，串起 EWC、Replay、知识蒸馏、GEM、动态架构五条经典路线，再延伸到 Online、Open-World、PEFT、VLM、LLM 与 Lifelong Agent 等现代方向。
---

> 本文系统整理持续学习（Continual Learning）的主线：从灾难性遗忘出发，串起 EWC、Replay、知识蒸馏、GEM、动态架构五条经典路线，再延伸到 Online、Open-World、PEFT、VLM、LLM 与 Lifelong Agent 等现代方向。
>
> 一句话概括核心矛盾：**如何在 Stability（稳定性）与 Plasticity（可塑性）之间取得平衡**——既要学得进新知识，又不要忘掉旧能力。

## 一、灾难性遗忘与持续学习

### 1.1 学新忘旧的悖论

深度学习模型有一个看起来有些反常的问题：人可以一边学习新知识、一边保留过去学到的东西，但神经网络往往做不到。假设我们先训练一个模型识别猫和狗（Task 1，准确率 95%），随后不给它猫狗数据，而是继续让它学习汽车和飞机（Task 2，准确率 94%），这时重新测试猫狗，准确率可能已经从 95% 掉到了 60% 甚至更低。也就是说，**学习新知识会破坏旧知识**，这种现象被称为 **Catastrophic Forgetting（灾难性遗忘）**：

$$
\text{学习新知识}\quad\Longrightarrow\quad\text{破坏旧知识}
$$

持续学习（Continual Learning，CL）研究的核心问题就是：如何让一个模型在数据、任务和环境不断变化的情况下持续学习，同时尽可能保留过去已经学到的能力。

### 1.2 持续学习是什么：不是重新训练

传统机器学习通常假设我们在训练开始之前就拥有完整的数据集 $D=\{(x_i,y_i)\}_{i=1}^{N}$，然后用它统一训练模型 $\theta^*=\arg\min_\theta L(D;\theta)$。但现实世界显然不是这样：真实数据更可能不断到来，$D_1\rightarrow D_2\rightarrow D_3\rightarrow\cdots$。

比如一个机器人：第一个月学会在办公室移动，第二个月进入仓库，第三个月学习新的机械操作，第四个月遇到以前从未见过的物体。我们当然不希望每出现一点新数据，就收集所有历史数据、重新训练整个模型、再重新部署。理想情况应该是：

$$
Model_t + D_{t+1} \rightarrow Model_{t+1}
$$

同时做到两件事：**Learn New Knowledge（学会新知识）** 和 **Remember Old Knowledge（记住旧知识）**。

### 1.3 Stability 与 Plasticity 的矛盾

持续学习背后其实存在一个非常核心的矛盾：**Stability（稳定性）** 表示模型保留旧知识的能力，**Plasticity（可塑性）** 表示模型学习新知识的能力。如果过度强调 Stability，模型几乎不愿意改变，自然学不好新东西；如果过度强调 Plasticity，模型就容易发生灾难性遗忘。因此几乎所有持续学习方法，本质上都在解决同一个问题：

$$
\boxed{\text{如何在 Stability 和 Plasticity 之间取得平衡？}}
$$

不同方法的区别，只是选择了不同的解决方式。

## 二、经典方法：五种思路

### 2.1 Regularization：重要参数不要乱改

最经典的持续学习思想之一，是学习新任务的时候尽量不要修改那些对旧任务非常重要的参数，代表方法是 **EWC（Elastic Weight Consolidation）**。假设模型在旧任务训练结束后得到参数 $\theta^*$，其中有些参数对识别猫非常重要、有些影响很小、有些对识别狗非常重要，那么学习新任务时：影响小的参数可以大胆修改，重要的参数尽量少动。

于是 EWC 的 Loss 可以写成：

$$
L(\theta)=L_{\text{new}}+\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^*)^2
$$

其中 $F_i$ 表示第 $i$ 个参数对于旧任务的重要程度，通常利用 Fisher Information 进行估计。因此 EWC 的思想可以非常简单地概括成：**越重要的旧参数，修改成本越高**。这类方法后来被统称为 Regularization-based Continual Learning，代表工作包括 EWC、Synaptic Intelligence、MAS，它们的共同思想都是通过给参数更新增加约束来保护旧知识。

### 2.2 Replay：一边学新一边复习

另一条非常经典、而且直到今天依然十分强大的路线叫做 **Replay**，它的思想甚至比 EWC 更容易理解。假设模型之前学过猫、狗、汽车、飞机，那么我们从旧数据中保存一小部分作为 Memory Buffer（比如每类 20 张），现在开始学习苹果、香蕉时，训练数据不只使用新数据，而是：

$$
D_{\text{train}}=D_{\text{new}}\cup D_{\text{memory}}
$$

也就是一边学新题、一边复习旧题，这其实和人的学习方式非常相似。

后来 Replay 又发展出了很多不同形式：

- **Experience Replay**：最直接，保存过去的真实样本 $Memory=\{x_{\text{old}},y_{\text{old}}\}$；
- **Feature Replay**：不保存原图，而是保存模型以前提取出来的 Feature $f(x)$，减少存储成本；
- **Generative Replay**：旧数据一个都不存，而是训练一个生成模型 $z\rightarrow G\rightarrow\hat{x}_{\text{old}}$，需要复习的时候由生成模型重新生成过去的数据。

Replay 最大的问题自然是 **Memory Size** 不可能无限增加，因此「Memory 里面到底应该保存哪些样本」直到现在仍然是 Online Continual Learning 中的重要研究问题。经典代表工作包括 iCaRL 和后来的 Dark Experience Replay（DER）。

### 2.3 Knowledge Distillation：让新模型记住旧模型的行为

另一个非常经典的方法是 **Knowledge Distillation**，代表工作是 Learning without Forgetting（LwF）。新模型除了学习新的任务之外，还需要模仿旧模型：$L=L_{\text{new}}+\lambda L_{\text{distill}}$，其中 $L_{\text{distill}}$ 要求新模型在旧知识上的行为尽量接近旧模型——旧模型产生 soft target，新模型去模仿。这个思想和普通知识蒸馏非常接近，区别在于这里 Knowledge Distillation 的主要目标不是压缩模型，而是**防止模型遗忘**。因此持续学习和知识蒸馏实际上有非常紧密的联系。

### 2.4 Gradient Constraint：这个梯度会不会伤害旧任务

前面的方法是在限制「参数应该怎么改」，另一类方法则直接去研究「梯度应该往哪里走」，最经典的是 **GEM（Gradient Episodic Memory）**。假设新任务的梯度是 $g_{\text{new}}$、旧任务梯度为 $g_{\text{old}}$，如果 $g_{\text{new}}^T g_{\text{old}}<0$，说明学习新任务的更新方向与旧任务产生了冲突：新任务往右走、旧任务往左走，继续按新任务的梯度更新很可能导致旧任务性能下降。于是 GEM 会寻找一个新的梯度 $\tilde g$，使它尽可能学习新任务，同时又不会明显增加旧任务 Loss。

所以 GEM 的核心思想就是：**检测任务之间的梯度冲突，并修改更新方向**。这条思想后来也大量出现在 Multi-Task Learning 和 Parameter-Efficient Continual Learning 中。

### 2.5 Dynamic Architecture：给新知识加新参数

前面的几种方法有一个共同特点：新旧任务仍然在共享一套网络。那么另一种非常直接的做法是：既然共享参数容易发生干扰，那就减少共享。例如 **Progressive Neural Networks**：新任务来了以后增加新的 Network Column，同时冻结旧网络，于是旧知识理论上就很难被覆盖。

这种方法的优势非常明显（Forgetting ↓），但问题也同样明显：$Tasks\uparrow\Rightarrow Parameters\uparrow$——如果任务无限增加，模型也会越来越大（Task 1 → Module 1，Task 2 → Module 2，……Task 100 → Module 100）。现代的 Adapter、Modular Network、Mixture-of-Experts、Dynamic Expansion 实际上都能看到类似思想。
## 三、在线与开放世界的持续学习

### 3.1 Online Continual Learning：让模型真正在线学习

传统持续学习往往仍然比较理想化：Task 1 训练很多 Epoch、Task 2 训练很多 Epoch、Task 3 再训练很多 Epoch。但现实世界中的数据更像 $x_1\rightarrow x_2\rightarrow x_3\rightarrow x_4\rightarrow\cdots$ 这样流式到达，每个样本可能只出现一次，而且可能根本没有明确的「Task 1 结束、Task 2 开始」的边界。这就是 **Online Continual Learning**，它的典型特点包括：

- 数据流式到达；
- 每个数据只能访问很少次数；
- Memory 有限；
- Distribution 不断变化；
- Task Boundary 可能未知。

相比传统 Class-Incremental Learning，它更加接近真正部署后的 AI 系统。

### 3.2 Open-World Continual Learning：遇到没见过的东西怎么办

传统 Class-Incremental Learning 一般默认测试数据属于模型已经学过的某个类别。比如模型已经学过 $\{\text{猫},\text{狗},\text{汽车}\}$，突然出现一只羊驼，普通分类器可能给出 狗 0.52、猫 0.31、汽车 0.17 的分布，然后强行预测成「狗」。但真正能够长期运行的系统应该先意识到 **I don't know**：羊驼 → Unknown → 获得新数据 → 学习「羊驼」→ 以后可以正确识别。

因此 Open-World Continual Learning 实际上把 **OOD Detection + Open Set Recognition + Continual Learning** 结合了起来。2025 年发表在 *Artificial Intelligence* 的工作甚至从理论上把 OOD Detection 和 Continual Learning 联系了起来。从现实应用来看，这是非常自然的一步：真正的持续学习系统不仅应该学习新东西，还应该首先意识到自己遇到了新东西。

## 四、Foundation Model 时代的持续学习

### 4.1 模型变大了：PEFT + Continual Learning

Foundation Model 带来了另一个现实问题：模型太大了。以前一个几十 MB 的模型做 Full Fine-tune 问题可能不大，但现在 ViT、CLIP、LLM、VLM 都包含数亿甚至数十亿参数，每来一个新任务都重新 Fine-tuning 整个模型显然不现实。于是出现了 **PEFT + Continual Learning**：使用 LoRA、Adapter、Prompt Tuning、Prefix Tuning 等方法，把 Backbone 冻结，只更新极少量参数来学习新任务。

但是一个新的问题随之出现：**Adapter 应该共享还是每个任务单独建立？** 如果每个 Task 都增加 Adapter，$Tasks\uparrow\Rightarrow Parameters\uparrow$，还是会无限增长；如果所有 Task 共用 Adapter，Forgetting 又可能回来。因此「参数效率 + Stability + Plasticity」成为 Foundation Model Continual Learning 的重要问题。例如 CVPR 2024 的 **Semantically-Shifted Incremental Adapter-Tuning** 就研究了如何利用共享 Adapter 进行 ViT 的 Class-Incremental Learning，并尽量避免模型不断膨胀。

### 4.2 CLIP / VLM 的持续学习

多模态模型又产生了一个很有意思的问题：普通持续学习是 $Image\rightarrow Class$，而 CLIP 是 $Image\leftrightarrow Text$。假设一个通用 CLIP 不断学习 Medical → Satellite → Industrial → Fashion，最终我们不仅希望新领域性能提高，还希望原本 CLIP 的 Zero-shot 能力不要明显下降。因此 Multimodal Continual Learning 不能只评价 Old Task Accuracy，还需要关心 **Zero-shot Capability** 和 **Image-Text Matching**。

例如 ICLR 2025 的 **C-CLIP** 就构建了多模态持续学习 Benchmark，同时评估 Image-Text Matching 和原始 Zero-shot 能力的遗忘。这也是 Foundation Model 时代持续学习与传统 CIL 的一个明显区别：我们保护的不只是过去几个分类任务，而是一个预训练模型原本拥有的大量通用能力。

### 4.3 LLM 持续学习：三个阶段

到了 LLM，问题又进一步扩大。一个 LLM 的知识是静态预训练得到的（2023 数据、2024 数据训练出一个 LLM），但现实世界的知识一直在变化（2025、2026、2027……），显然不可能每一年都从零重新训练一个 LLM。因此需要 **LLM Continual Learning**，目前通常可以分成几个阶段：

- **Continual Pre-training**：不断给模型加入新的语料，$LLM_t+Corpus_{t+1}\rightarrow LLM_{t+1}$，目标是更新世界知识，同时减少旧知识和通用能力下降；
- **Continual Instruction Tuning**：模型不断学习新的能力（Math → Coding → Tool Use → Agent），最终希望 Math + Code + Tool + Agent 全都要，而不是学了 Agent 丢掉 Math；
- **Continual Alignment**：模型的用户偏好、Safety Policy、Alignment Goal 同样会随时间变化，$Alignment_1\rightarrow Alignment_2\rightarrow Alignment_3$，需要研究新 Alignment 是否会破坏已有 Alignment。

近年来的 LLM Continual Learning Survey 基本都已经开始按照这些阶段重新整理持续学习。
## 五、超越参数：记忆、检索与 Lifelong Agent

### 5.1 学习不一定要修改参数

LLM 和 Agent 出现以后，持续学习出现了一个更加有意思的变化。传统观点是 Learning = Update Parameters，但现在完全可以：新知识 → External Memory → 保存 → 以后 Retrieval → LLM 使用。整个过程中参数 $\theta$ 完全没有发生变化，但从系统层面看，AI 确实学会了新的知识。因此现在的 Lifelong Learning 已经逐渐从 Parameter Updating 扩展为：

$$
\boxed{\text{Parameter} + \text{Memory} + \text{Retrieval} + \text{Tools}}
$$

这也是 LLM 时代一个非常值得关注的研究方向。

### 5.2 Lifelong Agent：真正长期运行的 AI

如果进一步考虑 Agent（LLM + Memory + Tools + Environment），持续学习就不再只是「模型参数有没有忘」，而会变成：Agent 执行任务 → 发生错误 → 总结经验 → 保存 Memory → 以后遇到类似情况利用历史经验。于是新的问题出现了：

- **什么经验值得保存？** Memory 无限增长显然不现实；
- **错误经验怎么办？** 如果 Agent 得出了错误结论并永久保存进 Memory，就可能形成长期错误；
- **Memory 冲突怎么办？** 比如 Memory 1 说「A 是正确的」、Memory 2 说「A 是错误的」，模型应该相信谁？
- **什么知识应该主动遗忘？** 人类并不会永久保存所有记忆，那么 AI 是否也应该有 Active Forgetting（主动遗忘）机制？

因此未来 Lifelong Agent 研究的重点可能不再只是解决 Catastrophic Forgetting，还包括 Memory Selection（记忆选择）、Memory Consolidation（记忆巩固）、Memory Updating（记忆更新）、Memory Conflict（记忆冲突）以及 Memory Forgetting（记忆遗忘）。这让持续学习从一个传统机器学习问题，逐渐变成了一个长期智能系统的问题。

## 六、总结：从防遗忘到长期智能系统

持续学习最早主要解决的问题是 Catastrophic Forgetting，因此出现了几个经典技术路线：

```text
Continual Learning
│
├── Regularization
│     └── EWC
│
├── Replay
│     └── iCaRL / DER
│
├── Knowledge Distillation
│     └── LwF
│
├── Gradient Constraint
│     └── GEM
│
└── Dynamic Architecture
      └── Progressive Networks
```

这些方法虽然看起来不同，但本质都是在寻找 Stability 和 Plasticity 之间的平衡。

而随着 Foundation Model 的出现，持续学习正在逐渐扩展为：

```text
Continual Learning
│
├── Online Continual Learning
│
├── Open-World Continual Learning
│
├── PEFT Continual Learning
│
├── Vision-Language Continual Learning
│
├── LLM Continual Learning
│
└── Lifelong Agent
      ├── Memory
      ├── Retrieval
      ├── Tools
      └── Experience
```

因此未来的 Continual Learning 很可能不只是研究「怎样防止一个神经网络忘记」，而是在回答一个更加宏观的问题：

$$
\boxed{\text{如何让一个 AI 系统在长期运行过程中不断获得知识、修正知识、保留重要经验，并适应一个持续变化的世界？}}
$$

这可能才是 Continual Learning 最终真正想解决的问题。

## 参考资料

- Kirkpatrick et al., _Overcoming catastrophic forgetting in neural networks_, PNAS, 2017：https://www.pnas.org/doi/10.1073/pnas.1611835114
- Li & Hoiem, _Learning without Forgetting_, ECCV 2016：https://arxiv.org/abs/1606.09282
- Rebuffi et al., _iCaRL: Incremental Classifier and Representation Learning_, CVPR 2017：https://arxiv.org/abs/1611.07725
- Lopez-Paz & Ranzato, _Gradient Episodic Memory for Continual Learning_, NeurIPS 2017：https://arxiv.org/abs/1706.08840
- Rusu et al., _Progressive Neural Networks_, 2016：https://arxiv.org/abs/1606.04671
- Buzzega et al., _Dark Experience for General Continual Learning_, NeurIPS 2020：https://arxiv.org/abs/2004.07211
- Tan et al., _Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer_, CVPR 2024：https://arxiv.org/abs/2403.19979
- Kim et al., _Open-world continual learning: Unifying novelty detection and continual learning_, Artificial Intelligence, 2025：https://arxiv.org/abs/2304.10038
- Liu et al., _C-CLIP: Multimodal Continual Learning for Vision-Language Model_, ICLR 2025
- Wu et al., _Continual Learning for Large Language Models: A Survey_, 2024
- Shi et al., _Continual Learning of Large Language Models: A Comprehensive Survey_, ACM Computing Surveys, 2025
- Yang et al., _Recent Advances of Foundation Language Models-based Continual Learning: A Survey_, ACM Computing Surveys, 2025
- Zheng et al., _Lifelong Learning of Large Language Model based Agents: A Roadmap_, 2025：https://arxiv.org/abs/2501.07278
