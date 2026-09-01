---
author: 芙芙
pubDatetime: 2026-08-31
title: 迁移学习（Transfer Learning）：把已经学会的知识迁移到新任务
featured: false
draft: false
tags:
  - 迁移学习
  - Transfer Learning
  - Fine-tuning
  - Domain Adaptation
  - 学习笔记
category: 深度学习
description: 系统整理迁移学习（Transfer Learning）主线：从「旧知识能不能帮到新任务」出发，讲清 Pretraining + Fine-tuning 范式、Inductive / Transductive / Unsupervised 三种类型、Domain Shift 与 Domain Adaptation，再到各领域的应用、Negative Transfer 与现代 Foundation Model 适配。
---

> 本文系统整理迁移学习（Transfer Learning）的主线：从「旧知识能不能帮到新任务」出发，讲清 Pretraining + Fine-tuning 范式、Inductive / Transductive / Unsupervised 三种类型、Domain Shift 与 Domain Adaptation，再到各领域的应用、Negative Transfer 与现代 Foundation Model 适配。
>
> 一句话概括：**迁移学习就是把已经学会的知识，用到一个新的相关问题上**；最典型的模式就是 Pretraining → Fine-tuning。

## 一、什么是迁移学习

### 1.1 为什么不能每个任务都从头训练

在传统机器学习中，我们通常默认每一个任务都要单独准备数据、单独训练模型：猫狗分类就准备猫狗数据从头训练，换成肺部 CT 分类就重新准备数据重新训练。但问题在于，现实世界中很多任务并没有足够的数据——医疗影像标注需要医生、工业缺陷数据很少、新设备故障数据不足、新语言语料较少、机器人真实环境数据采集成本很高。

于是就有了一个很自然的想法：**能不能把模型以前已经学会的知识，拿到新的任务上继续使用？** 这就是 Transfer Learning（迁移学习）。经典综述通常将其定义为：利用源领域和源任务中的知识，帮助目标领域中的目标任务学习：

$$
\boxed{\text{旧任务中学到的知识}\rightarrow\text{帮助解决新的相关任务}}
$$

### 1.2 一个最简单的例子：ImageNet → 猫狗分类

假设需要训练一个猫狗分类器。如果从头开始，就是 Random Initialization → 猫狗数据集 → Training → 模型；但实际上我们已经有大量在 ImageNet 上预训练好的模型（ResNet、DenseNet、EfficientNet、ViT），它们通过数百万张自然图片学到了大量视觉知识：边缘、纹理、轮廓、形状、局部结构、物体语义。

于是可以这样做：ImageNet Pretrained ResNet → 猫狗数据 → Fine-tuning → 猫狗分类器。这就是一个典型的迁移学习过程。这里最关键的一点是：**ImageNet 分类和猫狗分类并不是完全相同的任务，但它们之间存在大量可以复用的视觉知识**，因此模型不需要从零开始。

### 1.3 迁移的到底是什么

迁移的并不一定是最终预测结果，真正被迁移的通常是 Feature（特征）、Representation（表示）、Model Parameters（模型参数），甚至可以是 Policy、Knowledge、Samples。因此迁移学习真正想回答的问题是：

$$
\boxed{\text{哪些知识可以被复用？}}\qquad\boxed{\text{这些知识应该如何迁移？}}
$$

例如一个 ImageNet 预训练 CNN 的浅层通常已经学到了 Edge、Texture、Color、Shape，这些特征在很多视觉任务里都是通用的——即使目标任务变成工业零件缺陷检测，这些视觉表示仍然可能有用。

### 1.4 Source Domain 与 Target Domain

迁移学习中有两个最基本的概念：**Source Domain（源领域）** $D_s$，指已经拥有大量数据或已经学习过知识的领域（比如 ImageNet）；**Target Domain（目标领域）** $D_t$，指我们真正希望解决的新问题所在的领域（比如肺部 CT）。除了 Domain 还有 Source Task $T_s$ 和 Target Task $T_t$，因此一个完整的迁移学习问题可以写成：

$$
(D_s,T_s)\rightarrow(D_t,T_t)
$$

比如 ImageNet → CT 就是 Source Domain → Target Domain 的迁移。

## 二、迁移学习的主要类型

### 2.1 最常见的形式：Pretraining + Fine-tuning

现代深度学习中最常见的迁移学习形式是：

$$
\boxed{Pretraining\rightarrow Fine\text{-}tuning}
$$

计算机视觉是 ImageNet → Pretrained ResNet / ViT → Downstream Task → Fine-tuning；NLP 是 Large-scale Text → BERT → Sentiment Analysis / QA / NER。也就是先在一个大规模通用数据集上学习通用知识，再在特定任务上微调。

### 2.2 Fine-tuning 就等于迁移学习吗

不完全等价：$Fine\text{-}tuning\subset Transfer\ Learning$。Fine-tuning 只是迁移学习的一种实现方式，迁移学习还包括 Feature Transfer、Parameter Transfer、Instance Transfer、Domain Adaptation、Knowledge Transfer、Representation Alignment 等。所以：**迁移学习是思想，Fine-tuning 是其中一种常见技术路线**。

### 2.3 Feature Extraction 和 Fine-tuning 的区别

假设现在有一个 ImageNet 上训练好的 ResNet，有两种常见用法：

- **Feature Extraction**：冻结 Backbone（$\theta_{backbone}=fixed$），只训练新的分类器。优点是训练快、数据需求小、不容易破坏已有特征；缺点是对目标任务适应能力有限。
- **Fine-tuning**：允许 Backbone 参数继续更新（$\theta_{backbone}\rightarrow Update$），比如替换分类头后用较小学习率训练整个模型。优点是更适合目标任务、通常性能上限更高；缺点是需要更多数据、更容易过拟合、可能破坏预训练知识。

### 2.4 三种类型：Inductive / Transductive / Unsupervised

经典迁移学习根据 Source 和 Target 的任务、领域关系分类：

- **Inductive Transfer Learning**：源任务和目标任务不同（$T_s\neq T_t$），比如 ImageNet 图像分类迁移到目标检测、Language Modeling 迁移到 Sentiment Classification；
- **Transductive Transfer Learning**：任务相同（$T_s=T_t$）但数据领域不同（$D_s\neq D_t$），比如晴天道路 → 雨天道路的语义分割，这类问题后来发展出了非常重要的 **Domain Adaptation** 方向；
- **Unsupervised Transfer Learning**：Target Domain 没有标签或只有很少标签（Source 有大量有标签数据、Target 有大量无标签数据），模型依赖 Source 的知识帮助 Target 学习，现在的 Self-supervised Learning 也和这类思想密切相关。
## 三、Domain Shift 与 Domain Adaptation

### 3.1 Domain Shift：训练和部署不是同一个分布

现实世界中经常存在 Domain Shift（领域偏移），即训练数据和真实部署环境分布不同：$P_s(X)\neq P_t(X)$。比如训练时用晴天街景、测试时是雨天街景，虽然任务都是车辆检测，但输入分布变了。常见的 Domain Shift 包括：天气变化（Sunny → Rainy）、摄像头变化（Camera A → Camera B）、风格变化（Photo → Cartoon）、合成数据到真实数据（Simulation → Real World）、不同传感器（RGB → Thermal）。这些都需要把 Source 中学到的知识迁移到 Target。

### 3.2 Domain Adaptation：任务没变，数据分布变了

**Domain Adaptation（域适应）** 是迁移学习的一个重要子方向：$Domain\ Adaptation\subset Transfer\ Learning$。它通常假设 $T_s=T_t$（任务没变）但 $P_s(X)\neq P_t(X)$（数据分布变了），比如真实晴天街景 → 真实雨天街景的车辆检测。训练时通常可以看到 Target Domain（$Source+Target\rightarrow Training$），目标是解决分布不一致的问题。

### 3.3 迁移学习、Domain Generalization 与 Test-time Adaptation

这几个概念非常容易混，可以简单理解成：迁移学习是总称，下面包括 Pretraining + Fine-tuning、Domain Adaptation、Domain Generalization、Test-time Adaptation。

- **Transfer Learning（迁移学习）**：最宽泛，利用旧任务的知识帮助新任务；
- **Domain Adaptation（域适应）**：训练时可以看到 Target Domain，目标是解决 $P_s(X)\neq P_t(X)$；
- **Domain Generalization（域泛化）**：训练阶段看不到目标域——比如 Train 用 Sunny / Foggy / Cloudy、Test 用从未见过的 Rainy，希望模型能泛化到 Unseen Domain；
- **Test-time Adaptation（测试时适应）**：模型已经训练完成，部署时根据当前 Target Test Data 在测试阶段实时 Adapt 再 Prediction。

它们都可以看成：**如何将已有知识迁移到新的数据分布**。

## 四、迁移学习的应用领域

### 4.1 计算机视觉

计算机视觉是迁移学习最经典的应用领域之一，最常见的路线是 ImageNet → Pretrained Backbone → Downstream Task：图像分类（ImageNet ResNet → 鸟类分类）、目标检测（ImageNet Backbone → Faster R-CNN / YOLO → 工业缺陷检测）、语义分割（Pretrained Backbone → Medical Segmentation，或 ImageNet → Cityscapes）、人脸识别（先在大规模人脸数据集 Pretraining，再迁移到具体人员识别任务）。

### 4.2 医疗影像

医疗影像是迁移学习特别典型的应用场景，因为医学数据非常难标注：普通图片「这是不是一只猫」普通用户就能标注，但「CT 是否存在早期肺癌」往往需要专业医生，所以医学领域的数据集经常比较小。此时可以使用 ImageNet Pretraining → Medical Images，应用覆盖 CT、MRI、X-ray、Ultrasound、病理切片、眼底图像、皮肤病图像，任务包括 Classification、Detection、Segmentation。近年来医学领域也越来越多采用 Self-supervised Pretraining → Fine-tuning，因为大量医学图像其实没有人工标签。

### 4.3 NLP 与 LLM

NLP 其实已经把迁移学习变成了一种标准训练范式：过去是 Sentiment Analysis 训一个模型、NER 训另一个模型、QA 再训一个模型；后来变成 Large-scale Corpus → Pretraining → BERT → Different Downstream Tasks，一个通用预训练模型可以把语言知识迁移到情感分析、问答、NER、文本分类、自然语言推理等大量下游任务。

现在的大语言模型虽然更常使用 Pretraining、Instruction Tuning、Alignment 这些术语，但背后的核心思想依然和迁移学习高度相关：Internet-scale Corpus → Pretraining → General Language Model → Math / Code / QA / Translation / Law。模型先学语言规律、世界知识、推理模式、表达方式，再把这些知识迁移到新任务上。所以从更广的角度看，**Foundation Model 可以看作把 Transfer Learning 做到了极致**。

### 4.4 语音与遥感

语音领域也大量使用迁移学习：Large Speech Dataset → Speech Representation，然后迁移到 Speech Recognition、Speaker Recognition、Emotion Recognition、Accent Recognition，还可以做 English → Low-resource Language、Clean Speech → Noisy Speech。典型模型包括 wav2vec 2.0、HuBERT、Whisper，它们本质上也遵循 Large-scale Pretraining → Downstream Transfer。

遥感数据则非常容易发生 Domain Shift：Satellite A → Satellite B、Region A → Region B、RGB → SAR。虽然任务仍然是土地覆盖分类、建筑检测、道路提取、灾害识别，但数据分布可能差异很大，因此迁移学习和 Domain Adaptation 在遥感领域非常常见。

### 4.5 机器人、强化学习与时间序列

机器人领域中一个特别经典的迁移问题是 **Sim2Real**：$Simulation\rightarrow Real\ World$。因为真实机器人训练贵、慢、危险、数据难采，所以可以先在 MuJoCo、Isaac Sim、CARLA 这类仿真环境中训练，再将知识迁移到真实机器人，常见方法包括 Domain Randomization、Domain Adaptation、Policy Transfer、Representation Transfer。

强化学习中同样存在迁移：简单地图 → 复杂地图、Robot A → Robot B、Game A → Game B，可迁移的内容包括 Policy、Value Function、Representation、World Model，相关方向有 Transfer RL、Multi-task RL、Meta-RL、Sim2Real RL。

时间序列的迁移在工业场景中也很常见：Machine A → Machine B 的 Fault Diagnosis——某台机器积累了大量故障数据，但新机器几乎没有故障样本，这时就可以 Source Machine → Target Machine，应用包括 Bearing Fault Diagnosis、Predictive Maintenance、ECG、EEG、Power Load Forecasting、Industrial Sensor Data。

### 4.6 跨模态迁移

更加复杂的迁移问题是 Source 和 Target 甚至不是同一种模态：RGB → Depth、RGB → Thermal、甚至 Image → Text。这类问题通常属于 **Heterogeneous Transfer Learning（异构迁移学习）**，研究如何在特征空间、模态甚至数据维度不同的情况下实现知识迁移。
## 五、迁移学习的边界

### 5.1 Negative Transfer：迁移不一定总是有效

一个很重要的问题是 **Negative Transfer（负迁移）**。假设 Source 是猫狗分类、Target 是 MRI 肿瘤识别，两个任务差异非常大，如果强行迁移（$Knowledge_s\rightarrow Target$），可能不仅没有帮助，反而导致 $Performance\downarrow$。因此迁移学习并不是「只要有预训练模型就一定更好」，真正的问题是：

$$
\boxed{\text{Source 和 Target 是否真的具有可迁移知识？}}
$$

### 5.2 什么样的特征更容易迁移

在 CNN 中有一个很经典的规律：浅层通常学习 Edge、Texture、Color、Simple Shape 这些通用特征，深层逐渐学习 Dog Face、Car Wheel、Human Face 这些和训练任务更相关的特征。因此通常 **Low-level Feature 更容易迁移，High-level Feature 更加 Task-specific**——这也是为什么很多 Fine-tuning 方法会冻结前几层、微调后几层，而不是完全从头训练。

### 5.3 一个直观类比：骑自行车与可迁移性

迁移学习很像人学习新技能：已经会骑自行车的人学骑摩托车，不会从「什么叫保持平衡」重新学，而是直接迁移平衡感、转向、路况判断、空间感，再重新学习油门、离合、档位：

$$
Old\ Knowledge + New\ Learning = Transfer\ Learning
$$

但如果目标变成游泳，骑自行车学到的知识就不一定特别有用——这对应 **Transferability（可迁移性）**。如果只是从工程角度，load 一个预训练模型然后在目标数据集上 fit 其实不难；真正具有研究价值的问题是：哪一层特征最适合迁移？哪些样本值得迁移？Source 和 Target 差多远？什么时候会产生 Negative Transfer？如何衡量 Transferability？如何只迁移有用知识？因此「可迁移性」是迁移学习研究中的核心问题之一。

## 六、现代迁移学习

### 6.1 现代研究关注什么

现在单纯 ImageNet → Fine-tune 更多已经是一种基础操作，现代研究更关心：Domain Adaptation、Domain Generalization、Test-time Adaptation、Test-time Training、Continual Transfer、Cross-modal Transfer、Parameter-efficient Transfer、Foundation Model Adaptation、Transferability Estimation、Negative Transfer、Source-free Adaptation。

尤其是在 Foundation Model 时代，一个重要的问题变成：**如何以最低成本，把一个巨大的预训练模型迁移到特定任务？** 因此出现了 Adapter、Prompt Tuning、LoRA、PEFT 这类 Parameter-Efficient Fine-tuning 方法。

### 6.2 推荐后续学习路线

如果继续学习迁移学习，可以按照这条路线：

```text
Transfer Learning
→ Pretraining + Fine-tuning
→ Domain Adaptation
→ Domain Generalization
→ Test-time Adaptation
→ Test-time Training
```

从最宽泛的迁移学习出发，先掌握最主流的 Pretraining + Fine-tuning 范式，再逐步深入到解决「分布变了」的 Domain Adaptation、面对「没见过的新域」的 Domain Generalization，再到部署后实时适应的 Test-time Adaptation 与 Test-time Training。这样后面再接 OOD、TTA、TTT 就会比较自然。

## 七、总结

如果只用一句话解释迁移学习：

> **迁移学习就是把已经学会的知识，用到一个新的相关问题上。**

最典型的模式就是 Pretraining → Fine-tuning，例如 ImageNet → Medical Images，或者 Large-scale Text → BERT → Sentiment Analysis。迁移学习已经广泛应用于 Computer Vision、NLP、Large Language Models、Medical Imaging、Speech、Remote Sensing、Robotics、Reinforcement Learning、Time Series、Cross-modal Learning 等领域。

而真正值得研究的问题并不是「怎么加载一个预训练模型」，而是：

> **什么知识应该迁移？什么时候应该迁移？怎样避免 Negative Transfer？**

从这个角度看，Domain Adaptation、Domain Generalization、Test-time Adaptation 乃至今天的 Foundation Model Adaptation，其实都可以放在同一条主线上理解：**让模型已经学会的知识，更好地适应新的任务和新的数据分布。**

## 参考资料

- Pan, S. J., & Yang, Q., _A Survey on Transfer Learning_, IEEE TKDE, 2010：https://doi.org/10.1109/TKDE.2009.191
- Csurka, G., _Domain Adaptation for Visual Applications: A Comprehensive Survey_：https://arxiv.org/abs/1702.05374
- Wang, M., & Deng, W., _Deep Visual Domain Adaptation: A Survey_：https://arxiv.org/abs/1802.03601
- Devlin et al., _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_：https://arxiv.org/abs/1810.04805
- Baevski et al., _wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations_：https://arxiv.org/abs/2006.11477
