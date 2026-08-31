---
author: 芙芙
pubDatetime: 2026-08-30
title: 学习笔记：PPO 与 GRPO 原理（从 Policy Gradient 到 LLM 强化学习）
featured: false
draft: false
tags:
  - 强化学习
  - RL
  - PPO
  - GRPO
  - 学习笔记
category: 深度学习
description: 整理自 B 站「大白话 04」视频：承接 RL 基本原理，从 Policy Gradient 的更新过猛问题讲起，推导 Importance Sampling、PPO 的 Clip 目标、Critic 与 GAE，再讲 GRPO 为什么能用组内相对比较省掉 Value Model，并给出 PPO vs GRPO 的完整对比与数字例子。
---

> 承接上一篇[《强化学习 RL 基本原理》](https://titourpast.github.io/blog/posts/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E4%BB%8E-mdpbellman-%E6%96%B9%E7%A8%8B%E5%88%B0-monte-carlotd-%E4%B8%8E-q-learning/)：那里讲完了 MC → TD → SARSA → Q-Learning → DQN → Policy Gradient → Advantage → Actor-Critic。
>
> 本篇是 **04 期**的完整笔记：PPO 为什么这么设计，GRPO 又为什么能把 PPO 改得更适合 LLM。([哔哩哔哩][1])

如果说 03 是整个 RL 地图，那么 04 就把镜头拉到现在 LLM RL 中非常重要的一条线上：

$$
Policy\ Gradient
\rightarrow PPO
\rightarrow GRPO
$$

---

## 一、Policy Gradient 的基本问题

我们现在有：

$$
\pi_\theta(a|s)
$$

目标就是：

$$
\max_\theta
J(\theta)
$$

Policy Gradient 最经典的形式之一：

$$
\nabla_\theta J(\theta)
=
\mathbb E[
\nabla_\theta\log\pi_\theta(a|s)
Q^\pi(s,a)
]
$$

直觉非常简单。

如果：

$$
Q(s,a)>0
$$

说明这个动作不错，那么：

$$
\uparrow \pi(a|s)
$$

反过来，如果动作很差：

$$
\downarrow \pi(a|s)
$$

所以 RL 本质上就是：

> **好动作以后多做，坏动作以后少做。**

### 但问题来了：更新一次太猛会怎样？

Policy Gradient 有一个很大的问题：

> **策略一次更新太猛，可能直接把模型训崩。**

例如原来：

$$
\pi_{\text{old}}(a|s)=0.2
$$

一次梯度更新：

$$
0.2\rightarrow0.9
$$

变化太大。

RL 数据又是由当前 Policy 自己采出来的，因此 Policy 变化过大以后，旧数据和新 Policy 的分布差距会越来越大——**用旧策略采的数据去更新新策略，本身就不可靠了**。

PPO 的核心思想就是：

> **可以更新，但不要一步走太远。**

这也是 Proximal Policy Optimization 里 **Proximal（近端）** 的含义。

---

## 二、为什么需要 Advantage？（复习 + 深化）

但这里还有个前置问题。假设奖励：

```text
回答 A：100
回答 B：90
```

单看 Reward：

> 两个都很好。

但实际上 A 比 B 更好。

所以我们希望判断：

> **这个 Action 相对于正常水平到底好多少？**

于是引入：

$$
A(s,a)
=
Q(s,a)-V(s)
$$

也就是 **Advantage Function**。

如果：

$$
A>0
$$

这个动作比平均水平好。

如果：

$$
A<0
$$

这个动作比平均水平差。

于是优化目标可以理解成：

$$
\nabla_\theta
\log\pi_\theta(a|s)
A(s,a)
$$

> 注意：这里把 $Q$ 换成了 $A$。为什么？$Q$ 是绝对值，$A$ 是相对值——**一个动作好不好，要看它比「正常水平」好多少，而不是看它拿的奖励有多高**。上一篇的考试例子（小明 +2 vs 小红 +30）就是同一个道理。

---

## 三、Importance Sampling：为什么 PPO 里出现概率比值？

PPO 中会出现一个非常重要的概率比值：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

为什么？

因为数据是：

$$
\pi_{\text{old}}
$$

采出来的。

但我们正在训练：

$$
\pi_\theta
$$

所以需要用概率比进行修正——**把「旧策略采的数据」折算成「新策略下大概会怎样」**。

于是基本目标：

$$
L(\theta)
=
\mathbb E[
r_t(\theta)A_t
]
$$

### 数字例子

比如旧策略：

$$
\pi_{old}(攻击|s)=0.2
$$

新策略：

$$
\pi_\theta(攻击|s)=0.22
$$

那么：

$$
r=\frac{0.22}{0.2}=1.1
$$

意思就是：

> 新模型选择「攻击」的倾向是旧模型的 1.1 倍。

如果：

$$
0.2\rightarrow0.4
$$

那么：

$$
r=2
$$

说明：

> 变化太大了——旧数据算出来的目标会严重失真。

---

## 四、PPO 最著名的 Clip：限制更新幅度

PPO 的灵魂来了：

$$
L^{CLIP}(\theta)
=
\mathbb E
[
\min(
r_tA_t,
\operatorname{clip}
(r_t,1-\epsilon,1+\epsilon)A_t
)
]
$$

假设：

$$
\epsilon=0.2
$$

那么概率比最好限制在：

$$
[0.8,1.2]
$$

附近。

也就是：

```text
旧 Policy
    ↓
生成 trajectory
    ↓
计算 Reward
    ↓
计算 Advantage
    ↓
更新 Policy
    ↓
但不允许变化太猛
```

因此 PPO 的核心不是：

> **怎么让模型更新最快。**

而是：

> **怎么让 Policy 在持续变好的同时，不要突然发生巨大变化。**

### Clip 用数字看最容易懂

假设：

$$
\epsilon=0.2
$$

允许大致范围：

$$
[0.8,1.2]
$$

某个动作：

$$
A=+5
$$

说明：

> 这个动作不错，应该提高概率。

原来：

$$
P=20\%
$$

更新到：

$$
P=22\%
$$

那么：

$$
r=1.1
$$

OK——在范围内，正常受益。

但如果更新成：

$$
P=40\%
$$

那么：

$$
r=2
$$

PPO 会通过 clipped objective 限制继续从这种过大的变化中获益——**超过 1.2 的部分，梯度被截断，不再鼓励继续涨**。

所以 Clip 可以粗略理解成：

```text
模型：

「这个动作很好！
概率从20%调到90%！」

PPO：

「你先别激动。」
```

这其实就是 PPO 最核心的设计哲学：

$$
\boxed{
稳定地改进 Policy
}
$$

而不是：

$$
\boxed{
疯狂地改进 Policy
}
$$

---

## 五、PPO 中为什么有 Critic？——Advantage 从哪来

这里就和上一篇的：

$$
V(s)
$$

接上了。

PPO 通常采用 Actor-Critic：

```text
Actor
  ↓
πθ(a|s)
负责：我要做什么？

Critic
  ↓
Vφ(s)
负责：现在这个状态值多少？
```

Critic 帮助估计 Advantage。

比如：

$$
A_t
\approx
R_t+\gamma V(S_{t+1})-V(S_t)
$$

**注意**：这里的 $R_t+\gamma V(S_{t+1})-V(S_t)$ 正是上一篇 TD Error 的形式！Critic 输出 $V$，我们就能用「现实奖励 + 下一状态价值估计 − 当前状态价值估计」来近似 Advantage，不需要等整局结束。

更实际的 PPO 中通常会结合 **GAE（Generalized Advantage Estimation）**：

$$
\hat A_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
$$

其中：

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

### GAE 推导直觉

把 $\delta_t$ 展开：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

一步的 Advantage 估计：

$$
\hat A_t^{(1)} = \delta_t
$$

两步的（把下一步的 TD Error 也加权进来）：

$$
\hat A_t^{(2)} = \delta_t + \gamma\lambda\,\delta_{t+1}
$$

无限展开：

$$
\hat A_t = \delta_t + \gamma\lambda\,\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

- $\lambda=0$：只看一步 TD（高偏差低方差）
- $\lambda=1$：退化成蒙特卡洛式（低偏差高方差）
- $\lambda$ 介于之间：**在偏差和方差之间取平衡**

所以你会发现：

**上一篇里面讲 TD Error 不是白讲的。**

它一路延伸到了 PPO 的 GAE。

---

## 六、PPO 用到 LLM 上是什么样？

到了 LLM：

$$
State
$$

可以理解成 Prompt + 已生成 token；

$$
Action
$$

就是下一个 token；

$$
Policy
$$

就是语言模型：

$$
\pi_\theta(y|x)
$$

例如：

```text
Prompt
  ↓
LLM 生成 Response
  ↓
Reward Model / Verifier
  ↓
Reward
  ↓
计算 Advantage
  ↓
PPO 更新 LLM
```

但 PPO 用在 LLM 上有个明显的问题：

> **Critic / Value Model 很重。**

尤其模型参数量很大时，多维护一个 Value Model 会增加训练成本——本来 LLM 就大，还要再养一个和它差不多大的 Critic。

这就引出了 GRPO。

---

## 七、GRPO 的核心思想：不要 Critic，让同组回答互相比

GRPO：

$$
\boxed{
Group\ Relative\ Policy\ Optimization
}
$$

它和 PPO 一个非常重要的区别是：

> **不用额外的 Value Model 来估计 baseline，而是在同一个问题的多个回答之间进行相对比较。**

这一点也与公开的 GRPO 介绍一致：GRPO 对同一个 prompt 生成多个 completions，根据各自 reward 在组内做归一化，从而构造 Advantage，省掉 PPO 中用于 Advantage 估计的 Value Model。([哔哩哔哩][2])

例如给一个问题：

```text
2 + 3 = ?
```

模型采样 4 个回答：

```text
Response 1 → Reward = 1
Response 2 → Reward = 0
Response 3 → Reward = 1
Response 4 → Reward = 0
```

然后计算组内均值：

$$
\mu=\frac{1}{G}\sum_i r_i
$$

和组内标准差：

$$
\sigma
=
\sqrt{
\frac1G
\sum_i(r_i-\mu)^2
}
$$

最后做组内归一化（Relative Advantage）：

$$
A_i
=
\frac{r_i-\mu}{\sigma}
$$

也就是：

> 不问「这个回答绝对价值是多少」，而问「这个回答相比同组其他回答好不好」。

这就是 **Group Relative**。

### 数字例子

4 个回答的奖励是 $[1,0,1,0]$：

$$
\mu = \frac{1+0+1+0}{4} = 0.5
$$

$$
\sigma = \sqrt{\frac{(1-0.5)^2+(0-0.5)^2+(1-0.5)^2+(0-0.5)^2}{4}} = 0.5
$$

于是：

```text
A（r=1）：(1-0.5)/0.5 = +1.0  → 比同组好，提高概率
B（r=0）：(0-0.5)/0.5 = -1.0  → 比同组差，降低概率
C（r=1）：+1.0                  → 提高
D（r=0）：-1.0                  → 降低
```

**PPO 用 Critic 估计「正常水平」，GRPO 直接让同组的回答互相当参照物**——省掉了整个 Value Model。

---

## 八、PPO vs GRPO 最关键区别

|               | PPO      | GRPO           |
| ------------- | -------- | -------------- |
| Policy Model  | ✓        | ✓              |
| Reward        | ✓        | ✓              |
| Advantage     | ✓        | ✓              |
| Value/Critic  | **需要** | **通常不需要** |
| Baseline 来源 | $V(s)$   | 同组回答       |
| 多回答采样    | 不要求   | **核心设计**   |
| Clip          | ✓        | ✓              |
| KL 约束       | 常用     | 常用           |
| LLM 训练成本  | 较高     | 相对降低       |

所以可以非常粗暴地理解：

#### PPO

```text
这个回答 Reward = 8

Critic：
「正常情况下大概应该得 5 分」

Advantage：
8 - 5 = +3

→ 增加这个回答的概率
```

#### GRPO

```text
同一道题生成：

A：8
B：4
C：6
D：2

组内平均 ≈ 5

A：比同组平均好
D：比同组平均差

→ 增加 A 类回答概率
→ 降低 D 类回答概率
```

---

## 九、为什么 GRPO 特别适合数学/代码/推理？

因为这类任务经常可以比较方便地设计 Reward。

比如数学：

```text
Prompt：
证明/计算一道数学题

模型生成：

Response 1 → 正确 → 1
Response 2 → 错误 → 0
Response 3 → 正确 → 1
Response 4 → 错误 → 0
Response 5 → 错误 → 0
Response 6 → 正确 → 1
```

那么同一个 Prompt 下就天然形成：

$$
\text{Group}
$$

然后进行组内比较。

所以 GRPO 的思想非常适合：

> **一个问题 → 多次采样 → 可以验证答案 → 组内相对评价。**

---

## 十、回到小游戏：PPO 和 GRPO 在同一个例子里

还是上一篇那个小游戏：

```text
S0 → S1 → S2 → 宝箱
                ↘ 陷阱
```

### PPO

> Actor-Critic 可以，但 Policy 每次不要变化太大。

所以使用概率比值：

$$
\frac{\pi_{new}}{\pi_{old}}
$$

以及 Clip 等机制约束更新——**每步更新都限制在 [1-ε, 1+ε] 范围内**。

### GRPO

> LLM 的 Critic 太贵了。

那同一道题生成：

$$
G
$$

个答案。

通过组内 Reward：

$$
R_1,R_2,\cdots,R_G
$$

计算 Relative Advantage：

$$
A_i = \frac{R_i-\mu}{\sigma}
$$

于是：

$$
\boxed{
不再依赖单独的 Value/Critic 模型
}
$$

---

## 十一、把整条线真正串起来

如果你是为了**学 RL，而不是单纯看懂视频**，我建议脑子里保留下面这一条主线：

```text
强化学习
│
├── Agent 与 Environment 交互
│
├── State / Action / Reward
│
├── Return
│
├── Value
│     ├── V(s)
│     └── Q(s,a)
│
├── Bellman Equation
│
├── Value-based
│     ├── MC
│     ├── TD
│     ├── SARSA
│     └── Q-Learning
│
└── Policy-based
      │
      ├── Policy Gradient
      │
      ├── Advantage
      │
      └── Actor-Critic
             ↓
            PPO
             │
             ├── Importance Sampling
             ├── Probability Ratio
             ├── Clip
             ├── Critic / Value
             └── GAE
                    ↓
                   GRPO
                    │
                    ├── 一个 Prompt 多次采样
                    ├── Group Reward
                    ├── Relative Advantage
                    ├── 去掉 Value Model
                    └── PPO-style Clip / KL
```

这里面我认为**最值得吃透的不是 PPO 最后那条大公式**，而是下面这 6 个概念：

$$
\boxed{
Return
\rightarrow
V/Q
\rightarrow
TD
\rightarrow
Policy\ Gradient
\rightarrow
Advantage
\rightarrow
PPO
}
$$

这条链真正懂了以后，GRPO 其实非常自然：

> **PPO：用 Critic 告诉我「正常水平」是多少。**
> **GRPO：我不养 Critic 了，同一道题多生成几个答案，让它们互相当参照物。**

这也是为什么这两期适合连续看：03 解决「RL 的 Value、Policy、TD 到底是什么」，04 则把这些概念组合成 PPO，再解释 GRPO 为什么能在 LLM 场景下省掉 Value Model。([哔哩哔哩][3])

### 算法一句话记忆表

| 方法                | 你脑子里应该出现的一句话                          |
| ------------------- | ------------------------------------------------- |
| **MC**              | **等整局结束，用真实结果复盘**                    |
| **TD**              | **不等结束，用下一状态的估计更新当前状态**        |
| **SARSA**           | **按照我实际执行的 Policy 学 Q**                  |
| **Q-Learning**      | **按照最优动作学 Q，不管我实际探索了什么**        |
| **DQN**             | **用神经网络近似 Q**                              |
| **Policy Gradient** | **不学 Q 再选动作了，直接学习动作概率**           |
| **Actor-Critic**    | **Actor 做动作，Critic 判断这个动作是否超出预期** |
| **PPO**             | **Policy 可以变好，但每次别变得太猛**             |
| **GRPO**            | **不要 Critic，同一道题生成多个答案互相比**       |

而真正的知识依赖关系，我建议记成：

$$
\boxed{
MC
\rightarrow TD
\rightarrow
\begin{cases}
SARSA\\
Q\text{-Learning}\rightarrow DQN
\end{cases}
}
$$

然后另一条线：

$$
\boxed{
Policy\ Gradient
\rightarrow
Actor\text{-}Critic
\rightarrow
PPO
\rightarrow
GRPO
}
$$

两条线之间不是完全割裂的。**Value、TD Error、Advantage 这些思想最后都会重新出现在 Actor-Critic/PPO 里面。**

尤其是你之后继续看 PPO/GRPO，我建议重点搞懂 **Advantage 到底怎么从「TD Error」一步步发展到 GAE**。这个地方一旦搞懂，PPO 那个看起来很吓人的 Loss 就会突然变得非常合理。

> 顺带一提：DeepSeek-R1 这类推理模型的「深度思考」训练正是 GRPO 在 LLM 上的典型应用，可参考 [《推理模型深度思考原理笔记》](https://titourpast.github.io/blog/posts/%E6%8E%A8%E7%90%86%E6%A8%A1%E5%9E%8B%E6%B7%B1%E5%BA%A6%E6%80%9D%E8%80%83%E5%8E%9F%E7%90%86%E7%AC%94%E8%AE%B0/)。

## 参考资料

- **视频 04**：〖大白话 04〗一文理清 PPO 和 GRPO 算法流程 ｜ [哔哩哔哩 BV15cZYYvEhz](https://www.bilibili.com/video/BV15cZYYvEhz/?utm_source=chatgpt.com)
- **GRPO 算法介绍**：LLM 技术系列之 GRPO 算法介绍 ｜ [哔哩哔哩专栏](https://www.bilibili.com/opus/1045560825596608518?utm_source=chatgpt.com)

[1]: https://www.bilibili.com/video/BV15cZYYvEhz/?utm_source=chatgpt.com "〖大白话04〗一文理清PPO和GRPO算法流程_哔哩哔哩_bilibili"
[2]: https://www.bilibili.com/opus/1045560825596608518?utm_source=chatgpt.com "LLM技术系列之-GRPO算法介绍 - 哔哩哔哩"
[3]: https://www.bilibili.com/opus/1109918805995290672?utm_source=chatgpt.com "〖大白话03〗一文理清强化学习RL基本原理 | 原理图解+公式推导 - 哔哩哔哩"
