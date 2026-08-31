---
author: 芙芙
pubDatetime: 2026-08-31
title: 从 PPO 到 DPO、GRPO：大模型强化学习到底在优化什么？
featured: false
draft: false
tags:
  - 强化学习
  - PPO
  - DPO
  - GRPO
  - 学习笔记
category: 强化学习
description: 整理自 B 站「大白话04」视频并补充 DPO 内容：从 Policy Gradient 与 Advantage 讲起，用直观例子拆解 PPO 的 Clip 与 GAE、DPO 的直接偏好优化，以及 GRPO 如何用组内相对 Advantage 省掉 Critic，串起大模型后训练的三种主流强化学习方法。
---

> 本文主要整理自 B 站视频《【大白话04】一文理清强化学习 PPO 和 GRPO 算法流程 | 原理图解》([哔哩哔哩][1])，并补充了 DPO 的内容和更多直观例子。
>
> 上一篇 [《强化学习基础：从 MDP、Bellman 方程到 Monte Carlo、TD 与 Q-Learning》](https://titourpast.github.io/blog/posts/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E4%BB%8E-mdpbellman-%E6%96%B9%E7%A8%8B%E5%88%B0-monte-carlotd-%E4%B8%8E-q-learning/) 从基本框架出发，讲到了 Monte Carlo、TD、SARSA、Q-Learning 与 Policy Gradient。这一篇继续沿着 Policy-based 方法往下走，重点理解三个在 LLM 后训练中非常常见的方法：
>
> - PPO：Proximal Policy Optimization
> - DPO：Direct Preference Optimization
> - GRPO：Group Relative Policy Optimization
>
> 与其直接背三个复杂的 Loss，不如先搞清楚它们分别在解决什么问题。

## 一、从 Policy Gradient 到 Advantage：让好动作的概率变大

### 1.1 Policy-based：直接学策略，不绕价值这一层

上一篇文章里的 Q-Learning、DQN 都属于 Value-based 方法：先学习动作价值 $Q(s,a)$，再选择 $a=\arg\max_a Q(s,a)$。而 Policy-based 方法不再绕「给动作打分」这一层，而是直接学习策略 $\pi_\theta(a|s)$——也就是「在状态 $s$ 下，动作 $a$ 应该以多大的概率被选中」。

比如一个简单的战斗场景「前方有敌人」，一个策略可能给出：攻击 0.6、逃跑 0.3、防御 0.1。我们的目标就是不断调整参数 $\theta$，让那些能带来高 Reward 的动作概率上升。把一条完整轨迹的累计回报记为 $R(\tau)$，目标函数可以写成：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\big[R(\tau)\big]
$$

然后对 $\theta$ 做梯度上升：

$$
\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)
$$

这就是 Policy Gradient 最基本的思想：**让好轨迹的概率变大，坏轨迹的概率变小。**

### 1.2 REINFORCE：奖励好的动作，惩罚差的动作

Policy Gradient 最经典的形式是 REINFORCE，它的梯度可以写成：

$$
\nabla_\theta J(\theta)=\mathbb E\big[G_t\nabla_\theta\log\pi_\theta(a_t|s_t)\big]
$$

乍一看有点吓人，直觉其实非常简单：$G_t$ 是从时刻 $t$ 开始这条轨迹的总回报。如果 $G_t>0$，说明这个动作最后带来了不错的结果，那就应该提高 $\pi_\theta(a_t|s_t)$ 以后被选中的概率；反过来如果 $G_t<0$，就降低它出现的概率。所以 REINFORCE 可以粗暴地理解为：

```text
做了一件事 → 看看最后结果 → 结果好就以后多做，结果差就以后少做
```

### 1.3 但只有 Reward 还不够

考虑两个学生：A 平时考 60、这次考 80；B 平时考 95、这次考 80。两个人的 Reward 都是 80，但意义完全不同——对 A 是超常发挥，对 B 是发挥失常。这说明只看绝对 Reward 无法判断「这次行为比正常水平好多少」，于是需要引入 Advantage（优势函数）。

### 1.4 Advantage：不是问「好不好」，而是问「比平均好多少」

Advantage 最基本的定义是：

$$
A(s,a)=Q(s,a)-V(s)
$$

其中 $Q(s,a)$ 是「在状态 $s$ 执行动作 $a$，未来大概能拿到多少奖励」，$V(s)$ 是「在状态 $s$ 下按正常策略行动，平均能拿到多少奖励」。两者一减：$A(s,a)>0$ 说明这个动作比平均水平好，$A(s,a)<0$ 说明比平均水平差。

举个例子：假设 $V(s)=5$，即这个状态下一般来说能拿 5 分。动作 A 的 $Q(s,A)=10$，于是 $A(s,A)=10-5=5$，非常不错；动作 B 的 $Q(s,B)=2$，于是 $A(s,B)=2-5=-3$——虽然 B 可能还是带来了正奖励，但相比正常水平它其实是个比较差的动作。这就是 Advantage 的意义：**它不是问「好不好」，而是问「比平均好多少」。**

### 1.5 Actor-Critic：一个负责做，一个负责评价

那么 $V(s)$ 从哪里来？这就产生了 Actor-Critic：Actor 是演员，负责执行动作（策略 $\pi_\theta(a|s)$）；Critic 是评论家，负责评价动作（价值 $V_\phi(s)$）。整个过程可以这样看：

```text
Actor 觉得应该向左走 → 环境给 +5 → Critic 说这个状态正常水平只有 +2
→ Advantage = 5 - 2 = +3 → Actor 提高以后选择向左的概率
```

于是 **Actor 学策略、Critic 学价值**，这也成为后面 PPO 的重要基础。

## 二、PPO：近端策略优化，一次别改太狠

### 2.1 为什么还需要 PPO？

到这里看起来已经能训练了：采样 → 算 Reward → 算 Advantage → 更新 Policy。但普通 Policy Gradient 有一个严重问题：**一次更新可能走得太远**。假设当前策略是攻击 0.50 / 逃跑 0.50，某次训练发现攻击获得了很高奖励，如果梯度更新太猛，策略可能瞬间变成攻击 0.99 / 逃跑 0.01。而一次采样得到的 Reward 本来就带噪声，因为一次好结果就把整个策略彻底改掉，很容易导致训练崩溃。

所以我们希望：**可以学习，但每次别学太猛**。这就是 PPO（Proximal Policy Optimization，近端策略优化）——其中 Proximal 的含义就是：更新后的策略不要离旧策略太远。

### 2.2 概率比：新策略到底变化了多少

要限制「更新幅度」，先得度量「变化了多少」。设旧策略为 $\pi_{\theta_{old}}$、新策略为 $\pi_\theta$，定义概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

它衡量新旧策略对当前动作态度的变化：$r_t=1$ 表示策略没变；$r_t=1.2$ 表示新策略选这个动作的概率是旧策略的 1.2 倍；$r_t=0.7$ 表示降到了原来的 70%。

这里还有一个容易困惑的点：我们的数据 $(s_t,a_t)$ 通常是用旧策略 $\pi_{\theta_{old}}$ 采样得到的，但训练时优化的是 $\pi_\theta$，两者分布并不一致。这个概率比（Importance Sampling Ratio）可以在一定程度上修正这种差异，因此 PPO 能重复使用旧策略采集的数据，而不必参数每更新一次就重新和环境交互采样。

### 2.3 PPO 最核心的 Clip

对于好动作（$A_t>0$），我们希望 $\pi_\theta$ 上升、即 $r_t>1$；但能不能无限增大？不能。PPO 引入 clip 把 $r_t$ 限制在 $[1-\epsilon,1+\epsilon]$——例如 $\epsilon=0.2$ 时就是 $[0.8,1.2]$：

$$
\operatorname{clip}\big(r_t,1-\epsilon,1+\epsilon\big)
$$

举一个直观的例子：假设 $A_t>0$、原策略 $\pi_{old}(a|s)=0.5$。新策略更新到 0.55 时，$r=0.55/0.5=1.1$，在允许范围内，可以继续鼓励；但如果更新到 0.9，$r=0.9/0.5=1.8$，明显过头——PPO 会通过 Clip 告诉模型：可以奖励它，但别一下从 0.5 干到 0.9。

### 2.4 PPO 的目标函数

PPO 最经典的 Clipped Objective：

$$
L^{CLIP}(\theta)=\mathbb E_t\left[\min\left(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right]
$$

不用背，拆开看就行：它比较「不限制的 $r_t\hat A_t$」和「clip 之后的版本」，取更小的那个。当 $A_t>0$ 时，策略可以提高 $r_t$，但超过 $1+\epsilon$ 后收益不再增加——好动作提高概率可以，疯狂提高不行；当 $A_t<0$ 时，策略可以降低 $r_t$，但低于 $1-\epsilon$ 后 PPO 不再鼓励继续降——坏动作降低概率可以，直接把概率干成 0 不行。

所以 PPO 可以用一句话记忆：

> **好动作提高概率，坏动作降低概率，但每一次都别改太狠。**

## 三、GAE 与 PPO 完整流程

### 3.1 PPO 里的 Critic 在干什么

前面还有一个问题没解决：$\hat A_t$ 到底怎么算？PPO 一般会训练一个 Critic $V_\phi(s_t)$ 来估计当前状态的价值，再利用 Reward 和 Value 构造 Advantage。一种常见做法就是 GAE（Generalized Advantage Estimation）。

### 3.2 从 TD Error 到 GAE

先看 TD Error：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

它表示「实际观察到的新信息与之前预测之间的差距」：$\delta_t>0$ 说明事情发展得比 Critic 预计的好，那么这个动作很可能值得鼓励。

GAE 把多个未来 TD Error 组合起来：

$$
\hat A_t=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

$\lambda$ 负责平衡 Bias 和 Variance。具体感受一下两种极端：$\lambda=0$ 时 $\hat A_t=\delta_t$，只用「这一步」的 TD Error，Bias 大、容易被单步噪声误导；$\lambda=1$ 时把之后所有 TD Error 都按 $(\gamma\lambda)^l$ 累加，接近 Monte Carlo，Variance 大、要等完整轨迹结束才知道结果；取中间值（比如实践中常用的 $\lambda=0.95$）就是两者的折中。

### 3.3 把 PPO 整体流程串起来

PPO 大致分五步：

1. **旧策略采样**：用 $\pi_{\theta_{old}}$ 与环境交互，得到 $(s_t,a_t,r_t,s_{t+1})$；
2. **Critic 估值**：$V_\phi(s_t)$ 预测每个状态的价值；
3. **计算 Advantage**：例如用 GAE 得到 $\hat A_t$；
4. **Actor 更新**：计算概率比 $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$，用 PPO Clip Loss 更新；
5. **Critic 更新**：让 $V_\phi(s)$ 更加接近真实 Return。

整个过程可以浓缩成一条流水线：

```text
旧 Actor → 与环境交互 → Trajectory → Reward → Critic → Value → GAE → Advantage → PPO Clip → 更新 Actor
```
## 四、进入大语言模型：State、Trajectory 与 Reward Model

### 4.1 到大语言模型里，State 和 Action 是什么？

传统 RL 里，State 是游戏画面、Action 是左/右/跳/攻击；而大语言模型里，State 可以理解成「Prompt + 当前已经生成的 Token」，Action 就是「下一个 Token」。比如给定「中国的首都是」，模型从词表（北京、上海、广州、南京……）里选一个。模型实际输出的是 $P(\text{token}|\text{context})$，本质上就是 $\pi_\theta(a|s)$。所以：

> **语言模型天然就是一个 Policy。**

一整段回答就是一条 Trajectory。比如 Prompt 是「1 + 1 等于多少？」，模型生成「1 + 1 = 2。」，RL 视角下可以看成：State 1 = Prompt，Action 1 = "1"，State 2 = Prompt + "1"，Action 2 = "+"，……一直生成到 EOS。所以整个 Completion $y=(y_1,y_2,\cdots,y_T)$ 就是一条轨迹。

### 4.2 问题来了：LLM 的 Reward 从哪里来？

游戏里 Reward 很自然（赢 +1、输 -1；走到目标 +10、摔倒 -10），但大语言模型呢？模型生成「巴黎是德国的首都。」，它究竟应该拿 $-0.7$ 还是 $-2.3$？没有天然的答案。于是 **Reward Model 登场**：$R_\phi(x,y)$ 输入 Prompt + Response，输出一个 Reward 标量。比如「巴黎是法国首都。」得 $4.3$，「巴黎是德国首都。」得 $-2.1$，PPO 就可以根据这个分数更新语言模型。

### 4.3 Reward Model 怎么训练？

一个经典做法不是让人类直接打分（「7.345 分」这种绝对分数很难打准），而是让人类比较：「回答 A 和回答 B 哪个更好？」。收集偏好数据 $(x,y_w,y_l)$，其中 $y_w$ 是 preferred / winner，$y_l$ 是 rejected / loser。Reward Model 希望满足 $R(x,y_w)>R(x,y_l)$，常见用 Bradley-Terry 风格的概率模型：

$$
P(y_w\succ y_l)=\sigma\big(R(x,y_w)-R(x,y_l)\big)
$$

然后训练 Reward Model。

### 4.4 RLHF 中经典 PPO 流程

于是经典 RLHF 逐渐形成：Pretraining → Base LLM → SFT → SFT Model → 偏好数据 → Reward Model → PPO → Aligned LLM。展开来看就是：Prompt 输入 Policy Model → 模型生成 Response → Reward Model $R(x,y)$ 打分 → Critic $V(s)$ 估计 Value → 计算 Advantage → 用 $L^{CLIP}$ 更新 Policy。

### 4.5 为什么还需要 Reference Model？

如果 Reward Model 特别喜欢「答案越长越好」，模型可能逐渐发现：疯狂输出废话就能拿高分，这就是 **Reward Hacking**；或者策略训练太猛，模型直接偏离原本的语言能力。因此通常保留一个 Reference Model $\pi_{ref}$，限制 $\pi_\theta$ 不要离它太远，方法就是加 KL Penalty：

$$
R_{total}=R_{RM}-\beta D_{KL}\big(\pi_\theta\|\pi_{ref}\big)
$$

直觉是：Reward Model 说「你这个答案不错，+10」，但 Reference Model 发现「你现在说话方式和原来差太远了」，KL 扣 3 分，最终 $10-3=7$。所以 KL 的作用之一是**防止模型为了追逐 Reward，把原本已经学好的语言能力全部丢掉**。

### 4.6 PPO 用在 LLM 上为什么这么重？

大模型 PPO 需要同时维护四个大模型：Policy / Actor（真正要训练的）、Critic / Value Model（估计 $V(s)$）、Reference Model（算 KL）、Reward Model（给回答评分）。直观感受一下：训练一个 7B 的 Actor，通常还要再加载一个同量级的 Critic、一个 Reference Model 和一个 Reward Model，显存与算力开销是单模型训练的很多倍。

更麻烦的是，LLM 的 Value 很难估计：模型解数学题推到一半，Critic 要预测「推理到这里以后，最终答对的期望 Reward 是多少」。一个 Token「因此」到底让最终解题成功概率增加了 $0.02$ 还是 $0.0001$，实际上很难有准确答案。而数学、代码等推理任务的 Reward 又经常只在最后才能确定（最终答案正确 1、错误 0）。**长序列 + 稀疏 Reward，使 Critic 的学习更加困难**，这也是后来 GRPO 等方法的动机之一。

## 五、DPO：直接偏好优化

### 5.1 第一条改进路线：DPO

既然 PPO 这么复杂，问题就来了：我们真的一定需要 Reward Model + PPO 吗？假设手里已经有人类偏好数据（Prompt + 回答 A 好 / 回答 B 差，即 $(x,y_w,y_l)$）。PPO 的路线是 偏好数据 → 训练 Reward Model → Reward → PPO → Policy；而 DPO 的思想是：**既然最终目的就是让好回答概率上升、坏回答概率下降，那能不能直接学？**

比如 Prompt 是「法国首都是哪里？」，Winner 是「法国首都是巴黎。」，Loser 是「法国首都是柏林。」。我们希望训练后 $\pi_\theta(y_w|x)$ 上升、$\pi_\theta(y_l|x)$ 下降：不再显式训练 Reward Model，也不再进行 PPO rollout，直接从已有偏好里学。

### 5.2 DPO 为什么还需要 Reference Model？

如果只是无脑提高 $\pi(y_w)$、降低 $\pi(y_l)$，还是可能让模型改得太狠。所以 DPO 依然会把当前 Policy 和 Reference Policy 比较，核心关注：

$$
\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

意思是「相比原来的模型，现在到底把这个回答的概率提高了多少」。对于 Winner，希望这个比值更大；对于 Loser，则希望相对更小。

### 5.3 DPO Loss

经典 DPO Objective 可以写成：

$$
L_{DPO}=-\mathbb E\left[\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]
$$

公式很长，但核心就一句：**相对于 Reference Model，让 Winner 的概率提升幅度超过 Loser**。记住「Winner ↑、Loser ↓、同时不能离 Reference 太远」就够了。

再给一个示意数字：假设 Reference 模型对 Winner 和 Loser 的概率分别是 $0.4$ 和 $0.3$，两者差不多；训练后我们希望 $\pi_\theta$ 让 Winner 相对提升更多，比如变成 $0.55$ 和 $0.2$。于是上面括号里的量从「接近 $0$」变成「明显大于 $0$」——这正是 DPO 在推动的事情。

### 5.4 PPO 和 DPO 最大区别

PPO 是 Online / RL 风格：模型自己生成 → 得到 Reward → 更新 → 再生成，训练过程中不断产生新的行为。DPO 是 Offline Preference Learning 风格：使用已经收集好的 $(x,y_w,y_l)$ 直接学习。两者的区别可以看这张表：

| PPO | DPO |
|---|---|
| 需要 Reward | 需要 Preference Pair |
| 通常需要 Reward Model | 不需要显式 Reward Model |
| 需要 Rollout | 通常基于已有数据 |
| 有 Critic | 不需要 Critic |
| 训练较复杂 | 相对简单 |
| 可以继续在线探索 | 主要受已有偏好数据限制 |

### 5.5 DPO 的代价是什么？

DPO 简单了很多，但也牺牲了一件重要的事情：**在线探索**。如果训练数据里从来没有一个特别优秀的新解法，DPO 只能从已有的 Winner / Loser 中学习，不会像 RL 那样：模型自己生成新答案 → 环境打分 → 发现一个以前没有出现过的新策略 → 继续强化。这对数学推理、代码生成、Agent、可验证任务尤其重要。

因此在需要模型不断「探索 → 获得 Reward → 改进」的场景里，RL 仍然非常有价值——于是来到 GRPO。
## 六、GRPO：组相对策略优化

### 6.1 核心思想：如果我们把 Critic 删了呢？

PPO 最重的组件之一是 Critic $V_\phi(s)$。GRPO 问了一个非常直接的问题：**能不能不预测「这条回答绝对值多少钱」，而直接比较「这一批回答谁更好」？** 答案是可以。

核心操作是同一道题多做几遍。假设 Prompt 是「求方程 $x^2-5x+6=0$」，让模型一次生成 $G=5$ 个回答：回答 1「$x=2,3$」、回答 2「$x=1,6$」、回答 3「$x=2,3$」、回答 4「推导错误」、回答 5「$x=2,3$」，然后用 Reward Function 打分：$R_1=1,R_2=0,R_3=1,R_4=0,R_5=1$。现在不需要 Critic 告诉我们「这个 State 的 Value = 0.7238」，直接看这个回答在**同一组里**表现怎么样——这就是 Group Relative。

### 6.2 GRPO 的 Advantage

假设一组 Reward 是 $R_1,R_2,\cdots,R_G$，先计算均值 $\mu_R=\frac{1}{G}\sum_{i=1}^{G}R_i$ 和标准差 $\sigma_R$，然后：

$$
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}
$$

这其实就是 Z-score 标准化。举个例子：$R=[2,4,6]$，均值 $\mu=4$。只看 $R_i-\mu$：回答 1 是 $2-4=-2$（比组平均差，抑制），回答 2 是 $4-4=0$（平均水平，基本不动），回答 3 是 $6-4=+2$（比平均好，强化）。这就是 GRPO 最关键的思想。

### 6.3 为什么相对比较比绝对估值容易？

回到学生考试：问「张三数学水平到底是多少？」，你可能很难精确说出 82.37；但如果同样一道题五个人一起做（A 完全正确、B 完全错误、C 正确、D 漏了一步、E 正确），你很容易判断 A 比 B 好。PPO 的 Critic 要回答「这个状态绝对值多少」，较难；GRPO 只要回答「同一道题的几份答案谁更好」，相对容易。对 LLM 尤其如此——同一个 Prompt 本来就可以通过 Sampling 轻松生成很多个 Completion。

### 6.4 GRPO 并不是把 PPO 全扔了

GRPO 并不是和 PPO 完全没关系，它保留了 PPO 的很多思想：依然有概率比 $r_{i,t}(\theta)=\pi_\theta(o_{i,t}|q,o_{i,<t})/\pi_{old}(o_{i,t}|q,o_{i,<t})$，依然用 clip 限制更新范围。最大的改动是 **Advantage 的来源变了**：PPO 是 Critic → GAE → Advantage，GRPO 是 Group Rewards → Group Relative Advantage → PPO Clip。

核心差别用一句话概括：PPO 是「跟我预计你应该达到的水平比较」，GRPO 是「跟这一组同学的平均水平比较」。

### 6.5 GRPO Objective

一个简化的 GRPO Objective 可以写成：

$$
J_{GRPO}(\theta)=\mathbb E\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_t\left(\min\big(r_{i,t}A_i,\ \operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i\big)-\beta D_{KL}\right)\right]
$$

看起来复杂，但拆开只有四部分：**Group Sampling**（同一个 Prompt $q$ 生成 $o_1,o_2,\cdots,o_G$）、**Reward**（得到 $R_1,R_2,\cdots,R_G$）、**Relative Advantage**（$A_i=(R_i-\mu)/\sigma$）、**PPO-style Update**（$\min(rA,\ clip(r)A)$ 加 KL 约束）。所以 GRPO 仍然继承了 PPO 风格的 clipped policy objective 和 KL 约束，只是用组内相对奖励替换了 Critic/GAE 产生的优势。

### 6.6 GRPO 完整流程

GRPO 的完整流程很直观：采样 Prompt（比如「求 $24\times17$」）→ 生成 $G$ 个回答 → 计算 Reward（数学题可以直接验证最终答案，正确 1、错误 0）→ 计算 Group Advantage $A_i=(R_i-\operatorname{mean}(R))/\operatorname{std}(R)$ → 计算概率比 $r_{i,t}=\pi_\theta/\pi_{old}$ → clip 限制 $r_{i,t}$ 不要变化太猛 → 用 KL 约束 $\pi_\theta$ 不要过度偏离 $\pi_{ref}$ → 更新参数。最终效果就是：好于组平均的回答提高概率，差于组平均的回答降低概率。

### 6.7 为什么 GRPO 特别适合数学和代码？

因为这些任务的 **Reward 容易验证**：数学题可以直接把最终答案和 Ground Truth 比较（比如答案是不是 42），代码可以运行 Test Cases（全部通过 1、失败 0）。于是不一定需要复杂的人类偏好 Reward Model，可以直接用 Rule-based Reward / Verifiable Reward。

这也是一个非常重要的变化：从「人觉得好」到「环境验证对」。传统 RLHF 是 回答 → 人类偏好 → Reward Model → Reward；而 Reasoning RL 很多时候可以变成 回答 → Verifier → Correct / Incorrect → Reward。比如数学检查最终答案、代码运行单元测试、Agent 检查任务是否真正完成、Search 检查是否找到正确答案。这类 Verifiable Reward 是近几年大模型推理强化学习非常重要的一条路线。

### 6.8 为什么 GRPO 不需要 Critic？以及它的局限

PPO 需要 Critic，是因为它要知道 $A=Q-V$——「这次表现相对于预计水平好多少」；GRPO 直接拿同一 Prompt 的其他回答作为 Baseline：$A_i\approx R_i-\operatorname{mean}(R)$。于是 **Critic 的 Baseline 变成了 Group Mean**，不再需要额外训练一个 Value Network，这正是 GRPO 能降低 LLM RL 训练资源开销的重要原因。

但 GRPO 也不是完美的，问题恰恰出在 Group 上。假设同一道题采样 $G=8$ 个回答、8 个全错，Reward 全是 $[0,0,0,0,0,0,0,0]$：此时 $\mu=0$、$\sigma=0$，所有回答的相对 Advantage 都没有有效区分——**整组完全没有差异，就没有什么可学习的相对信号**。全部答对也一样：$R=[1,1,1,1,1,1,1,1]$ 时大家都一样好，GRPO 不知道哪个回答应该被更强地强化。所以 GRPO 比较喜欢「有些答对、有些答错」或 Reward 有明显差异的 Prompt，Prompt 难度、Sampling Temperature、Group Size 都会影响效果。

Group Size 也有权衡：$G=2$ 时 Group Mean 很容易受到偶然性的影响；$G=16$ 时相对排名通常更稳定，但每个 Prompt 要生成 16 个 Completion，Rollout 成本也变成很多倍。所以 GRPO 并不是没有额外成本，而是把 PPO 中很重的 Critic Training 换成了更多 Rollout，本质上仍然存在计算权衡。

## 七、PPO、DPO、GRPO 的关系与总结

### 7.1 PPO、DPO、GRPO 到底是什么关系？

现在把三者真正串起来：

- **PPO**：根据 Reward 在线更新 Policy，并用 Clip 防止策略改变太快。需要 Policy、Critic、Reward、Reference 四个组件，强大、通用，但是复杂且昂贵。
- **DPO**：已经有 Winner / Loser，就直接让 Winner 概率高于 Loser。省掉 Reward Model、Critic、Online Rollout，简单、稳定，但是基本依赖已有 Preference Data。
- **GRPO**：保留在线 RL，但用同一 Prompt 下多个回答的相对 Reward 替代 Critic。省掉 Critic，但仍然需要 Rollout、Reward、Policy Update，尤其适合可以自动验证 Reward 的 Reasoning Task。

### 7.2 一个表格彻底区分 PPO / DPO / GRPO

| | PPO | DPO | GRPO |
|---|---|---|---|
| 是否属于在线 RL | 是 | 通常否 | 是 |
| 是否需要 Rollout | 是 | 通常不需要 | 是 |
| 是否需要 Critic | 是 | 否 | 否 |
| 是否显式需要 Reward | 是 | 否 | 是 |
| 数据形式 | Prompt + Rollout + Reward | Winner / Loser | Prompt + Group Rollouts + Reward |
| Advantage | Critic + GAE | 无传统 Advantage | Group Relative Advantage |
| Policy Constraint | PPO Clip / KL | Reference Model | PPO Clip / KL |
| 能否探索新策略 | 强 | 弱 | 强 |
| 训练成本 | 高 | 较低 | 中等 |
| 典型场景 | RLHF、通用 RL | Preference Alignment | Math / Code / Reasoning RL |

### 7.3 一个班级考试的比喻

如果还是容易混，可以用一个统一比喻：

- **PPO：老师 + 学生**。学生做题，老师（Critic）告诉他「按照你的水平，这道题正常应该拿 70 分」，结果学生拿了 90 分，Advantage = +20，于是强化这次做法。
- **DPO：只看两份答案**。老师不给绝对分，只告诉你「答案 A 比答案 B 好」，于是模型学习以后更像 A、少像 B。
- **GRPO：全班一起考试**。不请老师预测「这个学生理论上应该考多少」，直接让一组学生一起做：成绩 60、70、85、95、40，平均 70。95 明显高于平均 → 强烈奖励；85 高于平均 → 奖励；70 是平均 → 基本不动；40 低于平均 → 惩罚。

这就是 Group Relative Policy Optimization。

### 7.4 从 PPO 到 GRPO / 从 DPO 到 GRPO，改了什么

GRPO 并不是把 PPO 从头到尾重写，真正改变最大的地方其实是 **Advantage Estimation**：PPO 是 $\text{Critic}\rightarrow\text{GAE}\rightarrow\text{Advantage}$，GRPO 是 $\text{Group Reward}\rightarrow\text{Normalization}\rightarrow\text{Advantage}$；而 Probability Ratio、Clip、KL Constraint、Policy Update 这些 PPO 风格的思想依然存在。所以以后再看到 GRPO Loss，第一反应不应该是「又来了一个全新的强化学习算法」，而应该是「它主要把 PPO 的 Critic-based Advantage 换成了 Group-based Advantage」。

DPO 和 GRPO 虽然都去掉了 Critic，但思想完全不同：DPO 是「数据已经在那里了，Winner / Loser 直接学」，更偏从已有偏好中学习；GRPO 是「模型自己不断产生新回答 → Reward → Learn → 再生成」，更偏通过环境反馈不断探索。这也是强化学习重新在 LLM Reasoning 中受到大量关注的原因。

### 7.5 最后把整个发展路径串起来

从上一篇 RL 基础开始，整个发展路径是这样的：

```text
MDP
│
├── Value-based
│   ├── Monte Carlo
│   ├── TD
│   ├── SARSA
│   ├── Q-Learning
│   └── DQN
│
└── Policy-based
    ├── Policy Gradient
    ├── REINFORCE
    └── Actor-Critic
            │
            ▼
           PPO
            ├───────────────┐
            ▼               ▼
           DPO             GRPO
            │               │
     Preference Pair    Group Rollout
            │               │
      Direct Update     Relative Reward
```

换一个角度，也可以看成一串「问题 → 解法」的递进：

```text
Policy Gradient
      │  问题：训练不稳定
      ▼
     PPO
      │  问题：LLM 上太重
      ├──────────────────────┐
      ▼                      ▼
     DPO                    GRPO
      │                      │
不要在线 RL          保留在线 RL
直接学偏好           但删除 Critic
```

### 7.6 最值得记住的公式

**Policy Ratio**：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

含义：新策略相对于旧策略，把这个 Action 的概率改了多少。

**PPO Clip**：

$$
L^{CLIP}=\mathbb E\left[\min\big(r_tA_t,\ clip(r_t,1-\epsilon,1+\epsilon)A_t\big)\right]
$$

含义：好动作增加概率，坏动作降低概率，但不要改太猛。

**Advantage**：

$$
A(s,a)=Q(s,a)-V(s)
$$

含义：这个动作相比正常水平到底好多少。

**GAE**：

$$
\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},\qquad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

**DPO**，核心看：

$$
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
$$

含义：相比 Reference Model，让 Winner 的提升超过 Loser。

**GRPO Advantage**：

$$
A_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)+\epsilon}
$$

含义：这个回答比同一 Prompt 下其他回答平均水平好多少。

### 7.7 最后总结

如果完全不记公式，只记下面这些也够用了：

- **Policy Gradient**：Reward 高的 Action，以后多做。
- **Advantage**：不看绝对 Reward，而看「比正常水平好多少」。
- **Actor-Critic**：Actor 负责行动，Critic 负责评价。
- **PPO**：好动作增加概率，坏动作降低概率，但 Clip 限制每一次更新不要过猛。
- **Reward Model**：把人类偏好转换成强化学习可以使用的 Reward。
- **Reference Model**：限制训练后的模型不要离原来的模型太远。
- **DPO**：已经知道哪个回答更好了，就不要绕 Reward Model + PPO，直接提高 Winner、降低 Loser。
- **GRPO**：不训练 Critic；同一道题生成多个答案，拿组内平均水平作为 Baseline。

因此整个路线真正值得记住的是：

$$
\boxed{
PPO:
Critic\ tells\ you\ how\ good
}
$$

$$
\boxed{
DPO:
Preference\ tells\ you\ which\ is\ better
}
$$

$$
\boxed{
GRPO:
The\ group\ tells\ you\ how\ good
}
$$

这也是 PPO、DPO 和 GRPO 三种方法最核心的区别。

## 参考资料

- Bilibili：《【大白话04】一文理清强化学习 PPO 和 GRPO 算法流程 | 原理图解》：https://www.bilibili.com/video/BV15cZYYvEhz/
- 知乎：《看完能和外婆解释的 PPO, DPO, GRPO 强化学习》：https://zhuanlan.zhihu.com/p/1984387073625593089
- Schulman et al., _Proximal Policy Optimization Algorithms_, 2017：https://arxiv.org/abs/1707.06347
- Rafailov et al., _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_, 2023：https://arxiv.org/abs/2305.18290
- Shao et al., _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_, 2024：https://arxiv.org/abs/2402.03300
- DeepSeek-AI, _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning_, 2025：https://arxiv.org/abs/2501.12948

[1]: https://www.bilibili.com/video/BV15cZYYvEhz/ "〖大白话04〗一文理清强化学习PPO和GRPO算法流程 | 原理图解_哔哩哔哩_bilibili"
