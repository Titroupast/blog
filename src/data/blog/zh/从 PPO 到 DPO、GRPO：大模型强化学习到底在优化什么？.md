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

## 一、从 Policy Gradient 到 REINFORCE

### 1.1 从 Policy Gradient 开始

在 Value-based 方法中，我们学习的是：

$$
Q(s,a)
$$

然后选择：

$$
a=\arg\max_aQ(s,a)
$$

而 Policy-based 方法不再绕这一层，而是直接学习：

$$
\pi_\theta(a|s)
$$

即：

> 在状态 $s$ 下，动作 $a$ 应该以多大的概率被选择。

比如：

```text
当前状态：前方有敌人

攻击：0.6
逃跑：0.3
防御：0.1
```

我们的目标就是不断调整参数 $\theta$，让那些能够带来高 Reward 的动作概率提高。

因此目标可以写成：

$$
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
[R(\tau)]
$$

然后进行梯度上升：

$$
\theta
\leftarrow
\theta+\alpha\nabla_\theta J(\theta)
$$

这就是 Policy Gradient 最基本的思想。


### 1.2 REINFORCE：奖励好的动作，惩罚差的动作

Policy Gradient 最经典的形式之一是 REINFORCE。

它的梯度可以写成：

$$
\nabla_\theta J(\theta)
=
\mathbb E
[
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
]
$$

乍一看有点吓人，其实直觉非常简单。

关注核心：

$$
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

如果：

$$
G_t>0
$$

说明：

> 这个动作最后带来了不错的结果。

那么应该：

$$
\uparrow \pi_\theta(a_t|s_t)
$$

也就是提高以后选择它的概率。

反过来，如果：

$$
G_t<0
$$

就降低这个动作以后出现的概率。

所以 REINFORCE 可以粗暴理解成：

```text
做了一件事
      ↓
看看最后结果
      ↓
结果好 → 以后多做
结果差 → 以后少做
```


## 二、Advantage 与 Actor-Critic

### 2.1 但只有 Reward 还不够

考虑下面两个学生。

学生 A：

```text
平时考 60
这次考 80
```

学生 B：

```text
平时考 95
这次考 80
```

虽然两个人这次都是：

$$
Reward=80
$$

但显然意义完全不同。

对于 A：

> 这次表现非常好。

对于 B：

> 这次表现反而很差。

也就是说，单纯看：

$$
Reward
$$

无法知道：

> 这次行为到底比“正常水平”好多少？

因此需要引入一个非常重要的东西：

$$
Advantage
$$

优势函数。


### 2.2 Advantage：不是问“好不好”，而是问“比平均好多少”

最基本的 Advantage 定义为：

$$
A(s,a)
=
Q(s,a)-V(s)
$$

其中：

$$
Q(s,a)
$$

表示：

> 在状态 $s$ 执行动作 $a$，未来大概能得到多少奖励。

而：

$$
V(s)
$$

表示：

> 在状态 $s$ 下，按照正常策略行动，平均能得到多少奖励。

所以：

$$
A(s,a)>0
$$

说明：

> 这个动作比平均水平好。

反之：

$$
A(s,a)<0
$$

说明：

> 这个动作比平均水平差。


### 2.3 举个例子

假设：

$$
V(s)=5
$$

也就是说在这个状态下，一般来说能获得：

$$
5
$$

分。

现在执行动作 A：

$$
Q(s,A)=10
$$

那么：

$$
A(s,A)=10-5=5
$$

说明这个动作非常不错。

执行动作 B：

$$
Q(s,B)=2
$$

那么：

$$
A(s,B)=2-5=-3
$$

说明：

> 虽然 B 可能还是带来了正奖励，但是相比正常水平，它其实是一个比较差的动作。

这就是 Advantage 的意义。


### 2.4 Actor-Critic：一个负责做，一个负责评价

问题又来了：

$$
V(s)
$$

从哪里来？

于是产生 Actor-Critic。

可以理解成：

```text
Actor：演员
负责执行动作

Critic：评论家
负责评价动作
```

Actor：

$$
\pi_\theta(a|s)
$$

Critic：

$$
V_\phi(s)
$$

整个过程像这样：

```text
Actor：
我觉得现在应该向左走。

        ↓

Environment：
你走了之后得到了 +5。

        ↓

Critic：
等等，在这个状态下正常水平只有 +2。

        ↓

Advantage：
5 - 2 = +3

        ↓

Actor：
看来向左确实不错，
以后可以提高选择向左的概率。
```

于是：

> Actor 学策略，Critic 学价值。

这成为 PPO 的重要基础。


## 三、PPO：近端策略优化（Clip 的核心思想）

### 3.1 为什么还需要 PPO？

到这里似乎已经能训练了：

```text
采样
↓
计算 Reward
↓
计算 Advantage
↓
更新 Policy
```

但普通 Policy Gradient 存在一个非常严重的问题：

> **一次更新可能走得太远。**

假设当前策略：

```text
攻击：0.50
逃跑：0.50
```

一次训练中发现：

```text
攻击获得了很高奖励
```

如果梯度更新太猛，可能瞬间变成：

```text
攻击：0.99
逃跑：0.01
```

这就很危险。

因为：

> 一次采样得到的 Reward 本来就可能存在噪声。

如果因为一次好结果就把整个策略彻底改掉，很容易导致训练崩溃。

所以我们希望：

> 可以学习，但每次不要学得太猛。

这就是：

### 3.2 PPO：Proximal Policy Optimization（近端策略优化）

其中 Proximal 可以理解成：

> 更新后的策略不要离旧策略太远。


### 3.3 PPO 的核心问题：新策略到底变化了多少？

假设旧策略为：

$$
\pi_{\theta_{old}}
$$

新策略为：

$$
\pi_\theta
$$

定义一个概率比：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{old}}(a_t|s_t)
}
$$

这个东西一般被称为：

> Probability Ratio / Importance Sampling Ratio。

它其实很好理解。

如果：

$$
r_t=1
$$

意味着：

```text
新策略概率
=
旧策略概率
```

策略没变化。


如果：

$$
r_t=1.2
$$

意味着：

> 新策略选择这个动作的概率，是旧策略的 1.2 倍。


如果：

$$
r_t=0.7
$$

意味着：

> 新策略把这个动作的概率降低到了原来的 70%。

所以：

$$
r_t
$$

本质上就在衡量：

> **新旧 Policy 对当前动作的态度发生了多大的变化。**


### 3.4 重要性采样为什么会出现在这里？

这里还有一个容易困惑的问题。

我们的数据：

$$
(s_t,a_t)
$$

通常是通过旧策略：

$$
\pi_{\theta_{old}}
$$

采样得到的。

但是训练的时候我们正在优化：

$$
\pi_\theta
$$

也就是说：

```text
数据：
旧模型产生的

目标：
更新新模型
```

这就产生分布不一致。

Importance Sampling 的概率比：

$$
\frac{\pi_\theta}{\pi_{\theta_{old}}}
$$

可以在一定程度上修正这种差异。

所以 PPO 可以重复使用旧策略采集到的数据，而不是参数每更新一下，就必须重新与环境交互采样一次。


### 3.5 PPO 最核心的 Clip

如果某个动作：

$$
A_t>0
$$

说明：

> 它比平均水平好。

我们自然希望：

$$
\pi_\theta(a_t|s_t)
\uparrow
$$

即：

$$
r_t>1
$$

但是能不能无限增大？

不能。

所以 PPO 引入：

$$
\operatorname{clip}
(
r_t,
1-\epsilon,
1+\epsilon
)
$$

假设：

$$
\epsilon=0.2
$$

那么：

$$
r_t
$$

会被限制在：

$$
[0.8,1.2]
$$

附近。


### 3.6 一个很直观的例子

假设：

$$
A_t>0
$$

说明这是一个好动作。

原策略：

$$
\pi_{old}(a|s)=0.5
$$

新策略更新到了：

$$
\pi_\theta(a|s)=0.55
$$

那么：

$$
r=
\frac{0.55}{0.5}
=
1.1
$$

很好，可以继续鼓励。

但如果：

$$
\pi_\theta(a|s)=0.9
$$

那么：

$$
r=
\frac{0.9}{0.5}
=
1.8
$$

这说明：

> 你是不是因为一次奖励就兴奋过头了？

于是 PPO 通过 Clip 告诉模型：

```text
可以奖励它，

但是别一下从 0.5
干到 0.9。
```


### 3.7 PPO 的目标函数

PPO 最经典的 Clipped Objective：

$$
L^{CLIP}(\theta)
=
\mathbb E_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)
\hat A_t
\right)
\right]
$$

先不用背。

把它拆成：

$$
r_t(\theta)\hat A_t
$$

和：

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
$$

然后取：

$$
\min
$$

核心目的就是：

> **只允许策略往正确方向调整有限的幅度。**


### 3.8 Advantage > 0 时发生什么？

如果：

$$
A_t>0
$$

说明：

> 这个动作好。

所以应该提高它的概率：

$$
r_t\uparrow
$$

但是：

$$
r_t>1+\epsilon
$$

以后收益不再继续增加。

于是：

```text
好动作：

提高概率 ✔
疯狂提高概率 ✘
```


### 3.9 Advantage < 0 时发生什么？

如果：

$$
A_t<0
$$

说明：

> 这个动作不好。

应该：

$$
\pi(a|s)\downarrow
$$

也就是：

$$
r_t\downarrow
$$

但同样不能无限降低。

当：

$$
r_t<1-\epsilon
$$

以后，PPO 不再鼓励继续降低。

于是：

```text
坏动作：

降低概率 ✔
直接把概率干成 0 ✘
```


### 3.10 PPO 可以怎么记？

如果只用一句话记忆：

> **好动作提高概率，坏动作降低概率，但是每一次都别改太狠。**

这就是 PPO 的核心。

所以：

```text
Policy Gradient

好就奖励
坏就惩罚
       ↓
问题：一次可能改太多
       ↓
PPO
       ↓
Clip 限制策略更新幅度
```


## 四、GAE 与 PPO 完整流程

### 4.1 PPO 里的 Critic 在干什么？

前面还有一个问题没解决：

$$
\hat A_t
$$

到底怎么算？

PPO 一般还要训练 Critic：

$$
V_\phi(s_t)
$$

来估计当前状态的价值。

然后利用 Reward 和 Value 构造：

$$
Advantage
$$

一种常见方法就是：

> GAE，Generalized Advantage Estimation。


### 4.2 从 TD Error 到 GAE

先来看 TD Error：

$$
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
$$

它表示：

> 实际观察到的新信息，与之前预测之间的差距。

如果：

$$
\delta_t>0
$$

说明：

> 事情发展得比 Critic 预计的好。

那么这个 Action 很可能值得鼓励。


GAE 则把多个未来 TD Error 组合起来：

$$
\hat A_t
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+\cdots
$$

也可以写成：

$$
\hat A_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
$$

这里：

$$
\lambda
$$

负责平衡：

> Bias 和 Variance。

所以 PPO 训练中经常可以看到：

```text
Critic
   ↓
Value
   ↓
TD Error
   ↓
GAE
   ↓
Advantage
   ↓
PPO Loss
   ↓
更新 Actor
```


### 4.3 把 PPO 整体流程串起来

PPO 大致可以分为下面几步。

**第一步：旧策略采样**

使用：

$$
\pi_{\theta_{old}}
$$

与环境交互，得到：

$$
s_t,a_t,r_t,s_{t+1}
$$


**第二步：Critic 估值**

Critic：

$$
V_\phi(s_t)
$$

预测每个 State 的价值。


**第三步：计算 Advantage**

例如使用：

$$
GAE
$$

得到：

$$
\hat A_t
$$


**第四步：Actor 更新**

计算：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{old}}(a_t|s_t)
}
$$

然后使用 PPO Clip Loss。


**第五步：Critic 更新**

让：

$$
V_\phi(s)
$$

更加接近真实 Return。


整个过程：

```text
旧 Actor
   ↓
与环境交互
   ↓
Trajectory
   ↓
Reward
   ↓
Critic → Value
   ↓
GAE
   ↓
Advantage
   ↓
PPO Clip
   ↓
更新 Actor
```


## 五、进入大语言模型：State、Trajectory 与 Reward Model

### 5.1 到大语言模型里，State 和 Action 是什么？

现在终于来到 LLM。

传统 RL：

```text
State：
游戏画面

Action：
左 / 右 / 跳 / 攻击
```

而对于大语言模型：

**State：**

可以理解成：

> Prompt + 当前已经生成的 Token。

例如：

```text
中国的首都是
```


**Action：**

就是：

> 下一个 Token。

例如词表中：

```text
北京
上海
广州
南京
...
```

模型实际输出的是：

$$
P(token|context)
$$

这本质上就是：

$$
\pi_\theta(a|s)
$$

所以：

> **语言模型天然就是一个 Policy。**


### 5.2 一整段回答就是 Trajectory

假设 Prompt：

```text
1 + 1 等于多少？
```

模型生成：

```text
1 + 1 = 2。
```

对于 RL 来说，可以看成：

```text
State 1：
Prompt

Action 1：
"1"

State 2：
Prompt + "1"

Action 2：
"+"

State 3：
Prompt + "1 +"

...
```

一直生成到 EOS。

所以整个 Completion：

$$
y=(y_1,y_2,\cdots,y_T)
$$

就是一条：

$$
Trajectory
$$


### 5.3 问题来了：LLM 的 Reward 从哪里来？

游戏里很好理解：

```text
赢：+1
输：-1
```

机器人也可以：

```text
走到目标：+10
摔倒：-10
```

但大语言模型呢？

模型生成：

```text
巴黎是德国的首都。
```

它究竟应该获得：

$$
-0.7
$$

还是：

$$
-2.3
$$

？

这没有天然的 Reward。

于是：

> Reward Model 登场了。


### 5.4 Reward Model 是干什么的？

Reward Model：

$$
R_\phi(x,y)
$$

输入：

```text
Prompt + Response
```

输出：

$$
Reward
$$

例如：

```text
回答 A：
巴黎是法国首都。

Reward = 4.3
```

```text
回答 B：
巴黎是德国首都。

Reward = -2.1
```

那么 PPO 就可以根据 Reward 更新语言模型。


### 5.5 Reward Model 怎么训练？

一个经典做法不是让人类直接打：

```text
7.345 分
```

因为绝对打分很难。

相比之下，让人类判断：

```text
回答 A

vs

回答 B

哪个更好？
```

容易很多。

因此可以收集 Preference Data：

$$
(x,y_w,y_l)
$$

其中：

$$
y_w
$$

代表 preferred / winner。

$$
y_l
$$

代表 rejected / loser。

Reward Model 希望满足：

$$
R(x,y_w)
>
R(x,y_l)
$$

常见地可以使用 Bradley-Terry 风格的概率模型：

$$
P(y_w\succ y_l)
=
\sigma
(
R(x,y_w)-R(x,y_l)
)
$$

然后训练 Reward Model。


### 5.6 RLHF 中经典 PPO 流程

于是经典 RLHF 逐渐形成：

```text
Pretraining
     ↓
Base LLM
     ↓
SFT
     ↓
SFT Model
     ↓
Preference Data
     ↓
Reward Model
     ↓
PPO
     ↓
Aligned LLM
```

展开来看：

**第一步**

Prompt 输入 Policy Model。

**第二步**

模型生成 Response。

**第三步**

Reward Model：

$$
R(x,y)
$$

给 Response 打分。

**第四步**

Critic：

$$
V(s)
$$

估计 Value。

**第五步**

计算：

$$
Advantage
$$

**第六步**

使用 PPO：

$$
L^{CLIP}
$$

更新 Policy。


## 六、Reference Model 与 KL 约束

### 6.1 为什么还需要 Reference Model？

假设 Reward Model 特别喜欢：

```text
答案越长越好。
```

模型可能逐渐发现：

> 那我疯狂输出废话，是不是 Reward 就越来越高？

这就是：

> Reward Hacking。

或者策略训练得太猛以后，模型可能直接偏离原来的语言能力。

因此通常保留一个：

$$
\pi_{ref}
$$

Reference Model。

然后限制：

$$
\pi_\theta
$$

不要离：

$$
\pi_{ref}
$$

太远。


### 6.2 KL Penalty

通常可以加入：

$$
D_{KL}
(
\pi_\theta
\|
\pi_{ref}
)
$$

进行约束。

整体 Reward 可以理解成类似：

$$
R_{total}
=
R_{RM}
-
\beta D_{KL}
(
\pi_\theta
\|
\pi_{ref}
)
$$

直觉就是：

```text
Reward Model：
你这个答案不错，+10。

Reference Model：
但是你现在说话方式
已经和原来的模型差太远了。

KL：
扣你 3 分。

最终：
10 - 3 = 7
```

所以 KL 的作用之一就是：

> 防止模型为了追逐 Reward，把原本已经学好的语言能力全部丢掉。


## 七、PPO 用在 LLM 上为什么这么重

### 7.1 PPO 用在 LLM 上为什么这么重？

到这里看一下整个 PPO 系统。

大模型 PPO 可能需要：

**Policy / Actor**

真正需要训练的大语言模型。

**Critic / Value Model**

估计：

$$
V(s)
$$

**Reference Model**

计算 KL。

**Reward Model**

给回答评分。

于是形成：

```text
Actor
Critic
Reward Model
Reference Model
```

也就是同时需要维护多个大模型。

而 Critic 的规模往往也很大。

这会带来非常高的：

- 显存占用
- 计算成本
- 通信成本
- 训练复杂度

这也是后来很多方法试图解决的问题。


### 7.2 更麻烦的是：LLM 的 Value 很难估计

假设模型正在解一道数学题：

```text
设 x = 3，
因此我们可以得到……
```

现在 Critic 要预测：

$$
V(s_t)
$$

意思是：

> 模型推理到这里以后，未来最终答对的概率/期望 Reward 是多少？

问题是：

这个非常难预测。

甚至一个 Token：

```text
“因此”
```

到底让最终解题成功概率增加了：

$$
0.02
$$

还是：

$$
0.0001
$$

？

实际上很难有准确答案。

而数学、代码等推理任务的 Reward 又经常只在最后才能确定：

```text
最终答案正确：1
最终答案错误：0
```

所以：

> 长序列 + 稀疏 Reward，使 Critic 的学习更加困难。

视频资料也特别强调了 LLM PPO 中 Actor、Critic、Reward Model、Reference Model，以及 GAE、KL 等组件；而 GRPO 的关键动机之一，就是移除 Value Model，通过同一 Prompt 的多个回答构造相对 Advantage。


## 八、DPO：直接偏好优化

### 8.1 第一条改进路线：DPO

既然 PPO 这么复杂，问题就来了：

> 我们真的一定需要 Reward Model + PPO 吗？

假设我们手里已经有人类偏好数据：

```text
Prompt

回答 A：好
回答 B：差
```

也就是：

$$
(x,y_w,y_l)
$$

PPO 的路线大概是：

```text
Preference Data
       ↓
训练 Reward Model
       ↓
Reward
       ↓
PPO
       ↓
Policy
```

而 DPO 的思想是：

> **既然最终目的就是让好回答概率上升、坏回答概率下降，那能不能直接学？**

于是：

**DPO：Direct Preference Optimization（直接偏好优化）。**

直接偏好优化。


### 8.2 DPO 的核心思想

例如：

```text
Prompt：
法国首都是哪里？
```

两个回答：

```text
Winner：
法国首都是巴黎。

Loser：
法国首都是柏林。
```

我们希望训练以后：

$$
\pi_\theta(y_w|x)
\uparrow
$$

同时：

$$
\pi_\theta(y_l|x)
\downarrow
$$

也就是说：

> 好答案概率越来越高，差答案概率越来越低。

不再显式训练 Reward Model，也不再进行 PPO rollout。

所以可以粗略理解：

```text
PPO：

Preference
   ↓
Reward Model
   ↓
Reward
   ↓
RL


DPO：

Preference
   ↓
直接优化 Policy
```


### 8.3 DPO 为什么还需要 Reference Model？

如果只是无脑：

$$
\uparrow \pi(y_w)
$$

$$
\downarrow \pi(y_l)
$$

还是可能让模型改得太狠。

所以 DPO 依然会拿当前 Policy 和 Reference Policy 比较。

核心关注：

$$
\log
\frac{
\pi_\theta(y|x)
}{
\pi_{ref}(y|x)
}
$$

意思是：

> 相比原来的模型，现在到底把这个回答的概率提高了多少？

对于 Winner，希望：

$$
\log
\frac{
\pi_\theta(y_w|x)
}{
\pi_{ref}(y_w|x)
}
$$

更大。

对于 Loser，则希望相对更小。


### 8.4 DPO Loss

经典 DPO Objective 可以写成：

$$
L_{DPO}
=
-
\mathbb E
\left[
\log\sigma
\left(
\beta
\left[
\log
\frac{
\pi_\theta(y_w|x)
}{
\pi_{ref}(y_w|x)
}
-
\log
\frac{
\pi_\theta(y_l|x)
}{
\pi_{ref}(y_l|x)
}
\right]
\right)
\right]
$$

公式很长，但核心就一句：

> **相对于 Reference Model，让 Winner 的概率提升幅度超过 Loser。**

因此没必要死记公式。

记住：

```text
Winner
↑

Loser
↓

同时不能离 Reference 太远
```

就够了。


### 8.5 PPO 和 DPO 最大区别

PPO：

> Online / RL 风格。

模型：

```text
自己生成
↓
得到 Reward
↓
根据 Reward 更新
↓
再生成
```

训练过程中不断产生新的行为。


DPO：

> Offline Preference Learning 风格。

使用已经收集好的：

$$
(x,y_w,y_l)
$$

直接学习。

所以：

| PPO | DPO |
|---|---|
| 需要 Reward | 需要 Preference Pair |
| 通常需要 Reward Model | 不需要显式 Reward Model |
| 需要 Rollout | 通常基于已有数据 |
| 有 Critic | 不需要 Critic |
| 训练较复杂 | 相对简单 |
| 可以继续在线探索 | 主要受已有偏好数据限制 |


### 8.6 DPO 的代价是什么？

DPO 简单了很多，但也牺牲了一件重要的事情：

> **在线探索。**

假设训练数据里从来没有一个特别优秀的新解法。

DPO 只能从已有：

```text
Winner / Loser
```

中学习。

它不会像 RL 那样：

```text
模型自己生成新答案
↓
环境打分
↓
发现一个以前没有出现过的新策略
↓
继续强化
```

这对于：

- 数学推理
- 代码生成
- Agent
- 可验证任务

尤其重要。

因此在需要模型不断：

> 探索 → 获得 Reward → 改进

的场景中，RL 仍然非常有价值。

于是又来到 **GRPO**（Group Relative Policy Optimization）：



## 九、GRPO：组相对策略优化

### 9.1 GRPO：如果我们把 Critic 删了呢？

PPO 最大的重量级组件之一就是：

$$
V_\phi(s)
$$

也就是 Critic。

GRPO 问了一个非常直接的问题：

> 能不能不预测“这条回答绝对值多少钱”，而直接比较“这一批回答谁更好”？

答案是：

可以。


### 9.2 GRPO 的核心：同一道题多做几遍

假设 Prompt：

```text
求方程 x² - 5x + 6 = 0。
```

让模型一次生成：

$$
G=5
$$

个回答：

```text
回答 1：x = 2,3
回答 2：x = 1,6
回答 3：x = 2,3
回答 4：推导错误
回答 5：x = 2,3
```

然后 Reward Function 打分：

```text
R1 = 1
R2 = 0
R3 = 1
R4 = 0
R5 = 1
```

现在我们不需要 Critic 告诉我们：

```text
这个 State 的 Value = 0.7238
```

直接看：

> 你在同一组里面表现得怎么样？

这就是：

> Group Relative。


### 9.3 GRPO 的 Advantage

假设一组 Reward：

$$
R_1,R_2,\cdots,R_G
$$

先计算均值：

$$
\mu_R
=
\frac1G
\sum_{i=1}^{G}R_i
$$

再计算标准差：

$$
\sigma_R
$$

然后：

$$
A_i
=
\frac{
R_i-\mu_R
}{
\sigma_R+\epsilon
}
$$

这其实就是：

> Z-score 标准化。


### 9.4 一个具体例子

假设：

$$
R=[2,4,6]
$$

均值：

$$
\mu=4
$$

暂时忽略标准差，先只看：

$$
R_i-\mu
$$

那么：

```text
回答 1：
2 - 4 = -2

回答 2：
4 - 4 = 0

回答 3：
6 - 4 = +2
```

意味着：

```text
回答 1：
比这组平均水平差
→ 抑制

回答 2：
平均水平
→ 基本不动

回答 3：
比平均水平好
→ 强化
```

这就是 GRPO 最关键的思想。

知乎相关资料将这一点概括为：同一 Prompt 生成多份回答，通过组内 Reward 的均值和标准差构造相对 Advantage，从而不依赖单独的 Critic。


### 9.5 为什么相对比较比绝对估值容易？

回到学生考试。

假设问：

> 张三数学水平到底是多少？

你可能很难精确说：

$$
82.37
$$

但是如果同样一道题，五个人一起做：

```text
A：完全正确
B：完全错误
C：正确
D：漏了一步
E：正确
```

你很容易判断：

> A 比 B 好。

这就是：

```text
PPO Critic：

这个状态绝对值多少？
        ↓
较难


GRPO：

同一道题的几份答案谁更好？
        ↓
相对容易
```

对于 LLM 尤其如此，因为同一个 Prompt 本来就很容易通过 Sampling 生成很多 Completion。


### 9.6 GRPO 并不是把 PPO 全扔了

这是一个非常容易误解的地方。

GRPO：

> 并不是完全和 PPO 没关系。

实际上它保留了 PPO 的很多思想。

例如依然有：

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}|q,o_{i,<t})
}{
\pi_{old}(o_{i,t}|q,o_{i,<t})
}
$$

也就是：

> 当前策略 / 旧策略概率比。

然后依然使用：

$$
\operatorname{clip}
$$

限制更新范围。

所以可以粗略理解成：

```text
PPO：

Critic
↓
GAE
↓
Advantage
↓
PPO Clip


GRPO：

Group Rewards
↓
Group Relative Advantage
↓
PPO Clip
```

最大的改动就是：

> **Advantage 的来源变了。**


### 9.7 PPO 与 GRPO 的核心差别

PPO：

$$
A_t
\approx
GAE(
V_\phi
)
$$

也就是：

> Critic 帮我判断当前 Action 比预期好多少。

GRPO：

$$
A_i
=
\frac{
R_i-\mu_R
}{
\sigma_R
}
$$

也就是：

> 同一道题里，你这个回答比其他回答平均水平好多少。

所以：

```text
PPO：
跟“我预计你应该达到的水平”比较。

GRPO：
跟“这一组同学的平均水平”比较。
```


### 9.8 GRPO Objective

一个简化的 GRPO Objective 可以写成：

$$
J_{GRPO}(\theta)
=
\mathbb E
\left[
\frac1G
\sum_{i=1}^G
\frac1{|o_i|}
\sum_t
\left(
\min
(
r_{i,t}A_i,
\operatorname{clip}
(r_{i,t},1-\epsilon,1+\epsilon)
A_i
)
-
\beta D_{KL}
\right)
\right]
$$

看起来非常复杂。

但拆开其实只有四个东西。


**第一部分：Group Sampling**

对于同一个 Prompt：

$$
q
$$

生成：

$$
o_1,o_2,\cdots,o_G
$$


**第二部分：Reward**

得到：

$$
R_1,R_2,\cdots,R_G
$$


**第三部分：Relative Advantage**

$$
A_i
=
\frac{
R_i-\mu
}{
\sigma
}
$$


**第四部分：PPO-style Update**

$$
\min
(
rA,
clip(r)A
)
$$

并使用 KL 限制模型不要偏离 Reference Model 太远。

因此 GRPO 仍然继承了 PPO 风格的 clipped policy objective 和 KL 约束，只是用组内相对奖励替换 Critic/GAE 产生的优势。


### 9.9 GRPO 完整流程

现在可以把 GRPO 写成一个非常直观的算法流程。

**Step 1：采样 Prompt**

```text
Prompt：
求 24 × 17。
```


**Step 2：生成 G 个回答**

```text
Response 1
Response 2
Response 3
...
Response G
```


**Step 3：计算 Reward**

例如数学题可以直接验证最终答案：

```text
正确：1
错误：0
```

得到：

$$
R_1,\cdots,R_G
$$


**Step 4：计算 Group Advantage**

$$
A_i
=
\frac{
R_i-\operatorname{mean}(R)
}{
\operatorname{std}(R)
}
$$


**Step 5：计算 Policy Ratio**

$$
r_{i,t}
=
\frac{
\pi_\theta
}{
\pi_{old}
}
$$


**Step 6：Clip**

限制：

$$
r_{i,t}
$$

不要变化太猛。


**Step 7：KL Constraint**

约束：

$$
\pi_\theta
$$

不要过度偏离：

$$
\pi_{ref}
$$


**Step 8：更新参数**

最终：

```text
好于组平均的回答
↑ Probability

差于组平均的回答
↓ Probability
```


### 9.10 GRPO 为什么特别适合数学和代码？

因为这些任务有一个特别好的性质：

> **Reward 容易验证。**

例如数学题：

```text
最终答案 = 42
```

直接和 Ground Truth 比较。

代码：

```text
运行 Test Cases
```

全部通过：

$$
Reward=1
$$

失败：

$$
Reward=0
$$

于是我们不一定需要：

> 一个复杂的人类偏好 Reward Model。

可以直接使用：

> Rule-based Reward / Verifiable Reward。


### 9.11 一个非常重要的变化：从“人觉得好”到“环境验证对”

传统 RLHF 更多是：

```text
回答
↓
人类偏好
↓
Reward Model
↓
Reward
```

而 Reasoning RL 很多时候可以变成：

```text
回答
↓
Verifier
↓
Correct / Incorrect
↓
Reward
```

例如：

**数学**

检查最终答案。

**Code**

运行 Unit Tests。

**Agent**

检查：

```text
任务是否真正完成？
```

**Search**

检查：

```text
是否找到正确答案？
```

这类 Reward 被称为：

> Verifiable Reward。

它也是近几年大模型推理强化学习非常重要的一条路线。


### 9.12 为什么 GRPO 不需要 Critic？

现在终于可以用一句话回答：

PPO 需要 Critic，是因为它要知道：

$$
A=Q-V
$$

即：

> 这次表现相对于“预计水平”好多少？

而 GRPO 直接拿同一 Prompt 的其他回答作为 Baseline：

$$
A_i
\approx
R_i-\operatorname{mean}(R)
$$

于是：

```text
Critic 的 Baseline

变成了

Group Mean
```

所以：

> 不再需要额外训练一个 Value Network。

这正是 GRPO 能够降低 LLM RL 训练资源开销的重要原因之一。


### 9.13 但 GRPO 也不是完美的

GRPO 最大的问题之一就在：

> Group。

假设同一道题采样：

$$
G=8
$$

个回答。

结果：

```text
8 个全错
```

Reward：

$$
[0,0,0,0,0,0,0,0]
$$

那么：

$$
\mu=0
$$

同时：

$$
\sigma=0
$$

所有回答的相对 Advantage 都没有有效区分。

也就是说：

> 如果整组完全没有差异，就没有什么可学习的相对信号。


### 9.14 如果全部答对也一样

例如：

$$
R=[1,1,1,1,1,1,1,1]
$$

那么：

```text
大家都一样好。
```

GRPO 同样不知道：

> 到底哪个回答应该比其他回答更加被强化。

因此 GRPO 比较喜欢：

```text
有些答对
有些答错
```

或者：

```text
Reward 有明显差异
```

的 Prompt。

这也是为什么：

> Prompt 难度、Sampling Temperature、Group Size 等都会影响 GRPO。


### 9.15 Group Size 有什么作用？

假设：

$$
G=2
$$

只有两份回答。

Group Mean 很容易受到偶然性的影响。

如果：

$$
G=16
$$

相对排名通常更加稳定。

但同时：

```text
每个 Prompt 要生成 16 个 Completion
```

Rollout 成本也变成原来的很多倍。

因此 GRPO 并不是：

> 完全没有额外成本。

而是把 PPO 中很重的：

```text
Critic Training
```

换成了：

```text
更多 Rollout
```

本质上仍然存在计算权衡。


## 十、PPO、DPO、GRPO 的关系与总结

### 10.1 PPO、DPO、GRPO 到底是什么关系？

现在把三者真正串起来。


**PPO**

核心思想：

> 根据 Reward 在线更新 Policy，并用 Clip 防止策略改变太快。

需要：

```text
Policy
Critic
Reward
Reference
```

特点：

> 强大、通用，但是复杂且昂贵。


**DPO**

核心思想：

> 已经有 Winner / Loser，就直接让 Winner 概率高于 Loser。

省掉：

```text
Reward Model
Critic
Online Rollout
```

特点：

> 简单、稳定，但是基本依赖已有 Preference Data。


**GRPO**

核心思想：

> 保留在线 RL，但用同一 Prompt 下多个回答的相对 Reward 替代 Critic。

省掉：

```text
Critic
```

但仍然需要：

```text
Rollout
Reward
Policy Update
```

特点：

> 尤其适合可以自动验证 Reward 的 Reasoning Task。


### 10.2 一个表格彻底区分 PPO / DPO / GRPO

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


### 10.3 一个班级考试的比喻

如果还是容易混，可以用一个统一比喻。

**PPO：老师 + 学生**

学生做题。

老师 Critic 告诉他：

```text
“按照你的水平，
这道题正常应该拿 70 分。”
```

结果学生拿：

```text
90 分。
```

所以：

$$
Advantage=+20
$$

强化这次做法。


**DPO：只看两份答案**

老师不给绝对分。

只告诉你：

```text
答案 A

比

答案 B

好。
```

于是模型学习：

```text
以后更像 A，
少像 B。
```


**GRPO：全班一起考试**

不请老师预测：

```text
“这个学生理论上应该考多少。”
```

直接让一组学生一起做。

成绩：

```text
60
70
85
95
40
```

平均：

$$
70
$$

那么：

```text
95：
明显高于平均 → 强烈奖励

85：
高于平均 → 奖励

70：
平均 → 基本不动

40：
低于平均 → 惩罚
```

这就是：

> Group Relative Policy Optimization。


### 10.4 从 PPO 到 GRPO，本质上改了什么？

很多时候我们会说：

> GRPO 是 PPO 的改进。

但更准确地说，它并不是把 PPO 从头到尾重写。

真正改变最大的地方其实是：

$$
\boxed{
Advantage\ Estimation
}
$$

PPO：

$$
\boxed{
Critic
\rightarrow
GAE
\rightarrow
Advantage
}
$$

GRPO：

$$
\boxed{
Group\ Reward
\rightarrow
Normalization
\rightarrow
Advantage
}
$$

而：

```text
Probability Ratio
Clip
KL Constraint
Policy Update
```

这些 PPO 风格的思想依然存在。

所以以后再看到 GRPO Loss，第一反应不要是：

> “又来了一个完全新的强化学习算法。”

而应该是：

> “它主要把 PPO 的 Critic-based Advantage 换成了 Group-based Advantage。”

这样会好理解很多。


### 10.5 从 DPO 到 GRPO，本质区别又是什么？

这两个虽然都去掉了 Critic，但思想完全不同。

DPO：

```text
数据已经在那里了。

Winner
Loser

直接学。
```

GRPO：

```text
模型自己不断产生新回答。

Generation
↓
Reward
↓
Learn
↓
Generation
↓
Reward
↓
Learn
```

所以：

> DPO 更偏“从已有偏好中学习”。

而：

> GRPO 更偏“通过环境反馈不断探索”。

这也是为什么强化学习重新在 LLM Reasoning 中受到大量关注。


### 10.6 最后把整个发展路径串起来

从上一篇 RL 基础开始：

```text
MDP
│
├── Value-based
│
│   ├── Monte Carlo
│   ├── TD
│   ├── SARSA
│   ├── Q-Learning
│   └── DQN
│
└── Policy-based
    │
    ├── Policy Gradient
    │
    ├── REINFORCE
    │
    └── Actor-Critic
            │
            ▼
           PPO
            │
            ├───────────────┐
            │               │
            ▼               ▼
           DPO             GRPO
            │               │
     Preference Pair    Group Rollout
            │               │
      Direct Update     Relative Reward
```

再换一个角度：

```text
Policy Gradient
      │
      │ 问题：训练不稳定
      ▼
     PPO
      │
      │ 问题：LLM 上太重
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
     DPO                    GRPO
      │                      │
不要在线 RL          保留在线 RL
直接学偏好           但删除 Critic
```


### 10.7 最值得记住的公式

**Policy Ratio**

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{old}(a_t|s_t)
}
$$

含义：

> 新策略相对于旧策略，把这个 Action 的概率改了多少。


**PPO Clip**

$$
L^{CLIP}
=
\mathbb E
[
\min
(
r_tA_t,
clip(r_t,1-\epsilon,1+\epsilon)A_t
)
]
$$

含义：

> 好动作增加概率，坏动作降低概率，但是不要改太猛。


**Advantage**

$$
A(s,a)
=
Q(s,a)-V(s)
$$

含义：

> 这个动作相比正常水平到底好多少。


**GAE**

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


**DPO**

核心看：

$$
\log
\frac{
\pi_\theta(y_w|x)
}{
\pi_{ref}(y_w|x)
}
-
\log
\frac{
\pi_\theta(y_l|x)
}{
\pi_{ref}(y_l|x)
}
$$

含义：

> 相比 Reference Model，让 Winner 的提升超过 Loser。


**GRPO Advantage**

$$
A_i
=
\frac{
R_i-\operatorname{mean}(R)
}{
\operatorname{std}(R)+\epsilon
}
$$

含义：

> 这个回答比同一 Prompt 下其他回答平均水平好多少。


### 10.8 最后总结

如果完全不记公式，只记下面这些也够用了。

**Policy Gradient**

> Reward 高的 Action，以后多做。

**Advantage**

> 不看绝对 Reward，而看“比正常水平好多少”。

**Actor-Critic**

> Actor 负责行动，Critic 负责评价。

**PPO**

> 好动作增加概率，坏动作降低概率，但 Clip 限制每一次更新不要过猛。

**Reward Model**

> 把人类偏好转换成强化学习可以使用的 Reward。

**Reference Model**

> 限制训练后的模型不要离原来的模型太远。

**DPO**

> 已经知道哪个回答更好了，就不要绕 Reward Model + PPO，直接提高 Winner、降低 Loser。

**GRPO**

> 不训练 Critic；同一道题生成多个答案，拿组内平均水平作为 Baseline。

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

1. 《【大白话04】一文理清强化学习 PPO 和 GRPO 算法流程 | 原理图解》
   - Bilibili
   - BV15cZYYvEhz

2. 《看完能和外婆解释的 PPO, DPO, GRPO 强化学习》
   - 知乎专栏
   - https://zhuanlan.zhihu.com/p/1984387073625593089

3. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.

4. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023.

5. Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024.

6. DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025.

[1]: https://www.bilibili.com/video/BV15cZYYvEhz/ "〖大白话04〗一文理清强化学习PPO和GRPO算法流程 | 原理图解_哔哩哔哩_bilibili"
