---
author: 芙芙
pubDatetime: 2026-09-03
title: 从调和级数到欧拉–马歇罗尼常数 γ：两个发散量为什么能“减”出一个常数？
featured: false
draft: false
tags:
  - 调和级数
  - 欧拉常数
  - 数学分析
  - 学习笔记
category: 数学
description: 整理自 B 站视频：从调和级数发散讲起，用积分比较解释为什么 H_n 的增长速度是 ln n，再严格证明 H_n−ln n 收敛，最终引出欧拉–马歇罗尼常数 γ≈0.5772。
---

> 本文整理自 B 站的一期关于欧拉–马歇罗尼常数的讲解视频([哔哩哔哩][1])，把「为什么两个发散量能减出一个常数」这个问题完整梳理一遍：调和级数为什么发散、发散速度为什么是 $\ln n$、$H_n-\ln n$ 为什么收敛，以及最后得到的欧拉–马歇罗尼常数 $\gamma$ 到底是什么。
>
> 核心结论先放在这里：**调和级数虽然不断向无穷增长，但它比 $\ln n$「多出来」的部分，最终稳定成了一个有限的常数 $\gamma\approx0.5772156649$。**

## 一、调和级数与一个神秘常数

### 1.1 两个发散量为什么能减出一个常数

在数学分析中有这样一个常数 $\gamma\approx0.577215664901532\cdots$，被称为**欧拉–马歇罗尼常数（Euler–Mascheroni constant）**，它来源于这样一个极限：

$$
\gamma=\lim_{n\to\infty}\left(1+\frac12+\frac13+\cdots+\frac1n-\ln n\right)
$$

乍一看这个式子有些奇怪：$1+\frac12+\frac13+\cdots+\frac1n$ 随着 $n$ 增大趋向无穷，$\ln n$ 同样趋向无穷——也就是说我们似乎是在计算一个 $\infty-\infty$。可神奇的是，这两个发散量的差并没有发散，而是逐渐稳定在 $0.5772156649\cdots$ 附近，这个数就是 $\gamma$。这篇文章就来回答：为什么能「减」出这个常数？

### 1.2 调和级数：发散却慢得惊人

先从最基本的调和级数开始：$1+\frac12+\frac13+\frac14+\frac15+\cdots=\sum_{n=1}^{\infty}\frac1n$，它的部分和 $H_n=\sum_{k=1}^{n}\frac1k$ 被称为第 $n$ 个**调和数（Harmonic Number）**。例如 $H_1=1$、$H_2=1+\frac12=1.5$、$H_3\approx1.8333$、$H_{10}\approx2.92897$。

一个非常经典的结论是：尽管 $\frac1n\to0$，调和级数依然发散——**一个无穷级数的每一项趋向于 0，并不能保证整个级数收敛**。证明可以用分组法：把级数重新分组为

$$
1+\frac12+\left(\frac13+\frac14\right)+\left(\frac15+\cdots+\frac18\right)+\cdots
$$

其中第三组 $\frac13+\frac14>\frac14+\frac14=\frac12$，下一组 $\frac15+\cdots+\frac18>4\times\frac18=\frac12$，再下一组有 8 项、每一项至少是 $\frac1{16}$，所以 $\frac19+\cdots+\frac1{16}>8\times\frac1{16}=\frac12$……依此类推，整个调和级数至少大于 $1+\frac12+\frac12+\frac12+\cdots$，而右边显然趋向无穷。因此 $\sum_{n=1}^{\infty}\frac1n$ 发散，即 $H_n\to+\infty$。

但调和级数发散得非常慢：$H_{10}\approx2.929$、$H_{100}\approx5.187$、$H_{1000}\approx7.485$、$H_{1000000}\approx14.393$——即使加到一百万项，结果也只有大约 $14.4$。那么它到底是按照什么速度增长的呢？答案是 $\ln n$：事实上 $H_n$ 和 $\ln n$ 的增长速度几乎完全相同。这也正是欧拉–马歇罗尼常数出现的地方。

![调和级数 Hₙ 与 ln n 的增长对比（两者几乎平行，相差常数 γ）](https://raw.githubusercontent.com/Titroupast/blog-img/master/euler-harmonic-growth.png)

## 二、为什么增长速度是 ln n

### 2.1 联系来自 1/x 的积分

这里最关键的联系来自函数 $f(x)=\frac1x$。我们知道 $\frac{d}{dx}\ln x=\frac1x$，所以 $\int_1^n\frac1x\,dx=\ln n$；另一方面 $H_n=\sum_{k=1}^{n}\frac1k$。因此我们实际上是在比较两个东西：

$$
\sum_{k=1}^{n}\frac1k\qquad\text{与}\qquad\int_1^n\frac1x\,dx=\ln n
$$

也就是在比较 $H_n$ 与 $\ln n$。

### 2.2 几何视角：矩形近似曲线面积

考虑曲线 $y=\frac1x$。由于 $1/x$ 是单调递减函数，在区间 $[k,k+1]$ 上始终有 $\frac1{k+1}<\frac1x<\frac1k$，对整个区间积分得到 $\frac1{k+1}<\int_k^{k+1}\frac1x\,dx<\frac1k$。而 $\int_k^{k+1}\frac1x\,dx=\ln(k+1)-\ln k$，所以：

$$
\frac1{k+1}<\ln\frac{k+1}{k}<\frac1k
$$

这就是调和级数和对数之间最直接的联系。调和和 $H_n$ 可以理解成：用一个个面积为 $\frac1k\times1$ 的矩形去近似曲线 $y=\frac1x$ 下方的面积，而真正的曲线面积就是 $\int_1^n\frac1x\,dx=\ln n$。因此 $H_n-\ln n$ 本质上可以看作：**用矩形近似 $1/x$ 曲线面积时，长期积累下来的误差**。

![单位宽矩形近似曲线 y=1/x：矩形面积和 = Hₙ，间隙累积即误差来源](https://raw.githubusercontent.com/Titroupast/blog-img/master/euler-rectangle-approx.png)
## 三、γ 的定义与收敛证明

### 3.1 定义数列 a_n = H_n − ln n

现在定义数列 $a_n=H_n-\ln n$，也就是 $a_n=1+\frac12+\cdots+\frac1n-\ln n$。计算前几项：$a_1=1$、$a_{10}\approx0.626383$、$a_{100}\approx0.582207$、$a_{1000}\approx0.577716$。继续增大 $n$，会发现 $a_n$ 越来越接近 $0.577215664901532\cdots$。于是定义：

$$
\gamma=\lim_{n\to\infty}(H_n-\ln n)
$$

这个 $\gamma$ 就是欧拉–马歇罗尼常数。注意 $\gamma$ 本身只是一个常数，严格来说我们不能说「$\gamma$ 收敛」——真正收敛的是数列 $H_n-\ln n$，它的极限恰好被我们命名为 $\gamma$。

### 3.2 为什么 H_n − ln n 一定收敛：单调性

要证明数列收敛，最常用的工具是**单调有界定理**：如果一个实数数列单调且有界，它就一定收敛。所以目标就是证明 $a_n$ 单调递减且有下界。

先证明单调递减。计算 $a_{n+1}-a_n$：因为 $a_{n+1}=H_{n+1}-\ln(n+1)$、$a_n=H_n-\ln n$，且 $H_{n+1}-H_n=\frac1{n+1}$，所以：

$$
a_{n+1}-a_n=\frac1{n+1}-\ln(n+1)+\ln n=\frac1{n+1}-\ln\left(1+\frac1n\right)
$$

接下来利用积分。$\ln\left(1+\frac1n\right)=\int_n^{n+1}\frac1x\,dx$，在区间 $[n,n+1]$ 中 $x<n+1$，所以 $\frac1x>\frac1{n+1}$，于是：

$$
\int_n^{n+1}\frac1x\,dx>\int_n^{n+1}\frac1{n+1}\,dx=\frac1{n+1}
$$

即 $\ln\left(1+\frac1n\right)>\frac1{n+1}$。代回原式得 $a_{n+1}-a_n<0$，所以 $a_{n+1}<a_n$，也就是 **$a_n$ 单调递减**。

### 3.3 有下界

还需要证明 $a_n$ 不会一直减到负无穷，仍然使用积分比较。因为 $\frac1x$ 在每个区间 $[k,k+1]$ 上单调递减，所以 $\frac1k>\int_k^{k+1}\frac1x\,dx$。从 $k=1$ 加到 $k=n$：

$$
\sum_{k=1}^{n}\frac1k>\sum_{k=1}^{n}\int_k^{k+1}\frac1x\,dx=\int_1^{n+1}\frac1x\,dx
$$

右边是 $\ln(n+1)$，于是 $H_n>\ln(n+1)$，因此：

$$
a_n=H_n-\ln n>\ln(n+1)-\ln n=\ln\left(1+\frac1n\right)>0
$$

所以 $a_n$ 至少有一个下界 $0$。

### 3.4 单调有界，必然收敛

现在我们已经证明：$a_{n+1}<a_n$（单调递减），且 $a_n>0$（有下界）。根据单调有界定理，$a_n=H_n-\ln n$ 一定收敛。于是存在某个有限常数 $L$，使得 $\lim_{n\to\infty}(H_n-\ln n)=L$，我们就把这个常数定义为 $\gamma$：

$$
\gamma=\lim_{n\to\infty}\left(\sum_{k=1}^{n}\frac1k-\ln n\right)\approx0.577215664901532\cdots
$$

![Hₙ − ln n 单调递减且有下界，收敛到 γ ≈ 0.5772](https://raw.githubusercontent.com/Titroupast/blog-img/master/euler-gamma-convergence.png)

## 四、∞−∞：一个未定式

### 4.1 两个发散量之差可以是有限数

这里有一个看起来矛盾但非常重要的现象：$H_n\to\infty$、$\ln n\to\infty$，但 $H_n-\ln n\to\gamma$。所以「$\infty-\infty$」并不能直接认为是 $0$、无穷或者「不存在」，它只是一个**未定式**——两个函数都趋向无穷，并不意味着它们的差也趋向无穷。例如 $(n+1)-n=1$：虽然 $n+1\to\infty$、$n\to\infty$，但它们的差始终为 $1$。

$H_n$ 和 $\ln n$ 的情况类似：它们虽然都趋于无穷，但发散的主要部分几乎完全相同，最终只留下一个固定的差 $\gamma$。这就是「两个发散量减出一个常数」背后的直觉。
## 五、H_n 的渐近展开与 γ 的深层含义

### 5.1 H_n = ln n + γ + o(1)

当 $n$ 很大时，可以近似认为 $H_n\approx\ln n+\gamma$。利用 Euler–Maclaurin 公式，还能得到更精确的渐近展开：

$$
H_n=\ln n+\gamma+\frac1{2n}-\frac1{12n^2}+\frac1{120n^4}-\cdots
$$

因此 $H_n-\ln n=\gamma+\frac1{2n}-\frac1{12n^2}+\cdots$。当 $n\to\infty$ 时，后面的 $\frac1{2n}$、$\frac1{12n^2}$、$\cdots$ 全部趋近于 0，于是只剩下 $\gamma$。从这个角度看，$\gamma$ 其实就是调和数渐近展开中的**常数项**：

$$
H_n=\ln n+\gamma+o(1)
$$

### 5.2 离散与连续之间的差

$\gamma$ 还有一个非常漂亮的理解方式。调和和 $\sum_{k=1}^{n}\frac1k$ 是一个**离散求和**，而 $\int_1^n\frac1x\,dx$ 是一个**连续积分**。由于 $\int_1^n\frac1x\,dx=\ln n$，所以：

$$
\gamma=\lim_{n\to\infty}\left(\sum_{k=1}^{n}\frac1k-\int_1^n\frac1x\,dx\right)
$$

因此可以把 $\gamma$ 理解成**离散求和与连续积分之间长期累积留下的有限误差**——这也是为什么 $\gamma$ 会频繁出现在分析学、数论、概率论以及特殊函数中。

## 六、γ 的有理数之谜与总结

### 6.1 γ 是有理数还是无理数？

这里还有一个非常著名的问题。我们已经知道 $\gamma=0.577215664901532\cdots$，但人类目前仍然不知道 $\gamma$ 是有理数还是无理数：既没有证明存在整数 $p,q$ 满足 $\gamma=p/q$，也没有证明这样的整数不存在。更进一步，$\gamma$ 是否为超越数同样未知。

因此不要把 $\gamma$ 和另一个常被称为「欧拉常数」的数 $e=2.718281828\cdots$ 混淆：对于 $e$，我们已经知道它不仅是无理数而且还是超越数；而对于 $\gamma$，这些问题至今仍未解决。

### 6.2 总结

整个逻辑可以浓缩成下面这条路线：

1. 调和级数 $H_n=1+\frac12+\frac13+\cdots+\frac1n$ 虽然发散（$H_n\to\infty$），但它的增长速度与 $\ln n$ 非常接近，因为 $\ln n=\int_1^n\frac1x\,dx$，而 $H_n$ 正是用单位宽矩形去近似这条曲线下方面积的结果；
2. 考虑二者之差 $a_n=H_n-\ln n$；
3. 证明 $a_{n+1}<a_n$（$a_n$ 单调递减），同时 $a_n>0$（$a_n$ 有下界）；
4. 根据单调有界定理，$a_n$ 一定收敛，于是定义：

$$
\gamma=\lim_{n\to\infty}(H_n-\ln n)\approx0.5772156649
$$

最终，对于足够大的 $n$ 有 $H_n\approx\ln n+\gamma$。所以，从某种意义上来说：

> **调和级数虽然不断向无穷增长，但它比 $\ln n$「多出来」的部分，却最终稳定成了一个有限的常数。**

这个常数，就是欧拉–马歇罗尼常数 $\gamma$。

## 参考资料

- Bilibili：从调和级数到欧拉–马歇罗尼常数的讲解：https://www.bilibili.com/video/BV1jSwueEEMe/

[1]: https://www.bilibili.com/video/BV1jSwueEEMe/
