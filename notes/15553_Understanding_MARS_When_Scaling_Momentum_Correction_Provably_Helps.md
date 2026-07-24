---
date: "2026-07-24"
paper_id: "未提供（本地文档编号：15553）"
title: "Understanding MARS: When Scaling Momentum Correction Provably Helps"
authors: "Egor Shulgin, Tamaz Gadaev, Sarit Khirirat, Peter Richtarik"
domain: "优化算法 / 大模型训练"
tags:
  - "论文笔记"
  - "随机优化"
  - "MARS"
  - "动量"
  - "方差缩减"
  - "LLM训练"
quality_score: "7.8/10"
created: "2026-07-24"
updated: "2026-07-24"
status: "analyzed"
source: "../markdown/15553_Understanding_MARS_When__2080601524336103424.md"
---

# Understanding MARS: When Scaling Momentum Correction Provably Helps

## 核心信息

- **作者**：Egor Shulgin、Tamaz Gadaev、Sarit Khirirat、Peter Richtarik。
- **研究主题**：为 MARS（Momentum with Adaptive Residual Scaling）相对 MVR（Momentum-based Variance Reduction）的效果提供收敛复杂度层面的解释，并检验其在 LLM 预训练中的表现。
- **论文类型**：理论分析为主，辅以 CIFAR-10 探针实验和 124M GPT 风格模型预训练实验。
- **版本信息**：所给 Markdown 未包含可核验的 arXiv 编号、发表日期、会议或作者单位；文中致谢提及 KAUST 资助，不能据此推断所有作者的机构。
- **原文**：[论文 Markdown](../markdown/15553_Understanding_MARS_When__2080601524336103424.md)。

> **一句话结论**
> MARS 的缩放系数 $\gamma$ 不是“越接近 1 越好”：减小 $\gamma$ 会降低梯度差异项的失配，但也会付出偏离 MVR 的代价；论文用 $\gamma$-similarity 把这项权衡写进复杂度上界。

## 摘要翻译

MARS 通过缩放基于动量的方差缩减（MVR）中的校正项，已成为大语言模型训练中颇有竞争力的优化器。然而，已有理论无法解释：为什么这一修改会比未缩放的 MVR（$\gamma=1$）收敛得更好。本文提出 **$\gamma$-similarity（$\gamma$-相似性）**，刻画缩放系数如何与随机梯度差结构相互作用。该条件在 $\gamma=1$ 时退化为标准相似性，在 $\gamma=0$ 时退化为光滑性。基于此，作者给出固定 $\gamma$ 的 MARS 收敛保证，复杂度显式依赖于 $\gamma$ 和对应的 $\gamma$-相似性常数。上界表明，小 $\gamma$ 可能带来收益：相似性项的下降可以超过偏离 MVR 的惩罚。作者进一步证明，在一定条件下优化 $\gamma$ 能获得低于 MVR 的复杂度保证。124M GPT 风格 LLM 预训练实验也显示，合适的小 $\gamma$ 可改善相对 $\gamma=1$ 和 AdamW 的 token 效率。

## 研究背景与问题

### 为什么需要 MARS

对非凸随机优化

$$
\min_{x\in\mathbb{R}^d} f(x):=\mathbb{E}_{\xi}[f_\xi(x)],
$$

SGD 每步代价低，却受随机梯度方差限制；经典 SVRG 一类方法虽然有理论优势，但往往依赖全梯度，难以直接用于大规模深度学习。MVR 将“同一样本在相邻参数处的随机梯度差”加入动量更新，在该类理论下可达到优于常规 SGD 的速率。

MARS 在 MVR 的梯度差校正项前引入缩放 $\gamma$。已有工作观察到它经验上很有效，但以下问题尚未被回答：

1. 为什么缩放校正项有机会优于标准 MVR 的 $\gamma=1$？
2. 这种改善能否在统一的收敛复杂度表达式中显式呈现？
3. 是否必须采用不可计算、随时间变化的理论最优 $\gamma_t$？

### 本文答案

论文的核心回答是：固定 $\gamma$ 也可以分析，且合适的 $\gamma\in[0,1]$ 能降低一个由“校正误差”和“偏离 MVR 代价”共同决定的上界；但这不是对任意任务、任意 $\gamma$ 的无条件优越性承诺。

## 方法概述

### MARS 更新

令

$$
\Delta_t=\nabla f_{\xi_t}(x_t)-\nabla f_{\xi_t}(x_{t-1}).
$$

两梯度版本的 MARS 使用

$$
x_{t+1}=x_t-\eta g_t,
$$

$$
g_t=(1-\beta)\bigl(g_{t-1}+\gamma\Delta_t\bigr)+\beta\nabla f_{\xi_t}(x_t).
$$

- $\eta$：步长；$\beta$：动量/混合系数。
- $\gamma=1$：退化为 MVR。
- $\gamma=0$：去掉梯度差校正，退化为带动量的 SGD。
- 原文聚焦每步需要两个同样本随机梯度的版本；复用前一步梯度的单梯度变体更便宜，但文中指出其理论速率较弱。

### 核心概念：$\gamma$-similarity

对任意 $x,y$，记随机与全梯度差为

$$
d_\xi(x,y)=\nabla f_\xi(x)-\nabla f_\xi(y),\qquad
d(x,y)=\nabla f(x)-\nabla f(y).
$$

定义

$$
\delta_\gamma^2:=\sup_{x\ne y}
\frac{\mathbb{E}_{\xi}\left[\lVert\gamma d_\xi(x,y)-d(x,y)\rVert^2\right]}
{\lVert x-y\rVert^2}.
$$

直觉上，它衡量“缩放后的随机梯度差”逼近“真实梯度差”的最坏归一化均方误差。

| 取值 | 恢复的对象 | 含义 |
|---|---|---|
| $\gamma=1$ | 标准相似性 $\delta^2$ | 使用完整 MVR 校正时，随机梯度差与全梯度差的失配 |
| $\gamma=0$ | 光滑性常数 $L^2$ | 完全不使用随机差分校正时，只剩全梯度变化 |
| $0<\gamma<1$ | 插值式失配 | 允许在两个端点间寻找更小的误差常数 |

在光滑性、无偏性与有界方差等标准假设下，论文证明

$$
\delta_\gamma^2\leq \gamma^2\delta^2+(1-\gamma)^2L^2.
$$

右侧的最小化解为

$$
\gamma_\star=\frac{L^2}{\delta^2+L^2},
\qquad
\delta_{\gamma_\star}^2\leq\frac{L^2\delta^2}{L^2+\delta^2}.
$$

这表明只看相似性常数时，内点 $\gamma_\star$ 的上界可同时低于两个端点对应的 $L^2$ 与 $\delta^2$。但真正的算法复杂度还包含偏离 $\gamma=1$ 的附加项，因此不能只最小化 $\delta_\gamma$。

### 收敛结论与关键权衡

在目标函数 $L$-光滑、随机梯度无偏且方差至多为 $\sigma^2$ 的条件下，论文给出的固定 $\gamma$ MARS 随机梯度复杂度为

$$
\mathcal{O}\left(
\frac{\sigma^2}{\epsilon^2}
+\frac{L\Delta}{\epsilon^2}
+\frac{\delta_\gamma\Delta\sigma}{\epsilon^3}
+\frac{|\gamma-1|L\Delta\sigma^2}{\epsilon^4}
\right),
$$

其中 $\Delta=f(x_0)-f_{\inf}$，目标为使 $\mathbb{E}\lVert\nabla f(\hat{x}_T)\rVert^2\leq4\epsilon^2$。

这四项的解释：

- 前两项与 $\gamma$ 无关，是共同的基础复杂度。
- 第三项会随 $\delta_\gamma$ 变小而降低，支持把 $\gamma$ 设为小于 1。
- 第四项是偏离 MVR 的代价，在 $\gamma=1$ 时消失；当要求极高精度（小 $\epsilon$）时，它可能很重要。

因此，MARS 并非单纯“减弱校正”，而是解决一个双目标权衡。

## MARS 相对 MVR 的理论优势

忽略共同项后，作者以

$$
A=\frac{\Delta\sigma}{\epsilon^3},\qquad
B=\frac{L\Delta\sigma^2}{\epsilon^4}
$$

定义替代目标

$$
J(\gamma)=A\sqrt{\gamma^2\delta^2+(1-\gamma)^2L^2}+B(1-\gamma).
$$

MVR 对应 $J(1)=A\delta$。在 $B<A\delta$ 的条件下，文中给出一个位于 $[0,1]$ 的显式 $\gamma_\star$，并证明 $J(\gamma_\star)\leq J(1)$。

### 正确解读

- 结论比较的是**所展示的上界表达式**，不是证明所有训练实例中 MARS 的实际损失都严格更低。
- $B<A\delta$ 等价于 $L\sigma<\epsilon\delta$。这更偏向中等目标精度、较高异质性/相似性常数的区域。
- 当 $L$ 或噪声 $\sigma$ 增大，优势出现的区域会缩小；图中的边界也反映了这个趋势。
- 论文的贡献是让“何时缩放有益”可见，而不是宣称固定小 $\gamma$ 普适最优。

## 实验结果

### CIFAR-10：与理论量对齐的探针

**目的**：验证训练轨迹中局部梯度差统计是否偏好 $\gamma<1$，不是追求最强的 CIFAR-10 泛化结果。

**设置**：

| 项目 | 配置 |
|---|---|
| 模型 | 小型 CNN，约 $8.1\times10^5$ 参数 |
| 数据集 | CIFAR-10 |
| 更新 | 原文理论分析的两梯度 $\gamma$-MVR |
| 训练 | 20 epochs，batch size 128 |
| 学习率 / 动量 | 0.05 / 0.99 |
| 其余 | 无 weight decay，单随机种子 |
| 局部估计 | 每 500 步，用 $M=512$ 个采样 mini-batch |

局部预测缩放系数为

$$
\widehat{\gamma}_t^\star=
\frac{\lVert d_t\rVert^2}
{\frac{1}{M}\sum_{m=1}^{M}\lVert d_{B_m,t}\rVert^2}.
$$

原文报告该量在训练中大约落在 $0.10$ 到 $0.68$，中位数约为 $0.44$，始终低于 MVR 的 1；固定 $\gamma$ 扫描的最小训练损失出现在 $\gamma=0.25$。

| $\gamma$ | 最终训练损失 | 最终测试准确率 | 最佳测试准确率 |
|---:|---:|---:|---:|
| 0 | 0.1925 | 72.90 | 75.23 |
| 0.025 | 0.1392 | **76.17** | **76.17** |
| 0.1 | 0.1023 | 74.25 | 74.68 |
| **0.25** | **0.0854** | 72.10 | 73.70 |
| 0.5 | 0.0911 | 70.37 | 71.79 |
| 0.75 | 0.1168 | 70.60 | 71.95 |
| 1.0（MVR） | 0.1415 | 70.69 | 71.72 |

**解读**：训练损失最佳点和测试准确率最佳点不同，因此该实验确实只能支持“$\gamma$ 对优化行为很重要”，不能支持“$\gamma=0.25$ 是最佳泛化超参数”。单种子也限制了统计结论的强度。

### 124M LLM 预训练：MARS-AdamW

| 项目 | 配置 |
|---|---|
| 模型 | Llama 风格 Transformer，约 124M 参数（12L / 12H / 768） |
| 数据 | SlimPajama |
| 序列长度 | 512 |
| batch | 256 条序列 |
| token 预算 | 约 2.10B |
| $\gamma$ 扫描 | $\{1,0.1,0.04,0.025,0.02,0.01\}$ |
| 学习率 | AdamW：$10^{-3}$；MARS：$3\cdot10^{-3}$ |
| Betas | AdamW：$(0.8,0.999)$；MARS：$(0.95,0.99)$ |
| 其他 | weight decay 0.1、2000 warmup steps、cosine scheduler |

作者从验证损失-已处理 token 曲线得到三点观察：

1. 多个小 $\gamma$ 设置在训练的大部分阶段可低于 AdamW，说明缩放有实际潜力。
2. 早期和后期的 $\gamma$ 排序不同，暗示最合适的缩放可能随训练所处区域变化。
3. 在这组固定超参数下，$\gamma=1$ 的 MVR 曲线出现明显不稳定；这说明它可能需要专门重调，不能直接归因为算法本体一定更差。

原文没有以表格给出所有曲线的最终精确数值，也没有报告多随机种子方差或每种 $\gamma$ 的重新调参结果。因此这里的证据主要是定性且受固定预算约束的 token-efficiency 比较。

## 贡献与价值

### 主要贡献

1. **概念统一**：用 $\gamma$-similarity 将标准相似性（$\gamma=1$）与光滑性（$\gamma=0$）纳入同一表达式。
2. **显式复杂度**：给出任意固定 $\gamma$ 的复杂度上界，而非依赖不可直接获得的时变 $\gamma_t$。
3. **条件性比较**：将 MARS 与 MVR 的差异压缩为一维权衡 $J(\gamma)$，说明小 $\gamma$ 为什么可能值得尝试。
4. **实践关联**：在 CIFAR-10 的局部诊断和 124M LLM 预训练中观察到 $\gamma<1$ 的合理性。

### 技术路线定位

```text
SGD / 动量 SGD
      -> MVR（动量 + 随机梯度差校正，gamma = 1）
      -> MARS（对校正项引入 gamma）
      -> 本文：gamma-similarity 与 fixed-gamma 复杂度解释
      -> 潜在方向：在线或阶段性地选择 gamma
```

它属于“随机优化中的方差缩减与新型动量”路线，和 LLM 优化器的连接点在于：MARS 类方法可作为 AdamW/Muon 等更新规则中梯度估计或动量部分的结构化修改。

## 局限性与批判性阅读

1. **理论优势依赖条件和上界松紧度**：$J(\gamma_\star)\leq J(1)$ 是替代上界上的比较，实际 $\delta_\gamma$ 是否紧贴其上界、实际训练是否对应理论区域，均未由定理自动保证。
2. **两梯度成本**：主分析每次迭代需要同一样本在 $x_t,x_{t-1}$ 的两次随机梯度评估。若按梯度评估而非更新步数比较，应把额外计算成本纳入实际吞吐分析。
3. **固定 $\gamma$ 与阶段变化的矛盾尚未解决**：LLM 曲线暗示最佳 $\gamma$ 随训练阶段变化，但论文主定理与实验扫描均使用固定 $\gamma$，尚未给出可实现的自适应规则。
4. **LLM 证据仍有限**：仅一个 124M 模型、一个数据集和约 2.10B tokens，且不同 $\gamma$ 没有单独重调学习率、betas 等超参数；结果不能直接外推到更大模型或不同训练配方。
5. **CIFAR-10 探针非泛化基准**：单随机种子，且最佳训练损失与最佳测试准确率不一致。作者也明确没有将它解释为调优后的泛化比较。
6. **理论假设与深度网络现实的距离**：全局 $L$-光滑、有界噪声方差、全局/最坏情况相似性常数在实际深网训练中难以直接验证；局部代理量只是诊断，并非全局 $\delta_\gamma$ 的一致估计。

## 可复用的实践启示

- 把 $\gamma$ 当作一等超参数，而非默认设为 MVR 的 $\gamma=1$。
- 若实现允许，先做对数尺度的小 $\gamma$ 扫描，并同时观察早期 token efficiency、末期验证损失和稳定性。
- 使用局部梯度差诊断时，可估计 $\widehat{\gamma}_t^\star$；但该量需要额外 mini-batch / 全梯度近似，只适合作为研究或离线调参信号。
- 评估时应报告每步的梯度次数、wall-clock、显存与多随机种子，而不只比较迭代步数或单条 loss 曲线。

## 未来工作

1. **可计算的自适应缩放**：利用低开销 mini-batch 统计构造稳定的在线 $\gamma_t$，并分析其误差与收敛性。
2. **单梯度版本的强化理论**：研究是否能缩小单梯度 MARS 与两梯度版本之间的速率/实际效率差距。
3. **大规模实证**：在多个模型尺寸、数据配方和计算预算下，将 MARS 与重新调优的 AdamW、MVR、Muon/MARS-M 做 token、FLOP 与 wall-clock 的公平比较。
4. **从局部统计到全局理论**：解释局部 $\widehat{\gamma}_t^\star$ 与可观测训练稳定性之间的关系，评估其在非平稳训练轨迹中的可靠性。

## 我的综合评价

**总体评分：7.8/10。** 这是一个把经验观察转化为可检查理论权衡的扎实工作：定义简洁，端点解释清楚，也给出了和 LLM 训练相连的初步实验。其主要不足不是理论结构本身，而是理论改善的条件性、两梯度成本，以及实证范围尚不足以证明跨规模、跨配方的普适收益。

| 维度 | 评分 | 理由 |
|---|---:|---|
| 创新性 | 8.0 | 用 $\gamma$-similarity 把缩放校正的作用显式化，概念与分析目标贴合。 |
| 技术质量 | 8.0 | 给出固定 $\gamma$ 的明确复杂度和端点恢复关系；结论应按“上界优势”理解。 |
| 实验充分性 | 6.5 | 有理论对齐的探针和 LLM 实验，但规模、种子、调参和量化汇总仍有限。 |
| 写作质量 | 8.0 | 问题、定义、权衡和适用范围的主线较清晰。 |
| 实用性 | 7.0 | 对优化器调参有直接启发，但尚缺低成本自适应方案与广泛系统验证。 |

> **阅读时不要混淆**
> 论文证明的是某些条件下 MARS 的**复杂度上界**可优于 MVR；它没有证明小 $\gamma$ 在所有深度学习任务、所有超参数设置中都优于 $\gamma=1$ 或 AdamW。

## 相关论文

- Yuan et al. (2025), *MARS: Unleashing the Power of Variance Reduction for Training Large Models*：提出 MARS；本文补足其相对 MVR 的复杂度解释。
- Cutkosky and Orabona (2019), *Momentum-Based Variance Reduction in Non-Convex SGD*：MVR 的理论基础。
- Liu et al. (2025), *MARS-M: When Variance Reduction Meets Matrices*：把 MARS 思路引入 Muon 类优化器。
- Semenov et al. (2025), *Benchmarking Optimizers for Large Language Model Pretraining*：本文 124M 预训练协议的参考来源。

## 我的笔记

- 想复现实验时，首先确认 MARS-AdamW 的具体实现是否与两梯度理论算法一致。
- 可将 $J(\gamma)$ 作为分析工具，而不是直接当成可用的训练期目标；其中 $L,\delta,\sigma,\Delta$ 在真实训练中并不可直接获得。
