# Gradient-Free Optimizer for Large Model Training：2024–2026 研究进展与优化器设计方向

> **调研日期：** 2026-09-19  
> **核心目标：** 面向“大模型训练优化器”研究，重点关注 **Gradient-Free / Derivative-Free / Zeroth-Order (ZO) Optimizer**，而不是泛泛讨论所有 memory-efficient optimizer。  
> **重点问题：** 如何设计一个在大模型上同时具备 **低显存、低 query complexity、低估计方差、快速收敛、自适应、可扩展到 pretraining** 的 gradient-free optimizer。  
> **邻近工作用途：** `Parameter-free Clipped Gradient Descent Meets Polyak`、`PowerStep`、`Powell-Style Model-Based Derivative-Free Optimization` 等并非都属于 LLM-ZO，但其 **step-size、clipping、geometry、trust-region、model-based search** 思想对新型 ZO optimizer 很有启发。

---

# 目录

1. [研究范围与基本概念](#1-研究范围与基本概念)
2. [为什么 Gradient-Free 对大模型值得研究](#2-为什么-gradient-free-对大模型值得研究)
3. [统一的 Gradient-Free Optimizer 设计框架](#3-统一的-gradient-free-optimizer-设计框架)
4. [2023：MeZO——大模型 ZO 的起点](#4-2023mezo大模型-zo-的起点)
5. [2024–2026 核心论文总表](#5-20242026-核心论文总表)
6. [方向一：降低 Gradient Estimator 的方差与偏差](#6-方向一降低-gradient-estimator-的方差与偏差)
7. [方向二：降低 Effective Dimension——Low-rank / Sparse / Subspace](#7-方向二降低-effective-dimensionlow-rank--sparse--subspace)
8. [方向三：Adaptive / Momentum / Preconditioned ZO](#8-方向三adaptive--momentum--preconditioned-zo)
9. [方向四：Curvature-aware 与 Layer-wise ZO](#9-方向四curvature-aware-与-layer-wise-zo)
10. [方向五：Quantized / Hardware-aware / Distributed ZO](#10-方向五quantized--hardware-aware--distributed-zo)
11. [方向六：Gradient-Free Pretraining](#11-方向六gradient-free-pretraining)
12. [2026 新趋势](#12-2026-新趋势)
13. [邻近优化器工作：哪些思想值得迁移到 Gradient-Free](#13-邻近优化器工作哪些思想值得迁移到-gradient-free)
14. [真正做“新优化器”时应重新设计哪些组件](#14-真正做新优化器时应重新设计哪些组件)
15. [值得重点研究的 6 个选题方向](#15-值得重点研究的-6-个选题方向)
16. [推荐的研究路线与实验方案](#16-推荐的研究路线与实验方案)
17. [论文阅读顺序](#17-论文阅读顺序)
18. [结论](#18-结论)
19. [参考文献与链接](#19-参考文献与链接)

---

# 1. 研究范围与基本概念

## 1.1 本报告中的 Gradient-Free 是什么？

对于大模型训练，本报告主要讨论：

\[
\boxed{\text{Gradient-Free} \approx \text{Derivative-Free} \approx \text{Zeroth-Order Optimization}}
\]

其核心特征是：

> **优化算法不通过 backpropagation 得到参数梯度，而主要依赖 loss / objective function evaluation 来决定更新方向。**

最典型的是 two-point zeroth-order estimator：

\[
\hat g_t
=
\frac{
f(\theta_t+\mu z_t)-f(\theta_t-\mu z_t)
}{
2\mu
}z_t,
\]

其中：

- \(\theta_t\in\mathbb R^d\)：模型参数；
- \(z_t\)：随机扰动方向；
- \(\mu\)：扰动半径；
- \(f(\theta)\)：训练 loss；
- \(\hat g_t\)：仅利用函数值估计得到的伪梯度。

模型更新为：

\[
\theta_{t+1}
=
\theta_t-\eta_t\hat g_t.
\]

因为不需要执行 backward：

\[
\boxed{\text{不保存 activation + 不保存真实 gradient}}
\]

所以显存可接近 inference。

---

## 1.2 哪些工作不是严格意义上的 Gradient-Free？

需要特别区分：

### Parameter-free Clipped Gradient Descent Meets Polyak

它仍然使用：

\[
\nabla f(x),
\]

因此属于 **first-order optimizer / optimization theory**，而不是 gradient-free。

但它研究：

- parameter-free stepsize；
- gradient clipping；
- Polyak stepsize；
- \((L_0,L_1)\)-smoothness；

这些机制非常适合迁移到 ZO optimizer。

---

### PowerStep

PowerStep 同样依赖真实 gradient，因此：

\[
\boxed{\text{PowerStep 不是 Gradient-Free}}
\]

它的价值在于：

> 不保存 Adam 的 second moment，通过 \(\ell_p\)-norm steepest descent 的几何结构获得 coordinate-wise adaptivity。

这对设计 **memory-efficient adaptive ZO optimizer** 很有启发。

---

### Powell-Style Model-Based Derivative-Free Optimization

这一类则是真正的 derivative-free optimization。

但它主要针对传统连续优化，并非 billion-parameter neural network。

它的重要价值是：

\[
\boxed{\text{Gradient-Free 不一定等于随机 finite difference}}
\]

还可以：

- 拟合局部线性/二次模型；
- trust region；
- interpolation geometry；
- random subspace；
- 利用过去 function evaluations。

因此值得作为下一代 LLM gradient-free optimizer 的方法学来源。

---

# 2. 为什么 Gradient-Free 对大模型值得研究？

传统 AdamW 大模型训练需要存储：

\[
\text{parameters}
+
\text{activations}
+
\text{gradients}
+
m_t
+
v_t.
\]

其中：

\[
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t,
\]

\[
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2.
\]

大模型训练最重要的显存负担包括：

1. activation；
2. gradient；
3. optimizer states。

而 ZO 可以消除：

\[
\boxed{
\text{activation-for-backward}
+
\text{gradient tensor}
}
\]

进一步结合 in-place perturbation 后，可以避免显式保存完整：

\[
z_t\in\mathbb R^d.
\]

因此理论上：

\[
M_{\mathrm{ZO}}
\approx
M_{\mathrm{inference}}
+
M_{\mathrm{small\ optimizer\ state}}.
\]

MeZO 展示了这一点：在单张 80GB A100 上，可以对 30B 级 OPT 模型进行 ZO fine-tuning，而同样预算下传统 Adam fine-tuning 可处理的模型显著更小。

但其代价是：

\[
\boxed{
\text{低显存}
\Longleftrightarrow
\text{高方差 + 高 query complexity + 慢收敛}
}
\]

因此当前研究真正的问题不是：

> “ZO 能不能训练 LLM？”

而是：

> **“能否设计一个具有 inference-level memory，同时具有接近 first-order optimizer 收敛效率的 gradient-free optimizer？”**

---

# 3. 统一的 Gradient-Free Optimizer 设计框架

为了做“优化器”而不仅是做一个 MeZO trick，建议把 ZO optimizer 写成如下统一结构。

设每一步采样 \(K_t\) 个方向：

\[
z_{t,k}\sim q_t(z),
\qquad
k=1,\ldots,K_t.
\]

构造 estimator：

\[
\hat g_t
=
\mathcal E
\left(
f(\theta_t+\mu_tz_{t,k}),
f(\theta_t-\mu_tz_{t,k}),
z_{t,k}
\right).
\]

经过 preconditioner / geometry：

\[
d_t
=
P_t\hat g_t.
\]

再通过 optimizer dynamics：

\[
m_t
=
\beta_t m_{t-1}
+
(1-\beta_t)d_t.
\]

最后更新：

\[
\theta_{t+1}
=
\theta_t
-
\eta_t\,
\mathcal C_t(m_t),
\]

其中 \(\mathcal C_t\) 可以代表 clipping / normalization / nonlinear transform。

因此一个 Gradient-Free Optimizer 至少有 **六个可以独立研究的模块**：

| 模块 | 数学对象 | 核心问题 |
|---|---|---|
| Perturbation distribution | \(q_t(z)\) | 应该向哪里搜索？ |
| Estimator | \(\mathcal E\) | 如何降低 bias / variance？ |
| Search geometry | \(P_t\) / subspace | 如何摆脱参数维度 \(d\)？ |
| Optimizer dynamics | momentum / adaptive state | 如何利用历史信息？ |
| Step-size / clipping | \(\eta_t,\mu_t,c_t\) | 如何减少超参数敏感性？ |
| Query budget | \(K_t\) | 每步应该花多少 forward？ |

这六个部分基本可以覆盖 2024–2026 年绝大多数工作。

---

# 4. 2023：MeZO——大模型 ZO 的起点

## Fine-Tuning Language Models with Just Forward Passes

- **方法：** MeZO
- **会议：** NeurIPS 2023
- **CCF：** A
- **定位：** LLM ZO optimization 的基础工作

标准 ZO-SGD：

\[
\hat g_t
=
\frac{
f(\theta_t+\mu z_t)-f(\theta_t-\mu z_t)
}{
2\mu
}z_t.
\]

如果直接实现，需要保存一个与模型相同大小的 \(z_t\)。

MeZO 使用随机种子重新生成：

\[
z_t,
\]

从而完成：

1. \(\theta+\mu z\)；
2. forward；
3. 恢复参数；
4. \(\theta-\mu z\)；
5. forward；
6. 通过相同 seed 重建 \(z\)；
7. in-place update。

因此不需要保存完整 perturbation。

### MeZO 的根本问题

传统 ZO estimator 方差具有维度依赖：

\[
\mathbb E\|
\hat g-\nabla f
\|^2
\sim
O(d).
\]

对于：

\[
d=10^9\sim10^{11},
\]

这看起来应该是灾难性的。

因此整个后续研究几乎都在解决：

\[
\boxed{
d,\quad
\operatorname{Var}(\hat g),
\quad
K,
\quad
\text{optimizer state}
}
\]

之间的矛盾。

---

# 5. 2024–2026 核心论文总表

> CCF 等级按 **CCF 官网当前人工智能目录（2026-09-19 检索）**。  
> 当前官网列出 NeurIPS / ICML / ICCV / AAAI 为 A，EMNLP 为 B，AISTATS 为 C。  
> ICLR 虽然是机器学习领域公认顶级会议，但目前 CCF 官网人工智能目录页面中未列出，因此这里记为“—”，避免自行赋级。  
> ACL Findings 不等同于 ACL Main Conference，单独标注 Findings。

| 年份 | 工作 | 发表 | CCF | 是否严格 Gradient-Free | 主要贡献 |
|---|---|---|---|---|---|
| 2023 | MeZO | NeurIPS 2023 | A | 是 | inference-level memory 的 LLM ZO |
| 2024 | ZO-LLM Benchmark | ICML 2024 | A | 是 | block-wise / hybrid / sparsity benchmark |
| 2024 | MeZO-SVRG | ICML 2024 | A | 是 | SVRG variance reduction |
| 2024 | DPZero | ICML 2024 | A | 是 | nearly dimension-independent private ZO |
| 2024 | Parameter-free Clipped GD Meets Polyak | NeurIPS 2024 | A | **否** | parameter-free clipping / Polyak，重要邻近思想 |
| 2025 | LOZO | ICLR 2025 | — | 是 | low-rank ZO estimator |
| 2025 | Transferable Static Sparsity | ICLR 2025 | — | 是 | 0.1% sensitive parameters + quantization |
| 2025 | Addax | ICLR 2025 | — | 部分 | FO + ZO hybrid |
| 2025 | R-AdaZO | ICML 2025 | A | 是 | adaptive ZO / moment refinement |
| 2025 | SubZero | ICCV 2025 | A | 是 | random low-rank subspace |
| 2025 | Sparse MeZO | NeurIPS 2025 | A | 是 | parameter selection / sparse perturbation |
| 2025 | DiZO | NeurIPS 2025 | A | 是 | layer-wise divergence adaptation |
| 2025 | PaZO | NeurIPS 2025 | A | 是 | preconditioned accelerated ZO |
| 2025 | Unbiased ZO Estimator | NeurIPS 2025 | A | 是 | function-only unbiased gradient estimator |
| 2025 | ZO Finds Flat Minima | NeurIPS 2025 | A | 是 | ZO implicit regularization |
| 2025 | PseuZO | NeurIPS 2025 | A | 广义 ZO | composite structure 减弱 dimension dependence |
| 2025 | Bilevel-ZOFO | NeurIPS 2025 | A | 混合 | ZO backbone + FO PEFT |
| 2025 | MUZO | EMNLP 2025 | B | 是 | multi-query + momentum/Adam |
| 2025 | HELENE | EMNLP 2025 | B | 是 | Hessian + layer-wise clipping |
| 2025 | QuZO | EMNLP 2025 | B | 是 | INT4/INT8 ZO |
| 2025 | TeZO | arXiv | — | 是 | model + temporal low-rank |
| 2026 | LOREN | AAAI 2026 | A | 是 | low-rank curvature / anisotropic perturbation |
| 2026 | ConMeZO | AISTATS 2026 | C | 是 | momentum-centered cone sampling |
| 2026 | FZOO | ICLR 2026 | — | 是 | batched one-sided ZO，降低 forward passes |
| 2026 | QZO | ICLR 2026 | — | 是 | perturb quantization scale |
| 2026 | LMAO | ICLR 2026 | — | 混合 | 3 forward + 1 backward alternating optimization |
| 2026 | Distributed ZO | ACL Findings 2026 | Findings | 是 | distributed high-throughput ZO |
| 2026 | KronZO | SN Computer Science | — | 是 | Kronecker ZO for pretraining |
| 2026 | SubZero+ | arXiv 2026 | — | 是 | multi-query subspace + subspace Adam |
| 2026 | Powell-style Model-Based DFO | arXiv 2026 | — | 是 | model-based trust region / random subspace |
| 2026 | PowerStep | arXiv 2026 | — | **否** | \(\ell_p\) steepest descent，邻近优化器思想 |

---

# 6. 方向一：降低 Gradient Estimator 的方差与偏差

---

## 6.1 MeZO-SVRG — ICML 2024，CCF-A

论文：

**Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models**

核心问题：

\[
\hat g_t
=
g_t+\xi_t,
\]

其中 \(\xi_t\) 是 ZO estimation noise。

MeZO-SVRG 将 SVRG 思想应用到 ZO：

\[
\hat g_t^{\mathrm{VR}}
\approx
\hat g(x_t;B_t)
-
\hat g(\tilde x;B_t)
+
\hat g(\tilde x;D).
\]

通过 reference point：

\[
\tilde x
\]

消除部分 stochastic noise。

论文报告：

- 在多个 LM fine-tuning 任务上优于 MeZO；
- 部分任务测试准确率提升可达约 20%；
- 达到 MeZO 峰值性能所需 GPU-hours 可降低约 \(2\times\)。

### 对新 optimizer 的启示

ZO 最大的瓶颈之一不是：

\[
\text{memory},
\]

而是：

\[
\boxed{\text{variance per forward FLOP}}.
\]

因此以后评价 optimizer 时不应只看：

\[
\operatorname{Var}(\hat g),
\]

还应该看：

\[
\boxed{
\frac{\text{variance reduction}}
{\text{additional forward queries}}
}
\]

---

## 6.2 MUZO — EMNLP 2025，CCF-B

MUZO 使用多个 perturbation：

\[
\hat g_t
=
\frac1K
\sum_{k=1}^K
\frac{
f(\theta+\mu z_k)-f(\theta-\mu z_k)
}{
2\mu
}
z_k.
\]

如果方向近似独立：

\[
\operatorname{Var}(\hat g)
\approx
O\left(\frac{1}{K}\right).
\]

并进一步结合 momentum / Adam。

### 问题

query cost：

\[
2K
\]

个 forward / step。

因此 optimizer 设计的真正目标应该从：

\[
\min T
\]

改变成：

\[
\boxed{
\min
\text{Total Forward FLOPs to Target Loss}
}
\]

---

## 6.3 Unbiased ZO Estimator — NeurIPS 2025，CCF-A

传统 two-point estimator 本质上估计 smoothing objective：

\[
\nabla f_\mu(x),
\]

而不是真正：

\[
\nabla f(x).
\]

因此存在 smoothing bias。

该工作通过 telescoping series 和随机化构造 **只使用 function evaluations 的无偏 estimator**。

### 重要意义

未来 Gradient-Free optimizer 可以把：

\[
\text{bias}
\]

和：

\[
\text{variance}
\]

作为独立设计对象：

\[
\operatorname{MSE}
=
\operatorname{Bias}^2
+
\operatorname{Variance}.
\]

这比简单固定 finite difference 更有研究空间。

---

## 6.4 FZOO — ICLR 2026

FZOO 的问题定位非常重要：

> MeZO 显存低，但是需要太多 optimizer steps / forward passes。

FZOO 使用：

- batched one-sided estimates；
- batch loss standard deviation 自适应 step；
- normalized-SGD-like update；
- GPU-friendly batching。

传统：

\[
f(\theta+\mu z),
\quad
f(\theta-\mu z)
\]

需要 two-sided evaluation。

FZOO 试图通过 one-sided / batch estimator 降低实际 forward 次数。

### 对 optimizer 研究的意义

ZO optimizer 必须同时考虑：

\[
\boxed{
\text{statistical efficiency}
\times
\text{hardware efficiency}
}
\]

算法 iteration 少并不等于 wall-clock 快。

---

# 7. 方向二：降低 Effective Dimension——Low-rank / Sparse / Subspace

这是当前最重要的一条研究线。

经典 ZO 的困难：

\[
\operatorname{Var}(\hat g)
\propto d.
\]

因此核心问题变成：

\[
\boxed{
\text{是否可以把 }d
\text{ 替换成 }
d_{\mathrm{eff}}\ll d?
}
\]

---

## 7.1 LOZO — ICLR 2025

论文：

**Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures**

假设 Transformer weight：

\[
W\in\mathbb R^{m\times n}.
\]

普通 perturbation：

\[
Z\in\mathbb R^{m\times n}.
\]

LOZO 用：

\[
Z=UV^\top,
\]

其中：

\[
U\in\mathbb R^{m\times r},
\quad
V\in\mathbb R^{n\times r},
\quad
r\ll \min(m,n).
\]

因此搜索方向从完整 \(mn\) 维降到 low-rank structure。

### 启示

不是：

\[
\text{模型参数少},
\]

而是：

\[
\boxed{
\text{寻找真正有用的 optimization subspace}
}
\]

---

## 7.2 SubZero — ICCV 2025，CCF-A

SubZero 使用 random low-rank subspace。

论文理论上证明其 estimator：

- 比传统 ZO 方差更低；
- 可以更接近 backprop gradient；
- 保持 convergence guarantee。

这进一步支持：

\[
d_{\mathrm{eff}}
\ll d
\]

这一 LLM fine-tuning 假设。

---

## 7.3 Sparse MeZO — NeurIPS 2025，CCF-A

Sparse MeZO 发现：

> ZO estimation error 加到 magnitude 较大的参数上造成的伤害更明显。

因此只对一部分参数使用 ZO perturbation。

本质：

\[
z_i
=
\begin{cases}
\epsilon_i, & i\in S_t,\\
0, & i\notin S_t.
\end{cases}
\]

使：

\[
|S_t|\ll d.
\]

论文报告：

- inference-level memory；
- 可在单 A100 上 fine-tune LLaMA-30B；
- 在 RTE 上相对 MeZO 有 9 个百分点 accuracy improvement；
- 约 3.5× speedup。

### 研究机会

现有 sparse mask 往往依赖：

- magnitude；
- sensitivity；
- static selection。

更值得研究的是：

\[
\boxed{
S_t
=
\text{dynamic active set}
}
\]

即训练过程中动态改变 perturbation parameters。

---

## 7.4 Transferable Static Sparsity — ICLR 2025

该方法只选择约：

\[
0.1\%
\]

的 sensitive parameters 做 ZO fine-tuning，并把其它参数量化。

作者展示 LLaMA2-7B 可在：

\[
<8\text{GB}
\]

GPU memory 下进行 ZO fine-tuning。

它说明可能存在：

\[
\boxed{
\text{跨任务共享的 optimization-sensitive subspace}
}
\]

这是很值得进一步理论化的问题。

---

## 7.5 TeZO — 2025 arXiv

TeZO 不仅考虑单步 gradient 的 low-rank：

\[
G_t\approx U_tV_t^\top,
\]

还假设不同时间：

\[
G_1,G_2,\ldots,G_T
\]

大致位于相似 subspace。

即：

\[
\boxed{\text{temporal low-rankness}}
\]

通过 tensor / CP decomposition 重用 spatial factors。

### 启示

未来 perturbation 不应该：

\[
z_t\overset{iid}{\sim}\mathcal N(0,I)
\]

每一步完全重新随机。

训练历史本身包含非常有价值的：

\[
\boxed{\text{search direction prior}}
\]

---

## 7.6 SubZero+ — 2026 arXiv

SubZero+ 进一步结合：

- layer-specific low-rank subspace；
- multi-query；
- subspace Adam；
- QR sign correction；
- large learning rate stability。

它代表一个很明显的趋势：

\[
\boxed{
\text{ZO optimizer state 不一定存储在 } \mathbb R^d,
\text{可以只存储在低维 subspace}
}
\]

这对设计新 optimizer 非常重要。

---

# 8. 方向三：Adaptive / Momentum / Preconditioned ZO

---

## 8.1 R-AdaZO — ICML 2025，CCF-A

Adam 的核心：

\[
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t,
\]

\[
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2.
\]

但在 ZO 中：

\[
\hat g_t
=
g_t+\xi_t.
\]

于是：

\[
\hat g_t^2
=
g_t^2
+
2g_t\xi_t
+
\xi_t^2.
\]

这意味着 second moment：

\[
v_t
\]

会系统性吸收 ZO noise。

R-AdaZO 的核心思想：

1. 利用 first moment 本身的 averaging 降低 gradient estimator variance；
2. 用 variance-reduced estimate 构造更可靠的 second moment；
3. 建立 variance-aware adaptive ZO analysis。

### 研究意义

这提出了一个很重要的问题：

\[
\boxed{
\text{Adam-style adaptivity 在 ZO 中应该如何重新定义？}
}
\]

直接把：

\[
g_t\rightarrow\hat g_t
\]

代入 Adam 很可能不是最优设计。

---

## 8.2 ConMeZO — AISTATS 2026，CCF-C

普通 MeZO：

\[
z_t\sim\mathcal N(0,I).
\]

即每次搜索方向是 isotropic random。

ConMeZO 利用 momentum：

\[
m_t
\]

作为 prior，把下一次 perturbation 限制在以 momentum 为中心的 cone 中。

概念上：

\[
z_t
\sim
q(z\mid m_{t-1}).
\]

而不是：

\[
z_t
\sim
q(z).
\]

### 这是一个非常值得继续做的方向

历史信息既可以用于：

\[
\text{update},
\]

也可以用于：

\[
\boxed{\text{决定下一次去哪里 query}}
\]

这和 Adam 完全不同。

对于 ZO 来说：

> optimizer 不只是在拿到 gradient estimator 后决定怎么更新；它还能够决定下一次 gradient estimator 怎么采样。

这使 ZO optimizer 的设计空间比 FO optimizer 更大。

---

# 9. 方向四：Curvature-aware 与 Layer-wise ZO

---

## 9.1 PaZO — NeurIPS 2025，CCF-A

**Preconditioned Accelerated Zeroth-Order Optimization for Fine-Tuning LLMs**

PaZO 从理论上强调 preconditioning 对 ZO 的必要性。

核心形式：

\[
d_t
=
P_t\hat g_t.
\]

如果：

\[
P_t
\approx
H_t^{-1/2},
\]

就可以减弱不同 curvature directions 导致的 conditioning 问题。

实际版本使用：

- diagonal Hessian estimate；
- moving average；
- preconditioned SPSA。

### 意义

这是从：

\[
\text{“估计一个更准的 gradient”}
\]

转向：

\[
\boxed{
\text{“改变 search geometry”}
}
\]

---

## 9.2 HELENE — EMNLP 2025，CCF-B

HELENE 引入：

- diagonal Hessian estimation；
- layer-wise clipping；
- gradient annealing。

其核心观察：

\[
H_1,
H_2,
\ldots,H_L
\]

不同 layer 的 curvature 差别很大。

因此统一：

\[
\eta
\]

和统一 clipping threshold 并不合理。

应该：

\[
\theta_{l,t+1}
=
\theta_{l,t}
-
\eta_{l,t}
P_{l,t}\hat g_{l,t}.
\]

论文在 RoBERTa-large 和 OPT-1.3B 上报告：

- 相对 MeZO 最高约 20× speedup；
- 平均约 1.5% accuracy improvement。

---

## 9.3 DiZO — NeurIPS 2025，CCF-A

DiZO 比较 first-order 和 zeroth-order 的 layer-wise update magnitude。

发现：

> ZO 往往给不同 layer 产生较同质的 update magnitude，而 FO training 中不同 layer 的更新尺度有明显差异。

因此使用 layer-wise divergence 来 rescale ZO update。

其本质可以写成：

\[
d_{l,t}
=
\alpha_{l,t}
\hat g_{l,t}.
\]

论文报告 GPU-hours 最高降低约 48%。

### 很重要的趋势

2025 年后越来越多工作都在从：

\[
\boxed{\text{global ZO}}
\]

转向：

\[
\boxed{\text{layer-wise / block-wise ZO}}
\]

原因很简单：

Transformer 参数本身高度结构化，不应该把整个：

\[
\theta\in\mathbb R^d
\]

当成一个同质向量。

---

## 9.4 LOREN — AAAI 2026，CCF-A

LOREN：

**Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-tuning**

普通 ZO：

\[
z\sim\mathcal N(0,I).
\]

LOREN 转向 anisotropic perturbation：

\[
z\sim\mathcal N(0,\Sigma_t).
\]

其中：

\[
\Sigma_t
\]

由 low-rank block-diagonal curvature information 决定。

结合：

- Natural Evolution Strategies；
- low-rank curvature；
- RLOO variance reduction。

这意味着未来的 ZO optimizer 很可能不是：

\[
\boxed{
\text{estimate gradient}
\rightarrow
\text{precondition gradient}
}
\]

而是：

\[
\boxed{
\text{直接改变 perturbation distribution}
}
\]

即：

\[
q_t(z)
\]

本身成为 optimizer state。

这是非常值得深入研究的方向。

---

# 10. 方向五：Quantized / Hardware-aware / Distributed ZO

---

## 10.1 QuZO — EMNLP 2025，CCF-B

QuZO 面向：

\[
\text{INT4 / INT8 forward}.
\]

传统 FO low-bit training 经常需要 STE：

\[
\frac{\partial Q(w)}{\partial w}
\approx1.
\]

而 ZO 不需要：

\[
\frac{\partial Q}{\partial w}.
\]

因此理论上更适合不可微 quantization operator。

QuZO 使用 stochastic rounding 控制低精度 perturbation 的 bias。

论文报告 LLaMA-7B fine-tuning 中，相比 quantized FO 方法：

\[
2.94\times\sim5.47\times
\]

memory reduction。

---

## 10.2 QZO — ICLR 2026

QZO 进一步发现：

如果直接 perturb 量化后的离散权重：

\[
Q(w+\mu z),
\]

小 perturbation 可能完全被 quantization 抹掉。

因此它不 perturb quantized weight，而是 perturb continuous quantization scale：

\[
s
\rightarrow
s+\mu z_s.
\]

论文报告 4-bit LLM 下，相比 16-bit full fine-tuning：

\[
>18\times
\]

total memory reduction，并可以单张 24GB GPU fine-tune LLaMA-2-13B。

---

## 10.3 Distributed ZO — ACL Findings 2026

ZO 天然拥有一个系统优势：

不同方向：

\[
z_1,z_2,\ldots,z_K
\]

对应的 function evaluations 可以并行。

因此：

\[
f(\theta+\mu z_1),
\ldots,
f(\theta+\mu z_K)
\]

可以 distributed execution。

这非常适合：

\[
\boxed{
\text{data parallel}
+
\text{query parallel}
}
\]

组合。

未来 Gradient-Free optimizer 不应只研究算法 complexity：

\[
O(KT),
\]

更应该考虑：

\[
\boxed{
\text{parallel query complexity}
}
\]

---

# 11. 方向六：Gradient-Free Pretraining

这是目前最值得关注、同时最困难的问题。

现有绝大多数 LLM ZO 工作是：

\[
\boxed{\text{fine-tuning}}
\]

而不是：

\[
\boxed{\text{pretraining from scratch}}.
\]

原因之一是 pretrained model 已经位于较好的 parameter region：

\[
\theta_0
\approx
\theta^\star_{\text{useful representation}}.
\]

fine-tuning 只需要：

\[
\Delta\theta
\]

处于相对低维 task-specific subspace。

但 pretraining：

\[
\theta_0
\sim\text{random initialization}
\]

必须探索更大的空间。

因此：

\[
d_{\mathrm{eff,pretrain}}
\gg
d_{\mathrm{eff,finetune}}.
\]

---

## 11.1 KronZO — 2026

**Zeroth-order Kronecker Optimization for Pretraining Language Models**

KronZO 研究了 pretraining gradient spectrum。

作者观察到：

> pretraining 阶段 gradient information 分布在更多 singular directions 上。

因此简单 low-rank ZO：

\[
Z=UV^\top
\]

可能丢失大量信息。

KronZO 使用 Kronecker structured perturbation，在较小存储成本下实现更接近 full-rank exploration。

在 GPT-2 Small / OpenWebText 从头训练实验中：

- 优于此前 ZO baselines；
- GPU memory 较低；
- 但最终 loss 仍落后 first-order optimizer。

### 这说明

\[
\boxed{
\text{ZO pretraining 仍然是开放问题}
}
\]

而不是已经解决的问题。

---

# 12. 2026 新趋势

## 12.1 SubZero+：低维 optimizer state

SubZero+ 把：

- subspace；
- multi-query；
- Adam；

真正结合起来。

一个很关键的思想是：

如果：

\[
g_t
\approx
B_t a_t,
\]

其中：

\[
B_t\in\mathbb R^{d\times r},
\quad
a_t\in\mathbb R^r,
\quad
r\ll d,
\]

那么不需要存：

\[
m_t,v_t\in\mathbb R^d.
\]

只需要存：

\[
m_t^{(a)},v_t^{(a)}\in\mathbb R^r.
\]

因此：

\[
\boxed{
\text{adaptive optimizer state can live in subspace}
}
\]

这可能是未来 ZO optimizer 的关键结构。

---

## 12.2 ConMeZO：sampling distribution 本身就是 optimizer

传统 optimizer 的思路：

\[
g_t
\rightarrow
\text{update}.
\]

但 ZO：

\[
q_t(z)
\rightarrow
\hat g_t
\rightarrow
\text{update}.
\]

所以：

\[
q_t(z)
\]

本身就是 optimizer 的一部分。

未来可以学习：

\[
q_t(z)
=
\mathcal N(m_t,\Sigma_t)
\]

或者更结构化的：

\[
q_t(z\mid
\text{layer},
\text{history},
\text{curvature}
).
\]

---

## 12.3 FZOO：优化目标从 step complexity 转向 forward complexity

对 LLM 来说：

\[
1\text{ step}
\]

不是统一成本。

如果一个方法：

\[
K=16
\]

而另一个：

\[
K=1,
\]

比较 iteration number 没有意义。

应该比较：

\[
\boxed{
\text{Forward Passes to Target Loss}
}
\]

进一步应该比较：

\[
\boxed{
\text{FLOPs / tokens / GPU-hours to Target Loss}
}
\]

---

# 13. 邻近优化器工作：哪些思想值得迁移到 Gradient-Free

这一部分不把这些论文当成核心 Gradient-Free literature，而是把它们视作：

\[
\boxed{
\text{新 ZO optimizer 的“思想库”}
}
\]

---

## 13.1 Parameter-free Clipped Gradient Descent Meets Polyak

- **会议：** NeurIPS 2024
- **CCF：** A
- **是否 gradient-free：** 否

论文研究 parameter-free clipped GD，并提出 Inexact Polyak Stepsize。

其重要背景是：

\[
(L_0,L_1)\text{-smoothness}.
\]

与传统：

\[
\|\nabla f(x)-\nabla f(y)\|
\le
L\|x-y\|
\]

不同，\((L_0,L_1)\)-smoothness 允许局部 smoothness 随 gradient magnitude 增长。

该工作说明：

> gradient clipping 不应该只是 heuristic；可以和 adaptive stepsize / Polyak 思想形成严格优化理论。

### 如何迁移到 ZO？

ZO 中真正可观测的是：

\[
f(\theta+\mu z),
\quad
f(\theta),
\quad
f(\theta-\mu z).
\]

可以定义 directional derivative：

\[
s_t
=
\frac{
f(\theta+\mu z)-f(\theta-\mu z)
}{
2\mu
}.
\]

传统 ZO：

\[
\hat g_t=s_tz_t.
\]

可以研究：

\[
\tilde s_t
=
\operatorname{clip}(s_t,c_t)
\]

从而：

\[
\hat g_t^{\mathrm{clip}}
=
\tilde s_tz_t.
\]

真正有研究价值的问题不是固定：

\[
c_t=c,
\]

而是：

\[
\boxed{
c_t
=
\text{function-value-driven adaptive clipping threshold}
}
\]

并同时让：

\[
\eta_t
\]

parameter-free。

这可以形成：

\[
\boxed{
\text{Parameter-Free Clipped Zeroth-Order Optimizer}
}
\]

非常符合你希望“做 optimizer”的研究定位。

---

## 13.2 PowerStep

论文：

**PowerStep: Memory-Efficient Adaptive Optimization via \(\ell_p\)-Norm Steepest Descent**

- **时间：** 2026-05 arXiv
- **是否 Gradient-Free：** 否

PowerStep 的核心是：

> Adam 的 coordinate-wise adaptivity 不一定必须依赖 second moment \(v_t\)。

从 \(\ell_p\)-norm steepest descent 出发，可以得到非线性更新方向：

\[
d_i
\propto
-\operatorname{sign}(m_i)
|m_i|^{q-1},
\]

其中：

\[
\frac1p+\frac1q=1.
\]

通过对 momentum 做 power transformation，获得 adaptive behavior，而不保存 second moment。

### 为什么对 ZO 很重要？

R-AdaZO / MUZO-Adam 等方法想把 Adam 引入 ZO 时，往往需要：

\[
v_t\in\mathbb R^d.
\]

这破坏 ZO 的极致低内存优势。

如果使用 low-dimensional ZO representation：

\[
\hat g_t=B_ta_t,
\]

可以只对：

\[
a_t\in\mathbb R^r
\]

维护 momentum：

\[
m_t^{(a)}
\]

并使用 PowerStep-like transform：

\[
u_t
=
\operatorname{sign}(m_t^{(a)})
\left|m_t^{(a)}\right|^\gamma.
\]

最终：

\[
\Delta\theta_t
=
-B_tu_t.
\]

这是一条非常自然的研究方向：

\[
\boxed{
\text{Power / Norm Geometry + Subspace ZO}
}
\]

注意：这是**研究设想**，不是 PowerStep 论文已有结论。

---

## 13.3 Powell-Style Model-Based Derivative-Free Optimization

论文：

**Powell-Style Model-Based Derivative-Free Optimization with Complexity Guarantees**

- **时间：** 2026-09 arXiv
- **是否 Gradient-Free：** 是
- **当前状态：** 预印本

传统 LLM-ZO 基本只利用：

\[
f(\theta+\mu z_t),
\quad
f(\theta-\mu z_t)
\]

然后下一步就丢掉这些 function evaluations。

Powell-style DFO 的思路完全不同：

> 保存多个历史 query points，并拟合局部 surrogate model。

例如：

\[
m_t(s)
=
f(\theta_t)
+
g_t^\top s
+
\frac12s^\top H_ts.
\]

这里的：

\[
g_t,H_t
\]

不是 backprop 得到，而是通过 interpolation/function evaluations 建模。

再解 trust-region：

\[
\min_{\|s\|\le\Delta_t}
m_t(s).
\]

### 为什么不能直接用于 LLM？

因为完整：

\[
H_t\in\mathbb R^{d\times d}
\]

完全不可行。

但是它可以与 subspace 结合：

\[
s=B_tu,
\quad
u\in\mathbb R^r.
\]

只在：

\[
r\ll d
\]

空间拟合：

\[
m_t(u)
=
c_t
+
b_t^\top u
+
\frac12u^\top A_tu.
\]

这样：

\[
A_t\in\mathbb R^{r\times r}.
\]

这就产生一个非常有意思的方向：

\[
\boxed{
\text{Subspace Model-Based DFO for LLM}
}
\]

即：

> 不再每一步随机估计完整 gradient，而利用过去的 forward evaluations 在低维 active subspace 内不断建立局部优化模型。

这是目前 LLM ZO 研究中相对少见的思路。

---

# 14. 真正做“新优化器”时应重新设计哪些组件

从现有工作看，不建议再简单提出：

\[
\text{MeZO + 一个小 trick}.
\]

更适合“optimizer paper”的方法应该有统一 update rule。

一个比较好的抽象是：

\[
\boxed{
\theta_{t+1}
=
\theta_t
+
B_tu_t
}
\]

其中：

\[
B_t\in\mathbb R^{d\times r}
\]

表示当前 search geometry。

---

## 14.1 Search Space

选择：

\[
B_t.
\]

可以来自：

- random low-rank；
- sparse mask；
- Kronecker；
- historical directions；
- momentum；
- curvature；
- layer-wise blocks。

研究问题：

\[
\boxed{
\text{How should the optimizer learn its search space?}
}
\]

---

## 14.2 Query Points

在 subspace 中：

\[
u_{t,k}\in\mathbb R^r.
\]

query：

\[
f(\theta_t+\mu_tB_tu_{t,k}).
\]

方向可以：

- Gaussian；
- Rademacher；
- orthogonal；
- quasi-Monte-Carlo；
- cone sampling；
- curvature-aware sampling。

---

## 14.3 Gradient / Local Model Estimation

方法 A：

two-point gradient：

\[
\hat a_t
=
\frac1K
\sum_k
\frac{
f(\theta+\mu Bu_k)
-
f(\theta-\mu Bu_k)
}{
2\mu
}u_k.
\]

然后：

\[
\hat g_t=B_t\hat a_t.
\]

方法 B：

model-based：

\[
m_t(u)
=
c+b^\top u
+
\frac12u^\top Au.
\]

---

## 14.4 Historical State

传统 MeZO 基本无 state。

新 optimizer 可以维护：

\[
m_t^{(r)}
\in\mathbb R^r,
\]

或者：

\[
H_t^{(r)}
\in\mathbb R^{r\times r}.
\]

而不是：

\[
m_t,v_t\in\mathbb R^d.
\]

这可以实现：

\[
\boxed{
\text{adaptive optimizer without }O(d)\text{ state}
}
\]

---

## 14.5 Step Size 与 Clipping

现有 ZO 对：

\[
\eta,
\mu
\]

非常敏感。

因此可以让：

\[
\eta_t
=
\Phi(
f_t,
\Delta f_t,
\|\hat a_t\|,
\operatorname{Var}(\hat a_t)
).
\]

同时：

\[
c_t
=
\Psi(
\text{directional derivatives}
).
\]

最终：

\[
u_t
=
-\eta_t
\operatorname{clip}
(P_t\hat a_t,c_t).
\]

这是 Polyak / clipping 思想最适合进入 ZO 的位置。

---

## 14.6 Query Budget

不要固定：

\[
K_t=K.
\]

可以根据 SNR：

\[
\operatorname{SNR}_t
=
\frac{
\|\mathbb E[\hat a_t]\|^2
}{
\operatorname{Var}(\hat a_t)
}
\]

调整：

\[
K_t.
\]

例如：

\[
K_t=
\begin{cases}
1, & \text{高 SNR},\\
2, & \text{中等 SNR},\\
4\sim8, & \text{低 SNR}.
\end{cases}
\]

优化目标变成：

\[
\boxed{
\min
\sum_{t=1}^T K_t
}
\]

subject to：

\[
f(\theta_T)\le f_{\mathrm{target}}.
\]

这比固定 multi-query 更像一个真正的 optimizer。

---

# 15. 值得重点研究的 6 个选题方向

下面按“与 Gradient-Free 主线的贴合程度 + 做 optimizer 的空间”排序。

---

## 方向 1：Parameter-Free Adaptive ZO Optimizer

### 核心问题

现有 MeZO 类方法需要手动调：

\[
\eta,\quad
\mu,\quad
c,\quad
K.
\]

这些超参数之间高度耦合。

可以研究：

\[
\boxed{
\text{Parameter-Free Zeroth-Order Optimizer}
}
\]

利用：

- Polyak-style loss gap；
- directional derivative；
- online loss variance；
- trust ratio；
- adaptive clipping。

例如：

\[
s_t
=
\frac{
f(\theta+\mu z)-f(\theta-\mu z)
}{
2\mu
}.
\]

构造：

\[
\tilde s_t
=
\operatorname{clip}
(s_t,c_t),
\]

其中：

\[
c_t
\]

由历史：

\[
|s_1|,\ldots,|s_t|
\]

自动估计。

### 优点

非常符合“做 optimizer”而不是“做 fine-tuning trick”。

### 难点

ZO estimator 本身有 noise，所以 Polyak / clipping theory 不能直接照搬 FO。

---

## 方向 2：Subspace Power ZO

结合：

- LOZO / SubZero；
- R-AdaZO；
- PowerStep。

假设：

\[
\hat g_t=B_ta_t,
\quad
a_t\in\mathbb R^r.
\]

只维护：

\[
m_t
=
\beta m_{t-1}
+
(1-\beta)a_t.
\]

不存 second moment。

用 power geometry：

\[
u_t
=
\operatorname{sign}(m_t)
|m_t|^\gamma.
\]

然后：

\[
\theta_{t+1}
=
\theta_t-\eta B_tu_t.
\]

### 目标

实现：

\[
\boxed{
\text{Adam-like adaptivity}
+
O(r)\text{ optimizer state}
+
\text{gradient-free}
}
\]

这是一个很干净的 optimizer 研究问题。

---

## 方向 3：Dynamic Curvature-Aware ZO

LOREN / PaZO / HELENE 已经说明 curvature 很重要。

可以进一步让：

\[
B_t,
\Sigma_t,P_t
\]

都动态学习。

ZO 本身可以估计 directional curvature：

\[
z^\top H z
\approx
\frac{
f(\theta+\mu z)
-
2f(\theta)
+
f(\theta-\mu z)
}{
\mu^2
}.
\]

因此可从 function values 得到 Hessian sketch。

使用最近：

\[
K
\]

个方向建立：

\[
\tilde H_t.
\]

然后：

\[
P_t
\approx
(\tilde H_t+\lambda I)^{-1/2}.
\]

### 可以理解为

\[
\boxed{
\text{ZO version of Shampoo / K-FAC / Sophia}
}
\]

但不能存 full Hessian，只能：

- diagonal；
- block；
- low-rank；
- Kronecker。

---

## 方向 4：Dynamic Active Subspace ZO

当前 LOZO/SubZero 往往假设 subspace 相对固定或随机。

但训练中：

\[
\mathcal S_t
\]

可能不断变化。

因此：

\[
B_{t+1}
=
\operatorname{Update}
(B_t,
\Delta f_t,
u_t).
\]

利用过去 successful directions：

\[
u_i
\]

建立：

\[
\boxed{\text{online active subspace}}
\]

可使用：

- Oja/PCA；
- low-rank covariance；
- evolutionary search covariance；
- L-BFGS-style direction memory；
- trust-region interpolation geometry。

这个方向与 Powell-style DFO 的结合尤其自然。

---

## 方向 5：Adaptive Query ZO

当前很多工作通过增大：

\[
K
\]

减少 variance。

但：

\[
K
\]

应该是动态变量。

可以根据：

\[
\widehat{\operatorname{Var}}(s_t),
\quad
|\Delta f_t|,
\quad
\text{direction agreement},
\quad
\text{curvature}
\]

决定：

\[
K_t.
\]

训练初期可能：

\[
K_t\gg1,
\]

稳定阶段：

\[
K_t=1.
\]

也可能相反，取决于 empirical SNR。

### 评价标准

\[
\boxed{
\text{forward FLOPs to target loss}
}
\]

而不是 steps。

---

## 方向 6：Gradient-Free Pretraining Optimizer

这是难度最高、潜在价值也最高的方向。

核心问题：

\[
\boxed{
\text{How to achieve full-rank exploration with low-dimensional estimation cost?}
}
\]

可以考虑：

\[
\text{Kronecker}
+
\text{dynamic subspace}
+
\text{low-rank curvature}
+
\text{adaptive query}
\]

组合。

例如对 weight matrix：

\[
W\in\mathbb R^{m\times n},
\]

构造：

\[
Z_t
=
A_tR_tB_t^\top,
\]

其中：

- \(R_t\) 低存储随机核心；
- \(A_t,B_t\) 动态更新；
- 保持较高 effective rank。

目标不是把 gradient 强行压成 low-rank，而是：

\[
\boxed{
\text{structured full-rank exploration}
}
\]

这更适合 pretraining。

---

# 16. 推荐的研究路线与实验方案

如果目标是最终发表一个 Gradient-Free Optimizer 工作，不建议直接从 7B/30B 模型开始。

---

## Phase 1：建立 ZO optimizer benchmark

模型：

\[
\text{RoBERTa-large}
\]

和：

\[
\text{GPT-2 124M/350M}.
\]

Baseline：

1. MeZO；
2. MeZO-SVRG；
3. LOZO；
4. SubZero；
5. Sparse MeZO；
6. R-AdaZO；
7. DiZO；
8. PaZO；
9. FZOO。

First-order reference：

- SGD；
- AdamW。

---

## Phase 2：研究 estimator quality

在小模型阶段允许计算真实 gradient：

\[
g_t^{\mathrm{FO}}
=
\nabla f(\theta_t).
\]

虽然正式 optimizer 不使用它，但可以用于研究。

建议记录：

### Cosine similarity

\[
\cos(
\hat g_t,
g_t^{\mathrm{FO}}
)
=
\frac{
\hat g_t^\top g_t
}{
\|\hat g_t\|
\|g_t\|
}.
\]

### Relative error

\[
\frac{
\|\hat g_t-g_t\|
}{
\|g_t\|
}.
\]

### Direction success rate

\[
P(
f(\theta-\eta\hat g)
<
f(\theta)
).
\]

### Effective rank

对 layer gradient：

\[
G_l
\]

分析 singular value spectrum。

这样才能回答：

> 为什么你的 optimizer 更好？

而不是只报告 accuracy。

---

## Phase 3：扩展至 1B–7B

建议：

- OPT-1.3B / 2.7B；
- Pythia；
- Qwen 1.5B / 3B；
- LLaMA-family 7B。

测试：

- full parameter；
- LoRA；
- long-context；
- low-bit。

---

## Phase 4：Pretraining

首先：

\[
\text{GPT-2 Small}
\]

数据：

- OpenWebText；
- FineWeb subset；
- C4 subset。

主要指标应该是：

\[
\boxed{
\text{training loss vs tokens}
}
\]

以及：

\[
\boxed{
\text{training loss vs total FLOPs}
}
\]

而不是只看 downstream accuracy。

---

# 17. 推荐的评价指标

新的 Gradient-Free Optimizer 至少应该报告：

| 指标 | 原因 |
|---|---|
| Peak GPU Memory | ZO 的核心优势 |
| GPU-hours | 真正资源成本 |
| Forward passes | 核心 query complexity |
| Total FLOPs | 比 iteration 更公平 |
| Tokens processed | LLM training 核心横轴 |
| Steps-to-target-loss | 优化效率 |
| FLOPs-to-target-loss | 最重要指标之一 |
| Final loss / perplexity | 最终训练能力 |
| Downstream accuracy | fine-tuning quality |
| Seed variance | ZO 高噪声，需要报告 |
| Estimator variance | 解释算法机制 |
| Gradient cosine similarity | 诊断 estimator |
| Effective rank | 验证 low-rank 假设 |
| Optimizer state memory | 检查 adaptive 方法是否破坏低内存优势 |
| Throughput tokens/s | 系统效率 |

特别建议论文中画：

\[
\text{Loss}
\quad\text{vs}\quad
\text{Forward FLOPs}
\]

而不是只画：

\[
\text{Loss}
\quad\text{vs}\quad
\text{Iteration}.
\]

---

# 18. 论文阅读顺序

如果准备真正进入这个方向，建议按下面顺序精读。

## 第一层：建立基础

### 1. MeZO — NeurIPS 2023

必须完全理解：

- seed-based perturbation；
- in-place update；
- memory analysis；
- two-point ZO estimator；
- why fine-tuning works。

---

### 2. ZO-LLM Benchmark — ICML 2024

建立整体地图：

- block-wise；
- sparsity；
- hybrid；
- task alignment；
- forward gradient。

---

### 3. MeZO-SVRG — ICML 2024

重点理解：

\[
\operatorname{Var}(\hat g)
\]

如何决定 convergence。

---

## 第二层：解决维度问题

### 4. LOZO — ICLR 2025

重点：

\[
\text{low-rank estimator}.
\]

### 5. SubZero — ICCV 2025

重点：

\[
\text{random subspace + variance}.
\]

### 6. Sparse MeZO — NeurIPS 2025

重点：

\[
\text{parameter selection}.
\]

---

## 第三层：真正进入 optimizer design

### 7. R-AdaZO — ICML 2025

重点：

\[
\text{moment + variance}.
\]

### 8. PaZO — NeurIPS 2025

重点：

\[
\text{preconditioner}.
\]

### 9. DiZO — NeurIPS 2025

重点：

\[
\text{layer-wise adaptation}.
\]

### 10. HELENE — EMNLP 2025

重点：

\[
\text{curvature + clipping}.
\]

### 11. LOREN — AAAI 2026

重点：

\[
\text{anisotropic perturbation distribution}.
\]

### 12. ConMeZO — AISTATS 2026

重点：

\[
\text{history-guided directional sampling}.
\]

### 13. FZOO — ICLR 2026

重点：

\[
\text{forward query efficiency}.
\]

---

## 第四层：进入 pretraining

### 14. KronZO — 2026

重点理解：

\[
\text{为什么 fine-tuning low-rank assumptions 在 pretraining 不完全成立}.
\]

---

## 第五层：跨方向吸收 optimizer 思想

### 15. Parameter-free Clipped Gradient Descent Meets Polyak

思考：

\[
\eta_t,\mu_t,c_t
\]

如何 parameter-free。

### 16. PowerStep

思考：

\[
\boxed{
\text{adaptive ZO 是否真的需要 second moment?}
}
\]

### 17. Powell-Style Model-Based DFO

思考：

\[
\boxed{
\text{历史 function evaluations 为什么每步都要丢掉？}
}
\]

能否构建：

\[
\text{LLM subspace trust-region optimizer}.
\]

---

# 19. 研究方向优先级建议

如果目标是未来做“大模型 Gradient-Free Optimizer”，我建议优先级如下。

## 第一优先级

\[
\boxed{
\text{Adaptive / Parameter-Free Subspace ZO}
}
\]

原因：

- 直接属于 gradient-free；
- 不是简单 application；
- optimizer 味道强；
- 可以做理论；
- 可以做 LLM experiments；
- memory advantage 清晰。

---

## 第二优先级

\[
\boxed{
\text{Curvature-Aware Gradient-Free Optimizer}
}
\]

可以连接：

- PaZO；
- HELENE；
- LOREN；
- Powell trust-region；
- Hessian sketch。

---

## 第三优先级

\[
\boxed{
\text{Gradient-Free Pretraining}
}
\]

研究价值最高，但实验成本和理论难度也最大。

核心问题：

\[
\text{full-rank exploration}
+
\text{low query complexity}
+
\text{low memory}.
\]

---

## 一个值得尝试的统一框架

可以考虑：

\[
\boxed{
\text{Parameter-Free Adaptive Subspace Zeroth-Order Optimizer}
}
\]

暂时写成：

\[
\text{PF-ASZO}.
\]

每一步：

### Step 1：构造 active subspace

\[
B_t
=
\operatorname{SubspaceUpdate}
(
B_{t-1},
u_{t-1},
\Delta f_{t-1}
).
\]

### Step 2：在 subspace 中 query

\[
f(
\theta_t
+
\mu_t B_tz_{t,k}
).
\]

### Step 3：估计低维 directional gradient

\[
a_t
=
\frac1{K_t}
\sum_k
\hat s_{t,k}z_{t,k}.
\]

### Step 4：momentum

\[
m_t
=
\beta_t m_{t-1}
+
(1-\beta_t)a_t.
\]

### Step 5：adaptive geometry

可选择：

\[
P_t
\]

为：

- diagonal curvature；
- low-rank curvature；
- power transform；
- trust-region metric。

### Step 6：parameter-free clipping

\[
\tilde m_t
=
\operatorname{clip}
(
P_tm_t,
c_t
),
\]

其中：

\[
c_t
\]

从历史 directional statistics 自动获得。

### Step 7：adaptive query / step

\[
K_t
=
\Phi(
\operatorname{SNR}_t
),
\]

\[
\eta_t
=
\Psi(
f_t,
\Delta f_t,
\|\tilde m_t\|
).
\]

### Step 8：更新

\[
\theta_{t+1}
=
\theta_t
-
\eta_tB_t\tilde m_t.
\]

这个框架把：

- LOZO/SubZero 的 subspace；
- R-AdaZO 的 moment；
- LOREN/PaZO 的 curvature；
- ConMeZO 的 adaptive sampling；
- FZOO 的 query efficiency；
- Parameter-free Polyak/clipping；
- PowerStep 的 low-state adaptivity；
- Powell-style 的 history reuse；

统一到了一个真正的：

\[
\boxed{
\text{Gradient-Free Optimizer Design Problem}
}
\]

中。

---

# 20. 最终判断

近两年的 LLM Gradient-Free Optimization 已经从：

\[
\text{“如何不用 backward？”}
\]

发展到：

\[
\boxed{
\text{“如何利用大模型 loss landscape 的结构设计一个真正的 optimizer？”}
}
\]

当前几条主要路线可以概括为：

\[
\boxed{
\begin{aligned}
&\text{Variance Reduction}\\
+&\text{Effective Dimension Reduction}\\
+&\text{Adaptive Sampling}\\
+&\text{Momentum / Preconditioning}\\
+&\text{Curvature Awareness}\\
+&\text{Parameter-Free Step Control}\\
+&\text{Adaptive Query Budget}\\
+&\text{Hardware-aware Execution}
\end{aligned}
}
\]

其中最大的长期问题仍然是：

\[
\boxed{
\text{Gradient-Free Optimizer for LLM Pretraining}
}
\]

因为 fine-tuning 的成功在很大程度上受益于 pretrained representation 和低 effective dimension，而从随机初始化进行 pretraining 需要更强的 exploration ability。

因此，如果你的目标是 **“做大模型训练优化器”而不是做一个 fine-tuning 技巧**，最值得持续追踪的主问题是：

> **如何让 Gradient-Free Optimizer 在不存储反向传播 activation / full gradient / full-size adaptive states 的前提下，通过结构化搜索空间、历史信息、curvature 与 parameter-free step control，达到接近 first-order optimizer 的 compute efficiency，并最终扩展到大模型 pretraining。**

---

# 21. 参考文献与链接

## 基础工作

1. Malladi et al. **Fine-Tuning Language Models with Just Forward Passes (MeZO).** NeurIPS 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html

2. Zhang et al. **Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark.** ICML 2024.  
   https://proceedings.mlr.press/v235/zhang24ad.html

3. Gautam et al. **Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models.** ICML 2024.  
   https://proceedings.mlr.press/v235/gautam24a.html

4. Zhang et al. **DPZero: Private Fine-Tuning of Language Models without Backpropagation.** ICML 2024.  
   https://proceedings.mlr.press/v235/zhang24af.html

---

## Low-rank / Subspace / Sparsity

5. Chen et al. **Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures (LOZO).** ICLR 2025.  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ccc9d814d3dee4750debaf23061e733-Abstract-Conference.html

6. Yu et al. **Zeroth-Order Fine-Tuning of LLMs in Random Subspaces (SubZero).** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.html

7. Liu et al. **Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning.** NeurIPS 2025.  
   https://proceedings.nips.cc/paper_files/paper/2025/hash/1e5c2efbddc02c1d971e2f19ccdb07d0-Abstract-Conference.html

8. Guo et al. **Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity.** ICLR 2025.  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html

9. Sun et al. **TeZO: Empowering the Low-Rankness on the Temporal Dimension in the Zeroth-Order Optimization for Fine-tuning LLMs.** arXiv 2025.  
   https://arxiv.org/abs/2501.19057

10. Yu et al. **SubZero+: Efficient Zeroth-Order LLM Fine-Tuning via Large Learning Rates.** arXiv 2026.  
    https://arxiv.org/abs/2608.15665

---

## Adaptive / Momentum / Curvature

11. Shu et al. **Refining Adaptive Zeroth-Order Optimization at Ease (R-AdaZO).** ICML 2025.  
    https://proceedings.mlr.press/v267/shu25b.html

12. Peng et al. **MUZO: Leveraging Multiple Queries and Momentum for Zeroth-Order Fine-Tuning of Large Language Models.** EMNLP 2025.  
    https://aclanthology.org/2025.emnlp-main.432/

13. Tan et al. **Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning (DiZO).** NeurIPS 2025.  
    https://proceedings.nips.cc/paper_files/paper/2025/hash/ffd4f5a2ea6b93e9bf5af9264d568cf2-Abstract-Conference.html

14. Zhao et al. **PaZO: Preconditioned Accelerated Zeroth-Order Optimization for Fine-Tuning LLMs.** NeurIPS 2025.  
    https://proceedings.neurips.cc/paper_files/paper/2025/hash/a14193e9d9fb0b03af0b717de1cac8ac-Abstract-Conference.html

15. Zhao et al. **HELENE: Hessian Layer-wise Clipping and Gradient Annealing for Accelerating Fine-tuning LLM with Zeroth-order Optimization.** EMNLP 2025.  
    https://aclanthology.org/2025.emnlp-main.1323/

16. Seung et al. **Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-tuning (LOREN).** AAAI 2026.  
    https://ojs.aaai.org/index.php/AAAI/article/view/39715

17. Behric et al. **ConMeZO: Adaptive Descent-Direction Sampling for Gradient-Free Finetuning of Large Language Models.** AISTATS 2026.  
    https://proceedings.mlr.press/v300/behric26a.html

---

## Estimator Theory

18. Ma & Huang. **On the Optimal Construction of Unbiased Gradient Estimators for Zeroth-Order Optimization.** NeurIPS 2025.  
    https://proceedings.neurips.cc/paper_files/paper/2025/hash/7f7eade8b69c853e3137cab80df3ccf6-Abstract-Conference.html

19. Zhang et al. **Zeroth-Order Optimization Finds Flat Minima.** NeurIPS 2025.  
    https://proceedings.nips.cc/paper_files/paper/2025/hash/ebc62a3af9342eb4ebc728e5c5bc4cca-Abstract-Conference.html

20. Yue et al. **PseuZO: Pseudo-Zeroth-Order Algorithm for Training Deep Neural Networks.** NeurIPS 2025.  
    https://papers.nips.cc/paper_files/paper/2025/hash/9a9afa70eead1805f00e3a0df2a41157-Abstract-Conference.html

---

## Hybrid / System / Quantization

21. Li et al. **Addax: Utilizing Zeroth-Order Gradients to Improve Memory Efficiency and Performance of SGD for Fine-Tuning Language Models.** ICLR 2025.  
    https://proceedings.iclr.cc/paper_files/paper/2025/hash/03560f68b1238221e7c07ad01c4b47aa-Abstract-Conference.html

22. Shirkavand et al. **Bilevel ZOFO: Efficient LLM Fine-Tuning and Meta-Training.** NeurIPS 2025.  
    https://proceedings.nips.cc/paper_files/paper/2025/hash/5f999632c48f87cffb214e575581e4a9-Abstract-Conference.html

23. Zhou et al. **QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models.** EMNLP 2025.  
    https://aclanthology.org/2025.emnlp-main.271/

24. Shang et al. **Fine-tuning Quantized Neural Networks with Zeroth-order Optimization (QZO).** ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/d17ef435b405a25715898c26dab3ed67-Abstract-Conference.html

25. Dang et al. **FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed.** ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/b8d4c82b365d9361863000f8d1e86f1b-Abstract-Conference.html

26. Zhang et al. **Three Forward, One Backward: Memory-Efficient Full-Rank Fine-Tuning of Large Models via Extra Forward Passes (LMAO).** ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/d5e9cf50dc182447a916bc56d4d42942-Abstract-Conference.html

27. Wang et al. **High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing.** Findings of ACL 2026.  
    https://aclanthology.org/2026.findings-acl.2128/

---

## Pretraining

28. Allaire et al. **Zeroth-order Kronecker Optimization for Pretraining Language Models (KronZO).** SN Computer Science, 2026.  
    https://www.gerad.ca/en/papers/G-2025-44

---

## 邻近优化器思想

29. Takezawa et al. **Parameter-free Clipped Gradient Descent Meets Polyak.** NeurIPS 2024.  
    https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ebba705ffdee81e0a638c99fe066ce2-Abstract-Conference.html

30. Lu et al. **PowerStep: Memory-Efficient Adaptive Optimization via \(\ell_p\)-Norm Steepest Descent.** arXiv 2026.  
    https://arxiv.org/abs/2605.10335

31. Chaudhry et al. **Powell-Style Model-Based Derivative-Free Optimization with Complexity Guarantees.** arXiv 2026.  
    https://arxiv.org/abs/2609.09441

---

## CCF 等级来源

32. 中国计算机学会：人工智能领域推荐国际学术会议和期刊目录。  
    https://www.ccf.org.cn/Academic_Evaluation/AI/

> 注：论文录用状态、会议等级和预印本状态会随时间发生变化；进行正式投稿选题或开题报告时，建议再次核对论文最终版本、会议 proceedings 和 CCF 最新目录。
