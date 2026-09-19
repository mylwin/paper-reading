# Powell-Style Model-Based DFO with Complexity Guarantees：全局推理（过程稿）

> 本文件由 `paper-sgd-reading` 的**全局推理模式**生成：按论文行文与依赖顺序连续推导全部公式、假设与算法，不逐轮等待提问。
> 配套交互笔记：[Powell_Style_..._精读笔记.md](Powell_Style_Model_Based_Derivative_Free_Optimization_with_Complexity_Guarantees_精读笔记.md)
> 定稿见 `04-equation_problem/Powell_Style_Model_Based_Derivative_Free_Optimization_with_Complexity_Guarantees/全局推理.md`。

## 0. 论文信息与记号约定

- 作者：A. Chaudhry, K. Scheinberg, Scholar Sun。致谢处资助为 ONR award N00014-22-1-215 与 Gary C. Butler Family Foundation。
- 类型：**理论 / 理论算法**（无导数优化 DFO 的模型基信赖域复杂度分析），附实现与数值对比。
- 被扩展的前作 [11]（Chaudhry & Scheinberg, ICM 2026 论文集）给出简化版算法的复杂度；本文把几何处理扩展到 Powell 风格（保留 trial step 作样本点 + 二次模型），并把分析扩展到确定性噪声 oracle 与随机子空间含噪两种情形。
- MSC：90C30, 90C56。

**编号还原说明（重要）**：Markdown 转换丢失了论文公式的右侧编号，本文按正文交叉引用反推，下列编号与论文自述一致：

| 编号 | 内容 | 依据 |
| --- | --- | --- |
| (2.1) | 模型 $m_k$ 的定义 | 第 2 节首个展示式 |
| (2.2) | 充分 Cauchy 下降 | 正文 "From (2.2) we have $m_k(x_k)-m_k(x_k+s_k)\ge \kappa_{fcd}\Vert g_k\Vert\Delta_k/2$" |
| (2.3) | $\Vert H_k\Vert\le\kappa_{bhm}$ | 与 (2.2) 同处 Assumption 2.2 |
| (2.4) | 引理 2.3 中 $\Delta_k$ 的双侧条件 | 正文 "From (2.4) we conclude" |
| (2.5) | 引理 2.4 的单步下降 | 正文 "(2.5) can be further stated as" |
| (2.8) | $\Vert\nabla m(x)-\nabla\phi(x)\Vert\le\kappa_{eg}\Delta$ | 正文 "all we need is to ensure (2.8)" |
| (2.9) | 有限差分梯度估计 | 正文 "$g_k$ is computed via (2.9)" |
| (3.1)(3.2) | 精确插值条件与模型定义 | 第 3 节 |
| (3.3) | 含噪插值条件 | 第 3 节 |
| (4.1) | 约束最小二乘建模型 | 第 4 节 |
| (4.4) | Lagrange 多项式更新公式 | 第 4 节 |
| (5.1)–(5.9) | 子空间模型、定义 5.1 的两条件、$\tilde C_1$ | 第 5 节 |
| (5.10) | 子空间有限差分 | 正文 "$g_k$ as in (5.10)" |

## 1. 推理计划与依赖链

```
假设 1.1 + 假设 1.2
   └─(D1)(D2) 下降引理 ──┬─→ 引理 2.9 (κ_eg ⇒ κ_ef)
                          ├─→ 引理 2.4 (单步下降 C_2)
                          ├─→ (2.10) 有限差分误差界
                          └─→ 定理 3.3/3.5 的 ‖y_iᵀ∇φ − Δ̄φ(y_i)‖ 界
定义 2.1 (fully-linear) + 假设 2.2 ((2.2)(2.3))
   └─→ 引理 2.3 (小 Δ ⇒ 成功，给出 C_1)
          ├─→ 引理 2.5 (Δ_k ≥ γC_1ε)
          ├─→ 引理 2.6 (|S_ε| 界)   ┐
          ├─→ 引理 2.7 (|U_ε| 界)   ┴─→ 定理 2.8 (|S|+|U| 界)
                                                    │
第 3 节 定理 3.3 / 3.5 + 推论 3.4 / 3.6 (κ_eg, κ_ef 的显式维度依赖)
                                                    │
第 4 节 引理 4.3 (2n 步 ⇒ 14^n-poised) ┐
        引理 4.2 (Λ_0 ⇒ Λ 的迭代数)    ┴─→ 定理 4.1 (连续 M 迭代 ≤ O(n log n))
                                                    │
                          定理 2.8 + 定理 4.1 ─→ 推论 4.4 (O(n^{3/2} log n · ε^{-2}))
                                                    │
第 5 节 引理 5.2 → 引理 5.5 (C̃_1) → 引理 5.6 (−4ε_f) → 引理 5.8–5.11 → 定理 5.12
        5.2 (5.10) → 定理 5.13 (O(nε^{-2})) ；5.3 算法 4 → 推论 5.14 (O(nq log q·ε^{-2}))
                                                    │
第 6 节 假设 6.1 + 分辨率下限 σ_k ─→ 实现仍保持同一阶（代价：对数因子）
```

---

# 阶段 A：目标函数性质与复杂度目标

## 单元 A1：问题、$\epsilon$-平稳点与噪声 oracle

论文要解 $\min_{x\in\mathbb{R}^n}\phi(x)$，其中 $\phi:\mathbb{R}^n\to\mathbb{R}$ 光滑但**可以非凸**。逐符号：

- $\phi$：目标函数，**未知量**（只能靠 oracle 采样，不能求导）。
- $x$：自变量，$\mathbb{R}^n$ 中的向量，**模型参数**（算法迭代更新的就是它）。
- $n$：问题维度，**已知量**（DFO 的复杂度显式依赖 $n$，这是它区别于有导优化的核心痛点）。
- $\mathbb{R}^n$：$n$ 维实向量空间，无约束域。

终止目标是 $\Vert\nabla\phi(x_\epsilon)\Vert\le\epsilon$。逐符号：

- $\nabla\phi$：$\phi$ 的梯度，$n$ 维列向量，**未知量**（这正是不可直接访问的对象）。
- $\Vert\cdot\Vert$：全文默认 Euclidean 范数（$\ell_2$），对向量取长度、对矩阵取谱范数（最大奇异值）。
- $\epsilon$：目标平稳度阈值，**超参数**（用户给定）。
- $x_\epsilon$：算法终止时输出的点，**中间结果**。

非凸问题不可能保证收敛到全局最小，所以用梯度范数做判据；一阶平稳是必需条件而非充分条件。

**oracle 模型**：算法只能通过 $f(x)\approx\phi(x)$ 取值，且满足

$$
|f(x) - \phi(x)| \le \epsilon_f \quad \forall x
$$

逐符号：

- $f$：噪声（不精确）零阶 oracle，**已知量**（可调用，每次调用计一次复杂度）。
- $\epsilon_f > 0$：oracle 的**绝对**误差上界，**已知量/超参数**（第 5 节算法 3 直接把 $2\epsilon_f$ 写进 $\rho_k$，所以它必须是可给定的上界；论文说 "any upper estimate of it can be used"）。
- $\forall x$：逐点一致成立，**不是**随机噪声的某种分布假设；论文称之为 deterministic noise。

**复杂度目标的格式**：$\mathcal{C}_\epsilon$ 是达到 $\Vert\nabla\phi(x_\epsilon)\Vert\le\epsilon$ 所需的 oracle 调用总次数，要证

$$
\forall \epsilon > \psi(\epsilon_f), \quad \mathcal{C}_\epsilon \le \Psi(n,\epsilon)
$$

（确定性算法），或 $\mathbb{E}[\mathcal{C}_\epsilon]\le\Psi(n,\epsilon)$（随机算法）。逐符号：

- $\mathcal{C}_\epsilon$：随机/确定性算法的 oracle 调用总数，**中间结果**（要上界的对象）。
- $\psi(\epsilon_f)$：**可达到的最优平稳度门槛**，**中间结果**——因为噪声给定了下界，$\epsilon$ 不能任意小。
- $\Psi(n,\epsilon)$：复杂度界，**中间结果**。
- $\mathbb{E}[\cdot]$：期望，取在算法内部随机性（子空间抽取）上。

论文自己声明：所有此类方法 $\Psi(\cdot) = \mathcal{O}(\epsilon^{-2})$，所以全文的关注点是 $\Psi$ 对 $n$ 的依赖，以及 $\psi$ 对 $n,\epsilon_f$ 的依赖。**这一句是判断"本文净增量在哪里"的关键**：不要指望 $\epsilon$ 的阶有改进，要看 $n$ 的幂和 $\log n$。

## 单元 A2：假设 1.1、1.2 与下降引理 (D1)(D2)

**假设 1.1**（$\phi$ 下有界）：存在 $\phi^\star$ 使 $\phi(x)\ge\phi^\star$，$\forall x$。$\phi^\star$ 是**未知常数**，只用于把"函数值总下降量"写成望远镜和的右端。

**假设 1.2**（梯度 $L$-Lipschitz）：$\phi$ 连续可微且

$$
\Vert \nabla \phi(y) - \nabla \phi(x) \Vert \le L \Vert y - x \Vert \quad \forall x, y \in \mathbb{R}^n
$$

$L \ge 0$ 是**未知的模型属性常数**，**不出现在任何算法输入里**（算法 1/2/3/4/5 的输入都不含 $L$）；$L$ 只进入"$\epsilon$ 允许多小"的前提。

**(D1)(D2) 的完整推导**（交互笔记单元 1 已详证，此处以推导线形式保留，后文全部误差界的种子）：

1. 限制到线段：令 $\varphi(\tau) = \phi(x + \tau(y-x))$，$\tau\in[0,1]$。由微积分基本定理 $\varphi(1)-\varphi(0) = \int_0^1 \varphi'(\tau)d\tau$，即

$$
\phi(y) - \phi(x) = \int_0^1 \nabla \phi(x + \tau (y-x))^\top (y - x) d\tau
$$

2. 加减同一项（纯代数，依据内积对第一变元的线性）：

$$
= \int_0^1 \nabla \phi(x)^\top (y-x) d\tau + \int_0^1 \left(\nabla \phi(x + \tau (y-x)) - \nabla \phi(x)\right)^\top (y - x) d\tau
$$

3. 第一项被积函数与 $\tau$ 无关，$\int_0^1 d\tau = 1$，移项得

$$
\phi(y) - \phi(x) - \nabla \phi(x)^\top (y - x) = \int_0^1 \left(\nabla \phi(x + \tau (y-x)) - \nabla \phi(x)\right)^\top (y - x) d\tau
$$

4. 取范数 + 积分三角不等式 + Cauchy–Schwarz：左端 $\le \int_0^1 \Vert\nabla\phi(x+\tau(y-x)) - \nabla\phi(x)\Vert\cdot\Vert y-x\Vert d\tau$。
5. 代入假设 1.2（把 $x+\tau(y-x)$ 与 $x$ 视作一对点）：$\Vert\nabla\phi(x+\tau(y-x))-\nabla\phi(x)\Vert\le L\Vert\tau(y-x)\Vert = L\tau\Vert y-x\Vert$。
6. 积分 $\int_0^1 \tau d\tau = \frac{1}{2}$，得

$$
\left| \phi(y) - \phi(x) - \nabla \phi(x)^\top (y - x) \right| \le \frac{L}{2} \Vert y - x \Vert^2
$$

(D1) 是去掉绝对值的上界形式；(D2) 是带绝对值的双侧形式。**$\frac12$ 完全来自 $\int_0^1\tau d\tau$**，在 $\phi(t)=\frac{a}{2}t^2$（$L=a$）上逐点取等，故 $\frac{L}{2}$ 不可换小。

**(S1) 推论（对 (D1) 右端关于 $y$ 最小化）**：令 $y = x - \frac{1}{L}\nabla\phi(x)$，

$$
\phi(x) - \phi\left(x - \tfrac{1}{L}\nabla\phi(x)\right) \ge \frac{1}{2L}\Vert\nabla\phi(x)\Vert^2
$$

**为什么要 (D2) 而不只是 (D1)**：引理 2.9 与推论 3.4/3.6 要界的是 $|m-\phi|$（绝对值），需要把模型与函数的一阶泰勒余项都夹住，(D1) 只有单侧。

---

# 阶段 B：信赖域框架与 fully-linear 模型

## 单元 B1：式 (2.1)，模型

每次迭代 $k\in\{0,1,\ldots\}$ 构造二次模型

$$
m_k(x_k + s) = \phi(x_k) + g_k^\top s + \frac{1}{2} s^\top H_k s
$$

逐符号（按式子从左到右）：

- $m_k$：第 $k$ 步的**模型**，$\mathbb{R}^n\to\mathbb{R}$ 的二次函数，**中间结果**（每步重算）。
- $x_k$：第 $k$ 个迭代点，**模型参数**。
- $s$：从 $x_k$ 出发的**位移向量**，$s\in\mathbb{R}^n$，$\Vert s\Vert\le\Delta_k$，**中间结果**。写成 $x_k+s$ 是把自变量平移到以 $x_k$ 为中心。
- $\phi(x_k)$：常数项。论文在此写 $\phi(x_k)$；含噪时实际可取的是 $f(x_k)$（见单元 D1 与 F2 的讨论——**这是本文第一处实质性不严谨**）。
- $g_k := \nabla m_k(x_k)$：模型梯度，$n$ 维列向量，**中间结果**，是 $\nabla\phi(x_k)$ 的替身。
- $H_k := \nabla^2 m_k(x_k)$：模型 Hessian，$n\times n$ **对称**矩阵（二次型 $\frac12 s^\top H s$ 只有对称部分有意义），**中间结果**。
- $\top$：转置；$g_k^\top s$ 是内积（标量），$s^\top H_k s$ 是二次型（标量）。
- $\frac12$：使 $\nabla_s(\frac12 s^\top H s) = Hs$、$\nabla_s^2 = H$，从而 $H_k$ 恰好是模型 Hessian（无量纲修正）。

**$\rho_k$ 只用差值**：$m_k(x_k)-m_k(x_k+s_k) = g_k^\top s_k + \frac12 s_k^\top H_k s_k$ 与常数项无关，所以 $\rho_k$ 的定义对 "$\phi(x_k)$ 还是 $f(x_k)$" 不敏感；但定义 2.1 与引理 2.9 是**绝对**误差界，敏感。

## 单元 B2：定义 2.1（fully-linear 模型）

给定球 $B(x,\Delta)$，称 $m(x+s)$ 是 $\phi(x+s)$ 在 $B(x,\Delta)$ 上的 $\kappa_{ef},\kappa_{eg}$-fully-linear 模型，若

$$
\Vert \nabla m(x) - \nabla \phi(x) \Vert \le \kappa_{eg} \Delta
$$

且

$$
| m(x+s) - \phi(x+s) | \le \kappa_{ef} \Delta^2 \quad \forall \Vert s \Vert \le \Delta
$$

逐符号：

- $\Delta > 0$：**信赖域半径**，本文取 Euclidean 球半径，**模型参数**（算法每步按 $\gamma,\gamma^{-1}$ 更新它）。
- $B(x,\Delta) = \{z: \Vert z-x\Vert\le\Delta\}$：闭球；本文后文把 $\Delta$ 与 $\Delta_k$ 混用，同一个量。
- $\kappa_{eg} \ge 0$：gradient error 常数，**未知的模型属性**（由插值理论事后界定，不是算法输入）。
- $\kappa_{ef} \ge 0$：function value error 常数，同上。
- 下标 $eg$ = error-gradient、$ef$ = error-function，**记号约定**而非变量。

**为什么一个 $\Delta$ 一次、一个 $\Delta^2$**：量纲论证。$\nabla m-\nabla\phi$ 与 $\kappa_{eg}\Delta$ 都是"梯度"量纲；$m-\phi$ 是"函数值"量纲，由 (D2) 知用一阶模型近似光滑函数的函数值偏差天然是 $\frac{L}{2}\Vert s\Vert^2 = \mathcal{O}(\Delta^2)$。所以 fully-linear 的两条**不是两个独立假设**：引理 2.9 表明第二条由第一条加 (2.3) 推出。**这是本节最重要的结构性事实**：全文只需守住 $\kappa_{eg}$。

**$\Delta$ 的双重角色（本框架与经典 TR 的分岔点）**：$\Delta$ 既限制步长、又控制模型精度（$\Delta$ 越小，$\kappa_{eg}\Delta$ 越小）。经典 TR 里 $\nabla m_k = \nabla\phi$，第二重角色不存在。

## 单元 B3：算法 1 逐行

**输入**：

- 零阶 oracle $f(x)\approx\phi(x)$：**已知量**。
- 起点 $x_0$、初始半径 $\Delta_0$：**初始值**。
- $\eta_1\in(0,1)$：接受比阈值，**超参数**，实践中取接近 0（论文数值用 0.01）。
- $\eta_2 > 0$：模型梯度与半径的比例阈值，**超参数**，**全文唯一由算法自选并可随维度调整的常数**（$\eta_2$ 取常数还是 $\sqrt n$ 决定了 $n^{2}$ 还是 $n^{3/2}$）。
- $\gamma\in(0,1)$：收缩因子，**超参数**；扩张用 $\gamma^{-1} > 1$，故扩张与收缩成对（一个失败步需要若干个成功步才能补回）。
- 判定"模型是否 fully-linear"的机制：**作为黑盒输入**，本节故意不给实现（第 3、4 节才实现它）。

**主循环** $k = 0,1,2,\ldots$（无限循环，靠 $\epsilon$-平稳分析终止性）。

**第 1 行**：算 $m_k$ 与试验步 $s_k$，$s_k \approx \arg\min_{s}\{m_k(x_k+s): s\in B(0,\Delta_k)\}$。

- $B(0,\Delta_k)$：**以原点为心**的球——步长域，不是 $x_k$ 的球；与 $B(x_k,\Delta_k) = x_k + B(0,\Delta_k)$ 等价。
- $\arg\min$：取最小点的集合，$\approx$ 表示**只需近似解**，精度由假设 2.2 的 (2.2) 规定。
- $s_k$：试验步，**中间结果**；$\Vert s_k\Vert\le\Delta_k$ 由约束保证。

**第 2 行**：

$$
\rho_k = \frac{f(x_k) - f(x_k + s_k)}{m_k(x_k) - m_k(x_k + s_k)}
$$

- 分子：**实际**下降（用可观测的 $f$），**已知量**。
- 分母：**模型预测**下降，**中间结果**；由 (2.2) 在 $\Vert g_k\Vert$ 不当时为正，所以分母不为 0（$\rho_k$ 的定义要求分母 $>0$，这一点论文未显式说明，但 (2.2) 右端在 $g_k\ne 0$ 时严格为正，$g_k=0$ 时 $\rho_k$ 分子也接近 0，实际实现按 $\Vert g_k\Vert\ge\eta_2\Delta_k$ 门槛先跳过——见算法 5 第 7 行）。
- $\rho_k$：实际下降 / 预测下降，**中间结果**，是"模型可信度"的经验度量。

**第 3 行**：三分支更新 $(x_{k+1},\Delta_{k+1})$。

$$
(x_{k+1},\Delta_{k+1}) \leftarrow
\begin{cases}
(x_k + s_k, \gamma^{-1}\Delta_k) & \text{若 } \rho_k \ge \eta_1 \text{ 且 } \Vert g_k\Vert \ge \eta_2 \Delta_k \\
(x_k, \Delta_k) & \text{否则，若模型在 } B(x_k,\Delta_k) \text{ 上非 FL} \\
(x_k, \gamma \Delta_k) & \text{其余情形}
\end{cases}
$$

（上面 $\text{若/否则/且}$ 处论文用英文 `\text`，写入笔记时按规范把文字挪出公式，此处保留分枝结构；实际渲染时分支文字写在正文里。）

- 第一支 = **成功迭代**：接受步并把半径**放大**。两个条件缺一不可。
- 第二支 = **模型改进迭代**（model improving）：拒绝步、半径**不变**，去修模型几何。
- 第三支 = **失败迭代**（unsuccessful）：拒绝步并把半径**缩小**。
- $\Vert g_k\Vert\ge\eta_2\Delta_k$ 这个条件在经典 TR 里没有，首次出现于随机模型 TR 论文 [3]。作用见单元 C1 与下面的"研究审计"。

**第 4 行**：若合适则执行模型改进步。第 3 节把"是否 FL"落实为"$\mathcal{Y}_k$ 是否 $\Lambda$-poised"，第 4 节给出实现（算法 2）。

**三分支带来的分类学**（这是第 2 节计数的对象）：给定 $\epsilon > 0$，令 $K_\epsilon$ 为**第一个**满足 $\Vert\nabla\phi(x_k)\Vert\le\epsilon$ 的迭代号，并定义

$$
\mathcal{S}_\epsilon := \{k\in\{0,\ldots,K_\epsilon-1\}: k \text{ 成功}\}
$$

$$
\mathcal{M}_\epsilon := \{k\in\{0,\ldots,K_\epsilon-1\}: k \text{ 是模型改进}\}, \qquad
\mathcal{U}_\epsilon := \{k\in\{0,\ldots,K_\epsilon-1\}: k \text{ 失败}\}
$$

- $K_\epsilon$：**停时（的确定性版本）**，**中间结果**；注意求和范围到 $K_\epsilon-1$，即"达到 $\epsilon$-平稳**之前**"的所有迭代。
- $\vert\mathcal{S}_\epsilon\vert$ 等：集合基数 = 迭代次数，**中间结果**。
- 三分支互斥且穷尽 $\{0,\ldots,K_\epsilon-1\}$，故 $\vert\mathcal{S}_\epsilon\vert+\vert\mathcal{M}_\epsilon\vert+\vert\mathcal{U}_\epsilon\vert = K_\epsilon$。

## 单元 B4：假设 2.2（(2.2) 充分 Cauchy 下降 + (2.3) Hessian 有界）

**(1)** 存在 $\kappa_{fcd}\in(0,1)$ 使

$$
m_k(x_k) - m_k(x_k + s_k) \ge \frac{\kappa_{fcd}}{2} \Vert g_k \Vert \min\left\{ \frac{\Vert g_k \Vert}{\Vert H_k \Vert}, \Delta_k \right\}
$$

逐符号：

- $\kappa_{fcd}$：fraction of Cauchy decrease，**未知的算法属性常数**（取精确解或 Cauchy 点时可为 1）。
- $\Vert H_k\Vert$：模型 Hessian 的谱范数（**对矩阵取最大奇异值**，非 Frobenius）。
- $\frac{\Vert g_k\Vert}{\Vert H_k\Vert}$：**无约束最速下降的"自然步长"**——沿 $-g_k$ 方向的二次模型极小点到中心的距离（对 $\tau\mapsto m_k(x_k-\tau g_k)$ 求极小得 $\tau^\star = \frac{\Vert g_k\Vert^2}{g_k^\top H_k g_k}$，步长 $\tau^\star\Vert g_k\Vert = \frac{\Vert g_k\Vert^3}{g_k^\top H_kg_k}\ge\frac{\Vert g_k\Vert}{\Vert H_k\Vert}$）。
- $\min\{\cdot,\Delta_k\}$：自然步长超出信赖域时取边界步。
- $\frac{\kappa_{fcd}}{2}\Vert g_k\Vert\cdot(\cdot)$：Cauchy 下降量的分数版本。
- $\Vert H_k\Vert = 0$（线性模型）时约定 $\frac{\Vert g_k\Vert}{\Vert H_k\Vert} = +\infty$，$\min$ 取 $\Delta_k$。**论文未写明这一约定**，但全文多次用 $\Vert g_k\Vert\ge\max\{\kappa_{bhm},\eta_2\}\Delta_k \Rightarrow \min\{\cdot\} = \Delta_k$，隐含依赖它。

**(2)** 存在 $\kappa_{bhm} > 0$ 使 $\Vert H_k\Vert\le\kappa_{bhm}$，对所有算法 1 产生的 $x_k$。

- $\kappa_{bhm}$：bounded hessian of model，**未知的模型属性**。论文自己在定理 2.8 后说明：理想情形 $\kappa_{bhm}\sim\mathcal{O}(L)$，线性模型时可为 0，"若允许"也可以很大——**这句话是第 6 节 $K = 10^{100}$ 参数字选择的理论接口，也是审计要点**。

**为什么 Cauchy 点满足 $\kappa_{fcd}=1$**：Cauchy 点是 (2.2) 右端在 $\tau\in[0,1]$ 上精确最小化的结果，取 $s^c = -\tau^\star\Delta_k\frac{g_k}{\Vert g_k\Vert}$，代入 $m_k(x_k+s) - m_k(x_k) = -\tau\Delta_k\Vert g_k\Vert + \frac12\tau^2\Delta_k^2\frac{g_k^\top H_kg_k}{\Vert g_k\Vert^2}$ 并最小化，得下降量 $\frac{\Delta_k\Vert g_k\Vert}{2}\cdot\min\{1,\frac{\Delta_k g_k^\top H_kg_k}{\Vert g_k\Vert^3}\}\cdot\frac{\Vert g_k\Vert^2}{\Delta_k g_k^\top H_kg_k}\cdots$；标准结论（[12, §6.3.2]）为

$$
m_k(x_k)-m_k(x_k+s^c) \ge \frac{\Vert g_k\Vert^2}{2}\min\left\{\frac{\Delta_k}{\Vert g_k\Vert}, \frac{\Vert g_k\Vert}{\Vert H_k\Vert}\right\} = \frac{\Vert g_k\Vert}{2}\min\left\{\Delta_k, \frac{\Vert g_k\Vert}{\Vert H_k\Vert}\right\}
$$

即 (2.2) 取 $\kappa_{fcd}=1$。

## 单元 B5：本节的"研究审计"

- **贡献角色**：定义 2.1、假设 2.2、算法 1 全部是**已有工具**（Conn–Scheinberg–Vicente 教材 [13]；$\Vert g_k\Vert\ge\eta_2\Delta_k$ 条件来自 [3]）。本文第 2 节的新东西只有：引理 2.3 里 $C_1$ 与 $C_0=\max\{\eta_2,\kappa_{ef}\}$ 的**具体组合方式**（把噪声项 $\epsilon_f/\Delta_k^2$ 折进常数），以及推论 2.10 里 $\eta_2$ **随维度增长**这一观察。
- **$\Vert g_k\Vert\ge\eta_2\Delta_k$ 的真实作用**：取消 criticality step。经典 DFO 复杂度分析（[15]）需要额外的"临界步"专门把 $\Delta$ 压到与 $\Vert g_k\Vert$ 同阶；这个门槛把半径与模型梯度绑定，使 $\Vert g_k\Vert$ 变小时 $\Delta$ 自动跟上。**代价**：$\eta_2$ 出现在 $C_1,C_2$ 里，直接改变 $n$ 的幂（$n^2$ vs $n^{3/2}$）。
- **最强假设**：假设 2.2 的 (2.3)（Hessian 一致有界）与"存在 FL 判定机制"这两条被当作输入黑盒；本文的贡献正是把后者实现为 $\Lambda$-poisedness 并给出计数（第 3、4 节）。
- **可证伪问题**：$\eta_2$ 的最优取值是 $\sqrt n$ 吗？由 $C_1^{-1}\ge\max\{\eta_2,\kappa_{bhm},\frac{2\kappa_{ef}+C_0}{(1-\eta_1)\kappa_{fcd}}\}+\kappa_{eg}$ 与 $C_2\propto\eta_2\min\{\eta_2/\kappa_{bhm},1\}$，对 $\eta_2$ 求极小可判断 $\sqrt n$ 是否就是最优解（阶段 D2 处理）。

---

# 阶段 C：$C_1$、$C_2$ 与两类迭代的计数

## 单元 C1：引理 2.3（小 $\Delta_k$ 蕴含成功步）——$C_1$ 的来历

**命题**：在假设 2.2 下，若 $m_k$ 是 $\kappa_{ef},\kappa_{eg}$-fully-linear 且

$$
\sqrt{\frac{2 \epsilon_f}{C_0}} \le \Delta_k \le C_1 \Vert \nabla \phi(x_k) \Vert
\qquad \text{其中} \qquad
C_1 = \left(\max\left\{\eta_2, \kappa_{bhm}, \frac{2\kappa_{ef} + C_0}{(1-\eta_1)\kappa_{fcd}}\right\} + \kappa_{eg}\right)^{-1}
$$

$C_0$ 为任意常数、取 $C_0 = \max\{\eta_2,\kappa_{ef}\}$，则 $\rho_k\ge\eta_1$ 且 $\Vert g_k\Vert\ge\eta_2\Delta_k$，故迭代 $k$ 成功、$x_{k+1} = x_k+s_k$。这就是 (2.4)。

逐符号（只解释新出现的）：

- $\sqrt{2\epsilon_f/C_0}$：**噪声地板**，**中间结果**。$\Delta_k$ 必须**不小于**它，否则模型误差被噪声淹没。
- $C_0$：人为引入的**辅助常数**，取 $\max\{\eta_2,\kappa_{ef}\}$，作用是把 $\epsilon_f/\Delta_k^2$ 折成常数。
- $C_1$：**半径–梯度比例常数**，$\le$ 各分量的倒数，**中间结果**。

**推导（逐步，不跳）**：

**第 1 步（$\Delta_k$ 上界 $\Rightarrow$ 两个门槛）**。由 $\Delta_k\le C_1\Vert\nabla\phi(x_k)\Vert$ 得 $C_1^{-1}\Delta_k\le\Vert\nabla\phi(x_k)\Vert$。由 $C_1^{-1}$ 的定义，

$$
C_1^{-1} \ge \max\{\kappa_{bhm},\eta_2\} + \kappa_{eg}
$$

（因为 $\max$ 里含 $\eta_2$ 与 $\kappa_{bhm}$ 两项）。于是

$$
(\max\{\kappa_{bhm},\eta_2\} + \kappa_{eg})\Delta_k \le \Vert \nabla \phi(x_k) \Vert
$$

由 fully-linear 的第一条与三角不等式：$\Vert\nabla\phi(x_k)\Vert = \Vert g_k + (\nabla\phi(x_k)-g_k)\Vert \le \Vert g_k\Vert + \Vert\nabla\phi(x_k)-g_k\Vert \le \Vert g_k\Vert + \kappa_{eg}\Delta_k$。夹在一起：

$$
(\max\{\kappa_{bhm},\eta_2\} + \kappa_{eg})\Delta_k \le \Vert g_k \Vert + \kappa_{eg}\Delta_k
\;\Rightarrow\;
\max\{\kappa_{bhm}, \eta_2\} \Delta_k \le \Vert g_k \Vert
$$

两侧消去同一个 $\kappa_{eg}\Delta_k$（实数不等式的合法操作）。**结论 1**：$\Vert g_k\Vert\ge\eta_2\Delta_k$（成功的第一条件）成立；**结论 2**：$\Vert g_k\Vert\ge\kappa_{bhm}\Delta_k$。

**第 2 步（(2.2) 的 $\min$ 塌缩）**。由结论 2 与 (2.3)：$\Vert H_k\Vert\le\kappa_{bhm}\le\frac{\Vert g_k\Vert}{\Delta_k}$，取倒数乘 $\Vert g_k\Vert$ 得 $\frac{\Vert g_k\Vert}{\Vert H_k\Vert}\ge\Delta_k$，故 $\min\{\frac{\Vert g_k\Vert}{\Vert H_k\Vert},\Delta_k\} = \Delta_k$，代入 (2.2)：

$$
m_k(x_k) - m_k(x_k + s_k) \ge \frac{\kappa_{fcd}}{2} \Vert g_k \Vert \Delta_k
$$

论文直接写"thus using ... we have $\ge \kappa_{fcd}\Vert g_k\Vert\Delta_k/2$"，**$\min$ 为何塌缩这一步被略过**，此处补齐。

**第 3 步（$\rho_k$ 的恒等分解）**。分子加减 $m_k(x_k)$、$m_k(x_k+s_k)$、$\phi(x_k)$、$\phi(x_k+s_k)$（纯代数）：

$$
\rho_k = \frac{m_k(x_k) - m_k(x_k+s_k)}{\text{den}} + \frac{(\phi - m_k)(x_k) - (\phi - m_k)(x_k+s_k)}{\text{den}} + \frac{(f-\phi)(x_k) - (f-\phi)(x_k+s_k)}{\text{den}}
$$

其中 $\text{den} := m_k(x_k)-m_k(x_k+s_k) > 0$。第一项恰为 1。

**关键事实（论文未点明）**：$m_k(x_k) = \phi(x_k)$ 按 (2.1) **精确成立**（模型以中心值为常数项），所以 $(\phi-m_k)(x_k) = 0$，第二项只剩 $-\frac{(\phi-m_k)(x_k+s_k)}{\text{den}}$，其绝对值 $\le\frac{\kappa_{ef}\Delta_k^2}{\text{den}}$（用 fully-linear 第二条，$\Vert s_k\Vert\le\Delta_k$）。**这就是为什么论文只写了一个 $\kappa_{ef}\Delta_k^2$ 而不是两个**。第三项 $\ge -\frac{2\epsilon_f}{\text{den}}$。合计（论文排版本把第三项写成加号，是排版符号错误；取绝对值界后不影响结论）：

$$
\rho_k \ge 1 - \frac{\kappa_{ef}\Delta_k^2}{\text{den}} - \frac{2 \epsilon_f}{\text{den}}
$$

**第 4 步（代入第 2 步的下界）**。$\text{den}\ge\frac{\kappa_{fcd}}{2}\Vert g_k\Vert\Delta_k$ 且右端为正，取倒数反向：

$$
\rho_k \ge 1 - \frac{\kappa_{ef}\Delta_k^2 + 2\epsilon_f}{\kappa_{fcd}\Vert g_k\Vert\Delta_k / 2} = 1 - \frac{(2\kappa_{ef} + 2\epsilon_f/\Delta_k^2)\Delta_k}{\kappa_{fcd}\Vert g_k \Vert \Delta_k \cdot \Delta_k / 2}
$$

把 $\Vert g_k\Vert$ 换成 $\Vert\nabla\phi(x_k)\Vert$：由三角不等式与 fully-linear，$\Vert g_k\Vert\ge\Vert\nabla\phi(x_k)\Vert-\kappa_{eg}\Delta_k$（与第 1 步同一条不等式的另一端），得论文的形式

$$
\rho_k \ge 1 - \frac{(2 \kappa_{ef} + 2 \epsilon_f / \Delta_k^2) \Delta_k}{\kappa_{fcd} \left( \Vert \nabla \phi(x_k) \Vert - \kappa_{eg} \Delta_k \right)}
$$

**第 5 步（用噪声地板吸收 $\epsilon_f$）**。由 (2.4) 的**下界** $\Delta_k^2\ge 2\epsilon_f/C_0$ 得 $\frac{2\epsilon_f}{\Delta_k^2}\le C_0$，于是

$$
\rho_k \ge 1 - \frac{(2 \kappa_{ef} + C_0) \Delta_k}{\kappa_{fcd} \left( \Vert \nabla \phi(x_k) \Vert - \kappa_{eg} \Delta_k \right)}
$$

**第 6 步（用 $\Delta_k$ 的上界收尾）**。要 $\rho_k\ge\eta_1$，只需

$$
\frac{(2 \kappa_{ef} + C_0)\Delta_k}{\kappa_{fcd}(\Vert \nabla \phi(x_k) \Vert - \kappa_{eg} \Delta_k)} \le 1 - \eta_1
$$

分母为正需 $\Vert\nabla\phi(x_k)\Vert > \kappa_{eg}\Delta_k$（下面由 $C_1^{-1}\ge\kappa_{eg}$ 保证）。整理（两边乘正分母、除以 $1-\eta_1 > 0$）：

$$
\Vert \nabla \phi(x_k) \Vert \ge \left( \frac{2 \kappa_{ef} + C_0}{(1-\eta_1)\kappa_{fcd}} + \kappa_{eg} \right) \Delta_k
$$

而这正是 $C_1^{-1}\ge\frac{2\kappa_{ef}+C_0}{(1-\eta_1)\kappa_{fcd}}+\kappa_{eg}$ 与 $\Delta_k\le C_1\Vert\nabla\phi(x_k)\Vert$ 的直接推论。$\blacksquare$

**论文此处的一处不精确**：证明末句写"the last inequality follows from (2.4) since $\Vert\nabla\phi(x_k)\Vert\ge(\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}+\kappa_{eg})\Delta_k$"，**漏掉了 $C_0$**；按第 6 步需要的是 $\frac{2\kappa_{ef}+C_0}{(1-\eta_1)\kappa_{fcd}}$。$C_1$ 的定义里是含 $C_0$ 的，所以只是末句笔误，结论成立。

**为什么 $C_0$ 要取 $\max\{\eta_2,\kappa_{ef}\}$**：既要 $\ge\kappa_{ef}$（使 $2\kappa_{ef}+C_0 \le 3\kappa_{ef}$ 型吸收），又要 $\ge\eta_2$（使引理 2.4 之后的地板比较 $\sqrt{4\epsilon_f/C_2}$ 统一，见单元 C3）。

## 单元 C2：引理 2.4（成功步蕴含函数值下降）——$C_2$ 的来历

**命题**：假设 1.2 与 2.2 下，若 $k$ 成功，则

$$
\phi(x_k) - \phi(x_{k+1}) \ge C_2 \Delta_k^2 - 2 \epsilon_f,
\qquad
C_2 = \frac{\eta_1 \eta_2 \kappa_{fcd}}{2} \min\left\{ \frac{\eta_2}{\kappa_{bhm}}, 1 \right\}
$$

否则 $x_{k+1} = x_k$，下降量为 0。这就是 (2.5)。

**推导**：

1. 成功 $\Rightarrow x_{k+1} = x_k + s_k$ 且 $\rho_k\ge\eta_1$、$\Vert g_k\Vert\ge\eta_2\Delta_k$。
2. 由 $\rho_k\ge\eta_1$（用**原始**定义，不含 $+2\epsilon_f$）：$f(x_k)-f(x_k+s_k) = \rho_k\cdot\text{den} \ge \eta_1\,\text{den}$。
3. 噪声换算：$f(x_k)\ge\phi(x_k)-\epsilon_f$，$f(x_k+s_k)\le\phi(x_k+s_k)+\epsilon_f$，相减得 $\phi(x_k)-\phi(x_k+s_k) \ge f(x_k)-f(x_k+s_k) - 2\epsilon_f$。
4. 结合 2、3：$\phi(x_k)-\phi(x_{k+1}) \ge \eta_1\,\text{den} - 2\epsilon_f$。
5. 由 (2.2) 与 $\Vert g_k\Vert\ge\eta_2\Delta_k$，且 $u\mapsto u\min\{u/\kappa_{bhm},\Delta_k\}$ 在 $u\ge0$ 上单调增：

$$
\text{den} \ge \frac{\kappa_{fcd}}{2} \eta_2 \Delta_k \min\left\{ \frac{\eta_2 \Delta_k}{\kappa_{bhm}}, \Delta_k \right\} = \frac{\eta_2 \kappa_{fcd}}{2} \min\left\{ \frac{\eta_2}{\kappa_{bhm}}, 1 \right\} \Delta_k^2
$$

（提出 $\Delta_k$：$\Delta_k\min\{\frac{\eta_2\Delta_k}{\kappa_{bhm}},\Delta_k\} = \Delta_k^2\min\{\frac{\eta_2}{\kappa_{bhm}},1\}$。）乘 $\eta_1$ 即得 $\eta_1\text{den}\ge C_2\Delta_k^2$。$\blacksquare$

**$\tau$ 改写**：附加条件 $\Delta_k \ge \sqrt{\frac{2\epsilon_f}{\tau C_2}}$，$\tau\in(0,1)$ 时 $2\epsilon_f \le \tau C_2\Delta_k^2$，故

$$
\phi(x_k) - \phi(x_{k+1}) \ge (1-\tau) C_2 \Delta_k^2
$$

取 $\tau = \frac12$（论文"Henceforth"处固定）：条件变 $\Delta_k \ge \sqrt{4\epsilon_f/C_2}$，结论变 $\ge\frac{C_2}{2}\Delta_k^2$。论文说"由于 $C_0\ge\eta_2\ge C_2$，$\Delta_k$ 的下界化简为 $\sqrt{4\epsilon_f/C_2}$"——核验 $\eta_2\ge C_2$：$C_2 = \eta_2\cdot\frac{\eta_1\kappa_{fcd}}{2}\min\{\eta_2/\kappa_{bhm},1\}\le\eta_2$，因为 $\eta_1,\kappa_{fcd}\le1$、$\min\{\cdot,1\}\le1$ ✓。核验 $\sqrt{4\epsilon_f/C_2}\ge\sqrt{2\epsilon_f/C_0}$：$\iff C_0 \ge C_2/2$ ✓（$C_0\ge\eta_2\ge C_2$）。**所以两个地板（引理 2.3 需要的与引理 2.4 需要的）被同一个 $\Delta_k\ge\sqrt{4\epsilon_f/C_2}$ 一并满足**——这就是 $C_0$ 取 $\max\{\eta_2,\kappa_{ef}\}$ 的真实目的。

## 单元 C3：引理 2.5（$\Delta_k$ 的下界）

**命题**：对任意 $\epsilon > \sqrt{\frac{4 \epsilon_f}{\gamma^2 C_2 C_1^2}}$，若 $\Delta_0 > \gamma C_1 \epsilon$，则对所有 $k\in\{0,\ldots,K_\epsilon-1\}$ 有 $\Delta_k \ge \gamma C_1 \epsilon$。

**推导（反证 + 归纳，补齐论文省略的前提）**：

设 $\bar k$ 为**第一个**使 $\Delta_{\bar k} < \gamma C_1\epsilon$ 的指标（若不存在则命题已成立）。因 $\Delta_0 > \gamma C_1\epsilon$，$\bar k \ge 1$，且 $\Delta_{\bar k - 1} \ge \gamma C_1 \epsilon$。$\Delta$ 能变小**只有**在失败分支（第三支，乘 $\gamma$）；成功分支放大、模型改进分支不变。故迭代 $\bar k - 1$ 失败，且

$$
\gamma^2 C_1 \epsilon \le \gamma \Delta_{\bar k - 1} = \Delta_{\bar k} < \gamma C_1 \epsilon
\;\Rightarrow\;
\Delta_{\bar k - 1} < C_1 \epsilon
$$

因为 $\bar k - 1 \le K_\epsilon - 1$，有 $\Vert\nabla\phi(x_{\bar k-1})\Vert > \epsilon$，于是 $\Delta_{\bar k-1} < C_1\epsilon < C_1\Vert\nabla\phi(x_{\bar k-1})\Vert$：(2.4) 的**上界**成立。

再核对 (2.4) 的**下界**：$\Delta_{\bar k-1} \ge \gamma\Delta_{\bar k}\cdot\gamma^{-1}$… 直接用 $\Delta_{\bar k-1}\ge\gamma C_1\epsilon > \gamma\cdot\sqrt{4\epsilon_f/(\gamma^2C_2C_1^2)} = \sqrt{4\epsilon_f/C_2} \ge \sqrt{2\epsilon_f/C_0}$ ✓（第一个 $>$ 用了 $\epsilon$ 的下界假设，最后一个 $\ge$ 见单元 C2 末尾）。

**还差一个前提**：引理 2.3 需要 $m_k$ 是 fully-linear。若 $m_{\bar k-1}$ 非 FL，则第 3 行走第二支，$\Delta$ **不变**，与"$\bar k-1$ 失败（$\Delta$ 缩小）"矛盾 ✓。故 $m_{\bar k-1}$ 必为 FL，引理 2.3 全部前提满足，推出迭代 $\bar k - 1$ **成功**——与"失败"矛盾。$\blacksquare$

论文原文只用两行带过（"Thus no iteration can be unsuccessful when $\Delta_k \le C_1\epsilon$"），**FL 前提与噪声地板的核对被完全省略**，此处补齐。

## 单元 C4：引理 2.6（成功迭代数的界）

$$
\vert \mathcal{S}_\epsilon \vert \le \frac{2 (\phi(x_0) - \phi^\star)}{C_2 (\gamma C_1 \epsilon)^2}
$$

**推导**：望远镜求和。$\phi(x_0) - \phi^\star \ge \phi(x_0) - \phi(x_{K_\epsilon}) = \sum_{k=0}^{K_\epsilon-1}(\phi(x_k)-\phi(x_{k+1}))$（假设 1.1 给出 $\phi(x_{K_\epsilon})\ge\phi^\star$；等号是逐项相消的恒等式）。由引理 2.4 与 $\tau=\frac12$：成功项贡献 $\ge\frac{C_2}{2}\Delta_k^2$，非成功项贡献 $\ge 0$（$x_{k+1}=x_k$ 时下降量恰为 0），故

$$
\phi(x_0) - \phi^\star \ge \sum_{k\in\mathcal{S}_\epsilon} \frac{C_2}{2} \Delta_k^2 \ge \frac{\vert \mathcal{S}_\epsilon \vert}{2} C_2 (\gamma C_1 \epsilon)^2
$$

最后一步用引理 2.5 的 $\Delta_k \ge \gamma C_1\epsilon$（对一切 $k<K_\epsilon$ 成立）。解出 $\vert\mathcal{S}_\epsilon\vert$。$\blacksquare$

**注意符号错误**：论文在推论 2.10 的证明里把这一界写成 $f(x_0) - f^\star$，应为 $\phi(x_0)-\phi^\star$（$f^\star$ 全文未定义）。阶不受影响。

## 单元 C5：引理 2.7（失败迭代数的界）

$$
\left\vert \mathcal{U}_\epsilon \right\vert \le \left\vert \mathcal{S}_\epsilon \right\vert + \left\lceil \log_\gamma \frac{C_1 \epsilon}{\Delta_0} \right\rceil
$$

**推导**：

**第 1 步（$\Delta$ 的显式表达）**：$\Delta$ 的每次更新是乘 $\gamma^{-1}$（成功）、$\gamma$（失败）或 $1$（模型改进）。故

$$
\Delta_{K_\epsilon} = \gamma^{-\vert \mathcal{S}_\epsilon \vert} \gamma^{\vert \mathcal{U}_\epsilon \vert} \Delta_0
$$

**第 2 步（关键：迭代 $K_\epsilon-1$ 必成功）**：$x$ **只在成功时改变**。$K_\epsilon$ 是第一个 $\Vert\nabla\phi(x_k)\Vert\le\epsilon$ 的指标，故 $\Vert\nabla\phi(x_{K_\epsilon-1})\Vert > \epsilon \ge \Vert\nabla\phi(x_{K_\epsilon})\Vert$，从而 $x_{K_\epsilon}\ne x_{K_\epsilon-1}$，只能是成功步。（论文写"the $K_\epsilon-1$-th iteration must be successful"但没给理由，理由是这一条。）

**第 3 步（下界）**：由第 2 步 $\Delta_{K_\epsilon} = \gamma^{-1}\Delta_{K_\epsilon-1}$，配合引理 2.5 的 $\Delta_{K_\epsilon-1}\ge\gamma C_1\epsilon$，得 $\Delta_{K_\epsilon}\ge C_1\epsilon$ ✓（这正是论文"$\Delta_{K_\epsilon} = \cdots \ge C_1\epsilon$"的完整论证）。

**第 4 步（取对数）**：$\gamma^{\vert\mathcal{U}_\epsilon\vert - \vert\mathcal{S}_\epsilon\vert}\Delta_0 \ge C_1\epsilon$，即

$$
\gamma^{\vert\mathcal{U}_\epsilon\vert - \vert\mathcal{S}_\epsilon\vert} \ge \frac{C_1 \epsilon}{\Delta_0}
$$

$\gamma\in(0,1)$，$\log_\gamma$ 严格**递减**，取对数**翻转**不等号：

$$
\left\vert \mathcal{U}_\epsilon \right\vert - \left\vert \mathcal{S}_\epsilon \right\vert \le \log_\gamma \frac{C_1 \epsilon}{\Delta_0}
$$

因 $\Delta_0 > \gamma C_1\epsilon \Rightarrow \frac{C_1\epsilon}{\Delta_0} < \gamma^{-1}$，该对数 $> -1$ 有限；左边是整数，故 $\le\lceil\cdot\rceil$。$\blacksquare$

**结构意义**：引理 2.7 是**纯粹的半径记账**，与 $\phi$、噪声、模型精度全无关，只依赖"$\Delta$ 只能按 $\gamma^{\pm1}$ 变化"+"$\Delta$ 有下界"。所以第 5 节随机子空间版本里，这一论证失效（半径下界不再自动成立），必须显式加 $\Delta_{min}$。

## 单元 C6：定理 2.8（合并界）

在引理 2.5/2.6/2.7 的条件下（$\epsilon > \sqrt{4\epsilon_f/(\gamma^2C_2C_1^2)}$，$\Delta_0 > \gamma C_1\epsilon$）：

$$
\left\vert \mathcal{S}_\epsilon \right\vert + \left\vert \mathcal{U}_\epsilon \right\vert \le \frac{4 (\phi(x_0) - \phi^\star)}{C_2 (\gamma C_1 \epsilon)^2} + \left\lceil \log_\gamma \frac{C_1 \epsilon}{\Delta_0} \right\rceil
$$

**推导**：$\vert\mathcal{S}\vert+\vert\mathcal{U}\vert \le \vert\mathcal{S}\vert + \vert\mathcal{S}\vert + \lceil\log\rceil = 2\vert\mathcal{S}\vert + \lceil\log\rceil$，代入引理 2.6 的 $2\vert\mathcal{S}\vert \le \frac{4(\phi(x_0)-\phi^\star)}{C_2(\gamma C_1\epsilon)^2}$。$\blacksquare$

**这个界不含 $\vert\mathcal{M}_\epsilon\vert$** —— 第 2 节故意把"如何保证 FL"留白，所以定理 2.8 只数成功+失败。$\vert\mathcal{M}_\epsilon\vert$ 的界是第 4 节定理 4.1 的任务，二者在推论 4.4 合并。**理解全文复杂度的钥匙**：

$$
\mathcal{C}_\epsilon \;\lesssim\; \underbrace{1\cdot(\vert\mathcal{S}_\epsilon\vert + \vert\mathcal{U}_\epsilon\vert)}_{\text{定理 2.8}} + \underbrace{2\cdot\vert\mathcal{M}_\epsilon\vert}_{\text{定理 4.1}} , \qquad \vert\mathcal{M}_\epsilon\vert \le (\vert\mathcal{S}_\epsilon\vert+\vert\mathcal{U}_\epsilon\vert + 1)\times(\text{每次连续 M 迭代的上界})
$$

## 单元 C7：常数审计（$C_1$、$C_2$ 的敏感度）

取 $\gamma,\eta_1,\kappa_{fcd}$ 为接近 1 的固定常数（论文建议 0.9 量级；数值实验 $\eta_1 = 0.01$、$\gamma_{dec}=0.8$、$\gamma_{inc}=1.3$ 与"接近 1"不符，见阶段 H）：

- $C_1^{-1} = \max\{\eta_2,\kappa_{bhm},\frac{2\kappa_{ef}+C_0}{(1-\eta_1)\kappa_{fcd}}\} + \kappa_{eg}$，$C_2 = \frac{\eta_1\eta_2\kappa_{fcd}}{2}\min\{\eta_2/\kappa_{bhm},1\}$。
- 界随 $\epsilon^{-2}$：$\frac{1}{C_2 C_1^2}$。**关键**：$C_1$ 里 $\kappa_{ef},\kappa_{eg}$ 与 $\eta_2$ 走 $\max$，**不是相加**，所以只要 $\kappa_{eg}$ 主导（$\kappa_{eg}\gg\eta_2$），$\eta_2$ 增大只会通过 $C_2$ **改善**界。这正是推论 2.10 用 $\eta_2 = \sqrt n$ 的机理。
- $(1-\eta_1)$ 在分母：$\eta_1\to1$ 时 $C_1\to0$，界爆炸。所以"接受阈值越严、复杂度越差"是**结构性**的，不是常数细节。

---

# 阶段 D：$\kappa_{eg} \Rightarrow \kappa_{ef}$ 与有限差分基线

## 单元 D1：引理 2.9

**命题**：假设 1.2 与 2.2 下，若 $\Vert\nabla m(x)-\nabla\phi(x)\Vert\le\kappa_{eg}\Delta$（即 (2.8)），则 $m$ 是 $\kappa_{ef},\kappa_{eg}$-fully-linear，且

$$
\kappa_{ef} = \kappa_{eg} + \frac{L + \kappa_{bhm}}{2}
$$

**推导**：取 $\Vert s\Vert\le\Delta$。用 (D2) 展开 $\phi$、用 (2.1) 展开 $m$：

$$
\phi(x+s) = \phi(x) + \nabla\phi(x)^\top s + r_\phi, \qquad |r_\phi| \le \frac{L}{2} \Vert s \Vert^2
$$

$$
m(x+s) = m(x) + \nabla m(x)^\top s + \frac{1}{2} s^\top H s
$$

（这里 $\nabla m(x) = g$、$\nabla^2 m = H$，且 $\frac12 s^\top Hs$ 的二次型误差正是 $|\frac12 s^\top H s| \le \frac{\kappa_{bhm}}{2}\Vert s\Vert^2$，用了 $H$ 对称 + 谱范数定义 $|v^\top H v|\le\Vert H \Vert\Vert v \Vert^2$。）两式相减：

$$
m(x+s) - \phi(x+s) = \underbrace{[m(x) - \phi(x)]}_{(*)} + [\nabla m(x) - \nabla \phi(x)]^\top s + \frac{1}{2} s^\top H s - r_\phi
$$

- $(*)$：**必须为 0**，否则要额外加一项 $\epsilon_f$。这正是"$m(x)=\phi(x)$"的作用。
- 第二项用 Cauchy–Schwarz：$\le\kappa_{eg}\Delta\Vert s\Vert \le \kappa_{eg}\Delta^2$。
- 第三、四项：$\le \frac{\kappa_{bhm} + L}{2}\Vert s \Vert^2 \le \frac{\kappa_{bhm}+L}{2}\Delta^2$。

合计 $|m(x+s)-\phi(x+s)| \le (\kappa_{eg} + \frac{L+\kappa_{bhm}}{2})\Delta^2$。$\blacksquare$

**审计（本稿发现的实质性问题）**：论文在含噪情形按 (3.3) 用 $f$ 造模型，此时 $m(x_k) = f(x_k)$，$(*) = f(x_k)-\phi(x_k)$，$|(*)|\le\epsilon_f$。要仍写成 $\kappa_{ef}\Delta^2$ 需要 $\epsilon_f \le \frac{L+\kappa_{bhm}}{2}\Delta^2$ 或把它并入 $\kappa_{eg}\Delta^2$，即需要 $\epsilon_f \le \kappa_{ef}\Delta^2$ 型条件。论文**没有**在引理 2.9 或推论 3.6 里显式处理这一项，只在推论 3.6 用 $\Delta_k \ge \sqrt{4\epsilon_f\Lambda/(L+\kappa_{bhm})}$ 处理了 $\kappa_{eg}$ 中的噪声项。**结论**：引理 2.9 在含噪情形需把 $\kappa_{ef}$ 改成 $2\kappa_{eg} + \frac{L+\kappa_{bhm}}{2}$ 一类的松弛（因为 $\epsilon_f \le \frac{L+\kappa_{bhm}}{4\Lambda}\Delta^2 \le \kappa_{eg}\Delta^2$ 在推论 3.6 的条件下成立），**阶不变**，但严格性上有缺口。这属于"评审推断"，作者大概率视为常数级 slack。

## 单元 D2：有限差分模型 (2.9) 与误差界 (2.10)

$$
g(x) = \sum_{i=1}^n \frac{f(x + \delta u_i) - f(x)}{\delta} u_i
$$

逐符号：

- $\delta > 0$：差分步长，**超参数**；本文取 $\delta = \Delta_k$。
- $u_i$：第 $i$ 个**坐标方向**（$\{u_i\}$ 是 $\mathbb{R}^n$ 的一组标准正交基，实际实现取旋转后的坐标向量，见 6.2 节），$\Vert u_i \Vert = 1$，$u_i^\top u_j = \delta_{ij}$。
- $\frac{f(x+\delta u_i)-f(x)}{\delta}$：第 $i$ 个**前向差分商**，标量，**已知量**（2 次 oracle 调用，但 $f(x)$ 在 $i$ 之间共享，故共 $n+1$ 次）。
- 求和 $\sum_{i=1}^n$：把 $n$ 个差分商按 $u_i$ 方向拼回向量。

**误差界 (2.10) 的推导**：

$$
\Vert g(x) - \nabla \phi(x) \Vert \le \frac{\sqrt{n} L \delta}{2} + \frac{2 \sqrt{n} \epsilon_f}{\delta}
$$

1. 加减真值：$\frac{f(x+\delta u_i)-f(x)}{\delta} = \frac{\phi(x+\delta u_i)-\phi(x)}{\delta} + \frac{(f-\phi)(x+\delta u_i) - (f-\phi)(x)}{\delta}$。
2. **偏差项**：由 (D2)，$|\phi(x+\delta u_i) - \phi(x) - \delta \nabla\phi(x)^\top u_i| \le \frac{L}{2}\delta^2\Vert u_i\Vert^2 = \frac{L}{2}\delta^2$，除以 $\delta$：$\left|\frac{\phi(x+\delta u_i)-\phi(x)}{\delta} - \nabla\phi(x)^\top u_i\right| \le \frac{L\delta}{2}$。记 $d_i$ 为该误差，$|d_i|\le\frac{L\delta}{2}$。
3. 用完备性 $\sum_i u_i u_i^\top = I$：$\sum_i \frac{\phi(x+\delta u_i)-\phi(x)}{\delta}u_i = \sum_i (\nabla\phi(x)^\top u_i) u_i + \sum_i d_i u_i = \nabla \phi(x) + \sum_i d_i u_i$，且 $\Vert\sum_i d_i u_i\Vert = \sqrt{\sum_i d_i^2} \le \frac{\sqrt n L\delta}{2}$（正交基下坐标 $\ell_2$ 等距）。
4. **噪声项**：$\Vert\sum_i \frac{e_i}{\delta}u_i\Vert = \frac{1}{\delta}\sqrt{\sum_i e_i^2}$，$|e_i|\le|{(f-\phi)(x+\delta u_i)|}+|(f-\phi)(x)|\le 2\epsilon_f$，故 $\le\frac{1}{\delta}\sqrt{n\cdot 4\epsilon_f^2} = \frac{2\sqrt n \epsilon_f}{\delta}$。
5. 三角不等式合并 3、4 得 (2.10)。$\blacksquare$

**取 $\delta = \Delta_k$ 后的 fully-linear 常数**：(2.10) 右端要写成 $\kappa_{eg}\Delta_k$。若 $\Delta_k \ge 2\sqrt{\epsilon_f/L}$，则 $\frac{2\sqrt n\epsilon_f}{\Delta_k} \le \frac{\sqrt n L \Delta_k}{2}$，于是

$$
\Vert g(x) - \nabla \phi(x) \Vert \le \frac{\sqrt n L}{2} \Delta_k + \frac{\sqrt n L}{2} \Delta_k = \sqrt n L \Delta_k
\;\Rightarrow\;
\kappa_{eg} = \sqrt n L, \quad \kappa_{ef} = \frac{L + \kappa_{bhm}}{2} + \sqrt n L
$$

（$\kappa_{ef}$ 由引理 2.9。）**注意 $\kappa_{eg}$ 里没有 $\frac12$**：那 $\frac12$ 被噪声项"用掉了"——这是 $\Delta_k \ge 2\sqrt{\epsilon_f/L}$ 条件的来源，而不是随便取的。

## 单元 D3：基线复杂度 $\mathcal{O}(n^2\epsilon^{-2})$ 与推论 2.10 的 $\mathcal{O}(n^{3/2}\epsilon^{-2})$

**基线（$\eta_2$ 与维度无关）**：有限差分每步给 FL 模型，代价 $n+1$ 次调用，**无模型改进步**（$\mathcal{M}_\epsilon = \emptyset$）。$C_1^{-1} = \Theta(\sqrt n)$（因 $\kappa_{eg} = \sqrt n L$、$\kappa_{ef} = \Theta(\sqrt n)$ 主导 $\max$）、$C_2 = \Theta(1)$（$\eta_2,\kappa_{bhm}$ 常数）。于是

$$
\mathcal{C}_\epsilon \le (n+1)(\vert \mathcal{S}_\epsilon \vert + \vert \mathcal{U}_\epsilon \vert) = \mathcal{O}\left( n \cdot \frac{1}{C_2 C_1^2 \epsilon^2} \right) = \mathcal{O}\left( n \cdot \frac{n}{\epsilon^2} \right) = \mathcal{O}(n^2 \epsilon^{-2})
$$

**推论 2.10（$\eta_2 = \sqrt n$，$\kappa_{bhm} \le \mathcal{O}(\sqrt n)$）**：此时

$$
C_1^{-1} = \max\left\{ \sqrt n, \kappa_{bhm}, \frac{2\kappa_{ef} + \max\{\sqrt n, \kappa_{ef}\}}{(1-\eta_1)\kappa_{fcd}} \right\} + \kappa_{eg} = \Theta(\sqrt n)
$$

$$
C_2 = \frac{\eta_1 \kappa_{fcd}}{2} \cdot \sqrt n \cdot \min\left\{ \frac{\sqrt n}{\kappa_{bhm}}, 1 \right\} = \Theta(\sqrt n)
$$

（$C_0 = \max\{\eta_2,\kappa_{ef}\} \le \Theta(\sqrt n)$，被 $\max$ 吸收；$\min$ 项在 $\kappa_{bhm} = \Theta(\sqrt n)$ 时为 $\Theta(1)$。）代入定理 2.8：

$$
\left\vert \mathcal{S}_\epsilon \right\vert + \left\vert \mathcal{U}_\epsilon \right\vert \le \mathcal{O}\left( \frac{\phi(x_0) - \phi^\star}{\sqrt n \cdot (\epsilon/\sqrt n)^2} \right) + \mathcal{O}\left( \log \frac{\epsilon}{\sqrt n \Delta_0} \cdot \frac{1}{\log \gamma} \right) = \mathcal{O}\left( \frac{\sqrt n}{\epsilon^2} + \log \frac{\sqrt n \Delta_0}{\epsilon} \right) = \mathcal{O}\left( \frac{\sqrt n}{\epsilon^2} \right)
$$

乘 $n+1$ 得 $\mathcal{C}_\epsilon \le \mathcal{O}(n^{3/2}\epsilon^{-2})$。$\blacksquare$

**对数项可吸收的条件**：$\log(\sqrt n\Delta_0/\epsilon) \le \mathcal{O}(\sqrt n \epsilon^{-2})$ 对 $\epsilon$ 有下界 $\Omega(\sqrt{n\epsilon_f})$ 时成立（论文在推论证明里直接把对数写成低阶项，未核对；核验：$\epsilon \ge c\sqrt{n\epsilon_f}$ 时 $\log$ 项 $\le \log(\text{const}\cdot \epsilon^{-1}\sqrt n)$ 关于 $\epsilon^{-2}$ 是低阶 ✓）。

**$\epsilon$ 的下界（$\psi(\epsilon_f)$）**：论文指出"$L$ 不随 $n$ 变，$C_1^{-1} = \Theta(\sqrt n)$，故 $\eta_2$ 无论取常数还是 $\sqrt n$，$\epsilon$ 的下界都是 $\Omega(\sqrt{n\epsilon_f})$"。核验：条件 $\epsilon > \sqrt{4\epsilon_f/(\gamma^2\min\{C_2,L\}C_1^2)}$，$C_1^2 = \Theta(1/n)$ ⇒ $\epsilon > \Theta(\sqrt{n\epsilon_f/\min\{C_2,L\}}) = \Omega(\sqrt{n\epsilon_f})$（$\min\{C_2,L\}$ 与 $n$ 无关）✓。

**本阶段研究审计**：

- **净增量**：推论 2.10 的 $\mathcal{O}(n^{3/2}\epsilon^{-2})$ 是**已知事实的重新整理**——有限差分 + 定理 2.8 直接给出；[11] 已有 $\mathcal{O}(n^2\epsilon^{-2})$ 的对照。真正的新内容是把 $\eta_2$ 当**可调维度参数**这一观察写清楚。
- **非空泛性**：$\mathcal{O}(n^{3/2}\epsilon^{-2})$ 只在 $\epsilon \ge \Omega(\sqrt{n\epsilon_f})$ 区域成立；$\epsilon_f = 0$ 时下界退化为 0，界仍成立。
- **可比性**：DFO 的已知下界（维度相关）是 $\mathcal{O}(n^2\epsilon^{-2})$ 与 $\mathcal{O}(n\epsilon^{-2})$ 两档，本文第 5 节才触及 $n\epsilon^{-2}$。
- **可证伪问题**：$\eta_2$ 取 $\Theta(n^{\alpha})$，一般地 $\vert\mathcal{S}\vert+\vert\mathcal{U}\vert = \mathcal{O}(n^{(1-2\alpha)/2}\epsilon^{-2}\cdot\max\{1,n^{(\alpha-1/2)_+}\})$ 型权衡，是否存在 $\alpha$ 使总复杂度优于 $n^{3/2}$？（答案受 $\kappa_{eg} = \Theta(\sqrt n)$ 是硬下界限制——推论 4.4 与第 5 节正是绕开它：不改 $\kappa_{eg}$，改**每步代价** $n+1 \to \mathcal{O}(1)$ 或 $q$。）

## 阶段 E：第 3 节 —— Lagrange 多项式与 fully-linear 模型

### E0. 本节在证明链中的位置

第 2 节把 $\kappa_{eg}$、$\kappa_{ef}$ 当成**已知常数**写进假设 2.2 与定理 2.8，但没有说这两个常数从哪来。第 3 节唯一任务就是回答这个问题：给定采样集 $\mathcal{Y}_k$ 满足什么几何条件时，插值模型 (2.1) 是 fully-linear 的，并且把 $\kappa_{eg}$ 显式写成 $n$、$L$、$\kappa_{bhm}$、$\Lambda$、$\Delta_k$ 的函数。

由前面已经推过的引理 2.9，判定 fully-linear **只需要**验证梯度误差界 (2.8)，即

$$
\Vert \nabla \phi ( x _ { k } ) - g _ { k } \Vert \le \kappa _ { e g } \Delta _ { k } .
$$

$g _ { k }$ 是模型 (2.1) 里的一次项系数（中间结果：由采样集与函数值解出，不是真实梯度）；$\Delta _ { k }$ 是第 $k$ 步信赖域半径（模型参数：由算法更新）；$\kappa _ { e g }$ 是待定常数（超参数性质：定理一旦证完它就是只依赖 $n , L , \kappa _ { b h m } , \Lambda$ 的量）。这就是本节所有定理的目标形式。

OCR 说明：原文 (3.1) 附近出现的 $\ddot { \phi } ( x )$ 应还原为 $\phi ( x )$；正文中的 $\mathcal { V }$、$\mathcal { D } _ { k }$ 应还原为 $\mathcal { Y }$、$\mathcal { Y } _ { k }$；$\mathcal { \ V } _ { k }$、$\mathcal { \mathcal { V } }$ 同样是 $\mathcal { Y } _ { k }$、$\mathcal { Y }$ 的识别变形。

---

### E1. Definition 3.1（Lagrange 多项式基）

**原文**：给定多项式空间 $\mathcal { P }$（维数 $p$）与点集 $\mathcal { Y } = \{ y _ { 1 } , \ldots , y _ { p } \} \subset \mathbb { R } ^ { n }$，若 $\{ \ell _ { j } ( s ) \} _ { j = 1 } ^ { p } \subset { \mathcal { P } }$ 满足

$$
\ell _ { j } ( y _ { i } ) = \delta _ { i j } = \left\{ \begin{array} { l l } { 1 } & { i = j , } \\ { 0 } & { i \neq j , } \end{array} \right.
$$

则称它是 $\mathcal { Y }$ 上（关于 $\mathcal { P }$）的 Lagrange 多项式基；若这组基存在，称 $\mathcal { Y }$ 是 poised（适定）的。

**逐符号**：

- $\mathcal { P }$：所选多项式的线性空间（已知量：算法设计时固定，不随迭代变）。本文只有两种取法：$p = n$ 的**齐次线性**多项式全体 $\{ s \mapsto a ^ { \top } s : a \in \mathbb { R } ^ { n } \}$（注意：不含常数项），和 $p = \frac { n ( n + 1 ) } { 2 }$ 的二次多项式空间。
- $p$：$\mathcal { P }$ 的维数，同时也是**采样点个数**——定义要求 $| \mathcal { Y } | = p$（超参数性质：由 $\mathcal { P }$ 决定，第 4 节里 $p$ 指 $\mathcal { Z }$ 的容量 $\frac { n ( n - 1 ) } { 2 }$，同名不同值，注意区分）。
- $y _ { i } \in \mathbb { R } ^ { n }$：第 $i$ 个采样点，**相对当前中心 $x _ { k }$ 的位移**（模型参数：会被几何修正替换）。下标 $i$ 取值 $1 , \ldots , p$。
- $\ell _ { j } ( s )$：第 $j$ 个 Lagrange 多项式，自变量 $s \in \mathbb { R } ^ { n }$ 是**任一点相对中心的位移**（中间结果：由 $\mathcal { Y }$ 唯一决定）。
- $\delta _ { i j }$：Kronecker 记号（已知量：$i = j$ 时为 1，否则为 0）。
- poised：中文可译"适定"，此处是术语，指"这组点足以唯一确定 $\mathcal { P }$ 中的插值系数"。

**存在性推导（原文未展开，这里补）**：取 $\mathcal { P }$ 的一组基 $\{ \psi _ { 1 } , \ldots , \psi _ { p } \}$。每个 $\ell _ { j }$ 可写成 $\ell _ { j } ( s ) = \Psi ( s ) ^ { \top } a _ { j }$，其中 $\Psi ( s ) = ( \psi _ { 1 } ( s ) , \ldots , \psi _ { p } ( s ) ) ^ { \top } \in \mathbb { R } ^ { p }$（中间结果：基的值向量），$a _ { j } \in \mathbb { R } ^ { p }$ 是待求系数向量。代入插值条件：对每个 $i$ 有 $\Psi ( y _ { i } ) ^ { \top } a _ { j } = \delta _ { i j }$。把这 $p$ 个标量方程排成矩阵，定义**广义 Vandermonde 矩阵** $A \in \mathbb { R } ^ { p \times p }$，$A _ { i j } = \Psi ( y _ { j } ) _ { i }$（中间结果），则条件成为 $A ^ { \top } a _ { j } = e _ { j }$，$e _ { j }$ 为 $\mathbb { R } ^ { p }$ 的第 $j$ 个坐标单位向量（已知量）。所以

$$
\exists ! \{ \ell _ { j } \} \iff A \text{ 可逆} \iff \{ y _ { 1 } , \dots , y _ { p } \} \text{ 在该基下线性无关（作为 } \Psi \text{-坐标）} .
$$

这解释了为什么"poised"是几何条件而不是分析条件：它只关于点的位置，与 $\phi$ 无关。

**齐次线性情形（本节主线，$p = n$）的显式解**：取 $\Psi ( s ) = s$，即基就是坐标函数。令 $Y \in \mathbb { R } ^ { n \times n }$ 为**以 $y _ { i }$ 为第 $i$ 列**的矩阵（中间结果；全篇 $Y$ 都用这个列约定），则 $A = Y ^ { \top }$，$a _ { j } = Y ^ { - \top } e _ { j }$ = 矩阵 $Y ^ { - T }$ 的第 $j$ 列。记 $( Y ^ { - T } ) _ { j }$ 为该列，得到本节最重要的一个恒等式

$$
\ell _ { j } ( s ) = a _ { j } ^ { \top } s = ( Y ^ { - T } ) _ { j } ^ { \top } s .
$$

$\ell _ { j }$ 是 $s$ 的线性函数，其"斜率向量"就是 $( Y ^ { - T } ) _ { j }$。$Y$ 可逆当且仅当 $\mathcal { Y }$ poised。

**由恒等式立刻得到 poisedness 的等价刻画（后面 Lemma 4.2 与推论 3.6 都靠它）**：

$$
\max _ { s \in B ( 0 , \Delta ) } \left\vert \ell _ { j } ( s ) \right\vert = \max _ { \Vert s \Vert \le \Delta } \left\vert ( Y ^ { - T } ) _ { j } ^ { \top } s \right\vert = \Delta \left\Vert ( Y ^ { - T } ) _ { j } \right\Vert .
$$

第一个等号是 $B ( 0 , \Delta ) = \{ s : \Vert s \Vert \le \Delta \}$ 的定义（已知量：中心在位移原点的闭球）。第二个等号分两头：由 Cauchy–Schwarz 不等式 $| a ^ { \top } s | \le \Vert a \Vert \Vert s \Vert$，取 $a = ( Y ^ { - T } ) _ { j }$ 得上界 $\Delta \Vert ( Y ^ { - T } ) _ { j } \Vert$；反向取 $s ^ { \star } = \Delta \cdot ( Y ^ { - T } ) _ { j } / \Vert ( Y ^ { - T } ) _ { j } \Vert$（若该列为 0 则两端都是 0，等式平凡成立），它满足 $\Vert s ^ { \star } \Vert = \Delta \in B ( 0 , \Delta )$，代入得 $( Y ^ { - T } ) _ { j } ^ { \top } s ^ { \star } = \Delta \Vert ( Y ^ { - T } ) _ { j } \Vert$，两头相等，故最大值恰好取到。所以**"Lagrange 多项式在球上的最大绝对值"与"$Y ^ { - T }$ 各列的欧氏范数"是同一个东西**。

---

### E2. Definition 3.2（$\Lambda$-poisedness）

**原文**：给定 $\mathcal { P }$（维数 $p$）、$\Lambda > 0$、集合 $B \subset \mathbb { R } ^ { n }$。若 poised 集 $\mathcal { Y } = \{ y _ { 1 } , \ldots , y _ { p } \}$ 满足 $\mathcal { Y } \subset B$ 且

$$
\Lambda \ge \max _ { j = 1 , \dots , p } \max _ { s \in { \mathcal { B } } } | \ell _ { j } ( s ) | ,
$$

则称 $\mathcal { Y }$ 在 $B$ 中是 $\Lambda$-poised 的。

**逐符号**：

- $\Lambda$：poisedness 阈值（超参数：人工设定，第 4 节 Table 6.1 取 $1 0 0 0$，理论取 $1 + \Theta ( 1 / n )$），要求 $\Lambda \ge 1$（下面推导）。
- $B$：约束 Lagrange 多项式取值的那个区域（已知量），本文统一取 $B ( 0 , \Delta )$，即位移空间中半径 $\Delta$ 的球。
- 外层 $\max _ { j = 1 , \dots , p }$：对**所有** Lagrange 多项式取最坏；内层 $\max _ { s \in B }$：对该多项式在球上取最坏。两个 max 缺一不可——只要求某一个 $j$ 小没有用。
- $| \ell _ { j } ( s ) |$：标量绝对值（$\ell _ { j }$ 取值在 $\mathbb { R }$）。

**为什么必须 $\Lambda \ge 1$**：$y _ { j } \in \mathcal { Y } \subset B ( 0 , \Delta )$，故 $y _ { j }$ 本身是允许的内层取点，于是 $\max _ { s \in B ( 0 , \Delta ) } | \ell _ { j } ( s ) | \ge | \ell _ { j } ( y _ { j } ) | = \delta _ { j j } = 1$。所以对每个 $j$ 内层最大值都 $\ge 1$，任何有限的 poisedness 常数必须 $\ge 1$。

**$\Lambda = 1$ 可达（这保证定义不空、也保证"推论里 $\Lambda = 1 + O ( 1 / n )$"有意义）**：取 $p = n$、$\Delta = 1$、$\mathcal { Y } = \{ e _ { 1 } , \ldots , e _ { n } \}$，则 $Y = I$，$( Y ^ { - T } ) _ { j } = e _ { j }$，由 E1 的恒等式 $\max _ { B ( 0 , 1 ) } | \ell _ { j } | = 1 \cdot \Vert e _ { j } \Vert = 1$，且 $| \ell _ { j } ( s ) | = | s _ { j } | \le \Vert s \Vert \le 1$。所以 $\Lambda = 1$ 恰好取到。反面例子（说明 $\Lambda$ 可以任意大）：取 $n = 2$，$y _ { 1 } = ( 1 , 0 )$，$y _ { 2 } = ( \cos \theta , \sin \theta )$，$Y ^ { - T }$ 的行范数随 $\sin \theta \to 0$ 发散，于是两点几乎共线时 $\Lambda \to \infty$。这正好对应论文第 6 节说的"高维下 $2 n + 1$ 个点越来越稀疏、越接近退化"。

**$\Lambda$ 的几何含义**（严格陈述，不用比喻）：由 E1 恒等式，$\Lambda$-poised $\iff$ 对所有 $j$ 有 $\Vert ( Y ^ { - T } ) _ { j } \Vert \le \Lambda / \Delta$。即 **$Y ^ { - T }$ 的每一列都不能太长**，等价于 $Y ^ { - 1 }$ 的每一行都不能太长。$\Lambda$ 越小，$Y ^ { - 1 }$ 越小，$Y$ 的列越接近正交、越"铺开"。

---

### E3. 模型构造 (3.1)(3.2)

**(3.1)**：设 $\mathcal { Y } _ { k } = \{ y _ { 1 } , \ldots , y _ { p } \}$ 在 $B ( 0 , \Delta )$ 中 poised，取 $g _ { k } , H _ { k }$ 满足 $\Vert H _ { k } \Vert \le \kappa _ { b h m }$ 且

$$
g _ { k } ^ { \top } y + \frac { 1 } { 2 } y ^ { \top } H _ { k } y = \phi ( x _ { k } + y ) - \phi ( x _ { k } ) , \quad \forall y \in \mathcal { Y } _ { k } .
$$

**(3.2)**：$m _ { k } ( x ) = \phi ( x _ { k } ) + g _ { k } ^ { \top } s + \frac { 1 } { 2 } s ^ { \top } H _ { k } s$，$s = x - x _ { k }$。

**逐符号**：

- $g _ { k } \in \mathbb { R } ^ { n }$：模型一次项系数（中间结果：由 $\mathcal { Y } _ { k }$ 上的 $p$ 个函数值解出），本节要证 $\Vert \nabla \phi ( x _ { k } ) - g _ { k } \Vert \le \kappa _ { e g } \Delta _ { k }$。
- $H _ { k } \in \mathbb { R } ^ { n \times n }$：模型 Hessian，本文取对称（已知量：二次型 $s ^ { \top } H _ { k } s$ 只依赖 $( H _ { k } + H _ { k } ^ { \top } ) / 2$，反对称部分对二次型贡献为 0，所以总可假设对称）。
- $\Vert H _ { k } \Vert$：**谱范数**，即最大奇异值（对称时=最大特征值绝对值）。范数种类必须点明：Lemma 4.2 用了 Frobenius 范数 $\Vert \cdot \Vert _ { F }$，定理 3.3 用谱范数，两者关系 $\Vert M \Vert \le \Vert M \Vert _ { F } \le \sqrt { n } \Vert M \Vert$。
- $\kappa _ { b h m }$：模型 Hessian 界（假设 2.2 第二条里的常数；已知量层面它由算法约束 $\Vert H _ { k } \Vert \le K$ 保证，见第 4 节 (4.1)）。
- 左端 $g _ { k } ^ { \top } y + \frac { 1 } { 2 } y ^ { \top } H _ { k } y$：模型在位移 $y$ 处相对中心的增量；右端 $\phi ( x _ { k } + y ) - \phi ( x _ { k } )$：真实函数在同一位移上的增量。所以 (3.1) 是"模型在采样点上精确复现函数的**增量**"。

**为什么写成增量而不是写成 $m _ { k } ( y _ { i } ) = \phi ( y _ { i } )$**：(3.1) 是 $p$ 个标量方程，未知数是 $g _ { k }$ 的 $n$ 个分量加上 $H _ { k }$ 的 $\frac { n ( n + 1 ) } { 2 }$ 个独立分量。$p = n$ 时未知数远多于方程——**$H _ { k }$ 基本是自由参数**，任何满足 $\Vert H _ { k } \Vert \le \kappa _ { b h m }$ 的取值都可以配上合适的 $g _ { k }$。这正是论文说"$\mathcal { P }$ 也可以是预设 Hessian 稀疏结构的二次多项式空间"的原因：自由度富余才允许施加稀疏结构。$m _ { k } ( x _ { k } ) = \phi ( x _ { k } )$（中心精确插值）由 (3.2) 的常数项 $\phi ( x _ { k } )$ 保证，这个性质在阶段 D 里是引理 2.9 只用一个 $\kappa _ { e g } \Delta _ { k } ^ { 2 }$ 项就能覆盖 (2.7) 的关键。

**(3.1) 不可实现**：右端含 $\phi$。论文随即给出可计算版本 (3.3)（见 E7）。

---

### E4. Theorem 3.3 完整推导

**原文表述**：设 $\mathcal { Y } = \{ y _ { 1 } , \ldots , y _ { n } \}$ 使 $Y$ 在 $B ( 0 , \Delta )$ 中 $\Lambda$-poised；$g , H$ 满足 $\Vert H \Vert \le \kappa _ { b h m }$ 与 $g ^ { \top } y + \frac { 1 } { 2 } y ^ { \top } H y = \phi ( x + y ) - \phi ( x )$ 对每个 $y \in \mathcal { Y }$。则

$$
\Vert \nabla \phi ( x ) - g \Vert \le \frac { 1 } { 2 } \left( L + \kappa _ { b h m } \right) \Delta \sqrt { n } \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } .
$$

特别地 $\Lambda = 1 + O ( 1 / n )$ 时 $\Vert \nabla \phi ( x ) - g \Vert = O ( \sqrt { n } ) \Delta$。

**逐符号**（新出现的）：$n$：问题维数，同时是这里 $| \mathcal { Y } |$ 与 $\mathcal { P }$ 的维数（已知量）；$\Delta$：poisedness 区域的半径，本定理里与后面 $\Delta _ { k }$ 对应（超参数性质：由算法当前状态给出）；$L$：假设 1.2 的梯度 Lipschitz 常数（已知量，不进算法）；$\sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$：$\Vert Y ^ { - T } D \Vert$ 的上界（中间结果，来自 [11, Thm 4.3]，见步骤 6 的诚实说明）。

**步骤 1（中心化）**：定义 $\bar { \phi } ( s ) = \phi ( x + s ) - \phi ( x )$（中间结果：一个新函数）。则

- $\bar { \phi } ( 0 ) = \phi ( x ) - \phi ( x ) = 0$；
- 由链式法则，$\nabla _ { s } \bar { \phi } ( s ) = \nabla \phi ( x + s )$，故 $\nabla \bar { \phi } ( 0 ) = \nabla \phi ( x )$。

这两条让我们可以**把中心搬到原点**，之后所有范数都在原点处取。

**步骤 2（逐点写成矩阵）**：把 (3.1) 在 $y = y _ { i }$ 处写出，$i = 1 , \ldots , n$：

$$
g ^ { \top } y _ { i } = \phi ( x + y _ { i } ) - \phi ( x ) - \textstyle { \frac { 1 } { 2 } } y _ { i } ^ { \top } H y _ { i } = \bar { \phi } ( y _ { i } ) - \textstyle { \frac { 1 } { 2 } } y _ { i } ^ { \top } H y _ { i } .
$$

定义三个向量（都是中间结果）：$\bar { \phi } ( Y ) \in \mathbb { R } ^ { n }$，第 $i$ 个分量 $\bar { \phi } ( y _ { i } )$；$h ( Y ) \in \mathbb { R } ^ { n }$，第 $i$ 个分量 $\frac { 1 } { 2 } y _ { i } ^ { \top } H y _ { i }$；$D = \operatorname { d i a g } ( \Vert y _ { 1 } \Vert , \ldots , \Vert y _ { n } \Vert ) \in \mathbb { R } ^ { n \times n }$（对角矩阵，对角元是采样点半径长度）。把 $n$ 个方程合成一个：$Y ^ { \top } g = \bar { \phi } ( Y ) - h ( Y )$（因为 $( Y ^ { \top } g ) _ { i } = y _ { i } ^ { \top } g = g ^ { \top } y _ { i }$）。

**步骤 3（解出 $g$）**：$\mathcal { Y }$ poised $\Rightarrow$ $Y$ 可逆 $\Rightarrow$ $Y ^ { \top }$ 可逆，两边左乘 $Y ^ { - \top }$：

$$
g = Y ^ { - T } \bar { \phi } ( Y ) - Y ^ { - T } h ( Y ) .
$$

**步骤 4（拆成两项误差）**：用步骤 1 的 $\nabla \phi ( x ) = \nabla \bar { \phi } ( 0 )$ 和三角不等式 $\Vert a - b \Vert \le \Vert a - c \Vert + \Vert c - b \Vert$（取 $a = \nabla \phi ( x )$，$b = g$，$c = Y ^ { - T } { \bar { \phi } } ( Y )$）：

$$
\Vert \nabla \phi ( x ) - g \Vert \le \underbrace { \Vert \nabla \bar { \phi } ( 0 ) - Y ^ { - T } \bar { \phi } ( Y ) \Vert } _ { \text{项 I：一阶 Taylor 截断误差} } + \underbrace { \Vert Y ^ { - T } h ( Y ) \Vert } _ { \text{项 II：模型二次项带来的误差} } .
$$

（注：这里的 $\underbrace$ 只是我给自己看的分工标注，写进定稿时改成文字。两项的分工是：**项 I 吸收 $L$，项 II 吸收 $\kappa _ { b h m }$**，最后合成 $L + \kappa _ { b h m }$。）

**步骤 5（一个反复使用的矩阵范数不等式）**：对任意可逆 $Y$、任意对角可逆 $D$ 和向量 $v$，

$$
\Vert Y ^ { - T } v \Vert = \Vert Y ^ { - T } D \, ( D ^ { - 1 } v ) \Vert \le \Vert Y ^ { - T } D \Vert \, \Vert D ^ { - 1 } v \Vert \le \Vert Y ^ { - T } D \Vert \, \sqrt { n } \Vert D ^ { - 1 } v \Vert _ { \infty } .
$$

依据逐条：第一个等号是 $D D ^ { - 1 } = I$；第一个不等号是算子范数的**次可乘性** $\Vert M x \Vert \le \Vert M \Vert \Vert x \Vert$（谱范数定义即 $\textstyle \max _ { \Vert x \Vert = 1 } \Vert M x \Vert$）；第二个不等号是范数等价关系 $\Vert u \Vert \le \sqrt { n } \Vert u \Vert _ { \infty }$（因为 $\textstyle \Vert u \Vert ^ { 2 } = \sum _ { i } u _ { i } ^ { 2 } \le n \operatorname* { m a x } _ { i } u _ { i } ^ { 2 }$）。

这一步的**动机**（为什么要把 $D$ 塞进去再拆出来）：poisedness 给出的是"每个 Lagrange 多项式在每个点上的值"的控制，天然带 $\Vert y _ { i } \Vert$ 的尺度；引入 $D$ 把"绝对尺度"和"归一化尺度"分开，之后可以逐个分量做 $\le \frac { L } { 2 } \Delta$ 型的标量估计。

**步骤 6（项 I 的界）**：论文写 "By the proof of Theorem 4.3 in [11], we have $\Vert Y ^ { - T } D \Vert \le \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$ and (项 I) $\le \frac { 1 } { 2 } \sqrt { n } L \Delta \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$。"

**必须诚实区分两件事**：

(a) **项 I 的界可以自证**，不依赖 [11]（除了 $\Vert Y ^ { - T } D \Vert$ 那个因子）。取 $v = Y ^ { \top } \nabla \bar { \phi } ( 0 ) - \bar { \phi } ( Y )$（中间结果），注意 $\nabla \bar { \phi } ( 0 ) = Y ^ { - \top } Y ^ { \top } \nabla \bar { \phi } ( 0 )$，于是

$$
\nabla \bar { \phi } ( 0 ) - Y ^ { - T } \bar { \phi } ( Y ) = Y ^ { - T } \left( Y ^ { \top } \nabla \bar { \phi } ( 0 ) - \bar { \phi } ( Y ) \right) = Y ^ { - T } v .
$$

用步骤 5 得 $\le \sqrt { n } \Vert Y ^ { - T } D \Vert \Vert D ^ { - 1 } v \Vert _ { \infty }$。而 $D ^ { - 1 } v$ 的第 $i$ 分量是

$$
\frac { \nabla \bar { \phi } ( 0 ) ^ { \top } y _ { i } - \bar { \phi } ( y _ { i } ) } { \Vert y _ { i } \Vert } = \frac { \nabla \phi ( x ) ^ { \top } y _ { i } - \left( \phi ( x + y _ { i } ) - \phi ( x ) \right) } { \Vert y _ { i } \Vert } .
$$

（等号用 $\nabla\bar\phi(0)=\nabla\phi(x)$ 与 $\bar\phi$ 的定义；这一步就是"一阶 Taylor 余项"。）由阶段 A 推的 (D2)：$| \phi ( x + y ) - \phi ( x ) - \nabla \phi ( x ) ^ { \top } y | \le \frac { L } { 2 } \Vert y \Vert ^ { 2 }$，分子 $\le \frac { L } { 2 } \Vert y _ { i } \Vert ^ { 2 }$，除以 $\Vert y _ { i } \Vert$ 得 $\le \frac { L } { 2 } \Vert y _ { i } \Vert \le \frac { L } { 2 } \Delta$（最后一步因 $\mathcal { Y } \subset B ( 0 , \Delta )$）。故 $\Vert D ^ { - 1 } v \Vert _ { \infty } \le \frac { L } { 2 } \Delta$，

$$
( \text{项 I} ) \le \textstyle { \frac { 1 } { 2 } } \sqrt { n } \, L \, \Delta \, \Vert Y ^ { - T } D \Vert .
$$

对照论文形式：只差把 $\Vert Y ^ { - T } D \Vert$ 换成 $\sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$。定理 3.5 的证明里作者写"identically as in the previous proof"，说明作者自己就是按这条路径推的。

(b) **$\Vert Y ^ { - T } D \Vert \le \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$ 本文没有证明**，是 [11, Theorem 4.3] 的结论，属于**无缝搬用的外部引理**。我能从本文给出的信息独立推出的只有较弱的一条：

$$
\left\| Y ^ { - T } D \right\| \le \left\| Y ^ { - T } D \right\| _ { F } = \left( \sum _ { i = 1 } ^ { n } \left\| Y ^ { - T } d _ { i } \right\| ^ { 2 } \right) ^ { 1 / 2 } = \left( \sum _ { i = 1 } ^ { n } \Vert y _ { i } \Vert ^ { 2 } \left\| ( Y ^ { - T } ) _ { i } \right\| ^ { 2 } \right) ^ { 1 / 2 } \le \sqrt { n } \Lambda ,
$$

依据：谱范数不超过 Frobenius 范数；$d _ { i } = \Vert y _ { i } \Vert e _ { i }$ 故 $Y ^ { - T } d _ { i } = \Vert y _ { i } \Vert ( Y ^ { - T } ) _ { i }$；$\Vert y _ { i } \Vert \le \Delta$ 且由 E1 恒等式 $\Delta \Vert ( Y ^ { - T } ) _ { i } \Vert = \max _ { B ( 0 , \Delta ) } | \ell _ { i } | \le \Lambda$。

**这个差距是有代价的，不是纯技术细节**：$\sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$ 比 $\sqrt { n } \Lambda$ 小，且当 $\Lambda = 1 + \Theta ( 1 / n )$ 时前者是 $\Theta ( 1 )$ 而后者是 $\Theta ( \sqrt { n } )$。见 E5 的具体代入。**结论：[11, Thm 4.3] 的这条锐化界直接决定了本文复杂度里 $n ^ { 3 / 2 }$ 的指数，用它替换我推出的 $\sqrt { n } \Lambda$ 会把复杂度退回 $n ^ { 2 }$。** 待核验项：想彻底自足地复核本节，需要读 [11] 的 Theorem 4.3 原文（ICM 2026 论文集，见参考文献 [11]）。

**步骤 7（项 II 的界）**：同样用步骤 5，

$$
( \text{项 II} ) = \Vert Y ^ { - T } h ( Y ) \Vert \le \sqrt { n } \Vert Y ^ { - T } D \Vert \Vert D ^ { - 1 } h ( Y ) \Vert _ { \infty } .
$$

$D ^ { - 1 } h ( Y )$ 第 $i$ 分量 $= \frac { y _ { i } ^ { \top } H y _ { i } } { 2 \Vert y _ { i } \Vert }$。用 $| y ^ { \top } H y | \le \Vert H \Vert \Vert y \Vert ^ { 2 }$（推导：$| y ^ { \top } ( H y ) | \le \Vert y \Vert \Vert H y \Vert$（Cauchy–Schwarz），$\Vert H y \Vert \le \Vert H \Vert \Vert y \Vert$（谱范数定义），相乘得结论），得该分量绝对值 $\le \frac { \Vert H \Vert \Vert y _ { i } \Vert ^ { 2 } } { 2 \Vert y _ { i } \Vert } = \frac { \Vert H \Vert } { 2 } \Vert y _ { i } \Vert \le \frac { \kappa _ { b h m } } { 2 } \Delta$（最后用假设 $\Vert H \Vert \le \kappa _ { b h m }$ 与 $\Vert y _ { i } \Vert \le \Delta$）。故

$$
( \text{项 II} ) \le \textstyle { \frac { 1 } { 2 } } \sqrt { n } \, \kappa _ { b h m } \, \Delta \, \Vert Y ^ { - T } D \Vert \le \textstyle { \frac { 1 } { 2 } } \sqrt { n } \, \kappa _ { b h m } \, \Delta \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } .
$$

**步骤 8（合并）**：步骤 6 + 步骤 7，公因子 $\frac { 1 } { 2 } \Delta \sqrt { n } \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$ 提出来，$L$ 与 $\kappa _ { b h m }$ 相加：

$$
\Vert \nabla \phi ( x ) - g \Vert \le \frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \Delta \sqrt { n } \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } .
$$

与原文结论逐项一致。$\blacksquare$

---

### E5. "特别地" 那句话的代数，以及 $\Lambda$ 到底要多小

令 $\Lambda = 1 + \frac { c } { n }$（$c > 0$ 常数），代入 $n ( \Lambda ^ { 2 } - 1 ) + 2$：

$$
\Lambda ^ { 2 } - 1 = \left( 1 + { \frac { c } { n } } \right) ^ { 2 } - 1 = { \frac { 2 c } { n } } + { \frac { c ^ { 2 } } { n ^ { 2 } } } \implies n ( \Lambda ^ { 2 } - 1 ) = 2 c + { \frac { c ^ { 2 } } { n } } ,
$$

所以 $n ( \Lambda ^ { 2 } - 1 ) + 2 = 2 c + 2 + \frac { c ^ { 2 } } { n }$，是 $\Theta ( 1 )$。于是定理 3.3 右端

$$
\frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \Delta \sqrt { n } \cdot \Theta ( 1 ) = \Theta ( \sqrt { n } ) \Delta ,
$$

即 $\kappa _ { e g } = \Theta ( \sqrt { n } )$，与有限差分模型的 $\kappa _ { e g } = \sqrt { n } L$（阶段 D 的 (2.10)）同阶。这就是论文声称"用更松的采样集也拿到有竞争力的 $\kappa _ { e g }$"的确切含义。

**边界情形检验**：

- $\Lambda = 1$（E2 已验证可达）：$n ( 1 - 1 ) + 2 = 2$，$\kappa _ { e g } = \frac { \sqrt { 2 } } { 2 } ( L + \kappa _ { b h m } ) \sqrt { n } = \frac { 1 } { \sqrt { 2 } } ( L + \kappa _ { b h m } ) \sqrt { n }$，比有限差分的 $\sqrt { n } L$ 还小一个常数（在 $\kappa _ { b h m } \le L$ 时）。
- $\Lambda$ 为固定大常数（如实践用的 $1 0 0 0$）：$n ( \Lambda ^ { 2 } - 1 ) + 2 = \Theta ( n \Lambda ^ { 2 } )$，于是 $\kappa _ { e g } = \Theta ( L \, n \, \Lambda )$ ——**$n$ 的指数从 $1 / 2$ 涨到 $1$**，复杂度直接从 $n ^ { 3 / 2 }$ 掉到 $n ^ { 2 }$。这是第 6 节把 $\Lambda$ 取 $1 0 0 0$ 与理论之间的真实张力的量化形式（阶段 H 会回到这里）。

**数值自检（$n = 2$，$\Lambda = 1$，$\Delta = 1$，$H = 0$）**：$\phi ( u ) = \frac { L } { 2 } u ^ { \top } u$，中心 $x = 0$，故 $\nabla \phi ( x ) = 0$。取 $\mathcal { Y } = \{ e _ { 1 } , e _ { 2 } \}$，$H = 0$，(3.1) 给 $g ^ { \top } e _ { i } = \phi ( e _ { i } ) - \phi ( 0 ) = \frac { L } { 2 }$，即 $g = ( \frac { L } { 2 } , \frac { L } { 2 } )$，$\Vert g \Vert = \frac { L } { \sqrt { 2 } } \approx 0.707 L$。定理论右端 $= \frac { 1 } { 2 } L \cdot 1 \cdot \sqrt { 2 } \cdot \sqrt { 2 } = L$。$0.707 L \le L$ 成立，且比值约 $0.7$——界不是紧的，但**阶是对的**：$\kappa _ { e g }$ 里 $\sqrt { n }$ 的必要性可由 $H = 0$ 时误差就是 $\Vert g - \nabla \phi ( x ) \Vert$ 这一事实看出，$\frac { L } { 2 } \Delta \sqrt { n }$ 是"每个分量各贡献 $\frac { L } { 2 } \Delta$ 的 Taylor 余项、$n$ 个分量平方求和再开方"的自然结果。

---

### E6. Corollary 3.4：从定理 3.3 到 fully-linear

**原文**：若 $m _ { k }$ 按 (3.2) 定义且 $\mathcal { Y } _ { k }$ 是 $\Lambda$-poised，则 $m _ { k }$ 是 $\kappa _ { e f } , \kappa _ { e g }$-fully-linear，其中

$$
\kappa _ { e g } = \frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \sqrt { n } \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } , \qquad \kappa _ { e f } = \kappa _ { e g } + \frac { L + \kappa _ { b h m } } { 2 } .
$$

**展开证明（原文一句 "follows from (3.2), Theorem 3.3 and Lemma 2.9"）**：fully-linear（定义 2.1）要求两条，这里都验证。

- **梯度条件 (2.8)**：定理 3.3 取 $x = x _ { k }$、$\Delta = \Delta _ { k }$（并把 $\mathcal { Y } _ { k } \subset B ( 0 , \Delta _ { k } )$ 作为 poisedness 的一部分）直接给出 $\Vert \nabla \phi ( x _ { k } ) - g _ { k } \Vert \le \kappa _ { e g } \Delta _ { k }$。
- **函数值条件 (2.7)**：即 $| m _ { k } ( x _ { k } + s ) - \phi ( x _ { k } + s ) | \le \kappa _ { e f } \Delta _ { k } ^ { 2 }$ 对**所有** $s \in B ( 0 , \Delta _ { k } )$。由引理 2.9（阶段 D 已完整重推），只要 (i) 模型中心精确 $m _ { k } ( x _ { k } ) = \phi ( x _ { k } )$、(ii) 采样点精确 $m _ { k } ( x _ { k } + y ) = \phi ( x _ { k } + y )$、(iii) $\Vert H _ { k } \Vert \le \kappa _ { b h m }$，则 $\kappa _ { e f } = \kappa _ { e g } + \frac { L + \kappa _ { b h m } } { 2 }$ 即可。(i) 由 (3.2) 的常数项 $\phi ( x _ { k } )$；(ii) 由 (3.1) 与 (3.2) 相减；(iii) 由 (3.1) 的显式约束。
- 假设 2.2 的另一半（充分 Cauchy 下降 (2.2)）不由本节负责——它由第 4 节信赖域子问题的解法保证（阶段 B4 已推导：Cauchy 点给 $\kappa _ { f c d } = 1$，与 $\Vert H _ { k } \Vert \le \kappa _ { b h m }$ 一起成立）。

**注意 $\kappa _ { e f }$ 与 $\kappa _ { e g }$ 的对称结构**：两者只差一个**不含 $\sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$ 的** $\frac { L + \kappa _ { b h m } } { 2 }$。含义：函数值误差在 $s$ 很小时比梯度误差低一阶（$\Delta ^ { 2 }$ vs $\Delta$），那多出来的一项正是把"梯度误差 $\times \Delta $"换成"函数值误差"时的二阶余项，与 $n$ 无关。

---

### E7. (3.3) 与 Theorem 3.5：可计算模型 + 含噪界

**原文 (3.3)**：

$$
g _ { k } ^ { \top } y + \frac { 1 } { 2 } y ^ { \top } H _ { k } y = f ( x _ { k } + y ) - f ( x _ { l } ) \quad \forall y \in \mathcal { Y } _ { k } .
$$

**OCR/记号问题**：右端第二项 $f ( x _ { l } )$ 的下标 $l$ 与左端的 $k$ 不一致，而定理 3.5 证明里用的是 $f ( x ) - \phi ( x )$（即在**当前中心**取值）。正确写法应是 $f ( x _ { k } )$，$l$ 是 $k$ 的识别错误。**核验依据**：紧接着定理 3.5 证明中 $E$ 的分量定义为 $( f ( x + y _ { i } ) - \phi ( x + y _ { i } ) ) - ( f ( x ) - \phi ( x ) )$，两处函数值都在同一中心 $x$ 取。若中心不一致（$x _ { l } \ne x _ { k }$），$E _ { i }$ 会多出一项 $- ( f ( x _ { l } ) - \phi ( x _ { l } ) )$，噪声界 $\Vert E \Vert _ { \infty } \le 2 \epsilon _ { f }$ 仍成立（变成 $3 \epsilon _ { f }$，因为三个点各自绝对误差 $\le \epsilon _ { f }$），但**插值的中心就错了**，$m _ { k } ( x _ { k } ) = \phi ( x _ { k } )$ 不再成立，引理 2.9 的前提失效。所以这里必须读作 $x _ { k }$。

**逐符号**：$f$：零阶 oracle（已知量：唯一能调用的对象，满足 $| f ( u ) - \phi ( u ) | \le \epsilon _ { f }$）；$\epsilon _ { f }$：绝对噪声界（超参数性质：问题给定，不进算法）。与 (3.1) 的差别只有把 $\phi$ 换成 $f$。

**Theorem 3.5 结论**（$\mathcal { Y }$ $\Lambda$-poised in $B ( 0 , \Delta )$，$g , H$ 满足 (3.3)，$\Vert H \Vert \le \kappa _ { b h m }$）：

$$
\Vert \nabla \phi ( x ) - g \Vert \le \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } \left( \frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \sqrt { n } \Delta + \sqrt { n } \frac { 2 \epsilon _ { f } \Lambda } { \Delta } \right) .
$$

比定理 3.3 多出括号里第二项：$\propto \epsilon _ { f } \Lambda / \Delta$，且**随 $\Delta$ 减小而增大**——这是含噪插值模型的本质退化，与阶段 D 里有限差分模型的噪声项 $\frac { 2 \sqrt { n } \epsilon _ { f } } { \Delta ^ { 2 } }$（除以 $\Delta ^ { 2 }$，更坏）对比可见插值模型对噪声更宽容一档。

**证明逐步展开**：

**步骤 1（噪声进入插值条件）**：由 (3.3) 和 $\bar { \phi }$ 的定义，对 $i = 1 , \ldots , n$，

$$
g ^ { \top } y _ { i } = f ( x + y _ { i } ) - f ( x ) - \textstyle { \frac { 1 } { 2 } } y _ { i } ^ { \top } H y _ { i } = \bar { \phi } ( y _ { i } ) - \textstyle { \frac { 1 } { 2 } } y _ { i } ^ { \top } H y _ { i } + \underbrace { \left( f ( x + y _ { i } ) - \phi ( x + y _ { i } ) \right) - \left( f ( x ) - \phi ( x ) \right) } _ { E _ { i } } .
$$

推导方式：在 $f ( x + y _ { i } ) - f ( x )$ 中加减 $\phi ( x + y _ { i } ) - \phi ( x )$（纯代数，加零）。

**步骤 2（矩阵形式）**：$Y ^ { \top } g = \bar { \phi } ( Y ) - h ( Y ) + E$，$E = ( E _ { 1 } , \ldots , E _ { n } ) ^ { \top } \in \mathbb { R } ^ { n }$（中间结果），左乘 $Y ^ { - \top }$：

$$
g = Y ^ { - T } \bar { \phi } ( Y ) - Y ^ { - T } h ( Y ) + Y ^ { - T } E .
$$

**步骤 3（误差分解）**：用 $\nabla \phi ( x ) = \nabla \bar { \phi } ( 0 ) = Y ^ { - \top } Y ^ { \top } \nabla \bar { \phi } ( 0 )$，

$$
\nabla \phi ( x ) - g = Y ^ { - T } \left( Y ^ { \top } \nabla \bar { \phi } ( 0 ) - \bar { \phi } ( Y ) - E \right) + Y ^ { - T } h ( Y ) ,
$$

取范数并用三角不等式：

$$
\Vert \nabla \phi ( x ) - g \Vert \le \Vert Y ^ { - T } ( Y ^ { \top } \nabla \bar { \phi } ( 0 ) - \bar { \phi } ( Y ) - E ) \Vert + \Vert Y ^ { - T } h ( Y ) \Vert .
$$

**步骤 4（第一次用步骤 5 型不等式 + $\infty$ 范数拆项）**：对 $w = Y ^ { \top } \nabla \bar { \phi } ( 0 ) - \bar { \phi } ( Y ) - E$，$\Vert Y ^ { - T } w \Vert \le \Vert Y ^ { - T } D \Vert \Vert D ^ { - 1 } w \Vert \le \sqrt { n } \Vert Y ^ { - T } D \Vert \Vert D ^ { - 1 } w \Vert _ { \infty }$；再用 $\infty$ 范数的三角不等式和 $\Vert D ^ { - 1 } E \Vert _ { \infty } \le \Vert D ^ { - 1 } \Vert _ { \infty } \Vert E \Vert _ { \infty }$（对角矩阵的 $\infty$ 范数 $= \operatorname* { m a x } _ { i } ( D ^ { - 1 } ) _ { i i } = \operatorname* { m a x } _ { i } \frac { 1 } { \Vert y _ { i } \Vert }$，等号是 $\infty$ 范数=最大绝对行和、对角矩阵行和只有一个非零元）：

$$
\Vert \nabla \phi ( x ) - g \Vert \le \sqrt { n } \Vert Y ^ { - T } D \Vert \left( \Vert D ^ { - 1 } Y ^ { \top } \nabla \bar { \phi } ( 0 ) - D ^ { - 1 } \bar { \phi } ( Y ) \Vert _ { \infty } + \Vert D ^ { - 1 } \Vert _ { \infty } \Vert E \Vert _ { \infty } \right) + \Vert Y ^ { - T } h ( Y ) \Vert .
$$

**步骤 5（三个因子逐个界住）**：

- (i) $\Vert D ^ { - 1 } Y ^ { \top } \nabla \bar { \phi } ( 0 ) - D ^ { - 1 } \bar { \phi } ( Y ) \Vert _ { \infty } \le \frac { L } { 2 } \Delta$：与 E4 步骤 6(a) 完全同一条推导（逐分量 Cauchy–Schwarz 后用 (D2)）。
- (ii) $\Vert Y ^ { - T } h ( Y ) \Vert \le \frac { 1 } { 2 } \sqrt { n } \kappa _ { b h m } \Delta \Vert Y ^ { - T } D \Vert$：与 E4 步骤 7 同。
- (iii) $\Vert E \Vert _ { \infty } \le 2 \epsilon _ { f }$：$| E _ { i } | \le | f ( x + y _ { i } ) - \phi ( x + y _ { i } ) | + | f ( x ) - \phi ( x ) | \le \epsilon _ { f } + \epsilon _ { f }$（三角不等式 + 噪声假设对 $x$ 与 $x + y _ { i }$ 都成立，$y \in \mathcal { Y } \cup \{ 0 \}$ 那句就是这个意思）。**注意**：这里 $2 \epsilon _ { f }$ 而不是 $\epsilon _ { f }$，来源是"两个点的噪声之差"，不是噪声更大。
- (iv) $\Vert D ^ { - 1 } \Vert _ { \infty } \le \frac { \Lambda } { \Delta }$：**这一步只用 poisedness，是全节最精巧的一行，展开成四小步**。
  1. Lagrange 基的定义给出 $\ell _ { i } ( y _ { i } ) = 1$（Definition 3.1，取 $j = i$）。
  2. 齐次线性情形 $\ell _ { i } ( s ) = ( Y ^ { - T } ) _ { i } ^ { \top } s$ 是线性映射，故对任意标量 $\alpha$ 有 $\ell _ { i } ( \alpha y _ { i } ) = \alpha \ell _ { i } ( y _ { i } ) = \alpha$（一次齐性；**这条在二次 Lagrange 多项式下不成立**，见下面的说明）。取 $\alpha = \frac { \Delta } { \Vert y _ { i } \Vert } > 0$ 得 $\ell _ { i } \left( \frac { \Delta y _ { i } } { \Vert y _ { i } \Vert } \right) = \frac { \Delta } { \Vert y _ { i } \Vert }$。
  3. 点 $\frac { \Delta y _ { i } } { \Vert y _ { i } \Vert }$ 的范数恰为 $\Delta$，故属于 $B ( 0 , \Delta )$，可被 poisedness 定义里的内层 $\max$ 取到：$\left| \ell _ { i } \left( \frac { \Delta y _ { i } } { \Vert y _ { i } \Vert } \right) \right| \le \Lambda$。
  4. 合并 2、3：$\frac { \Delta } { \Vert y _ { i } \Vert } \le \Lambda$ 即 $\frac { 1 } { \Vert y _ { i } \Vert } \le \frac { \Lambda } { \Delta }$，对所有 $i$ 取 max 得 (iv)。
  **推论**：$\Vert y _ { i } \Vert \ge \frac { \Delta } { \Lambda }$ ——**$\Lambda$-poised 的集合里不允许有离中心太近的点**。这与 E1 恒等式的"列不能太长"是同一件事的两面，也解释了为什么第 4 节 (iv) 步"用 $s$ 替换 $\mathcal { Z }$ 中更远的点"是在维护几何质量。

**步骤 6（合并）**：把 (i)(iii)(iv) 代入步骤 4 括号、再统一乘 $\sqrt { n } \Vert Y ^ { - T } D \Vert$，加上 (ii)，得

$$
\Vert \nabla \phi ( x ) - g \Vert \le \sqrt { n } \Vert Y ^ { - T } D \Vert \left( \frac { L } { 2 } \Delta + \frac { 2 \epsilon _ { f } \Lambda } { \Delta } + \frac { \kappa _ { b h m } } { 2 } \Delta \right) ,
$$

再用 $\Vert Y ^ { - T } D \Vert \le \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$（[11, Thm 4.3]，同 E4 步骤 6(b)），并把 $\frac { L } { 2 } \Delta + \frac { \kappa _ { b h m } } { 2 } \Delta$ 合写，即原文结论。$\blacksquare$

**与 E4 步骤 6(b) 相同的不诚实风险**：$\Vert Y ^ { - T } D \Vert$ 的锐化界本文未证；换成 $\sqrt { n } \Lambda$ 会让噪声项变成 $\sqrt { n } \Lambda \cdot \frac { 2 \epsilon _ { f } \Lambda } { \Delta } = \frac { 2 \sqrt { n } \epsilon _ { f } \Lambda ^ { 2 } } { \Delta }$，$\Lambda$ 的幂次从 $1$ 涨到 $2$。

**关于二次 Lagrange 的说明**（论文没写，但它决定了第 4 节为什么只维护**线性** Lagrange 多项式）：若 $\mathcal { P }$ 取二次空间（$p = \frac { n ( n + 1 ) } { 2 }$），$\ell _ { i }$ 不再是线性函数，步骤 5(iv) 的第 2 小步失效（不能把 $\alpha$ 从自变量里提出来），$\Vert D ^ { - 1 } \Vert _ { \infty } \le \frac { \Lambda } { \Delta }$ 这条界需要额外的 $\Delta$ 齐次化论证。本文的做法正是绕开这一点：$\mathcal { Y } _ { k }$ 只保证**线性** $\Lambda$-poised，二次项全部塞进 $\mathcal { Z } _ { k }$ 的最小二乘里，而 $\mathcal { Z } _ { k }$ 的几何只用"到中心距离"管理。这是"Powell 风格但可分析"的结构性关键。

---

### E8. Corollary 3.6：噪声版 fully-linear 常数

**原文**：若 $m _ { k }$ 按 (3.2)、$\mathcal { Y } _ { k }$ $\Lambda$-poised，且

$$
\Delta _ { k } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } } ,
$$

则 $m _ { k }$ 是 $\kappa _ { e f } , \kappa _ { e g }$-fully-linear，且

$$
\kappa _ { e g } = ( L + \kappa _ { b h m } ) \sqrt { n } \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } , \qquad \kappa _ { e f } = \kappa _ { e g } + \frac { L + \kappa _ { b h m } } { 2 } .
$$

**半径条件的来历（把噪声项压下去）**：要求"含噪的第二项 $\le$ 无噪的第一项"，

$$
\sqrt { n } \frac { 2 \epsilon _ { f } \Lambda } { \Delta _ { k } } \le \frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \sqrt { n } \Delta _ { k } \iff 2 \epsilon _ { f } \Lambda \le \frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \Delta _ { k } ^ { 2 } \iff \Delta _ { k } ^ { 2 } \ge \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } ,
$$

正是原文那条（两边乘 $\sqrt { n }$ 前后等价，$\sqrt n$ 可约去）。此时定理 3.5 右端 $\le 2 \cdot \frac { 1 } { 2 } ( L + \kappa _ { b h m } ) \sqrt { n } \Delta _ { k } \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$，即 $\kappa _ { e g } \Delta _ { k }$ 形式，$\kappa _ { e g }$ 与 (3.4) 相比**恰好差一个因子 2**，这因子 2 就是"噪声吃掉一份干净误差"的定量代价。

**$\kappa _ { e f }$ 为什么和定理 3.3 一样**：引理 2.9 的结构 $\kappa _ { e f } = \kappa _ { e g } + \frac { L + \kappa _ { b h m } } { 2 }$ 只依赖 Lipschitz 与 Hessian 界，不依赖模型是用 $\phi$ 还是用 $f$ 建的；噪声全部体现在 $\kappa _ { e g }$ 里。但注意一个我在阶段 D 记下的**潜在缺口**：引理 2.9 的证明要求 $m _ { k } ( x _ { k } ) = \phi ( x _ { k } )$ 与 $m _ { k } ( x _ { k } + y ) = \phi ( x _ { k } + y )$ **精确**成立。按 (3.3) 建的模型只满足 $m _ { k } ( x _ { k } + y ) - m _ { k } ( x _ { k } ) = f ( x _ { k } + y ) - f ( x _ { k } )$，与 $\phi$ 的差正是那 $2 \epsilon _ { f }$。所以严格说，从 (3.3) 到 (2.7) 需要把噪声通过**梯度误差 + (D2)** 间接传进来（即 (2.7) 的左端 $| m _ { k } ( x _ { k } + s ) - \phi ( x _ { k } + s ) |$ 用 E4 步骤 6 的分解重新估计，会得到 $\kappa _ { e f } \Delta _ { k } ^ { 2 } + 2 \epsilon _ { f }$ 型余项），而不是简单引用引理 2.9。**论文在此处的表述是宽松的，属于跳步。** 这个缺口对复杂度结论的实际影响是"$\epsilon$ 的下界要再加一项 $\epsilon _ { f }$ 的倍数"，与 $\epsilon \ge \Omega ( \sqrt { n \Lambda \epsilon _ { f } } )$ 相比在 $\epsilon _ { f }$ 很小时不是主项，但严格性上应当补。

**半径下界与算法的耦合（本节最要紧的可验证性问题）**：
定理 3.3/3.5 只在 $\Delta _ { k }$ 不小于某个正数时才给出 fully-linear。而收敛分析要求 $\Delta _ { K _ { \epsilon } } \le \frac { \epsilon } { C _ { 1 } }$（阶段 C 引理 2.5 的逆否）。两条同时成立需要

$$
\frac { \epsilon } { C _ { 1 } } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } } .
$$

用 $C _ { 1 } ^ { - 1 } = \Theta ( \kappa _ { e g } ) $（阶段 C7 已推：$C _ { 1 } ^ { - 1 }$ 的主导项来自 $\kappa _ { e g } \Delta _ { k }$ 与 $\eta _ { 2 } \Delta _ { k }$ 的 max 结构）并把 $\kappa _ { e g } = \Theta ( ( L + \kappa _ { b h m } ) \sqrt { n } )$ 代入，得到

$$
\epsilon \ge \Omega \left( \sqrt { n \Lambda \epsilon _ { f } } \right) .
$$

与阶段 D3 得到的 $\epsilon \ge \Omega ( \sqrt { n \epsilon _ { f } } )$ 一致（那里 $\Lambda = \Theta ( 1 )$）。**新信息是 $\Lambda$ 出现在噪声门槛里**：$\Lambda$ 越大，可保证收敛的 $\epsilon$ 下限越高。这条定量关系是评判第 6 节取 $\Lambda = 1 0 0 0$ 的直接尺子。

---

### E9. 第 3 节研究审计

**净增量（相对 [11]）**：
1. 模型允许带**二次项**（$g , H$ 由 (3.1)/(3.3) 联合决定，$H$ 只需满足谱范数界），而 [11] 的对应结果是纯线性模型。论文明说 "This result is an extension of a similar result in [11] which allows $m _ { k }$ to include a quadratic term"——**扩展方向是本文允许二次项**，[11] 不允许。
2. 定理 3.5 / 推论 3.6 是**带噪声 oracle** 的版本，[11] 只处理精确 $\phi$。这一条是本文对复杂度理论最实质的增量（因为 DFO 实践里 $f$ 一定是近似的），代价是 $\kappa _ { e g }$ 翻倍、且要求 $\Delta _ { k }$ 有 $\sqrt { \epsilon _ { f } \Lambda }$ 型下界。
3. $\Lambda$-poisedness 的定义本身、$\ell _ { i } ( s ) = ( Y ^ { - T } ) _ { i } ^ { \top } s$ 恒等式、$\max | \ell _ { i } | = \Delta \Vert ( Y ^ { - T } ) _ { i } \Vert$ 都是教科书内容（来自 [13]，原文注明 "The concepts and the definitions below can be found in [13]"），**不是本文贡献**。

**最强近邻对比表**：

| 项 | [11]（本文所基于的会议论文） | 本文第 3 节 | 有限差分（阶段 D） |
| --- | --- | --- | --- |
| 模型次数 | 线性 | 线性插值 + 任意满足谱范数界的二次项 | 线性 |
| 采样点数 | $n$（全 poised） | $n$（线性 poised）+ $\le \frac { n ( n - 1 ) } { 2 }$（仅按距离管理） | $n + 1$ |
| oracle | 精确 $\phi$ | 带噪 $f$，$\vert f - \phi \vert \le \epsilon _ { f }$ | 带噪 $f$ |
| $\kappa _ { e g }$ | $\Theta ( \sqrt { n } )$（$\Lambda = 1 + O ( 1 / n )$） | $\Theta ( \sqrt { n } )$，噪声时多因子 2 | $\sqrt { n } L$，噪声时 $+ \frac { 2 \sqrt { n } \epsilon _ { f } } { \Delta ^ { 2 } }$ |
| 额外前提 | 无 | $\Delta _ { k } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } }$ | $\Delta _ { k } \ge 2 \sqrt { \epsilon _ { f } / L }$ |

**隐藏依赖**：$\kappa _ { e g }$ 显含 $\Lambda$ 的平方根项 $n ( \Lambda ^ { 2 } - 1 ) + 2$。理论要求 $\Lambda$ 接近 $1 + \Theta ( 1 / n )$，但第 4 节的 Lemma 4.3 给出的**可达** poisedness 是 $1 4 ^ { n }$ 级别——把 $\Lambda = 1 4 ^ { n }$ 代入，$n ( \Lambda ^ { 2 } - 1 ) + 2 \approx n \cdot 1 4 ^ { 2 n }$，$\kappa _ { e g }$ 变成 $\exp ( O ( n ) )$ 级别。**这个矛盾在第 4 节必须靠"连续 model improving 迭代次数有限"来控制，而不是靠最终 $\Lambda$ 小**：定理 4.1 的界里带 $| \log \log \Lambda |$，正是为了让 $\Lambda$ 很大时代价只以对数-对数形式出现。理解这一点是读懂第 4 节的前提。

**非空泛区域**：$\epsilon \in \left[ \Omega ( \sqrt { n \Lambda \epsilon _ { f } } ) , \mathcal { O } ( 1 ) \right]$；在此区间外，(2.8) 的推导链条断掉，本节结论不提供任何保证。

**可证伪问题（下一节先检验它）**：$1 4 ^ { n }$ 这个底数（来自 Lemma 4.3 中 $7 \Lambda$ 的归纳与 $\sqrt { 4 0 } < 7$ 的松弛）是本质还是松弛？若能构造一类二维 poised 集，其任意一次几何修正后 $\Lambda$ 至少乘 $\Lambda ^ { c }$（$c > 1$ 与维度无关），则 $1 4 ^ { n }$ 的指数形式不可改进；否则引理 4.3 有收紧空间，进而第 4 节 $\log n$ 因子的数量级会变。

## 阶段 F：第 4 节 —— 基于 Lagrange 多项式的 model improving 迭代

### F0. 本节攻击的目标（先说清楚"为什么不是改 $\kappa_{eg}$"）

阶段 D3 已确认：$\vert\mathcal S_\epsilon\vert+\vert\mathcal U_\epsilon\vert$ 的量级由 $\frac{1}{C_2 C_1^2}$ 决定，而 $C_1^{-1}$ 与 $\kappa_{eg}$ 同阶。第 4 节**不去压低 $\kappa_{eg}$**（$\Theta(\sqrt n)$ 已被有限差分与插值两条路同时达到，像是硬墙），改攻两个别的量：

1. **每步模型代价**：有限差分每步要 $n+1$ 次 oracle（阶段 D），总复杂度 $(n+1)\cdot\frac{1}{C_2C_1^2}\cdot\epsilon^{-2}$。第 4 节的算法每步只花 $1$ 或 $2$ 次 oracle，把因子 $n+1$ 换掉。
2. **代价**：换掉 $n+1$ 之后，多出第三类迭代 $\mathcal M_\epsilon$（model improving），必须给它一个**连续长度上界**，否则复杂度失控。本节的全部技术内容就是这个上界：定理 4.1 的 $4n\log n + 8n + 2n\vert\log\log\Lambda\vert$。

所以第 4 节的净增量是**算法层面**的（一个可实现、带 $\mathcal Z$ 集、带 self-correction 的 Powell 风格几何管理程序）+ **计数层面**的（引理 4.2、4.3）。几何与 poisedness 的分析工具全部来自第 3 节。

---

### F1. (4.1)：约束最小二乘模型

**原文**：

$$
\min _ { g , H } \sum _ { z \in \mathcal { Z } _ { k } } \left( f ( x _ { k } ) + g ^ { \top } z + \frac { 1 } { 2 } z ^ { T } H z - f ( x _ { k } + z ) \right) ^ { 2 }
$$

subject to $\Vert H \Vert \le K$，以及

$$
g ^ { \top } y + \frac { 1 } { 2 } y ^ { \top } H y = f ( x _ { k } + y ) - f ( x _ { k } ) , \quad \forall y \in \mathcal { Y } _ { k } .
$$

**逐符号**：

- $g \in \mathbb R^n$、$H \in \mathbb R^{n\times n}$（对称）：优化变量（中间结果：解出后记为 $g_k,H_k$，进入模型 (2.1)）。
- $\mathcal Z_k$：第二个采样集，$|\mathcal Z_k| \le p = \frac {( n - 1 ) n } { 2 }$（模型参数：随迭代更新）。**只按到中心的距离管理几何**，不维护 Lagrange 多项式。
- $\mathcal Y_k$：$|\mathcal Y_k| = n$，维护**线性** Lagrange 多项式与 $\Lambda$-poisedness（模型参数）。
- $K$：Hessian 谱范数上限（超参数：算法显式设定；Table 6.1 里取 $1 0 ^ { 1 0 0 }$）。
- $f ( x _ { k } ) + g ^ { \top } z + \frac { 1 } { 2 } z ^ { T } H z - f ( x _ { k } + z )$：**残差**，模型在 $z$ 处的预测值减 oracle 观测值。平方求和是最小二乘拟合误差。
- 常数项固定为 $f ( x _ { k } )$，**不作为未知数**——这与 Powell 的 underdetermined 模型一致，也正是 $m_k(x_k) = f(x_k)$ 的来源（引理 2.9 需要的中心精确性）。

**这条问题性质（原文未说明，但决定可实现性）**：

- 目标函数对 $(g,H)$ 是**凸二次**：残差里 $g ^ { \top } z$ 线性于 $g$；$z ^ { \top } H z = \mathrm{tr}(H z z ^ { \top })$ 线性于 $H$（因为 $z^\top H z = \sum_{ij}H_{ij}z_iz_j$ 是 $H$ 的线性函数）。所以平方和是仿射函数的平方和 ⟹ 凸二次。
- 等式约束对 $(g,H)$ **线性**（同上）。
- $\{ H : \Vert H \Vert \le K \}$ 是凸的（谱范数是范数，范数球凸；等价地写成半定约束 $- K I \preceq H \preceq K I$）。
- ⟹ (4.1) 是**凸 QP（带谱范数球约束）**，局部最优即全局最优，可用标准内点法解。
- **解的存在性**（原文未证，一行可补）：可行集非空需要 $g$ 能被 $\mathcal Y_k$ 唯一解出——由 poisedness，$Y_k$ 可逆，给定 $H$ 有 $g = Y _ { k } ^ { - \top } ( \bar { f } ( Y _ { k } ) - h ( H ) )$。于是可行集同构于 $\{ H : \Vert H \Vert \le K \}$，该集在 $\mathbb R^{n\times n}$ 中**闭且有界**（谱范数球），$g$ 连续依赖于 $H$，故可行集紧；连续函数在紧集上取到下界 ⟹ 最优解存在。

**$K$ 与 $\kappa_{bhm}$ 的关系**：任何解都满足 $\Vert H_k \Vert \le K$，故 $\kappa _ { b h m } \le K$（取 $\kappa_{bhm} = K$ 即可满足假设 2.2 第二条）。原文还指出：若存在一个 Hessian 满足 $\Vert H \Vert \le \kappa_{bhm} \le K$ 的二次模型同时插值 $\mathcal Y \cup \mathcal Z$ 全部点，则它必是最优解（残差全 0，目标最小可能值）。

**自由度计数（原文只用一句话带过，这里给出可核对的账）**：
未知数 $n$（$g$ 的分量）$+ \frac { n ( n + 1 ) } { 2 }$（对称 $H$ 的独立分量）$= \frac { n ( n + 3 ) } { 2 }$。等式约束 $| \mathcal Y _ { k } | = n$ 个；$\mathcal Z _ { k }$ 只给最小二乘残差（不进等式）。取 $| \mathcal Z _ { k } | = p = \frac { n ( n - 1 ) } { 2 }$ 时，"总信息量" $n + p = \frac { n ( n + 1 ) } { 2 }$，仍比未知数少 $n$。
所以严格按计数，$p = \frac { n ( n - 1 ) } { 2 }$ 时问题**仍欠定 $n$ 个自由度**，而原文说 "When $p < \frac { n ( n - 1 ) } { 2 }$ then the problem may have multiple optimal solutions"（即 $p$ 取满时不再有多解）。两种读法：

- **读法 A（我倾向，也符合 Powell）**：$\frac { n ( n - 1 ) } { 2 }$ 恰好是**严格非对角**元素个数。设计意图是"$\mathcal Y$（含中心）确定 $g$ 与 $H$ 的对角部分，$\mathcal Z$ 确定非对角部分"，此时未知数 $n + \frac{n(n-1)}{2} = \frac{n(n+1)}{2}$ 与方程数匹配。但**约束里没有"对角部分由 $\mathcal Y$ 决定"这条显式机制**（$\mathcal Y$ 的 $n$ 个方程对 $g$ 与 $H$ 是耦合的），所以这条意图必须靠附加规则实现——原文接下来正是给了两条附加规则：取 Frobenius 范数最小解（[13]），或取与上一轮 Hessian 最接近的解（Powell [21]）。
- **读法 B（严格）**：应写成 $p \le \frac { n ( n - 1 ) } { 2 }$ 时可能多解。

**论文在此处未给出计数论证**，属于需要读代码才能确认的地方（第 6 节说 MFN/MCFN 两种模型都测了，"MFN/MCFN" 就是上述两条附加规则的缩写）。这条不影响任何定理：定理只用到 $\Vert H _ { k } \Vert \le K$ 与 $\mathcal Y _ { k }$ 上的精确插值，两者在所有最优解上都成立。

---

### F2. Algorithm 2 逐行（geometry-correcting algorithm）

**输入**：零阶 oracle $f ( x ) \approx \phi ( x )$；$p = \frac { ( n - 1 ) n } { 2 }$；$\Delta _ { 0 }$；$x _ { 0 }$；$\gamma \in ( 0 , 1 )$；$\eta _ { 1 } > 0$；$\eta _ { 2 } > 0$；$\Lambda > 1$；$\Lambda _ { s c } \ge 1$。

逐符号（只列 Algorithm 1 里没有的）：

- $\Lambda _ { s c }$：**self-correction 触发阈值**（超参数，Table 6.1 取 $2$）。与 $\Lambda$ 的分工：$\Lambda$ 是"poised 与否"的判定线（(iii) 用），$\Lambda _ { s c }$ 是"试步点值够不够大、值不值得插入 $\mathcal Y$"的判定线（(ii) 用）。原文强调"analysis applies as long as $\Lambda _ { s c } \geq 1$"，而实践取 2。
- $\gamma \in ( 0 , 1 )$：**收缩因子**（超参数，Table 6.1 取 $0.8$）；成功时半径用 $\gamma ^ { - 1 } > 1$ 扩张。注意与 Algorithm 1 的记号一致（那里也写 $\Delta_{k+1}=\gamma^{-1}\Delta_k$ / $\gamma\Delta_k$），但 Table 6.1 里的"expansion factor $\gamma_{inc}=1.3$"是**另一个数**，$1/\gamma = 1.25 \ne 1.3$——实现里扩张与收缩不互为倒数，理论分析假设的是互为倒数的那版。这是**理论与实现的一处真实不一致**（不影响阶，影响常数）。
- $I _ { i m p } \in \{ 0 , 1 \}$：**本轮是否做了模型改进的标志位**（中间结果：本轮内被 (i)/(iii) 置 1）。它决定半径是否保持不动，并决定后续步骤是否执行。

**初始化**：$\mathcal Y _ { 0 }$（$n$ 个点）、$\mathcal Z _ { 0 }$（$\le p$ 个点）、$f ( x _ { 0 } )$ 与所有 $f ( x _ { 0 } + y _ { i } )$、$f ( x _ { 0 } + z _ { i } )$、以及 $\mathcal Y _ { 0 }$ 的线性 Lagrange 多项式集 $\{ \ell _ { i } \} _ { i = 1 } ^ { n }$。oracle 代价 $1 + n + | \mathcal Z _ { 0 } |$。

**第 1 行（建模型）**：原文写 "Build a quadratic model $m _ { k } ( x _ { k } + s )$ as in (3.2) using $f ( x _ { k } )$ and $f ( x _ { k } + y _ { i } )$，$f ( x _ { 0 } + z _ { i } )$"。
两处记号问题：(a) 真正的构造方式是 **(4.1)**（(3.2) 只是模型的形状 $f ( x _ { k } ) + g ^ { \top } s + \frac { 1 } { 2 } s ^ { \top } H _ { k } s$，不含求解规则）；(b) $f ( x _ { 0 } + z _ { i } )$ 的下标应为 $k$。
**为什么这里可以用 $\mathcal Z$ 的值却不破坏分析**：$\mathcal Z$ 只进目标函数（最小二乘），不进等式约束；等式只有 $\mathcal Y$ 的 $n$ 条，所以 $g _ { k }$ 仍由 $g _ { k } = Y _ { k } ^ { - \top } ( \bar { f } ( Y _ { k } ) - h ( H _ { k } ) )$ 精确给出，第 3 节全部推导（$g$ 的这个表达式是起点）只依赖 $\mathcal Y$ 上的等式，与 $\mathcal Z$ 无关。这是 Y-Z 分离设计的核心：**分析只读 $\mathcal Y$，实践只靠 $\mathcal Z$ 提精度**。

**第 2 行（试步与比率）**：$s _ { k }$ 与 $\rho _ { k }$ 同 Algorithm 1，即

$$
\rho _ { k } = \frac { m _ { k } ( x _ { k } ) - m _ { k } ( x _ { k } + s _ { k } ) } { f ( x _ { k } ) - f ( x _ { k } + s _ { k } ) } , \qquad \Vert s _ { k } \Vert \le \Delta _ { k } .
$$

分母用 oracle 值（**这才是 DFO**：$\phi$ 不可知），分子纯模型计算。这一行**恰好花 1 次 oracle**（$f ( x _ { k } + s _ { k } )$；$f ( x _ { k } )$ 上一轮已有）。
关键副产品：$f ( x _ { k } + s _ { k } )$ 已经算出来了 ⟹ **把 $s _ { k }$ 插入采样集是免费的**（0 次额外 oracle）。这正是 (ii) self-correction 的实践价值所在，也是本节开头引述的论证（"步被拒正是因为模型与函数在试步处不一致，所以这个点最可能改进模型"）。

**第 3 行（成功迭代）**：条件 $\rho _ { k } \ge \eta _ { 1 }$ **且** $\Vert g _ { k } \Vert \ge \eta _ { 2 } \Delta _ { k }$。

- 两个条件缺一不可。第二个就是"用 $\Vert g _ { k } \Vert \ge \eta _ { 2 } \Delta _ { k }$ 取代 criticality step"——若模型梯度相对半径太小，说明模型没资格判断"该往哪走"，此时接受位移会毁掉下降量下界（引理 2.4 的 $C _ { 2 } \Delta _ { k } ^ { 2 }$ 靠 $\Vert g_k\|\ge\eta_2\Delta_k$ 与 (2.2) 相乘得到）。
- 动作：$x _ { k + 1 } = x _ { k } + s _ { k }$；$\Delta _ { k + 1 } = \gamma ^ { - 1 } \Delta _ { k }$（扩张）。
- $j _ { k } ^ { * } = \arg \max _ { j = 1 , \dots , n } \Vert y _ { j } - s _ { k } \Vert$，$i _ { k } ^ { * } = \arg \max _ { i = 1 , \dots , p } \Vert z _ { i } - s _ { k } \Vert$。
  **这两行的真实含义**：换了中心之后，旧位移 $y$ 变成新位移 $y - s _ { k }$，所以 $\Vert y _ { j } - s _ { k } \Vert$ **就是新中心下第 $j$ 个 $\mathcal Y$ 点的半径**。$j_k^*$ = 新中心下最远的 $\mathcal Y$ 点，$i_k^*$ = 新中心下最远的 $\mathcal Z$ 点。
- 分支：若 $\Vert y _ { j _ { k } ^ { * } } - s _ { k } \Vert > \Vert z _ { i _ { k } ^ { * } } - s _ { k } \Vert$ **且** $\vert \ell _ { j _ { k } ^ { * } } ( s _ { k } ) \vert > 0$，则 $\mathcal Y _ { k + 1 } = ( \mathcal Y _ { k } \setminus \{ y _ { j _ { k } ^ { * } } \} \cup \{ 0 \} ) - s _ { k }$，并重算 Lagrange 多项式；否则对 $\mathcal Z$ 做同样操作（$| \mathcal Z _ { k } | = p$ 时替换，$< p$ 时添加）。
  **拆解那个集合式**（顺序很重要）：先从 $\mathcal Y _ { k }$ 去掉最远点，加入 $0$（代表新中心 $x _ { k + 1 }$ 的位移），然后整体平移 $- s _ { k }$。平移后新加入的点变成 $- s _ { k }$，即**旧中心 $x _ { k }$ 以位移 $- s _ { k }$ 的身份留在集合里**——它的函数值 $f ( x _ { k } )$ 上一轮已算过，所以这步**0 次额外 oracle**。集合大小保持 $n$。所以"成功迭代时 $\mathcal Y$ 怎么更新"的答案是：**丢最远的、留最老的**。
  **$\vert \ell _ { j _ { k } ^ { * } } ( s _ { k } ) \vert > 0$ 这条守卫为什么必须有**：(4.4) 的更新公式要除以 $\ell _ { j ^ { * } } ( \tilde s )$；更重要的是由 Cramer 恒等式（F4 会推）$\det Y ^ { + } = \det Y \cdot \ell _ { j } ( \tilde s )$，故 $\ell _ { j } ( \tilde s ) = 0$ **恰好等价于**替换后 $Y ^ { + }$ 奇异、poisedness 被破坏。脚注 a 说"$\mathcal Z _ { k } \ne \emptyset$ 时 $\Vert y _ { j _ { k } ^ { * } } - s _ { k } \Vert > 0$，程序见 [13]"，指的是避免替换成重复点的实现细节。
  **原文未解释为什么比的是 $\mathcal Y$ 与 $\mathcal Z$ 各自最远点的大小**。可读的理由：替换 $\mathcal Z$ 无需重算 Lagrange 多项式（更便宜），替换 $\mathcal Y$ 能改善 poisedness（更有价值）；把名额给"更过分的那一个"是在两种代价之间的贪心折中。这属于实现选择，不影响定理（定理只要求 $\mathcal Y \subset B ( 0 , \Delta )$ 与 poisedness 被维护）。

**第 4 行（model improving 或 unsuccessful）**：条件 $\rho _ { k } < \eta _ { 1 }$ **或** $\Vert g _ { k } \Vert < \eta _ { 2 } \Delta _ { k }$（第 3 行两个条件的否命题，正确）。置 $x _ { k + 1 } = x _ { k }$（**中心不动，所以所有位移无需平移**）、$I _ { i m p } = 0$，然后执行下面所有适用的步骤。

**(i) 几何修正：替换 $\mathcal Y$ 中的远点。**
$j _ { k } ^ { * } = \arg \max _ { j = 1 , \dots , n } \Vert y _ { j } \Vert$（中心没动，所以直接按旧位移范数取最远）。若 $\Vert y _ { j _ { k } ^ { * } } \Vert > \Delta _ { k }$：令 $s ^ { * } := y _ { j _ { k } ^ { * } }$（被丢弃的点转而候选给 $\mathcal Z$，它的值已知）、$I _ { i m p } := 1$。
插入点选择：若 $\vert \ell _ { j _ { k } ^ { * } } ( s _ { k } ) \vert > 0$ 取 $s _ { k } ^ { * } := s _ { k }$（**免费**，值已算）；否则取 $s _ { k } ^ { * } := \arg \max _ { s \in B ( 0 , \Delta _ { k } ) } \vert \ell _ { j _ { k } ^ { * } } ( s ) \vert$ 并**新算** $f ( x _ { k } + s _ { k } ^ { * } )$。
**这个 argmax 有闭式解**（第 3 节 E1 恒等式的直接推论，原文没点明其计算价值）：$\ell _ { j } ( s ) = ( Y ^ { - T } ) _ { j } ^ { \top } s$ 在 $\Vert s \Vert \le \Delta$ 上的最大值在 $s = \Delta \cdot ( Y ^ { - T } ) _ { j } / \Vert ( Y ^ { - T } ) _ { j } \Vert$ 处取到，代价 $\mathcal O ( n )$。并且该点**恰好落在球面** $\Vert s \Vert = \Delta$ 上——引理 4.3 的归纳基础（"新点在边界上 ⟹ 集合含单位向量"）就靠这一条。
更新 $\mathcal Y _ { k + 1 } = \mathcal Y _ { k } \setminus \{ y _ { j _ { k } ^ { * } } \} \cup \{ s _ { k } ^ { * } \}$ 并重算 Lagrange 多项式。
**为什么先做这一步**：$\Lambda$-poisedness 的定义含 $\mathcal Y \subset B ( 0 , \Delta )$。只要有点出界，"poised" 这个命题就不成立、$\Lambda$-检查也无从谈起，所以出界点必须最先清除。

**(ii) Self-correction：用"试步处 Lagrange 值大"的点替换 $\mathcal Y$ 中一点。** 前置条件 $I _ { i m p } = 0$（(i) 没做才轮到它）。
$j _ { k } ^ { * } = \arg \max _ { j = 1 , \dots , n } \vert \ell _ { j } ( s _ { k } ) \vert$。若 $\vert \ell _ { j _ { k } ^ { * } } ( s _ { k } ) \vert > \Lambda _ { s c }$：$s ^ { * } := y _ { j _ { k } ^ { * } }$，$\mathcal Y _ { k + \frac 1 2 } = \mathcal Y _ { k } \setminus \{ y _ { j _ { k } ^ { * } } \} \cup \{ s _ { k } \}$，重算 Lagrange；否则 $s ^ { * } := s _ { k }$，$\mathcal Y _ { k + \frac 1 2 } = \mathcal Y _ { k }$。
**为什么 $\vert \ell _ { j } ( s _ { k } ) \vert$ 大 = 这一步值得做**（原文在此处只说"can be quantified by"，给出严格理由）：模型与函数在第 2 行的**残差**是 $r := f ( x _ { k } + s _ { k } ) - m _ { k } ( x _ { k } + s _ { k } )$。若在 $\tilde { \mathcal Y } = \mathcal Y \setminus \{ y _ { j } \} \cup \{ s _ { k } \}$ 上重解插值，新模型的残差变化量至少是 $\vert \ell _ { j } ( s _ { k } ) \vert \cdot \vert r \vert$ 级别——因为 $\ell _ { j }$ 正是"在 $s_k$ 处插入一个单位脉冲、在其余点处为 0"的那个基函数（Definition 3.1 的 $\delta _ { i j }$ 性质）。故 $\vert \ell _ { j } ( s _ { k } ) \vert$ 是"该点携带多少未被模型吸收的信息"的精确度量，$\Lambda _ { s c } \ge 1$ 表示"至少要有原来那么大的信息量才值得动集合"。
**$\mathcal Y _ { k + \frac 1 2 }$ 这个半下标**：表示"本轮 $\mathcal Y$ 的中间状态"，(iii) 在它基础上继续改，(iv) 用 $s^*$（(ii) 丢弃的点）候选给 $\mathcal Z$。注意 (ii) **不置 $I_{imp}=1$**：self-correction 不算"几何改进"，不冻结半径？——不对，$I_{imp}$ 只在 (i) 和 (iii) 里被置 1，(ii) 里不改。所以做过 self-correction 但没做过 (i)/(iii) 的轮次，$I_{imp}$ 仍为 0，**半径照样收缩**。这是有意为之：self-correction 不保证改善 poisedness（它改善的是拟合残差），所以不能因此免除半径收缩的责任。这一点原文没写，是我从 $I_{imp}$ 的赋值位置读出来的。

**(iii) 几何修正：替换"坏"点。** 前置 $I _ { i m p } = 0$。
$( j _ { k } ^ { * } , s _ { k } ^ { * } ) = \arg \max _ { j = 1 , \dots , p , \; s \in B ( 0 , \Delta _ { k } ) } \vert \ell _ { j } ( s ) \vert$。
**上标 $p$ 是笔误**：线性 Lagrange 多项式只有 $n$ 个（$|\mathcal Y_k| = n$），引理 4.3 的第一个 claim 里同一个算子写成 $\arg \max _ { j = 1 , \dots , n }$，可交叉验证。
若 $\vert \ell _ { j _ { k } ^ { * } } ( s _ { k } ^ { * } ) \vert > \Lambda$：算 $f ( x _ { k } + s _ { k } ^ { * } )$（**1 次新 oracle**），$\mathcal Y _ { k + 1 } = \mathcal Y _ { k + \frac 1 2 } \setminus \{ y _ { j _ { k } ^ { * } } \} \cup \{ s _ { k } ^ { * } \}$，重算 Lagrange，$I _ { i m p } := 1$。否则 $\mathcal Y _ { k + 1 } = \mathcal Y _ { k + \frac 1 2 }$（$I_{imp}$ 保持 0）。
**这一行是"poisedness 判定"的执行体**：$\max _ { j } \max _ { s \in B ( 0 , \Delta _ { k } ) } \vert \ell _ { j } ( s ) \vert$ 正是 Definition 3.2 里被 $\Lambda$ 界住的那个量。它 $\le \Lambda$ ⟺ 集合已 $\Lambda$-poised ⟹ 什么都不做、$I_{imp}=0$ ⟹ 半径收缩。它就是第 4 节开头"若无法改进，则集合已 poised，故本轮 unsuccessful"那句的形式化。
**注意 $s _ { k } ^ { * }$ 总是取在球面上**（同 (i) 的闭式解），所以新点半径 $\ge \Delta_k/\Lambda$ 且 $= \Delta _ { k }$，与引理 4.3 基础情形一致。

**(iv) 尝试用现成的点改进 $\mathcal Z$。对每个已定义的 $s \in \{ s ^ { * } , s _ { k } ^ { * } \}$ 重复**：

- 若 $| \mathcal Z _ { k } | < p$：$\mathcal Z _ { k + 1 } = \mathcal Z _ { k } \cup \{ s \}$（**免费**：$s$ 必是已求过值的点）。
- 否则 $i _ { k } ^ { * } = \arg \max _ { i = 1 , \dots , n } \Vert z _ { i } \Vert$（原文这里写 $i = 1 , \dots , n$，**应为 $1 , \dots , p$**：$\mathcal Z$ 有 $p$ 个点；同段前一处 $\arg\max_{i=1,\dots,p}$ 是对的）。若 $\Vert z _ { i _ { k } ^ { * } } \Vert > \Vert s \Vert$ 则 $\mathcal Z _ { k + 1 } = \mathcal Z _ { k } \setminus \{ z _ { i _ { k } ^ { * } } \} \cup \{ s \}$。
- 否则 $\mathcal Z$ 不变。

**这一步不置 $I _ { i m p }$，也不影响半径**——它纯粹是"把已经花钱算到的信息塞进模型"的免费增益。替换准则"换掉最远的"与 E2 推论 $\Vert y _ { i } \Vert \ge \Delta / \Lambda$ 的方向一致：$\mathcal Z$ 上没有 poisedness 约束，但仍希望点不要离中心太远，否则 (4.1) 的残差是在 $\Delta _ { k }$ 球外拟合的，对球内的模型精度帮助有限。

**半径更新（第 4 行末尾）**：$I _ { i m p } = 1 \implies \Delta _ { k + 1 } = \Delta _ { k }$；$I _ { i m p } = 0 \implies \Delta _ { k + 1 } = \gamma \Delta _ { k }$。
**为什么 model-improving 时半径不动**：本轮失败的原因可能是模型几何差，而不是半径太大。先固定半径把几何修好，才有资格判断"是不是 $\Delta$ 太大"。这条正是 Algorithm 1 的三分支逻辑的细化，也保证了一串连续 model-improving 迭代中 $\Delta _ { k }$ **恒定**——引理 4.2/4.3 能把 $\Delta$ 归一成 1 全靠这一点（否则球在变，"$\Lambda$-poised in $B ( 0 , \Delta _ { k } )$"的判定目标也在变）。

**每轮 oracle 代价 $\le 2$（第 4 节开头只对 [11] 陈述过，这里对 Algorithm 2 重算一遍）**：
第 2 行的 $f ( x _ { k } + s _ { k } )$ 恒 1 次。(i) 最多再加 1 次（仅当 $\vert \ell _ { j ^ { * } } ( s _ { k } ) \vert = 0$ 需要另取 $s_k^*$）；(i) 一旦执行就置 $I _ { i m p } = 1$，(ii)(iii) 全被前置条件挡住 ⟹ 分支互斥。(ii) 只用已算好的 $s _ { k }$，0 次。(iii) 只在 $I _ { i m p } = 0$ 时执行，最多加 1 次。(iv) 0 次。合计 $\le 1 + 1 = 2$。
成功轮与 unsuccessful 轮各恰好 1 次。于是

$$
\mathcal { C } _ { \epsilon } \le | \mathcal { S } _ { \epsilon } | + | \mathcal { U } _ { \epsilon } | + 2 | \mathcal { M } _ { \epsilon } | \le ( 1 + 2 L _ { \max } ) ( | \mathcal { S } _ { \epsilon } | + | \mathcal { U } _ { \epsilon } | ) , \quad L _ { \max } := 2 n \log n + 4 n + n | \log \log \Lambda | ,
$$

第二个不等号用到 $| \mathcal { M } _ { \epsilon } | \le L _ { \max } ( | \mathcal { S } _ { \epsilon } | + | \mathcal { U } _ { \epsilon } | )$：每一段连续 model-improving 迭代都以一次非 model-improving 迭代结束（超过 $L _ { \max }$ 后集合必已 poised，(iii) 检查失败 ⟹ $I_{imp}=0$ ⟹ 该轮 unsuccessful），而引理 2.7 已说明第 $K _ { \epsilon } - 1$ 轮必是成功的，序列不会以 M 段结尾。

---

### F3. Algorithm 2 与 Algorithm 1 的差异表

| 项 | Algorithm 1 | Algorithm 2 |
| --- | --- | --- |
| 模型来源 | 抽象假设 fully-linear | (4.1) 显式构造 |
| 采样集 | 未指定 | $\mathcal Y$（$n$ 点，管 poisedness）+ $\mathcal Z$（$\le p$ 点，只管距离） |
| 失败时动作 | 直接判 unsuccessful / 笼统 model-improving | (i)(ii)(iii) 三级瀑布 + (iv) 免费增益 |
| 半径为 0 下界？ | 无 $\sigma_k$ | 无（$\sigma _ { k }$ 是 Algorithm 5 才引入的实现件） |
| criticality step | 由 $\Vert g _ { k } \Vert \ge \eta _ { 2 } \Delta _ { k }$ 取代 | 同样由该条件取代 |
| 每轮 oracle | 1 | $\le 2$ |

**随机性来源**：Algorithm 2 **完全确定**（$\arg\max$ 打破平局的规则除外），第 4 节没有任何概率论证；随机性只在第 5 节（随机子空间 $Q$）出现。这一点关系到"期望复杂度 vs 最坏情形复杂度"的区分：第 4 节给的是**逐样本最坏情形上界**，第 5 节给的是 $\mathbb E [ \mathcal C _ { \epsilon } ]$。

---

### F4. (4.4)：Lagrange 多项式的秩一更新

原文：替换 $y _ { j _ { k } ^ { * } } \to \tilde { s }$ 后

$$
\ell _ { j _ { k } ^ { * } } ^ { + } ( x ) = \frac { \ell _ { j _ { k } ^ { * } } ( x ) } { \ell _ { j _ { k } ^ { * } } ( \tilde { s } ) } , \qquad \ell _ { i } ^ { + } ( x ) = \ell _ { i } ( x ) - \ell _ { i } ( \tilde { s } ) \ell _ { j _ { k } ^ { * } } ^ { + } ( x ) \quad ( i \ne j _ { k } ^ { * } ) .
$$

**逐符号**：$\tilde { s }$ 是本轮新插入的点（$s _ { k }$ 或 $s _ { k } ^ { * }$，取决于 (i)/(ii)/(iii)）；上标 $+$ 表示"关于新集 $\tilde { \mathcal Y } = \mathcal Y \setminus \{ y _ { j ^ { * } } \} \cup \{ \tilde { s } \}$ 的 Lagrange 基"；分母 $\ell _ { j ^ { * } } ( \tilde { s } ) \ne 0$ 由 (i)/(ii)/(iii) 各处的守卫保证。

**验证它确实是新集的 Lagrange 基**（四条，一条不漏；$\tilde {\mathcal Y}$ 有 $n$ 个点，$n$ 个基函数，一一对应）：

1. $\ell _ { j ^ { * } } ^ { + } ( \tilde { s } ) = \ell _ { j ^ { * } } ( \tilde { s } ) / \ell _ { j ^ { * } } ( \tilde { s } ) = 1$。
2. $i \ne j ^ { * }$：$\ell _ { i } ^ { + } ( \tilde { s } ) = \ell _ { i } ( \tilde { s } ) - \ell _ { i } ( \tilde { s } ) \cdot 1 = 0$。
3. $m \ne j ^ { * }$，在旧点 $y _ { m }$ 处：$\ell _ { j ^ { * } } ^ { + } ( y _ { m } ) = 0 / \ell _ { j ^ { * } } ( \tilde { s } ) = 0$（因 $\ell _ { j ^ { * } } ( y _ { m } ) = \delta _ { j ^ { * } m } = 0$）；$\ell _ { m } ^ { + } ( y _ { m } ) = 1 - \ell _ { m } ( \tilde { s } ) \cdot 0 = 1$；$i \ne m$、$i \ne j ^ { * }$ 时 $\ell _ { i } ^ { + } ( y _ { m } ) = 0 - \ell _ { i } ( \tilde { s } ) \cdot 0 = 0$。
4. 每次赋值都是线性组合，$\ell ^ { + }$ 仍属 $\mathcal P$（齐次线性）。

因为 poised 集的 Lagrange 基唯一（E1 的存在性推导：$Y ^ { \top } a _ { j } = e _ { j }$ 有唯一解），(4.4) 就是**唯一正确的**更新式，计算代价 $\mathcal O ( n ^ { 2 } )$（$n$ 个基、每个基是 $n$ 维系数向量的一次线性组合），远小于重解 $n \times n$ 线性系统的 $\mathcal O ( n ^ { 3 } )$。

**与 poisedness 守卫的等价性（把 (4.4) 和引理 4.2 缝在一起的那条恒等式）**：
设 $Y ^ { + }$ 是把 $Y$ 的第 $j$ 列换成 $\tilde s$ 的矩阵。写 $\tilde { s } = Y a$，$a = Y ^ { - 1 } \tilde { s }$。按列替换的行列式性质（行列式对第 $j$ 列是线性映射，且 $\det [ y _ { 1 } , \dots , Y a , \dots , y _ { n } ] = \det ( Y ) \cdot a _ { j }$，因为把 $a _ { j } y _ { j }$ 之外的 $a _ { i } y _ { i }$ 项加到第 $j$ 列不改变行列式）：

$$
\det Y ^ { + } = \det Y \cdot a _ { j } = \det Y \cdot e _ { j } ^ { \top } Y ^ { - 1 } \tilde { s } = \det Y \cdot ( Y ^ { - T } e _ { j } ) ^ { \top } \tilde { s } = \det Y \cdot \ell _ { j } ( \tilde { s } ) .
$$

三重含义一次到手：(a) $\ell _ { j } ( \tilde { s } ) \ne 0 \iff \det Y ^ { + } \ne 0 \iff$ 新集 poised ⟹ **守卫条件不是实现细节而是数学必需**；(b) $\ell _ { j } ( \tilde { s } )$ 的大小正是"替换带来的行列式（=体积）放大倍数"；(c) 引理 4.2 里 $| \det Y ^ { + } | = | \det Y | \, | \ell _ { i } ( s ) |$ 就是这一条，原文称"by Cramer's rule"，用 Cramer 法则推同一式子同样成立。

**原文末尾那句 "we assume that $\mathcal { V } _ { 0 }$ is poised then so are all consequent ${ \mathcal { V } } _ { k }$ sets by construction"** 现在有了严格内容：所有 $\mathcal Y$ 替换都带 $\ell \ne 0$ 守卫，故归纳保持可逆。但要**注意**：初值 $\mathcal Y _ { 0 }$ 由 §6 的规则 $\{ \Delta _ { 0 } u _ { i } \}$（随机旋转后的坐标向量）给出，$u_i$ 正交 ⟹ $Y_0 = \Delta_0 U$ 正交可逆，poised 成立，且 $\Lambda = 1$（$Y _ { 0 } ^ { - T } = U / \Delta _ { 0 }$，列范数 $1 / \Delta _ { 0 }$，$\max _ { B ( 0 , \Delta _ { 0 } ) } | \ell _ { i } | = \Delta _ { 0 } \cdot ( 1 / \Delta _ { 0 } ) = 1$）——**初始集恰好是 1-poised 的**，这条对实践很重要：算法开局几何质量最优，之后只在被拒绝步的扰动下退化。

---

### F5. Lemma 4.2：从任意 $\Lambda _ { 0 }$-poised 到 $\Lambda$-poised 的迭代数

**原文**：$\mathcal P$ 为线性多项式（$p = n$）。若 $\mathcal Y _ { k }$ 在 $B ( 0 , \Delta )$ 中 $\Lambda _ { 0 }$-poised，则至多再需

$$
\left\lceil n \log n + n \left| \log \log \Lambda _ { 0 } \right| + n \left| \log \log \Lambda \right| \right\rceil
$$

次连续 model-improving 迭代。

**$\Delta = 1$ 的归约（原文 "w.l.o.g."，这里给验证）**：令 $Y ^ { \prime } = Y / \Delta$，则 $\ell _ { i } ^ { \prime } ( x ) = ( Y ^ { \prime - T } ) _ { i } ^ { \top } x = \Delta ( Y ^ { - T } ) _ { i } ^ { \top } x$，于是 $\max _ { x \in B ( 0 , 1 ) } | \ell _ { i } ^ { \prime } ( x ) | = \Delta \max _ { x \in B ( 0 , 1 ) } | ( Y ^ { - T } ) _ { i } ^ { \top } x | = \max _ { x \in B ( 0 , \Delta ) } | \ell _ { i } ( x ) |$（最后一个等号是把 $x \to \Delta x$ 的缩放写开）。两端的最大值相同，且 $\mathcal Y \subset B ( 0 , \Delta ) \iff \mathcal Y ^ { \prime } \subset B ( 0 , 1 )$。**poisedness 常数在整体缩放下不变**，故可设 $\Delta = 1$。这也是引理 4.3 里同样写"assume $\Delta _ { k } = 1$"的合法性来源。

**步骤 1（Hadamard 不等式给出体积下界）**。Hadamard：对列向量 $m _ { i }$ 构成的 $M$，$| \det M | \le \prod _ { i } \Vert m _ { i } \Vert$。理由（一句话可核对）：Gram–Schmidt 把 $M$ 写成 $Q R$（$Q$ 正交、$R$ 上三角），$| \det M | = \prod _ { i } | R _ { i i } |$，而 $| R _ { i i } |$ 是第 $i$ 列正交化后的高度 $\le \Vert m _ { i } \Vert$。
取 $M = Y ^ { - T }$，其列为 $( Y ^ { - T } ) _ { i }$，由 E1 恒等式与 $\Lambda _ { 0 }$-poised：$\Vert ( Y ^ { - T } ) _ { i } \Vert = \max _ { B ( 0 , 1 ) } | \ell _ { i } | \le \Lambda _ { 0 }$。所以

$$
| \det ( Y ^ { - T } ) | \le \prod _ { i } \Vert ( Y ^ { - T } ) _ { i } \Vert \le \Lambda _ { 0 } ^ { n } \implies | \det Y | \ge \Lambda _ { 0 } ^ { - n } ,
$$

用了 $| \det ( Y ^ { - T } ) | = | \det Y | ^ { - 1 }$。

**步骤 2（体积上界）**：$\mathcal Y \subset B ( 0 , 1 )$ 给 $\Vert y _ { i } \Vert \le 1$，对 $Y$ 再用 Hadamard：$| \det Y | \le \prod _ { i } \Vert y _ { i } \Vert \le 1$。

**步骤 3（"坏度"与体积的关系——原文此处跳步，补完）**：要证

$$
\max _ { i \in [ n ] } \max _ { x \in B ( 0 , 1 ) } | \ell _ { i } ( x ) | \le | \det Y | ^ { - 1 } ,
$$

即 $\max _ { i } \Vert ( Y ^ { - T } ) _ { i } \Vert \le | \det Y | ^ { - 1 }$。**用"高"的几何论证**：记 $\tilde { h } _ { i }$ 为 $y _ { i }$ 到 $\mathrm { span } \{ y _ { j } : j \ne i \}$ 的距离。
(a) $( Y ^ { - T } ) _ { i }$ 垂直于该子空间：对 $j \ne i$，$( Y ^ { - T } ) _ { i } ^ { \top } y _ { j } = \ell _ { i } ( y _ { j } ) = \delta _ { i j } = 0$。
(b) 由 $\ell _ { i } ( y _ { i } ) = 1$，$( Y ^ { - T } ) _ { i }$ 在 $y _ { i }$ 方向上的分量满足 $\Vert ( Y ^ { - T } ) _ { i } \Vert \cdot \tilde { h } _ { i } \ge | ( Y ^ { - T } ) _ { i } ^ { \top } y _ { i } | = 1$，而反向地由 (a)，$y _ { i }$ 沿法向的分量恰为 $\tilde { h } _ { i }$ 乘以单位法向量，故 $( Y ^ { - T } ) _ { i }$ 与该单位法向量平行，等号成立：$\Vert ( Y ^ { - T } ) _ { i } \Vert = 1 / \tilde { h } _ { i }$。
(c) 记 $\mathrm { base } _ { i }$ 为 $\{ y _ { j } : j \ne i \}$ 张成的 $( n { - } 1 )$ 维平行多面体的体积，则平行多面体体积 $V = | \det Y | = \mathrm { base } _ { i } \cdot \tilde { h } _ { i }$（底乘高；$n = 2$ 时可手算核对），故 $\mathrm { base } _ { i } = V \Vert ( Y ^ { - T } ) _ { i } \Vert$。
(d) Hadamard 给 $\mathrm { base } _ { i } \le \prod _ { j \ne i } \Vert y _ { j } \Vert \le 1$，所以 $V \Vert ( Y ^ { - T } ) _ { i } \Vert \le 1$，即 $\Vert ( Y ^ { - T } ) _ { i } \Vert \le V ^ { - 1 } = | \det Y | ^ { - 1 }$。$\blacksquare$（(b) 的"等号"与 (c)(d) 合起来才给出精确常数，绕开 $\sqrt n$ 松弛。）
由步骤 3：**只要** $| \det Y | \ge \Lambda ^ { - 1 }$，就有 $\max _ { i } \max _ { B ( 0 , 1 ) } | \ell _ { i } | \le \Lambda$，即 $\Lambda$-poised。这把"几何条件"彻底换成了"体积条件"——这是整个引理 4.2 的设计动机。

**步骤 4（一次 (iii) 修正让体积至少变成 $|\det Y|^{1-1/n}$）**：由 F4 的 Cramer 恒等式与 (iii) 的选点规则（**故意**挑 $\ell$ 最大的那个 $j$ 和那个 $s$）：

$$
| \det Y ^ { + } | = | \det Y | \cdot | \ell _ { j ^ { * } } ( s _ { k } ^ { * } ) | = | \det Y | \cdot \max _ { i \in [ n ] } \Vert ( Y ^ { - T } ) _ { i } \Vert .
$$

再用 AM–GM 链（原文一行，这里拆到每一步；记 $u _ { i } = ( Y ^ { - T } ) _ { i }$）：

$$
\max _ { i } \Vert u _ { i } \Vert ^ { 2 } \ge \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Vert u _ { i } \Vert ^ { 2 } = \frac { 1 } { n } \Vert Y ^ { - T } \Vert _ { F } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } \ge \left( \prod _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } \right) ^ { 1 / n } = | \det Y ^ { - T } | ^ { 2 / n } = | \det Y | ^ { - 2 / n } ,
$$

依据依次为：最大值 $\ge$ 平均值；Frobenius 范数定义 $\Vert M \Vert _ { F } ^ { 2 } = \sum _ { i } \Vert M _ { : i } \Vert ^ { 2 } = \sum _ { i } \sigma _ { i } ^ { 2 }$（奇异值分解 + 正交不变性）；$n$ 个非负数的算术平均 $\ge$ 几何平均；$\prod _ { i } \sigma _ { i } = | \det |$。
开方并代回：**$\vert\det Y^+\vert\ge\vert\det Y\vert^{1-1/n}$**。（原文把 $\sqrt [ n ] { \sigma _ { i } ( Y ^ { - T } ) ^ { 2 } }$ 写成这样，指标 $i$ 在根号里是笔误，应为 $\sqrt [ n ] { \prod _ { i } \sigma _ { i } ^ { 2 } }$。）

**步骤 5（对数尺度上的等比收缩）**：两边取 $\log$，注意 $| \det Y | \le 1$ 故 $\log | \det Y | \le 0$：

$$
\log | \det Y ^ { + } | \ge \left( 1 - \frac { 1 } { n } \right) \log | \det Y | .
$$

令 $d _ { k } := - \log | \det Y _ { k } | \ge 0$，则 $d _ { k + 1 } \le ( 1 - 1 / n ) d _ { k }$，即 $d _ { t } \le ( 1 - 1 / n ) ^ { t } d _ { 0 }$。
**两次修正的复合（(ii) 后接 (iii)）为什么同样收缩**：(ii) 让 $\log | \det Y ^ { + } | = \log ( | \det Y | \, | \ell _ { j ^ { * } } ( s _ { k } ) | ) \ge \log | \det Y |$（因触发条件 $\vert \ell _ { j ^ { * } } ( s _ { k } ) \vert > \Lambda _ { s c } \ge 1$），即体积**不减**（对数不减、且都 $\le 0$）；(iii) 再乘上 $( 1 - 1 / n )$（注意乘 $(1-1/n)<1$ 使负数**变大**，即更接近 0）。合起来仍是每轮至少 $( 1 - 1 / n )$ 收缩。原文写成一行链式不等式，逻辑就是这个。

**步骤 6（数迭代）**：目标是 $d _ { t } \le \log \Lambda$。充分条件 $( 1 - 1 / n ) ^ { t } d _ { 0 } \le \log \Lambda$。取对数：$t \log ( 1 - 1 / n ) + \log d _ { 0 } \le \log ( \log \Lambda )$，注意 $\log ( 1 - 1 / n ) < 0$，**除过去要变号**：

$$
t \ge \frac { \log \left( d _ { 0 } / \log \Lambda \right) } { - \log \left( 1 - \frac { 1 } { n } \right) } .
$$

（原文此处写成分母 $\log ( 1 - 1 / n )$、没写负号，**符号是乱的**；但结论式正确，按上面这条读即可。）
再用 $- \log ( 1 - \frac { 1 } { n } ) = \log ( 1 + \frac { 1 } { n - 1 } ) \ge \frac { 1 } { n }$（依据 $\log ( 1 + x ) \ge \frac { x } { 1 + x }$，$x > - 1$；取 $x = \frac { 1 } { n - 1 }$ 得 $\frac { x } { 1 + x } = \frac { 1 } { n }$），于是分母 $\ge 1 / n$ ⟹ 整体 $\le n \log \frac { d _ { 0 } } { \log \Lambda }$。
代入 $d _ { 0 } \le n \log \Lambda _ { 0 }$（步骤 1）：

$$
t \le \left\lceil n \log \frac { n \log \Lambda _ { 0 } } { \log \Lambda } \right\rceil = \left\lceil n \log n + n \log \log \Lambda _ { 0 } - n \log \log \Lambda \right\rceil \le \left\lceil n \log n + n | \log \log \Lambda _ { 0 } | + n | \log \log \Lambda | \right\rceil ,
$$

与原文一致。$\blacksquare$

**两个必须提醒读者的性质**：

1. $\log \log \Lambda$ 只在 $\Lambda > 1$ 时有定义且 $\log \Lambda > 0$；Algorithm 2 的输入条件正好要求 $\Lambda > 1$。$\Lambda \to 1 ^ { + }$ 时 $\log \log \Lambda \to - \infty$，所以这一项 $\to + \infty$ ——**追求完美几何的代价是发散的**，这正是实践把 $\Lambda$ 放宽到 $1 0 0 0$ 的理论出口。定量：$\Lambda = 1 + \frac { 1 } { n }$ 时 $\log \Lambda \approx \frac { 1 } { n }$，$\left| \log \log \Lambda \right| \approx \log n$，于是额外项 $n \left| \log \log \Lambda \right| \approx n \log n$ ——**这个 $\log n$ 就是推论 4.4 里 $\log n$ 的全部来源**。
2. 引理 4.2 只数了"从 $\Lambda _ { 0 }$ 到 $\Lambda$"，**不含**把出界点清掉的迭代，那部分在引理 4.3 里。

---

### F6. Lemma 4.3：$2 n$ 轮内达到 $1 4 ^ { n }$-poised

**原文**：$p = n$。至多 $2 n$ 次连续 model-improving 迭代后 $\mathcal Y _ { k }$ 是 $1 4 ^ { n }$-poised。

**阶段划分**：前 $n$ 轮由 (i) 清除出界点——每轮 (i) 恰好删掉一个 $\Vert y \Vert > \Delta _ { k }$ 的点并插入球内（$s _ { k }$ 满足 $\Vert s _ { k } \Vert \le \Delta _ { k }$；(i) 的备选点取在球面 $\Vert s \Vert = \Delta _ { k }$），至多 $n$ 轮清空，代价至多 $n$ 次 oracle。此后 $\Delta _ { k }$ 在这串迭代里恒定（$I _ { i m p } = 1 \implies$ 半径不动），故可归一 $\Delta _ { k } = 1$。

**子空间 poisedness（原文在这里临时扩展定义）**：称 $\mathcal Y$ 在子空间 $S$ 中 $\Lambda$-poised，若 $\max _ { x \in B ( 0 , 1 ) \cap S } | \ell _ { i } ( x ) | \le \Lambda$ 对 $\ell _ { i } ( s ) = ( Y ^ { - T } ) _ { i } ^ { \top } s$ 成立。
**基础事实**：若 $\mathcal Y$ 含单位向量 $y _ { 1 }$，则 $\mathcal Y$ 在 $S = \mathrm { span } \{ y _ { 1 } \}$ 中 $1$-poised。验证：$B ( 0 , 1 ) \cap S = \{ \alpha y _ { 1 } : | \alpha | \le 1 \}$，$\ell _ { 1 } ( \alpha y _ { 1 } ) = \alpha \ell _ { 1 } ( y _ { 1 } ) = \alpha$（齐次线性 + Definition 3.1），$| \alpha | \le 1$；$i \ne 1$ 时 $\ell _ { i } ( \alpha y _ { 1 } ) = 0$。所以子空间上最大值 $\le 1$，即 $1$-poised。

**Claim 1（(ii) 至多把子空间 poisedness 放大 2 倍）**。设 $\mathcal Y$ 在 $S$ 中 $\Lambda$-poised，(ii) 用 $s _ { k }$ 替换 $y _ { j ^ { * } }$，触发条件给 $\vert \ell _ { j ^ { * } } ( s _ { k } ) \vert > \Lambda _ { s c } \ge 1$，且 $j ^ { * } = \arg \max _ { j } \vert \ell _ { j } ( s _ { k } ) \vert$ 给 $\left| \ell _ { i } ( s _ { k } ) \right| \le \left| \ell _ { j ^ { * } } ( s _ { k } ) \right|$。由 (4.4)：

$$
\max _ { x \in B ( 0 , 1 ) \cap S } \left| \ell _ { j ^ { * } } ^ { + } ( x ) \right| = \frac { \max _ { x } | \ell _ { j ^ { * } } ( x ) | } { | \ell _ { j ^ { * } } ( s _ { k } ) | } \le \frac { \Lambda } { 1 } \le 2 \Lambda ,
$$

$$
\max _ { x \in B ( 0 , 1 ) \cap S } \left| \ell _ { i } ^ { + } ( x ) \right| \le \max _ { x } | \ell _ { i } ( x ) | + \frac { | \ell _ { i } ( s _ { k } ) | } { | \ell _ { j ^ { * } } ( s _ { k } ) | } \max _ { x } | \ell _ { j ^ { * } } ( x ) | \le \Lambda + 1 \cdot \Lambda = 2 \Lambda ,
$$

第二个用了三角不等式 $\left| a - b c \right| \le \left| a \right| + \left| b \right| \left| c \right|$（把 $\ell _ { i } ^ { + } ( x ) = \ell _ { i } ( x ) - \frac { \ell _ { i } ( s _ { k } ) } { \ell _ { j ^ { * } } ( s _ { k } ) } \ell _ { j ^ { * } } ( x )$ 里的系数当常数）。$\blacksquare$（注意原文写 $\frac { \Lambda } { 1 } \le 2 \Lambda$ 是故意留了松弛，为的是让 (ii)+(iii) 相乘时凑出 $2 \times 7 = 1 4$。）

**Claim 2（(iii) 让 poisedness 放大至多 7 倍，同时子空间升一维）**。设 $\mathcal Y$ 在**真**子空间 $S$ 中 $\Lambda$-poised。
先处理简单分支：若 $\mathcal Y$ 在**全空间**已 $2 \Lambda$-poised，则 $\max _ { B ( 0 , 1 ) } | \ell _ { j ^ { * } } ^ { + } | = 1$（因分母=分子的最大值，商为 1）且 $\max | \ell _ { i } ^ { + } | \le 2 \Lambda + 2 \Lambda = 4 \Lambda \le 7 \Lambda$，故 $S ^ { + }$ 可任取高一维的子空间。⟹ 以下假设**不是**全空间 $2 \Lambda$-poised。
记 $P _ { S }$、$P _ { S } ^ { \perp }$ 为正交投影（$P_S$ 对称幂等，$P_S+P_S^\perp=I$），$\ell ( x ) = ( \ell _ { 1 } ( x ) , \dots , \ell _ { n } ( x ) ) ^ { \top }$，$\Vert \ell ( x ) \Vert _ { \infty } = \max _ { j } | \ell _ { j } ( x ) |$。
(a) $\Vert \ell ( s _ { k } ^ { * } ) \Vert _ { \infty } > 2 \Lambda$：因为"不是全空间 $2 \Lambda$-poised"⟹ (iii) 会取到 $\max _ { j , s } | \ell _ { j } ( s ) | > 2 \Lambda$（否则它检查的值 $\le\Lambda$，(iii) 根本不会触发；$2\Lambda$ 是后面归纳用的量）。
(b) $\Vert \ell ( P _ { S } s _ { k } ^ { * } ) \Vert _ { \infty } \le \Lambda$：**需要 $P _ { S } s _ { k } ^ { * } \in B ( 0 , 1 ) \cap S$**——正交投影不增范数（$\Vert P _ { S } v \Vert ^ { 2 } = v ^ { \top } P _ { S } ^ { \top } P _ { S } v = v ^ { \top } P _ { S } v \le \Vert v \Vert ^ { 2 }$，用 $P _ { S }$ 对称幂等 + Cauchy–Schwarz），故 $\Vert P _ { S } s _ { k } ^ { * } \Vert \le \Vert s _ { k } ^ { * } \Vert \le 1$；再用在 $S$ 中的 $\Lambda$-poisedness。
(c) 三角不等式（方向 $\Vert a + b \Vert \ge \Vert a \Vert - \Vert b \Vert$，这里写成原文的形式）：

$$
2 \Lambda < \Vert \ell ( s _ { k } ^ { * } ) \Vert _ { \infty } = \Vert \ell ( P _ { S } ^ { \perp } s _ { k } ^ { * } ) + \ell ( P _ { S } s _ { k } ^ { * } ) \Vert _ { \infty } \le \Vert \ell ( P _ { S } ^ { \perp } s _ { k } ^ { * } ) \Vert _ { \infty } + \Lambda \implies \Vert \ell ( P _ { S } ^ { \perp } s _ { k } ^ { * } ) \Vert _ { \infty } > \Lambda .
$$

第一步等号用 $\ell$ 的线性（齐次线性多项式的向量值映射是线性的）与 $P _ { S } ^ { \perp } s + P _ { S } s = s$。
(d) 齐次 + 最优性：$v : = P _ { S } ^ { \perp } s _ { k } ^ { * }$ 满足 $\ell ( v ) = \Vert v \Vert \ell ( v / \Vert v \Vert )$（一次齐性），且 $v / \Vert v \Vert$ 是单位向量 $\in B ( 0 , 1 )$，故由 $( j _ { k } ^ { * } , s _ { k } ^ { * } )$ 的 $\arg \max$ 定义，$\Vert \ell ( v / \Vert v \Vert ) \Vert _ { \infty } \le \Vert \ell ( s _ { k } ^ { * } ) \Vert _ { \infty }$。两边除：

$$
\Vert v \Vert = \frac { \Vert \ell ( v ) \Vert _ { \infty } } { \Vert \ell ( v / \Vert v \Vert ) \Vert _ { \infty } } \ge \frac { \Vert \ell ( v ) \Vert _ { \infty } } { \Vert \ell ( s _ { k } ^ { * } ) \Vert _ { \infty } } \overset{(a),(c)}{\ge} \frac { \Vert \ell ( v ) \Vert _ { \infty } } { \Vert \ell ( v ) \Vert _ { \infty } + \Lambda } > \frac { \Lambda } { \Lambda + \Lambda } = \frac { 1 } { 2 } ,
$$

用到 $t \mapsto \frac { t } { t + \Lambda }$ 在 $t > 0$ 上单调增（求导 $\frac { \Lambda } { ( t + \Lambda ) ^ { 2 } } > 0$）。**结论：新点必须有至少 $\frac { 1 } { 2 }$ 的"离子空间距离"**，否则它没把维度撑起来。
(e) 归纳用界：先由 Claim 1 的同一论证（此处触发值是 $\Vert \ell ( s _ { k } ^ { * } ) \Vert _ { \infty } > 2 \Lambda \ge 1$，条件 $\ge 1$ 已够）得 $\max _ { B ( 0 , 1 ) \cap S } \Vert \ell ^ { + } \Vert _ { \infty } \le 2 \Lambda$。再由 $\ell ^ { + } ( s _ { k } ^ { * } ) = e _ { j ^ { * } }$（F4 验证的第 1、2 条，$\infty$ 范数 $= 1$）：

$$
\Vert \ell ^ { + } ( v ) \Vert _ { \infty } = \Vert \ell ^ { + } ( s _ { k } ^ { * } ) - \ell ^ { + } ( P _ { S } s _ { k } ^ { * } ) \Vert _ { \infty } \le 1 + 2 \Lambda \le 3 \Lambda \implies \left\| \ell ^ { + } \left( \frac { v } { \Vert v \Vert } \right) \right\| _ { \infty } = \frac { \Vert \ell ^ { + } ( v ) \Vert _ { \infty } } { \Vert v \Vert } \le \frac { 3 \Lambda } { 1 / 2 } = 6 \Lambda ,
$$

这里 $3 \Lambda \le 6 \Lambda$ 用了 $\Lambda \ge 1$ 与 (d) 的 $\Vert v \Vert \ge 1 / 2$。
(f) 升维：令 $S ^ { + } = S \oplus \mathrm { span } \{ v \}$（直和：$v \perp S$）。任意 $x \in B ( 0 , 1 ) \cap S ^ { + }$ 唯一写成 $x = \alpha x ^ { \prime } + \beta u$，$u = v / \Vert v \Vert$，$x ^ { \prime } \in S$，$\alpha ^ { 2 } + \beta ^ { 2 } \le 1$，并且可取 $x ^ { \prime } \in B ( 0 , 1 ) \cap S$（$\Vert x ^ { \prime } \Vert \le \Vert x \Vert / \sqrt { \alpha ^ { 2 } + \beta ^ { 2 } } \le 1$ 当 $\alpha \ne 0$；$\alpha = 0$ 时无所谓）。由 $\ell ^ { + }$ 的线性 + Cauchy–Schwarz（对 $( | \alpha | , | \beta | )$ 与 $( 2 \Lambda , 6 \Lambda )$）：

$$
\Vert \ell ^ { + } ( x ) \Vert _ { \infty } \le | \alpha | \cdot 2 \Lambda + | \beta | \cdot 6 \Lambda \le \sqrt { \alpha ^ { 2 } + \beta ^ { 2 } } \cdot \sqrt { 2 ^ { 2 } + 6 ^ { 2 } } \cdot \Lambda \le \sqrt { 4 0 } \Lambda < 7 \Lambda ,
$$

$\sqrt { 4 0 } < 7 \iff 4 0 < 4 9$ ✓。**这就是 $7$ 这个数的来历**（原文写作 $\left\| { \left[ \begin {array} { l } { 2 } \\ { 6 } \end{array} \right] } \right\| \Lambda$）。

**归纳**：$t = 1$ 基础情形——(iii) 第一次执行时，新点取在球面 $\Vert s \Vert = 1$ 上（(i) 与 (iii) 的备选点都是 $\arg \max$ on 球，线性函数最大必在边界，因若 $\Vert s \Vert < 1$ 则可再乘 $1 / \Vert s \Vert$ 放大），所以 $\mathcal Y$ 含单位向量 ⟹ 由"基础事实"，$1$-poised 于一维子空间 $= 1 4 ^ { 0 }$-poised。
归纳步：第 $t + 1$ 轮若同时做 (ii)(iii)，$\Lambda \to 2 \Lambda \to 1 4 \Lambda$；若只做 (iii)，$\Lambda \to 7 \Lambda \le 1 4 \Lambda$。故 $1 4 ^ { t } \to 1 4 ^ { t + 1 }$，维度 $\ge t \to \ge t + 1$。$t = n$ 时维度 $\ge n$ 即全空间，得 $1 4 ^ { n }$-poised。合计：$n$ 轮清远点 $+ n$ 轮升维 $= 2 n$。$\blacksquare$

**这条引理的两个诚实评估**：
1. 它给的 $\Lambda _ { 0 } = 1 4 ^ { n }$ 与推论所需的 $\Lambda = 1 + \frac { 1 } { n }$ 相差指数级；引理 4.2 用 $| \log \log \Lambda _ { 0 } | = \log ( n \log 1 4 ) = \log n + \log \log 1 4$ 把指数**压成对数**——这是"指数级几何恶化只花 $\mathcal O ( n \log n )$ 步"的关键机制。
2. $1 4 = 2 \times 7$、$7 = \lceil \sqrt { 4 0 } \rceil$ 都是**分析松弛**而非下界；把 (e) 里 $1 + 2 \Lambda \le 3 \Lambda$ 与 $\sqrt { 2 ^ { 2 } + 6 ^ { 2 } }$ 两处收紧，底数可以显著下降。定理的实际内容只是"$\exp ( \mathcal O ( n ) )$ 级 poisedness，$\mathcal O ( n )$ 轮达到"。

---

### F7. Theorem 4.1：oracle 计数

**原文**：连续 model-improving 迭代串中的 oracle 调用数 $\le 4 n \log n + 8 n + 2 n | \log \log \Lambda |$。
**推导**：引理 4.3 给前 $2 n$ 轮（同时给出 $\Lambda _ { 0 } = 1 4 ^ { n }$），接引理 4.2：

$$
2 n + \left\lceil n \log n + n | \log \log 1 4 ^ { n } | + n | \log \log \Lambda | \right\rceil .
$$

展开 $| \log \log 1 4 ^ { n } |$：$\log 1 4 ^ { n } = n \log 1 4$，再取 $\log$ 得 $\log n + \log \log 1 4$。因 $\log 1 4 \approx 2 . 6 4$、$\log \log 1 4 \approx 0 . 9 7 \in ( 0 , 1 )$，且 $n \ge 1$ 时 $n \log 1 4 \ge \log 1 4 > 1$，故绝对值可以直接去掉：$n | \log \log 1 4 ^ { n } | = n \log n + n \log \log 1 4 \le n \log n + n$。代回：

$$
2 n + \left\lceil 2 n \log n + n \log \log 1 4 + n | \log \log \Lambda | \right\rceil \le 2 n + 2 n \log n + n + n | \log \log \Lambda | + 1 \le 2 n \log n + 4 n + n | \log \log \Lambda | ,
$$

最后一步：取整多出的 $1 \le n$，$2 n + n + 1 \le 4 n$ ✓（用 $2 n \log n \ge 0$，$n \ge 1$）。
每轮 $\le 2$ 次 oracle（F2 末尾已重算）⟹ 乘 2 得定理陈述。**迭代数上界** $2 n \log n + 4 n + n | \log \log \Lambda |$ 与**oracle 数上界** $4 n \log n + 8 n + 2 n | \log \log \Lambda |$ 相差恰好 2 倍，读定理时别混。

**一处逻辑缺口（值得记录）**：引理 4.2 的前提是"当前集合是 $\Lambda _ { 0 }$-poised"，而引理 4.3 结论"至多 $2 n$ 轮后达到 $1 4 ^ { n }$-poised"里的 $2 n$ 与后面的引理 4.2 计数是**串接**的（先 $2n$，再 $n\log n+\dots$），这个串接默认了"清远点与升维阶段不会破坏 $\Lambda_0$-poisedness 的前提"。原文用 $\Delta$ 恒定与 $|\det Y|$ 单调不减论证了升维段内部自洽，但**没有单独论证"$2 n$ 轮结束时集合确实 poised"**（它由 Claim 1/2 的归纳隐含给出：每一步都在保持 poised 的前提下更新——因为守卫 $\ell \ne 0$）。复核时应把这句补上。

---

### F8. Corollary 4.4：Algorithm 2 的总复杂度

**原文**：在与定理 2.8 相同假设下，对任意

$$
\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } \min \{ C _ { 2 } , L + \kappa _ { b h m } \} C _ { 1 } ^ { 2 } } } ,
$$

（原文括注"即 $\epsilon \ge \Omega ( \sqrt { n \epsilon _ { f } } ) $"），Algorithm 2 的总 oracle 复杂度满足：$\eta _ { 2 } = \sqrt n$ 时 $\mathcal { C } _ { \epsilon } \le \mathcal { O } ( n ^ { 3 / 2 } \log n \epsilon ^ { - 2 } )$；$\eta _ { 2 }$ 为常数时 $\mathcal { C } _ { \epsilon } \le \mathcal { O } ( n ^ { 2 } \log n \epsilon ^ { - 2 } )$。

**$\epsilon$ 门槛的两条来源（先合并，再检查原文常数）**：需要同时成立
(1) 引理 2.5/2.6/2.7 自己的前提 $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } C _ { 2 } C _ { 1 } ^ { 2 } } }$（保证 $\Delta _ { k } \ge \gamma C _ { 1 } \epsilon \ge \sqrt { \frac { 4 \epsilon _ { f } } { C _ { 2 } } }$，从而 $C _ { 2 } \Delta _ { k } ^ { 2 } - 2 \epsilon _ { f } \ge \frac { 1 } { 2 } C _ { 2 } \Delta _ { k } ^ { 2 }$）；
(2) 推论 3.6 的 fully-linearity 前提 $\Delta _ { k } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } }$，由 $\Delta _ { k } \ge \gamma C _ { 1 } \epsilon$ 只需 $\gamma C _ { 1 } \epsilon \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } }$。
合并（取较严者，$\frac { 1 } { \min \{ a , b \} } = \max \{ \frac { 1 } { a } , \frac { 1 } { b } \}$）：

$$
\epsilon \;\ge\; \frac { 2 \sqrt { \epsilon _ { f } } } { \gamma C _ { 1 } } \max \left\{ \frac { 1 } { \sqrt { C _ { 2 } } } , \; \sqrt { \frac { \Lambda } { L + \kappa _ { b h m } } } \right\} = \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } C _ { 1 } ^ { 2 } \min \{ C _ { 2 } , \; ( L + \kappa _ { b h m } ) / \Lambda \} } } .
$$

**与原文的差别**：原文分母里是 $\min \{ C _ { 2 } , L + \kappa _ { b h m } \}$，**丢了 $\Lambda$**。因 $\Lambda = 1 + 1 / n \le 2$，差异至多 $\sqrt { 2 }$ 倍，不影响阶，但严格陈述应带 $\Lambda$。
**顺带修正阶段 E8 的一处写法**：那里我写成 $\frac { \epsilon } { C _ { 1 } } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } }$，方向反了；正确是 $\gamma C _ { 1 } \epsilon \ge \sqrt { \cdots }$（引理 2.5 给的是半径下界 $\Delta _ { k } \ge \gamma C _ { 1 } \epsilon$，$C _ { 1 }$ 在**分子**位置）。按正确方向，$C _ { 1 } = \Theta ( n ^ { - 1 / 2 } )$ 时门槛是 $\epsilon \ge \Theta ( \sqrt n ) \cdot \sqrt { \frac { \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } }$，即若 $\kappa _ { b h m } = \mathcal O ( 1 )$ 得 $\Omega ( \sqrt { n \Lambda \epsilon _ { f } } )$；若 $\kappa _ { b h m } = \Theta ( \sqrt n )$ 则降为 $\Omega ( n ^ { 1 / 4 } \sqrt { \Lambda \epsilon _ { f } } )$。

**再修正阶段 C7/D3 的一处归因（重要）**：我此前说"$\eta _ { 2 }$ 增大通过 $C _ { 1 } ^ { - 1 }$ 的 max 结构改善复杂度"。重算 $C _ { 2 } = \frac { \eta _ { 1 } \eta _ { 2 } \kappa _ { f c d } } { 2 } \min \{ \frac { \eta _ { 2 } } { \kappa _ { b h m } } , 1 \}$ 后应为：

- $\eta _ { 2 } \ge \kappa _ { b h m }$ 时 $\min \{ \cdot , 1 \} = 1$，$C _ { 2 } = \Theta ( \eta _ { 2 } )$；
- $\eta _ { 2 } < \kappa _ { b h m }$ 时 $C _ { 2 } = \Theta ( \eta _ { 2 } ^ { 2 } / \kappa _ { b h m })$（被 Hessian 界**卡住**，二次地差）。
而 $C _ { 1 } ^ { - 1 } = \max \{ \eta _ { 2 } , \kappa _ { b h m } , \frac { 2 \kappa _ { e f } + \max \{ \eta _ { 2 } , \kappa _ { e f } \} } { ( 1 - \eta _ { 1 } ) \kappa _ { f c d } } \} + \kappa _ { e g }$ 在 $\kappa _ { e f } , \kappa _ { e g } = \Theta ( \sqrt n )$ 时**无论 $\eta _ { 2 }$ 是常数还是 $\sqrt n$ 都是 $\Theta ( \sqrt n )$**。
⟹ 定理 2.8 的界 $\frac { 1 } { C _ { 2 } ( \gamma C _ { 1 } \epsilon ) ^ { 2 } }$ 里，$\eta _ { 2 } = \sqrt n$ 带来的 $\sqrt n$ 改进**完全来自 $C _ { 2 }$**：
$\eta _ { 2 }$ 常数 ⟹ $C _ { 2 } = \Theta ( 1 )$、$C _ { 1 } ^ { 2 } = \Theta ( 1 / n )$ ⟹ $\frac { 1 } { C _ { 2 } C _ { 1 } ^ { 2 } } = \Theta ( n )$；
$\eta _ { 2 } = \sqrt n$（并 $\kappa _ { b h m } \le \mathcal O ( \sqrt n )$ 保证 $\min$ 不塌陷）⟹ $C _ { 2 } = \Omega ( \sqrt n )$ ⟹ $\frac { 1 } { C _ { 2 } C _ { 1 } ^ { 2 } } = \Omega ( \sqrt n \cdot n ^ { - 1 } \cdot n ) = \Theta ( \sqrt n )$。
与原文证明里"$C _ { 1 } ^ { - 1 } = \Theta ( \sqrt { n } )$ and $C _ { 2 } = \Theta ( \sqrt { n } )$"完全一致 ✓。这也解释了为什么推论 2.10 必须**额外假设** $\kappa _ { b h m } \le \mathcal O ( \sqrt n )$：$\kappa _ { b h m }$ 更大时 $\frac { \eta _ { 2 } } { \kappa _ { b h m } } < 1$，$C _ { 2 }$ 掉到 $\Theta ( \eta _ { 2 } ^ { 2 } / \kappa _ { b h m } )$，$\eta_2$ 的技巧失效。

**总账（把 F2 的每轮代价与定理 2.8 接起来）**：

$$
\mathcal { C } _ { \epsilon } \le ( 1 + 2 L _ { \max } ) \left( \frac { 4 ( \phi ( x _ { 0 } ) - \phi ^ { \star } ) } { C _ { 2 } ( \gamma C _ { 1 } \epsilon ) ^ { 2 } } + \left\lceil \log _ { \gamma } \frac { C _ { 1 } \epsilon } { \Delta _ { 0 } } \right\rceil \right) , \qquad L _ { \max } = 2 n \log n + 4 n + n | \log \log \Lambda | .
$$

取 $\Lambda = 1 + \frac { 1 } { n }$ ⟹ $| \log \log \Lambda | = \Theta ( \log n )$ ⟹ $L _ { \max } = \mathcal O ( n \log n )$；$\kappa _ { e g } = \Theta ( \sqrt n )$（推论 3.6 那条 $\Theta ( { \sqrt { n } } )$ 的推导：$( L + \kappa _ { b h m } ) \sqrt n \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 } = \sqrt n \cdot \Theta ( \sqrt { n } ) \cdot \Theta ( 1 ) \cdot \Theta ( 1 ) \cdot L$-因子——注意 $\kappa_{bhm}$ 与 $L$ 一起算）。于是
$\eta _ { 2 } = \sqrt n$：$\mathcal C_\epsilon = \mathcal O ( n \log n \cdot \sqrt n \epsilon ^ { - 2 } ) = \mathcal O ( n ^ { 3 / 2 } \log n \epsilon ^ { - 2 } )$ ✓；
$\eta _ { 2 }$ 常数：$\mathcal O ( n \log n \cdot n \epsilon ^ { - 2 } ) = \mathcal O ( n ^ { 2 } \log n \epsilon ^ { - 2 } )$ ✓。

**与推论 2.10 的正面对比（这才是第 4 节的成绩单）**：

| | 每轮 oracle | $\vert\mathcal S\vert+\vert\mathcal U\vert$（$\eta_2=\sqrt n$） | 总复杂度（$\eta_2=\sqrt n$） |
| --- | --- | --- | --- |
| 有限差分（推论 2.10） | $n + 1$ | $\mathcal O ( \sqrt n \epsilon ^ { - 2 } )$ | $\mathcal O ( n ^ { 3 / 2 } \epsilon ^ { - 2 } )$ |
| Algorithm 2（推论 4.4） | $\le 2$ | $\mathcal O ( \sqrt n \epsilon ^ { - 2 } )$ | $\mathcal O ( n ^ { 3 / 2 } \log n \epsilon ^ { - 2 } )$ |

**结论要直说**：就 $n$ 与 $\epsilon$ 的**阶**而言，Algorithm 2 相对有限差分**没有增益**，只多付一个 $\log n$。它的价值在于：(a) 模型是**二次**的（有限差分只能给线性模型），实践中每步下降量更大、常数 $C_2$ 之外的隐藏常数更好（第 6 节的数据证实这点）；(b) 它是**第一个**把 Powell 式 self-correction + 几何修正 + 二次模型放进 $\epsilon^{-2}$ 复杂度框架的算法，即"实践有效的东西可被分析"这一方法论主张。$n ^ { 3 / 2 }$ 的突破要到第 5 节的随机子空间（$\mathcal O ( n \epsilon ^ { - 2 } )$）才出现。

---

### F9. 第 4 节研究审计

**净增量**：(4.1) 的 Y-Z 分离 + 三级修正瀑布 (i)(ii)(iii) + 免费的 (iv) 是**新的算法设计**；引理 4.2 的"$| \det Y |$ 等比收缩"论证是对 [11]/Powell 已知论证的整理（$h _ { i }$-高度那一步原文直接跳过，F5 步骤 3 是我补的）；引理 4.3 的 $1 4 ^ { n }$ 归纳**是本文新的**（[11] 只需 $3 n$ 轮，因为它每次几何修正都要求全集合 poised，不需要"从含单位向量出发逐维升上去"）。
**理论-实现一致性**：定理要求 $\Lambda = 1 + 1 / n$、$\eta _ { 2 } = \sqrt n$，而 Algorithm 2 的输入里 $\Lambda$、$\eta _ { 2 }$ 是自由超参数——**算法不需要知道 $n$ 以外的未知量**（$L$、$\kappa_{bhm}$ 都不进输入），这点是合格的；但 Table 6.1 的实现取 $\Lambda = 1 0 0 0 \ne 1 + 1 / n$、$\eta _ { 2 } = 5 \times 1 0 ^ { - 9 } \ne \sqrt n$，即**实验用的参数不满足定理的条件**。论文在 6.2 节末尾承认了这点（"our analysis addresses the worst-case"）。这是合法的立场，但要清楚：**实验结果不构成对定理的经验验证，只是对算法实用性的验证**。
**公平成本**：每轮 $\le 2$ 次 oracle，但 (4.1) 是 $n + \frac { n ( n + 1 ) } { 2 }$ 变量的凸 QP，**单步墙钟代价 $\mathcal O ( n ^ { 6 } )$ 量级**（朴素内点法）；引理里的"oracle 复杂度"不含这个。第 6 节之所以要在实现里用 (4.4) 的秩一更新 + Powell 式 $\mathcal O ( n ^ { 2 } )$ 求解，正是为了补这一块，而**理论部分完全没有讨论计算时间与内存**，属于论文自己划定的边界（标题就是 complexity of oracle calls）。
**失效区域**：$\kappa _ { b h m } \gg \eta _ { 2 }$（真 Hessian 极大，如 §6 提到 POWERSUM 的 $1 0 ^ { 1 0 0 }$）时 $C _ { 2 }$ 二次地塌陷 ⟹ 迭代数按 $\kappa _ { b h m } / \eta _ { 2 } ^ { 2 }$ 恶化；$\epsilon$ 低于 $\Omega ( \sqrt { n \Lambda \epsilon _ { f } } )$ 时 fully-linearity 不再成立 ⟹ 界失效（不是"界变松"，而是**不成立**）。
**最小判别实验**：把 Algorithm 2 的 (ii) self-correction 关掉（只用 (i)(iii)），保持 $\Lambda$、$\eta _ { 2 }$、模型与求解器完全相同。若性能不变，则第 4 节开头关于"self-correction 是实践关键"的论证只是**理论装饰**；若显著变差，才支持"被拒步的 $\vert \ell _ { j } ( s _ { k } ) \vert$ 是信息量度量"这一机制解释。§6 的 GC-YZ-LIN vs GC-YZ-V 对照（后者移植 NEWUOA 的 self-correction 规则）**恰好是这个实验的一半**，缺的是"完全关掉"那一臂。

---

# 阶段 G：第 5 节——随机子空间中的含噪模型基信赖域

## G0 本节在贡献链中的位置，以及参考文献补录

第 2–4 节把最坏情形 oracle 复杂度压到 $\mathcal{O}(\frac{n^{3/2}\log n}{\epsilon^2})$（推论 4.4），仍然是 $n$ 的超线性函数。第 5 节换一条攻击路线：**不再优化单次建模的误差常数 $\kappa_{eg}$，而是把建模工作限制在一个 $q$ 维随机子空间里**，于是每次建模只需要 $\mathcal{O}(q)$ 次函数求值。代价是子空间可能"不朝向"真实梯度，需要随机地重抽。

本节相对前作的关键增量（作者自述，第 686 行）：文献 [11] 的子空间分析**不含噪声**；[14] 的噪声是随机的、可以按信赖域半径任意压缩。本文要做的是**固定精度的确定性噪声 oracle** $\vert f(x)-\phi(x)\vert\le\epsilon_f$。困难点在于：第 2–3 节所有 $\kappa_{eg}$ 的界都形如 $\kappa_{eg}\Delta_k$，而有限差分与插值的误差里都含有 $\frac{\epsilon_f}{\Delta_k}$ 这一项，所以必须有一个**半径下界** $\Delta_k\ge c>0$ 才能得到有限的 $\kappa_{eg}$。在 Algorithm 1 里这个下界是"免费"的：$\Delta_k\ge\gamma C_1\epsilon$（引理 2.5），只要 $\epsilon$ 足够大就自动成立。而在子空间方法里**不成立**——即使 $\Delta_k$ 已经很小，只要抽到的 $Q_k$ 与梯度近乎正交，迭代照样失败，于是半径会一路缩到 $0$。作者的结论（第 686 行）：必须人为引入两处算法改动——**放宽的接受判据**（$\rho_k$ 里加 $2\epsilon_f$）和**强制的半径下界** $\Delta_{min}$。这两处改动就是第 5 节的全部算法新意。

Markdown 转换丢失了参考文献表，下面从 `01-raw` 的 PDF 第 30–32 页补录本节与下节真正用到的条目（后文引用时只用编号）：

| 编号 | 条目 | 在本文中的作用 |
| --- | --- | --- |
| [3] | Bandeira, Scheinberg, Vicente (2014), SIOPT | $\Vert g_k\Vert\ge\eta_2\Delta_k$ 替代 criticality step 的出处 |
| [4] | Berahas, Cao, Choromanski, Scheinberg (2021) | 有限差分误差界 $\frac{\sqrt n L\delta}{2}+\frac{2\sqrt n\epsilon_f}{\delta}$ 的出处 |
| [5] | Berahas, Cao, Scheinberg (2021), SIOPT | 含噪线搜索的高概率复杂度（第 890 行类比对象） |
| [7] | Cao, Berahas, Scheinberg (2023), Math. Prog. | 含噪信赖域；引理 5.6 自称"类似 [7, Lem 4.3]" |
| [8] | Cartis, Roberts (2023), Math. Prog. | JL 嵌入子空间模型基 TR，$\mathcal{O}(n^2\epsilon^{-2})$ |
| [9] | Cartis, Roberts (2026) | 注记：把 [8] 的 JL 变换重新缩放即可达到 $\mathcal{O}(n\epsilon^{-2})$ |
| [10] | Cartis, Scheinberg (2018), Math. Oper. Res. | 概率模型的高概率/期望复杂度；$I_t,A_t,B_t$ 随机过程与引理 5.10、5.11 的出处 |
| [11] | Chaudhry, Scheinberg, ICM 2026 论文集 | 本文第 3–5 节框架的直接前作；引理 5.2/5.4、定理 3.3 的 $\Lambda$ 界、[11, Lem 6.7] 的 $\theta$ 与 $\kappa_g$ 均出自此 |
| [13] | Conn, Scheinberg, Vicente (2009) | 教材：fully-linear 模型、fcd、信赖域复杂度标准形式 |
| [14] | Dzahini, Wild (2024), SIOPT | 随机子空间 + 随机模型基 TR |
| [16] | Gratton, Royer, Vicente, Zhang (2018), IMA JNA | 子空间直接搜索把 $\mathcal{O}(n^2\epsilon^{-2})$ 降到 $\tilde{\mathcal{O}}(n\epsilon^{-2})$ |
| [21] | Powell (2004), NEWUOA | 数值对比基线；二次 Lagrange 多项式 self-correction 规则的原型 |
| [26] | Scheinberg, Toint (2010), SIOPT | self-correcting geometry 的原始出处（Algorithm 2 的第 (ii) 步） |
| [27] | Zhang (2023), PRIMA | NEWUOA 的参考实现 |

## G1 子空间嵌入与式 (5.1)：链式法则、$L_Q\le L$、以及 $Q_kQ_k^\top g_k=g_k$ 为什么"不失一般性"

**设定**：$Q\in\mathbb{R}^{n\times q}$，$q\le n$，列正交，即 $Q^\top Q=I_q$（$I_q$ 是 $q$ 阶单位阵，已知量）。$Q$ 的列张成 $\mathbb{R}^n$ 的一个 $q$ 维子空间，记作"由 $Q$ 诱导的子空间"。这里 $q$ 是**超参数**（子空间维数），$Q$ 是**算法每步随机抽取的中间结果**。

**限制到子空间上的函数**：给定中心 $x$，定义

$$
\hat{\phi}(v)=\phi(x+Qv),\qquad v\in\mathbb{R}^q.
$$

$\hat\phi$ 是 $\phi$ 沿子空间方向的 $q$ 维"切片"，是已知函数的复合，因此可求导。用一元链式法则逐分量写出它的梯度：对 $i=1,\dots,q$，

$$
\frac{\partial \hat{\phi}}{\partial v_i}(v)=\nabla\phi(x+Qv)^\top \frac{\partial(x+Qv)}{\partial v_i}=\nabla\phi(x+Qv)^\top (Qu_i),
$$

其中 $u_i$ 是 $\mathbb{R}^q$ 的第 $i$ 个坐标单位向量（已知量），$Qu_i$ 就是 $Q$ 的第 $i$ 列。把 $i=1,\dots,q$ 排成向量，正好是矩阵乘法：

$$
\nabla\hat{\phi}(v)=Q^\top\nabla\phi(x+Qv).
$$

在 $v=0$ 处取，左乘 $Q$，得到论文第 688 行断言的等式

$$
Q\nabla\hat{\phi}(0)=QQ^\top\nabla\phi(x).
$$

含义：$\nabla\hat\phi(0)\in\mathbb{R}^q$ 是**子空间内梯度的坐标表示**，而 $QQ^\top\nabla\phi(x)\in\mathbb{R}^n$ 是它在 $\mathbb{R}^n$ 里的向量表示，两者通过 $Q$ 这个"坐标—向量"同构一一对应。$P_Q=QQ^\top$ 是到该子空间的**正交投影矩阵**：它对称（$P_Q^\top=(QQ^\top)^\top=QQ^\top$）且幂等（$P_Q^2=Q(Q^\top Q)Q^\top=QQ^\top=P_Q$），这两个性质在下面每一步都会用到。

同理定义 ${\hat m}(v)=m(x+Qv)$，于是 $Q\nabla{\hat m}(0)=QQ^\top\nabla m(x)$。

**子空间模型 (5.1)**：算法在第 $k$ 步用

$$
m_k(x_k+Q_kv)=\phi(x_k)+g_k^\top Q_kv+\frac{1}{2}v^\top Q_k^\top H_kQ_kv.
\tag{见正文 (5.1)}
$$

逐个符号：$\phi(x_k)$ 是真实目标在中心的函数值（**不可得**，实现里换成 $f(x_k)$，与第 2 节同一处不一致）；$g_k\in\mathbb{R}^n$ 是模型梯度（中间结果，由子空间有限差分或子空间插值给出）；$H_k\in\mathbb{R}^{n\times n}$ 是模型 Hessian（中间结果），受假设 2.2 的 $\Vert H_k\Vert\le\kappa_{bhm}$ 约束；$v\in\mathbb{R}^q$ 是子空间内的自变量；$\frac12 v^\top Q_k^\top H_kQ_kv$ 是把 $\mathbb{R}^n$ 的二次型限制到子空间上。注意 $m_k$ 只在点集 $\{x_k+Q_kv\}$ 上有定义——它不是 $\mathbb{R}^n$ 上的函数，只是子空间上的函数，这一点与第 2 节的 $m_k$ 定义（在整个 $\mathbb{R}^n$ 上）不同。

**"不失一般性"假设 $Q_kQ_k^\top g_k=g_k$**：论文第 696 行说，对任意 $v$ 有 $g_k^\top Q_kv=g_k^\top Q_kQ_k^\top Q_kv$，所以可以假设 $g_k$ 落在子空间里。展开验证：$Q_k^\top Q_k=I_q$，故 $Q_kQ_k^\top Q_k=Q_k(I_q)=Q_k$，于是

$$
g_k^\top Q_kv=g_k^\top(Q_kQ_k^\top Q_k)v=(Q_kQ_k^\top g_k)^\top Q_kv,
$$

最后一步用了 $Q_kQ_k^\top$ 对称。所以把 $g_k$ 替换成 $Q_kQ_k^\top g_k$，式 (5.1) 对**每一个** $v$ 的取值都不变。这不是额外的假设，而是**恒等式**：模型只依赖 $g_k$ 在子空间内的分量，子空间外的分量 $ (I-Q_kQ_k^\top)g_k$ 从不出现在任何计算中。因此可以（并且必须，见 G4）取代表元 $g_k=Q_kQ_k^\top g_k$。

**为什么这条恒等式后面是必需的**：引理 5.4 的第二部分 (5.7) 把"子空间内梯度误差 $\le\kappa_{eg}\Delta_k$"翻译成"$\Vert g_k-Q_kQ_k^\top\nabla\phi(x_k)\Vert\le\kappa_{eg}\Delta_k$"，那个翻译**只用** $g_k=Q_kQ_k^\top g_k$（下面 G4 会看到具体用法）。所以这不是装饰性假设。

**子空间信赖域**：$B_{Q_k}(x_k,\Delta_k)=\{z:z=x_k+Q_kv,\ \Vert v\Vert\le\Delta_k\}$。因为 $Q_k$ 列正交，$\Vert Q_kv\Vert^2=v^\top Q_k^\top Q_kv=\Vert v\Vert^2$，所以 $\Vert z-x_k\Vert=\Vert v\Vert$：**在子空间里量出的距离与在 $\mathbb{R}^n$ 里量出的距离相同**，半径 $\Delta_k$ 的含义与第 2 节完全一致。这是"用正交矩阵而不是任意矩阵"的全部好处。

**$L_Q$ 与它和 $L$ 的关系**：引理 5.2 里的 $L_Q$ 是映射 $x\mapsto QQ^\top\nabla\phi(x)$ 的 Lipschitz 常数，即满足

$$
\Vert QQ^\top(\nabla\phi(a)-\nabla\phi(b))\Vert\le L_Q\Vert a-b\Vert\quad\text{对一切 }a,b
$$

的最小常数。由假设 1.2，$\nabla\phi$ 是 $L$-Lipschitz 的；而正交投影不放大范数：$\Vert QQ^\top w\Vert=\Vert Q(Q^\top w)\Vert=\Vert Q^\top w\Vert\le\Vert w\Vert$（最后一个不等式因为 $\Vert Q^\top\Vert$ 的谱范数是 $1$：$Q^\top Q=I_q$ 意味着 $Q$ 把单位球映成单位球上的等距嵌入）。取 $w=\nabla\phi(a)-\nabla\phi(b)$ 得

$$
\Vert QQ^\top(\nabla\phi(a)-\nabla\phi(b))\Vert\le\Vert\nabla\phi(a)-\nabla\phi(b)\Vert\le L\Vert a-b\Vert,
$$

所以 $L_Q\le L$。**结论**：$L_Q$ 是比 $L$ 更精细的光滑常数（投影后的梯度场可能比原场更平滑），后面所有界都用 $L_Q$，而 $\hat\phi$ 自身的光滑常数至多 $L$（下面 G2 用到）。

## G2 定义 5.1 与引理 5.2：$\kappa_{ef}=\kappa_{eg}+\frac{L_Q+\kappa_{bhm}}{2}$

**定义 5.1（子空间 fully-linear 模型）**：称 $m(x+Qv)$ 是 $\phi(x+Qv)$ 在 $B_Q(x,\Delta)$ 上的 $\kappa_{ef},\kappa_{eg}$-fully-linear 模型，若

$$
\Vert\nabla{\hat m}(0)-\nabla{\hat\phi}(0)\Vert\le\kappa_{eg}\Delta \tag{5.3}
$$

$$
\vert{\hat m}(v)-{\hat\phi}(v)\vert\le\kappa_{ef}\Delta^2\quad\text{对所有 }\Vert v\Vert\le\Delta. \tag{5.4}
$$

符号：$\nabla{\hat m}(0)=Q^\top g$，$\nabla{\hat\phi}(0)=Q^\top\nabla\phi(x)$（都是 $\mathbb{R}^q$ 中的向量，范数是 $\mathbb{R}^q$ 的欧氏范数）；$\kappa_{eg},\kappa_{ef}$ 是与 $x,\Delta,v$ 无关的正常数（待求量）。

**与定义 2.1 的唯一区别，也是本节全部复杂度差异的来源**：(5.3) 比较的是**投影后**的梯度 $Q^\top g$ 与 $Q^\top\nabla\phi(x)$，而不是 $g$ 与 $\nabla\phi(x)$ 本身。误差度量从 $\mathbb{R}^n$ 换到了 $\mathbb{R}^q$，所以有限差分的维度因子从 $\sqrt n$ 变成 $\sqrt q$（G10 会看到）。但代价是：模型只约束了梯度在子空间内的分量，子空间外的 $\frac{n-q}{n}$ 分量完全没被约束——这个损失由"良对齐"条件（定义 5.3）在别处补回来。

**引理 5.2**：在假设 1.2 与 2.2 下，若 (5.3) 成立，则 $m$ 在 $B_Q(x,\Delta)$ 上是 fully-linear 的，且

$$
\kappa_{ef}=\kappa_{eg}+\frac{L_Q+\kappa_{bhm}}{2}.
$$

论文直接写"Lemma 6.6 from [11]"并省略证明。下面完整重推（结构与引理 2.9 相同，但要用 $L_Q$ 而不是 $L$，这正是论文里 $L_Q$ 这个符号出现的唯一理由）。

**第一步：模型的常数项精确等于 $\phi(x)$。** 由 (5.1)，${\hat m}(0)=m(x)=\phi(x)+g^\top Q\cdot0+\frac12\cdot0=\phi(x)$，而 ${\hat\phi}(0)=\phi(x+Q\cdot0)=\phi(x)$。所以

$$
{\hat m}(0)-{\hat\phi}(0)=0.
$$

这是"精确居中"，与我在阶段 D 里为引理 2.9 补出的关键观察完全同构：**正因为模型在中心的取值是硬约束成 $\phi(x)$ 的，函数值界里才不会出现 $\epsilon_f$ 项**。

**第二步：四段拆分。** 对任意 $\Vert v\Vert\le\Delta$，加减配平：

$$
{\hat m}(v)-{\hat\phi}(v)=\underbrace{[{\hat m}(v)-{\hat m}(0)-\nabla{\hat m}(0)^\top v]}_{T_1}+{\hat m}(0)-{\hat\phi}(0)+\underbrace{[\nabla{\hat m}(0)-\nabla{\hat\phi}(0)]^\top v}_{T_3}+\underbrace{[{\hat\phi}(v)-{\hat\phi}(0)-\nabla{\hat\phi}(0)^\top v]}_{T_4}.
$$

（这里 $\underbrace$ 只用于本稿标注，写入定稿时改为文字说明。）

**第三步：$T_1$。** ${\hat m}(v)-{\hat m}(0)=g^\top Qv+\frac12 v^\top Q^\top HQv$，而 $\nabla{\hat m}(0)^\top v=(Q^\top g)^\top v=g^\top Qv$，两者相减只剩二次项：

$$
T_1=\frac{1}{2}v^\top Q^\top HQv=\frac{1}{2}(Qv)^\top H(Qv).
$$

用二次型的基本界 $\vert y^\top Hy\vert\le\Vert H\Vert\Vert y\Vert^2$（依据：$H$ 对称时 $\vert y^\top Hy\vert\le\max_i\vert\lambda_i(H)\vert\Vert y\Vert^2=\Vert H\Vert\Vert y\Vert^2$，$\Vert\cdot\Vert$ 是谱范数），配合假设 2.2 的 $\Vert H\Vert\le\kappa_{bhm}$ 和 $\Vert Qv\Vert=\Vert v\Vert\le\Delta$：

$$
\vert T_1\vert\le\frac{\kappa_{bhm}}{2}\Delta^2.
$$

**第四步：$T_3$。** 直接用 Cauchy–Schwarz 与 (5.3)：

$$
\vert T_3\vert\le\Vert\nabla{\hat m}(0)-\nabla{\hat\phi}(0)\Vert\cdot\Vert v\Vert\le\kappa_{eg}\Delta\cdot\Delta=\kappa_{eg}\Delta^2.
$$

**第五步：$T_4$ 要用 $L_Q$ 而不是 $L$——这是本节唯一需要新技巧的一步。** 用积分形式写出余项：

$$
T_4=\int_0^1\big[\nabla{\hat\phi}(tv)-\nabla{\hat\phi}(0)\big]^\top v\,dt.
$$

（依据：一元微积分基本定理，把 $t\mapsto{\hat\phi}(tv)$ 在 $[0,1]$ 上积分其导数 $\nabla{\hat\phi}(tv)^\top v$。）对被积函数作恒等改写，$\nabla{\hat\phi}(tv)=Q^\top\nabla\phi(x+tQv)$，所以

$$
\big[\nabla{\hat\phi}(tv)-\nabla{\hat\phi}(0)\big]^\top v=\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)^\top Qv.
$$

现在插入 $P_Q=QQ^\top$。把左边的向量分解成"子空间内 + 子空间外"：

$$
\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)^\top Qv=\big[(I-P_Q)\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)\big]^\top Qv+\big[P_Q\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)\big]^\top Qv.
$$

**第一个中括号为零**：$(I-P_Q)$ 的值域是子空间的正交补，而 $Qv$ 在子空间内，正交向量内积为 $0$。于是

$$
\big[\nabla{\hat\phi}(tv)-\nabla{\hat\phi}(0)\big]^\top v=\big[QQ^\top\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)\big]^\top Qv.
$$

**但注意 $\nabla\phi(x)$ 没有被投影**，而 $L_Q$ 的定义里投影的是**同一个** $x$ 处的梯度。补一项：把 $\nabla\phi(x)$ 换成 $P_Q\nabla\phi(x)$ 是否改变结果？

$$
\big[(I-P_Q)\nabla\phi(x)\big]^\top Qv=0
$$

同样因为正交。所以 $\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)^\top Qv=\big(\nabla\phi(x+tQv)-P_Q\nabla\phi(x+tQv)+P_Q\nabla\phi(x+tQv)-P_Q\nabla\phi(x)\big)^\top Qv$，其中第一项又是 $0$，只剩

$$
=\big[QQ^\top\big(\nabla\phi(x+tQv)-\nabla\phi(x)\big)\big]^\top Qv.
$$

（这一步等价于说：沿子空间方向移动时，只有梯度的子空间分量参与内积，因此可以用"投影梯度场"的光滑常数。）现在用 $L_Q$ 的定义（取 $a=x+tQv$，$b=x$）与 Cauchy–Schwarz：

$$
\big|\big[\nabla{\hat\phi}(tv)-\nabla{\hat\phi}(0)\big]^\top v\big|\le L_Q\Vert tQv\Vert\cdot\Vert Qv\Vert=tL_Q\Vert v\Vert^2.
$$

对 $t$ 从 $0$ 到 $1$ 积分：$\vert T_4\vert\le\frac{L_Q}{2}\Vert v\Vert^2\le\frac{L_Q}{2}\Delta^2$。

**第六步：合并。** 四段相加：

$$
\vert{\hat m}(v)-{\hat\phi}(v)\vert\le\Big(\kappa_{eg}+\frac{L_Q+\kappa_{bhm}}{2}\Big)\Delta^2,
$$

即引理 5.2 的 $\kappa_{ef}$。$\blacksquare$

**与引理 2.9 的对照（审计）**：结构完全平行，只是 $L$ 换成 $L_Q\le L$，所以 $\kappa_{ef}$ 只会**变小或相等**：$\kappa_{ef}^{sub}\le\kappa_{ef}^{full}$。但注意引理 2.9 里我补出的"额外 $\max\{\eta_2,\kappa_{ef}\}$ 项"在这里**没有**出现——因为这里的 $\kappa_{ef}$ 只服务于一处（引理 5.5 的 $\rho_k$ 下界），而那里的误差被 $+2\epsilon_f$ 吸收了。这个差别正是 $\tilde C_1$ 与 $C_1$ 形式不同的根源（G5）。

## G3 算法 3 逐行：$+2\epsilon_f$ 与 $\Delta_{min}$ 各自解决什么

**输入**（第 733 行）：含噪零阶 oracle $\vert f(x)-\phi(x)\vert\le\epsilon_f$（$\epsilon_f$ 是**已知上界**——这是本节最重的实用假设，见 G12）；最小半径 $\Delta_{min}$（**新增超参数**）；初值 $x_0$、$\Delta_0\ge\Delta_{min}$；初始正交矩阵 $Q_0\in\mathbb{R}^{n\times q}$；$\eta_1\in(0,1)$（接受比阈值）、$\eta_2>0$（模型梯度阈值）、$\gamma\in(0,1)$（收缩因子）。后三个与算法 1 同名同义。

**第 1 行**：对当前 $Q_k$ 按 (5.1) 建模。与算法 1 的差别只在模型定义域是子空间。

**第 2 行**：$s_k=Q_kv_k$，其中 $v_k\approx\arg\min\{{\hat m}_k(v):\Vert v\Vert\le\Delta_k\}$。注意**优化只在 $q$ 维里做**：这是子空间方法真正的计算收益所在（$q\times q$ 的信赖域子问题，而不是 $n\times n$）。$v_k$ 满足假设 2.2 的 (2.2) 充分 Cauchy 下降，但其中的球是 $\mathbb{R}^q$ 里的球。

**第 3 行（核心改动 1）**：

$$
\rho_k=\frac{f(x_k)-f(x_k+s_k)+2\epsilon_f}{m_k(x_k)-m_k(x_k+s_k)}.
$$

分母 $m_k(x_k)-m_k(x_k+s_k)={\hat m}_k(0)-{\hat m}_k(v_k)>0$，只要 $g_k\ne0$：由 (2.2)，它 $\ge\frac{\kappa_{fcd}}{2}\Vert g_k\Vert\min\{\Vert g_k\Vert/\kappa_{bhm},\Delta_k\}>0$。而 $\Vert g_k\Vert=0$ 时第 $\Vert g_k\Vert\ge\eta_2\Delta_k$ 的门槛直接判失败，不会走到除法。所以分母恒正，下面的不等式方向都成立。

**这条 $\rho_k$ 的"乐观"性质**（论文未证明，我补上）：由 $\vert f-\phi\vert\le\epsilon_f$，

$$
f(x_k)-f(x_k+s_k)+2\epsilon_f\ge\big(\phi(x_k)-\epsilon_f\big)-\big(\phi(x_k+s_k)+\epsilon_f\big)+2\epsilon_f=\phi(x_k)-\phi(x_k+s_k).
$$

即**分子是真实下降量 $\phi(x_k)-\phi(x_k+s_k)$ 的一个上界估计的相反数**（更准确说：分子 $\ge$ 真实下降量）。除以同一个正分母得

$$
\rho_k\ge\frac{\phi(x_k)-\phi(x_k+s_k)}{m_k(x_k)-m_k(x_k+s_k)}.
$$

含义：$\rho_k\ge\eta_1$ 比"真实下降/模型预测 $\ge\eta_1$"**更容易满足**，所以放宽了接受判据。为什么必须放宽？若用标准 $\rho_k^{std}=\frac{f(x_k)-f(x_k+s_k)}{m_k(x_k)-m_k(x_k+s_k)}$，分子可能比真实下降量小 $2\epsilon_f$；当 $\Delta_k$ 小时真实下降量本身是 $\mathcal{O}(\Delta_k^2)$，$\frac{2\epsilon_f}{\mathcal{O}(\Delta_k^2)}\to\infty$ 的相对误差会让 $\rho^{std}$ 系统性偏小，于是半径一路缩到底（引理 2.3 的"小半径必成功"论证失效）。加 $2\epsilon_f$ 就是把这层系统性偏差在**分子上**补回来。

**代价**：$\rho_k\ge\eta_1$ 不再蕴含 $\phi$ 下降。引理 5.6 会看到下降量右边多出 $-4\epsilon_f$，即**目标函数可能在"成功"步上升**。这是第 5 节分析相对第 2 节的第二个结构性变化（论文第 882 行明说）。

**第 4 行（核心改动 2）**：三分支更新，与算法 1 同构，只有第三支不同：

- 若 $\rho_k\ge\eta_1$ 且 $\Vert g_k\Vert\ge\eta_2\Delta_k$：$(x_{k+1},\Delta_{k+1})=(x_k+s_k,\gamma^{-1}\Delta_k)$（成功，扩张）；
- 否则，若模型在 $B_{Q_k}(x_k,\Delta_k)$ 上不是 fully-linear：$(x_{k+1},\Delta_{k+1})=(x_k,\Delta_k)$（保持不动，即"模型改进步"）；
- 否则：$(x_{k+1},\Delta_{k+1})=(x_k,\max\{\gamma\Delta_k,\Delta_{min}\})$（失败，收缩但**不越过地板**）。

$\Delta_{min}$ 只出现在第三支。它的存在使 $\Delta_k\ge\Delta_{min}$ 对一切 $k$ 成立（归纳：$\Delta_0\ge\Delta_{min}$，扩张支增大、保持支不变、收缩支取 $\max\{\cdot,\Delta_{min}\}\ge\Delta_{min}$）。**这就是 G10 里 $\kappa_{eg}=\frac{\sqrt qL}{2}+\frac{2\sqrt q\epsilon_f}{\Delta_{min}^2}$ 有意义的唯一原因**：有限差分的噪声项 $\frac{\epsilon_f}{\delta}$ 需要一个 $\delta$ 的下界才能折成 $\kappa_{eg}\Delta_k$ 的形式。

**第 5 行**：$Q_{k+1}\leftarrow$ 随机（若成功，或若模型已 fully-linear）；否则 $Q_{k+1}=Q_k$。含义：**只要不是"模型改进步"，就换子空间**。这一点决定了后面 $t$ 指标的语义（G7）。

**第 6 行**："Perform some model improvement steps"——与算法 1 同样含糊，第 5 节自身不指定怎么改进；§5.2 用有限差分（无改进步），§5.3 的算法 4 用几何校正。

**与算法 1 的差异汇总表**：

| 项 | 算法 1 | 算法 3 |
| --- | --- | --- |
| 模型定义域 | 全空间 $\mathbb{R}^n$ | 子空间 $B_{Q_k}(x_k,\Delta_k)$ |
| 子问题规模 | $n$ 维 | $q$ 维 |
| $\rho_k$ 分子 | $f(x_k)-f(x_k+s_k)$ | 加 $2\epsilon_f$ |
| 收缩 | $\gamma\Delta_k$，无下界 | $\max\{\gamma\Delta_k,\Delta_{min}\}$ |
| 半径下界来源 | 自动：$\Delta_k\ge\gamma C_1\epsilon$ | 人工：$\Delta_{min}$，且需 $\Delta_{min}\le\gamma\hat C_1\epsilon$ |
| $Q_k$ | 无 | 每步（除模型改进步）随机重抽 |

## G4 定义 5.3 与引理 5.4：良对齐 ⟺ 勾股恒等式

**定义 5.3（良对齐子空间）**：若

$$
\Vert QQ^\top\nabla\phi(x)-\nabla\phi(x)\Vert\le\kappa_g\Vert\nabla\phi(x)\Vert \tag{5.5}
$$

对某个 $\kappa_g\in[0,1)$ 成立，则称 $Q$ 诱导的子空间与 $\nabla\phi(x)$ 是 $\kappa_g$-well aligned。符号：$\kappa_g$ 是**对齐损失系数**（超参数/中间结果，由 $Q$ 的随机性与 $q/n$ 的比值决定），$\kappa_g$ 越接近 $0$ 表示梯度几乎全在子空间内，越接近 $1$ 表示梯度几乎与子空间正交（此时该子空间上的模型对下降毫无用处）。$\kappa_g<1$ 是**硬性要求**：$\kappa_g=1$ 意味着允许投影完全丢掉梯度，(5.5) 就退化成恒真式。

**引理 5.4 第一部分**：第 $k$ 步上，$Q_k$ 与 $\nabla\phi(x_k)$ 是 $\kappa_g$-well aligned **当且仅当**

$$
\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert^2\ge(1-\kappa_g^2)\Vert\nabla\phi(x_k)\Vert^2. \tag{5.6}
$$

证明只用正交投影的勾股定理。记 $u=\nabla\phi(x_k)$，$P=P_{Q_k}=Q_kQ_k^\top$。分解 $u=Pu+(I-P)u$。**两正交性**：$(Pu)^\top((I-P)u)=u^\top P(I-P)u=u^\top(P-P^2)u=0$，用到 $P^2=P$。于是

$$
\Vert u\Vert^2=\Vert Pu\Vert^2+\Vert(I-P)u\Vert^2.
$$

又 $(I-P)u=Pu-u$，所以 (5.5) 就是 $\Vert(I-P)u\Vert\le\kappa_g\Vert u\Vert$，两边平方并用上式：

$$
\Vert u\Vert^2-\Vert Pu\Vert^2\le\kappa_g^2\Vert u\Vert^2\iff\Vert Pu\Vert^2\ge(1-\kappa_g^2)\Vert u\Vert^2.
$$

$\blacksquare$ 两个方向都成立，所以是"当且仅当"。

**推论（$q$ 与对齐的物理关系）**：若 $Q$ 在 $q$ 维子空间的 Grassmann 流形上均匀分布，则对固定 $u$，$\mathbb{E}\Vert Pu\Vert^2=\frac{q}{n}\Vert u\Vert^2$（由对称性，$n$ 个坐标方向地位相同，而 $\operatorname{tr}P=q$）。所以"平均而言" $1-\kappa_g^2=\frac{q}{n}$。文献 [11, Lem 6.7] 给出的可用版本是 $1-\kappa_g^2=\frac{q}{10n}$，即只取平均值的十分之一，换取一个与 $n,q$ 无关的成功概率 $\theta\ge\frac{243}{443}$（见 G7）。**这个 $10$ 就是"用对齐质量换概率"的显式价格。**

**引理 5.4 第二部分**：若 $m(x_k+s)$ 是 $\phi(x_k+s)$ 在 $B_{Q_k}(x_k,\Delta_k)$ 上的 $\kappa_{ef},\kappa_{eg}$-fully-linear 模型，则

$$
\Vert g_k-Q_kQ_k^\top\nabla\phi(x_k)\Vert\le\kappa_{eg}\Delta_k. \tag{5.7}
$$

证明：由定义 5.1 的 (5.3)，

$$
\Vert Q_k^\top g_k-Q_k^\top\nabla\phi(x_k)\Vert=\Vert\nabla{\hat m}_k(0)-\nabla{\hat\phi}(0)\Vert\le\kappa_{eg}\Delta_k.
$$

（这里用了 $\nabla{\hat m}_k(0)=Q_k^\top g_k$：由 (5.1)，${\hat m}_k(v)=\phi(x_k)+g_k^\top Q_kv+\frac12 v^\top Q_k^\top H_kQ_kv$，对 $v$ 求梯度得 $Q_k^\top g_k+Q_k^\top H_kQ_kv$，在 $v=0$ 取即 $Q_k^\top g_k$。）左边 $=\Vert Q_k^\top(g_k-\nabla\phi(x_k))\Vert$。现在用 $g_k=Q_kQ_k^\top g_k$：

$$
g_k-Q_kQ_k^\top\nabla\phi(x_k)=Q_kQ_k^\top g_k-Q_kQ_k^\top\nabla\phi(x_k)=Q_kQ_k^\top\big(g_k-\nabla\phi(x_k)\big).
$$

而对任意 $w\in\mathbb{R}^n$，$\Vert QQ^\top w\Vert=\Vert Q(Q^\top w)\Vert=\Vert Q^\top w\Vert$（第一个等号是结合律，第二个等号是 $Q$ 列正交给出的等距性 $\Vert Qz\Vert=\Vert z\Vert$，取 $z=Q^\top w$）。取 $w=g_k-\nabla\phi(x_k)$ 得

$$
\Vert g_k-Q_kQ_k^\top\nabla\phi(x_k)\Vert=\Vert Q_k^\top(g_k-\nabla\phi(x_k))\Vert\le\kappa_{eg}\Delta_k.
$$

$\blacksquare$ **注意**：若没有 $g_k=Q_kQ_k^\top g_k$，等距那一步就不成立，(5.7) 只能写成 $\Vert Q_k^\top(\cdot)\Vert$ 的形式，后面 G5 的三角不等式链就接不上。**这就是 G1 那条"不失一般性"的真正用途。**

## G5 引理 5.5：$\tilde C_1$ 的两条来源，以及它为什么比 $C_1$ 大

**命题**：在假设 1.2、2.2 下，若 $Q_k$ 与 $\nabla\phi(x_k)$ 是 $\kappa_g$-well aligned、$m_k$ 在 $B_{Q_k}(x_k,\Delta_k)$ 上 fully-linear，且

$$
\Delta_k\le\sqrt{1-\kappa_g^2}\,\tilde C_1\Vert\nabla\phi(x_k)\Vert, \tag{5.9}
$$

$$
\tilde C_1=\Big(\max\Big\{\eta_2,\kappa_{bhm},\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}\Big\}+\kappa_{eg}\Big)^{-1}, \tag{5.8}
$$

则 $\rho_k\ge\eta_1$、$\Vert g_k\Vert\ge\eta_2\Delta_k$，从而第 $k$ 步成功。

符号：$\tilde C_1$ 是"成功所需的半径上界系数"（中间结果），与定理 2.8 的 $C_1$ 同族但更小（下面比较）。

**第一步：把 (5.9) 右端换成子空间内的梯度。** 由 (5.6)，$\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert\ge\sqrt{1-\kappa_g^2}\Vert\nabla\phi(x_k)\Vert$，代入 (5.9) 右端：

$$
\Delta_k\le\sqrt{1-\kappa_g^2}\tilde C_1\Vert\nabla\phi(x_k)\Vert\le\tilde C_1\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert.
$$

即 $\tilde C_1^{-1}\Delta_k\le\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert$。展开 $\tilde C_1^{-1}$：

$$
\Big(\max\Big\{\eta_2,\kappa_{bhm},\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}\Big\}+\kappa_{eg}\Big)\Delta_k\le\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert. \tag{$\ast$}
$$

**第二步：用 (5.7) 把右边换成 $\Vert g_k\Vert$。** 三角不等式：

$$
\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert\le\Vert g_k\Vert+\Vert Q_kQ_k^\top\nabla\phi(x_k)-g_k\Vert\le\Vert g_k\Vert+\kappa_{eg}\Delta_k.
$$

代回 $(\ast)$，两边消去 $\kappa_{eg}\Delta_k$：

$$
\max\Big\{\eta_2,\kappa_{bhm},\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}\Big\}\Delta_k\le\Vert g_k\Vert. \tag{$\ast\ast$}
$$

论文第 815–822 行把这两步合并成一句，并只保留了 $\max\{\kappa_{bhm},\eta_2\}\Delta_k\le\Vert g_k\Vert$。**$(\ast\ast)$ 里剩下的第三个分量在后面还要单独用**（第四步），这就是 $\tilde C_1$ 的 $\max$ 里为什么要有 $\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$。

**第三步：门槛 $\Vert g_k\Vert\ge\eta_2\Delta_k$ 到手**，由 $(\ast\ast)$ 的 $\eta_2$ 分量直接读出。同时由假设 2.2 的 (2.2)：

$$
m_k(x_k)-m_k(x_k+s_k)\ge\frac{\kappa_{fcd}}{2}\Vert g_k\Vert\min\Big\{\frac{\Vert g_k\Vert}{\kappa_{bhm}},\Delta_k\Big\}.
$$

由 $(\ast\ast)$ 的 $\kappa_{bhm}$ 分量，$\kappa_{bhm}\Delta_k\le\Vert g_k\Vert$，即 $\frac{\Vert g_k\Vert}{\kappa_{bhm}}\ge\Delta_k$，$\min$ 取 $\Delta_k$：

$$
m_k(x_k)-m_k(x_k+s_k)\ge\frac{\kappa_{fcd}}{2}\Vert g_k\Vert\Delta_k. \tag{$\dagger$}
$$

**第四步：$\rho_k$ 的下界，逐步展开。** 论文第 827 行的长链有四处跳步，逐条补：

(a) 分子恒等改写（加减 $\phi$ 与 $m_k$）：

$$
f(x_k)-f(x_k+s_k)+2\epsilon_f=\big[m_k(x_k)-m_k(x_k+s_k)\big]+\big[f(x_k)-m_k(x_k)\big]-\big[f(x_k+s_k)-m_k(x_k+s_k)\big]+2\epsilon_f.
$$

(b) 把 $f$ 换成 $\phi$。论文第 827 行第一个不等号写成 $\ge\frac{m_k(x_k)-m_k(x_k+s_k)+(\phi(x_k)-m_k(x_k))-(\phi(x_k+s_k)-m_k(x_k+s_k))}{m_k(x_k)-m_k(x_k+s_k)}$。核对：$f(x)-m_k(x)=\big(f(x)-\phi(x)\big)+\big(\phi(x)-m_k(x)\big)$，而 $\vert f-\phi\vert\le\epsilon_f$ 给出 $f(x)-m_k(x)\ge\phi(x)-m_k(x)-\epsilon_f$；对 $x_k+s_k$ 同理给出 $-\big(f(x_k+s_k)-m_k(x_k+s_k)\big)\ge-\big(\phi(x_k+s_k)-m_k(x_k+s_k)\big)-\epsilon_f$。两个 $-\epsilon_f$ 相加正好抵掉分子的 $+2\epsilon_f$。**这就是 $+2\epsilon_f$ 的精确用途**：它不是随手加的余量，而是**恰好**把两处 oracle 误差全部抵消，使 $\rho_k$ 的下界变成一个**只含 $\phi$ 与 $m_k$** 的量。

(c) 用 fully-linear 的函数值界 (5.4)。注意 $s_k=Q_kv_k$ 且 $\Vert v_k\Vert\le\Delta_k$，所以 $x_k+s_k\in B_{Q_k}(x_k,\Delta_k)$，(5.4) 给出

$$
\vert\phi(x_k)-m_k(x_k)\vert\le\kappa_{ef}\Delta_k^2,\qquad\vert\phi(x_k+s_k)-m_k(x_k+s_k)\vert\le\kappa_{ef}\Delta_k^2.
$$

于是分子 $\ge$ 模型下降 $-2\kappa_{ef}\Delta_k^2$。论文第 827 行写成 $1-\frac{2\kappa_{ef}\Delta_k^2}{m_k(x_k)-m_k(x_k+s_k)}$ 之前先写了 $1-\frac{\kappa_{ef}\Delta_k^2}{\cdots}$，**少了一个因子 2**：它的第一个分式分子里 $(\phi(x_k)-m_k(x_k))-(\phi(x_k+s_k)-m_k(x_k+s_k))$ 只被 (5.4) 约束为 $\ge-2\kappa_{ef}\Delta_k^2$，但论文第二行直接写 $\ge1-\frac{\kappa_{ef}\Delta_k^2}{m_k(x_k)-m_k(x_k+s_k)}$。核对论文原文第 827 行：它写的是"$\ge1-\frac{2\kappa_{ef}\Delta_k^2}{m_k(x_k)-m_k(x_k+s_k)}$"（Markdown 转换中该 $2$ 与后续 $\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}(\cdots)}$ 一致，见下一行确实带 $2$）。所以**没有丢因子**，是我在核对时把两行看串了；记录在此以免定稿误改。

(d) 用 $(\dagger)$ 与 $(\ast\ast)$ 把分母换成梯度：

$$
1-\frac{2\kappa_{ef}\Delta_k^2}{\frac{\kappa_{fcd}}{2}\Vert g_k\Vert\Delta_k}=1-\frac{4\kappa_{ef}\Delta_k}{2\kappa_{fcd}\Vert g_k\Vert}\cdot\frac{2}{2}=1-\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}\Vert g_k\Vert}.
$$

（即 $\frac{2\kappa_{ef}\Delta_k^2}{\kappa_{fcd}\Vert g_k\Vert\Delta_k/2}=\frac{4\kappa_{ef}\Delta_k}{\kappa_{fcd}\Vert g_k\Vert}$——**这里出现因子 4 而不是论文写的 2**。再核对论文第 827 行：它写 $\ge1-\frac{\kappa_{ef}\Delta_k^2}{m_k(x_k)-m_k(x_k+s_k)}\ge1-\frac{\kappa_{ef}\Delta_k^2}{\kappa_{fcd}\Vert g_k\Vert\Delta_k/2}\ge1-\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}(\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert-\kappa_{eg}\Delta_k)}$。第一处 $\kappa_{ef}\Delta_k^2$ 无因子 $2$ 说明**论文的 (b) 步只用了单侧误差 $\phi(x_k)-m_k(x_k)\ge-\kappa_{ef}\Delta_k^2$ 与 $\phi(x_k+s_k)-m_k(x_k+s_k)\le+\kappa_{ef}\Delta_k^2$，两者相加得 $\ge-2\kappa_{ef}\Delta_k^2$**——所以正确值应带 $2$。但紧接着 $\frac{\kappa_{ef}\Delta_k^2}{\kappa_{fcd}\Vert g_k\Vert\Delta_k/2}=\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}\Vert g_k\Vert}$ 已经带 $2$ 了，若分子是 $2\kappa_{ef}\Delta_k^2$ 则应为 $\frac{4\kappa_{ef}\Delta_k}{\kappa_{fcd}\Vert g_k\Vert}$。**结论：论文在 (c)→(d) 之间确实丢了一个因子 2。**）

**这个因子 2 的后果**：最后一步要求 $\ge\eta_1$，即需要 $\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}\Vert Q_kQ_k^\top\nabla\phi\Vert_{\rm eff}}\le1-\eta_1$；若正确值是它的两倍，则 $\tilde C_1$ 的定义里 $\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$ 应改成 $\frac{4\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$。**只改常数，不改任何阶**：$\tilde C_1^{-1}$ 仍是 $\Theta(\max\{\eta_2,\kappa_{bhm},\kappa_{ef}\}+\kappa_{eg})$，$\hat C_1^{-1}$ 仍是 $\Theta(\sqrt n(L+\frac{\epsilon_f}{\Delta_{min}^2}))$，定理 5.13 与推论 5.14 的 $\mathcal{O}(n\epsilon^{-2})$、$\mathcal{O}(nq\epsilon^{-2})$、$\mathcal{O}(nq\log q\,\epsilon^{-2})$ 全部不变。所以这是一处**非致命**的常数错误。

(e) 把 $\Vert g_k\Vert$ 换回 $\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert$（论文倒数第二个分式）：由 (5.7) 的反向三角不等式，$\Vert g_k\Vert\ge\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert-\kappa_{eg}\Delta_k$。分母变小 ⟹ 分数变大 ⟹ $1-\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}\Vert g_k\Vert}\ge1-\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}(\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert-\kappa_{eg}\Delta_k)}$ ✓ 方向正确。

(f) 最后一步 $\ge\eta_1$ 的依据：由 $(\ast)$ 单独取第三个分量，

$$
\Big(\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}+\kappa_{eg}\Big)\Delta_k\le\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert\iff\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}\Delta_k\le\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert-\kappa_{eg}\Delta_k,
$$

即论文第 830 行末尾"because ... follows from (5.9)"那句被省略的推导。把它代入 (e) 的分母：

$$
\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}\big(\Vert Q_kQ_k^\top\nabla\phi(x_k)\Vert-\kappa_{eg}\Delta_k\big)}\le\frac{2\kappa_{ef}\Delta_k}{\kappa_{fcd}\cdot\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}\Delta_k}=1-\eta_1,
$$

所以 $1-(\cdot)\ge\eta_1$ ✓。$\blacksquare$

**$\tilde C_1$ 的构造性解释（论文未写，我反推）**：$\tilde C_1^{-1}$ 不是拍脑袋写的，它是**三条独立需求的分母的极大值**：

- 需求 1（门槛）：$\eta_2\Delta_k\le\Vert g_k\Vert$，代价 $\eta_2$；
- 需求 2（$\min$ 取 $\Delta_k$）：$\kappa_{bhm}\Delta_k\le\Vert g_k\Vert$，代价 $\kappa_{bhm}$；
- 需求 3（$\rho_k\ge\eta_1$）：代价 $\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$（按论文）或 $\frac{4\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$（按我修正后的因子）；

三条都要 $\Delta_k\le\frac{\Vert Q_kQ_k^\top\nabla\phi\Vert}{\text{代价}+\kappa_{eg}}$（$\kappa_{eg}$ 是三条共用的公共项，来自 (5.7) 的换元），取最严的即分母取 $\max$ 再加 $\kappa_{eg}$。这正是 $\tilde C_1^{-1}$ 的形式。**同一个反推用在定理 2.8 的 $C_1^{-1}$ 上，可以解释它为什么多一项 $\max\{\eta_2,\kappa_{ef}\}$**：那里 $\rho_k$ 没有 $+2\epsilon_f$，分子上多一处未被抵消的误差，需求 3 的代价变成 $\frac{2\kappa_{ef}+\max\{\eta_2,\kappa_{ef}\}}{(1-\eta_1)\kappa_{fcd}}$。

**大小关系**：$\tilde C_1^{-1}\le C_1^{-1}$（$\max$ 的第三项更小），所以

$$
\tilde C_1\ge C_1.
$$

含义：算法 3 的"半径足够小 ⟹ 必成功"门槛**比算法 1 更宽松**（允许更大的 $\Delta_k$ 就保证成功）。这是 $+2\epsilon_f$ 换来的好处，与 $\Delta_{min}$ 的引入是同一件事的两面。

## G6 引理 5.6：成功步上的净下降 $-4\epsilon_f$

**命题**：在算法 3 中，若 $\rho_k\ge\eta_1$ 且 $\Vert g_k\Vert\ge\eta_2\Delta_k$，则

$$
\phi(x_k)-\phi(x_{k+1})\ge C_2\Delta_k^2-4\epsilon_f,\qquad C_2=\frac{\eta_1\eta_2\kappa_{fcd}}{2}\min\Big\{\frac{\eta_2}{\kappa_{bhm}},1\Big\}.
$$

（$C_2$ 与引理 2.4 的 $C_2$ **完全同一个表达式**，论文在此重新写了一遍而没有引用，容易让读者以为是新的常数。它没有 $\kappa_{ef},\kappa_{eg}$ 依赖，所以子空间化对 $C_2$ 无影响。）

**第一步：$\rho_k\ge\eta_1$ 的可 rearrange 形式。** 由第 4 行分支逻辑，两个条件同时成立意味着走第一支，故 $x_{k+1}=x_k+s_k$，$\phi(x_{k+1})=\phi(x_k+s_k)$。由 $\rho_k\ge\eta_1$ 与分母为正：

$$
\eta_1\big(m_k(x_k)-m_k(x_k+s_k)\big)\le f(x_k)-f(x_k+s_k)+2\epsilon_f.
$$

右边用 $\vert f-\phi\vert\le\epsilon_f$ 上界：$f(x_k)-f(x_k+s_k)+2\epsilon_f\le\big(\phi(x_k)+\epsilon_f\big)-\big(\phi(x_k+s_k)-\epsilon_f\big)+2\epsilon_f=\phi(x_k)-\phi(x_k+s_k)+4\epsilon_f$。**这就是 $4\epsilon_f$ 的全部来源：$2\epsilon_f$（$\rho$ 里人为加的）$+\epsilon_f+\epsilon_f$（两端 oracle 误差）**。整理：

$$
\phi(x_k)-\phi(x_k+s_k)\ge\eta_1\big(m_k(x_k)-m_k(x_k+s_k)\big)-4\epsilon_f.
$$

**第二步：模型下降换成 $\Delta_k^2$。** 由 (2.2) 与 $\Vert g_k\Vert\ge\eta_2\Delta_k$：

$$
\eta_1\big(m_k(x_k)-m_k(x_k+s_k)\big)\ge\frac{\eta_1\kappa_{fcd}}{2}\Vert g_k\Vert\min\Big\{\frac{\Vert g_k\Vert}{\kappa_{bhm}},\Delta_k\Big\}\ge\frac{\eta_1\kappa_{fcd}}{2}\eta_2\Delta_k\min\Big\{\frac{\eta_2\Delta_k}{\kappa_{bhm}},\Delta_k\Big\}.
$$

（第一个不等号用了 $\Vert g_k\Vert\ge\eta_2\Delta_k$ 两次：一次提出前面的 $\eta_2\Delta_k$，一次放进 $\min$ 里——$\min$ 是单调的，所以可以这样替换。）提取 $\Delta_k^2$：

$$
=\frac{\eta_1\eta_2\kappa_{fcd}}{2}\Delta_k^2\min\Big\{\frac{\eta_2}{\kappa_{bhm}},1\Big\}=C_2\Delta_k^2.
$$

$\blacksquare$（与引理 2.4 的推导逐字相同，我在此重做一遍是因为它的 $\min$ 处理是 $C_2$ 对 $\eta_2$ 的依赖关系的关键，G12 的 $\eta_2$ 优化分析要用。）

**注意符号**：$C_2\Delta_k^2-4\epsilon_f$ 在 $\Delta_k$ 很小时是**负**的，即"成功步可以让 $\phi$ 上升，最多上升 $4\epsilon_f$"。这正是 G7 里引理 5.8 必须把"大步成功"与"小步成功"分开计数的原因。

## G7 §5.1 的随机过程：$t$ 的语义、滤链、以及 $\Delta_{min}\le\gamma\hat C_1\epsilon$ 的确切作用

**$t$ 指标的定义（第 858 行，必须先读对）**：模型改进步（分支 b）不改变 $Q_k$、不改变 $\Delta_k$、不改变 $x_k$。定义 $Q_t$ 为第 $t$ 个**随机矩阵**，$x_t,\Delta_t$ 是与之对应的迭代点与半径。所以 $t$ 数的正是第 5 行"重抽 $Q$"的那些步，即分支 (a) 与分支 (c)；分支 (b) 被完全排除在 $t$ 之外。论文的说法是"$t$ 数的是跟随在一次成功或一次失败之后的迭代"。

**由此得到两条精确的半径动力学**（后面所有计数引理的地基）：

$$
A_t=1\Rightarrow\Delta_{t+1}=\gamma^{-1}\Delta_t,\qquad A_t=0\Rightarrow\Delta_{t+1}=\max\{\gamma\Delta_t,\Delta_{min}\}\le\gamma\Delta_t. \tag{L}
$$

（$A_t=0$ 时用的是分支 (c)，因为分支 (b) 不产生 $t$ 的增量。）

**三个指示随机变量**：

$$
I_t=\mathbb{1}\{Q_t\text{ 与 }\nabla\phi(x_t)\text{ 是 }\kappa_g\text{-well aligned}\},
$$

$$
A_t=\mathbb{1}\{Q_t\text{ 导致成功迭代，即 }\Delta_{t+1}=\gamma^{-1}\Delta_t\},
$$

$$
B_t=\mathbb{1}\{\Delta_t>\hat C_1\Vert\nabla\phi(x_t)\Vert\},\qquad \hat C_1=\sqrt{1-\kappa_g^2}\,\tilde C_1.
$$

符号：$\mathbb{1}\{\cdot\}$ 是指示函数（条件真取 1，假取 0）；$B_t=1$ 读作"半径相对于当前梯度太大"，即引理 5.5 的充分条件 (5.9) **尚未满足**（$\hat C_1$ 正是 (5.9) 右端的系数）。$I_t,A_t,B_t$ 都是 $0$-$1$ 变量，所以 $1-I_t$ 等就是补事件的指示。

**滤链与可测性（第 872 行）**：$\mathcal{F}_{t-1}=\sigma(Q_0,\dots,Q_{t-1})$。断言 $x_t,\Delta_t$ 是 $\mathcal{F}_{t-1}$-可测的——成立，因为 $x_t,\Delta_t$ 完全由此前抽到的矩阵和此前的 oracle 回答决定，而**噪声 oracle 是确定性的**（$\phi$ 的固定扰动，不是随机扰动），所以给定 $x$ 就有唯一的 $f(x)$，不引入新随机源。这一点很重要：**若噪声是随机的，$\mathcal{F}_{t-1}$ 必须同时包含 oracle 的回答，而 [10] 的分析是否仍成立需要重新检查。**论文没提这个前提，我在此点明。$m_t,s_t,\rho_t$ 是 $\mathcal{F}_t$-可测的（它们依赖 $Q_t$）。$T_\epsilon=\min\{t:\Vert\nabla\phi(x_t)\Vert\le\epsilon\}$ 关于 $\{\mathcal{F}_{t-1}\}$ 是停时：$\{T_\epsilon\le t-1\}$ 只涉及 $x_0,\dots,x_t$，都在 $\mathcal{F}_{t-1}$ 里。

**假设 5.7**：存在 $\theta\in(\frac12,1]$ 使 $\mathbb{P}\{I_t=1\mid\mathcal{F}_{t-1}\}\ge\theta$。

为什么这条假设**不是**显然的、也不是免费的：$I_t$ 同时依赖 $Q_t$（新鲜随机）和 $x_t$（$\mathcal{F}_{t-1}$-可测）。给定 $\mathcal{F}_{t-1}$，$x_t$ 已固定，而 $Q_t$ 与历史独立，于是条件概率退化为"对固定的方向 $u=\nabla\phi(x_t)$，随机子空间与 $u$ 的对齐程度满足 (5.5) 的无条件概率"。所以假设 5.7 等价于一个**纯几何的概率下界**，与算法无关。[11, Lem 6.7] 给出：取 $Q_t$ 诱导的子空间在 Grassmann 流形上均匀分布时，可取 $\kappa_g=\sqrt{1-\frac{q}{10n}}$ 且 $\theta\ge\frac{243}{443}\approx0.5485$。**$\theta>\frac12$ 是硬性的**：下面引理 5.10 出现 $\frac{1}{2\theta-1}$，$\theta\le\frac12$ 时发散。直观解释：对齐的步要**平均比不对齐的步多**，随机方法才有净进展；$\theta>\frac12$ 就是"多数子空间是有用的"。

**关键不等式 $A_t\ge I_t(1-B_t)$（第 885 行）**：逐情形验证。$I_t(1-B_t)$ 只在 $\{I_t=1,B_t=0\}$ 时为 $1$。此时 (5.5) 成立（$I_t=1$）且 $\Delta_t\le\hat C_1\Vert\nabla\phi(x_t)\Vert=\sqrt{1-\kappa_g^2}\tilde C_1\Vert\nabla\phi(x_t)\Vert$，正是引理 5.5 的条件 (5.9)，故 $\rho_t\ge\eta_1$ 且 $\Vert g_t\Vert\ge\eta_2\Delta_t$，即 $A_t=1$ ✓。其余三种情形 $I_t(1-B_t)=0\le A_t$ 自动成立。

**读法**：$A_t\ge I_t(1-B_t)$ 是"$t$ 步成功的**充分**条件"的概率版本——**对齐 + 半径够小 ⟹ 必成功**。注意它只在模型已 fully-linear 时才有意义（引理 5.5 的第二个前提），而模型不 fully-linear 时走分支 (b)，不产生 $t$ 增量，所以不需要担心。

**$\Delta_{min}\le\gamma\hat C_1\epsilon$ 的确切作用（我补出的推导，论文只在第 882 行断言"不改变动力学的主要性质"）**：

第一步，先证 $B_t\le\bar B_t$，其中 $\bar B_t=\mathbb{1}\{\Delta_t\ge\gamma\hat C_1\epsilon\}$。若 $B_t=1$，则 $\Delta_t>\hat C_1\Vert\nabla\phi(x_t)\Vert$；对 $t<T_\epsilon$ 有 $\Vert\nabla\phi(x_t)\Vert>\epsilon$（$T_\epsilon$ 的定义），故 $\Delta_t>\hat C_1\epsilon\ge\gamma\hat C_1\epsilon$（因 $\gamma<1$），即 $\bar B_t=1$ ✓。（论文第 892 行括号里写 "$B_t=1\Rightarrow\bar B_1=1$"，下标 $1$ 是 $t$ 的排版错误。）

第二步，**地板不会咬住**：若 $B_t=1$，由 $\Delta_{min}\le\gamma\hat C_1\epsilon<\gamma\Delta_t$ 得 $\gamma\Delta_t>\Delta_{min}$，于是 (L) 的第二式取等：

$$
B_t=1\Rightarrow\Delta_{t+1}=\gamma\Delta_t\quad(\text{严格收缩，}\max\text{ 不起作用}).
$$

**这才是 $\Delta_{min}\le\gamma\hat C_1\epsilon$ 的全部功能**：它保证在"半径太大"的那些步上，半径仍然严格按 $\gamma$ 的几何阶梯下降，从而"从 $\Delta_0$ 缩到 $\hat C_1\epsilon$ 以下最多需要 $\log_\gamma$ 那么多步"这个阶梯计数引理（引理 5.9）不被地板破坏。若没有这个条件，半径会卡在 $\Delta_{min}>\hat C_1\epsilon$ 上，$B_t$ 恒为 $1$ 而阶梯不动，引理 5.9 与 5.10 立刻失效。**反过来说**：地板在 $\{B_t=0\}$ 区域（半径已经够小）确实会咬住，此时 $A_t=0$ 的步可以无限次停在 $\Delta_{min}$ 而不产生阶梯运动——引理 5.11 恰好要用到这个区域的计数，而它被直接引用为 [10] 的结论（见 G8 的诚实标注）。

## G8 引理 5.8–5.11：哪些能自证，哪些是继承的

### 引理 5.8（大步成功的总数被势函数控制）

设 $\epsilon>\sqrt{\frac{8\epsilon_f}{C_2\gamma^2\hat C_1^2}}$。对任意 $l\in\{0,\dots,T_\epsilon-1\}$ 与算法的**一切实现**（逐样本成立，不需要期望）：

$$
\sum_{t=0}^{l}\bar B_tI_tA_t\le\sum_{t=0}^{l}\bar B_tA_t\le\frac{\phi(x_0)-\phi^\star+4\epsilon_f\sum_{t=0}^{l}(1-\bar B_t)A_t}{\frac{1}{2}C_2(\gamma\hat C_1\epsilon)^2}.
$$

**左边的不等号**：$I_t\le1$，且 $\bar B_tA_t\ge0$，逐项支配 ✓（平凡）。

**右边的证明（论文只给了三句话，我补完）**：

第一步，把 $\epsilon$ 的条件翻译成半径条件。$\epsilon>\sqrt{\frac{8\epsilon_f}{C_2\gamma^2\hat C_1^2}}\iff\gamma\hat C_1\epsilon>\sqrt{\frac{8\epsilon_f}{C_2}}$。所以 $\bar B_t=1$（即 $\Delta_t\ge\gamma\hat C_1\epsilon$）蕴含

$$
\Delta_t\ge\sqrt{\frac{8\epsilon_f}{C_2}}\iff\Delta_t^2\ge\frac{8\epsilon_f}{C_2}\iff C_2\Delta_t^2\ge8\epsilon_f\iff C_2\Delta_t^2-4\epsilon_f\ge\frac{1}{2}C_2\Delta_t^2.
$$

（最后一步：$C_2\Delta_t^2-4\epsilon_f\ge C_2\Delta_t^2-\frac12 C_2\Delta_t^2$，因为 $4\epsilon_f\le\frac12 C_2\Delta_t^2$。）于是对 $\bar B_tA_t=1$ 的步，用引理 5.6：

$$
\phi(x_t)-\phi(x_{t+1})\ge C_2\Delta_t^2-4\epsilon_f\ge\frac{1}{2}C_2\Delta_t^2\ge\frac{1}{2}C_2(\gamma\hat C_1\epsilon)^2,
$$

（最后一个不等号又用了 $\bar B_t=1$。）对 $(1-\bar B_t)A_t=1$ 的步（小成功步），引理 5.6 只给 $\phi(x_t)-\phi(x_{t+1})\ge-4\epsilon_f$。对 $A_t=0$ 的步，$x_{t+1}=x_t$，变化量为 $0$。

第二步，望远镜求和。$\sum_{t=0}^{l}\big(\phi(x_t)-\phi(x_{t+1})\big)=\phi(x_0)-\phi(x_{l+1})\ge\phi(x_0)-\phi^\star$（用假设 1.1 的下界 $\phi^\star=\inf_x\phi(x)$）。把上一步的逐类下界代入：

$$
\phi(x_0)-\phi^\star\le\sum_{t=0}^{l}\big(\phi(x_t)-\phi(x_{t+1})\big)\le\ (\text{无上限，需反向})
$$

**方向核对**：我们要的是 $\sum\bar B_tA_t$ 的**上界**，所以要用 $\phi(x_0)-\phi^\star\ge\sum_t(\text{下降})$，而 $\sum_t(\text{下降})\ge\sum_t\big[\bar B_tA_t\cdot\frac12 C_2(\gamma\hat C_1\epsilon)^2-(1-\bar B_t)A_t\cdot4\epsilon_f\big]$（大步用正下界，小步用 $-4\epsilon_f$ 下界，$A_t=0$ 项用 $0$，统一写成"下界之和"）。于是

$$
\phi(x_0)-\phi^\star\ge\frac{1}{2}C_2(\gamma\hat C_1\epsilon)^2\sum_{t=0}^{l}\bar B_tA_t-4\epsilon_f\sum_{t=0}^{l}(1-\bar B_t)A_t,
$$

移项除系数即得。$\blacksquare$ 论文第 897 行的分式正是这一步的结果，只是把"移项"跳过了。

**这个引理的结构意义**：它把"大步成功的次数"用一个**势函数**（$\phi$ 的总下降量 $\phi(x_0)-\phi^\star$）除以一个**每步最小下降量**（$\frac12 C_2(\gamma\hat C_1\epsilon)^2$）来界住，多出来的 $+4\epsilon_f\sum(1-\bar B_t)A_t$ 是"小步成功可能抬高 $\phi$"的**记账项**——它会在定理 5.12 里变成 $C_3$，并被 $\epsilon$ 的下界压成 $\le\theta-\frac12$。整个第 5 节的含噪分析难点全在这一个记账项上。

### 引理 5.9（失败步数 ≤ 成功步数 + 阶梯层数）

设 $\Delta_0\ge\hat C_1\epsilon$ 且 $\Delta_{min}\le\gamma\hat C_1\epsilon$。则对一切 $l$ 与一切实现：

$$
\sum_{t=0}^{l}B_t(1-A_t)\le\sum_{t=0}^{l}\bar B_tA_t+\log_\gamma\Big(\frac{\hat C_1\epsilon}{\Delta_0}\Big).
$$

论文写"easily follows from the dynamics"，**没有给证明**。我按阶梯论证重建，并说明它为什么成立：

- 由 G7 第一步，$B_t\le\bar B_t$，所以左边 $\le\sum\bar B_t(1-A_t)$。
- 考虑"层"的编号 $j\in\mathbb{Z}$，把 $\Delta_t\in[\gamma^{j+1}\hat C_1\epsilon,\ \gamma^{j}\hat C_1\epsilon)$ 记作第 $j$ 层。$\bar B_t=1$ 等价于 $\Delta_t\ge\gamma\hat C_1\epsilon$，即 $j\ge0$（层号非负）。
- 由 (L) 与 G7 第二步：在 $\{B_t=1\}$ 上，$A_t=0$ 使层号 $-1$（严格乘 $\gamma$，地板不干预），$A_t=1$ 使层号 $+1$。
- 从初始层 $j_0$（由 $\Delta_0\ge\hat C_1\epsilon$ 知 $j_0\ge0$）出发，要在非负层内累计 $N_{fail}$ 次"层号 $-1$"，必须有至少 $N_{fail}-|j_0|$ 次"层号 $+1$"（否则半径早就掉出 $\bar B=1$ 区域，那些步不再计入左边）。而"层号 $+1$"的步必然同时满足 $\bar B_t=1$（因为它从非负层出发）与 $A_t=1$。
- $j_0$ 与 $\Delta_0$ 的关系：$\gamma^{j_0+1}\hat C_1\epsilon\le\Delta_0$ ⟹ $j_0\ge\log_\gamma\frac{\Delta_0}{\hat C_1\epsilon}-1$。用 $\gamma<1$ 换底整理，可下降的层数预算就是 $\log_\gamma\frac{\hat C_1\epsilon}{\Delta_0}\ (\ge0)$。

**诚实标注**：上面第三、四步的"$-1$ 与取整"细节我给的是标准 [10, Lem 3.x] 式论证的骨架，**论文正文与我的重建都没有精确到取整符号**。但结论的阶与形式是清楚的，且引理 5.10 只用到它的这个形式。**若 $\Delta_{min}>\gamma\hat C_1\epsilon$，第二条的"严格乘 $\gamma$"失效，整个论证崩塌**——这与 G7 的结论一致，也说明论文把这个条件写出来是必要的、不是多余的。

### 引理 5.10（"半径太大"的总步数）

论文先引用 [10] 的一个结论：在假设 5.7 下

$$
\mathbb{E}\Big(\sum_{t=0}^{T_\epsilon-1}\bar B_t(1-I_t)\Big)\le\frac{1-\theta}{\theta}\mathbb{E}\Big(\sum_{t=0}^{T_\epsilon-1}\bar B_tI_t\Big), \tag{$\ddagger$}
$$

再由它推出

$$
\mathbb{E}\Big(\sum_{t=0}^{T_\epsilon-1}B_t\Big)\le\frac{1}{2\theta-1}\Big(\sum_{t=0}^{T_\epsilon-1}\bar B_tA_t+\log_\gamma\Big(\frac{\hat C_1\epsilon}{\Delta_0}\Big)\Big).
$$

**先记两处问题**：(i) 右边第一个求和**没有 $\mathbb{E}$**，左边有——左边是数、右边是随机变量，量纲不符，**应为 $\mathbb{E}\big[\sum\bar B_tA_t\big]$**（定理 5.12 的证明里确实代入了引理 5.8 的确定性上界，所以是排版漏写，不影响结果）。(ii) $(\ddagger)$ 是 [10] 的结论，本文未证。

**$(\ddagger)$ 的机制（我按 [10] 的思路补证，因为它只用假设 5.7）**：$\bar B_t(1-I_t)$ 数的是"半径大且子空间不对齐"的步。对每个 $t$，把 $\mathcal{F}_{t-1}$ 固定，$\bar B_t$ 就变成常数，而 $\mathbb{E}[1-I_t\mid\mathcal{F}_{t-1}]=1-\mathbb{P}\{I_t=1\mid\mathcal{F}_{t-1}\}\le1-\theta$，$\mathbb{E}[I_t\mid\mathcal{F}_{t-1}]\ge\theta$。于是 $\frac{\mathbb{E}[\bar B_t(1-I_t)]}{\mathbb{E}[\bar B_tI_t]}$ 的逐层比较给出 $\mathbb{E}[\sum\bar B_t(1-I_t)]\le\frac{1-\theta}{\theta}\mathbb{E}[\sum\bar B_tI_t]$（严格证明需要对 $t$ 做条件期望的逐层求和，或用"每次对齐的试验成功后才允许下一次"的负相关结构；这是 [10] 的核心引理之一，我在此只确认**它只依赖假设 5.7 与 $\bar B_t$ 的 $\mathcal{F}_{t-1}$-可测性**，不依赖噪声、不依赖 $\Delta_{min}$，所以可以直接继承）。

**从 $(\ddagger)$+5.9 到 5.10 的这一步，我给出可自证的弱化版**：

$$
\sum B_t=\sum B_t(1-A_t)+\sum B_tA_t\overset{5.9}{\le}\sum\bar B_tA_t+\log_\gamma(\cdot)+\sum B_tA_t,
$$

再用 $B_t\le\bar B_t$ 得 $\sum B_tA_t\le\sum\bar B_tA_t$，于是

$$
\sum B_t\le2\sum\bar B_tA_t+\log_\gamma(\cdot).
$$

论文的形式是 $\frac{1}{2\theta-1}$ 倍。两者比较：$\frac{1}{2\theta-1}\ge2\iff2\theta-1\le\frac12\iff\theta\le\frac34$。**在 $\theta\in(\frac12,\frac34]$ 区间（包含本文实际使用的 $\theta\ge\frac{243}{443}\approx0.5485$ 的下端），我这条初等推导已经足够，甚至比论文的形式更紧**（$2$ 倍优于 $\frac{1}{2\theta-1}$ 倍）。只有当 $\theta>\frac34$（子空间几乎必然对齐）时，才需要 $(\ddagger)$ 提供的更精细的补偿。**结论：引理 5.10 的正确性不依赖 [10] 的 $(\ddagger)$，至少在实际参数区间内不依赖**；论文在此引 [10] 是为了覆盖整个 $\theta\in(\frac12,1]$。**这是一处"引用过度"，可以简化。**

### 引理 5.11（"半径不太大"的步数占时间的比例）

$$
\mathbb{E}\Big(\sum_{t=0}^{T_\epsilon-1}(1-B_t)\Big)\le\frac{1}{2\theta}\mathbb{E}[T_\epsilon].
$$

论文写"shown in [10] ... since $A_t\ge I_t(1-B_t)$ and by the dynamics of $\Delta_t$"，**完全未证**。我能自证的是弱化版：

**弱化版（只用假设 5.7 与 $A_t\ge I_t(1-B_t)$）**：分解 $(1-B_t)=(1-B_t)I_t+(1-B_t)(1-I_t)$。第一项由 $A_t\ge I_t(1-B_t)$ 得 $(1-B_t)I_t\le A_t$，更精确地 $(1-B_t)I_t\le(1-B_t)A_t$（因为 $I_t(1-B_t)\le A_t$ 且两边已含 $(1-B_t)$ 因子——严格说：在 $\{I_t=1,B_t=0\}$ 上 $A_t=1$，故 $(1-B_t)I_t\le(1-B_t)A_t$ 逐样本成立 ✓）。对第二项取条件期望：$\mathbb{E}[(1-B_t)(1-I_t)]=\mathbb{E}[(1-B_t)\mathbb{E}[1-I_t\mid\mathcal{F}_{t-1}]]\le(1-\theta)\mathbb{E}[\sum(1-B_t)]$。合并：

$$
\mathbb{E}\Big[\sum(1-B_t)\Big]\le\mathbb{E}\Big[\sum(1-B_t)A_t\Big]+(1-\theta)\mathbb{E}\Big[\sum(1-B_t)\Big]\Rightarrow\theta\,\mathbb{E}\Big[\sum(1-B_t)\Big]\le\mathbb{E}\Big[\sum(1-B_t)A_t\Big]\le\mathbb{E}\Big[\sum A_t\Big].
$$

**$\frac{1}{2\theta}$ 版还需要的另一半**：$\mathbb{E}[\sum A_t]\le\frac12\mathbb{E}[T_\epsilon]$。这一半**不真**：若每个 $Q_t$ 都对齐（$\theta=1$），则 $B_t=0$ 的步全部成功，$\sum A_t$ 可以接近 $T_\epsilon$。但此时半径按 $\gamma^{-1}$ 指数增长，几步之内必然 $B_t=1$，所以"$B_t=0$ 且 $A_t=1$"的**连续段长度**被 $\log_\gamma\frac{\Delta_{min}}{\hat C_1\epsilon}$ 界住——这正是"by the dynamics of $\Delta_t$"的含义。**然而地板 $\Delta_{min}$ 的存在恰恰破坏了这个阶梯论证在小半径侧的可用性**（半径可以长期停在 $\Delta_{min}$，此时 $B_t=0$、$A_t=0$ 可以无限重复，只要 $I_t=0$）。

**诚实结论**：引理 5.11 在**有地板**的算法 3 中不是 [10] 结论的直接搬运。我给出的可自证版本是

$$
\mathbb{E}\Big[\sum(1-B_t)\Big]\le\frac{1}{\theta}\mathbb{E}\Big[\sum(1-B_t)A_t\Big]\le\frac{1}{\theta}\Big(\text{地板以下的最长连续成功段}\Big)^{+}\cdots
$$

即**必须额外假设/证明"$\{B_t=0,A_t=0\}$ 的步数被 $\{B_t=0,A_t=1\}$ 的步数按 $\frac{1-\theta}{\theta}$ 比例界住"**（这由 $(\ddagger)$ 的同类论证给出：在 $B_t=0$ 区域，$A_t\ge I_t$，故 $\{B_t=0,A_t=0\}\subseteq\{B_t=0,I_t=0\}$，于是 $\mathbb{E}[\sum(1-B_t)(1-A_t)]\le\frac{1-\theta}{\theta}\mathbb{E}[\sum(1-B_t)I_t]\le\frac{1-\theta}{\theta}\mathbb{E}[\sum(1-B_t)A_t]$，从而 $\mathbb{E}[\sum(1-B_t)]\le\frac{1}{\theta}\mathbb{E}[\sum(1-B_t)A_t]$ 与 $\mathbb{E}[\sum(1-B_t)]=\mathbb{E}[\sum(1-B_t)A_t]+\mathbb{E}[\sum(1-B_t)(1-A_t)]$ 联立解出）——**解出来正是 $\frac{1}{2\theta}$ 吗？** 设 $X=\mathbb{E}[\sum(1-B_t)A_t]$，$Y=\mathbb{E}[\sum(1-B_t)(1-A_t)]$，则 $Y\le\frac{1-\theta}{\theta}X$，且 $\mathbb{E}[\sum(1-B_t)]=X+Y\le X\big(1+\frac{1-\theta}{\theta}\big)=\frac{X}{\theta}$。又 $X\le\mathbb{E}[\sum A_t]$，而 $\sum A_t+\sum(1-A_t)=T_\epsilon$。要把 $X$ 换成 $\frac12 T_\epsilon$ 需要 $\mathbb{E}[\sum A_t]\ge\mathbb{E}[\sum(1-A_t)]$，即"成功步不少于失败步"——**这在 $\theta>\frac12$ 时由上一条（失败步被 $\frac{1-\theta}{\theta}<1$ 倍的成功步界住）恰好成立**：$\mathbb{E}[\sum(1-A_t)]$ 中落在 $\{B_t=0\}$ 的部分被 $\frac{1-\theta}{\theta}\mathbb{E}[\sum(1-B_t)A_t]$ 界住，落在 $\{B_t=1\}$ 的部分由引理 5.9 的阶梯界住（**这一步在 $\{B_t=1\}$ 区域是安全的，因为 G7 已证地板不咬住 $B_t=1$**）。忽略阶梯的加性 $\log_\gamma$ 项，得 $\mathbb{E}[\sum A_t]\ge\frac{\theta}{2\theta-1}\cdot(\cdot)$……

**最终判定**：$\frac{1}{2\theta}$ 的形式**可以**从"$B_t=1$ 区域用阶梯（地板无害）+ $B_t=0$ 区域用 $(\ddagger)$ 式比例 + 忽略加性对数项"拼出来，但**拼出的结果会带一个加性 $\log_\gamma\frac{\Delta_{min}}{\hat C_1\epsilon}$ 项**，论文与 [10] 的陈述里都没有这一项。所以：**引理 5.11 在有地板时缺一项目测的对数加性项；由于定理 5.12 的最终界里本来就有 $\log_\gamma\frac{\hat C_1\epsilon}{\Delta_0}$ 项，补上这一项不改变任何阶**。这是一处**非致命**的陈述缺项，且是"地板破坏 [10] 动力学"这一风险的唯一实际落点。

## G9 定理 5.12：期望迭代数的完整代数

**命题**：设假设 1.2、2.2、5.7 成立。对任意 $\epsilon>\sqrt{\frac{16\epsilon_f}{(2\theta-1)^2C_2\hat C_1^2\gamma^2}}$，在 $\Delta_0\ge\hat C_1\epsilon$ 与 $\Delta_{min}\le\gamma\hat C_1\epsilon$ 下，

$$
\mathbb{E}[T_\epsilon]\le\frac{4\theta}{(2\theta-1)^2}\Big(\frac{\phi(x_0)-\phi^\star}{\frac{1}{2}C_2(\gamma\hat C_1\epsilon)^2}+\log_\gamma\Big(\frac{\hat C_1\epsilon}{\Delta_0}\Big)\Big).
$$

**第一步：$(1-\bar B_t)A_t\le(1-B_t)$。** 因为 $\bar B_t\ge B_t$（G7），$1-\bar B_t\le1-B_t$，再乘 $A_t\le1$ ✓。于是引理 5.8 的记账项满足

$$
\sum_{t=0}^{l}(1-\bar B_t)A_t\le\sum_{t=0}^{l}(1-B_t).
$$

**第二步：把引理 5.8 代入引理 5.10**（取 $l=T_\epsilon-1$，两边取期望）：

$$
\mathbb{E}\Big(\sum B_t\Big)\le\frac{1}{2\theta-1}\Big(\underbrace{\frac{\phi(x_0)-\phi^\star}{\frac{1}{2}C_2(\gamma\hat C_1\epsilon)^2}}_{\text{记作 }U}+C_3\,\mathbb{E}\Big[\sum(1-B_t)\Big]+\log_\gamma\Big(\frac{\hat C_1\epsilon}{\Delta_0}\Big)\Big),
$$

其中把 $4\epsilon_f$ 除进分母得到的系数正是论文第 948 行的

$$
C_3=\frac{4\epsilon_f}{(2\theta-1)\cdot\frac{1}{2}C_2(\gamma\hat C_1\epsilon)^2}.
$$

（核对：引理 5.8 给出 $\sum\bar B_tA_t\le U+\frac{4\epsilon_f}{\frac12 C_2(\gamma\hat C_1\epsilon)^2}\mathbb{E}[\sum(1-B_t)]$；代进引理 5.10 的 $\frac{1}{2\theta-1}$ 倍里，$\frac{4\epsilon_f}{(2\theta-1)\frac12 C_2(\gamma\hat C_1\epsilon)^2}=C_3$ ✓。）

**第三步：$T_\epsilon$ 的分解。** $T_\epsilon=\sum_{t=0}^{T_\epsilon-1}1=\sum(1-B_t)+\sum B_t$，故

$$
\mathbb{E}[T_\epsilon]\le U+\log_\gamma(\cdot)\over(2\theta-1)+(1+C_3)\mathbb{E}\Big[\sum(1-B_t)\Big].
$$

（论文的写法把 $\frac{1}{2\theta-1}$ 只乘在 $U+\log$ 上 ✓。）

**第四步：用引理 5.11 吸收最后一项。** $\mathbb{E}[\sum(1-B_t)]\le\frac{1}{2\theta}\mathbb{E}[T_\epsilon]$：

$$
\mathbb{E}[T_\epsilon]\le\frac{1}{2\theta-1}\Big(U+\log_\gamma(\cdot)\Big)+(1+C_3)\frac{1}{2\theta}\mathbb{E}[T_\epsilon].
$$

**第五步：$C_3\le\theta-\frac12$ 的来源（论文第 960 行一句话，展开）**：由 $\epsilon$ 的下界 $\epsilon^2>\frac{16\epsilon_f}{(2\theta-1)^2C_2\hat C_1^2\gamma^2}$，取倒数（两边正，方向翻转）：

$$
\frac{1}{\epsilon^2}<\frac{(2\theta-1)^2C_2\hat C_1^2\gamma^2}{16\epsilon_f}.
$$

代入 $C_3=\frac{4\epsilon_f}{(2\theta-1)\frac12 C_2\gamma^2\hat C_1^2\epsilon^2}$：

$$
C_3<\frac{4\epsilon_f}{(2\theta-1)\frac12 C_2\gamma^2\hat C_1^2}\cdot\frac{(2\theta-1)^2C_2\hat C_1^2\gamma^2}{16\epsilon_f}=\frac{4(2\theta-1)}{16\cdot\frac12}=\frac{2\theta-1}{2}=\theta-\frac{1}{2}.
$$

✓ 所以 $\epsilon$ 的那个"看起来莫名其妙"的下界**唯一的作用**就是让 $C_3\le\theta-\frac12$，从而让噪声记账项可被吸收。

**第六步：合并系数。** $1+C_3\le1+\theta-\frac12=\theta+\frac12=\frac{2\theta+1}{2}$，除以 $2\theta$：

$$
\frac{1+C_3}{2\theta}\le\frac{2\theta+1}{4\theta}.
$$

移项：$1-\frac{2\theta+1}{4\theta}=\frac{4\theta-2\theta-1}{4\theta}=\frac{2\theta-1}{4\theta}$，于是

$$
\frac{2\theta-1}{4\theta}\mathbb{E}[T_\epsilon]\le\frac{1}{2\theta-1}\Big(U+\log_\gamma(\cdot)\Big)\Rightarrow\mathbb{E}[T_\epsilon]\le\frac{4\theta}{(2\theta-1)^2}\Big(U+\log_\gamma(\cdot)\Big).\ \blacksquare
$$

**$\theta=\frac{243}{443}$ 时的实际常数（自算，论文未给）**：$2\theta-1=\frac{486-443}{443}=\frac{43}{443}\approx0.0971$。

$$
\frac{4\theta}{(2\theta-1)^2}=\frac{4\cdot243\cdot443}{43^2}=\frac{430596}{1849}\approx232.9.
$$

再加上 $\frac{1}{\frac12 C_2(\gamma\hat C_1\epsilon)^2}$ 里的 $C_2,\gamma$ 与 $\log_\gamma$，**定理 5.12 的常数在 $\theta$ 取下界时约为 $233$ 倍的 $\frac{\phi(x_0)-\phi^\star}{C_2\gamma^2\hat C_1^2\epsilon^2/2}$**。这是一个**非空泛但很大**的常数；$\theta\to\frac12^+$ 时以 $(2\theta-1)^{-2}$ 的速度爆炸。审计含义：本文的随机子空间界在 $\theta$ 只比 $\frac12$ 大一点时**在实践上不可用**，这也解释了 §6 里 GC-sub 表现差（虽然作者把主因归为"每步要重抽 $q$ 个点"）。

## G10 §5.2：子空间有限差分 (5.10)、$\Delta_{min}^2$ 为什么是平方、以及 $\sqrt q$ 与 $\sqrt{n/q}$ 的精确抵消

**子空间有限差分（第 973 行）**：

$$
\hat g(0)=\sum_{i=1}^{q}\frac{f(x+\delta Qu_i)-f(x)}{\delta}u_i,\qquad g(x)=Q\hat g(0),
$$

$u_i$ 是某个 $q\times q$ 正交矩阵的第 $i$ 列（已知量，取 $I_q$ 即标准基）。$Qu_i\in\mathbb{R}^n$ 是子空间里的第 $i$ 个搜索方向，$\Vert Qu_i\Vert=1$。$\delta>0$ 是差分步长（超参数，取 $\delta=\Delta_k$）。

**oracle 代价**：需要 $f$ 在 $x$ 与 $x+\delta Qu_i$（$i=1,\dots,q$）处的值，共 $q+1$ 个；但 $f(x_k)$ 在上一轮已经算过（失败步 $x_{k+1}=x_k$），成功步时 $f(x_{k+1})=f(x_k+s_k)$ 也算过了，所以**每轮只需 $q$ 个新值**（第 990 行"either $q$ or $q+1$"）。

**误差界推导（论文写"by similar analysis as in [4]"，我完整给出）**：$\hat g(0)$ 的第 $j$ 分量（因为 $\{u_i\}$ 正交，$\hat g(0)^\top u_j=\frac{f(x+\delta Qu_j)-f(x)}{\delta}$）与 $\nabla\hat\phi(0)^\top u_j=Qu_j\cdot\nabla\phi(x)$ 比较。一元 Taylor 带 Lagrange 余项：

$$
\phi(x+\delta Qu_j)=\phi(x)+\delta\nabla\phi(x)^\top(Qu_j)+\frac{\delta^2}{2}(Qu_j)^\top\nabla^2\phi(\xi_j)(Qu_j),\quad \xi_j\in[x,x+\delta Qu_j].
$$

（依据：$\phi$ 二次连续可微，由假设 1.2。）除以 $\delta$ 并减去 $\nabla\phi(x)^\top(Qu_j)$：

$$
\Big\vert\frac{\phi(x+\delta Qu_j)-\phi(x)}{\delta}-\nabla\phi(x)^\top(Qu_j)\Big\vert\le\frac{\delta}{2}\Vert\nabla^2\phi(\xi_j)\Vert\Vert Qu_j\Vert^2\le\frac{\delta L}{2},
$$

（用 $\Vert\nabla^2\phi\Vert\le L$，这是假设 1.2 中 $L$-光滑与 Hessian 存在的标准推论。）再加上 oracle 噪声：$\vert f-\phi\vert\le\epsilon_f$ 在两个点各贡献一次，$\frac{2\epsilon_f}{\delta}$。于是逐分量

$$
\vert\hat g(0)_j-\nabla\hat\phi(0)_j\vert\le\frac{L\delta}{2}+\frac{2\epsilon_f}{\delta}.
$$

对 $j=1,\dots,q$ 取 $\ell_2$ 范数（$q$ 个相同量级的分量）：

$$
\Vert\nabla{\hat m}(0)-\nabla{\hat\phi}(0)\Vert=\Vert\hat g(0)-\nabla{\hat\phi}(0)\Vert\le\sqrt q\Big(\frac{L\delta}{2}+\frac{2\epsilon_f}{\delta}\Big)=\frac{\sqrt qL\delta}{2}+\frac{2\sqrt q\epsilon_f}{\delta}.
$$

✓ 即第 981 行。注意 $\nabla{\hat m}(0)=Q^\top g_k=Q^\top Q\hat g(0)=\hat g(0)$ ✓（又用了 $Q^\top Q=I$），所以"模型梯度"就是 $\hat g(0)$，无需额外论证。

**$\kappa_{eg}$ 的提取，以及为什么分母是 $\Delta_{min}^2$（这里极易读错，我逐步做）**：fully-linear 要求的形状是 $\Vert\nabla{\hat m}(0)-\nabla{\hat\phi}(0)\Vert\le\kappa_{eg}\Delta_k$。取 $\delta=\Delta_k$ 后上面得到的是

$$
\frac{\sqrt qL\Delta_k}{2}+\frac{2\sqrt q\epsilon_f}{\Delta_k}=\Big(\frac{\sqrt qL}{2}+\frac{2\sqrt q\epsilon_f}{\Delta_k^{\,2}}\Big)\Delta_k,
$$

所以 $\kappa_{eg}=\frac{\sqrt qL}{2}+\frac{2\sqrt q\epsilon_f}{\Delta_k^2}$，再用 $\Delta_k\ge\Delta_{min}$ 放大第二项（$\Delta_k^2\ge\Delta_{min}^2$ ⟹ 分数变小，取 $\Delta_{min}$ 得一致上界）：

$$
\kappa_{eg}=\frac{\sqrt qL}{2}+\frac{2\sqrt q\epsilon_f}{\Delta_{min}^2},\qquad \kappa_{ef}=\kappa_{eg}+\frac{L_Q+\kappa_{bhm}}{2}.
$$

✓ 与第 987 行一致。**平方完全合理**：分子是 $\frac{\epsilon_f}{\Delta_k}$，要写成 $\kappa_{eg}\Delta_k$ 就必须再除一次 $\Delta_k$。这也解释了为什么 $\Delta_{min}$ 是**必需**的：$\Delta_{min}\to0$ 时 $\kappa_{eg}\to\infty$，整个分析崩塌。**顺带修正阶段 D 的一处含糊**：全空间那版（第 261 行）写 $\kappa_{eg}=\sqrt nL$，我当时指出"$\frac{\sqrt nL\Delta_k}{2}+\frac{2\sqrt n\epsilon_f}{\Delta_k}$ 不是常数"。现在可以精确补完：按同样算法 $\kappa_{eg}=\frac{\sqrt nL}{2}+\frac{2\sqrt n\epsilon_f}{\Delta_k^2}$，再代入论文给的下界 $\Delta_k\ge2\sqrt{\frac{\epsilon_f}{L}}$（即 $\Delta_k^2\ge\frac{4\epsilon_f}{L}$）：

$$
\kappa_{eg}\le\frac{\sqrt nL}{2}+\frac{2\sqrt n\epsilon_f}{4\epsilon_f/L}=\frac{\sqrt nL}{2}+\frac{\sqrt nL}{2}=\sqrt nL,
$$

✓✓ **正好是论文的 $\kappa_{eg}=\sqrt nL$**，且 $\kappa_{ef}=\kappa_{eg}+\frac{L+\kappa_{bhm}}{2}=\frac{L+\kappa_{bhm}}{2}+\sqrt nL$ ✓ 与第 261 行完全吻合。所以论文第 261 行**没有错**，只是把"除以 $\Delta_k$ 再用下界"这一步完全省略了。阶段 D 里我标的"疑点"到此关闭。

**定理 5.13 的推导链**（第 992–1000 行，逐步验证）：

1. $\tilde C_1^{-1}=\Theta(\kappa_{eg})$。核对：$\tilde C_1^{-1}=\max\{\eta_2,\kappa_{bhm},\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}\}+\kappa_{eg}$。定理假设 $\eta_1,\eta_2,\kappa_{bhm},\gamma$ 为常数，$L\ge1$。由 $\kappa_{eg}\ge\frac{\sqrt qL}{2}\ge\frac{L}{2}\ge\frac12$ 与 $\kappa_{ef}=\kappa_{eg}+\frac{L_Q+\kappa_{bhm}}{2}\le\kappa_{eg}+\frac{L}{2}+\frac{\kappa_{bhm}}{2}\le2\kappa_{eg}+\frac{\kappa_{bhm}}{2}=\Theta(\kappa_{eg})$，$\max$ 的第三项 $\Theta(\kappa_{eg})$ 支配前两项（常数），故 $\tilde C_1^{-1}=\Theta(\kappa_{eg})=\Theta(\sqrt q(L+\frac{\epsilon_f}{\Delta_{min}^2}))$ ✓。
2. [11, Lem 6.7]：$\sqrt{1-\kappa_g^2}=\Theta(\sqrt{q/n})$（具体是 $\sqrt{q/(10n)}$）。
3. $\hat C_1^{-1}=\tilde C_1^{-1}/\sqrt{1-\kappa_g^2}=\Theta(\sqrt q(L+\cdot))\cdot\Theta(\sqrt{n/q})=\Theta(\sqrt n(L+\frac{\epsilon_f}{\Delta_{min}^2}))$ ✓。
4. **$q$ 完全抵消**。这是本节最重要的结构性事实：子空间把 $\kappa_{eg}$ 缩小 $\sqrt{q/n}$ 倍（收益），同时把对齐质量 $\sqrt{1-\kappa_g^2}$ 也缩小 $\sqrt{q/n}$ 倍（损失），两者在 $\hat C_1^{-1}$ 里**精确相消**。所以**迭代数与 $q$ 无关**，$q$ 只出现在每步的 oracle 代价里。
5. 迭代界：$\mathbb{E}[K_\epsilon]\le\mathcal{O}(\hat C_1^{-2}\epsilon^{-2})=\mathcal{O}\big(\frac{n}{\epsilon^2}(L+\frac{\epsilon_f}{\Delta_{min}^2})^2\big)$ ✓（$C_2$ 是常数，$\theta$ 是常数，$\log$ 项被 big-O 吸收，论文第 998 行明说"suppresses constant factors and an additive logarithmic term"）。注意 $K_\epsilon$（按 $k$ 计）与 $T_\epsilon$（按 $t$ 计）在这里相同，因为有限差分**没有模型改进步**（第 990 行）——这正是第 1000 行"k and t are equivalent"的理由。
6. $\epsilon$ 的下界合并：定理 5.12 要求 $\epsilon>\sqrt{\frac{16\epsilon_f}{(2\theta-1)^2C_2\hat C_1^2\gamma^2}}=\Theta(\hat C_1^{-1}\sqrt{\epsilon_f})=\Omega(\sqrt n(L+\frac{\epsilon_f}{\Delta_{min}^2})\sqrt{\epsilon_f})$；另要求 $\Delta_{min}\le\gamma\hat C_1\epsilon$ 即 $\epsilon\ge\frac{\Delta_{min}}{\gamma\hat C_1}=\Omega(\sqrt n(L+\frac{\epsilon_f}{\Delta_{min}^2})\Delta_{min})$。两条取大 ⟹ $\epsilon>\Omega(\sqrt n(L+\frac{\epsilon_f}{\Delta_{min}^2})(\sqrt{\epsilon_f}+\Delta_{min}))$ ✓ 与第 992 行一致。
7. $\Delta_0\ge\hat C_1\epsilon$ 代入最小可行 $\epsilon$：$\hat C_1\epsilon_{min}=\Omega(\sqrt{\epsilon_f}+\Delta_{min})$ ⟹ $\Delta_0\ge\mathcal{O}(\sqrt{\epsilon_f}+\Delta_{min})$ ✓。

**$\Delta_{min}$ 的最优选取（论文第 1002 行只说"approximately optimized by taking $\Delta_{min}=\Theta(\sqrt{\epsilon_f})$"，我把它算出来）**：要最小化 $h(\Delta)=(L+\frac{\epsilon_f}{\Delta^2})(\sqrt{\epsilon_f}+\Delta)$。展开 $h=L\sqrt{\epsilon_f}+L\Delta+\epsilon_f^{3/2}\Delta^{-2}+\epsilon_f\Delta^{-1}$，求导置零：

$$
h'(\Delta)=L-2\epsilon_f^{3/2}\Delta^{-3}-\epsilon_f\Delta^{-2}=0\iff L=\frac{2\epsilon_f^{3/2}}{\Delta^3}+\frac{\epsilon_f}{\Delta^2}.
$$

令 $\Delta=\alpha\sqrt{\epsilon_f}$：$L=\frac{2}{\alpha^3}+\frac{1}{\alpha^2}$。数值解：$L=1\Rightarrow\alpha\approx1.52$；$L=10\Rightarrow\alpha\approx0.63$；$L=1000\Rightarrow\alpha\approx0.126$。**所以 $\alpha$ 只随 $L$ 缓变（$L$ 大时 $\alpha\approx(2/L)^{1/3}$），$\Delta_{min}=\Theta(\sqrt{\epsilon_f})$ 在 $L=\Theta(1)$ 下是对的**，且调优 $\alpha$ 相对 $\alpha=1$ 只带来不到 $10\%$ 的常数改善（$L=10$ 时 $h$ 从 $22\sqrt{\epsilon_f}$ 降到 $20.4\sqrt{\epsilon_f}$）。论文用"approximately"是诚实的。

取 $\Delta_{min}=\sqrt{\epsilon_f}$ 后：$\frac{\epsilon_f}{\Delta_{min}^2}=1$，$\epsilon>\Omega(\sqrt n(L+1)\sqrt{\epsilon_f})$，$\mathbb{E}[K_\epsilon]\le\mathcal{O}(\frac{n}{\epsilon^2}(L+1)^2)=\mathcal{O}(\frac{n}{\epsilon^2})$（$L$ 视为常数）✓ 第 1005 行。每步 $\mathcal{O}(q)$ 次 oracle ⟹

$$
\mathbb{E}[\mathcal{C}_\epsilon]\le\mathcal{O}\Big(\frac{nq}{\epsilon^2}\Big)
$$

✓ 第 1011 行。

**与第 2、4 节的正面对照（这是评价本文贡献的关键一张表）**：

| 方案 | 迭代数 | 每迭代 oracle | 总 oracle | 精度下限 |
| --- | --- | --- | --- | --- |
| 算法 1 + 有限差分，$\eta_2$ 常数（[11] 的界） | $\mathcal{O}(n\epsilon^{-2})$ | $n+1$ | $\mathcal{O}(n^2\epsilon^{-2})$ | $\epsilon\ge\Omega(\sqrt{\epsilon_f})$ |
| 算法 1 + 有限差分，$\eta_2=\sqrt n$（推论 2.10） | $\mathcal{O}(\sqrt n\epsilon^{-2})$ | $n+1$ | $\mathcal{O}(n^{3/2}\epsilon^{-2})$ | 同上 |
| 算法 2 + 插值，$\eta_2=\sqrt n$（推论 4.4） | $\mathcal{O}(\sqrt n\epsilon^{-2})$ | $1+\mathcal{O}(\log n)$ 摊销 | $\mathcal{O}(n^{3/2}\log n\,\epsilon^{-2})$ | $\epsilon\ge\Omega(\sqrt{n\Lambda\epsilon_f})$ |
| 算法 3 + 子空间有限差分（定理 5.13） | $\mathcal{O}(n\epsilon^{-2})$ | $q$ | $\mathcal{O}(nq\epsilon^{-2})$ | $\epsilon\ge\Omega(\sqrt{n\epsilon_f})$ |
| 算法 4 + 子空间几何校正（推论 5.14） | $\mathcal{O}(n\epsilon^{-2})$ | $q$ 摊销 $\mathcal{O}(q\log q)$ | $\mathcal{O}(nq\log q\,\epsilon^{-2})$ | 同上 |

**机制解释（我给出，论文未给）**：为什么子空间的迭代数是 $n$ 而全空间能到 $\sqrt n$？因为迭代数 $\propto\frac{\hat C_1^{-2}}{C_2}$，而 $\hat C_1^{-1}$ 里带 $\sqrt{n/q}$ 的**对齐放大**。全空间没有这一项（$\kappa_g=0$），于是 $\eta_2=\sqrt n$ 可以"免费"把 $C_2$ 抬到 $\Theta(\sqrt n)$（$\tilde C_1^{-1}$ 已经被 $\kappa_{eg}=\Theta(\sqrt n)$ 支配，$\eta_2$ 塞进 $\max$ 不增加它）。**子空间里 $\eta_2$ 不再免费**：$\kappa_{eg}=\Theta(\sqrt qL)$ 是常数级（$q,L$ 固定，$\Delta_{min}=\sqrt{\epsilon_f}$），所以 $\max$ 会被 $\eta_2$ 抓住，$\eta_2$ 再乘上 $\sqrt{n/q}$ 的放大。设 $\eta_2\ge\kappa_{bhm}$（此时 $C_2=\frac{\eta_1\kappa_{fcd}}{2}\eta_2$），迭代数 $\propto\frac{(\max\{\eta_2,\Theta(1)\})^2\cdot(n/q)}{\eta_2}=\frac{n\eta_2}{q}$，**关于 $\eta_2$ 单调递增 ⟹ 最优就是 $\eta_2=\Theta(1)$**。

**这条分析有两个价值**：(1) 它证明定理 5.13 里"$\eta_2$ 为常数"的假设**在本分析框架内是最优选择**，不是偷懒——与推论 2.10/4.4 恰好相反，值得写进定稿；(2) 它给出一个可证伪的更强主张：**若要子空间方法同时拿到 $\sqrt n$ 的迭代收益，必须改进 $\hat C_1$ 的定义**（例如把 $\sqrt{1-\kappa_g^2}$ 从 $\tilde C_1$ 的门槛里挪出来，用"对齐时按投影梯度、不对齐时按全梯度"的分层判据），这是一个具体的研究缺口。

## G11 §5.3 算法 4 与推论 5.14 的计数

**算法 4 相对算法 3 的三处改动**：

1. 第 1 行：模型不再用有限差分，而是用 (3.2) 的插值模型，样本集 $\mathcal{Y}_k\subset\mathbb{R}^q$（$|\mathcal{Y}_k|=q$）与 $\mathcal{Z}_k$（$|\mathcal{Z}_k|\le p$），全部住在**子空间坐标**里。
2. 第 4 行第二支的判据从"模型 not fully-linear"换成"$\mathcal{Y}_k$ 不是 $\Lambda$-poised"——这是**可检验的**（只需算 $\max_j\max_{\Vert s\Vert\le\Delta}\vert\ell_j(s)\vert$），比算法 1/3 的不可检验判据前进了一步（但算法 5 的实现里又混用了两者）。
3. **新增第 6 行**：若成功或 $\mathcal{Y}_k$ 已 $\Lambda$-poised，则 $(\mathcal{Y}_{k+1},\mathcal{Z}_{k+1})\leftarrow(\mathcal{Y}_0,\mathcal{Z}_0)$，即**重置样本集**。配合第 5 行的"成功或已 poised 就重抽 $Q$"，语义是：**每个子空间最多用一轮，一轮之内最多花 $\mathcal{O}(q\log q)$ 步把几何修好，然后整个丢掉**。

作者在第 1016 行明确承认这个设计的代价："This initialization choice is somewhat arbitrary"、"Our computational results show that this is quite expensive"，并提到可以把 $\mathcal{V}_0$ 缩到 1 个点、$\mathcal{Z}_0$ 清空（技术上简单但实践未知）。**这是一处理论与实现互相让步的诚实交代**：算法 4 的分析形式最干净，但它是全节里实践上最贵的方案。

**推论 5.14 的计数（把第 1068 行的一句话展开成四项）**：设 $\Lambda=1+\frac1q$，$\Delta_{min}=\sqrt{\epsilon_f}$，$|\mathcal{Z}_0|=q$。

- **子空间个数**：定理 5.12 的 $\mathbb{E}[T_\epsilon]=\mathcal{O}(\frac{n}{\epsilon^2})$（$\kappa_{eg},\kappa_{ef}=\mathcal{O}(\sqrt q(L+\frac{\epsilon_f}{\Delta_{min}^2}))$ 由定理 3.5 + $\Lambda=1+\frac1q$ 给出，与有限差分同阶，故 $\hat C_1^{-1}=\Theta(\sqrt n(L+\cdot))$ 也同阶）。
- **每个子空间的初始化代价**：$\mathcal{O}(q)$ 次新 oracle（算 $f(x_k+Q_ky_i)$ 与 $f(x_k+Q_kz_i)$）。
- **每个子空间的几何校正步数**：定理 4.1 把 $n$ 换成 $q$、$\Lambda=1+\frac1q$：$2q\log q+4q+q\vert\log\log(1+\frac1q)\vert$，而 $\log(1+\frac1q)\approx\frac1q$、$\log\log(1+\frac1q)\approx-\log q$，故 $=\mathcal{O}(q\log q)$。每步至多 2 次 oracle ⟹ $\mathcal{O}(q\log q)$ 次。
- **合计**：$\mathcal{O}(\frac{n}{\epsilon^2})\times\big[\mathcal{O}(q)+\mathcal{O}(q\log q)\big]=\mathcal{O}(\frac{nq\log q}{\epsilon^2})$ ✓。

**审计结论**：与算法 3 的 $\mathcal{O}(\frac{nq}{\epsilon^2})$ 相比，**几何校正版本在阶上更差（多一个 $\log q$）**。这与第 4 节的结论完全同构（推论 4.4 的 $n^{3/2}\log n$ 差于推论 2.10 的 $n^{3/2}$）。**全文的一致性结论**：本文的插值/几何校正机制在**最坏情形 oracle 计数**上从未赢过有限差分；它赢的地方是（a）每步的**计算**代价（解 $q\times q$ 或 $n\times n$ 的小问题，而不是 $n+1$ 次求值——但求值才是 DFO 的代价度量，所以这不算赢），（b）**常数**（$\kappa_{eg}$ 的 $\Lambda$-依赖在 $\Lambda$ 小时优于有限差分的粗界），(c) 实践性能（§6 的数据）。**作者在第 1072 行与第 1248 行的措辞（"mainly theoretical"）与这个结论是一致的，没有夸大**。

## G12 第 5 节审计

- **条件**：假设 1.2（$L$-光滑、下有界）、假设 2.2（fcd + 有界模型 Hessian）、假设 5.7（$\mathbb{P}\{I_t=1\mid\mathcal{F}_{t-1}\}\ge\theta>\frac12$）、外加三条**可行性约束** $\Delta_0\ge\hat C_1\epsilon$、$\Delta_{min}\le\gamma\hat C_1\epsilon$、$\epsilon$ 大于噪声下限。
- **结论**：$\mathbb{E}[T_\epsilon]\le\frac{4\theta}{(2\theta-1)^2}(\cdot)$；$\mathbb{E}[\mathcal{C}_\epsilon]\le\mathcal{O}(\frac{nq}{\epsilon^2})$（有限差分）/ $\mathcal{O}(\frac{nq\log q}{\epsilon^2})$（几何校正）；精度下限 $\epsilon\ge\Omega(\sqrt{n\epsilon_f})$。
- **相对最强近邻（[11] 的子空间分析、[8] 的 JL 版本、[14] 的随机噪声版本）放宽了什么**：第一次在**随机子空间 + 确定性固定精度噪声**下给出期望复杂度。技术上放宽的是"噪声可任意压缩"这一 [14] 式假设。
- **加强了什么**：需要**已知** $\epsilon_f$ 的上界（且算法里显式使用它）；需要额外的 $\Delta_{min}$ 超参数及其与 $\epsilon$ 的耦合条件。
- **隐藏依赖**：$\theta$ 与 $\kappa_g$ 全部来自 [11, Lem 6.7]（需 $q\ge3$，定理 5.13 明写）；$\hat C_1$ 通过 $\kappa_{ef}$ 依赖引理 5.2 的 $L_Q\le L$；$\epsilon_f$ 的**高估**会以两条路径恶化结果：$\Delta_{min}$ 要跟着放大（$\propto\sqrt{\epsilon_f^{used}}$），$\epsilon$ 的下界 $\propto\sqrt{\epsilon_f^{used}}$。所以"保守取 $\epsilon_f$"直接线性地抬高可达精度，论文第 756 行有此提醒但没给量化。
- **非空泛区域**：$\epsilon_f=0$ 时 $\Delta_{min}$ 可取 $0$，$-4\epsilon_f$ 与 $+2\epsilon_f$ 全部消失，定理 5.12/5.13 退化为 [11] 的 $\mathcal{O}(n\epsilon^{-2})$ ✓（一致性检验通过）。$q=n$ 时 $\kappa_g=0$、$\hat C_1=\tilde C_1$、每步 oracle 变 $n$，应回到第 2 节的界——但注意**此时 $A_t$ 的"重抽 $Q$"没有意义**，且定理 5.13 的迭代数仍是 $n$（因为 $\eta_2$ 取常数），与推论 2.10 的 $\sqrt n$ 差一个 $\sqrt n$，差别**完全**来自 $\eta_2$ 的取法（G10 已解释）。
- **本节最可能失效处（按可信度排序）**：
  1. **引理 5.11 的 $\frac{1}{2\theta}$**：如 G8 所析，地板 $\Delta_{min}$ 使 [10] 的阶梯论证在小半径侧需要额外论证，我判断结论应补一个加性 $\log_\gamma\frac{\Delta_{min}}{\hat C_1\epsilon}$ 项。非致命（不改阶），但**是本文自己的新机制与继承结论之间未被验证的接缝**，是最值得先查的一处。
  2. **引理 5.5 的因子 2**：$\kappa_{ef}\Delta_k^2$ 应带 $2$，导致 $\tilde C_1$ 里 $\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$ 应为 $\frac{4\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$。非致命。
  3. **引理 5.10 右边漏 $\mathbb{E}$**：排版级，非致命。
  4. **$\Delta_{min}\le\gamma\hat C_1\epsilon$ 与 $\epsilon$ 下限的循环依赖**：$\hat C_1$ 依赖 $\kappa_{ef},\kappa_{eg}$，后两者依赖 $\Delta_{min}$，而 $\Delta_{min}$ 又要 $\le\gamma\hat C_1\epsilon$。代入 $\Delta_{min}=\sqrt{\epsilon_f}$ 后两条约束化为 $\epsilon\ge\Omega(\sqrt n(L+\cdot)\sqrt{\epsilon_f})$，**自洽**（G10 第 6 步已验证），不构成循环。此项排除。
  5. **终止性**：定理 5.12 界的是 $\mathbb{E}[T_\epsilon]$，但 $\Vert\nabla\phi(x_t)\Vert\le\epsilon$ 在零阶设定下**不可检验**，算法 3/4 都没有给出可用的停机判据。这是整个第 2–5 节共同的缺口（第 2 节的 $K_\epsilon$ 同样不可检验），§6 用 $\sigma_{end}=10^{-12}$ 代替。
- **足以支撑作者主张吗**：主张是"把 [11] 的子空间分析扩展到确定性噪声"。**成立**：$\mathcal{O}(nq\epsilon^{-2})$ 与噪声下界 $\Omega(\sqrt{n\epsilon_f})$ 都是新的、且证明骨架完整（除上面 1、2 两处可修复的漏洞）。但主张的**分量**有限：噪声只改变了精度下限和一个加性记账项，没有改变 $n,q,\epsilon$ 的任何指数。
- **可证伪的后续问题**：把 $\hat C_1$ 的门槛改成"分层判据"——在 $I_t=1$ 时用 $\tilde C_1\Vert Q_tQ_t^\top\nabla\phi\Vert$、在 $I_t=0$ 时不要求收缩——能否把 $\hat C_1^{-1}$ 里的 $\sqrt{n/q}$ 放大去掉，从而在子空间情形也允许 $\eta_2=\sqrt n$、把迭代数从 $n$ 降到 $\sqrt n$？判别实验：在合成二次函数上跑 $n\in\{100,400\}$、$q=5$，比较 $\eta_2\in\{1,\sqrt n\}$ 两组的**实际迭代数**（不是 oracle 数）。若 $\eta_2=\sqrt n$ 组迭代数按 $\sqrt n$ 而非 $n$ 增长，则本节的"抵消论证"（G10 第 4 步）被证伪，界可改进。

---

# 阶段 H：第 6 节——实现（GC-YZ-LIN / GC-YZ-V / GC-sub）与数值结果

## H0 第 6 节要立的主张，以及"阶保持不变"这句话的确切含义

作者在第 1072 行的主张是一句**弱主张**：Algorithm 5 在 Algorithm 2 之上加了若干实用部件（分辨率下限、两道 gate、hedge 模型、$K$ 截断），"这些附加特征改善实践性能但使分析更繁琐，**然而最终最坏情形复杂度的阶是保持的**"。所以第 6 节的理论部分只需要检查一件事：**每个新增部件要么不增加 oracle 计数，要么只增加对数因子**。下面逐个验证，并指出论文没验证到的地方。

三条与实现有关、但论文没有明说的对照事实（我从 PDF 第 30–32 页的参考文献表与第 1196 行的实验设定核对得到）：
- self-correction 的思想不是本文的，出处是 [26]（Scheinberg & Toint 2010, "Self-correcting geometry in model-based algorithms for DFO"）；本文的贡献是**把它和 $\mathcal{Y}$–$\mathcal{Z}$ 分离、线性 Lagrange 多项式、以及复杂度分析**绑在一起。
- NEWUOA [21] 用**二次** Lagrange 多项式管理整个插值集的几何；本文的 GC-YZ-LIN 只用**线性** Lagrange 多项式管理 $\mathcal{Y}$ 的子集。第 1082 行把 GC-YZ-LIN 定位成"DFOTR 与 NEWUOA 之间的中间地带"，这个定位是准确的。
- 实验"所有信赖域子问题精确求解"（第 1196 行）——这一点在下面 H3 里是决定性的，它让假设 6.1 以 $\kappa_{step}=1$ 成立。

## H1 分辨率下限 $\sigma_k$：与 $\Delta_{min}$ 的本质差别

**定义（第 1086 行）**：$\sigma_k$ 是 $\Delta_k$ 的**自适应**下界。规则：当 $\Delta_k=\sigma_k$、几何已经好（$\vert\ell_{j^*}(s^*)\vert\le\Lambda$）、且迭代仍不成功时，令 $\sigma_{k+1}=\theta\sigma_k$（Algorithm 5 输入里 $\theta\in(0,\gamma)$，Table 6.1 取 $0.1$）。在此之前 $\Delta_k$ 不得越过 $\sigma_k$ 下降。初始化 $\sigma_0=\Delta_0$；终止时 $\sigma_{k+1}=\max\{\theta\sigma_k,\sigma_{end}\}$，$\sigma_{end}=10^{-12}$。

**与算法 3 的 $\Delta_{min}$ 对比**：

| | $\Delta_{min}$（算法 3） | $\sigma_k$（算法 5） |
| --- | --- | --- |
| 是否随时间变 | 固定常数 | 单调不增，可降到 $\sigma_{end}$ |
| 作用 | 保证 $\kappa_{eg}$ 有限（噪声分析必需） | 允许半径在"几何已经好但还没成功"时继续缩 |
| 动机 | 理论 | 实践：高维下"每次收缩前都先修好几何"太贵，改成**只在若干离散分辨率上修几何** |

**复杂度代价（我算出来，论文只说"some logarithmic factors"）**：$\sigma$ 的取值层数是

$$
\Big\lceil\log_{1/\theta}\frac{\Delta_0}{\sigma_{end}}\Big\rceil+1.
$$

代入 Table 6.1：$\theta=0.1$，$\Delta_0=0.5$，$\sigma_{end}=10^{-12}$，$\frac{\Delta_0}{\sigma_{end}}=5\times10^{11}$，$\log_{10}=11.7$ ⟹ **13 层**。每一层内部可以套用第 2 节的整套论证（在该层上把 $\Delta_k$ 的下界换成 $\sigma$），所以总界变成

$$
\mathcal{C}_\epsilon\le\mathcal{O}\Big(\log_{1/\theta}\frac{\Delta_0}{\sigma_{end}}\cdot\frac{n^{3/2}\log n}{\epsilon^2}\Big)=\mathcal{O}\Big(\frac{n^{3/2}\log n}{\epsilon^2}\Big)
$$

（因为 $\log_{1/\theta}\frac{\Delta_0}{\sigma_{end}}$ 与 $n$ 无关，是绝对常数 13）。**所以"阶保持"成立**，且代价是一个可量化的 13 倍。论文没给这个数，但它对评价实现有实际意义：**$\sigma_{end}=10^{-12}$ 与 $\theta=0.1$ 直接决定"最多多花 13 倍"**。

## H2 两道 gate：省 oracle 的地方，也是唯一"改变分支逻辑"的地方

**小模型梯度 gate（Line 7）**：若 $\Vert g_k\Vert<\eta_2\Delta_k$，**不去求值** $f(x_k+s_k)$，而是直接做几何检查并可能收缩半径。省一次 oracle。
**小步 gate（Line 9）**：若 $\Vert s_k\Vert<c_g\sigma_k$，同样跳过求值。省一次 oracle。

两者的理论依据相同：**在这些条件下，即使去求值，迭代也必然被判为不成功**，所以跳过不改变 $(x_k,\Delta_k)$ 的演化轨迹（只要分支逻辑与"求值后 $\rho_k<\eta_1$"的分支一致）。Line 7 的三个子分支：

1. $\vert\ell_{j_k^*}(s_k^*)\vert>\Lambda$：做"几何校正 $\mathcal{Y}$（替换坏点）"——与 Algorithm 2 的第 (iii) 步同一动作；
2. $\vert\ell_{j_k^*}(s_k^*)\vert\le\Lambda$ 且 $\Delta_k\le\sigma_k$：做**分辨率更新** $\sigma_{k+1}=\max\{\theta\sigma_k,\sigma_{end}\}$，$\Delta_{k+1}=\max\{\frac12\Delta_k,\sigma_{k+1}\}$；
3. $\vert\ell_{j_k^*}(s_k^*)\vert\le\Lambda$ 且 $\Delta_k>\sigma_k$：$\Delta_{k+1}\leftarrow\max\{\gamma\Delta_k,\sigma_k\}$。

**两处要指出的不一致**：
- 子分支 2 用 $\frac12$ 收缩，子分支 3 用 $\gamma$（Table 6.1 里 $\gamma_{dec}=0.8$）。**同一算法里两个不同的收缩因子**。对分析无影响（任何固定 $<1$ 的因子都只改 $\log_\gamma$ 的底），但引理 2.5/2.7/5.9 那类"阶梯"论证必须统一按 $\min\{\gamma,\frac12\}=\frac12$ 来写，论文没有做这个交代。
- 子分支 2 里 $\Delta_k\le\sigma_k$ 与 $\Delta_{k+1}=\max\{\frac12\Delta_k,\sigma_{k+1}\}$：因为 $\sigma_{k+1}=\theta\sigma_k\le0.1\Delta_k<\frac12\Delta_k$（$\theta=0.1<\frac12$），$\max$ 实际总取 $\frac12\Delta_k$。**即 $\sigma_{k+1}$ 的 $\max$ 保护在 Table 6.1 的参数下从不生效**，$\Delta$ 与 $\sigma$ 会保持 $\Delta>\sigma$ 的间隙。这解释了为什么需要 $\theta\in(0,\gamma)$ 这个输入约束——它保证一次分辨率下降后半径不会立刻又贴住地板。

## H3 假设 6.1 与 $\kappa_{step}$：它其实不是假设，而是引理

**假设 6.1（第 1094 行）**：对每个 $k$，

$$
\Vert s_k\Vert\ge\kappa_{step}\min\Big\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\Big\}.
$$

符号：$\kappa_{step}>0$ 是**新增常数**（待定量）；$\min\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\}$ 正是 fcd 条件 (2.2) 里出现的那个量。

论文第 1098 行给的两个值："精确求解 ⟹ $\kappa_{step}=1$；Cauchy 步 ⟹ $\kappa_{step}=\frac{\kappa_{fcd}}{3}$"。下面分别验证，并给出一个更强的结论。

**验证 1（精确解 ⟹ $\kappa_{step}=1$）**：精确解满足 $\Vert s_k\Vert\le\Delta_k$（可行性），且 $m_k(x_k)-m_k(x_k+s_k)\ge m_k(x_k)-m_k(x_k+s_k^{Cauchy})$（最优性，Cauchy 点是可行候选）。对精确解，标准论证给出 $m_k(x_k)-m_k(x_k+s_k)\ge\frac{\Vert g_k\Vert}{2}\min\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\}$（即 (2.2) 对 $\kappa_{fcd}=\frac12$ 成立）。但**这不能直接给出 $\Vert s_k\Vert$ 的下界**——需要另走一条路：若 $\Vert s_k\Vert<\Delta_k$（内点解），则 $H_k$ 半正定且 $s_k=-H_k^{-1}g_k$（当 $H_k$ 正定），于是 $\Vert g_k\Vert=\Vert H_ks_k\Vert\le\kappa_{bhm}\Vert s_k\Vert$，即 $\Vert s_k\Vert\ge\frac{\Vert g_k\Vert}{\kappa_{bhm}}$；若 $\Vert s_k\Vert=\Delta_k$ 则 $\Vert s_k\Vert=\Delta_k$。两支取 $\min$ 即 $\Vert s_k\Vert\ge\min\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\}$ ⟹ $\kappa_{step}=1$ ✓。（$H_k$ 半定奇异时 $s_k$ 沿负曲率方向走到边界 $\Vert s_k\Vert=\Delta_k$，同样成立。）**论文这一步没写，我补上。**

**验证 2（Cauchy 点 ⟹ $\kappa_{step}=1$，比论文的 $\frac{\kappa_{fcd}}{3}$ 更好）**：Cauchy 点 $s_k^c=-t_kg_k$，$t_k=\min\{\frac{\Vert g_k\Vert^2}{g_k^\top H_kg_k},\frac{\Delta_k}{\Vert g_k\Vert}\}$（当 $g_k^\top H_kg_k>0$；否则取 $t_k=\frac{\Delta_k}{\Vert g_k\Vert}$）。于是

$$
\Vert s_k^c\Vert=t_k\Vert g_k\Vert=\begin{cases}\Delta_k,&g_k^\top H_kg_k\le0\ \text{或}\ \frac{\Vert g_k\Vert^3}{g_k^\top H_kg_k}\ge\Delta_k,\\[4pt]\dfrac{\Vert g_k\Vert^3}{g_k^\top H_kg_k}\ge\dfrac{\Vert g_k\Vert^3}{\kappa_{bhm}\Vert g_k\Vert^2}=\dfrac{\Vert g_k\Vert}{\kappa_{bhm}},&\text{否则}.\end{cases}
$$

（第二行用 $g_k^\top H_kg_k\le\Vert H_k\Vert\Vert g_k\Vert^2\le\kappa_{bhm}\Vert g_k\Vert^2$。）两支合起来正是 $\Vert s_k^c\Vert\ge\min\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\}$ ⟹ **$\kappa_{step}=1$ 对 Cauchy 点也成立**。

**验证 3（只有 (2.2) 时 ⟹ $\kappa_{step}=\frac{2\kappa_{fcd}}{3}$，这解释了论文为什么会写出 $\frac{\kappa_{fcd}}{3}$）**：论文第 1098 行的"Cauchy step"我理解是指"任何满足 fcd 条件 (2.2) 的步"（而不是字面的 Cauchy 点），因为论文在假设 2.2 的框架下工作，那里只假定了 (2.2)。此时：

$$
m_k(x_k)-m_k(x_k+s_k)=-g_k^\top s_k-\frac{1}{2}s_k^\top H_ks_k\le\Vert g_k\Vert\Vert s_k\Vert+\frac{\kappa_{bhm}}{2}\Vert s_k\Vert^2.
$$

与 (2.2) 联立，记 $\mu=\min\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\}$：

$$
\kappa_{fcd}\Vert g_k\Vert\mu\le\Vert g_k\Vert\Vert s_k\Vert+\frac{\kappa_{bhm}}{2}\Vert s_k\Vert^2.
$$

分两支。**支 A**：$\mu=\Delta_k$（即 $\Delta_k\le\frac{\Vert g_k\Vert}{\kappa_{bhm}}$）。用 $\Vert s_k\Vert\le\Delta_k$ 得 $\frac{\kappa_{bhm}}{2}\Vert s_k\Vert^2\le\frac{\kappa_{bhm}}{2}\Delta_k\Vert s_k\Vert\le\frac{\Vert g_k\Vert}{2}\Vert s_k\Vert$，于是 $\kappa_{fcd}\Vert g_k\Vert\Delta_k\le\frac32\Vert g_k\Vert\Vert s_k\Vert$ ⟹ $\Vert s_k\Vert\ge\frac{2\kappa_{fcd}}{3}\Delta_k=\frac{2\kappa_{fcd}}{3}\mu$。
**支 B**：$\mu=\frac{\Vert g_k\Vert}{\kappa_{bhm}}$。令 $a=\Vert s_k\Vert$，解二次不等式 $\frac{\kappa_{bhm}}{2}a^2+\Vert g_k\Vert a-\frac{\kappa_{fcd}\Vert g_k\Vert^2}{\kappa_{bhm}}\ge0$，取正根：

$$
a\ge\frac{\Vert g_k\Vert}{\kappa_{bhm}}\Big(\sqrt{1+2\kappa_{fcd}}-1\Big)=\mu\big(\sqrt{1+2\kappa_{fcd}}-1\big).
$$

对 $\kappa_{fcd}\in(0,1]$ 逐点数：$\kappa_{fcd}=0.1$：$\sqrt{1.2}-1=0.0954>\frac{2\cdot0.1}{3}=0.0667$；$\kappa_{fcd}=0.5$：$0.414>0.333$；$\kappa_{fcd}=1$：$0.732>0.667$。**支 B 总不支配，所以**

$$
\boxed{\text{假设 2.2}+\text{可行性}\Rightarrow\text{假设 6.1 以 }\kappa_{step}=\frac{2\kappa_{fcd}}{3}\text{ 成立}}
$$

（定稿里这条不用 boxed，写成正文。）**结论：假设 6.1 是假设 2.2 的推论，不是独立假设**；论文写的 $\frac{\kappa_{fcd}}{3}$ 是 $\frac{2\kappa_{fcd}}{3}$ 的一半，保守但正确。**把它列为"Assumption"是呈现上的弱点**——它让读者以为需要额外的算法条件，实际上任何满足 fcd 的步都自动满足。

**gate 相容性条件 $c_g\le\kappa_{step}$ 的实际含义（论文第 1098 行一句话，我展开）**：论文第 1100–1104 行的链条是

$$
\kappa_{bhm}\Delta_k\le\max\{\kappa_{bhm},\eta_2\}\Delta_k\le\Vert g_k\Vert\Rightarrow\min\Big\{\Delta_k,\frac{\Vert g_k\Vert}{\kappa_{bhm}}\Big\}=\Delta_k,
$$

（第一个不等号因为 $\max\ge$ 每个分量；第二个不等号是引理 2.3 里"$\Delta_k$ 足够小 ⟹ 门槛成立"的结论。）于是 $\mu=\Delta_k$，假设 6.1 给 $\Vert s_k\Vert\ge\kappa_{step}\Delta_k$。又 $\Delta_k\ge\sigma_k$ 恒成立（$\sigma$ 只在 $\Delta\le\sigma$ 时下降，且 $\Delta_{k+1}\ge\sigma_{k+1}$），所以

$$
\Vert s_k\Vert\ge\kappa_{step}\Delta_k\ge\kappa_{step}\sigma_k\ge c_g\sigma_k,
$$

即**小步 gate 永远不会在"本该成功"的那些步上误触发** ✓。**但关键数值检查**：Table 6.1 取 $c_g=0.5$。
- 若用精确解（$\kappa_{step}=1$）：$1\ge0.5$ ✓ 相容——这正是第 1196 行"所有信赖域子问题精确求解"所保证的。
- 若只用 fcd 保证（$\kappa_{step}=\frac{2\kappa_{fcd}}{3}$）：需要 $\kappa_{fcd}\ge0.75$。而标准 Cauchy 点给出的 $\kappa_{fcd}$ 通常 $\le\frac12$ ⟹ $\kappa_{step}\le\frac13<0.5=c_g$，**gate 会破坏理论**。
- 若按论文写的 $\kappa_{step}=\frac{\kappa_{fcd}}{3}$：需要 $\kappa_{fcd}\ge1.5$，而 $\kappa_{fcd}\le1$ 恒成立（fcd 常数不会超过 1，因为 (2.2) 左边 $\le$ 模型最大下降），**则该条件在任何实现下都不可能满足**。

**这是一处实质性的表述错误**：论文写的 $\kappa_{step}=\frac{\kappa_{fcd}}{3}$ 与 $c_g\le\kappa_{step}$ 联立后要求 $\kappa_{fcd}\ge3c_g=1.5$，**不可能成立**；只有改用我推的 $\frac{2\kappa_{fcd}}{3}$（要求 $\kappa_{fcd}\ge0.75$，仍偏严）或精确解的 $\kappa_{step}=1$（要求 $c_g\le1$ ✓ 与 Table 6.1 一致）才自洽。**正确读法**：因为实验用精确解，$\kappa_{step}=1$，$c_g=0.5\le1$ ✓，所以实现是合法的；但论文正文里"$\kappa_{fcd}/3$"与"$c_g\le\kappa_{step}$"两句放在一起会让读者得出"实现不满足假设"的错误结论。**非致命（不影响正确性），但是必须修正的表述。**

## H4 算法 5 逐行（GC-YZ-LIN）

**输入**：零阶 oracle $f(x)\approx\phi(x)$；$\Delta_0,x_0$；$\gamma\in(0,1)$；$\eta_1>0,\eta_2>0$；$\Lambda>1$；$\Lambda_{sc}\ge1$；$\theta\in(0,\gamma)$；$\sigma_{end}\in(0,\Delta_0]$；$c_g\in(0,1]$；模型 Hessian 上界 $K$；平均权重 $\beta\in(0,1]$。

**符号冲突警告**：这里的 $\theta$ 是**分辨率收缩因子**，与第 5 节假设 5.7 里的对齐概率 $\theta$ **同名不同义**。定稿里我把第 6 节的改称 $\theta_\sigma$ 并注明。

**初始化**：$|\mathcal{Y}_0|=n$（对线性多项式 poised），$|\mathcal{Z}_0|\le\frac{n(n+1)}{2}$；求值 $f(x_0)$、$f(x_0+y_i)$、$f(x_0+z_i)$；$\sigma_0=\Delta_0$；线性 Lagrange 多项式 $\{\ell_i\}$；模型误差 $e_C,e_F\leftarrow$ undefined；$H_{prev}\leftarrow0$；活动模型标签 $\alpha\leftarrow F$。

**Line 1（建模）**：同时构造 MFN 模型 $(g_F,H_F)$ 与 MCFN 模型 $(g_C,H_C)$，然后 $H_{prev}=H_C$（为下一轮的 MCFN 做准备）。下标 $F$ = "Fresh"（最小 Frobenius 范数，$H_{prev}=0$），$C$ = "Change"（相对上一轮改动最小）。

**Line 2–4（模型切换）**：若 $e_C$ 或 $e_F$ 未定义则用 $F$；否则带 **0.8 滞回**的切换：$\alpha=F$ 且 $e_C<0.8e_F$ ⟹ 切到 $C$；$\alpha=C$ 且 $e_F<0.8e_C$ ⟹ 切回 $F$。$0.8$ 的作用是**迟滞（hysteresis）**：只有当备选模型明显更好（误差低 $20\%$ 以上）才切换，避免在两个模型间来回抖动。Line 4 按标签取 $(g_k,H_k)$。

**Line 5（Hessian 有界化）**：若 $\Vert H_k\Vert>K$ 则改用 (4.1) 的约束最小二乘解。Table 6.1 取 $K=10^{100}$ ⟹ **几乎永不触发**（作者的理由：CUTEst 里 SSBRYND/SCURLY10 的真 Hessian 范数 $>10^{20}$，POWERSUM 可超 $10^{100}$，小 $K$ 会截掉有用曲率）。**代价见 H6。**

**Line 6（几何度量）**：$(j_k^*,s_k^*)=\arg\max_{j=1,\dots,n,\ s\in B(0,\Delta_k)}\vert\ell_j(s)\vert$。由阶段 E1 的闭式解，$\max_{\Vert s\Vert\le\Delta_k}\vert\ell_j(s)\vert=\Delta_k\Vert(Y^{-\top})_j\Vert$ 且在球面取到，所以这一步是 $n$ 次范数计算，**不需要优化**。（论文正文写"$s\in B(0,\Delta_k)$"是对的；Markdown 转换里 (4.4) 那处 $\arg\max_{j=1,\dots,p}$ 的下标错误在阶段 F 已记。）

**Line 7（小模型梯度 gate）**、**Line 9（小步 gate）**：见 H2。
**Line 8**：按算法 1 求 trial step。**Line 10**：求值并算 $\rho_k$（按算法 1 的标准定义，**没有** $+2\epsilon_f$——因为第 6 节处理的是无噪声的 CUTEst 问题）。
**Line 11（模型打分，EWMA）**：对 $\mu\in\{F,C\}$，

$$
p_\mu\leftarrow(g_\mu)^\top s_k+\frac{1}{2}s_k^\top H_\mu s_k,\qquad f_c\leftarrow f(x_k+s_k)-f(x_k),
$$

$$
\varepsilon_\mu\leftarrow\frac{\vert p_\mu-f_c\vert}{\max\{\vert f_c\vert,\vert p_\mu\vert\}},\qquad e_\mu\leftarrow\begin{cases}\varepsilon_\mu,&e_\mu\text{ 未定义},\\ \beta e_\mu+(1-\beta)\varepsilon_\mu,&\text{否则}.\end{cases}
$$

符号：$p_\mu$ 是模型 $\mu$ 预测的下降量（取负号约定：$m_k(x_k)-m_k(x_k+s_k)=-\big(g^\top s+\frac12 s^\top Hs\big)$，所以论文这里 $p_\mu$ 与 $f_c$ 的符号约定是"预测增量 vs 实测增量"，两者都是"增加量"，一致 ✓）；$f_c$ 是实测增量；$\varepsilon_\mu\in[0,\infty)$ 是**相对误差**（分母用 $\max$ 防止除零并给出尺度不变的相对量）；$e_\mu$ 是指数加权滑动平均，$\beta=0.8$（Table 6.1）⟹ 有效记忆长度 $\frac{1}{1-\beta}=5$ 步。

**Markdown 转换错误（已由 PDF 确认）**：转换后的正文写成"otherwise $e_\mu=\beta e_\mu+(1-\beta)e_\mu$"，右边 $=\beta e_\mu+(1-\beta)e_\mu=e_\mu$，**恒等式，等于什么都没做**。PDF 第 26 页确认应为 $(1-\beta)\varepsilon_\mu$。这是转换丢字符，不是论文错误。

**Line 12–13**：成功 ⟹ 按算法 2 的成功更新（含 (4.4) 的 Lagrange 秩一更新与 $\mathcal{Y}$ 的 $(\mathcal{Y}\setminus\{y_{j^*}\}\cup\{0\})-s_k$ 式更新）；不成功/模型改进 ⟹ 执行算法 2 中所有适用的 (i)(ii)(iii)(iv)，并按 $I_{imp}$ 与 $\Delta_k$ vs $\sigma_k$ 的三种组合决定 $\Delta_{k+1}$（模型改进步：$\max\{\gamma\Delta_k,\sigma_k\}$；地板以上失败：同上；地板处失败：走 H1 的分辨率更新）。

**与算法 2 的差异（一张表）**：

| 部件 | 算法 2 | 算法 5 | 对 oracle 计数的影响 |
| --- | --- | --- | --- |
| 建模 | (4.1) 约束最小二乘 | MFN/MCFN 二选一（hedge），仅当 $\Vert H\Vert>K$ 才回落到 (4.1) | 无（都不求值） |
| 半径下界 | 无 | $\sigma_k$ 自适应 | 乘 $\approx13$ 层（H1） |
| 求值前拦截 | 无 | 两道 gate | **减少**求值 |
| 收缩因子 | $\gamma$ | $\gamma$ 与 $\frac12$ 混用 | 无（只改对数底） |
| $\rho_k$ | 标准 | 标准（无噪声） | — |
| self-correction 判据 | $\Lambda_{sc}$ | 同 | 无 |

## H5 GC-YZ-V 的打分规则，以及它为什么不在理论覆盖内

**第 1119 行**：

$$
\mathrm{score}(y_i)=\max\Big(1,\frac{\Vert y_i\Vert^2}{\max(0.1\Delta_k,\sigma_k)^2}\Big)^{3}\cdot\ell_i(s_k)^{2},
$$

替换掉 score 最大的点 $y_i$ 为 $s_k$。逐个符号：$\Vert y_i\Vert$ 是样本点相对信赖域中心的距离（$\mathcal{Y}$ 住在 $B(0,\Delta_k)$ 的尺度里）；分母 $\max(0.1\Delta_k,\sigma_k)$ 是"当前有效分辨率尺度"（取 $\Delta_k$ 的十分之一与地板中较大者，防止 $\Delta_k$ 很小时把近处的点也判成"远"）；立方 $3$ 是**经验指数**（作者未给理由）；$\ell_i(s_k)^2$ 是该点线性 Lagrange 多项式在被拒步处的取值平方——**这一项才是理论上重要的**（阶段 F 的引理 4.3/定理 4.1 全靠 $\vert\ell_j(s_k)\vert$ 控制 $\det Y$ 的收缩）。

**为什么不在理论内**：定理 4.1 与引理 4.2 的分析要求 $\mathcal{Y}$ 始终是**线性** poised 集，且 self-correction 用线性 $\ell_j$。GC-YZ-V 的 $\ell_i$ 是对 $\mathcal{Y}\cup\mathcal{Z}$ 的**二次** Lagrange 多项式（NEWUOA 式），此时 $\vert\ell_i(s_k)\vert$ 大**不再**蕴含 $\det Y$ 的改进，引理 4.2 的 AM-GM 链条断裂。作者在第 1122 行明确写"not currently accompanied by any theoretical guarantees"，第 1244 行再次声明。**这是全文最干净的一处"理论/实践分离"声明。**

**GC-sub**：算法 4 的直接实现 + 一个"patience"参数（连续成功/失败多少次才重抽子空间），理论对应 patience $=1$。作者说实验上 $1$ 最好，且明说"our contribution in terms of subspace TR method in this paper is mainly theoretical"（第 1248 行）。

## H6 Table 6.1 的三个参数与定理条件的正面冲突（量化）

Table 6.1（PDF 第 28 页）：$\eta_1=0.01$、$\eta_2=5\times10^{-9}$、$\gamma_{inc}=1.3$、$\gamma_{dec}=0.8$、$\theta_\sigma=0.1$、$\Lambda=1000$、$\Lambda_{sc}=2$、$\beta=0.8$、$K=10^{100}$、$c_g=0.5$；GC-sub：$\eta_1=0.1$、$q=5$、容量 $2q$。

定理要求 $\eta_2=\sqrt n$（推论 2.10/4.4）与 $\Lambda=1+\frac1n$；实现取 $\eta_2=5\times10^{-9}$、$\Lambda=1000$。**把实现的参数代进定理的界，会得到什么？** 以 $n=100$、POWERSUM 类问题（$\kappa_{bhm}\sim10^{100}$）为例：

$$
C_2=\frac{\eta_1\eta_2\kappa_{fcd}}{2}\min\Big\{\frac{\eta_2}{\kappa_{bhm}},1\Big\}=\frac{0.01\times5\times10^{-9}\times\kappa_{fcd}}{2}\times\big(5\times10^{-109}\big)\approx1.25\times10^{-119}\,\kappa_{fcd},
$$

$$
C_1^{-1}\ge\kappa_{bhm}=10^{100}\Rightarrow(\gamma C_1\epsilon)^2=\Big(\frac{0.8\epsilon}{10^{100}}\Big)^{2}=6.4\times10^{-213}\ (\epsilon=10^{-6}),
$$

$$
|\mathcal{S}_\epsilon|\le\frac{2(\phi(x_0)-\phi^\star)}{C_2(\gamma C_1\epsilon)^2}\approx\frac{2\times10^{2}}{1.25\times10^{-119}\times6.4\times10^{-213}}\approx2.5\times10^{333}.
$$

**结论：在论文自己给出的实验参数下，第 2–4 节的所有界都是空泛的（vacuous）**——不是"松"，而是给出的数字远大于任何可枚举量。这本身**不是错误**（最坏情形界普遍如此，作者在第 1200 行也承认"our analysis addresses the worst-case"），但它有一个可操作的推论：**§6 的实验结果与定理之间没有任何定量联系**，因此（a）实验不能算作对定理的验证；（b）$\eta_2$、$\Lambda$、$K$ 的"实践最优值"完全由经验决定，与理论的 $\eta_2=\sqrt n$、$\Lambda=1+\frac1n$ 建议**方向相反**。

**$\kappa_{bhm}$ 的角色被 $K=10^{100}$ 掏空**：定理里 $\kappa_{bhm}$ 是假设 2.2 的输入（有界模型 Hessian），$C_1,C_2,\kappa_{ef}$ 全都依赖它。实现里 $K$ 大到几乎不触发 ⟹ 模型 Hessian 由 MFN/MCFN 决定 ⟹ $\kappa_{bhm}$ 等于**数据里出现过的最大 MCFN Hessian 范数**，一个事先未知的量。作者第 1114 行的辩护是"large $\kappa_{bhm}$ 只在 $L$ 同样大时出现，所以不影响阶"——**这个辩护只在把 $L$ 与 $\kappa_{bhm}$ 视为同阶时才成立**，而复杂度界对 $\kappa_{bhm}$ 的依赖是 $C_2\propto\min\{\eta_2/\kappa_{bhm},1\}$，即 $\kappa_{bhm}$ 大时迭代数**线性**恶化，与 $L$ 无关。所以更准确的说法是：**$\kappa_{bhm}$ 大 ⟹ 界恶化，只是实践中 $\eta_2$ 取得很小所以从不触发 $\min$ 的第二支**。

## H7 数值结果：论文的四条观察与我能核对的部分

**AUC 数据（我从 PDF 第 28–31 页的图例逐个读出，$\tau$ 依次为 $10^{-2},10^{-4},10^{-6}$）**：

| 测试集 | GC-YZ-LIN | DFOTR | NEWUOA | GC-YZ-V |
| --- | --- | --- | --- | --- |
| $2\le n\le5$ | 0.874 / 0.780 / 0.712 | 0.855 / 0.774 / 0.714 | 0.872 / 0.771 / 0.701 | 0.863 / 0.776 / 0.704 |
| $n=30$ | 0.805 / 0.755 / 0.839 | 0.778 / 0.747 / 0.602 | 0.803 / 0.747 / 0.814 | 0.806 / 0.852 / 0.865 |
| $n=100$ | 0.801 / 0.650 / 0.439 | 0.178 / 0.615 / 0.337 | 0.875 / 0.626 / 0.464 | 0.896 / 0.852 / 0.486 |
| $n=200$ | 0.877 / 0.785 / 0.838 | 0.873 / 0.747 / 0.522 | 0.875 / 0.747 / 0.814 | 0.775 / 0.751 / 0.704 |

hedge 对照（图 6.2，$n=30$，$\tau=10^{-4}$）：GC-YZ-LIN $0.741$（hedge）对 $0.659$（MFN）；DFOTR $0.744$ 对 $0.639$；NEWUOA $0.748$ 对 $0.717$。
子空间对照（图 6.6，GC-YZ-LIN+hedge 对 GC-sub）：$n=30$：$0.861/0.674/0.505$ 对 $0.750/0.498/0.309$；$n=100$：$0.795/0.570/0.386$ 对 $0.659/0.378/0.194$。

**逐条审计论文的观察**：

1. **"新的自适应 Hessian 拟合改善所有求解器的性能"——支持。** 三个求解器在图 6.2 上 hedge 都高于 default，提升幅度 $+0.082$（GC-YZ-LIN）、$+0.105$（DFOTR）、$+0.031$（NEWUOA）。**这是第 6 节最结实的一条结论**，而且它独立于本文的整套 poisedness 理论（hedge 是纯建模技巧）。
2. **"几何校正方法（GC-YZ-LIN/V、NEWUOA）在高维优于 DFOTR"——支持，但证据强度不均。** $n=30$：$0.778$ 对 $0.805$（小幅）；$n=100$：$\tau=10^{-2}$ 时 DFOTR 崩到 $0.178$（大幅）；但 **$n=200$ 时 DFOTR 恢复到 $0.873$，与 GC-YZ-LIN 的 $0.877$ 基本持平**。所以"维度越高差距越大"这条**单调趋势不成立**；$n=200$ 的反常说明 DFOTR 的崩塌是**问题集相关**（某些问题上半径塌缩）而不是维度相关的系统性现象。作者第 1242 行的机制解释（"$2n+1$ 个点在高维越来越稀疏、趋于退化"）无法解释 $n=200$ 的回升，**这一处解释与数据不完全一致**。
3. **"GC-YZ-V 几乎严格优于 NEWUOA 与 GC-YZ-LIN"——在 $n=30,100$ 成立，在 $n=200$ 不成立。** $n=200$：GC-YZ-V 三项 $0.775/0.751/0.704$ 全部**低于** GC-YZ-LIN 的 $0.877/0.785/0.838$ 与 NEWUOA 的 $0.875/0.747/0.814$（只有 $\tau=10^{-4}$ 略高于 NEWUOA）。所以"almost strict improvement"这个措辞**只在 $n\le100$ 成立**，而论文正文（第 1244 行）是在 $n=30,100,200$ 三张图的段落后写的。**这是第 6 节里唯一一处措辞强于证据的地方。**
4. **"随机子空间的实用实现仍难与全空间竞争"——支持，且证据很强。** 图 6.6 六个格子里 GC-sub 全部显著落后，且**差距随 $\tau$ 变小而扩大**（$n=30$：$-0.111$、$-0.176$、$-0.196$；$n=100$：$-0.136$、$-0.192$、$-0.192$）。作者的机制解释（每重抽一次要 $\mathcal{O}(q)$ 次求值、重抽次数的界不松）与 G10/G11 的分析一致 ✓；但**理论预测"维度越高子空间方法越有竞争力"被数据反驳**（$n=100$ 的差距比 $n=30$ 更大），作者自己也承认"this is not something that is immediately evident"。**这是全文最明确的理论—实践背离。**

## H8 第 6 节审计

- **主张**：实现保留 Algorithm 2 的全部要素并加实用部件，最坏情形 oracle 阶不变。
- **证据**：H1（$\sigma_k$ 只乘 $\approx13$ 层）、H2（两道 gate 只**减少**求值）、H3（假设 6.1 在精确解下 $\kappa_{step}=1$，与 $c_g=0.5$ 相容）。**结论：主张成立，但论文没有写出证明，只写了"follows very closely"**。我补的 H1 定量（13 倍）与 H3 的 $\kappa_{step}=\frac{2\kappa_{fcd}}{3}$ 是这一节真正缺的技术内容。
- **最强假设**：假设 6.1（实为推论）+ $c_g\le\kappa_{step}$。
- **隐藏依赖**：Line 7 子分支 2 的 $\frac12$ 与 $\gamma$ 混用；$\theta$ 与第 5 节同名；$K$ 的选择把 $\kappa_{bhm}$ 变成事后量。
- **非空泛区域**：H6 已量化——**在论文自己的实验参数下界是空泛的**。这不是缺陷而是定位：本文是复杂度论文，§6 是实用性论证。
- **失效区域**：$\kappa_{fcd}$ 小（近似退化子问题）时 $\kappa_{step}$ 小 ⟹ $c_g=0.5$ 的 gate 破坏理论；$\sigma_{end}$ 取得比 $\epsilon_f$ 相关的理论地板还小时（无噪声下无此问题）。
- **可证伪后续**：把 DFOTR 在 $n=200$ 回升的现象按问题分类（哪些问题的曲率极端），检验"DFOTR 崩塌 $\propto$ 问题条件数 $\times$ 维度"还是"$\propto$ 维度"。若前者成立，则第 1242 行的解释需要重写。

---

# 收尾：全文证明骨架、问题清单与后续

## 一、证明骨架（假设 ⟹ 引理 ⟹ 定理 的完整依赖链）

```
假设 1.1（φ 下有界，φ* = inf φ）
假设 1.2（∇φ 存在且 L-Lipschitz）
   │
   ├─(D1) 下降引理 |φ(y)-φ(x)-∇φ(x)ᵀy| ≤ (L/2)‖y‖²
   ├─(D2) ‖∇φ(y)-∇φ(x)‖ ≤ L‖y-x‖
   └─(S1) ‖²φ‖ ≤ L
        │
定义 2.1（fully-linear：κ_eg, κ_ef）＋ 假设 2.2（(2.2) fcd ＋ (2.3) ‖H_k‖≤κ_bhm）
        │
        ├─ 引理 2.9：κ_ef = κ_eg + (L+κ_bhm)/2        [用 (D1) + 精确居中]
        ├─ 引理 2.4：单步下降 ≥ C_2Δ_k²，C_2 = (η₁η_2κ_fcd/2)·min{η₂/κ_bhm,1}
        ├─ 引理 2.3：Δ_k ≤ C_1‖φ‖ ⇒ 成功，C_1⁻¹ = max{η₂,κ_bhm,(2κ_ef+max{η₂,κ_ef})/((1-η₁)κ_fcd)} + κ_eg
        │     └─ 引理 2.5：k<K_ε ⇒ Δ_k ≥ γC_1ε        [阶梯]
        │           ├─ 引理 2.6：|S_ε| ≤ 2(φ(x₀)-φ*)/(C_2(γC_1ε)²)   [用 2.4+2.5]
        │           └─ 引理 2.7：|U_ε| ≤ |S_ε| + ⌈log_γ(C_1ε/Δ₀)⌉    [用 2.3+2.5]
        │                 └─ 定理 2.8：|S_ε|+|U_ε| ≤ 2·(上式)          [2.6+2.7]
        │                       └─ 引理 2.9 + (2.9)FD ⇒ 推论 2.10：C_ε ≤ O(n^{3/2}ε⁻²) [η₂=√n]
        │
第 3 节（κ_eg 的显式维度依赖）
        ├─ 定义 3.1（poised）＋ 定义 3.2（Λ-poised）＋ 恒等式 max_{B(0,Δ)}|ℓ_j| = Δ‖(Yᵀ)_j‖
        ├─ 定理 3.3：κ_eg = ½(L+κ_bhm)√n·√(n(Λ²-1)+2)   [用 (D2) + 引理 4.x of [11] 的 ‖Y⁻ᵀD‖ 精化]
        ├─ 推论 3.4：Λ=1+1/n ⇒ κ_eg = Θ(√n)
        ├─ (3.3) 含噪插值 ＋ 定理 3.5：κ_eg 再加 √n·2ε_fΛ/Δ
        └─ 推论 3.6：Δ_k ≥ √(4ε_fΛ/(L+κ_bhm)) ⇒ κ_eg,κ_ef = Θ(√nΛ)   [因子 2 = 噪声代价]
                │
第 4 节（把每步 oracle 降到 ≤2）
        ├─ (4.1) 凸 QP（残差线性 + 谱范数球，SDP 可表示）
        ├─ 算法 2 的 (i)(ii)(iii)(iv) 瀑布 + I_imp 门控
        ├─ (4.4) Lagrange 秩一更新（四条 Lagrange 条件逐一验证）
        ├─ 引理 4.2：Λ₀-poised ⇒ Λ-poised 需 ⌈n·log(logΛ₀/logΛ)⌉ 步
        │     [det Y⁺ = det Y·ℓ_j(s̃) → Hadamard → ‖(Y⁻ᵀ)_i‖ ≤ |det Y|⁻¹ → AM-GM |det Y⁺| ≥ |det Y|^{1-1/n}]
        ├─ 引理 4.3：去远点后 ≤2n 步 ⇒ 14ⁿ-poised
        │     [子空间 poised 归纳：self-correct ×2，几何校正 ×7（√40<7）]
        ├─ 定理 4.1：连续模型改进步 ≤ 2n log n + 4n + n|log log Λ|
        └─ 推论 4.4：C_ε ≤ O(n^{3/2}log n·ε⁻²) [η₂=√n] / O(n²log n·ε⁻²) [η₂ 常数]
                │
第 5 节（随机子空间 + 确定性噪声）
        ├─ 定义 5.1（子空间 fully-linear）＋ 引理 5.2：κ_ef = κ_eg + (L_Q+κ_bhm)/2，L_Q ≤ L
        ├─ 算法 3：ρ_k 加 2ε_f（吸收两处 oracle 误差）＋ Δ_min（保证 κ_eg 有限）
        ├─ 定义 5.3（κ_g-well aligned）＋ 引理 5.4（勾股 ⟺ (5.6)；(5.7) 靠 QQᵀg_k=g_k）
        ├─ 引理 5.5：Δ_k ≤ Ĉ₁‖∇φ ⇒ 成功，Ĉ₁ = √(1-κ_g²)·C̃₁，C̃₁¹ = max{η₂,κ_bhm,2κ_ef/((1-η₁)κ_fcd)} + κ_eg
        ├─ 引理 5.6：成功步 φ 下降 ≥ C_2Δ_k² − 4ε_f     [可能为负！]
        ├─ 随机过程 I_t,A_t,B_t；A_t ≥ I_t(1−B_t)；B_t=1 ⇒ Δ_{t+1}=γΔ_t（地板不咬）
        ├─ 引理 5.8（势函数界大步成功）＋ 5.9（阶梯界失败步）＋ 5.10 ＋ 5.11
        ├─ 定理 5.12：E[T_ε] ≤ (4θ/(2θ−1)²)(·)          [C₃ ≤ θ−½ 由 ε 下界给出]
        ├─ 定理 5.13（子空间 FD）：E[K_ε] ≤ O((n/ε²)(L+ε_f/Δ_min²)²)；Δ_min=Θ(√ε_f) ⇒ O(nε⁻²)
        └─ 推论 5.14（算法 4）：E[C_ε] ≤ O(nq log q·ε⁻²)
                │
第 6 节（实现）
        ├─ 假设 6.1（实为引理 2.2 的推论，κ_step = 2κ_fcd/3；精确解/Cauchy 点 κ_step = 1）
        ├─ σ_k 分辨率下限 ⇒ 乘 ⌈log_{1/θ}(Δ₀/σ_end)⌉+1 ≈ 13 层
        └─ 两道 gate 只减少求值 ⇒ 阶不变
```

## 二、全文复杂度机制的统一解释

所有界都可以写成同一个模板：

$$
\mathcal{C}_\epsilon\le(\text{每迭代 oracle 数})\times\frac{C_1^{-2}}{C_2}\cdot\epsilon^{-2},\qquad C_2=\frac{\eta_1\eta_2\kappa_{fcd}}{2}\min\Big\{\frac{\eta_2}{\kappa_{bhm}},1\Big\}.
$$

（第 5 节把 $C_1$ 换成 $\hat C_1$。）**三个自由度**：
1. **$\kappa_{eg},\kappa_{ef}$ 的大小**决定 $C_1^{-1}$（第 3 节的全部工作，$\Lambda$-poisedness 是它的度量工具）；
2. **$\eta_2$ 的取法**决定 $C_2$（$\eta_2$ 增大使 $C_2$ 增大，但也可能使 $C_1^{-1}$ 增大）；
3. **每迭代 oracle 数**（第 4 节的 self-correction + 几何校正把它从 $n+1$ 降到摊销 $\mathcal{O}(\log n)$；第 5 节用子空间降到 $q$）。

**$\eta_2$ 的权衡是全文最微妙的一点**：
- 全空间 + FD：$C_1^{-1}\ge\kappa_{eg}=\Theta(\sqrt n)$ 已经支配 $\max$，所以 $\eta_2=\sqrt n$ **免费**塞进 $\max$，$C_2$ 白得 $\Theta(\sqrt n)$ ⟹ 迭代数从 $n$ 降到 $\sqrt n$（推论 2.10）。
- 子空间 + FD：$\kappa_{eg}=\Theta(\sqrt qL)$ 是常数级，$\hat C_1^{-1}$ 带 $\sqrt{n/q}$ 放大 ⟹ $\eta_2$ **不免费**，最优就是 $\eta_2=\Theta(1)$（G10）。
- **同一个参数在两个情形下的最优取法相反**，而论文在两处都只给出结论、没有解释这个对称性。这是本稿最有增量价值的一处分析。

## 三、问题清单（按严重性分级）

**A 级：影响某个具体陈述的正确性，但不影响任何阶结论**

| 编号 | 位置 | 问题 | 修复 |
| --- | --- | --- | --- |
| A1 | 引理 5.5 证明 (c)→(d) | 两处 $\kappa_{ef}\Delta_k^2$ 应带系数 2 | $\tilde C_1^{-1}$ 里 $\frac{2\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$ 改 $\frac{4\kappa_{ef}}{(1-\eta_1)\kappa_{fcd}}$ |
| A2 | 引理 5.10 陈述 | 右边 $\sum\bar B_tA_t$ 漏 $\mathbb{E}$ | 补期望号 |
| A3 | 引理 5.11 | 有 $\Delta_{min}$ 地板时 [10] 的阶梯论证在小半径侧需补一项 $\log_\gamma\frac{\Delta_{min}}{\hat C_1\epsilon}$ | 补加性项，阶不变 |
| A4 | 第 1098 行 | $\kappa_{step}=\frac{\kappa_{fcd}}{3}$ 与 $c_g\le\kappa_{step}$、$c_g=0.5$ 联立要求 $\kappa_{fcd}\ge1.5$，不可能 | 改为精确解的 $\kappa_{step}=1$（实验正是精确求解）或我推的 $\frac{2\kappa_{fcd}}{3}$ |
| A5 | 第 892 行 | "$B_t=1\Rightarrow\bar B_1=1$" 下标错 | $\bar B_t$ |
| A6 | 推论 4.4 | $\epsilon$ 下限漏 $\Lambda$（应为 $\min\{C_2,(L+\kappa_{bhm})/\Lambda\}$） | 补 $\Lambda$ |
| A7 | 定理 4.1 证明 | "$\le2n\log n+4n+n\vert\log\log\Lambda\vert$" 与上一行的取整关系未交代（迭代数与 oracle 数差恰为 2） | 明确区分两个计数 |

**B 级：论文跳步、我已在推导中补完（不是错误）**

| 编号 | 位置 | 跳步内容 |
| --- | --- | --- |
| B1 | 第 261 行 | $\kappa_{eg}=\sqrt nL$ 需要"$\frac{\sqrt nL}{2}+\frac{2\sqrt n\epsilon_f}{\Delta_k^2}$ 再代入 $\Delta_k\ge2\sqrt{\epsilon_f/L}$"，已补（G10） |
| B2 | 引理 5.2 | 全证缺失，已用四段拆分 + $L_Q$ 余项完整重推（G2） |
| B3 | 引理 5.4 | 全证缺失，已用勾股 + $\Vert QQ^\top w\Vert=\Vert Q^\top w\Vert$ 完整重推（G4） |
| B4 | 引理 5.6 | $4\epsilon_f=2\epsilon_f+\epsilon_f+\epsilon_f$ 的构成未拆，已拆（G6） |
| B5 | 定理 5.12 | $C_3\le\theta-\frac12$ 的代入未写，已写（G9 第五步） |
| B6 | 引理 5.9 | "easily follows"，我给出阶梯重建（G8），取整细节仍留白 |
| B7 | 假设 6.1 | 实为假设 2.2 的推论，已证（H3） |
| B8 | 第 1098 行 $\kappa_{step}=1$ | 精确解/Cauchy 点两支论证缺失，已补（H3） |
| B9 | 定理 3.3 | $\Vert Y^{-\top}D\Vert$ 的精化依赖 [11, Thm 4.3]，我只自证到 $\le\sqrt n\Lambda$（弱一维），已诚实标注（阶段 E4） |

**C 级：结构性 / 概念性问题（值得在评审意见里写）**

| 编号 | 问题 | 说明 |
| --- | --- | --- |
| C1 | **插值/几何校正机制在最坏情形 oracle 计数上从未赢过有限差分** | 推论 2.10 的 $n^{3/2}$ 对推论 4.4 的 $n^{3/2}\log n$；定理 5.13 的 $nq$ 对推论 5.14 的 $nq\log q$。两处都是**多一个 $\log$**。论文标题与摘要暗示的贡献是"Powell 风格 + 复杂度保证"，但复杂度上并不优于简单 FD。作者用 §6 的实践性能补这一票，是合理的选择，但**读者若只看理论部分会高估第 4 节的收益**。 |
| C2 | **算法 1/2/3/4 的分支判据不可实现** | "模型是否 fully-linear"、"$\Vert\nabla\phi(x_k)\Vert\le\epsilon$" 在零阶设定下都不可判定。第 3 节把判据换成 $\Lambda$-poisedness 是实质进步（可计算），但算法 3 又退回"not fully-linear"，算法 5 混用两种。 |
| C3 | **$\epsilon_f$ 必须已知上界** | 第 5 节整个算法依赖 $\rho_k$ 里的 $2\epsilon_f$ 与 $\Delta_{min}$ 的选取。高估 $\epsilon_f$ 会同时抬高 $\Delta_{min}$、$\kappa_{eg}$ 与精度下限，而论文只给了定性提醒（第 756 行）。 |
| C4 | **引理 4.3 的 $14^n$ 与引理 4.2 的 $n\log\log\Lambda$ 之间的接缝** | 定理 4.1 直接把 $\Lambda_0=14^n$ 代入引理 4.2，但引理 4.2 要求 $\Lambda>\Lambda_0$ 才有意义（$\log\log$ 单调），而这里 $\Lambda=1+\frac1n\ll14^n$。**方向是"从坏到好"，需要引理 4.2 的陈述支持 $\Lambda<\Lambda_0$ 的情形**——我在阶段 F7 标注了这一处逻辑接缝。 |
| C5 | **第 6 节参数与定理条件相反**（H6） | 界在实验参数下空泛（$\sim10^{333}$ 次迭代）。 |
| C6 | **观察 3 的措辞强于证据**（H7） | GC-YZ-V 在 $n=200$ 全面落后。 |
| C7 | **理论完全不覆盖墙钟/内存** | (4.1) 是 $\big(n+\frac{n(n+1)}{2}\big)$ 变量、SDP 可表示的凸 QP，朴素内点法单步 $\mathcal{O}(n^6)$；§6 的实现改用 (4.4) 秩一更新 + Powell 式 $\mathcal{O}(n^2)$ 求解，**理论与实现之间这块是断开的**（标题限定为 oracle complexity，属自我划界）。 |

## 四、可证伪的后续问题（按性价比排序）

1. **子空间方法能否也拿到 $\sqrt n$ 的迭代收益？**（G12）把 $\hat C_1$ 的门槛改成分层判据（对齐时按投影梯度、不对齐时不要求收缩），检验能否去掉 $\sqrt{n/q}$ 放大。判别实验：$n\in\{100,400\}$、$q=5$ 的合成二次函数上比较 $\eta_2\in\{1,\sqrt n\}$ 两组的**迭代数**增长率。
2. **关掉 self-correction 会怎样？**（阶段 F9）这是判定第 4 节 (ii) 步是"机制必需"还是"理论装饰"的最小实验；§6 的 GC-YZ-LIN vs GC-YZ-V 只做了其中一半。
3. **$\Lambda$-poisedness 的 $14^n$ 是否是分析松弛？**（阶段 E9）若能构造一个 $\Lambda$-poised 集使 $\kappa_{eg}\ge c\,n\Lambda$（而非 $\sqrt n\Lambda$），则第 3 节的 $\sqrt n$ 收益全部来自 [11, Thm 4.3] 的精化，本文的界在 $\Lambda$ 不小时会退化。判别方式：随机生成 $\Lambda$-poised 集，测 $\kappa_{eg}$ 对 $n$ 的经验指数。
4. **DFOTR 的崩塌是维度驱动还是问题条件数驱动？**（H7 观察 2）按问题条件数分层重做图 6.3–6.5。
5. **引理 5.11 的地板缺项是否真实存在？**（A3）构造一个 $\Delta_{min}$ 卡住、$I_t$ 持续为 0 的实例，测 $\sum(1-B_t)$ 与 $T_\epsilon$ 的比值是否超过 $\frac{1}{2\theta}$。

## 五、待核验项

| 项 | 状态 |
| --- | --- |
| [11] = Chaudhry & Scheinberg, ICM 2026 论文集 pp.208–228 | 已从 PDF 第 31 页参考文献表核实 |
| [10] = Cartis & Scheinberg, Math. Oper. Res. 169(2):337–375, 2018 | 已核实；引理 5.10/5.11 与 $(\ddagger)$ 的具体编号仍需查原文 |
| [7] = Cao, Berahas, Scheinberg, Math. Prog. 207:573–624, 2023 | 已核实；引理 5.6 自称"类似 [7, Lem 4.3]"，两处的 $-4\epsilon_f$ 是否同形式待查 |
| [4] = Berahas, Cao, Choromanski, Scheinberg, FoCM 2021 | 已核实；FD 误差界 $\frac{\sqrt nL\delta}{2}+\frac{2\sqrt n\epsilon_f}{\delta}$ 的原文形式（是否含 $\sqrt n$、噪声项系数 2）待查 |
| [11, Thm 4.3]（$\Vert Y^{-\top}D\Vert$ 精化）与 [11, Lem 6.7]（$\theta\ge243/443$、$\kappa_g=\sqrt{1-q/10n}$） | **未核验原文**，本文推导直接引用；这两条分别是 $n^{3/2}$ 与 $\mathcal{O}(n\epsilon^{-2})$ 的承重墙，优先级最高 |
| [26] = Scheinberg & Toint, SIOPT 20(6):3512–3532, 2010 | 已核实为 self-correcting geometry 的出处 |
| 参考文献表 | Markdown 转换整体丢失，已从 `01-raw` PDF 第 30–32 页补录到 G0 |

## 六、与交互笔记的关系

本文件是全局推理模式的过程草稿，含我的自查与自我修正记录（阶段 F8 修正了阶段 C7/D3 关于"$\sqrt n$ 收益来自 $C_1$ 还是 $C_2$"的归因、以及阶段 E8 的一处不等式方向）。逐轮问答记录见同目录的 `Powell_Style_..._精读笔记.md`。定稿（去掉问答与自查痕迹、按 Typora 可渲染规范书写）见 `../../04-equation_problem/Powell_Style_Model_Based_Derivative_Free_Optimization_with_Complexity_Guarantees/全局推理.md`。
