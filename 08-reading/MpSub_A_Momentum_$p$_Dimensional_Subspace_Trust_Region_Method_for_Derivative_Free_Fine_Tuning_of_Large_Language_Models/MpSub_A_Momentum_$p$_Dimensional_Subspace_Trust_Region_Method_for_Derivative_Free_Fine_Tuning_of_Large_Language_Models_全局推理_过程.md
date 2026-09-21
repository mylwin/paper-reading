# MpSub 全局推理：过程稿

> **本文件由全局推理模式生成**（用户指令："帮我全局推理该文献中的公式、定理、假设和算法等"）。
> 内容为推导草稿、跳步补证记录与量级审计，**不作教材使用**。定稿见 [全局推理.md](../../04-equation_problem/MpSub_A_Momentum_$p$_Dimensional_Subspace_Trust_Region_Method_for_Derivative_Free_Fine_Tuning_of_Large_Language_Models/全局推理.md)。
> 配套交互问答笔记：[MpSub_..._精读笔记.md](MpSub_A_Momentum_$p$_Dimensional_Subspace_Trust_Region_Method_for_Derivative_Free_Fine_Tuning_of_Large_Language_Models_精读笔记.md)（本次为全局推理模式，该文件未被写入）。

## 0. 论文信息与编号还原

- 标题：MpSub: A Momentum $p$-Dimensional Subspace Trust-Region Method for Derivative-Free Fine-Tuning of Large Language Models。
- 作者与单位：Yuyang Wang、Haoyu Yao（西安交通大学）、Pengcheng Xie（LBNL，通讯作者）。
- 状态：**Preprint**，arXiv:2609.07666v1，v1 日期 2026-09-07（cs.LG）。AMS：90C56（无导数优化）、65K05（数学优化算法）、90C26（随机规划）。
- 类型：**理论 / 理论算法**（随机子空间信赖域 + 零阶 LLM 微调），附小规模数值实验。
- 记号约定：论文中向量与矩阵为粗体，本稿用普通字母（ $x$ 、 $g$ 、 $D$ ）； $\Vert \cdot \Vert$ 不带下标时指 $\Vert \cdot \Vert_2$ （向量取 Euclidean 范数、矩阵取谱范数）。

公式编号与 PDF 一致（PDF 保留了右侧编号，无需还原）：第 2 节 (2.1)–(2.18)，第 3 节 (3.1)–(3.22)。

## 1. 推理计划与依赖链

处理顺序按论文行文，但**推导顺序按依赖关系**。

```
第 2 节（算法本体）
  (2.1) 问题 → Algorithm 2.1
    ├─ Step1 子空间框架: (2.2) 参数化 ← (2.3) 动量方向 ← (2.4) 高斯方向 ← (2.5) 近正交性
    ├─ Step2 子空间梯度: (2.6) 中央差分 → g_k → (2.7) 线性模型
    ├─ Step3 试探步:     (2.8) 信赖域子问题 → (2.9) 闭式解 → (2.10) x^+ → (2.11) pred_k
    ├─ Step4 接受/半径:  (2.12) rho_k → (2.13) 接受规则、(2.14) 半径更新
    └─ LLM 实现:         (2.15) minibatch → (2.16) 同批 rho_k → (2.17) 种子重建 → (2.18) 位移累加

第 3 节（理论链）
  Assumption 3.1 / (3.2) L-光滑
    ├─[D1] 下降引理（由 (3.2) 积分推出，论文未单列）
    ├─ Lemma 3.2 (3.3)(3.4) 有限差分误差，含 Q_k          ┐
    ├─ Lemma 3.3 (3.5)(3.6)(3.7) 高斯梯度捕获 + 好事件 G_k ┴─→ Prop 3.4 (3.8)(3.9) 一般 frame 的单步下降与 rho 下界
    │                                                        │
    ├─ Lemma 3.5 (3.10)(3.11)(3.12) 半径平方可求和 ←──────────┘（只用 rho 定义 + (3.1) safeguard）
    │        │
    ├─ Theorem 3.6 (3.13)–(3.18) liminf ‖∇f(x_k)‖ = 0 a.s.（反证 + 鞅大数律 + 对数增速）
    │        │
    ├─ Lemma 3.7 (3.19) Σ_{K_eps} Δ_k < ∞ a.s.（**论文此处跳步**：renewal）
    │        │
    └─ Theorem 3.8 (3.20)(3.21)(3.22) ‖∇f(x_k)‖ → 0 a.s.（矩条件 + 梯度 Lipschitz + 行程可求和）
```

外部知识清单（在使用处当场展开）：下降引理、中央差分的 Taylor 余项、 $\chi^2$ 分布可加性、独立高斯向量的近正交性矩计算、Hoeffding–Azuma 不等式 + Borel–Cantelli（鞅强大数律）、鞅收敛定理（二次变差有限 $\Rightarrow$ a.s. 收敛）、逆三角不等式与激变（excursion）构造。

---

## 2. 第 2 节推导要点

### 2.1 (2.5) 近正交性：必须自己算

$d_i^{(k)\top} d_j^{(k)} = z_i^\top z_j / n$ ， $z_i, z_j$ 独立且分量为 $N(0,1)$ 。

- 一阶： $\mathbb{E}[z_i^\top z_j] = \sum_a \mathbb{E}[z_{i,a}]\mathbb{E}[z_{j,a}] = 0$ 。
- 二阶： $\mathbb{E}[(z_i^\top z_j)^2] = \sum_{a,b}\mathbb{E}[z_{i,a}z_{i,b}]\mathbb{E}[z_{j,a}z_{j,b}] = \sum_{a,b}\delta_{ab}\delta_{ab} = n$ ，故 $\mathbb{E}[(d_i^\top d_j)^2] = 1/n$ 。

RMS 内积 $= n^{-1/2}$ ； $n \sim 10^8$ 时为 $10^{-4}$ 。**注意**：这只说明"两两内积的均方根小"，不等于 $\beta_k = \Vert D_k\Vert_2$ 小； $\beta_k$ 的量级由第 3 节 (3.20) 前的矩条件处理（ $\beta_k^2 \le \sum_i \Vert d_i\Vert^2 \approx p$ ）。这是"跳过 Gram–Schmidt"的真实代价，见 §5 审计。

### 2.2 (2.9) 闭式解：Cauchy–Schwarz 取等条件

对任意 $\Vert s_s\Vert_2 \le \Delta_k$ ： $g_k^\top s_s \ge -\Vert g_k\Vert_2 \Vert s_s\Vert_2 \ge -\Delta_k \Vert g_k\Vert_2$ 。
第一个不等号取等 $\iff$ $s_s = -\lambda g_k$ （ $\lambda \ge 0$ ）；第二个取等 $\iff$ $\lambda \Vert g_k\Vert_2 = \Delta_k$ 。合并得唯一解 (2.9)。故 $g_k \neq 0$ 时解唯一且在边界上（Cauchy 点退化到边界，因为模型是线性的、无曲率约束）。

### 2.3 (2.17) 种子重建的单射条件

$\mathrm{seed}(k,i) = (s_0 C_1 + k C_2 + i C_3)$ 再对 $M$ 取模，其中 $s_0$ 是全局 `base_seed`， $C_1,C_2,C_3,M$ 为固定素数。
不同的 $(k,i)$ 需要不同种子，否则两条探索方向逐元素相同 $\Rightarrow$ frame 秩亏 $\Rightarrow$ Lemma 3.3 的"条件独立"与 (3.5) 的 $\chi^2_{p-1}$ 分布同时失效。
充分条件： $K_{\max} C_2 + p C_3 < M$ （ $K_{\max}$ 为最大迭代号）。论文未给出 $C_1,C_2,C_3,M$ 的具体值，故**无法核验该单射在实际超参下成立**；实验 $K_{\max}=200$ 、 $p\le 30$ ，只要 $M$ 取 31 位素数且 $C_2,C_3$ 为中等素数即自动成立。列为待核项。

### 2.4 内存与预算核算（论文数字自洽性检查）

- 显式存 $D_k$ ： $pn$ 个 FP32 数 $= 20 \times 1.25\times 10^8 \times 4$ 字节 $= 10$ GB。✓ 与"超过 OPT-125M 权重本身（0.5 GB）"一致。
- MpSub 实际额外存储： $m_k$ 与 $u_k$ 各 $n$ 维 $= 2 \times 1.25\times 10^8 \times 4$ 字节 $= 1$ GB。所以"零额外开销"应理解为"相对 $pn$ 为零"，而非绝对零。
- 前向预算： $2p+2 = 42$ ， $8400/42 = 200$ 步。✓ 与 §4.1 一致。

---

## 3. 第 3 节推导要点

### 3.1 [D1] 下降引理（论文默认已知，此处展开）

由 (3.2) 与微积分基本定理沿段 $x + \theta(y-x)$ ：

$$

f(y) - f(x) = \int_0^1 \nabla f(x + \theta (y-x))^\top (y-x) d\theta

$$

$$

\Rightarrow \left\vert f(y) - f(x) - \nabla f(x)^\top (y-x) \right\vert \le \int_0^1 L \theta \Vert y-x\Vert_2^2 d\theta = \frac{L}{2} \Vert y-x\Vert_2^2

$$

取 $y = x + t d$ 即得 Lemma 3.2 证明第一行的不等式；取上界方向即得 Prop 3.4 与 Lemma 3.5 使用的 $f(x_k^+) \le f(x_k) + \nabla f(x_k)^\top h_k + \frac{L}{2}\Vert h_k\Vert_2^2$ 。

### 3.2 Lemma 3.2

$f(x_k + \Delta_k d_i) = f(x_k) + \Delta_k \hat g_{k,i} + R_{i,+}$ ， $f(x_k - \Delta_k d_i) = f(x_k) - \Delta_k \hat g_{k,i} + R_{i,-}$ ， $\vert R_{i,\pm}\vert \le \frac{L\Delta_k^2}{2}\Vert d_i\Vert_2^2$ 。
两式相减除以 $2\Delta_k$ ： $g_{k,i} - \hat g_{k,i} = (R_{i,+} - R_{i,-})/(2\Delta_k)$ ，三角不等式给出 (3.3)。
(3.4)： $\Vert g_k - \hat g_k\Vert_2^2 = \sum_i (g_{k,i}-\hat g_{k,i})^2 \le \frac{L^2\Delta_k^2}{4}\sum_i \Vert d_i\Vert_2^4 = \frac{L^2\Delta_k^2}{4}Q_k^2$ 。✓

**关键点**：误差项 $L\Delta_k Q_k/2$ 与 $n$ 无关，而被捕获信号 $\Vert \hat g_k\Vert$ 随 $n$ 缩小（3.5）。这是全篇理论的核心张力。

Hessian  Lipschitz 版本： $f(x \pm \Delta d)$ 三阶展开相消后中央差分误差 $= \frac{\Delta^2}{6}\nabla^3 f[d,d,d] + O(\Delta^4)$ ， $\vert \nabla^3 f[d,d,d]\vert \le M \Vert d\Vert_2^3$ ，得 $\frac{M\Delta_k^2}{6}\Vert d_i\Vert_2^3$ 。论文声明后文不用。

### 3.3 Lemma 3.3

$v_k^\top d_i^{(k)} = v_k^\top z_i/\sqrt{n}$ ； $v_k^\top z_i \sim N(0, \Vert v_k\Vert_2^2)$ （一维高斯线性泛函），除以 $\sqrt n$ 得 $N(0, \Vert v_k\Vert_2^2 / n)$ 。✓ 论文最后一行。
$\{(v_k^\top d_i)^2\}_{i=2}^p$ 条件独立同分布，且 $(n/\Vert v_k\Vert^2)(v_k^\top d_i)^2 \sim \chi^2_1$ ，由 $\chi^2$ 可加性：

$$

\sum_{i=2}^{p}\left( v_k^\top d_i^{(k)} \right)^2 = \frac{\Vert v_k\Vert_2^2}{n} \chi^2_{p-1}

$$

（上式应读作"左端随机变量与右端同分布"，论文原文用带上标 $d$ 的等号表示这一点。）取期望 $\mathbb{E}[\chi^2_{p-1}] = p-1$ 得 (3.6)。

(3.7) 的构造：取 $\tau$ 使 $\mathbb{P}(\chi^2_{p-1} \ge \tau) \ge q$ （ $\chi^2_{p-1}$ 在 0 处密度可积、 $\tau \to 0$ 时概率 $\to 1$ ，故对任意 $q<1$ 可行），令 $\alpha = \sqrt{\tau/n}$ 。
$R$ ： $\mathbb{P}(\max_{2\le i\le p} \Vert z_i\Vert_2^2 / n > R^2) \le (p-1)\mathbb{P}(\chi^2_n > nR^2) \to 0$ （ $R>1$ 固定、 $p$ 有限、大偏差），故 $R$ 可行。动量方向 $\Vert d_1\Vert_2 = 1 \le R$ 。联合得 (3.7)。

### 3.4 Proposition 3.4

$h_k = D_k s_k$ ， $\hat g_k = D_k^\top \nabla f(x_k) \Rightarrow \nabla f(x_k)^\top h_k = \hat g_k^\top s_k$ 。
$\hat g_k^\top s_k = g_k^\top s_k + (\hat g_k - g_k)^\top s_k \le -\Delta_k \Vert g_k\Vert_2 + \Vert \hat g_k - g_k\Vert_2 \Delta_k \le -\Delta_k\Vert g_k\Vert_2 + \frac{L Q_k \Delta_k^2}{2}$ 。
$\Vert h_k\Vert_2 \le \beta_k \Delta_k$ 。代入 [D1] 得 (3.8)；除以预测下降 $\Delta_k \Vert g_k\Vert_2$ 得 (3.9)。✓ 全部展开无跳步。

### 3.5 Lemma 3.5

$k \in \mathcal{S} \Rightarrow f(x_k) - f(x_k^+) = \rho_k \Delta_k \Vert g_k\Vert_2 \ge \eta \kappa \Delta_k^2 > 0$ （用 $\rho_k \ge \eta$ 与 $\Vert g_k\Vert_2 \ge \kappa \Delta_k$ ）。
严格下降 $\Rightarrow$ (2.13) 接受 $\Rightarrow$ $x_{k+1} = x_k^+$ 。又拒绝步满足 $f(x_{k+1}) = f(x_k)$ ，故 $\{f(x_k)\}$ 单调不增，逐项非负：

$$

\sum_{k\in\mathcal{S}, k\le N} \Delta_k^2 \le \frac{1}{\eta\kappa}\sum_{k=0}^{N}\left( f(x_k) - f(x_{k+1}) \right) = \frac{f(x_0) - f(x_{N+1})}{\eta\kappa} \le \frac{f(x_0) - f_{low}}{\eta\kappa}

$$

递推： $\Delta_{k+1}^2 \le \gamma_1^2 \Delta_k^2 + \gamma_2^2 \Delta_k^2 1_{\{k\in\mathcal{S}\}}$ （两种情形分别验证； $\Delta_{\min}=0$ 保证 $\max(\gamma_1\Delta_k, 0) = \gamma_1\Delta_k$ 无截断）。求和移项：

$$

(1-\gamma_1^2)\sum_{k=1}^{N}\Delta_k^2 + \Delta_{N+1}^2 \le \gamma_1^2 \Delta_0^2 + \gamma_2^2 \sum_{k\in\mathcal{S}} \Delta_k^2 \le \gamma_1^2 \Delta_0^2 + \frac{\gamma_2^2 (f(x_0) - f_{low})}{\eta\kappa}

$$

$1-\gamma_1^2 > 0$ （因 $\gamma_1 \in (0,1)$ ） $\Rightarrow$ (3.11)，正项级数收敛 $\Rightarrow$ 通项 $\to 0$ 得 (3.12)。✓

### 3.6 Theorem 3.6（反证骨架）

1. 设 (3.14) 成立。由 Lemma 3.3 取 $\alpha, R, q$ ，事件 $G_k$ 条件概率 $\ge q$ 。
2. $G_k$ 上： $Q_k \le \sqrt p R^2$ ， $\beta_k = \Vert D_k\Vert_2 \le (\sum_i \Vert d_i\Vert_2^2)^{1/2} \le R\sqrt p$ （论文印为 $pR^2$ ，是更松但同样有效的界）。
3. $\Vert g_k\Vert_2 \ge \alpha \epsilon - \frac{L\sqrt p R^2}{2}\Delta_k \to \alpha\epsilon$ ，故 $k$ 足够大时 (3.16) $\Vert g_k\Vert_2 \ge \alpha\epsilon/2$ 。
4. (3.9)： $\rho_k \ge 1 - \frac{L\Delta_k (Q_k + \beta_k^2)}{2\Vert g_k\Vert_2} \to 1$ ，故 $k$ 足够大时 $\rho_k \ge \eta$ ；同时 $\Vert g_k\Vert_2 \ge \kappa\Delta_k$ 。 $\Rightarrow$ $G_k$ 发生时 $k \in \mathcal{S}$ 。
5. **需要补的一句**： $\Delta_{k+1} = \min(\gamma_2\Delta_k, \Delta_{\max})$ 要等于 $\gamma_2\Delta_k$ ，需 $\gamma_2\Delta_k \le \Delta_{\max}$ ；由 (3.12) 与 $\Delta_{\max} > 0$ 固定， $k$ 足够大后自动成立。论文未提，属可补的平凡细节。
6. (3.17) 取对数： $\log \Delta_{k+1} - \log \Delta_k \ge I_k \log \gamma_2 + (1-I_k)\log \gamma_1$ 。
7. 鞅强大数律： $I_k - \mathbb{E}[I_k\mid\mathcal{F}_k]$ 有界（ $\le 1$ ）鞅差序列，Hoeffding–Azuma + Borel–Cantelli $\Rightarrow$ $\frac1N \sum (I_k - \mathbb{E}[I_k\mid\mathcal{F}_k]) \to 0$ a.s.，又 $\mathbb{E}[I_k\mid\mathcal{F}_k] \ge q$ ，得 $\liminf \frac1N \sum I_k \ge q$ a.s.
8. 平均 (3.17)： $\liminf \frac{\log \Delta_{K+N} - \log \Delta_K}{N} \ge q\log\gamma_2 + (1-q)\log\gamma_1$ 。
   (3.15) 的等价性： $q > \frac{\log(1/\gamma_1)}{\log(\gamma_2/\gamma_1)} \iff q\log\gamma_2 + (1-q)\log\gamma_1 > 0$ 。逐步验证：
   $q(\log\gamma_2 - \log\gamma_1) > -\log\gamma_1 \iff q\log\gamma_2 > (q-1)\log\gamma_1 \iff q\log\gamma_2 + (1-q)\log\gamma_1 > 0$ 。✓（注意 $\log\gamma_1 < 0 < \log\gamma_2$ ，右端系数 $q>1/2$ 保证 $q - 1 < 0$ 。）
9. 于是 $\Delta_{K+N} \ge \Delta_K e^{cN}$ （ $c>0$ ） $\to \infty$ ，与 (3.12) 矛盾 $\Rightarrow$ (3.13)。

### 3.7 Lemma 3.7 —— **论文唯一的实质跳步**

论文原文：由 $q > 1/2$ 与"standard renewal property for probabilistically accurate models"把 $\sum_{\mathcal{K}_\epsilon \cap G_k}\Delta_k < \infty$ 升级为 $\sum_{\mathcal{K}_\epsilon}\Delta_k < \infty$ ，**无引用、无证明**。

本稿补一个自足证明（用鞅收敛定理替代 renewal）：

- 记 $p_k = \mathbb{P}(G_k \mid \mathcal{F}_k) \ge q$ ， $w_k = \Delta_k 1_{\{k \in \mathcal{K}_\epsilon\}} 1_{\{k \ge k_0\}}$ ， $w_k$ 是 $\mathcal{F}_k$-可测（ $x_k$ 、 $\Delta_k$ 、 $\epsilon$ 判据都在采样新方向之前已知）。
- 已知一： $\sum_k w_k 1_{G_k} < \infty$ a.s.。因为 $G_k \cap \mathcal{K}_\epsilon \cap \{k \ge k_0\}$ 上 $\Vert g_k\Vert_2 \ge \alpha\epsilon/2$ 且 $\rho_k \ge \eta$ ，故接受且 $f(x_k) - f(x_{k+1}) = \rho_k \Delta_k \Vert g_k\Vert_2 \ge \frac{\eta\alpha\epsilon}{2}\Delta_k$ ；这些非负增量之和 $\le f(x_0) - f_{low}$ （子级数 $\le$ 全级数，因 $\{f(x_k)\}$ 单调不增）。
- 已知二： $\sum_k w_k^2 \le \sum_k \Delta_k^2 < \infty$ a.s.（Lemma 3.5）。
- 鞅 $M_N = \sum_{k\le N} w_k(1_{G_k} - p_k)$ 的二次变差 $\sum_k w_k^2 p_k(1-p_k) \le \sum_k w_k^2 < \infty$ a.s.，且增量 $\vert w_k(1_{G_k}-p_k)\vert \le w_k \to 0$ 。由局部平方可积鞅收敛定理： $M_N$ a.s. 收敛到有限极限。
- 于是 $\sum_{k\le N} w_k p_k = \sum_{k\le N} w_k 1_{G_k} - M_N \le C(\omega) + \sum_k w_k1_{G_k} < \infty$ a.s.，再用 $p_k \ge q$ ： $\sum_k w_k \le q^{-1}\sum_k w_k p_k < \infty$ a.s.。证毕。

结论：(3.19) 成立，且**证明只用 $q > 0$**（不需要 $q > 1/2$！$q>1/2$ 只出现在论文那句未证 renewal 里）。这是一个可写进评审理由的技术观察：给定鞅论证后，引理的条件被放宽了。

### 3.8 Theorem 3.8

矩条件： $\beta_k^2 \le \sum_i \Vert d_i\Vert_2^2$ ， $\mathbb{E}[\Vert d_i\Vert_2^2 \mid \mathcal{F}_k] = 1$ （ $i\ge 2$ ）与 $\Vert d_1\Vert_2 = 1$ $\Rightarrow$ $\sup_k \mathbb{E}[\beta_k^2\mid\mathcal{F}_k] \le p$ ， $\sup_k \mathbb{E}[\beta_k \mid \mathcal{F}_k] \le \sqrt p$ （Jensen）。
(3.20)： $\mathbb{E}[\sum_k \beta_k^2 \Delta_k^2] = \sum_k \mathbb{E}[\Delta_k^2 \mathbb{E}[\beta_k^2\mid\mathcal{F}_k]] \le p \mathbb{E}[\sum_k \Delta_k^2] < \infty$ $\Rightarrow$ $\sum_k \beta_k^2\Delta_k^2 < \infty$ a.s. $\Rightarrow$ $\beta_k\Delta_k \to 0$ a.s.
(3.21)：Cauchy–Schwarz， $\sum_{k\in\mathcal{K}_\epsilon}\beta_k\Delta_k \le (\sum_k \beta_k^2\Delta_k^2)^{1/2} (\sum_{k\in\mathcal{K}_\epsilon}\Delta_k)^{1/2} < \infty$ a.s.（用 Lemma 3.7）。
激变构造（论文省略，需补）：若 $\lim_k \Vert\nabla f(x_k)\Vert_2 \ne 0$ ，结合 (3.13) 有 $\limsup > 0$ ；取 $\epsilon \in (0, \limsup/2)$ ，对无穷多个 $b_j$ （ $\Vert\nabla f(x_{b_j})\Vert_2 > 2\epsilon$ ）令 $a_j = \max\{k < b_j : \Vert\nabla f(x_k)\Vert_2 \le \epsilon\}$ （由 $\liminf = 0$ 该集合非空），则 $a_j < k < b_j$ 上 $\Vert\nabla f(x_k)\Vert_2 > \epsilon$ ，即 $k \in \mathcal{K}_\epsilon$ 。
$\epsilon < \Vert\nabla f(x_{b_j})\Vert_2 - \Vert\nabla f(x_{a_j})\Vert_2 \le L\sum_{k=a_j}^{b_j-1}\Vert x_{k+1}-x_k\Vert_2 \le L\sum_{k=a_j}^{b_j-1}\beta_k\Delta_k$
$= L\beta_{a_j}\Delta_{a_j} + L\sum_{k=a_j+1}^{b_j-1}\beta_k\Delta_k \to 0$ （第一项用 (3.20)，第二项是 (3.21) 级数的尾部）。矛盾 $\Rightarrow$ (3.22)。✓

---

## 4. 第 4 节：实验中的隐含数量关系

- 匹配预算口径：MpSub $2p+2 = 42$ 前向/步 $\times$ 200 步 $= 8400$ ；MeZO 2 前向/步 $\times$ 4200 步 $= 8400$ 。开发集评估"另计并排除在预算外"——但 §4.3 说 MeZO 的学习率网格搜索额外花 42000（OPT-125M，5 个学习率）与 25200（OPT-350M，3 个学习率 $\times$ 3 种子）前向，说明**总墙钟预算并不匹配**，只有训练目标前向匹配。论文对此是诚实的（明写了）。
- 表 4.1：OPT-125M 上 MpSub 开发损失 0.674 对 MeZO 0.760，但测试精度 0.673 对 0.655（差异在 min–max 区间内，56 样本测试集上 1 个样本 $= 1.8$ 个百分点）；OPT-350M 上测试 0.690 对 0.685、开发精度 0.787 对 0.685。**结论的强度依赖测试集规模**：3 种子 $\times$ 56 样本，二项噪声 $\approx \pm 6$ 个百分点，故"可比精度"成立、"优于 MeZO"不成立。
- 表 4.2： $p$ 从 5 到 30，开发精度 0.70→0.78→0.76（ $p=20$ 峰值），单步时延 3.0→10.1 s 单调上升（RTX 4060 Ti）。

---

## 5. 审计与量级核算（本稿核心增量）

### 5.1 provably-informative 半径的量级

在 $G_k$ 上，(3.9) 给出 $\rho_k \ge \eta$ 的充分条件：

$$

\Delta_k \le \frac{2(1-\eta)\Vert g_k\Vert_2}{L (Q_k + \beta_k^2)} \approx \frac{2(1-\eta)\alpha \Vert \nabla f\Vert_2}{L p R^2}

$$

其中用了 $\alpha \approx \sqrt{(p-1)/n}$ ： $\alpha = \sqrt{\tau/n}$ ， $\tau$ 取 $\chi^2_{p-1}$ 的 $q$ 分位数， $p$ 大时 $\tau \approx (p-1) - z_q\sqrt{2(p-1)}$ 。
代 $n = 1.25\times 10^8$ 、 $p = 20$ 、 $\eta = 0.1$ 、 $R \approx 1.1$ ： $\alpha \approx 3.9\times 10^{-4}$ ， $Q_k + \beta_k^2 \approx 28$ ，得

$$

\Delta_k \le 2.5 \times 10^{-5} \cdot \frac{\Vert \nabla f\Vert_2}{L}

$$

即**理论上模型可信的半径比论文默认 $\Delta_0 = 10^{-1}$ 小约 4 个数量级**。而 §4.3 显示精度在 $\Delta_0 \in [10^{-3}, 3\times 10^{-1]}$ 上平坦——也就是说，实验里起作用的那一段恰是理论**保证失效**的区域。
推断（非作者主张）：在 $\Delta_0 = 10^{-1}$ 处 MpSub 的表现并非由 $\rho_k \ge \eta$ 的信赖域机制驱动，而更可能由 (2.13) 的"只要下降就接受"退化成带动量方向的随机搜索提供。
可证伪诊断：记录 $\Delta_k$ 轨迹。若机制为信赖域， $\Delta_k$ 应先收缩到 $\sim 10^{-5}$ 再稳定扩张；若 $\Delta_k$ 单调贴到 $\Delta_{\min}$ 而精度仍平坦，则机制解释被推翻。

### 5.2 FP32 分辨率造成的吸收态

$\Delta_k$ 触到 $\Delta_{\min} = 10^{-12}$ 时，扰动量 $\Delta_k d_{i,a} \sim 10^{-12} \times n^{-1/2} \approx 3\times 10^{-17}$ ，而 FP32 权重 $\sim 10^{-2}$ 的分辨率约 $10^{-9}$ 。正负扰动点权重逐位相同 $\Rightarrow$ $f_i^+ = f_i^-$ $\Rightarrow$ $g_k = 0$ $\Rightarrow$ 走 Algorithm 2.1 第 12 行：不评估试探点、 $\Delta_{k+1} = \max(\gamma_1\Delta_k, \Delta_{\min}) = \Delta_{\min}$ 。此后每步都停在同一状态：**半径与迭代点双双冻结**（ $x_{k+1} = x_k$ 因为 $s_s$ 未定义/分支直接 continue）。
从 $\Delta_0 = 10^{-1}$ 以 $\gamma_1 = 0.5$ 连续失败到 $10^{-12}$ 只需 $\approx 47$ 步，而实验只有 200 步。这使"实验未冻结"成为一个**可检验的事实断言**：若某次运行连续失败约 47 次，训练即停止推进。论文与代码均未报告 $\Delta_k$ 轨迹，列为待核 + 复现首要观测项。
理论侧对应： $\Delta_{\min} = 0$ 时该状态退化为 $\Delta_k \to 0$ 的几何收缩，与 (3.13)(3.22) 的"收敛到稳定点"在数学上一致，但**其收敛机制本身（半径 $\to 0$ ）就是实际停滞**——这是所有信赖域全局收敛定理的通病，在零阶 + 高维有限差分下被 $\sqrt{p/n}$ 因子放大到数值不可达。

### 5.3 跳过正交化的代价，以及 $p$ 饱和的解释

- 论文对 $p > 20$ 不再提升的解释是"捕获能量已足够"（引 Lemma 3.3）。但 (3.6) 说期望捕获能量 $\propto (p-1)/n$ **线性增长**，按此解释精度应继续改善。
- 本稿的替代解释：非正交 frame 使 $\beta_k^2 \le p R^2$ 随 $p$ 线性增长，进入 (3.8)(3.9) 的误差项，抵消了能量增益；由 5.1 的公式，provably-informative 半径 $\propto \sqrt{p}/p = 1/\sqrt p$ ，即**$p$ 越大，理论允许的半径越小**。
- 可证伪预测：若把 $d_2,\dots,d_p$ 改为关于 $d_1$ 正交归一（代价： $O(p^2 n)$ 逐元素运算 + $p^2$ 次张量重建，相对 $2p+2$ 次前向可忽略；且**不需要**额外前向评估），则 $\beta_k = 1$ 、 $Q_k = 1$ ，(5.1) 的半径阈值改善约 $\sqrt p$ 倍（ $p=20$ 时 $\sim 4.5$ 倍），且表 4.2 的 $p$-饱和点应右移。这是一个直接可测的改进方向，也是本论文最自然的后续实验。

### 5.4 主张-证据-限制（摘要相关）

| 作者主张 | 论文证据 | 评审推断 / 限制 | 置信度 |
| --- | --- | --- | --- |
| 无需学习率即可达到与调参 MeZO 可比的精度 | 表 4.1、图 4.2 右： $\Delta_0$ 跨 300 倍精度 0.655–0.678 | 成立（在 1 数据集 / 2 模型 / 3 种子 / 56 测试样本下）；"可比"未被证伪，"更优"未被证实 | 中 |
| 有限差分误差被半径线性控制 | Lemma 3.2 (3.3)(3.4)，证明完整 | 正确；但误差与 $n$ 无关而信号 $\propto 1/\sqrt n$ ，是高维下的结构性弱点 | 高 |
| 梯度范数 a.s. 收敛到零 | Thm 3.6 + 3.8，链完整 | 依赖 (3.1) 的 safeguard（含未给值的 $\kappa$ ）与 $\Delta_{\min}=0$ ，两者都与算法 2.1 / 实验设置不一致；Lemma 3.7 有一处未证 renewal（本稿 §3.7 已补证） | 中高（确定性目标下） |
| 每迭代 $2p+2$ 次评估、成本随 $p$ 线性 | §2 与表 4.2 时延列 | 成立；但训练内存并非零额外： $m_k + u_k$ 共 $2n$ 个 FP32 | 高 |
| 理论覆盖 LLM 微调 | §3 开头自述"固定确定性目标" | **不覆盖**： $f_k$ 随 minibatch 变化，作者自己声明不主张总体稳定点收敛 | 高 |

### 5.5 与最强近邻的差异（净增量）

- 相对 2D-MoSub [12]（ $p=2$ 二次插值）：把插值点数从 $\frac12(p+1)(p+2)$ （ $p=20$ 时 231）降到 $2p+2$ （42），代价是模型从二次降为线性——**放弃了曲率信息**，因此没有步长内蕴估计能力，全部尺度自适应压在 $\Delta_k$ 上。净增量成立，但"线性模型足够"这一前提论文只用实验间接支持。
- 相对 MeZO [9]：从单随机方向 $\to$ $p$ 维 frame + 信赖域尺度自适应。MeZO 的步长是超参，MpSub 的 $\Delta_k$ 是自校准——这是真实差异。
- 相对随机子空间信赖域族（Cartis–Fowkes–Shao [2]、Cartis–Roberts [3]、Kozak 等 [8]、Menickelly [10]）：这些工作在中等维度用**正交**随机基 + 梯度可用；MpSub 的差异化点是"高斯非正交 frame + 纯前向 + $n\sim 10^8$"，而 (2.5) 的近正交性论证正是为省掉 QR 而写。**待核**：论文未与 [2][3][10] 做定量条件对比（例如是否要求 $\mathbb{E}[\beta_k^2]$ 有界）。

---

## 6. 会话小结

- 本次为全局推理模式，覆盖第 2 节全部 (2.1)–(2.18) 与 Algorithm 2.1、第 3 节 Assumption 3.1 至 Theorem 3.8 全部 (3.1)–(3.22)、第 4 节实验数量关系。
- 补证了论文两处跳步：下降引理（§3.1）、Lemma 3.7 的 renewal（§3.7，鞅收敛定理替代，且把条件从 $q>1/2$ 放宽到 $q>0$ ）。
- 标注了三处实质缺口：(3.1) 与 (2.14) 不一致（safeguard 与 $\kappa$ 、 $\Delta_{\min}$ ）、provably-informative 半径与实验默认半径差 4 个数量级、FP32 下 $\Delta_{\min}$ 吸收态。
- 待核项： $C_1,C_2,C_3,M$ 是否保证 (2.17) 单射；MeZO 基线是"一点估计 + 缓存基准损失"还是"两点估计"（论文称两点，其原论文实现口径待查）； $\Delta_k$ 轨迹实测。
- 定稿已写入 `04-equation_problem/<标题>/全局推理.md`。任何一步都可以继续追问，或说"回到交互模式"逐个细讲。
