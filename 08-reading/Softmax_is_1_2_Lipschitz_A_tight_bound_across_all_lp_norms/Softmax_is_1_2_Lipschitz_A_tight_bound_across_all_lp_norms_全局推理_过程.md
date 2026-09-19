# 全局推理（过程稿）：Softmax is 1/2-Lipschitz

- 由 `paper-sgd-reading` 全局推理模式生成；按论文行文顺序连续推导，未逐单元等待提问。
- 定稿（可独立阅读版）：`../../04-equation_problem/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/全局推理.md`
- 本文件保留推导时的取舍、复核记录与逐处审计；公式规范已自查。

---

> 本文件由 `paper-sgd-reading` 技能的**全局推理模式**生成：按论文行文顺序连续推导全部编号公式、引理、定理及其附录证明，中途不停顿等待提问。
> 配套交互笔记：`Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms_精读笔记.md`（若尚未创建，说明本次会话未走交互模式）。
> 用户精读笔记：`../../03-notes/2026-09/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/精读.md`（只读，用于对齐术语；本文对其两处结论做了独立复算并给出修正）。

## 论文信息

- 标题：Softmax is 1/2-Lipschitz: A tight bound across all $\ell_p$ norms
- 作者：Pravin Nair（独作，IIT Madras, Electrical Engineering）
- 出处：arXiv:2510.23012；Published in Transactions on Machine Learning Research (12/2025)，OpenReview forum `6dowaHsa6D`
- 类型：**理论 / 理论算法**（文献校准型，无新算法、无新数据集）
- 本文核验到的外部原始表述（用于溯源，非论文正文）：
  - Gao & Pavel (2017) arXiv:1704.00805：Prop 1（$\sigma = \nabla\mathrm{lse}$）、Prop 2（式 27，Jacobian）、Prop 4（式 35，$\lambda$-Lipschitz）、Prop 3（式 33，单调性）、Cor 2 + Theorem 1（Baillon–Haddad）、Cor 3（$\lambda$-Lipschitz 且 $\frac{1}{\lambda}$-co-coercive；$\lambda=1$ 非扩张且 firmly nonexpansive；$\lambda\in(0,1)$ 压缩）—— 已用 `pdftotext` 从 arXiv PDF 直接抽取原文核对。
  - Qi et al. (2023) LipsFormer arXiv:2304.09856：Theorem 1 的两个分支，$\ell_\infty$ 分支与 $\ell_2$ 分支的完整表达式与 $\epsilon$ 的定义 —— 已从 ar5iv HTML 抽取核对。

## 记号约定（与论文的差异，一次性说明）

论文用**粗体**表示向量。本文不用粗体，改用小写字母表示向量（$x, y, s, v \in \mathbb{R}^n$）、大写字母表示矩阵（$A, M, J \in \mathbb{R}^{n\times n}$），并在需要时用上下标区分实例。$\mathbf{1}$ 表示全 1 向量。除此之外与论文记号一致：$\sigma_\lambda$ 为带温度 softmax，$s = \sigma_\lambda(x)$，$\Delta_n$ 为概率单纯形，$\Delta_n^\circ$ 为其内部，$\partial\Delta_n$ 为其边界，$\Vert\cdot\Vert_p$ 同时表示向量 $\ell_p$ 范数与矩阵的 $\ell_p$ 诱导算子范数。

## 推理计划与依赖关系

论文证明链只有两个真正独立的技术件：**(i) softmax Jacobian 的行绝对和可解析求出且被 $1/2$ 控制**，**(ii) $\ell_1$–$\ell_\infty$ 插值不等式**。其余全部是这两个件的机械后果。依赖图如下（箭头 = 被使用）：

```
式(1) 定义 + 平移不变性 + 满射性 ──┐
                                  ├─→ Lemma 3 (式 4: 变分式) ──┐
Lemma 1 (L_p = sup_x ‖J_f(x)‖_p) ─┤                           │
Lemma 2 (J = λ(diag(s) − ssᵀ)) ───┘                           ├─→ Theorem 1(a) (‖J‖_p ≤ 1/2)
Prop 1(a)(b) (列和/行和) ──┐                                  │        │
                           ├─→ Prop 1(c) (插值) ──────────────┘        │
                                                                       ├─→ Theorem 1(b) (全局界 λ/2)
                                                                       │        │
Thm 1(a) + Prop1(a)(b) ─→ Prop 2(a) (p=1,∞ 内点达到 1/2)              │        ├─→ Cor 1 (Gao-Pavel Cor 3 改进)
                                                                       │        ├─→ Thm 2 (SCSA 界, 审计)
Lemma 4 (支撑集刻画) ─→ Prop 2(b) (1<p<∞ 只取极限) ←─ 式(8) 严格性     │        └─→ Thm 3 (DSFP, 审计)
        ↑                                                                   │
   Prop 1(c) 取等条件 ──────────────────────────────────────────────────────┘
Example 1 / Example 2 ─→ 紧性的构造性证据
式(7) ─→ §5 经验验证（方法论审计）
```

处理顺序：Part 0 前置件（式 1、定义 2.1–2.4、Lemma 1、Lemma 2）→ Part 1 范数工具（Prop 1）→ Part 2 主定理（Lemma 3、Thm 1）→ Part 3 紧性与可达性（Prop 2、Lemma 4、Example 1/2）→ Part 4 下游结果（Cor 1、Thm 2、Thm 3、式 7）→ Part 5 骨架总结与主张审计。

---

## Part 0：前置件

### 单元 0.1 式 (1)：softmax 定义，以及论文没写但后面要用到的三条性质

论文式 (1) 定义 $\sigma_\lambda : \mathbb{R}^n \to \Delta_n^\circ$：

$$
\sigma_\lambda(x)_i = \frac{\exp(\lambda x_i)}{\sum_{j=1}^{n} \exp(\lambda x_j)}, i = 1, \ldots, n .
$$

逐个符号（按式子从左到右）：

- $\sigma_\lambda$ —— softmax 算子，下标 $\lambda$ 是其唯一参数。属于**模型参数/超参数**（在强化学习里是策略的温度倒数，在注意力里通常是 $1/\sqrt{d_k}$ 或 $1/\tau$）。
- $\lambda > 0$ —— inverse-temperature（逆温度）系数。$\lambda$ 越大输出越"尖"（集中在最大分量），越小越平。它是**已知常数**（分析时给定）。
- $x \in \mathbb{R}^n$ —— 输入向量，论文里叫 logits / 注意力分数 / Q 值。$x_i$ 是它的第 $i$ 个分量。
- $i$ —— 输出分量的指标，取值 $1,\dots,n$；$j$ 是求和指标（哑标），同样取 $1,\dots,n$。
- $n$ —— softmax 归一化的维度（分类问题的类别数、注意力的一行 token 数、动作空间大小）。**它是变量**，这也是"$p,n$ 一致"这一卖点要覆盖的对象之一。
- $\exp(\lambda x_i)$ —— 指数函数，保证分子严格为正。
- 分母 $\sum_j \exp(\lambda x_j)$ —— 配分函数（partition function），保证输出各分量和为 1。
- $\Delta_n = \{u \in \mathbb{R}^n : u_i \ge 0, \sum_i u_i = 1\}$ —— 概率单纯形；$\Delta_n^\circ = \{u : u_i > 0, \sum_i u_i = 1\}$ 是其**相对内部**（在仿射超平面 $\{\sum u_i = 1\}$ 内的内部）。因为分子恒正，$\sigma_\lambda$ 的值域确实落在 $\Delta_n^\circ$，取不到边界 $\partial\Delta_n = \{u : \exists i, u_i = 0\}$。

这一"值域是开集 $\Delta_n^\circ$、不含边界"的事实后面承担实质作用：Prop 2(b) 的极值点恰好在边界上，于是"上确界存在但取不到"。

现在补论文**未显式列出、但 Lemma 3 与 Theorem 1(b) 都依赖**的三条性质，并逐条证明。

**性质 0.1a（平移不变性）。** 对任意常数 $c \in \mathbb{R}$：

$$
\sigma_\lambda(x + c\mathbf{1}) = \sigma_\lambda(x) .
$$

证明：分子乘 $\exp(\lambda c)$，分母每一项也乘 $\exp(\lambda c)$，整体约掉。

**性质 0.1b（$\lambda$ 可提到自变量上）。**

$$
\sigma_\lambda(x) = \sigma_1(\lambda x) .
$$

证明：把 $\lambda x_i$ 看成一个整体即为 $\sigma_1$ 的定义式。

这条把"任意 $\lambda$ 的界"归约成"$\lambda=1$ 的界乘一个 $\lambda$"，是 Theorem 1(a) 只需处理 $\sigma_1$、Theorem 1(b) 再补 $\lambda$ 的原因。

**性质 0.1c（满射性：$\sigma_\lambda(\mathbb{R}^n) = \Delta_n^\circ$）。** 对任意 $s \in \Delta_n^\circ$，取 $x_i = \ln s_i$（因为 $s_i>0$，$\ln s_i$ 有限，故 $x\in\mathbb{R}^n$），则

$$
\sigma_1(x)_i = \frac{\exp(\ln s_i)}{\sum_j \exp(\ln s_j)} = \frac{s_i}{\sum_j s_j} = s_i ,
$$

最后一步用了 $\sum_j s_j = 1$。

这条论文完全没写，但 **Lemma 3 把 $\sup_{x\in\mathbb{R}^n}$ 换成 $\sup_{s\in\Delta_n^\circ}$ 必须同时用到性质 0.1c（每个 $s$ 都有原像）和 Jacobian 只通过 $s$ 依赖 $x$（Lemma 2）**。属于论文的实质跳步之一，见 Part 2。

### 单元 0.2 定义 2.1 / 2.2 / 2.4：$\ell_p$ 范数、诱导算子范数、全局与局部 Lipschitz 常数

**定义 2.1。** 对 $1 \le p < \infty$，$\Vert x \Vert_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$；对 $p = \infty$，$\Vert x \Vert_\infty = \max_{1\le i\le n}|x_i|$。其中 $|x_i|$ 是绝对值，$p$ 是范数阶（本论文要求 $p \ge 1$，以保证三角不等式成立、$\Vert\cdot\Vert_p$ 真是范数而非仅准范数）。

矩阵的诱导范数：$\Vert A \Vert_p = \sup_{v\ne 0} \frac{\Vert Av\Vert_p}{\Vert v\Vert_p} = \sup_{\Vert v\Vert_p = 1}\Vert Av\Vert_p$。符号 $A \in \mathbb{R}^{m\times n}$，$Av$ 是矩阵乘向量。这里论文说"overload $\Vert\cdot\Vert_p$ 同时表示向量范数与诱导范数"——约定，不是错误。

一个论文未说但推导要用的小事实：因为 $v \mapsto \Vert Av\Vert_p$ 连续、单位球 $\{v : \Vert v\Vert_p = 1\}$ 在有限维是**紧集**，所以上确界其实是最大值，可写成 $\max$。这个"$\sup$ 可换成 $\max$"在后面 Lemma 4 修补严格性时会用到（论文正文没写，导致 Case (iii) 的证明链在"逐点严格推得上确界严格"这一环缺一跳）。

**定义 2.2（式 2、式 3）。** $f : \mathbb{R}^n \to \mathbb{R}^m$ 关于 $\Vert\cdot\Vert_p$ 的 Lipschitz 常数是满足

$$
\Vert f(x) - f(y) \Vert_p \le L_p \Vert x - y \Vert_p , \forall x, y \in \mathbb{R}^n
$$

的最小 $L_p \ge 0$，等价地由式 (3) 给出

$$
L_p = \sup_{x, y \in \mathbb{R}^n, x\ne y} \frac{\Vert f(x) - f(y) \Vert_p}{\Vert x - y \Vert_p} .
$$

符号：$x, y$ 是两个自变量取值，$L_p$ 是待求的**未知量**（本文的目标就是把它算出来），$\sup$ 表示所有点对上的比值上确界。$f$ 为 contractive 指 $L_p<1$，non-expansive 指 $L_p\le1$。

**定义 2.4（局部 Lipschitz 常数）。**

$$
L_p(x) := \limsup_{y \to x, y\ne x} \frac{\Vert f(y) - f(x) \Vert_p}{\Vert y - x \Vert_p} .
$$

符号：$\limsup_{y\to x}$ 是"$y$ 沿任意方式趋于 $x$ 时，比值 eventual 的上极限"，即任意小邻域内比值的最大可能极限。若 $f$ 在 $x$ 可微，则 $L_p(x) = \Vert J_f(x)\Vert_p$。

这一点需要展开，因为论文当"已知"使用。$f$ 在 $x$ 可微意味着

$$
f(y) - f(x) = J_f(x)(y - x) + r(y)
$$

其中 $r(y)$ 是 Fréchet 余项，满足 $\frac{\Vert r(y) \Vert_p}{\Vert y - x \Vert_p} \to 0$（当 $y \to x$）。于是

$$
\frac{\Vert f(y) - f(x) \Vert_p}{\Vert y - x \Vert_p} = \left\Vert J_f(x)\frac{y-x}{\Vert y-x\Vert_p} \right\Vert_p + o(1) .
$$

当 $y$ 遍历 $x$ 的邻域时，$u = \frac{y-x}{\Vert y-x\Vert_p}$ 恰好遍历单位球面 $\{\Vert u\Vert_p = 1\}$（任意方向、任意小步长都取得到），故上极限就是 $\max_{\Vert u\Vert_p=1}\Vert J_f(x)u\Vert_p = \Vert J_f(x)\Vert_p$。证毕。注意这里"$u$ 遍历整个单位球面"是关键：如果定义域不是开集（例如限制在单纯形内），就不能取遍所有方向，$\Vert J_f(x)\Vert_p$ 只是 $L_p(x)$ 的上界而非等式——论文定义在 $\mathbb{R}^n$ 上，所以没问题。

### 单元 0.3 Lemma 1：$L_p = \sup_x L_p(x) = \sup_x \Vert J_f(x)\Vert_p$

论文引 Hytönen et al. (2016) 直接陈述。两个方向都要证，方向不同难度也不同。

**方向 $\le$（$L_p \ge \sup_x \Vert J_f(x)\Vert_p$）。** 由式 (3)，对任意固定的 $x$ 和任意 $y \ne x$，

$$
\frac{\Vert f(y) - f(x) \Vert_p}{\Vert y - x \Vert_p} \le L_p .
$$

两边取 $\limsup_{y\to x}$，左端即 $L_p(x) = \Vert J_f(x)\Vert_p$（单元 0.2 末），故 $\Vert J_f(x)\Vert_p \le L_p$ 对一切 $x$ 成立，取 $\sup_x$ 得

$$
\sup_{x \in \mathbb{R}^n} \Vert J_f(x) \Vert_p \le L_p .
$$

**方向 $\ge$（$L_p \le \sup_x \Vert J_f(x)\Vert_p$）。** 记 $K := \sup_x \Vert J_f(x)\Vert_p$。任取 $x \ne y$，令线段 $z(t) = x + t(y - x)$，$t \in [0,1]$，并定义标量函数

$$
h(t) := \left\Vert f(z(t)) - f(x) \right\Vert_p .
$$

$h$ 是两个 Lipschitz 函数的复合（$t \mapsto f(z(t))$ 局部 Lipschitz，范数映射 Lipschitz 常数为 1，因为 $|\Vert a\Vert_p - \Vert b\Vert_p| \le \Vert a-b\Vert_p$ 是三角不等式的直接推论），故 $h$ 绝对可微几乎处处存在。对几乎处处的 $t$：

$$
h'(t) \le \left\Vert \frac{\mathrm{d}}{\mathrm{d}t} f(z(t)) \right\Vert_p = \left\Vert J_f(z(t)) (y - x) \right\Vert_p \le K \Vert y - x \Vert_p ,
$$

第一个不等号是"范数的导数以方向的范数导数为上界"（链式法则加反向三角不等式），第二个不等号是 $K$ 的定义结合诱导范数定义。两边从 $0$ 积到 $1$：

$$
h(1) - h(0) \le K \Vert y - x \Vert_p \int_0^1 \mathrm{d}t = K \Vert y - x \Vert_p ,
$$

而 $h(0) = 0$、$h(1) = \Vert f(y) - f(x)\Vert_p$，故 $\Vert f(y)-f(x)\Vert_p \le K\Vert y-x\Vert_p$ 对一切 $x,y$ 成立，由式 (3) 的最小性 $L_p \le K$。

两方向合并即 Lemma 1。这里必须强调**它要求 $f$ 在整个 $\mathbb{R}^n$ 上连续可微且定义域凸**（论文说 $C^1$，凸性由 $\mathbb{R}^n$ 自动满足）：若定义域非凸，线段论证会跑出定义域，只得到 $L_p \le K$ 的局部版本。softmax 满足前提，故可用。

**Lemma 1 的一个直接后果（论文用它做"不能更小"的论证）**：$L_p = \sup_x \Vert J_f(x)\Vert_p$ 意味着只要找到**一个** $x$ 使 $\Vert J_f(x)\Vert_p \ge c$，就有 $L_p \ge c$。这是排除 Xu et al. (2022) 的 $1/4$ 的最短路径（见单元 2.4）。

### 单元 0.4 Lemma 2：softmax 的 Jacobian

目标：$J_{\sigma_\lambda}(x) = \lambda\left(\mathrm{diag}(s) - s s^{\top}\right)$，其中 $s = \sigma_\lambda(x)$。论文引 Gao & Pavel 的 Prop 2，但这里手动求导一遍，因为后面所有代数都建立在这个逐元素形式上。

记 $Z = \sum_j \exp(\lambda x_j)$（配分函数），则 $\sigma_\lambda(x)_i = \exp(\lambda x_i)/Z_i$ 的分母与 $i$ 无关。先算两个原料：

$$
\frac{\partial}{\partial x_j} \exp(\lambda x_i) = \lambda \exp(\lambda x_i)  \delta_{ij}, \frac{\partial Z}{\partial x_j} = \lambda \exp(\lambda x_j),
$$

其中 $\delta_{ij}$ 是 Kronecker 记号：$i = j$ 时为 1，否则为 0。用商法则：

$$
\frac{\partial \sigma_i}{\partial x_j} = \frac{\lambda \exp(\lambda x_i)\delta_{ij} \cdot Z - \exp(\lambda x_i)\cdot \lambda \exp(\lambda x_j)}{Z^2} = \lambda \frac{\exp(\lambda x_i)}{Z}\delta_{ij} - \lambda \frac{\exp(\lambda x_i)}{Z}\cdot\frac{\exp(\lambda x_j)}{Z} .
$$

两项恰好都是 $\sigma$ 的分量，于是

$$
\left[J_{\sigma_\lambda}(x)\right]_{ij} = \lambda\left(s_i \delta_{ij} - s_i s_j\right),
$$

即对角元为 $\lambda s_i (1 - s_i)$，非对角元（$i \ne j$）为 $-\lambda s_i s_j$。

写成矩阵形式即 $\lambda(\mathrm{diag}(s) - s s^{\top})$。符号：$\mathrm{diag}(s)$ 是把 $s$ 放在对角线、其余为 0 的方阵；$s s^{\top}$ 是外积，第 $(i,j)$ 元为 $s_i s_j$。

三条立即可用、且是全文真正承重的结构性质：

1. **对称性**：$J_{\sigma_\lambda}(x)$ 对称（$\mathrm{diag}(s)$ 与 $s s^{\top}$ 都对称）。这是 Theorem 1(a) 里 $\Vert J\Vert_1 = \Vert J\Vert_\infty$ 的唯一原因，也是全文"$p$ 一致"的起点。
2. **只通过 $s$ 依赖 $x$**：Jacobian 是 $s = \sigma_\lambda(x)$ 的函数，不是 $x$ 的其它函数的显式依赖。这是 Lemma 3 换元的第二个原料。
3. **行和为 0**：$\sum_j [J]_{ij} = \lambda(s_i - s_i\sum_j s_j) = \lambda(s_i - s_i) = 0$，即 $J\mathbf{1} = 0$。这与性质 0.1a（平移不变性）是同一件事的两个侧面，也说明 $\mathbf{1}$ 是零特征值特征向量。论文没用到这一条，但它解释了为什么 $M(s)$ 是半正定的（它是分类分布 $\mathrm{Cat}(s)$ 的协方差矩阵：$\mathbb{E}[(e_I - s)(e_I - s)^\top] = \mathrm{diag}(s) - ss^\top$，$e_I$ 是 one-hot 向量）。半正定性给了 $\Vert M(s)\Vert_2 = \lambda_{\max}(M(s)) \le \mathrm{tr}(M(s)) = 1 - \sum_i s_i^2 \le 1 - 1/n$（最后一步 Cauchy–Schwarz：$(\sum s_i)^2 \le n\sum s_i^2$），这正是 LipsFormer 在 $\ell_2$ 分支用 $(n-1)/n$ 的来源，也是论文 Remark 里 Yudin et al. 那条路线的骨架。可见"$1/2$"与"$(n-1)/n$"的差距，本质是"用不用迹"的差距。

---

---

## Part 1：范数工具（Proposition 1）

### 单元 1.1 Prop 1(a)(b)：$\ell_1$ 诱导范数 = 最大列绝对和，$\ell_\infty$ = 最大行绝对和

设 $A = (A_{ij}) \in \mathbb{R}^{n\times n}$（论文只写方阵，实际矩形 $m\times n$ 亦成立，此处保持方阵以贴合 $J_{\sigma_1}$）。

**(a)** $\Vert A \Vert_1 = \max_{1\le j\le n}\sum_{i=1}^n |A_{ij}|$。

上界：

$$
\Vert Ax \Vert_1 = \sum_{i=1}^n \left| \sum_{j=1}^n A_{ij} x_j \right| \le \sum_{i=1}^n \sum_{j=1}^n |A_{ij}| |x_j| = \sum_{j=1}^n |x_j| \left( \sum_{i=1}^n |A_{ij}| \right) \le \left( \max_{j} \sum_{i} |A_{ij}| \right) \Vert x \Vert_1 ,
$$

三个步骤依次是：三角不等式、交换两个有限和的顺序、用 $\sum_i|A_{ij}| \le \max_k\sum_i|A_{ik}|$ 一致放缩。取 $\Vert x\Vert_1 = 1$ 再上确界得 $\Vert A\Vert_1 \le \max_j\sum_i|A_{ij}|$。

下界（可达性，论文略去，这里必须补，否则"$=$"没证）：设 $j^\star$ 是列和最大的列指标，取 $x = e_{j^\star}$（第 $j^\star$ 个标准基向量），则 $\Vert x\Vert_1 = 1$ 且 $Ax$ 就是 $A$ 的第 $j^\star$ 列，故 $\Vert Ax\Vert_1 = \sum_i|A_{ij^\star}| = \max_j\sum_i|A_{ij}|$。两侧夹住，等号成立。

**(b)** $\Vert A \Vert_\infty = \max_{1\le i\le n}\sum_{j=1}^n |A_{ij}|$。

上界：对固定行 $i$，$\left|\sum_j A_{ij}x_j\right| \le \sum_j|A_{ij}||x_j| \le \left(\sum_j|A_{ij}|\right)\max_j|x_j| \le \left(\max_i\sum_j|A_{ij}|\right)\Vert x\Vert_\infty$；再对 $i$ 取最大。

下界：设 $i^\star$ 为行和最大的行，取 $x_j = \mathrm{sgn}(A_{i^\star j})$（$+1$ 或 $-1$，若 $A_{i^\star j} = 0$ 任取 $1$），则 $\Vert x\Vert_\infty = 1$ 且 $\left|(Ax)_{i^\star}\right| = \sum_j|A_{i^\star j}|$。等号成立。

符号补充：$e_j$ 是第 $j$ 个分量为 1、其余 0 的向量；$\mathrm{sgn}(t)$ 为符号函数。这两步是"诱导范数在 $\ell_1/\ell_\infty$ 端点有闭式"的标准结论，出处 Golub & Van Loan (2013)。

**(b) 的一个直接推论，对全文很重要**：若 $A$ **对称**，则 $A$ 的第 $i$ 行绝对和等于 $A^\top$ 的第 $i$ 列绝对和，于是 $\Vert A\Vert_\infty = \Vert A^\top\Vert_1 = \Vert A\Vert_1$。Theorem 1(a) 的第一步"$\Vert J_{\sigma_1}(x)\Vert_1 = \Vert J_{\sigma_1}(x)\Vert_\infty$"就是这么来的——它完全建立在单元 0.4 的对称性上，不依赖 softmax 的其它任何细节。

### 单元 1.2 Prop 1(c)：$\ell_1$–$\ell_\infty$ 插值不等式（附录 A.1 逐步展开）

**命题。** 对 $1 < p < \infty$：

$$
\Vert A \Vert_p \le \Vert A \Vert_1^{1/p} \Vert A \Vert_\infty^{1 - 1/p} .
$$

符号：$\Vert A\Vert_1$、$\Vert A\Vert_\infty$ 是单元 1.1 的两个闭式量；指数 $1/p$ 与 $1-1/p$ 互补（和为 1），所以这是一个**几何平均型**界。$p=1$ 与 $p=\infty$ 时退化成恒等式（论文说"trivially satisfies"，确实如此：$1/p = 1$、$1-1/p = 0$，右边就是 $\Vert A\Vert_1$）。

论文说这是 Riesz–Thorin 插值定理的特例，但给了自包含证明。下面把附录 A.1 的每一步补全，包括它一笔带过的"By Hölder's inequality"。

固定 $x \in \mathbb{R}^n$，$1 < p < \infty$。第一步，按定义展开：

$$
\Vert Ax \Vert_p^p = \sum_{i=1}^n \left| (Ax)_i \right|^p .
$$

第二步，逐行三角不等式：$\left|(Ax)_i\right| = \left|\sum_j A_{ij}x_j\right| \le \sum_j |A_{ij}||x_j|$。因为 $t \mapsto t^p$ 在 $t\ge0$ 单调增，可以对不等式两边取 $p$ 次幂。

第三步（**论文只写"By Hölder's inequality"，这里展开**）：对每个固定的 $i$，把 $|A_{ij}||x_j|$ 拆成两个因子的乘积，即把 $|A_{ij}|$ 的幂 $1$ 写成 $\frac{p-1}{p} + \frac{1}{p}$：

$$
|A_{ij}| |x_j| = |A_{ij}|^{(p-1)/p} \cdot \left( |A_{ij}|^{1/p}|x_j| \right) .
$$

对共轭指数 $q = \frac{p}{p-1}$（满足 $\frac{1}{p}+\frac{1}{q}=1$）用 Hölder 不等式 $\sum_j a_j b_j \le \left(\sum_j a_j^{q}\right)^{1/q}\left(\sum_j b_j^{p}\right)^{1/p}$，取 $a_j = |A_{ij}|^{(p-1)/p}$、$b_j = |A_{ij}|^{1/p}|x_j|$：

$$
\sum_j a_j^{q} = \sum_j |A_{ij}| , \sum_j b_j^{p} = \sum_j |A_{ij}| |x_j|^{p} ,
$$

且 $1/q = \frac{p-1}{p} = 1 - 1/p$。代回即得论文那一步：

$$
\sum_{j=1}^n |A_{ij}||x_j| \le \left( \sum_{j=1}^n |A_{ij}| \right)^{1 - 1/p} \left( \sum_{j=1}^n |A_{ij}| |x_j|^p \right)^{1/p} .
$$

取 $p$ 次幂并对 $i$ 求和：

$$
\Vert Ax \Vert_p^p \le \sum_{i=1}^n \left( \sum_j |A_{ij}| \right)^{p-1} \left( \sum_j |A_{ij}| |x_j|^p \right) .
$$

第四步（论文式 (8) 的第一处"把 max 提出求和号"）：对每个 $i$，$\sum_j|A_{ij}| \le \max_k\sum_j|A_{kj}| = \Vert A\Vert_\infty$（最后等号用 Prop 1(b)），且剩余因子 $\sum_j|A_{ij}||x_j|^p \ge 0$，故

$$
\Vert Ax \Vert_p^p \le \Vert A \Vert_\infty^{p-1} \sum_{i=1}^n \sum_{j=1}^n |A_{ij}| |x_j|^p .
$$

**记这一步为 $(\star)$，它在 Part 3 的 Lemma 4 里是主角。** 论文式 (8) 的左半就是这个放缩的逐点版。

第五步（恒等变形，交换求和顺序）：

$$
\sum_{i=1}^n \sum_{j=1}^n |A_{ij}| |x_j|^p = \sum_{j=1}^n |x_j|^p \left( \sum_{i=1}^n |A_{ij}| \right) ,
$$

这一步是**等式**（有限和重排），不是不等式。

第六步（式 (8) 的第二处"提出 max"）：$\sum_i|A_{ij}| \le \max_k\sum_i|A_{ik}| = \Vert A\Vert_1$（Prop 1(a)），故

$$
\sum_j |x_j|^p \left(\sum_i|A_{ij}|\right) \le \Vert A \Vert_1 \sum_j |x_j|^p = \Vert A\Vert_1 \Vert x \Vert_p^p .
$$

合并第四至六步：

$$
\Vert Ax \Vert_p^p \le \Vert A \Vert_\infty^{p-1} \Vert A \Vert_1 \Vert x \Vert_p^p .
$$

两边开 $p$ 次方（$p>0$，保序）：$\Vert Ax\Vert_p \le \Vert A\Vert_\infty^{1-1/p}\Vert A\Vert_1^{1/p}\Vert x\Vert_p$。最后对 $\Vert x\Vert_p = 1$ 取上确界，即 Prop 1(c)。证毕。

**外部知识点的精确出处。** 论文称此不等式是 Riesz–Thorin 的特例。核对：Riesz–Thorin 给出 $\Vert T\Vert_{p_\theta\to q_\theta} \le \Vert T\Vert_{p_0\to q_0}^{1-\theta}\Vert T\Vert_{p_1\to q_1}^{\theta}$，取 $p_0=q_0=1$、$p_1=q_1=\infty$、$\frac{1}{p_\theta} = (1-\theta)\cdot 1 + \theta\cdot 0 = 1-\theta$，即 $\theta = 1-\frac1p$，得 $\Vert A\Vert_{p\to p}\le\Vert A\Vert_1^{1/p}\Vert A\Vert_\infty^{1-1/p}$ ——与论文的初等证明结论完全一致，论文自证是对的，不必依赖 Riesz–Thorin。**结论：这是直接套用经典结果的矩阵有限维特例，未做修改。**

**取等条件（附录没有讨论，但 Prop 2(b) 与 Lemma 4 全部依赖它）**。上面链条要处处取等，至少要：

- $(\star)$ 取等 $\Rightarrow$ 对**所有**满足 $\sum_j|A_{ij}||x_j|^p > 0$ 的行 $i$，都有 $\sum_j|A_{ij}| = \max_k\sum_j|A_{kj}|$（即"对该方向 $x$ 有贡献的行，行和全部相等"）；
- 第六步取等 $\Rightarrow$ 对所有满足 $x_j \ne 0$ 的列 $j$，都有 $\sum_i|A_{ij}| = \max_k\sum_i|A_{ik}|$。

对**算子范数**取上确界时，这些条件必须在某个极方向（或极方向序列）上成立。这个观察是理解 Lemma 4 "只有 $\mathrm{supp}(s)\le2$ 才能取等"的钥匙。

---

## Part 2：主定理

### 单元 2.1 Lemma 3：把全局常数写成单纯形上的变分问题（式 4）

**目标式 (4)**：

$$
L_p = \lambda \sup_{s \in \Delta_n^\circ} \left\Vert \mathrm{diag}(s) - s s^{\top} \right\Vert_p .
$$

符号：$L_p$ 是 $\sigma_\lambda$ 关于 $\ell_p$ 的全局 Lipschitz 常数；$s$ 现在是**积分变量**（遍历整个 $\Delta_n^\circ$），不再是某个特定输入的 softmax 输出。

推导：由 Lemma 1（单元 0.3），

$$
L_p = \sup_{x \in \mathbb{R}^n} \Vert J_{\sigma_\lambda}(x) \Vert_p .
$$

由 Lemma 2，$J_{\sigma_\lambda}(x) = \lambda M(s)$，其中记 $M(s) := \mathrm{diag}(s) - ss^\top$、$s = \sigma_\lambda(x)$。由诱导范数的正齐次性（$\Vert \lambda A\Vert_p = \lambda\Vert A\Vert_p$，$\lambda>0$）：

$$
L_p = \lambda \sup_{x \in \mathbb{R}^n} \left\Vert M(\sigma_\lambda(x)) \right\Vert_p .
$$

论文到此直接写"令 $s = \sigma_\lambda(x)$ 即得式 (4)"。**这一跳需要两条理由，论文一条都没写**：

1. $\left\Vert M(\sigma_\lambda(x))\right\Vert_p$ 只通过 $s = \sigma_\lambda(x)$ 依赖 $x$（单元 0.4 结构性质 2），故可以把 $x$ 的集合压成 $s$ 的取值集合；
2. $\sigma_\lambda$ 的值域正好是整个 $\Delta_n^\circ$（性质 0.1c 的满射性）。

有了这两条，$\{\sigma_\lambda(x) : x \in \mathbb{R}^n\} = \Delta_n^\circ$，于是 $\sup_x$ 与 $\sup_{s\in\Delta_n^\circ}$ 是同一个上确界。

**边界为什么可以"忽略"（论文的说法需要精确化）**。论文说"边界点不影响 Lipschitz 常数的表述，因为 softmax 输出分量严格为正"。这只是**表述层面**成立：$\sup$ 在开集 $\Delta_n^\circ$ 上取值，与在闭包 $\Delta_n$ 上取值的差别是"可能取不到"。因为 $s \mapsto M(s)$ 逐元素连续、$\Vert\cdot\Vert_p$ 连续，复合 $s\mapsto\Vert M(s)\Vert_p$ 连续，而 $\Delta_n$ 紧，所以

$$
\sup_{s \in \Delta_n^\circ} \Vert M(s) \Vert_p = \max_{s \in \Delta_n} \Vert M(s)\Vert_p ,
$$

即"开集上的上确界 = 闭包上的最大值"。这个等式后面在 Prop 2 里被反向使用：极值点若只落在 $\partial\Delta_n$，则开集上取不到，但上确界仍是同一个数。论文没写这层连续性论证，属于可补的跳步。

### 单元 2.2 Theorem 1(a)：$\Vert J_{\sigma_1}(x)\Vert_p \le 1/2$ 对一切 $p \ge 1$

**第一步：端点范数相等。** 由单元 0.4 结构性质 1，$J_{\sigma_1}(x)$ 对称；由单元 1.1 的推论，$\Vert J_{\sigma_1}(x)\Vert_1 = \Vert J_{\sigma_1}(x)\Vert_\infty$。这一步是"$p$ 一致"的全部来源，值得单独标注：它只用到对称，没用到 softmax——任何对称矩阵都有这个等式。

**第二步：行绝对和的闭式（全文技术核心）。** 由 Prop 1(b) 与 Lemma 2（取 $\lambda = 1$，$s = \sigma_1(x)$）：

$$
\Vert J_{\sigma_1}(x) \Vert_\infty = \max_{1\le i\le n} \sum_{j=1}^n \left| \left[J_{\sigma_1}(x)\right]_{ij} \right| .
$$

符号：$[J]_{ij}$ 表示矩阵 $J$ 的第 $(i,j)$ 元；$\max_i$ 是对 $n$ 个行和取最大。固定行 $i$，逐元素取绝对值。对角元 $\left|s_i(1-s_i)\right| = s_i(1-s_i) \ge 0$（因 $0 < s_i < 1$）；非对角元 $\left|-s_is_j\right| = s_is_j \ge 0$。故

$$
\sum_{j=1}^n \left|[J]_{ij}\right| = s_i(1-s_i) + \sum_{j\ne i} s_i s_j .
$$

处理第二项。因为 $\sum_j s_j = 1$，

$$
\sum_{j \ne i} s_i s_j = s_i \left( \sum_j s_j - s_i \right) = s_i (1 - s_i) .
$$

代回：

$$
\sum_{j=1}^n \left|[J]_{ij}\right| = 2 s_i (1 - s_i) .
$$

这正是论文附录 A.2 那条三行计算（它写成 $s_i(1-s_i) - s_i^2 + \sum_j s_is_j$；因 $\sum_j s_is_j = s_i^2$，减掉 $s_i^2$ 即去掉对角重复项，结果同为 $2s_i(1-s_i)$。两种写法等价，我核对过，论文此处无误）。于是

$$
\Vert J_{\sigma_1}(x) \Vert_1 = \Vert J_{\sigma_1}(x) \Vert_\infty = \max_{1\le i\le n} 2 s_i (1 - s_i) .
$$

**第三步：标量最值。** $g(t) = 2t(1-t)$ 是开口向下的二次函数，$g'(t) = 2 - 4t$，令其为 0 得 $t = 1/2$，$g(1/2) = 1/2$。因 $s_i \in (0,1)$，

$$
\max_i 2s_i(1-s_i) \le \frac{1}{2},
$$

且等号成立当且仅当 存在某个 $i$ 使 $s_i = 1/2$。这个"当且仅当"是 Prop 2(a) 的全部内容，先记住。

**第四步：插值覆盖中间 $p$。** 由 Prop 1(c) 与第二步、第三步：

$$
\left\Vert J_{\sigma_1}(x) \right\Vert_p \le \left(\frac12\right)^{1/p}\left(\frac12\right)^{1-1/p} = \frac12 .
$$

指数相加为 1，所以两个 $\frac12$ 的几何平均仍是 $\frac12$。**这就是"范数一致"的全部机制：不是对每个 $p$ 分别分析，而是两端点相等 + 插值。** 若两端点不等（一般非对称矩阵），插值只给一个依赖 $p$ 的界，"一致"就没了。所以本文的"$p$ 一致"是 Prop 1(c) 的机械推论，技术含量集中在端点等式（= 对称性）上。

**数值复核（我实际跑的）**：对 $n = 2,3,10$、每 $n$ 抽 400 个随机 $x$（尺度在 $0.1$–$30$ 间随机），$p \in \{1,2,3,\infty\}$，观测 $\Vert J_{\sigma_1}(x)\Vert_p$ 最大值均不超过 $0.500000$；行和恒等式 $\sum_j|[M(s)]_{ij}| = 2s_i(1-s_i)$ 在 2000 次随机抽样下 0 失配。

### 单元 2.3 Theorem 1(b)：全局界 $\lambda/2$

$$
\left\Vert \sigma_\lambda(x) - \sigma_\lambda(y) \right\Vert_p \le \frac{\lambda}{2} \Vert x - y \Vert_p , \forall x, y \in \mathbb{R}^n, \forall p \in [1,\infty].
$$

推导：由 Lemma 2，$J_{\sigma_\lambda}(x) = \lambda M(\sigma_\lambda(x))$，而 $\sigma_\lambda(x) = \sigma_1(\lambda x)$（性质 0.1b）。故

$$
\sup_{x} \Vert J_{\sigma_\lambda}(x)\Vert_p = \lambda \sup_{x} \left\Vert M(\sigma_\lambda(x)) \right\Vert_p = \lambda \sup_{z} \Vert J_{\sigma_1}(z)\Vert_p \le \frac{\lambda}{2} ,
$$

中间等号用了 $\lambda x$ 随 $x$ 遍历 $\mathbb{R}^n$。再由 Lemma 1（单元 0.3 的"方向 $\ge$"折线论证）把 Jacobian 上界转成全局 Lipschitz 界，得结论。

**审计：附录 A.2 有一处范围笔误。** 论文 A.2 的 Proof of Theorem 1(b) 先写"$\sup_x \Vert J_{\sigma_\lambda}(x)\Vert_p \le \lambda/2$ for all $1 < p < \infty$"，随后又写"$\le \frac{\lambda}{2}\Vert x-y\Vert_p$ for all $1\le p\le\infty$"。前者漏掉端点，而端点恰是 Theorem 1(a) 第一步直接算出的，结论无恙。排版笔误，不影响任何下游结果。

### 单元 2.4 "$1/4$ 不可能"：一个必须显式写出的反例

论文用 Prop 2 回答"能不能是 Xu et al. (2022) 报的 $1/4$"，但没给最短反例。补上：取 $n = 2$、$x = (0,0)$，则 $s = \sigma_1(0,0) = (1/2,1/2) \in \Delta_2^\circ$，此时 $M(s)$ 的四个元素为 $[M]_{11} = [M]_{22} = 1/4$、$[M]_{12} = [M]_{21} = -1/4$。它可以精确分解为

$$
M(s) = \frac{1}{4} u u^{\top}, u = (1, -1) \in \mathbb{R}^2 .
$$

这是秩 1 矩阵，非零特征值为 $\frac14 \Vert u\Vert_2^2 = \frac14\cdot 2 = \frac12$，故 $\Vert M(s)\Vert_2 = 1/2$。由 Lemma 1，$L_2 \ge 1/2$。所以任何小于 $1/2$ 的"全局 Lipschitz 常数"（含 $1/4$）都必然为假。$\lambda = 1$ 时 $1/4$ 错；正确值是 $\lambda/2$，$1/4$ 仅在 $\lambda = 1/2$ 这一个特定温度下才碰巧成立。

这条推理还顺带说明 $\lambda/2$ 不只是上界，而是最小可行常数（配合 Prop 2 的可达/逼近分类，见 Part 3）。

---

---

## Part 3：紧性与可达性（Prop 2、Lemma 4、Example 1/2）

Prop 2 是本文相对已有 $\ell_2$ 结果真正的增量所在：已有工作给出 $\le 1/2$，本文给出"什么时候恰好取到 $1/2$"的完整分类。

### 单元 3.1 Prop 2(a)：$p = 1$ 与 $p = \infty$ 时上确界在单纯形内部取到

**待证问题**：

$$
\sup_{x \in \mathbb{R}^n} \Vert J_{\sigma_1}(x) \Vert_p = \sup_{s \in \Delta_n^\circ} \Vert \mathrm{diag}(s) - s s^{\top} \Vert_p .
$$

左边是 $\sigma_1$ 的 Jacobian 诱导范数上确界，等号由 Lemma 3（单元 2.1）在 $\lambda = 1$ 时给出。

**证明分两块：上界 + 构造点。**

上界已由单元 2.2 第二步给出：$\Vert M(s)\Vert_1 = \Vert M(s)\Vert_\infty = \max_i 2s_i(1-s_i) \le 1/2$。

构造点：取 $x = (\ln(n-1), 0, 0, \ldots, 0) \in \mathbb{R}^n$。符号：第 1 个分量为 $\ln(n-1)$（要求 $n \ge 2$ 才有定义；$n=2$ 时它是 $\ln 1 = 0$），其余 $n-1$ 个分量为 0。代入式 (1)（$\lambda=1$）：

$$
s_1 = \frac{\exp(\ln(n-1))}{\exp(\ln(n-1)) + \sum_{j=2}^n \exp(0)} = \frac{n-1}{(n-1) + (n-1)} = \frac{1}{2} ,
$$

$$
s_j = \frac{1}{2(n-1)} > 0, ,  j = 2, \ldots, n .
$$

校验它确实在 $\Delta_n^\circ$：所有分量严格为正，且 $\sum_j s_j = \frac12 + (n-1)\cdot\frac{1}{2(n-1)} = \frac12+\frac12 = 1$ ✓。

由单元 2.2 第二步末尾的"当且仅当"：$s_1 = 1/2$ 使 $\max_i 2s_i(1-s_i) \ge 2\cdot\frac{1}{2}\cdot\frac{1}{2} = \frac{1}{2}$，结合上界即

$$
\Vert M(s) \Vert_1 = \Vert M(s) \Vert_\infty = \frac{1}{2} .
$$

当 $n \ge 3$ 时，其余行的行和 $2s_j(1-s_j) = \frac{1}{n-1}\left(1-\frac{1}{2(n-1)}\right) < \frac12$，故最大值由第 1 行单独取到；$n = 2$ 时 $s = (1/2, 1/2)$，两行同时取等，结果仍是 $1/2$。两种情形都不与上界 $\le 1/2$ 冲突。

**数值复核**：$n = 2, 3, 10, 50$ 时该构造点均给出 $s_1 = 0.5$、$\Vert M\Vert_1 = \Vert M\Vert_\infty = 0.5$ ✓。

**这条结果的确切含义（容易被读歪）**：它说"$p\in\{1,\infty\}$ 时，存在具体的输入 $x$ 使**局部** Lipschitz 常数恰为 $1/2$"。它**不**说"存在点对 $(x,y)$ 使全局比值 $\frac{\Vert\sigma(x)-\sigma(y)\Vert_p}{\Vert x-y\Vert_p}$ 恰好等于 $1/2$"——后者需要 Jacobian 界沿线段处处取等，是更强命题，论文没有主张，本文也不应替它主张。全局常数 $=1/2$ 由 Lemma 1 的"$L_p = \sup_x$"保证（上确界意义下）。

### 单元 3.2 Lemma 4 的充分方向：支撑集为 2 时的显式极方向

**Lemma 4（论文原述）**：设 $M(s) := \mathrm{diag}(s) - ss^\top$，$s \in \Delta_n$，$1 < p < \infty$，$n > 2$。则 $\Vert M(s)\Vert_p = 1/2$ 当且仅当 $s$ 是 $(1/2, 1/2, 0, \ldots, 0)$ 的某个排列。

**充分性（$\Leftarrow$）**。设 $s_i = s_j = 1/2$（$i \ne j$），其余为 0。则

$$
M(s) = \frac{1}{4}\left(e_i e_i^{\top} + e_j e_j^{\top} - e_i e_j^{\top} - e_j e_i^{\top}\right) = \frac{1}{4}(e_i - e_j)(e_i - e_j)^{\top} .
$$

符号：$e_k$ 是第 $k$ 个标准基向量；$(e_i-e_j)(e_i-e_j)^\top$ 是秩 1 外积。逐元素即 $M_{ii}=M_{jj}=1/4$、$M_{ij}=M_{ji}=-1/4$、其余 0，与论文 A.3 末尾一致。

要证 $\Vert M(s)\Vert_p \ge 1/2$，只需造一个 $\Vert v\Vert_p = 1$ 使 $\Vert M(s)v\Vert_p \ge 1/2$。取

$$
v = 2^{-1/p}(e_i - e_j) .
$$

先验归一化：$\Vert v \Vert_p^p = 2\cdot\left(2^{-1/p}\right)^p = 2\cdot 2^{-1} = 1$ ✓。再算像：

$$
M(s) v = \frac{1}{4}(e_i - e_j)\left[(e_i - e_j)^{\top} v\right], ,  (e_i-e_j)^{\top} v = 2^{-1/p}(1-(-1)) = 2^{1-1/p} ,
$$

故 $M(s)v = \frac14\cdot 2^{1-1/p}(e_i-e_j)$，其 $\ell_p$ 范数为

$$
\Vert M(s) v \Vert_p = \frac{1}{4}\cdot 2^{1-1/p}\cdot \Vert e_i - e_j\Vert_p = \frac{1}{4}\cdot 2^{1-1/p}\cdot 2^{1/p} = \frac{1}{4}\cdot 2 = \frac{1}{2} .
$$

（用到 $\Vert e_i-e_j\Vert_p = (1^p+1^p)^{1/p} = 2^{1/p}$。）结合 Theorem 1(a) 的 $\Vert M(s)\Vert_p\le 1/2$，得 $\Vert M(s)\Vert_p = 1/2$。✓

这里有一个值得点出的结构事实：**这个极方向 $v$ 对一切 $1<p<\infty$ 都工作**，且 $2^{1-1/p}\cdot 2^{1/p} = 2$ 与 $p$ 无关。所以"边界点 $(1/2,1/2,0,\dots)$ 的诱导范数对所有 $p$ 都是 $1/2$"不是巧合，而是 $v$ 与 $M(s)$ 都只在一个二维 $\{+,-\}$ 块上活动、$\ell_p$ 归一化的两个因子恰好互补所致。

### 单元 3.3 Lemma 4 的必要方向 + 论文证明链里的一个真跳步

**必要性（$\Rightarrow$）**。设 $\Vert M(s)\Vert_p = 1/2$。

**第一步（端点范数被迫取等）。** 由 Prop 1(c) 与 Theorem 1(a)：

$$
\frac{1}{2} = \Vert M(s) \Vert_p \le \Vert M(s) \Vert_1^{1/p} \Vert M(s) \Vert_\infty^{1-1/p} \le \left(\frac{1}{2}\right)^{1/p}\left(\frac{1}{2}\right)^{1-1/p} = \frac{1}{2} .
$$

两端相同，故中间必取等；又 $\Vert M(s)\Vert_1 = \Vert M(s)\Vert_\infty$（对称），两个 $\le 1/2$ 的非负数的几何平均等于 $1/2$ 只能是二者都等于 $1/2$：

$$
\Vert M(s) \Vert_1 = \Vert M(s) \Vert_\infty = \frac{1}{2} .
$$

**第二步（分量必须有一个等于 $1/2$）。** 由单元 2.2 第二步，$\Vert M(s)\Vert_\infty = \max_i 2s_i(1-s_i)$。解标量方程 $2t(1-t) = 1/2$：等价于 $4t^2 - 4t + 1 = 0$，即 $(2t-1)^2 = 0$，唯一解 $t = 1/2$。故存在 $i_1$ 使 $s_{i_1} = 1/2$。

**第三步（按支撑集大小分类）。** 回忆定义 $\mathrm{supp}(s) = \{i : s_i \ne 0\}$（论文 Definition A.1）。

- **Case (i)，$|\mathrm{supp}(s)| = 1$**：$s = e_i$，则 $M(s) = \mathrm{diag}(e_i) - e_ie_i^\top = 0$，范数为 0，矛盾。
- **Case (ii)，$|\mathrm{supp}(s)| = 2$**：$s = (a, 1-a, 0,\ldots)$（重排后）。$\max_i 2s_i(1-s_i) = 2a(1-a) = 1/2$ 由第二步迫使 $a = 1/2$，即 $s$ 是 $(1/2,1/2,0,\ldots,0)$ 的排列 ✓。
- **Case (iii)，$|\mathrm{supp}(s)| \ge 3$**：必须排除。

Case (iii) 是全文最细的一段，论文的处理不完整。先复述论文的论证，再指出缺口并补齐。

**论文论证**：由第一步，$\Vert M(\hat s)\Vert_1 = \Vert M(\hat s)\Vert_\infty = 1/2$，故插值不等式必须取等；论文转而证明式 (8) 的"把 max 提出求和号"这一步严格。因 $|\mathrm{supp}(\hat s)|\ge3$ 且 $\hat s_{i_1}=1/2$（唯一，因为若再有 $\hat s_{i_3}=1/2$ 则质量已用完，与 $|\mathrm{supp}|\ge3$ 矛盾），存在 $i_2\in\mathrm{supp}(\hat s)$、$i_2\ne i_1$，$0<\hat s_{i_2}<1/2$，于是该行绝对和 $2\hat s_{i_2}(1-\hat s_{i_2}) < 1/2 = \max_i\sum_j|M_{ij}|$，式 (8) 严格。

**缺口所在**：上面证的是"Prop 1(c) 证明链中，对**行指标 $i_2$** 的那一次放缩严格"。但 $\Vert M\Vert_p$ 是对 $v$ 取 $\sup$ 得到的**算子**量，逐点放缩严格并不自动蕴含上确界严格——理论上仍可能存在一列方向 $v^{(k)}$ 使严格量趋于 0，从而 $\sup$ 仍等于 $1/2$。论文从"逐点严格"直接跳到"$\Vert M(\hat s)\Vert_p < 1/2$"，缺一次一致化。

**补齐（三步，只用有限维事实）**：

(1) **上确界可取到。** $v \mapsto \Vert M(s)v\Vert_p$ 连续，单位球面 $\{v : \Vert v\Vert_p = 1\}$ 在有限维紧，故 $\Vert M(s)\Vert_p = \max_{\Vert v\Vert_p=1}\Vert M(s)v\Vert_p$，存在极方向 $v^\star$。（这一条单元 0.2 已指出论文没写。）

(2) **对每个单位 $v$，链条 $(\star)$ 处严格。** 记 $R_i = \sum_j |M_{ij}|$、$T_i(v) = \sum_j |M_{ij}||v_j|^p$，则 $(\star)$ 那一步的形状是

$$
\sum_i R_i^{p-1} T_i(v) \le R_{\max}^{p-1} \sum_i T_i(v) .
$$

因 $p > 1$，$t\mapsto t^{p-1}$ 严格增；取等当且仅当对每个 $T_i(v) > 0$ 的 $i$ 都有 $R_i = R_{\max}$。所以 $T_{i_2}(v) > 0$ 就直接给出 $(\star)$ 严格（$R_{i_2} < R_{\max}$）。

而 $T_{i_2}(v) = 0$ 意味着：对所有 $M_{i_2 j} \ne 0$ 有 $v_j = 0$。$M(s)$ 的第 $i_2$ 行中，$M_{i_2 i_2} = s_{i_2}(1-s_{i_2}) > 0$，且对每个 $j \in \mathrm{supp}(s)$ 有 $M_{i_2 j} = -s_{i_2}s_j < 0$。故 $T_{i_2}(v) = 0 \Rightarrow v_j = 0$ 对所有 $j \in \mathrm{supp}(s)$，即 $v$ 的支撑落在 $\mathrm{supp}(s)$ 之外。

(3) **另一种情形像为 0。** 若 $\mathrm{supp}(v) \cap \mathrm{supp}(s) = \emptyset$，则

$$
M(s) v = \mathrm{diag}(s) v - s (s^{\top} v) = 0 - s\cdot 0 = 0 ,
$$

因为 $\mathrm{diag}(s)v$ 的第 $j$ 元是 $s_jv_j$，两个支撑不交故全为 0；$s^\top v = \sum_j s_jv_j = 0$ 同理。此时 $\Vert M(s)v\Vert_p = 0 < 1/2$ 也严格。

合并 (2)(3)：**每个**单位 $v$ 都使链条严格，故 $\Vert M(s)v\Vert_p < R_{\max}^{1/p}C_{\max}^{1-1/p}\le 1/2$。再由 (1) 取到最大值的 $v^\star$，

$$
\Vert M(s) \Vert_p = \Vert M(s) v^\star \Vert_p < \frac{1}{2} .
$$

Case (iii) 排除完毕，Lemma 4 得证。

**评估**：这是一个可补的跳步而非错误——补进去只用到了"紧集上连续函数取最大"与"$t^{p-1}$ 严格增"两条，不引入新假设，Lemma 4 的结论与论文一致。但论文原样的证明在审查意义上是不完整的，若审稿人追问"$<$ 与 $\sup<$ 的区别"会站不住。

### 单元 3.4 Prop 2(b)：$1<p<\infty$ 时只能取极限

由 Lemma 4，$\Delta_n^\circ$ 内的点全部满足 $|\mathrm{supp}(s)| = n > 2$，故 $\Vert M(s)\Vert_p < 1/2$ 对一切内点成立，故上确界在 $\Delta_n^\circ$ 内**不取到**。

另一方面，$\hat s = (1/2,1/2,0,\ldots,0) \in \partial\Delta_n$ 满足 $\Vert M(\hat s)\Vert_p = 1/2$（单元 3.2）。取一列 $s_k \in \Delta_n^\circ$ 使 $s_k \to \hat s$（构造见单元 3.5），由 $s \mapsto \Vert M(s)\Vert_p$ 连续：

$$
\lim_{k\to\infty} \Vert M(s_k) \Vert_p = \Vert M(\hat s) \Vert_p = \frac{1}{2} ,
$$

故 $\sup_{\Delta_n^\circ} \Vert M(s)\Vert_p \ge 1/2$，配合 Theorem 1(a) 的 $\le 1/2$ 得上确界恰为 $1/2$，与端点情形同值。$n = 2$ 时 $(1/2,1/2)$ 本身就在 $\Delta_2^\circ$，所以"内点取到"（与 Prop 2(a) 的构造点在 $n=2$ 时是同一个点，自洽）。

**这条结果是全文最有信息量的一处分类**，它说明"$\sigma$ 是 $1/2$-Lipschitz"在 $1<p<\infty$、$n>2$ 时是一句"常数最小但没有任何点达到"的陈述。对使用者的实际影响：任何"取最难输入构造对抗样本 / 证明最坏情形下界"的下游论证，不能靠"存在某个 $x$ 使局部常数为 $1/2$"来实现，只能用逼近序列，相应地下界论证会带 $-\epsilon$ 项。

### 单元 3.5 Example 2 复算：论文这里有一处实算错误

论文的构造（附录 A.3 末）：给定 $\epsilon \in (0,1/2)$，选 $\delta \in (0,1/2)$ 满足 $2\delta(1-\delta) = \epsilon$，令

$$
s_1 = s_2 = \frac{1}{2} - \delta, ,  s_j = \frac{2\delta}{n-2}, ,  j \ge 3, ,  v = \left(2^{-1/p}, -2^{-1/p}, 0, \ldots, 0\right) .
$$

先确认这是合法的：$\sum_i s_i = 2\left(\frac12-\delta\right) + (n-2)\cdot\frac{2\delta}{n-2} = 1 - 2\delta + 2\delta = 1$ ✓，且 $\delta\in(0,1/2)$ 保证全部分量 $>0$，即 $s \in \Delta_n^\circ$ ✓；$\Vert v\Vert_p = 1$（同单元 3.2）✓。

论文声称：

$$
\left[M(s)v\right]_1 = -\left[M(s)v\right]_2 = \frac{2}{2^{1/p}}\left(\frac{1}{2} - \delta\right)^2  \Longrightarrow \Vert M(s)v \Vert_p = \frac{1}{2} - 2\delta(1-\delta) = \frac{1}{2} - \epsilon .
$$

**实际重算。** 关键的第一步是 $s$ 与 $v$ 的内积：

$$
s^{\top} v = s_1\cdot 2^{-1/p} + s_2\cdot\left(-2^{-1/p}\right) = (s_1 - s_2) 2^{-1/p} = 0 ,
$$

因为构造里 $s_1 = s_2$。代回 $M(s) = \mathrm{diag}(s) - ss^\top$ 的定义，外积项整列消失：

$$
M(s) v = \mathrm{diag}(s) v - s (s^\top v) = \mathrm{diag}(s) v ,
$$

所以逐分量 $[M(s)v]_i = s_i v_i$，只有 $i=1,2$ 非零：

$$
\left[M(s)v\right]_1 = \left(\frac{1}{2} - \delta\right) 2^{-1/p}, ,  \left[M(s)v\right]_2 = -\left(\frac{1}{2} - \delta\right) 2^{-1/p}, ,  \left[M(s)v\right]_j = 0  (j \ge 3) .
$$

于是

$$
\Vert M(s) v \Vert_p = \left[ 2\left(\left(\frac{1}{2}-\delta\right)2^{-1/p}\right)^p \right]^{1/p} = \left(\frac{1}{2} - \delta\right)\cdot\left(2\cdot 2^{-1}\right)^{1/p} = \frac{1}{2} - \delta .
$$

**结论：论文显示的 $\frac{2}{2^{1/p}}(1/2-\delta)^2$ 与 $1/2 - 2\delta(1-\delta)$ 都不对，正确值是 $\frac{1}{2^{1/p}}(1/2-\delta)$ 与 $1/2-\delta$。** 错误的是把"行和减半"的因子 $(1/2-\delta)$ 又乘了一次——$[Mv]_1$ 只有一个 $s_1$ 因子（来自 $\mathrm{diag}(s)v$），不是 $s_1^2$。

**数值复核（$\delta = 0.1$，$p = 2$）**：$[M(s)v]_1$ 直接算 $= 0.282843$，修正公式给 $0.282843$ ✓，论文公式给 $0.226274$ ✗；$\Vert M(s)v\Vert_p$ 直接算 $= 0.4000000 = 1/2-\delta$ ✓，论文公式给 $0.32$ ✗。

**结论是否受影响？不受。** 论文要证的是"对每个 $\epsilon\in(0,1/2)$，存在 $s\in\Delta_n^\circ$ 使 $1/2-\epsilon \le \Vert M(s)\Vert_p < 1/2$"。修正后 $\Vert M(s)\Vert_p \ge \Vert M(s)v\Vert_p = 1/2-\delta$，而论文的取法 $2\delta(1-\delta)=\epsilon$ 给出

$$
\delta = \frac{1-\sqrt{1-2\epsilon}}{2} \le \epsilon ,  (\epsilon \in (0,1/2)),
$$

故 $1/2 - \delta \ge 1/2 - \epsilon$ ✓，上界 $<1/2$ 由 Lemma 4 保证 ✓。所以 Prop 2(b) 成立，只是这个"唯一的构造性证明"的中间算式写错了，读者照抄会推不出结论。实际上更简单的取法是直接令 $\delta = \epsilon$，不需要解 $2\delta(1-\delta)=\epsilon$ 这个多余的方程。

**审计级别**：计算瑕疵，不影响任何定理结论，但影响 Example 2 的可读性与可复核性。

### 单元 3.6 Example 1 复算：与论文数值一致

论文 Example 1：$K = 20$，$\varepsilon = 10^{-4}$，$x = (0,0,-K,\ldots,-K) \in \mathbb{R}^{10}$，$y = x + \varepsilon v$，$v$ 为 $J_{\sigma_1}(x)$ 最大特征值对应特征向量，声称比值 $\approx 0.49999999504472$ 对一切 $p\ge1$ 成立。

结构解释（为什么要这样选 $x$）：$s = \sigma_1(x)$ 满足 $s_1 = s_2 = \frac{e^{20}}{2e^{20}+8}$，$s_3,\ldots,s_{10} = \frac{1}{2e^{20}+8}$。即 $s$ 无限接近边界极值点 $(1/2,1/2,0,\ldots,0)$ 但严格在内部——正是 Prop 2(b) 的逼近序列的一个具体实例。而 $J_{\sigma_1}(x)$ 对称，所以"$v$ 取最大特征向量"对 $p=2$ 就是精确极方向；对 $p\ne2$ 论文没声称它是极方向，只是声称这个方向**碰巧**对所有 $p$ 都接近极（原因见单元 3.2 末尾：极方向是 $(+,-)$ 二维块，$\ell_p$ 因子互补）。

**数值复核**（$\lambda_{\max}(J_{\sigma_1}(x)) = 0.49999999587769$，对应 $v \approx (0.707107, -0.707107, 0, \ldots, 0)$）：

| $p$ | 本文复算比值 | 论文值 |
| --- | --- | --- |
| 1 | 0.499999995044 | 0.49999999504472 |
| 2 | 0.499999995044 | 同上 |
| 3 | 0.499999995044 | 同上 |
| 4 | 0.499999995044 | 同上 |
| $\infty$ | 0.499999995045 | 同上 |

前 11 位小数一致，末位差异属浮点运算次序。**Example 1 确认无误。** 偏差量级也可解释：$0.5 - \lambda_{\max} = 4.12\times10^{-9}$ 恰为 $\frac{4}{2e^{20}+8}$（$s$ 偏离 $1/2$ 的量），而 $\varepsilon = 10^{-4}$ 带来的有限差分二阶修正 $\sim \varepsilon^2$ 量级更小，不主导。

### 单元 3.7 Theorem 1 之后那条 Remark 的一处表述过强

论文 Remark 写："for $1<p<\infty$, the interpolation inequality in Proposition 1 is indeed strict for $J_{\sigma_1}(x)$ for all $x\in\mathbb{R}^n$"。这句话对 $n = 2$ 为假：取 $x = (0,0)$，$s = (1/2,1/2)$，则 $\Vert M(s)\Vert_1 = \Vert M(s)\Vert_\infty = 1/2$ 且（单元 3.2 的计算）$\Vert M(s)\Vert_p = 1/2$，插值不等式取等。正确表述需加 $n > 2$（Prop 2(b) 正文确实加了，Remark 漏了）。属于表述与定理精确度的落差，不是错误结论。

---

---

## Part 4：下游结果与方法论审计（§4、§5）

### 单元 4.1 推论 1：Gao & Pavel Corollary 3 的改进版（含 Baillon–Haddad 完整推导）

**论文推论 1**：对任意 $\lambda > 0$，

$$
\Vert \sigma_\lambda(x) - \sigma_\lambda(y) \Vert_p \le \frac{\lambda}{2} \Vert x - y \Vert_p ,  \forall p \ge 1 ,
$$

以及

$$
\left\langle \sigma_\lambda(x) - \sigma_\lambda(y), x - y \right\rangle \ge \frac{2}{\lambda} \Vert \sigma_\lambda(x) - \sigma_\lambda(y) \Vert_2^2 .
$$

符号：$\langle a, b\rangle = \sum_i a_ib_i$ 是欧氏内积（所以第二式**只在 $\ell_2$ 结构下有意义**，与第一式的"一切 $p$"不同层次，论文对此是清楚的）；$\frac{2}{\lambda}$ 称为 co-coercivity（余强制）常数。

论文说"证明直接来自 Gao & Pavel 的分析，把 Lipschitz 常数换成 $\lambda/2$"。我把这条链完整展开，因为它涉及一个论文没写的外部定理。

**原料 1：softmax 是某个凸函数的梯度。** 定义 log-sum-exp 势函数

$$
\phi_\lambda(x) := \frac{1}{\lambda} \log \sum_{j=1}^n \exp(\lambda x_j) .
$$

求梯度：$\frac{\partial \phi_\lambda}{\partial x_i} = \frac{1}{\lambda}\cdot\frac{\lambda\exp(\lambda x_i)}{\sum_j\exp(\lambda x_j)} = \sigma_\lambda(x)_i$，即 $\nabla\phi_\lambda = \sigma_\lambda$（Gao & Pavel Prop 1）。$\phi_\lambda$ 凸（对数配分函数是凸的，或由下面原料 2 的 Hessian 半正定）。

**原料 2：Hessian 与它的半正定性。** $\nabla^2\phi_\lambda(x) = J_{\sigma_\lambda}(x) = \lambda M(s)$，且对任意 $v$，

$$
v^{\top} M(s) v = \sum_i s_i v_i^2 - \left( \sum_i s_i v_i \right)^2 = \mathbb{E}_{I \sim \mathrm{Cat}(s)}\left[ v_I^2 \right] - \left( \mathbb{E}_{I\sim\mathrm{Cat}(s)}[v_I] \right)^2 = \mathrm{Var}(v_I) \ge 0 ,
$$

最后一步是方差非负。所以 $M(s)$ 半正定、$\phi_\lambda$ 凸。（这也解释了单元 0.4 提到的 $J\mathbf{1}=0$：常数向量方向上的随机变量无波动，方差为 0。）

**这里正是 Gao & Pavel 丢掉因子 2 的确切位置。** 他们的 Prop 4 用

$$
v^{\top}\nabla^2\phi_\lambda(x) v = \lambda\left( \sum_i s_i v_i^2 - \left(\sum_i s_i v_i\right)^2 \right) \le \lambda \sum_i s_iv_i^2 \le \lambda \Vert v \Vert_2^2\max_i s_i \le \lambda \Vert v \Vert_2^2 ,
$$

即**丢掉非负的平方项** $-\left(\sum_i s_i v_i\right)^2$、再用 $\max_i s_i \le 1$。这两步各自在什么情形下代价多大，可以精确算出来：取最坏构型 $s = (1/2, 1/2)$、$v = (1,-1)/\sqrt{2}$，则 $\left(\sum_i s_iv_i\right)^2 = 0$（丢这一项**不付出代价**），$\sum_i s_iv_i^2 = 1/2$（真值就是 $1/2$），但最后一步把 $\max_i s_i = 1/2$ 放大成 $1$，于是他们的链条给出 $v^\top \nabla^2\phi v \le \lambda$ 而非 $\lambda/2$。**结论：因子 2 完全丢在 $\max_i s_i \le 1$ 这一步。** 反过来，若 $s$ 接近确定性（$s_1\to1$），$\max_i s_i\to1$ 近乎不损失，但那时真值 $\lambda_{\max}(M(s))\to0$，损失转移到丢掉平方项那一步。也就是说，他们的两步放缩在**不同的 $s$ 区域**上各自松，无法同时收紧，这正是"逐元素放缩"型证明的典型特征；本文的行和 + 插值路线绕开了这个问题。

**原料 3：Baillon–Haddad 定理（论文与 Gao & Pavel 都引用而不证，这里给出可自查的证明）。**

命题：设 $\phi : \mathbb{R}^n \to \mathbb{R}$ 凸、$C^1$，且 $\nabla\phi$ 关于 $\Vert\cdot\Vert_2$ 是 $L$-Lipschitz，则

$$
\left\langle \nabla\phi(u) - \nabla\phi(v), u - v \right\rangle \ge \frac{1}{L} \Vert \nabla\phi(u) - \nabla\phi(v) \Vert_2^2 ,  \forall u, v \in \mathbb{R}^n .
$$

证明分四步。

第一步（下降引理）。$\nabla\phi$ 的 $L$-Lipschitz 性给出：对任意 $w, u$，

$$
\phi(w) \le \phi(u) + \left\langle \nabla\phi(u), w - u \right\rangle + \frac{L}{2} \Vert w - u \Vert_2^2 .
$$

推导：$\phi(w)-\phi(u) = \int_0^1 \left\langle \nabla\phi(u+t(w-u)), w-u \right\rangle \mathrm{d}t$，减去 $\langle\nabla\phi(u),w-u\rangle$ 后用 Cauchy–Schwarz，被积函数 $\le \Vert \nabla\phi(u+t(w-u))-\nabla\phi(u)\Vert_2\Vert w-u\Vert_2 \le Lt\Vert w-u\Vert_2^2$，对 $t$ 从 0 积到 1 得 $\frac{L}{2}\Vert w-u\Vert_2^2$。

第二步（对右端关于 $w$ 取最小）。右端是 $w$ 的强凸二次函数，极小点为 $w^\star = u - \nabla\phi(u)/L$，最小值 $\phi(u) - \frac{1}{2L}\Vert \nabla\phi(u)\Vert_2^2$。故

$$
\phi(w^\star) \le \phi(u) - \frac{1}{2L} \Vert \nabla\phi(u) \Vert_2^2 .
$$

第三步（用凸性给 $\phi(w^\star)$ 一个下界，再整理）。凸性在 $v$ 处：$\phi(w^\star) \ge \phi(v) + \langle\nabla\phi(v), w^\star - v\rangle$。与第二步合并，并把 $w^\star - v = (u-v) - \nabla\phi(u)/L$ 代入，得到下面这条不等式，记作 (BH)：

$$
\phi(u) - \phi(v) - \left\langle \nabla\phi(v), u - v \right\rangle \ge \frac{1}{2L} \Vert \nabla\phi(u) \Vert_2^2 - \frac{1}{L}\left\langle \nabla\phi(v), \nabla\phi(u) \right\rangle .
$$

第四步（把 (BH) 用在平移后的函数上）。令 $\tilde\phi(z) := \phi(z) - \langle \nabla\phi(v), z \rangle$。$\tilde\phi$ 仍凸、梯度仍 $L$-Lipschitz，且 $\nabla\tilde\phi(z) = \nabla\phi(z) - \nabla\phi(v)$，特别地 $\nabla\tilde\phi(v) = 0$。把 (BH) 用于 $\tilde\phi$ 的点对 $(u,v)$：左端不变（因为 $\nabla\tilde\phi(v)=0$，且 $\tilde\phi(u)-\tilde\phi(v) = \phi(u)-\phi(v)-\langle\nabla\phi(v),u-v\rangle$），右端成为 $\frac{1}{2L}\Vert\nabla\phi(u)-\nabla\phi(v)\Vert_2^2 - \frac{1}{L}\langle 0, \cdot \rangle$，于是

$$
\phi(u) - \phi(v) - \left\langle \nabla\phi(v), u - v \right\rangle \ge \frac{1}{2L} \Vert \nabla\phi(u) - \nabla\phi(v) \Vert_2^2 .
$$

把 $u, v$ 互换得到另一条：$\phi(v) - \phi(u) + \langle\nabla\phi(u), u-v\rangle \ge \frac{1}{2L}\Vert\nabla\phi(u)-\nabla\phi(v)\Vert_2^2$。两式相加，$\phi$ 项全部消掉，左端剩 $\langle \nabla\phi(u)-\nabla\phi(v), u-v\rangle$，右端剩 $\frac{1}{L}\Vert\nabla\phi(u)-\nabla\phi(v)\Vert_2^2$。证毕。

**装回 softmax**：$\phi = \phi_\lambda$、$\nabla\phi = \sigma_\lambda$、$L = \lambda/2$（Theorem 1），得余常数 $\frac{1}{L} = \frac{2}{\lambda}$，即推论 1 第二式 ✓。

**三个分类结论逐条核对**：

- nonexpansive（$\Vert\cdot\Vert_p$，一切 $p$）：需要 $L_p \le 1$，即 $\lambda/2 \le 1$，亦即 $\lambda \le 2$ ✓。
- firmly nonexpansive（$\ell_2$）：定义为 $\Vert f(x)-f(y)\Vert_2^2 \le \langle f(x)-f(y), x-y\rangle$，由余强制 $\langle\cdot,\cdot\rangle\ge\frac{2}{\lambda}\Vert\cdot\Vert_2^2$，只要 $\frac{2}{\lambda}\ge 1$，即 $\lambda\le 2$ ✓。
- contractive：需要 $\lambda/2 < 1$，即 $\lambda < 2$ ✓。

**与原始 Gao & Pavel Corollary 3 的逐条对照**（我从 arXiv PDF 抽取的原文为："$\sigma$ is $\lambda$-Lipschitz and $\frac{1}{\lambda}$-co-coercive ... Nonexpansive and firmly nonexpansive for $\lambda = 1$ ... Contractive for $\lambda\in(0,1)$"）：

| 性质 | Gao & Pavel (2017) Cor 3 | 本文推论 1 | 变化 |
| --- | --- | --- | --- |
| Lipschitz 常数 | $\lambda$ | $\lambda/2$ | 减半 |
| 余强制常数 | $1/\lambda$ | $2/\lambda$ | 加倍（不等式更强） |
| 非扩张 + firmly 非扩张 | $\lambda = 1$ | $\lambda \in (0,2]$ | 由孤立点扩成区间 |
| 压缩 | $\lambda \in (0,1)$ | $\lambda \in (0,2)$ | 窗口翻倍 |

**数值复核**：对 $\lambda \in \{0.5, 1, 2, 4\}$，每 $\lambda$ 抽 3000 组随机点对（$n=6$），计算相对余量 $\frac{\langle \Delta\sigma, \Delta x\rangle - \frac{2}{\lambda}\Vert\Delta\sigma\Vert_2^2}{\frac{2}{\lambda}\Vert\Delta\sigma\Vert_2^2}$，最小值为 $+0.1999 > 0$，即未见违反（随机点对不接近极构型，故留有余量是预期的）。

**研究审计**：这是本文下游结果里最干净的一个——不依赖任何未定义符号，改进是"允许的温度窗口扩大一倍"，可直接被"选 $\lambda$ 使不动点迭代压缩"类论证使用（熵正则 RL 的 Boltzmann 探索温度、软价值迭代）。非空泛区域为 $\lambda\in(1,2]$：这一段原来被判为非扩张/非压缩，现在被判为压缩，是净新增可用区间。可证伪推论：任何用 Gao & Pavel Cor 3 做收敛性证明的论文，把温度窗口放宽 2 倍后，其收敛率常数应改善 $2$ 或 $4$ 倍（取决于其界中 $\lambda$ 的幂次），且这一改善可在其数值实验里直接读出。

### 单元 4.2 Theorem 2：SCSA 界的精化（逐系数审计）

**先确认结构。** 标准自注意力（论文式 5）：$\mathrm{Att}(X;W^Q,W^K,W^V) = \mathrm{softmax}\left(\frac{XW^Q(XW^K)^\top}{\sqrt{d_k}}\right)XW^V$，其中 $\mathrm{softmax}(\cdot)$ 表示**逐行**作用 $\sigma_1$。符号：$X \in \mathbb{R}^{n\times d}$ 是 $n$ 个 token、每个 $d$ 维特征；$W^Q, W^K \in \mathbb{R}^{d\times d_k}$、$W^V \in \mathbb{R}^{d\times d_v}$；$d_k$ 是 head 维度，$\sqrt{d_k}$ 是缩放因子（把 $1/\sqrt{d_k}$ 塞进 softmax 的输入里，因此论文把整分数矩阵作为 softmax 输入、对应 $\lambda = 1$）。

论文指出：$X \mapsto \mathrm{Att}(X)$ 在未限定输入域上**不是**全局 Lipschitz（引 Kim et al. 2021，其结论是任意 $\ell_p$ 下常数为 $\infty$）。可验证的原因：把 $X$ 沿某方向放大 $t$ 倍，分数矩阵按 $t^2$ 放大，softmax 输出被压在单纯形内故有界，而右乘的 $XW^V$ 按 $t$ 放大；分子有界、分母按 $\Vert X\Vert$ 增长的模式使比值无界。因此改用 SCSA（论文式 6）：把 $q,k$ 归一化到单位范数、外乘温度 $\tau$ 与尺度 $\nu$，softmax 输入有界，整个映射才可能有界。

**LipsFormer Theorem 1 的 $\ell_2$ 分支原文**（我从 arXiv PDF 与 ar5iv 双向核对）：

$$
\mathrm{Lip}(\mathrm{SCSA})_2 \le 2N(N-1)\nu\tau\epsilon^{-1/2}\Vert W^K \Vert_2 + 2(N-1)\nu\tau\epsilon^{-1/2}\Vert W^Q \Vert_2 + 2N\nu\epsilon^{-1/2}\Vert W^{V\top} \Vert_2 ,
$$

其中 $N$ 是 token 数，$\epsilon$ 是"防止 $q,k,v$ 归一化时分母为 0 的平滑因子"（LipsFormer 自定义）。**本文 Theorem 2 直接沿用 $\epsilon$ 却完全没解释它是什么**——这是一处可读者性缺陷，必须回查被引论文才能补上。

**本文 Theorem 2 的陈述**（把 $N$ 改写成 $n$）：

$$
L_2(\mathrm{SCSA}) \le n^2 \nu\tau\epsilon^{-1/2} \Vert W^K \Vert_2 + n\nu\tau\epsilon^{-1/2} \Vert W^Q \Vert_2 + 2n\nu\epsilon^{-1/2} \Vert W^{V^\top} \Vert_2 .
$$

**替换是否合法？我用 LipsFormer 附录 H 的项结构验证过，合法。** 其式 (16) 显示：三个求和项里，含 softmax **Jacobian** 范数 $\Vert P^{(i)}\Vert$ 的只有两个带 $\tau$ 的项；第三项只含 $\nu\Vert P_{ii}\Vert\Vert V_i\Vert$，用的是 softmax **输出**矩阵的范数（$\le 1$），与 Jacobian 常数无关。故可反推 LipsFormer 前两项的结构为 $\nu\tau\epsilon^{-1/2}\Vert W\Vert\times 2N^2\times c_J$，其中 $c_J$ 是他们代入的 softmax Jacobian 常数：取 $c_J = (N-1)/N$ 给出 $2N^2\cdot\frac{N-1}{N} = 2N(N-1)$ ✓（与原文第一项吻合）、取 $c_J = \frac12$ 给出 $2N^2\cdot\frac12 = N^2$ ✓（与本文 Theorem 2 第一项吻合）。第二项同理：$2N\cdot\frac{N-1}{N} = 2(N-1)$ ✓、$2N\cdot\frac12 = N$ ✓。

**顺带补一个论文没给的推导：LipsFormer 为什么用 $(N-1)/N$。** 由单元 0.4 的性质 3，$M(s)$ 半正定，故 $\Vert M(s)\Vert_2 = \lambda_{\max}(M(s)) \le \mathrm{tr}(M(s)) = \sum_i s_i(1-s_i) = 1 - \sum_i s_i^2$。再由 Cauchy–Schwarz：$1 = \left(\sum_i s_i\right)^2 \le N\sum_i s_i^2$，即 $\sum_i s_i^2 \ge 1/N$，故 $\Vert M(s)\Vert_2 \le 1 - 1/N = (N-1)/N$。这是一条"用迹"的路线，天然比"用行和 + 插值"松（$N$ 大时 $(N-1)/N \to 1$，压缩性完全丧失）。

**改进倍数的准确核算**（论文称 "improves the estimate by a factor of 2, removing the extra factor 2"）：

| 项 | LipsFormer | 本文 Thm 2 | 改进倍数（旧/新） |
| --- | --- | --- | --- |
| $\Vert W^K\Vert_2$ 项 | $2N(N-1)$ | $N^2$ | $2(N-1)/N$ |
| $\Vert W^Q\Vert_2$ 项 | $2(N-1)$ | $N$ | $2(N-1)/N$ |
| $\Vert W^{V^\top}\Vert_2$ 项 | $2N$ | $2N$ | $1$（无改进） |

所以：只有两项被改进，倍数是 $\frac{2(N-1)}{N} = 2 - \frac{2}{N}$，**严格小于 2**，随 $N$ 增大才趋于 2；$N = 2$ 时倍数为 1，即**整个界一字未改**（与 Prop 2 里"$n=2$ 时 $(n-1)/n$ 已经是真值 $1/2$"完全自洽）。论文 §4.2 与结论里的"改进 2 倍 / 去掉多余的因子 2"是**过强表述**，准确说法是"$\tau$ 相关的两项各改进 $2-2/n$ 倍，第三项不变；整体改进严格介于 1 与 2 之间"。

**研究审计**：Theorem 2 是"换常数"型改进，不改变界的任何阶（$N^2$ 与 $N$ 的阶、$\epsilon^{-1/2}$、$\tau$ 的线性度全部不变）。其实际价值取决于下游是否用这个界做认证——若用，鲁棒半径按 $\sqrt{2-2/N}$ 放大（当 Lipschitz 常数出现在分母时）。另外，LipsFormer 的 $\ell_\infty$ 分支（其附录 H.1 式 (17)）已经使用了 $\mathrm{Lip}$ 估计里的 $\Vert P^{(i)}\Vert_\infty = \max_i 2(P_{ii}-P_{ii}^2) \le \frac12$，即本文 Theorem 1(a) 在 $p=\infty$ 处的同一行和恒等式，2023 年即已见诸文献；本文未与之划清（见单元 5.2 首创性一行）。

### 单元 4.3 Theorem 3：DSFP 压缩条件——这里有一处实质性错误

**问题设置。** 二人零和支付矩阵 $A \in \mathbb{R}^{n\times m}$，熵正则化为

$$
\min_{x \in \Delta_n} \max_{y \in \Delta_m} x^{\top} A y + \tau\left(H(x) - H(y)\right),  H(p) = -\sum_i p_i \log p_i ,
$$

符号：$x$ 是行（min 方）混合策略、$y$ 是列（max 方）策略、$\tau>0$ 是正则强度、$H$ 是 Shannon 熵。DSFP 迭代（论文 §4.3）：

$$
y_{k+1} \leftarrow (1-\alpha) y_k + \alpha \sigma_{1/\tau}\left( A^{\top} \sigma_{1/\tau}(-A y_k) \right),  x_k = \sigma_{1/\tau}(-A y_k) .
$$

先做维度校验（我核查任何迭代式的第一步）：$y_k \in \mathbb{R}^m \Rightarrow A y_k \in \mathbb{R}^n \Rightarrow \sigma_{1/\tau}(-Ay_k) = x_k \in \Delta_n \Rightarrow A^\top x_k \in \mathbb{R}^m \Rightarrow$ 外层 softmax 回到 $\Delta_m^\circ$ ✓ 自映射成立；$(1-\alpha)y+\alpha T(y)$ 是 $\Delta_m$ 内两点的凸组合，仍在 $\Delta_m$ ✓。

**再确认这个不动点确实是正则化博弈的解**（论文只断言、不证，我补上）：对 $\min_{x\in\Delta_n} x^\top Ay + \tau H(x)$ 写 Lagrange 条件，$\nabla_x\left(x^\top Ay + \tau H(x)\right) = Ay - \tau(\log x + \mathbf{1}) = \mu\mathbf{1}$，即 $\log x_i = -\frac{1}{\tau}(Ay)_i + c$；由平移不变性（性质 0.1a）得 $x = \sigma_{1/\tau}(-Ay)$ ✓。对 $\max_y$ 一侧同理得 $y = \sigma_{1/\tau}(A^\top x)$ ✓。所以 $T$ 的不动点正是熵正则 Nash（QRE），论文的问题陈述自洽、符号约定也对。

**论文附录 A.4 的证明与其错误。** 令 $T(y) = \sigma_{1/\tau}(A^\top\sigma_{1/\tau}(-Ay))$，逐层用 Theorem 1(b)（$\lambda = 1/\tau$，故每层 softmax 常数 $\frac{1}{2\tau}$）：

$$
\Vert T(y_1) - T(y_2) \Vert_p \le \frac{1}{2\tau} \Vert A^{\top} \Vert_p \cdot \frac{1}{2\tau} \Vert A \Vert_p \Vert y_1 - y_2 \Vert_p .
$$

到这里都对（Lipschitz 常数的复合规则）。**下一步论文写 "Since $\Vert A^\top \Vert_p = \Vert A \Vert_p$"，这一步对 $p \ne 2$ 是错的。** 正确的是伴随算子范数恒等式：

$$
\Vert B \Vert_p = \Vert B^{\top} \Vert_q ,  \frac{1}{p} + \frac{1}{q} = 1 ,
$$

证明：$\Vert Bv \Vert_p = \max_{\Vert u\Vert_q \le 1} u^{\top}Bv$（$\ell_p$ 的对偶刻画），故 $\Vert B\Vert_p = \max_{\Vert v\Vert_p\le1}\max_{\Vert u\Vert_q\le1} u^\top B v = \max_{\Vert u\Vert_q\le1}\Vert B^\top u\Vert_q = \Vert B^\top\Vert_q$；再把 $B$ 换成 $B^\top$ 即得 $\Vert A^\top\Vert_p = \Vert A\Vert_q$。

**反例（我实际算过）**：取 $A_{11}=1$、$A_{12}=2$、$A_{21}=3$、$A_{22}=4$。则 $\Vert A\Vert_1 = \max(1+3, 2+4) = 6$（最大列和），而 $\Vert A^\top\Vert_1 = \Vert A\Vert_\infty = \max(1+2, 3+4) = 7$（最大行和）。$6 \ne 7$，论文所用等式在 $p=1$ 下即被推翻；同时它精确印证修正后的恒等式 $\Vert A^\top\Vert_1 = \Vert A\Vert_\infty = 7$ ✓。补充：$\Vert R^\top\Vert_2 = \Vert R\Vert_2$ 恒成立（奇异值相同），而 $\Vert R^\top\Vert_p = \Vert R\Vert_p$ 对 $p \ne 2$ 一般不成立（随机矩阵测试中 $p=1$ 全部不相等）。

**修正后的结论。** 压缩因子应为

$$
c = \frac{\Vert A \Vert_q \Vert A \Vert_p}{4\tau^2},  \frac{1}{p}+\frac{1}{q}=1 ,
$$

条件应为 $\tau > \frac{1}{2}\sqrt{\Vert A\Vert_p \Vert A\Vert_q}$。$p = 2$ 时 $q=2$，退化为论文原述 $\tau > \Vert A\Vert_2/2$ ✓ ——即论文结论在 $\ell_2$ 下完全正确，只是"对一切 $1\le p\le\infty$"的范数一致宣称不成立。$p=1$（等价地 $p=\infty$）时应为 $\tau > \frac{1}{2}\sqrt{\Vert A\Vert_1\Vert A\Vert_\infty}$。

其余部分无恙：$c<1$ 时 $T$ 是完备度量空间 $(\Delta_m,\Vert\cdot\Vert_p)$ 上的 Banach 压缩；松弛迭代 $(1-\alpha)I+\alpha T$ 的常数为 $(1-\alpha)+\alpha c < 1$（$\alpha\in(0,1]$），故线性收敛到唯一不动点；$x_k = \sigma_{1/\tau}(-Ay_k)$ 由 softmax 连续性收敛到 $x^\star$ ✓。

**研究审计**：这是本文唯一一处"实质错误"，影响范围限于 §4.3 与附录 A.4（一处等式、一处条件），不触及 Theorem 1、Prop 2、推论 1、Theorem 2。修复成本极低（$\Vert A\Vert_p^2$ 换成 $\Vert A\Vert_p\Vert A\Vert_q$）。但它的性质值得注意：这是"把别人证明里的一个常数换掉"这类工作最容易犯的错误类型——换常数时顺手保留了自身论证里并不成立的对称性。

### 单元 4.4 式 (7)：经验 Lipschitz 常数的定义与它到底验证了什么

论文式 (7)：

$$
L_p^{\mathrm{emp}} = \max_{1\le i\le M} \frac{\Vert \sigma_\lambda(x_i) - \sigma_\lambda(x_i + \delta x_i) \Vert_p}{\Vert \delta x_i \Vert_p} ,
$$

符号：$M$ 是采样的输入向量条数；$x_i$ 是第 $i$ 个 softmax 输入（ViT/GPT-2/Qwen3 里是一行的 pre-softmax 注意力分数 $QK^\top/\sqrt{d_k}$；ResNet-50 里是分类 logits；RL 里是某状态的 $Q(s,\cdot)$）；$\delta x_i$ 是对 $x_i$ 的随机扰动，论文要求它被归一化为 $\Vert \delta x_i \Vert_p = \epsilon$，$\epsilon$ 是扰动幅度；分母就是 $\epsilon$。

式 (7) 与式 (3) 的关系：式 (3) 是**所有**点对的上确界（真值 $L_p$），式 (7) 是**有限个特定点对**的最大值。因此

$$
L_p^{\mathrm{emp}} \le L_p ,
$$

因为 (7) 的取最大集合是 (3) 的上确界集合的子集。即 $L_p^{\mathrm{emp}}$ 是 $L_p$ 的**下界样本**。由此得到两条方法论结论：

1. 若某次实验得到 $L_p^{\mathrm{emp}} > 1/2$，则**直接证伪** Theorem 1 —— 这是有效的 falsification 测试。论文报告"从未超过 $1/2$"，属于这一意义上的有效验证 ✓。
2. 但"始终低于 $1/2$"**不能**用来"验证 $L_p = 1/2$"，因为任何不小于样本最大值的常数都能同样解释这些数据。要主张"紧"，必须靠定理（Prop 2 的构造性/逼近性论证）。论文的紧性主张确实建立在 Prop 2 上，§5 只是"逼近 $1/2$"的观察 ✓ 逻辑上不越界；但摘要里 "validate the sharpness" 的措辞容易让读者误以为经验证据能确立紧性。

其它可核对点：

- 编号引用不一致：§5 开头说"用 equation 3 的定义"，§5.1/§5.2/§5.3 又称 "Definition 7"，而 (7) 实际是编号公式不是定义。
- 扰动方向：$\delta x_i$ 随机且只固定范数。对 $p \ne 2$，$\ell_p$ 诱导范数的极方向并非各向同性，随机方向命中近极方向的概率很小；论文在 PIQA/GPT-2 上得到 0.4999 的经验值，等价于**这些注意力行本身已高度接近边界构型** $(1/2,1/2,0,\ldots,0)$（即"两个 token 分掉几乎全部注意力"）。这是一个可检验的推论，论文没做：应直接报告这些行的 top-2 softmax 权重。若 top-2 权重不接近 $(1/2,1/2)$，则 0.4999 需要另一种解释（例如 $\epsilon$ 不够大/不够小使比值落在局部常数之外）。
- $\lambda$ 的处理：§5.1/§5.2 隐含 $\lambda = 1$（$1/\sqrt{d_k}$ 已算进分数），§5.3 显式变 $\lambda$ 并与 $\lambda/2$ 比较 ✓ 一致。
- 规模与可复现性：§5.1 报告 $N\times H\times M\times 12$ 个向量（$N$ token、$H$ head、12 层），§5.2 为 $H\times 100\times 12$ 每 token 每 prompt。论文无代码、无随机种子、无逐点数值表，图只能读近似值。§5 的**数值层**不可复算；其**逻辑层**（不违反上界）可复算——我已在单元 3.6 用 Example 1 复核到 11 位小数一致。

---

## Part 5：整体骨架、主张审计与结论

### 单元 5.1 证明骨架（假设 → 引理 → 定理，含每条边的核验状态）

```
式(1) softmax ─┬─ 性质 0.1a 平移不变（补证）
               ├─ 性质 0.1b σ_λ(x)=σ_1(λx)（补证）
               └─ 性质 0.1c 满射到 Δ_n°（论文未写，本文补证）─┐
                                                             │
Lemma 2 Jacobian = λ(diag(s)−ssᵀ) ─┬─ 对称 ──→ ‖·‖_1 = ‖·‖_∞ │
                                    ├─ 只通过 s 依赖 x ───────┼─→ Lemma 3 式(4) 变分式
                                    └─ 行和为 0 ⇒ M(s) 半正定 ─┘        │
Lemma 1 L_p = sup_x ‖J_f(x)‖_p（双向，本文补证）────────────────────────┤
                                                                        ▼
Prop 1(a)(b) 列和/行和闭式 ─┐                              Thm 1(a) 行和 = 2s_i(1−s_i) ≤ 1/2
                             ├─ Thm 1(a) 第四步插值 ─────────────────────┐
Prop 1(c) 插值（本文全展开）┘                                           ▼
                                                          Thm 1(b) 全局界 λ/2
Thm 1(a) + 构造点 x=(ln(n−1),0,…) ─→ Prop 2(a) p=1,∞ 内点取到 1/2      │
                                                                        ├─→ 推论 1（余强制 2/λ；λ∈(0,2] 非扩张）
Lemma 4 支撑集刻画：极值点必为 (½,½,0,…) ─→ Prop 2(b) 1<p<∞,n>2 不可达  ├─→ Thm 2（改进 2−2/n，非 2）
        ↑ 需 式(8) 严格 + 单位球紧（论文跳步，本文补齐）                 └─→ Thm 3（p=2 正确；一般 p 需 ‖A‖_p‖A‖_q）
Example 2 逼近序列（论文算式有误，本文修正为 1/2−δ）
Example 1 数值（本文复算 11 位一致）
```

一句话概括增量：**"范数一致"完全由"Jacobian 对称 + 端点插值"两步机械给出；真正的技术内容是可达性分类（Prop 2 + Lemma 4），它依赖对插值取等条件的精细分析。**

### 单元 5.2 主张-证据-限制矩阵（本文独立核验后的版本）

| 论文主张 | 证据位置 | 论文证据是否充分 | 我这一轮的独立核验 | 置信度 |
| --- | --- | --- | --- | --- |
| $\sigma_\lambda$ 在所有 $\ell_p$（$p\ge1$）下为 $\lambda/2$-Lipschitz | Thm 1 + A.1/A.2 | 充分（Lemma 3 换元、单位球取最大两处跳步可补） | 证明链逐步重推无误；$n=2,3,10$、$p\in\{1,2,3,\infty\}$ 各 400 随机点数值均未越 $1/2$ | 高 |
| $\lambda/2$ 紧（不可更小），$1/4$ 错 | Prop 2 + Lemma 4；单元 2.4 反例 | 充分 | $n=2$、$x=0$ 时 $\Vert J\Vert_2=1/2$ 精确复算成立，故任何小于 $1/2$ 的常数必错 | 高 |
| $p=1,\infty$ 内点达到；$1<p<\infty,n>2$ 只能取极限 | Prop 2 + A.3 | 结论充分，但 Case (iii) 由"逐点严格"跳到"上确界严格"缺一跳 | 已用"紧集取最大 + 行和 $R_{i_2}<R_{\max}$ + 支撑不交时 $Mv=0$"补齐 | 高（补齐后） |
| Example 1 比值 $\approx 0.499999995$ 对所有 $p$ | §3 Example 1 | 充分 | 复算 $p=1,2,3,4,\infty$ 均为 0.499999995044，前 11 位一致 | 高 |
| Example 2 的逼近序列构造 | A.3 Example 2 | **不充分（算式错）** | 正确值 $[Mv]_1 = 2^{-1/p}(1/2-\delta)$、$\Vert Mv\Vert_p = 1/2-\delta$；论文写 $(1/2-\delta)^2$ 与 $1/2-2\delta(1-\delta)$。$\delta=0.1$、$p=2$：直接算 0.400000 对论文式 0.320000 | 高（错误确证）；结论仍成立 |
| 推论 1：余强制 $2/\lambda$、$\lambda\in(0,2]$ 非扩张、$(0,2)$ 压缩 | §4.1 | 充分（依赖 Baillon–Haddad，论文未展开） | 已从 $\sigma_\lambda=\nabla\phi_\lambda$ 完整重推 B–H；12000 组随机点对未违反，相对余量为正 | 高 |
| Theorem 2：SCSA 界"改进 2 倍" | §4.2 | **表述过强** | 与 LipsFormer Thm 1 逐系数比对：$\tau$ 相关两项改进 $2-2/n$ 倍、$W^V$ 项不变；$n=2$ 时无改进 | 高 |
| Theorem 3：压缩因子 $\Vert A\Vert_p^2/(4\tau^2)$、条件 $\tau>\Vert A\Vert_p/2$ 对所有 $p$ | §4.3 + A.4 | **不成立**（$p\ne2$） | 反例 $\Vert A\Vert_1=6$、$\Vert A^\top\Vert_1=7$；正确为 $\tau>\frac{1}{2}\sqrt{\Vert A\Vert_p\Vert A\Vert_q}$ | 高 |
| "首个范数一致的 softmax Lipschitz 分析" | 摘要 + §1 | 需收敛表述 | LipsFormer (2023) 附录 H.1 式 (17) 已有 $\max_i 2(P_{ii}-P_{ii}^2)\le 1/2$，即本文 Thm 1(a) 在 $p=\infty$（由对称性亦含 $p=1$）的同一行和恒等式；本文未与之划清 | 高 |
| §5 经验常数"从未超过 $1/2$" | §5 图 1–4 | 有限采样足以证伪、不足以确立紧性 | 未复算（无代码/无种子）；Example 1 已独立复算 | 中 |

### 单元 5.3 对"我做优化器 / 鲁棒性理论"的可迁移结论

1. **可以直接用的**：任何含 softmax 的 Lipschitz 复合界，把 softmax 因子从 $\lambda$ 换成 $\lambda/2$（$\ell_p$ 任意 $p\ge1$）；任何依赖 Gao & Pavel Cor 3 的温度条件，$\lambda$ 窗口翻倍。零风险替换。
2. **要小心用的**：$\ell_p$ 一致性对 $1<p<\infty$、$n>2$ 是"取不到"的，所以任何"存在最坏输入"式论证不能引用它，只能带 $\epsilon$ 余量。
3. **不要照抄的两处**：涉及 $\Vert A^\top\Vert_p$ 的对称性化简（本文 Thm 3 就是这么错的）；涉及显式极方向的中间算式（本文 Example 2 就是这么错的）。两处指向同一条纪律：**在"换常数"型工作里，凡引入新的范数等价或转置对称性，必须单独验证，不能沿用被引论文的写法。**
4. **可继续挖的研究缺口（可证伪）**：
   - 同一套"对称 + 行和闭式 + 插值"能否搬到 sparsemax、entmax、top-$k$、Sinkhorn 行归一化？这些算子的 Jacobian 也对称，但行和不再是 $2s_i(1-s_i)$（sparsemax/entmax 的支撑集会随输入切换），需要重算极值构型。这是一个边界清楚、可在两周内验证的技术命题。
   - 矩阵输入上"逐行 softmax + 跨行耦合"的界：本文只处理了逐行独立性（每行一个 $\sigma_1$），完全没触及行间耦合；Kim et al. (2021) 的"常数为 $\infty$"结果在 SCSA 之外的哪些缩放下可救，仍是空白。

## 论文资产

- **公式推理**：[全局推理.md](../../04-equation_problem/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/全局推理.md)
- **思考过程**：[Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms_全局推理_过程.md](Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms_全局推理_过程.md)、[Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms_精读笔记.md](Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms_精读笔记.md)
