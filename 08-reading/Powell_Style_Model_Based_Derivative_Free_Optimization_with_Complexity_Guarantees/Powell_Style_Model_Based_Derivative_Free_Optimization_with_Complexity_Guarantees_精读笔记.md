# Powell_Style_Model_Based_Derivative_Free_Optimization_with_Complexity_Guarantees_精读笔记

- 生成时间：2026-09-19 14:09

---

# 论文精读笔记：Powell-Style Model-Based Derivative-Free Optimization with Complexity Guarantees

- 作者/来源：A. Chaudhry, K. Scheinberg, Scholar Sun；ONR 资助，引用 [11] 为 ICM 2026 论文集
- 所属子方向：无导数优化（DFO）/ 模型基信赖域 / 随机子空间 / 非凸一阶平稳点复杂度
- 精读开始日期：2026-09-19
- 模式：交互模式（一次一个单元，用户要求"从头推导"）

---

## 目录（随讲解进度更新）

- [x] 单元 1：假设 1.2（梯度 $L$-Lipschitz）+ 由它推出的下降引理
- [x] 单元 2：噪声 oracle 与复杂度目标 $\mathcal{C}_\epsilon \le \Psi(n, \epsilon)$
- [x] 单元 3：模型 (2.1) 与 Definition 2.1（fully-linear 模型）
- [x] 单元 4：Algorithm 1 逐行（含 $\rho_k$ 与三分支更新）
- [ ] 单元 5：Assumption 2.2（充分 Cauchy 下降 + Hessian 有界）
- [ ] 单元 6：Lemma 2.3（小 $\Delta_k$ 蕴含成功步）与 $C_1$ 的来历
- [ ] 单元 7：Lemma 2.4（成功步蕴含函数值下降）与 $C_2$ 的来历
- [ ] 单元 8：Lemma 2.5 / 2.6 / 2.7（$\Delta_k$ 下界与两类迭代计数）
- [ ] 单元 9：Theorem 2.8（合并界）
- [ ] 单元 10：Lemma 2.9（$\kappa_{eg}$ 推出 $\kappa_{ef}$）
- [ ] 单元 11：有限差分模型与 Corollary 2.10（$\mathcal{O}(n^{3/2}\epsilon^{-2})$）
- [ ] 单元 12：Def 3.1 / 3.2（Lagrange 多项式与 $\Lambda$-poisedness）
- [ ] 单元 13：Theorem 3.3 与 Corollary 3.4
- [ ] 单元 14：Theorem 3.5 与 Corollary 3.6（含噪插值界）
- [ ] 单元 15：(4.1) 约束最小二乘模型与 Algorithm 2 逐行
- [ ] 单元 16：(4.4) Lagrange 多项式更新公式
- [ ] 单元 17：Lemma 4.2（$\det$ 与 poisedness 的计数）
- [ ] 单元 18：Lemma 4.3（子空间归纳与 $14^n$-poised）
- [ ] 单元 19：Theorem 4.1 + Corollary 4.4
- [ ] 单元 20：第 5 节子空间记号、Definition 5.1、Lemma 5.2
- [ ] 单元 21：Definition 5.3 / Lemma 5.4（well-aligned 子空间）
- [ ] 单元 22：Lemma 5.5（$\tilde{C}_1$）与 Algorithm 3 的两处改动
- [ ] 单元 23：Lemma 5.6（$-4\epsilon_f$ 的来源）
- [ ] 单元 24：5.1 节随机过程 $I_t, A_t, B_t$ 与 Lemma 5.8–5.11
- [ ] 单元 25：Theorem 5.12（$\mathbb{E}[T_\epsilon]$ 界）
- [ ] 单元 26：Theorem 5.13 与 Corollary 5.14
- [ ] 单元 27：Assumption 6.1 与 $\kappa_{step}$（实现如何保住理论）

---

## 单元 1：假设 1.2（梯度 $L$-Lipschitz）+ 下降引理

**原文表述**：$\phi$ 连续可微，且

$$
\Vert \nabla \phi(y) - \nabla \phi(x) \Vert \le L \Vert y - x \Vert
$$

对所有 $x, y \in \mathbb{R}^n$ 成立。

**讲解要点摘要**：

- 作用：全文唯一被反复使用的目标函数性质。它本身不出现在任何定理结论里，但下面 (D1)(D2) 两条推论是 Lemma 2.4、Lemma 2.9、Theorem 3.3 以及有限差分界每一处误差界的种子。
- 关键推导来源：限制到线段 $\varphi(\tau) = \phi(x + \tau(y-x))$ → 微积分基本定理 → 加减 $\nabla \phi(x)$ → 积分三角不等式 + Cauchy–Schwarz → 代入假设 1.2 → $\int_0^1 \tau d\tau = \frac{1}{2}$。
- 三条推论：
  - (D1) 上界形式：函数值不超过"一阶泰勒展开 + $\frac{L}{2}$ 距离平方"
  - (D2) 双侧形式：上述偏差的绝对值同样被 $\frac{L}{2}$ 距离平方界住（Lemma 2.9 需要双侧）
  - (S1) 对 (D1) 右端关于 $y$ 最小化，得到下降量 $\frac{1}{2L}\Vert \nabla \phi(x) \Vert^2$
- 数值例子结论：$\phi(x) = \frac{a}{2}x^2$ 时 (D2) 取等号，界是紧的；$L$ 趋于 $0$ 时 $\phi$ 退化为仿射函数，(D1) 变成等式。
- 论文里 $C_2 \Delta_k^2$ 的形状就是 (S1) 的形状：下降量正比于半径的平方。

**断点图（去掉假设 1.2 会断在哪）**：

没有 1.2 → (D2) 不成立 → Lemma 2.9 中"$\kappa_{eg}$ 加上 $\frac{L+\kappa_{bhm}}{2}$ 就是 $\kappa_{ef}$"推不出 → Lemma 2.3 里 $\rho_k \ge \eta_1$ 的估计失去控制 → Lemma 2.4 的单步下降、Lemma 2.6 的求和 telescoping 全部断裂。

**可验证性与隐藏前提（研究审计）**：

- $L$ 不出现在 Algorithm 1 的输入里（输入只有 $f$、$x_0$、$\Delta_0$、$\eta_1$、$\eta_2$、$\gamma$），所以算法本身不需要知道 $L$；$L$ 只进入"允许的 $\epsilon$ 有多小"这一前提。
- 论文在 Corollary 2.10 之后明确写了 "we assume that L does not scale with n"。这是它能比较 $\mathcal{O}(n^2\epsilon^{-2})$ 与 $\mathcal{O}(n^{3/2}\epsilon^{-2})$ 的先决条件，属于实质性建模假设而非无害技术假设。
- 相比 [11]，本文没有在假设 1.1/1.2 上做任何放宽，放宽的是算法允许保留哪些样本点。

### 追问 1：步骤 4 到底怎么代入的

**问**：步骤 4 到底怎么带入的？

**答要点**：

- 先纠正定位：步骤 4 **没有**用假设 1.2，用假设发生在步骤 6。步骤 4 只有两个纯代数动作——(a) 把步骤 2 的 $\varphi'(\tau)$ 代入步骤 3 的积分；(b) 在被积函数里同时加一项、减一项 $\nabla \phi(x)^\top (y-x)$。
- 动作 (a) 的结果（记为 $(\star)$）：

$$
\phi(y) - \phi(x) = \int_0^1 \nabla \phi(x + \tau (y-x))^\top (y - x) d\tau
$$

- 动作 (b) 的依据只是内积对第一个变元的线性：对任意向量 $a$、$b$、$c$ 有 $a^\top c = b^\top c + (a - b)^\top c$。取 $a$ 为动点梯度 $\nabla \phi(x + \tau(y-x))$、$b$ 为**固定**的 $\nabla \phi(x)$、$c$ 为 $y - x$。
- 拆开后用积分线性性分成两个积分；第一个积分的被积函数与 $\tau$ 无关，于是

$$
\int_0^1 \nabla \phi(x)^\top (y-x) d\tau = \nabla \phi(x)^\top (y-x) \int_0^1 d\tau = \nabla \phi(x)^\top (y-x) \cdot 1
$$

  其中 $\int_0^1 d\tau = 1$ 就是区间 $[0,1]$ 的长度。移项后左端恰好是一阶泰勒余项。
- **动机（本单元唯一需要判断的地方）**：假设 1.2 的左端形状是"两个梯度之差取范数"，而 $(\star)$ 的被积函数里只有一个梯度，无处可套。加一项减一项就是为了把这个形状造出来。选 $\nabla \phi(x)$ 而不是别的点，因为 $x$ 是已有的展开中心，且不依赖 $\tau$，能整项提出积分号。
- 数值自检（$\phi(t) = t^2$，故 $L = 2$；$x = 1$，$y = 3$，动点为 $1 + 2\tau$）：

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 左端 $\phi(y) - \phi(x)$ | $9 - 1$ | $8$ |
| $(\star)$ 右端积分 | $\int_0^1 (4 + 8\tau) d\tau$ | $8$ |
| 拆分第一项 | $\int_0^1 4 d\tau$ | $4$ |
| 拆分第二项 | $\int_0^1 8\tau d\tau$ | $4$ |
| 余项与 $\frac{L}{2}(y-x)^2$ | $8 - 4$ 与 $\frac{2}{2} \cdot 4$ | $4 = 4$，取等 |

  逐步取等说明两件事：$\frac{1}{2}$ 完全来自 $\int_0^1 \tau d\tau$，不是拟合出来的；二次函数上步骤 5、6 的不等号逐点取等，所以 $\frac{L}{2}$ 不可能换成更小的常数。
- 三个易错点：不要在步骤 4 就套假设 1.2；不要把 $\int_0^1 d\tau = 1$ 当"什么都没做"（若把线段参数写成 $\tau \in [0, 2]$，整条界错一倍）；$b$ 必须取固定点的梯度，取随 $\tau$ 变的第三点就提不出积分号。若误取 $b = \nabla \phi(y)$，余项变成 $y$ 处梯度减去动点处梯度、因子由 $\tau$ 变成 $1 - \tau$，积分仍是 $\frac{1}{2}$，结论不变。

---

## 单元 2：噪声 oracle 与复杂度目标

**原文表述**（第 1.1 节末，紧接假设 1.2）：

$$
| f(x) - \phi(x) | \le \epsilon_f
$$

对所有 $x$ 成立，其中 $\epsilon_f > 0$。目标（第 67 段）：对所有 $\epsilon > \psi(\epsilon_f)$，确定性算法要求 $\mathcal{C}_\epsilon \le \Psi(n, \epsilon)$，随机算法要求 $\mathbb{E}[\mathcal{C}_\epsilon] \le \Psi(n, \epsilon)$。

**讲解要点摘要**：

- $\phi(x)$ 本身也读不到（零阶 + 噪声 ⟹ 只有 $f$）；假设 1.1/1.2 是对 $\phi$ 说的，不是对 $f$。误差界对**所有** $x$ 一致、与调用次数无关、不随迭代衰减，且 $f$ 是确定函数 ⟹ **不能靠重复采样平均消噪**。
- 核心推导（补全论文跳步，其结论是式 (2.10)）：对正交基 $\{u_i\}$ 与步长 $\delta$，令 $g(x) = \sum_{i=1}^{n} \frac{f(x + \delta u_i) - f(x)}{\delta} u_i$，则

  $$
  g(x) - \nabla \phi(x) = \sum_{i=1}^{n} c_i u_i
  $$

  其中 $c_i = \frac{f(x + \delta u_i) - f(x)}{\delta} - u_i^{\top} \nabla \phi(x)$。

  正交性给出 $\Vert g(x) - \nabla \phi(x) \Vert = \sqrt{\sum_{i=1}^{n} c_i^2}$；把 $c_i$ 加减 $\phi$ 拆成截断项与噪声项：截断项由单元 1 的 (D2) 取 $y = x + \delta u_i$、用 $\Vert \delta u_i \Vert = \delta$ 得绝对值 $\le \frac{L\delta}{2}$；噪声项由 oracle 界 + 三角不等式得 $\le \frac{2\epsilon_f}{\delta}$。故

  $$
  \Vert g(x) - \nabla \phi(x) \Vert \le \sqrt{n}\left(\frac{L\delta}{2} + \frac{2\epsilon_f}{\delta}\right) = \frac{\sqrt{n} L \delta}{2} + \frac{2\sqrt{n} \epsilon_f}{\delta}
  $$

  $\sqrt{n}$ 的来源是"$n$ 个同量级坐标误差的平方和开根号"，不是单维误差变大；它就是后文 $\kappa_{eg} = \sqrt{n} L$ 与复杂度 $n$ 幂次的源头之一。
- 对 $\delta$ 求最优：提出 $\sqrt{n}$，AM–GM 给 $h(\delta) \ge 2\sqrt{n L \epsilon_f}$，乘积中 $\delta$ 恰好约掉 ⟹ 地板与 $\delta$ 无关；等号条件 $\frac{L\delta}{2} = \frac{2\epsilon_f}{\delta}$ 给 $\delta^{\ast} = 2\sqrt{\frac{\epsilon_f}{L}}$。
- 论文第 261 行的 $\Delta_k \ge 2\sqrt{\frac{\epsilon_f}{L}}$ **正是** $\delta^{\ast}$，且论文取"大于等于"而非"等于"：它要 $\frac{2\sqrt{n}\epsilon_f}{\Delta_k} \le \frac{\sqrt{n} L \Delta_k}{2}$（两边乘 $\frac{2\Delta_k}{\sqrt{n}}$ 得 $4\epsilon_f \le L\Delta_k^2$），目的是让界保持 $\kappa_{eg}\Delta_k$ 的**线性形式**以便进入 fully-linear 理论与 Lemma 2.9。取舍是牺牲绝对精度换界的形式，代价就是 $\psi(\epsilon_f) > 0$。
- $\psi$ 的平方根形状 = "$\epsilon_f$ 被 $\delta$ 除一次"的放大效应。数值：$n = 100$、$L = 1$、$\epsilon_f = 10^{-4}$ 时 $\delta^{\ast} = 0.02$，两项各 $0.1$，和 $0.2 = 2\sqrt{n L \epsilon_f}$ 取等；$\epsilon_f$ 降 $100$ 倍地板只降 $10$ 倍。
- $\mathcal{C}_\epsilon$ 数的是 **oracle 调用次数**（不是迭代数）：一次完整坐标有限差分要 $n + 1$ 次调用，这是 $\mathcal{O}(n^2\epsilon^{-2})$ 的第一块地基。
- 达标判据是**外部**的：第 872 行 $T_\epsilon$ 用真梯度 $\nabla \phi$ 定义，算法自己的更新只看 $\rho_k$、$\Vert g_k \Vert$、$\Delta_k$，从不计算 $\nabla \phi$。
- 确定性 vs 随机的区别不是"有无噪声"，而是**路径是否唯一**：Alg 1/2 逐点断言；Alg 3/4 中 $Q_t$ 随机 ⟹ $x_t, \Delta_t, T_\epsilon, \mathcal{C}_\epsilon$ 全是随机变量，且 $\mathbb{E}$ 只对算法内部随机性取（噪声无分布可积）。期望界由 Markov 只给 $\Pr[\mathcal{C}_\epsilon > \lambda \Psi] \le \frac{1}{\lambda}$ 的松尾，不如所引 [7]、[18] 的高概率界。

**断点图（去掉 $\psi$ 这条量词限制会断在哪）**：

$\Delta_k < 2\sqrt{\frac{\epsilon_f}{L}}$ ⟹ (2.10) 的噪声项超过截断项 ⟹ 无法取 $\kappa_{eg}\Delta_k$ 形式、模型不再可证 fully-linear ⟹ Lemma 2.3 的 $\rho_k \ge \eta_1$ 失控 ⟹ Lemma 2.4 单步下降与 Theorem 2.8 的计数 $\frac{4(\phi(x_0) - \phi^{\star})}{C_2(\gamma C_1 \epsilon)^2}$ 第一步就断。Algorithm 3/4 的输入 $\Delta_{\min} = \sqrt{\epsilon_f}$ 与定理里的 $\psi$ 是同一件事的两种写法。

**可验证性与隐藏前提（研究审计）**：

- $\epsilon_f$ 是**算法必须知道的输入**：Alg 3/4 的 Inputs 显式列出这条界，且 Alg 4 的 $\rho_k$ 分子写成 $f(x_k) - f(x_k + s_k) + 2\epsilon_f$（第 1041 行）。若 $\epsilon_f$ 未知，该算法不可实现。
- 目标函数性质 $L$ 不进算法输入（只进 $\epsilon$ 允许范围），但 $\epsilon_f$ **进**输入——两条噪声/光滑常数的地位不对称，值得记住。
- 原文第 67 行有两处松动："reaching a point $x$" 与 $\Vert \nabla \phi(x_\epsilon) \Vert$ 混用；"that guarantees reaching" 语法上修饰 algorithm。采用读法：$\mathcal{C}_\epsilon$ = 从开始到**首次**满足判据为止的累计调用数。
- 三区分开：作者称 $\psi$ 给出 "best achievable optimality criterion"（主张）；论文只证到"第 261 行的充分条件在半径过小时失效"（证据，属算法自洽所需）；$\Omega(\sqrt{n L \epsilon_f})$ 对更广算法类是否成立本文无下界证明（推断）。
- $\Psi$ 里 $\epsilon^{-2}$ 的形状预告：成功步下降量 $\Theta(\Delta^2)$（(S1) 形状）+ 停止前 $\Delta \ge \gamma C_1 \epsilon$ ⟹ 计数 $\sim \epsilon^{-2}$；单元 9 严格化。

**本部分问答记录**：

（等待用户提问）

---

## 单元 3：模型 (2.1) 与 Definition 2.1（fully-linear 模型）

**原文表述**：

$$
m_k(x_k + s) = \phi(x_k) + g_k(x_k)^{\top} s + \frac{1}{2} s^{\top} H_k(x_k) s
$$

Definition 2.1：$m(x+s)$ 是 $\phi(x+s)$ 在 $B(x, \Delta)$ 上的 $\kappa_{ef}, \kappa_{eg}$-fully-linear 模型，若

$$
\Vert \nabla m(x) - \nabla \phi(x) \Vert \le \kappa_{eg} \Delta
$$

且

$$
\left| m(x+s) - \phi(x+s) \right| \le \kappa_{ef} \Delta^2
$$

对所有满足 $\Vert s \Vert \le \Delta$ 的 $s$ 成立。

**讲解要点摘要**：

- 自洽性验证（论文未写、必须自己做的）：以 $z = x_k + s$ 为自变量求导，$\nabla m_k(z) = g_k + H_k(z - x_k)$、$\nabla^2 m_k(z) = H_k$。故 $g_k = \nabla m_k(x_k)$、$H_k = \nabla^2 m_k(x_k)$ 是**恒等式**而非假设；副产品 $\nabla m_k(x_k+s) = g_k + H_k s$（模型梯度随位移线性变化），以及 $m_k(x_k) = \phi(x_k)$（中心精确插值）。
- 两条界的**量词范围不对称**：梯度那条只在球心一点比，函数值那条在整个球每一点比。后者是算法能用 $\rho_k$ 的前提（见"断点图"）。
- $\kappa_{ef}$ 与 $\kappa_{eg}$ 的关系当场推出（即论文 Lemma 2.9，原文留白）：令 $e(s) = m_k(x_k+s) - \phi(x_k+s)$，四步：
  1. $e(0) = 0$（中心插值）；
  2. $\Vert \nabla e(0) \Vert = \Vert \nabla m_k(x_k) - \nabla \phi(x_k) \Vert \le \kappa_{eg}\Delta$（定义第一条）；
  3. $\nabla e$ 的 Lipschitz 常数：模型部分 $\Vert H_k(s-t) \Vert \le \kappa_{bhm}\Vert s-t\Vert$（Assumption 2.2 第 2 条 + 谱范数定义），真函数部分 $L\Vert s-t\Vert$（假设 1.2），相加得 $(L+\kappa_{bhm})$-Lipschitz；
  4. 对 $e$ 套单元 1 的 (D1)：$e(s) \le 0 + \nabla e(0)^{\top}s + \frac{L+\kappa_{bhm}}{2}\Vert s\Vert^2 \le \kappa_{eg}\Delta^2 + \frac{L+\kappa_{bhm}}{2}\Delta^2$（Cauchy–Schwarz + $\Vert s\Vert \le \Delta$）；对 $-e$ 重复（齐次性使 Lipschitz 常数不变）得绝对值版本。
  结论：$\kappa_{ef} = \kappa_{eg} + \frac{L + \kappa_{bhm}}{2}$。
- **"为什么一个 $\Delta$ 一个 $\Delta^2$"的答案**：梯度误差 $\Theta(\Delta)$ 乘位移 $\Theta(\Delta)$ 得 $\Theta(\Delta^2)$，泰勒二阶余项本身也是 $\Theta(\Delta^2)$，两来源同阶。若第二条改写成 $\kappa_{ef}\Delta$（一阶），数学成立但没用——Lemma 2.4/2.6 需要误差比每步下降量 $C_2\Delta^2$ 高阶，一阶界压不住，求和计数不闭合。
- $\kappa_{eg}, \kappa_{ef}, \kappa_{bhm}$ 必须与 $k$、$\Delta$、$s$ 无关（第 99 行 "fixed throughout the iterations"），$\Delta$ 的幂次全部显式写出。
- 边界情形：线性模型 $H_k = 0 \Rightarrow \kappa_{bhm} = 0$，$\kappa_{ef} = \kappa_{eg} + \frac{L}{2}$（第 235 行明确允许，且是本文主线）；$\Delta \to \infty$ 时两条界都失去信息量 ⟹ fully-linear 只在小半径有用，这解释了 Lemma 2.3 的形式和半径放大为什么要附加 $\Vert g_k \Vert \ge \eta_2 \Delta_k$。

**含噪 oracle 下 Definition 2.1 的硬伤（本单元核心审计）**：

第二条在 $s = 0$ 处取值 ⟹ $\left| m(x) - \phi(x) \right| \le \kappa_{ef}\Delta^2$，即定义**强制**中心常数项误差达 $\Delta^2$ 阶。但实现只有 $f$，误差 $\le \epsilon_f$ 且不随 $\Delta$ 缩小 ⟹ 必须 $\Delta \ge \sqrt{\frac{\epsilon_f}{\kappa_{ef}}}$。这与单元 2 的 $\delta^{\ast} = 2\sqrt{\frac{\epsilon_f}{L}}$ **量级相同但来源不同**（一个来自截断/噪声平衡，一个来自中心插值精度要求）。两个独立理由都指向 $\Delta$ 的下界量级 $\sqrt{\epsilon_f}$，这解释了 Alg 3/4 为何把 $\Delta_{\min} = \sqrt{\epsilon_f}$ 做成输入。

论文第 369 行承认："constructing the model to satisfy (3.1) is not possible unless $\phi(x+y)-\phi(x)$ can be computed exactly"，随后式 (3.2)/(4.1) 改用差值拟合 $g_k^{\top}y + \frac{1}{2}y^{\top}H_k y = f(x_k+y) - f(x_l)$：右端作差把不可知的绝对常数项**消掉**，代价是两次读数各进一次噪声（放大成 $2\epsilon_f$）。这是三处修正的共同来源：单元 14 含噪插值界、Alg 4 的 $\rho_k$ 分子 $+2\epsilon_f$、Lemma 5.6 的 $-4\epsilon_f$。

**断点图（去掉 Def 2.1 第二条的全称量词会断在哪）**：

$s_k$ 的位置由模型决定、事先未知；$\rho_k$ 的分子是真实（含噪）下降、分母是模型预测下降，二者要在**同一个未知点** $x_k + s_k$ 上可比 ⟹ 必须有该点的模型误差界。若第二条只在 $s = 0$ 成立，$\phi(x_k) - \phi(x_k+s_k)$ 的端点误差无从控制 ⟹ Lemma 2.4 的单步下降 $C_2\Delta_k^2 - 2\epsilon_f$ 推不出 ⟹ Theorem 2.8 的计数断裂。

**数值自检（$n = 1$，$\phi(t) = \frac{a}{2}t^2$，$a = L = 2$；模型梯度带常数误差 $\eta$、$H_k = a$）**：

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 第一条要求的 $\kappa_{eg}$ | $\frac{\eta}{\Delta} = \frac{0.005}{0.1}$ | $0.05$ |
| 真实模型误差 | $\eta \Delta + \frac{1}{2}(H_k - a)\Delta^2 = \eta \Delta$ | $5 \times 10^{-4}$ |
| 第二条实际需要的 $\kappa_{ef}$ | $\frac{5 \times 10^{-4}}{0.01}$ | $0.05$（等于 $\kappa_{eg}$） |
| Lemma 2.9 给的 $\kappa_{ef}$ | $0.05 + \frac{2+2}{2}$ | $2.05$ |
| 松弛倍数 | $\frac{2.05}{0.05}$ | $41$ |

表中 $\eta$、$\Delta$ 均取正数，故绝对值符号已省略；因 $H_k = a$，二次项恰好为零，真实误差只剩 $\eta \Delta$。

$\frac{L+\kappa_{bhm}}{2}$ 这一项是通用证明**不知道** $H_k$ 与 $\nabla^2 \phi$ 吻合而付出的代价；$\kappa_{ef}$ 直接进入 $C_1$（Theorem 2.8 中 $\frac{2\kappa_{ef} + \max(\eta_2, \kappa_{ef})}{(1-\eta_1)\kappa_{fcd}}$ 那一项），$C_1$ 又同时决定 $\Delta_k$ 下界与 $\psi(\epsilon_f)$ 地板，所以"界不紧"会按倍数放大调用数与精度地板。

**可验证性与隐藏前提（研究审计）**：

- Definition 2.1 是把"收敛分析"与"几何/poisedness"解耦的唯一接口：第 2 节只用 $\kappa_{ef}, \kappa_{eg}, \kappa_{bhm}, \kappa_{fcd}$ 四个数，第 3/4/5 节全部工作是为了**给出**这四个数。
- 论文第 99 行明说"如何验证模型 fully-linear"作为**输入黑箱**给出，本节不讨论 ⟹ 待还的债，检查点是单元 12–14（poisedness ⟹ fully-linear）与 17–18（算法如何维持 poisedness）是否闭环。
- $\kappa_{bhm}$ 是全文唯一"由模型构造决定、却要求与维度无关"的量。第 235 行说"理想上应与 $L$ 同阶"，Corollary 2.10 实际只假设 $\kappa_{bhm} \le \mathcal{O}(\sqrt{n})$；"理想"与"所用"之间的落差要靠 poisedness 去填（单元 14、18），不是靠假设。
- 三区分：作者主张（fully-linear 框架 + Powell 式几何维护即可同时得 $\mathcal{O}(n^{3/2}\epsilon^{-2})$ 与 $\mathcal{O}(n\epsilon^{-2})$）／论文证据（Def 2.1 只依赖四个常数；Lemma 2.9 把 $\kappa_{ef}$ 归约到 $\kappa_{eg}$ 与 $\kappa_{bhm}$；第 369 行承认含噪下 (2.1) 不可直接实现）／评审推断（加性形式 $\kappa_{eg} + \frac{L+\kappa_{bhm}}{2}$ 可按一维特例收紧，收紧会直接改 $C_1$、$\psi$、$\Psi$ 的常数，本文未做）。

**本部分问答记录**：

（等待用户提问）

---

## 单元 4：Algorithm 1 逐行（含 $\rho_k$ 与三分支更新）

**原文表述**（第 102–110 行）：输入 $f(x) \approx \phi(x)$、$x_0$、$\Delta_0$、$\eta_1 \in (0,1)$、$\eta_2 > 0$、$\gamma \in (0,1)$，外加"判定模型是否 fully-linear 的机制"；对 $k = 0, 1, 2$ 持续循环，步骤 1 求模型与试验步、步骤 2 算 $\rho_k$、步骤 3 三分支更新、步骤 4 视需要改模型。

$$
\rho_k = \frac{f(x_k) - f(x_k + s_k)}{m_k(x_k) - m_k(x_k + s_k)}
$$

**讲解要点摘要**：

- **无停止条件**：算法里没有 $\epsilon$、也从不计算 $\nabla \phi$。第 135 行在算法之外定义 $K_\epsilon$ 为首个使 $\Vert \nabla \phi(x_k) \Vert \le \epsilon$ 的 $k$（与 Alg 3 的 $T_\epsilon$ 同一角色）。实践中靠 $\Delta_{\min}$ 或调用上限硬截断。
- 输入不对称：**Alg 1 的输入里没有 $\epsilon_f$**（Alg 3/4 才有），但 $\epsilon_f$ 通过定理前提 $\epsilon > \sqrt{\frac{4\epsilon_f}{\gamma^2 C_2 C_1^2}}$ 悄悄进入；$\Delta_0 > \gamma C_1 \epsilon$ 同样只在定理里出现。
- 步骤 1 的球是 $B(0, \Delta_k)$（球心原点，变量是位移 $s$），与 Definition 2.1 的 $B(x, \Delta)$（球心是点）同心性不同但半径同源，读证明时不要混成两个球。
- $\approx$ 有精确含义 = Assumption 2.2 第 1 条 (2.2)。**原文缺口 (i)**：线性模型 $\Vert H_k \Vert = 0$ 时 $\frac{\Vert g_k \Vert}{\Vert H_k \Vert}$ 无定义，必须补约定"记作 $+\infty$、$\min$ 取 $\Delta_k$"，否则 (2.2) 对论文主线的线性模型不成立。
- $\rho_k$ 的噪声账（可现在就算清）：分子减真实下降量 $= \left[f(x_k)-\phi(x_k)\right] - \left[f(x_k+s_k)-\phi(x_k+s_k)\right]$，两项各 $\le \epsilon_f$ ⟹ 分子误差 $\le 2\epsilon_f$；分母纯模型计算，无噪声。Alg 1 未修正 ⟹ Lemma 2.4 写成 $C_2\Delta_k^2 - 2\epsilon_f$；Alg 4 把分子改成加 $2\epsilon_f$（第 1041 行）⟹ 单元 23 处变 $-4\epsilon_f$。
- **原文缺口 (ii)**：$\Vert g_k \Vert = 0$ 时 (2.2) 右端为 $0$，分母可能恰为 $0$，$\rho_k$ 无定义；算法文本未讨论。
- 分支 1 为什么要两个条件：模型梯度误差 $\le \kappa_{eg}\Delta_k$ 与半径同阶，半径大于梯度时误差不少于信号。附加条件 $\Vert g_k \Vert \ge \eta_2 \Delta_k$ 给出严格推论

  $$
  \Delta_{k+1} = \gamma^{-1}\Delta_k \le \gamma^{-1}\eta_2^{-1}\Vert g_k \Vert
  $$

  即放大后的半径仍被放大前的模型梯度控制——这就是第 131 行"半径双重职能（控步长 + 控模型精度）、从而免掉 criticality step"的内容，该条件最早来自 [3]（概率模型 TR）。
- 分支 2 的 "else" 是顺序语义：仅在分支 1 失败后才检查。设计理由是错误归因——$\rho_k$ 低可能因为模型不可信，此时缩半径是错罚（更小半径里模型照样不可信），交给步骤 4 改模型；第 133 行据此命名 model improving iteration。
- 分支 3 缩半径 $\gamma\Delta_k$，Alg 1 **无下界**；含噪版第 747、1045 行改成 $\max(\gamma\Delta_k, \Delta_{\min})$。**原文缺口 (iii)**：有限差分模型需要 $\Delta_k \ge 2\sqrt{\frac{\epsilon_f}{L}}$（第 261 行）而 Alg 1 无任何机制维持它，分支 3 可无限收缩 ⟹ 只能靠定理前提的 $\epsilon$ 地板代偿。第 41 行"additional complication requires algorithmic modifications"指的就是补 $\Delta_{\min}$（在单元 22 核对）。
- 复杂度分解：$\mathcal{C}_\epsilon \le$（每次迭代最多调用数）乘（$\left| \mathcal{S}_\epsilon \right| + \left| \mathcal{U}_\epsilon \right|$ 与模型改进步的全部调用数）；前者归 Theorem 2.8（单元 9），后者归 Theorem 4.1（单元 19，指标集记作 $\mathcal{M}_\epsilon$）。
- Alg 1 无随机性 ⟹ 第 2 节结论都是逐点形式 $\mathcal{C}_\epsilon \le \Psi$；$\mathbb{E}$ 直到第 5 节抽 $Q_t$ 才出现。

**手动跑一遍（$n = 1$，$\phi(t) = t^2$ 故 $L = 2$，$\epsilon_f = 0$，线性模型 $H_k = 0$，$\delta = \Delta_k$）**：

$g_k = \frac{(x_k+\Delta_k)^2 - x_k^2}{\Delta_k} = 2x_k + \Delta_k$，取 $s_k = -\Delta_k$，得闭式

$$
\rho_k = \frac{2x_k - \Delta_k}{2x_k + \Delta_k}
$$

取 $\eta_1 = 0.5$ 时，$\rho_k \ge \eta_1$ 等价于 $\Delta_k \le \frac{2x_k}{3}$。

取 $\eta_2 = 1$、$\gamma = 0.5$、$x_0 = 3$、$\Delta_0 = 1$：

| $k$ | $x_k$ | $\Delta_k$ | $g_k$ | $\rho_k$ | 判定 | $x_{k+1}$ | $\Delta_{k+1}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 1 | 7 | 0.71 | 分支 1 | 2 | 2 |
| 1 | 2 | 2 | 6 | 0.33 | 分支 3 | 2 | 1 |
| 2 | 2 | 1 | 5 | 0.60 | 分支 1 | 1 | 2 |
| 3 | 1 | 2 | 4 | 0.00 | 分支 3 | 1 | 1 |
| 4 | 1 | 1 | 3 | 0.33 | 分支 3 | 1 | 0.5 |
| 5 | 1 | 0.5 | 2.5 | 0.60 | 分支 1 | 0.5 | 1 |

- $k = 1$ 的失败原因**就是单元 2 的截断误差**：$g_1 = 6$ 而 $\phi'(2) = 4$，误差 $2 = \frac{L\delta}{2}$ 逐点取等（二次函数）。半径越大梯度高估越多 ⟹ 模型预测下降 $12$ vs 实际 $4$ ⟹ $\rho$ 低 ⟹ 缩半径。
- $k = 3$ 跨过谷底：$x=1$ 走 $-2$ 到 $-1$，真实下降恰为 $0$，模型预测 $8$。
- 本例 $\eta_2$ 条件恒不起作用（$g_k = 2x_k+\Delta_k \ge \Delta_k$）。要"$\rho$ 达标但 $\eta_2$ 不达标"需 $\frac{2x_k}{\eta_2 - 1} < \Delta_k \le \frac{2x_k}{3}$，区间非空当且仅当 $\eta_2 > 4$。取 $\eta_2 = 5$、$x_k = 2$、$\Delta_k = 1.2$：$\rho_k = 0.54 \ge 0.5$ 通过、$g_k = 5.2 < 6 = \eta_2\Delta_k$ 失败 ⟹ 明明在下降仍不许放大半径。$\eta_2$ 用一次局部接受换"半径与梯度同步"的全局性质。
- 9 步里 4 次成功（$x$: 3 → 2 → 1 → 0.5 → 0.25），每次成功前需 1–2 次失败把半径压到 $\frac{2x}{3}$ 以下，而半径按 $\gamma$ 的幂走 ⟹ 压到位只需对数次。这是 Theorem 2.8 中"多项式项 + 对数项 $\log_\gamma$"结构的经验原型。

**可验证性与隐藏前提（研究审计）**：

- 参数身份三分：超参数 $\eta_1, \eta_2, \gamma, \Delta_0, x_0$（算法显式用）；模型属性 $\kappa_{eg}, \kappa_{ef}, \kappa_{bhm}, \kappa_{fcd}$（算法不用但界依赖）；问题属性 $L, \phi^{\star}$。$\epsilon_f$ 属第四类——算法不用、却必须被知道（Alg 4）或被前提代偿（Alg 1）。
- 与 [11] 的分支结构完全相同，差异全在步骤 4 与半径下界；这使"本文贡献在几何维护与噪声"这一自我定位可被机械核对。
- 三处缺口 (i)(ii)(iii) 中只有 (iii) 影响定理表述（被 $\psi$ 前提吸收），(i)(ii) 是实现层面的表述遗漏。
- 可检验的后续问题：Alg 1 + 只加半径地板 $\Delta_{\min} = c\sqrt{\epsilon_f}$ 而**不**修正 $\rho_k$，能否仍得 Theorem 2.8 同阶界？若可，则 $+2\epsilon_f$ 修正是多余的。检验路径：构造分子因噪声变号的二次例子，比较两版判定是否分歧。

**本部分问答记录**：

（等待用户提问）

