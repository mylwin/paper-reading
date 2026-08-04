# 论文精读笔记：Understanding MARS: When Scaling Momentum Correction Provably Helps

- 作者/来源：Egor Shulgin, Tamaz Gadaev, Sarit Khirirat, Peter Richtárik（KAUST 等，2026，编号 15553）
- 所属子方向：方差缩减（Momentum-based Variance Reduction, MVR）+ 动量加速 + 非凸收敛性分析
- 精读开始日期：2026-07-26

---

## 论文一句话总结

MARS 优化器在 MVR 的"动量修正项"前乘了一个缩放系数 $\gamma$ 。本文提出 **γ-相似性（γ-similarity）** 这个新条件，
首次从理论上解释了：为什么取 $\gamma < 1$ （而不是 MVR 的 $\gamma = 1$ ）能让收敛复杂度**严格更低**。

---

## 目录（随讲解进度更新）

- [x] 公式 (1)：随机优化问题定义 $\min f(x) = \mathbb{E}[f_{\xi}(x)]$
- [x] 公式 (2)(3)：MVR 更新规则
- [x] 公式 (4) + 算法 1：MARS 更新规则
- [x] 假设 1：L-光滑性
- [x] 假设 2：无偏 + 有界方差
- [x] 定义 1 / 公式 (5)：γ-相似性
- [x] 公式 (6)(7)：标准相似性与二阶相似性
- [x] 引理 1：$\delta_\gamma^2$ 的上界与最优 $\gamma_\star$
- [x] 定理 1 / 公式 (8)：MARS 的梯度复杂度
- [x] 推论 1 / 公式 (9)：$T$ -依赖参数下的收敛率
- [x] 推论 2：MARS 严格优于 MVR 的条件
- [x] 7.1 节：$\hat{\gamma}_t^\star$ 的经验估计公式

---

## 公式 (1)：随机优化问题的定义

**原文表述**（即论文公式 (1)）：

$$
\min_{x \in \mathbb{R}^d} f(x) := \mathbb{E}_{\xi} [ f_{\xi}(x) ]
$$

**讲解要点摘要**：

- 作用：定义全文要解决的问题——最小化"对所有数据的平均损失"，但每次只能通过随机采样 $\xi$ 看到一小部分数据。
- 关键点：$f(x)$ 是"看不见的真实目标"（期望），$f_{\xi}(x)$ 是"每次能算的随机样本损失"；训练神经网络的 mini-batch 损失就是 $f_{\xi}$ 。
- 数值例子结论：$n = 3$ 个样本的均值损失场景中，$f(x)$ 为平均损失，每次随机抽 1 个样本算 $f_{\xi}$ ，平均下来等于 $f$ 。

**本部分问答记录**：

（无提问，用户直接进入下一个）

---

## 公式 (2)(3)：MVR（动量方差缩减）的更新规则

**原文表述**（即论文公式 (2) 与 (3)）：

公式 (2)，参数更新：

$$
x_{t+1} = x_t - \eta g_t
$$

公式 (3)，梯度估计器更新：

$$
g_t = (1-\beta) \left( g_{t-1} + [ \nabla f_{\xi_t}(x_t) - \nabla f_{\xi_t}(x_{t-1}) ] \right) + \beta \nabla f_{\xi_t}(x_t)
$$

**讲解要点摘要**：

- 作用：(2) 是标准参数更新，(3) 是 MVR 的核心——梯度估计器 $g_t$ 由三部分组成：旧估计 $g_{t-1}$ 、**同一样本在两点的梯度差（修正项）**、当前新鲜随机梯度。
- 等价改写（STORM 形式）：$g_t = \nabla f_{\xi_t}(x_t) + (1-\beta)(g_{t-1} - \nabla f_{\xi_t}(x_{t-1}))$ ，即"当前梯度 + 打折后的历史误差"，误差每步乘 $(1-\beta)$ 收缩 → 方差被压低。
- 与 SGD 关系：$\beta = 1$ 退化为 SGD；去掉修正项退化为 SGD+动量（EMA）。修正项的作用是把旧梯度信息"搬运"到新位置，消除普通动量的陈旧偏差。
- 关键细节：修正项两个梯度用**同一个样本 $\xi_t$** ，同一样本在相邻两点的梯度差噪声很小（这正是后文相似性常数 $\delta$ 度量的东西）。
- 收敛率：MVR 达 $\mathcal{O}(1/T^{1/3})$ ，优于 SGD 与 SGD+动量的 $\mathcal{O}(1/T^{1/4})$ ；代价是每步算 2 个随机梯度。
- 数值例子结论：三样本二次函数、$x_1 = 4.2$ 处真实梯度 $2.4$ ，纯 SGD 估计 $-3.6$ （误差 $6.0$ ）、动量估计 $5.68$ （误差 $3.28$ ）、MVR 估计 $4.4$ （误差 $2.0$ ），MVR 最准。
- 溯源：即 Cutkosky & Orabona (2019) 的 STORM，原文 $a_t$ 对应本文 $\beta$ ，$d_t$ 对应 $g_t$ ；本文用固定 $\beta$ ，STORM 原文用自适应 $a_t$ 。

**本部分问答记录**：

1. 问：为什么要额外花一次反向传播去算 $\nabla f_{\xi_t}(x_{t-1})$ （用当前样本在上一个点算梯度）？
   答：核心是让修正项的两个梯度用**同一个样本**，从而让样本的"个性噪声"在相减时抵消。
   - 把随机梯度拆成"真实梯度 + 该样本的噪声"：同样本相减后剩下的噪声是 $n_{\xi}(x_t) - n_{\xi}(x_{t-1})$ ，两点很近时近似抵消，量级 $\le \delta \Vert x_t - x_{t-1} \Vert = \delta \eta \Vert g_{t-1} \Vert$ ，随步长缩小而缩小。
   - 若复用上一步存下的 $\nabla f_{\xi_{t-1}}(x_{t-1})$ （不同样本，省一次计算），两份独立噪声不抵消反而叠加，方差约为 $2\sigma^2$ ，是不随迭代衰减的常数——这就是论文 1.3 节的单梯度变体，收敛率掉回 $\mathcal{O}(1/T^{1/4})$ 。
   - 数值验证：真实梯度差为 $-1.6$ ；同样本（ $\xi = 3$ ）差为 $-1.6$ （几乎精确）；跨样本 $\nabla f_3(4.2) - \nabla f_1(5) = -11.6$ （完全坏掉）。
   - 代价：需保存上一步参数 $x_{t-1}$ ，对同一 batch 多做一次前向+反向，计算量约 2 倍；理论上用 2 倍计算换收敛率从 $1/T^{1/4}$ 提到 $1/T^{1/3}$ ，划算。

2. 问：表格里的收敛率 $\mathcal{O}(1/T^{1/4})$ 、$\mathcal{O}(1/T^{1/3})$ 是如何计算出来的？
   答：通用四步套路：① 由 L-光滑得 descent lemma；② 代入更新式取期望，得"单步下降 $\ge (\eta/2) \Vert \nabla f \Vert^2$ 减去噪声项"；③ 对 $t$ 求和望远镜相消（telescoping），总下降量被 $\Delta = f(x_0) - f_{\inf}$ 封顶；④ 选步长/动量参数使"优化项 $\Delta/(\eta T)$ "与"噪声项"两项相等（balancing）。
   - SGD：平均 $\Vert \nabla f \Vert^2 \le 2\Delta/(\eta T) + L \eta \sigma^2$ ，两项平衡取 $\eta \propto 1/\sqrt{T}$ ，得 $\Vert \nabla f \Vert^2 \sim 1/\sqrt{T}$ ，即 $\Vert \nabla f \Vert \sim 1/T^{1/4}$ 。
   - SGD+动量：平均降噪（乘 $\beta$ ）与陈旧偏差（正比于 $L\eta/\beta$ ）两效应相消，率不变。
   - MVR：定义误差 $e_t = g_t - \nabla f(x_t)$ ，得递推 $\mathbb{E} \Vert e_t \Vert^2 \le (1-\beta)^2 \mathbb{E} \Vert e_{t-1} \Vert^2 + 2(1-\beta)^2 \delta^2 \eta^2 \mathbb{E} \Vert g_{t-1} \Vert^2 + 2\beta^2 \sigma^2$ ；稳态误差约为 $\delta^2 \eta^2 \mathbb{E} \Vert g \Vert^2 / \beta + \beta \sigma^2$ 。取 $\eta \propto \sqrt{\beta}/\delta$ 吸收第一项，剩 $\Delta\delta/(\sqrt{\beta} T) + \beta\sigma^2$ ，对 $\beta$ 平衡得 $\beta = (\Delta\delta/(\sigma^2 T))^{2/3}$ ，代回得 $\Vert \nabla f \Vert^2 \sim (\Delta\delta\sigma)^{2/3}/T^{2/3}$ ，即 $\Vert \nabla f \Vert \sim 1/T^{1/3}$ 。
   - 这个 $\beta$ 的表达式正是推论 1 里 $\beta_T$ 的第二项来源；下界（Arjevani et al. 2023）证明 $1/T^{1/3}$ 已基本最优。
   - 关键直觉：噪声项不可避免，能做的只有"让两项相等"；MVR 的噪声项从 $\sigma^2$ 量级降到 $\beta\sigma^2$ 量级且可控，换来更好的平衡点。

---

## 公式 (4) + 算法 1：MARS（γ-MVR）的更新规则

**原文表述**（即论文公式 (4)）：

$$
g_t = (1-\beta) \left( g_{t-1} + \gamma \Delta_t \right) + \beta \nabla f_{\xi_t}(x_t)
$$

其中修正项 $\Delta_t = \nabla f_{\xi_t}(x_t) - \nabla f_{\xi_t}(x_{t-1})$ ，$\gamma \ge 0$ 为修正缩放系数。

**讲解要点摘要**：

- 与 MVR 唯一区别：修正项前多乘一个 $\gamma$ （"音量旋钮"）。$\gamma = 1$ 退化为 MVR，$\gamma = 0$ 退化为 SGD+动量。
- 凸组合视角（ $\gamma \in [0,1]$ ）：MARS 估计器 = $\gamma$ 倍 MVR 更新 + $(1-\gamma)$ 倍纯动量更新，两者共享同一个 $g_{t-1}$ 。
- 为什么要缩小修正：修正项自身带噪声。$\gamma$ 在"欠修正偏差（正比于 $1-\gamma$ ）"与"修正噪声（随 $\gamma$ 放大）"之间做权衡，本质是收缩估计（shrinkage）；综合误差正好由 γ-相似性常数 $\delta_\gamma$ 度量。
- 算法 1 逐行：①输入 $\eta, \gamma, \beta, x_0, g_0$ ；③更新参数；④抽新样本；⑤同样本两点梯度差；⑥公式 (4) 组装；⑧输出均匀随机抽一个历史迭代点（对应分析中"平均梯度平方"的界）。
- 异质曲率数值例（ $n = 3$ ，$\nabla f_1 = \nabla f_2 = 0$ ，$\nabla f_3 = 3x$ ，$f(x) = x^2/2$ ）：修正目标 $d = -0.3$ 时，各 $\gamma$ 的修正项均方误差：$\gamma = 1$ （MVR）为 $0.18$ ，$\gamma = 0$ （纯动量）为 $0.09$ ，$\gamma = 1/3$ （最优）为 $0.06$ ——样本异质性强时"不修正"能胜过"全量修正"，最优在中间，与引理 1 的 $\gamma_\star = L^2/(\delta^2 + L^2) = 1/3$ 完全吻合。
- 重要反例：若各样本曲率相同（如前面三样本 $(x-c_i)^2$ 例子），则 $\delta = 0$ ，$\gamma_\star = 1$ ，MVR 已最优—— $\gamma < 1$ 只在"梯度差结构异质（ $\delta > 0$ ）"时才有益。
- 实践对应：论文分析的是"裸版" γ-MVR；实际 MARS-AdamW 再套 Adam 预处理，理论覆盖核心更新。

**本部分问答记录**：

（待补充）

---

## 假设 1（L-光滑 + 下有界）与假设 2（无偏 + 有界方差）

**原文表述**：

假设 1：$f$ 下有界（ $f_{\inf} > -\infty$ ）且 L-光滑：

$$
\Vert \nabla f(x) - \nabla f(y) \Vert \le L \Vert x - y \Vert, \forall x, y
$$

假设 2：随机梯度无偏且方差有界：

$$
\mathbb{E}_{\xi} [ \nabla f_{\xi}(x) ] = \nabla f(x)
$$

且

$$
\mathbb{E}_{\xi} \Vert \nabla f_{\xi}(x) - \nabla f(x) \Vert^2 \le \sigma^2
$$

**讲解要点摘要**：

- 假设 1 两半部分各司其职：下有界给望远镜相消提供有限预算 $\Delta = f(x_0) - f_{\inf}$ ；L-光滑保证梯度变化速度有顶（地形无尖角/无无限弯曲），是 descent lemma 的来源，也决定步长上限 $\eta \sim 1/L$ 。
- 二阶可导时 L-光滑等价于 Hessian 特征值全在 $[-L, L]$ 。满足：二次函数（ $L = \vert a \vert$ ）、$\sin x$ 、logistic 损失；不满足：$x^4$ 、$e^x$ （曲率无界）、$\vert x \vert$ （尖角）；只满足光滑不满足下有界：$f(x) = -x$ 。ReLU 网络严格说不光滑，属标准理想化。
- 假设 2 无偏：有限和+均匀抽样时自动成立（期望=平均=全梯度）；mini-batch 也无偏且方差降为 $\sigma^2/B$ 。破坏无偏的操作：不加权的非均匀抽样、梯度裁剪。
- 三样本例子 $(x-c_i)^2$ 的 $\sigma^2$ ：偏差向量 $2(3-c_i) = (4, 2, -6)$ ，$\sigma^2 = (16+4+36)/3 = 56/3 \approx 18.7$ ，与 $x$ 无关，有界 ✓。
- 异质曲率例（ $\nabla f_i \in \{0, 0, 3x\}$ ）的方差为 $2x^2$ ，随 $\Vert x \Vert$ 增长而无界！——说明"绝对噪声有界（ $\sigma^2$ ）"和"相对噪声按距离比例有界（相似性 $\delta$ ）"是两类不同条件；定理 1 同时用到两者，各控制不同误差源。
- 若去掉：无 L-光滑 → descent lemma 失效，任何步长都可能不降反升；无下有界 → 预算无穷，梯度范数界无法建立；无无偏 → 系统性漂移到错误点；无有界方差 → 噪声项 $\beta\sigma^2$ 失控。
- 在本文公式中的落点：$L$ 出现在 $\eta = 1/(L + \sqrt{a})$ 、复杂度项 $L\Delta/\epsilon^2$ 、$\delta_0 \le L$ 、$\gamma_\star = L^2/(\delta^2 + L^2)$ ；$\sigma$ 出现在 $\beta = \min\{1, \epsilon^2/\sigma^2\}$ 及复杂度含 $\sigma$ 的三项。

**本部分问答记录**：

1. 问：光滑性为什么可以用 $\Vert \nabla f(x) - \nabla f(y) \Vert \le L \Vert x - y \Vert$ 这个公式定义？无偏估计等基础概念也忘了，要完整推理过程。
   答：分四层重建：
   - Lipschitz 连续 = "变化有限速"：变化量的范数 $\le$ 常数乘输入变化量的范数。光滑性就是把这个限速要求施加在"梯度"上（而非函数值上）：坡度不能突变 = 地形无尖角、曲率有顶；一维二阶可导时等价于 $\vert f'' \vert \le L$ 。
   - 为什么这个定义"有用"：它能推出 descent lemma。一维推导链：微积分基本定理 $f(y) - f(x) = \int_x^y f'(t) dt$ （总变化=沿途坡度累加）；由 Lipschitz，沿途坡度 $f'(t) \le f'(x) + L(t-x)$ ；逐点代入积分得 $f(y) \le f(x) + f'(x)(y-x) + L(y-x)^2/2$ 。高维用 $\varphi(\tau) = f(x + \tau(y-x))$ 降到一维 + Cauchy-Schwarz。
   - 无偏估计：估计量是随机变量，无偏 = 它的期望等于真值（平均不偏心，不是每次都准）。均匀抽样自动无偏：期望=等权平均=全梯度；用三样本例在 $x = 5$ 验证：$(8+6-2)/3 = 4 = \nabla f(5)$ 。意义：噪声随时间相互抵消而不累积；证明中的落点是 $\mathbb{E} \langle \nabla f, g \rangle = \Vert \nabla f \Vert^2$ 。
   - 方差 = $\mathbb{E} \Vert X - \mathbb{E}X \Vert^2$ （平均意义的散度）；靶心图类比：无偏=弹着点以靶心为中心，方差=弹着散布范围；SGD 无偏高方差，动量降方差但引入偏差，MVR 两者兼顾。
   - $\sup$ 的含义：对所有点对取"最坏比值"，因为证明时不知道迭代轨迹经过哪里，常数必须全域成立。

---

## 定义 1 / 公式 (5)：γ-相似性（全文核心）

**原文表述**（即论文公式 (5)）：

$$
\delta_{\gamma}^2 := \sup_{x \ne y} \frac{\mathbb{E}_{\xi} \left[ \Vert \gamma d_{\xi}(x,y) - d(x,y) \Vert^2 \right]}{\Vert x - y \Vert^2}
$$

其中两个记号：$d_{\xi}(x,y)$ 表示样本 $\xi$ 在两点的梯度之差（同样本梯度差），$d(x,y)$ 表示全梯度在两点之差（真实梯度差）。

**讲解要点摘要**：

- 作用：它就是 MARS 修正项的"质量指标"——缩放后的修正 $\gamma d_{\xi}$ 追踪真实目标 $d$ 的均方误差，按距离平方归一化后取全域最坏值。它是**定义**而非假设：每个问题都有自己的 $\delta_{\gamma}$ 曲线。
- 三步"发明"过程：① MVR/MARS 误差递推中需要控制的量正是 $\mathbb{E} \Vert \gamma \Delta_t - d_t \Vert^2$ ；② 该噪声随两点距离成比例缩小，故除以 $\Vert x - y \Vert^2$ 归一化才能得到常数；③ 轨迹未知，故取 $\sup$ 保证全域可用。与 L-光滑定义的母模板完全同构。
- 两个端点逐步验证：$\gamma = 0$ 时分子无 $\xi$ ，期望直接脱落，$\delta_0^2 = \sup \Vert d \Vert^2 / \Vert x-y \Vert^2 \le L^2$ （恰为光滑常数平方）；$\gamma = 1$ 时分子为 $\mathbb{E} \Vert d_{\xi} - d \Vert^2$ ，由于 $\mathbb{E}[d_{\xi}] = d$ （无偏性的推论），这正是 $d_{\xi}$ 的方差，即标准相似性 $\delta^2$ （公式 (6)）。
- 核心分解（通向引理 1 的桥）：固定 $x, y$ ，把 $\gamma d_{\xi} - d$ 拆成 $\gamma(d_{\xi} - d) + (\gamma - 1) d$ ，展开平方后交叉项因 $\mathbb{E}[d_{\xi} - d] = 0$ 消失，得 $\mathbb{E} \Vert \gamma d_{\xi} - d \Vert^2 = \gamma^2 \mathbb{E} \Vert d_{\xi} - d \Vert^2 + (1-\gamma)^2 \Vert d \Vert^2$ ，即"$\gamma^2$ 倍方差 + $(1-\gamma)^2$ 倍偏差平方"——γ 的偏差-方差权衡的精确数学化身，除以距离平方取 sup 即得引理 1 的上界 $\gamma^2 \delta^2 + (1-\gamma)^2 L^2$ 。
- 异质曲率例精确计算（ $\nabla f_i \in \{0, 0, 3x\}$ ）：$\delta_{\gamma}^2 = (2 + (3\gamma - 1)^2)/3$ 。端点验证：$\gamma = 0$ 得 $1 = L^2$ ；$\gamma = 1$ 得 $2 = \delta^2$ ；最小在 $\gamma = 1/3$ 处得 $2/3$ 。与公式 (4) 小节的 MSE 数字（ $0.18, 0.09, 0.06 = \delta_{\gamma}^2 \times 0.09$ ）完全一致。
- 在论文中的位置：MVR 分析里所有 $\delta$ 的位置被 $\delta_{\gamma}$ 替换，外加一笔与 $1-\gamma$ 成正比的偏差费 → 定理 1；对 $\gamma$ 最小化 $\delta_{\gamma}$ 的上界 → 引理 1；两者合力 → 推论 2 的"MARS 严格优于 MVR"。

**本部分问答记录**：

1. 问：分解式除以距离平方后取 $\sup$ 那一步（"和的 sup ≤ sup 的和"）是如何得到的？引理 1 是什么？
   答：
   - $\sup(A+B) \le \sup A + \sup B$ 的证明：对每个点对，$A \le \sup A$ 且 $B \le \sup B$ ，所以 $A+B \le \sup A + \sup B$ 处处成立；右边是一个固定数且是 $A+B$ 的上界，而 $\sup(A+B)$ 是最小上界，自然不超过它。可以严格小于：A 和 B 的"最坏点"可以不在同一处（例：A=(3,1)、B=(1,3) 在两个点上，sup(A+B)=4 < 3+3=6）。
   - 常数提出：$\sup(cA) = c \sup A$ （ $c \ge 0$ ），所以 $\gamma^2$ 、$(1-\gamma)^2$ 可以拿到 sup 外。
   - 两个 sup 分别认身份：方差比值的 sup 就是标准相似性 $\delta^2$ （定义）；真实梯度差比值的 sup 就是 $\delta_0^2 \le L^2$ （光滑性）。
   - 引理 1 三句话：① 任意 $\gamma$ 有 $\delta_\gamma^2 \le \gamma^2 \delta^2 + (\gamma-1)^2 L^2$ （即上述推导）；② 右边抛物线最低点在 $\gamma_\star = L^2/(\delta^2 + L^2) \in [0,1]$ （求导置零）；③ 最低点处 $\delta_{\gamma_\star}^2 \le \delta^2 L^2/(\delta^2 + L^2)$ ，即"并联电阻"结构 $1/(1/\delta^2 + 1/L^2)$ ，严格小于 $\delta^2$ 和 $L^2$ 两者。

---

## 公式 (6)(7)：标准相似性与二阶相似性 + 引理 1（正式单元）

**原文表述**：

公式 (6)，标准相似性（即定义 1 取 $\gamma = 1$ ）：

$$
\delta^2 := \delta_1^2 = \sup_{x \ne y} \frac{\mathbb{E}_{\xi} \left[ \Vert d_{\xi}(x,y) - d(x,y) \Vert^2 \right]}{\Vert x - y \Vert^2}
$$

公式 (7)，有限和情形（ $\xi$ 均匀取自 $\{1, \dots, n\}$ ）退化为二阶相似性：

$$
\frac{1}{n} \sum_{i=1}^{n} \Vert d_i(x,y) - d(x,y) \Vert^2 \le \delta^2 \Vert x - y \Vert^2
$$

引理 1（三条结论）：① $\delta_{\gamma}^2 \le \gamma^2 \delta^2 + (\gamma - 1)^2 L^2$ ；② 右边最小点 $\gamma_\star = L^2/(\delta^2 + L^2) \in [0, 1]$ ；③ $\delta_{\gamma_\star}^2 \le \delta^2 L^2/(\delta^2 + L^2)$ 。

**讲解要点摘要**：

- (6)→(7) 只是把期望写成等权平均：均匀抽样时 $\mathbb{E}_{\xi}[\cdot] = \frac{1}{n} \sum_i [\cdot]$ ，无新内容。
- "二阶"名字的由来：梯度差近似等于曲率乘位移（二次函数时精确：$d_i = A_i h$ ），所以 (7) 实际度量的是**各样本曲率（Hessian）相对平均曲率的离散度**，而不是梯度本身的离散度（那是 $\sigma^2$ ）。
- 一维验证：异质曲率例曲率为 $0, 0, 3$ ，均值 $1$ ，曲率方差 $= (1 + 1 + 4)/3 = 2 = \delta^2$ ✔——有限和一维情形下 $\delta^2$ 就是曲率的方差。
- 引理 1 证明链（已在定义 1 问答中完成）：偏差-方差分解 → 除距离平方 → sup 两性质 → 抛物线求导置零 → 代回得并联电阻式 $1/(1/\delta^2 + 1/L^2)$ ，严格小于 $\delta^2$ 与 $L^2$ 。
- 取等条件：二次函数时上界取等（分解式本身是恒等式，且两个 sup 在同一方向上同时达到）。
- 论文 Example 1 完整验算（ $f_i = \frac{1}{2} x^T A_i x$ ，$A_i = \hat{L} e_i e_i^T$ ，$d = n$ ）：$d_i(x,y) = \hat{L} h_i e_i$ ，$d(x,y) = (\hat{L}/n) h$ ；逐步平均后 $\delta_{\gamma}^2 = (\gamma^2 n - 2\gamma + 1) \hat{L}^2 / n^2$ ，端点：$\delta^2 = (n-1)\hat{L}^2/n^2$ ，$L^2 = \hat{L}^2/n^2$ ；$\gamma_\star = 1/n$ ，$\delta_{\gamma_\star}^2 = \frac{n-1}{n} \cdot \frac{\hat{L}^2}{n^2}$ ，改进倍数 $\delta^2 / \delta_{\gamma_\star}^2 = n$ ——样本越多改进越大，可以任意大。
- 文献定位：(6) 比 star similarity、expected similarity 等条件更一般；若 $\mathbb{E} \Vert d_{\xi} \Vert^2 \le \hat{L}^2 \Vert x-y \Vert^2$ 则 $\delta \le \hat{L}$ 。定位性内容，不影响主线。

**本部分问答记录**：

1. 问：Example 1 的推理太快看不懂，请重新慢速推一遍。
   答：先用 $n = 2$ 、$\hat{L} = 2$ 的迷你版具体数字算穿，再升到一般 $n$ ：
   - 读懂构造：$e_i e_i^T$ 是"只在第 $(i,i)$ 位置有 1"的矩阵，所以 $f_i(x) = (\hat{L}/2) x_i^2$ ——**第 $i$ 个样本只关心第 $i$ 个坐标**，其余方向完全平坦；全目标 $f = (\hat{L}/2n) \Vert x \Vert^2$ ，每坐标平均曲率 $\hat{L}/n$ ，故 $L = \hat{L}/n$ 。
   - $n = 2$ 具体算：取 $h = (1, 2)$ ，则 $d_1 = (2, 0)$ ，$d_2 = (0, 4)$ ，$d = (1, 2)$ ；验证平均正好是 $d$ （无偏）。各 $\gamma$ 的均方误差：$\gamma = 1$ 为 $5$ ，$\gamma = 0$ 为 $5$ ，$\gamma = 1/2$ 为 $2.5$ ——两端点一样坏，中点减半，印证 $\delta^2 = L^2 = 1$ 、$\gamma_\star = 1/2$ 、改进倍数 $= 2 = n$ 。
   - 一般 $n$ 的逐坐标分析：追踪误差向量 $\gamma d_i - d$ 在坐标 $i$ 上为 $\hat{L}(\gamma - 1/n) h_i$ （看得见的坐标，可能过修），在坐标 $j$ 上为 $-\hat{L} h_j / n$ （看不见的坐标，完全漏修）；平方求和后对 $i$ 平均，两项分别得 $(\gamma - 1/n)^2 \Vert h \Vert^2 / n$ 与 $(n-1) \Vert h \Vert^2 / n^3$ ，比值与 $h$ 无关，展开化简得 $\delta_{\gamma}^2 = (\gamma^2 n - 2\gamma + 1) \hat{L}^2 / n^2$ 。
   - $\gamma_\star = 1/n$ 的直觉：单样本在自己坐标上的曲率是 $\hat{L}$ ，而平均曲率只有 $\hat{L}/n$ ——它的"意见"在自己方向上比真相大 $n$ 倍，乘 $1/n$ 恰好把看得见的坐标修正得一分不差；看不见的坐标反正注定漏修，与 $\gamma$ 基本无关。$\gamma = 1$ 的灾难在于把看得见的坐标过修 $n$ 倍。

2. 问（针对附录推导文档公式 (18)）：$\delta_{\gamma}^2 = (\gamma^2 n - 2\gamma + 1) \hat{L}^2/n^2$ 如何直接得到，为什么是等号而非不等号？
   答：
   - 等号的关键：对 $i$ 平均后的矩阵 $M = \frac{1}{n} \sum_i (\gamma A_i - \bar{A})^{\top}(\gamma A_i - \bar{A})$ 每个坐标的对角元都相同（坐标 $j$ 只在 $i = j$ 时拿到特殊值、其余 $n-1$ 次拿普通值，与 $j$ 无关），故 $M = c I_n$ 是单位阵的倍数。于是 $h^{\top} M h = c \Vert h \Vert^2$ 对**所有** $h$ 精确成立（不是上界），比值恒等于 $c$ 与点对无关，sup 处处取到——两个潜在放缩点（二次型上界、取 sup）都退化为等式。
   - 算术逐步：$c = \frac{\hat{L}^2}{n} [ (\gamma - 1/n)^2 + (n-1)/n^2 ]$ ；展开 $(\gamma - 1/n)^2 = \gamma^2 - 2\gamma/n + 1/n^2$ ，又 $(n-1)/n^2 = 1/n - 1/n^2$ ，两个 $1/n^2$ 相消得 $\gamma^2 - 2\gamma/n + 1/n$ ；乘 $\hat{L}^2/n$ 得 $\hat{L}^2 (\gamma^2/n - 2\gamma/n^2 + 1/n^2)$ ；通分（ $\gamma^2/n = \gamma^2 n/n^2$ ）即 $(\gamma^2 n - 2\gamma + 1) \hat{L}^2/n^2$ 。
   - 数值校验：$n = 2, \hat{L} = 2, \gamma = 1/2$ 代入得 $(2 \cdot 1/4 - 1 + 1) \cdot 4/4 = 1/2$ ，与精读时直接逐样本计算的 $2.5/5 = 0.5$ 一致。

3. 问：从 $M = c I_n$ 到 $\delta_{\gamma}^2 = c$ 这一步没看懂，为什么直接等于 $c$ ？
   答：拆三小步：
   - ① 二次型化简：$h^{\top} (c I_n) h = c \sum_j h_j^2 = c \Vert h \Vert^2$ （逐坐标展开，$I_n$ 不改变向量）。
   - ② 比值化简：定义 1 里的比值分子就是 $h^{\top} M h$ ，故比值 $= c \Vert h \Vert^2 / \Vert h \Vert^2 = c$ 对**每一个**非零 $h$ （即每一对 $x \ne y$ ）都成立——比值的取值集合是单点集 $\{c\}$ 。
   - ③ 对单点集取 sup：$c$ 是上界且任何小于 $c$ 的数都不是上界，故 $\sup = c$ 。一般矩阵的比值会随 $h$ 方向在 $[\lambda_{\min}, \lambda_{\max}]$ 内变化，sup 才需要"找最坏方向"；$M = c I_n$ 时所有方向等价，sup 退化为常数本身。
   - 方向无关性的数值演示（ $n = 2, \hat{L} = 2, \gamma = 1/2$ ）：取 $h = (1, 2)$ 比值 $2.5/5 = 0.5$ ；换 $h = (1, 0)$ ：两样本误差平方分别为 $0$ 与 $1$ ，均值 $0.5$ ，比值 $0.5/1 = 0.5$ ——不同方向同一比值。

---

## 定理 1 / 公式 (8)：MARS 的梯度复杂度

**原文表述**（前置条件完整清单）：问题 (1) + 算法 1；假设 1（L-光滑且下有界，$\Delta = f(x_0) - f_{\inf}$ ）；假设 2（无偏、方差界 $\sigma^2$ ）；$\delta_{\gamma}$ 取自定义 1（需有限）；初始化 $g_0$ 为 $B_{init} = \lceil 1/\beta \rceil$ 个样本的平均梯度；若 $\sigma = 0$ 取 $\beta = 1$ ，若 $\sigma > 0$ 取 $\beta = \min \{ 1, \epsilon^2 / \sigma^2 \}$ 与 $\eta = 1/(L + \sqrt{a})$ ，其中（取 $\gamma \in [0,1]$ 写法）

$$
a = \left( \frac{(1-\beta)^3 (1-\gamma)^2}{\beta} L^2 + 2 (1-\beta)^2 \delta_{\gamma}^2 \right) \frac{1}{\beta}
$$

迭代数 $T \ge \lceil 2\Delta/(\eta \epsilon^2) + \sigma^2/\epsilon^2 \rceil$ 时，输出满足梯度范数平方期望 $\le 4\epsilon^2$ ，总梯度评估次数为公式 (8)：

$$
\mathcal{O} \left( \frac{\sigma^2}{\epsilon^2} + \frac{L \Delta}{\epsilon^2} + \frac{\delta_{\gamma} \Delta \sigma}{\epsilon^3} + \frac{(1-\gamma) L \Delta \sigma^2}{\epsilon^4} \right)
$$

**讲解要点摘要**：

- 参数来历：$\beta = \epsilon^2/\sigma^2$ 来自噪声地板 $\beta \sigma^2 \le \epsilon^2$ ；$B_{init} = \lceil 1/\beta \rceil$ 是热启动，使初始误差 $\sigma^2/B_{init} \le \beta \sigma^2 \le \epsilon^2$ 与稳态同量级；$\eta = 1/(L+\sqrt{a})$ 中 $\sqrt{a}$ 把误差递推里正比于 $\eta^2$ 的两项吸收进 descent lemma。
- 误差递推（与 MVR 同构，中间项换成 $\gamma \Delta_t - d_t$ ）：$e_t = (1-\beta) e_{t-1} + (1-\beta)(\gamma \Delta_t - d_t) + \beta \times$ 新鲜梯度噪声。关键新现象：$\gamma \Delta_t - d_t$ 均值不为零，拆成零均值部分（由 $\delta_{\gamma}$ 控制，定义 1 在此入场）+ 确定性偏差 $(\gamma - 1) d_t$ （由光滑性控制，量级 $(1-\gamma) L \eta \Vert g \Vert$ ）。
- 偏差比噪声贵一个 $1/\beta$ ：零均值噪声在递推展开中"方差相加"（得 $1/\beta$ 一次），偏差"幅度相加"后再平方（得 $1/\beta$ 两次）——这就是 $a$ 中偏差项多一个 $1/\beta$ 、最终落在 $\epsilon^4$ 上的原因。
- 从 $T$ 到 (8) 的算术：$\sqrt{a} \le (1-\gamma) L / \beta + \sqrt{2} \delta_{\gamma} / \sqrt{\beta}$ （根号拆项）；代 $\beta = \epsilon^2/\sigma^2$ 得 $1/\eta \le L + (1-\gamma) L \sigma^2/\epsilon^2 + \sqrt{2} \delta_{\gamma} \sigma/\epsilon$ ；乘 $2\Delta/\epsilon^2$ 逐项得 $T$ 的四块；总评估 $= 2T + B_{init}$ ，即 (8)。
- 四项账单身份：$\sigma^2/\epsilon^2$ 热启动+采样下限；$L\Delta/\epsilon^2$ 确定性下降（无噪也要付）；$\delta_{\gamma} \Delta \sigma / \epsilon^3$ 修正噪声费（MARS 可优化项）；$(1-\gamma) L \Delta \sigma^2/\epsilon^4$ 欠修正偏差费（γ 偏离 1 的代价）。后两项此消彼长，推论 2 处理最优折中。
- 特例校验：$\gamma = 1$ 时第四项消失，得 MVR 的 $\mathcal{O}(1/\epsilon^3)$ ；$\gamma = 0$ 时 $\delta_0 \le L$ ，第四项主导，得 SGD+动量的 $\mathcal{O}(1/\epsilon^4)$ ；$\sigma = 0$ 时 $\beta = 1$ 退化为确定性梯度下降 $\mathcal{O}(L\Delta/\epsilon^2)$ 。复杂度↔收敛率换算：$1/\epsilon^3 \Leftrightarrow T^{-1/3}$ ，$1/\epsilon^4 \Leftrightarrow T^{-1/4}$ 。
- 相比 Yuan et al. (2025) 的三点改进：任意固定 $\gamma$ 均成立（而非不可计算的 $\gamma_t$ ）；给出显式梯度复杂度（而非仅中间量的率）；同时覆盖 MVR 与 SGD+动量两端。

**本部分问答记录**：

1. 问：表中两个量级（ $\delta_{\gamma}^2 \eta^2 \Vert g \Vert^2$ 与 $(1-\gamma) L \eta \Vert g \Vert$ ）如何得到？偏差贵 $1/\beta$ 的机理没听懂；第 4 节算术仍有跳步。
   答：
   - 量级的钥匙只有一把：更新式 (2) 给出 $x_t - x_{t-1} = -\eta g_{t-1}$ ，所以两点距离 $= \eta \Vert g_{t-1} \Vert$ 。把它代进定义 1（在点对 $(x_t, x_{t-1})$ 上应用）得噪声二阶矩 $\le \delta_{\gamma}^2 \eta^2 \Vert g_{t-1} \Vert^2$ ；代进假设 1 得 $\Vert d_t \Vert \le L \eta \Vert g_{t-1} \Vert$ ，再乘标量 $(1-\gamma)$ 即偏差幅度。噪声报平方（随机量看二阶矩），偏差报幅度（确定量）。
   - 机理用数字看：递推展开 $e_t = \sum_s (1-\beta)^{t-s} z_s + \ldots$ ，几何权重之和约 $1/\beta$ ，权重平方和约 $1/(2\beta)$ 。取 $\beta = 0.1$ （记忆约 10 步）：每步方差 1 的零均值噪声，总方差约 $5$ （交叉项归零，方差相加，随机游走 10 步只走出 $\sqrt{10}$ ）；每步幅度 1 的同向偏差，总幅度约 $10$ ，平方 $100$ （向东走 10 步就是 10）。平方尺度上：噪声 $\times 1/\beta$ ，偏差 $\times 1/\beta^2$ ——多出的 $1/\beta$ 由此而来，与 $a$ 的两项系数一一对应。
   - 第 4 节补齐的出处：$T \ge 2\Delta/(\eta \epsilon^2)$ 来自"优化项 $2\Delta/(\eta T) \le \epsilon^2$ "的移项；$\sqrt{u+v} \le \sqrt{u} + \sqrt{v}$ 两边平方即证（差一个非负项 $2\sqrt{uv}$ ，例 $u=9, v=16$ ：$5 \le 7$ ）；$(1-\beta)$ 的各次幂 $\le 1$ 直接丢弃；三项逐一乘 $2\Delta/\epsilon^2$ 展开，再加 $2T$ 与 $B_{init}$ 合计。

2. 问：定理 1 之后关于 $\gamma = 1$ 的那段正文没有分析。
   答：那段话三层意思：
   - 代入层：$\gamma = 1$ 时第四项系数归零、$\delta_1 = \delta$ ，(8) 退化为 $\mathcal{O}(\sigma^2/\epsilon^2 + L\Delta/\epsilon^2 + \delta \Delta \sigma/\epsilon^3)$ ——与 Cutkosky & Orabona (2019) 的 MVR 复杂度同形，且 $\epsilon^3$ 项系数是更精细的 $\delta$ （相似性版本，$\delta \le \hat{L}$ 时比经典光滑版更紧）。这是定理的"回归测试"：新理论在老端点必须复现老结果。
   - 下界层："consistent with the known optimal-rate picture"指 $1/\epsilon^3$ 形态已贴着 Arjevani et al. (2023) / Fradin et al. (2026) 的下界——下界的含义是：在该问题类上**任何算法**都不可能把 $\epsilon$ 的幂次做得更好，所以 MARS 对 MVR 的改进只可能发生在**系数**（ $\delta \to \delta_{\gamma}$ ）而非指数上。
   - 诚实声明层："not a new lower bound ... but refined upper bound"——作者明确本文比较的是**两个上界表达式**（MARS 的保证不差于 MVR 的保证），而不是证明 MVR 算法本身一定更慢；上界比较是理论担保的比较，推论 2 的措辞（displayed bound expression）也是这个分寸。实验（第 7 节）才是对"算法真更快"的直接证据。

---

## 推论 1 / 公式 (9)：$T$ -依赖参数下的收敛率

**原文表述**（取 $\gamma \in [0,1]$ 写法）：给定预算 $T$ ，要求 $\beta_T \le 1$ ，选

$$
\beta_T = \sqrt{\frac{(1-\gamma) L \Delta}{\sigma^2 T}} + 2^{-1/3} \left( \frac{\delta_{\gamma} \Delta}{\sigma^2 T} \right)^{2/3}, \qquad \eta = \frac{1}{L + \frac{(1-\gamma) L}{\beta_T} + \frac{\sqrt{2} \delta_{\gamma}}{\sqrt{\beta_T}}}
$$

热启动 $B_{init} = \lceil 1/\beta_T \rceil$ ，则输出点梯度范数平方期望满足公式 (9)：

$$
\frac{2 L \Delta}{T} + \frac{4 \sigma \sqrt{(1-\gamma) L \Delta}}{\sqrt{T}} + \frac{3 \cdot 2^{2/3} (\delta_{\gamma} \Delta \sigma)^{2/3}}{T^{2/3}} + \frac{\sigma^2}{T}
$$

**讲解要点摘要**：

- 与定理 1 的关系：同一台证明机器，只是把"给定 $\epsilon$ 解 $T$ "翻转成"给定 $T$ 读出精度"；$\eta$ 的分母就是定理 1 里 $L + \sqrt{a}$ 的拆项上界。
- 母不等式（固定 $\beta$ 时的汇总）：梯度范数平方期望 $\le \frac{2L\Delta}{T} + \frac{P}{\beta} + \frac{Q}{\sqrt{\beta}} + 2\beta\sigma^2 + \frac{\sigma^2}{T}$ ，其中 $P = \frac{2(1-\gamma)L\Delta}{T}$ ，$Q = \frac{2\sqrt{2}\delta_{\gamma}\Delta}{T}$ ；五块来源：$2\Delta/(\eta T)$ 展开成前三块，噪声地板 $2\beta\sigma^2$ ，热启动残留 $\sigma^2/T$ （初始误差 $\beta\sigma^2$ 除以记忆长度 $\beta T$ ）。
- 两次平衡 + "取和"技巧：$\beta_1 = \sqrt{(1-\gamma)L\Delta/(\sigma^2 T)}$ 平衡 $P/\beta$ 与 $2\beta\sigma^2$ （各出 $2\sigma\sqrt{(1-\gamma)L\Delta}/\sqrt{T}$ ，合计系数 4）；$\beta_2 = 2^{-1/3}(\delta_{\gamma}\Delta/(\sigma^2 T))^{2/3}$ 平衡 $Q/\sqrt{\beta}$ 与 $2\beta\sigma^2$ （分别出 $2^{5/3}$ 与 $2^{2/3}$ 倍的 $(\delta_{\gamma}\Delta\sigma)^{2/3}/T^{2/3}$ ，$2^{5/3} + 2^{2/3} = 3 \cdot 2^{2/3}$ ）。取 $\beta_T = \beta_1 + \beta_2$ ：递减项在更大的 $\beta$ 下只会更小，递增项可加性拆成两份——四块两两配对即得 (9)。
- 率的读法：$\gamma = 1$ 时第二项消失，主导项 $T^{-2/3}$ （MVR 率）；$\gamma = 0$ 时 $\delta_0 \le L$ ，主导项 $T^{-1/2}$ （SGD+动量率）；四项衰减速度 $T^{-1} < T^{-2/3} < T^{-1/2}$ ，大 $T$ 时最慢的项说了算。
- 条件 $\beta_T \le 1$ 的作用：$\beta$ 是动量参数须在 $(0,1]$ 内，$T$ 足够大时自动满足。

**本部分问答记录**：

1. 问：母不等式（五块汇总）是如何得到的？
   答：六步总装：
   - ① 单步：descent lemma + 内积恒等式 $\langle u, v \rangle = \frac{1}{2}(\Vert u \Vert^2 + \Vert v \Vert^2 - \Vert u - v \Vert^2)$ （取 $u$ 为真梯度、$v = g_t$ ，差正好是误差 $e_t$ ），得：单步下降 $\ge \frac{\eta}{2}$ 梯度平方 $- \frac{\eta}{2} \Vert e_t \Vert^2 + \frac{\eta}{2}(1 - L\eta) \Vert g_t \Vert^2$ ——最后一项是负项"吸收池"，先留着。
   - ② 求和望远镜，除以 $\eta T / 2$ ：平均梯度平方 $\le \frac{2\Delta}{\eta T} + \frac{1}{T} \sum_t \mathbb{E} \Vert e_t \Vert^2 - (1 - L\eta) \cdot \frac{1}{T} \sum_t \mathbb{E} \Vert g_t \Vert^2$ 。
   - ③ 误差递推对 $t$ 求和（几何级数，泄漏率 $1 - (1-\beta)^2 \approx 2\beta$ ），平均误差拆三源：初始 $\frac{\beta\sigma^2}{2\beta T} \approx \frac{\sigma^2}{T}$ （热启动使 $\Vert e_0 \Vert^2 \le \beta\sigma^2$ ）；移动噪声+偏差合计 $a \eta^2 \times$ 平均 $\Vert g \Vert^2$ （ $a$ 就是定理 1 里那个）；新鲜噪声 $\frac{2\beta^2\sigma^2}{2\beta} = \beta\sigma^2$ （记账为 $2\beta\sigma^2$ ）。
   - ④ 吸收：$\eta = \frac{1}{L + \sqrt{a}}$ 等价于 $1 - L\eta = \sqrt{a}\, \eta$ ；移动项 $a\eta^2 \le \sqrt{a}\, \eta$ （因 $\sqrt{a}\, \eta \le 1$ ），正好被①留的负项吃掉——这就是 $\eta$ 分母里 $\sqrt{a}$ 的全部使命。
   - ⑤ 展开 $\frac{2\Delta}{\eta T}$ ：$\frac{1}{\eta} = L + \frac{(1-\gamma)L}{\beta} + \frac{\sqrt{2}\delta_{\gamma}}{\sqrt{\beta}}$ 乘开即得 $\frac{2L\Delta}{T} + \frac{P}{\beta} + \frac{Q}{\sqrt{\beta}}$ 。
   - ⑥ 输出均匀抽样 = 平均，五块合体即母不等式。

2. 问：为什么说推论 1 包含了 MVR（ $\gamma = 1$ 时 $\mathcal{O}(T^{-2/3})$ ）与动量 SGD（ $\gamma = 0$ 时 $\mathcal{O}(T^{-1/2})$ ）的速率？
   答：公式 (9) 是四项之和，大 $T$ 时**衰减最慢的非零项决定率**（ $T^{-1/2}$ 慢于 $T^{-2/3}$ 慢于 $T^{-1}$ ，如 $T = 10^6$ 时三者分别为 $10^{-3}, 10^{-4}, 10^{-6}$ ）：
   - $\gamma = 1$ ：第二项系数 $\sqrt{(1-\gamma)L\Delta} = 0$ ，$T^{-1/2}$ 项**整个消失**；剩余项中最慢的是 $T^{-2/3}$ 项（系数含 $\delta_1 = \delta$ ），故率为 $\mathcal{O}(T^{-2/3})$ ，即 MVR。括号里的 $\sigma^2/T$ 是热启动残留（初始误差 $\beta\sigma^2$ ÷ 记忆长度 $\beta T$ ），经典 MVR 陈述里没有它，故作者声明"up to"；但它衰减为 $T^{-1}$ 快于主导项，不改变率。
   - $\gamma = 0$ ：第二项系数 $\sqrt{L\Delta} > 0$ ，$T^{-1/2}$ 项活着且最慢 → 主导，率为 $\mathcal{O}(T^{-1/2})$ ；第三项用 $\delta_0 \le L$ 仍受控但衰减更快。算法本体此时就是动量 SGD，界复现其已知率，自洽。
   - 微妙点：任何固定 $\gamma < 1$ 渐进意义上都会被 $T^{-1/2}$ 项主导（系数 $\sqrt{1-\gamma}$ 小但非零）——所以 $\gamma < 1$ 的收益在**中等精度/有限预算**区间；精度要求极高时最优 γ 趋近 1，与推论 2 条件 $B < A\delta$ （即 $\epsilon > L\sigma/\delta$ ）及图 1 的对角分界线一致。

---

## 推论 2：MARS 严格优于 MVR 的条件

**原文表述**：在定理 1 设定下取 $\gamma \in [0,1]$ ，记 $A = \Delta\sigma/\epsilon^3$ ，$B = L\Delta\sigma^2/\epsilon^4$ ，$D = \delta^2 + L^2$ ，定义替代函数

$$
J(\gamma) = A \sqrt{\gamma^2 \delta^2 + (1-\gamma)^2 L^2} + B (1-\gamma)
$$

若 $B < A\delta$ 且 $\gamma_\star = \frac{L^2}{D} + \frac{B L \delta}{D \sqrt{A^2 D - B^2}}$ ，则 $\gamma_\star \in [0,1]$ 且 $J(\gamma_\star) \le J(1) = A\delta$ ；从而用引理 1 替换 $\delta_{\gamma}$ 后，MARS 的复杂度上界表达式 (8) 不大于 MVR（ $\gamma = 1$ ）的对应表达式。

**讲解要点摘要**：

- 设置：(8) 中 $\sigma^2/\epsilon^2$ 与 $L\Delta/\epsilon^2$ 与 γ 无关（公共项），只需比较非公共部分 $A \delta_{\gamma} + B(1-\gamma)$ ；用引理 1 的 $\delta_{\gamma} \le \sqrt{\gamma^2\delta^2 + (1-\gamma)^2 L^2}$ 替换即得 $J(\gamma)$ ；基线 $J(1) = A\delta$ 即 MVR 非公共项。
- 驻点推导：记 $R(\gamma) = \sqrt{\gamma^2\delta^2 + (1-\gamma)^2 L^2}$ ，则 $J'(\gamma) = A(\gamma\delta^2 - (1-\gamma)L^2)/R - B$ 。代换 $u = \gamma D - L^2$ （即 $\gamma = (u+L^2)/D$ ），有恒等式 $R^2 = (u^2 + \delta^2 L^2)/D$ ；驻点方程 $Au = BR$ 两边平方解得 $u = B\delta L/\sqrt{A^2 D - B^2}$ ，回代即 $\gamma_\star$ 公式。
- $\gamma_\star \le 1 \Leftrightarrow u \le \delta^2 \Leftrightarrow B^2 L^2 \le \delta^2(A^2 D - B^2) \Leftrightarrow B^2 D \le A^2 \delta^2 D \Leftrightarrow B \le A\delta$ ——条件 $B < A\delta$ 恰好就是"最优点落在可行区内"的充要条件。
- $J(\gamma_\star) \le J(1)$ ：$J$ 凸（凸二次式的平方根为凸，加线性项仍凸），唯一驻点即全局最小点；且 $J'(1) = A\delta - B > 0$ 说明 $\gamma = 1$ 处函数仍在上升，向左移严格下降，改进严格。
- $\gamma_\star$ 结构：第一项 $L^2/D$ 是引理 1 的纯 $\delta_{\gamma}$ 最小化点；第二项 $\ge 0$ 是偏差费 $B(1-\gamma)$ 把最优点向 $1$ 方向拉回的修正量，$B$ 越大拉得越近 1。
- 条件解读：$B/A = L\sigma/\epsilon$ ，故 $B < A\delta \Leftrightarrow \epsilon > L\sigma/\delta$ ——异质性 $\delta$ 越大、噪声-光滑乘积 $L\sigma$ 越小、精度目标越温和，MARS 越容易净赚；即图 1 对角分界线 $L\sigma/(\epsilon\delta)$ 。
- 极限校验：$B \to 0$ 时 $\gamma_\star \to L^2/D$ （退化为引理 1）；$B \to A\delta$ 时 $u \to \delta^2$ 即 $\gamma_\star \to 1$ （退回 MVR）；$\delta = 0$ 时条件永不成立（MVR 已最优，自洽）。
- 数值验证（ $\delta^2 = 2, L^2 = 1, D = 3, A = 1, B = 0.5$ ，$B < A\delta = 1.414$ ✓）：$\gamma_\star = 1/3 + 0.5\sqrt{2}/(3\sqrt{2.75}) \approx 0.476$ ；$J(\gamma_\star) \approx 1.115 < J(1) = 1.414$ ；且 $J(1/3) \approx 1.150 > J(\gamma_\star)$ ，印证偏差费确实把最优点从引理 1 的 $1/3$ 拉向 1。
- 结论分寸：比较的是两个**上界表达式**（displayed bound expressions），非算法本体；与 $\gamma = 1$ 段落的诚实声明一贯。

**本部分问答记录**：

1. 问：$A\delta_{\gamma} + B(1-\gamma) \le A\sqrt{\gamma^2\delta^2 + (1-\gamma)^2 L^2} + B(1-\gamma)$ 这个替换如何得到？
   答：
   - 只动了第一项：不等式两边的 $B(1-\gamma)$ 完全相同（不含 $\delta_{\gamma}$ ），只把左边的 $\delta_{\gamma}$ 换成了右边的 $\sqrt{\gamma^2\delta^2 + (1-\gamma)^2 L^2}$ 。
   - 依据是引理 1 第一条：$\delta_{\gamma}^2 \le \gamma^2\delta^2 + (1-\gamma)^2 L^2$ ，两边开正平方根（均非负，开方保序）得 $\delta_{\gamma} \le \sqrt{\gamma^2\delta^2 + (1-\gamma)^2 L^2}$ 。
   - 乘以非负系数 $A = \Delta\sigma/\epsilon^3 \ge 0$ 不改变不等号方向，得 $A\delta_{\gamma} \le A\sqrt{\ldots}$ ；两边同加 $B(1-\gamma)$ 保序，即得目标不等式，右边定义为 $J(\gamma)$ 。
   - 作用：$\delta_{\gamma}$ 一般不可计算，换成只含可算常数 $\delta, L$ 的显式函数 $J(\gamma)$ ，才能对 $\gamma$ 求最小；代价是不等式可能不紧（二次函数时取等）。基线 $J(1) = A\delta$ 处取等（ $\delta_1 = \delta$ 无放缩）。

---

## 6.2 节：MARS 相对 MVR 的加速图景（图 1）

**原文要点**：定理 1 上界中唯一依赖 γ 的部分是 $A\delta_{\gamma} + B(1-\gamma)$ ；用引理 1 替换后由 $J(\gamma)$ 控制；MVR 基线 $J(1) = A\delta$ ；用比值 $A\delta / J(\gamma_\star)$ 度量"界推出的加速"，图 1 画 $\log_{10}$ 下的热力图（横轴 $\epsilon$ 、纵轴 $\delta$ ，行变 $L$ 、列变 $\sigma$ ，虚线为 $\gamma_\star$ 等高线）。

**讲解要点摘要**：

- 度量的归一化：$A, B$ 均正比于 $\Delta$ ，比值 $A\delta/J(\gamma_\star)$ 中 $\Delta$ 约掉——图 1 注里写 $A = \sigma/\epsilon^3$ 、$B = L\sigma^2/\epsilon^4$ （省去 $\Delta$ ）的原因。进一步，比值只依赖三个量：$\delta, L$ 与 $B/A = L\sigma/\epsilon$ 。
- 加速上限：$B \to 0$ 极限下 $\gamma_\star \to L^2/D$ ，$J(\gamma_\star) \to A \delta L/\sqrt{D}$ ，比值 $\to \sqrt{D}/L = \sqrt{\delta^2 + L^2}/L \approx \delta/L$ （ $\delta \gg L$ 时）——**界推加速的天花板约为异质性与光滑度之比 $\delta/L$** ，与 Example 1 的 $n$ 倍改进一致（那里 $\delta/L = \sqrt{n-1}$ ，对应常数平方比 $n$ ）。
- 三个图案的成因：① 右上亮：$\epsilon, \delta$ 增大 → $B/A = L\sigma/\epsilon$ 相对变小，偏差费可忽略，比值趋近天花板 $\sqrt{D}/L$ ；同时 $\gamma_\star$ 的第二项缩小，等高线向小 γ 方向移。② 对角分界：改进条件 $B < A\delta \Leftrightarrow L\sigma/(\epsilon\delta) < 1$ ；在双对数坐标下 $\log\delta = \log(L\sigma) - \log\epsilon$ 是斜率 $-1$ 的直线，即图中对角线；线下方暗（无收益）。③ 增大 $L$ 或 $\sigma$ → $L\sigma$ 增大 → 分界线 $\epsilon\delta \approx L\sigma$ 外移，需更大的 $\epsilon\delta$ 才进入收益区。
- 数值点验：$L = 1, \sigma = 0.01, \epsilon = 0.1, \delta = 1$ ：$B/A = 0.1 < \delta$ ✓；取 $A = 1, B = 0.1$ 算得 $\gamma_\star \approx 0.535$ ，$J(\gamma_\star) \approx 0.755$ ，比值 $\approx 1.32$ （ $\log_{10} \approx 0.12$ ），低于天花板 $\sqrt{2} \approx 1.41$ 。
- 分寸："bound-implied speedup"——① 比的是上界表达式；② 只比非公共项，若公共项 $\sigma^2/\epsilon^2 + L\Delta/\epsilon^2$ 占主导，总复杂度的实际加速会被稀释；③ 引理 1 替换在非二次问题上可能不紧。
- 实践映射：LLM 预训练处于中等精度、高数据异质区域，正落在亮区——为第 7 节实验选择铺垫。

**本部分问答记录**：

（本节为应用性讨论，暂无追问）

---

## 第 7 节：实验（7.1 CIFAR-10 探针 + 7.2 MARS-AdamW 预训练）

**原文要点**：

7.1：小 CNN + CIFAR-10，用**被分析的原始双梯度 γ-MVR**。在检查点 $t$ 处，记全梯度差 $d_t$ 与 minibatch 梯度差 $d_{B,t}$ ，用 $M = 512$ 个 batch 估计

$$
\hat{\gamma}_t^\star = \frac{\Vert d_t \Vert^2}{\frac{1}{M} \sum_{m=1}^{M} \Vert d_{B_m, t} \Vert^2}
$$

结果（图 2）：$\hat{\gamma}_t^\star \in [0.10, 0.68]$ ，中位数约 0.44，全程低于 1 且随训练变化；固定-γ 扫描中最终训练损失在 $\gamma = 0.25$ 最小，$\gamma = 1$ 更差。

7.2：MARS-AdamW，124M Llama 型模型（12L/12H/768），Chinchilla 预算约 2.10B token，沿用 Semenov et al. (2025) 已调优协议，所有超参跨 γ 固定；扫 $\gamma \in \{0.01, 0.025, 0.04, 0.1, 1\}$ + AdamW 基线。三现象（图 3）：① 小 γ 可胜 AdamW（晚期曲线更低）；② 不同小 γ 表现不同、存在最优值，早/晚期排序不同；③ $\gamma = 1$ 对超参敏感、未专门调优时不稳定。

**讲解要点摘要**：

- 估计量的推导（为何是引理 1 最优点的经验版）：固定点对，最小化 $\mathbb{E} \Vert \gamma d_B - d \Vert^2 = \gamma^2 \mathbb{E} \Vert d_B \Vert^2 - 2\gamma \Vert d \Vert^2 + \Vert d \Vert^2$ （用无偏性 $\mathbb{E} d_B = d$ 化简交叉项），求导置零得 $\gamma_{opt} = \Vert d \Vert^2 / \mathbb{E} \Vert d_B \Vert^2$ ；把期望换成 $M$ 个 batch 的经验平均即 $\hat{\gamma}_t^\star$ 。又由 $\mathbb{E} \Vert d_B \Vert^2 = \Vert d \Vert^2 + \mathbb{E} \Vert d_B - d \Vert^2$ ，得 $\gamma_{opt} = \Vert d \Vert^2 / (\Vert d \Vert^2 + \mathrm{Var})$ ，代入 $\Vert d \Vert^2 \approx L^2 \Vert h \Vert^2$ 、$\mathrm{Var} \approx \delta^2 \Vert h \Vert^2$ 即引理 1 的 $\gamma_\star = L^2/(L^2 + \delta^2)$ 的逐点（局部）版本。
- 7.1 结果成因：真实数据局部 $\delta > 0$ （minibatch 梯度差能量系统性大于全梯度差）→ $\hat{\gamma}_t^\star < 1$ 全程成立；随训练变化是因为它是局部量（非 sup），非凸地形沿轨迹的 $L_t, \delta_t$ 在变；扫描最优 $\gamma = 0.25$ 落在 $\hat{\gamma}_t^\star$ 范围 [0.10, 0.68] 内，估计量与端到端最优互证。
- 7.1 的定位："theorem-aligned probe"——用的正是定理 1 分析的算法本体，验证链条无缺口；代价是任务小、非竞争性 benchmark。
- 7.2 三现象的理论对应：① AdamW 相当于无修正端，小 γ 加入受控修正 → 推论 2 收益区（中等精度、高异质）内担保更优，token 效率 = 同损失更少 token ↔ 更低复杂度；② 内部最优 γ 存在 = 推论 2 的 $\gamma_\star \in (0,1)$ ；早/晚期排序不同↔ 7.1 中 $\hat{\gamma}_t^\star$ 随轨迹变化（局部 $\delta_{\gamma}$ 项与 $(1-\gamma)$ 罚项的权衡随位置漂移）；扫描值全在 0.1 以下含义：LLM batch 异质性极强（ $\delta/L$ 大），类 Example 1 情形 $\gamma_\star = L^2/(\delta^2+L^2)$ 极小；③ $\gamma = 1$ 不稳：定理 1 要求 $\eta$ 随 $\sqrt{a}$ 缩小（ $a$ 含 $2\delta^2/\beta$ ，$\gamma = 1$ 时 $\delta_1 = \delta$ 最大化了噪声项），超参沿用 AdamW 调优值而未按 MVR 约束缩小步长 → 稳定性条件被破坏；原版 MARS 靠裁剪等机制兜底，此处被控制变量排除。
- 证据边界（论文自述）：非穷举调参，超参固定以隔离 γ 效应；小 γ 之间终损失差异较小；7.2 的 MARS-AdamW 含 Adam 预处理，不在定理 1 字面覆盖范围内——理论保真性靠 7.1，实践相关性靠 7.2，两实验互补。

**本部分问答记录**：

（待补充）

---

## 全文推导链复盘（读完后的总结）

目标命题：存在 $\gamma \in (0,1)$ 使 MARS 的梯度复杂度担保严格低于 MVR，且给出显式 $\gamma_\star$ 与成立条件。

九步链条：
1. 问题与工具箱：随机非凸优化 (1)，仅用标准假设 1（L-光滑+下有界）与 2（无偏+有界方差）。
2. 基线机制：MVR 靠同样本修正项消除动量陈旧偏差，误差递推每步收缩，达 $\mathcal{O}(1/\epsilon^3)$ 。
3. 缺口：修正项自身带噪；MARS 乘 $\gamma$ 缩放，但旧理论（Yuan et al.）只有不可计算的 $\gamma_t$ 且率与 MVR 无差别。
4. 关键发明——定义 1（γ-相似性）：度量 $\gamma d_{\xi}$ 追踪 $d$ 的归一化均方误差；端点 $\delta_0 \le L$ （光滑）、$\delta_1 = \delta$ （相似性），把两套分析连成连续谱。
5. 引理 1：偏差-方差分解（交叉项因无偏性归零）→ $\delta_{\gamma}^2 \le \gamma^2\delta^2 + (1-\gamma)^2 L^2$ → 求导得 $\gamma_\star = L^2/D$ → 最优值 $\delta^2 L^2/D$ （调和式），严格小于 $\delta^2$ 与 $L^2$ ；二次函数取等，Example 1 给出 $n$ 倍任意大改进。
6. 定理 1：误差递推中间项换为 $\gamma\Delta_t - d_t$ ，拆零均值部分（由 $\delta_{\gamma}$ 控，累积乘 $1/\beta$ ）+ 确定偏差（由光滑性控，累积乘 $1/\beta^2$ ）；四步框架汇总得账单 (8) 四项。
7. 推论 1：同机器改按预算 $T$ 记账得 (9)；$\gamma = 1$ 复现 $T^{-2/3}$ （MVR），$\gamma = 0$ 复现 $T^{-1/2}$ （动量 SGD）——回归测试通过。
8. 推论 2：非公共项经引理 1 替换为凸函数 $J(\gamma)$ ；驻点 $\gamma_\star = L^2/D + BL\delta/(D\sqrt{A^2D - B^2})$ ；条件 $B < A\delta \Leftrightarrow \epsilon > L\sigma/\delta$ 恰为可行性判据；$J'(1) = A\delta - B > 0$ 给出严格改进。命题得证。
9. 实验闭环：7.1 探针实测 $\hat{\gamma}_t^\star \in [0.10, 0.68] < 1$ 且扫描最优 $0.25$ 落在其内；7.2 小 γ 胜 AdamW 与 $\gamma = 1$ ，且 $\gamma = 1$ 不稳（步长约束被破坏）。

最终作用：① 首个解释 MARS 优于 MVR 的理论，且仅用标准假设、对任意固定 γ 成立；② 指数已被下界封顶，改进常数（ $\delta \to \delta_{\gamma}$ ）是唯一战场，本文把"算法修改"精确映射为"常数改进"；③ 实践指导：异质性越强 γ 应越小，$\hat{\gamma}_t^\star$ 可沿轨迹估计，小 γ 同时扩大稳定超参区；④ 方法论范式：为算法修改量身定制相似性型度量，可推广到其他缩放动量/方差缩减方法。

---

## 本次会话小结（每次会话结束前更新）

- 已讲完：全部主线内容——公式 (1)(2)(3)(4)+算法 1、假设 1/2、定义 1/(5)、(6)(7)、引理 1（含 Example 1）、定理 1/(8)、推论 1/(9)、推论 2、6.2 节/图 1、第 7 节实验。正文精读完毕。
- 还剩（可选）：附录证明细节（定理 1 完整证明、Proposition 1）、附录 G 实验细节。
- 下次可以从这里接着讲：如需深入附录证明，从定理 1 的完整误差递推引理开始；或转入复盘/组会汇报准备。
