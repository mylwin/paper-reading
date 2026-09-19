# 论文精读笔记：Parameter-free Clipped Gradient Descent Meets Polyak

- 作者/来源：Yuki Takezawa, Han Bao, Ryoma Sato, Kenta Niwa, Makoto Yamada。NeurIPS 2024 Main Conference Track，arXiv:2405.15010
- 所属子方向：自适应学习率 / Polyak 步长 + 梯度裁剪收敛性分析（$(L_0,L_1)$ 光滑，确定性设定）
- 精读开始日期：2026-09-18
- 模式：交互模式（一次一个知识单元，从论文第一行开始）
- 配套定稿：[全局推理.md](../../04-equation_problem/Parameter_Free_Clipped_Gradient_Descent_Meets_Polyak/全局推理.md)（此前全局推理模式产出的完整推导手稿，可对照）

---

## 精读路线图（按论文行文顺序）

- [x] 单元 1：式 (1) 问题设定 + 三个隐含前提 + 凸性一阶条件
- [x] 单元 2：梯度下降更新式 + Assumption 1（$L$ 光滑）+ 下降引理
- [ ] 单元 3：Theorem 1（Nesterov, Corollary 2.1.2）—— 最优步长下的 $\mathcal{O}(L/T)$
- [ ] 单元 4：式 (4) 裁剪梯度下降 + Assumption 2（$(L_0,L_1)$ 光滑）
- [ ] 单元 5：Theorem 2（Koloskova et al., Thm 2.3）—— 调好参的裁剪率
- [ ] 单元 6：式 (7) Polyak 步长的来源推导 + Theorem 3（Hazan–Kakade, Thm 1）
- [ ] 单元 7：Proposition 1（本文新增）—— Polyak 步长的隐式裁剪下界
- [ ] 单元 8：Proposition 2（Jiang–Stich, Lemma 15）对照
- [ ] 单元 9：Theorem 4（本文主定理一）+ 附录 A 证明（Lemma 1、Lemma 2、两区分情形）
- [ ] 单元 10：式 (13)/(14) 与 Algorithm 1（Inexact Polyak Stepsize）
- [ ] 单元 11：Theorem 5（本文主定理二）+ 附录 B 证明（Lemma 4、Lemma 5）
- [ ] 单元 12：Table 1 三种 parameter-free 方法逐项对比与审计
- [ ] 单元 13：附录 C（Lemma 6 + Proposition 3，合成函数为何满足 $(L_0,L_1)$ 光滑）
- [ ] 单元 14：实验（合成函数与神经网络）设置与主张-证据核对

---

## 单元 1：式 (1) 问题设定与它隐含的三个前提

**原文表述**：

$$
\min _ {x \in \mathbb {R} ^ {d}} f (x),
$$

其中论文在 §1 开头写明 "the loss function $f$ is convex and lower bounded"。

**讲解要点摘要**：

- 作用：界定全文研究对象——无约束、确定性、凸的损失最小化；$\min$ 上加 $\arg$ 才是最优点，$\min$ 本身是最优值。
- 关键推导来源：本文自己提出的设定；但"凸 + 下有界 + 最优点存在"这三条是后面每一个定理的前提清单（Theorem 2、4、5 都逐条重述）。
- 本单元导出两个全文反复使用的工具：
  1. 凸性一阶条件 $\nabla f(x)^{\top}(x - x^{\star}) \ge f(x) - f^{\star}$，是所有距离递推中"唯一一次不等号放缩"；
  2. 零梯度即全局最优 $\nabla f(x_t) = 0 \Rightarrow x_t$ 为最优点，用来处理 Polyak 步长分母为零的边界。
- 统一记号（沿用定稿）：$\Delta_t = f(x_t) - f^{\star}$、$G_t = \Vert \nabla f(x_t) \Vert$、$R_t = \Vert x_t - x^{\star} \Vert$。
- 强调：本文主定理是**确定性**分析，$x_{t+1} = x_t - \eta_t \nabla f(x_t)$ 用的是完整梯度，全程没有期望、没有随机梯度 $g_t$、没有方差假设。

**本部分问答记录**：

（待用户提问后追加）

---

## 单元 2：梯度下降更新式、Assumption 1（$L$ 光滑）与下降引理

**原文表述**：§2.1 的更新式与 Assumption 1（式 (2)），以及附录 A 的 Lemma 1（式 (16)）。

**讲解要点摘要**：

- 更新式 $x _ {t+1} = x _ {t} - \eta _ {t} \nabla f (x _ {t})$ 用**完整梯度**，全文确定性设定；$\eta _ {t}$ 允许随 $t$ 变化，这正是"parameter-free"要自动决定的东西。
- 工作马（全文所有证明的第一行）：

$$
\Vert x _ {t+1} - x^ {\star} \Vert ^ {2} \le \Vert x _ {t} - x^ {\star} \Vert ^ {2} - 2 \eta _ {t} \Delta _ {t} + \eta _ {t} ^ {2} G _ {t} ^ {2}
$$

  来源：更新式减 $x^{\star}$、按平方范数恒等式展开、只用一次凸性一阶条件（武器 1）。
  ⚠️ 这一步**完全没有用到光滑性**，论文 §2.3 直接给出未推导，我们已补全。
- Assumption 1：$\Vert \nabla f (x) - \nabla f (y) \Vert \le L \Vert x - y \Vert$，即梯度映射 Lipschitz 连续；二阶可导时等价于 Hessian 特征值上界为 $L$。
- 下降引理（由 Assumption 1 完整推出，链式法则 + Cauchy–Schwarz + $\int _ {0} ^ {1} \theta d \theta$ 型积分）：

$$
f (y) \le f (x) + \nabla f (x) ^ {\top} (y - x) + \frac {L}{2} \Vert y - x \Vert ^ {2}
$$

- Lemma 1（式 (16)）由下降引理取 $y = x - \frac {1}{L} \nabla f (x)$ 得到：

$$
\frac {1}{2 L} \Vert \nabla f (x) \Vert ^ {2} \le f (x) - f^ {\star}
$$

  关键性质：**只需 $L$ 光滑，不需要凸性**，所以它是光滑函数的定理而不是额外假设，方向与 PL 条件相反（PL 是假设，此式是结论）。
- 溯源：Assumption 1 与下降引理是 Nesterov (2018, Lectures on Convex Optimization) 的标准内容；Lemma 1 论文标注"See Lemma 2.28 in Garrigos and Gower (2023)"，属直接套用。handbook 中的**具体编号我们未逐字核对**（本地无 PyYAML，未联网取原文；结论已独立重推验证）。
- 数值代入要点：
  - $f (x) = x ^ {2} / 2$、$L = 1$：下降引理与 Lemma 1 都**取等号**（二次函数的二次上模型是精确的）。
  - $f (x) = \log (1 + e ^x)$、$L = 1/4$：$x = 0$、$y = 1$ 处 $1.31326 \le 1.31815$ ✓；Lemma 1 给 $0.5 \le 0.69315$ ✓，且此处 $f^{\star}$ 取不到（下确界 0 在 $x$ 趋于负无穷处），说明 Lemma 1 不依赖最优点存在。
  - $f (x) = x ^ {4}$ 在整条实轴上对任何有限 $L$ 都**不满足** $L$ 光滑（需 $L \ge 4 a ^2$ 对所有 $a$ 成立），这正是 Assumption 2 要解决的情形，也解释了 §6.1 为什么用四次函数做合成实验。
- 埋线：把 §2.1 那一步的右端看成 $\eta$ 的二次函数 $q ( \eta ) = - 2 \eta \Delta _ {t} + \eta ^ {2} G _ {t} ^ {2}$，顶点在 $\eta = \Delta _ {t} / G _ {t} ^ {2}$ —— 这就是式 (7) Polyak 步长的**唯一来源**（论文 §2.3 原话 "By minimizing the right-hand side"），单元 6 正式讲。

**讲解标准（2026-09-18 本会话两次修正后的最终版）**：

1. 不只推公式——**论文提到的每一个符号都要解释，算法要逐行解释**，目标是完全理解全文数学理论。
2. 但交付方式必须是**渐进式**：符号**随着公式一步一步就地解释**，写出某个式子后立刻按"从左到右、按推导中首次用到的顺序"逐个说明该式里的每个符号（字母 / 下标 / 上标 / 范数取哪一种 / 求和与期望的范围 / 论文自定义的常数与缩写），并标注它属于已知量、未知量、模型参数、超参数还是中间结果；同一符号再次出现时一句回指即可。
3. **明确取消**"先给一张全文符号总表（变量 / 常数 / 记号 / 编号对象 / OCR 坑）再往下讲"的做法——那样解释与推导脱钩。单元结尾可以放一个只含本单元符号的小结表用于复习，但它不替代就地解释。
4. 因此**单元 2 需要按新标准重讲**（原讲解里那张前置总表不作数）；OCR 还原、编号勘误这类信息改到"用到它的那一步"当场说明。

**单元 2 新增审计发现（已 grep 原文核对）**：

- `DoG` 只在 §6.2 Results（第 307 行）与 Fig. 3 / Fig. 5 图例出现，全文**从未定义**其更新式，参考文献仅有 Ivgi et al. (2023)。而 §6.2 的主张之一"DoG 在 LSTM 上最好、在 Nano-GPT 上不稳定"依赖一个论文没给定义的算法，复现必须回原文。
- §6.2 正文写 "Figure 4 shows the loss curves"，但主文 loss curves 是 **Figure 3**；Figure 4 在附录 E 且只含 LSTM 与 Nano-GPT。同段 "For T5 ... as shown in Fig. 4" 同样对不上。属图表交叉引用错误。
- §6.2 写 "For Inexact Polyak Stepsize, Theorem 4 requires the selection of the best parameters"，而 best-iterate 规则属于 **Theorem 5 与 Algorithm 1**。定理编号引错。
- OCR / 排版坑（读 `02-markdown` 时需人工还原）：§1 中 "$\mathbf{\Delta}\mathbf{x}^{\star}$ is the optimal solution" 应为 $x^{\star}$；Theorem 3 中 $\bar{x}$ 被识别成 $\bar{\mathbf{r}}$；§6.1 合成函数分母 "$7\mathcal{D}$" 应为 $72$（与 Proposition 3 对照）；附录 A 证明里多处 $\nabla f ( \pmb { x } )$ 丢了下标 $t$。

**本部分问答记录**：

（暂无追问）

---

## 本次会话小结（每次会话结束前更新）

- 已讲完：单元 1、单元 2
- 还剩：单元 3 及以后
- 下次可以从这里接着讲：单元 3（Theorem 1，Nesterov 2018, Corollary 2.1.2：$\eta_t = 1/L$ 时的 $\mathcal{O}(L R_0^2 / T)$，以及它为什么是"最优步长"）
