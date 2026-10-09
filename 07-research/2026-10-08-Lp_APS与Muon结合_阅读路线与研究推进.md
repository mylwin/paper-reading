---
title: Lp、APS 与 Muon 结合：推荐阅读顺序与研究推进
date: 2026-10-08
scope: 接续个人知识库、第3周周报和现有NanoGPT实验
status: 文献阅读路线与待验证研究假设
---

# Lp、APS 与 Muon 结合：推荐阅读顺序与研究推进

**先读 PowerStep → SMuon → Distance-Aware Muon → ZFO → MEKA，再读低成本基线与实际Muon理论。** 这条顺序分别回答：Lp到底改变什么、哪些组合已经做过、步长已有何种自适应、额外探测是否值得、为什么局部最优可能没有训练收益。

本路线接续[完整审计与方案](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案.md)和[第3周周报](2026-10-05-第3周周报.md)。原报告的日志证据、实验预算和回退规则仍有效。阅读顺序是针对你的当前课题安排，不是通用重要性排名；有笔记的论文按下面的指定部分回读即可。

## 1. 第一轮：确定问题与创新边界

如果对NS实现仍不熟，先用20–30分钟看[Muon作者说明](https://kellerjordan.github.io/posts/muon/)。对照现有代码，区分动量、Nesterov输入、有限NS、矩阵形状缩放和辅助参数路由。下面的时长仅是阅读安排估计。

| 顺序 | 推荐阅读与原文链接 | 重点部分 | 要解决的问题与读后产出 |
|---|---|---|---|
| 1 | **PowerStep: Memory-Efficient Adaptive Optimization via ℓp-Norm Steepest Descent，v2**：[HTML](https://arxiv.org/html/2605.10335v2) · [PDF](https://arxiv.org/pdf/2605.10335v2) | §2、§3、§5.6；证明需要时看附录D。已有笔记者先核对v2改动，约1小时 | 写清“动量后坐标保号幂”与pbSGDM“先变换梯度再累积”的区别。§5.6已有Muon隐藏矩阵配PowerStep辅助参数，故这类组合不能作为首次提出。新版理论含噪声残差，不能直接沿用旧版最优速率表述。 |
| 2 | **From SGD to Muon: Adaptive Optimization via Schatten-p Norms（SMuon）**：[HTML](https://arxiv.org/html/2605.19781v1) · [PDF](https://arxiv.org/pdf/2605.19781v1) | §2.1、§3–4、附录A.2.5–A.2.6及B，约2–3小时 | 区分坐标幂与奇异值幂。该文已自动选谱几何、低频刷新底层统计，并推导代理最优步长。必须读附录B的步长尖峰现象；产出“已覆盖/仍需验证”对照表。 |
| 3 | **Distance-Aware Muon: Adaptive Step Scaling for Normalized Optimization**：[HTML](https://arxiv.org/html/2605.18999v1) · [PDF](https://arxiv.org/pdf/2605.18999v1) | §2–5，接着§6.1，约2小时 | 将轨迹半径、动量下降证书和recentered标量搜索分别写成公式。圈出各自依赖的光滑性、局部化/星凸条件。它是“Muon自适应尺度”的直接近邻，不能省略。 |
| 4 | **Trust the Direction, Search the Step: Zero-and-First-Order Methods for LLM Fine-Tuning（ZFO）**：[HTML](https://arxiv.org/html/2610.02190v1) · [PDF](https://arxiv.org/pdf/2610.02190v1) | §2、§3.1、§4.2、§5，约1.5–2小时 | 固定一阶方向，用当前梯度及两次额外前向构造一维模型。画清公共随机数、搜索区间与调用预算。回答“为什么要多一次反向，而不只做前向探测？”微调成本不能直接外推到NanoGPT预训练。 |
| 5 | **Self-Tuning Stochastic Optimization with Curvature-Aware Gradient Filtering（MEKA，2020）**：[正式页面](https://proceedings.mlr.press/v137/chen20a.html) · [PDF](https://proceedings.mlr.press/v137/chen20a/chen20a.pdf) | 先摘要与实验/讨论，再按需回读滤波和选步公式，约45–60分钟 | 曲率校正和自动选步有合理模型，却没有在深度学习上超过调好的基线。列出本课题必须检验的“估计更准却训练不更好”失败模式。无需先复现完整Kalman滤波。 |

**第2篇最值得优先精读。** 它同时与你的Lp兴趣、Muon方向和自适应几何最接近。第3、4篇决定标量步长路线的直接对照；第5篇用于防止将局部拟合收益误当训练收益。

## 2. 第二轮：确定基线、理论对象与噪声边界

| 顺序 | 推荐阅读与原文链接 | 重点部分 | 对本课题的用途 |
|---|---|---|---|
| 6 | **AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates（AdaGO，OPT2025）**：[PDF](https://www.opt-ml.org/papers/2025/paper99.pdf) | §2/Algorithm 1、§3假设、实验配置 | 标量历史梯度范数自适应，不需要额外梯度，是“探测值得吗”的成本基线。不能把小模型实验表述为已验证大规模LLM收益。 |
| 7 | **Convergence of Practical Muon with Finite Newton–Schulz Iterations and Nesterov Momentum**：[HTML](https://arxiv.org/html/2609.39595v1) · [PDF](https://arxiv.org/pdf/2609.39595v1) | §2、§3.2/Proposition 3.3与Theorem 3.4、§4.1及4.3 | 核对实际有限NS方向的对齐与能量，而非用精确极因子替代代码。标出理论采用的时域相关学习率/动量条件，不将其当固定0.95动量配方的现成证明。 |
| 8 | **DGA-Muon: Decoupled Geometry-Aligned Adaptive Scaling for Muon，v2**：[HTML](https://arxiv.org/html/2610.06578v2) · [PDF](https://arxiv.org/pdf/2610.06578v2) | §3–4、Algorithm 1；§5先读假设、§6看配置 | 看原始梯度为何提供正交后统计缺失的信息；分别核对宽/高矩阵缩放。几何对齐的行列缩放与联合标量半径不同，是当代自适应Muon基线。保持互相垂直不等于保持单位范数和平坦奇异值。 |
| 9 | **Muon Under Gradient Noise and the Limits of Orthogonalization Near Optima**：[HTML](https://arxiv.org/html/2609.32861v1) · [PDF](https://arxiv.org/pdf/2609.32861v1) | §3–6、§7.4及§8；定理证明后读 | 区分平均响应、残差方差和实际下降。小信号下的噪声机制给响应校准提供动机，但不能直接推出LLM上某个切换规则最优。与第3周坐标幂路线衔接。 |
| 10 | **Muon is Not That Special: Random or Inverted Spectra Work Just as Well（Freon/Kaon）**：[HTML](https://arxiv.org/html/2605.11181v1) · [PDF](https://arxiv.org/pdf/2605.11181v1) | 方法总览、附录C、H.3；必要时看I的诊断 | 区分诊断量和可在线预测的控制量。逐步选幂与最优局部步长应结合其随机特征设定理解，不能把单步贪心自动选择当作自然保证收益。 |

第6–10篇不必全部读完证明才开始机制验证。先抽取算法、关键条件、费用和失败结果，每篇形成半页对照；等确定要用的证明工具再精读附录。

## 3. 本次继续核验后，对方案的实质修正

### 3.1 SMuon不止选择谱幂，也处理代理最优步长

[SMuon附录A.2.5–A.2.6与B](https://arxiv.org/html/2605.19781v1)已在随机特征代理中同时分析谱几何和标量步长。附录B还报告，直接采用代理最优步长可能在几何改变时出现loss尖峰。因此，“用对齐/曲率选步”“自动选幂”“低频刷新”都不能单独成为本课题的新颖性主张。

**本次推论：** 首轮固定实际NS5方向并限制半径，能减少同时改变方向和尺度造成的混淆；但这只是较清楚的检验方式，并不自动构成新算法贡献。可能的增量须来自反馈可靠性、摊销成本或跨配方迁移的实证与分析。

### 3.2 将主要假设写成一个可被否定的问题

> 在固定实际Muon方向和辅助优化器的条件下，稀疏同batch、同随机数的方向梯度差，能否比原APS、历史梯度范数和前向探测，在同总成本下更可靠地选择更新半径？

与近邻的区别及尚缺证据如下。这里“区别”不等于“首次提出”。

| 近邻 | 已有信息来源 | 本方案要单独证明什么 |
|---|---|---|
| SMuon | 梯度、动量、激活形成层级随机特征代理，并选择谱几何 | 真实网络联合方向割线反馈是否更有预测力，能否抵偿额外反向费用 |
| Distance-Aware | 轨迹距离、动量下降证书、光滑上界与标量majorization | 实际随机训练中测得并缓存的局部反馈，是否优于现成尺度控制 |
| ZFO | 同batch当前梯度及两次额外前向的一维局部模型 | 梯度差相对loss差是否数值更稳、样本外更有效；比较测得的开销，不能先认定更省 |
| AdaGO | 当前及累计梯度范数 | 方向曲率提供的额外信息，是否确实超出低成本范数统计 |
| DGA-Muon | 原始梯度的行/列二阶统计 | 保持原方向的联合标量控制，在何种配方下具有独立作用 |

### 3.3 保留最小公式，暂时不引入新谱幂控制器

令 $D_t$ 是代码实际产生的联合隐藏矩阵方向，包含有限NS与各块形状缩放；定义

$$
Q_t=\frac{D_t}{\|D_t\|_F},\qquad
r_t^{\mathrm{base}}=\eta_t^{\mathrm{Muon}}\|D_t\|_F.
$$

这里 $Q_t$ 是单位联合方向，$r_t^{\mathrm{base}}$ 是原Muon的物理更新距离。所有隐藏矩阵同时按该方向探测，辅助参数冻结；同batch、同随机数计算

$$
G_t^{\mathrm{probe}}=\nabla F_{S_t}(W_t-\delta_tQ_t),\qquad
a_t=\langle G_t,Q_t\rangle,\qquad
\widehat L_t=\frac{\langle G_t-G_t^{\mathrm{probe}},Q_t\rangle}{\delta_t}.
$$

$a_t$是一阶下降信息，$\widehat L_t$是探测线段上的平均方向曲率，含隐藏层间耦合；它不是整个后续更新区间的曲率上界。同样本差分不消除采样目标本身的随机性。用有效正曲率缓存 $\bar L_t$ 暂时控制

$$
r_t=\min\left(r_t^{\mathrm{base}},\;\tau\frac{\max(a_t,0)}{\bar L_t}\right),\qquad
W_{t+1}=W_t-r_tQ_t.
$$

起始设置沿用原报告：$\tau=1/2$，每32步探测，先只允许缩步；非法曲率、缓存过期、非下降方向按原报告§6.3回退。方向为零时跳过矩阵更新。warmdown只在基准半径中作用一次。该规则尚无随机LLM收敛证明，也不是无超参数方法。

## 4. 读完以后具体做什么

用约5个工作单元推进；这是任务顺序，不要求每日读完固定篇数，也不预先启动GPU训练。

1. **公式对齐。** 读1–2，列pbSGDM、Lp-SGDMA、PowerStep、Muon、SMuon的动量输入、非线性位置、方向尺度与实际更新。特别标清不同论文的 $p$、幂指数和动量符号不相同。
2. **近邻与成本对齐。** 读3–5，形成一页比较：信息来源、额外forward/backward、缓存/矩阵状态、成立条件、目标任务。将SMuon附录B与MEKA负结果写入失败假设。
3. **现有APS机制消融。** 优先设计原APS与固定 $1/c$ 对照；保持首步、方向、实际调度一致，区分省去第二反向与最终质量。沿用原报告E0，历史日志不得覆盖。现有饱和统计支持做这个实验，不等于已经证明APS无用。
4. **冻结checkpoint的预测检验。** 沿用原报告E1的早/中/末checkpoint、独立选择/评分batch；用同一方向比较APS、割线、ZFO式反馈。先检查探测距离减半时的稳定性，再看独立batch和16步分支收益。所有分支从相同参数及优化器状态恢复。
5. **通过机制门槛再闭环。** 稀疏曲率只有在有样本外预测力、缓存可用且开销可控时才进入短跑。正式报告用等tokens与等总时间、多配对种子、相同辅助优化器与调参次数；加入至少一个直接标量近邻及ZFO成本对照。

如果只预测当前batch、相对固定/时间半径没有增益，或仅每步额外反向有效且等墙钟无收益，应停止将这条路线作为主线，回到第3周已有的响应校准坐标幂路线。不要为维持假设继续叠加控制器。

## 5. 需要扩展时再读的分支

| 触发问题 | 追加阅读 | 阅读目的 |
|---|---|---|
| 决定研究谱幂随时间变化 | [DynMuon v3](https://arxiv.org/html/2605.17109v3)、[Muonᵖ](https://arxiv.org/html/2606.13867v1) | 先对照已有时间策略，避免把可由日程解释的变化称作反馈优势 |
| 标量半径不足，决定改方向 | [Beyond the Matrix Sign: Quadratic Spectral Descent](https://arxiv.org/abs/2609.07597)、[MALT/MALTER](https://arxiv.org/html/2608.05088v1) | 核对曲率改变方向的近邻、状态及开销；这会扩大课题范围 |
| 认为谱bulk分配比半径重要 | [Spectral Allocation / SAMuon](https://arxiv.org/abs/2608.25990v1) | 先与简单静态谱先验比较 |
| 需要交替几何节省计算 | [LionMuon](https://arxiv.org/abs/2609.35297v1) | 比较交替策略与真实通信/计算预算 |
| 变化主要来自embedding/head | [MuonIO](https://arxiv.org/abs/2610.02705v1) | 单独研究辅助参数，避免混淆隐藏矩阵机制 |
| 将来转向MoE | [ExpertMuon-Compass](https://arxiv.org/abs/2610.04140v1) | 进一步完整核验专家对齐步长；当前只有摘要级结论，不作为稠密NanoGPT性能证据 |

这些不是开始第一轮机制检验前的必读清单。原报告已登记ORCA、MeqMuon、DeVA与SoftServe等近邻，按实际机制需求再展开。

## 6. 知识库中的已有入口与版本说明

截图中的Lp-SGDA/SGDMA原始论文尚未准确定位。PowerStep是相关幂变换近邻，不是已确认的截图出处；现阶段对原APS的判断来自截图、实际代码与日志，不能据此替代对原论文定理和假设的核验。

- [PowerStep已有精读](../03-notes/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent/精读.md)：本地原PDF为v1，本次建议对照在线v2；没有覆盖你的原PDF或笔记。
- [MEKA已有精读](../03-notes/2026-07/Self_Tuning_Stochastic_Optimization_with_Curvature_Aware_Gradient_Filtering/精读.md)：2026-07是入库月份，正式论文为2020年；这次回读重点是深度学习负结果。
- [DGA-Muon本地PDF](../01-raw/2026-10/DGA_Muon_Decoupled_Geometry_Aligned_Adaptive_Scaling_for_Muon.pdf)：方法对应原报告附录A入口。
- [第3周周报](2026-10-05-第3周周报.md)：§五已有响应校准坐标幂，附录D已有方向/尺度分离与预算要求。本次路线接着推进，不将已有想法重新包装。

**截至2026-10-08的核验状态：** 主读清单的原文网页/PDF均已取得；新补读SMuon附录A.2.5–A.2.6及B，并复核MEKA正式出版信息。本文记录正文/定点核验结果，不宣称全文所有定理和实验均已独立复现。扩展清单的精读深度以原报告为准。

**仍待核验：** 日报中的[Can Muon Adapt Its Stepsize Without Knowing the Solution Distance?](https://openreview.net/forum?id=Ghfc8IRXdy)；[PDF入口](https://openreview.net/pdf?id=Ghfc8IRXdy)。forum/PDF读取未成功，续查公开API返回403并要求challenge verification，未取得题录或正文。因此不将它当作已确认独立论文，也不推断它与Distance-Aware是否为同一工作；论文新颖性判断仍须补齐此项。
