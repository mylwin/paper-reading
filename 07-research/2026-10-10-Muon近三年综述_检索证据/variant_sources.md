---
title: Muon 变体近邻与原始来源核验
date: 2026-10-10
cutoff: 2026-10-10 Asia/Shanghai
scope: 2025—2026；Muon + APS + Lp 的直接近邻
---

# Muon 变体：来源、版本与可行性限制

本文件是综述的检索证据附录。原始 arXiv 题录和指定正文已于 2026-10-10 重新访问；旧知识库笔记只用来发现 URL，不作为论文事实的证据。阅读等级为“摘要/题录”“方法定点阅读”，不表示已重审全文证明或复现实验。所有下列最新版本均早于截止日。Chrome CDP 前置检查显示未连接；公开静态检索仍可进行。OpenReview 的未访问成功条目单列，不能视作排除。

## 1. 版本与正文定位表

| 工作 | 首次提交 / 截止日可见最新版本 | 原始来源与阅读位置 | 核验等级 |
|---|---|---|---|
| AdaMuon，Si / Zhang / Shen | 2025-07-15 / v3 2025-12-24 | [题录](https://arxiv.org/abs/2507.11005)、[v3 Alg.1 / §3](https://arxiv.org/html/2507.11005v3) | 方法定点 |
| AdaGO，Zhang / Liu / Schaeffer | 2025-09-03 / v2 2025-09-06 | [题录](https://arxiv.org/abs/2509.02981)、[OPT2025 原文 p.3–4 / Alg.1](https://www.opt-ml.org/papers/2025/paper99.pdf) | 方法定点；工作坊版 |
| DeVA，Song / Bai / Zhang / Bullins / Gleich | 2026-02-06 / v2 2026-05-26 | [题录](https://arxiv.org/abs/2602.06880)、[v2 §3 / Alg.2、附录矩估计变体](https://arxiv.org/html/2602.06880v2) | 方法定点 |
| Freon / Kaon，Shumaylov et al. | 2026-05-11 / v1 | [题录](https://arxiv.org/abs/2605.11181)、[v1 摘要、附录 H.2–H.3](https://arxiv.org/html/2605.11181v1) | 方法定点；主实验未全审 |
| DynMuon，Wu / Shah / Silwal / Zhang | 2026-05-16 / v3 2026-06-01 | [题录](https://arxiv.org/abs/2605.17109)、[v3 §2、谱日程消融](https://arxiv.org/html/2605.17109v3) | 方法/消融定点 |
| Distance-Aware Muon，Demidovich et al. | 2026-05-18 / v1 | [题录](https://arxiv.org/abs/2605.18999)、[v1 §2–5 / Alg.1–3](https://arxiv.org/html/2605.18999v1) | 方法/条件定点 |
| SMuon，Massena / Friedrich / Serrurier | 2026-05-19 / v1 | [题录](https://arxiv.org/abs/2605.19781)、[v1 §3–5、附录 A.2.5–A.2.6 / B](https://arxiv.org/html/2605.19781v1) | 方法/尺度失败定点 |
| Muon^p，Dong / Sawin | 2026-06-11 / v1 | [v1 题录](https://arxiv.org/abs/2606.13867v1)、[v1 §2](https://arxiv.org/html/2606.13867v1) | 方法定点；无版本 abs 首次请求失败，指定 v1 成功 |
| MALT / MALTER，Wu / Dong / Sun / Ma | 2026-08-05 / v1 | [题录](https://arxiv.org/abs/2608.05088)、[v1 §3.5 / Alg.2](https://arxiv.org/html/2608.05088v1) | 方法定点 |
| QSD，Zhang / Sun / Liu | 2026-09-07 / v2 2026-09-26 | [题录](https://arxiv.org/abs/2609.07597)、[v2 §2–3.3、附录 E / I / J / K](https://arxiv.org/html/2609.07597v2) | 方法/近邻定点；非全文证明审计 |
| ZFO，McGee / Bergou / Dutta | 2026-10-01 / v1 | [题录](https://arxiv.org/abs/2610.02190)、[v1 §2 / Alg.1](https://arxiv.org/html/2610.02190v1) | 方法定点；题录自报 NeurIPS2026 接收，未独立核会务名单 |
| DGA-Muon，Zhang / Yu | 2026-10-05 / v2 2026-10-07 | [题录](https://arxiv.org/abs/2610.06578)、[v2 §3–5](https://arxiv.org/html/2610.06578v2) | 方法/理论条件定点 |

## 2. 按研究问题组织的近邻

### 2.1 固定方向，控制标量尺度

**AdaGO**（原文 p.3–4）：积累截断梯度范数平方，并用当前截断梯度范数 / 累计平方根给正交动量方向选标量；含步长下限和初始累计量。增加一个标量，不额外计算第二梯度。实验是 CIFAR-10 与函数回归，不能写成已大规模验证 LLM。证明使用理想 Orth 与光滑、无偏有界方差等条件；有限 NS 实现与精确正交不同。[原文](https://www.opt-ml.org/papers/2025/paper99.pdf)

**Distance-Aware**（§2–5）：给出轨迹距离适配、梯度/动量下降证书、重心化信赖域与一维 majorized 搜索三类算法。非凸驻点结论需有界轨迹；末迭代目标差结论使用 star-convex 及局部条件或相应不同条件；理论保留光滑常数，不是所有参数全自动。实验包含 GPT-124M/WikiText-103，仍不能直接覆盖现有 NS5 配方。[原文](https://arxiv.org/html/2605.18999v1)

**ZFO**（§2）：固定一阶更新的单位方向，以同样本两个对称前向估计二、三阶方向导数，在有限区间比较 Taylor / Padé 候选，遇极点回退。是“信任方向、搜索半径”的直接先例；主要 LLM 场景是微调，不能借其成本结论直接推断预训练开销。[原文](https://arxiv.org/html/2610.02190v1)

父任务另行核验 **MuonMax-Momo**（Crawshaw et al., [2510.09827](https://arxiv.org/abs/2510.09827)，[ICML2026 原文](https://proceedings.mlr.press/v306/crawshaw26a.html)）。其核范数尺度、Momo 模型步长及低频估计也必须约束“自适应步长 + 单状态”新颖性。本附录不冒称独立读完该论文。

### 2.2 谱幅度、谱幂与自动选几何

**SMuon**（§3–5、A.2.5–A.2.6、B）：Schatten-$(p+1)$ 方向为 $U\Sigma^{1/p}V^T$。随机特征代理以 alignment² / curvature 选层级 $p$，EMA 平滑基础谱统计且可低频刷新；另有 Adam 二阶矩版本。附录已推导代理联合最优步长 $\eta^*=kN(p)/D(p)$，但直接用代理倍率会造成 loss spikes。故谱幂、自动选指数、低频控制和方向曲率式选步均已有覆盖。[原文](https://arxiv.org/html/2605.19781v1)

**DynMuon**（§2、消融）：以 $U\Sigma^pV^T$ 从正指数平滑转向轻微负指数；分析局部曲率、信号、噪声与训练阶段的权衡。日程实现及精确 SVD 对照已存在，固定负指数和突变日程较差。该 mode-wise 理论使用局部理想化假设，不能写成一般训练阶段的必然规律。[原文](https://arxiv.org/html/2605.17109v3)

**Muon^p**（§2）：有理 $p\in(0,1)$ 的 $U\Sigma^pV^T$ 对应 Schatten-$q$，$q=1+1/p$；用保留原输入的双变量递推实现，不是简单把任意 NS 单变量多项式目标改掉。摘要及方法强调微调收益与适用区间，不保证全部预训练都优于 Muon。[原文](https://arxiv.org/html/2606.13867v1)

**Freon/Kaon**：前者涵盖 Schatten 与 quasi-norm 区域，使用 QDWH 近似；后者随机化奇异值，提醒几何精确性未必是主要性能因子。附录 H.3 已给逐步网格选指数与解析谱斜率选指数，再用精确代理步长。§3.3 对比单 batch 与完整数据方向景观；附录 C 明确报告逐步追踪最优指数因跨 batch 波动大而未获稳定贪心日程。该失败仅限论文尝试与范围，不能泛化为所有自适应指数不可能。[原文](https://arxiv.org/html/2605.11181v1)

### 2.3 矩适配、预条件与几何保持

**AdaMuon v3**（Alg.1）：先对动量逐元素 Sign，再有限 NS；对正交化输出维护完整逐元素二阶矩，缩放后 RMS 匹配。该非均匀缩放不等于数学上保持原半正交结构。不能把“坐标幂后 NS”直接视为空白，也不能把它归为仅增加标量状态。[原文](https://arxiv.org/html/2507.11005v3)

**DeVA v2**（§3 / Alg.2）：分离方差适配与尺度不变项，并桥接向量/谱优化；实现含不同矩统计及矩阵基变换。附录另给行列分解二阶状态版本和 sign 消融。它不同于固定有限 NS 方向的一维步长；主方法完整成本不能仅由 factorized 二阶状态推断。[原文](https://arxiv.org/html/2602.06880v2)

**MALT/MALTER**（§3.5）：左右对角梯度统计预条件，NS 后映回原空间，并用 norm grafting 控制幅度；MALTER 再加预条件空间范数的 Adam 式标量。原文 Alg.2 的 HTML 中 $v/\nu$ 名称和学习率乘法有待对照源码，报告不照抄为可运行公式。它已覆盖“低状态曲率/噪声适配 Muon”的宽泛方向。[原文](https://arxiv.org/html/2608.05088v1)

**DGA v2**（§3–5）：从原始梯度算缩放，宽矩阵行缩放、长矩阵列缩放，分析 NorMuon 精确/近似正交化下适配退化。几何保持指适当一侧向量彼此垂直，不是都保持单位范数；收敛章节明确精确正交化，不能当作任意 NS5 的证明。[原文](https://arxiv.org/html/2610.06578v2)

### 2.4 QSD v2：比旧笔记更直接的方向曲率近邻

QSD 在谱范数球解 PSD 二次模型，K-FAC 定义曲率，FW 会改变奇异幅度和子空间；零初始化的首个原子是 Muon，其一次截断线搜索已是固定 Muon 射线曲率选步。§3.3 / 附录 I 在已接受层方向上用 logits 前向差分近似 $Jd$、softmax Hessian 算 GGN，匹配 K-FAC 方向曲率；不用额外反传，逐层前向校准。124M 每 192 步、350M 每 384 步；小样本、clip / EMA、FP32。理论下降与 $O(1/K)$ 属二次代理和内求解器，非真实 LLM 损失保证。[v2 原文 §3.3 / I / J / K](https://arxiv.org/html/2609.07597v2)

## 3. 从近邻推导的研究限制与未验证空间

以下是本次研究判断，不是任何单篇论文宣称的定理。

**最直接的四组对照**取决于最终主张。若保持现有实际有限 NS 方向而只改半径：AdaGO（廉价累计范数）、Distance-Aware（证书/半径）、ZFO（前向曲率探测）、QSD 的 zero-start 单 FW 步（截断二次射线）都应进入机制审计；MuonMax-Momo 是新增必须核对的模型步长对照。若引入谱幂：SMuon、DynMuon、Muon^p、Freon 更直接。

不能再以以下宽泛说法立项：首次自适应 Muon；首次方向与半径分离；首次谱幂或时间变化谱幂；首次 alignment / curvature 比值；首次低频有限差分校准；首次低状态标量步长。QSD v2 已使“方向曲率仍为空白”尤其不成立。

较窄的可检验问题仍是：**在固定实际 NS5、固定辅助参数配方与固定总计算预算下，用户现有 APS 的同样本两点梯度信息，能否比无额外反传标量法、前向 logits/GGN 法、函数值曲率法更可靠地预测样本外下降，并在低频复用后提供总墙钟收益。** 这不是目前已证明的新颖性，也不保证正结果。若目标是纯一维控制，不必先引入 QSD 的 K-FAC 全矩阵状态或谱方向求解器。

需区分三个曲率对象：两点梯度割线近似真实样本 Hessian 的一段平均射线曲率，可能为负；QSD 的 softmax-GGN 项为 PSD，漏掉模型二阶导项；随机特征模型的 activation-curvature 为局部代理。它们可能给出不同选择，不应以“都是曲率”混成一个方法。

可执行审计建议：用同 checkpoint / 同实际 NS5 方向，比较原 APS、梯度割线、函数值中央差分、logits-GGN 探测、调好固定半径；独立 batch 给评分，记录探测 oracle 调用、恢复精度和总时间。若更贵的梯度反馈无样本外预测优势，或稀疏缓存消灭收益，应停止它作为主创新。

**矩阵 Lp 的位置还必须明确**：逐元素坐标幂依赖坐标基；谱幂是 Schatten，保持输入奇异向量；polar 之后再施加谱幂不产生任何变化（非零极因子奇异值为 1），power 前再精确 polar 则丢弃非零谱幅度；有限 NS 残差可依赖输入谱，但必须作为实际数值映射研究。不要把这种残差误称精确正交后的新谱几何。

## 4. 未闭合来源和访问结果

候选标题 **Can Muon Adapt Its Stepsize Without Knowing the Solution Distance?**，[OpenReview forum](https://openreview.net/forum?id=Ghfc8IRXdy)。本次精确标题/ID 检索未找到可读一手文本；forum、[PDF](https://openreview.net/pdf?id=Ghfc8IRXdy)、[公开 API](https://api2.openreview.net/notes?id=Ghfc8IRXdy) 均未通过当前工具读取。shell curl 还因环境 DNS 限制返回状态 000，此结果仅代表当前访问失败，不证明论文不存在。不能认定它与 Distance-Aware 同一版本，也不能宣称已排除其创新冲突。要正式判定新颖性前需补齐。

本附录未把检索器给出的相关非一手材料当作事实证据，未新训练任何模型，未检查论文所有代码 commit，也未据 arXiv 自述确认所有会议接收状态。
