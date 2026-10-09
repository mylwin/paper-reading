---
title: Lp、APS 与 Muon 结合：文献审计与具体研究方案
date: 2026-10-08
scope: 大模型优化器设计；以现有 NanoGPT 实验为起点
status: 调研与可执行方案，尚未验证新方案的训练收益
---

# Lp、APS 与 Muon 结合：文献审计与具体研究方案

后续阅读安排见[推荐阅读顺序与研究推进](2026-10-08-Lp_APS与Muon结合_阅读路线与研究推进.md)，含指定章节、原文链接及2026-10-08补充核验。

**建议的主线：先固定 Muon 的实际有限 Newton–Schulz 更新方向，研究低频同批次梯度反馈能否校准它的更新半径。先解决步长与方向尺度失配，再决定是否引入坐标幂或谱幂。**

这条路线直接利用已有 Lp/APS 代码和实验，问题比“把三个优化器与 Muon 拼起来”更清楚。但它还不是已确认的新算法贡献：方向曲率选步、标量自适应 Muon、自动选择谱幂均有先行工作。现阶段适合确定一个可被实验否定的课题，不能承诺论文新颖性或一定超过 Muon。

## 1. 调研范围与证据边界

截至2026年10月8日，本次覆盖：

- 三份研究周报：[第1周](2026-09-21-第1周周报.md)、[第2周](2026-09-28-第2周周报.md)、[第3周](2026-10-05-第3周周报.md)，以及 [lp范数三篇联读](2026-09-23-lp范数三篇联读_从算子界到复杂度插值.md)。第3周文件包含10月7日追加内容，不能只按文件名判断版本。
- 18份日报：9月17、18、19、20、21、22、23、24、25、28、29、30日；10月2、3、4、6、7、8日。扫描主题与全量候选，重点阅读 Muon、幂变换、Polyak、噪声、曲率及相关失败结果；不是把所有推荐论文都重新精读。
- 核心笔记及公式材料：PowerStep、SignSGD、Parameter-free Polyak、MARS、MEKA。对比笔记、解析文本、原始PDF及新版原文，避免沿用过时定理。
- 原文方法/边界定点阅读：DGA-Muon、ORCA、SAMuon、QSD、LionMuon、MeqMuon、MuonIO、Muon参数区域理论、SoftServe，以及新版PowerStep、SMuon、Distance-Aware Muon、ZFO。其余相关工作按下表标明为正文核验、历史核验或摘要级。
- 现有实现：`power_aps_optimizers.py`、`internal_optimizers.py`、优化器工厂、NanoGPT训练入口、预设与原始完整运行日志。未修改优化器，未启动训练。
- 本次仅新增可复现的标量代数校验：9组二次函数尺度不变性检查，及APS在小梯度下的数值示例。它们不提供LLM性能证据。

日报预筛分、AI精读稿、资产归档和周报结论分别属于检索、二次分析、存在性与研究判断证据，不能互相替代。论文入库月份也不等于发表日期，例如MEKA原文是2020年工作坊论文。

## 2. 你的现有研究已走到哪里

| 材料 | 已有判断 | 本次如何接着推进 |
|---|---|---|
| 第1周周报 | 从零阶/子空间/信赖域出发，重视前向调用与墙钟预算 | 保留预算意识；当前主线遵循10月7日收窄要求，聚焦LLM优化器 |
| 第2周周报 | 单状态更新；噪声结构是否决定每块应保留多少幅值信息 | 不能再次把“噪声决定幂指数”包装成新想法 |
| 第3周§五 | 已有响应校准的坐标幂、独立batch预测、低频迟滞控制，并完成Gaussian标量校验 | 本次新增现有APS实测饱和证据及矩阵方向的尺度失配分析，先检验步长反馈 |
| 第3周附录D | 方向与步长必须分离；额外反传须有预算收益；机械换分母不能继承理论 | 用固定NS5方向控制归因；不同时增加SAM、谱幂、动量修正和预条件 |
| MARS与MEKA材料 | 同样本两点梯度包含参数变化信息；更准的局部估计未必改善最终训练 | 利用现有第二梯度测方向曲率，同时保留样本外与多步验证 |

最关键的缺口并不是再找一种更新公式，而是建立这条证据链：**可测反馈 → 对未来实际下降的预测 → 固定预算验证损失 → 总训练时间收益。**

## 3. 现有实验真正说明了什么

以下直接读取 `runs/fig2/nano/*/run_summary.json`，并重算两个Lp调优运行的 `train_metrics.jsonl`。均完成5100步、334,233,600训练tokens；有效batch64、序列长度1024、验证tokens10,485,760。它们不是配对多种子试验，配方与参数路由也不完全相同，不能据微小差值宣称显著胜出。

| 当前本地运行 | 最终验证交叉熵 ↓ | 总墙钟/小时 ↓ | 日志tokens/s ↑ |
|---|---:|---:|---:|
| Adam | 4.060439 | 4.116 | 28,675 |
| Muon | 3.626272 | 4.147 | 28,473 |
| SOAP | 3.643856 | 4.801 | 23,868 |
| NSISA | 3.667547 | 5.972 | 18,247 |
| pbSGDM，调优版 | 4.458026 | 3.990 | 29,877 |
| Lp-SGDA，调优版 | 4.779711 | 7.140 | 14,807 |
| Lp-SGDMA，调优版 | 4.030326 | 6.543 | 16,424 |

墙钟来自整个运行摘要；日志tokens/s有自己的计时区间，不能用它代替 `总tokens/完整墙钟`。峰值显存和吞吐也不能仅由优化器持久状态量推断。

**Lp-SGDMA已明显修复原始配方的严重不稳定，最终接近本地Adam，但仍落后当前Muon且耗时更多。** 这支持继续研究，尚不支持直接把Lp方向加到Muon便会改善。

两个调优Lp运行都使用 $c=300$：

| 指标 | Lp-SGDA | Lp-SGDMA |
|---|---:|---:|
| $\eta_t>0.99/c$ 的train_step占比 | 99.882% | 99.765% |
| 实际步长中位数 | 0.0033331483 | 0.0033332867 |
| $1/c$ | 0.0033333333 | 0.0033333333 |

因此现在首先要问：**第二次梯度是否给了有用的自适应信息，还是调好的上限几乎决定了全部步长？** 这需要固定 $1/c$ 消融，不能仅凭命中率就断言第二梯度完全无用；少量早期非饱和步也可能改变轨迹。

同时，现有Lp参数更新读取 `step_size`，普通LambdaLR改写的是 `lr`。所以不能把图上的名义学习率衰减当成实际APS步长衰减。修正此接口属于工程对照，不能单独算算法创新，也不能回填覆盖历史结果。

原始证据与代码：

- [Muon运行摘要](../../graduate-thesis-code/reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/runs/fig2/nano/muon/full-v2/run_summary.json)
- [Lp-SGDMA调优运行摘要](../../graduate-thesis-code/reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/runs/fig2/nano/lp-sgdma%28优化参数%29/full/run_summary.json)
- [Lp-SGDA调优逐步日志](../../graduate-thesis-code/reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/runs/fig2/nano/lp-sgda%28优化参数%29/full/train_metrics.jsonl)
- [APS实现](../../graduate-thesis-code/reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/code/power_aps_optimizers.py)，125–189行；[Muon实现](../../graduate-thesis-code/reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/code/internal_optimizers.py)，135–201行。
- [Nano训练入口](../../graduate-thesis-code/reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/code/NanoGPT/train_gpt_sisa_lower_no_2ndgradient.py)，1460–1503行：缓存同batch、随机数重放和第二次反传。
- [本次重算结果](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案/run_evidence.json)。

## 4. 文献查重：哪些组合已经有人做过

| 工作与核验范围 | 已覆盖的内容 | 对本方案的约束 |
|---|---|---|
| [PowerStep v2，9/29，§2、§3、§5.6](https://arxiv.org/html/2605.10335v2)；本地PDF为v1 | 动量后坐标保号幂；已有Muon隐藏矩阵+PowerStep辅助参数组合。v2定理含噪声残差；组合实验有验证协议差异 | 不能主张首次“Lp+Muon”或凭旧版主张最优随机收敛 |
| [SMuon，§2–4、附录A.2.5–A.2.6及B](https://arxiv.org/html/2605.19781v1) | Schatten族连接SGD与Muon；按alignment²/曲率代理自动选几何，低频刷新及EMA统计，另有二阶矩版本。续查附录A.2.5–A.2.6已推导代理最优步长；附录B报告直接使用该步长可能造成loss尖峰 | 谱幂插值、自动选p、低频刷新均不是空白 |
| [DynMuon](https://arxiv.org/html/2605.17109v3)、[Muon$^p$](https://arxiv.org/html/2606.13867v1)、[Freon/Kaon](https://arxiv.org/html/2605.11181v1)；第3周已有正文核验 | 谱幂日程、后期切换、自适应指数尝试；Freon报告逐batch贪心选幂失败 | 不能只靠“随训练变化的指数”；须击败简单时间策略并排除选参噪声 |
| [AdaMuon v3](https://arxiv.org/abs/2507.11005)，本次摘要核验 | 正交化方向上的二阶矩适配；新版包含正交化前sign处理与RMS缩放 | “先坐标非线性再NS”也需查重，且它新增完整二阶矩状态 |
| [AdaGO，OPT2025，Alg.1](https://www.opt-ml.org/papers/2025/paper99.pdf) | 保持正交方向，以梯度范数累计量调整标量步长；无需额外梯度 | 是低状态标量步长强对照；文中实验主要为小模型，不能称已有LLM大规模验证 |
| [Distance-Aware Muon，§3–5](https://arxiv.org/html/2605.18999v1) | 轨迹半径、动量/梯度下降证书、标量majorization搜索 | 方向—半径分离及对齐保护不是新贡献；理论条件与实际随机训练分开 |
| [DGA-Muon v2，§3–4](https://arxiv.org/abs/2610.06578v2)，本地原文 | 原始梯度提供缩放统计；宽矩阵按行、高矩阵按列，兼容正交结构 | 不能把几何对齐缩放当新方法；本次割线反馈与它不同，但需公平比较 |
| [MALT/MALTER](https://arxiv.org/abs/2608.05088)、[DeVA](https://arxiv.org/abs/2602.06880)，方法/动机段核验、实验摘要级 | 前者两侧对角预条件和噪声自适应标量；后者分离方差适配与尺度不变更新 | 泛称“曲率/噪声感知Muon”不够；后续立项需读实现与完整消融 |
| [ZFO，§2–4](https://arxiv.org/html/2610.02190v1)，10/1 | FO方向+两次额外前向，拟合一维模型选步；包括Muon接口 | 本方案要与前向探测成本比较，不能主张首次混合一阶/零阶选步；其微调成本不能直接当预训练开销 |
| [ExpertMuon-Compass](https://arxiv.org/abs/2610.04140v1)，10/2，本次摘要级 | MoE专家的对齐因子和标量半径，保留Muon方向与动量 | 只加gradient/update cosine门控的创新空间很小；本方案测曲率，且先研究稠密模型 |
| [SAMuon](https://arxiv.org/abs/2608.25990)、[QSD](https://arxiv.org/abs/2609.07597)，本地原文 | 前者用静态谱先验增强bulk；后者在同一谱范数球内解二次局部模型，并可改变奇异方向 | 各谱方向不同幅度、局部曲率、较少频率刷新都已有先例 |
| [LionMuon](../01-raw/2026-09/LionMuon_Alternating_Spectral_and_Sign_Descent_for_Efficient_Training.pdf)，本地方法段 | 交替谱与坐标sign更新 | 几何切换本身不足以主张新颖性；切换须有额外可验证依据 |
| [ORCA](https://arxiv.org/abs/2610.06116v1)，本地方法段 | 早期施加参数W的软正交正则，之后撤除 | 它控制权重谱，不是简单对更新奇异值做幂日程；本课题首版不加该正则 |
| [MeqMuon](../01-raw/2026-10/MeqMuon_Matrix_Equilibrating_Muon_for_LLM_Pretraining.pdf)、[MuonIO](../01-raw/2026-10/MuonIO_Principled_Norm_Aware_Descent_for_Embedding_Tables_and_Language_Model_Heads.pdf)，本地方法段 | 单状态矩阵均衡、embedding/head专用几何 | 隐藏矩阵与I/O贡献必须拆开，不能靠同时换辅助优化器制造收益 |
| [SoftServe](../01-raw/2026-10/SoftServe_A_Scalable_Quasi_Newton_Method_for_Deep_Learning.pdf)，§2–3 | 同样本割线、软约束保持正定、对角/Kronecker曲率 | 割线与负曲率处理早已是成熟路线；首版只保留一维信息，不开发完整QN状态 |

**未闭合的查重项：** 日报列出 [Can Muon Adapt Its Stepsize Without Knowing the Solution Distance?](https://openreview.net/forum?id=Ghfc8IRXdy)。本次forum和PDF读取失败，精确标题检索未取得一手正文；续查公开API返回403并要求challenge verification，仍未取得题录或正文。不能把它视作已排除，也不能确认它是否为Distance-Aware工作的另一版本。此项在论文新颖性判断前必须补齐。

## 5. 一个能解释“不能直接套APS”的尺度问题

下文APS专指仓库/用户截图的规则，不把所有Polyak家族都混为同一个方法。设 $D_t$ 为当步更新方向，$G_t^+$ 为沿该方向更新后、在同batch上计算的梯度，则现有代码使用

$$
A_t=|\langle G_t^+,D_t\rangle|,\qquad
\eta_{t+1}=\frac{A_t}{\|G_t^+\|_F^2+cA_t+\epsilon}.
$$

内积按所有参与参数求和；现有实现并不是逐矩阵分别取绝对值再求和。忽略很小的数值项时，可写为

$$
\eta_{t+1}=\frac{1}{c+\|G_t^+\|_F^2/A_t}.
$$

这解释了为什么当 $cA_t$ 主导时步长接近 $1/c$。但步长接近常数，不等于参数位移一定接近常数，关键还在 $D_t$。

### 5.1 幂方向与Muon方向的齐次性不同

用固定局部梯度对的缩放模型说明问题：令 $G^+=a\bar G$，并让方向构造的输入也按 $a>0$ 缩放。如果 $D=a^h\bar D$，则

$$
\eta_{\rm APS}(a)=\frac{1}{c+a^{1-h}K},\qquad
K=\frac{\|\bar G\|_F^2}{|\langle\bar G,\bar D\rangle|},
$$

前提是内积非零，且暂时忽略 $\epsilon$。这是固定局部量的尺度分析，不是对整个随机训练轨迹的收敛证明。

- Lp幂方向 $\operatorname{sign}(M)|M|^\gamma$ 的次数为 $h=\gamma>0$。即使 $\eta\approx1/c$，位移仍随 $a^\gamma$ 缩小。
- 理想Muon极因子 $\operatorname{polar}(M)$ 的次数为 $h=0$。此时 $\eta\to1/c$，方向幅度又不随 $a$ 缩小，因而存在近零信号时仍维持更新半径的机制风险。
- 代码NS5先做Frobenius归一化，在输入范数明显大于归一化epsilon时也近似0次齐次；极小输入及有限精度下不完全成立。

**所以把APS原式接Muon，可能把Lp中的“步长饱和但方向渐小”变成“步长与方向幅度都不渐小”。** 不能据此断言必然发散：warmdown、epsilon、噪声和实际参数轨迹都会影响结果。这是需要验证的机制假设。

固定 $\epsilon>0$ 时，极限 $a\to0$ 最终由epsilon主导，APS步长会变小；不能在保留epsilon时仍声称严格极限恒等于 $1/c$。关键是训练中实际处于哪个尺度区间。

### 5.2 现有APS也不具备方向重标度不变性

同一条更新射线，把方向改为 $D'=kD$ 时，希望步长相应除以 $k$，使实际位移不变。现有APS一般不满足这一点。因此Muon的shape scaling、RMS matching与原APS的 $c$ 会耦合，不能把Lp调出的 $c=300$ 直接复制过去。

另一方面，$\langle G^+,D\rangle$ 取绝对值仅保证标量非负，不能证明当前 $-D$ 是下降方向。需要检查的是更新前 $\langle G,D\rangle>0$，两者也不能混同。

[本次代数校验](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案/mechanism_checks.json)展示了上述尺度区间，并验证下节二次模型公式的重标度性质；没有模拟或复现LLM训练。

## 6. 具体方案：固定实际方向，低频校准半径

工作名暂定“有限NS方向的稀疏曲率校准”，仅作实验标签，不是论文命名或新颖性声明。

**研究问题：** 在一个完整动量状态、少量标量和预先限定的探测预算下，同batch两点反馈是否比原APS饱和上限、已调Muon日程及既有标量适配，更可靠地控制有限NS方向的实际位移？

### 6.1 方向保持现有Muon实现

对隐藏矩阵块 $b$，先按现有EMA与Nesterov形式构造

$$
M_{t,b}=\beta M_{t-1,b}+(1-\beta)G_{t,b},\qquad
N_{t,b}=(1-\beta)G_{t,b}+\beta M_{t,b},
$$

其中 $G_{t,b}$ 为当前batch梯度。再令

$$
D_{t,b}=s_b\,\operatorname{NS}_5(N_{t,b}),
$$

$s_b=\sqrt{\max(1,\mathrm{rows}_b/\mathrm{cols}_b)}$ 与当前代码一致。首版不改变动量、不改变NS系数、不增加Powerball，不把NS5当精确polar。

把所有Muon隐藏矩阵方向作为一个联合向量，记其Frobenius总范数为 $\|D_t\|_F$。为了让探测距离有明确单位，定义

$$
Q_t=\frac{D_t}{\|D_t\|_F}.
$$

$Q_t$只是重新表示同一射线；正常Muon的基准位移半径为
$r_t^{\rm base}=\eta_t^{\rm Muon}\|D_t\|_F$。
**若始终取 $r_t=r_t^{\rm base}$，更新 $-r_tQ_t$ 精确恢复原Muon，不能把它误称为新的归一化算法。** 零范数时跳过矩阵更新。

### 6.2 同样本探测实际方向曲率

只在探测步临时移动隐藏矩阵：

$$
W^{\rm probe}=W_t-\delta_tQ_t.
$$

$\delta_t$是物理位移距离，可先取当步 $r_t^{\rm base}$ 的一半；该系数是待验证实验设置。embedding、共享head及其余辅助参数在探测过程中冻结。对相同训练batch、相同随机数，得到 $G_t^{\rm probe}$，并计算

$$
a_t=\langle G_t,Q_t\rangle,\qquad
\widehat L_t=\frac{\langle G_t-G_t^{\rm probe},Q_t\rangle}{\delta_t}.
$$

$a_t$表示当前方向的一阶下降增益；$\widehat L_t$是沿联合矩阵方向的平均曲率。若目标二次可微，且探测时辅助参数固定，则

$$
\widehat L_t=\int_0^1
\langle Q_t,H(W_t-u\delta_tQ_t)[Q_t]\rangle\,du.
$$

这是含跨隐藏层耦合的联合方向曲率。不能把联合移动产生的某层梯度差称作该层独立Hessian曲率。使用同batch/同RNG可去掉“更换样本”产生的差分噪声，但无法去掉采样目标本身的随机性。

先记录 $a_t$ 与 $\widehat L_t$ 的原始带符号值；不把负曲率取绝对值后称为可信正曲率。诊断阶段比较 $\delta_t$ 与 $\delta_t/2$，检查有限差分偏差与精度；不要先假定探测越小越好。

### 6.3 半径规则和回退必须具体

局部二次模型为

$$
F(W_t-rQ_t)\approx F(W_t)-ra_t+\frac{r^2}{2}L_t.
$$

若 $a_t>0$ 且曲率可信为正，模型建议 $r_t=a_t/L_t$。初始实验采用保守截断

$$
r_t=\min\left\{r_t^{\rm base},\;\tau\frac{[a_t]_+}{\bar L_t}\right\},
\qquad W_{t+1}=W_t-r_tQ_t,
$$

其中 $[a]_+=\max(a,0)$，$\bar L_t$为最近有效探测的正曲率标量，$\tau$为安全系数。**首版固定 $\tau=1/2$、探测间隔32步，不同时搜索所有控制变量。** 非探测步使用当前 $a_t$ 和缓存 $\bar L_t$；第一次有效探测前恢复原Muon日程。EMA平滑只作后续消融，避免预先加入新参数。

- 曲率非正、非有限或尺度不可靠：标为无效，不覆盖上一有效值；首版当步回到原Muon半径。若 $a_t\le0$，改用当前梯度的NS5方向重新检查，仍不对齐则跳过矩阵更新。不要以abs掩盖反向方向。
- 缓存曲率很旧：超过两次预定探测机会未得到有效值就失效，回到原日程；记录fallback比例。32和两次机会均为起始工程设定，不是理论最优值。
- warmup/warmdown已经在 $r_t^{\rm base}$ 中，不能再额外乘一次相同scheduler因子。
- 首版只能减小原Muon半径，目的先是校准和降低配方敏感性。若希望允许增步，在机制成立后再把上限改为 $2r_t^{\rm base}$，单独记为不同实验。
- 如果绝大多数步仍取基准上限，方法基本退化成已调Muon；必须报告，不能仅展示validation略有变化。

这种全矩阵组标量缩放不会改变各块NS方向的相对奇异值形状。它保持的是**已有方向的几何结构**，并不使有限NS输出突然成为精确正交矩阵。

### 6.4 为什么这个公式比机械套APS更合适

在精确正曲率二次模型、未截断时，不归一化也可写步长

$$
\alpha_t^*=\frac{\langle G_t,D_t\rangle}
{\langle D_t,H[D_t]\rangle}.
$$

方向重标度 $D_t\mapsto kD_t$ 时，分子乘 $k$、分母乘 $k^2$，故 $\alpha_t^*D_t$ 不变；目标整体乘正数时，分子和分母同乘该数，位移也不变。探测距离必须按物理位移设置，数值阈值也须采用相对尺度，才能在实现中保留这些性质。截断版还要求物理半径上限保持相同；若任意放大D却保持名义lr不变，基准半径本身已经改变，不能再宣称整个截断算法不变。

这正是经典方向二次模型/线搜索思想，不是新发明。潜在研究增量在于：对**实际有限NS方向**，稀疏、带预算的随机反馈是否可靠，何时失效，以及是否获得等墙钟收益。

### 6.5 工程实现与成本

方向准备、动量更新和辅助Adam状态更新各只执行一次。探测仅临时改权重，不能调用完整 `optimizer.step()` 两次。探测后恢复权重，保留已经准备好的原始方向再提交正式更新；辅助组仍采用同一固定日程。

原Nano closure只在更新后重放梯度，不能原样当作带“探测—恢复—正式更新”的事务接口。新实验需独立实现并记录：exact restore、随机数恢复、累积梯度归一化、DDP同步、探测不消耗新训练数据、辅助梯度不被第二次backward误用。

额外持久状态可仅为有效曲率、探测时点和少量诊断标量；但临时方向、权重快照、batch缓存以及梯度仍占显存。**不能称零额外内存，也不能把整个混合优化器称为只有一份状态：辅助Adam仍有一阶/二阶矩。**

若一次探测增加一遍完整前后向，间隔32步时，理想算术摊销约为普通训练步的 $1/32\approx3.1\%$；这不含复制、重放、通信、额外验证和编译成本，实际只能测量。先对比间隔1（上界诊断）与32（预算版），再决定64是否值得做。

ZFO式两次前向也必须纳入成本对照。稠密预训练没有其某些微调实验的rollout/reward耗时来稀释探测成本；不能直接引用其“开销小”作为本方案预算依据。

## 7. Lp具体放在哪里：明确三条路线的优先级

| 路线 | 技术做法 | 本次判断 |
|---|---|---|
| A：方向不变，校准半径 | Muon NS5 + 上述方向曲率反馈 | **首选可检验课题**，与现有APS成本问题最直接；先验证，不预宣称创新 |
| B：Muon隐藏矩阵 + Lp/PowerStep辅助参数 | embedding/head及一维参数改用单动量幂 | 可作工程组合与内存对照；PowerStep已报告该组合，不能作为主创新 |
| C：改变隐藏矩阵谱几何 | $M=U\Sigma V^\top$，使用 $U\Sigma^\gamma V^\top$；再校准尺度 | 幂与谱的自然连接，但SMuon/DynMuon/Muon$^p$/Freon已覆盖甚多，A有证据后才展开 |

$\ell_p$坐标幂与Schatten谱幂不可混用：前者逐元素作用，依赖所选坐标基；后者作用于奇异值，保持该矩阵的奇异向量。理想Schatten-$p$的幂指数为 $\gamma=1/(p-1)$，而SMuon论文用Schatten-$(p+1)$记号，指数为 $1/p$；比较公式时要对齐记号。

三个常见组合陷阱：

1. 对精确谱幂结果再做polar，$\operatorname{polar}(U\Sigma^\gamma V^\top)=UV^\top$，非零奇异子空间上的幂信息被消掉。有限NS有残余依赖，但不能把数值近似误差当清楚的设计原理。
2. 正交化之后逐元素power通常破坏原半正交结构；正交化之前逐元素power一般改变奇异向量。两者都不等同于谱幂。
3. 相同 $\gamma$、学习率、$c$ 不能跨pbSGDM、Lp-SGDMA和Muon直接复制：变换与动量的顺序、heavy-ball/EMA比例、方向幅度及辅助参数路由都不同。

建议第一阶段把三个现有优化器作为**诊断与消融对象**，而不是强行让三个都生成一个混合方法。

## 8. 最小实验：按信息增益逐级花预算

### E0：先确认现有APS到底有没有用

相同初始化、数据序列、batch、辅助参数处理和实际调度，比较：

- 原调优Lp-SGDMA/SGDA；
- 首步保持0.0006，之后固定 $1/c$，其余方向完全相同；
- 固定 $1/c$ + 明确作用在实际更新上的warmdown；
- 原APS + 同样实际warmdown。

固定步长省去第二次反传；分别报告等tokens与等总时间。这些实验回答“动态反馈”和“调度修正”各贡献多少。不要只拿同时改了两项的新版本和旧日志比较。

若固定上限基本重现原APS，当前最合理结论是：此配置下复杂步长的额外成本尚未获得收益，而不是认定所有APS都无效。

### E1：冻结checkpoint，先验证方向曲率是否能预测

从同一Muon训练轨迹选早、中、末三个checkpoint。每点取8份独立训练batch，前4份作选择，后4份作评分；参数、动量和优化器状态均从相同快照恢复。方向由完整训练batch构造，避免半batch改变噪声规模后冒充实际方向。

在选择数据上比较原APS标量、$a_t$、割线曲率与建议半径；在独立评分batch上，用相同更新方向测真实loss变化。独立评分batch不参与选步，也不是正式验证集。先检验 $\delta$ 与 $\delta/2$ 一致性，再以半径 $\{0.25,0.5,1\}r^{\rm base}$ 的真实下降作为参考。

重要诊断：

- APS接近上限时，方向曲率是否仍有可重复变化？
- 同batch预测能否外推到独立batch和16步分支，而非仅拟合一次loss？
- NS3/NS5与小矩阵精确SVD诊断是否给出不同反馈？不在整个124M模型上每步做SVD。
- 是否只是梯度/更新尺度差异？做方向乘常数、loss乘常数的位移不变性检查。
- 当前 $a_t$ 在缓存曲率下是否仍有预测力？观察不同间隔的失配，不凭“曲率缓慢变化”直接假设可稀疏化。

三点×8batch只是初筛，不提供稳定置信区间。若排序对batch分组极敏感，先增加独立batch复核；仍不可识别就停止闭环控制。

### E2：小模型闭环，只保留能解释归因的组

第一轮最多保留：已调Muon、Muon+原APS、Muon+每步割线、Muon+32步割线、与复杂方案等预算调过的固定/时间半径、ZFO式前向探测。首轮单种子短跑只排除错误/发散；保留方案完成至少3个配对种子、同5100步预算。

正式比较至少包含一个直接近邻标量方法（AdaGO或Distance-Aware的可实现配方）；DGA-Muon作为当代强基线。若进一步声称自动选几何，再加入SMuon，而不是只对比原始Muon。为控制总预算，可按任务主张删减候选，不可删掉最接近的强对照。

按本地Muon完整运行约4.15小时估算，保留3种方法、每种3种子就约需37小时串行A100时间，另加探测、验证及调参；这是粗略规划值，编译与短跑成本不线性。先完成无需新训练的日志检查和冻结探测，再确定正式运行数，不能一次展开全部候选。

初始只用现有batch64。若通过，再用batch32或128之一做迁移，各方法获得相同LR/上限搜索次数；不能认为某个通用缩放法已保证公平。

日志至少包括实际位移、名义lr、$a_t$、原始曲率、探测距离、上限命中率、fallback比例、额外forward/backward次数、总tokens、总墙钟、峰值显存与验证交叉熵。所有探测和失败调参均记预算。

### 预先约定成功与停止条件

建议沿用第3周的决策尺度：固定tokens验证损失改善至少0.01 nats/token，检查配对种子差值与不确定性；同质量总时间改善至少5%，或在质量不劣时显著降低跨batch调参成本。它们是预设决策标准，不是预计效果。

下列情况应停止将A当主线：

- 反馈仅预测选择batch，无法预测独立batch或短分支；
- 校准后不胜过等预算固定/时间半径；
- 大多数步仍被上限决定，曲率统计没有独立解释力；
- 只有每步二次反传有效，稀疏化失效且等墙钟无收益；
- 收益消失于同辅助优化器/同实际warmdown/配对种子对照；
- 完整近邻核验发现与已有算法实质相同。

不应仅为保住假设继续叠加行列二阶矩、SAM或多动量。

## 9. 理论能做到哪里，不能承诺什么

最小可证命题是确定性下降，而不是完整LLM收敛率：若当前单位方向 $Q$ 的射线上有真实曲率上界 $L_Q$，且 $a=\langle\nabla F,Q\rangle>0$，则对 $0\le r\le a/L_Q$，

$$
F(W-rQ)\le F(W)-ra+\frac{L_Qr^2}{2}\le F(W)-\frac{ra}{2}.
$$

单次割线是射线一段上的平均曲率，**不是该步全区间的上界**；缓存曲率更不是下一方向的上界。所以该下降命题不能自动套给提出的随机稀疏算法。后续理论必须明确方向误差、采样误差、曲率漂移、无效探测回退及额外oracle调用。

相关理论使用条件也需分开：PowerStep v2为带噪声残差的有限时域保证；Parameter-free Polyak核心理论为凸确定性设定；Muon参数区域论文的SoftMuon结论含PL及动量限制；Practical Muon的有限NS/Nesterov保证使用与总步数耦合的日程。都不是现有固定 $\beta=0.95$、NS5、warmdown组合的现成证明。

可参考的本地理论资产：[Polyak全局推理](../04-equation_problem/Parameter_Free_Clipped_Gradient_Descent_Meets_Polyak/全局推理.md)、[MARS公式伴读](../04-equation_problem/Understanding_MARS_When_Scaling_Momentum_Correction_Provably_Helps/公式伴读.md)、[MEKA精读](../03-notes/2026-07/Self_Tuning_Stochastic_Optimization_with_Curvature_Aware_Gradient_Filtering/精读.md)。先独立推出上面的单步式，再讨论随机项；不需要先承诺最优非凸速率。

## 10. 当前具体决策

**优先验证A：实际有限NS方向的低频曲率校准。** 它把当前APS几乎贴上限、两次反传昂贵的观察，转成一个直接可检验的问题；比立刻开发新的谱幂混合优化器更容易判断对错。

**保留第3周响应校准坐标幂为第二路线。** 若E0表明APS贡献很小，而方向/噪声更能解释差异，就回到该路线，并用SMuon、Freon、Muon噪声论文完善近邻对照。当前不需要同时优化两个控制器。

可以写入开题/组会的表述：

> 研究有限Newton–Schulz矩阵优化器中方向尺度与自适应半径的耦合。以已有Lp-APS实现的步长饱和现象为起点，检验同样本方向曲率反馈的样本外预测能力，并评估其低频使用在固定状态、训练时间及调参预算下的收益与失效边界。

不能写成“首次提出Lp与Muon融合”“首次自适应谱幂”“无需调参”“零额外成本”，也尚不能宣称新方案优于Muon。

下一项最值得执行的工作是E0与E1，不是7B训练。若它们给出反证，及时保留负面机制结论并转向第二路线，本次调研仍然减少了后续试错成本。


## 附录A：本地原文的精确核验入口

以下页码为本地PDF的物理页，从1开始。用于定位，不表示独立重审全部附录证明。

| 本地原文 | 版本与定位 | 需要核对的关键点 |
|---|---|---|
| [DGA-Muon](../01-raw/2026-10/DGA_Muon_Decoupled_Geometry_Aligned_Adaptive_Scaling_for_Muon.pdf) | v2；p.7 Alg.1，p.8–9，p.14–15，p.21 | 行列缩放保持互相垂直而非单位范数；理论精确polar条件；实验超参同时变化 |
| [ORCA](../01-raw/2026-10/ORCA_The_Annealed_Spectral_Conditioning_Optimizer_for_Faster_Better_LLM_Training.pdf) | v1；p.5 Eq.3.1–3.4，p.6 | 权重软正交惩罚及早期撤除；不是更新谱幂 |
| [SAMuon](../01-raw/2026-09/Spectral_Allocation_Why_Muon_Outperforms_Adam_and_How_to_Improve_Muon.pdf) | v1；p.5–8，p.10–12 | held-out谱探针、静态bulk增强、单种子边界与真实开销 |
| [QSD](../01-raw/2026-09/Beyond_the_Matrix_Sign_Quadratic_Spectral_Descent.pdf) | 本地v1；p.4–6，p.10–12；线上已有v2 | 同一二次surrogate内的比较不是真实训练保证；内循环率不等于外循环率 |
| [LionMuon](../01-raw/2026-09/LionMuon_Alternating_Spectral_and_Sign_Descent_for_Efficient_Training.pdf) | v1；p.4 Alg.1，p.8–11 | 一个共享buffer、周期交替；分布式达标时间受互联条件影响 |
| [MeqMuon](../01-raw/2026-10/MeqMuon_Matrix_Equilibrating_Muon_for_LLM_Pretraining.pdf) | v1；p.4 Table1，p.5–6 | 实际NS后行列不均衡仍可明显存在；按CV选择归一化 |
| [MuonIO](../01-raw/2026-10/MuonIO_Principled_Norm_Aware_Descent_for_Embedding_Tables_and_Language_Model_Heads.pdf) | v1；p.6 Alg.1，p.7，p.9 | I/O专用范数；不能按张量二维性无差别路由 |
| [Muon参数区域理论](../01-raw/2026-10/Convergence_guarantees_for_Muon_New_parameter_regimes_and_generalizations.pdf) | v1；p.3–6，p.8–9 | canonical/ideal/soft三个对象；PL、动量限制与实际配方差距 |

## 附录B：证据复算与复核状态

- [复算脚本](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案/recompute_evidence.py)：仅标准Python库，读取已有运行日志，不训练、不修改源码。传入 `--code-root` 可指定代码仓库；`--output-dir` 写入新结果，已存在同名结果时拒绝覆盖。
- [运行结果](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案/run_evidence.json)、[输入日志SHA256](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案/run_input_manifest.json)、[标量机制校验](2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案/mechanism_checks.json)。
- 7份运行摘要、2份逐步日志已重新读取；9组确定性二次尺度校验通过。现有训练日志不是本次新跑的结果。
- 本报告31个本地相对链接已独立校验，0断链；13组独立展示公式的分隔符成对。全库官方索引checker因可用Python环境缺PyYAML未完成，不能宣称已验证全库索引。
- 新算法没有实现或训练；没有证明它优于直接近邻；OpenReview候选及部分摘要级近邻仍需补齐。报告明确保留这些不确定性。
