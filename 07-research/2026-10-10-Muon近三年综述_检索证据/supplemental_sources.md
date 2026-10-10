# 补充来源、检索路线与访问边界

截止：2026-10-10，Asia/Shanghai。配套主报告：`../2026-10-10-Muon优化器近三年研究综述与APS_Lp可行性探索.md`。本文件记录主任务补读的来源层级，便于区分新核验、旧知识库审计和未闭合候选；不宣称穷尽搜索或完整证明复核。

## 1. 检索路线

1. 从本地Muon、APS、PowerStep、Schatten与谱幂命中恢复实际定义和已有研究路线。
2. 起源链：作者博客、官方源码、Modular Norm、Old Optimizer New Norm；规模化链：Moonshot与Essential AI。
3. 理论链：精确polar、谱几何、有限Taylor-NS、实际有限NS/Nesterov；检查驻点度量、光滑性及计算oracle。
4. 组合查重：标量范数、Momo、距离适配、谱幂、自适应指数、两点前向、QSD新版方向GGN。
5. 负面链：公平Shampoo调参、局部贪心谱幂、噪声残差、外部任务排名、训练终点。
6. 当日日报补充：SignMuon、谱平坦候选；失败的OpenReview条目保留，不以未读代替排除。

公开搜索用于发现，结论引用arXiv版本正文、作者博客、PMLR或官方代码。Chrome CDP未连接，但公开静态检索可用；未登录网站或修改外部应用。没有把搜索引擎自动生成日期当作首次发表日期。

## 2. 主任务补读来源

| 工作 | 原始来源 | 本次读取与版本边界 |
|---|---|---|
| MuonMax–Momo / An Exploration of Non-Euclidean Gradient Descent | [arXiv v1](https://arxiv.org/html/2510.09827v1)、[版本页](https://arxiv.org/abs/2510.09827)、[ICML2026正式页](https://proceedings.mlr.press/v306/crawshaw26a.html) | 首发2025-10-10；§3—5与附录B、表4定点。截断线性模型、模型式动量、损失下界和块对偶范数尺度。附录D表4最佳调参124M/FineWeb1B/3 seeds（mean±std）中MuonAdam–Momo 3.5546±.0004、MuonAdam 3.5592±.0014、MuonMax–Momo 3.5779±.0007。鲁棒性与最佳点分开报告，未重新训练。 |
| Convergence of Practical Muon with Finite NS and Nesterov Momentum | [v1正文](https://arxiv.org/html/2609.39595v1)、[题录](https://arxiv.org/abs/2609.39595v1) | 首发2026-09-30；§3/4.3定点。分析联合层级有限映射及单系数Nesterov；Frobenius平均梯度率与总时域耦合日程，系数映射对齐/能量条件。epsilon、形状增益、精确算术与噪声假设需保留。未全证明复核。 |
| Muon Under Gradient Noise and the Limits of Orthogonalization Near Optima | [v1正文](https://arxiv.org/html/2609.32861v1)、[题录](https://arxiv.org/abs/2609.32861v1) | 首发2026-09-26；§7—8及结论。Gaussian局部响应匹配、非线性残差、冻结Transformer梯度；不提供完整训练切换实验。 |
| PowerStep | [v2正文](https://arxiv.org/html/2605.10335v2)、[版本页](https://arxiv.org/abs/2605.10335) | 首发2026-05-11，v2 2026-09-29；§2/3/5.6与既有审计对照。新摘要明确噪声残差、精确无正则有限时域；本地精读仍为v1。再次访问指定abs-v2失败，无版本abs成功，正文此前成功；不由一次失败否定来源。 |
| Stacey | [v1题录](https://arxiv.org/abs/2506.06606v1) | 首发2025-06-07；摘要级。坐标Lp与插值原/对偶迭代背景，未审实验或证明。 |
| Polar Express | [版本页](https://arxiv.org/abs/2505.16932) | 首发2025-05-22，当前v5 2026-05-04；摘要级。有限多项式及低精度实现，不将最坏polar误差当端到端收益保证。 |
| Scion / Norm-Constrained LMOs | [ICML2025正式页](https://proceedings.mlr.press/v267/pethick25a.html) | 正式题录/摘要；报告写正式记录年份，不冒充首次arXiv提交日期。 |
| Sign Compression for Muon | [v1题录](https://arxiv.org/abs/2607.29674v1) | 首发2026-07-31；摘要级。作者报告算子组合次序反例与误差反馈限制，本次不声称自行复现论文反例。 |
| SAMuon / Spectral Allocation | [v1题录](https://arxiv.org/abs/2608.25990v1) | 首发2026-08-26；摘要/题录重核。原文方法和单种子边界沿用10月8日已有p.5—12审计，本次不重复称全读。 |
| MeqMuon | [v1题录](https://arxiv.org/abs/2609.35701v1) | 首发2026-09-28；摘要/题录重核，方法位置见10月8日p.4—6审计。 |
| MuonIO | [v1题录](https://arxiv.org/abs/2610.02705v1) | 首发2026-10-02；摘要/题录重核，已有Alg.1方法审计。仅引用I/O专用范数方向，不混同全训练状态/时间比例。 |
| ORCA | [v1题录](https://arxiv.org/abs/2610.06116v1) | 首发2026-10-05；摘要/题录重核；权重软正交正则的边界来自旧方法审计。 |

## 3. 本地证据与复算限制

- 初始Markdown扫描61个命中文件，路径、摘要片段和SHA256见`local_search_manifest.json`；扫描后又读当天日报JSON。文件数含同论文多种资产，不能改写为论文篇数。
- 沿旧模块兼容链重新只读`reusable_optimizers/pisa/power_aps.py`与`muon.py`（旧长目录名称和旧链接已失效），核对APS旧方向反馈、真实`step_size`、动量形式、形状增益、NS5系数与辅助Adam路由。
- 10月8日`run_evidence.json`保存7份历史运行摘要。本次调用旧`recompute_evidence.py`时，在外部`runs/fig2/nano/adam/full/run_summary.json`触发FileNotFoundError；后续搜索未找到原始runs目录。没有重写历史摘要，没有把旧结果当新实验。
- 新增`combination_checks.py/json`只验证组合的代数关系：逐元素幂不保正交、谱幂后polar恒等、APS重标度失配、标量二次方向割线尺度不变。它不证明随机非凸收敛、不衡量Transformer性能。

## 4. 未闭合项

[自适应步长OpenReview候选](https://openreview.net/forum?id=Ghfc8IRXdy)与[谱平坦OpenReview候选](https://openreview.net/forum?id=PI7PN0x6oA)未获得可读原文。保留冲突可能，后续新颖性判定前补齐。本文已有足够证据设计E0/E1，但不宣称文献查重闭合或组合首次提出。

部分前沿工作的全部代码commit、数据可用性、完整seed结果和理论证明未审计。这里的不足影响普遍性能与新颖性结论，不阻止提出明确可停止的小规模探索。
