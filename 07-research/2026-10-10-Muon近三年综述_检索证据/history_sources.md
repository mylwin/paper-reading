# Muon 起源、规模化实证与理论：一手检索证据

检索截止：2026-10-10；覆盖窗口：2023-10-10—2026-10-10。日期以一手页面首次公开日期为准。本文供主综述引用，属于定点证据审计，不宣称逐篇完整精读。已加载 web-access skill；Chrome CDP 未连接的前置检查由主任务完成，公开页面采用搜索发现与静态原文读取。搜索摘要仅作发现入口，下面标注“摘要级”的条目均已直接打开 arXiv 原始摘要页。

## 结论先行

Muon 的**正式命名、公开配方和竞争性实证起于 2024 年**，不是从 2023 年连续存在的三年技术线。2023 年窗口主要是前驱背景；Newton–Schulz、谱最陡下降、Shampoo 与正交梯度有更早来源，不能把这些数学部件写成 Muon 首创。2025 年关键转折是 Moonshot 的权重衰减与形状尺度校正，以及 Essential AI 对“大 batch 下计算量—时间前沿”的独立研究。理论支持主要解释矩阵结构、谱几何、动量与衰减的作用，**没有证明 Muon 普遍优于调好的 AdamW、SOAP 或 Shampoo**。2026 年直接出现了重要的受控反证与更细的几何改造。

## 18 个一手来源

### H01 — Scalable Optimization in the Modular Norm

- 作者：Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein。
- 首次日期：2024-05-23。
- [原始摘要](https://arxiv.org/abs/2405.14813)；[官方代码](https://github.com/jxbz/modula)。
- 核验层级：**摘要级**。
- 核心证据：递归定义与网络架构匹配的 modular norm，用它归一化基础优化器的更新，使学习率能跨宽度和深度迁移；在“well-behaved atomic modules”条件下给出梯度 Lipschitz 常数的递归形式。
- 意义与限制：为“不同参数角色应该用不同几何”和宽度尺度迁移提供前驱理论；不能直接作为 Muon 在某数据集上的优势证据。

### H02 — SOAP: Improving and Stabilizing Shampoo using Adam

- 作者：Nikhil Vyas 等。
- 首次日期：2024-09-17。
- [摘要](https://arxiv.org/abs/2409.11321)；[v1 全文](https://arxiv.org/html/2409.11321v1)；[作者代码](https://github.com/nikhilvyas/SOAP)。
- 核验层级：**全文入口和摘要核验，方法定点读取**。
- 核心证据：SOAP 在 Shampoo 预条件器的特征基中运行 Adam，持续更新该基下的二阶矩；增加的主要超参数是预条件器更新频率。
- 对照结论：原始摘要在 360M/660M LM 大 batch 实验中报告相对 AdamW 超过 40% 迭代减少、超过 35% wall-clock 减少，相对 Shampoo 两项约 20%；这些数字是作者摘要报告，未逐表重建，主文若使用需附设置边界。
- 限制：论文先于 Muon 的系统论文，原始结果不是 SOAP 对 Muon 的直接对照。不能由 SOAP 优于 AdamW 推导其必然优于或劣于 Muon。

### H03 — Old Optimizer, New Norm: An Anthology

- 作者：Jeremy Bernstein, Laker Newhouse。
- 首次日期：2024-09-30。
- [原始论文](https://arxiv.org/abs/2409.20325)。
- 核验层级：**摘要级；与 H04 的原文引用交叉核对**。
- 核心证据：关掉指数移动平均后，Adam、Shampoo、Prodigy 可以分别视为特定范数的最陡下降；建议按张量在网络中的作用分配不同算子范数。
- Muon 关联：Jordan 直接说明其 Appendix A 提供了用 Newton–Schulz 实现 Shampoo 的计算策略，并影响 Muon；Muon 无动量且无历史累积时可与 accumulation-free Shampoo 联系。
- 限制：关掉 EMA 后的等价不等于“实用 Muon 与完整 Shampoo 等价”，也不意味着 Muon 具有完整曲率估计。

### H04 — Muon: An optimizer for hidden layers in neural networks

- 作者：Keller Jordan；贡献者包含 Jeremy Bernstein、Laker Newhouse、Vlado Boza、Yuchen Jin 等。
- 博客日期：2024-12-08；原文参考文献记录首次公开更新规则为 **2024-10-04**，讨论段记录 speedrun 换成 Muon 的日期为 **2024-10-15**。两者不要混为博客发布日期。
- [作者原文](https://kellerjordan.github.io/posts/muon/)。
- 核验层级：**全文定点，定义、NS、运行成本、前驱、实证标准和开放问题均已读取**。
- 核心证据：对 SGD/Nesterov 动量矩阵执行近似极分解；NS 系数为 (3.4445, −4.7750, 2.0315)，默认 5 步，允许最终非零奇异值近似分布在 0.7—1.3。隐藏矩阵使用 Muon；embedding、head、标量和向量使用 AdamW；Q/K/V 分开处理优于拼在一个矩阵中是作者经验。
- 限制：博客中“罕见方向被放大”明确属于猜想；理论 FLOP 开销低不等于实际 wall-clock 开销低。后续添加早期谱梯度研究的段落标注为 2025-07-12，当前页面不完全是 2024 年原貌。

### H05 — KellerJordan/Muon 官方实现

- 类型：持续更新的官方源码，引用条目为 2024；本次读取日期 2026-10-10，不用搜索引擎自动生成的“Published 2.8 years ago”断言仓库起源。
- [官方仓库](https://github.com/KellerJordan/Muon)。
- 核验层级：**README 定点**。
- 核心证据：当前示例采用 MuonWithAuxAdam，隐藏权重和其余参数分别配置学习率；默认 momentum=0.95、nesterov=True、ns_steps=5；仅学习率和权重衰减通常需要调参。
- 限制：这是可变 master 分支的用法示例，不能将当前代码与历史 2024 配方无版本区分地比较。实际复现应锁定 commit、NS 系数、更新形状校正和权重衰减定义。

### H06 — A Note on the Convergence of Muon and Further

- 作者：Jiaxiang Li, Mingyi Hong。
- 首次日期：2025-02-05；当前摘要页标题带 “and Further”，本次读取的 v2 HTML 标题仍为 **A Note on the Convergence of Muon**。
- [摘要及版本历史](https://arxiv.org/abs/2502.02900)；[已读 v2 全文](https://arxiv.org/html/2502.02900v2)。
- 核验层级：**摘要+全文引言/方法定点**。
- 核心证据：将 Muon 联系到谱范数下目标二次近似的更新方向，分析相关版本的收敛。
- 限制：并非完整实用 hybrid Muon 在非凸 LLM 上优于 AdamW 的定理；主综述引用时必须标明版本和简化设置，不混用范数下的驻点指标。

### H07 — Muon is Scalable for LLM Training

- 作者：Jingyuan Liu, Jianlin Su 等（Moonshot AI / UCLA）。
- 首次日期：2025-02-24。
- [摘要](https://arxiv.org/abs/2502.16982)；[已核验 v1 全文](https://arxiv.org/html/2502.16982v1)。
- 核验层级：**全文定点，§2.2、§3.2、表 2/3、Appendix A/B**。
- 核心证据：加入 decoupled weight decay 控制长训练中的权重/激活尺度增长；采用 0.2√max(A,B) 更新形状校正，使 RMS 接近 AdamW。§3.2 中以 399M—1.5B 非 embedding 参数的 dense Llama 做 scaling；AdamW 先做 compute-optimal 超参数搜索，再把其设置迁移至校正后的 Muon。原文报告匹配 AdamW 损失只需约 **52% training FLOPs**。
- 规模化：Moonlight 是约 3B activated / 16B total MoE，训练 5.7T tokens。
- 限制：52% 来自该拟合和设置，不能写成所有网络 wall-clock 快 2 倍；proprietary data mix、MoE 架构改造与优化器贡献不能混同；最优 Muon 独立完整调参和最优 AdamW 的预算关系需透明报告。

### H08 — MoonshotAI/Moonlight 官方工程材料

- 类型：与 H07 同一研究的代码/模型发布材料，2025 年发布线；持续更新页面。
- [官方仓库](https://github.com/MoonshotAI/Moonlight)。
- 核验层级：**README 定点**。
- 核心证据：公开分布式 Muon 实现，README 说明 ZeRO-1 风格设计以优化状态显存和通信，提供 pretrained / instruction-tuned / intermediate 模型材料。
- 限制：同一项目的论文、代码和模型卡不是三个独立实验来源；“memory optimal”是作者对其设计的说明，需具体到分片策略，不推导所有 ZeRO/FSDP 场景的性能。

### H09 — Practical Efficiency of Muon for Pretraining

- 作者：Essential AI / Ishaan Shah 等。
- 首次日期：2025-05-04。
- [摘要](https://arxiv.org/abs/2505.02222)；[已核验 v1 全文](https://arxiv.org/html/2505.02222v1)。
- 核验层级：**全文定点，§2.1—2.5、§3、Appendix B/G**。
- 核心证据：100M—4B 参数，DCLM 文本和过滤后的 Stack V2 Python 数据，TPU v5p；500M 上跨 128K—16M tokens batch 比较等损失下 compute-time Pareto 曲线；Muon 在大 batch 下保留相对数据效率。独立研究 muP 超参数迁移，并用 telescoping 减少有限宽度与搜索误差。
- 公平性：小尺度细调，500M 验证，1B 跨 batch 小 sweep 后迁移至大尺度；AdamW 使用标准 Optax 实现。参数非全部用 Muon。
- 限制：原文把新 batch training loss 作为 generalization proxy，不是完全独立 held-out 验证；4B 在 §2 批量实验仅 50B tokens，低于标称 Chinchilla-optimal budget。本文对“second-order”的用词不等于算法显式估计 Hessian。

### H10 — PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective

- 作者：Tim Tsz-Kit Lau, Qi Long, Weijie Su。
- 首次日期：2025-05-27；摘要页截至本次为 v4，2026-02-05 修订；本次核验 v1。
- [摘要](https://arxiv.org/abs/2505.21799)；[v1 全文](https://arxiv.org/html/2505.21799v1)；[官方代码](https://github.com/timlautk/polargrad)。
- 核验层级：**全文方法与实验配置定点**。
- 核心证据：从 gradient-anisotropy preconditioning 统一矩阵极分解梯度方法；考察 PolarGrad/PolarSGDM/PolarMuon 与 Muon 的联系，分析数值极分解策略。附录中 GPT-2 Small/Medium、Qwen2.5 等设置明确分开隐藏参数与其他参数的优化器。
- 限制：其“超越 Adam/Muon”结论是作者在自己的矩阵问题与 LM 设置的实验；不能泛化。本文可用于提醒“Muon+AdamW 混合配方本身不是唯一可能端点”，也用于约束 NS 精度和数值实现对比较的影响。

### H11 — On the Convergence Analysis of Muon

- 作者：Wei Shen, Ruichuan Huang, Minhui Huang, Cong Shen, Jiawei Zhang。
- 首次日期：2025-05-29。
- [摘要](https://arxiv.org/abs/2505.23737)；[已核验 v1 全文](https://arxiv.org/html/2505.23737v1)。
- 核验层级：**全文定点，Assumptions 3.1—3.3/4.5、Theorem 4.3 等**。
- 核心证据：普通 Frobenius smoothness 下，作者明确说 Muon 相对 (S)GD 的收敛率没有直接明显优势；改为谱范数 smoothness，且 Hessian 可由低秩/近似 block diagonal 结构控制时，Muon 的核范数驻点复杂度界可更好。
- 关键限制：理论优势依赖不同几何常数与 Hessian 假设，比较对象主要 GD/SGD；不构成对 AdamW/SOAP 的一般速度保证；核范数梯度与 Frobenius 梯度的 epsilon 不能当同一个指标。

### H12 — Muon Optimizes Under Spectral Norm Constraints

- 作者：Lizhang Chen, Jonathan Li, Qiang Liu。
- 首次日期：2025-06-18。
- [摘要](https://arxiv.org/abs/2506.15054)；[已核验 v1 全文](https://arxiv.org/html/2506.15054v1)。
- 核验层级：**全文定点，§3、§5—7，未逐行复核全部证明**。
- 核心证据：Muon 作为取核范数的 Lion-K；带 decoupled weight decay 的理想更新对应权重谱约束，§7.1 给 σ_i(X)≤1/λ；包含动量、Lyapunov 与 KKT 视角。
- 实用边界：全文 Eq. 2 写的是隐式更新（衰减使用 X_{t+1}），且用精确 matrix sign；实际代码的显式衰减与 5 步低精度 NS 不能无说明照搬定理。若更新另乘形状尺度，谱半径尺度也必须相应调整；不能机械把任何实现的 λ 解释为 1/λ 硬约束。

### H13 — Clarifying Shampoo: Adapting Spectral Descent to Stochasticity and the Parameter Trajectory

- 作者：Runa Eschenhagen, Anna Cai, Tsung-Hsien Lee, Hao-Jun Michael Shi。
- 首次日期：2026-02-10。
- [摘要](https://arxiv.org/abs/2602.09314)；[已核验 v1 全文](https://arxiv.org/html/2602.09314v1)。
- 核验层级：**全文定点，§2—2.2、表 1/4、Appendix B.1/B.1.2**。
- 核心证据：C4 dense Llama 320M/1.5B，不同 token budgets，同一 PyTorch Distributed Shampoo 代码路径；最佳设置重跑 10 seeds。Shampoo 变体在 token 效率上匹配或超过 SVD Muon，且 NS 版本也作消融。调 ε 可以翻转 Shampoo 与 Muon 的排序。
- 数值点：表 4，320M/1× Chinchilla/B=64，Shampoo^(1/2) perplexity 25.31±0.08，NS Muon 26.31±0.14；这些是作者 mean±2σ，不是标准差。
- 限制：token 更好不等于 wall-clock 更好；仅 C4/Llama、固定 schedule，weight decay 未完整调参。SOAP 相关排序综述为他文引述，不能当该论文的直接 SOAP 实验。

### H14 — Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning

- 作者：Yechen Zhang 等。
- 首次日期：2026-03-10；当前 v2 修订 2026-04-01。
- [原始摘要与版本](https://arxiv.org/abs/2603.09697)。
- 核验层级：**摘要级**。
- 核心证据：在 Shampoo Kronecker 统计定义的白化坐标中做 NS，尝试补足单纯均匀谱更新忽略方向曲率的问题。
- 作者摘要报告：160M—800M LM 中相对 Muon 约 12% training steps 减少；本次未对其原始表逐项复核，主综述宜只写“提出曲率白化后的正交更新”，不把百分比当已复现实证。
- 限制：作者关于 Stiefel 的表述不代表“Muon 把权重强制在 Stiefel 上”；Muon 主要正交化的是更新。Mousse 对 Muon+APS+LP 的近邻意义需要按 APS/LP 的实际定义判断。

### H15 — Reassessing Muon for Matrix Factorization

- 作者：Ali Parviz, Gal Mishne, Alex Cloninger。
- 首次日期：2026-07-14；v2 修订 2026-08-01。
- [原始摘要](https://arxiv.org/abs/2607.13246)。
- 核验层级：**摘要级**。
- 核心证据：低秩矩阵分解的受控问题，与仔细调参的 adaptive baselines 比较，Muon 并不稳定超过 AdamW，既有优势对超参数敏感。
- 限制：这是受控外推失败证据，不是否定 LM 实证；没有核验具体任务/表数值，主报告不宜附未经核验的优势百分比。

### H16 — Optimizers for Diffusion Models: A Controlled Benchmark

- 作者：Arman Bolatov 等。
- 首次日期：2026-09-19。
- [原始摘要](https://arxiv.org/abs/2609.23055)；[作者代码](https://github.com/armanbolatov/diffusion-baselines)。
- 核验层级：**摘要级**。
- 核心证据：四种 diffusion formulations，对比 AdamW/Lion/Muon/SOAP/MARS/MARS-M/Schedule-Free，共享搜索方案，各胜者完整预算 3 seeds 重训；Muon、MARS-M、SOAP 各自至少一种 formulation 优于调好 AdamW，胜者随 formulation 改变。
- 限制：是模型目标依赖的证据，不能用“至少一种胜出”写成普遍优势；本次未按 formulation 提取最佳算法与原始误差条。

### H17 — AMUSE: Anytime Muon with Stable Gradient Evaluation

- 作者：Jueun Kim, Baekrok Shin, Jihun Yun, Beomhan Baek, Minhak Song, Chulhee Yun。
- 首次日期：2026-05-21；v2 修订：2026-07-14。
- [原始摘要与版本历史](https://arxiv.org/abs/2605.22432)。
- 核验层级：**摘要级**。
- 核心证据：以 river-valley 视角讨论正交更新增强低曲率 bulk 进展、同时放大 dominant-direction 噪声；以 schedule-free averaging 稳定梯度评估，插值系数随时间从 fast iterate 逐渐转向 averaged iterate。
- 限制：作者报告改善性能—迭代 Pareto 前沿，本次没有逐表核验，不能转换成 wall-clock 或 FLOP 优势；它是动态平均/梯度评估位置的近邻，并不能自动覆盖任何名称为 APS 的方法。

### H18 — Anytime Training with Schedule-Free Spectral Optimization

- 作者：Anuj Apte, Pranav Deshpande, Niraj Kumar, Shouvanik Chakrabarti, Junhyung Lyle Kim。
- 首次日期：2026-05-21；v2 修订：2026-09-25。
- [原始摘要与版本历史](https://arxiv.org/abs/2605.23061)；知识库已有 PDF：`01-raw/2026-10/Anytime_Training_with_Schedule_Free_Spectral_Optimization.pdf`（本证据条目没有宣称精读此 PDF）。
- 核验层级：**最新原始摘要级**。
- 核心证据：SF-NorMuon 用一个超参数配置在 125M/772M LM、1—8× Chinchilla horizons 匹配或超过调参 AdamW；fast iterate 上的 weight decay 对长 horizon 稳定性关键。
- 限制：v2 摘要同时明确说仍落后 horizon-aware cosine NorMuon 约 0.03 nats，完全匹配 scheduled spectral optimizer 仍开放。不能把“胜过调好 AdamW”写成“已消除所有 schedule-free 差距”。horizon 是未来实验的重要独立轴。

## 主报告可用的证据分层与实验约束（研究者综合）

1. **强直接实证**：H07 支持 Muon 在其 LLM scaling 设置的 FLOP 优势；H09 支持大 batch 下更好的计算量—时间前沿；H13 则显示在受控 C4/Llama 条件下，调好的 Shampoo 仍能有更高 token 效率。三者互不矛盾，因为目标、预算、数值算法和成本口径不同。
2. **原理证据**：H03/H11/H12 能解释矩阵谱几何、结构 smoothness 与衰减，但不证明所有模型都优于 AdamW，也不自动覆盖有限 NS、低精度、参数分片、混合 AdamW 配方。
3. **研究可行性推论**：Muon+APS+LP 如果意图改变谱、曲率、更新幅度或低秩结构，应分别控制“几何方向变换”与“单纯步长/正则强度变化”。建议固定公平搜索预算，先在小模型比较 AdamW/Muon/SOAP/Shampoo，再进行 APS/LP 的 2×2 消融；并报告等 token、等训练 FLOPs、等 wall-clock 三种结果。
4. **数值与预算审计**：锁定 NS 系数/步数/精度，记录 exact polar 与实际 NS 的差别；分别列出 auxiliary AdamW 的参数类。训练 LR decay 后再判断排序，不能只用早期 train loss。
5. **命名警戒**：此文件不推断 APS、LP 的含义；“LP=low precision / layerwise preconditioning / linear probing”等不同定义会改变对应近邻与可行性。必须结合知识库确认后写主综述。
