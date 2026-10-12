# IEEE TPAMI 写作与润色档案

核验日期：2026-10-11；语料窗口：2024–2025。**当前为官方scope与编辑建议的工作档案，尚未完成50篇全文风格归纳。** 50篇候选锁定，公开版本、发表身份、版本内容及语义审阅须分别核验，实际计数见 `corpus/tpami/validation.json`。

## 已核验官方内容与待核验规范

[官方Call for Papers](https://www.computer.org/digital-library/journals/tp/cfp-ieee-pattern-analysis-machine-intelligence)将scope描述为计算机视觉、图像理解、模式分析识别及部分机器智能，强调用于模式分析的机器学习；鼓励在IEEE DataPort存储可连接论文的数据以支持复现。此scope不应被改写成仅适用于优化器或某种网络。

该页引导作者到专属Information and Submission Guidelines；但当前Computer Society网页访问返回403/418，先前下载的作者页只包含网站壳标题，**不是完整指南证据**。当前匿名类型、常规稿投稿最大页数、收费页数阈值和摘要字数仍待从现行一手指南核验。历史14页投稿与12页收费说法来自不同政策口径，不能合并成一个上限；不把TNNLS或TAI规则套用过来。

模板由独立TPAMI注册项按官方选择器管理。模板可以先编译核验，但规范字段未核实时不能称“已可直接投稿”；不要通过缩小字号或边距弥补缺失规范。

## 10维写作操作档案

以下是编辑建议及一篇全文先导观察，不能称为TPAMI已完成50篇统计的特色。

| 维度 | 可执行动作 | 证据类型与边界 |
|---|---|---|
| 读者与目标 | 解释贡献如何推进视觉、模式分析或机器智能的理解与方法；不只列应用表现。 | scope官方，高；贡献组织编辑建议 |
| 标题与摘要 | 用对象、机制和核心结果组织摘要；将数据集范围、评价协议与结论限定写清。 | 编辑建议；摘要硬字数待核验 |
| 引言与贡献 | 问题、已有能力、仍未解决的限制、所提机制、证据依次衔接；贡献逐项对应正文结果。 | 编辑建议 |
| 相关工作 | 在任务/假设/机制/证据上定位；针对直接近邻给具体差异，避免笼统贬低所有前人。 | 编辑建议；独立节位置无语料结论 |
| 方法与理论 | 从整体机制到组成模块、训练/推断、理论条件；公式解释对象与作用，证明条件不可在压缩时删除。 | 编辑建议 |
| 实验与结果 | 明确任务协议与评价指标；主比较、消融、资源成本和失败案例各自回应已提出主张。 | 编辑建议；不虚构官方数据集/基线数量要求 |
| 讨论与局限 | 区分实测范围、未覆盖条件及有证据的失败机制；结论不引入新结果。 | 编辑建议 |
| 段落与措辞 | 段首先说待解决的问题或观察，再给机制/证据和解释；增强措辞必须相应增强证据。 | 编辑建议 |
| 正文与补充 | 正文保留可理解论证和关键设置，详细证明/协议可迁入补充，保留结果与证据映射。 | 编辑建议；补充页数政策待核验 |
| 改写与检查 | 先检查主张与证据，再调整组织和英文；跨venue迁移保留数字、假设、公式和引用键。 | 编辑建议 |

## 支持、反例与留出状态

**一篇归纳先导已完成：** TP25-07 *PC-SRGAN: Physically Consistent Super-Resolution Generative Adversarial Network for General Transient Simulations*（[正式DOI](https://doi.org/10.1109/TPAMI.2025.3596647)，[公开accepted稿](https://arxiv.org/pdf/2505.06502)）全文11页含附录A–D已读。其页眉明确accepted身份且DOI匹配，故按作者接受稿记录，不能将OpenAlex的publishedVersion标签当终排版事实。§I融合相关工作和具体缺口，§II方法，§III同时包含数值误差分析、比较和消融，§IV讨论及结论；附录补模型、离散化与划分、滤波器、时间积分、架构训练和代码commit。反例：没有独立Related Work或Conclusion；理论分析置于Results。13%数据效率仅对应PSNR阈值，MSE需17.7%，不得改写成所有指标或所有问题只需13%。TableII去掉边界loss时MSE/MSGE改善，与正文概括存在张力，须保留数值并提示，不能复制增强表述。置信为单篇组织高、venue频率极低。

10篇留出尚未验证。正式发表元数据与作者公开稿须逐篇建立对应关系；submittedVersion不自动等于正式录用版本。另一候选 *Dynamic Inference by Model Reduction* 的OpenAlex acceptedVersion标签与2023未审预印本页眉相冲突，已记版本排除，不能计入2025全文完成数。

## 原创改写示例

原稿：`Our framework solves visual reasoning and is better than previous work in every aspect.`

改写（仅在材料支持时）：`The framework models relations between detected objects before predicting the answer. On the evaluated visual reasoning benchmarks, this design improves accuracy over the specified baselines; the ablation isolates the contribution of relational aggregation.`

若没有消融或未控制基线协议，移除相应证据声明并标记需要补充的实验。
