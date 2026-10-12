# IEEE TNNLS 写作与润色档案

核验日期：2026-10-11；语料窗口：2024–2025。**目前为已核验官方要求和编辑建议，尚无完成的50篇全文风格归纳。** 50篇候选锁定、公开版本下载与语义审阅分开计数；作者公开稿还须核对内容版本。实际进度见 `corpus/tnnls/validation.json`。

## 官方要求与版本边界

[当前完整作者指南](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls)已阅读全文，快照保存在调研证据 `sources/tnnls/author-info.html` 和 `.txt`。当前明确 **double-anonymous**；至少两位独立审稿人。作者联系方式在投稿系统填写，不应混入匿名正文。旧2012作者说明PDF的single-blind与13页阈值不能代替当前规范。

- 常规全文：正常10页，允许收费超长但**总稿最多15页**；综述15/21页；brief 6/9页；comments至多3页。总稿上限不含supplementary materials，不能把收费起点当作允许的无限篇幅。
- 使用标准IEEE双栏、单倍行距PDF，嵌入字体且全部页编号。4–5个关键词；摘要须信息充分、包括重要结果或结论，当前专属指南未给摘要硬字数，不从其他期刊移植。
- Full Paper 要求可归档、充分论证且完整解决一个问题的贡献；Brief Paper 可报告有用的扩展，但贡献仍须有结论性。投稿后不会在类别间自动转换。
- 会议稿须实质性增强；具体增量以材料列出，不把补背景或改变措辞作为研究增量。详细推导鼓励放appendix。图注应能独立理解，坐标轴不能只有未说明的符号。
- 当前门户为 [IEEE Author Portal](https://ieee.atyponrex.com/journal/tnnls)。终稿ZIP包括全部源文件、图、参考文献与相应类别所需作者传记；匿名投稿和含作者终稿不能混用。
- 官方指南要求披露AI生成文字及涉及章节。匿名阶段披露的位置须结合门户规则，不擅自保留能够识别作者的致谢。

## 10维执行档案

| 维度 | 可执行规则 | 类别、支持与边界 |
|---|---|---|
| 读者与目标 | 面向神经网络及学习系统读者说明理论、设计或应用上的技术贡献；问题解决要充分、明确。 | 官方要求，高；指南Scope/Types |
| 标题与摘要 | 标题指出对象与机制；摘要交代问题、机制、可核实结果与结论。 | 摘要功能为官方要求；标题组织为编辑建议，未完成语料统计 |
| 引言与贡献 | 每项贡献对应定义清晰的理论或实验结果；扩展稿明确已发表基础及新增工作。 | 增量要求官方，高；贡献组织编辑建议 |
| 相关工作 | 根据与当前贡献的关系比较假设、机制、任务和证据，不用引用数量代替定位。 | 编辑建议；暂无语料频率或独立节位置结论 |
| 方法、算法、理论 | 正文解释机制与必需条件，详细推导可置附录；算法输入、步骤、输出对应方法文字。 | 附录建议官方，高；算法检查编辑建议 |
| 实验与解释 | 把每组实验连接到一个主张，保留数据、协议、基线与误差含义；应用结果应说明学习系统的技术贡献。 | 编辑建议；不声称某个基线数量是官方规定 |
| 讨论与结论 | 把已支持结论与未覆盖设置分开；不把单任务观察增强成一般保证。 | 编辑建议 |
| 段落、密度、措辞 | 在总稿限制内删重复背景，保留技术条件和证据限定；避免无证据的首次、全面、普适。 | 编辑建议 |
| 正文、附录、补充 | 结果条件留正文、推导可附录；supplement不计总稿上限，但不是隐藏核心论证的理由。 | 页数/推导位置官方，高；证据分配编辑建议 |
| 建议与改写 | 先建主张–证据表再润色；类别和阶段选定后加载独立期刊配置。 | 编辑建议 |

## 语料证据与留出验证

**留出1/10已完成全文审阅：** TN24-23 *Hybrid Quantum-Classical Convolutional Neural Network Model for Image Classification*（[正式DOI](https://doi.org/10.1109/TNNLS.2023.3312170)，[机构公开early-access全文](https://elib.dlr.de/199742/1/TNNLS_2022_P_25391.pdf)）15页无附录。页眉确认内容已final、仅分页可变，DOI匹配；在线2023与print2024分别记录。按此前编辑建议独立检查：§V的8×8下采样、类别子集、三次重复及Wilcoxon结果限定了比较结论；§VI p13明确卷积操作效率不包含编码/测量成本；§VII将硬件收益标为未来可能。三条建议相容，但尚无40篇归纳形成的经验规则，因此不能称完整风格验证。组织反例：Background在Related Work前；没有附录；专门的复杂度分析位于实验后。该留出观察仅用于验证和边界展示，不据其新增venue规则。

下载成功不证明author-accepted版本身份；元数据指向accepted/published的文件也须正文核对。其他9篇留出待完成。上表编辑建议适用于方法/理论/应用稿，不能当作TNNLS独有风格。

## 原创改写示例

原稿：`The proposed network provides a universal solution and significantly improves all existing systems.`

改写（需要材料支持）：`The proposed network separates the two inference stages. Under the stated evaluation protocol, this design reduces inference latency while retaining classification accuracy on the tested datasets.`

数字和显著性检验未提供时不补造；若只测准确率，删去latency主张。
