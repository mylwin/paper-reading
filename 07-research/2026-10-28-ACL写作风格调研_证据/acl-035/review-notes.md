# SubLIME: Subset Selection via Rank Correlation Prediction for Data-Efficient LLM Evaluation

完整审阅PDF 22页及科学附录；逐维证据见对应semantic-review。

**reader_argument**（PDF [1, 2]）：以评测成本与新benchmark冷启动缺历史结果定位，目标是排名保真而非新模型准确率。 建议：先定义要保持的决策结果，再说明减少哪种评测投入。 适用边界：少样本评测不是免除anchor全量评测。

**title_abstract**（PDF [1]）：标题明确subset/rank prediction/data-efficient；摘要动机、信号来源、选子集与跨benchmark结果。 建议：摘要交代输入预算、保真目标与作用对象。 适用边界：80–99%缩样本率不能直接当含预处理/anchor的总成本节省。

**introduction**（PDF [1, 2, 3]）：规模成本→warm-up缺口→RCP→两个贡献；另设IMO类比章节。 建议：类比用于解释选取原则，尽快落到可计算指标。 适用边界：类比不提供排序稳定性的理论保证。

**related_work**（PDF [2]）：分训练数据选择和高效评测，后者按需要历史数据/初始化条件定位。 建议：同任务前作按可用信息与评测预算比较。 适用边界：source自称其他方法一概需100+不能未经核实照搬。

**method_theory**（PDF [3, 4, 5, 6]）：SubMS、特征矩阵、RCP训练与新benchmark推断；图2/3与三算法，模型划分和benchmark五折分开。 建议：多个学习/选择环节明确各环节能访问的数据。 适用边界：GunningFog/Flesch为代理，不等于任务真难度/标注质量。

**evidence**（PDF [7, 8, 12, 13, 16, 20]）：100未见模型上比较RC/SDC，分别看anchor5/10/20；附录给弱组件SubMS、分布与模型列表。 建议：核心表展示每benchmark结果，重复采样检查预算稳定性。 适用边界：RCP回归R²、Spearman排名、Pearson分数相关三者不可混称accuracy。

**discussion_limits**（PDF [7, 8, 9]）：低收益/高难benchmark与随机baseline强的情形明确提及；局限anchor成本/代理稳健性/代表性/新域。 建议：正面突出低anchor条件下的用途，边界放在影响应用的条件。 适用边界：原文冗长为无优势辩护与over98%全称结论不应学成风格要求。

**paragraph_language**（PDF [3, 7, 8]）：主题句解释指标/结果，粗体关键词便于浏览；部分段落将相关放大为确保。 建议：报告实测比较直接简洁，原因解释留为解释而非保证。 适用边界：避免把数据集冗余作为输给随机baseline的已证机制。

**main_appendix**（PDF [4, 5, 8, 10, 11, 12, 13, 14, 15, 16, 20, 22]）：正文流程及主表，A附录给采样、分布、弱组件对照、313模型列表。 建议：核心评测划分与预算在正文，完整实例/清单移附录。 适用边界：附录A.4.3/4标题下内容依赖跨页浮动表，须追到16/22页。

**actionable**（PDF [7, 8]）：GPQA低anchor仍非高相关，方法优势应定位比较和设置。 建议：原创例：With five anchor models and a 20% subset, SubLIME reaches a Spearman correlation of 0.620 on GPQA, compared with 0.560 for TinyBenchmarks. 适用边界：同设置主表提供真实比较；不把模型回归R².8996写最终模型排名.8996。

原创改写：SubLIME uses a small anchor panel to select evaluation subsets that preserve model rankings; the gains vary across benchmarks and anchor budgets.

审计问题独立保留在逐篇记录，不要求复制为论文免责声明。
