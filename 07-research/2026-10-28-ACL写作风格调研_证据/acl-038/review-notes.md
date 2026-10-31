# M2RC-EVAL: Massively Multilingual Repository-level Code Completion Evaluation

完整审阅PDF 24页及科学附录；逐维证据见对应semantic-review。

**reader_argument**（PDF [1, 2, 3]）：问题是现有repo补全覆盖语言少且平均数遮蔽细粒度能力，资源与诊断双目标。 建议：benchmark引言先说现有评测无法回答什么，再解释新增标注怎样回答。 适用边界：multilingual指18编程语言，非18自然语言。

**title_abstract**（PDF [1]）：标题命名repo补全评测范围，摘要两个缺口对应18语言/两类标注并引出训练集。 建议：资源、任务与评测用途都可作为贡献，不必杜撰算法创新。 适用边界：comprehensive/effectiveness泛称应在新摘要用可核实发现替代。

**introduction**（PDF [1, 2]）：图1展示3语言实例、输入上下文、cursor、答案及两种标签，随后3贡献。 建议：新任务用一个完整输入输出样例对齐术语，再列资源及分析贡献。 适用边界：源码标注示例不等同人类语义准确性验证。

**related_work**（PDF [3]）：CodeLLM与repo补全两组，Table1按语言/细粒度/训练/仓库数比较。 建议：资源对比表选择直接对应缺口的维度。 适用边界：first massively multilingual自称需独立查新，不从本稿推出所有前作能力受限。

**method_theory**（PDF [3, 4, 5, 13, 14, 19, 20, 21]）：来源许可/星数文件数筛选→AST节点选cursor→质量筛→分割统计→深度bucket与语义映射。 建议：明确样本单位和筛选再报告数量，标注定义配实例。 适用边界：Tree-sitter解析与人工定义映射不同于全量人工标注，AST深度是代理难度。

**evidence**（PDF [5, 6, 7, 8, 13, 14, 16, 17, 18, 22, 23, 24]）：baseline/retrieval/retrieval+tuning对照，随后数据/模型规模、bucket、语义、长度、CodeBLEU、syntax与案例。 建议：先资源用途主结果，再用新增标注给出读者原来得不到的诊断。 适用边界：syntax静态解析非execution；长度趋势不是已拟合普适scaling law。

**discussion_limits**（PDF [7, 8, 9, 13]）：模型在浅层/特殊语言结构/多行有弱项；局限超参成本、任务范围与真实工程复杂性。 建议：负结果可作为benchmark价值；限制聚焦部署解释条件。 适用边界：not seen any risks不是安全保证，不复制为技术稿免审条款。

**paragraph_language**（PDF [5, 6, 7, 8]）：Analysis on作重复路标，观察→读数→解释→用途；HTML规则简单用assume。 建议：段落对应诊断问题，直接写训练集改善与失败类，不累积模糊修饰。 适用边界：原文significant没有检验支撑不可自动当统计显著。

**main_appendix**（PDF [4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]）：正文主表/总体诊断，附录细语种曲线、18taxonomy、六完整案例和分桶理由。 建议：主文保留标签定义与总体验证，完整清单及成功失败案例放附录。 适用边界：附录大量科学图并非参考文献后可跳过内容，必须看到24末图。

**actionable**（PDF [5, 6, 7, 8]）：细粒度诊断补足总分，retrieval和tuning效果应按指标拆开。 建议：原创例：Retrieval increases StarCoder’s reported average exact match from 21.0% to 24.1%, while edit similarity falls from 52.0% to 50.0%. Retrieval with tuning raises both metrics to 44.5% and 72.2%. 适用边界：平均列和18逐语种数值有不一致待核；明确reported，非自算已认证均值。

原创改写：The benchmark pairs repository-level completion with AST-based diagnostics, exposing differences that aggregate scores conceal.

审计问题独立保留在逐篇记录，不要求复制为论文免责声明。
