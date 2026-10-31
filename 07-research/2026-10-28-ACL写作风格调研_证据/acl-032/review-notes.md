# Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets

完整审阅PDF 18页及科学附录；逐维证据见对应semantic-review。

**reader_argument**（PDF [1, 3]）：多数据集适配中的负迁移与多adapter成本共同构成问题。 建议：以读者实际权衡组织论证，同时交代性能与资源。 适用边界：分类任务、多固定数据集；非生成任务通用最优结论。

**title_abstract**（PDF [1]）：标题方法与使用场景明确，摘要串联估计器误差、估计成本和最终收益。 建议：摘要按方法依赖顺序报告最有辨识度的量化证据。 适用边界：105×仅估计步骤FLOPs比，不能写最终训练或推理105倍加速。

**introduction**（PDF [1, 2, 3]）：应用→常见方案成本→近初始化观察→任务分组→ensemble→多项验证；图1分输入、分组、组合。 建议：复杂方法用流程图与层级概览降低后文负担。 适用边界：不强制三条贡献；本文是自然段总结。

**related_work**（PDF [8, 9, 15, 16]）：完整相关工作后置，关键技术来源在引言和方法及时引用；附录对梯度相似作专门比较。 建议：近邻差异围绕实际设计轴展开。 适用边界：first comprehensive自称不可复制为新稿未经核实的首创性。

**method_theory**（PDF [3, 4, 5, 13, 14]）：先用负迁移案例，再给affinity定义、局部线性估计、SDP与boosting；两个算法配套。 建议：先说明中间量怎样产生、用于何决策，再给实现；复杂推导可附录。 适用边界：近初始化小距离是经验观察；JL保距不证明整个预测近似或泛化。

**evidence**（PDF [5, 6, 7, 8, 14, 15, 16, 17, 18]）：估计精度→端到端性能/FLOPs/内存→规模扩展→sharpness→梯度相似/transfer。 建议：按方法依赖安排验证，每个中间组件和最终用途各有证据。 适用边界：3seed均值SD并非显著性；用官方dev作test且训练另划validation。

**discussion_limits**（PDF [8, 9, 10]）：结论回收方法与经验分析，局限明确权重梯度访问、固定任务、OOD未验证。 建议：局限集中于会改变使用方式的条件，正文先报告真实收益。 适用边界：无直接社会影响的自述不是技术稿自动免除伦理讨论依据。

**paragraph_language**（PDF [4, 6, 7, 8]）：结果段直接陈述数值，sharpness相关用suggest，经验观察与机制解释层级有别。 建议：保留实测改善力度，对解释采用适当语气，不逐句may。 适用边界：微小归一化距离本身不构成泰勒余项上界。

**main_appendix**（PDF [1, 5, 7, 13, 14, 15, 16, 17, 18]）：正文提供两个算法和主结果，附录A推导/成本、B完整模型与扩展实验。 建议：主文保留能判断权衡的信息，附录补可复核细节和对照。 适用边界：附录含新科学分析必须一起审阅，录用版布局不替代当前投稿页限。

**actionable**（PDF [6, 7]）：性能与资源三轴呈现，可保留方法收益也可显示相对baseline不同成本。 建议：原创例：The ensemble raises average accuracy from 78.6% to 88.6% in the reported setting, with FLOPs increasing from 4.3×10^18 to 4.7×10^18 and GPU memory from 12.6 to 21.4 GB. 适用边界：+10是百分点；不可称内存无代价或所有任务最佳。

原创改写：The ensemble improves average accuracy with a modest increase in FLOPs; its memory cost remains higher than single-adapter QLoRA.

审计问题独立保留在逐篇记录，不要求复制为论文免责声明。
