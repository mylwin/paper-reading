# JMLR 写作与润色档案

核验日期：2026-10-10。语料窗口：2024–2025。**此档案目前是官方规则与三篇归纳全文与一篇独立留出审阅形成的工作版本，不是已完成50篇的统计总结。** 下载和文本解析不等于语义审阅；实际状态见调研证据 `corpus/jmlr/validation.json`。

## 官方要求

[作者要求](https://jmlr.org/author-info.html)强调稿件完整而精炼，明确主张并以理论或可复现实验支撑，解释发现如何推进理解，交代前人贡献。正文使用官方JMLR样式。摘要不超过200词；正文超过35页可能延迟审理，超过50页（含附录）须在cover letter解释。会议扩展稿必须明确引用与实质增量。最终生成工程须读取模板注册表与相应阶段配置，不根据历史发表PDF反推当前硬限制。

## 10维风格与执行建议

| 维度 | 当前证据及可执行动作 | 类别与置信 |
|---|---|---|
| 读者与目标 | 面向更广的ML读者解释问题为何值得理解；专业符号前给必要定义。 | 官方要求，高 |
| 标题与摘要 | 标题可突出方法，也可陈述理论对象；摘要写缺口、解决机制、结果与适用边界，不强制先写数字。 | 先导观察，中低；J25-16摘要限定连续非负损失，J25-17摘要介绍图跳跃机制 |
| 引言与贡献 | 先说明现有解释/方法为何不足，再把贡献对应到可定位的结果；贡献清单可用也可不用。 | 先导观察，中低；J25-16 §1采用章节路线而非bullet list |
| 相关工作 | 用对比说明概念、假设和证据差异；独立节不是硬要求。 | 先导观察，中低；J25-16 §1/§5、J25-17 §1均融合相关工作 |
| 方法与理论 | 定义、直觉、定理及边界构成主线；主定理条件写在正文，冗长证明可放附录；例子应解释定义。 | 官方原则＋先导观察，中；J25-16 §2–5交替Example与定理，J25-17 §2–3先机制后性质 |
| 实验与结果 | 实验回答具体理论/机制问题，并区分示例、模拟与真实应用；纯理论稿无需虚构benchmark。 | 先导观察，中低；J25-17 §4模拟与§5应用，J25-16无实证实验 |
| 讨论、局限、结论 | 明确结论依赖什么，讨论未覆盖设置；可以专设Discussion，也可合入结论。 | 先导观察，中低；J25-17 §6，J25-16 §6放宽假设＋§7结论 |
| 语言与段落 | 技术条件决定语气；“only”或“improved”必须有对应条件和对象，使用段首目的句引导论证。 | 编辑建议；不据小样本推断全刊句法频率 |
| 正文与附录 | 主文保留完整可理解的论证，附录按正文结果映射；附录不能补救正文遗漏的主张条件。 | 编辑建议，适用于理论/方法稿 |
| 改写与检查 | 建立主张→假设→定理/实验→图表→附录映射，检查重述是否增强结论。 | 编辑建议 |

## 已审支持样本与反例

- **J25-16**，*Bias-variance decompositions: the exclusive privilege of Bregman divergences*：[官方论文](https://jmlr.org/papers/v26/25-0211.html)。全文30页已阅读写作组织、条件与证明衔接（未复证定理）。§2限定损失域，§3定义clean分解，§4列可行例子与等式约束反例，§5唯一性，§6放宽假设，§7结论，附录A逐项证明。其标题很强，摘要和正文却给出条件，反驳“JMLR标题必须保守且无强结论”这种规则；无实证实验，反驳固定实验章节。
- **J25-17**，*Graph-accelerated Markov Chain Monte Carlo using Approximate Samples*：[官方论文](https://jmlr.org/papers/v26/24-1024.html)。全文25页已阅读（未复证定理）。§2用toy example和Remark解释机制，§3.1明确只比较混合时间上界而非实际混合时间，§3.2给(A1–A3)，§4重复模拟、§5应用、§6讨论；附录A证明、B优化、C图选型数值比较、D基线渐近近似。不能把toy case的约100倍提升改成普遍100倍加速。正文与附录定理编号有不一致，视为具体源稿问题，不当作风格模仿。

**留出验证1/10：** J24-21 *Contextual Bandits with Packing and Covering Constraints: A Modular Lagrangian Approach via Regression*（[正式论文](https://jmlr.org/papers/v25/24-1220.html)，全文37页含附录A–C）与“条件化主张、相关工作位置可变、纯理论无需实验、附录结果映射”四条已形成的先导规则相容。其主要证明大量保留主文、参考文献位于附录前，构成正文分配的变体，不能要求所有JMLR稿将全部证明移附录。10篇留出尚未全部语义审阅；上述先导结论不得标为50篇验证的venue特征。50篇最终样本中软件文章Aequitas Flow已排除并替换，排除日志保留。

- **J24-04**，*Regimes of No Gain in Multi-class Active Learning*（[正式论文](https://jmlr.org/papers/v25/23-0234.html)），全文31页已读。§1用相邻的margin定义解释缺口，Table1和两项贡献区分密度条件；§3结果概览、§4完整证明占主文、§5模拟、§6结论。摘要和§1 p3明确“no gain”仅指最坏情况速率，不否认实际常数收益；§5报告10次重复与标准差。该文没有附录，反驳固定把证明放附录的规则。

## 原创改写示例

原稿：`Our method is universally faster and solves high-dimensional sampling.`

改写（仅在材料支持时）：`Under the stated regularity and coverage assumptions, the proposed transition improves the conductance bound. In the evaluated posterior models, it also increases effective sample size per iteration.`

这里区分理论界与实测结果；缺少对应证据时保留待补标记，不生成此断言。
