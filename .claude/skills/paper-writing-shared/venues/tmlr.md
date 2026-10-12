# TMLR 写作与润色档案

核验日期：2026-10-11。语料窗口2024–2025。**官方规则可用，50篇候选已锁定；已取得3篇固定版本作者公开稿，2篇连续PDF阅读，其中1篇正文及相关附录核查完成。** OpenReview当前PDF有浏览器验证；不能声称50篇已审或作者稿与出版商当前PDF字节相同。

## 官方依据

- [作者指南](https://jmlr.org/tmlr/author-guide.html)：双匿名；必须用TMLR官方LaTeX模板生成PDF；没有统一硬页数，但内容须证明长度合理。附录放参考文献之后，其阅读由审稿人决定。显著潜在伤害的工作需broader impact声明。
- [接收标准](https://jmlr.org/tmlr/acceptance-criteria.html)：主张是否由准确可信证据支持，以及目标读者能否清楚理解并从发现中受益。新颖性、SOTA或主观重大影响不是独立必要条件；证据不足也可通过收窄主张解决。

## 10维执行规则

| 维度 | 操作 | 来源性质 |
|---|---|---|
| 读者与目标 | 明确谁能从发现中学到什么，避免只为声称突破而放大意义。 | 官方标准 |
| 标题与摘要 | 标题指向实际发现；摘要交代研究问题、证据与边界，不虚构SOTA或新颖性。 | 编辑建议，来自官方证据要求 |
| 引言与贡献 | 说明发现对研究群体的价值，允许复现、负结果或已有方法的系统分析。 | 官方标准的执行解释 |
| 相关工作 | 对齐问题、设置与证据，承认前人成果；不通过攻击已有工作制造“新颖”。 | 编辑建议 |
| 方法与理论 | 技术主张与假设在主文说清；读者不读附录也应理解主要结论的条件。 | 官方标准＋编辑建议 |
| 实验与结果 | 给出可复现设置与结论所需证据；不支持的比较结论降低强度。 | 官方标准 |
| 讨论、局限与结论 | 区分观察、解释与猜测，描述失败条件；有显著风险时给具体影响声明。 | 官方指南＋编辑建议 |
| 语言与段落 | 用问题→证据→发现组织，精确而直接，减少不承载信息的宣传性形容词。 | 编辑建议 |
| 正文与附录 | 附录置于参考文献后；核心论证不得依赖审稿人必须阅读补充材料。 | 官方指南＋编辑建议 |
| 改写与检查 | 检查每个“improves”“robust”“generalizes”能否定位相应设置与结果。 | 编辑建议 |

上述建议不能称为2024–2025语料的频率观察。完整归纳和10篇留出验证仍待完成；本批先导证据见下表。没有概率抽样，不给虚构频率或置信区间。

## 连续全文先导：基准稿与复现稿

[G4SATBench审阅](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/tmlr/semantic-reviews/tmlr-2024-01.json)：arXiv2309.16941v2，20页全读，含A–D附录，页首Published TMLR05/2024与审稿forum、作者、题名及官方发表索引一致。[CroPA审阅](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/tmlr/semantic-reviews/tmlr-2025-01.json)：arXiv2506.22982v1，40页全读，Published TMLR06/2025及forum吻合；p40另引详细prompt清单，作者仓库已检查但未确认与归档补充对应，**暂不计正文＋相关附录完整数**。这些是作者存档版本的观察，不能用于推断期刊最终排版。

| 规则／类别 | 支持与位置 | 反例、条件及置信 | 执行方式 |
|---|---|---|---|
| T-P1：研究问题决定结果组织（语料观察） | G4SATBench p6–10按可比基准与行为诊断；CroPA p4列四项原主张，p15–22按复现主张和新增工作组织 | 两篇都是评估类，不能推广成全部TMLR必须有Claim编号；单篇位置高置信，全刊低置信 | 基准稿分公平比较和诊断；复现稿显式区分原主张、复现协议、复现结论与扩展 |
| T-P2：负结果与条件是主要发现的一部分（观察＋编辑建议） | G4SATBench Table5预训练负结果；AppendixD Table12限定solver预算 | 限制32flips的结果不支持优于无限预算经典solver；泛化1seed不同于一般benchmark3seeds | 保留预算、重复次数、失败种子及无收益结果；把机制解释写为假说而非证明 |
| T-P3：长文需可追踪主张，而非重复强调意义（观察＋建议） | CroPA scope/复现标签、AppendixB算法、D配置；G4SATBench Figure1框架及A–D | 附录细节不能代替正文中有效性条件；本批不能证明长文更受偏好 | 给方法、结果与附录明确回链，精简重复背景，保留必要协议 |
| T-P4：原文不作为正确性金标准（编辑建议，由反例触发） | CroPA p18–24正文与Tables3–8、p11公式与p33算法 | 有单项指标退步、CutMix/SCMix说法反向、百分点计算和主表/附录协议未对齐；符号冲突需作者确认，不等于已证明算法错误 | 输出具体位置、冲突值和作者查询；不能为润色静默选有利数字、翻转符号或强化all metrics |

原创改写示例，输入事实为“总体0.8433→0.9685，有一项VQA退步，未确认跨模型协议”：`The overall score increases by 12.52 percentage points under the reported protocol, while one VQA measure decreases. The cross-model results require protocol reconciliation before they support a broader transferability claim.` 这是编辑示例，不为用户论文补造数据。

## 原创改写示例

原稿：`We introduce a groundbreaking new method that dominates existing approaches.`

改写（依证据填充）：`We evaluate [method] under [specified conditions] and identify [supported finding]. The comparison in [table] indicates [measured effect], while [counterexample] limits its scope.`

缺少证据时保留方括号，不填写虚构结果。
