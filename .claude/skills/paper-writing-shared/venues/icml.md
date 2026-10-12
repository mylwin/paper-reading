---
venue: ICML
corpus_years: [2024, 2025]
sample_count: 50
induction_count: 40
holdout_count: 10
reviewed_count: 4
review_status: incomplete-writing-focused-full-text-audit
full_text_and_external_appendix_review_complete: false
verified_date: 2026-10-11
---

# ICML：以贡献类型组织可检验的机器学习论证

## 证据与适用范围

50篇主会研究论文在阅读前锁定，2024、2025各25篇，每年20归纳＋5留出。全部已取得PMLR官方发表版PDF及逐页文字解析，当前完成4篇正文及所含相关附录的写作审阅，留出验证尚未完成。**此档案是有证据的初版，不能表述为50篇全文验证的风格结论。**完成量以[validation](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/icml/validation.json)为准，逐篇见[manifest](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/icml/manifest.json)和[审阅记录](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/icml/reviews.json)。文字解析本身不算审阅；数学条件和证明组织已读，不代表重新证明或复现实验。

采样从完整PMLR235/267研究论文列表，按标题代理主题分层，排除Position题名，确定性选择且尽量避免重复第一作者。代理主题不是论文真实类型，语义审阅后才更正类型。样本不是全会风格的概率估计，不能据小样本宣称频率或区别于其他会议。

## 官方要求：2026主会、阶段分别适用

投稿正文8页，匿名；终稿正文9页，使用`accepted`模式。引用及附录不限页并入同一PDF，但关键评审证据需在正文。投稿50MB、终稿20MB；US Letter和年度官方LaTeX样式。终稿摘要单段、理想4–6句，标题及节标题使用内容词首字母大写。主会必需Impact Statement；终稿存在金融等实质利益冲突时在引言末标明Conflict of Interest Disclosure。这些是版本规则，不能由2024–2025录用稿倒推更改。[2026作者指南](https://icml.cc/Conferences/2026/AuthorInstructions)

代码与数据的可复现性说明、匿名链接、既有论文区别和准确引用需随真实材料填写。相应政策加载目标年份官方文件；已验证最新届次为2026，未获得2027指南，不自行改年份。样式包、模板检查和编译结论由配套模板包提供。

## 十维规则：观察与建议分开

以下`D`=af0821a5诊断论文，`P`=4207e021线性排序，`M`=09851122混合专家统计理论，`V`=fc8aa210 Verilog应用；均为2024。每项**语料观察置信度低**，只涵盖所列样本。**编辑建议**是可选论证方法，不是官方强制章节模板。

| 维度 | 语料观察、支持与反例 | 编辑建议及适用条件 |
|---|---|---|
| 1.读者与目标 | D pp.1–2用角色互换例子引入组合性；P p.1先正式定义排序问题；V pp.1–3解释EDA专业流程。反例：M pp.2–3要求概率模型背景。 | 面向目标读者交代任务及价值；专业应用说明评估量，理论稿保留形式对象。不能统一改成大众科学报道。 |
| 2.标题与摘要 | D为任务＋视角，P短描述，M明示理论对象，V为系统＋机制。四篇摘要均不以数值提升为必要信息。 | 摘要交代问题、缺口、贡献与证据类型；定理写条件和保证，诊断写发现范围；有可靠关键数字时再加入，不造数字。 |
| 3.引言与贡献 | D分三个问题、三个发现、贡献；P四项贡献及最佳速度；M设置、挑战、表1和贡献；V机制到应用任务。反例：贡献顺序并不相同。 | 贡献逐项对应定义、证明、数据或实验；引言中的强结论需能定位后文证据。 |
| 4.相关工作 | D分VLM、组合性、交互；P对比对称聚类与非对称排序；V分领域生成与可控生成。反例：M把相关文献整合入引言。 | 围绕差异及未解缺口组织，允许独立栏目、背景合并或引言整合；不机械要求早期独立Related Work。 |
| 5.方法、算法、理论 | P条件、命题、短证明、算法和不兼容反例在正文；M条件、结果表、证明路线在正文，A–C完整推导；V流程图和代码例子。 | 按贡献类型选择表达；理论条件紧邻结论；借用定理需归属明确，证明直觉不得替代完整证明。 |
| 6.实验与证据 | D按三个问题交织方法及结果；P同时报告求解速度和固定变量；V分功能、节点及验证耗时。反例：M全部合成实验位于附录D。 | 每项实验回答一个主张，保留数据选择、评价量、对照和运行成本；理论稿可以以证明为主，不强制排行榜实验。 |
| 7.讨论、局限、结论 | D G承认理论与实践距离；P p.9保留慢于基线实例；M p.9明说模型正确指定假设；V p.8说明闭源概率不可见及额外开销。 | 结论回到已验证对象；未来跨域应用要明确未验证；不能把限制当冗词删除。 |
| 8.段落、措辞强度 | P以条件和存在量限定，D以may描述解释；M区分统计率与EM经验行为。反例：V有groundbreaking/any-task扩张措辞。 | 以证据强度决定prove/show/suggest，明确比较对象；接受稿中的宣传词不是写作规范。段落承担一个论证动作。 |
| 9.正文与附录 | D A–G含额外诊断及设置；P A证明B实验C实例；M A–C证明D实验；V无文字附录。 | 正文留关键条件、机制与判别证据，细推导和补充测试显式引用附录；不要因为模板有Appendix而填无关内容。 |
| 10.可执行修改 | M的样本量估计率与优化迭代率不相同；V的综合节点不是全流程PPA；P最佳加速不代表每个实例。 | 起草建贡献—证据—条件表；润色不改变对象、数值、定理条件及引用键；压缩先去重复再调整图表组织，不改官方版式。 |

## 留出验证

10篇划分已固定，尚未完成全文审阅；不能宣称上述建议通过留出验证。验证须逐篇记录哪些规则适用、哪些需修订，以及反例是否来自论文类型、年份或论证结构，不能只报“通过”。

## 原创改写示例

给定事实：两个指定合成模型在模型正确指定和有界参数条件下有统计率保证；优化器仅作数值实验。

- 弱稿：`Our optimizer guarantees fast convergence for all mixture-of-experts models.`
- 理论组织：`For the two specified mixture models with bounded parameters, we establish the stated sample-size rates under correct specification. The numerical optimizer is evaluated separately and carries no global iteration-complexity guarantee.`
- 实证组织：`The numerical study compares the two specified mixture models under the same initialization protocol. Its observed optimization behavior complements the statistical analysis without extending that analysis to general neural mixtures.`

例句原创，非摘抄或用户事实补全。此处差异由论文类型决定；尚无证据声称其他venue不适用。

## 待完成

余下46篇全文及相关附录审阅、2025覆盖、全部10留出验证、跨venue控制论文类型比较。完成后按真实新证据更新置信度；现阶段通用编辑建议可以执行，不能将此初版当作语料研究验收已通过。
