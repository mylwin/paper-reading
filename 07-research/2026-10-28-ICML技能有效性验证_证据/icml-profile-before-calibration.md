---
venue: ICML
corpus_years: [2024, 2025]
sample_count: 50
induction_count: 40
holdout_count: 10
reviewed_count: 37
review_status: incomplete-writing-focused-full-text-audit
full_text_and_external_appendix_review_complete: false
verified_date: 2026-10-11
corpus_review_updated: 2026-10-28
official_rules_reverified_this_batch: false
---

# ICML：以贡献类型组织可检验的机器学习论证

## 证据与适用范围

50篇主会研究论文在阅读前锁定，2024、2025各25篇，每年20归纳＋5留出。全部已取得PMLR官方发表版PDF及逐页文字解析，当前完成37篇正文及所含相关附录的写作审阅（2024年19篇、2025年18篇，均为归纳），留出验证尚未完成。**此档案是有证据的初版，不能表述为50篇全文验证的风格结论。**完成量以[validation](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/icml/validation.json)为准，逐篇见[manifest](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/icml/manifest.json)和[审阅记录](../../../../07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/icml/reviews.json)。文字解析本身不算审阅；数学条件和证明组织已读，不代表重新证明或复现实验。

采样从完整PMLR235/267研究论文列表，按标题代理主题分层，排除Position题名，确定性选择且尽量避免重复第一作者。代理主题不是论文真实类型，语义审阅后才更正类型。样本不是全会风格的概率估计，不能据小样本宣称频率或区别于其他会议。

## 官方要求：2026主会、阶段分别适用

投稿正文8页，匿名；终稿正文9页，使用`accepted`模式。引用及附录不限页并入同一PDF，但关键评审证据需在正文。投稿50MB、终稿20MB；US Letter和年度官方LaTeX样式。终稿摘要单段、理想4–6句，标题及节标题使用内容词首字母大写。主会必需Impact Statement；终稿存在金融等实质利益冲突时在引言末标明Conflict of Interest Disclosure。这些是版本规则，不能由2024–2025录用稿倒推更改。[2026作者指南](https://icml.cc/Conferences/2026/AuthorInstructions)

代码与数据的可复现性说明、匿名链接、既有论文区别和准确引用需随真实材料填写。相应政策加载目标年份官方文件；已验证最新届次为2026，未获得2027指南，不自行改年份。样式包、模板检查和编译结论由配套模板包提供。

## 十维规则：观察与建议分开

以下`D`=af0821a5诊断论文，`P`=4207e021线性排序，`M`=09851122混合专家统计理论，`V`=fc8aa210 Verilog应用；前四篇均为2024；后续跨年度新增见后节。每项**语料观察置信度低**，只涵盖所列样本。**编辑建议**是可选论证方法，不是官方强制章节模板。

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

## 2025年新增两篇：规则适用核对

语料审阅更新于2026-10-21；官方格式要求仍沿用原快照，本轮没有重新联网核验或修改模板。两篇均是预先锁定的归纳样本，不是正式留出；下面为**归纳规则适用核对**，不是盲测或独立生成行为评测。全venue置信度仍低，不从先导样本估计风格频率。

| 新样本及完整范围 | 支持的组织方式 | 润色或压缩必须保留的边界 |
|---|---|---|
| `icml-2025-1bf35893`，隐私图稀疏化，正式14页含A/B；视觉核对2–5、12–14页 | 引言内Related Work→Technical Contribution/比较表→工具与算法→隐私、概率、误差分项分析→新分解子模块→应用和开放问题；A补高概率证明，B回链下游推论 | edge adjacency与公开节点、非负权、允许乘性误差、(ε,δ)-DP、高概率和参数依赖构成整体。多项式时间不是实测速度，理论下游推论不是实验。Table1比较不同隐私类型，不能只按指数排名。 |
| `icml-2025-94972568`，关键token与cDPO，正式13页含A/B；视觉核对3、4、6–8、12–13页 | 先定义与100例诊断→可扩展估计→训练目标→主结果→Related Work→结论；A给受限分布解释/β消融，B补温度/模型 | 64次rollout零成功是有限样本结果；masking错误prefix不是整体测试准确率。0.002%为估算前向次数比，不是实测耗时。cDPO与基线学习率不同；同方差Gaussian解释不是一般模型保证。主实验β=1.0不能静默改成消融最优观察点。 |

两篇对现有十维建议的支持和修订：

- 题目和摘要按证据写：图算法摘要有关键误差指数，cDPO摘要主要交代对象和流程；都不能推出ICML要求或禁止数字摘要。
- 章序与贡献形式可变：前者将Related Work放引言子节，后者在结果之后；理论主定理加比较表与经验三项贡献列表均可成立，不强制统一栏目或“先Related Work”。
- 理论、诊断、识别近似、最终训练效果和成本估计分开。正文保留决定结论的条件，附录按定理、公式和实验指针回链；纯理论稿无需补造实验。
- 新的实证反例要求收窄措辞：MATH500中19.6相对19.4的结果不能泛称每个设置均大幅提升；Table3虽标p<0.005，本轮未独立复算检验或补造其流程。
- 原页核验是必要补充：图算法的绝对值/根号范围、cDPO的表格列和乱码图例不能只靠文本抽取判断。发现可疑数值或数学含义时列作者核对项，不静默修正。

逐篇十维记录位于 `corpus/icml/semantic-reviews/`，也由既有 `reviews.json`统一索引。完整来源、规则适用矩阵、清理与检查见[ICML专项报告](../../../../07-research/2026-10-21-ICML论文风格验证.md)。两篇科学附录均包含在已读PDF中；本轮未发现另指外部科学附录，也未重新抓取落地页或运行代码。已读显示推导不代表独立正确性认证。

## 留出验证

10篇划分已固定，尚未完成全文审阅；不能宣称上述建议通过留出验证。验证须逐篇记录哪些规则适用、哪些需修订，以及反例是否来自论文类型、年份或论证结构，不能只报“通过”。

## 原创改写示例

给定事实：两个指定合成模型在模型正确指定和有界参数条件下有统计率保证；优化器仅作数值实验。

- 弱稿：`Our optimizer guarantees fast convergence for all mixture-of-experts models.`
- 理论组织：`For the two specified mixture models with bounded parameters, we establish the stated sample-size rates under correct specification. The numerical optimizer is evaluated separately and carries no global iteration-complexity guarantee.`
- 实证组织：`The numerical study compares the two specified mixture models under the same initialization protocol. Its observed optimization behavior complements the statistical analysis without extending that analysis to general neural mixtures.`

例句原创，非摘抄或用户事实补全。此处差异由论文类型决定；尚无证据声称其他venue不适用。

## 待完成

余下36篇全文及相关附录审阅、补充2025不同类型覆盖、全部10留出验证、跨venue控制论文类型比较。完成后按真实新证据更新置信度；现阶段通用编辑建议可以执行，不能将此初版当作语料研究验收已通过。


另两个原创编辑示例，仅在研究材料具有对应事实时使用：

- 理论：`Under edge adjacency, the stated guarantee combines multiplicative and additive cut errors with high probability. Its simplified exponent suppresses the reported privacy and approximation parameters.`
- 实证：`The reported efficiency comparison counts estimated forward passes; it is not a wall-clock benchmark. The distributional illustration additionally assumes equal-variance Gaussian models.`

这些例句不是论文结果的独立认证，不可将其内容补作用户的新研究事实。


## 持续审阅：ICML第7–8篇归纳样本

2026-10-21继续完整审阅`icml-2025-29ddd3f6`（TARL，12页含A/B）与`icml-2024-a5ae56e8`（IRED，13页含A/B），原页核验14页。当前累计8/50（2024五篇、2025三篇）；40归纳和10留出的验收尚未完成。完整记录与后续进度见[全量续研报告](../../../../07-research/2026-10-21-论文写作风格全量续研.md)。

- TARL：区分训练期offline数据、测试状态使用和无需测试reward。熵过滤与KL消融显示强超参数依赖，不能把降低动作熵写成OOD消除或安全保证。D4RL以随机/专家为归一化锚点，分数可高于100；score差和relative return分母须保留。σ的含义及KL式号有原版不一致，作为作者核对项而非静默修改。
- IRED：训练/推理伪代码并列，三类实验分别按任务指标、难度变化、可视化和消融组织，A/B补训练设置与模型结构。更难Sudoku上的优势不等于普通/visual Sudoku所有设置领先；路径规划只评第一步是否缩短到goal的距离。结论明示无效答案可能低能、推理开销、无额外记忆等局限。

这两篇扩充经验方法的证据组织观察，不构成独立留出或跨venue风格差异证明。新的可执行检查是保留反馈类型、归一化基准、任务指标与难度条件；抽象的“泛化更强”必须回链到具体测量。


## 持续审阅：第9篇DOMINO（2024）

`icml-2024-ccca6fde`正式16页含A–D全部读完，关键9页原图核验。格式合法率、任务准确率、困惑度和吞吐分别报告；正文定义与scanner/parser图连接方法，附录公开语法、提示、重分词过程及额外F1结果。不是所有论文都需要独立Related Work章，本篇将技术比较融入挑战章节。

可执行建议：保留词表桥接token与最小干预的定义，不能把有限lookahead一律写成fully aligned；注明速度不含语法预计算及离线已知语法条件，C等设置仍慢于无约束生成。附录简化C语法不代表全C语义保证；Llama的CP结果与GUIDANCE并列，不能改成严格领先。原版算法符号和不同表基线存在待作者核对项，保持源数值。

上述是样本观察与编辑建议，尚无全venue频率或留出验证依据；官方模板要求未在此批重核。


## 持续审阅：第10篇GaussianPro（2024）

`icml-2024-cea7b10d`18页含A–D全部读完、13页原页核验。先以渲染失败图建立问题，再按预备知识、2D/3D几何传播、平面损失及多种消融组织；附录补天空mask、邻视图、推导、成本与逐场景表。

可执行建议：对齐初始化再归因。MipNeRF360原3DGS27.21、更好SfM重训27.88、本文27.92不能合并为方法独立获得0.71dB。性能是质量/训练/FPS/点数的多维权衡，Waymo平均提升不代表每场景胜出，ZipNeRF在Mip平均质量仍较高。`3DGS*`在不同表有不同定义，必须保留局部释义；源式符号/索引和表内数值疑点列作者核对。无动态建模的限制应保留，数字更好不等于已做统计显著性检验。

这仍是归纳样本观察，ICML正式留出尚未开始，官方模板规则未重核。


## 持续审阅：第11篇Double Variance Reduction（2024）

`icml-2024-3c86a8a5`19页含A/B完整证明与C四数据集超参数全部读完，核看13页。理论稿按误差来源→新估计器→算法→三项Lyapunov递推→参数选择→复杂度→数值实验组织。

可执行建议：每步期望O(1)不能写成总成本不随维度增长；强凸光滑有限和加凸prox项、平滑偏差与精度依赖需保留，固定v界含残差。实验证据是逻辑回归的SZO归一化曲线，LLM只是应用背景引用；附录两个数据集采用周期快照，不能悄悄称完全相同的loopless更新。局部prox系数、符号/γ正负与引理条件疑点作为作者核对项，不静默改数学。本轮阅读全文和证明组织不构成独立重证。


## 持续审阅：第12篇Minerva（2025）

`icml-2025-c37cd014`19页含A–D全部读完，核看13页。基准稿按能力构念、原子/组合测试、分项诊断、输出错误和有限变体组织，没有强制贡献bullet或新模型定理，Related Work在结果后。完整模板、难度/样本数、计分器、长度与提示变体放附录。

可执行建议：固定1110合成样本快照不等于整个可编程空间。保留默认4k/state约1.5k与输出上限，exact match、ROUGE-L F1/recall和Jaccard分开；Theory of Mind在此仅多agent集合状态操作。错误原因用限定解释，微小提示变体不能推出prompt无关；表中正差值反驳“所有模型均下降”，CI与不完整变体/计数疑点作为作者核对项。真实日历/医疗等只是动机，不能补成已验证部署。


## 持续审阅：第13篇ARS（2025）

`icml-2025-cd2d22c4`19页含A–F全部读完，核看15原页。通过单任务与多任务固定缩放的失效实验建立动机，再将历史buffer均值缩放、周期reset和增强变体拆开；附录保留成功定义、LoRA、完整曲线及horizon/PopArt等消融。

可执行建议：基础ARS、ARS-LN、ARS-LoRA及网络宽度必须分别报告，不将所有增益归因给单组件。均值均衡不代表完整分布相同；scale仅在reset前更新，正均值条件不能省。平均、bottom-k、ESTR和48/50子集均值不互换；8seed的±标准差不是CI，百分点与相对百分比需区分。匹配horizon重跑基线是比较条件，不能混用150/500步与不同容量来归因。源LoRA双零初始化、reset对照及图标签疑点只列作者核对。经验稳定不升级成理论收敛保证，仿真不升级为真实部署。


## 持续审阅：第14篇StableMask（2024；2026-10-22完成）

`icml-2024-2ff98956`正式20页全部读完，A–F理论/实现/训练配置完整；G只有标题。经PMLR→作者仓库→arXiv链取得同题作者v1，补读其F的7页注意力可视化。正式15页＋作者7页原页核验，不将作者旧版充作正式模板，不声称重读作者版25页。来源和版本映射见逐篇记录及专项source-audit。

可执行建议：位置表示定理是特定权重、一个head与两embedding维的存在构造，不能写成训练必然获得该表示。causal isotropy严格且仅为说明，BOS破同输入对称的例外应保留。原SM不能KV缓存，SM-I后缀改写才能得到近基线延迟；无新增可学习参数仍有γ与额外计算。PIQA/ARCC并列不写严格领先。τ边界、γ表标题与融合伪代码的疑点列作者核对，阅读不等于数值证明/实现审计。补充头图缺完整采样/对照标签，不估计全head的DA消除比例。


## 持续审阅：第15篇DP Prediction Certification（2025）

`icml-2025-24d32f9f`20页及A–E区间传播、敏感度网格、全证明和实验配置全部读完，核看16原页。由私有预测访问边界到参数区间、稳定充分条件、机制与查询预算曲线组织，不把训练模型可公开混入预测隐私。

可执行建议：认证失败不等于证明不稳定；元素级裁剪不改成DP-SGD的L2裁剪。GPT-2实验是冻结公共嵌入上的小分类器，医疗仅敏感Drusen微调；AMEX95%稳定覆盖不是准确率。SS在大Q/大集成和OCT蒸馏并非总优，运行与k网格成本保留。D.6任意数据依赖上界替换平滑敏感度的论证需进一步核对相邻数据集尺度控制；正文/附录集成K公式、D.4归一化及softmax上下标差异列作者问题。写作审阅不构成独立隐私证明认证，不将其未核实数学结论转为技能公理。


## 持续审阅：第16篇EnzyGen（2024）

`icml-2024-d85e13c4`21页及A–D完整读完，14原页核验。跨学科应用稿由领域定义、联合生成/三loss、数据和评价量，到问题驱动的消融、zero-shot、案例与独立Discussion组织；等变性证明放附录，不能代替实际催化能力的验证。

可执行建议：ESP、Gnina docking和AF2 pLDDT都是计算代理，不能改成湿实验/催化活性/热力学稳定性。3157是PDB子集，均值优不等于全部类优；主表有基线更好的类，微调有8类退步。规模对照混合预训练与随机初始化，RFDiffusion用发布权重而其他基线重训，不能只归因模型大小或组件。两新类/两新底物top32是有限zero-shot。保留位点坐标/长度输入，核对MSA与split信息流、top-k选择和均值口径。λ似然、证明范数/编号及附表重复行作为作者核对项，不静默改数学或数字；隐藏图文字不能当可视版式缺陷。


## 持续审阅：第17篇MFSC（2025）

`icml-2025-515dbc4f`22页含A–F全部读完，17原页核验。先给MV-MDP/度量/条件性值误差界，再给融合与mask机制，实验由五个具体问题串联，附录补九项对照、资源和PPO/SAC伪代码。

可执行建议：c≥γ、ε聚类、Markov和值假设紧邻理论结论，不当成实际训练已达界或任意缺失鲁棒证明。缺失仅随机去一个视角；不互补的信息仍无法恢复。GradCAM/top2.5%梯度和MINE关联不证明因果识别。Ant可比、DMControl落后TACO、CARLA成功率/撞击反例与更多训练/显存代价保留。F2C文献值、gripper信号、PPO/SAC及4seedCI/4seedstd/3seedstd分别说明；时序单视角不称多物理相机。正文/伪代码loss、λ方向与动态输入疑点列作者核对，不静默改数学或声称代码复现。


## 持续审阅：第18篇Scientific Generative Agent（2024）

`icml-2024-064d513a`23页含A全提示、B流程/任务边界、C完整比较/长迭代、D.1–D.9代码及分子列表全部读完，15原页核验。跨领域动机后用明确E/Θ/θ/Φ接口与双层目标，再按任务、机制消融、骨干和案例组织；Related Work在实验后，结论先列六限制。

可执行建议：仿真轨迹和UniMol代理低loss不等于唯一真实物理定律、量化实验或可合成分子；归一化gap−2不是实际负能隙。数学argmin不冒充已解全局最优，80候选次数对齐不等于等算力，专家反馈与有效初例保留。5seed阴影min/max不能称CI，h去exploitation更好、d并列、Mixtral局部更好均保留。正文/附录SR表归属错位、Improvement比例定义和应力代码括号疑点作为作者核对项，不静默改表/公式或执行代码。历史模型/费用不当当前推荐；探索利用与更多轮数的结果不推广为一般收敛定理。


## 持续审阅：第19篇SAC Regularization / Bitter Lesson（2024）

`icml-2024-c83edc89`23页含A统计、B架构/超参、C.1–C.8完整实证附录全读，19原页核验。此类诊断研究以三问题/动机分组、64跨类组合与逐阶边缘化为主，不需要伪造新模型或理论贡献；每小节Study description→Results→Takeaways可作可选组织。

可执行建议：代理指标与相关性不当因果机制，干预动机分类不等于实际作用互相独立；LN的DMC/MW差异、WD与reset交互及TOP例外保留。主SAC/proprio结论与附录DrQ像素失败同时说明，不照搬“仅proprio”遗漏补测；ReDO崩溃前取值和Dog先选top3延长4M需披露。NR只施加critic，RR2/16、最后10eval平均、IQM/SE与DrQ3seed口径不可混用。边缘化有效样本数、缺失超参和像素表只列4/所述6任务作为核对项，不补数据或声称所有任务/配置鲁棒。


## 持续审阅：第20篇Sundial（2025）

`icml-2025-56165d19`23页及A–F全部读完，18原页核验，包含19–23页所有示例图。以continuous/native与distribution/flexible两轴定位，先prelim归属，再模块/采样loss、数据、榜单、scaling、目标消融、test-time预算、adaptation与工程消融；限制放附录，须载入写作边界。

可执行建议：不用目标参数密度先验不等于没有高斯源；配置≤2880、F16/720不同训练horizon和univariate/covariate限制明确。GIFT MASE改善但CRPS与meanrank非第一，MSE消融/ECL概率及扩大ETTh1数据有反例，不写所有场景单调胜出。80%样本预测区间不是参数CI或覆盖认证，更多样本/CLT不能保证真实校准。zero-shot与聚合finetune分开，20/100samples与K50、设备差异/CPU近1秒不能合成无条件毫秒结论。主表与消融CRPS设置、median/Mean图例、噪声符号和缺失训练配置列核对；不静默改数字，也不将外部leaderboard链接当全矩阵已审计。


## 持续审阅：第21篇Diffusion Fisher（2025）

`icml-2025-557f816a`25页含A.1–A.10/B.1–B.4/C.1–C.7全读，21原页核验。先定义解析对象及外积结构，再分trace/operator两种访问接口、学习/复用终点近似、条件误差界与下游实验；OT基础与证明、成本及限制分层放附录。

可执行建议：DF-TM新增网络约24h/8V100训练成本不能藏于单次访问加速之后；DF-EA无需额外终点计算只适用已有adjoint流程。总体最优不代表训练收敛，有误差界不证明比VJP必然更准；DF-EA界不消失，t0奇异、有限二阶矩与有界数据条件分开。Hutchinson未实测、351NFE非一般必要下界；NLL/BPD不等同感知质量。二维单个非仿射例不证明全部non-affine非OT。归一化d、σ幂次、AB/AᵀB、score符号/α、正定/半正定及asym分母列作者核对，不静默改数学或认证其证明。


## 持续审阅：第22篇Neural Collapse meets Differential Privacy（2024）

`icml-2024-c7f1dbe3`27页及A–I附录全部读完，20原页核验。两个研究问题连接结构性0-1误分类界、不同扰动、缓解方案和三组实证；复杂度总表置前，相关工作整合prelim与分布假设，证明/扩展置附录。

可执行建议：末层binary零初始化单步NoisyGD与10epoch NoisySGD实证分开；β分布、阈值、额外坐标独立、perfect multiclass与总kρ预算紧邻结论。经验类均值β不是已知理想ETF距离或uniform上界，抽/复制768维不是骨干扩容。私有特征普通PCA消融不能证明全流程DP，k100无效及缺seed/误差条保留；k10和K−1文案需对齐。汇总表/定理样本量、Gaussian尾概率、投影界方向、旋转后L∞条件等只列作者核对，不静默改数学或独立认证隐私证明。未来随机初始化、非NC/all-layer训练与合成数据仍为开放方向。


## 持续审阅：第23篇Sparse Spectral Training（2025）

`icml-2025-152b386e`27页及A–L全部读完，21原页核验，含全部八图。方法以基线限制→谱层/更新→解释→显存交换→跨任务实验组织，附录补实际采样、reset/warmup、完整超参、剪枝、GaLore/DoRA/VeRA与资源权衡。

可执行建议：活跃参数比例、冻结因子、表中内存、真实训练时间、effective step与gap缩小分别报告。LLaMA1.3B的97.4%是相对选定基线的PPL差距缩小；OPT1.3B的SST9345.72MB/387.2h高于LoRA7133.24MB/324.8h，不能写更少显存且更快。真实采样混合均匀，优化器周期全reset不可漏；ReLoRA*与原版warmstart/99%prune不同。A梯度为0与范数相关不证明鞍点/方向等价/收敛。Airport r2、GaLore d256r32和zero-shot/Full反例保留。奇异值累计和非能量；剪枝非全比例更好，4434.3%效率比受近零分母放大。NaN=0标失败，±类型和消融不一数值列作者核对，不静默改表或补统计保证。


## 持续审阅：第24篇Likelihood Based Distribution Regression（2025）

`icml-2025-c7b2de50`27页及A–L/L.1/L.2全读，21原页核验。理论稿按general entropy/bias定理→稀疏/全连接特化→Wasserstein去卷积→learnable class→数值补充组织；related work前置，长anisotropic逼近和借用子网引理归属在附录。

可执行建议：观测P的Hellinger与无噪Q的W1、integrated X与uniform、几何d与结构t*分开；率指数的intrinsic依赖不消除D常数或p/α条件。sieveMLE近优化与CVAE实证的variational/optimization gap不能混同。FD3 SD输CKDE、较大注噪劣化、Ellipse有限n非单调保留，均值/SD的MSE不当完整分布保证。50samples SD非重训CI、Sinkhorn与像素距离口径需说明。多chart明确未保证gk regularity，可测生成存在不自动满足smoothness；噪声合成/分支、transport条件与归一化疑点只列作者核对，不静默改定理或认证其最优率。


## 持续审阅：第25篇Overestimation in LLM Evaluation（2025）

`icml-2025-c5f868c8`28页及A–G全读，24原页核验含全部21图/训练轨迹。以受控检测/剔除、同检查点分支和mode/timing/copies设计建立比较，结果用六个研究问题展开，Limitations在结论前；附录保留另一指标、完整曲线、绝对表及检测统计/例子。

可执行建议：去污染是按8gram/70%从test删例，不认证训练语料或语义无重叠；同checkpoint减少差异不等于多独立seed。最终BLEU点、瞬时分数、百分比与2.5×平均分开，MetricX方向相反。WMT23vs24差分与任务generalization并存，不将partial效果小写成普遍无影响。single init/order、≤8B/decoder-only/MT范围、Λ形语言与nearzero小变化保留。42/46条件数、实际语言集、删样本计数和缺CI的表注列作者核对；不补显著性或宣称所有Full干预每语言都优。53.6%复用预算不当实测硬件速度或能耗降低。


## 持续审阅：第26篇Maximal Local Disparity（2024）

`icml-2024-1dd24fde`30页含A–F全读，24原页核验。指标定义→性质/精确及近似计算→训练→tradeoff验证组织；RelatedWork在实验后，不固定章序。区分population估计、固定经验metric近似、a.e.平滑和minmax训练四层保证，不据前三层声称训练全局收敛。

二元score demographic parity范围，ε0变换不变性与ε>0性质分开；O(K²/ε)还需CDF排序，近0误差用绝对值。AP与其他fairness反例、验证95%ERM选择、五seedSTD保留。附录单调性方向/期望界、ΔDP定义、未列超参、代码np.max与×100单位问题附页码列作者核对，未执行代码或独立重证明，不把源文疑点改写成技能规则。


## 持续审阅：第27篇Categorical Diversity Designs（2024）

`icml-2024-eb3f98b9`30页含A–D.5全读，25原页核验。问题/三指标→精确integerprogram→greedy近似→成本→多任务证据组织；RelatedWork并入引言，理论论文摘要不强塞榜单数字。

写作规则：PR/CR/AC方向、单位与改善基准分别定义；GAC保证是d/2−AC改善比例，exact farthest-point子问题不等于整体精确解。完整域oracle与>20维子采样实现、整数规划成本/超时/内存、积分分位与实验STD分开。GAC在Load/LAB/Anon等反例保留。打印Prop3.2在d3/n4与70集合枚举所得PR*=CR*=1冲突；entropy界、PR/2、GP/SCIP符号、Q矩阵内存和伪码问题按页列源文查询，算例结果及复查脚本保留。不将有疑点公式当已验证技能知识；未复现作者SCIP/GP/GNN代码。


## 持续审阅：第28篇Tight and Fast Bounds（2025）

`icml-2025-573f6027`30页含A.1–A.5.4全读，23原页公式核验。纯理论无实验/图表：assumptions→lemma→theorem→conditioned comparisons，附录先proofsketch再分步展开；实验与matchinglowerbound为未来工作，不补实证或最优性。

写作建议：local结果R≤2Rhat+remainder，不改成excessrisk或统一generalizationgap速率；MacroAUC用r0少数正/负概率有效样本，仍sqrt速率。up to logarithms、B/M/γ等隐含依赖保留。独立权重合并范数的关键projection步骤、可学习映射/尺度条件、稀有标签/构对/取整查询附页码；n1/c4有限算例在指定解耦解释下与打印上界冲突，检查脚本与结果保留。本技能提取论证组织，不将尚待澄清的bounds当已核实通用知识。


## 持续审阅：第29篇Density Guidance（2025）

`icml-2025-70f7fda5`30页含A–N全读，27原页核验含22图、算法与listing。既有现象→有效条件/反例→连续精确控制→实用近似→随机扩展组织，实验随论证穿插，RelatedWork后置。

写作建议：原flow的density沿新路径数值不是新sampler边缘；SA失败反例和有限97%/100%观察分开；非零score、显式quantile初始化、隐式高维近似、stochastic忽略Rayleigh与finite-step误差须保留。>.99相关不是绝对准确，PNG细节与NIQE质量代理不同，noextraODEcost不含trace/校准/SDE。源文q方向、posterior权重/CDF逆、投影证明、重印方程及同名u代码问题附页码查询，不静默修复或宣称复现。无需将本篇组织强套固定实验章或全数值摘要。


## 持续审阅：第30篇Pessimism Meets Risk（2024）

`icml-2024-10e608ae`31页含A–G全部读完，24原页核验。误差传播先于算法，变换统一风险符号后，依次给基础pessimism、reference-advantage和variance-weighting界，正文proofsketch与附录分层证明对应。主文无实验章，但G末两页有小规模模拟，不能归为完全没有实验。

写作建议：coverage-free instance bound不等任意数据恢复最优；覆盖kappa、K阈值、已知特征/确定奖励、独立aux与variance floor必须随结论保留。R_beta仍含指数风险/时域依赖，比较上界不等实际策略全面胜出，offline下界仍open。模拟仅Alg1、正beta、20试验和80%CI，不扩展到加权算法、riskaverse或金融应用；最后H20曲线未到0。log前clip、负beta公式/值域、加权范数、D_ref命名及奖励状态名冲突按页列核对，未静默修复或独立认证证明。


## 持续审阅：第31篇Timer（2024）

`icml-2024-bb636262`31页含A–F全读，26原页核验，包含21图/18表。数据基础设施→S3表示→decoder/MSE→三任务适配→规模/架构/长度分析→zero-shot组织，逐任务Setups/Results与完整表对应；Limitations在附录中间，后续仍有科学结果。

写作建议：fewshot比例微调、one-series训练与ICL、任务专用checkpoint与通用架构分开；1B/28B为timepoints不是参数。44配置胜率与MSE减少比例、相对scratch与SOTA、参与数据不同的family rank不能混用。ECL/Weather/ETTm2负迁移、单数据scaling/zero-shot非单调与失败预测保留；异常top-alpha排序命中非概率confidence或在线延迟认证。S24的4G/12G冲突、阈值胜SOTA文案、数据域/频率/heldout及公式维数列核对。无概率预测/分类/显式跨变量适配，固定seed不补CI，未执行代码或审计数据泄漏。


## 持续审阅：第32篇GeoRCG（2025；2026-10-23完成）

`icml-2025-2a7aabb2`33页含A–E全读，28原页核验12图/12表及全部理论。表示/分子两阶段、并行训练/串行采样、额外正则/预训练、对称性/TV/条件reward与代理实证分别组织；末三页随机分子图已读，不靠caption计完整。

写作建议：表3相对EDM平均减少50.44%，对各属性所列最佳已有方法平均36.70%，两种分母的算术复核和脚本保留。1000→100是分子步数，不等包含表示1000步/CFG的总NFE或10×速度；实际耗时/额外训练、低温多样性代价、GEOM及unique反例、主表与调参最佳分别保留。MAE/MSE标签、split/预训练范围、指标分母/值、conditional/mixture正则和线性Gaussian条件、MI及打印公式疑点按页列核，不以泛用理论解释替代实测或独立证明认证。


## 持续审阅：第33篇Relative Error Fair Clustering（2025）

`icml-2025-70af507a`36页含A–D全部读完，27原页核验，包括完整assignment与一般z证明、唯一实验图。主定理先集中陈述，技术解释再按无约束→fair的依赖逆序教学，长证明分结构/重环/轻环/组合；无单独结论或限制章，不能当venue必需结构。

写作建议：强query、weakquery、coresetsize和最终求解时间分开，后处理可能指数/额外查询；fair kmedian和无fair一般z、持久噪声与metric/组定义条件保留。实验先50k/30k缩到2000，再100coreset，200SO的10%只对子样本；signedfairtree相对cost不等OPT或allC保证，不补CI/实际公平结果。Λ/rowmass、估权、真实环diam、capacity/浓缩和z幂次打印问题附页码查询，不静默修复或认证所有证明。


## 持续审阅：第34篇Manifold Perspective on GNN Generalization（2025）

`icml-2025-e3193968`38页及A–J.3.8全部读完，32原页核验包括全部证明、完整实验表/曲线和末页Roman拟合。理论解释型稿按两个对象prelim→采样/谱assumptions→node/graph两主定理→依赖/tradeoff→实证组织，不要求新架构或排行榜贡献；假设讨论/额外related/完整fit放附录。

写作建议：图-流形风险界与真实GCN划分、iid几何条件与citationgraph、固定权重synthetic和训练实验分开。更小CL给更小上界不等每任务accuracy提升，强正则反例、CiteSeer/Physics较弱loss相关和trainaccuracy<95%选域拟合保留；不把降序Laplacian谱当流形假设证明。polynomial/heat响应、零与重特征值、bandlimit/normalization、标签积分和非线性推导缺口按页列查询；原版??和图caption问题不复制为排版规范，未独立认证所有证明或复现实验。


## 持续审阅：第35篇Configurable Mirror Descent（2024）

icml-2024-56926c9a 全40页及A–F.9.2读完，30原页核验所有26图/9表/4算法和KKT推导。框架稿用六设计要求映射三贡献及RQ1–6；统一框架与具体15小游戏实验分开，不强制数字摘要或真实场景故事。补充材料公开pergame参数搜索、全部曲线/轨迹、失败例及时间内存。

保留preliminary/empirical：MCC团队BR仅100步近似，低估计gap不能认证精确均衡；SW提高不保证NE，magnet/不同Bregman非每任务最佳。迭代、2D候选/κ预算与wallclock不混写，资源表显示额外成本。KKT符号、epsilon投影/逆域、EU归一化和SW目标方向等原页查询保存，不默改结论或宣称重证算法。仅35篇先导观察，正式留出未用。


## 持续审阅：第36篇Learning Causal Dynamics Models in Object-Oriented Environments（2024）

icml-2024-c8c985f6 全42页及A–J读完，34原页核验全部11图/14表/3算法/证明和末页超参。表征先验→识别条件→共享结构/注意力→CMI/训练→预测/规划/迁移分别组织，正文保留关键条件，附录补非对称/噪声/弱点。未强制编号贡献或数值摘要。

建议严格分开population识别和有限样本神经检验、先验给定与像素表征学习、结构准确与预测/回报。StarCraft无GT、DZB回报/WalkerID反例、范围内47/53组合迁移与真正数量外推区分。设备/样本不等预算、weights bytes、噪声和统计量口径保留；C.13测量噪声追加一个Gaussian解析边界检查，记录原页证明/实现查询，不将其广义鲁棒性写成认证规则。仅36篇先导，正式留出未用。


## 持续审阅：第37篇Lean and Mean Adaptive Optimization（2025）

icml-2025-29bd8228 全46页及A–E读完，39原页核验完整科学图表/算法/证明。两瓶颈→两模块→分别理论保证→多尺度实验→配置消融形成清楚论证，首页图和量化摘要直接表达收益。可借鉴机制—保证—实测的映射，积极陈述已有证据的贡献。

理论针对累计SN及固定P的SGD-SM，组合SNSM由实验支撑；状态内存、峰值显存、每步耗时与达质量阈值耗时各有数据，清楚命名即可。原文公式/推导查询保存在研究记录，避免把逐项审计告警机械放入用户稿件。37篇先导，正式留出未用。
