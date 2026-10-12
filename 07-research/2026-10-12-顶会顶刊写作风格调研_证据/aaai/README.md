# AAAI 跨年度全文写作风格审阅（2026-10-12）

本批连续语义审阅两篇预先锁定的归纳样本，没有改动留出划分、共享计数、manifest、records、技能档案或旧评测档案。两篇都是AAA​​I Press正式发表版本；当前投稿模板与这些发表稿的排版必须分别管理。以下规则属于样本观察和编辑建议，不是官方要求，也不足以估计AAAI全会风格频率。

| 样本 | 主PDF全读 | 实际相关文本附录 | 视觉检查 | 计数建议 |
|---|---|---|---|---|
| aaai-2025-12 在线联盟学习 | 9页，含参考文献 | 作者补充A–F共7页全读；1949索引行无缺行 | 主文4、5、6、7页 | 语义全文完成；补充二进制归档与视觉核对未完成 |
| aaai-2024-22 KD-Club | 8页，含参考文献 | 正文无引用外部文本附录；官网仅列PDF，作者链接为代码 | 主文3、6、7页 | 语义全文完成 |

“完成”仅指实际全部文本的写作风格语义阅读。没有执行代码、复现实验、独立核验下界或验证全部证明正确性。补充二进制PDF下载先出现DNS失败，升级联网后仍解析超时；网页PDF解析内容可读，截图请求未返回可用的可见图片。因此保留真实文本完成状态，同时登记档案和视觉限制，不宣称补充公式排版已验收。

## 来源与版本

- 在线联盟学习：[AAAI官方发表页](https://ojs.aaai.org/index.php/AAAI/article/view/33498)，DOI 10.1609/aaai.v39i13.33498，2025-04-11，13709–13717页；[正文明确引用的作者补充](https://u.cs.biu.ac.il/~agmon/CohenAAAI25Sup.pdf)，同标题、作者及AAAI2025标识。
- KD-Club：[AAAI官方发表页](https://ojs.aaai.org/index.php/AAAI/article/view/30061)，DOI 10.1609/aaai.v38i18.30061，2024-03-24，20735–20742页；[作者代码仓库](https://github.com/JHL-HUST/KD-Club)为实现产物，不是遗漏的文本附录。

本地PDF SHA-256与锁定记录一致，正文标题、作者、会议信息与发表页相符。本次未重新下载并逐字节比较出版方PDF，不能把缓存身份核验表述成新的远端字节一致性证明。

## aaai-2025-12：Online Learning of Coalition Structures by Selfish Agents

[逐篇结构化证据](aaai-2025-12.json)。

| 维度 | 样本观察 | 证据位置与适用边界 |
|---|---|---|
| 读者与论证目标 | 先用学习小组例子降低联盟博弈进入门槛，再明确非合作、自利、Nash稳定性与共享数据平台。一般AI读者得到动机，理论读者得到评价目标。 | main pp1–2；main pp3–4。仅适用于该非合作在线博弈理论样本；不能据此规定所有AAAI都要生活类比。 |
| 标题和摘要 | 标题描述问题和行为主体，不带算法缩写；摘要按初始未知偏好→非合作建模→两类反馈→Nash regret与样本复杂度排列，结论是近似稳定及对数因素内最优。 | main p1。理论工作；没有实验数字不能虚构实证验证。 |
| 引言与贡献 | 例子→hedonic games与集中式已知偏好局限→在线未知偏好→Nash稳定目标→共享平台；Contributions段明确既有潜在博弈方法对联合策略空间线性而导致n的指数依赖。 | main pp1–2。保留ASHG对称偏好和共享平台；不是无通信分布式学习。 |
| 相关工作 | 独立Related Work占pp2–3，按博弈表示/稳定复杂度、在线与动态、PAC、匹配市场、潜在博弈学习分组，各组以信息/合作性/策略规模差异定位。 | main pp2–3。与KD-Club将相关工作融入引言形成结构反例；不可强制所有AAAI独立该节。 |
| 方法、算法与理论 | Preliminaries先给四步交互流程及反馈、regret定义；4.1 bandit与4.2 semi-bandit分开估计器，再由potential结构、ε-BRD和估计误差导向4.3 regret。主文有短证明/关键推导，附录补长证明。 | main pp3–7；supplement A–F pp1–7。该样本完全理论，无实验节；形式正确性未独立证明。 |
| 证据链 | 证据链是potential-game引理、ε-BRD步数、置信误差界、Nash-regret定理、online-to-batch推论，无经验实验。两个反馈的n依赖不同，不能用同一O(sqrt(T))隐藏全部参数。 | main p6 Theorem1/Lemma4；main p7 Theorem2 Eqs8–9 and Remark2；supplement C–F。最优性是作者陈述与引用支持，本次未独立核验下界及证明。 |
| 讨论与边界 | Conclusions and Future Work将扩展范围写成其他hedonic类、其他解概念、无共享平台的分布式算法、其他信息模型及任意对手下no-regret。 | main p7。本例没有单列Limitations；不得伪装该节为官方必需。 |
| 段落与措辞 | 解释段先给概念直觉再数学对象，段末常指出与已有技术差异；“optimal”紧邻feedback及log因素，使用显式“in contrast”“unlike”定位。 | main pp4–7。重复对照句可能增加篇幅；这是观察而非要求复刻连接词。 |
| 正文和附录 | 主文pp1–7保留模型、两段算法、核心结果和证明骨架，pp8–9参考文献；补充A–F共7页承接完整证明。补充B Theorem2与主文Theorem2不同，补充F Theorem3对应主文bandit regret分支。 | main p2 proof pointer；main p6 Remark1；main p7 appendix pointers；supplement pp1–7。补充编号不完全同步构成需要修复的反例，不是可学习的风格规则。 |
| 可执行建议 | 可复用的是对信息反馈与评价目标条件化的论证组织，以及正式结果/证明分配。 | main pp3–7。只有作者材料提供证明时才能填入保证；本例问题不是所有AAAI稿件的研究方向。 |

可执行建议：

- 贡献写成具体依赖改进及机制，而非将常规UCB本身说成新算法思想。
- 保留定理条件、反馈类型、ε和δ；建立引理→子程序→主定理依赖图；不静默修补原证明。
- 理论稿可以用完整条件化的定理和附录证据，不为适配AAAI虚构实验；同时保留n、T、δ依赖。
- 主附录用带语义名称的对应表核对，不只按定理编号自动链接；长证明移出正文时保留条件。

反例与需保留的编辑审计：

- **main p7 Eqs8–9**：bandit与semi-bandit的n依赖不同，即使均为sqrt(T)量级也不能润色为相同regret bound。 保留两个公式及反馈标签。
- **main p6 Theorem1 and proof; supplement A p1**：定理文字写for any game，而证明依赖potential函数；在介绍稿中不得扩张成任意博弈均保证收敛。 标记范围/假设待作者确认，不自行添加或删除条件；未开展独立定理正确性证明。
- **main p6 Remark1/main p7 Theorem2; supplement B and F**：同名Theorem2不是同一结论，supplement F标Theorem3承接主文Theorem2 bandit分支。 核对所有cross-reference，保留原版本差异并提出作者确认。
- **main p7 Eq10; supplement D Eq37 and C.2 Eq26**：ε-NS的近似保证在推导中写成等式，正文公式视觉确认确为等号；需要核对是否应写相应不等式。 记为实质性公式审计项，不能作为语言润色自动改写；补充glyph尚未视觉认证。
- **main p7 Conclusions/Future Work**：无共享平台的distributed algorithm只是未来工作。 不得在摘要和贡献中宣传已实现communication-free算法。

## aaai-2024-22：KD-Club: An Efficient Exact Algorithm with New Coloring-Based Upper Bound for the Maximum k-Defective Clique Problem

[逐篇结构化证据](aaai-2024-22.json)。

| 维度 | 样本观察 | 证据位置与适用边界 |
|---|---|---|
| 读者与论证目标 | 引言从clique的苛刻完整连接到允许缺边的dense subgraph，说明k-defective介于clique与(k+1)-plex，再聚焦精确求解和BnB上界。 | main pp1–2。组合搜索算法样本；不能把其问题背景作为AAAI通用主题。 |
| 标题和摘要 | 标题含算法名、exact、coloring-based upper bound和问题全名；摘要是任务→BnB瓶颈→忽略边信息→CLUB→两处集成→求解数/树/时间证据。 | main p1。标题形式与2025-12无算法名标题不同，无统一AAAI命名要求。 |
| 引言与贡献 | 相关方法嵌入引言，逐个指出MADEC+大k上界退化与KDBB忽略候选集内部缺边的机制；贡献段解释颜色独立集如何提供被忽略的缺边下界。 | main pp1–2。未使用独立贡献列表或Related Work节；不可强制固定模板。 |
| 相关工作 | Introduction比较RDS/branch-and-price及两种最近BnB；Preliminaries的Revisiting the Reduction Rules of KDBB细化直接baseline定义和Rules1–2。 | main pp1–2。本例直接baseline不是同硬件上原始论文数字；KDBB为本文作者重实现。 |
| 方法、算法与理论 | Lemma1短证明→CLUB定义Eq1→七顶点图例→ColorBound/Extract→KD-Club主框架→预处理Rules3–4→BnB。示例显式比较旧界7和新界5相对LB6的剪枝。 | main pp2–5；main p3 Fig1。理论的bound dominance不等于运行时间dominance；不声称独立证明所有算法正确性。 |
| 证据链 | 341图×6个k=2046实例；硬件、1800s cut-off和baseline来源在正文；Table1分图集/k汇总解出数，Table2给20代表实例的约简后规模、树大小与秒数，Fig2拆分预处理和BnB两个作用点。 | main pp5–7。Table2是选择的代表实例；两种k消融曲线不能代表全部2046实例的细粒度排名。 |
| 讨论与边界 | Conclusion总结更充分缺边检测及两处应用；future work提出推广其他relaxed clique/BnB问题。实验正文承认dense图预处理无约简、大k预处理作用小。 | main p6 dense cases；main p7 ablation discussion/conclusion。无独立Limitations节；不以此取消用户目标venue当前官方栏目要求。 |
| 段落与措辞 | 每小节先说明目的再给式或algorithm、最后解释对应line ranges；性能陈述常用significantly/excellent，而表格提供局部例外，因而润色应收紧量词。 | main pp3–7。主文没有统计检验或多seed报告；不能自动扩成statistically significant。 |
| 正文和附录 | 八页发表稿直接容纳全部关键Lemma证明、五个算法、图例、实验设置和结果；pp1–7正文，p8致谢与参考文献，无附录承接。 | main pp1–8。与2025-12外部七页proof supplement不同；篇幅分配取决于论文类型和官方规则。 |
| 可执行建议 | 可执行组织是机制缺口→界证明/例子→两阶段集成→分层比较/消融，并保留局部败例。 | main pp2–7。例外来自本样本表格，不是venue强制措辞；不得把精确求解改写成polynomial-time guarantee。 |

可执行建议：

- 写可定位的机制缺口，将CLUB的两个作用点与后续实验指标对应。
- 给核心界的推导和可手算例子，算法以输入输出和line references连接正文；保留精确性依据。
- 实验组织按总体覆盖→可解释性能机制→模块消融；保留图数据、k、cut-off、单位、NA超时语义与重实现来源。
- 对于短可展开的界推导可放正文，把算法和必要实验参数留在可阅读位置；勿为贴近Nature而删去technical proof。

反例与需保留的编辑审计：

- **main p6 Table1, DIMACS2 k=1**：KD-Club解31，MADEC+解32；不是每个group都最多。 保留most groups限定与cut-off。
- **main p6 Table1, SNAP/DIMACS10 k=5**：KD-Club解37，KDBB解38；该group发生反向差异。 保留局部失败条件，不将overall改写成uniform dominance。
- **main p7 Fig2(a), k=3; p6 general ablation prose**：实际曲线显示去预处理变体KD-Club^-Pre低于KDBB，另一变体KD-Club^-BnB也低于KDBB；“两variants优于KDBB”的笼统表述不能无条件采纳。 逐k读图，模块贡献具有相互作用；不能从完整算法更强推导每个消融变体都更强。
- **main p6 baseline KDBB paragraph**：KDBB是本文作者重实现，文中声称比原报道结果更好；不等于使用原作者实现或独立复现认证。 保留baseline provenance，比较对象可重复性待代码实验核验。
- **main p7 Table2 and main p6 dense-case explanation**：dense代表图约简前后规模相同，仍可通过BnB得到速度收益；不能将全部收益归因于预处理压缩。 分离预处理机制与搜索机制；代表实例并非完整样本。

## 两文比较与下一步适配

两文都以明确机制支撑贡献，但结构不能强行统一。在线联盟学习采用独立相关工作、较长模型定义、分反馈形式的定理和外部证明；KD-Club将相关工作置于引言及baseline复述，正文直接保留短界证明与手算图例，再给两阶段集成和分层实验。二者区别首先由论文类型解释，尚不能归因于2024与2025的年度风格变化。

写作技能可按稿件类型选择论证骨架；润色技能应优先检查主张中的量词、反馈/信息条件、指标量纲与附录引用。不能从发表事实推导所有原稿表述都正确：KD-Club表格的两处局部败例和k=3消融曲线应保留；在线联盟学习的主附录定理编号及范围/近似等式需要作者确认，不能当作可复刻的写作模板。

置信划分：逐篇结构及主文视觉确认高；样本内反例高；AAAI普遍频率未估计；官方要求本批未新增。正式留出验证没有执行。

本批原始网页文本和7张主文渲染位于本目录；sha256-manifest.json保存附件完整性记录。
