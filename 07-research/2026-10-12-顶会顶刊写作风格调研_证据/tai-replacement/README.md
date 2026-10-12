# TAI 2024 替代候选资格证据

目前是**已核验候选，待主任务正式登记**。仅建议修复归纳槽 `tai-2024-03` 误入的 Front Cover；原始锁定清单与留出划分仍保留，不把候选附件下载或阅读算作语料累计完成。

推荐 **Efficient Structure Slimming for Spiking Neural Networks**，Yaxin Li、Xuanye Fang、Yuyuan Gao、Dongdong Zhou、Jiangrong Shen、Jian K. Liu、Gang Pan、Qi Xu。出版社存入的 [Crossref DOI 数据](https://api.crossref.org/works/10.1109%2FTAI.2024.3352533)确认正式刊出是 2024 年 8 月、5(8):3823–3831；[Birmingham 一手机构记录](https://research.birmingham.ac.uk/en/publications/efficient-structure-slimming-for-spiking-neural-networks/)和实际 PDF 封面共同确认是**已接收作者稿**，不是仅凭 OpenAlex 标签确认版本。

候选完整材料：

- [资格、确定性筛选路径及限制](replacement-candidate.json)。原框先排序，再依次排除编辑/封面/目录/综述并优先已确认 accepted/published edition；SS-TrBoosting 的正式发表身份已确认，但实际 arXiv edition 暂不能独立确认已接收版本，仍保留为候选。
- [原归档来源中未入既有清单的 2024 定位框](original-frame-unused-2024.json)，定位元数据不能代替全文版本核验。
- [已接收稿](../../../.claude/skills/paper-writing-shared/tmp/source-materials/07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai-replacement/snn-slimming-accepted-author.pdf)及[文本](../../../.claude/skills/paper-writing-shared/tmp/source-materials/07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai-replacement/snn-slimming-accepted-author.txt)：PDF 10 页，其中第 1 页是机构封面，第 2–10 页为论文 1–9 页；论文没有引用另行附录或补充文本。
- [连续全文十维观察与证据反例](candidate-semantic-review.json)，[关键页渲染](renders/pdf-p8.png)，[附件哈希](candidate-files-sha256.json)。不包含独立证明或实验复现。

先前 OpenAlex 将机构 locator 标为 submittedVersion，与实际封面及一手机构页面的已接收身份不符。Crossref 正式刊期是 2024 年 8 月；机构 early online 为 2024-01-11，两者含义不同且不冲突。当前 PDF 的机构封面带下载日期，哈希只承诺本次保存附件，未声称与出版社最终 PDF 字节一致。

写作证据中特别保留了反例：摘要“约 10% 参数留存且损失不到 1%”需限定到 CIFAR10 配置；能耗估计不能写为实测部署；原文 4.82 times 与表中 4.82% 不能当作相同单位。最终规范化建议取自这些可定位证据，不能把任何单篇语言错误当作 TAI 风格规则。

机构附件限本地私人/非商业调研，不发布或向外重分发全文。出版社 Xplore 本次无法访问，未确认未列出的资产或 publisher/current edition 等价。
