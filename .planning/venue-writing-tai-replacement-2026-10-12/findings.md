# 结论

推荐候选 Efficient Structure Slimming for Spiking Neural Networks，DOI 10.1109/TAI.2024.3352533。Crossref 出版社存入记录确认 2024 年 8 月正式刊出，5(8):3823–3831；Birmingham 记录和实际机构封面确认这是已接收作者稿，匹配 8 位作者和 DOI，实际 PDF 共 10 页，论文为后 9 页。

最先遇到的原创候选 SS-TrBoosting 正式 2024 刊出无疑，但 arXiv 实际稿没有接收/发表身份抬头，不能仅凭 related DOI 断言此 edition 是 accepted。保留待核验而非假排除，之后按原抽样 accepted/published-preferred 原则提出最先可独立确认的候选。

机构 PDF 解析和渲染会发出 xref/path 重建警告，但均正常退出，全部十页有文本，关键标题和双栏对照表正常渲染。没有独立代码复现或证明验证。

论文所说约 10% 留存且损失 <1% 必须限定到 CIFAR10 的 channel+fine-tune 配置；CIFAR100 接近留存率并不满足。能耗为操作数乘引用每操作耗能的估计，不是本论文实测部署。原文 4.82 times 与表的 4.82% 有单位冲突，不应作为可学习的写法。
