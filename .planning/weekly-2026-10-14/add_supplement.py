from pathlib import Path
import json
R=Path(__file__).parent;ROOT=R.resolve().parents[1]
section=r'''## 附录C：10月16日新增日报的补充证据

收尾校验期间工作区新增了[10月16日检索结果](<../08-daily/2026-10-16/search_result.json>)，因此补入该日信息，不改变10月8—14日主窗口的统计。该日检索265篇唯一论文、97篇历史命中、168篇候选；前十保留一篇Workshop误入记录，正式阅读建议为9篇。该次计数与主窗口358个原始题目字符串存在重叠，**不能相加为623篇**。三篇PDF已落盘，但不计入主窗口16篇。补充快照见[记录](<../.planning/weekly-2026-10-14/supplement-2026-10-16.json>)。

| 补充归档论文 | 一手来源与本次层级 | 对本报告决策的影响 |
|---|---|---|
| Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon | [官方摘要](https://arxiv.org/abs/2610.19844v1)；[本地PDF](<../01-raw/2026-10/Memory_Efficient_Subspace_Optimization_with_Residual_Correction_From_Momentum_SGD_to_Muon.pdf>)；未审计证明/实验 | “低秩状态＋残差＋周期换基”已有PMRC/SMRC覆盖，应优先核验其boundary correction，继续暂缓宽泛重复方案 |
| From Parallel to Sequential: Small-Batch Noise Restores Simplicity Bias of Muon | [原文§3–4、§6、Appendix G](https://arxiv.org/html/2610.16868v1)定点核验；[本地PDF](<../01-raw/2026-10/From_Parallel_to_Sequential_Small_Batch_Noise_Restores_Simplicity_Bias_of_Muon.pdf>) | 增加了噪声平滑谱算子、改变学习顺序的机制；更应把batch与谱控制共同检验，不能只最大化更新秩 |
| How Inner Optimizers Shape Replica Scaling | [官方摘要](https://arxiv.org/abs/2610.19064v1)；[本地PDF](<../01-raw/2026-10/How_Inner_Optimizers_Shape_Replica_Scaling.pdf>)；未核验完整协议 | 伪梯度协方差、噪声尺度和内部白化成为分布式近邻；当前没有多副本实测，不另开昂贵路线 |

**小batch论文的具体边界。** Theorem 4.2的连续动力学建立在白化数据、奇异向量对齐、平衡初始化及特定噪声结构上；无动量spectral flow与实际LLM实现不同。Appendix G语言实验使用162M模型、3.2B tokens、seq512和D-Muon，比较B=256/512/1024及LR缩放；模型有效秩变化不是所有规模上的泛化保证。该结果为方向C提供新的直接近邻，也限制了“简单降低噪声或把所有弱方向放大必定更好”的解释。本文不声称已经完成其完整理论审计。

**新增的判别实验。** 在C原有两带gain×batch分支中，加入归一化梯度/SGDM与Muon的对照，匹配实际更新幅度、token、数据次序和LR搜索预算；观察子空间重建或层权重的有效秩随时间变化，并以独立验证loss检验。先在低秩矩阵重建复核顺序学习，再决定是否扩大LLM试验；不能仅因有效秩较低就判断更好。batch改变的是噪声、步数和动量时间尺度的组合，原研究计划仍要求跨batch公平调优。

10月16日前十还出现了配对时间/数据/调参预算基准等候选；本次仅由日报元数据定位，部分远程页面未成功取得，不引用其具体数值，也不据此新增算法排名。它们应进入下一轮全文核验，而不是被当成已复现证据。
'''
replacements={
'supplement_checked_until: 2026-10-16':'supplement_checked_until: 2026-10-16\nsupplement_new_pdfs: 3',
'10月15—16日没有新增本地日报或新论文提交记录，不能据此判断外部没有新论文。':'补查初期未见15—16日新增；最终校验时工作区出现10月16日日报与3篇新归档，已单列附录C，不与主窗口统计混合。',
'本周归档16篇原文，新增解析和精读各1篇':'主窗口归档16篇原文，新增解析和精读各1篇',
'下阶段只选一条路线进入闭环训练，先在70M—350M排错和证伪，再过1B门槛，最后决定是否投入7B。':'下阶段只选一条路线进入闭环训练，先在70M—350M排错和证伪，再过1B门槛，最后决定是否投入7B。收尾新增的10月16日日报另补3篇原文，其中小batch噪声论文进一步要求将噪声与谱增益联合验证，见附录C。',
'| 翻译资产 | 新精读论文有双栏对比PDF |':'| 10月16日补充 | 1天日报、3篇新归档 | 独立统计，不与主窗口重复相加 |\n| 翻译资产 | 新精读论文有双栏对比PDF |',
'10月10日日报尚未提交，但属于本地检索记录；已提交原文数量另按Git统计。':'10月10日日报属于本地检索记录；已提交原文数量另按Git统计。10月16日补充记录不纳入主窗口358个标题字符串。',
'现有算法是否已支持拟议控制目标':'现有算法是否已支持拟议控制目标',
'| 跨batch谱gain控制 | 高风险跟踪 |':'| Small-Batch Noise原文 | 精读 | 核查噪声平滑、实际动量与seq512实验的边界 | 半天到1天 | 能否在相同预算下区分噪声、LR与谱增益 |\n| 跨batch谱gain控制 | 高风险跟踪 |',
'### 暂缓的路线':'方向C还需对照补充的[Small-Batch Noise原文](https://arxiv.org/html/2610.16868v1)，区分谱增益与噪声诱导的学习顺序；附录C给出条件与判别实验。\n\n### 暂缓的路线',
'这些工作不追加进“本周16篇本地归档”的数量。':'这些工作不追加进“主窗口16篇本地归档”的数量。PMRC在10月16日又有本地归档，见附录C，不能重复计数。',
}
for file in [ROOT/'07-research/2026-10-12-第4周周报.md',R/'write_report.py']:
 s=file.read_text()
 for old,new in replacements.items():
  if old!=new:assert old in s,old[:40];s=s.replace(old,new,1)
 if file.suffix=='.md':s+='\n'+section
 else:
  s=s.replace("OUT.write_text(BODY,encoding='utf-8')",'BODY += '+repr('\n'+section)+"\nOUT.write_text(BODY,encoding='utf-8')")
 file.write_text(s)
