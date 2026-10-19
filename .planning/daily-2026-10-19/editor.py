"""Persist research judgments based only on source titles and abstracts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).parent
DAY = ROOT / '08-daily/2026-10-19'
D = json.loads((DAY / 'search_result.json').read_text())
P = json.loads((RUN / 'initial-screening.json').read_text())['top_papers']
order = [1, 6, 0, 5, 4, 7, 3, 8, 2, 9]
reasons = {
 1: '直接改变Muon的范数几何并以全局通道预算控制结构稀疏，与当前Lp/Muon设计最接近；线性因子化命题和1B模型剪枝提供可核验入口。先读oracle和预算，再评估剪枝收益是否值得额外求解成本。',
 2: '原始页面注明NeurIPS 2026 Workshop，按共享配置excluded_keywords排除正式推荐；保留预筛与内容初评供审计，不以会场标签推断质量。',
 6: 'LMO理论明确覆盖归一化、符号和精确谱方向，可能为当前几何优化器引入低成本梯度跟踪；q与预测误差的方差权衡可先在小实验验证，再检查有限NS实现是否受保证。',
 0: '超参数跨宽度迁移与免调参目标相关，但DP噪声及逐样本裁剪是专门条件；优先读SNR条件与DP-Muon谱缩放，不把DP结论直接移植到普通预训练。',
 5: '提供GD、SignGD、近似Adam、Muon、Shampoo及K-FAC的统一奇异模态比较，适合建立几何机制对照；显式平衡与对齐假设限制直接外推。',
 4: '混合Transformer-SSM中μP与AdamW归一化的角色可启发尺度迁移；宽度、深度和8倍迁移有具体核验点，但重点是架构与参数化而非新更新规则。',
 7: '局部/共享预条件混合给出异质性控制、反例及收敛率；Muon非线性与动量并不自动属于SPD理论，选读这一证据边界与实验协议。',
 3: '同时支持压缩通信和局部步的谱联邦方法可补充系统路线；摘要未给收敛阶、算法名、压缩器和通信量，先跟踪，不因一般效率措辞安排复现。',
 8: 'AFOR在线调节二阶矩遗忘系数，对非平稳自适应更新有局部启发；只有EEG跨被试证据，暂缺LLM尺度和状态成本信息，降级跟踪。',
 9: '原始页面注明GlobalSouthAI Workshop，按配置排除正式推荐；内容主要是词表扩展，AdamW衰减意外改动冻结行作为负面案例保留，不安排归档。'
}
E = {
 'assessment_scope': 'metadata_and_abstract',
 'confidence': '低；摘要初评，待全文核验',
 'overview': {
  'trend': '本次检索日期为2026-10-19，近期窗口2026-09-19至2026-10-19，补充窗口2025-10-19至2026-09-18，并非只收录今日首发。213篇唯一记录中90篇命中已有PDF、笔记或历史推荐，123篇未命中候选。预筛前十中两篇在原始页面注明Workshop，按配置排除正式推荐，保留8项阅读建议。仅据本次候选摘要，直接有用的线索是把Muon几何与可压缩性连接、用随机传输降低动量跟踪成本，以及在明确DP噪声模型下研究超参数跨宽度迁移；不能由此断言全领域已形成这一趋势。前三组成几何预算、oracle预算与尺度迁移组合，DP专门场景的结论不能直接推广到普通预训练。arXiv取相关性前200条，OpenReview按15配置关键词各取首页100条；Semantic Scholar两次429，高影响力补充未取得结果。',
  'hotspots': [
   {'title': '范数几何与结构预算', 'text': 'BMuon以跨块共享通道预算连接谱更新与结构稀疏；需区分剪枝后的收益、未剪枝性能和额外训练开销。'},
   {'title': '超参数的跨宽度可迁移性', 'text': 'μP候选分别讨论DP噪声和混合SSM架构下的学习率迁移；优化器跨阶段切换项因Workshop配置排除，只保留审计记录。'},
   {'title': '梯度跟踪的成本与方差', 'text': '随机动量传输用概率q节省校正调用，额外方差由预测误差决定；能否覆盖实际有限NS需要另行核验。'}
  ]
 },
 'domain_summaries': {'大模型优化器设计': {
  'summary': '前十逐篇读摘要并核对原始页面后，以BMuon、随机动量传输和DP-μP为前三；奇异模态动力学、混合架构LR迁移与FedMIX-P定点选读，联邦压缩和EEG遗忘控制跟踪。跨阶段切换及词表扩展两篇Workshop按配置排除，正式阅读建议8项。主题计数是关键词命中量，含Muon粒子物理及宽泛训练误匹配，不等于直接相关研究量。',
  'subfields': {
   '符号与范数几何优化': 'BMuon从范数oracle约束通道预算；随机动量传输覆盖精确谱LMO；模态动力学用于对照几何的作用。含Muon同词异义记录，应结合附录状态排除。',
   '免调参与梯度裁剪': 'DP-μP讨论裁剪与加噪共同影响跨宽度学习率缩放，属于有明确噪声模型的迁移方案，不能据此称普通训练完全免调参。',
   '自适应与状态内存': 'AdamW/Muon跨阶段实证、混合架构的逐坐标归一化、FedMIX-P共享预条件及AFOR遗忘系数可提供诊断线索；前三未声称新的状态内存压缩收益。',
   '大模型训练与优化器设计': '切换代价需要将预训练、持续预训练和SFT分别比较；词表扩展的冻结行衰减问题属于实现负面案例。',
   '未细分': '部分命中只含宽泛训练词或学科分类，不能直接算作优化器设计证据；先检查更新规则与当前问题的联系。'
  }
 }},
 'top_papers': {},
 'semantic_reranking': []
}
A = E['top_papers']
A[P[0]['title']] = {
 'title_zh': '迈向差分隐私优化中的超参数迁移',
 'abstract_zh': '缩放规律与超参数调优仍是大规模模型训练的核心挑战，尤其是在差分隐私（DP）约束下，噪声注入和逐样本梯度裁剪会使优化进一步复杂化。虽然最大更新参数化（$μ$P）能够在标准训练中实现零样本超参数迁移，但我们表明，由于高维噪声与低秩梯度具有不同的谱性质，它在DP约束下会失效。在本工作中，我们提出DP-$μ$P，这是一个有理论依据的框架，将$μ$P扩展到隐私保护设置。通过分析梯度裁剪、噪声注入与特征学习之间的相互作用，我们推导了一个决定逐层学习率缩放的信噪比（SNR）条件。将它与我们提出的谱缩放表述结合后，我们为DP-SGD、DP-Adam和DP-Muon优化器给出具体的缩放规则。在实证方面，我们表明，DP-$μ$P可在多种架构（MLP、ViT、GPT-2）上实现有效的跨模型宽度零样本学习率迁移，在显著减少计算成本的同时，达到或超过逐个调优基线的性能。',
 'summary': '作者从DP高维噪声与低秩梯度的谱差异出发，为DP-SGD/Adam/Muon构建跨宽度学习率缩放规则。',
 'contributions': ['作者指出普通μP在DP裁剪和加噪下失效，并提出DP-μP扩展。', '以特征学习和逐层SNR条件推导谱缩放规则，覆盖三类DP优化器。'],
 'method': '分析裁剪、加噪和特征学习如何共同影响逐层信噪比，再结合谱缩放构建学习率迁移规则；优先读噪声/梯度尺度假设、裁剪阈值和宽度参数化。',
 'result': '摘要报告MLP、ViT和GPT-2中跨宽度零样本学习率迁移达到或超过独立调优基线，同时降低计算成本；未给模型规模、隐私预算、改善幅度或成本数字。',
 'selection_reason': reasons[0],
 'evidence_limit': '初步判断，仅摘要。需全文核验DP噪声分布、隐私预算与裁剪设置、SNR条件是否可测、DP-Muon谱缩放是否使用精确正交化，以及迁移宽度和调参预算。零样本学习率迁移不等于所有超参数无需选择，也不是对普通非DP预训练的保证。',
 'reading_decision': '精读'
}
A[P[1]['title']] = {
 'title_zh': 'BMuon：为Muon的谱几何分配预算以实现结构化可压缩性',
 'abstract_zh': 'Muon已成为神经网络训练中表现强劲的优化器，并可解释为谱范数几何下的最速下降。然而，我们发现，Muon训练的Transformer在训练后剪去其MLP隐藏通道时，性能可能急剧下降。为兼顾Muon的优化性能与结构化可压缩性，我们提出Budget Muon（BMuon），修改Muon的谱几何，使不同块中的隐藏通道竞争一个共享的分配预算。所得oracle恰好对应于一族范数下的归一化最速下降，这族范数在Muon几何与结构稀疏几何之间插值。我们控制稀疏性的主要手段是一个可解释的全局通道分配预算，而非依赖具体设置的稀疏惩罚系数。对于线性因子化模型，我们表明，Muon允许具有冗余隐藏宽度的最小代价精确表示；而在BMuon的oracle几何和足够紧缺的预算下，每一个最小代价表示都具有最小宽度。为用于实际训练，我们为精确分配问题推导了一个下界化–最大化（minorization-maximization）代理。在ImageNet-1K上的ViT-S/16中，采用渐进式结构化MLP剪枝时，BMuon优于Muon，且其学习到的分配比权重幅值提供了更有效的剪枝准则。在FineWeb上训练的一个1B参数LLaMA风格模型中，采用训练后MLP通道剪枝时，BMuon也优于Muon。',
 'summary': '作者把Muon更新几何与全局通道预算连接起来，以训练时分配改善后续结构化MLP剪枝。',
 'contributions': ['作者构造在谱几何与结构稀疏几何之间插值的范数族及其归一化最速下降oracle。', '作者在线性因子化中比较最小代价表示的隐藏宽度，并提出实际分配问题的minorization-maximization代理。'],
 'method': '用跨块共享通道预算约束更新方向，精确分配oracle作为几何定义，再用代理求解落地。优先读预算如何归一化、近似求解是否保持理论属性以及剪枝准则。',
 'result': '摘要报告ViT-S/16在ImageNet-1K渐进结构剪枝下优于Muon、分配评分优于权重幅值，并在FineWeb训练的1B LLaMA风格模型训练后MLP剪枝下优于Muon；未给具体提升、未剪枝代价及训练时长。',
 'selection_reason': reasons[1],
 'evidence_limit': '初步判断，仅摘要。需全文核验“预算足够紧缺”的阈值、最小代价表示的假设、oracle与实际代理的差距，预算搜索成本、相同剪枝率/训练预算/超参下的公平比较，以及未剪枝性能是否下降。该文不是免调参证明。',
 'reading_decision': '精读'
}
A[P[2]['title']] = {
 'title_zh': 'LLM训练阶段之间切换优化器的短暂代价',
 'abstract_zh': '近期工作提示，在预训练和后续训练阶段之间切换优化器会带来性能代价，但尚缺乏对这一现象的系统研究。我们研究AdamW与Muon优化器之间的不匹配：从头预训练1.2B Llama-3模型，同时改变两种优化器各自执行更新的比例；在持续预训练和监督微调（SFT）期间，也类似地在两种优化器之间进行插值。出乎意料的是，我们几乎没有发现具有实际意义的优化器不匹配惩罚的证据：最终性能主要可由预训练优化器和后训练优化器各自的独立影响解释，且随着SFT期间Muon更新比例增加，下游性能总体上有所改善。虽然匹配预训练优化器可以减少SFT期间的遗忘，但这并未转化为最终基准性能的提高。我们进一步在Llama和Qwen家族、参数规模最高达14B的开放权重模型上评估AdamW和Muon的通用任务SFT；尽管这些模型用AdamW预训练，Muon依然可与AdamW竞争。综合而言，我们的结果提示，后训练优化器的选择可以在很大程度上独立于预训练所用优化器。',
 'summary': '在作者覆盖的Llama/Qwen训练协议中，AdamW与Muon的跨阶段不匹配对最终表现的独立惩罚证据有限。',
 'contributions': ['作者以1.2B Llama-3从零预训练和各阶段更新比例插值，分离预训练与后训练优化器的影响。', '区分SFT遗忘与最终基准表现，并把通用任务SFT比较扩展到最高14B的Llama/Qwen开放模型。'],
 'method': '改变AdamW与Muon更新比例，在预训练、持续预训练和SFT分别比较，再在开放权重模型上做SFT评估。优先查状态初始化、学习率/调参预算和阶段插值的具体定义。',
 'result': '摘要报告不匹配惩罚证据很少，SFT中Muon更新占比提高通常改善下游表现；匹配可减少遗忘但不改善最终基准，最高14B模型上Muon保持竞争力。摘要没有数值效应量或不确定性。',
 'selection_reason': reasons[2],
 'evidence_limit': '初步判断，依据摘要。少量不匹配证据不等于所有阶段和任务都无代价；需核验基准集合、种子、调参公平性、动量/二阶状态的切换处理及短暂损失波动。AdamW↔Muon的SFT结论不直接验证前日报噪声主导时Muon→SGD的局部理论。',
 'reading_decision': '精读'
}
A[P[6]['title']] = {
 'title_zh': '利用低成本预测器的随机化动量传输',
 'abstract_zh': r'STORM和二阶动量等动量方法通过向动量加入传输校正来跟踪梯度，代价是每次迭代多调用一次oracle。我们提出随机化动量传输，以概率$q$计算校正，否则使用低成本预测，并通过重要性加权保留条件均值，同时向方差中加入$(q^{-1}-1)$倍的预测损失。在传输偏差、二阶矩和残差控制假设下，对每个固定的$q>0$，包括归一化方向、符号方向和精确谱（Muon）方向在内的线性最小化oracle（LMO）更新，在采用单梯度初始化时，均保留$O(T^{-1/3})$的平稳性界。对于未归一化的STORM，当且仅当梯度差分噪声模量超过光滑性常数时，某个$q<1$才能降低我们所给充分梯度预算的首项系数，降低幅度至多约为$10\%$。一种在线对角预测器可得到从遗憾界到收敛性的保证。在oracle成本近似相同时，仅查询四分之一的校正在MNIST1D上可将最终训练损失降低$18$至$43\%$，并将Hessian–向量传输在CIFAR-10上的测试准确率提高$1.5$个百分点。',
 'summary': '作者用概率校正加便宜预测保留梯度传输均值，并明确量化节省oracle调用所增加的方差。',
 'contributions': ['作者推导随机传输重要性加权后的方差增量，并在特定假设下把O(T^{-1/3})界扩展到归一化、符号及精确谱LMO。', '给出未归一化STORM预算系数改善的充要条件、约10%上限和在线对角预测器的遗憾到收敛保证。'],
 'method': '以q稀疏调用真实校正，以预测器补足其余步骤，通过重要性加权无偏化；检查q、预测误差和校正成本的联合权衡。先用小规模消融比较固定oracle预算。',
 'result': '摘要报告单梯度初始化、固定q>0下O(T^{-1/3})，以及近似同oracle成本下MNIST1D损失下降18–43%、CIFAR-10测试准确率提高1.5个百分点；未报告LLM预训练。',
 'selection_reason': reasons[6],
 'evidence_limit': '初步判断，仅摘要。需全文核验残差控制条件能否实现、q趋小时常数如何恶化、预测器成本、gradient预算与墙钟时间的差异。理论中的精确谱oracle不自动覆盖有限Newton–Schulz；STORM预算首项约10%的改善与实验损失改善是不同指标。',
 'reading_decision': '精读'
}
rest = {
 5: ('在平衡与对齐假设下比较GD、SignGD、近似Adam、Muon、Shampoo和K-FAC的模态出生/饱和/衰减；选读而非直接外推Transformer。', '选读'),
 4: ('实用混合Transformer-SSM中原始μP加AdamW报告宽度256–2048、深度4–32及8倍LR迁移；诊断不通过而迁移有效值得核验。', '选读'),
 7: ('FedMIX-P以共享/局部预条件混合降低算子失配，给出O(R^{-1/2})、固定混合反例及实验；SPD理论不自动涵盖完整Muon。', '选读'),
 3: ('谱联邦方法同时结合压缩和局部步并声称去掉有界异质性假设；摘要缺收敛阶、压缩量及语言任务证据，先跟踪。', '跟踪'),
 8: ('AFOR用RASS/AFC在线调节二阶矩遗忘，三项EEG跨被试基准优于Adam；LLM适用性和额外状态代价未知，跟踪。', '跟踪'),
 9: ('【配置排除：Workshop】词表扩展报告token/词下降38.6%及Indic/English权衡；仅保留原预筛记录，不属正式推荐。', '跟踪')
}
for idx, (summary, decision) in rest.items():
 A[P[idx]['title']] = {'summary': summary, 'reading_decision': decision, 'selection_reason': reasons[idx], 'evidence_limit': '仅摘要初评，待全文核验。'}
for rank, idx in enumerate(order, 1):
 p = P[idx]
 p['candidate_screening_rank'] = idx + 1
 p['semantic_rank'] = rank
 p['formal_recommendation'] = idx not in [2, 9]
 if not p['formal_recommendation']:
  p['editorial_status'] = '配置排除（workshop）'
  A[p['title']]['reading_decision'] = '跟踪'
  if not A[p['title']]['summary'].startswith('【配置排除'):
   A[p['title']]['summary'] = '【配置排除：Workshop】' + A[p['title']]['summary'] + '仅保留预筛记录，不属正式推荐。'
 E['semantic_reranking'].append({'title': p['title'], 'screening_rank': p['screening_rank'], 'candidate_screening_rank': idx + 1, 'semantic_rank': rank, 'reason': reasons[idx], 'reading_decision': A[p['title']]['reading_decision'], 'formal_recommendation': p['formal_recommendation']})
D['top_papers'] = [P[idx] for idx in order]
D['formal_recommendation_count'] = 8
D['search_limitations'] = [
 'arXiv相关性前200条，非窗口内全部论文。',
 'OpenReview按15配置关键词分别取首页100条，多数存在截断，覆盖详见openreview-coverage.json。',
 'Semantic Scholar过去一年补充两次429，未取得高影响力与引用数据，不代表无相关论文。',
 '按现有真实PowerStep PDF核验过的LaTeX ell_p标题别名去重；当天输出不作为历史。原始HTTP记录离线重放，未修改技能评分。',
 'screening_rank保留脚本全量预筛名次；candidate_screening_rank为历史去重后的原前十名次；semantic_rank为人工阅读次序。',
 'Muon粒子物理误命中保留全量记录但标记不推荐；大方向和细分计数为关键词匹配，不等于直接相关论文数。',
 '两篇原始页面注明Workshop，按配置排除正式推荐；保留原预筛信息，不从会场标签推断质量。',
 '所有研究结论限于题目与摘要；PDF仅核验身份和可读性，未进行全文评审。'
]
(DAY / 'daily-editorial.json').write_text(json.dumps(E, ensure_ascii=False, indent=2) + '\n')
(DAY / 'search_result.json').write_text(json.dumps(D, ensure_ascii=False, indent=2) + '\n')
print('Saved ten semantic ranks, selection reasons and three complete translations')
