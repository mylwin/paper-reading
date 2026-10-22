import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / '08-daily/2026-10-22'
data = json.loads((DAY / 'search_result.json').read_text())
top = data['top_papers']

editorial = {
    'assessment_scope': 'metadata_and_abstract',
    'confidence': '低；仅题目与摘要初评，待全文核验',
    'overview': {
        'trend': '本轮窗口为2026-09-22至2026-10-22，合并去重202篇（arXiv 183、OpenReview 19），107篇命中已有资产或历史推荐，95篇未命中。逐篇审阅预筛前十后，6篇有直接或条件性阅读价值，4篇因研究对象不符作语义排除；前十均保留用于审计，不从第11名补选。前三选择HyMUON、CAST和AlgoPerf，分别研究动量时间尺度、历史自适应谱裁剪和跨任务免调参评测。它们首次提交于10月20日、14日和9日，今日新入库不等于今日首发。该样本含Muon粒子物理同名误命中，不能据数量推断全领域趋势。Semantic Scholar年度窗口两次429，OpenReview的AdamW查询429，覆盖均有缺口。',
        'hotspots': [
            {'title': '历史信息如何进入更新', 'text': 'HyM将随迭代增长的线性加权历史与短期EMA结合，HyMUON再正交化该方向；核心增量应与单一EMA的调参预算、额外状态和算子成本一起核验。'},
            {'title': '裁剪尺度与阈值共同自适应', 'text': 'CAST从AdamW前一时刻的二阶矩确定行列尺度，并从裁剪输出的谱范数历史设置矩阵阈值；7.2%是达到指定训练损失所需步数的减少，尚不是墙钟加速。'},
            {'title': '免调参需要跨任务证据', 'text': 'AlgoPerf用固定硬件上的达标墙钟时间和跨任务单一配置评测优化器；它为比较PowerStep、Muon变体与Schedule-Free AdamW提供协议依据，基准排名不应直接外推LLM预训练。'}
        ]
    },
    'domain_summaries': {
        '大模型优化器设计': {
            'summary': '优先阅读可落到更新规则的HyMUON与CAST，并用AlgoPerf检查调参与墙钟预算是否公平。初始化Hessian结构论文提供正交化设计的局部理论线索，不能替代训练中证据；算术简化和OCL专用优化器适合定点选读。自动匹配的总量包含宽泛AdamW和Muon同名记录，不是直接相关论文数量。',
            'subfields': {
                '符号与范数几何优化': 'HyMUON更改正交化前的历史方向；初始化分析解释梯度自外积的对角占优。前者有训练实证线索，后者只覆盖初始化与渐近维度条件。',
                '免调参与梯度裁剪': 'CAST自适应行列尺度和矩阵阈值，AlgoPerf强调跨任务单一配置。自适应规则仍可能含超参数，应分别核验设计常数与任务调参成本。',
                '自适应与状态内存': 'HyM新增长期历史可能带来状态成本，CAST复用AdamW二阶矩但仍有谱估计开销。前三没有直接给出总显存节省证据；算术简化不等于状态压缩。',
                '大模型训练与优化器设计': 'HyMUON报告LLaMA-1.3B/C4损失比较，CAST报告19.08B总参数MoE、83.89B token训练。AlgoPerf九类工作负载结果提供通用评测参考，不代表上述LLM设置下的胜负。',
                '未细分': '仅分类或宽泛词命中的记录保留在附录追溯。结构修复、主动偏好采样、扩散隐私与RF芯片代理不计正式优化器阅读建议。'
            }
        }
    },
    'top_papers': {},
}

editorial['top_papers'][top[0]['title']] = {
    'title_zh': '用于深度学习模型优化的混合动量方法',
    'abstract_zh': '动量是深度学习中随机优化的核心，包括大语言模型的训练。经典动量方法通常依赖一种快速的指数移动平均（EMA）机制，它按固定时间尺度衰减过去梯度的权重，并可能低估仍然包含有效信息的较早梯度。这使保留长期历史与适应近期变化之间产生了权衡。\n\n  我们提出混合动量（Hybrid Momentum，HyM），将过去梯度的线性加权平均与按几何方式衰减的快速动量结合。前者赋予梯度与其迭代序号成正比的权重，并保留跨度不断增长的历史，而后者强调近期梯度。这种混合构造具有模块化特性，可以与自适应预条件和其他更新变换结合。\n\n  具体而言，我们提出HyMUON，对混合方向施加Newton–Schulz正交化。示例与受控实验展示了结合这两种时间尺度下梯度信息的益处。在图像分类、机器翻译和语言建模中，HyM与HyMUON达到或超过强基线的表现。HyMUON在IWSLT14上优于Muon，在GPT-2 124M于OpenWebText上的100k步和600k步两种设置中也优于Muon。\n\n  对于在C4上的LLaMA-1.3B预训练，HyMUON取得2.777的最终验证损失，而Muon为2.903。',
    'summary': '将长期线性加权历史与短期EMA混合，再施加Muon式正交化，直接检验历史方向设计能否改善LLM训练。',
    'contributions': [
        '作者提出可组合的HyM方向规则，分开长期历史保留与近期适应，并构造正交化变体HyMUON。',
        '摘要给出跨图像、翻译与语言建模的比较线索，包括LLaMA-1.3B/C4上的明确损失数值。'
    ],
    'method': '维护按迭代序号线性加权的历史平均与几何衰减快速动量，将二者混合；HyMUON对所得方向执行Newton–Schulz正交化。先读混合系数、递推实现及状态数，再与只用EMA的Muon作相同预算比较。',
    'result': '摘要报告HyMUON在IWSLT14、GPT-2 124M/OpenWebText的100k与600k步设置中优于Muon；在LLaMA-1.3B/C4上最终验证损失为2.777，对照Muon为2.903。摘要未报告误差条、同等墙钟时间结果或新增状态大小。',
    'selection_reason': '直接改变Muon的输入方向，可与现有Muon、PowerStep的方向与步长分工对照；长期历史与短期适应的拆分提供可消融机制，信息增益高于泛泛稳定性描述。',
    'evidence_limit': '初步判断，仅题目与摘要，待全文核验。需检查混合参数是否扫参、两个历史的额外状态、学习率与权重衰减公平性、训练token与时间预算、种子方差和基线实现。损失差不能直接解释为下游能力提升，也不能外推更大模型。',
    'reading_decision': '精读'
}
editorial['top_papers'][top[1]['title']] = {
    'title_zh': 'CAST：用于大语言模型预训练的历史自适应谱裁剪',
    'abstract_zh': '软谱裁剪（soft spectral clipping，SSC）会减小梯度奇异值，但其效果取决于梯度尺度与裁剪阈值。我们提出CAST（Clipping with Adaptive Scales and Thresholds，采用自适应尺度与阈值的裁剪），它是SSC的一种历史自适应扩展。AdamW前一时刻的二阶矩用于设置行尺度和列尺度。裁剪输出的估计谱范数的指数移动平均用于设置每个矩阵的阈值。CAST在AdamW更新之前恢复原始坐标。\n\n  我们训练了一个总参数量为19.08B的混合专家（MoE）模型，训练使用83.89B个token。与九种梯度处理基线相比，CAST获得了更低的训练损失和平均验证损失，并且相对梯度尖峰最少。它用少7.2%的步数达到最优基线最终的平滑训练损失。',
    'summary': '用AdamW二阶矩设置行列尺度、用裁剪输出的谱范数历史设置矩阵阈值，形成可接入预训练的自适应谱处理模块。',
    'contributions': [
        '作者把SSC对尺度和阈值的依赖拆开，通过历史二阶矩与谱范数EMA分别自适应设置。',
        '摘要报告19.08B总参数MoE、83.89B token实验及相对九个梯度处理基线的结果。'
    ],
    'method': '先由AdamW上一时刻二阶矩构造行列尺度，在相应坐标中执行软谱裁剪，用裁剪输出估计谱范数的EMA更新各矩阵阈值，再恢复原坐标执行AdamW。重点核验是否引入循环依赖、谱估计精度及额外矩阵算子。',
    'result': '摘要报告训练损失与平均验证损失低于九个梯度处理基线，相对梯度尖峰最少；达到最佳基线最终平滑训练损失所需步数减少7.2%。19.08B是总参数数，摘要未给激活参数量；步数减少不等于端到端耗时减少。',
    'selection_reason': '直接命中梯度裁剪和自适应更新，且有MoE预训练证据入口。与HyMUON的方向历史形成互补，可用于设计谱处理、尺度恢复与步长的独立消融。',
    'evidence_limit': '初步判断，仅题目与摘要，待全文核验。需检查九个基线的阈值和预算、谱范数估计次数、EMA与尺度常数的调参、额外FLOPs与墙钟时间、MoE负载及种子方差，并确认7.2%的停止准则与平滑窗口。不能由摘要推断无需调参或总显存下降。',
    'reading_decision': '精读'
}
editorial['top_papers'][top[2]['title']] = {
    'title_zh': '迈向更易调参的训练算法：AlgoPerf的新结果',
    'abstract_zh': '评估深度学习优化器具有挑战性，因为超参数调节以及优化器与模型协同设计等混杂因素往往会掩盖算法的真实表现。我们给出更新后的AlgoPerf V1基准的结果，该基准在固定硬件上，以九种工作负载达到目标所需的墙钟时间评估一组新的训练算法。\n\n  本报告聚焦该基准自调参规则集的结果，该规则集要求提交方案在所有工作负载上使用同一套配置。在PyTorch和JAX的13个提交方案中，Schedule-Free AdamW排名第一。\n\n  除墙钟时间得分外，我们还分析了步数效率与墙钟执行时间之间的权衡，发现矩阵预条件方法和双循环方法具有良好的样本效率，但需要更积极的计算优化才能在运行时间上具有竞争力。\n\n  最后，我们的超参数显著性与保留分析表明，学习率和动量在不同领域与算法中都高度敏感，这凸显出对超参数调节更稳健、且能良好泛化到新任务的方法的必要性。',
    'summary': '以固定硬件、跨九任务单一配置的达标墙钟时间评测训练算法，为免调参与效率主张提供统一比较协议。',
    'contributions': [
        '作者报告AlgoPerf V1自调参规则集的13个提交，明确区分步数效率与运行时间。',
        '通过超参数显著性与保留分析检查跨任务敏感性，为优化器设计提供评测依据。'
    ],
    'method': '固定硬件与目标阈值，在九种工作负载上用单一配置比较达标墙钟时间，分别分析步数收益、算子开销和学习率/动量敏感性；将其作为后续PowerStep和Muon变体实验的协议参考。',
    'result': '摘要报告Schedule-Free AdamW在13个PyTorch/JAX提交中排名第一；矩阵预条件及双循环方法的样本效率未自动转化为运行时间竞争力，学习率与动量仍高度敏感。未报告所有LLM预训练设置下的排名。',
    'selection_reason': '当前目标包含免调参，单任务损失改进不足以检验该目标。该报告能直接改变训练预算和比较方案，与今日两篇新更新机制形成方法—评测组合。',
    'evidence_limit': '初步判断，仅题目与摘要，待全文核验。需检查九种任务的覆盖、目标阈值、失败运行计分、实现与编译成本、配置选择是否使用测试反馈以及提交者投入。跨任务单一配置不等于算法没有超参数；总体排名也不能代替LLM规模和硬件上的复测。',
    'reading_decision': '精读'
}

rest = [
    ('以初始化处梯度自外积的渐近对角占优解释Muon预条件器结构；可为几何设计提供局部依据，选读。', '四类典型模型和显式缩放率提供可核验线索，但初始化、期望与过参数化极限不能直接推出训练全程的曲率适配。', '选读'),
    ('将AdaGrad、Adam与AdamW转为只含加减乘和二幂除法的算术简化形式；适合有硬件实现目标时定点选读。', '直接更改优化器算术，但静态分母s=S(T)与通常自适应更新的差异必须核验；摘要没有端到端LLM训练效率证据。', '选读'),
    ('SPIN以快塑性序列和自适应稳定序列插值应对在线持续学习；仅在研究遗忘与历史机制时选读。', '有更新规则迁移线索，但核心是严格单遍OCL，不是LLM预训练；逆Hessian解释与内存预算需查全文。', '选读'),
    ('【语义排除】研究功能保护修复与LoRA结构更新，AdamW只是实现所用优化器，不计正式优化器阅读建议。', '主要干预对象是可训练结构及修复策略；不能因摘要出现AdamW和Qwen就判为优化器设计直接竞争工作。', '跟踪'),
    ('【语义排除】NAOD设计带偏LLM裁判下的偏好比较采样，未研究训练权重的更新规则。', '核心是采集设计、偏差干扰调整和策略风险，Frank–Wolfe是求解工具；与当前优化器设计研究对象不同。', '跟踪'),
    ('【语义排除】用一次性输出扰动实现扩散模型的重构隐私，梯度裁剪仅为DP-SGD背景比较。', '核心为PAC隐私适配机制，不是LLM优化器或裁剪更新设计；不计正式阅读建议。', '跟踪'),
    ('【语义排除】RFChipAgent用代理编排芯片设计及TPE/CMA-ES尺寸搜索，属于设计自动化。', 'adaptive optimization指电路搜索而非LLM权重优化；不计当前方向的正式阅读建议。', '跟踪')
]
for paper, (summary, reason, decision) in zip(top[3:], rest):
    editorial['top_papers'][paper['title']] = {'summary': summary, 'selection_reason': reason, 'reading_decision': decision, 'evidence_limit': '仅题目与摘要初评，待全文核验。'}

editorial['semantic_reranking'] = [
    {'title': p['title'], 'screening_rank': p['screening_rank'], 'candidate_screening_rank': p['candidate_screening_rank'], 'semantic_rank': p['semantic_rank'], 'formal_recommendation': p['formal_recommendation'], 'reason': editorial['top_papers'][p['title']]['selection_reason']}
    for p in top
]
limits = [
    'arXiv官方API以15个配置关键词查询submittedDate窗口，按相关性返回前200条；查询未施加category过滤，且Muon会命中粒子物理。因此这是有界关键词样本，202篇不是直接相关论文总量。',
    'OpenReview默认拼接查询持续翻页而中止，随后逐15关键词各取第一页至多100条，14项成功、AdamW一项HTTP 429。只保留pdate/cdate/tcdate可解析且在窗口内的记录，合并36篇；该日期未必等于首次公开日期。',
    'Semantic Scholar年度窗口2025-10-22至2026-09-21，两次HTTP 429，年度高影响力补充为0，不代表没有相关论文。',
    '10篇预筛候选均逐篇审阅题目和摘要，6篇保留阅读建议、4篇语义排除；top_papers仍保留10项供审计，正式推荐数量另记，不从第11名补选。',
    '前三分析仅为摘要初评，已核对arXiv官方摘要页。PDF归档和首页身份校验不等于完成全文评审。',
    '全库链接检查在本次执行前已有9项悬空链接，均位于旧07-research文档；本次只要求新增文件与链接无新增问题，并单独记录基线。'
]
editorial['source_limitations'] = limits
data['search_limitations'] = limits
data['sources_attempted'] = ['arxiv', 'openreview', 'semantic_scholar']
data['metadata_corrections'] = [{'field': 'arxiv_id', 'reason': '从官方API id/abs URL提取稳定编号，未改写评分或screening_rank。'}]
for p in top[:3]:
    p['source_verified'] = True
    p['source_verification_url'] = p['url'].replace('http:', 'https:')
    p['source_verification_date'] = '2026-10-22'
# The API author list split a single author into two records; verified abs page
# and PDF identify the full name. Preserve raw metadata for review.
p = top[2]
p.setdefault('raw_authors', p['authors'])
p['authors'] = ['Priya Kasimbeg', 'Ahmed Khaled', 'Aahlad Chandrabhatta', 'Yifan Wang', 'Andy Liu', 'Niccolò Ajroldi', 'David Tweedle', 'Xingyu (Alice) Yang', 'Isaac Patole', 'Frank Schneider', 'George E. Dahl']
for p2 in data['all_papers']:
    if p2['title'] == p['title']:
        p2.setdefault('raw_authors', p2['authors'])
        p2['authors'] = p['authors']
data['metadata_corrections'].append({'title': p['title'], 'field': 'authors', 'reason': '官方摘要页将Xingyu (Alice) Yang列为一位作者；保留raw_authors并修正API拆分记录。', 'source': p['source_verification_url']})
(DAY / 'daily-editorial.json').write_text(json.dumps(editorial, ensure_ascii=False, indent=2) + '\n')
(DAY / 'search_result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
coverage = {'date': '2026-10-22', 'arxiv': {'keywords': 15, 'returned': 200, 'limit': 200, 'sort': 'relevance', 'category_filter': False}, 'openreview': json.loads((ROOT / '.planning/daily-2026-10-22/openreview-coverage.json').read_text()), 'semantic_scholar': {'attempts': 2, 'status': 'HTTP 429', 'returned': 0}, 'limitations': limits}
(DAY / 'source-coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
