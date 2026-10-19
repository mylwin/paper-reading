import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DAY=ROOT/'08-daily/2026-10-10';RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
from paper_config import clean_stem
D=json.loads((DAY/'search_result.json').read_text());titles=[p['title'] for p in D['top_papers']]
def related(fragment,shared):
 for base,pattern in [('03-notes','*/*/精读.md'),('01-raw','*/*.pdf')]:
  for path in sorted((ROOT/base).glob(pattern)):
   stem=path.parent.name if base=='03-notes' else path.stem
   if fragment.lower() in stem.lower():
    return {'title':stem.replace('_',' '),'path':str(path),'source':'note' if base=='03-notes' else 'pdf','shared_terms':shared}
 raise ValueError(fragment)
D['top_papers'][0]['related_papers']=[related('Understanding_MARS',['动量设计','历史梯度']),related('Anytime_Training',['调度依赖','训练时长'])]
D['top_papers'][1]['related_papers']=[related('When_and_Why_SignSGD',['符号更新','范数几何']),related('PowerStep_',['幅值信息','低状态更新']),related('Beyond_the_Matrix_Sign',['谱方向','矩阵更新'])]
D['top_papers'][2]['related_papers']=[related('DGA_Muon',['方向与尺度分离']),related('ORCA_',['谱条件化','更新尺度']),related('PowerStep_',['范数几何','步长'])]
# Avoid declaring a conceptual relationship from an isolated common title word.
for p in D['all_papers']:
 p['related_papers']=[r for r in p.get('related_papers',[]) if len(r.get('shared_terms',[]))>=2]
 if p['title'] in titles[:3]:p['related_papers']=next(q['related_papers'] for q in D['top_papers'][:3] if q['title']==p['title'])
for p in D['top_papers'][:3]:
 verify=RUN/('verify-'+p['id']+'.json')
 if verify.exists():
  n=next((n for n in json.loads(verify.read_text()).get('notes',[]) if n.get('forum',n.get('id'))==p['id']),None)
  if n:
   assert n['content']['abstract']['value']==p['summary']
   p['license']=n.get('license');p['metadata_verified']=True
D['top_papers'][1]['authors']=['Maria Smirnova','Alexey Kravatskiy']
D['top_papers'][1]['authors_status']='public_arxiv_and_OPT_record'
D['top_papers'][0]['authors_status']='anonymous_OpenReview; public_preprint_author_list_separate'
D['top_papers'][2]['authors_status']='anonymous_submission_not_disclosed'
analyses={
 titles[0]:{
 'title_zh':'带动量语言模型规模扩展的对数时间调度',
 'summary':'ADana 让 AdamW 的梯度记忆时长随训练增长，并用显式阻尼抑制长记忆不稳定，是与谱方向设计互补的优化器路线。',
 'abstract_zh':r'在实践中，AdamW 的超参数 $(\beta_1,\beta_2)$ 和权重衰减 $\lambda$ 通常保持固定值。是否有理由采用其他做法？我们表明，对于大规模语言模型训练，答案是肯定的：利用语言数据的幂律结构，可以为 $(\beta_1,\beta_2,\lambda)$ 设计随时间变化的调度，从而带来显著的性能收益。我们研究对数时间调度，在这种调度中，优化器的梯度记忆时长随训练时间增长。尽管其朴素变体不稳定，我们表明，适当的阻尼机制能够恢复稳定性，同时保留更长记忆带来的收益。在此基础上，我们提出 ADana，一种类似 AdamW 的优化器，将对数时间调度与显式阻尼结合，以平衡稳定性和性能。我们在不同规模的 Transformer（4500 万到 44 亿参数）上实证评估 ADana，并与 AdamW、Muon 和 Ademamix 比较。在适当调参时，相较于调优后的 AdamW，ADana 可实现最高 40\% 的计算效率收益；随着模型规模增长，这些收益持续存在，甚至进一步提高。我们还表明，将对数时间调度用于 Ademamix 时也会产生类似收益，而仅采用对数时间权重衰减就能带来显著改进。最后，我们提出 ADana 的若干变体，以缓解潜在失效模式并提高稳健性。',
 'contributions':['作者提出随训练时间增长的梯度记忆时长，并通过显式阻尼修复朴素长记忆方案的不稳定。','作者将动量、二阶矩和权重衰减调度放入同一设计框架，报告 ADana 及若干防失效变体。'],
 'method':'以语言数据的幂律结构为动机，把固定 beta1/beta2 与固定衰减改成对数时间调度；阻尼用于平衡历史信息与稳定性。摘要没有给出具体调度公式，应优先阅读系数随时间的定义、初始化与偏差修正，而不把“更长记忆”单独视为有效机制。',
 'result':'OpenReview 摘要报告 45M–4.4B Transformer，对照 AdamW、Muon、Ademamix，适当调参时相对调优 AdamW 最高 40% 计算效率收益；仅调度权重衰减也可改进。公开 arXiv 2602.05298v2（2026-02-18）报告最大 2.6B，属于较早版本，本地归档该版，不能用它证明 4.4B 实验。',
 'selection_reason':'与昨日 SF-NorMuon 形成直接对照：后者解除训练终点依赖，ADana 改变历史梯度的有效记忆时标；两个问题不同。与已精读 MARS 比较，可把收益归因拆成历史梯度、阻尼和衰减三个组件。对优化器设计的新增信息高于只增加一种谱滤波器。',
 'evidence_limit':'依据 OpenReview CC BY 4.0 原始摘要及同名 arXiv 元数据，低置信度初评，待全文核验。需核对幂律假设、阻尼公式、beta2 调度与一阶动量的独立消融、搜索预算与固定 token/墙钟公平性、40% 的精确定义、种子方差以及超参跨规模迁移。它仍要求适当调参，不能称免调参。匿名 ICLR 2027 投稿与公开早期预印本属于不同版本；最新实验需读取投稿 PDF。',
 'reading_decision':'精读'},
 titles[1]:{
 'title_zh':'Muon 的符号压缩：SignMuon、MuonSign 与误差反馈的局限',
 'summary':'符号算子与 Muon 谱方向的组合可能破坏下降性；误差反馈放在梯度端可恢复理论保证，但实测排序却偏向没有一般保证的输出端压缩。',
 'abstract_zh':r'SignMuon 对 Muon 的线性最小化预言机（Linear Minimization Oracle，LMO）方向逐元素取符号，将 Muon 更新压缩至每个参数一个比特：这是给矩阵感知优化器提供一比特预算的最直接方式。它在实证上优于 SignSGD，却可能连在线性函数上都沿上升方向更新。在预言机之前而不是之后对梯度取符号也不能解决这一问题：我们给出一个小型显式实例，其中先取符号的方式（MuonUSign）和两侧均取符号的方式（MuonSign）也都会上升，因此在预言机周围的任何取符号位置都不能保证一般情况下的下降。误差反馈同样不能修复后取符号的方式：若将其应用于预言机的输出，无论光滑性常数、步长和动量取何值，它都可能失败。将其应用于梯度则确实有效；对于光滑非凸问题，EF21-MuonUSign 和 EF21-MuonSign 达到梯度范数平方的标准 $\mathcal{O}(T^{-1/2})$ 收敛率，后者在每个通信方向上均使用一比特。随后，我们的实验颠倒了这一排序：在 CIFAR-10、其联邦划分和 nanoGPT speedrun 中，最强的压缩方法是后取符号，恰恰是我们证明可能发散的那个位置，而具有可证明收敛性的变体落后于它。',
 'contributions':['作者报告 sign-before、sign-after 与两侧取符号都可能破坏下降性的显式反例。','作者区分预言机输出端与梯度端的误差反馈，后者支持 EF21 变体的非凸收敛保证；同时报告实验排序与一般理论保证相反。'],
 'method':'把符号压缩放置在 Muon LMO 的不同位置，先用小型反例检查更新与原梯度的对齐，再比较输出端和梯度端误差反馈。理论讨论的是覆盖问题类的一般保证，不能将存在反例误写为典型 LLM 训练必然发散。',
 'result':'摘要报告输出端误差反馈可在任意光滑常数、步长和动量下失败；梯度端 EF21-MuonUSign/EF21-MuonSign 对梯度范数平方给出 O(T^-1/2)。CIFAR-10、联邦划分和 nanoGPT speedrun 的压缩方法排序反而偏向 sign-after；摘要未给出损失、墙钟或通信降幅数值。',
 'selection_reason':'直接连接已精读 SignSGD 与 PowerStep 的范数几何：两个各自合理的方向算子，组合后未必仍是下降方向。这为设计低比特 Muon、压缩动量或省状态更新提供可证伪的检查，而不仅是性能对比。可以先复现反例，再在小型语言模型上测梯度–更新夹角。',
 'evidence_limit':'依据 OpenReview 摘要、公开 arXiv 2607.29674v1 和作者元数据，低置信度初评，待全文核验。需核对反例的归一化与动量规则、EF21 的收缩参数和维度常数、理论所控制的迭代点与实验用模型是否一致、误差反馈新增状态与每个通信方向的真实比特数、基线调参和随机种子。最强压缩方法不等于胜过未压缩 Muon；一比特通信不等于一比特优化器内存。OPT 2026 条目与七月预印本是近期公开记录/早期版本的区别，不能称今天首次发表。',
 'reading_decision':'精读'},
 titles[2]:{
 'title_zh':'仅展平谱还不够：Muon 何时改善学习率与收敛',
 'summary':'精确极分解 Muon 的可下降步长受梯度奇异值尺度控制，而跨参数块的尺度差异仍影响全局收敛。',
 'abstract_zh':r'Muon 通过展平每个梯度的奇异值、同时保留其奇异方向，更新矩阵值参数。尽管它在实践中取得成功，控制其学习率承受能力与收敛行为的机制仍未得到充分理解。我们通过精确极分解更新，分析确定性的全批量 Muon。首先，我们推导保证下降的学习率阈值，并表明该阈值由梯度矩阵的、按参数块大小加权的平均奇异值尺度控制。这一刻画明确呈现梯度幅值的作用，并揭示目标函数缩放如何影响允许的步长。随后，我们分析 Muon 的收敛行为，识别控制其收敛率的谱性质。我们的分析表明，仅展平各个参数块内部的谱，还不足以确定全局收敛，因为参数块之间的尺度差异仍然具有影响。综合来看，这些结果从理论上解释了谱展平如何控制 Muon 的学习率承受能力和收敛行为。',
 'contributions':['作者推导与参数块大小加权平均奇异值尺度有关的下降学习率阈值。','作者指出块内谱展平不足以确定全局收敛，保留跨块尺度差异和目标缩放的影响。'],
 'method':'在确定性全批量与 exact-polar 更新下分析下降阈值和谱相关的收敛行为。核心是将块内奇异值均衡与块间梯度尺度分开；摘要没有给出阈值公式和完整假设，不根据题目补写全局收敛阶。',
 'result':'摘要报告下降阈值的控制量和跨块尺度的影响，没有量化 LLM 训练结果、具体收敛率公式或随机梯度结论。它提供理想化更新的理论依据，不是有限 Newton–Schulz 实现优于 AdamW 的实证证明。',
 'selection_reason':'直接审查 DGA-Muon、ORCA 和当前方向–步长分离思路：固定谱方向后如何确定尺度，以及全局单一学习率是否覆盖不同参数块。与 SignMuon 的算子组合反例互补，前者讨论组合是否下降，本篇讨论保留 exact-polar 时下降步长的边界。',
 'evidence_limit':'依据 OpenReview CC BY 4.0 原始摘要，低置信度初评，待全文核验；匿名 ICLR 2027 投稿不等于录用。需核对光滑性使用的范数、阈值是否可计算或依赖未知常数、梯度秩亏点、块大小权重、目标乘常数后的步长变换，以及动量、随机噪声和有限 Newton–Schulz 误差。PDF HTTP403，尚未获得全文，不填虚假定理编号或实验证据。',
 'reading_decision':'选读'}
}
rest=[
 'AdamW/SGDM 在固定 token 预算下导出 B∝T^(2/3) 的长时批量规则，FineWeb 最大1.4B；先核对 μ-KL 与下界条件，再与已有批量依赖论文对照。',
 'Pythia-160M 用固定文本探针区分优化器步数与历史梯度相关性；适合作为 ADana 的诊断线索，但预测改进尚不能证明新优化器收益。',
 'Turbo-Muon 预条件化 Newton–Schulz 初值，移除常用5次迭代中的1次，摘要报告约3%端到端加速；核对额外开销、有限极分解误差及基准方差。',
 'Muon 光谱匹配把数据子集选择与实际更新方向对齐；关注极分解线性化误差、样本多样性和选择成本，属于优化器与数据选择耦合。',
 'DP-FedAdamW 修正隐私噪声引入的二阶矩方差/偏差与客户端漂移；Tiny-ImageNet ε=1报告5.83%提升，联邦隐私条件不直接等于通用LLM预训练。',
 'Conv-NS 在卷积算子几何中逼近极因子，却未明显胜过reshape Muon；可借鉴几何对齐与经验收益的对照，卷积不是当前大模型主线。',
 'Muon式动量谱驯化用于非凸、非全局梯度Lipschitz的Gibbs采样，报告不变测度和指数混合；这是采样算法，暂作数学工具跟踪。'
]
for title,summary in zip(titles[3:],rest):analyses[title]={'summary':summary}
editorial={
 'overview':{
 'trend':'本次聚焦优化器设计的三个可拆分机制：梯度记忆的时标、符号与谱方向的组合顺序，以及跨参数块的步长尺度。306篇为关键词候选而非306篇直接相关论文；其中62篇命中已有资产或历史推荐，244篇未命中，取预筛前10并完成语义复排。检索日期为2026-10-10，前三篇为今日新增推荐，不是今日首次发表。',
 'hotspots':[
 {'title':'覆盖与时间边界','text':'近30天窗口2026-09-10至2026-10-10；arXiv按相关性最多200条，OpenReview按配置15关键词每词第1页100条。候选池的最新arXiv提交为10月8日，不能由此断言10月9日或10日没有新论文。含物理Muon等词义误命中，仅作附录记录。Semantic Scholar近一年窗口2025-10-10至2026-09-09两次HTTP429，无高影响力补充与引用数据。'},
 {'title':'与既有阅读形成对照','text':'ADana与昨日SF-NorMuon比较记忆调度和训练终点依赖；SignMuon与已有SignSGD/PowerStep比较算子组合和幅值保留；跨块尺度理论与DGA-Muon/ORCA比较方向归一化和步幅选择。优先记录可检验差异，不从投稿场所或摘要收益判断正确性。'},
 {'title':'版本与PDF状态','text':'OpenReview前三篇PDF均HTTP403。ADana存在同名arXiv v2，但最大模型为2.6B，不同于投稿摘要的4.4B；SignMuon存在arXiv v1。归档时记录公开版本，摘要分析继续以本次检索摘要为准。Spectral Flattening没有取得可归档全文，保留远程入口与失败原因。'},
 {'title':'新近候选线索','text':'候选池检出10月8日 Rounding in Preconditioner Space: Redesigning 4-bit AdamW Optimizer-State Quantization、10月7日 Spectrally Targeted Muon 等。它们仍列于附录；本次未进入focus预筛前10，没有替换已选前三篇，也未做全文判断。'}]},
 'domain_summaries':{'大模型优化器设计':{
 'summary':'结合已有谱优化、PowerStep与SignSGD阅读，今日重点是哪些历史信息应保留、压缩应放在哪里、尺度应如何跨块设置。把效率、理论保证和经验稳定性分别核验。直接LLM证据优先，隐私联邦、卷积和Langevin采样候选只作相邻问题跟踪。',
 'subfields':{
 '符号与范数几何优化':'SignMuon反例提醒合理的符号方向与谱方向不能任意组合；跨块尺度理论提醒块内谱均衡不代表全局尺度均衡。优先检验夹角、有限NS误差和跨层步幅。',
 '免调参与梯度裁剪':'ADana和BST讨论调度/超参迁移，不是无可调参数；对Polyak、裁剪候选仍需检查未知损失下界、阈值和噪声假设。',
 '自适应与状态内存':'ADana改变记忆时长，What Ages a Gradient提供诊断；低比特通信、误差反馈残差、二阶矩量化和峰值显存必须分别核算。',
 '大模型训练与优化器设计':'训练时长、batch与数据选择都可能改变优化器排序。固定token预算与固定墙钟是不同实验，应保留调参成本和模型规模边界。',
 '未细分':'宽泛focus命中但未落到配置细分关键词的条目，不能自动解释为大模型优化器贡献。'}}},
 'top_papers':analyses,
 'semantic_reranking':[{'title':p['title'],'screening_rank':p['screening_rank'],'candidate_screening_rank':p['candidate_screening_rank'],'semantic_rank':p['semantic_rank'],'reason':analyses[p['title']].get('selection_reason',analyses[p['title']]['summary']),'assessment_scope':'metadata_and_abstract'} for p in D['top_papers']],
 'abstract_translation_audit':{'source':'前三篇采用本次OpenReview原始英文摘要，前两篇公开arXiv版本另行标注。ADana/谱展平投稿记录CC BY4.0已核验；SignMuon公开arXiv许可CC BY4.0。中文摘要是完整翻译改编，原文与翻译分列。','checked':['按英文信息顺序完整翻译','方法名称与变量','45M/4.4B/40%与版本差异','一比特通信方向','O(T^-1/2)控制梯度范数平方','全批量确定性限制','实验反转理论排序的限定结论']}}
(DAY/'daily-editorial.json').write_text(json.dumps(editorial,ensure_ascii=False,indent=2))
(DAY/'search_result.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
print('已保存完整摘要翻译、10篇复排理由和来源边界')
