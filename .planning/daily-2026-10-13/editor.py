"""Editorial assessment of the corrected top ten; original scores are retained."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-13'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import paper_config as pc
D=json.loads((DAY/'search_result.json').read_text());top=D['top_papers'];A={}
A[top[0]['title']]={
 'title_zh':'谱定向 Muon',
 'summary':'用阈值选择性正交化分离小奇异方向放大和大奇异方向收缩，初步证据将训练收益与参数谱变平区分开。',
 'abstract_zh':r'''Muon 优化器对每个更新矩阵进行正交化，将其所有奇异值设为一，并且已在大语言模型训练中展现出很强的效果。然而，这种成功究竟来自放大梯度下降所忽略的小奇异方向，还是抑制干扰训练的大而退化的方向，仍不清楚。我们提出 Spectrally Targeted Muon，它只对高于或低于阈值 $\tau$ 的奇异值进行正交化，因此改变 $\tau$ 就能在归一化 SGD 与 Muon 之间插值。它通过对平移后的 Gram 矩阵执行 Newton–Schulz 迭代所得的投影，分离相关的奇异子空间，因此无需 SVD。我们在 CIFAR-10 和 NanoGPT speedrun 上评估这些变体，跟踪梯度、更新和权重矩阵的有效秩，以及一个新指标：更新与权重矩阵等谱流形切空间的对齐程度。我们有三项发现。第一，动量的小奇异值并非噪声。对每个矩阵除少数最大奇异值之外的所有奇异值进行正交化，几乎可以达到 Muon 的表现，尽管它只触及动量的一小部分；而只对顶部奇异值进行正交化则明显落后，尽管顶部包含了几乎全部动量。在语言模型中，每个动量奇异值都远小于一，因此定向正交化只能放大；Muon 通过使那些在原始尺度下小到无法用于训练的方向变得可训练而获胜。第二，收缩最大的奇异值才是保持参数谱平坦的原因；这项操作成本低，却不是驱动损失变化的因素。第三，AdamW 与 Muon 的主要区别在于建立结构的速度较慢，这解释了它较慢的起步，以及为什么 warmup 对 AdamW 有帮助，却只会损害 Muon。''',
 'contributions':['作者以奇异值阈值控制正交化子空间，构造归一化SGD与Muon之间的干预，而非只比较最终优化器排名。','作者区分小方向放大对损失的作用与大方向收缩对参数谱的作用，并提出等谱流形切空间对齐诊断。'],
 'method':'对平移Gram矩阵做Newton–Schulz投影，选择阈值上下的子空间，再比较有效秩、损失和切空间对齐；摘要报告CIFAR-10与NanoGPT speedrun。',
 'result':'摘要称除少数顶部奇异方向以外的正交化近似匹配Muon，而仅顶部方向明显落后；大方向收缩维持平坦参数谱。摘要没有给出损失差值、训练时间或置信区间，warmup结论只按所研究设置理解。',
 'selection_reason':'直接检验已有BulkBoost/Freon谱塑形思路最关键的因果问题：应保留或放大小方向，还是主要抑制大方向。无需SVD的子空间干预可成为设计新谱更新的对照；先读阈值实现与范数匹配实验，再决定是否组合裁剪。',
 'evidence_limit':'基于arXiv v1原始题目、摘要与元数据的低置信度初评，待全文核验。需核对阈值选法、投影近似误差、更新范数/学习率/算力与调参预算是否匹配；小奇异值非噪声的结论需检查参考方向、统计误差及时间变化。CIFAR-10与NanoGPT不能支持所有规模LLM或所有warmup策略的普遍结论。',
 'reading_decision':'精读'}
A[top[1]['title']]={
 'title_zh':'在预条件器空间中舍入：重新设计4-bit AdamW优化器状态量化',
 'summary':'ZIP-SR和ZE-EDEN分别从舍入概率与码本校准控制二阶矩量化后的预条件器误差，提供低比特AdamW的直接对照。',
 'abstract_zh':'''量化 AdamW 的优化器状态可以减少持久存储，但量化误差会通过矩递推传播，并扰动后续的自适应更新。我们从“舍入空间”的角度重新设计 AdamW 的 4-bit 优化器状态量化：舍入空间是量化器在相邻重构级别之间作出选择时所使用的坐标。对于二阶矩，对邻近零的量化单元进行局部分析表明，较小的平均状态误差未必意味着下一步的平均预条件器误差也较小。一维二次构造进一步表明，在状态空间和预条件器空间进行舍入会产生性质不同的优化动力学。这些结果促成了包含零的预条件器空间随机舍入（Zero-Inclusive Preconditioner-space Stochastic Rounding，ZIP-SR）：它在二阶矩码本中保留零，并在预条件器空间计算随机舍入概率。作为互补路线，排除零的 EDEN 校准（Zero-Excluding EDEN calibration，ZE-EDEN）使用不含零的二阶矩码本，并对量化后的二阶矩块重新缩放，以缓解正量化下限引起的预条件器失真。两种配置的一阶矩都采用 4-bit NormalFloat（NF4），并在训练最后 10% 的阶段，对 LM-head 的一阶矩实施定向随机舍入。在参数规模从 130M 到 2.7B 的 GPT 和 Llama 风格预训练实验中，两种方法都在每个被评估的模型规模上缩小了 TorchAO 4-bit AdamW 相对于 32-bit AdamW 的平均验证损失差距，所报告的最大差距缩减达到 70%。在全参数监督微调中，两种方案均取得了低于 TorchAO 的验证损失，同时在下游任务上保持接近 32-bit AdamW 的表现。''',
 'contributions':['作者提出状态误差与预条件器误差可分离的一维局部分析/动力学构造，说明仅优化状态无偏性并不足够。','作者提出保留零的ZIP-SR和排除零后再校准的ZE-EDEN两条路线，并结合LM-head末段一阶矩随机舍入。'],
 'method':'围绕邻零二阶矩量化单元分析后续预条件器失真；在预条件器空间计算随机舍入概率，或使用正下限码本并进行块校准；一阶矩为NF4，LM-head最后10%训练另作定向处理。',
 'result':'摘要报告130M至2.7B的GPT/Llama预训练中每个评估规模均缩小对32-bit AdamW的平均验证损失差距，最大缩减70%；全参数SFT验证损失低于TorchAO、下游接近32-bit。70%是损失差距的缩减，不是总体损失下降或内存缩减。',
 'selection_reason':'与已有PowerStep“取消二阶状态”的路线形成保留二阶状态但低比特存储的对照，可帮助区分收益来自状态删减、量化还是更新几何。邻零分析能直接指导数值稳定性检查；需先复现一维构造和LM-head消融，再核算端到端资源。',
 'evidence_limit':'基于arXiv v1摘要的低置信度初评，待全文核验。需审查epsilon、偏差修正、码本/块大小、量化递推误差、随机舍入开销以及TorchAO版本与公平调参；隔离LM-head末段处理和二阶矩设计各自贡献。摘要未报告净显存、吞吐、多种子方差或大于2.7B结果，不把4-bit直接等同端到端8倍内存收益。',
 'reading_decision':'精读'}
A[top[2]['title']]={
 'title_zh':'MoARa：面向低秩大语言模型预训练的模块感知秩分配与结构保持分解',
 'summary':'按模块敏感度分配投影秩并拆分梯度幅值/方向，报告在Llama 2 7B达到GaLore同等困惑度所需步骤与时间减少。',
 'abstract_zh':'''低秩梯度投影降低了大语言模型（LLM）预训练的优化器状态内存成本，但达到目标质量所需的步数和实际墙钟时间仍是有意义的改进维度。我们将这一问题归因于现有方法中的两项设计选择：投影秩预算在投影敏感度各不相同的 Transformer 模块之间被均匀分配；对原始梯度进行投影会同时削弱其幅值和方向。我们提出 MoARa，它结合基于静态性能剖析的模块感知投影秩分配与分块的幅值–方向分解；默认块大小设在注意力头维度附近。在涵盖 Llama、Qwen 和 DeepSeek 的五种 Transformer 架构、300M 至 7B 的规模上，在 Llama 2 7B 中，配合 MoARa 的 GaLore 以少 37% 的步数和少 34% 的实际墙钟时间达到标准 GaLore 的最终困惑度，在标准计算图编译下仅增加 0.2% 的峰值预留内存开销。在我们评估的六种低秩预训练方法中，仅采用模块感知秩分配，就在全部六种方法上带来了方向一致的步数减少。在兼容的宿主方法上，两组件设计的步数减少最多达到 41.7%，实际墙钟时间减少最多达到 37.1%。''',
 'contributions':['作者把总秩预算从均匀分配改为基于模块敏感度的静态分配，并提供跨六种低秩方法的单组件对照。','作者用分块幅值–方向分解减少原始梯度投影的结构损失，并在兼容方法中联合使用。'],
 'method':'先做模块敏感度剖析并分配投影秩，再以接近注意力头维度的块分离幅值和方向；在GaLore及其他低秩预训练方法上检验。',
 'result':'摘要报告五架构、300M至7B；Llama 2 7B达到标准GaLore最终困惑度时步数少37%、墙钟少34%、峰值reserved memory仅多0.2%；单组件六方法均减少步骤，兼容宿主的双组件最高减少41.7%步骤/37.1%时间。最高值不适用于全部设置。',
 'selection_reason':'为当前状态内存研究补充“同一预算如何分配”的可操作路线，并报告7B端到端时间对照。与低比特AdamW共同形成实用资源轴，避免前三全是Muon变体；可先复现静态秩分配，成本低于同时重写全部更新规则。',
 'evidence_limit':'基于arXiv v1题目、摘要与元数据的低置信度初评，待全文核验。需核对敏感度剖析成本是否计入时间、总秩/显存/训练token/学习率/质量目标的预算匹配、编译与硬件设置、种子方差及哪些宿主兼容双组件。0.2%是峰值预留内存额外开销，不是优化器状态压缩比例；论文在近30天窗口内，并非今日首发。',
 'reading_decision':'精读'}
rest=[
 '稀有token的非对称梯度与自适应归一化会改变固定点；给出RMSProp周期到达闭式偏差、SGD/AMSGrad对照，优先选读词表参数机制，不能把unigram结论直接外推所有LLM。',
 '逐样本裁剪系数与梯度的协方差会扭曲Muon极因子，短模型历史平均缓解该残差；私有GPT-2微调有配种子对照，选读裁剪几何及额外计算/隐私预算。',
 'Muon损失中性边界与时间方向对齐分离；无动量随机理论、130M与1B诊断提供大步长机制线索，需核对有动量实现与边界估计，暂列选读。',
 '有限p阶矩下比较裁剪/不裁剪平均SGD，给出偏差与置信度权衡及非对称噪声极限点偏移；作为裁剪设计的理论反例选读，不作为大模型训练实证。',
 '行列log-RMS尺度场连接Transformer权重通道组织与AdamW二阶矩；冻结checkpoint编辑提供干预线索，可选读通道增益设计，尚未提出直接优化器收益。',
 '用连续流平衡与随机探针替代NS稠密矩阵乘；仅10.95M模型/电路模拟，探针两种子且验证交叉熵较控制高0.0188，硬件实现证据需跟踪。',
 '校准LLM judge并比较量化/early-exit输出质量，属于推理优化测量而非训练参数更新；对当前优化器设计弱相关，降为跟踪。']
for p,summary in zip(top[3:],rest):A[p['title']]={'summary':summary,'reading_decision':'跟踪' if p['semantic_rank']>=9 else '选读'}
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
assets=[]
for rel in tracked:
 p=ROOT/rel
 if p.is_file() and ((rel.startswith('03-notes/') and p.name=='精读.md') or (rel.startswith('01-raw/') and p.suffix=='.pdf')):
  assets.append((pc.clean_stem(p.parent.name if p.name=='精读.md' else p.stem),p))
# Only verifiable library neighbors; shared topic does not establish inheritance.
neighbors=[['Does Muon Need Fine','Freon'],['PowerStep','Minimal-State Optimization'],['Rethinking Memory-Efficient Optimizers','Low Rank Curvature']]
for p,terms in zip(top[:3],neighbors):
 related=[]
 for term in terms:
  matches=[path for norm,path in assets if pc.clean_stem(term) in norm]
  matches.sort(key=lambda path:(path.name!='精读.md',str(path)))
  if matches:
   path=matches[0];title=path.parent.name if path.name=='精读.md' else path.stem
   related.append({'title':title,'path':str(path),'shared_terms':[term]})
 # Search tracked recommendation history when no standalone asset matches.
 for term in terms:
  if any(term in x.get('shared_terms',[]) for x in related):continue
  for rel in sorted(tracked):
   path=ROOT/rel
   if rel.startswith('08-daily/') and path.name=='search_result.json' and path.is_file():
    history=json.loads(path.read_text())
    match=next((x for x in history.get('top_papers',[]) if pc.clean_stem(term) in pc.clean_stem(x.get('title',''))),None)
    note=path.parent/'今日检索.md'
    if match and note.is_file() and note.relative_to(ROOT).as_posix() in tracked:
     related.append({'title':match['title'],'path':str(path),'shared_terms':[term]});break
 p['related_papers']=related
top[2]['matched_subdomain']='自适应与状态内存'
D['semantic_subdomain_corrections']=[{'title':top[2]['title'],'matched_subdomain':'自适应与状态内存','reason':'摘要的optimizer-state memory cost与共享配置的optimizer state memory同义，按语义归类；不修改预筛分。'}]
for p in D['all_papers']:
 if p['title']==top[2]['title']:p['matched_subdomain']='自适应与状态内存'
 if p['title'] in {q['title'] for q in top[:3]}:p['related_papers']=next(q['related_papers'] for q in top[:3] if q['title']==p['title'])
E={
 'overview':{'trend':'本次检索264篇唯一论文，82篇已知、182篇未命中已有资产或历史推荐。语义复排优先读谱子空间干预、低比特预条件器误差与低秩预算分配，分别对应更新机制、状态压缩和端到端效率；这些是当前候选的初步观察，不是全领域趋势统计。今日指检索/入库日，前三首发分别为10月7日、10月8日、9月14日。',
 'hotspots':[{'title':'覆盖边界','text':'近期窗口2026-09-13至2026-10-13；arXiv相关性前200条，OpenReview15关键词各首100条，13个查询截断；过去一年补充两次429，未取得引用信号。不能据此宣称今日没有新论文。'},
 {'title':'把更新几何和资源预算分别检验','text':'谱定向干预可区分小方向放大和大方向收缩；ZIP-SR/ZE-EDEN检验低比特更新误差；MoARa检验模块秩预算。组合实验应匹配更新范数、质量目标和算力。'},
 {'title':'去重审计','text':'PowerStep的LaTeX ell_p写法与已有lp文件名未被自动归一；核验其已有PDF后标记已知，从原第11候选补齐。保留原始与修正预筛，未修改脚本分数。'}]},
 'domain_summaries':{'大模型优化器设计':{'summary':'前三形成谱机制、量化误差与投影预算的互补组合；稀有token固定点与裁剪协方差作为下一批机制阅读。按匹配关键词归类的统计会包含弱相关和Muon粒子物理误命中，不等于直接相关工作的数量。',
 'subfields':{'符号与范数几何优化':'Spectrally Targeted Muon将小方向放大与谱变平区分；Muon稳定边界和Physical Muon提供动力学/实现视角，需分别审查适用规模与成本。',
 '免调参与梯度裁剪':'私有Muon分析逐样本裁剪诱发的几何残差；重尾平均SGD给出裁剪的统计收益与代价。本次前三没有直接免调参方法。',
 '自适应与状态内存':'ZIP-SR/ZE-EDEN保留低比特二阶矩，MoARa改变低秩预算；稀有token工作揭示坐标自适应的固定点偏差。已有PowerStep作为对照，不再重复入队。',
 '大模型训练与优化器设计':'MoARa有300M至7B预训练与墙钟结果，低比特AdamW覆盖130M至2.7B；不要将小模型谱诊断外推至所有大模型。',
 '未细分':'宽泛优化关键词可能命中推理、求解和其他问题；前十中的推理质量测量降为跟踪，全量结果只作为检索记录。'}}},
 'top_papers':A,
 'semantic_reranking':[{'title':p['title'],'screening_rank':p['screening_rank'],'candidate_screening_rank':p['candidate_screening_rank'],'semantic_rank':p['semantic_rank'],'reason':A[p['title']].get('selection_reason',A[p['title']]['summary']),'assessment_scope':'metadata_and_abstract'} for p in top],
 'abstract_translation_audit':{'scope':'前三原始摘要完整翻译，信息顺序、方法名、数字、比较对象和限定词逐项核对；保留原始英文摘要。','checked':['Spectrally Targeted：阈值上下/归一化SGD至Muon、shifted Gram/NS无SVD、CIFAR-10/NanoGPT、有效秩/等谱切空间、三项发现与warmup边界','ZIP-SR/ZE-EDEN：邻零局部分析/一维构造、保留零与排除零两路线、NF4/LM-head最后10%、130M至2.7B、每规模改善/最大70%差距缩减、全参数SFT','MoARa：均匀预算/幅值方向共同衰减、静态剖析/块大小近head维度、五架构/300M至7B/Llama-Qwen-DeepSeek、37%步骤/34%时间/0.2%reserved overhead、六方法单组件、兼容宿主最高41.7%/37.1%']}}
(DAY/'daily-editorial.json').write_text(json.dumps(E,ensure_ascii=False,indent=2)+'\n')
(DAY/'search_result.json').write_text(json.dumps(D,ensure_ascii=False,indent=2)+'\n')
