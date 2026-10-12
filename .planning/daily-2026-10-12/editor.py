import json, sys, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; DAY=ROOT/'08-daily/2026-10-12'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import paper_config as pc
D=json.loads((DAY/'search_result.json').read_text()); top=D['top_papers']; A={}
A[top[0]['title']]={
 'title_zh':'从几何到泛化：为什么行归一化可以优于 Adam 和 Muon',
 'summary':'在受控高维分类模型中，行归一化保留类决策边界方向，而坐标与谱几何可引入不消失的失真；优势依赖协方差与坐标轴的关系。',
 'abstract_zh':'''不同优化器可以拟合同一份训练数据，却选择几何性质显著不同的分类器，但这种差异是否会以可证明的方式影响总体表现，仍不清楚。我们表明，在高维多分类中，逐行归一化可以实现严格高于全批量 Adam、随机重排 Adam 的一种代理方法，以及精确 SVD Muon 的总体准确率。在各向同性高斯云数据模型下，这种优势产生于行归一化的逐类欧氏几何渐近地保留了总体决策边界的方向，而 Adam 的逐坐标几何和 Muon 的谱几何则引入了不会消失的失真。超出各向同性情形，当类均值协方差与测试噪声协方差具有独立取向时，这种优势在对类均值进行全批量训练的情形下仍然存在。对于类均值指数小于一的幂律谱，即使测试噪声高度各向异性，该优势也成立。当两种协方差均为对角矩阵且足够接近时，相对 Adam 的优势可能反转；而对二者施加相同的随机旋转，只改变它们与 Adam 坐标轴的对齐关系，就能恢复该优势。合成实验及语言模型最后一层实验支持所预测的优势。''',
 'contributions':['作者建立行、坐标和谱三种更新几何与总体分类准确率之间的条件性比较。','作者给出协方差取向、幂律谱以及对角且接近时的优势反转，区分几何机制与普遍优越性。'],
 'method':'在高维多分类数据模型中追踪总体决策边界方向，比较行归一化、Adam及精确SVD Muon引起的失真；用共同随机旋转隔离坐标轴对齐的影响。',
 'result':'摘要报告各向同性高斯云、独立取向协方差及类均值谱指数小于一时的严格准确率优势；对角且接近时可反转。合成与语言模型最后一层实验支持，未给出数字。',
 'selection_reason':'直接回应已有 The Row Normalization Puzzle in Muon 的行尺度问题，新增信息是总体泛化和可反转边界。适合先复核数据模型与反转构造，再决定是否将行几何引入优化器；90页全文建议定点阅读定理和最后一层实验。',
 'evidence_limit':'基于arXiv v1题目、摘要与元数据，低置信度初步判断，待全文核验。需区分全批量Adam、随机重排代理与实际AdamW；核对高维极限、初始化、正则化、类均值训练假设和协方差独立取向条件。最后一层实验不等于端到端LLM预训练优势，摘要无准确率差值与多种子信息。',
 'reading_decision':'精读'}
A[top[1]['title']]={
 'title_zh':'Muon 需要精细的谱塑形吗？',
 'summary':'BulkBoost仅用一个主体谱与尖峰谱的相对增益，在保持更新Frobenius范数时与精细谱映射竞争，为谱塑形提供低维对照。',
 'abstract_zh':r'''Muon 将当前和过去的梯度组合成矩阵动量。对于 $M=UΣV^\top$，理想化的极分解更新 $Q=UV^\top$ 赋予每个奇异方向相同的权重。我们将其称为平坦剖面。近来的若干优化器用精细谱映射替代这种平坦剖面，为每个方向赋予各自的增益。我们考察 Muon 更新需要多少这样的谱细节。我们的谱诊断表明，约 $94$--$97\%$ 的被测奇异模态位于估计的噪声边缘以下，但它们整体上仍与参考梯度正向对齐。

我们提出 BulkBoost，这是一种包含固定秩和噪声校准两种变体的两段谱重加权框架。后者使用拆分小批量的梯度差，校准 Muon 的 Nesterov 输入所对应的 Marchenko--Pastur 参考边缘，从而将边缘以下的主体谱与边缘以上的尖峰谱分离。两种变体都通过一个共享增益提高主体谱的相对权重，同时保持每个矩阵未经重加权方向的 Frobenius 范数。对于固定的分段，我们的理论给出了将权重移向主体谱能降低损失的一阶条件。它还量化了在所有逐模态重分配中，两段谱可以捕获的最大一阶改善速率的比例。在涵盖 Pythia-14M 至 410M 以及六个语料库的 30 种继续预训练设置中，两段重加权与 Freon 的精细幂律剖面具有竞争力，并优于 Spectra。以 Muon 的平坦剖面为参照，Freon 平均将最终损失降低了适配前损失的 $0.022\%$，而两段变体实现了 $0.073$--$0.147\%$ 的降低。这些观察表明，有益的平坦剖面偏离具有出人意料的低维性质：单一的主体谱对尖峰谱增益，所捕获的收益至少与精细谱剖面一样多。''',
 'contributions':['作者用固定秩或噪声校准分段，提出保范数的两段谱重加权，减少逐模态映射自由度。','作者给出固定分段下一阶下降条件及可捕获改善比例，并与Freon、Spectra作继续预训练比较。'],
 'method':'用拆分小批量梯度差估计噪声参考边缘，增大主体谱相对尖峰谱的权重，再保留原方向的Frobenius范数；固定秩变体提供不依赖该校准的对照。',
 'result':'摘要报告94–97%的奇异模态低于估计边缘且整体正向对齐；30种Pythia-14M至410M、六语料继续预训练中，两段变体相对平坦剖面降低适配前损失的0.073–0.147%，Freon为0.022%。分母是适配前损失，不是最终损失。',
 'selection_reason':'与已有Freon等谱塑形工作形成直接竞争，适合检验收益是否需要精细谱曲线。对优化器设计最有用的是保范数的简单对照，能减少步长混淆；先复现固定秩版本，再评估噪声估计的额外成本。',
 'evidence_limit':'基于arXiv v1原始摘要，低置信度初步判断，待全文核验。需读参考梯度构造、噪声边缘估计适用条件、分段与增益调参预算、统计误差、随机种子和校准额外计算。固定分段的一阶结论不直接保证动态分段、多步随机训练收敛；继续预训练410M以内不能直接推广到从头预训练的大模型。',
 'reading_decision':'精读'}
A[top[2]['title']]={
 'title_zh':'采用有限步 Newton–Schulz 正交化的 Muon 训练损失保证',
 'summary':'在宽两层ReLU与正定NTK条件下，用动量对齐及谱范数界证明有限步NS Muon可到达指定训练损失。',
 'abstract_zh':r'''现有 Muon 收敛分析要么假设精确正交化，要么分析经典 Newton--Schulz 多项式，并且只保证达到平稳性，因此 Muon 的五步调优 Newton--Schulz 迭代保留了什么、这是否足以达到预先指定的神经网络训练损失，仍未解决。我们建立了一项有限时间训练保证，同时考虑正交化之前的动量累积和调优后的有限步更新。对于输出权重固定且随机、极限神经切线核正定的足够宽两层 ReLU 网络的全批量训练，我们证明：以关于初始化的高概率，Muon 可以达到任意目标经验平方损失 $\varepsilon>0$。对于每个动量参数 $μ\in[0,1)$，在其他问题参数固定时，与 $(1-μ)\sqrt{\varepsilon}$ 成比例的、依赖目标的常数学习率，给出 $O((1-μ)^{-1}\varepsilon^{-1/2})$ 的到达时间上界。充分宽度与目标精度及动量均无关。分析表明，调优后的 Newton--Schulz 映射在限制更新谱范数的同时，保留了与动量缓冲的对齐。对初始化附近梯度变化的控制，将这种对齐传递到当前梯度，从而无需精确正交化就能确保在到达目标之前持续下降。数值实验在低于充分理论阈值的宽度下支持这些机制：梯度与更新的对齐始终高于分析参考值；在固定教师–学生数据集上，涵盖六种宽度与五种学生初始化的全部 30 次运行均达到目标损失，同时保持核的正定性。''',
 'contributions':['作者将动量累积与调优五步NS同时纳入有限时间训练损失分析，超出只证明平稳性。','作者用谱范数控制和梯度对齐转移联系有限步矩阵近似与下降机制。'],
 'method':'先证明有限NS更新与动量对齐并控制谱范数，再在初始化邻域限制梯度变化，使对齐转移到当前梯度，结合正定NTK达到目标平方损失。',
 'result':r'摘要给出所有 $μ\in[0,1)$ 下到达时间 $O((1-μ)^{-1}\varepsilon^{-1/2})$，学习率正比 $(1-μ)\sqrt{\varepsilon}$，充分宽度不依赖这两者；固定数据集六宽度、五初始化共30次运行达到目标。其他参数固定，不能忽略隐藏常数。',
 'selection_reason':'与昨日自适应NS计算及已有有限NS平滑分析互补：本篇问近似方向是否足以完成训练，而非如何减少迭代。22页、机制明确，可先核对五步多项式与对齐引理，作为改造谱算子的理论检查模板。',
 'evidence_limit':'基于arXiv v1题目与摘要，低置信度初步判断，待全文核验。条件限于全批量、宽两层ReLU、固定随机输出与正定极限NTK；学习率依赖目标，不能称免调参。需核验宽度中的样本量、核特征值和置信度依赖、有限NS系数/精度、初始化邻域保持以及动量趋近1的退化。单一教师–学生数据集不支持LLM随机预训练外推。',
 'reading_decision':'选读'}
rest=[
 'Soren以有界单调sigmoid替代谱完全压平，并用SNS多项式实现；相对光滑与metric PL条件及谱近似误差值得选读，摘要无实证量化，不能认定优于简单两段谱。',
 '全批量极分解更新的周期2振荡中，AGOP可继续对齐教师子空间；提供损失平台与表征变化分离的机制，但小头部/维度条件及深层冻结设置限制迁移。',
 '梯度交换子的logit投影读取训练顺序记忆；四LLM顺序识别92%、Qwen-3-4B十token降权缩小中位32%差距，优化器改造用途间接，先选读干预对照。',
 '一般范数与次高斯分布的良性过拟合分析给出p>1正例和1范数非高斯失败；可作为范数几何外推边界，尚非LLM训练算法结果。',
 '小于59M的nanoGPT/TinyStories重校准表明Step Law形式保留而系数改变，原学习率中位高估4倍；只作小规模调参对照，单管线限制外推。',
 'LinUCB小间隙采样趋向D最优设计，并给出渐近遗憾常数和推断结论；属于在线实验设计，与当前优化器设计弱相关，降为跟踪。',
 'SDRL训练LLM选择工业优化求解策略；adaptive optimization指求解工作负载而非参数更新优化器，对当前课题低相关，降为跟踪。']
for x,s in zip(top[3:],rest):A[x['title']]={'summary':s,'reading_decision':'跟踪' if x['semantic_rank']>=9 else '选读'}
# Related links must survive the requested Git publication: use tracked assets or today's PDFs.
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
assets=[]
for rel in tracked:
 p=ROOT/rel
 if (rel.startswith('03-notes/') and p.name=='精读.md') or (rel.startswith('01-raw/') and p.suffix=='.pdf'):
  assets.append((pc.clean_stem(p.parent.name if p.name=='精读.md' else p.stem),p))
neighbors=[['The Row Normalization Puzzle in Muon','PowerStep'],['Freon','Cost-free Spectral Estimation'],['Muon with Finite Newton-Schulz','Cost-free Spectral Estimation']]
for paper,terms in zip(top[:3],neighbors):
 related=[]
 for term in terms:
  matches=[p for norm,p in assets if pc.clean_stem(term) in norm]
  matches.sort(key=lambda p:(p.name!='精读.md',str(p)))
  if matches:
   p=matches[0]; title=p.parent.name if p.name=='精读.md' else p.stem
   related.append({'title':title,'path':str(p),'shared_terms':[term]})
 paper['related_papers']=related
for x in D['all_papers']:
 if x['title'] in {p['title'] for p in top[:3]}:x['related_papers']=next(p['related_papers'] for p in top[:3] if p['title']==x['title'])
E={'overview':{'trend':'本次267篇唯一候选、74篇已有命中、193篇未命中；保持脚本前十候选与分数，语义复排后优先读行几何泛化、低维谱塑形与有限NS训练保证。今日指入队/入库日，前三首发分别为10月8日、10月5日、10月2日。以下只做摘要初评，待全文核验；并非整个领域的趋势统计。','hotspots':[{'title':'覆盖边界','text':'最近窗口2026-09-12至2026-10-12；arXiv相关性前200条，OpenReview15关键词各第一页100条，有截断；过去一年窗口2025-10-12至2026-09-11的Semantic Scholar两次429，无引用补充。不能据此称今日无新论文。'},{'title':'谱塑形是否需要复杂曲线','text':'BulkBoost两段增益与Soren连续sigmoid提供可证伪对照：先匹配更新范数、调参及计算预算，再比较下降和验证损失。'},{'title':'几何优势与条件边界','text':'行归一化优势可随协方差轴对齐反转；有限NS训练保证依赖NTK和全批量条件。算法设计应同时验证收益机制与失效区域。'}]},'domain_summaries':{'大模型优化器设计':{'summary':'优先组合关注几何泛化、谱重加权和近似更新对齐；宽泛优化词及粒子物理Muon误命中保留在全量附录，不计作直接研究证据。','subfields':{'符号与范数几何优化':'行归一化、BulkBoost、Soren与有限NS提供不同更新几何与谱近似视角；良性过拟合和AGOP平台分析提供条件性泛化/表征边界。','免调参与梯度裁剪':'今日前三没有直接给出免调参或裁剪新方法；有限NS学习率依赖目标精度，小模型Step Law重拟合也不是免调参。','自适应与状态内存':'候选含AdamW调参及自适应关键词，但前三未报告优化器状态内存压缩结果，不将低维谱参数误写为状态压缩。','大模型训练与优化器设计':'BulkBoost继续预训练最大410M，行归一化只含语言模型最后一层实证；与全批量两层ReLU理论分开理解，不直接推广到端到端大模型。','未细分':'LinUCB和工业优化求解器属于弱相关宽泛匹配，在前十中降级为跟踪。'}}},'top_papers':A,'semantic_reranking':[{'title':p['title'],'screening_rank':p['screening_rank'],'candidate_screening_rank':p['candidate_screening_rank'],'semantic_rank':p['semantic_rank'],'reason':A[p['title']].get('selection_reason',A[p['title']]['summary']),'assessment_scope':'metadata_and_abstract'} for p in top],'abstract_translation_audit':{'scope':'前三原始摘要完整翻译；保持信息顺序、BulkBoost两段落、公式与全部限定词','checked':['行归一化：全批量/随机重排代理/SVD、独立协方差取向、指数小于1、优势反转','BulkBoost：94–97%、分小批量、Nesterov/MP、保Frobenius范数、固定分段一阶结论、30设置/六语料/14M至410M、0.022%与0.073–0.147%及分母','有限NS：五步、全批量/宽两层ReLU/固定随机输出/正定NTK、高概率、所有μ、目标学习率/到达时间、宽度独立性、六宽度五初始化30次、固定数据集']}}
(DAY/'daily-editorial.json').write_text(json.dumps(E,ensure_ascii=False,indent=2)+'\n');(DAY/'search_result.json').write_text(json.dumps(D,ensure_ascii=False,indent=2)+'\n')
