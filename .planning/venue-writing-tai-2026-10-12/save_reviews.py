from pathlib import Path
import json, hashlib, shutil

ROOT = Path('/Users/myl/Documents/paper')
OUT = ROOT / '07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai'
OUT.mkdir(parents=True, exist_ok=True)

def dim(observation, pages, recommendation, counterexample, condition):
    return dict(observation=observation, pdf_pages=pages, recommendation=recommendation,
                counterexample=counterexample, condition=condition,
                rule_kind='corpus_observation_with_editorial_advice',
                confidence='high_for_this_paper; low_for_venue_generalization')

sudoku = {
 'id':'tai-2025-16','title':'Efficient Solution Validation of Constraint Satisfaction Problems on Neuromorphic Hardware: The Case of Sudoku Puzzles',
 'doi':'10.1109/TAI.2025.3536428','year':2025,'split':'derivation',
 'pdf':'.claude/skills/paper-writing-shared/tmp/corpus/tai/pdf/Efficient_Solution_Validation_of_Constraint_Satisfaction_Problems_on_Neuromorphic_Hardware_The_Case_of_Sudoku_Puzzles.pdf',
 'review_date':'2026-10-12','mode':'continuous_all_page_semantic_writing_style_review',
 'main_text_style_review_complete':True,'full_text_style_review_complete':True,
 'external_appendix_review_complete':False,'full_text_and_relevant_appendix_reviewed':False,
 'coverage':{'pdf_pages':list(range(1,13)),'cover':1,'article_pages':'PDF 2–12 = manuscript 1–11','body':'PDF 2–11','references_and_biographies':'PDF 11–12','embedded_appendix':'none','missing':'独立 Supplementary Material：Latin square 方法对比，正文结论 PDF 11 明确引用'},
 'version_audit':{'original_locator_label':'submittedVersion','verified_version':'institutional_author_accepted_manuscript','eligibility':'published_research_identity_verified; relevant_supplement_pending','authors':['Riccardo Pignari','Vittorio Fra','Enrico Macii','Gianvito Urgese'],'sources':['https://iris.polito.it/handle/11583/2992746','https://api.crossref.org/works/10.1109%2Ftai.2025.3536428'],'evidence':'机构封面及正式记录明确 IEEE postprint / Author\'s Accepted Manuscript；Crossref 和机构记录一致为 2025, 6(8), 2061–2072。PDF 篇内保留 JOURNAL OF LATEX CLASS FILES 2021 / SHELL 样例页眉，属于未清理模板，不能作为发表身份或推荐格式。','publisher_vor_byte_equivalence_verified':False},
 'ten_dimensions':{
 'reader_argument':dim('用 CSP、SNN、神经形态平台逐级解释，为 AI 与硬件交叉读者建立共同术语；论证新增验证/停机模块如何减少平台外操作。',[2,3,4],'交叉领域工程稿先给读者足以理解瓶颈的定义，再落到具体新增模块。','不是所有 CSP 都需数独背景；本文无通用最优性证明。','硬件/系统方法论文'),
 'title_abstract':dim('标题将广义 CSP 范围与数独个案并列；摘要按平台基础、三种改动、先软件后硬件验证、三类收益组织，最后给提取 spike 与提取时间范围。',[2],'摘要数字必须带具体平台、测量量与实例范围。','Impact Statement 声称能效提升，但正文未直接报告焦耳/功率测量；传输和提取时间不是整体能耗。','有实际平台测量的工程稿'),
 'introduction':dim('应用范围→既有 spike 求解器→外部验证开销→图1新旧管线→选择 Sudoku 个案。贡献以模块职责展开，不依赖固定贡献列表。',[2,3],'用原系统与改进系统对照图支撑差异，写清验证位置变化。','不要将减少外部依赖写成完全没有外部控制/传输。','系统改造研究'),
 'related_work':dim('Background 中按 CSP 形式化、网络映射、既有三项不足组织引文；没有独立 Related Work 章。结论指向补充的更广 Latin square 对比。',[3,11],'相关工作位置由读者理解依赖决定，合并背景也可；保留补充比较的正文指针。','未取得补充对比，不能声称已审查其基线完整性。','领域背景和技术依赖紧密结合的稿件'),
 'method_theory':dim('Implementation 分网络单元、吸引子动力学、约束稳定、停机与内置验证、端到端复述。图2/3与参数表I/II配合，逐块解释信息和抑制连接。',[4,5,6],'模块图、变量定义与参数表要互相引用，先局部机制再解释全管线。','没有正式定理，不应为适配期刊强加理论保证；同一模块在III-C和III-D反复说明可酌情压缩。','模块化工程方法'),
 'evidence':dim('GeNN 四配置×九题×300次共10800，先消融；SpiNNaker 两配置×九题×100次共1800，后部署；另5400次检查数值精度。表III/IV报告逐题成功数，图示分难度汇总。',[7,8,9],'把软件消融、硬件迁移、精度诊断分层，保留各自重复次数和实例数。','实际硬件 easy 成功率82.33%降至78.67%，不能润色成所有难度均提升；重复运行不是更多独立题目。','软件与硬件双阶段验证'),
 'discussion_limits':dim('Discussion 解释初始线索失效、停机只在找到解后触发、Memory 写入延迟及 SpiNNaker 144步限制；明确仅适用于 Latin square，其他约束需专门设计。',[8,9,10,11],'讨论把部署差异连接到具体机制，局限写清适用问题类和必要改造。','浮点16/32/64实验仅支持精度敏感性，不能独立证明定点量化是唯一原因。','存在平台迁移差异的工程稿'),
 'paragraph_language':dim('段落多以模块功能或观测结果开头，用图表回指解释因果机制；正文承认硬件 easy 类下降。部分形容词如 huge/disruptively 强于测量范围。',[5,8,10,11],'保留直接结果与有条件机制解释，将夸张形容词改为具体指标范围。','不要模仿强语气；推测和实测应标记证据等级。','实证方法与系统稿'),
 'main_appendix':dim('正文容纳大图、神经参数表、硬件修正值、逐题结果和限制；更广方法对比外移 Supplementary Material。',[4,5,6,7,8,10,11],'正文保留足以理解新增机制与主结果的参数；外移材料仍需实际版本和可追溯定位。','外部 Supplementary Material 尚未取得，不能将其内容当证据或标完整。','版面有限而部署参数关键的稿件'),
 'actionable':dim('从主张、配置、平台和指标四项核对摘要/Impact Statement/结果/结论。',[2,8,9,11],'原创示例：On the nine evaluated Sudoku instances, the SpiNNaker pipeline reduced spike extraction time while improving medium and hard success rates; easy-instance success decreased.','不能写 The pipeline improves every class and demonstrates measured energy savings。','仅复用本研究实际提供的指标，不推广为 TAI 强制措辞')},
 'visual_checks':[{'pdf_page':8,'file':'renders/sudoku-pdf8.png','checked':'Table IV and Figure 6; hardware easy decline and medium/hard success counts'}],
 'independent_experiment_or_proof_verification':False,'holdout_validation':None}

drugmap = {
 'id':'tai-2025-01','title':'DrugMAP: Deep Multimodal Transformers for Drug-Target Mechanism of Action Prediction',
 'doi':'10.1109/TAI.2025.3565671','year':2025,'split':'derivation',
 'pdf':'.claude/skills/paper-writing-shared/tmp/corpus/tai/pdf/DrugMAP_Deep_Multimodal_Transformers_for_Drug_Target_Mechanism_of_Action_Prediction.pdf',
 'review_date':'2026-10-12','mode':'continuous_all_page_semantic_writing_style_review',
 'main_text_style_review_complete':True,'full_text_style_review_complete':True,
 'external_appendix_review_complete':False,'full_text_and_relevant_appendix_reviewed':False,
 'coverage':{'pdf_pages':list(range(1,16)),'cover':1,'article_pages':'PDF 2–15 = manuscript 1–14','body':'PDF 2–13','references':'PDF 13–15','embedded_appendix':'none','missing':'正文 S1–S7 各项未完成内容审阅；HAL API已找到真实附件目录，正在获取'},
 'version_audit':{'original_locator_label':'submittedVersion','verified_version':'author_accepted_manuscript_clean_copy','eligibility':'published_research_identity_verified; relevant_supplements_pending','sources':['https://hal.science/hal-05067564v1','https://api.archives-ouvertes.fr/search/?q=halId_s%3Ahal-05067564&fl=*&wt=json','https://api.crossref.org/works/10.1109%2Ftai.2025.3565671'],'evidence':'实际PDF各篇内页带 ACCEPTED MANUSCRIPT / CLEAN COPY。HAL 封面/一手 API title+7作者+DOI一致，Crossref正式记录为2025年11月,6(11),3087–3099。HAL accepted稿14页与正式页数13不同，不宣称版面等价。HAL 2025-10-13 Submitted on 是仓储提交日期，不是期刊审稿提交状态。','author_name_note':'PDF/HAL Bandopadhyay；正式DOI作者 Bandyopadhyay。保存拼写差异，不静默纠正原稿。','publisher_vor_byte_equivalence_verified':False},
 'ten_dimensions':{
 'reader_argument':dim('先解释 DTI 二分类与作用机制 MoA 的区别，面向药理与机器学习交叉读者；论证新数据资源与多模态建模缺一不可。',[2,3,4],'先明确目标输出与现有任务的区别，再介绍表示学习。','不要把分类机制标签当作因果机制被验证或临床效果被测量。','应用领域的新任务/数据集方法稿'),
 'title_abstract':dim('命名方法＋网络类型＋目标任务；摘要现实成本、现有二分类不足、数据缺口、融合架构、资源与改进数字；独立 Impact Statement 解释潜在应用。',[2],'现实意义用潜力表述，并将摘要数字绑定表中任务/指标。','AUC提高3.5%的表述实际0.950→0.985为3.5个百分点，非相对3.5%；临床影响没有前瞻实验。','数据资源与模型同时贡献的稿件'),
 'introduction':dim('药物机制识别需求→已有DTA/DTI未覆盖MoA→预训练与跨注意力动机→四条数据、表示、编码器与消融贡献。',[2,3],'贡献条目对应后文可检查的数据、模块与实验，不把堆叠组件数量当创新证据。','预训练模型来源与本文训练要区分，不能把借用模型写为本文新提出。','复用预训练编码器的方法稿'),
 'related_work':dim('按DTI二分类/回归分任务，再补充tuple输入与全图输入在未见节点上的区别，定位MoA标签缺口。',[3,4],'相关工作先按问题与数据假设分类，再解释本稿选择的条件。','不得宣称所有图输入都无法处理新节点；这是文中的限制论证而非普适定理。','多个邻近预测任务并存'),
 'method_theory':dim('独立Dataset Description在Methodology之前，数据来源版本、过滤、类合并、增强、负样本先清楚；方法再分预训练编码器、WRX-Attn、融合预测头，Eq3–10与图1–5逐块对应。',[4,5,6,7,8,9,10],'保留自建数据集的规则和任务定义，区分标准公式与本文新增WRX模块。','正文IV称predict affinity但III说明无定量亲和力且为分类，润色应标记任务用词冲突，不能暗加回归能力。','自建数据集是贡献核心'),
 'evidence':dim('Experiments先基准、训练、指标；Results按自建数据、公共二分类数据、五个测试集案例、编码器/融合/增强消融分开。共享设置说明基线重新训练，80:20并train内5折。',[10,11,12,13],'每张表说明DTI还是MoA，样本划分单位、增强时机、基线适配和指标聚合需要可核验。','正文VI-A写MoA AUC0.940/AUPR0.931，实际表II为0.985/0.940；不能静默选数字。新增药物/靶标泛化没有cold split在主文明确说明。','多任务、多数据集报告'),
 'discussion_limits':dim('无独立Discussion，结论总结数据+方法+主指标并写长蛋白token限制，另III说明简单离子/极长序列被排除。病例是选自测试集的预测与文献解释。',[5,12,13],'允许局限放结论但应连到前文过滤与任务边界；保持case studies来源明确。','五个测试集案例不是新分子实验验证或临床试验；嵌入相似0.9973不直接证明突变保留生物功能。','监督分类应用稿'),
 'paragraph_language':dim('先给模块作用再解释表示和公式，较多领域教程；论证中robust/generalize词语较强，因果机制和跨模态增益需要消融支撑。',[6,7,8,9,12,13],'压缩重复的transformer教程，保留新模块的必要维度与操作；likely解释不要提升为确定因果。','不能复用潜在临床作用强度作为普通语言优化；新增生物/因果结论需新证据。','交叉领域应用稿'),
 'main_appendix':dim('主体保留数据处理、机制定义、网络与比较主表；S1排除清单、S2修正、S3基线参数、S4消融、S5逐类曲线、S6划分/超参、S7案例细节外置。',[4,11,12,13],'建立每个S文件与正文主张映射；重要消融结论不能只靠文件名字验证。','已定位HAL附件，但当前还不能确认S6归属或完成审阅，不能计全文+相关补充完成。','补充文件分散的期刊工程'),
 'actionable':dim('用任务×数据划分×指标对齐文字、表、摘要与结论，清楚区分“百分点”与相对提升。',[2,11,12,13],'原创示例（作者核对数值后）：On the constructed MoA dataset, Table II reports an AUC of 0.985; the absolute increase over MMA-DPI is 0.035.','在正文-表格冲突未解决前只能写编辑查询，不自行把全稿改成0.985。','数值一致性审计建议，非TAI独特措辞规范')},
 'counterexamples':[{'pdf_pages':[11,12,13],'issue':'Table II MoA AUC/AUPR=0.985/0.940；VI-A文本0.940/0.931；结论再次为98.5%/94.0%。','action':'保留数字冲突及定位，询问作者确认，禁止语言润色静默改值。'},{'pdf_pages':[11,12,13],'issue':'0.985-0.950=0.035，为3.5个百分点；0.998-0.985=0.013，为1.3个百分点，而移除graph编码器段落写1.6%。','action':'标记指标单位与减法不一致；补充审阅后再判断是否不同设置。'}],
 'visual_checks':[{'pdf_page':11,'file':'renders/drugmap-pdf11.png','checked':'Table II and III; DTI vs MoA columns and numeric prose conflict'}],
 'independent_experiment_or_proof_verification':False,'holdout_validation':None}

for obj in (sudoku,drugmap):
 p=ROOT/obj['pdf'];obj['pdf_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 obj['support_sample_ids']=[obj['id']]
 (OUT/(obj['id']+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
render=OUT/'renders';render.mkdir(exist_ok=True)
for name in ['sudoku-pdf8.png','drugmap-pdf11.png']:
 shutil.copy2(ROOT/'.planning/venue-writing-tai-2026-10-12'/name,render/name)
print('saved two all-page reviews; both exclude missing external supplements from full completion')
