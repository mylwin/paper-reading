"""Persist completed human-in-the-loop style reading; never infer reading from extraction."""
import json, hashlib, shutil
from pathlib import Path

ROOT=Path.cwd()
BASE=ROOT/'07-research/2026-10-10-顶会顶刊写作风格调研_证据'
OUT=ROOT/'07-research/2026-10-11-顶会顶刊写作风格全文续研_证据'
OUT.mkdir(exist_ok=True)
SCRATCH=ROOT/'.planning/venue-writing-next-research-2026-10-11'

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

identities={
 'aaai-2024-09':dict(pages=9,full=True,version='official AAAI Press published PDF',section_pages={'1':'Abstract, Introduction','2':'Background and Related Work; event extraction','3–5':'event processing, commonsense generation and filtering','5–7':'evaluation, human assessment, planning tasks, Conclusion','7–9':'References'},supplement='Official article lists PDF and Video/Poster; no textual supplement link. The complete nine-page PDF does not refer to an appendix or supplementary prose. Prior event-extraction publication is cited prior work, not this paper appendix.',visual=[7]),
 'aaai-2025-16':dict(pages=9,full=True,version='official AAAI Press published PDF',section_pages={'1–2':'Abstract, Introduction, Background','3–6':'factored MDP, specifications, algorithms, conditional theorem, refinement','6–7':'code generation, LP certification, evaluation, Discussion, Related Work','7–9':'References'},supplement='No prose appendix in PDF or official article. Referenced Isabelle formalization repository README inspected as executable proof artifact. Its complete source and machine proofs were not independently verified; this writing-style reading completion does not assert proof verification.',visual=[4,6]),
 'tmlr-2024-01':dict(pages=20,full=True,version='arXiv2309.16941v2; author-deposited TMLR published-header version, not byte-verified against current OpenReview PDF',section_pages={'1–3':'Abstract, Introduction, Related Work, Preliminaries','3–6':'dataset, GNN representation, training objectives and inference','6–10':'benchmark and diagnostic questions, results','10–13':'Discussion, limitations, Conclusion, References','14–20':'Appendices A–D: generator settings, model equations, complete results, diagnostics and solver budgets'},supplement='Appendices A–D embedded and continuously read. Author G4SATBench repository inspected for artifact context; no separately referenced prose supplement located. Matching Published TMLR05/2024 and Reviewed-on7VB5db72lr header, authors/title and official publication index. Current publisher PDF byte equality remains unknown.',visual=[20]),
 'tmlr-2025-01':dict(pages=40,full=False,version='arXiv2506.22982v1; author-deposited TMLR published-header version, not byte-verified against current OpenReview PDF',section_pages={'1–4':'Abstract, Introduction, Related Work, scope of reproducibility','4–14':'original method, extensions, objectives, threat models','15–22':'Claims1–4 reproduction and extended experiments','23–26':'Discussion, author communication, broader impact, limitations','26–32':'References, appendix contents and figures','32–39':'Appendices A–D: environment, algorithms, prompt roles, training and extra results','40':'Appendix E: detailed prompt list referred to external supplementary material'},supplement='All40PDF pages read, including embedded appendices. Appendix E refers to a detailed external prompt list. Author repository fixed commit419d92a93a4f2883599d7cddc25ccb30d3303c4e inventoried; prompt-generating functions in utils/eval_tools.py and dataset-loader code read. data/prompt.txt is only a CLI invocation. Repository correspondence with archived OpenReview supplement remains unverified; external supplement completeness therefore false.',visual=[11,19,21,32,33,39]),
 'tai-2024-07':dict(pages=10,full=True,version='UWA accepted author manuscript; explicit IEEE accepted declaration, not final typeset published PDF',section_pages={'1–3':'Abstract, Impact Statement, Introduction and Background','3–5':'three-module framework, causal inference and modified Q-learning algorithms','5–6':'validation setting and learning-efficiency results','6–8':'scale comparison and NYC dataset routing experiments','8–9':'dynamic-traffic illustrations, application discussion and Conclusion','9–10':'References'},supplement='Complete10page accepted-author PDF contains no appendix or reference to external supplementary prose. UWA landing page explicitly labels Accepted author manuscript and confirms journal publication. PDF provisional VOL05NO10OCT2024 is not final bibliographic metadata; final record5(12):6186–6195,Dec2024.',visual=[1,7,8])
}

dims={
 'aaai-2024-09':{
 'reader_argument':'面向规划与更广AI读者，p1用叙事句子与动作模型的差距建立任务；不先堆语言模型技术名词。',
 'title_abstract':'标题为系统名＋可核验功能；p1摘要从领域模型瓶颈转到自动系统和比较，sometimes限定与半自动方法的关系。',
 'introduction':'p1–2明确新贡献是由既有事件抽取发展到动作模型，不把全部流水线伪称首次提出；叙事实例与PDDL表示连接动机和输出。',
 'related_work':'p2背景与相关工作合并，围绕规划表示与获取来源组织；这是系统稿可选结构，不是AAAI统一要求。',
 'method_theory':'p2–5按事件抽取、指代/短语动词/条件因果处理、常识预测和过滤展开，阈值与选择逻辑留在正文；无正式定理，不能补造证明。',
 'evidence':'p5–7分别检验动作覆盖、专家对前提/效果的适切性、改变初始事实后的规划可解性；9名专家、每域6份评估，不把人评当绝对正确性。',
 'discussion_limits':'p7结论回收自动获取与比较，承认负效果选择及故事上下文仍有限；未来规模用途不等于已部署验证。',
 'paragraph_language':'解释步骤接输入/操作/作用；比较对象明确自动、半自动、人工，结论的sometimes不可删除。',
 'main_appendix':'9页完整正式PDF，无另引附录；必要生成与评估细节在正文。无附录不代表所有类型都应无附录。',
 'actionable':'系统稿先展示输入与输出差距，再把生成质量、人工判断和下游效用分开；标明不可复现基线的近似来源及每项比较的分母。'
 },
 'aaai-2025-16':{
 'reader_argument':'p1–2从安全相关软件正确性和大状态空间连接交互式证明用途，任务是可验证算法开发，不是宣称全新RL算法。',
 'title_abstract':'标题直接验证对象；摘要依次陈述方法论、API案例、可执行细化、实践评估和LP证书。',
 'introduction':'说明表示复杂性与分解验证难点；贡献是验证过程及模块化工具，明确沿用与修改前人算法。',
 'related_work':'背景p2解释ITP/HOL；专门Related Work在p7，与NaRuto早期合并节形成结构反例。',
 'method_theory':'p3–6用7项Specification、6个Algorithm和locale依赖组织；Theorem1 p4要求返回标志True即权重收敛；不保证任意停止状态。',
 'evidence':'形式规范、细化关系与可执行评估承担不同主张；p6Table1分ring/star拓扑、总时间与LP时间、约束与变量规模。',
 'discussion_limits':'p6–7外部LP求解器本身未验证，结果经证书核验；精确有理数有成本，大ring实例求解失败，未来浮点精度需误差界。',
 'paragraph_language':'规范先说明功能，再给条件与公式，后细化实现；数学正确性与实现性能分开，不把verified扩大到未验证LP求解器。',
 'main_appendix':'9页论文无独立附录；部分完整细节由公开Isabelle形式化承载。已审论文与仓库README，不声称重放机器证明或审计全代码。',
 'actionable':'验证类稿件保留定理前提、模块保证和外部组件可信边界；性能表按拓扑/精度口径组织，不强加神经网络消融。'
 },
 'tmlr-2024-01':{
 'reader_argument':'p1–2把不统一任务/数据造成的比较失真作为基准动机，兼顾SAT与GNN读者；先说明能比较什么。',
 'title_abstract':'标题是基准＋推进的功能；摘要依次缺口、7类数据/难度/模型评估和负发现，承认与CDCL差距。',
 'introduction':'贡献围绕数据、可比较流程、诊断发现与公开产物，不以单个SOTA数值替代意义。',
 'related_work':'p2–3区分独立求解与神经辅助求解，讨论既有数据缺陷，然后补SAT预备知识；不是强制所有TMLR稿都这样排列。',
 'method_theory':'p3–6将Dataset/GNN/Tasks/Training/Inference分开，Figure1流程地图＋关键消息传递式，模型全变体放AppendixB；无正式定理。',
 'evidence':'p6–8统一比较，p8–10按行为诊断问题解释；预训练负结果Table5保留。一般benchmark3seeds，而AppendixC.1明确泛化实验1seed，不能统一改为三次。',
 'discussion_limits':'p10–11限制与未来分开，承认小型合成数据、独立GNN与工业求解差距；不把探讨机制写成证明。',
 'paragraph_language':'问题标签导向结果和解释；失败的训练种子在表中保留“-”，不删失败样本来强化稳定性。',
 'main_appendix':'A p14–15生成参数/统计；B p15–16模型；C p15–19完整结果；D p19–20诊断与solver预算，全部读完。',
 'actionable':'基准稿按可比性与研究问题组织，正文给关键口径、附录给完整配置；solver比较明确预算、分摊时间与无预算结果。'
 },
 'tmlr-2025-01':{
 'reader_argument':'p1–4先分复现目标与新增工作，读者可判断哪些发现来自原方法、哪些来自扩展；这不是所有TMLR论文的固定类型。',
 'title_abstract':'题名直接说明复现与增强；摘要回顾原方法、复现结论及扩展，不隐藏与前人成果的关系。',
 'introduction':'p2–3用ReproducibilityStudy/ExtendedWork标签划分贡献；p4单列Scope给原始四项主张。',
 'related_work':'p3–4按攻击转移轴与VLM场景定位，而非泛泛列举模型；事实归属应保留。',
 'method_theory':'原方法p4–6与扩展p7–14分别解释；攻击对象、prompt/model/image迁移及targeted/untargeted需逐协议对齐，不能用一句统一覆盖。',
 'evidence':'p15–22逐Claims1–4记录复现，再报告扩展；下述正文/表格/算法矛盾未解决，不将原文表述视为可信规则。',
 'discussion_limits':'p23–26明确作者沟通、局限与broader impact；给复现偏差解释，应区分观察与待证原因。',
 'paragraph_language':'复现标签及实验问题提高可追踪性，但all metrics、百分数和算法符号必须与具体证据核对；原文不是语言正确性金标准。',
 'main_appendix':'40页PDF全读含A–E；E在p40另指详细外部prompt清单，作者仓库函数已读但归档对应未确认，因此只记主文与内置附录完成。',
 'actionable':'对复现/扩展稿建立原主张—复现协议—结论—新增证据矩阵；遇数字/符号冲突给作者核查项，不静默更改算法或挑有利数字。'
 },
 'tai-2024-07':{
 'reader_argument':'p1–3以人类学习类比引入model-based/free和因果模型，再连接路由问题；降低入口术语负担，有背景教程但未验证其全刊普遍性。',
 'title_abstract':'标题方法＋框架；摘要报告NYC数据与76%/66%，p1独立Impact Statement讲意义，不与摘要合并；first/domain-agnostic的强度应另审。',
 'introduction':'p1–2按已知因果结构局限→离线发现与在线学习整合→三项贡献，先解释架构作用。',
 'related_work':'引言和Background嵌入相关方法比较，没有独立Related Work章节；不能从官方要求推导固定节名。',
 'method_theory':'p3–5结构发现、BBN推断、修改Q-learning三模块；Algorithms1–2给输入和选择条件，先验目标顺序与人工结构约束实际使用，未给普适收敛证明。',
 'evidence':'p5–8分学习效率、规模与NYC路由；Taxiv3离线500000随机步，NYC1000000采样＋100000训练episodes，531行程；训练和离线数据成本不得隐去。',
 'discussion_limits':'p8随机交通仅示例，p9把新领域列未来；实证限路由与离散任务，不能把医疗/金融用途写成已实现收益。',
 'paragraph_language':'每项系统特点有解释，但agnostic to prior knowledge仅指不需完整地图，不等于无先验；76%同最短距离是观察比例，不是全输入最优保证。',
 'main_appendix':'10页接受作者稿无引用外部附录；保留IEEE接受声明，不能用其临时卷期、作者信息及页眉反推当前匿名/终稿格式。',
 'actionable':'应用系统稿分模块、成本和评价层级，Impact说明证据支持的用途；曲线实测与外推线分别标注，优势按规模/先验/指标限定。'
 }
}

counterexamples={
 'aaai-2024-09':[
  {'location':'p7 Table2','finding':'NaRuto(L) West3.57超过StoryFramer3.26，但Aladdin3.34低于4.04；人工GT在两域更高。','editing_rule':'不能把sometimes on par强化为超过所有人工/半自动模型。'},
  {'location':'p5–7 baselines and Table3','finding':'H2020不可用时由描述近似；不同模型的Total Tasks不同；规划器每任务仅一任意plan。','editing_rule':'说明近似基线、分母及distinct plans下界，不能当共同任务集成功率。'}],
 'aaai-2025-16':[
  {'location':'p4 Theorem1','finding':'API返回True的收敛前提不能删除。','editing_rule':'定理结论条件化，非任意timeout结果都获该界。'},
  {'location':'p6–7 LP certification/evaluation','finding':'LP solver未验证，只验证其证书；star2^40规模不能泛化成所有ring或任意MDP。','editing_rule':'验证对象、精确求解成本和拓扑分别陈述。'}],
 'tmlr-2024-01':[
  {'location':'p20 Table12; AppendixD','finding':'NeuroSAT在限制32flips时优于Sparrow；无限预算Sparrow与CaDiCaL均求解全部。GPU时间还按batch分摊。','editing_rule':'不能把限定预算胜率改成全面替代经典solver或同口径更快。'},
  {'location':'p6 vs p16 AppendixC.1','finding':'一般benchmark3seeds，泛化实验1seed。','editing_rule':'重复次数按实验注明，不能统一表述。'},
  {'location':'p7 general statement vs p8 Table2 and AppendixC.3','finding':'medium kClique的NeuroSAT有训练失败“-”。','editing_rule':'stable all等表述需保留例外，失败结果不可移除。'}],
 'tmlr-2025-01':[
  {'location':'p18 prose vs p19 Tables3–4','finding':'all metrics的表述与unknownVQA0.9680→0.8824、0.9880→0.8920及specificVQA0.9457→0.9342不符。','status':'observable prose/table discrepancy','editing_rule':'单项退步和整体改善同时保留，不照抄全指标优势。'},
  {'location':'p20 prose vs p21 Table5 and p23 discussion','finding':'表中SCMix overall0.7055高于CutMix0.5565，与p20CutMix优于SCMix说法不一致。','status':'unresolved','editing_rule':'给作者查询，不自行选择解释原因。'},
  {'location':'p21 Table6 vs p24 discussion','finding':'0.8433→0.9685是12.52百分点，相对14.85%；与13.5%的讨论措辞不一致。','status':'arithmetic check','editing_rule':'明确百分点/相对百分比并核对摘要正文。'},
  {'location':'p22 Table8 vs p39 AppendixD.4.5','finding':'Flamingo迁移至BLIP2/InstructBLIP，DUAP主表0.1013/0.0938、附录0.00/0.00；可能协议不同，当前未清楚对齐。','status':'protocol correspondence unresolved','editing_rule':'不得默认为数据错误或擅选有利表；确认targeted/untargeted等配置。'},
  {'location':'p11 Eqs11–12 vs p33 AppendixB Algorithm3','finding':'主文Ld=sum(1-cos)被减去，与最小化时促进对齐的解释不直接一致；附录累积cos并用descent，主算法另用ascent。视觉已确认符号。','status':'objective/algorithm correspondence requires author confirmation; not a correctness proof','editing_rule':'公式与实现说明冲突单独报出，不静默翻符号。'}],
 'tai-2024-07':[
  {'location':'p6–8 data collection, Fig6','finding':'Q-Cogni先有离线随机采样与训练；Fig6低规模明显更慢，大规模有外推虚线。','editing_rule':'分摊离线/训练/在线成本并标实测/外推，不能声称全规模更快。'},
  {'location':'p4 and p6 constraints; p7–8 NYC setting','finding':'实验有人工结构约束、排序目标和预先拟合BBN；不需完整地图不等于无先验。','editing_rule':'按实际使用的先验限定自动/agnostic主张。'},
  {'location':'p8 outcomes and traffic examples; p9 future','finding':'531行程76%与Dijkstra同距离、24%更长；66%比Q-learning短；交通只给2示例，未验证燃油/成本/医疗收益。','editing_rule':'写经验比例及样本场景；部署和社会效果留作未验证可能性。'}]
}

ledger=[]
for venue in ['aaai','tmlr','tai']:
    directory=BASE/'corpus'/venue
    rows=[json.loads(l) for l in (directory/'records.jsonl').read_text().splitlines()]
    for row in rows:
        pid=row['id']
        if pid=='tai-2024-03':
            row.update(eligibility='ineligible_nonresearch_front_cover',type_status='metadata title and absence of authors confirmed nonresearch',replacement_status='pending; original locked record retained for audit',exclusion_date='2026-10-11')
        if pid not in identities: continue
        meta=identities[pid]
        if pid=='tai-2024-07':
            row.update(pdf_path='.claude/skills/paper-writing-shared/tmp/corpus/tai/pdf/Q_Cogni_An_Integrated_Causal_Reinforcement_Learning_Framework.pdf',markdown_path='.claude/skills/paper-writing-shared/tmp/corpus/tai/markdown/Q_Cogni_An_Integrated_Causal_Reinforcement_Learning_Framework.md',page_count=10)
        pdf=ROOT/row['pdf_path']; md=ROOT/row['markdown_path']
        digest=sha(pdf)
        if row.get('pdf_sha256'): assert row['pdf_sha256']==digest
        previous={k:row.get(k) for k in ['semantic_review_status','version','acquired_version','version_verification','eligibility']}
        if venue=='tmlr':
            row.update(acquired_version='author_arxiv_deposit_with_matching_TMLR_publication_header',version_verification='title/authors/date/review_forum match publication; current publisher bytes not checked',eligibility='accepted_published_identity_content_confirmed_author_deposit')
        elif venue=='tai':
            row.update(acquired_version='accepted_author_manuscript_verified_by_PDF_declaration_and_UWA_repository',version_verification='content accepted declaration and DOI plus institution accepted-manuscript label; OpenAlex submittedVersion contradicted',eligibility='accepted_author_version_content_confirmed')
        row.update(pdf_sha256=digest,full_text_style_review_complete=True,external_appendix_review_complete=meta['full'],semantic_review_status='continuous complete PDF writing-style reading; relevant appendix complete' if meta['full'] else 'continuous complete PDF reading; referenced external prompt supplement unresolved',reviewed_date='2026-10-11',semantic_reviewer='Codex root',review_granularity='Continuous all-page semantic writing-style reading, including references and embedded appendices; not independent theorem/code verification or experimental reproduction',appendix_coverage=meta['supplement'],semantic_review_path=str((directory/'semantic-reviews'/f'{pid}.json').relative_to(ROOT)))
        review=dict(id=pid,title=row['title'],venue=venue,year=row['year'],split=row['split'],review_date='2026-10-11',reviewer='Codex root',mode=row['review_granularity'],full_text_style_review_complete=True,external_appendix_review_complete=meta['full'],publication_identity_and_acquired_version=meta['version'],publisher_current_byte_equivalence='not established' if venue in ['tmlr','tai'] else 'downloaded from official DOI-linked article',pdf_path=row['pdf_path'],pdf_sha256=digest,markdown_path=row['markdown_path'],markdown_sha256=sha(md),coverage=[dict(version=meta['version'],pages=list(range(1,meta['pages']+1)),section_pages=meta['section_pages'],method='continuous reading; any truncated tool text reread separately; figures/formulas ambiguous in extraction checked in actual page render')],supplement_audit=dict(source=row.get('accepted_url',row.get('publication_url')),result=meta['supplement']),ten_dimensions=dims[pid],counterexamples=counterexamples[pid],confidence=dict(paper_observation='high for documented page content',venue_generalization='low; purposive pilot, no frequency estimate, no complete50-paper or heldout validation'),official_requirement=False,holdout_validation=None,previous_status=previous)
        write(directory/'semantic-reviews'/f'{pid}.json',review)
        ledger.append(dict(id=pid,venue=venue,year=row['year'],split=row['split'],pdf_pages=meta['pages'],main_complete=True,full_and_appendix_complete=meta['full'],review_path=row['semantic_review_path'],pdf_sha256=digest))
    (directory/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

write(OUT/'batch-ledger.json',dict(updated='2026-10-11',new_main_reviews=5,new_full_reviews=4,new_pdf_acquisitions=3,new_holdout_reviews=0,papers=ledger,eligibility_issue=dict(id='tai-2024-03',reason='Front Cover; not original research',replacement='pending; locked slot retained, does not count as research paper'),unfinished_agent_materials='AAAI/NeurIPS/TAI agent intermediates not certified by root and excluded; account usage interruption'))
write(OUT/'lock-integrity-baseline.json',{v:sha(BASE/'corpus'/v/'manifest.json') for v in ['aaai','tmlr','tai','neurips']})
shutil.copy2(SCRATCH/'identity-web-sources.json',OUT/'identity-web-sources.json')
for folder in ['additional-renders','tmlr-renders']:
    target=OUT/'visual-review'/folder
    if not target.exists(): shutil.copytree(SCRATCH/folder,target)
print(json.dumps(ledger,ensure_ascii=False,indent=2))
