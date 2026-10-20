import json,re,subprocess,sys
from pathlib import Path
from urllib.parse import urljoin
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-20'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import paper_config
D=json.loads((DAY/'search_result.json').read_text());E=json.loads((RUN/'editor-input.json').read_text())
for a in E['top_papers'].values():
 if a.get('abstract_zh'):a['abstract_zh']=a['abstract_zh'].replace('\\\\mathcal','\\mathcal')
original=D['top_papers']
(RUN/'initial-screening.json').write_text(json.dumps(D,ensure_ascii=False,indent=2)+'\n')
order=[2,3,1,10,6,7,8,9,4,5]
reasons={
 1:'复值步长机制有可迁移线索，但实值Transformer适用性未证实，排在直接LLM状态压缩与裁剪几何方案之后。',
 2:'直接研究极低秩LLM训练，子空间跟踪机制和1B预训练提供明确核验入口，列为第一。',
 3:'免问题参数、范数球裁剪和本地更新有直接机制联系；将通信收益与本地计算预算分开审查，列第二。',
 4:'Muon在本文指粒子，核心是π介子形状因子及g-2；并非优化器研究，排除。',
 5:'Muon是中微子散射输出粒子运动学，未研究Muon优化器，词义误命中，排除。',
 6:'闭包支持条件与稀疏更新直接相关，但OPT 2026经官网核验为Workshop，按配置排除正式推荐。',
 7:'协变量漂移与轨迹稳定性可补机制线索，但OPT 2026为Workshop，按配置排除。',
 8:'可训练边界和有限NS截断有诊断价值，但DynaFront经主办方页面核验为Workshop，按配置排除。',
 9:'符号投票差额与更新幅度直接相关且有LLM/硬件数字，但OPT 2026为Workshop，按配置排除。',
 10:'极分解后归一化直接关联Muon设计，适合定点对照；此为较早arXiv工作的近期OpenReview提交，不能称今日新发现的算法。'
}
for rank,p in enumerate(original,1):p['candidate_screening_rank']=rank
D['top_papers']=[original[i-1] for i in order]
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
raw={p['title']:p for source in ['arxiv','openreview'] for p in json.loads((RUN/(source+'-raw.json')).read_text())}
physics_patterns=['muon collider','muon beam','muon anomalous','muon magnetic','muon-to-electron','muon-to-positron','muography','muon detector','muon detection','muon mean lifetime','muon decay','muon physics','hadronic light-by-light','neutrino','pion','radiative muon','muone','muon cooling','muon propagation','muon bulb','muon flux']
for p in D['all_papers']+D['top_papers']:
 full=raw.get(p['title'],p);text=(p['title']+' '+full.get('summary','')).lower()
 cats=full.get('categories',[])
 phys=any(re.search(r'\b'+re.escape(t)+r'\b',text) for t in physics_patterns) or (any(c.startswith(('hep-','nucl-')) for c in cats) and 'optimizer' not in text)
 if phys:p['editorial_status']='词义误命中（不推荐）'
 venue=(full.get('venue') or '')+' '+full.get('source_comments','')
 if 'OPT 2026' in venue or 'DynaFront' in venue or 'workshop' in venue.lower():p['editorial_status']='配置排除（workshop）'
 if p.get('pdf_url','').startswith('/'):p['pdf_url']=urljoin('https://openreview.net',p['pdf_url'])
 related=[];seen=set()
 for q in p.get('related_papers',[]):
  path=Path(q.get('path',''))
  if not path.is_file() or q['title'] in seen:continue
  try:rel=path.relative_to(ROOT).as_posix()
  except ValueError:continue
  if rel not in tracked:continue
  if path.name=='search_result.json':
   note=path.parent/'今日检索.md'
   if not note.is_file() or note.relative_to(ROOT).as_posix() not in tracked or q['title'] not in note.read_text():continue
  related.append(q);seen.add(q['title'])
 p['related_papers']=[] if phys else related
for rank,p in enumerate(D['top_papers'],1):
 p['semantic_rank']=rank;p['formal_recommendation']=not bool(p.get('editorial_status'))
 n=p['candidate_screening_rank'];title=p['title']
 if title not in E['top_papers']:
  E['top_papers'][title]={'summary':reasons[n], 'selection_reason':reasons[n], 'reading_decision':'选读' if p['formal_recommendation'] else '跟踪', 'evidence_limit':'题目与摘要初评；待全文核验。'}
E['semantic_reranking']=[{'title':p['title'],'screening_rank':p['screening_rank'],'candidate_screening_rank':p['candidate_screening_rank'],'semantic_rank':p['semantic_rank'],'reason':reasons[p['candidate_screening_rank']],'formal_recommendation':p['formal_recommendation']} for p in D['top_papers']]
A=json.loads((DAY/'archive-result.json').read_text());V={x['title']:x for x in json.loads((RUN/'arxiv-source-verification.json').read_text())}
def relation(title,path,shared,why):
 asset=ROOT/path;assert asset.is_file() and path in tracked
 return {'title':title,'path':str(asset),'source':'note' if asset.name=='精读.md' else 'pdf','shared_terms':shared,'relationship':why}
refs=[
 [relation('Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon','01-raw/2026-10/Memory_Efficient_Subspace_Optimization_with_Residual_Correction_From_Momentum_SGD_to_Muon.pdf',['subspace','memory-efficient optimizer'],'对照：比较持续主子空间跟踪与残差校正的误差来源及相同秩/状态预算下的收益，尚未核实技术包含关系。')],
 [relation('Parameter-Free Clipped Gradient Descent Meets Polyak','03-notes/2026-09/Parameter_Free_Clipped_Gradient_Descent_Meets_Polyak/精读.md',['parameter-free','clipping'],'互补：比较裁剪/免问题参数的定义；集中式步长分析不能直接充当本地训练通信定理。'),relation('PowerStep: Memory-Efficient Adaptive Optimization via lp-Norm Steepest Descent','03-notes/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent/精读.md',['norm geometry'],'对照：范数球投影和范数最速方向属于不同更新构造，先核对计算代价与几何定义。')],
 [relation('PowerStep: Memory-Efficient Adaptive Optimization via lp-Norm Steepest Descent','03-notes/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent/精读.md',['adaptive optimization','update geometry'],'互补：分别审查更新方向与幅度的自适应信号，不预设复值逐参数规则适用于实值Transformer。')]
]
for p,rel in zip(D['top_papers'][:3],refs):
 a=next(x for x in A['archived'] if x['title']==p['title'])
 p.update(archive_status=a['status'],archive_detail=a['detail'],archive_path=a['path'],archive_month=a['month'],source_verified=True,license_urls_verified=V[p['title']]['license_urls'])
 p['related_papers']=rel
 for q in D['all_papers']:
  if q['title']==p['title']:q['related_papers']=rel
D['formal_recommendation_count']=sum(p['formal_recommendation'] for p in D['top_papers'])
D['physics_false_positive_count']=sum(p.get('editorial_status')=='词义误命中（不推荐）' for p in D['all_papers'])
D['search_limitations']=[
 'arXiv export API首次503、后续429/超时；官方普通搜索页15词中8词成功、7词SSL失败。普通页面会返回旧论文的近期修订，未将此直接当首发日期。',
 'arXiv完整15词高级检索502；短查询Muon/SignSGD/gradient clipping按首次提交日期取得143条。普通页结果仅保留该首发过滤集合，或2026-10编号的记录；其他无法核验窗口的9月记录保守排除。标题/摘要再按15配置关键词字面匹配，使用技能原评分筛选并限200；最终输入134条。各关键词首页上限200，非穷尽检索。',
 'OpenReview逐关键词首页上限100，14词成功、AdamW词429；并集窗口内35条。日期是OpenReview记录日期，不代表论文首次公开日期。详情接口403挑战，未绕过；本次前三均取自已核验的arXiv页面及PDF。',
 'Semantic Scholar两次429，过去一年高影响力补充未取得结果；未以旧论文近期修订冒充高影响力补充。',
 '预筛前十保留审计，其中2条粒子物理误命中、4条Workshop按配置排除，正式阅读建议4项；不从第十一名补选。'
]
E['overview']['trend']=f"检索日期2026-10-20，近期窗口2026-09-20至2026-10-20。{D['total_unique']}条唯一记录中{D['total_known']}条命中已有PDF、笔记或历史推荐，{D['total_candidates']}条未命中候选。预筛前十经逐篇题目/摘要复排后，排除2条粒子物理误命中与4条Workshop，保留4项阅读建议，前三为Ori、NBCLT和AURA。仅就这组摘要，可关注低秩状态预算、范数球裁剪与步长幅度自适应三条线索；该样本不足以证明全领域趋势。来源覆盖受限：arXiv采用官方HTML恢复并保守核验首发窗口，OpenReview部分首页截断，Semantic Scholar限流无年度补充；详见今日落盘和检索覆盖记录。"
E['overview']['hotspots']=[{'title':'低秩状态预算与子空间漂移','text':'Ori直接面向1B LLM预训练，先比较跟踪频率、秩预算和残差校正；差距缩小比例与训练加速不同。'},{'title':'裁剪几何与本地计算预算','text':'NBCLT连接预测校正、范数球投影与通信复杂度，问题参数无关结论依赖本地计算随时域缩放。'},{'title':'步长幅度与方向规则的分工','text':'AURA利用复值连续更新的一致性；Muon+提供极分解后归一化的对照。任务、几何和尺度各异，不能合并成统一实证结论。'}]
E['domain_summaries']['大模型优化器设计']={'summary':'直接LLM优化器设计优先Ori；NBCLT按定理条件选读，AURA按复值幅度自适应机制选读。Muon+属于较早工作的近期OpenReview提交，定点比较归一化；预筛中的粒子物理与Workshop排除项只作审计。主题计数是关键词命中量，含同词异义，不等于直接相关工作量。','subfields':{'符号与范数几何优化':'NBCLT的范数球欧氏投影、AURA的方向/幅度区分以及Muon+的极分解后归一化值得分别对照；OPT/DynaFront相关候选按配置排除。','免调参与梯度裁剪':'NBCLT不依赖问题参数的设置有预算缩放条件；不能据摘要写成所有超参数无需选择。','自适应与状态内存':'Ori提供低成本主子空间连续跟踪，重点查严格低秩下的精度、状态量和额外正交化代价；AURA摘要未给LLM结果。','大模型训练与优化器设计':'Ori为直接预训练/微调证据；其他任务中的机制需先小规模验证，不能直接外推大型Transformer。','未细分':'宽泛训练词及Muon同词异义只支持检索命中；附录状态标记用于区分相关性。'}}
(DAY/'search_result.json').write_text(json.dumps(D,ensure_ascii=False,indent=2)+'\n')
(DAY/'daily-editorial.json').write_text(json.dumps(E,ensure_ascii=False,indent=2)+'\n')
print('formal',D['formal_recommendation_count'],'physics false positives',D['physics_false_positive_count'])
