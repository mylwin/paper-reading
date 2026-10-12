import json
from pathlib import Path
root=Path.cwd();base=root/'07-research/2026-10-10-顶会顶刊写作风格调研_证据';out=root/'07-research/2026-10-12-顶会顶刊写作风格调研_证据'
before=json.loads((out/'progress-before.json').read_text());ledger=json.loads((out/'batch-ledger.json').read_text());after=json.loads(json.dumps(before))
for v in after['venues']:
 items=[p for p in ledger['papers'] if p['venue']==v['venue']]
 v['main_text_style_reviewed']+=len(items);v['full_text_and_relevant_appendix_reviewed']+=sum(p['full_and_appendix_complete'] for p in items);v['induction_complete']+=sum(p['full_and_appendix_complete'] and p['split'] in ['induction','derivation'] for p in items)
 new=sum(p.get('new_pdf_acquisition',False) for p in items);v['pdf_acquired']+=new;v['text_parsed']+=new
 if not items:continue
 path=base/'corpus'/v['venue']/'validation.json';data=json.loads(path.read_text())
 data.update(counts={k:n for k,n in v.items() if k!='venue'},updated='2026-10-12',
  main_text_review_ids=sorted(set(data.get('main_text_review_ids',[])+[p['id'] for p in items])),
  complete_review_ids=sorted(set(data.get('complete_review_ids',[])+[p['id'] for p in items if p['full_and_appendix_complete']])),
  acceptance_status='in_progress;50pervenue and40induction/10heldout acceptance not achieved')
 if v['venue']=='aaai':data['full_text_and_relevant_appendix_review_count']=v['full_text_and_relevant_appendix_reviewed']
 if v['venue']=='neurips':
  data.update(semantic_reviewed_count=v['full_text_and_relevant_appendix_reviewed'],pending_semantic_count=50-v['full_text_and_relevant_appendix_reviewed'],status='continuous semantic pilot2complete',review_scope='all scientific PDF appendices/checklists; code correctness and generic tool manuals not independently audited')
 if v['venue']=='tai':
  data.update(downloaded=v['pdf_acquired'],parsed=v['text_parsed'],semantic_fulltext_reviewed=v['full_text_and_relevant_appendix_reviewed'],known_ineligible_count=0,known_ineligible_ids=[],excluded_historical_nonresearch_ids=['tai-2024-03'],replacement_status='registered2026-10-12 as tai-2024-03-r1;50active rows and holdout preserved; original lock backup retained',official_fulltext_and_version_content_confirmation='4 actual accepted author manuscripts independently confirmed;2 full scientific reading,2 referenced supplements pending',content_confirmed_accepted_manuscripts=4)
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
after.update(updated='2026-10-12',known_ineligible_locked_slots=0,excluded_historical_nonresearch_records=1,eligibility_note='550 active positions after one documented nonresearch slot repair; original lock/history preserved. This is not certification of every remaining candidate qualification.',totals={k:sum(v[k] for v in after['venues']) for k in before['totals']});after['remaining_full_review']=550-after['totals']['full_text_and_relevant_appendix_reviewed']
for p in [base/'progress.json',out/'progress-after.json']:p.write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n')
path=base/'implementation-validation.json';d=json.loads(path.read_text());d.update(updated='2026-10-12',full_corpus_completed_count=23,full_corpus_count_as_of='2026-10-12; prior isolated behavior evaluation archive remains historical14paper snapshot',current_research_report='07-research/2026-10-12-顶会顶刊写作风格调研.md');path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
p=root/'07-research/2026-10-10-顶会顶刊写作风格调研.md';s=p.read_text()
replacements={
 '创建：2026-10-10；更新：2026-10-11。':'创建：2026-10-10；更新：2026-10-12。',
 '550 个锁定位置已满足年度/划分分配检查；2026-10-11发现TAI的1条Front Cover不符合原创研究资格，替换尚待完成，因此不称550篇合格研究论文。':'550个活动位置满足年度/划分分配检查；原TAI Front Cover于2026-10-12按独立修复记录替换为可核验接受稿，原锁定快照及排除行保留。其他候选的逐篇资格核验仍在进行，不声称已确认550篇均合格。',
 '| AAAI | 50 | 50 | 2 | 0 |':'| AAAI | 50 | 50 | 4 | 0 |',
 '| NeurIPS | 50 | 50 | 0 | 0 |':'| NeurIPS | 50 | 50 | 2 | 0 |',
 '| IEEE TAI | 50 | 23 | 1 | 0 |':'| IEEE TAI | 50 | 24 | 2 | 0 |',
 '| 合计 | **550** | **428** | **18** | **4** |':'| 合计 | **550** | **429** | **23** | **4** |',
 '18 篇指写作组织':'23篇指写作组织',
 '122 个锁定位置尚无可用全文；另有410个已解析位置尚未完成此审阅。已确认的非研究封面需要替换，不能靠阅读全文完成资格。':'121个活动位置尚无可用全文；另有406个已解析位置尚未完成此审阅。完整度以真实正文及相关科学附录阅读为准，引用但未读补充不计完整。',
 '- AAAI有50篇导航与段落观察、3篇主文连续审阅，其中2篇完整；24-06公开完整证明未定位，仍仅计正文。':'- AAAI有50篇导航与段落观察、5篇正文连续审阅，其中4篇完整；24-06完整证明未定位仍仅计正文。25-12实际7页作者补充已读，但二进制与视觉glyph核验缺口明确保留。',
 '目前TPAMI、TNNLS、TAI各1篇经过内容核验的接受作者稿计入完整审阅；TAI的Q-Cogni实际已接受声明纠正了OpenAlex submittedVersion标签。':'目前TPAMI、TNNLS各1篇、TAI2篇内容核验接受稿计入完整；TAI另2篇仅正文完成且引用补充未读。机构封面、submitted标签和临时Latex页眉不能代替实际版本核验。',
 '本次新增逐篇审阅、反例和版本纠正见[全文续研报告](2026-10-11-顶会顶刊写作风格全文续研.md)。':'连续调研见[10月11日续研](2026-10-11-顶会顶刊写作风格全文续研.md)及[10月12日调研](2026-10-12-顶会顶刊写作风格调研.md)。'
}
for old,new in replacements.items():
 assert old in s,old
 s=s.replace(old,new)
p.write_text(s)
p=root/'.claude/skills/paper-writing-shared/venues/overview.md';s=p.read_text().replace('更新：2026-10-11。','更新：2026-10-12。').replace('2篇完整先导，覆盖2024/2025；另1篇正文的完整证明仍未定位','4篇完整先导，覆盖2024/2025；另1篇正文的完整证明仍未定位').replace('无完整已审语料结论','2篇跨年度完整先导，均非正式留出验证').replace('独立 100–150 词 Impact Statement；1篇接受稿先导、1条封面待替换','独立100–150词Impact Statement；2篇完整、2篇补充待核验；封面修复记录保留');p.write_text(s)
p=root/'.planning/venue-writing-2026-10-10/task_plan.md';s=p.read_text().replace('428主PDF获取/解析，18篇完整已审；1条非研究封面待替换','429主PDF获取/解析，23篇完整已审；非研究封面已按审计记录修复');p.write_text(s)
print(json.dumps(after['totals'],ensure_ascii=False))
