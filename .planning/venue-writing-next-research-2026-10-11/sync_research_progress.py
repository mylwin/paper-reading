import json
from pathlib import Path
root=Path.cwd();base=root/'07-research/2026-10-10-顶会顶刊写作风格调研_证据'
out=root/'07-research/2026-10-11-顶会顶刊写作风格全文续研_证据'
baseline=json.loads((root/'.planning/venue-writing-next-research-2026-10-11/baseline-progress.json').read_text())
ledger=json.loads((out/'batch-ledger.json').read_text())
progress=json.loads(json.dumps(baseline))
for item in progress['venues']:
    v=item['venue']; new=[r for r in ledger['papers'] if r['venue']==v]
    item['main_text_style_reviewed']+=len(new)
    complete=[r for r in new if r['full_and_appendix_complete']]
    item['full_text_and_relevant_appendix_reviewed']+=len(complete)
    item['induction_complete']+=sum(r['split'] in ['induction','derivation'] for r in complete)
    item['holdout_complete']+=sum(r['split']=='holdout' for r in complete)
    if v=='tmlr':item['pdf_acquired']+=3;item['text_parsed']+=3
    if v not in ['aaai','tmlr','tai']:continue
    path=base/'corpus'/v/'validation.json'; data=json.loads(path.read_text())
    data.update(updated='2026-10-11',counts={k:n for k,n in item.items() if k!='venue'},acceptance_status='in_progress;50 eligible full reviews per venue and40induction/10holdout validation not achieved')
    prior_main=data.get('main_text_review_ids',[]);prior_full=data.get('complete_review_ids',[])
    data['main_text_review_ids']=sorted(set(prior_main+[r['id'] for r in new]))
    data['complete_review_ids']=sorted(set(prior_full+[r['id'] for r in complete]))
    if v=='aaai':data['full_text_and_relevant_appendix_review_count']=item['full_text_and_relevant_appendix_reviewed']
    else:
        data['downloaded']=item['pdf_acquired'];data['parsed']=item['text_parsed'];data['semantic_fulltext_reviewed']=item['full_text_and_relevant_appendix_reviewed']
    if v=='tmlr':data['version_note']='3 fixed author deposits acquired;2 publication identities confirmed in actual PDF headers,1 accepted current-version identity unresolved. OpenReview byte equivalence unknown;1 external prompt supplement pending.'
    if v=='tai':data.update(known_ineligible_count=1,known_ineligible_ids=['tai-2024-03'],replacement_status='pending; original locked manifest retained.550 slots must not be reported as550 eligible research papers',official_fulltext_and_version_content_confirmation='1 accepted author manuscript confirmed by PDF and institution; remaining candidates require audit')
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
progress['updated']='2026-10-11';progress['status']='in_progress'
progress['totals']={k:sum(v[k] for v in progress['venues']) for k in baseline['totals']}
progress['remaining_full_review']=550-progress['totals']['full_text_and_relevant_appendix_reviewed']
progress['known_ineligible_locked_slots']=1
progress['eligibility_note']='550 locked positions include1 confirmed nonresearch Front Cover pending replacement; other candidate qualification is not inferred from metadata alone.'
progress['qualification']='Continuous PDF prose/style review is not proof/code audit or experiment replication. Accepted author/published-header deposits may support writing organization, not final typesetting. Referenced unreviewed external appendices excluded from full count.'
for target in [base/'progress.json',out/'progress-after.json']:
 target.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
(out/'progress-before.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+'\n')
report=root/'07-research/2026-10-10-顶会顶刊写作风格调研.md'
text=report.read_text()
replacements={
 '| AAAI | 50 | 50 | 0 | 0 |':'| AAAI | 50 | 50 | 2 | 0 |',
 '| TMLR | 50 | 0 | 0 | 0 |':'| TMLR | 50 | 3 | 1 | 0 |',
 '| IEEE TAI | 50 | 23 | 0 | 0 |':'| IEEE TAI | 50 | 23 | 1 | 0 |',
 '| 合计 | **550** | **425** | **14** | **4** |':'| 合计 | **550** | **428** | **18** | **4** |',
 '550 条元数据已满足分配检查。':'550 个锁定位置已满足年度/划分分配检查；2026-10-11发现TAI的1条Front Cover不符合原创研究资格，替换尚待完成，因此不称550篇合格研究论文。',
 '14 篇指写作组织':'18 篇指写作组织',
 '125 篇尚无可用全文；另有 411 篇已解析但尚未完成此审阅。':'122 个锁定位置尚无可用全文；另有410个已解析位置尚未完成此审阅。已确认的非研究封面需要替换，不能靠阅读全文完成资格。',
 '- AAAI 有 50 篇导航与段落观察、1 篇主文连续审阅；其完整相关证明未取得，因此全文完成数为 0。':'- AAAI有50篇导航与段落观察、3篇主文连续审阅，其中2篇完整；24-06公开完整证明未定位，仍仅计正文。',
 '- TMLR 公开 PDF 指向 OpenReview，当前返回 403；不以摘要分析补足全文数。':'- TMLR当前OpenReview PDF仍有验证阻碍；已取得3篇固定版本作者公开稿，两篇与实际TMLR发表页眉吻合。G4SATBench完整20页含附录已审；CroPA完整40页PDF已审，但外部提示词清单的归档对应未确认，暂不计完整。',
 '只有两篇经过内容核验的接受稿目前计入完整审阅。':'目前TPAMI、TNNLS、TAI各1篇经过内容核验的接受作者稿计入完整审阅；TAI的Q-Cogni实际已接受声明纠正了OpenAlex submittedVersion标签。'
}
for old,new in replacements.items():
 if old not in text:raise ValueError('Expected exact report fragment missing: '+old)
 text=text.replace(old,new)
text=text.replace('## 11 套 LaTeX 包与阶段验证','本次新增逐篇审阅、反例和版本纠正见[全文续研报告](2026-10-11-顶会顶刊写作风格全文续研.md)。先前技能评测档案保留当时14篇语料的快照，不随本轮修改。\n\n## 11 套 LaTeX 包与阶段验证')
report.write_text(text)
overview=root/'.claude/skills/paper-writing-shared/venues/overview.md';text=overview.read_text()
text=text.replace('已有段落导航观察，完整相关补充材料审阅未完成','2篇完整先导，覆盖2024/2025；另1篇正文的完整证明仍未定位')
text=text.replace('全文未取得','3篇作者稿取得，1篇完整、另1篇外部prompt待核验')
text=text.replace('独立 100–150 词 Impact Statement |','独立 100–150 词 Impact Statement；1篇接受稿先导、1条封面待替换 |')
overview.write_text(text)
task=root/'.planning/venue-writing-2026-10-10/task_plan.md';text=task.read_text().replace('425主PDF获取/解析，14篇完整已审','428主PDF获取/解析，18篇完整已审；1条非研究封面待替换');task.write_text(text)
tai=root/'.claude/skills/paper-writing-shared/venues/tai.md';text=tai.read_text().replace('其余49条合格锁定位置','其余锁定记录的资格核验');tai.write_text(text)
print(json.dumps(progress['totals'],ensure_ascii=False))
