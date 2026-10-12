import hashlib, json, re, subprocess, zipfile
from pathlib import Path
from urllib.parse import unquote

root=Path.cwd();base=root/'07-research/2026-10-10-顶会顶刊写作风格调研_证据'
out=root/'07-research/2026-10-11-顶会顶刊写作风格全文续研_证据'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
checks=[]
def check(name,ok,details=None):
 checks.append(dict(check=name,passed=bool(ok),details=details))
 if not ok:print('FAILED',name,details)
ledger=json.loads((out/'batch-ledger.json').read_text())
progress=json.loads((base/'progress.json').read_text())
check('progress exact aggregate',all(progress['totals'][k]==sum(v[k] for v in progress['venues']) for k in progress['totals']))
check('batch counts conservative',ledger['new_main_reviews']==5 and ledger['new_full_reviews']==4 and ledger['new_holdout_reviews']==0 and sum(p['full_and_appendix_complete'] for p in ledger['papers'])==4)
expected={v['venue']:v for v in progress['venues']}
check('main21 full18 induction14 holdout4',progress['totals']['main_text_style_reviewed']==21 and progress['totals']['full_text_and_relevant_appendix_reviewed']==18 and progress['totals']['induction_complete']==14 and progress['totals']['holdout_complete']==4)
lockhash=json.loads((out/'lock-integrity-baseline.json').read_text())
check('locked manifests not mutated',all(sha(base/'corpus'/v/'manifest.json')==h for v,h in lockhash.items()))
for v in ['aaai','tmlr','tai']:
 d=base/'corpus'/v;manifest=json.loads((d/'manifest.json').read_text());old=manifest.get('records',manifest.get('papers'))
 rows=[json.loads(l) for l in (d/'records.jsonl').read_text().splitlines()]
 current={r['id']:r for r in rows}
 check(v+' identity/year/split unchanged',len(old)==len(rows)==50 and all((r['id'],r['year'],r['split'],r['title'])==(current[r['id']]['id'],current[r['id']]['year'],current[r['id']]['split'],current[r['id']]['title']) for r in old))
 check(v+' 25peryear20induction5holdout',all(sum(r['year']==y and r['split'] in ['induction','derivation'] for r in rows)==20 and sum(r['year']==y and r['split']=='holdout' for r in rows)==5 for y in [2024,2025]))
 check(v+' validation matches progress',json.loads((d/'validation.json').read_text())['counts']=={k:n for k,n in expected[v].items() if k!='venue'})
 for item in [x for x in ledger['papers'] if x['venue']==v]:
  r=current[item['id']];rev=json.loads((root/item['review_path']).read_text())
  pdf=root/r['pdf_path'];md=root/r['markdown_path']
  extracted=subprocess.run(['pdftotext','-layout',str(pdf),'-'],capture_output=True,text=True,check=True).stdout
  pages=len([s for s in extracted.split('\f') if s.strip()])
  check(item['id']+' identity and exact asset hashes',sha(pdf)==r['pdf_sha256']==rev['pdf_sha256']==item['pdf_sha256'] and sha(md)==rev['markdown_sha256'])
  check(item['id']+' continuous page coverage certificate',rev['coverage'][0]['pages']==list(range(1,pages+1)) and pages==item['pdf_pages'] and len(rev['ten_dimensions'])==10 and bool(rev['counterexamples']))
  check(item['id']+' no completion inference from parse',rev['full_text_style_review_complete'] and rev['external_appendix_review_complete']==item['full_and_appendix_complete'] and bool(rev['supplement_audit']['result']))

evaluation=root/'07-research/2026-10-11-论文写作润色技能评测_证据'
archive=json.loads((evaluation/'archive-sha256.json').read_text())
broken=[s for s,h in archive.items() if not (evaluation/s).is_file() or sha(evaluation/s)!=h]
check('prior evaluation archived bytes unchanged',not broken,dict(files_checked=len(archive),mismatches=broken))
promptprov=base/'corpus/tmlr/supplement-audit/cropa-prompt-file-provenance.json'
sourcefiles=json.loads(promptprov.read_text())
check('CroPA fixed source snapshots match provenance',all(sha(root/p['local_path'])==p['sha256'] for p in sourcefiles))
ziprecord=json.loads((out/'next-batch/supplement-acquisition.json').read_text())
zippath=root/ziprecord['local_path']
with zipfile.ZipFile(zippath) as z:crc=z.testzip()
check('NeurIPS supplement complete ZIP CRC and hash',crc is None and sha(zippath)==ziprecord['sha256'] and ziprecord['semantic_reading_complete'] is False)
images=[dict(path=str(p.relative_to(root)),sha256=sha(p),inspection='Codex root directly viewed actual PDF render; selected ambiguous/evidence-critical pages, not all-pages layout audit') for p in sorted((out/'visual-review').rglob('*.png'))]
save(out/'visual-review-manifest.json',images)
check('13 directly inspected actual page renders preserved',len(images)==13)

registry=json.loads((root/'.planning/venue-writing-next-research-2026-10-11/workspace-registry.json').read_text())['registry']
month=[p for p in registry if p['archive_month']=='2026-10']
check('new3 asset dates and no research-note promotion',all(any(p['paper_stem']==Path(r['pdf_path']).stem and p['archive_date']=='2026-10-11' and p['parse_status']=='已解析' and p['reading_status']=='未精读' for p in registry) for r in [json.loads(l) for l in (base/'corpus/tmlr/records.jsonl').read_text().splitlines()][:2]+[json.loads((base/'corpus/tmlr/records.jsonl').read_text().splitlines()[25])]))
headers={
 '01-raw':f'本目录保存 **2026-10** 首次入库的论文原始 PDF，共 {len(month)} 篇。',
 '02-markdown':f'本目录登记 **2026-10** 入库的 {len(month)} 篇论文，其中已解析 {sum(p["parse_status"]=="已解析" for p in month)} 篇。',
 '03-notes':f'本目录登记 **2026-10** 入库的 {len(month)} 篇论文，其中已精读 {sum(p["reading_status"]=="已精读" for p in month)} 篇。',
 '06-translation':f'本目录登记 **2026-10** 入库的 {len(month)} 篇论文，其中已翻译 {sum(p["translation_status"]=="已翻译" for p in month)} 篇。'}
for folder,header in headers.items():
 p=root/folder/'2026-10/README.md';text=p.read_text();lines=text.splitlines();lines[2]=header;p.write_text('\n'.join(lines)+'\n')

implementation=base/'implementation-validation.json';d=json.loads(implementation.read_text());d.update(full_corpus_completed_count=18,full_corpus_count_as_of='2026-10-11 full-text continuation; prior independent behavior evaluation retains14paper historical snapshot',current_research_report='07-research/2026-10-11-顶会顶刊写作风格全文续研.md');save(implementation,d)
save(out/'validation.json',dict(updated='2026-10-11',kind='research evidence integrity checks; not a new skill behavior or LaTeX compile evaluation',checks=checks,all_checks_passed=all(c['passed'] for c in checks),link_check='pending below'))
p=subprocess.run(['python','.claude/skills/paper-daily/scripts/sync_indexes.py','--check-links','--json'],capture_output=True,text=True)
links=json.loads(p.stdout)
save(out/'workspace-link-check.json',links)
preexisting={'07-research/2026-10-08-Lp_APS与Muon结合_文献审计与具体研究方案.md','07-research/2026-10-10-Muon优化器近三年研究综述与APS_Lp可行性探索.md'}
new=[r for r in links['dangling'] if r['file'] not in preexisting]
check('new task links resolve; known unrelated9 kept',not new and len(links['dangling'])==9 and not links['missing_readme'] and not links['consistency_errors'],dict(new_dangling=new,preexisting_dangling=len(links['dangling'])))
save(out/'validation.json',dict(updated='2026-10-11',kind='research evidence integrity checks; not a new skill behavior or LaTeX compile evaluation',checks=checks,all_checks_passed=all(c['passed'] for c in checks),link_check=dict(new_dangling_count=len(new),preexisting_unrelated_count=len(links['dangling']),global_exit_code=p.returncode)))
print(json.dumps(dict(checks=len(checks),passed=sum(c['passed'] for c in checks),new_links_broken=len(new),previous_archive_files=len(archive)),ensure_ascii=False))
if any(not c['passed'] for c in checks):raise SystemExit(1)
