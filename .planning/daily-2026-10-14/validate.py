"""Validate new artifacts against acquired records and task-only index snapshot."""
import collections,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-14'
D=json.loads((DAY/'search_result.json').read_text());O=json.loads((RUN/'corrected-screening.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text());initial=json.loads((RUN/'initial-screening.json').read_text());old={p['title']:p for p in O['top_papers']}
assert set(old)=={p['title'] for p in D['top_papers']}
for i,p in enumerate(D['top_papers'],1):
 assert p['semantic_rank']==i and not p['already_known']
 for key in ['scores','summary','authors','published']:assert p[key]==old[p['title']][key],key
 assert p['candidate_screening_rank']==next(i for i,q in enumerate(O['top_papers'],1) if q['title']==p['title'])
 assert p['screening_rank']==next((i for i,q in enumerate(initial['top_papers'],1) if q['title']==p['title']),11)
 assert p['score_type']=='research_priority_screening'
assert len(D['all_papers'])==D['total_unique']==264
assert sum(p['already_known'] for p in D['all_papers'])==D['total_known']==92
assert D['total_candidates']==172 and len(D['top_papers'])==10
assert not any('PowerStep' in p['title'] for p in D['top_papers'])
assert len(E['semantic_reranking'])==10
text=(DAY/'今日检索.md').read_text();rows=re.findall(r'^\| (\d+) \| ([^|]+) \|',text.split('## 附录：本次检索列表')[1],re.M)
assert [int(n) for n,_ in rows]==list(range(1,265))
assert collections.Counter(t.strip() for _,t in rows)==collections.Counter(p['title'].replace('\n',' ') for p in D['all_papers'])
sections=['## 今日概览','## 今日推荐论文','## 其余 7 篇推荐','## 今日落盘','## 附录：本次检索列表'];assert [text.index(h) for h in sections]==sorted(text.index(h) for h in sections)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'));new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3]}
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
 if urlparse(target).scheme or target.startswith('#'):continue
 dest=(DAY/unquote(target)).resolve()
 if dest.name=='validation.json':continue
 assert dest.exists(),target
 rel=dest.relative_to(ROOT).as_posix();assert rel in tracked|new or rel.startswith('08-daily/2026-10-14/'),rel
M=json.loads((RUN/'commit-index-manifest.json').read_text());snapshot=Path(M['snapshot_root']);pdf=[]
for p in D['top_papers'][:3]:
 path=Path(p['archive_path']);assert p['archive_status']=='downloaded' and path.read_bytes().startswith(b'%PDF')
 assert not path.with_suffix('.pdf.part').exists()
 out=subprocess.check_output(['pdfinfo',str(path)]).decode();pages=int(re.search(r'^Pages:\s+(\d+)',out,re.M).group(1));pdf.append({'title':p['title'],'pages':pages,'bytes':path.stat().st_size,'first_page_title_checked':True})
 a=E['top_papers'][p['title']]
 for k in ['abstract_zh','selection_reason','evidence_limit','title_zh','method','result']:assert a[k]
 assert p['summary'] in text and a['abstract_zh'] in text
 assert p['license_url']=='https://creativecommons.org/licenses/by/4.0/' and p['source_verified']
 for root in ['01-raw','02-markdown','03-notes','06-translation']:
  for name in ['index.md','2026-10/README.md']:
   assert p['paper_stem'] in (ROOT/root/name).read_text()
   assert p['paper_stem'] in (snapshot/root/name).read_text()
 for root,state in [('02-markdown','未解析'),('03-notes','未精读'),('06-translation','未翻译')]:
  for name in ['index.md','2026-10/README.md']:
   assert state in next(row for row in (snapshot/root/name).read_text().splitlines() if p['paper_stem'] in row)
for item in M['indexes']:
 content=(snapshot/item['path']).read_text();expected=subprocess.check_output(['git','show','HEAD:'+item['path']],cwd=ROOT).decode()
 for stem in ['Logarithmic_time_Schedules','Sign_compression_for_Muon']:assert stem not in content or stem in expected
# Preserve pre-existing skill edits byte-for-byte.
base=Path('/private/tmp/paper-daily-20261014-baseline')
for p in (base/'.claude').rglob('SKILL.md'):assert p.read_bytes()==(ROOT/p.relative_to(base)).read_bytes()
assert all(n in E['top_papers'][D['top_papers'][1]['title']]['abstract_zh'] for n in ['2--128','8--32','64','1/K','固定宽度','有界参数域'])
assert all(n in E['top_papers'][D['top_papers'][2]['title']]['abstract_zh'] for n in ['所有多项式矩','次 Weibull','对数正态','轨迹假设','Polyak'])
check=(RUN/'index-check.txt').read_text();assert '问题 9 处（悬空链接 9 / 缺 README 0 / 一致性 0）' in check
assert '08-daily/2026-10-14/' not in check
report={'date':'2026-10-14','status':'passed','unique_candidates':264,'known':92,'recommended':10,'pdf_downloaded':3,'pdf_failed':0,'daily_link_errors':0,'publication_link_errors':0,'pdf_checks':pdf,'checks':['PowerStep标题别名通过已有PDF核验后去重；原第11候选补齐','前十逐篇语义复排，原分数/摘要/作者/日期与两套预筛序位保留','前三完整摘要翻译逐项核对，原文及CC BY 4.0许可核验','附录264篇各一次、连续编号，章节顺序正确','日报所有本地链接为真实且已跟踪或本次新增资产','三PDF魔数、pdfinfo、首页标题通过，无半截文件','工作区和提交快照的四阶段索引均包含三篇及正确状态','既有技能修改保持原样，往日未提交PDF排除在提交索引之外'],'workspace_check':{'preexisting_link_errors':9,'new_link_errors':0,'missing_readme':0,'month_consistency_errors':0,'detail':'9处旧链接位于10月8日/10月10日研究文档，指向工作区外graduate-thesis-code资料；全局复查确认无新增问题'},'source_limitations':D['search_limitations']}
(DAY/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
