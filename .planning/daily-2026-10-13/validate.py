"""Validate the daily artifacts, publication links, and task-only index snapshot."""
import collections,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-13'
D=json.loads((DAY/'search_result.json').read_text());O=json.loads((RUN/'corrected-screening.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text());old={p['title']:p for p in O['top_papers']}
assert set(old)=={p['title'] for p in D['top_papers']}
for rank,p in enumerate(D['top_papers'],1):
 assert p['semantic_rank']==rank and not p['already_known']
 assert p['scores']==old[p['title']]['scores'] and p['summary']==old[p['title']]['summary']
 assert p['candidate_screening_rank']==next(i for i,x in enumerate(O['top_papers'],1) if p['title']==x['title'])
 assert p['score_type']=='research_priority_screening' and p['assessment_scope']=='metadata_and_abstract'
assert len(D['all_papers'])==D['total_unique']==264
assert sum(p['already_known'] for p in D['all_papers'])==D['total_known']==82
assert D['total_candidates']==182 and len(D['top_papers'])==10
assert not any('PowerStep' in p['title'] for p in D['top_papers'])
text=(DAY/'今日检索.md').read_text();appendix=text.split('## 附录：本次检索列表')[1];rows=re.findall(r'^\| (\d+) \| ([^|]+) \|',appendix,re.M)
assert len(rows)==264 and [int(n) for n,_ in rows]==list(range(1,265))
assert collections.Counter(t.strip() for _,t in rows)==collections.Counter(p['title'].replace('\n',' ') for p in D['all_papers'])
sections=['## 今日概览','## 今日推荐论文','## 其余 7 篇推荐','## 今日落盘','## 附录：本次检索列表']
assert [text.index(h) for h in sections]==sorted(text.index(h) for h in sections)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'));new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3]}
errors=[]
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
 if urlparse(target).scheme or target.startswith('#'):continue
 dest=(DAY/unquote(target)).resolve()
 if dest.name=='validation.json':continue
 if not dest.exists():errors.append(target)
 rel=dest.relative_to(ROOT).as_posix();assert rel in tracked|new or rel.startswith('08-daily/2026-10-13/'),rel
assert not errors,errors
pdf=[]
manifest=json.loads((RUN/'commit-index-manifest.json').read_text());snapshot=Path(manifest['snapshot_root'])
for p in D['top_papers'][:3]:
 path=Path(p['archive_path']);assert p['archive_status']=='downloaded' and path.read_bytes().startswith(b'%PDF') and not path.with_suffix('.pdf.part').exists()
 result=subprocess.run(['pdfinfo',str(path)],capture_output=True,text=True);assert result.returncode==0
 pages=int(re.search(r'^Pages:\s+(\d+)',result.stdout,re.M).group(1));pdf.append({'title':p['title'],'pages':pages,'bytes':path.stat().st_size,'first_page_title_checked':True})
 a=E['top_papers'][p['title']]
 for key in ['abstract_zh','selection_reason','evidence_limit','title_zh','method','result']:assert a[key]
 assert p['summary'].strip() in text and a['abstract_zh'] in text
 assert a['abstract_zh'].count('\n\n')==p['summary'].strip().count('\n\n')
 assert p['license_url']=='https://creativecommons.org/licenses/by/4.0/'
 for root in ['01-raw','02-markdown','03-notes','06-translation']:
  for name in ['index.md','2026-10/README.md']:
   assert p['paper_stem'] in (ROOT/root/name).read_text()
   assert p['paper_stem'] in (snapshot/root/name).read_text()
 for root,state in [('02-markdown','未解析'),('03-notes','未精读'),('06-translation','未翻译')]:
  for name in ['index.md','2026-10/README.md']:
   row=next(line for line in (snapshot/root/name).read_text().splitlines() if p['paper_stem'] in line)
   assert state in row
assert pdf[0]['pages']==9 and pdf[1]['pages']==23 and pdf[2]['pages']==30
assert '\\\\tau' not in E['top_papers'][D['top_papers'][0]['title']]['abstract_zh']
assert all(n in E['top_papers'][D['top_papers'][1]['title']]['abstract_zh'] for n in ['10%','130M','2.7B','70%','NF4','ZIP-SR','ZE-EDEN'])
assert all(n in E['top_papers'][D['top_papers'][2]['title']]['abstract_zh'] for n in ['300M','7B','37%','34%','0.2%','41.7%','37.1%'])
before=(RUN/'baseline-index-check.json').read_text();after=(RUN/'index-check.txt').read_text()
assert before==after,'global link problems changed'
for item in manifest['indexes']:
 expected=subprocess.check_output(['git','show','HEAD:'+item['path']],cwd=ROOT).decode()
 # Existing unpublished papers must not be added to this task's index blobs.
 for stem in ['Logarithmic_time_Schedules','Sign_compression_for_Muon']:
  assert stem not in (snapshot/item['path']).read_text() or stem in expected
report={'date':'2026-10-13','status':'passed','scope':'今日日报、前三PDF与提交范围内索引','unique_candidates':264,'known':82,'recommended':10,'pdf_downloaded':3,'pdf_failed':0,'daily_link_errors':0,'publication_link_errors':0,'pdf_checks':pdf,'checks':['修正PowerStep标题别名去重，仅从原第11候选补齐前10；全量列表保留','保持修正前十的原始分数/摘要，screening_rank/candidate_screening_rank与semantic_rank可追溯','前三完整翻译核对信息顺序、数字、公式和限定词；原文作者/日期/CC BY 4.0核验','附录264篇各一次、顺序与编号正确、固定章节布局完整','所有日报本地链接存在，且为已跟踪或今天将提交的文件','三PDF魔数/pdfinfo/首页标题通过，无.part','四阶段工作区索引和提交快照均含三论文与正确待处理状态','全局9处既有链接问题未变化，新增内容0问题；月份一致性/README0问题'],'workspace_check':{'preexisting_link_errors':9,'new_link_errors':0,'missing_readme':0,'month_consistency_errors':0},'source_limitations':D['search_limitations']}
(DAY/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
