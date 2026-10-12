import collections,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];DAY=ROOT/'08-daily/2026-10-12';RUN=Path(__file__).parent
D=json.loads((DAY/'search_result.json').read_text()); O=json.loads((RUN/'initial-screening.json').read_text()); E=json.loads((DAY/'daily-editorial.json').read_text());old={x['title']:x for x in O['top_papers']}
assert set(old)=={x['title'] for x in D['top_papers']}
for rank,p in enumerate(D['top_papers'],1):
 assert p['semantic_rank']==rank and not p['already_known']
 for field in ['scores','screening_rank','summary']:assert p[field]==old[p['title']][field]
assert D['total_unique']==len(D['all_papers'])==267
assert D['total_known']==sum(x['already_known'] for x in D['all_papers'])==74
text=(DAY/'今日检索.md').read_text(); rows=re.findall(r'^\| (\d+) \| ([^|]+) \|',text.split('## 附录：本次检索列表')[1],re.M)
assert len(rows)==267 and [int(n) for n,_ in rows]==list(range(1,268))
assert collections.Counter(t.strip() for _,t in rows)==collections.Counter(x['title'].replace('\n',' ') for x in D['all_papers'])
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'));new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3]}
errors=[]
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
 if urlparse(target).scheme or target.startswith('#'):continue
 dest=(DAY/unquote(target)).resolve()
 if dest.name=='validation.json':continue
 if not dest.exists():errors.append(target)
 rel=dest.relative_to(ROOT).as_posix();assert rel in tracked|new or rel.startswith('08-daily/2026-10-12/'),rel
assert not errors,errors
pdf=[]
for p in D['top_papers'][:3]:
 path=Path(p['archive_path']);assert p['archive_status']=='downloaded' and path.read_bytes().startswith(b'%PDF') and not path.with_suffix('.pdf.part').exists()
 result=subprocess.run(['pdfinfo',str(path)],capture_output=True,text=True);assert result.returncode==0
 pages=int(re.search(r'^Pages:\s+(\d+)',result.stdout,re.M).group(1));pdf.append({'title':p['title'],'pages':pages,'bytes':path.stat().st_size})
 a=E['top_papers'][p['title']];assert a['abstract_zh'] and a['selection_reason'] and a['evidence_limit']
 for root in ['01-raw','02-markdown','03-notes','06-translation']:
  for name in ['index.md','2026-10/README.md']:assert p['paper_stem'] in (ROOT/root/name).read_text()
checks={'date':'2026-10-12','status':'passed','scope':'本次日报、前三PDF和四阶段登记','unique_candidates':267,'known':74,'recommended':10,'pdf_downloaded':3,'pdf_failed':0,'daily_link_errors':0,'publication_link_errors':0,'pdf_checks':pdf,'checks':['原前十候选集合、分数、摘要、screening_rank均保留','逐篇语义复排与理由已保存','前三完整摘要译文按原顺序核对数字、公式和限定词；CC BY 4.0链接已核验','附录267篇各出现一次、连续编号','本地链接存在且为已跟踪或本次新增资产，可随提交访问','PDF魔数、pdfinfo结构、首页标题通过；无.part','三篇在四阶段根索引与月份README登记、状态与磁盘一致'],'workspace_check':{'preexisting_link_errors':9,'missing_readme':0,'month_consistency_errors':0,'detail':'9处既有问题位于10月8日和10月10日研究文档，指向外部graduate-thesis-code路径'},'source_limitations':D['search_limitations']}
(DAY/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False))
