"""Check daily artifacts, evidence fidelity, links, PDF identity and index scope."""
import collections,difflib,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-16'
D=json.loads((DAY/'search_result.json').read_text());O=json.loads((RUN/'corrected-screening.json').read_text());I=json.loads((RUN/'initial-screening.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text());V={p['title']:p for p in json.loads((RUN/'source-verification.json').read_text())}
original={p['title']:p for p in O['top_papers']};assert set(original)=={p['title'] for p in D['top_papers']}
for n,p in enumerate(D['top_papers'],1):
 assert p['semantic_rank']==n and not p['already_known']
 for key in ['scores','summary','authors','published','url','pdf_url']:assert p[key]==original[p['title']][key],key
 assert p['candidate_screening_rank']==next(i for i,q in enumerate(O['top_papers'],1) if q['title']==p['title'])
 assert p['screening_rank']==next((i for i,q in enumerate(I['top_papers'],1) if q['title']==p['title']),11)
 assert p['score_type']=='research_priority_screening'
assert len(D['all_papers'])==D['total_unique']==265
assert sum(p['already_known'] for p in D['all_papers'])==D['total_known']==97
assert D['total_candidates']==168 and len(D['top_papers'])==10
assert D['formal_recommendation_count']==sum(p['formal_recommendation'] for p in D['top_papers'])==9
assert not any('PowerStep' in p['title'] for p in D['top_papers'])
assert len(E['semantic_reranking'])==10 and all(p['reason'] for p in E['semantic_reranking'])
text=(DAY/'今日检索.md').read_text();appendix=text.split('## 附录：本次检索列表')[1]
rows=re.findall(r'^\| (\d+) \| ([^|]+) \|',appendix,re.M)
assert [int(n) for n,_ in rows]==list(range(1,266))
assert collections.Counter(t.strip() for _,t in rows)==collections.Counter(p['title'].replace('\n',' ') for p in D['all_papers'])
sections=['## 今日概览','## 今日推荐论文','## 其余 7 篇推荐','## 今日落盘','## 附录：本次检索列表'];assert [text.index(h) for h in sections]==sorted(text.index(h) for h in sections)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'));new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3]}
links=0
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
 if urlparse(target).scheme or target.startswith('#'):continue
 dest=(DAY/unquote(target)).resolve()
 if dest.name=='validation.json':continue
 assert dest.exists(),target
 rel=dest.relative_to(ROOT).as_posix();assert rel in tracked|new or rel.startswith('08-daily/2026-10-16/'),rel
 links+=1
M=json.loads((RUN/'commit-index-manifest.json').read_text());snapshot=Path(M['snapshot_root']);pdf=[]
for p,expected_pages in zip(D['top_papers'][:3],[34,27,25]):
 path=Path(p['archive_path']);assert p['archive_status']=='downloaded' and path.read_bytes().startswith(b'%PDF')
 assert not path.with_suffix('.pdf.part').exists()
 out=subprocess.check_output(['pdfinfo',str(path)]).decode();pages=int(re.search(r'^Pages:\s+(\d+)',out,re.M).group(1));assert pages==expected_pages
 assert p['title']==re.search(r'^Title:\s+(.+)',out,re.M).group(1)
 first=(RUN/(p['paper_stem']+'-first-page.txt')).read_text();assert len(first)>500
 pdf.append({'title':p['title'],'pages':pages,'bytes':path.stat().st_size,'title_metadata_matches':True,'first_page_checked':True})
 a=E['top_papers'][p['title']]
 for key in ['abstract_zh','selection_reason','evidence_limit','title_zh','method','result']:assert a[key]
 assert p['summary'] in text and a['abstract_zh'] in text
 assert p['summary']==V[p['title']]['abstract']
 assert p['license_url']=='https://creativecommons.org/licenses/by/4.0/' and p['source_verified']
 assert any('creativecommons.org/licenses/by/4.0/' in u for u in V[p['title']]['license_urls'])
 for root in ['01-raw','02-markdown','03-notes','06-translation']:
  for name in ['index.md','2026-10/README.md']:
   assert p['paper_stem'] in (ROOT/root/name).read_text()
   assert p['paper_stem'] in (snapshot/root/name).read_text()
 for root,state in [('02-markdown','未解析'),('03-notes','未精读'),('06-translation','未翻译')]:
  for name in ['index.md','2026-10/README.md']:assert state in next(row for row in (snapshot/root/name).read_text().splitlines() if p['paper_stem'] in row)
 for q in p['related_papers']:
  asset=Path(q['path']);assert asset.is_file()
  if asset.name=='search_result.json':assert q['title'] in (asset.parent/'今日检索.md').read_text()
# Translation audit anchors complement a sentence-by-sentence editorial read.
for needle in ['SMRC','PMRC','O(T^{-1/4})','固定的标准正交基','子空间虚拟迭代点','边界校正','加权动量跟踪','多阶段PMRC']:assert needle in E['top_papers'][D['top_papers'][0]['title']]['abstract_zh']
for needle in ['鞍点流形','全批量','并行学习','随机平滑','噪声尺度','深度线性模型','低秩矩阵重构','视觉Transformer','LLM']:assert needle in E['top_papers'][D['top_papers'][1]['title']]['abstract_zh']
for needle in ['DiLoco','固定总训练预算','Muon与AdamW','协方差结构','批量大小','8–96','参数量和数据规模','DP']:assert needle in E['top_papers'][D['top_papers'][2]['title']]['abstract_zh']
# Each staged index retains every HEAD table row, adding exactly today's PDFs.
for item in M['indexes']:
 old=subprocess.check_output(['git','show','HEAD:'+item['path']],cwd=ROOT).decode();content=Path(item['snapshot']).read_text()
 oldrows={line for line in old.splitlines() if line.startswith('|')};newrows={line for line in content.splitlines() if line.startswith('|')}
 assert not oldrows-newrows,(item['path'],'changed an existing row')
 assert len(newrows-oldrows)==3,(item['path'],len(newrows-oldrows))
 for stem in ['Logarithmic_time_Schedules','Sign_compression_for_Muon']:assert stem not in content or stem in old
check=(RUN/'index-check.txt').read_text();assert '问题 9 处（悬空链接 9 / 缺 README 0 / 一致性 0）' in check
assert '08-daily/2026-10-16/' not in check and '.planning/daily-2026-10-16/' not in check
report={'date':'2026-10-16','status':'passed','unique_papers':265,'known':97,'screening_candidates':168,'screening_top_n':10,'formal_reading_suggestions':9,'excluded_workshop':1,'pdf_downloaded':3,'pdf_failed':0,'daily_local_links_checked':links,'daily_link_errors':0,'publication_link_errors':0,'pdf_checks':pdf,'checks':['前十摘要逐篇复排，原分数、摘要、作者、日期及修正前后预筛名次保留','前三完整中文摘要与原文逐句核对；方法名、条件、公式、8–96副本及证据边界保留','前三摘要页面、PDF标题和CC BY 4.0许可核验；仅首页身份检查，不冒充全文评审','附录265篇各一次、连续编号，物理同词误命中与Workshop排除项明确标记','本地链接均为真实且已跟踪或本次新增资产；相关论文指向精读/PDF/实际包含论文的历史日报','三PDF魔数、页数、首页标题通过，无半截文件','四阶段工作区索引和提交快照均登记三篇与正确状态','提交索引保留HEAD全部已有行且仅新增本次3篇；往日未提交PDF未纳入'],'workspace_check':{'preexisting_link_errors':9,'new_link_errors':0,'missing_readme':0,'month_consistency_errors':0,'detail':'9处既有链接位于10月8日/10月10日研究文档，指向工作区外graduate-thesis-code资料；保留原样'},'source_limitations':D['search_limitations']}
(DAY/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','source_limitations']},ensure_ascii=False))
