import collections,hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-20'
D=json.loads((DAY/'search_result.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text());O=json.loads((RUN/'initial-screening.json').read_text())
V={p['title']:p for p in json.loads((RUN/'arxiv-source-verification.json').read_text())}
original={p['title']:p for p in O['top_papers']};assert len(original)==10
assert set(original)=={p['title'] for p in D['top_papers']}
for rank,p in enumerate(D['top_papers'],1):
 assert not p['already_known'] and p['semantic_rank']==rank
 for key in ['scores','screening_rank','summary','authors','published','url']:
  assert p[key]==original[p['title']][key],(p['title'],key)
 assert p['score_type']=='research_priority_screening' and p['assessment_scope']=='metadata_and_abstract'
assert D['total_unique']==len(D['all_papers'])==156
assert D['total_known']==sum(p['already_known'] for p in D['all_papers'])==70
assert D['total_candidates']==86
assert sum(p['formal_recommendation'] for p in D['top_papers'])==4
assert all(p['formal_recommendation'] for p in D['top_papers'][:3])
assert len(E['semantic_reranking'])==10
text=(DAY/'今日检索.md').read_text()
headers=['## 今日概览','## 今日推荐论文','## 其余 7 篇推荐','## 今日落盘','## 附录：本次检索列表']
assert [text.index(h) for h in headers]==sorted(text.index(h) for h in headers)
rows=re.findall(r'^\| (\d+) \| ([^|]+) \|',text.split(headers[-1])[1],re.M)
assert [int(n) for n,_ in rows]==list(range(1,157))
assert collections.Counter(t.strip() for _,t in rows)==collections.Counter(p['title'].replace('\n',' ') for p in D['all_papers'])
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3]}
links=0
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
 if urlparse(target).scheme or target.startswith('#'):continue
 dest=(DAY/unquote(target)).resolve()
 if dest==DAY/'validation.json':continue
 assert dest.is_file(),target
 rel=dest.relative_to(ROOT).as_posix()
 assert rel in tracked|new or rel.startswith('08-daily/2026-10-20/'),rel
 links+=1
manifest=json.loads((RUN/'commit-index-manifest.json').read_text());snapshot=Path(manifest['snapshot_root']);BASE=(RUN/'base-commit.txt').read_text().strip()
pdfs=[]
for p,pages in zip(D['top_papers'][:3],[27,64,22]):
 path=Path(p['archive_path']);assert path.read_bytes().startswith(b'%PDF') and not path.with_suffix('.pdf.part').exists()
 info=subprocess.run(['pdfinfo',str(path)],capture_output=True,text=True,check=True)
 assert int(re.search(r'^Pages:\s+(\d+)',info.stdout,re.M).group(1))==pages
 first=(RUN/(p['paper_stem']+'-first-page.txt')).read_text()
 assert p['arxiv_id'] in first and len(first)>500
 a=E['top_papers'][p['title']]
 for k in ['abstract_zh','title_zh','summary','contributions','method','result','selection_reason','evidence_limit','reading_decision']:assert a[k]
 assert a['abstract_zh'] in text and p['summary'] in text
 norm=lambda s:''.join(s.replace('ε',r'\epsilon').replace('https://anonymous.4open.science/r/Ori-Submission-2132.','this https URL.').split())
 assert norm(p['summary'])==norm(V[p['title']]['abstract'])
 assert p['source_verified'] and V[p['title']]['status']==200
 for root in ['01-raw','02-markdown','03-notes','06-translation']:
  for name in ['index.md','2026-10/README.md']:
   assert p['paper_stem'] in (ROOT/root/name).read_text()
   assert p['paper_stem'] in (snapshot/root/name).read_text()
 for root,state in [('02-markdown','未解析'),('03-notes','未精读'),('06-translation','未翻译')]:
  for name in ['index.md','2026-10/README.md']:
   assert state in next(row for row in (snapshot/root/name).read_text().splitlines() if p['paper_stem'] in row)
 pdfs.append({'title':p['title'],'pages':pages,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'identity':'首页arXiv编号、标题与作者核验','pdfinfo_warnings':info.stderr.strip()})
anchors=[['GaLore','Oja++','k','69.3','1B','LLaMA','16','10','Roberta-Base','https://anonymous.4open.science/r/Ori-Submission-2132'],['联邦','去中心化','欧氏投影','本地计算预算','时域','ε^{-2}','ε^{-4}','有界','静态','时变','网络常数','初始化成本'],['CVNN','长度','对齐','旋转方向','额外的梯度','Adam','Muon','四个','全连接','相同预算','无法带来改善']]
for p,terms in zip(D['top_papers'][:3],anchors):
 for term in terms:assert term in E['top_papers'][p['title']]['abstract_zh'],term
for item in manifest['indexes']:
 old=subprocess.check_output(['git','show',BASE+':'+item['path']],cwd=ROOT).decode();current=Path(item['snapshot']).read_text()
 oldrows={l for l in old.splitlines() if l.startswith('|')};newrows={l for l in current.splitlines() if l.startswith('|')}
 assert not oldrows-newrows,(item['path'],'modified existing row')
 assert len(newrows-oldrows)==3,(item['path'],'row additions')
report={'date':'2026-10-20','status':'passed','unique_papers':156,'known':70,'candidates':86,'screening_top_n':10,'formal_recommendations':4,'excluded_in_top10':{'workshop':4,'physics':2},'physics_false_positives_marked':60,'pdf_downloaded':3,'pdf_failed':0,'daily_local_links_checked':links,'new_link_errors':0,'pdf_checks':pdfs,'workspace_check':{'preexisting_link_errors':9,'new_link_errors':0,'missing_readme':0,'month_consistency_errors':0},'checks':['前十逐篇语义复排，保留脚本分数和screening_rank','前三完整中译人工逐句检查，数字、方法、限定条件已保留；脚本核对关键锚点','arXiv原摘要核对：NBCLT仅epsilon记法不同；Ori源网页超链接锚文本与实际URL等价','PDF魔数、页数、首页身份核验；AURA提取时有字体类型警告，仍成功提取','156条附录各出现一次，连续编号；其他7项明确标注排除项','本地链接均为真实已跟踪或本次新增文件；未链接未提交的MUON2精读','四阶段索引状态正确；提交快照仅添加本次三篇，已有表格行无改写'],'source_limitations':D['search_limitations']}
(DAY/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['pdf_checks','checks','source_limitations']},ensure_ascii=False))
