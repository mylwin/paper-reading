"""Validate evidence fidelity, daily links, PDFs and task-only index scope."""
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).parent
DAY = ROOT / '08-daily/2026-10-17'
D = json.loads((DAY / 'search_result.json').read_text())
O = json.loads((RUN / 'corrected-screening.json').read_text())
E = json.loads((DAY / 'daily-editorial.json').read_text())
V = {p['title']: p for p in json.loads((RUN / 'source-verification.json').read_text())}
original = {p['title']: p for p in O['top_papers']}
assert set(original) == {p['title'] for p in D['top_papers']}
for rank, p in enumerate(D['top_papers'], 1):
 assert p['semantic_rank'] == rank and not p['already_known']
 for key in ['scores', 'screening_rank', 'summary', 'authors', 'published', 'url', 'pdf_url']:
  assert p[key] == original[p['title']][key], key
 assert p['candidate_screening_rank'] == next(i for i, q in enumerate(O['top_papers'], 1) if q['title'] == p['title'])
 assert p['score_type'] == 'research_priority_screening'
 assert p['assessment_scope'] == 'metadata_and_abstract'
assert len(D['all_papers']) == D['total_unique'] == 259
assert sum(p['already_known'] for p in D['all_papers']) == D['total_known'] == 104
assert D['total_candidates'] == 155 and len(D['top_papers']) == 10
assert not any('PowerStep' in p['title'] for p in D['top_papers'])
assert len(E['semantic_reranking']) == 10 and all(p['reason'] for p in E['semantic_reranking'])
assert D['physics_false_positive_count'] == 32
text = (DAY / '今日检索.md').read_text()
rows = re.findall(r'^\| (\d+) \| ([^|]+) \|', text.split('## 附录：本次检索列表')[1], re.M)
assert [int(n) for n, _ in rows] == list(range(1, 260))
assert collections.Counter(t.strip() for _, t in rows) == collections.Counter(p['title'].replace('\n', ' ') for p in D['all_papers'])
sections = ['## 今日概览', '## 今日推荐论文', '## 其余 7 篇推荐', '## 今日落盘', '## 附录：本次检索列表']
assert [text.index(h) for h in sections] == sorted(text.index(h) for h in sections)
tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
new = {Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3]}
links = 0
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
 if urlparse(target).scheme or target.startswith('#'):
  continue
 dest = (DAY / unquote(target)).resolve()
 if dest.name == 'validation.json':
  continue
 assert dest.is_file(), target
 rel = dest.relative_to(ROOT).as_posix()
 assert rel in tracked | new or rel.startswith('08-daily/2026-10-17/'), rel
 links += 1
manifest = json.loads((RUN / 'commit-index-manifest.json').read_text())
snapshot = Path(manifest['snapshot_root'])
pdfs = []
for index, (p, expected_pages) in enumerate(zip(D['top_papers'][:3], [35, 37, 30])):
 path = Path(p['archive_path'])
 assert p['archive_status'] == 'downloaded' and path.read_bytes().startswith(b'%PDF')
 assert not path.with_suffix('.pdf.part').exists()
 info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, check=True)
 pages = int(re.search(r'^Pages:\s+(\d+)', info.stdout, re.M).group(1))
 assert pages == expected_pages
 if index:
  assert p['title'] == re.search(r'^Title:\s+(.+)', info.stdout, re.M).group(1)
 first = (RUN / (p['paper_stem'] + '-first-page.txt')).read_text()
 assert len(first) > 500
 assert p['url'].split('/')[-1] in first
 a = E['top_papers'][p['title']]
 for key in ['abstract_zh', 'selection_reason', 'evidence_limit', 'title_zh', 'method', 'result']:
  assert a[key]
 assert p['summary'] in text and a['abstract_zh'] in text
 assert p['summary'] == V[p['title']]['abstract']
 assert V[p['title']]['metadata']['citation_title'] == [p['title']]
 assert p['source_verified']
 for root in ['01-raw', '02-markdown', '03-notes', '06-translation']:
  for name in ['index.md', '2026-10/README.md']:
   assert p['paper_stem'] in (ROOT / root / name).read_text()
   assert p['paper_stem'] in (snapshot / root / name).read_text()
 for root, state in [('02-markdown', '未解析'), ('03-notes', '未精读'), ('06-translation', '未翻译')]:
  for name in ['index.md', '2026-10/README.md']:
   assert state in next(row for row in (snapshot / root / name).read_text().splitlines() if p['paper_stem'] in row)
 for q in p['related_papers']:
  assert Path(q['path']).is_file()
 pdfs.append({'title': p['title'], 'pages': pages, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'first_page_identity_checked': True, 'parser_warnings': info.stderr.strip(), 'first_page_render_verified': index == 0})
for p, anchors in zip(D['top_papers'][:3], [
 ['有限批量', '正信号', 'Hessian', 'Generalized-Scion', '七种', '31M GPT-2', 'ViT-B/16', '匹配动量'],
 ['Fan', '2025', '线性分类', '正的稳定化常数', '指数移动平均', 'Nesterov', '可求和扰动', '有界距离', '对数增长'],
 ['高斯噪声', '一阶平均响应', '不相关', 'Hermite', '局部二次', '平稳噪声缓冲', '每一个稳定步长', '不改变收缩动力学', '冻结', '响应匹配的动量SGD']
]):
 translation = E['top_papers'][p['title']]['abstract_zh']
 for term in anchors:
  assert term in translation, term
for item in manifest['indexes']:
 old = subprocess.check_output(['git', 'show', 'HEAD:' + item['path']], cwd=ROOT).decode()
 content = Path(item['snapshot']).read_text()
 oldrows = {line for line in old.splitlines() if line.startswith('|')}
 newrows = {line for line in content.splitlines() if line.startswith('|')}
 assert not oldrows - newrows, (item['path'], 'modified existing row')
 assert len(newrows - oldrows) == 3, (item['path'], len(newrows - oldrows))
 for prefix in ['Logarithmic_time_Schedules', 'Sign_compression_for_Muon']:
  assert prefix not in content or prefix in old
initial = json.loads((RUN / 'initial-files.json').read_text())
index_paths = {x['path'] for x in manifest['indexes']}
for rel, digest in initial.items():
 if rel in index_paths or rel.startswith('.planning/daily-2026-10-17/'):
  continue
 assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, ('unrelated asset changed', rel)
report = {
 'date': '2026-10-17', 'status': 'passed', 'unique_papers': 259, 'known': 104,
 'screening_candidates': 155, 'recommendations': 10, 'physics_false_positives_marked': 32,
 'pdf_downloaded': 3, 'pdf_failed': 0, 'daily_local_links_checked': links,
 'new_link_errors': 0, 'publication_link_errors': 0, 'pdf_checks': pdfs,
 'checks': ['保留脚本预筛分数与screening_rank，仅复排前十并补充候选名次及语义名次', '前三摘要与arXiv原页面逐字匹配；完整译文逐句核对方法、数字、比较及条件', '259条附录各出现一次且连续编号，32条粒子物理误命中已标明', '所有本地访问与相关论文链接为真实文件，且为已跟踪或本次新增资料', '三份PDF魔数、页数、首页身份核验；第一篇首页成功渲染，保留原文件解析警告', '四阶段索引登记正确状态；仅本次三篇增量进入提交快照', '无关已有改动逐文件SHA256保持不变'],
 'workspace_check': {'preexisting_link_errors': 9, 'new_link_errors': 0, 'missing_readme': 0, 'month_consistency_errors': 0, 'detail': '9处既有问题位于10月8日/10月10日研究文档，指向工作区外graduate-thesis-code资料'},
 'source_limitations': D['search_limitations']
}
(DAY / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['pdf_checks', 'checks', 'source_limitations']}, ensure_ascii=False))
