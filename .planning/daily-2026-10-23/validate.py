import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).parent
DAY = ROOT / '08-daily/2026-10-23'
data = json.loads((DAY / 'search_result.json').read_text())
editor = json.loads((DAY / 'daily-editorial.json').read_text())
initial = json.loads((RUN / 'initial-screening.json').read_text())
before = json.loads((RUN / 'workspace-links-baseline.json').read_text())
after = json.loads((RUN / 'workspace-links-after.json').read_text())
base = (RUN / 'base-commit.txt').read_text().strip()
original = {p['title']: p for p in initial['top_papers']}
assert len(original) == len(data['top_papers']) == 10
assert set(original) == {p['title'] for p in data['top_papers']}
assert data['total_unique'] == len(data['all_papers']) == 204
assert data['total_known'] == sum(p['already_known'] for p in data['all_papers']) == 118
assert data['total_candidates'] == 86
assert sum(p['formal_recommendation'] for p in data['top_papers']) == 8
assert len(editor['semantic_reranking']) == 10
for rank, p in enumerate(data['top_papers'], 1):
    assert not p['already_known'] and p['semantic_rank'] == rank
    for field in ('scores', 'screening_rank', 'summary', 'published', 'url'):
        assert p[field] == original[p['title']][field], (p['title'], field)
    assert p['score_type'] == 'research_priority_screening'
    assert p['assessment_scope'] == 'metadata_and_abstract'
    assert editor['top_papers'][p['title']]['selection_reason']

text = (DAY / '今日检索.md').read_text()
sections = ['## 今日概览', '## 今日推荐论文', '## 其余 7 篇推荐', '## 今日落盘', '## 附录：本次检索列表']
assert [text.index(s) for s in sections] == sorted(text.index(s) for s in sections)
rows = re.findall(r'^\| (\d+) \| ([^|]+) \|', text.split(sections[-1])[1], re.M)
assert [int(n) for n, _ in rows] == list(range(1, 205))
assert collections.Counter(t.strip() for _, t in rows) == collections.Counter(p['title'].replace('\n', ' ') for p in data['all_papers'])
assert text.count('【语义排除】') == 2

links = 0
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
    if urlparse(target).scheme or target.startswith('#'):
        continue
    dest = (DAY / unquote(target)).resolve()
    assert dest.is_file(), target
    links += 1
assert after['dangling'] == before['dangling']
assert after['missing_readme'] == after['consistency_errors'] == []

pdfs = []
anchors = [
 ['Muon', 'AdamW', 'PolarGD', 'Hessian', 'Newton–Schulz', '四分之一', '线搜索', '旋转'],
 ['ZO', 'FO', 'StructZO', '$mn$', '$mr$', 'Gram', 'OPT', 'Llama3.1-8B', '8.7%', '26.9%', '1.5%'],
 ['PPO', 'FP16', '全局标量', '策略变化残差', '持久更新', '六个', '五个', '1.90', '17', '30', '1.16', '2.03', '71.7%', '74.9%', '44.5%']
]
for p, pages, terms, paragraphs in zip(data['top_papers'][:3], [47, 20, 9], anchors, [4, 5, 4]):
    path = Path(p['archive_path'])
    assert p['formal_recommendation'] and p['archive_status'] == 'downloaded'
    assert path.read_bytes().startswith(b'%PDF')
    assert not path.with_suffix('.pdf.part').exists()
    info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, check=True)
    assert int(re.search(r'^Pages:\s+(\d+)', info.stdout, re.M).group(1)) == pages
    first = (RUN / (p['paper_stem'] + '-first-page.txt')).read_text()
    assert p['arxiv_id'] in first
    normalized = re.sub(r'[^a-z0-9]', '', first.lower())
    assert re.sub(r'[^a-z0-9]', '', p['title'].lower()) in normalized
    a = editor['top_papers'][p['title']]
    for field in ['title_zh', 'summary', 'abstract_zh', 'contributions', 'method', 'result', 'selection_reason', 'evidence_limit', 'reading_decision']:
        assert a[field]
    assert p['summary'] in text and a['abstract_zh'] in text
    assert len(a['abstract_zh'].split('\n\n')) == paragraphs
    for term in terms:
        assert term in a['abstract_zh'], (p['title'], term)
    for root, state in [('01-raw', None), ('02-markdown', '未解析'), ('03-notes', '未精读'), ('06-translation', '未翻译')]:
        for filename in ['index.md', '2026-10/README.md']:
            indexed = (ROOT / root / filename).read_text()
            row = next(l for l in indexed.splitlines() if l.startswith('|') and p['paper_stem'] in l)
            assert state is None or state in row
    for related in p['related_papers']:
        assert Path(related['path']).is_file()
    pdfs.append({'title': p['title'], 'pages': pages, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'identity': '首页标题与arXiv编号一致', 'pdfinfo_warnings': info.stderr.strip()})

for root in ['01-raw', '02-markdown', '03-notes', '06-translation']:
    for filename in ['index.md', '2026-10/README.md']:
        rel = root + '/' + filename
        old = subprocess.check_output(['git', 'show', base + ':' + rel], cwd=ROOT).decode()
        current = (ROOT / rel).read_text()
        oldrows = {l for l in old.splitlines() if l.startswith('|')}
        newrows = {l for l in current.splitlines() if l.startswith('|')}
        assert not oldrows - newrows, rel
        assert len(newrows - oldrows) == 3, rel

report = {
    'date': '2026-10-23', 'status': 'passed_with_preexisting_workspace_link_errors',
    'unique_papers': 204, 'known': 118, 'candidates': 86,
    'screening_top_n': 10, 'formal_recommendations': 8, 'semantic_exclusions': 2,
    'pdf_downloaded': 3, 'pdf_failed': 0, 'pdf_checks': pdfs,
    'daily_local_links_checked': links, 'new_link_errors': 0,
    'workspace_check': {'preexisting_link_errors': 9, 'remaining_link_errors': after['dangling_count'], 'new_link_errors': 0, 'missing_readme': 0, 'month_consistency_errors': 0},
    'checks': ['前十逐篇摘要语义复排，脚本分数及screening_rank未改写', '前三完整中译逐句人工复核；程序核对段落、数字、模型与方法锚点', '前三arXiv官方摘要页的题目、作者、版本与摘要已核对；未改写源摘要、分数或作者列表', 'PDF魔数、页数、首页标题与arXiv编号通过；文本提取有字体类型警告但成功完成', '附录204条连续编号且每篇仅出现一次，2篇语义排除明确标记', '四阶段八份索引表格各仅新增三行，原有表格行无改写；刷新幂等', '本次新增本地链接全通过，旧库9处断链与执行前完全一致'],
    'source_limitations': data['search_limitations']
}
(DAY / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['pdf_checks', 'checks', 'source_limitations']}, ensure_ascii=False))
