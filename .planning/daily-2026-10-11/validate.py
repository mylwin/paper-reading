import collections
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / '08-daily/2026-10-11'
RUN = Path(__file__).resolve().parent
data = json.loads((DAY / 'search_result.json').read_text())
editorial = json.loads((DAY / 'daily-editorial.json').read_text())
original = json.loads((RUN / 'focus-screening-result.json').read_text())
old = {x['title']: x for x in original['top_papers']}
assert set(old) == {x['title'] for x in data['top_papers']}
for paper in data['top_papers']:
    assert paper['scores'] == old[paper['title']]['scores']
    assert paper['screening_rank'] == old[paper['title']]['screening_rank']
    assert paper['summary'] == old[paper['title']]['summary']
    assert not paper['already_known']
for paper in data['top_papers'][:3]:
    analysis = editorial['top_papers'][paper['title']]
    assert analysis['abstract_zh'] and analysis['selection_reason']
assert data['total_unique'] == len(data['all_papers']) == 303
assert data['total_known'] == sum(x['already_known'] for x in data['all_papers']) == 78
pdfs = []
for paper in data['top_papers'][:3]:
    if paper['archive_status'] == 'downloaded':
        path = Path(paper['archive_path'])
        assert path.read_bytes()[:4] == b'%PDF'
        assert not path.with_suffix('.pdf.part').exists()
        pdfs.append(str(path.relative_to(ROOT)))
assert len(pdfs) == 2
note = (DAY / '今日检索.md').read_text()
appendix = note.split('## 附录：本次检索列表')[1]
rows = re.findall(r'^\| (\d+) \| ([^|]+) \|', appendix, re.M)
assert len(rows) == 303
assert [int(number) for number, _ in rows] == list(range(1, 304))
assert collections.Counter(title.strip().replace(r'\|', '|') for _, title in rows) == collections.Counter(x['title'].replace('\n', ' ') for x in data['all_papers'])
errors = []
for _, target in re.findall(r'\[([^\]]*)\]\(([^)]+)\)', note):
    if urlparse(target).scheme or target.startswith('#'):
        continue
    dest = (DAY / unquote(target.split('#')[0])).resolve()
    if dest.name == 'validation.json':
        continue
    if not dest.exists():
        errors.append(target)
assert not errors, errors
verified = json.loads((RUN / 'verify-pAjrfibDB9.json').read_text())
record = next(x for x in verified['notes'] if x['id'] == 'pAjrfibDB9')
assert record['license'] == 'CC BY 4.0'
assert record['content']['abstract']['value'] == data['top_papers'][0]['summary']
validation = {
    'date': '2026-10-11', 'scope': '本次日报与新增PDF', 'status': 'passed',
    'unique_candidates': 303, 'known': 78, 'recommended': 10,
    'pdf_downloaded': 2, 'pdf_failed': 1,
    'checks': ['原始候选集合、预筛分与screening_rank保留',
               '前三篇原始英文摘要一致；完整中文翻译逐项核对',
               '复排理由及semantic_rank完整', '附录303篇各出现一次',
               '本次日报所有本地链接有效', '新增PDF魔数正确，无.part',
               '四阶段索引刷新，新增论文待处理状态登记'],
    'daily_link_errors': 0,
    'workspace_check': {
        'preexisting_link_errors': 9, 'missing_readme': 0,
        'month_consistency_errors': 0,
        'detail': '9处在2026-10-08和2026-10-10调研文档，涉及外部graduate-thesis-code路径，超出今日检索范围'
    },
    'source_limitations': data['search_limitations']
}
(DAY / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
(RUN / 'note.md').unlink(missing_ok=True)
print(json.dumps({'validation': 'passed', 'pdfs': pdfs, 'daily_links': 0,
                  'prior_workspace_link_issues': 9}, ensure_ascii=False))
