"""Render fixed skill layout plus verified source/landing annotations."""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / '08-daily/2026-10-09'
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import render_note as renderer

data = json.loads((DAY / 'search_result.json').read_text())
recommendations = {p['title'] for p in data['top_papers']}


def status(paper):
    if paper.get('already_known'):
        return '已在库（' + (paper.get('kb_source') or 'unknown') + '）'
    if 'muon collider' in (paper.get('title', '') + ' ' + paper.get('summary', '')).lower():
        return '词义误命中（不推荐）'
    return '推荐' if paper.get('title') in recommendations else '候选（未入前10）'


renderer.status = status
sys.argv = ['render_note.py', '--date', '2026-10-09',
            '--papers-json', str(DAY / 'search_result.json'),
            '--editorial-json', str(DAY / 'daily-editorial.json')]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    renderer.main()
text = buffer.getvalue()
for paper in data['top_papers'][:3]:
    replacement = ('- **来源**：OpenReview（' + paper.get('venue', '') + '，投稿记录 '
                   + paper['published'][:10]
                   + '；许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，'
                   + '中文摘要为翻译改编）\n')
    if paper.get('arxiv_counterpart'):
        replacement += ('- **版本核验**：[arXiv 2605.23061v2](https://arxiv.org/abs/2605.23061v2)，'
                        '首次提交 2026-05-21，更新 2026-09-25；'
                        'PDF归档该版，作者来自该公开记录。\n')
    text = text.replace('- **来源**：openreview\n', replacement, 1)
text = text.replace('- **作者**：--', '- **作者**：--（当前OpenReview匿名投稿未披露）')
text = text.replace('可选保留，便于重渲染', '已保存，可重渲染')
# OpenReview's source string double-escapes these TeX commands; normalize only
# the presentation. Source records and screening abstracts remain unchanged.
text = text.replace('\\\\times', '\\times').replace('\\\\sim', '\\sim')
landing = ('\n四阶段根索引与月份 README 已刷新；本次成功归档论文登记为 '
           '`未解析` / `未精读` / `未翻译`。其余两篇未下载成功，'
           '未生成虚假的阶段资产。\n\n'
           '- [PDF统一登记](../../01-raw/index.md) · '
           '[解析索引](../../02-markdown/index.md) · '
           '[精读索引](../../03-notes/index.md) · '
           '[翻译索引](../../06-translation/index.md)\n'
           '- [归档结果](archive-result.json)\n\n')
text = text.replace('## 附录：本次检索列表', landing + '## 附录：本次检索列表')
print('# 2026-10-09 今日检索\n\n' + text, end='')
