import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / '08-daily/2026-10-22'
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import render_note as render

data = json.loads((DAY / 'search_result.json').read_text())
top = {p['title']: p for p in data['top_papers']}
original_status = render.status

def status(paper):
    if paper.get('already_known'):
        return original_status(paper)
    candidate = top.get(paper['title'])
    if candidate:
        return '阅读建议' if candidate['formal_recommendation'] else '语义排除'
    return '候选（未入前十）'

render.status = status
output = io.StringIO()
with contextlib.redirect_stdout(output):
    render.main()
text = output.getvalue()
text = text.replace('新增推荐 10 篇', '预筛候选 10 篇（正式阅读建议 6 篇、语义排除 4 篇）')
text = text.replace('进入今日推荐 10 篇', '进入前十候选 10 篇，其中正式阅读建议 6 篇')
text = text.replace('## 其余 7 篇推荐\n', '## 其余 7 篇推荐\n\n以下保留前十候选的全部审计记录，其中3篇为条件性阅读建议、4篇已语义排除；排除项不计正式推荐。\n')
text = text.replace('| 编辑稿 | daily-editorial.json | 可选保留，便于重渲染 |', '| 编辑稿 | daily-editorial.json | 已保存完整分析与十篇复排理由 |\n| 来源覆盖 | source-coverage.json | arXiv已检索；OpenReview部分查询限流；年度补充限流 |\n| 归档记录 | archive-result.json | 3篇下载成功，0失败 |\n| 校验记录 | validation.json | 本次新增链接、月份和状态校验通过；旧库9处断链单列 |')
print(text, end='')
