import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / '08-daily/2026-10-11'
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import render_note as renderer

data = json.loads((DAY / 'search_result.json').read_text())
editorial = json.loads((DAY / 'daily-editorial.json').read_text())
titles = {p['title'] for p in data['top_papers']}
raw = json.loads((Path(__file__).parent / 'arxiv-raw.json').read_text())
raw += json.loads((Path(__file__).parent / 'openreview-raw.json').read_text())
summaries = {p['title']: p.get('summary', '') for p in raw}

def status(paper):
    if paper.get('already_known'):
        return '已在库（' + (paper.get('kb_source') or 'unknown') + '）'
    text = (paper['title'] + ' ' + summaries.get(paper['title'], '')).lower()
    if any(word in text for word in ['muon collider', 'muon beam', 'muon magnetic',
                                    'lepton flavor', 'muon anomalous', 'muon-to-electron']):
        return '词义误命中（不推荐）'
    return '推荐' if paper['title'] in titles else '候选（未入前10）'

renderer.status = status
sys.argv = ['render_note.py', '--date', '2026-10-11', '--papers-json',
            str(DAY / 'search_result.json'), '--editorial-json',
            str(DAY / 'daily-editorial.json')]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    renderer.main()
text = buffer.getvalue()
for paper in data['top_papers'][:3]:
    summary = paper['summary']
    translation = editorial['top_papers'][paper['title']]['abstract_zh']
    text = text.replace('- **英文摘要**：' + renderer.cell(summary),
                        '- **英文摘要**：\n\n' + summary.strip().replace(r'\method{}', r'`\method{}`'), 1)
    text = text.replace('- **中文摘要**：' + translation,
                        '- **中文摘要（完整翻译）**：\n\n' + translation.replace(r'\method{}', r'`\method{}`'), 1)
    source = paper['source']
    label = ('OpenReview（ICLR 2027 匿名投稿，创建日期2026-09-18；投稿不等于录用）'
             if source == 'openreview' else 'arXiv（' + paper['published'][:10] + '首次提交，v1）')
    text = text.replace('- **来源**：' + source + '\n',
                        '- **来源**：' + label + '\n'
                        '- **摘要许可**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；中文为完整翻译改编。\n', 1)
text = text.replace('- **作者**：--\n', '- **作者**：--（当前匿名记录未披露）\n', 1)
text = text.replace('可选保留，便于重渲染', '已保存，可重渲染')
landing = '''四阶段根索引与月份 README 已刷新。两篇 PDF 的入库日期为2026-10-11，登记为 `未解析` / `未精读` / `未翻译`。AdamW 损失尖峰论文两个 OpenReview PDF 入口均 HTTP 403，保留推荐和远程入口，未生成本地占位文件。

- [PDF统一登记](../../01-raw/index.md) · [解析索引](../../02-markdown/index.md) · [精读索引](../../03-notes/index.md) · [翻译索引](../../06-translation/index.md)
- [归档结果](archive-result.json) · [验证记录](validation.json)

'''
text = text.replace('## 附录：本次检索列表', landing + '## 附录：本次检索列表')
print('# 2026-10-11 今日检索\n\n' + text, end='')
