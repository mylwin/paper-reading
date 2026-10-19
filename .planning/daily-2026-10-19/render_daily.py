"""Render through the skill, retaining abstracts and publication-safe links."""
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / '08-daily/2026-10-19'
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import render_note as r
D = json.loads((DAY / 'search_result.json').read_text())
E = json.loads((DAY / 'daily-editorial.json').read_text())
analyses = E['top_papers']
titles = set(analyses)
def status(p):
 if p.get('editorial_status'):
  return p['editorial_status']
 if p.get('already_known'):
  return '已在库（' + (p.get('kb_source') or 'unknown') + '）'
 if p['title'] in titles:
  return '推荐（跟踪）' if analyses[p['title']]['reading_decision'] == '跟踪' else '推荐'
 return '候选（未入前10）'
r.status = status
tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
new = {Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3] if p.get('archive_status') == 'downloaded'}
original_pdf, original_note = r.local_pdf, r.local_note
def available_pdf(directory, stem):
 path = original_pdf(directory, stem)
 return path if path and path.relative_to(ROOT).as_posix() in tracked | new else None
def available_note(directory, stem):
 path = original_note(directory, stem)
 return path if path and path.relative_to(ROOT).as_posix() in tracked else None
r.local_pdf, r.local_note = available_pdf, available_note
sys.argv = ['render_note.py', '--date', '2026-10-19', '--papers-json', str(DAY / 'search_result.json'), '--editorial-json', str(DAY / 'daily-editorial.json')]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
 r.main()
text = buffer.getvalue().replace('新增推荐 10 篇', '预筛前十10篇（正式阅读建议8项，配置排除2项）').replace('进入今日推荐 10 篇', '进入预筛前十10篇（正式阅读建议8项）')
text = text.replace('## 其余 7 篇推荐\n', '## 其余 7 篇推荐\n\n保留原预筛前十的其余7项供审计；两项Workshop已按配置排除，非正式推荐。\n')
for p in D['top_papers'][:3]:
 text = text.replace('- **英文摘要**：' + r.cell(p['summary']), '- **英文摘要**：\n\n' + p['summary'].strip(), 1)
 abstract_zh = analyses[p['title']]['abstract_zh']
 text = text.replace('- **中文摘要**：' + abstract_zh, '- **中文摘要（完整翻译）**：\n\n' + abstract_zh, 1)
 source = '- **来源**：arXiv（' + p['published'][:10] + '首次提交；检索版本见访问链接）\n'
 if p['license_urls_verified']:
  license_url = p['license_urls_verified'][0].replace('http:', 'https:')
  source += '- **摘要署名与许可**：作者见下，[来源许可](' + license_url + ')；中文为基于原始页面摘要的完整翻译。\n'
 text = text.replace('- **来源**：arxiv\n', source, 1)
 for q in p['related_papers']:
  shared = ' — 共同点：`' + ', '.join(q['shared_terms']) + '`'
  text = text.replace(shared, shared + '；' + q['relationship'], 1)
text = text.replace('可选保留，便于重渲染', '已保存，可重渲染')
text = text.replace('按研究大方向分节；每节内部按研究优先级从高到低。', '按配置关键词命中的研究大方向分节；Muon粒子物理误命中保留分组但标记不推荐。主题计数不等于直接相关论文数；同脉络为已核验的阅读入口，不代表已证实技术继承。每节内部按研究优先级从高到低。')
landing = '''
前三篇PDF已归档至2026-10，入库日2026-10-19。四阶段根索引及月份README已刷新，新增三篇均登记为 `未解析` / `未精读` / `未翻译`。

- [PDF统一登记](../../01-raw/index.md) · [解析索引](../../02-markdown/index.md) · [精读索引](../../03-notes/index.md) · [翻译索引](../../06-translation/index.md)
- [归档结果](archive-result.json) · [校验记录](validation.json)
- 两篇Workshop从正式推荐中排除；前三PDF均通过魔数、页数和首页身份校验，原始文件未改写。
- 全局链接检查仍有9处既有研究文档的工作区外链接问题，本次新增链接问题为0；详见校验记录。

'''
text = text.replace('## 附录：本次检索列表', landing + '## 附录：本次检索列表')
print(('# 2026-10-19 今日检索\n\n' + text).rstrip())
