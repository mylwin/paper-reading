import contextlib, io, json, sys, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; DAY=ROOT/'08-daily/2026-10-14'; RUN=Path(__file__).parent
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import render_note as r
D=json.loads((DAY/'search_result.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text()); titles={p['title'] for p in D['top_papers']}
raw=json.loads((RUN/'arxiv-raw.json').read_text())+json.loads((RUN/'openreview-raw.json').read_text()); summaries={p['title']:p.get('summary','') for p in raw}
def status(p):
 if p.get('already_known'):return '已在库（'+(p.get('kb_source') or 'unknown')+'）'
 text=(p['title']+' '+summaries.get(p['title'],'')).lower()
 if any(t in text for t in ['muon collider','muon beam','muon anomalous','muon magnetic','muon-to-electron','lepton flavor']):return '词义误命中（不推荐）'
 if p['title'] in titles:return '前10候选（跟踪）' if E['top_papers'][p['title']].get('reading_decision')=='跟踪' else '推荐'
 return '候选（未入前10）'
r.status=status
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
new_pdfs={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3] if p.get('archive_status')=='downloaded'}
original_local_pdf=r.local_pdf
def published_pdf(directory,stem):
 path=original_local_pdf(directory,stem)
 return path if path and path.relative_to(ROOT).as_posix() in tracked|new_pdfs else None
r.local_pdf=published_pdf
sys.argv=['render_note.py','--date','2026-10-14','--papers-json',str(DAY/'search_result.json'),'--editorial-json',str(DAY/'daily-editorial.json')]
b=io.StringIO()
with contextlib.redirect_stdout(b):r.main()
text=b.getvalue()
for p in D['top_papers'][:3]:
 text=text.replace('- **英文摘要**：'+r.cell(p['summary']),'- **英文摘要**：\n\n'+p['summary'].strip(),1)
 text=text.replace('- **中文摘要**：'+E['top_papers'][p['title']]['abstract_zh'],'- **中文摘要（完整翻译）**：\n\n'+E['top_papers'][p['title']]['abstract_zh'],1)
 text=text.replace('- **来源**：arxiv\n','- **来源**：arXiv（'+p['published'][:10]+' 首次提交，v1）\n- **摘要许可**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；中文为完整翻译改编。\n',1)
text=text.replace('可选保留，便于重渲染','已保存，可重渲染')
landing='''前三篇原始PDF均已归档，入库日期2026-10-14。四阶段根索引与月份README已刷新，登记为 `未解析` / `未精读` / `未翻译`。

- [PDF统一登记](../../01-raw/index.md) · [解析索引](../../02-markdown/index.md) · [精读索引](../../03-notes/index.md) · [翻译索引](../../06-translation/index.md)
- [归档结果](archive-result.json) · [验证记录](validation.json)

'''
text=text.replace('## 附录：本次检索列表',landing+'## 附录：本次检索列表')
print(('# 2026-10-14 今日检索\n\n'+text).rstrip())
