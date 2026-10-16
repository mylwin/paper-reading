"""Use the skill layout with honest candidate states and publication-safe links."""
import contextlib,io,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DAY=ROOT/'08-daily/2026-10-16'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import render_note as r
D=json.loads((DAY/'search_result.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text());titles={p['title'] for p in D['top_papers']}
def status(p):
 if p.get('editorial_status'):return p['editorial_status']
 if p.get('already_known'):return '已在库（'+(p.get('kb_source') or 'unknown')+'）'
 if p['title'] in titles:
  if not next(q.get('formal_recommendation',True) for q in D['top_papers'] if q['title']==p['title']):return '配置排除（workshop）'
  return '前10候选（跟踪）' if E['top_papers'][p['title']].get('reading_decision')=='跟踪' else '推荐'
 return '候选（未入前10）'
r.status=status
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3] if p.get('archive_status')=='downloaded'}
original=r.local_pdf
def publication_pdf(directory,stem):
 path=original(directory,stem)
 return path if path and path.relative_to(ROOT).as_posix() in tracked|new else None
r.local_pdf=publication_pdf
sys.argv=['render_note.py','--date','2026-10-16','--papers-json',str(DAY/'search_result.json'),'--editorial-json',str(DAY/'daily-editorial.json')]
b=io.StringIO()
with contextlib.redirect_stdout(b):r.main()
text=b.getvalue().replace('新增推荐 10 篇','预筛前十 10 篇（正式阅读建议9篇，配置排除1篇）')
text=text.replace('进入今日推荐 10 篇','进入预筛前十10篇（正式阅读建议9篇）')
text=text.replace('## 其余 7 篇推荐\n','## 其余 7 篇推荐\n\n以下保留预筛前十的其余7项，其中Workshop项已排除正式推荐；跟踪项尚不安排精读。\n')
for p in D['top_papers'][:3]:
 text=text.replace('- **英文摘要**：'+r.cell(p['summary']),'- **英文摘要**：\n\n'+p['summary'].strip(),1)
 text=text.replace('- **中文摘要**：'+E['top_papers'][p['title']]['abstract_zh'],'- **中文摘要（完整翻译）**：\n\n'+E['top_papers'][p['title']]['abstract_zh'],1)
 text=text.replace('- **来源**：arxiv\n','- **来源**：arXiv（'+p['published'][:10]+'首次提交，v1）\n- **摘要许可**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；作者署名见下，中文为完整翻译改编，英文摘要来自原文访问链接。\n',1)
 for q in p['related_papers']:
  original=' — 共同点：`'+', '.join(q['shared_terms'])+'`'
  text=text.replace(original,original+'；'+q['relationship'],1)
text=text.replace('可选保留，便于重渲染','已保存，可重渲染')
text=text.replace('| 编辑稿 | daily-editorial.json | 已保存，可重渲染 |\n','| 编辑稿 | daily-editorial.json | 已保存，可重渲染 |\n\n')
landing='''前三篇原始PDF已归档，入库日期2026-10-16。四阶段根索引与月份README已刷新，新增论文登记为 `未解析` / `未精读` / `未翻译`。

- [PDF统一登记](../../01-raw/index.md) · [解析索引](../../02-markdown/index.md) · [精读索引](../../03-notes/index.md) · [翻译索引](../../06-translation/index.md)
- [归档结果](archive-result.json) · [验证记录](validation.json)
- 全局索引检查有9处既有研究文档链接问题，位于10月8日/10月10日研究资料，指向工作区外资料；本次日报本地链接问题为0。详见验证记录。

'''
text=text.replace('## 附录：本次检索列表',landing+'## 附录：本次检索列表')
text=text.replace('按研究大方向分节；每节内部按研究优先级从高到低。','按配置关键词命中的研究大方向分节；Muon粒子物理同词误命中保留原分组并标记不推荐，同脉络仅为核验过的本地阅读入口，不代表已证实技术继承。每节内部按研究优先级从高到低。')
print(('# 2026-10-16 今日检索\n\n'+text).rstrip())
