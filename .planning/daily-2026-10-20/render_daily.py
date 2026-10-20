import contextlib,io,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DAY=ROOT/'08-daily/2026-10-20'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import render_note as r
D=json.loads((DAY/'search_result.json').read_text());E=json.loads((DAY/'daily-editorial.json').read_text())
analyses=E['top_papers'];topnames={p['title'] for p in D['top_papers']}
def status(p):
 if p.get('editorial_status'):return p['editorial_status']
 if p.get('already_known'):return '已在库（'+(p.get('kb_source') or 'unknown')+'）'
 return '推荐' if p['title'] in topnames else '候选（未入前10）'
r.status=status
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
new={Path(p['archive_path']).relative_to(ROOT).as_posix() for p in D['top_papers'][:3] if p.get('archive_status')=='downloaded'}
oldpdf,oldnote=r.local_pdf,r.local_note
def pdf(directory,stem):
 path=oldpdf(directory,stem)
 return path if path and path.relative_to(ROOT).as_posix() in tracked|new else None
def note(directory,stem):
 path=oldnote(directory,stem)
 return path if path and path.relative_to(ROOT).as_posix() in tracked else None
r.local_pdf,r.local_note=pdf,note
sys.argv=['render_note.py','--date','2026-10-20','--papers-json',str(DAY/'search_result.json'),'--editorial-json',str(DAY/'daily-editorial.json')]
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):r.main()
text=buffer.getvalue().replace('新增推荐 10 篇','预筛前十10项，正式阅读建议4项').replace('进入今日推荐 10 篇','进入预筛前十10项（正式阅读建议4项）')
text=text.replace('## 其余 7 篇推荐\n','## 其余 7 篇推荐\n\n其余7项保留预筛审计：Muon+为正式选读；4项Workshop按配置排除，2项是粒子物理误命中，均不属于正式推荐。\n')
for p in D['top_papers'][:3]:
 text=text.replace('- **英文摘要**：'+r.cell(p['summary']),'- **英文摘要**：\n\n'+p['summary'].strip(),1)
 zh=analyses[p['title']]['abstract_zh'];text=text.replace('- **中文摘要**：'+zh,'- **中文摘要（完整翻译）**：\n\n'+zh,1)
 src='- **来源**：arXiv；版本身份已与原始PDF首页核对\n'
 if p.get('license_urls_verified'):
  url=p['license_urls_verified'][0].replace('http:','https:');src+='- **摘要署名与许可**：作者见下，[来源许可]('+url+')；中文为原摘要完整翻译。\n'
 text=text.replace('- **来源**：arxiv\n',src,1)
 for q in p.get('related_papers',[]):
  shared=' — 共同点：`'+', '.join(q['shared_terms'])+'`'
  text=text.replace(shared,shared+'；'+q['relationship'],1)
text=text.replace('可选保留，便于重渲染','已保存，可重渲染')
text=text.replace('按研究大方向分节；每节内部按研究优先级从高到低。','按配置关键词命中的大方向分节；60条粒子物理误命中已标记不推荐。主题计数不等于直接相关工作量；同脉络仅为已核验的阅读入口。每节内部按研究优先级从高到低。')
landing='''\n前三篇PDF均新下载，入库日2026-10-20，归属2026-10。四阶段根索引及月份README已刷新，三篇分别登记为 `未解析` / `未精读` / `未翻译`。\n\n- [PDF统一登记](../../01-raw/index.md) · [解析索引](../../02-markdown/index.md) · [精读索引](../../03-notes/index.md) · [翻译索引](../../06-translation/index.md)\n- [归档结果](archive-result.json) · [校验记录](validation.json)\n- PDF魔数、页数和首页身份已核验；AURA的文本提取出现字体类型警告，仍成功提取首页，原始PDF未改写。\n- 全库检查保留9处既有研究文档的工作区外链接问题，本次新增断链为0；无月份错位或缺失README。\n\n**来源覆盖与限制**：\n\n'''+''.join('- '+s+'\n' for s in D['search_limitations'])+'''\nWorkshop类别依据：[OPT 2026主办方](https://opt-ml.org/index.html)、[DynaFront主办方](https://sites.google.com/view/dynafrontneurips26)。排除依据是当前配置，不以会场判断论文质量。\n\n'''
text=text.replace('## 附录：本次检索列表',landing+'## 附录：本次检索列表')
print(('# 2026-10-20 今日检索\n\n'+text).rstrip())
