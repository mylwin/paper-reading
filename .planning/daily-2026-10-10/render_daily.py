import contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DAY=ROOT/'08-daily/2026-10-10'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import render_note as renderer
D=json.loads((DAY/'search_result.json').read_text());titles={p['title'] for p in D['top_papers']}
def status(p):
 if p.get('already_known'):return '已在库（'+(p.get('kb_source') or 'unknown')+'）'
 text=(p['title']+' '+p.get('summary','')).lower()
 if any(t in text for t in ['muon collider','muon beam','muon magnetic','lepton flavor','electromagnetic isospin']):return '词义误命中（不推荐）'
 return '推荐' if p['title'] in titles else '候选（未入前10）'
renderer.status=status
sys.argv=['render_note.py','--date','2026-10-10','--papers-json',str(DAY/'search_result.json'),'--editorial-json',str(DAY/'daily-editorial.json')]
buf=io.StringIO()
with contextlib.redirect_stdout(buf):renderer.main()
text=buf.getvalue()
metadata=[
 '- **来源**：OpenReview（ICLR 2027 匿名投稿，创建日期2026-09-18；许可[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，中文摘要为翻译改编）\n- **版本核验**：同名公开预印本 [arXiv 2602.05298v2](https://arxiv.org/abs/2602.05298v2)，首次提交2026-02-05、更新2026-02-18。本地PDF为该版，摘要最大模型2.6B；下列投稿摘要最大4.4B，两版证据分开。公开预印本作者为Damien Ferbach、Courtney Paquette、Gauthier Gidel、Katie Everett、Elliot Paquette。\n',
 '- **来源**：OpenReview（OPT 2026 Poster条目，日期2026-10-02；公开作者信息）\n- **版本核验**：同名公开预印本 [arXiv 2607.29674v1](https://arxiv.org/abs/2607.29674v1)，首次提交2026-07-31；本地PDF归档该版。许可[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，中文摘要为翻译改编；下列英文摘要来自本次OpenReview检索。\n',
 '- **来源**：OpenReview（ICLR 2027 匿名投稿，创建日期2026-09-18；许可[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，中文摘要为翻译改编）\n- **版本核验**：本次未取得可归档全文；PDF接口HTTP403，保留来源入口。\n']
for replacement in metadata:text=text.replace('- **来源**：openreview\n',replacement,1)
text=text.replace('- **作者**：--','- **作者**：--（当前OpenReview匿名记录未披露）')
text=text.replace('可选保留，便于重渲染','已保存，可重渲染')
text=text.replace('\\\\times','\\times').replace('\\\\sim','\\sim')
landing='\n四阶段根索引与月份README已刷新。两篇已归档论文的入库日期均为2026-10-10，登记为`未解析` / `未精读` / `未翻译`；第三篇仅保留检索记录和PDF失败状态。\n\n- [PDF统一登记](../../01-raw/index.md) · [解析索引](../../02-markdown/index.md) · [精读索引](../../03-notes/index.md) · [翻译索引](../../06-translation/index.md)\n- [归档结果（含首次失败和备用版本）](archive-result.json)\n\n'
text=text.replace('## 附录：本次检索列表',landing+'## 附录：本次检索列表')
print('# 2026-10-10 今日检索\n\n'+text,end='')
