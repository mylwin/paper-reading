"""Attach archive outcomes and verified, relevant local relationships."""
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-16'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import paper_config
D=json.loads((DAY/'search_result.json').read_text());A=json.loads((DAY/'archive-result.json').read_text())
for p,a in zip(D['top_papers'][:3],A['archived']):
 assert p['title']==a['title'];p.update(archive_status=a['status'],archive_detail=a['detail'],archive_path=a['path'],archive_month=a['month'])
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
raw=json.loads((RUN/'arxiv-raw.json').read_text())+json.loads((RUN/'openreview-raw.json').read_text());summaries={p['title']:p.get('summary','') for p in raw}
physics_terms=['muon collider','muon beam','muon anomalous','muon magnetic','muon-to-electron','lepton flavor','muography','muon detector','muon mean lifetime','muon decay','muon physics','hadronic light-by-light','muon g-2','muon $g-2','muon neutrino']
for p in D['all_papers']:
 text=(p['title']+' '+summaries.get(p['title'],'')).lower()
 if any(t in text for t in physics_terms):p['editorial_status']='词义误命中（不推荐）';p['related_papers']=[]
 elif p['title']==D['top_papers'][8]['title']:p['editorial_status']='配置排除（workshop）'
 seen=set();related=[]
 for q in p.get('related_papers',[]):
  path=Path(q.get('path',''))
  if not path.is_file() or q['title'] in seen:continue
  rel=path.relative_to(ROOT).as_posix()
  if rel not in tracked:continue
  if path.name=='search_result.json':
   note=path.parent/'今日检索.md'
   if not note.is_file() or note.relative_to(ROOT).as_posix() not in tracked or q['title'] not in note.read_text():continue
  related.append(q);seen.add(q['title'])
 p['related_papers']=related

def relation(title,path,shared,relationship):
 asset=ROOT/path;assert asset.is_file();assert path in tracked
 if asset.name=='search_result.json':assert title in json.dumps(json.loads(asset.read_text()),ensure_ascii=False)
 return {'title':title,'path':str(asset),'stem':paper_config.sanitize_paper_title(title),'source':'daily' if asset.name=='search_result.json' else ('note' if asset.name=='精读.md' else 'pdf'),'shared_terms':shared,'relationship':relationship}
refs=[
 [relation('PowerStep: Memory-Efficient Adaptive Optimization via lp-Norm Steepest Descent','03-notes/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent/精读.md',['optimizer state memory'],'互补：比较免存二阶状态与子空间状态加正交残差的两种内存设计，不能仅凭摘要断言谁更优。'),relation('Accelerating Muon with Spectral Residual Correction','08-daily/2026-10-11/search_result.json',['Muon','residual correction'],'待对照：同涉Muon残差修正，但此处是否为相同残差定义与分析对象需读全文。')],
 [relation('Muon with Finite Newton-Schulz: The Smoothing Benefit in Nonsmooth Nonconvex Optimization','01-raw/2026-09/Muon_with_Finite_Newton_Schulz_The_Smoothing_Benefit_in_Nonsmooth_Nonconvex_Optimization.pdf',['Muon','smoothing'],'互补：有限Newton–Schulz近似与小批量随机噪声属于不同的平滑来源，应分开控制。'),relation('Convergence of Muon with Newton-Schulz','03-notes/2026-10/Convergence_of_Muon_with_Newton_Schulz/精读.md',['Muon','spectral descent'],'互补：将已有离散实现收敛分析与新文连续噪声动力学对照，具体假设并不自动兼容。')],
 [relation('SignMuon: Communication-Efficient Distributed Muon for LLM Pretraining','08-daily/2026-10-13/search_result.json',['distributed LLM training','Muon'],'互补：通信压缩与局部训练内部优化器白化是不同路线；需区分通信成本与固定总预算下的模型性能。')]
]
for p,related in zip(D['top_papers'][:3],refs):
 p['related_papers']=related
 for q in D['all_papers']:
  if q['title']==p['title']:q['related_papers']=related
D['source_notes']=[{'title':D['top_papers'][8]['title'],'url':'https://arxiv.org/abs/2610.21555','comments':'Accepted at NeurIPS 2026 Workshop (Optimization for Machine Learning)','verified_on':'2026-10-16','effect':'人工排除正式推荐，保留原预筛结果'}]
(DAY/'search_result.json').write_text(json.dumps(D,ensure_ascii=False,indent=2)+'\n')
print('Attached archive outcomes and verified relationships')
