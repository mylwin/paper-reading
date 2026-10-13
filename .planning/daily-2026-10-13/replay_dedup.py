"""Replay acquired records through the skill, with one verified title alias."""
import json,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import search_arxiv as s
original_scan=s.scan_known_papers
alias='PowerStep: Memory-Efficient Adaptive Optimization via $\\ell_p$-Norm Steepest Descent'
asset=ROOT/'01-raw/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent.pdf'
assert asset.is_file()
def known(*args,**kwargs):
 data=original_scan(*args,**kwargs)
 # Today's intermediate search result is not recommendation history.
 for key,path in list(data['by_daily'].items()):
  if Path(path).parent.name=='2026-10-13':
   del data['by_daily'][key]
   if key not in data['by_pdf'] and key not in data['by_note']:
    data['stems'].discard(key)
    data['reference'].pop(key,None)
 for title in [alias,alias.replace('\\ell','\\\\ell'),s.title_to_note_filename(alias)]:
  key=s.clean_stem(title);data['stems'].add(key);data['by_pdf'][key]=str(asset)
 return data
s.scan_known_papers=known
def cached(source,**kwargs):
 papers=json.loads((RUN/(source+'-raw.json')).read_text())
 for p in papers:
  if isinstance(p.get('published_date'),str):p['published_date']=datetime.fromisoformat(p['published_date'].replace('Z','+00:00'))
 return papers
s.fetch_with_paper_config=cached
s.search_hot_papers_from_categories=lambda **kwargs:[]
raise SystemExit(s.main())
