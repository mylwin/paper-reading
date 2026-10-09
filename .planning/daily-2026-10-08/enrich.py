import sys,json
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path('.claude/skills/paper-daily/scripts').resolve()))
import search_arxiv as s
focus=['large language model optimizer','optimizer design','Muon','gradient clipping','parameter-free optimization','optimizer state memory']
c=s.load_research_config(None)
raw=s.search_arxiv_by_keywords(focus,datetime(2026,9,8),datetime(2026,10,8),max_results=200)
scored=s.filter_and_score_papers(raw,c,datetime(2026,10,8),False,focus)
Path('.planning/daily-2026-10-08/arxiv-full.json').write_text(json.dumps(scored,ensure_ascii=False,indent=2,default=str))
