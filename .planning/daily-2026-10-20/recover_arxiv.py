import sys,json,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import search_arxiv as s
import logging
logging.basicConfig(level=logging.INFO)
c=s.load_research_config(None)
kw=list(dict.fromkeys(k for d in c['research_domains'].values() for k in d['keywords']))
papers=[];coverage=[]
for i in range(0,len(kw),5):
 batch=kw[i:i+5]
 got=s._search_arxiv_by_keywords_single(batch,datetime(2026,9,20),datetime(2026,10,20),max_results=100,max_retries=1,timeout=45)
 papers+=got;coverage.append({'keywords':batch,'records':len(got),'limit':100})
 (RUN/'arxiv-raw.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2,default=str)+'\n')
 (RUN/'arxiv-coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
 time.sleep(3)
