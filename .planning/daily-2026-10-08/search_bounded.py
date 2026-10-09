import sys,json,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path('.claude/skills/paper-daily/scripts').resolve()))
import search_arxiv as s

def bounded(keywords,start_date,end_date,max_results=200,**kwargs):
    def query(term):
        params={'term':term,'limit':100,'offset':0,'content':'all','source':'forum'}
        req=urllib.request.Request(s.OPENREVIEW_SEARCH_URL+'?'+urllib.parse.urlencode(params),headers={'User-Agent':s.OPENREVIEW_USER_AGENT})
        try:
            with urllib.request.urlopen(req,timeout=30) as r: data=json.load(r)
            papers=[]
            for n in data.get('notes',[]):
                p=s.parse_openreview_note(n)
                if p and (p.get('published_date') is None or start_date.date()<=p['published_date'].date()<=end_date.date()): papers.append(p)
            print(f'OpenReview bounded {term}: {len(data.get("notes",[]))} retrieved, {len(papers)} in window',file=sys.stderr)
            return papers
        except Exception as e:
            print(f'OpenReview bounded {term}: {e}',file=sys.stderr);return []
    seen=set(); result=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for batch in ex.map(query,['Muon','optimizer','gradient clipping','parameter-free']):
            for p in batch:
                key=p.get('id') or p['title']
                if key not in seen: seen.add(key);result.append(p)
    return result
s.SOURCE_REGISTRY['openreview']['fetch']=bounded
sys.exit(s.main())
