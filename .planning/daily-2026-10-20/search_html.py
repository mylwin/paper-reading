import concurrent.futures,html,json,re,sys,urllib.request,urllib.parse
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import search_arxiv as s
c=s.load_research_config(None)
kw=list(dict.fromkeys(k for d in c['research_domains'].values() for k in d['keywords']))
def plain(t):return ' '.join(html.unescape(re.sub('<[^>]+>','',t)).split())
def field(block,pattern):
 m=re.search(pattern,block,re.S);return m.group(1) if m else ''
def parse(raw):
 out=[]
 for b in re.findall(r'<li class="arxiv-result">(.*?)</li>',raw,re.S):
  url=field(b,r'<p class="list-title[^>]*><a href="([^"]+)"')
  ident=url.rsplit('/',1)[-1]
  title=plain(field(b,r'<p class="title[^>]*>(.*?)</p>'))
  abstract=plain(field(b,r'<span class="abstract-full[^>]*>(.*?)<a class="is-size-7"'))
  submitted=plain(field(b,r'>Submitted</span>(.*?);'))
  try:date=datetime.strptime(submitted,'%d %B, %Y')
  except ValueError:continue
  if not datetime(2026,9,20)<=date<=datetime(2026,10,20):continue
  authors=[plain(x) for x in re.findall(r'<a[^>]*>(.*?)</a>',field(b,r'<p class="authors">(.*?)</p>'),re.S)]
  comments=plain(field(b,r'<p class="comments[^>]*>(.*?)</p>'))
  out.append({'id':url,'arxiv_id':ident,'title':title,'summary':abstract,'authors':authors,'affiliations':[],'published':date.isoformat(),'published_date':date.isoformat(),'categories':re.findall(r'data-tooltip="[^"]*">([^<]+)</span>',b),'url':url,'pdf_url':'https://arxiv.org/pdf/'+ident,'source':'arxiv','source_comments':comments})
 return out

def query(term):
 url='https://arxiv.org/search/?'+urllib.parse.urlencode({'query':term,'searchtype':'all','abstracts':'show','order':'-announced_date_first','size':200})
 rec={'term':term,'url':url,'limit':200,'order':'newest announcement'}
 try:
  if term=='Muon':raw=Path('/tmp/paper-daily-arxiv-search.html').read_text()
  else:
   req=urllib.request.Request(url,headers={'User-Agent':'paper-skills/0.1 (local research)'})
   with urllib.request.urlopen(req,timeout=40) as f:raw=f.read().decode()
  papers=parse(raw);rec.update(status='ok',page_records=raw.count('<li class="arxiv-result">'),in_window=len(papers))
 except Exception as e:papers=[];rec.update(status='failed',detail=str(e))
 print(json.dumps(rec),file=sys.stderr,flush=True)
 return papers,rec
if __name__=='__main__':
 allpapers=[];coverage=[];seen=set()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
  for batch,record in e.map(query,kw):
   coverage.append(record)
   for p in batch:
    if p['arxiv_id'] not in seen:allpapers.append(p);seen.add(p['arxiv_id'])
   (RUN/'arxiv-raw.json').write_text(json.dumps(allpapers,ensure_ascii=False,indent=2)+'\n')
   (RUN/'arxiv-html-coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
 print('Unique arXiv in window:',len(allpapers))
