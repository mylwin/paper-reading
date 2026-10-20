import json,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
RUN=Path(__file__).parent
D=json.loads(Path('/tmp/paper-daily-20261020-candidates.json').read_text())
def verify(p):
 ident=urllib.parse.parse_qs(urllib.parse.urlparse(p['url']).query)['id'][0]
 url='https://api2.openreview.net/notes?'+urllib.parse.urlencode({'id':ident})
 try:
  with urllib.request.urlopen(url,timeout=35) as f:data=json.load(f)
  n=data['notes'][0]
  return {'title':p['title'],'status':'ok','url':url,'note':n}
 except Exception as e:return {'title':p['title'],'status':'failed','detail':str(e),'url':url}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:out=list(ex.map(verify,D['top_papers']))
(RUN/'source-verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for p in out:
 print(p['title'],p['status'])
 if p['status']=='ok':
  n=p['note'];print({k:n.get(k) for k in ['invitations','pdate','cdate','license']});print({k:v for k,v in n['content'].items() if k in ['venue','venueid','pdf','authors','title']})
