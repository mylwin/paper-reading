import json,urllib.request,urllib.parse
from pathlib import Path
from search_html import parse,kw,RUN
params={'advanced':'','date-filter_by':'date_range','date-from_date':'2026-09-20','date-to_date':'2026-10-20','date-date_type':'submitted_date_first','abstracts':'show','size':200,'order':''}
for i,k in enumerate(['Muon','SignSGD','gradient clipping']):
 params[f'terms-{i}-operator']='AND' if i==0 else 'OR'
 params[f'terms-{i}-term']='"'+k+'"' if ' ' in k else k
 params[f'terms-{i}-field']='all'
url='https://arxiv.org/search/advanced?'+urllib.parse.urlencode(params)
req=urllib.request.Request(url,headers={'User-Agent':'paper-skills/0.1 (local research)'})
with urllib.request.urlopen(req,timeout=60) as f:raw=f.read().decode()
(RUN/'arxiv-advanced.html').write_text(raw)
papers=parse(raw)
(RUN/'arxiv-first-date-filtered.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2)+'\n')
(RUN/'arxiv-advanced-coverage.json').write_text(json.dumps({'url':url,'filter':'first_submission','limit':200,'page_records':raw.count('<li class="arxiv-result">'),'parsed':len(papers),'status':'ok'},ensure_ascii=False,indent=2)+'\n')
print('Original submission date filtered:',len(papers))
