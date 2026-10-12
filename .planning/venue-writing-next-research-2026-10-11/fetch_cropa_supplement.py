import urllib.request,json,hashlib
from pathlib import Path
p=Path('07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/tmlr/supplement-audit');p.mkdir(exist_ok=True)
req=urllib.request.Request('https://api.github.com/repos/Swadesh06/Revisting_CroPA/git/trees/main?recursive=1',headers={'User-Agent':'paper-writing-research'})
with urllib.request.urlopen(req,timeout=40) as f:j=json.load(f)
(p/'cropa-repository-tree.json').write_text(json.dumps(j,indent=2))
print('commit',j['sha'],'truncated',j.get('truncated'))
for r in j.get('tree',[]):
 if r['type']=='blob' and ('prompt' in r['path'].lower() or r['path'].endswith('.pdf') or r['path'].endswith('.md')):print(r['path'],r.get('size'))
