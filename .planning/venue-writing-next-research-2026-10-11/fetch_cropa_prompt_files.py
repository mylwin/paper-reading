import json,urllib.request,hashlib
from pathlib import Path
p=Path('07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/tmlr/supplement-audit')
j=json.loads((p/'cropa-repository-tree.json').read_text());sha=j['sha'];log=[]
for key in ['README.md','data/prompt.txt','data/config.json','utils/eval_tools.py','utils/eval_datasets.py']:
 u=f'https://raw.githubusercontent.com/Swadesh06/Revisting_CroPA/{sha}/{key}'
 with urllib.request.urlopen(u,timeout=40) as f:b=f.read(200000)
 dest=p/('cropa-'+key.replace('/','--'));dest.write_bytes(b)
 log.append(dict(path=key,url=u,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),local_path=str(dest)))
(p/'cropa-prompt-file-provenance.json').write_text(json.dumps(log,indent=2))
print(json.dumps(log,indent=2))
