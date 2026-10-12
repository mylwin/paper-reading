from pathlib import Path
import urllib.request,json,hashlib
out=Path("07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai/sources")
urls={"drugmap-hal-full.json":"https://api.archives-ouvertes.fr/search/?q=halId_s%3Ahal-05067564&fl=*&wt=json", "drugmap-README-fixed.md":"https://raw.githubusercontent.com/JUCompBio/DTI-MOA/6d46ebf10c82a277c8a216be54a05f153759fce2/README.md", "sudoku-xplore.html":"https://ieeexplore.ieee.org/document/10887036/", "drugmap-xplore.html":"https://ieeexplore.ieee.org/document/10980435/"}
log=[]
for name,url in urls.items():
 try:
  req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
  with urllib.request.urlopen(req,timeout=15) as r: data=r.read(10_000_001);final=r.url
  if len(data)>10_000_000:raise RuntimeError("size limit")
  (out/name).write_bytes(data);log.append(dict(file=name,url=url,final_url=final,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));print(name,len(data),flush=True)
 except Exception as e:log.append(dict(file=name,url=url,error=str(e)));print(name,str(e),flush=True)
(out/"extra-fetch-log.json").write_text(json.dumps(log,ensure_ascii=False,indent=2))
