from pathlib import Path
import urllib.request,json,hashlib
out=Path("07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai/sources");out.mkdir(parents=True,exist_ok=True)
urls={"sudoku-institution.html":"https://iris.polito.it/handle/11583/2992746", "drugmap-hal-api.json":"https://api.archives-ouvertes.fr/search/?q=halId_s%3Ahal-05067564&wt=json", "drugmap-tree.json":"https://api.github.com/repos/JUCompBio/DTI-MOA/git/trees/main?recursive=1", "sudoku-crossref.json":"https://api.crossref.org/works/10.1109%2Ftai.2025.3536428", "drugmap-crossref.json":"https://api.crossref.org/works/10.1109%2Ftai.2025.3565671"}
log=[]
for name,url in urls.items():
 try:
  req=urllib.request.Request(url,headers={"User-Agent":"Academic-writing-style-audit/1.0","Accept":"application/json,text/html"})
  with urllib.request.urlopen(req,timeout=20) as r: data=r.read(10_000_001);final=r.url
  if len(data)>10_000_000: raise RuntimeError("size limit")
  (out/name).write_bytes(data);log.append(dict(file=name,url=url,final_url=final,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));print(name,"saved",len(data))
 except Exception as e: log.append(dict(file=name,url=url,error=str(e)));print(name,str(e))
(out/"source-fetch-log.json").write_text(json.dumps(log,ensure_ascii=False,indent=2))
