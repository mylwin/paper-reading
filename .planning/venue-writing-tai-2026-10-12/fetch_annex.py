from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
out=Path("07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai/sources")
d=json.loads((out/"drugmap-hal-full.json").read_text())["response"]["docs"][0]
a=list(enumerate(d["fileAnnexes_s"]))
def run(x):
 i,url=x;name="drugmap-annex-"+str(i+1)+(".xlsx" if url.endswith(".xlsx") else ".pdf")
 try:
  req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
  with urllib.request.urlopen(req,timeout=25) as r:data=r.read(20_000_001);content=r.headers.get("Content-Type");final=r.url
  if len(data)>20_000_000:raise RuntimeError("size limit")
  valid=data.startswith(b"%PDF") if name.endswith(".pdf") else data.startswith(b"PK")
  if not valid:raise RuntimeError("wrong content "+str(content)+" "+repr(data[:100]))
  (out/name).write_bytes(data);return dict(file=name,url=url,final_url=final,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return dict(file=name,url=url,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 log=list(ex.map(run,a))
(out/"annex-fetch-log.json").write_text(json.dumps(log,ensure_ascii=False,indent=2));print(json.dumps(log,ensure_ascii=False,indent=2))
