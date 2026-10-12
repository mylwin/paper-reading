import pathlib,urllib.request,hashlib,json,subprocess
p=pathlib.Path('07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai-replacement')
urls={'snn-slimming-accepted-author.pdf':'https://pure-oai.bham.ac.uk/ws/files/219032020/LiY24.pdf','snn-repository.html':'https://research.birmingham.ac.uk/en/publications/efficient-structure-slimming-for-spiking-neural-networks/','ss-trboosting-arxiv-v1.pdf':'https://arxiv.org/pdf/2412.03212v1','ss-trboosting-crossref.json':'https://api.crossref.org/works/10.1109%2FTAI.2024.3350543','snn-slimming-crossref.json':'https://api.crossref.org/works/10.1109%2FTAI.2024.3352533'}
out=[]
for name,url in urls.items():
 f=p/name
 if f.exists():out.append({'file':str(f),'status':'exists-not-overwritten'});continue
 try:
  data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PaperStyleResearch/1.0'}),timeout=35).read(25_000_001)
  if len(data)>25_000_000:raise RuntimeError('oversize')
  if name.endswith('.pdf') and not data.startswith(b'%PDF'):raise RuntimeError('not PDF')
  f.write_bytes(data);out.append({'url':url,'file':str(f),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  if name.endswith('.pdf'):subprocess.run(['pdftotext','-layout',str(f),str(f.with_suffix('.txt'))],check=True)
 except Exception as e:out.append({'url':url,'status':'failed','error':str(e)})
(p/'acquisition.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
