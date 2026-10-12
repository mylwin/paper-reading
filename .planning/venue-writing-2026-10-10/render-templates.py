from pathlib import Path
import subprocess,json,concurrent.futures,hashlib
workspace=Path(__file__).resolve().parents[2]
r=workspace/'07-research/2026-10-10-顶会顶刊写作风格调研_证据/templates'
build=r/'builds';out=r/'renders';out.mkdir(exist_ok=True)
previous={x['project']:x for x in json.loads((r/'render-results.json').read_text())} if (r/'render-results.json').exists() else {}
def render(p):
 name=p.parents[1].name+('' if p.stem=='main' else '-supplement');d=out/name;d.mkdir(exist_ok=True)
 proc=subprocess.run(['pdftoppm','-r','85','-png',str(p),str(d/'page')],capture_output=True,text=True)
 if proc.returncode:return {'project':name,'error':proc.stderr}
 # Restrict to the actual current page count so old render files cannot inflate coverage.
 import re
 info=subprocess.check_output(['pdfinfo',str(p)],text=True)
 count=int(re.search(r'Pages:\s*(\d+)',info).group(1));pages=sorted(d.glob('page-*.png'))[:count];sheets=[]
 for i in range(0,len(pages),4):
  target=d/f'contact-{i//4+1}.png';cmd=['magick','montage']
  for q in pages[i:i+4]:cmd+=['-label',name+' / '+q.stem,str(q)]
  cmd+=['-font','Helvetica','-pointsize','14','-background','white','-fill','black','-tile','2x','-geometry','600x+8+14',str(target)]
  proc=subprocess.run(cmd,capture_output=True,text=True)
  if proc.returncode:return {'project':name,'error':proc.stderr}
  sheets.append(str(target.relative_to(r)))
 result={'project':name,'pdf':str(p.relative_to(r)),'pages':len(pages),'renders':[str(p.relative_to(r)) for p in pages],'contact_sheets':sheets,'pdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'visual_review_status':'pending'}
 old=previous.get(name,{})
 if old.get('pdf_sha256')==result['pdf_sha256']:
  result.update({k:v for k,v in old.items() if k not in result or k=='visual_review_status'})
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 a=list(ex.map(render,sorted(build.glob('*/build/main.pdf'))+sorted(build.glob('*/build/supplement.pdf'))))
(r/'render-results.json').write_text(json.dumps(a,indent=2))
print(json.dumps([{'project':x['project'],'pages':x.get('pages'),'error':x.get('error')} for x in a],indent=2))
