import sys,json,hashlib,subprocess,urllib.request
from pathlib import Path
root=Path.cwd()
sys.path.insert(0,str(root/'.claude/skills/paper-daily/scripts'))
from paper_config import sanitize_paper_title
base=root/'07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus/tmlr'
rows=[json.loads(s) for s in (base/'records.jsonl').read_text().splitlines()]
urls={'tmlr-2024-01':'https://arxiv.org/pdf/2309.16941v2','tmlr-2024-02':'https://arxiv.org/pdf/2303.16133v2','tmlr-2025-01':'https://arxiv.org/pdf/2506.22982v1'}
log=[]
for r in rows:
 if r['id'] not in urls: continue
 stem=sanitize_paper_title(r['title']); dest=root/'.claude/skills/paper-writing-shared/tmp/corpus/tmlr/pdf'/f'{stem}.pdf'; url=urls[r['id']]
 dest.parent.mkdir(parents=True,exist_ok=True)
 try:
  if not dest.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
   with urllib.request.urlopen(req,timeout=40) as f:data=f.read(50*1024*1024+1)
   if not data.startswith(b'%PDF') or len(data)>50*1024*1024:raise ValueError('Not valid bounded PDF')
   dest.write_bytes(data)
  data=dest.read_bytes()
  p=subprocess.run(['pdftotext','-layout',str(dest),'-'],capture_output=True,text=True,check=True)
  pages=[t for t in p.stdout.split('\f') if t.strip()]
  out=root/'.claude/skills/paper-writing-shared/tmp/corpus/tmlr/markdown'/f'{stem}.md'
  out.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists():out.write_text('# '+r['title']+'\n\nPDF text extraction with page boundaries; equations/tables may be lossy. No image extraction performed.\n\n'+'\n\n'.join('## PDF page '+str(i+1)+'\n\n```text\n'+t+'\n```' for i,t in enumerate(pages)))
  r.update(download_ok=True,download_message='Acquired version-pinned author arXiv deposit; publication/content audit pending',acquisition_url=url,acquired_version='author_arxiv_deposit_pending_publication_audit',pdf_path=str(dest.relative_to(root)),pdf_sha256=hashlib.sha256(data).hexdigest(),markdown_path=str(out.relative_to(root)),parse_status='full_pdf_text_extracted',page_count=len(pages),version_verification='pending')
  log.append(dict(id=r['id'],status='acquired',url=url,pages=len(pages),first_page=pages[0][:500]))
 except Exception as e:log.append(dict(id=r['id'],status='failed',url=url,error=str(e)))
(base/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(root/'.planning/venue-writing-next-research-2026-10-11/tmlr-acquisition-log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2))
print(json.dumps(log,ensure_ascii=False,indent=2))
