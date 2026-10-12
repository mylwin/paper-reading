import urllib.request, zipfile, io, json, hashlib
from pathlib import Path
root=Path.cwd()
out=root/'07-research/2026-10-11-顶会顶刊写作风格全文续研_证据/next-batch'
out.mkdir(exist_ok=True)
url='https://papers.nips.cc/paper_files/paper/2024/file/19f461132d8ed04764f58f88b7a8e842-Supplemental-Conference.zip'
req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
try:
 with urllib.request.urlopen(req,timeout=40) as response:
  data=response.read(50*1024*1024+1)
  if len(data)>50*1024*1024: raise ValueError('Supplement exceeds bounded50MiB fetch')
 with zipfile.ZipFile(io.BytesIO(data)) as archive:
  bad=archive.testzip()
  if bad: raise ValueError('Failed CRC: '+bad)
  members=[dict(path=x.filename,bytes=x.file_size) for x in archive.infolist()]
 dest=out/'neurips-2024-2aada42a-complete.zip'
 if dest.exists() and dest.read_bytes()!=data: raise ValueError('Refuse overwrite differing existing complete archive')
 dest.write_bytes(data)
 record=dict(id='neurips-2024-2aada42a',url=url,downloaded_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local_path=str(dest.relative_to(root)),zip_crc='passed',members=members,semantic_reading_complete=False,note='Earlier interrupted agent ZIP lacks central-directory end; preserved in planning, excluded from complete supplement assets. New intact archive not semantically read and never counted as completed paper.')
except Exception as e:
 record=dict(id='neurips-2024-2aada42a',url=url,status='failed',error=str(e),semantic_reading_complete=False)
(out/'supplement-acquisition.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
