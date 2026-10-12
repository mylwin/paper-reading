#!/usr/bin/env python3
"""Navigation aid only. Generated packets never certify semantic review."""
import json,re,subprocess
from pathlib import Path
E=Path(__file__).resolve().parents[2];ROOT=E.parent.parent
for venue in ['aaai','ijcai','colt']:
 m=json.loads((E/'corpus'/venue/'manifest.json').read_text())
 folder=E/'corpus'/venue/'navigation';folder.mkdir(exist_ok=True)
 for r in m['records']:
  if 'pdf_path' not in r:continue
  txt=subprocess.run(['pdftotext',str(ROOT/r['pdf_path']),'-'],capture_output=True,text=True,check=True).stdout
  pages=txt.split('\f');packet=[f"# {r['id']} {r['title']} ({r['split']})",'Navigation aid: paragraph excerpts, NOT a completed full-text review.']
  for n,p in enumerate(pages):
   if not p.strip():continue
   paras=[re.sub(r'\s+',' ',x).strip() for x in re.split(r'\n\s*\n',p) if x.strip()]
   headings=[x for x in paras if len(x)<95 and len(x.split())<13 and re.search('[A-Za-z]{3}',x) and not re.search('Conference|Intelligence|University|@|Copyright|^[0-9]+$|^Figure|^Table',x)]
   # The longest prose paragraph avoids page headers, tables and isolated formulas.
   prose=sorted([x for x in paras if len(x)>200 and sum(c.isalpha() for c in x)/len(x)>.65],key=len,reverse=True)
   packet.append(f'\n## PDF p.{n+1}\nHeadings / short blocks: '+ ' | '.join(headings[:16]))
   if prose:packet.append('Prose locator: '+prose[0][:220]+' … '+prose[0][-140:])
  (folder/(r['id']+'.md')).write_text('\n'.join(packet)+'\n')
