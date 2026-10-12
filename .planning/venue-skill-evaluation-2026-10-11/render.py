"""Render retained actual PDFs; visual judgments are recorded separately."""
from pathlib import Path
import subprocess,json,hashlib
W=Path(__file__).resolve().parents[2]
E=W/'07-research/2026-10-11-论文写作润色技能评测_证据'
rows=[]
for name,stem in [('E02-icml-writing','main'),('E03-revised','article'),('E04-tai-migrated','main'),('E05-independent-iclr','main'),('E08-author-leak','main'),('E08-macro-title','main'),('E10-independent-migration/icml-polished','article'),('E10-independent-migration/tai-submission','article')]:
    pdf=E/'outputs'/name/'build'/f'{stem}.pdf'
    d=E/'renders'/name;d.mkdir(parents=True,exist_ok=True)
    subprocess.run(['pdftoppm','-r','85','-png',str(pdf),str(d/'page')],check=True,capture_output=True)
    pages=sorted(d.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
    for start in range(0,len(pages),4):
        subprocess.run(['magick','montage',*[str(p) for p in pages[start:start+4]],'-thumbnail','750x','-tile','2x','-geometry','+8+8','-depth','8',str(d/f'contact-{start//4+1}.png')],check=True,capture_output=True)
    text=subprocess.run(['pdftotext','-layout',str(pdf),'-'],capture_output=True,text=True,check=True).stdout
    (d/'pdf-text.txt').write_text(text)
    rows.append({'case':name,'pdf':str(pdf.relative_to(E)),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(pages),'contacts':[str(p.relative_to(E)) for p in sorted(d.glob('contact-*.png'))],'visual_review':'pending direct image inspection'})
(E/'runs/render-manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'case':r['case'],'pages':r['pages']} for r in rows],indent=2))
