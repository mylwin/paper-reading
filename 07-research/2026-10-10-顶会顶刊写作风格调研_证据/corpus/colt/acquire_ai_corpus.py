#!/usr/bin/env python3
"""Public proceedings acquisition. Extraction is NOT semantic review."""
import argparse, concurrent.futures, gzip, hashlib, html, json, re, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=Path(__file__).resolve().parents[2]
DATE='2026-10-10'

def plain(x):
    return html.unescape(re.sub(r'<[^>]+>','',x)).strip()

def topic(title):
    t=title.lower()
    for label,pattern in [('privacy','privat|differential|fair|strateg|calibrat'),('online-bandits-rl','bandit|regret|online|reinforcement|markov|mdp'),('optimization-sampling','optimiz|descent|sampling|gradient|convex|langevin'),('complexity-learnability','learnab|complexity|pac|lower bound|vc dimension|compression|halfspace'),('statistics-representation','regress|estimat|mixture|neural|network|clustering|distribution|test')]:
        if re.search(pattern,t):return label
    return 'other-theory'

def candidates(venue,year):
    src=EVIDENCE/'sources'/venue
    out=[]
    if venue=='colt':
        s=(src/f'{year}-index.html').read_text()
        s=s.split('<div class="paper">',1)[1]
        for block in s.split('<div class="paper">'):
            title=re.search(r'<p class="title">(.*?)</p>',block,re.S)
            pdf=re.search(r'href="([^"]+\.pdf)"',block)
            authors=re.search(r'<span class="authors">(.*?)</span>',block,re.S)
            pages=re.search(r'PMLR \d+:(\d+)-(\d+)',block)
            if not(title and pdf and pages):continue
            title=plain(title.group(1))
            if re.search('preface|extended abstract|open problem',title,re.I) or int(pages[2])-int(pages[1])<7:continue
            out.append(dict(title=title,pdf_url=pdf[1],accepted_url=re.search(r'href="([^"]+\.html)"',block)[1],authors=plain(authors[1]),stratum=topic(title),version='PMLR published full paper',source_index=f'https://proceedings.mlr.press/v{247 if year==2024 else 291}/'))
    elif venue=='ijcai':
        s=(src/f'{year}-index.html').read_text().split('<div class="section_title"><h3>Main Track</h3></div>',1)[1].split('<div class="section_title"><h3>',1)[0]
        parts=re.split(r'<div class="subsection_title">(.*?)</div>',s)
        for i in range(1,len(parts),2):
            for b in re.split(r'<div id="paper\d+" class="paper_wrapper">',parts[i+1])[1:]:
                title=plain(re.search(r'<div class="title">(.*?)</div>',b,re.S)[1]);pdf=re.search(r'href="(\d+\.pdf)"',b)[1]
                out.append(dict(title=title,pdf_url=f'https://www.ijcai.org/proceedings/{year}/{pdf}',accepted_url=f'https://www.ijcai.org/proceedings/{year}/{int(pdf[:-4])}',authors=plain(re.search(r'<div class="authors">(.*?)</div>',b,re.S)[1]),stratum=plain(parts[i]),version='IJCAI published main-track paper',source_index=f'https://www.ijcai.org/proceedings/{year}/'))
    else:
        for f in sorted(src.glob(f'{year}-*.html')):
            s=f.read_text()
            if 'Special Track' in plain(re.search(r'<h1[^>]*>(.*?)</h1>',s,re.S)[1]):continue
            parts=re.split(r'<h2[^>]*>(.*?)</h2>',s,flags=re.S)
            for i in range(1,len(parts),2):
                if 'AAAI Technical Track' not in plain(parts[i]):continue
                for b in parts[i+1].split('<div class="obj_article_summary">')[1:]:
                    a=re.search(r'<a id="article-(\d+)" href="([^"]+)">(.*?)</a>',b,re.S)
                    links=re.findall(r'href="(https://ojs\.aaai\.org/index.php/AAAI/article/view/\d+/\d+)"[^>]*>(.*?)</a>',b,re.S)
                    pdf=next((u.replace('/article/view/','/article/download/') for u,t in links if plain(t)=='PDF'),None)
                    auth=re.search(r'<div class="authors">(.*?)</div>',b,re.S)
                    if not (a and pdf and auth):continue
                    out.append(dict(title=plain(a[3]),pdf_url=pdf,accepted_url=a[2],authors=plain(auth[1]),stratum=plain(parts[i]).replace('AAAI Technical Track on ',''),version='AAAI Press published technical-track paper',source_index=f'https://ojs.aaai.org/index.php/AAAI/issue/view/{f.stem.split("-")[1]}'))
    return out

def lock(venue):
    dest=EVIDENCE/'corpus'/venue/'manifest.json'
    if dest.exists():return json.loads(dest.read_text())
    rows=[]
    for y in (2024,2025):
        groups={}
        for r in candidates(venue,y):groups.setdefault(r['stratum'],[]).append(r)
        for vals in groups.values():vals.sort(key=lambda r:hashlib.sha256((DATE+r['title']).encode()).hexdigest())
        chosen=[];first_authors=set()
        keys=sorted(groups)
        # Round-robin topic strata; author de-duplication within year.
        while len(chosen)<25:
            progressed=False
            for k in keys:
                vals=groups[k]
                while vals:
                    r=vals.pop(0);first=r['authors'].split(',')[0].strip()
                    if first not in first_authors:
                        chosen.append(r);first_authors.add(first);progressed=True;break
                if len(chosen)==25:break
            if not progressed:raise RuntimeError(f'Insufficient candidates: {venue} {y}')
        for n,r in enumerate(chosen):
            r.update(id=f'{venue}-{y}-{n+1:02d}',venue=venue,year=y,split='holdout' if n in (4,9,14,19,24) else 'induction',selection_order=n+1,selection_locked_at=DATE,semantic_review_status='pending',paper_type='pending semantic classification')
            rows.append(r)
    payload=dict(venue=venue,requested=50,selection_date=DATE,corpus_years=[2024,2025],method='Deterministic SHA256 order within source topical strata, round-robin; no repeated first author within year; holdout positions 5/10/15/20/25 locked before extraction. Sampling strata are metadata, not semantic conclusions.',records=rows)
    dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2));return payload

def fetch(r):
    stem=re.sub(r'[/\\:*?"<>|\x00-\x1f]','-',r['title']).strip().rstrip('.')
    existing=list((ROOT/'01-raw').glob(f'*/{stem}.pdf'))
    pdf=existing[0] if existing else ROOT/'01-raw'/'2026-10'/f'{stem}.pdf'
    try:
        if not pdf.exists():
            req=urllib.request.Request(r['pdf_url'],headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'})
            b=urllib.request.urlopen(req,timeout=40).read(50*1024*1024+1)
            if b[:2]==b'\x1f\x8b':b=gzip.decompress(b)
            if len(b)>50*1024*1024:raise ValueError('PDF exceeds 50 MB')
            if not b.startswith(b'%PDF'):raise ValueError('Response is not PDF')
            pdf.parent.mkdir(parents=True,exist_ok=True)
            with pdf.open('xb') as f:f.write(b)
        md=ROOT/'02-markdown'/pdf.parent.name/f'{stem}.md'
        if not md.exists():
            proc=subprocess.run(['pdftotext','-layout',str(pdf),'-'],capture_output=True,text=True,check=True)
            pages=proc.stdout.split('\f');md.parent.mkdir(parents=True,exist_ok=True)
            content='\n\n'.join(f'<!-- PDF page {i+1} -->\n\n{t.rstrip()}' for i,t in enumerate(pages) if t.strip())
            md.write_text(f'---\ntitle: {json.dumps(r["title"],ensure_ascii=False)}\nsource_pdf: {json.dumps(str(pdf.relative_to(ROOT)),ensure_ascii=False)}\nparser: pdftotext -layout\nparsed_date: {DATE}\nfigures: not extracted; text-only PDF parsing\n---\n\n'+content+'\n')
        text=md.read_text();info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True).stdout
        pg=re.search(r'^Pages:\s+(\d+)',info,re.M)
        r.update(acquisition_status='downloaded and text parsed',pdf_path=str(pdf.relative_to(ROOT)),markdown_path=str(md.relative_to(ROOT)),month=pdf.parent.name,pages=int(pg[1]) if pg else None,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),text_characters=len(text),extraction_review_status='automatic extraction only; not counted as semantic full-text review',appendix_coverage='pending semantic check; extracted all available PDF pages')
    except Exception as e:r.update(acquisition_status='failed',error=f'{type(e).__name__}: {e}')
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--venue',choices=['aaai','ijcai','colt'],required=True);ap.add_argument('--download',action='store_true');args=ap.parse_args()
    m=lock(args.venue)
    if args.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            for r in ex.map(fetch,m['records']):print(r['id'],r['acquisition_status'],r.get('pages'),flush=True)
        (EVIDENCE/'corpus'/args.venue/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
    (EVIDENCE/'corpus'/args.venue/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in m['records']))

if __name__=='__main__':main()
