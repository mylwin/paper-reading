#!/usr/bin/env python3
"""Public JMLR/TMLR corpus acquisition. Extraction is NOT semantic review."""
import argparse, concurrent.futures, hashlib, html, json, re, subprocess, sys
from pathlib import Path
from urllib.parse import urljoin

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
CORPUS_CACHE=ROOT/'.claude/skills/paper-writing-shared/tmp/corpus'
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
from fetch_pdfs import download_pdf,index_existing_pdfs
from paper_config import sanitize_paper_title,clean_stem

def group(title):
    t=title.lower()
    for name,pat in [('theory','bound|complexity|convergence|regret|estimat|minimax|identifi|theor'),('evaluation','benchmark|evaluat|reproduc|empirical|understand|analysis|analyz'),('learning','reinforcement|policy|bandit|agent|game|explor'),('generative','generat|diffusion|language|transformer|llm|text|flow|foundation'),('representation','graph|vision|image|feature|represent|classif|segment|multimodal')]:
        if re.search(pat,t): return name
    return 'other'

def candidates(v):
    out=[]
    if v=='jmlr':
        for year,vol in [(2024,25),(2025,26)]:
            s=Path(f'/tmp/jmlr{vol}.html').read_text()
            for b in re.findall(r'<dl>(.*?)</dl>',s,re.S):
                title=html.unescape(re.sub('<.*?>','',re.search(r'<dt>(.*?)</dt>',b,re.S).group(1))).strip()
                links=re.findall(r'href=[\'\"]([^\'\"]+)',b)
                pdf=next((x for x in links if x.endswith('.pdf')),None)
                ab=next((x for x in links if x.endswith('.html')),None)
                if pdf and ab and not re.search('survey|software|toolbox|package|library|editorial',title,re.I):
                    out.append(dict(title=title,year=year,pdf_url=urljoin('https://jmlr.org',pdf),publication_url=urljoin('https://jmlr.org',ab),authors=html.unescape(re.sub('<.*?>','',re.search(r'<i>(.*?)</i>',b,re.S).group(1))),version='publisher_final',source_index=f'https://jmlr.org/papers/v{vol}/'))
    else:
        s=Path('/tmp/tmlr-papers.html').read_text()
        for b in re.findall(r'<li class="item .*?</li>',s,re.S):
            m=re.search(r'</i>, ([A-Z][a-z]+) (202[45])',b)
            if not m: continue
            title=html.unescape(re.sub('<.*?>','',re.search(r'<b>(.*?)</b>',b,re.S).group(1))).strip()
            if re.search('survey|review of|position paper',title,re.I): continue
            pdf=re.search(r'href="(https://openreview.net/pdf[^\"]+)',b).group(1)
            pub=re.search(r'href="(https://openreview.net/forum[^\"]+)',b).group(1)
            out.append(dict(title=title,year=int(m.group(2)),month=m.group(1),pdf_url=html.unescape(pdf),publication_url=html.unescape(pub),authors=html.unescape(re.sub('<.*?>','',re.search(r'<i>(.*?)</i>',b,re.S).group(1))),version='official_published_openreview_current',source_index='https://jmlr.org/tmlr/papers/'))
    for x in out: x['provisional_stratum']=group(x['title'])
    return out

def lock(v):
    rows=candidates(v); selected=[]
    for year in [2024,2025]:
        pools={g:sorted([r for r in rows if r['year']==year and r['provisional_stratum']==g],key=lambda r:hashlib.sha256(r['publication_url'].encode()).hexdigest()) for g in sorted(set(r['provisional_stratum'] for r in rows))}
        sample=[];seen=set()
        while len(sample)<25:
            for g,pool in pools.items():
                while pool:
                    r=pool.pop(0);a=r['authors'].split(',')[0]
                    if a not in seen:
                        seen.add(a);sample.append(r);break
                if len(sample)==25:break
        for i,r in enumerate(sample):
            r.update(id=f'{v}-{year}-{i+1:02}',split='derivation' if i<20 else 'holdout',selection_locked='2026-10-10',semantic_review_status='not_reviewed',type_status='title_based_provisional')
        selected.extend(sample)
    d=BASE/v;d.mkdir(exist_ok=True)
    manifest=dict(venue=v,target=50,selection_method='Deterministic SHA256 rank within six title-based strata, round-robin; unique first authors per year; 20 derivation+5 holdout locked before downloads. Title strata are provisional, not semantic paper types.',publication_year_basis='official journal publication index',locked_before_analysis=True,papers=selected)
    (d/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    source=BASE.parent/'sources'/v;source.mkdir(exist_ok=True)
    files=['/tmp/jmlr25.html','/tmp/jmlr26.html'] if v=='jmlr' else ['/tmp/tmlr-papers.html']
    for f in files:(source/Path(f).name).write_bytes(Path(f).read_bytes())
    return selected

def acquire(r):
    existing=index_existing_pdfs(CORPUS_CACHE)
    stem=sanitize_paper_title(r['title'])
    venue=r['id'].split('-')[0]
    dest=existing.get(clean_stem(stem),CORPUS_CACHE/venue/'pdf'/f'{stem}.pdf')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():ok,msg=True,'reused existing PDF'
    else:ok,msg=download_pdf(r['pdf_url'],dest,45)
    r.update(download_ok=ok,download_message=msg)
    if ok and dest.stat().st_size>50*1024*1024:
        if msg!='reused existing PDF':dest.unlink()
        r.update(download_ok=False,download_message='over 50MB, rejected; replacement pending');return r
    if ok:
        r['pdf_path']=str(dest.relative_to(ROOT));r['pdf_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
        out=CORPUS_CACHE/venue/'markdown'/f'{dest.stem}.md';out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():
            p=subprocess.run(['pdftotext','-layout',str(dest),'-'],capture_output=True,text=True)
            if p.returncode==0:
                pages=p.stdout.split('\f')
                out.write_text('# '+r['title']+'\n\nPDF text extraction with page boundaries; equations/tables may be lossy. No image extraction performed.\n\n'+'\n\n'.join('## PDF page '+str(i+1)+'\n\n```text\n'+t+'\n```' for i,t in enumerate(pages) if t.strip()))
            else:r['parse_error']=p.stderr
        if out.exists():r['markdown_path']=str(out.relative_to(ROOT));r['parse_status']='full_pdf_text_extracted';r['semantic_review_status']='not_reviewed'
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('venue',choices=['jmlr','tmlr']);ap.add_argument('--download',action='store_true');a=ap.parse_args()
    d=BASE/a.venue
    papers=json.loads((d/'manifest.json').read_text())['papers'] if (d/'manifest.json').exists() else lock(a.venue)
    if a.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:rows=list(ex.map(acquire,papers))
        (d/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (d/'validation.json').write_text(json.dumps(dict(venue=a.venue,locked=len(rows),downloaded=sum(r.get('download_ok',False) for r in rows),parsed=sum('markdown_path' in r for r in rows),semantic_fulltext_reviewed=0,holdout_semantic_validation='pending',automated_extraction_is_not_fulltext_review=True),indent=2))
        print(a.venue, sum(r.get('download_ok',False) for r in rows),'downloaded',sum('markdown_path' in r for r in rows),'parsed; semantic review pending')
    else:print(a.venue,len(papers),'locked')
if __name__=='__main__':main()
