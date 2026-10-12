#!/usr/bin/env python3
"""Public official corpus acquisition; extracted text is NOT a semantic review."""
import argparse, concurrent.futures, hashlib, html, json, re, subprocess, urllib.request, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'07-research/2026-10-10-顶会顶刊写作风格调研_证据'
STRATA={
 'theory_statistics':r'\b(theor|provab|convergence|regret|bound|generalization|statistical|sample complexity|identifi|optimality)',
 'decision_rl':r'\b(reinforcement|policy|bandit|decision|planning|markov|control|robot)',
 'trustworthy':r'\b(robust|adversarial|privacy|private|fair|safety|causal|causality|uncertainty|calibration)',
 'generative':r'\b(diffusion|generative|generation|flow matching|autoregressive|denoising|synthesis)',
 'representation_application':r'.'
}
def plain(x): return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>','',x))).strip()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def candidates(venue,year,path):
 s=Path(path).read_text(); out=[]
 if venue=='icml':
  for b in re.findall(r'<div class="paper">(.*?)</div>',s,re.S):
   title=plain(re.search(r'<p class="title">(.*?)</p>',b,re.S)[1]);
   if title.startswith('Position:'):continue
   links=re.findall(r'href="([^"]+)"[^>]*>([^<]+)</a>',b)
   a=plain(re.search(r'<span class="authors">(.*?)</span>',b,re.S)[1])
   d=dict(title=title,authors=a,landing_url=next(u for u,t in links if t=='abs'),pdf_url=next(u for u,t in links if t=='Download PDF'))
   out.append(d)
 elif venue=='neurips':
  for b in re.findall(r'<li class="conference"[^>]*>(.*?)</li>',s,re.S):
   m=re.search(r'<a title="paper title" href="([^"]+)">(.*?)</a>',b,re.S)
   if not m:continue
   d=dict(title=plain(m[2]),authors=plain(re.search(r'class="paper-authors">(.*?)</span>',b,re.S)[1]),landing_url='https://papers.nips.cc'+m[1],pdf_url=None)
   out.append(d)
 elif venue=='iclr':
  for u,title in re.findall(r'<li><a href="(/virtual/\d{4}/poster/\d+)">(.*?)</a></li>',s,re.S):
   out.append(dict(title=plain(title),authors=None,landing_url='https://iclr.cc'+u,pdf_url=None))
 for d in out:
  d.update(venue=venue,year=year,track='main conference research',official_listing_url={'icml':f'https://proceedings.mlr.press/v{235 if year==2024 else 267}/','neurips':f'https://papers.nips.cc/paper_files/paper/{year}'+('/vol38-main-conference' if year==2025 else ''),'iclr':f'https://iclr.cc/virtual/{year}/papers.html'}[venue],sampling_stratum=next(k for k,p in STRATA.items() if re.search(p,d['title'],re.I)),paper_type='pending semantic classification')
 return out
def lock(venue,files):
 papers=[]; frames=[]
 for year,path in files:
  data=candidates(venue,year,path);frames.append(dict(year=year,candidates=len(data),source_file=path,source_complete='</html>' in Path(path).read_text().lower()))
  used_authors=set()
  for category in STRATA:
   pool=sorted([d for d in data if d['sampling_stratum']==category],key=lambda d:hashlib.sha256(('2026-10-10-venue-writing|'+d['title']).encode()).hexdigest())
   selected=[]
   for d in pool:
    author=(d['authors'] or d['title']).split(',')[0].strip().lower()
    if author in used_authors:continue
    used_authors.add(author);selected.append(d)
    if len(selected)==5:break
   if len(selected)<5:raise RuntimeError(f'{venue} {year}: insufficient {category} ({len(selected)})')
   for i,d in enumerate(selected):
    d['split']='holdout' if i==4 else 'induction'; d['id']=f'{venue}-{year}-'+hashlib.sha256(d['title'].encode()).hexdigest()[:8];papers.append(d)
 manifest=dict(schema_version=1,venue=venue,created='2026-10-10',sampling_method='five topic proxies from titles; deterministic SHA256 within each stratum; four induction and one holdout per stratum/year; distinct first authors where metadata available; semantic paper-type classification pending',sampling_frame=frames,split_locked_before_pdf_review=True,target_count=50,papers=papers)
 folder=BASE/'corpus'/venue;folder.mkdir(parents=True,exist_ok=True);dump(folder/'manifest.json',manifest)
 return manifest
def get(url,max_bytes=52_428_800):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (academic corpus verification)'})
 with urllib.request.urlopen(req,timeout=45) as f:
  data=f.read(max_bytes+1)
 if len(data)>max_bytes:raise RuntimeError('asset exceeds 50 MiB')
 return data
def acquire(d):
 d=dict(d);stem=re.sub(r'[/:*?"<>|\\]','-',d['title']).strip(' .'); d['paper_stem']=stem
 try:
  if not d['pdf_url']:
   cached=BASE/'sources'/d['venue']/(d['id']+('-proceedings' if 'proceedings.iclr.cc' in d['landing_url'] else '')+'.html')
   page=cached.read_text() if cached.exists() else get(d['landing_url']).decode('utf-8')
   if not cached.exists():cached.write_text(page)
   meta_authors=re.findall(r'<meta name="citation_author" content="([^"]+)"',page)
   if meta_authors:d['authors']=', '.join(html.unescape(x) for x in meta_authors)
   if d['venue']=='neurips' or 'proceedings.iclr.cc' in d['landing_url']:
    urls=re.findall(r'<meta name="citation_pdf_url" content="([^"]+)"',page)
    if not urls:urls=re.findall(r'href=[\'"]([^\'"]+)[\'"][^>]*>\s*Paper\s*</a>',page)
    if not urls:urls=re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]',page)
    if not urls:raise RuntimeError('No official paper PDF discovered on landing page')
    d['pdf_url']=urllib.parse.urljoin(d['landing_url'],html.unescape(urls[0]))
   else:
    ids=re.findall(r'https://openreview.net/forum\?id=([A-Za-z0-9_-]+)',page)
    if not ids:raise RuntimeError('No OpenReview link discovered')
    d['openreview_url']='https://openreview.net/forum?id='+ids[0];d['pdf_url']='https://openreview.net/pdf?id='+ids[0]
    a=re.search(r'class="authors">(.*?)</',page,re.S)
    if a:d['authors']=plain(a[1])
  existing=list((ROOT/'01-raw').glob('????-??/'+stem+'.pdf'))
  if existing:
   pdf=existing[0];d['asset_reused']=True
  else:
   data=get(d['pdf_url'])
   if not data.startswith(b'%PDF'):raise RuntimeError('Source returned non-PDF (HTML/challenge); not archived as PDF')
   pdf=ROOT/'01-raw/2026-10'/(stem+'.pdf');pdf.parent.mkdir(exist_ok=True,parents=True)
   with pdf.open('xb') as f:f.write(data)
   d['asset_reused']=False
  d.update(pdf_path=str(pdf.relative_to(ROOT)),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pdf_version='official proceedings PDF; exact bytes locked by SHA256' if d['venue']!='iclr' or 'proceedings.iclr.cc' in d['landing_url'] else 'OpenReview public PDF; final-version classification pending',download_status='available')
  month=pdf.parent.name;md=ROOT/'02-markdown'/month/(stem+'.md');md.parent.mkdir(parents=True,exist_ok=True)
  if not md.exists():
   p=subprocess.run(['pdftotext','-enc','UTF-8',str(pdf),'-'],capture_output=True,text=True,check=True)
   pages=p.stdout.split('\f');pages=pages[:-1] if not pages[-1].strip() else pages
   if sum(len(x.strip()) for x in pages)<2000:raise RuntimeError('Too little extracted text; OCR required')
   text='# '+d['title']+'\n\n> PDF纯文本解析（pdftotext）；保留全文与页码。公式和图表的视觉布局需回看原PDF。本文件没有插图或语义批注。\n\n'
   text+='\n\n'.join(f'## PDF page {i+1}\n\n'+t.strip() for i,t in enumerate(pages))+'\n'
   md.write_text(text)
  d.update(markdown_path=str(md.relative_to(ROOT)),parse_status='complete text extraction',semantic_review_status='pending',reviewed_pages=[],appendix_review_status='pending',structure=[],claims=[],counterexamples=[],semantic_review=None)
  d.pop('error',None)
 except Exception as e:d.update(download_status='failed' if 'pdf_path' not in d else 'available',parse_status=d.get('parse_status','pending'),semantic_review_status='pending',error=type(e).__name__+': '+str(e))
 return d
def run(manifest_path):
 path=Path(manifest_path);m=json.loads(path.read_text());results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  for d in ex.map(acquire,m['papers']):
   results.append(d);print(d['id'],d['download_status'],d.get('error',''),flush=True)
   (path.parent/'records.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results))
 m['papers']=results;dump(path,m)
 dump(path.parent/'validation.json',dict(venue=m['venue'],locked_count=len(results),downloaded_count=sum(x['download_status']=='available' for x in results),parsed_count=sum(x.get('parse_status')=='complete text extraction' for x in results),semantic_reviewed_count=0,holdout_reviewed_count=0,status='pending semantic review',note='Downloading and extracting full PDF text is not full-text semantic reading.'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('venue');a.add_argument('--source2024');a.add_argument('--source2025');a.add_argument('--acquire',action='store_true');args=a.parse_args()
 path=BASE/'corpus'/args.venue/'manifest.json'
 if args.source2024:lock(args.venue,[(2024,args.source2024),(2025,args.source2025)])
 if args.acquire:run(path)
