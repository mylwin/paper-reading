#!/usr/bin/env python3
"""Crossref-verified IEEE candidate harvesting; public version status remains explicit."""
import concurrent.futures,hashlib,json,re,urllib.request
from pathlib import Path
import journal_acquire as ja

def crossref(x):
    doi=x['doi'].replace('https://doi.org/','')
    try:
        req=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'paper-writing-style-research/1.0'})
        with urllib.request.urlopen(req,timeout=25) as f:j=json.load(f)['message']
        year=j.get('published-print',j.get('published',{})).get('date-parts',[[None]])[0][0]
        if year not in [2024,2025]:return None
        locs=[p for p in x['locations'] if p.get('pdf_url')]
        locs.sort(key=lambda p:({'acceptedVersion':0,'publishedVersion':1,'submittedVersion':2}.get(p.get('version'),3), 'ieeexplore' in p.get('pdf_url','')))
        if not locs:return None
        p=locs[0]
        return dict(title=j['title'][0],year=year,publication_url=j['URL'],pdf_url=p['pdf_url'],authors=', '.join((a.get('given','')+' '+a.get('family','')).strip() for a in j.get('author',[])),doi=doi,version=p['version'],version_metadata_source='OpenAlex locator, not independent content/version verification',publication_verified='Crossref publisher DOI deposit',crossref=j,openalex_id=x['id'],provisional_stratum=ja.group(j['title'][0]),semantic_review_status='not_reviewed',eligibility='pending_content_version_confirmation' if p['version']=='submittedVersion' else 'metadata_indicates_accepted_or_published')
    except Exception:return None

def main(v):
    files={'tpami':['/tmp/tpami-openalex-1.json','/tmp/tpami-openalex-more.json'],'tnnls':['/tmp/tnnls-openalex.json','/tmp/tnnls-openalex-2.json','/tmp/tnnls-openalex-expanded.json'],'tai':['/tmp/tai-openalex.json','/tmp/tai-openalex-expanded.json','/tmp/tai-openalex-page2.json']}[v]
    candidates={x['id']:x for f in files for x in json.loads(Path(f).read_text())['results']}
    cand=[x for x in candidates.values() if x.get('doi') and any(p.get('pdf_url') for p in x['locations']) and not re.search(r'survey|review|editorial|correction|erratum|guest|bibliometric',x['title'],re.I)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:verified=[r for r in ex.map(crossref,cand) if r]
    selected=[]
    for y in [2024,2025]:
        pools={g:sorted([r for r in verified if r['year']==y and r['provisional_stratum']==g],key=lambda r:(r['version']=='submittedVersion',hashlib.sha256(r['doi'].encode()).hexdigest())) for g in sorted(set(r['provisional_stratum'] for r in verified))}
        sample=[];seen=set()
        while len(sample)<25 and any(pools.values()):
            for g,pool in pools.items():
                while pool:
                    r=pool.pop(0);a=r['authors'].split(',')[0]
                    if a not in seen:seen.add(a);sample.append(r);break
                if len(sample)==25:break
        for i,r in enumerate(sample):r.update(id=f'{v}-{y}-{i+1:02}',split='derivation' if i<20 else 'holdout',selection_locked='2026-10-10',type_status='title_based_provisional')
        selected.extend(sample)
    d=ja.BASE/v;d.mkdir(exist_ok=True)
    (d/'manifest.json').write_text(json.dumps(dict(venue=v,target=50,papers=[{k:z for k,z in r.items() if k!='crossref'} for r in selected],locked_before_analysis=True,selection_method='Crossref publisher metadata, print-year preferred; public PDF locator; deterministic strata round robin, unique first author, accepted/published preferred inside each stratum; preprint content version remains pending.',formal_fulltext_target_fulfilled=False),ensure_ascii=False,indent=2))
    sd=ja.BASE.parent/'sources'/v;sd.mkdir(exist_ok=True)
    (sd/'crossref-selected.json').write_text(json.dumps({r['doi']:r['crossref'] for r in selected},ensure_ascii=False,indent=2))
    for f in files:(sd/Path(f).name).write_bytes(Path(f).read_bytes())
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        for r in ex.map(ja.acquire,selected):
            r.pop('crossref',None);rows.append(r)
    (d/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (d/'validation.json').write_text(json.dumps(dict(venue=v,locked=len(rows),downloaded=sum(r.get('download_ok',False) for r in rows),parsed=sum('markdown_path' in r for r in rows),metadata_accepted_or_published=sum(r['version']!='submittedVersion' for r in rows),semantic_fulltext_reviewed=0,holdout_semantic_validation='pending',official_fulltext_and_version_content_confirmation='pending',automated_extraction_is_not_fulltext_review=True),indent=2))
    print(v,len(rows),'locked',sum(r.get('download_ok',False) for r in rows),'downloaded; semantic review pending')
if __name__=='__main__':
    import sys
    main(sys.argv[1])
