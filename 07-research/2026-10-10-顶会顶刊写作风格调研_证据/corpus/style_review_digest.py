#!/usr/bin/env python3
"""A navigation aid, not an automated semantic evaluator or fulltext-reading certificate."""
import json,re,sys
from pathlib import Path
base=Path(__file__).resolve().parent
root=base.parents[2]
rows=[json.loads(x) for x in (base/sys.argv[1]/'records.jsonl').read_text().splitlines()]
start,end=map(int,sys.argv[2:4])
for r in rows[start:end]:
    if not r.get('markdown_path'):continue
    t=(root/r['markdown_path']).read_text();pages=re.split(r'## PDF page \d+\n\n```text\n',t)[1:]
    print('\n###',r['id'],r['title'],'split',r['split'],'PAGES',len(pages))
    abstract=pages[0].find('Abstract')
    print('OPENING',pages[0][abstract:abstract+1600] if abstract>=0 else pages[0][:1100])
    headings=[]
    refpage=next((i for i,s in enumerate(pages,1) if re.search(r'^References\s*$',s,re.M)),len(pages)+1)
    for p,s in enumerate(pages,1):
        for line in s.splitlines():
            x=line.strip()
            if re.match(r'^(?:[1-9](?:\.\d+)*\.?\s+[A-Za-z]{3}|Appendix [A-Z]\.|[A-Z]\.\d+\s+[A-Z]|[A-Z]\.\s+[A-Z][a-z]{2})',x) and len(x)<100 and p<refpage and not re.search(r'[=∈≤≥{},]|Appendix.* (?:is|are|to|of) ',x):
                pos=s.find(line);headings.append((p,x,s[pos+len(line):pos+len(line)+150].replace('\n',' ')))
    print('CHAPTER NAVIGATION')
    for p,h,s in headings:print(p,h,'|',s[:120])
    print('PAGE NAVIGATION (every page; extracts do not substitute for complete prose)')
    for p,s in enumerate(pages,1):
        if p>=refpage:
            if p==refpage:print(p,'Bibliography starts; remaining reference pages:',len(pages)-p+1)
            continue
        ls=[x.strip() for x in s.splitlines() if len(x.strip())>45]
        pick=next((x for x in ls[1:] if not re.search('doi:|arxiv|@|Journal of|Creative Commons',x,re.I)),s[:90])
        print(p,pick[:90])
    print('EVIDENCE / QUALIFICATION PASSAGES')
    alltext='\n'.join(pages)
    out=[]
    for m in re.finditer(r'(?i)(?:we (?:show|prove|demonstrate|find|evaluate)|under (?:the|these|mild)|limitation|future work|in conclusion|in summary|numerical experiments|experimental results|conclusion)',alltext):
        para=alltext[max(0,m.start()-80):m.start()+280].replace('\n',' ')
        if para not in out:out.append(para)
    for x in out[:5]:print(x)
