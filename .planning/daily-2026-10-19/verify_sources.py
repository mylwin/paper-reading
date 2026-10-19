import concurrent.futures,html,json,re,urllib.request
from pathlib import Path
R=Path(__file__).parent;D=json.loads(Path('08-daily/2026-10-19/search_result.json').read_text())
P=D['top_papers']
def check(p):
 url=p['url'].replace('http:','https:')
 with urllib.request.urlopen(url,timeout=45) as f: raw=f.read().decode();status=f.status
 meta={}
 for tag in re.findall(r'<meta\b[^>]*>',raw,re.I):
  attrs=dict(re.findall(r'([\w-]+)=[\"\x27](.*?)[\"\x27]',tag,re.S))
  if attrs.get('name','').startswith('citation_'):meta.setdefault(attrs['name'],[]).append(html.unescape(attrs.get('content','')))
 license_urls=sorted(set(re.findall(r'https?://creativecommons.org/[^\s\"<>]+',raw)))
 abstract=re.search(r'<blockquote[^>]*class="abstract[^\"]*"[^>]*>(.*?)</blockquote>',raw,re.S)
 abstract=html.unescape(re.sub('<[^>]+>','',abstract.group(1))).strip() if abstract else ''
 abstract=re.sub(r'^Abstract:\s*','',abstract)
 comments=re.search(r'<td[^>]*class="[^"]*comments[^"]*"[^>]*>(.*?)</td>',raw,re.S)
 comments=html.unescape(re.sub('<[^>]+>','',comments.group(1))).strip() if comments else ''
 return {'comments':comments,'title':p['title'],'url':url,'status':status,'metadata':meta,'license_urls':license_urls,'abstract':abstract,'source_summary':p['summary']}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:out=list(e.map(check,P))
(R/'source-verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for p in out:print(p['title'],p['license_urls'])
