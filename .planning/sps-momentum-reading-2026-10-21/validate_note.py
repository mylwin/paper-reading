from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

root=Path('/Users/myl/Documents/paper')
work=root/'.planning/sps-momentum-reading-2026-10-21'
stem='ICLR-2025-stochastic-polyak-step-sizes-and-momentum-convergence-guarantees-and-practical-performance-Paper-Conference'
note=root/'03-notes/2026-10'/stem/'精读.md'
text=note.read_text()
errors=[]
parts=text.split('$$')
if len(parts)%2 != 1:
    errors.append('Unbalanced display delimiters')
maths=parts[1::2]
for number, expression in enumerate(maths,1):
    depth=0
    for char in expression:
        if char=='{': depth+=1
        elif char=='}': depth-=1
        if depth<0: errors.append(f'Equation {number}: unmatched closing brace')
    if depth: errors.append(f'Equation {number}: unbalanced braces')
    if expression.count(r'\left') != expression.count(r'\right'):
        errors.append(f'Equation {number}: unbalanced scalable delimiters')
    if re.search(r'\\(?:qquad|quad|tag|big\w*|Big\w*|boxed|underbrace|overbrace|overset|underset)\b|\\[;,!|]',expression):
        errors.append(f'Equation {number}: unsupported command')
    if re.search(r'\\text\{[^}]*[\u4e00-\u9fff]',expression):
        errors.append(f'Equation {number}: Chinese in math text')
outside=''.join(parts[0::2])
if '$' in outside: errors.append('Unexpected single-dollar math')
links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
local=[]
for target in links:
    if re.match(r'https?://',target): continue
    path=(note.parent/unquote(target.split('#')[0])).resolve()
    local.append(str(path))
    if not path.is_file(): errors.append('Missing local link: '+target)
if re.search(r'\[待补充|TODO|PLACEHOLDER|\{\{',text): errors.append('Template placeholder remains')

manifest=json.loads((work/'figure-manifest.json').read_text())
source=root/manifest['source']
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['sha256']
before=json.loads((work/'indexes-before.json').read_text())
index_changes={}
for path,old in before.items():
    current=(root/path).read_text()
    old_rows={line for line in (old or '').splitlines() if line.startswith('|') and 'paper_stem' not in line and not line.startswith('|---')}
    # Existing table rows can gain columns/links for the selected paper only.
    removed=[line for line in old_rows if line not in current.splitlines() and stem not in line]
    if removed: errors.append(f'{path}: pre-existing paper rows changed ({len(removed)})')
    index_changes[path]={'new_paper_present':stem in current, 'existing_rows_preserved':not removed}

# Verify the generic distance inequality and the conservative replacement.
lam=1; a=1; b=-1
left=((1+lam)*a-lam*b)**2
claimed=(1+lam)*a*a+lam*b*b
safe=(1+2*lam)**2*max(a*a,b*b)
assert left==9 and claimed==3 and safe==9
report={'note':str(note), 'characters':len(text), 'display_equations':len(maths), 'local_links_checked':len(local), 'original_figures':6, 'proof_excerpt':1, 'pdf_sha256_unchanged':True, 'equation26_check':{'left':left,'claimed_right':claimed,'safe_bound':safe}, 'indexes':index_changes, 'errors':errors}
(work/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
