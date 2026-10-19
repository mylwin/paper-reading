import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent;DAY=ROOT/'08-daily/2026-10-17'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip()
M=json.loads((RUN/'commit-index-manifest.json').read_text())
for x in M['indexes']:
 if not x['changed']:continue
 blob=subprocess.check_output(['git','hash-object','-w',x['snapshot']],cwd=ROOT).decode().strip()
 subprocess.run(['git','update-index','--add','--cacheinfo','100644',blob,x['path']],cwd=ROOT,check=True)
A=json.loads((DAY/'archive-result.json').read_text());files=[str(p.relative_to(ROOT)) for p in DAY.iterdir() if p.is_file()]+[str(Path(p['path']).relative_to(ROOT)) for p in A['archived'] if p['status']=='downloaded']
# Keep provenance sufficient to repeat validation and explain source coverage.
files += [str(p.relative_to(ROOT)) for p in RUN.iterdir() if p.is_file() and p.name not in ['commit-index-manifest.json','initial-files.json','stage.py']]
subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
