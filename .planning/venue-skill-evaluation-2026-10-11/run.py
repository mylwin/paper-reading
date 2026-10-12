"""Retain real CLI integration outputs. Each phase writes a new result file."""
from pathlib import Path
import subprocess
import sys
import hashlib
import json
import shutil
import os
from datetime import datetime, timezone

W=Path(__file__).resolve().parents[2]
E=W/'07-research/2026-10-11-论文写作润色技能评测_证据'
S=W/'.claude/skills/paper-writing-shared/scripts/project.py'
phase=sys.argv[1]
rows=[]
def call(case,args):
    proc=subprocess.run([sys.executable,str(S),*map(str,args)],capture_output=True,text=True)
    rows.append({'case':case,'command':proc.args,'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'time':datetime.now(timezone.utc).isoformat()})
    return proc
def tree(p):
    return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and 'build' not in f.relative_to(p).parts}
if phase=='baseline':
    for name,main in [('E02-icml-writing','main.tex'),('E03-original','article.tex'),('E03-revised','article.tex'),('E04-tai-migrated','main.tex')]:
        p=E/'outputs'/name
        call(name,['compile','--project',p,'--main',main])
        call(name+'-check',['check','--project',p])
    (E/'source-tree-sha256.json').write_text(json.dumps({name:tree(E/'outputs'/name) for name in ['E03-original','E03-revised']},indent=2)+'\n')
    call('E03-compare',['compare','--before',E/'outputs/E03-original','--after',E/'outputs/E03-revised','--report',E/'runs/E03-compare.json'])
    call('E04-compare',['compare','--before',E/'outputs/E03-revised','--after',E/'outputs/E04-tai-migrated','--report',E/'runs/E04-compare.json'])
    for case,extra in [('unknown-venue',['--venue','unsupported']),('unknown-year',['--venue','icml','--version','2099'])]:
        target=E/'outputs'/case
        call('E06-'+case,['init',*extra,'--output',target])
        rows[-1]['target_created']=target.exists()
    target=E/'outputs/E03-original';before=tree(target)
    call('E06-existing',['init','--venue','icml','--output',target]);rows[-1]['original_unchanged']=before==tree(target)
    registry=json.loads((S.parent.parent/'registry.json').read_text())
    for venue in registry['venues']:
        for mode in ['submission','final']:
            p=E/'outputs/template-matrix'/f'{venue}-{mode}'
            call('E07-'+venue+'-'+mode,['init','--venue',venue,'--mode',mode,'--output',p])
            m=json.loads((p/'project.json').read_text())
            rows[-1]['official_hashes_match']=all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in m['official_files'].items())
            rows[-1]['stage_anonymous']=m['constraints']['anonymous']
            rows[-1]['unverified_stage_fields']=[k for k in ['main_page_limit','anonymous'] if m['constraints'].get(k) is None]
    call('E08-default-entrypoint',['compile','--project',E/'outputs/E03-revised'])
    probe=E/'outputs/E08-stale-pdf';shutil.copytree(E/'outputs/E03-revised',probe)
    with (probe/'sections/body.tex').open('a') as f:f.write('\nThis added text has never been compiled.\n')
    call('E08-stale-pdf',['check','--project',probe])
    probe=E/'outputs/E08-bib-drift';shutil.copytree(E/'outputs/E03-revised',probe)
    b=probe/'references.bib';b.write_text(b.read_text().replace('ICML 2026 Author Instructions','Unverified Replacement Reference'))
    call('E08-bib-drift',['compare','--before',E/'outputs/E03-revised','--after',probe,'--report',E/'runs/E08-bib-before.json'])
    probe=E/'outputs/E08-author-leak';shutil.copytree(E/'outputs/E04-tai-migrated',probe)
    t=probe/'main.tex';t.write_text(t.read_text().replace(r'\author{Anonymous submission}',r'\author{Ada Fixture, Example Institute, ada@example.invalid}'))
    call('E08-author-leak-compile',['compile','--project',probe]);call('E08-author-leak-check',['check','--project',probe])
    probe=E/'outputs/E08-macro-title';shutil.copytree(E/'outputs/E04-tai-migrated',probe)
    t=probe/'main.tex';t.write_text(t.read_text().replace(r'\title{A Fixed Confidence Gate for Selective Classification}',r'\newcommand{\LongTitle}{A Fixed Confidence Gate for Selective Classification With a Very Long Title That Exceeds the Stated Word Limit}'+'\n'+r'\title{\LongTitle}'))
    call('E08-macro-title-compile',['compile','--project',probe]);call('E08-macro-title-check',['check','--project',probe])
elif phase=='after-fix':
    call('E08-default-entrypoint',['compile','--project',E/'outputs/E03-revised'])
    call('E08-stale-pdf',['check','--project',E/'outputs/E08-stale-pdf'])
    call('E08-bib-drift',['compare','--before',E/'outputs/E03-revised','--after',E/'outputs/E08-bib-drift','--report',E/'runs/E08-bib-after.json'])
    call('E08-author-leak-check',['check','--project',E/'outputs/E08-author-leak'])
    call('E08-macro-title-check',['check','--project',E/'outputs/E08-macro-title'])
    for name in ['E02-icml-writing','E03-original','E03-revised','E04-tai-migrated']:
        p=E/'outputs'/name
        call(name+'-refresh-compile',['compile','--project',p])
        call(name+'-refresh-check',['check','--project',p])
    # A distinct regression: new entrypoint, cited database included from a subdirectory,
    # and a referenced asset change rather than a textual change.
    probe=E/'outputs/E09-heldout-entry';shutil.copytree(E/'outputs/E03-revised',probe)
    (probe/'article.tex').rename(probe/'camera.tex')
    m=json.loads((probe/'project.json').read_text());m['entrypoint']='camera.tex';(probe/'project.json').write_text(json.dumps(m,indent=2))
    call('E09-custom-entry',['compile','--project',probe])
    original_times=(probe/'references.bib').stat()
    (probe/'references.bib').write_text((probe/'references.bib').read_text().replace('2026','2027'))
    os.utime(probe/'references.bib',ns=(original_times.st_atime_ns,original_times.st_mtime_ns))
    call('E09-bib-stale',['check','--project',probe])
    call('E09-bib-compare',['compare','--before',E/'outputs/E03-revised','--after',probe,'--report',E/'runs/E09-bib-compare.json'])
    call('E04-compare-after',['compare','--before',E/'outputs/E03-revised','--after',E/'outputs/E04-tai-migrated','--report',E/'runs/E04-compare-after.json'])
else: raise SystemExit('Unknown phase')
out=E/'runs'/f'{phase}.json'
if out.exists():raise SystemExit('Refusing overwrite of retained run')
out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'case':r['case'],'exit_code':r['exit_code']} for r in rows],indent=2))
