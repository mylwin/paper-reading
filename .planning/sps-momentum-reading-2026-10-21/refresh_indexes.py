from pathlib import Path
import sys
import json

root = Path('/Users/myl/Documents/paper')
sys.path.insert(0, str(root / '.claude/skills/paper-daily/scripts'))
import sync_indexes as sync

# Skip a README header that the current archive parser can mistake for a paper.
# Leave the shared script and all existing paper metadata unchanged.
original_read = sync.read_archive_dates
def archive_without_header(workspace, config):
    return {key:value for key,value in original_read(workspace, config).items()
            if sync.clean_stem(key) != 'paperstem'}
sync.read_archive_dates = archive_without_header

stem = 'ICLR-2025-stochastic-polyak-step-sizes-and-momentum-convergence-guarantees-and-practical-performance-Paper-Conference'
overrides = {stem: {
    'date':'2026-10-21', 'month':'2026-10',
    'title':'Stochastic Polyak Step-sizes and Momentum: Convergence Guarantees and Practical Performance',
    'source':'用户提供；ICLR 2025',
}}
config = sync.load_config()
out = root / '.planning/sps-momentum-reading-2026-10-21'
paths = []
for stage in ['01-raw','02-markdown','03-notes','06-translation']:
    paths += [root/stage/'index.md', root/stage/'2026-10/README.md']
backup_path = out/'indexes-before.json'
if not backup_path.exists():
    backup_path.write_text(json.dumps({str(p.relative_to(root)):p.read_text() if p.exists() else None for p in paths}, ensure_ascii=False, indent=2)+'\n')
dry = '--write' not in sys.argv
result = sync.refresh(root, config, dry_run=dry, overrides=overrides)
selected = next(x for x in result['registry'] if x['paper_stem']==stem)
summary = {k:v for k,v in result.items() if k!='registry'}
summary['selected'] = selected
(out/('indexes-dry-run.json' if dry else 'indexes-refresh.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
