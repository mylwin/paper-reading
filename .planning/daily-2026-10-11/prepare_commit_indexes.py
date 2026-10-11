"""Generate task-only index blobs using the authoritative sync writer.

Other concurrent task assets stay in the working tree. The temporary snapshot
contains HEAD's tracked paper assets and the two PDFs archived by this run.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import paper_config
import sync_indexes

STAGE = Path(tempfile.mkdtemp(prefix='paper-daily-20261011-indexes-', dir='/private/tmp'))
roots = ['01-raw', '02-markdown', '03-notes', '06-translation']
paths = subprocess.check_output(['git', 'ls-files', '-z', '--', *roots], cwd=ROOT).decode().split('\0')
for rel in filter(None, paths):
    src, dest = ROOT / rel, STAGE / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Generated indexes/READMEs must start from HEAD, not concurrent edits.
    if src.name in ('README.md', 'index.md'):
        dest.write_bytes(subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT))
    elif src.exists():
        os.link(src, dest)

data = json.loads((ROOT / '08-daily/2026-10-11/search_result.json').read_text())
overrides = {}
for paper in data['top_papers'][:3]:
    if paper['archive_status'] != 'downloaded':
        continue
    src = Path(paper['archive_path'])
    dest = STAGE / src.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dest)
    overrides[paper['paper_stem']] = {
        'date': '2026-10-11', 'month': '2026-10',
        'title': paper['title'], 'source': paper['source']
    }
config = paper_config.load_config()
# Preserve tracked assets' Git dates while the snapshot has no repository.
original_git_range = sync_indexes._git_commit_range
def tracked_git_range(path, workspace):
    return original_git_range(ROOT / path.relative_to(STAGE), ROOT)
sync_indexes._git_commit_range = tracked_git_range
report = sync_indexes.refresh(STAGE, config, overrides=overrides)
output = []
for directory in roots:
    for name in ['index.md', '2026-10/README.md']:
        rel = directory + '/' + name
        content = (STAGE / rel).read_bytes()
        old = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
        output.append({'path': rel, 'snapshot': str(STAGE / rel),
                       'changed': content != old})
manifest = {'snapshot_root': str(STAGE), 'indexes': output}
(ROOT / '.planning/daily-2026-10-11/commit-index-manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'snapshot': str(STAGE), 'paper_count': report.get('total'),
                  'changed_indexes': sum(x['changed'] for x in output)}, ensure_ascii=False))
