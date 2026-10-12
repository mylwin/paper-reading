"""One-shot migration: isolate skill research assets without touching user papers."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).resolve().parent
SHARED = ROOT / '.claude/skills/paper-writing-shared'
TMP = SHARED / 'tmp'
assert subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip() == '6679515'
assert not (SHARED / 'corpus-locations.json').exists(), 'Already migrated; do not replay'
data = json.loads((PLAN / 'candidate-assets.json').read_text())
assets = data['assets']
assert sum(n.startswith('01-raw/') for n in assets) == 453
assert sum(n.startswith('02-markdown/') for n in assets) == 453
assert not any(x['preexisting_committed'] or x['has_user_notes_or_translation']
               for x in assets.values())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(p):
    dest = TMP / 'migration-before' / p.relative_to(ROOT)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


prior = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '-z',
                                    data['baseline_commit']]).decode().split('\0'))
protected = {n: sha(ROOT / n) for n in prior
             if n.startswith(('01-raw/', '02-markdown/', '03-notes/', '06-translation/'))
             and (ROOT / n).is_file() and Path(n).name not in {'README.md', 'index.md'}}
mapping = {}
entries = []
for name, info in sorted(assets.items()):
    p = ROOT / name
    venue = info['matched_samples'][0]['venue']
    kind = 'pdf' if name.startswith('01-raw/') else 'markdown'
    dest = TMP / 'corpus' / venue / kind / p.name
    assert not dest.exists()
    mapping[name] = str(dest.relative_to(ROOT))
    entries.append({'old_path': name, 'new_path': mapping[name], 'sha256': sha(p),
                    'bytes': p.stat().st_size, 'kind': kind, 'venue': venue})

# Actual research papers/supplements and full reading caches outside the main library.
extra = [
    ROOT / '07-research/2026-10-10-顶会顶刊写作风格调研_证据/sources/iclr/iclr-2024-710a9140-supplement.pdf',
    ROOT / '07-research/2026-10-10-顶会顶刊写作风格调研_证据/sources/iclr/iclr-2024-710a9140-supplement.txt',
]
rep = ROOT / '07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai-replacement'
extra += [rep / n for n in ['snn-slimming-accepted-author.pdf',
                           'ss-trboosting-arxiv-v1.pdf',
                           'snn-slimming-accepted-author.txt', 'ss-trboosting-arxiv-v1.txt']]
aaai = ROOT / '07-research/2026-10-12-顶会顶刊写作风格调研_证据/aaai'
extra += list(aaai.glob('supplement*text*.txt'))
for d in ROOT.glob('.planning/venue*'):
    for p in d.rglob('*.txt'):
        if ('-raw.txt' in p.name or '-full.txt' in p.name
                or p.name.startswith('supplement-')
                or re.match(r'tai-\d{4}-\d{2}\.txt$', p.name)):
            extra.append(p)
for p in sorted(set(extra)):
    if not p.is_file():
        continue
    name = str(p.relative_to(ROOT))
    dest = TMP / 'source-materials' / name
    assert not dest.exists()
    mapping[name] = str(dest.relative_to(ROOT))
    entries.append({'old_path': name, 'new_path': mapping[name], 'sha256': sha(p),
                    'bytes': p.stat().st_size, 'kind': 'research_source'})
for x in entries:
    p, q = ROOT / x['old_path'], ROOT / x['new_path']
    assert sha(p) == x['sha256']
    q.parent.mkdir(parents=True, exist_ok=True)
    p.rename(q)
    assert sha(q) == x['sha256']

# Remove only research-owned rows before refreshing user indexes.
stems = {Path(n).stem for n in assets}
index_changes = []
for directory in ['01-raw', '02-markdown', '03-notes', '06-translation']:
    for p in [ROOT / directory / 'index.md', ROOT / directory / '2026-10/README.md']:
        original = p.read_text()
        lines = original.splitlines(keepends=True)
        kept = [line for line in lines if not (line.startswith('|')
                and any(c.strip() in stems for c in line.split('|')))]
        assert len(lines) - len(kept) == 453, (p, len(lines) - len(kept))
        snapshot(p)
        p.write_text(''.join(kept))
        index_changes.append({'path': str(p.relative_to(ROOT)), 'rows_removed': 453})

ordered = sorted(mapping, key=len, reverse=True)


def replace_text(s):
    for old in ordered:
        s = s.replace(old, mapping[old])
    return s


def convert(obj, parent):
    if isinstance(obj, str):
        s = replace_text(obj)
        if s == obj and '\n' not in obj:
            rel = (parent / obj).resolve()
            try:
                old = str(rel.relative_to(ROOT))
            except ValueError:
                return obj
            if old in mapping:
                s = os.path.relpath(ROOT / mapping[old], parent)
        return s
    if isinstance(obj, list):
        return [convert(x, parent) for x in obj]
    if isinstance(obj, dict):
        return {k: convert(v, parent) for k, v in obj.items()}
    return obj


# Original lock/pre-migration snapshots remain immutable, with location-map resolution.
historical = lambda p: any(x in p.name for x in [
    'locked-before-', 'records.before-', 'baseline', 'progress-before'])
roots = [SHARED / 'venues', *ROOT.glob('07-research/*顶会顶刊*'),
         *ROOT.glob('.planning/venue*')]
changed = []
seen = set()
for root in roots:
    files = [root] if root.is_file() else list(root.rglob('*'))
    for p in files:
        if p in seen or not p.is_file() or TMP in p.parents:
            continue
        seen.add(p)
        if historical(p) or p.name in {'archive-sha256.json', 'sha256-manifest.json'}:
            continue
        if p.suffix not in {'.json', '.jsonl', '.md', '.py'}:
            continue
        original = p.read_text()
        if p.suffix in {'.json', '.jsonl'}:
            try:
                obj = [json.loads(s) for s in original.splitlines()] if p.suffix == '.jsonl' else json.loads(original)
            except ValueError:
                continue
            new = convert(obj, p.parent)
            if new == obj:
                continue
            output = (''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in new)
                      if p.suffix == '.jsonl' else json.dumps(new, ensure_ascii=False, indent=2) + '\n')
        elif p.suffix == '.md':
            def link(m):
                target = m[1].strip().strip('<>')
                if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target):
                    return m[0]
                path = unquote(target.split('#')[0])
                try:
                    old = str((p.parent / path).resolve().relative_to(ROOT))
                except ValueError:
                    return m[0]
                if old not in mapping:
                    return m[0]
                return '](' + os.path.relpath(ROOT / mapping[old], p.parent) + ')'
            output = re.sub(r'\]\(([^\n)]+)\)', link, replace_text(original))
        else:
            output = replace_text(original)
        if output != original:
            snapshot(p)
            p.write_text(output)
            changed.append(str(p.relative_to(ROOT)))

location = {'date': '2026-10-12', 'local_only': True,
            'root': '.claude/skills/paper-writing-shared/tmp',
            'purpose': 'Skill research corpus; never knowledge-base registration or Git publication',
            'baseline_commit': data['baseline_commit'], 'assets': entries,
            'original_snapshots': 'tmp/migration-before/',
            'qualification': 'Path relocation preserves paper bytes and study conclusions; historical locks unchanged.'}
(SHARED / 'corpus-locations.json').write_text(json.dumps(location, ensure_ascii=False, indent=2) + '\n')
(PLAN / 'migration-result.json').write_text(json.dumps({
    'assets_moved': len(entries), 'main_pdfs': 453, 'parsed_texts': 453,
    'updated_references': changed, 'index_changes': index_changes,
    'protected_knowledge_assets': protected,
    'all_moved_hashes_match': all(sha(ROOT / x['new_path']) == x['sha256'] for x in entries)
}, ensure_ascii=False, indent=2) + '\n')
assert all(sha(ROOT / n) == h for n, h in protected.items())
print(json.dumps({'moved': len(entries), 'main_pdfs': 453, 'parsed_texts': 453,
                  'references_updated': len(changed),
                  'original_user_assets_unchanged': len(protected)}, ensure_ascii=False))
