"""Read-only integrity checks; does not certify semantic reading or reproduce results."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import subprocess
import zipfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '07-research/2026-10-12-顶会顶刊写作风格调研_证据'
CORPUS = ROOT / '07-research/2026-10-10-顶会顶刊写作风格调研_证据/corpus'
EVAL = ROOT / '07-research/2026-10-11-论文写作润色技能评测_证据'
checks = []


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(name, ok, detail=None):
    checks.append({'check': name, 'passed': bool(ok), 'detail': detail})


before = read(OUT / 'progress-before.json')
after = read(OUT / 'progress-after.json')
ledger = read(OUT / 'batch-ledger.json')
check('canonical progress matches archived after snapshot',
      after == read(CORPUS.parent / 'progress.json'))
expected = {'locked': 550, 'pdf_acquired': 429, 'text_parsed': 429,
            'main_text_style_reviewed': 28,
            'full_text_and_relevant_appendix_reviewed': 23,
            'induction_complete': 19, 'holdout_complete': 4}
check('cumulative counts and remaining gap', after['totals'] == expected
      and after['remaining_full_review'] == 527)
check('venue totals sum to cumulative counts', all(
    sum(v[k] for v in after['venues']) == count for k, count in expected.items()))
check('batch progression equals declared new work', all(
    after['totals'][k] - before['totals'][k] == delta for k, delta in {
        'main_text_style_reviewed': 7,
        'full_text_and_relevant_appendix_reviewed': 5,
        'pdf_acquired': 1, 'text_parsed': 1, 'induction_complete': 5,
        'holdout_complete': 0}.items()))
check('seven main, five full, no holdout and one new asset',
      len(ledger['papers']) == 7
      and sum(p['main_complete'] for p in ledger['papers']) == 7
      and sum(p['full_and_appendix_complete'] for p in ledger['papers']) == 5
      and all(p['split'] != 'holdout' for p in ledger['papers'])
      and sum(p.get('new_pdf_acquisition', False) for p in ledger['papers']) == 1)
pages_total = 0
for item in ledger['papers']:
    review = read(ROOT / item['review_path'])
    pdf = ROOT / item['pdf_path']
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    pages_total += pages
    cov = review.get('coverage_pdf_pages')
    if cov is None:
        coverage = review['coverage']
        cov = coverage.get('pdf_pages') if isinstance(coverage, dict) else coverage[0]['pages']
    record = next(x for x in map(json.loads,
                  (CORPUS / item['venue'] / 'records.jsonl').read_text().splitlines())
                  if x['id'] == item['id'])
    check(item['id'] + ': PDF identity and full page coverage',
          sha(pdf) == item['pdf_sha256'] == review['pdf_sha256']
          and cov == list(range(1, pages + 1)), {'pages': pages})
    dims = review['ten_dimensions']
    check(item['id'] + ': ten nonempty dimensions and version/supplement scope',
          len(dims) == 10 and all(dims.values())
          and bool(review['supplement_audit'])
          and bool(review['publication_identity_and_acquired_version']))
    check(item['id'] + ': recorded full status matches ledger',
          bool(record.get('full_text_style_review_complete'))
          and bool(record.get('external_appendix_review_complete'))
          == item['full_and_appendix_complete'])
check('actual PDF page count and separate supplement accounting',
      pages_total == ledger['pdf_pages_continuously_read'] == 117
      and ledger['additional_author_supplement_pages_read'] == 7
      and ledger['institution_covers_in_pdf_page_count'] == 3)

active_ids = []
for p in sorted(CORPUS.glob('*/manifest.json')):
    d = read(p)
    rows = d.get('records', d.get('papers'))
    active_ids.extend(x['id'] for x in rows)
    counts = Counter((int(x['year']), 'holdout' if x['split'] == 'holdout'
                      else 'induction') for x in rows)
    check(p.parent.name + ': fifty positions with fixed yearly split',
          len(rows) == 50 and len({x['id'] for x in rows}) == 50
          and counts == Counter({(2024, 'induction'): 20,
                                (2025, 'induction'): 20,
                                (2024, 'holdout'): 5,
                                (2025, 'holdout'): 5}))
check('550 unique active identities across eleven venues',
      len(active_ids) == len(set(active_ids)) == 550)
locks = read(OUT / 'lock-integrity-baseline.json')
check('AAAI and NeurIPS locked manifests unchanged', all(
    sha(CORPUS / v / 'manifest.json') == locks[v] for v in ['aaai', 'neurips']))
amend = read(CORPUS / 'tai/selection-amendments.json')[0]
check('TAI original manifest and record bytes preserved',
      sha(ROOT / amend['original_manifest_backup']) == locks['tai']
      == amend['original_manifest_sha256']
      and sha(ROOT / amend['original_records_backup']) == amend['original_records_sha256'])
old = read(ROOT / amend['original_manifest_backup'])['papers']
new = read(CORPUS / 'tai/manifest.json')['papers']
old_map = {x['id']: x for x in old}
new_map = {x['id']: x for x in new}
check('TAI eligibility repair changes exactly one induction slot',
      set(old_map) - set(new_map) == {'tai-2024-03'}
      and set(new_map) - set(old_map) == {'tai-2024-03-r1'}
      and all(old_map[k] == new_map[k] for k in set(old_map) & set(new_map))
      and new_map['tai-2024-03-r1']['year'] == old_map['tai-2024-03']['year']
      and new_map['tai-2024-03-r1']['split'] == old_map['tai-2024-03']['split'])
check('TAI all ten holdout identities and locked records preserved',
      [x for x in old if x['split'] == 'holdout']
      == [x for x in new if x['split'] == 'holdout'])
check('post-pilot repair not represented as original immutable pre-analysis lock',
      read(CORPUS / 'tai/manifest.json')['locked_before_analysis'] is False
      and read(CORPUS / 'tai/manifest.json')['original_lock_before_analysis'] is True)

files = read(OUT / 'aaai/sha256-manifest.json')['files']
check('all AAAI delegated source evidence hashes unchanged',
      all(sha(OUT / 'aaai' / x['path']) == x['sha256'] for x in files),
      {'files': len(files)})
images = list(OUT.rglob('*.png'))
check('29 saved original-page images are real PNG files', len(images) == 29
      and all(p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for p in images))
zip_path = ROOT / ('07-research/2026-10-11-顶会顶刊写作风格全文续研_证据/'
                   'next-batch/neurips-2024-2aada42a-complete.zip')
with zipfile.ZipFile(zip_path) as z:
    crc = z.testzip()
check('repaired official NeurIPS supplement ZIP complete and CRC valid',
      crc is None and sha(zip_path)
      == '9655b463e5dd68ca2219ca65c07e15d531ef27e287e6681dab93e740ad48db2e')
archive = read(EVAL / 'archive-sha256.json')
check('all 654 historical evaluation evidence hashes unchanged',
      len(archive) == 654 and all((EVAL / p).is_file() and sha(EVAL / p) == s
                                for p, s in archive.items()))
baseline = read(EVAL / 'final-skills-sha256.json')
changed = sorted(p for p, s in baseline.items() if not (ROOT / p).is_file()
                 or sha(ROOT / p) != s)
allowed = sorted('.claude/skills/paper-writing-shared/venues/' + n + '.md'
                 for n in ['aaai', 'neurips', 'tai', 'tmlr', 'overview'])
check('template packs, project scripts and entry skills unchanged since evaluation',
      changed == allowed, {'changed': changed})
(OUT / 'skill-resource-delta.json').write_text(json.dumps(
    {'compared_to': str(EVAL.relative_to(ROOT) / 'final-skills-sha256.json'),
     'changed': changed, 'new_compilation_or_behavior_evaluation': False},
    ensure_ascii=False, indent=2) + '\n')

restore = read(ROOT / '.planning/paper-skills-preservation-2026-10-12/restoration.json')
check('21 restored old skill files match Git index recovery hashes',
      len(restore['restored']) == 21 and all(sha(ROOT / x['path']) == x['sha256']
                                         for x in restore['restored']))
entries = ['paper-analyze', 'paper-conf', 'paper-daily', 'paper-interests',
           'paper-weekly', 'paper-sgd-reading', 'paper-writing', 'paper-polishing']
check('six existing and two new skills discoverable through existing entry', all(
    (ROOT / '.agents/skills' / x / 'SKILL.md').is_file() for x in entries)
      and not (ROOT / '.claude/skills/paper-writing-shared/SKILL.md').exists())
new_asset = next(x for x in ledger['papers'] if x.get('new_pdf_acquisition'))
stem = Path(new_asset['pdf_path']).stem
check('new PDF and parse share stable stem and actual files',
      (ROOT / f'02-markdown/2026-10/{stem}.md').is_file())
registry = read(ROOT / '.planning/venue-writing-next-research-2026-10-12/'
                'workspace-registry-restored-skills.json')['registry']
new_rows = [x for x in registry if stem in str(x)]
check('new asset indexed in restored read-only index workflow', len(new_rows) == 1,
      {'row': new_rows})

# Provide real destinations before link scan; final archive manifest is written later.
(OUT / 'archive-sha256.json').write_text('{}\n')
(OUT / 'validation.json').write_text('{}\n')
cmd = ['python3', str(ROOT / '.claude/skills/paper-daily/scripts/sync_indexes.py'),
       '--workspace', str(ROOT), '--config', str(ROOT / '.claude/skills/config.yaml'),
       '--check-links', '--json']
import os
env = dict(os.environ, PYTHONPATH='/tmp/paper-writing-validation-deps')
links = subprocess.run(cmd, env=env, text=True, capture_output=True)
(OUT / 'workspace-link-check.json').write_text(links.stdout)
link_result = json.loads(links.stdout)
check('workspace link scanner returns structured result', isinstance(link_result, dict),
      {'exit_code': links.returncode,
       'scope': 'Existing unrelated broken links are reported separately, not a global pass.'})
scan = [ROOT / '07-research/2026-10-12-顶会顶刊写作风格调研.md']
scan.extend(ROOT / p for p in allowed)
broken = []
for p in scan:
    for dest in re.findall(r'\]\(([^\n)]+)\)', p.read_text()):
        dest = dest.strip().strip('<>')
        if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', dest) or dest.startswith('#'):
            continue
        target = unquote(dest.split('#', 1)[0])
        if not (p.parent / target).exists():
            broken.append({'file': str(p.relative_to(ROOT)), 'target': dest})
check('updated research report and five profiles have no broken local links',
      not broken, {'broken': broken})
result = {'date': '2026-10-12', 'scope': 'Evidence integrity and preservation; '
          'not independent semantic reading certification, compilation or behavior evaluation',
          'passed': sum(x['passed'] for x in checks), 'total': len(checks),
          'all_passed': all(x['passed'] for x in checks), 'checks': checks}
(OUT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passed': result['passed'], 'total': result['total'],
                  'failed': [x for x in checks if not x['passed']]}, ensure_ascii=False, indent=2))
raise SystemExit(0 if result['all_passed'] else 1)
