"""Attach archive outcomes and restrict local references to verified assets."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).parent
DAY = ROOT / '08-daily/2026-10-19'
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import paper_config

D = json.loads((DAY / 'search_result.json').read_text())
A = json.loads((DAY / 'archive-result.json').read_text())
V = {p['title']: p for p in json.loads((RUN / 'source-verification.json').read_text())}
tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
raw = json.loads((RUN / 'arxiv-raw.json').read_text()) + json.loads((RUN / 'openreview-raw.json').read_text())
summaries = {p['title']: p.get('summary', '') for p in raw}
physics = ['muon collider', 'muon beam', 'muon anomalous', 'muon magnetic', 'muon-to-electron', 'lepton flavor', 'muography', 'muon detector', 'muon mean lifetime', 'muon decay', 'muon physics', 'hadronic light-by-light', 'muon g-2', 'muon $g-2', 'muon neutrino']
for p in D['all_papers'] + D['top_papers']:
 text = (p['title'] + ' ' + summaries.get(p['title'], '')).lower()
 if any(term in text for term in physics):
  p['editorial_status'] = '词义误命中（不推荐）'
  p['related_papers'] = []
 seen, related = set(), []
 for q in p.get('related_papers', []):
  path = Path(q.get('path', ''))
  if not path.is_file() or q['title'] in seen or path.relative_to(ROOT).as_posix() not in tracked:
   continue
  if path.name == 'search_result.json':
   note = path.parent / '今日检索.md'
   if not note.is_file() or note.relative_to(ROOT).as_posix() not in tracked or q['title'] not in note.read_text():
    continue
  related.append(q)
  seen.add(q['title'])
 p['related_papers'] = related

def relation(title, path, shared, explanation):
 asset = ROOT / path
 assert asset.is_file() and path in tracked
 return {'title': title, 'path': str(asset), 'stem': paper_config.sanitize_paper_title(title), 'source': 'note' if asset.name == '精读.md' else 'pdf', 'shared_terms': shared, 'relationship': explanation}

refs = [
 [relation('PowerStep: Memory-Efficient Adaptive Optimization via lp-Norm Steepest Descent', '03-notes/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent/精读.md', ['norm geometry', 'optimizer design'], '互补：PowerStep关注优化器状态内存，BMuon关注模型通道可压缩性，两类预算和指标需要分开比较。')],
 [relation('Convergence Analysis of STORM Under Different Geometries', '01-raw/2026-10/Convergence_Analysis_of_STORM_Under_Different_Geometries.pdf', ['STORM', 'geometry'], '直接对照：既有不同几何STORM分析与随机化校正的额外预测方差、oracle预算差异，需核验两文具体假设。'), relation('Convergence of Muon with Newton-Schulz', '03-notes/2026-10/Convergence_of_Muon_with_Newton_Schulz/精读.md', ['Muon', 'finite Newton–Schulz'], '边界对照：精确谱LMO保证不自动涵盖有限NS实现，需另加近似误差分析。')],
 [relation('Is Weight Tying Still Beneficial for Decoder-Only LLMs in Private Settings Under DP-SGD', '01-raw/2026-10/Is_Weight_Tying_Still_Beneficial_for_Decoder_Only_LLMs_in_Private_Settings_Under_DP_SGD.pdf', ['DP-SGD', 'LLM'], '互补：参数共享与DP宽度缩放是不同维度，需先核验裁剪和噪声预算再设计联合比较。')]
]
for p, a, related in zip(D['top_papers'][:3], A['archived'], refs):
 assert p['title'] == a['title']
 p.update(archive_status=a['status'], archive_detail=a['detail'], archive_path=a['path'], archive_month=a['month'], source_verified=True)
 p['related_papers'] = related
 p['source_abstract_url'] = V[p['title']]['url']
 p['license_urls_verified'] = V[p['title']]['license_urls']
 for q in D['all_papers']:
  if q['title'] == p['title']:
   q['related_papers'] = related
D['source_notes'] = [{'title': x['title'], 'url': x['url'], 'comments': x['comments'], 'effect': '按配置排除正式推荐，保留原预筛记录'} for x in V.values() if 'workshop' in x.get('comments','').lower()]
for p in D['all_papers'] + D['top_papers']:
 if p['title'] in {x['title'] for x in D['source_notes']}:
  p['editorial_status'] = '配置排除（workshop）'
D['physics_false_positive_count'] = sum(p.get('editorial_status') == '词义误命中（不推荐）' for p in D['all_papers'])
(DAY / 'search_result.json').write_text(json.dumps(D, ensure_ascii=False, indent=2) + '\n')
print('Archive results and verified relationships saved; physics false positives:', D['physics_false_positive_count'])
