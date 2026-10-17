"""Attach archive outcomes and restrict local references to verified assets."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).parent
DAY = ROOT / '08-daily/2026-10-17'
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
 [relation('PowerStep: Memory-Efficient Adaptive Optimization via lp-Norm Steepest Descent', '03-notes/2026-09/PowerStep_Memory_Efficient_Adaptive_Optimization_via_lp_Norm_Steepest_Descent/精读.md', ['norm geometry', 'sign updates'], '互补：以已有Lp更新为候选几何，对照本文符号/谱路由；不能直接把二元判据推广到任意p。')],
 [relation('Convergence of Muon with Newton-Schulz', '03-notes/2026-10/Convergence_of_Muon_with_Newton_Schulz/精读.md', ['finite Newton–Schulz', 'Muon'], '互补：收敛保证与渐近最大间隔偏置是不同命题，需核对稳定常数和多项式条件。'), relation('Training Loss Guarantees for Muon with Finite-Step Newton-Schulz Orthogonalization', '01-raw/2026-10/Training_Loss_Guarantees_for_Muon_with_Finite_Step_Newton_Schulz_Orthogonalization.pdf', ['finite orthogonalization'], '对照：有限训练损失保证不自动意味着保留理想谱隐式偏置。')],
 [relation('From Parallel to Sequential: Small-Batch Noise Restores Simplicity Bias of Muon', '01-raw/2026-10/From_Parallel_to_Sequential_Small_Batch_Noise_Restores_Simplicity_Bias_of_Muon.pdf', ['Muon', 'minibatch noise'], '互补：特征学习中噪声的可能收益与最优点附近的平稳残差代价可并存，需区分阶段和模型。'), relation('Muon with Finite Newton-Schulz: The Smoothing Benefit in Nonsmooth Nonconvex Optimization', '01-raw/2026-09/Muon_with_Finite_Newton_Schulz_The_Smoothing_Benefit_in_Nonsmooth_Nonconvex_Optimization.pdf', ['Muon', 'smoothing'], '互补：有限NS近似与高斯梯度噪声来自不同机制，不应混为同一平滑结论。')]
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
D['physics_false_positive_count'] = sum(p.get('editorial_status') == '词义误命中（不推荐）' for p in D['all_papers'])
(DAY / 'search_result.json').write_text(json.dumps(D, ensure_ascii=False, indent=2) + '\n')
print('Archive results and verified relationships saved; physics false positives:', D['physics_false_positive_count'])
