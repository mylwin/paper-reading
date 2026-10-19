from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(20261019)
def polar(a):
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    return u @ vh

def offdiag(a):
    return float(np.linalg.norm(a - np.diag(np.diag(a))))

out = {'seed': 20261019, 'scope': 'Synthetic algebra checks; not LLM training replication', 'matrices': []}
for m, n in [(4, 9), (5, 5), (9, 4)]:
    g = rng.normal(size=(m, n))
    o = polar(g)
    s = np.linspace(.4, 1.6, min(m, n))
    if m <= n:
        z = s[:, None] * o
        wrong = o * np.linspace(.4, 1.6, n)[None, :]
        gram, wronggram = z @ z.T, wrong @ wrong.T
    else:
        z = o * s[None, :]
        wrong = np.linspace(.4, 1.6, m)[:, None] * o
        gram, wronggram = z.T @ z, wrong.T @ wrong
    nuclear = np.linalg.svd(g, compute_uv=False).sum()
    out['matrices'].append({'shape': [m,n], 'polar_row_energy_range': float(np.ptp(np.sum(o*o, axis=1))),
        'aligned_gram_offdiag': offdiag(gram), 'opposite_gram_offdiag': offdiag(wronggram),
        'aligned_inner_product': float(np.sum(g*z)), 'opposite_inner_product': float(np.sum(g*wrong)),
        'lower_alignment_bound': float(.4*nuclear)})
    assert offdiag(gram) < 1e-12
    assert float(np.sum(g*z)) >= .4*nuclear - 1e-12
    assert float(np.sum(g*wrong)) >= .4*nuclear - 1e-12

# Different residual patterns: a smaller norm need not produce less row variation.
o1 = np.sqrt(1.2)*np.eye(2)
o2 = np.diag(np.sqrt([1.1, .9]))
def nor_stats(o):
    v = np.mean(o*o, axis=1)
    return {'orthogonality_residual_2': float(np.linalg.norm(o@o.T-np.eye(2), 2)),
            'row_scale_range': float(np.ptp(1/np.sqrt(v)))}
out['nonmonotonic_residual_example'] = [nor_stats(o1), nor_stats(o2)]

# Same-step, zero-momentum DGA scaling without epsilon-dominated coordinates.
g = rng.normal(size=(3,5))
def dga_update(g):
    return polar(g) / np.sqrt(np.sum(g*g, axis=1))[:, None]
z1, z10 = dga_update(g), dga_update(10*g)
out['loss_rescaling'] = {'gradient_multiplier': 10, 'update_norm_ratio': float(np.linalg.norm(z10)/np.linalg.norm(z1))}

# Sum/mean changes a global layer factor, not relative row adaptivity (delta=0).
s_sum = 1/np.sqrt(np.sum(g*g,axis=1))
s_mean = 1/np.sqrt(np.mean(g*g,axis=1))
out['sum_mean'] = {'width': 5, 'scale_ratio_mean_over_sum': (s_mean/s_sum).tolist()}

out['reported_result_recalculation'] = {
 'BPB_absolute_reduction': .6924-.6910, 'BPB_relative_reduction_pct': 100*(.6924-.6910)/.6924,
 'train_loss_absolute_reduction': 2.2799-2.2708, 'train_loss_relative_reduction_pct': 100*(2.2799-2.2708)/2.2799,
 'CORE_absolute_gain': .3089-.2942, 'CORE_relative_gain_pct': 100*(.3089-.2942)/.2942,
}
sc = [4, 3, 3, 2, 2, 2, 4]
w = [.15,.20,.20,.20,.10,.05,.10]
mean = sum(x*y for x,y in zip(sc,w))
out['score'] = {'dimensions_1_to_5':sc, 'weights':w, 'weighted_mean': mean, 'unrounded_0_to_10': 2.5*(mean-1), 'rounded_to_half': round(2*2.5*(mean-1))/2}
p = Path(__file__).with_name('audit-results.json')
p.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
