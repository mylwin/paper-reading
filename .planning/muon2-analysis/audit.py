from pathlib import Path
import json, re, math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'02-markdown/2026-10/MUON2_Boosting_MUON_via_Adaptive_Second_Moment_Preconditioning.md'
s=source.read_text()
# Preserve original table cells for audit, including rowspan/colspan information.
tables=[]
for k,t in enumerate(re.findall(r'<table>.*?</table>',s,re.S),1):
 rows=[]
 for row in re.findall(r'<tr>(.*?)</tr>',t,re.S):
  cells=[re.sub(r'<[^>]+>','',x).strip() for x in re.findall(r'<td[^>]*>(.*?)</td>',row,re.S)]
  rows.append(cells)
 tables.append({'appearance_order':k,'rows':rows})
Path(__file__).with_name('source-tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2)+'\n')

results={}
model=['GPT-Small','GPT-Base','GPT-Large','LLaMA-60M','LLaMA-350M','LLaMA-1B','LLaMA-13B','MoE-7B-A1B']
mu3=np.array([32.70,24.69,21.13,26.37,14.91,11.63,10.54,11.93])
mu5=np.array([29.51,21.47,17.56,24.98,14.03,10.62,9.24,10.61])
x3=np.array([28.12,20.39,16.99,24.59,13.44,10.42,9.11,9.85])
x5=np.array([27.95,19.96,16.52,24.60,13.46,10.21,8.71,9.66])
results['PPL']=[{'model':model[i],'muon_3':mu3[i],'muon_5':mu5[i],'muon2_3':x3[i],'muon2_5':x5[i],
 'cross_steps_absolute_reduction':mu5[i]-x3[i],'cross_steps_relative_reduction_pct':100*(mu5[i]-x3[i])/mu5[i],
 'same_5_steps_relative_reduction_pct':100*(mu5[i]-x5[i])/mu5[i]} for i in range(8)]

llama=np.array([[24.23,42.80,32,44.37,71.22,53.43,36.52,22.95],
[26.79,46.09,33.20,49.75,72.25,55.41,40.99,22.96],
[26.37,47.81,31.80,50.23,72.20,55.01,42.50,22.97],
[27.13,49.03,35.20,52.61,74.05,55.88,45.08,22.95]])
gpt=np.array([[24.32,44.23,30.80,39.92,68.82,52.09,32.19,22.95],
[24.91,44.07,30.40,42.50,69.80,51.07,37.07,22.92],
[25.77,44.91,29.40,43.72,69.70,50.51,40.03,22.85],
[24.49,46.93,31.60,45.01,69.80,51.62,40.07,23.02]])
results['downstream']={}
for name,arr in [('LLaMA-1B',llama),('GPT-Large',gpt)]:
 d=arr[3]-arr[1];cross=arr[2]-arr[1]
 results['downstream'][name]={'recomputed_unweighted_means':arr.mean(axis=1).tolist(),
 'same_5_steps_diff_pp':d.tolist(),'same_5_steps_win_loss_tie':[int(sum(d>0)),int(sum(d<0)),int(sum(d==0))],
 'cross_steps_diff_pp':cross.tolist(),'cross_steps_win_loss_tie':[int(sum(cross>0)),int(sum(cross<0)),int(sum(cross==0))]}

results['speed']={
 'NS_iteration_reduction_pct':100*(5-3)/5,
 'same_K3_overhead_pct':100*(2971-2958)/2958,'same_K5_overhead_pct':100*(3085-3053)/3053,
 'cross_K_step_time_reduction_pct':100*(3053-2971)/3053,
 'body_matched_loss_GPUh_reduction_pct':100*(1042-796)/1042,
 'body_equal_steps_GPUh_reduction_pct':100*(1044-1016)/1044,
 'figure13_K3_GPUh_reduction_pct':100*(1041-816)/1041,
 'figure13_K5_GPUh_reduction_pct':100*(1041-797)/1041,
 'figure5_K3_training_steps_reduction_pct':100*(4796-3862)/4796,
 'figure5_K5_training_steps_reduction_pct':100*(4796-3632)/4796,
 'GPUh_from_256GPUs':{'muon5_4796steps':4796*3.053*256/3600,'muon2_3_3862steps':3862*2.971*256/3600,'muon2_5_3632steps':3632*3.085*256/3600}}
results['memory']={'1B_F_over_Muon_GB':18.57-18.56,'7B_F_over_Muon_GB':47.53-47.29,
 '7B_full_over_Muon_GB':60.66-47.29,'7B_F_savings_over_full_pct':100*(60.66-47.53)/60.66}

def polar(a):
 u,_,vt=np.linalg.svd(a,full_matrices=False);return u@vt
def ns(a,k):
 x=a/np.linalg.norm(a)
 for _ in range(k):
  z=x@x.T;x=3.4445*x-4.775*z@x+2.0315*z@z@x
 return x
def cos(a,b):return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
# First-step counterexample: exact second-moment scaling can collapse a full-rank all-positive matrix.
g=np.array([[1.,1.],[1.,2.]])
h=(.1*g)/(np.sqrt(.01*g*g)+1e-8)
results['first_step_counterexample']={'gradient':g.tolist(),'preconditioned':h.tolist(),
 'raw_condition_number':float(np.linalg.cond(g)),'preconditioned_condition_number':float(np.linalg.cond(h))}
# A positive element-wise scale can rotate the exact polar factor.
g2=np.array([[2.,1.],[1.,3.]])
v2=np.array([[1.,100.],[.01,1.]])
h2=g2/np.sqrt(v2)
results['changed_polar_target']={'original_polar':polar(g2).tolist(),'preconditioned_polar':polar(h2).tolist(),
 'cosine_of_exact_polar_targets':cos(polar(g2),polar(h2))}
# Spectral-only cosine formula.
rng=np.random.default_rng(20261019)
a=rng.normal(size=(4,7));u,sig,vt=np.linalg.svd(a,full_matrices=False);q=ns(a,3)
z=sig/np.linalg.norm(sig)
for _ in range(3):z=3.4445*z-4.775*z**3+2.0315*z**5
spectral_cos=float(z.sum()/math.sqrt(len(z)*np.sum(z*z)))
results['cosine_formula']={'matrix':cos(q,polar(a)),'spectral':spectral_cos,'CV_identity':float(1/math.sqrt(1+np.var(z)/np.mean(z)**2))}
assert abs(results['cosine_formula']['matrix']-spectral_cos)<1e-12
# Factorization is exact for separable squared energies.
p=np.array([1.,2.,3.]);qv=np.array([2.,4.,5.,6.]);v=np.outer(p,qv)
f=np.outer(v.sum(1),v.sum(0))/v.sum()
results['factorized_rank_one_error']=float(np.max(np.abs(f-v)))
# Full-rank but nonseparable matrix loses interactions under factorization.
v=np.array([[9.,1.],[1.,9.]])
f=np.outer(v.sum(1),v.sum(0))/v.sum()
results['nonseparable_second_moment']={'true':v.tolist(),'factorized':f.tolist()}
# Approximate zone thresholds for the exact stated Keller polynomial.
grid=np.geomspace(1e-6,1,100000);one=3.4445*grid-4.775*grid**3+2.0315*grid**5;five=grid.copy()
for _ in range(5):five=3.4445*five-4.775*five**3+2.0315*five**5
results['zone_thresholds_at_0.7']={'five_step_lower':float(grid[np.flatnonzero(five>=.7)[0]]),'one_step_lower':float(grid[np.flatnonzero(one>=.7)[0]])}
scores=[4,3,3,3.5,2.5,3,4];weights=[.15,.2,.2,.2,.1,.05,.1];avg=sum(a*b for a,b in zip(scores,weights))
results['score']={'scores':scores,'weights':weights,'weighted_mean':avg,'unrounded':2.5*(avg-1),'rounded':round(2*2.5*(avg-1))/2}
Path(__file__).with_name('audit-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k!='PPL'},ensure_ascii=False,indent=2))
print('Cross-step PPL reductions (%):',[round(x['cross_steps_relative_reduction_pct'],2) for x in results['PPL']])
