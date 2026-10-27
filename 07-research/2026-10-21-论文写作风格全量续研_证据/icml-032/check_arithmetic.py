"""Recompute six scalar ratios from Table3 and timings from Table6; not experiment replication."""
from pathlib import Path
import json
ours=[.86,325.2,202.2,257.9,.805,.475]
edm=[2.76,655,356,584,1.111,1.101]
best=[1.97,577,328,479,.844,.689]
def reductions(base):
 values=[100*(b-o)/b for b,o in zip(base,ours)]
 return {'property_reductions_percent':values,'unweighted_mean_percent':sum(values)/6}
d={'source':'locked official PDF pp8/9 Tables3/6; rounded printed scalars only','property_names':['alpha','gap','HOMO','LUMO','mu','Cv'],'ours':ours,'edm':edm,'best_listed_prior_per_property':best,'relative_to_edm':reductions(edm),'relative_to_per_property_best_listed_prior':reductions(best),'timing_ratios':[],'limits':'Simple arithmetic, no inference of significance/MAE-vs-MSE resolution, no reproduction or parametermatched quality comparison. Prior-best excludes dataset/random/N-only references and ours; chooses each prior metric minimum.'}
for steps,baseline,molecule,nocfg in [(100,610,1481,770),(50,310,690,380),(20,152,315,189)]:
 d['timing_ratios'].append({'steps':steps,'baseline_seconds':baseline,'representation_seconds':97,'molecule_with_cfg_seconds':molecule,'molecule_without_cfg_seconds':nocfg,'total_with_cfg_over_baseline':(97+molecule)/baseline,'total_without_cfg_over_baseline':(97+nocfg)/baseline})
p=Path(__file__).with_name('arithmetic-check.json');assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'edm_mean':d['relative_to_edm']['unweighted_mean_percent'],'prior_best_mean':d['relative_to_per_property_best_listed_prior']['unweighted_mean_percent']},ensure_ascii=False))
