"""Narrow exact checks for source queries, not a proof audit of the paper."""
from fractions import Fraction as F
from pathlib import Path
import json
beta=F(9,10);eta=F(1,1000);x=F(1);g=x;m=(1-beta)*g
actual=((x-eta*m)**2-x*x)/2
claimed=-eta*g*g+(3-beta)/(2*(1-beta))*eta*eta*g*g
assert actual>claimed
density=[]
for b in [F(0),F(1,2),F(9,10),F(1)]:
 k=F(7,5)*b-F(3,5)
 density.append({'beta':str(b),'noisy_group_count_exponent':str(k),'subset_size_exponent_from_d_beta_over_k':str(b-k),'count_exponent_nonnegative':k>=0})
d={'date':'2026-10-28','paper_id':'icml-2025-29bd8228','checks':{'D1_one_step':{'setup':'f(x)=x²/2, P=identity, m0=0, beta=.9, eta=.001, T=1, exact gradient/no noise','actual_change':str(actual),'printed_upper_bound':str(claimed),'violates_printed_D1':actual>claimed,'interpretation':'Query initialization/boundary terms of Lemma D.1 pp28–30; the printed recurrence drops finite geometric boundary factors.'},'C2_count_vs_size':density},'qualification':'Exact local examples to document author queries. No author code/data executed, no independent certification/refutation of every theorem or empirical finding. Keep queries in research records, not as repetitive warnings in generated manuscript prose.'}
p=Path(__file__).with_name('local-boundary-checks.json');assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print('2 local checks saved')
