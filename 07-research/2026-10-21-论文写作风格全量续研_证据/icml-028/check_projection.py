"""Finite illustrative check of Eq(8) under decoupled per-label weights."""
import itertools,math,json
from pathlib import Path
# n=1, c=4; phi_j=1 fixed, independent w_j in [-1,1], Lambda=A=B=1.
c=4
signs=list(itertools.product([-1,1],repeat=c))
values=[max(sum(e*w for e,w in zip(eps,w))/c for w in signs)for eps in signs]
r={'scope':'finite algebraic consistency check of printed projection step, not full theorem or source-code replication','source_position':'p18 Eq8, p15 Step3 and p28 Eq19','interpretation':'Eq1 allows alpha(w)=max_j|w_j|; independent label weights, fixed scalar phi_j=1. Product sample has n=1,c=4. Supremum uses the joint vector class as expanded in the author proof.','n':1,'c':c,'Lambda':1,'A':1,'B':1,'rademacher_patterns':len(signs),'hypotheses_vertices':len(signs),'actual_empirical_R':sum(values)/len(values),'printed_upper_B_over_sqrt_nc':1/math.sqrt(c),'conflicts_with_printed_upper_under_this_interpretation':True,'qualification':'Clarify whether a coupled joint norm or a different definition of P(F) was intended; the arithmetic does not determine validity of all main bounds.'}
assert r['actual_empirical_R']==1>r['printed_upper_B_over_sqrt_nc']==.5
p=Path(__file__).with_name('projection-consistency.json')
if p.exists():assert json.loads(p.read_text())==r
else:p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(r,ensure_ascii=False))
