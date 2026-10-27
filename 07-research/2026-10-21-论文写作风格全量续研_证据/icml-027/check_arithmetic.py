"""Tiny source-consistency audit; no author code, papers or full-text data."""
import itertools,math,json
from pathlib import Path
X=list(itertools.product([0,1],repeat=3));h=lambda x,y:sum(a!=b for a,b in zip(x,y))
sets=list(itertools.combinations(X,4))
pr=max(min(h(a,b)/2 for a,b in itertools.combinations(S,2)) for S in sets)
cr=min(max(min(h(a,b) for b in S)for a in X)for S in sets)
r=json.loads(Path(__file__).with_name('arithmetic-consistency.json').read_text())
assert len(sets)==70 and pr==r['optimal_PR_defined_with_half']==1
assert cr==r['optimal_CR']==1
assert pr>math.floor(3/2)-math.log2(3)/2
for c in r['entropy_checks']:
 x=c['x'];v=-x*math.log2(x)-(1-x)*math.log2(1-x)
 assert math.isclose(v,c['H2'])
 assert v>next(v for k,v in c.items() if k.startswith('printed_upper'))
assert r['dense_int8_Q_at_d20_bytes']==2**40>16*10**9
print('Tiny arithmetic consistency reproduced; not full theorem/code certification')
