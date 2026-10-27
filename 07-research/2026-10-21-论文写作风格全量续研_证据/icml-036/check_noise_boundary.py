"""Analytic toy scope check of C.13; no author data/code or full proof certification."""
from fractions import Fraction
from pathlib import Path
import json
cases=[]
for tau in [Fraction(1,10000),Fraction(1,100),Fraction(1)]:
 # Independent standard Z,u,v and independent measurement noises of variance tau.
 # W=Z+u; Y'=Z+v. W not a parent of Y'. Cov(W,Y' | Z)=0.
 # Measurements Zhat,What,Yhat add independent noise, invertible if noise known.
 cov=Fraction(1)-Fraction(1)/(Fraction(1)+tau)
 assert cov>0
 cases.append({'measurement_noise_variance':str(tau),'true_conditional_covariance_W_Y_given_Z':'0','observed_conditional_covariance_What_Yhat_given_Zhat':str(cov)})
out={'date':'2026-10-23','source_query':'icml-2024-c8c985f6 pp21–22 Theorem C.13/Eq32',
'setup':'Z,u,v independent N(0,1); W=Z+u; Y_next=Z+v; inputs=(Z,W), no W->Y_next. Add independent Gaussian measurement noises of variance tau to all variables. True next transition is conditionally independent (one output; additional independent output can be appended).',
'calculation':'Cov(What,Yhat|Zhat)=Cov(What,Yhat)-Cov(What,Zhat)*Cov(Zhat,Yhat)/Var(Zhat)=1-1/(1+tau). Gaussian nonzero conditional covariance implies conditional dependence.',
'cases':cases,'interpretation':'Noisy conditioning can leave an unblocked fork through latent true Z; restorability if the unobserved noise were given is insufficient for this observed-variable calculation. Query the additional scope/assumptions of C.13; do not use its broad restorable-noise claim as a certified writing rule.',
'qualification':'Independent analytic toy boundary check, not author experiment replication, independent full theorem certification or verdict on all main results.'}
p=Path(__file__).with_name('noise-boundary-check.json');assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'cases_checked':len(cases),'all_nonzero':True}))
