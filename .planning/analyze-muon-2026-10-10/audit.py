"""Independent arithmetic checks; no training or claim of full reproduction."""
from fractions import Fraction
from pathlib import Path
from math import comb, sqrt, log, expm1, log1p
import json

def cs(k):
    return [Fraction(comb(2*s,s),4**s) for s in range(k+1)]

def phi_coeffs(k):
    c=cs(k)
    return [Fraction((2*k+1)*c[k]*c[s],k+s+1) for s in range(k+1)]
for k in range(1,21):
    a=phi_coeffs(k)
    assert all(x>0 for x in a) and sum(a)==1

def ns(s,k,q):
    for _ in range(q):
        s*=sum(float(c)*(1-s*s)**i for i,c in enumerate(cs(k)))
    return s
rows=[]
for delta in [.5,.9,.99,.9999]:
    rows.append({'delta0':delta,'kappa':2,'q':3,'chi_upper':1/sqrt(-expm1(27*log(delta))),'chi_exact_singular':1/ns(sqrt(1-delta),2,3)})
equal=[]
for r in [16,64,256]:
    s=ns(1/sqrt(r),2,3)
    equal.append({'rank':r,'q':3,'kappa':2,'epsilon_exact':1-s,'chi_exact':1/s})
a,b,c=3.4445,-4.7750,2.0315
ad={'p_at_1':a+b+c,'tau_at_1':(a+b+c)**2,'tau_derivative_at_1':(a+b+c)**2+2*(a+b+c)*(b+2*c)}
results={'global_phi_coefficient_identity_checked_degrees':list(range(1,21)),'delta_sensitivity':rows,'equal_spectrum':equal,'ad_hoc':ad,'loss_differences':[]}
for task,time,sgd,svd,new in [('CIFAR-10',60,1.010,1.182,.876),('CIFAR-100',500,2.072,2.522,1.481),('Tiny-ImageNet',2000,3.857,3.520,1.800),('NanoGPT',600,4.594,7.125,.026),('GPT-1.3B',3600,2.0156,7.4930,.0286)]:
    results['loss_differences'].append({'task':task,'seconds':time,'sgd_loss':sgd,'svd_loss':svd,'ns_loss':new,'vs_sgd_absolute':sgd-new,'vs_sgd_relative_percent':100*(sgd-new)/sgd,'vs_svd_absolute':svd-new,'vs_svd_relative_percent':100*(svd-new)/svd})
p=Path(__file__).with_name('audit_results.json');p.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
