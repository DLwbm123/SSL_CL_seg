"""Immutable training-L local-function summaries in effective native/F coordinates."""
import copy
import math
import torch
from .protocol import MEMORY_BATCHES

COORDINATE = 'effective-native-weights-and-cumulative-F-at-fixed-domain-final'


def finite(value):
    if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def seal_site(domain, anchor, batches):
    """Each of exactly32 fixed-model batches supplies (class, CE, native/F grads, pixels).

    The future collector must build all observations with this SAME final anchor,
    not mix online trajectory gradients with a different final anchor.
    """
    if domain not in ('Siemens','Philips','GE','Canon') or len(batches)!=MEMORY_BATCHES:
        raise ValueError('exact registered domain and32 training-L batches required')
    if not anchor or not finite(anchor) or any(not isinstance(v,torch.Tensor) for v in anchor.values()):
        raise ValueError('invalid native/F anchor')
    classes=[]
    for c in range(4):
        observations=[]
        for batch in batches:
            indices=[row[0] for row in batch]
            if len(indices)!=len(set(indices)) or any(type(i)is not int or not 0<=i<4 for i in indices):
                raise ValueError('duplicate/invalid class observation')
            observations.extend(row for row in batch if row[0]==c)
        if not observations:
            raise ValueError('missing training-L class; no resampling or fabricated support')
        jac={n:torch.zeros_like(v,device='cpu',dtype=torch.float32) for n,v in anchor.items()}
        importance=copy.deepcopy(jac); losses=[]; support=0
        for _,loss,gradients,pixels in observations:
            if not math.isfinite(loss) or loss<0 or type(pixels)is not int or pixels<=0 or set(gradients)!=set(anchor) or not finite(gradients):
                raise ValueError('invalid class CE/gradient/support')
            for n,v in gradients.items():
                if v.shape!=anchor[n].shape: raise ValueError('native gradient shape mismatch')
                v=v.detach().cpu().float();jac[n]+=v;importance[n]+=v.square()
            support+=pixels;losses.append(loss)
        count=len(observations)
        classes.append(dict(mean_CE=sum(losses)/count,jacobian={n:v/count for n,v in jac.items()},
                            importance={n:v/count for n,v in importance.items()},observations=count,support=support))
    result=dict(domain=domain,coordinate=COORDINATE,
                anchor={n:v.detach().cpu().clone() for n,v in anchor.items()},classes=classes)
    if not finite(result):raise ValueError('nonfinite accumulated memory')
    return result


class History:
    def __init__(self, sites=(), expected_domains=()):
        self._sites=copy.deepcopy(list(sites))
        if list(expected_domains)!=list(('Siemens','Philips','GE','Canon')[:len(expected_domains)]):
            raise ValueError('history must be a registered domain prefix')
        if [s.get('domain') for s in self._sites]!=list(expected_domains):
            raise ValueError('historical domain order/coverage mismatch')
        for site in self._sites:
            if set(site)!= {'domain','coordinate','anchor','classes'} or site['coordinate']!=COORDINATE or len(site['classes'])!=4 or not finite(site):
                raise ValueError('invalid historical schema/coordinates')
            if not site['anchor'] or any(not isinstance(v,torch.Tensor) for v in site['anchor'].values()):
                raise ValueError('nonempty native/F tensor anchor required')
            for cls in site['classes']:
                if set(cls)!= {'mean_CE','jacobian','importance','observations','support'} or cls['mean_CE']<0 or type(cls['support'])is not int or cls['support']<=0 or type(cls['observations'])is not int or not 0<cls['observations']<=MEMORY_BATCHES:
                    raise ValueError('invalid class support/observation schema')
                for key in ('jacobian','importance'):
                    if set(cls[key])!=set(site['anchor']):raise ValueError('memory native key mismatch')
                    for n,v in cls[key].items():
                        if v.shape!=site['anchor'][n].shape or (key=='importance' and (v<0).any()):
                            raise ValueError('invalid class native coordinates/importance')

    def state(self):
        return copy.deepcopy(self._sites)

    def terms(self, values):
        terms={}
        for site in self._sites:
            if set(values)!=set(site['anchor']) or any(values[n].shape!=a.shape for n,a in site['anchor'].items()) or not finite(values):
                raise ValueError('current effective native/F coordinates mismatch')
            for c,cls in enumerate(site['classes']):
                diagonal=[]; signed=next(iter(values.values())).sum()*0
                for n,current in values.items():
                    anchor=site['anchor'][n].to(current); diff=current-anchor
                    weight=cls['importance'][n].to(current);jac=cls['jacobian'][n].to(current)
                    if weight.count_nonzero():
                        diagonal.append((weight*diff.square()).sum()/weight.sum()/anchor.square().mean().clamp_min(1e-8))
                    signed=signed+(jac*diff).sum()
                diag=torch.stack(diagonal).mean() if diagonal else signed*0
                terms[f"{site['domain']}/class{c}"]=diag+signed.clamp_min(0).square()/max(cls['mean_CE']**2,1e-6)
        return terms

    def penalty(self, values):
        terms=self.terms(values)
        return torch.stack(list(terms.values())).mean() if terms else next(iter(values.values())).sum()*0
