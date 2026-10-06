"""Two deployment rules; unchanged V7 native data, losses and student updates."""
import copy, fcntl, hashlib, json, time, traceback
from pathlib import Path
import torch
from experiments.qprompt_rl_v1.v7_unlabeled_coreset import worker as v
from . import protocol as p
q,e,w,io=v.q,v.e,v.w,v.io
ROOT,C=v.ROOT,v.C

def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def oldjob(name):return Path(C['old_campaign'])/'scopes/pilot/jobs'/name

def policy(seed):
    path=oldjob(f'learn_{C["domain"]}_{seed}')/'FROZEN_POLICY.private.pt'
    saved=load(path);assert saved['commit']==p.TRAINING and saved['domain']==C['domain'] and saved['controller']==seed
    assert filehash(path)==e.read(Path(C['root'])/'PROVENANCE_AUDIT.json')['policies'][f'{C["domain"]}/{seed}']['sha256']
    prior=e.rng_state();model=q.Policy();model.load_state_dict(saved['state']);model.eval();model.requires_grad_(False);e.rng_restore(prior)
    return model,path

class Uniform(torch.nn.Module):
    def forward(self,x,emb,chosen,k):
        logits=torch.zeros(len(x));logits[chosen]=-torch.inf;return logits

def sample(model,x,emb,domain,seed,block):
    rng=torch.Generator().manual_seed(e.stable('V7.1/deploy',domain,seed,block))
    return q.sequence(model,x,emb,p.old.k(domain),generator=rng)[0]

def choose(t,method,seed,block,model):
    if method=='GRPO_SAMPLE':x,emb=v.features(t)
    else:x,emb=torch.zeros(len(t.provider._u),q.FEATURES),torch.zeros(len(t.provider._u),1)
    start=time.monotonic()
    with torch.no_grad():chosen=sample(model,x,emb,C['domain'],seed,block)
    w.COST['selection_wall_seconds']+=time.monotonic()-start;w.COST['selection_calls']+=1
    return chosen

def qualification(ledger):
    t=v.create(ledger);entry=v.snapshot(t);checks=[]
    for method in p.METHODS:
        v.restore(t,entry);model,_=policy(601) if method=='GRPO_SAMPLE' else (Uniform(),None)
        selected=choose(t,method,601,0,model)
        assert e.same(entry,v.snapshot(t)),'selection changed native state'
        v.advance(t,ledger,'smoke',method+'/continuous',selected,4);expected=v.snapshot(t)
        v.restore(t,entry);again=choose(t,method,601,0,model);assert again==selected
        v.advance(t,ledger,'smoke',method+'/split-first',selected,2)
        v.save(t,ROOT/'resume.private.pt',selection=selected)
        saved=load(ROOT/'resume.private.pt');v.restore(t,saved['state']);assert saved['selection']==selected
        v.advance(t,ledger,'smoke',method+'/split-last',selected,2)
        assert e.same(expected,v.snapshot(t)),method+' exact native restore failed'
        checks.append(dict(method=method,status='PASS',physical_calls=8,exact_all_snapshot_fields=True))
    e.write(ROOT/'QUALIFICATION.json',dict(status='PASS',checks=checks,physical_student_updates=16,controller_updates=0))

def endpoint(ledger):
    t=v.create(ledger);v.save(t,ROOT/'ENTRY.private.pt');io.export(t,ROOT/'entry.pt')
    model,path=policy(C['controller']) if C['method']=='GRPO_SAMPLE' else (Uniform(),None)
    before=e.cpu(model.state_dict());hashbefore=filehash(path) if path else None
    started=time.time();history=[]
    for block in range(p.old.DOMAINS[C['domain']]//100):
        selected=choose(t,C['method'],C['controller'],block,model)
        previous=set(history[-1]) if history else set();current=set(selected)
        e.append(ROOT/'SELECTIONS.private.jsonl',dict(block=block,indices=selected))
        exposure=list(t.provider.exposure)
        v.advance(t,ledger,'endpoint',f'{C["method"]}/{block}',selected,100)
        delta=[a-b for a,b in zip(t.provider.exposure,exposure)]
        e.append(ROOT/'SELECTION_SUMMARY.jsonl',dict(block=block,size=len(selected),previous_jaccard=len(current&previous)/len(current|previous),actual_exposure_sorted=sorted(delta),sum_exposures=sum(delta)))
        history.append(selected);v.save(t,ROOT/'student_latest.private.pt',method=C['method'],controller=C['controller'])
        w.v4.status('ENDPOINT_TRAINING',step=t.step,physical_calls=dict(ledger.count))
    assert e.same(before,e.cpu(model.state_dict())) and all(t.grad is None for t in model.parameters())
    assert path is None or filehash(path)==hashbefore
    assert w.COST['reward_label_sample_accesses']==0
    dest=ROOT/'endpoint.pt';io.export(t,dest)
    e.write(ROOT/'ENDPOINT_REGISTRY.json',[dict(seed=168,domain=C['domain'],method=C['method'],controller=C['controller'],step=t.step,path=str(dest),training_wall_seconds=time.time()-started,selected_union=len(set(i for h in history for i in h)),U_size=len(t.provider._u),actual_U_image_accesses=sum(t.provider.exposure))])
    e.write(ROOT/'FREEZE.json',dict(status='FROZEN',time=time.time(),controller_updates=0,reward_label_sample_accesses=0,policy_hash_before=hashbefore,policy_hash_after=filehash(path) if path else None,commit=C['commit']))

def main():
    torch.set_num_threads(2);torch.cuda.set_device(0)
    lock=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(),'no retry'
    ledger=w.v4.Ledger(ROOT);ledger.caps=C['caps'];ledger.seen=set()
    started=time.time()
    try:
        if C['job']=='diagnosis':
            from .diagnosis import diagnosis
            diagnosis(ledger)
        else:dict(qualification=qualification,endpoint=endpoint,evaluate=w.evaluate)[C['job']](ledger)
        w.audit(ledger)
        e.write(ROOT/'FINAL.json',dict(status='COMPLETE',start=started,time=time.time(),physical_calls=dict(ledger.count),peak_cuda_allocated=torch.cuda.max_memory_allocated(),commit=C['commit']))
        w.v4.status('COMPLETE')
    except BaseException as exc:
        w.v4.status('FAILED',error=repr(exc),traceback=traceback.format_exc(),physical_calls=dict(ledger.count));raise
    finally:e.write(ROOT/'MEASURED_COSTS.json',dict(w.COST,job_wall_seconds=time.time()-started))

if __name__=='__main__':
    with w.NativeOperations(ROOT/'operations'):main()
