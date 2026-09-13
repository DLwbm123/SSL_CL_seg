"""Reuse the frozen common training loop; change only its contract and step kernel."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from . import core as c,contract as ct
from experiments.lcrseg.dpr_v0_1 import engine as common
from experiments.lcrseg.dpr_v0_1.engine import save,rng_state,restore_rng,Operations

# Common pending labels retain their old spelling, but contain only final-adopted weights.
initial=common.initial
diagnose=common.diagnose
native_pending=common.pending

def pending(root,model,ema,opt,p,counts,get_batches,**kwargs):
    if p['source']!='SYNTHETIC' and p['epoch']==20 and p['index']==p['steps']-1:
        tid=p['task']['task_id'].replace(p['task']['arm'],'F_CONV')
        expected=ct.read(root.parent.parent/'BASELINE_BINDING.json')['baselines'][tid]['warmup']
        actual=dict(student_hash=c.state_hash(model),EMA_hash=c.state_hash(ema),optimizer_hash=c.optimizer_hash(opt),label_order_hash=p['label_order_hash'],U_opens=p['U_opens'])
        checks={k:actual[k]==expected[k] for k in expected}
        c.write_json(root/'warmup.json',dict(status='PASS' if all(checks.values()) else 'FAIL',actual=actual,expected=expected,checks=checks))
        p['row']['checks']['historical_F_CONV_warmup_exact']=all(checks.values())
    return native_pending(root,model,ema,opt,p,counts,get_batches,**kwargs)

def train(*args,**kwargs):
    base=kwargs.get('base',args[0] if args else None);tid=kwargs.get('task_id',args[1] if len(args)>1 else None)
    root=Path(base)/'tasks'/tid
    def failure(model,ema,opt,counts,phase):
        path=root/'FAILED_PHASE_FORENSIC.pt'
        if path.exists():raise FileExistsError('failure evidence protected')
        save(path,dict(state='FORENSIC_ONLY_NOT_RESUMABLE',phase=phase,student=model.state_dict(),EMA=ema.state_dict(),optimizer=opt.state_dict(),rng=rng_state(next(model.parameters()).device),logical_counts=dict(counts),actual_operations='separate Operations receipt',recovery='pre_step.pt only unless latest is a final-adopted pending checkpoint'))
    def receipt(path,obj):
        if Path(path).name=='receipt.json' and obj.get('status')=='TRAINING_COMPLETE':
            obj['memory'].update(temporary_diagnostic_CPU_snapshot_bytes=3*438192*4,active_FP32_proposal_snapshots=3,pooled_response_fields=3,
                deployment='ordinary single student',current_EMA_queries=0)
        return c.parent.write_json(path,obj)
    with patch.multiple(common,c=c,ct=ct,pending=pending),patch.object(c,'FAIL_HOOK',failure),patch.object(c,'write_json',receipt):
        return common.train(*args,**kwargs)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','task_id','data','reference'):p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--resume',action='store_true');a=vars(p.parse_args());ct.neutral_subprocess_paths();root=Path(a['base'])/'tasks'/a['task_id']
    try:
        with Operations(root.parent/(root.name+'_operations')):train(**a,device=torch.device('cuda:0'))
    except BaseException as exc:
        c.write_json(root.parent/(root.name+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(exc),phase=c.PHASE,recovery='final pending first; raw/candidate forensic state never resumed'));raise
