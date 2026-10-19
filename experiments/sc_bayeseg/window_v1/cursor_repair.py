"""Append a reconciled terminal cursor without changing any training state."""
import copy,json,os
from pathlib import Path
import torch
from care_seg_study_v1.run import write,save
from sc_bayeseg.run import same
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);runroot=Path(s['fit']);out=runroot/'cursor_repair_v1'
    result=json.loads((runroot/'RESULTS.json').read_text())
    assert result['costs']==dict(optimizer=2404,main=2400,golden=4)
    assert json.loads((runroot/'EXIT.json').read_text())['exit_code']==0
    rows=[json.loads(v) for v in (runroot/'physical_updates.jsonl').read_text().splitlines()]
    assert [v['physical'] for v in rows]==list(range(1,2405))
    assert [v['main_step'] for v in rows if v['kind']=='main']==list(range(1,2401))
    a=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
    b=torch.load(runroot/'fullview/S2400.pt',map_location='cpu',weights_only=False)
    assert a['cursor']==4375 and b['cursor']==5575
    assert same(a,torch.load(runroot/'fullview/S0000.pt',map_location='cpu',weights_only=False))
    names=[n for n,_ in Model().named_parameters()]
    sa=a['trainer']['optimizer_state']['state'];sb=b['trainer']['optimizer_state']['state']
    for i,n in enumerate(names):
        if semantic(n):assert float(sb[i]['step']-sa[i]['step'])==2400
        else:assert same(sa[i],sb[i])
    assert all(torch.equal(v,b['model'][n]) for n,v in a['model'].items() if not semantic(n))
    assert b['trainer']['scheduler_state']['last_epoch']==6775
    assert b['trainer']['scheduler_state']['_step_count']-a['trainer']['scheduler_state']['_step_count']==2400
    for step in range(100,2401,100):
        p=torch.load(runroot/f'fullview/S{step:04}.pt',map_location='cpu',weights_only=False)
        assert p['cursor']==3175+step
        assert p['trainer']['scheduler_state']['last_epoch']==4375+step
    storage(s);out.mkdir(exist_ok=False)
    repaired=copy.deepcopy(b);repaired['cursor']=a['cursor']+2400
    save(out/'S2400_complete.pt',repaired)
    check=torch.load(out/'S2400_complete.pt',map_location='cpu',weights_only=False)
    assert check['cursor']==6775 and all(same(b[k],check[k]) for k in b if k!='cursor')
    receipt=dict(status='PASS',optimizer_updates=0,old_cursor=5575,reconciled_cursor=6775,
                 affected_periodic_checkpoints=24,source_cursor=4375,main_updates=2400,golden_updates=4,
                 model_trainer_rng_population_unchanged=True,semantic_adam_steps_added=2400,
                 nonsemantic_model_and_adam_frozen=True,scheduler_terminal_epoch=6775,
                 original_files_preserved=True,cause='periodic snapshot used old origin 3176+i; actual TrainerState used 4375+i',
                 space=storage(s))
    write(out/'RECEIPT.json',receipt);write(out/'EXIT.json',dict(exit_code=0))
    return receipt


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
