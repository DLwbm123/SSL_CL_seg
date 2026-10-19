"""One bounded, direction-qualified continuation of complete-view fitting."""
import copy,json,os,time,traceback
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState,seed_everything
from lcrseg.engine.checkpoint import capture_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.model import Model as OriginalModel
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.diagnostic_v1.run import DiagnosticMethod
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.guard_v1.run import frame
from sc_bayeseg.window_v1.common import storage


def flip(v,k):
    dims=([-1] if k&1 else [])+([-2] if k&2 else [])
    return v.flip(dims) if dims else v


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(optimizer=0,main=0,golden=0);active={};results={};paired_rng=[]
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());oldsel=json.loads((Path(s['supervised'])/'PRIVATE_SELECTION.json').read_text())
        assert len(set(sel['identities']))==4 and set(oldsel['indices'])==set(sel['train_indices']) and not set(oldsel['indices'])&set(sel['heldout_indices'])
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        lb4=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb4.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb4.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        valid=F.interpolate(lb.valid_mask.float(),(128,128),mode='nearest').bool();basebatch=replace(lb,image=image[:2],label=y[:2],valid_mask=valid)
        for k in range(4):
            assert all(torch.equal(flip(flip(v,k),k),v) for v in [basebatch.image,basebatch.label,basebatch.valid_mask])
        ancestor=json.loads((Path(s['augment'])/'ANCESTOR_AUDIT.json').read_text());assert ancestor['status']=='PASS'
        qualification=json.loads((Path(s['direction'])/'RESULTS.json').read_text());assert qualification['qualified'] and qualification['model_and_rng_unchanged']
        payload=torch.load(s['checkpoint'],map_location='cpu',weights_only=False);assert payload['cursor']==4375 and payload['population']==dict(labeled=2,unlabeled=0)
        for arm in ['fullview']:
            out=root/arm;out.mkdir(exist_ok=False);seed_everything(163)
            model=Model().cuda();method=DiagnosticMethod(model,2,'mean_ce');method.begin_site('Siemens',None,6775)
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
            restore(payload,method,trainer);assert same(payload['model'],model.state_dict()) and same(payload['trainer'],trainer.state_dict())
            named=list(model.named_parameters());assert len(named)==100 and optimizer.state_dict()['param_groups'][0]['params']==list(range(100))
            for n,p in named:p.requires_grad_(semantic(n))
            frozen={k:v.detach().clone() for k,v in model.state_dict().items() if not semantic(k)}
            frozen_ids=[i for i,(n,p) in enumerate(named) if not semantic(n)]
            frozen_adam={i:copy.deepcopy(optimizer.state_dict()['state'].get(i)) for i in frozen_ids}
            local=dict(main=0,golden=0);phase=['golden'];index=[0];goldens=[]
            def pre(opt,args,kwargs):
                assert costs['optimizer']<2404 and local[phase[0]]<(2400 if phase[0]=='main' else 4)
                assert opt.param_groups[0]['lr']==3e-4 and all(p.grad is None or torch.isfinite(p.grad).all() for _,p in named)
                assert all(p.grad is None for n,p in named if not semantic(n))
            def post(opt,args,kwargs):
                costs['optimizer']+=1;costs[phase[0]]+=1;local[phase[0]]+=1
                append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],kind=phase[0],kind_index=costs[phase[0]],arm=arm,main_step=index[0]+1,source_cursor=4375))
                active['cursor']=4375+index[0]+1
            pre_hook=optimizer.register_step_pre_hook(pre);post_hook=optimizer.register_step_post_hook(post)
            fullbatch=replace(basebatch,image=torch.cat([flip(basebatch.image,k) for k in range(4)]),label=torch.cat([flip(basebatch.label,k) for k in range(4)]),valid_mask=torch.cat([flip(basebatch.valid_mask,k) for k in range(4)]),case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
            assert set(fullbatch.patient_id)==set(lb.patient_id) and len(fullbatch.image)==8
            def batch_at(i):return fullbatch
            def update(i):
                index[0]=i
                return trainer.train_step(batch_at(i),None,state=TrainerState(global_step=4375+i,site_step=4375+i))
            def evaluate(step):
                row=dict(step=step,groups={part:frame(model,image[ix],y[ix])['metrics'] for part,ix in [('train',slice(0,2)),('heldout',slice(2,4))]},train_views={str(k):frame(model,flip(image[:2],k),flip(y[:2],k))['metrics'] for k in range(4)})
                append(out/'metrics.jsonl',row);return row
            initial=snapshot(method,trainer,4375);storage(s);save(out/'S0000.pt',initial)
            initial_metrics=evaluate(0);active.update(out=out,method=method,trainer=trainer,cursor=4375,arm=arm)
            try:
                for i in range(2400):
                    rng=capture_rng_state()
                    paired_rng.append(copy.deepcopy(rng))
                    if i in [0,2399]:
                        phase[0]='golden';before=snapshot(method,trainer,4375+i)
                        first=update(i);expected=snapshot(method,trainer,3176+i);restore(before,method,trainer)
                        second=update(i);observed=snapshot(method,trainer,3176+i)
                        equal=torch.equal(first.total_loss,second.total_loss) and same(expected,observed);assert equal
                        goldens.append(dict(main_step=i+1,full_state_and_loss_equal=True));write(out/'GOLDEN.json',goldens)
                        restore(before,method,trainer)
                    phase[0]='main';result=update(i)
                    append(out/'steps.jsonl',dict(step=i+1,loss=float(result.total_loss.detach()),views=[0,1,2,3],batch_images=8))
                    if (i+1)%100==0:
                        storage(s);save(out/f'S{i+1:04}.pt',snapshot(method,trainer,3176+i))
                        assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
                    if (i+1)%600==0:evaluate(i+1)
                    if i==0 or (i+1)%25==0:write(root/'STATUS.json',dict(status='RUNNING',arm=arm,step=i+1,costs=costs,loss=float(result.total_loss.detach()),space=storage(s)))
                state=optimizer.state_dict()['state'];assert all(same(state.get(i),v) for i,v in frozen_adam.items())
                assert local==dict(main=2400,golden=4) and len(goldens)==2
                final=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
                qualified=all(v>=.8 for view in final['train_views'].values() for row in view['stochastic']['dice'] for v in row)
                result=dict(status='COMPLETE',costs=local,initial=initial_metrics,final=final,training_semantic_qualified=qualified,nonsemantic_model_and_adam_frozen=True,golden=goldens)
                write(out/'RESULTS.json',result);results[arm]=result
            finally:pre_hook.remove();post_hook.remove()
            active.clear();del model,method,trainer;torch.cuda.empty_cache()
        assert costs==dict(optimizer=2404,main=2400,golden=4) and len(paired_rng)==2400
        storage(s);save(root/'PRIVATE_RNG_PAIRS.pt',paired_rng)
        write(root/'RESULTS.json',dict(status='COMPLETE',started=started,finished=time.time(),costs=costs,arms=results,ancestor=ancestor,source_rng_restored=True,recorded_rng_steps=2400,geometry_inverse_exact=True,source_cursor=4375,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:
            storage(s);save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
