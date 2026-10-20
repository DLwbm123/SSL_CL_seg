"""Class-mean risk at fixed-family oracle states, with sealed matched controls."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import cpu,write,append,save,draw_g,local_sample,local_kl,metrics
from sc_bayeseg.window_v1.same_family_oracle import posterior,ARMS,mixture
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    costs=dict(local_optimizer=0,local_reverse=0,global_optimizer=0,train_theta=0,train_g_forward=0,
      train_g_image_forward=0,evaluation_theta=0,evaluation_g_forward=0,evaluation_g_image_forward=0,reused_baseline_records=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        old=Path(s['oracle_source']);controlroot=Path(s['allocation_source'])
        assert json.loads((controlroot/'AUDIT.json').read_text())['status']=='PASS'
        controls=json.loads((controlroot/'RESULTS.json').read_text())['arms']
        files=[Path(s['checkpoint'])]+[old/a/'S1000.pt' for a in ARMS]
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
        payload=torch.load(files[0],map_location='cpu',weights_only=False);assert payload['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False);model.load_state_dict(payload['model']);model.eval();source=cpu(model.state_dict())
        baseline=json.loads((old/'RESULTS.json').read_text())['arms']
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']])
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)]).cuda()
        counts=torch.stack([(y==k).sum((1,2)) for k in range(4)],1);assert (counts>0).all()
        weights=counts.float().reciprocal().gather(1,y.flatten(1)).reshape_as(y)/4
        assert torch.allclose(weights.sum((1,2)),torch.ones(8,device='cuda'),atol=1e-6)
        write(root/'DESIGN.json',dict(class_counts=counts.cpu().tolist(),new_risk='per-view class-mean CE; changed objective',control_reused=2))
        results={}
        for arm in ARMS:
            state=torch.load(old/arm/'S1000.pt',map_location='cpu',weights_only=False)
            sealed_optimizer=copy.deepcopy(state['optimizer'])
            for mode,T,L in (('CLASS_BALANCED',2,2),):
                name=arm+'_'+mode;out=root/name;out.mkdir(exist_ok=False)
                raw=nn.ParameterList([nn.Parameter(v.cuda().clone()) for v in state['raw']])
                opt=torch.optim.Adam(raw,lr=.03,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
                opt.load_state_dict(copy.deepcopy(state['optimizer']));restore_rng_state(state['rng'])
                assert same(cpu(opt.state_dict()),sealed_optimizer) and same(cpu(tuple(raw)),state['raw'])
                matched=torch.load(controlroot/(arm+'_T2L2')/'S0000.pt',map_location='cpu',weights_only=False)
                assert same(matched['raw'],state['raw']) and same(matched['optimizer'],state['optimizer']) and same(matched['rng'],state['rng'])
                count=[0]
                def hook(optimizer,args,kwargs):
                    count[0]+=1;costs['local_optimizer']+=1;active['step']=count[0]
                    assert count[0]<=500 and costs['local_optimizer']<=1000
                    append(root/'physical_updates.jsonl',dict(index=costs['local_optimizer'],arm=arm,mode=mode,step=count[0],source_step=1000+count[0]))
                handle=opt.register_step_post_hook(hook);active=dict(name=name,raw=raw,optimizer=opt,step=0)
                def checkpoint(step):
                    return dict(arm=arm,mode=mode,step=step,source_step=1000+step,raw=tuple(raw),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs,source_generator_cursor=8675)
                save(out/'S0000.pt',checkpoint(0));costs['reused_baseline_records']+=1
                for step in range(1,501):
                    assert time.time()-s['original_started_unix']<85500
                    opt.zero_grad(set_to_none=True);q=posterior(tuple(raw));categories=[]
                    tg=torch.Generator(device='cuda').manual_seed(720163+1000+step)
                    lg=torch.Generator(device='cuda').manual_seed(820163+1000+step)
                    for _ in range(T):
                        draw_g(model,tg);costs['train_theta']+=1
                        for _ in range(L):
                            c,r=local_sample(q,lg);categories.append((F.cross_entropy(model.spatial_logits(c,r),y,reduction='none')*weights).sum((1,2)).mean())
                            costs['train_g_forward']+=1;costs['train_g_image_forward']+=8
                    kc,kr=local_kl(q);data=torch.stack(categories).mean();kl=(kc+kr).mean()/16384;loss=data+kl
                    assert torch.isfinite(loss);loss.backward();costs['local_reverse']+=1
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in raw)
                    assert all(v.grad is None for v in model.parameters())
                    norm=float(nn.utils.clip_grad_norm_(raw,1.));opt.step();model.clear_draw()
                    append(root/'steps.jsonl',dict(arm=arm,mode=mode,step=step,data=float(data.detach()),KL=float(kl.detach()),loss=float(loss.detach()),gradient_norm_before_clip=norm))
                    if step%25==0:write(root/'STATUS.json',dict(status='RUNNING',arm=arm,mode=mode,step=step,costs=costs))
                    if step%100==0:
                        storage(s,reserve=2_000_000_000);save(out/f'S{step:04}.pt',checkpoint(step))
                assert count[0]==500 and all(int(v['step'])==1500 for v in opt.state_dict()['state'].values())
                flows,logflows=mixture(model,posterior(tuple(raw)),costs)
                save(out/'PRED_0500.pt',dict(generator_flows=flows,mean_logprob_flows=logflows))
                main=metrics(torch.stack([v.cpu() for v in flows]).mean(0),y.cpu())
                perflow=[metrics(v.cpu(),y.cpu()) for v in flows]
                expected=[float(-v.gather(1,y[:,None]).mean()) for v in logflows]
                balanced=[float((-v.gather(1,y[:,None])[:,0]*weights).sum((1,2)).mean()) for v in logflows]
                kc,kr=local_kl(posterior(tuple(raw)));klc=float(kc.mean().detach()/16384);klr=float(kr.mean().detach()/16384)
                final=dict(expected_class_balanced_NLL_flows=balanced,class_balanced_A=sum(balanced)/4+klc+klr,oracle_metrics=main,flow_metrics=perflow,expected_category_NLL_flows=expected,KLc_per_pixel=klc,KLR_per_pixel=klr,semantic_A=sum(expected)/4+klc+klr,robust_qualified24=main['qualified_24'] and all(v['qualified_24'] for v in perflow))
                results[name]=dict(matched_control=controls[arm+'_T2L2'],initial=baseline[arm]['final'],final=final,optimizer=500,source_initial_step=1000,source_final_step=1500)
                write(out/'RESULTS.json',results[name]);write(out/'EXIT.json',dict(exit_code=0))
                handle.remove();active={}
                assert same(sealed_optimizer,state['optimizer']) and same(source,model.state_dict())
                del raw,opt;torch.cuda.empty_cache()
        assert costs==dict(local_optimizer=1000,local_reverse=1000,global_optimizer=0,train_theta=2000,train_g_forward=4000,train_g_image_forward=32000,evaluation_theta=64,evaluation_g_forward=512,evaluation_g_image_forward=4096,reused_baseline_records=2)
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in meta.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',arms=results,costs=costs,source_frozen=True,source_Adam_not_aliased=True,
          any_robust_oracle_qualified=any(v['final']['robust_qualified24'] for v in results.values()),
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(name=active['name'],step=active['step'],raw=tuple(active['raw']),optimizer=active['optimizer'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
