"""Label-conditioned local-q diagnostic for the frozen categorical generator."""
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import (Interface, STREAMS, cpu, write, append, save,
    draw_g, local_sample, local_kl, semantic_loss, metrics)
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


ARMS=('ZERO','R1_WARM')


def posterior(raw):
    cm,cs,rm,rs=raw
    return cm,-4+3*torch.tanh(cs/3),rm,-4+3*torch.tanh(rs/3)


@torch.no_grad()
def mixture(model,q,costs):
    """Oracle parameters already depend on Y; this is never an image-only API."""
    flows=[];logflows=[]
    try:
        for seed in STREAMS:
            tg=torch.Generator(device='cuda').manual_seed(seed)
            lg=torch.Generator(device='cuda').manual_seed(seed+1000000)
            prob=torch.zeros((8,4,128,128),device='cuda')
            meanlog=torch.zeros_like(prob)
            for _ in range(8):
                draw_g(model,tg);costs['evaluation_theta']+=1
                for _ in range(8):
                    c,r=local_sample(q,lg)
                    logits=model.spatial_logits(c,r)
                    prob+=logits.softmax(1)
                    meanlog+=logits.log_softmax(1)
                    costs['evaluation_g_forward']+=1
                    costs['evaluation_g_image_forward']+=8
            flows.append(prob/64);logflows.append(meanlog/64)
        return flows,logflows
    finally:
        model.clear_draw()


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(local_optimizer=0,local_reverse=0,global_optimizer=0,train_theta=0,
               train_g_forward=0,train_g_image_forward=0,evaluation_theta=0,
               evaluation_g_forward=0,evaluation_g_image_forward=0,
               initialization_encoder_image_forward=0,initialization_posterior_batch_forward=0,
               reused_prediction_records=0)
    active={};started=time.time()
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        storage(s,reserve=2_000_000_000);write(root/'CONTEXT.json',s)
        previous=Path(s['r_start']);unet=Path(s['qualified_unet'])
        assert json.loads((unet/'AUDIT.json').read_text())['status']=='PASS'
        assert json.loads((unet/'RESULTS.json').read_text())['qualified24']
        assert json.loads((previous/'AUDIT.json').read_text())['status']=='PASS'
        sourcefile=Path(s['checkpoint']);warmfile=previous/'R1/S0600.pt'
        source_stat=sourcefile.stat();warm_stat=warmfile.stat()
        payload=torch.load(sourcefile,map_location='cpu',weights_only=False)
        assert payload['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False)
        model.load_state_dict(payload['model']);source=cpu(model.state_dict());model.eval()
        histories={p:p.read_text() for p in (previous/'RESULTS.json',previous/'physical_updates.jsonl',unet/'RESULTS.json',unet/'AUDIT.json')}
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        assert len(sel['train_indices'])==2 and not set(sel['train_indices'])&set(sel['heldout_indices'])
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        batch=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(batch.image,(128,128),mode='bilinear',align_corners=False)
        labels=F.interpolate(batch.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(image,k) for k in range(4)])
        labels=torch.cat([flip(labels,k) for k in range(4)])
        before_rng=capture_rng_state()
        interface=Interface().cuda().requires_grad_(False)
        interface.load_state_dict(torch.load(warmfile,map_location='cpu',weights_only=False)['model'])
        with torch.no_grad():
            feature=interface.e(image);cm,cs=interface.hc(feature.mean((2,3))).chunk(2,1)
            spatial=feature.reshape(8,16,16,8,16,8).mean((3,5))
            rm,rs=interface.hr(spatial).chunk(2,1)
            warm=tuple(cpu(v) for v in (cm,cs,rm,rs))
            assert same(posterior(tuple(v.cuda() for v in warm)),interface.posterior(feature))
        costs['initialization_encoder_image_forward']=8
        costs['initialization_posterior_batch_forward']=1
        del interface,feature;restore_rng_state(before_rng)
        initial={'ZERO':tuple(torch.zeros_like(v) for v in warm),'R1_WARM':warm}
        save(root/'INITIAL_RAW.pt',initial)
        write(root/'QUALIFICATION.json',dict(status='PASS',warm_raw_reproduces_original_q=True,
              no_inverse_tanh=True,source_generator_cursor=8675,local_optimizer=0,local_reverse=0,
              oracle_label_conditioned=True,unet_qualified=True))
        results={}
        for arm in ARMS:
            out=root/arm;out.mkdir(exist_ok=False)
            raw=nn.ParameterList([nn.Parameter(v.to('cuda').clone()) for v in initial[arm]])
            opt=torch.optim.Adam(raw.parameters(),lr=.03,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            assert not opt.state_dict()['state']
            count=[0]
            def post(optimizer,args,kwargs):
                count[0]+=1;costs['local_optimizer']+=1
                assert count[0]<=1000 and costs['local_optimizer']<=2000
                active['step']=count[0]
                append(root/'physical_updates.jsonl',dict(index=costs['local_optimizer'],arm=arm,step=count[0]))
            hook=opt.register_step_post_hook(post)
            active=dict(arm=arm,raw=raw,optimizer=opt,step=0)
            def checkpoint(step):
                return dict(arm=arm,step=step,raw=tuple(raw),optimizer=opt.state_dict(),
                            rng=capture_rng_state(),costs=costs,source_generator_cursor=8675)
            def evaluate(step):
                q=posterior(tuple(raw))
                if arm=='R1_WARM' and step==0:
                    old=torch.load(previous/'R1/PRED_0600.pt',map_location='cpu',weights_only=False)
                    assert old['seeds']==list(STREAMS)
                    flows=old['generator_flows'];logflows=None
                    costs['reused_prediction_records']+=1
                    origin='R_START_R1_600_reused_without_generator_calls'
                else:
                    flows,logflows=mixture(model,q,costs)
                    save(out/f'PRED_{step:04}.pt',dict(generator_flows=flows,mean_logprob_flows=logflows,seeds=list(STREAMS)))
                    origin='new_oracle_readout'
                kc,kr=local_kl(q)
                kc=float(kc.mean().detach()/(128**2));kr=float(kr.mean().detach()/(128**2))
                probs=torch.stack([p.cpu() for p in flows])
                main=metrics(probs.mean(0),labels.cpu())
                perflow=[metrics(p,labels.cpu()) for p in probs]
                expected=None
                if logflows is not None:
                    expected=[float(-lp.gather(1,labels[:,None]).mean()) for lp in logflows]
                row=dict(arm=arm,step=step,origin=origin,oracle_metrics=main,flow_metrics=perflow,
                         KLc_per_pixel=kc,KLR_per_pixel=kr,expected_category_NLL_flows=expected,
                         expected_category_unavailable_reason='not stored in original sealed R1 baseline; not rerun' if expected is None else None,
                         semantic_A=(sum(expected)/4+kc+kr) if expected is not None else None,
                         robust_qualified24=main['qualified_24'] and all(v['qualified_24'] for v in perflow),
                         near_R_scale_upper_fraction=float((q[3].detach()>=-1.01).float().mean()))
                append(root/'metrics.jsonl',row)
                return row
            save(out/'S0000.pt',checkpoint(0));first=evaluate(0)
            for step in range(1,1001):
                assert time.time()-s['original_started_unix']<85500
                opt.zero_grad(set_to_none=True)
                loss,parts=semantic_loss(model,posterior(tuple(raw)),labels,step,costs)
                assert torch.isfinite(loss)
                loss.backward();costs['local_reverse']+=1
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in raw)
                assert all(v.grad is None for v in model.parameters())
                norm=float(nn.utils.clip_grad_norm_(raw.parameters(),1.))
                opt.step();model.clear_draw()
                append(root/'steps.jsonl',dict(arm=arm,step=step,loss=float(loss.detach()),
                       gradient_norm_before_clip=norm,**parts))
                if step%25==0:
                    write(root/'STATUS.json',dict(status='RUNNING',arm=arm,step=step,costs=costs,
                          original_elapsed_seconds=time.time()-s['original_started_unix']))
                if step%100==0:
                    storage(s,reserve=2_000_000_000);save(out/f'S{step:04}.pt',checkpoint(step))
                if step in (250,500,1000):
                    last=evaluate(step);assert same(source,model.state_dict())
            hook.remove()
            assert count[0]==1000 and all(int(v['step'])==1000 for v in opt.state_dict()['state'].values())
            results[arm]=dict(initial=first,final=last,local_optimizer=1000,oracle_only=True)
            write(out/'RESULTS.json',results[arm]);write(out/'EXIT.json',dict(exit_code=0))
            active={};del raw,opt;torch.cuda.empty_cache()
        assert costs==dict(local_optimizer=2000,local_reverse=2000,global_optimizer=0,train_theta=4000,
             train_g_forward=8000,train_g_image_forward=64000,evaluation_theta=224,
             evaluation_g_forward=1792,evaluation_g_image_forward=14336,
             initialization_encoder_image_forward=8,initialization_posterior_batch_forward=1,
             reused_prediction_records=1)
        assert same(source,model.state_dict()) and all(v.grad is None for v in model.parameters())
        for file,oldstat in ((sourcefile,source_stat),(warmfile,warm_stat)):
            st=file.stat();assert (st.st_size,st.st_mtime_ns)==(oldstat.st_size,oldstat.st_mtime_ns)
        for p,text in histories.items():assert p.read_text()==text
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,arms=results,
              source_generator_frozen=True,source_H_and_UNet_not_trained=True,old_history_unchanged=True,
              any_robust_oracle_qualified=any(v['final']['robust_qualified24'] for v in results.values()),
              phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],
              space=storage(s),interpretation='Label-conditioned finite-budget local-q diagnostic, never image-only or patient generalization'))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:
            save(root/'failure_partial.pt',dict(arm=active['arm'],step=active['step'],raw=tuple(active['raw']),
                 optimizer=active['optimizer'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()))
        write(root/'STATUS.json',dict(status='FAILED',costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
