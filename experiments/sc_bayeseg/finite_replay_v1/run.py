"""Exact short J replay, followed by bounded copy-only attribution."""
import copy,json,os,time,traceback,math
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState
from lcrseg.engine.checkpoint import restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.split_v1.run import evaluate
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.adam_audit_v1.run import candidate,dot,norm
from sc_bayeseg.diagnostic_v1.run import dice


def logits(model,image):
    mu=model.logit_encoder(model.shape_encoder(image).chunk(2,1)[0]).chunk(2,1)[0]
    return torch.cat([torch.zeros_like(mu[:,:1]),mu],1)


def values(z,y):
    ce=F.cross_entropy(z,y,reduction='none');mask=y==2
    return [ce.mean(),ce[mask].mean()]


def cpu(obj):
    if torch.is_tensor(obj):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu(v) for v in obj)
    return obj


def summaries(x,subsets):
    rows={}
    for name,mask in subsets.items():
        rows[name]={}
        for i in [-1]+list(range(len(x))):
            chosen=mask if i==-1 else mask[i:i+1]
            v=x if i==-1 else x[i:i+1]
            a=v[chosen].double()
            rows[name]['aggregate' if i==-1 else 'image_'+str(i)]=dict(count=a.numel(),mean=float(a.mean()) if a.numel() else None,negative=int((a<0).sum()))
    return rows


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(optimizer=0,main=0,golden=0,candidates=0,extra_reverse=0)
    active={};clip_original=torch.nn.utils.clip_grad_norm_;handles=[]
    def status(phase,**kw):write(root/'STATUS.json',dict(phase=phase,costs=costs.copy(),**kw))
    def grad(*args,**kwargs):
        assert costs['extra_reverse']<150
        costs['extra_reverse']+=1
        append(root/'reverse_calls.jsonl',dict(index=costs['extra_reverse'],time=time.time()))
        return torch.autograd.grad(*args,**kwargs)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(s['origin']);payload=torch.load(origin/'J'/'initial.pt',map_location='cpu',weights_only=False)
        assert payload['cursor']==3000 and payload['population']==dict(labeled=2,unlabeled=0)
        assert json.loads((origin/'EXIT.json').read_text())['exit_code']==0 and not (origin/'FAILURE.json').exists()
        oldledger=[json.loads(v) for v in (origin/'physical_updates.jsonl').read_text().splitlines()]
        assert [v['physical'] for v in oldledger]==list(range(1,1205))
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        sel=json.loads((origin/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=Model().cuda();method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600)
        params=list(model.parameters());assert len(params)==100 and all(p.requires_grad for p in params)
        optimizer=torch.optim.Adam(params,lr=3e-4,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(payload,method,trainer);assert same(snapshot(method,trainer,3000),payload)
        history={r['added_step']:r for r in map(json.loads,(origin/'J'/'telemetry.jsonl').read_text().splitlines())}
        metrics={r['added_step']:r for r in map(json.loads,(origin/'J'/'metrics.jsonl').read_text().splitlines())}
        baseline=evaluate(model,image,y);assert all(same(v,metrics[0][k]) for k,v in baseline.items())
        write(root/'RESTORE.json',dict(status='PASS',complete_state_equal=True,baseline_equal=True,population=payload['population'],parameter_tensors=100))
        save(root/'S00.pt',payload)
        if s.get('preflight_only'):
            with torch.no_grad():
                zz=logits(model,image);ff=values(zz,y)
            assert all(torch.isfinite(v) for v in ff)
            assert int((y==2).sum())==490
            assert same(snapshot(method,trainer,3000),payload)
            write(root/'ZERO_CHECK.json',dict(status='PASS',costs=costs,full_state_unchanged=True,baseline_equal=True,mean_ce=[float(v) for v in ff]))
            write(root/'EXIT.json',dict(exit_code=0));return
        phase=['main'];index=[0];stash={}
        def guard(opt,args,kwargs):
            assert costs['optimizer']<27 and costs[phase[0]]<(25 if phase[0]=='main' else 2)
            assert all(g['lr']==3e-4 for g in opt.param_groups)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            append(root/'attempts.jsonl',dict(attempt=costs['optimizer']+1,phase=phase[0],step=index[0],time=time.time()))
        def receipt(opt,args,kwargs):
            costs['optimizer']+=1;costs[phase[0]]+=1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],phase=phase[0],step=index[0],time=time.time()))
        handles=[optimizer.register_step_pre_hook(guard),optimizer.register_step_post_hook(receipt)]
        def clip(parameters,max_norm,*args,**kwargs):
            pp=list(parameters);v=clip_original(pp,max_norm,*args,**kwargs)
            stash.update(norm=float(v),clipped=[p.grad.detach().clone() for p in pp])
            return v
        torch.nn.utils.clip_grad_norm_=clip
        def step(k):return trainer.train_step(lb,None,state=TrainerState(global_step=3000+k-1,site_step=3000+k-1))
        telemetry=[]
        for k in range(1,26):
            index[0]=k;active.update(cursor=3000+k-1)
            r=step(k);assert torch.isfinite(r.total_loss)
            active['cursor']=3000+k
            row=dict(step=k,loss=float(r.total_loss),gradient_norm=stash['norm'],**r.scalars);telemetry.append(row)
            append(root/'telemetry.jsonl',row)
            save(root/f'G{k:02}.pt',cpu(stash['clipped']))
            save(root/f'S{k:02}.pt',cpu(snapshot(method,trainer,3000+k)))
            if k in history:
                assert row['loss']==history[k]['loss'] and abs(row['gradient_norm']-history[k]['gradient_norm_before_clip'])<=1e-4
                assert all(row[key]==value for key,value in history[k].items() if key in r.scalars and key not in ['training_step_seconds','peak_memory_bytes'])
            status('MAIN',step=k)
        historical=evaluate(model,image,y)
        assert all(same(v,metrics[25][k]) for k,v in historical.items())
        write(root/'HISTORICAL.json',dict(initial=baseline,step25=historical,exact_metrics_equal=True,telemetry_steps_1_25_equal=True))
        phase[0]='golden';golden=[]
        for k in [1,25]:
            index[0]=k;before=torch.load(root/f'S{k-1:02}.pt',weights_only=False,map_location='cpu');restore(before,method,trainer)
            r=step(k);observed=cpu(snapshot(method,trainer,3000+k));expected=torch.load(root/f'S{k:02}.pt',weights_only=False,map_location='cpu')
            equal=same(observed,expected);golden.append(dict(step=k,complete_state_equal=equal,loss_equal=float(r.total_loss)==telemetry[k-1]['loss']))
            assert equal and golden[-1]['loss_equal']
        write(root/'GOLDEN.json',golden)
        torch.nn.utils.clip_grad_norm_=clip_original
        for h in handles:h.remove()
        handles=[]
        # No optimizer invocation can occur in the offline section.
        def forbidden(*a,**kw):raise AssertionError('offline optimizer call forbidden')
        optimizer.step=forbidden
        state=lambda k:torch.load(root/f'S{k:02}.pt',map_location='cpu',weights_only=False)
        weights=lambda p:[p['model'][name].cuda() for name,_ in model.named_parameters()]
        def load(p):model.load_state_dict(p['model']);model.clear_draw()
        def getz(p):
            load(p)
            with torch.no_grad():return logits(model,image).detach()
        zs=[getz(state(k)).cpu() for k in range(26)]
        z0=zs[0].cuda();mask=y==2
        allcomp=z0.masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf).argmax(1)
        margin0=z0[:,2]-z0.gather(1,allcomp[:,None])[:,0]
        pad=F.pad(mask[:,None].float(),(1,1,1,1),mode='replicate')[:,0]
        interior=mask & pad[:,1:-1,:-2].bool() & pad[:,1:-1,2:].bool() & pad[:,:-2,1:-1].bool() & pad[:,2:,1:-1].bool()
        subsets=dict(all_class2=mask,near_decision=mask&(margin0.abs()<=1),near_background=mask&(z0[:,2].abs()<=1),spatial_inner_boundary=mask&~interior)
        assert [int(v.sum()) for v in subsets.values()]==[490,74,51,293]
        closure=[];crosscache={};metricrows=[]
        for k,z in enumerate(zs):
            z=z.cuda();f=values(z,y)
            metricrows.append(dict(step=k,ce=float(f[0]),class2_ce=float(f[1]),dice=dice(z.argmax(1),y),class2_confusion=[int((mask&(z.argmax(1)==c)).sum()) for c in range(4)]))
        def candidate_receipt(kind,**kw):
            assert costs['candidates']<14
            costs['candidates']+=1
            append(root/'candidate_evaluations.jsonl',dict(index=costs['candidates'],kind=kind,**kw))
        for k in range(1,26):
            old,new=state(k-1),state(k);load(old)
            oldp=weights(old);newp=weights(new);delta=[b-a for a,b in zip(oldp,newp)]
            gg=torch.load(root/f'G{k:02}.pt',weights_only=False,map_location='cuda')
            ost=old['trainer']['optimizer_state'];group=ost['param_groups'][0]
            with torch.no_grad():
                cand,_,_,_,info=candidate(params,gg,[ost['state'][i] for i in group['params']],group)
                err=max(float((a-b).abs().max()) for a,b in zip(cand,delta));assert err<=1e-7
            z=logits(model,image);f=values(z,y)
            gf=[grad(v,params,allow_unused=True,retain_graph=True) for v in f]
            linear=[dot(g,delta) for g in gf]
            probe=torch.zeros_like(z,requires_grad=True)
            jt=grad(z,params,grad_outputs=probe,create_graph=True,retain_graph=True,allow_unused=True)
            prod=sum((a*b).sum() for a,b in zip(jt,delta) if a is not None)
            dz=grad(prod,probe)[0].detach()
            actual=[metricrows[k][key]-metricrows[k-1][key] for key in ['ce','class2_ce']]
            comp=z.detach().masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf).argmax(1)
            m_old=z[:,2].detach()-z.detach().gather(1,comp[:,None])[:,0]
            zn=zs[k].cuda();m_fixed=zn[:,2]-zn.gather(1,comp[:,None])[:,0]
            newcomp=zn.masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf).argmax(1)
            m_new=zn[:,2]-zn.gather(1,newcomp[:,None])[:,0]
            ma=dz[:,2]-dz.gather(1,comp[:,None])[:,0];mr=m_fixed-m_old-ma;ms=m_new-m_fixed
            assert float(ms.max())<=1e-6
            residual=m_new-m_old-ma-mr-ms;assert float(residual.abs().max())<=1e-5
            oldpred=z.detach().argmax(1);newpred=zn.argmax(1)
            transitions={}
            for name,chosen in subsets.items():
                transitions[name]={}
                for i in [-1,0,1]:
                    ch=chosen if i==-1 else chosen[i:i+1];a=oldpred if i==-1 else oldpred[i:i+1];b=newpred if i==-1 else newpred[i:i+1]
                    transitions[name]['aggregate' if i==-1 else f'image_{i}']={f'{aa}_to_{bb}':int((ch&(a==aa)&(b==bb)).sum()) for aa in range(4) for bb in range(4)}
            row=dict(step=k,actual=actual,linear=linear,remainder=[a-b for a,b in zip(actual,linear)],candidate_actual_max_error=err,delta_norm=norm(delta),
                     margins={name:summaries(v,subsets) for name,v in [('actual',m_new-m_old),('linear',ma),('remainder',mr),('switch',ms)]},transitions=transitions,margin_closure_max=float(residual.abs().max()))
            closure.append(row);append(root/'attribution.jsonl',row)
            if k in [1,2,25]:crosscache[k]=dict(base=[float(v.detach()) for v in f],gf=[[None if g is None else g.detach().cpu() for g in grads] for grads in gf],actual=actual,linear=linear)
            if k in [1,2,5,10,25]:
                for alpha in [1/16,1/4]:
                    candidate_receipt('scaled',step=k,alpha=alpha)
                    with torch.no_grad():
                        for p,a,d in zip(params,oldp,delta):p.copy_(a+alpha*d)
                        fc=[float(v) for v in values(logits(model,image),y)]
                        landed=[p.detach()-a for p,a in zip(params,oldp)]
                    E=[v-float(b.detach())-alpha*l for v,b,l in zip(fc,f,linear)]
                    append(root/'scaled.jsonl',dict(step=k,alpha=alpha,actual=[v-float(b.detach()) for v,b in zip(fc,f)],linear=[alpha*l for l in linear],E=E,E_per_alpha=[v/alpha for v in E],landed_delta_norm=norm(landed),rounding_error_norm=norm([a-alpha*d for a,d in zip(landed,delta)])))
            status('OFFLINE',step=k)
        cross=[]
        for k in [2,25]:
            cells={'00':dict(Q=crosscache[1]['actual'],A=crosscache[1]['linear']), '11':dict(Q=crosscache[k]['actual'],A=crosscache[k]['linear'])}
            for key,sk,nk in [('01',0,k-1),('10',k-1,0)]:
                p=state(sk);load(p);restore_rng_state(state(nk)['rng'])
                result=method.training_step(lb,None,3000+sk,3000+sk)
                full=grad(result.total_loss,params,allow_unused=True)
                assert all(g is not None and torch.isfinite(g).all() for g in full)
                nn=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2);clipped=torch._foreach_mul(full,(1/(nn+1e-6)).clamp(max=1.))
                opt=p['trainer']['optimizer_state'];group=opt['param_groups'][0]
                with torch.no_grad():d,_,_,_,_=candidate(params,clipped,[opt['state'][i] for i in group['params']],group)
                cache=crosscache[1 if sk==0 else k]
                a=[dot([None if g is None else g.cuda() for g in grads],d) for grads in cache['gf']]
                candidate_receipt('state_noise_cross',anchor=k,cell=key)
                before=weights(p)
                with torch.no_grad():
                    for param,base,dd in zip(params,before,d):param.copy_(base+dd)
                    fnew=[float(v) for v in values(logits(model,image),y)]
                cells[key]=dict(Q=[v-b for v,b in zip(fnew,cache['base'])],A=a,full_gradient_norm=float(nn))
            for cell in cells.values():cell['R']=[q-a for q,a in zip(cell['Q'],cell['A'])]
            contrast={}
            for term in ['Q','A','R']:
                v={key:cell[term] for key,cell in cells.items()}
                contrast[term]={name:expr for name,expr in [('total',[b-a for a,b in zip(v['00'],v['11'])]),('state',[b-a for a,b in zip(v['00'],v['10'])]),('noise',[b-a for a,b in zip(v['00'],v['01'])]),('interaction',[d-b-c+a for a,b,c,d in zip(v['00'],v['10'],v['01'],v['11'])])]}
                assert all(abs(t-s-n-i)<1e-10 for t,s,n,i in zip(contrast[term]['total'],contrast[term]['state'],contrast[term]['noise'],contrast[term]['interaction']))
            cross.append(dict(anchor=k,cells=cells,contrast=contrast))
        write(root/'CROSS.json',cross);write(root/'METRICS.json',metricrows)
        aggregates={}
        for label,rows in [('step1',closure[:1]),('steps2_25',closure[1:]),('steps1_25',closure)]:
            aggregates[label]={}
            for term in ['actual','linear','remainder']:
                aggregates[label][term]=dict(net=[sum(r[term][i] for r in rows) for i in [0,1]],harm=[sum(max(r[term][i],0) for r in rows) for i in [0,1]],benefit=[-sum(min(r[term][i],0) for r in rows) for i in [0,1]])
        for i,key in enumerate(['ce','class2_ce']):assert abs(aggregates['steps1_25']['actual']['net'][i]-(metricrows[-1][key]-metricrows[0][key]))<1e-10
        assert costs==dict(optimizer=27,main=25,golden=2,candidates=14,extra_reverse=104)
        ledger=[json.loads(v) for v in (root/'physical_updates.jsonl').read_text().splitlines()]
        assert [v['physical'] for v in ledger]==list(range(1,28))
        assert len((root/'attempts.jsonl').read_text().splitlines())==27
        assert len((root/'reverse_calls.jsonl').read_text().splitlines())==104
        assert len((root/'candidate_evaluations.jsonl').read_text().splitlines())==14
        result=dict(status='COMPLETE',costs=costs,started=started,finished=time.time(),aggregates=aggregates,metric_initial=metricrows[0],metric_final=metricrows[-1],candidate_max_error=max(r['candidate_actual_max_error'] for r in closure),margin_max_closure=max(r['margin_closure_max'] for r in closure),historical_exact=True,golden_exact=True,original_ledger_unchanged=len((origin/'physical_updates.jsonl').read_text().splitlines())==1204)
        write(root/'RESULTS.json',result);status('COMPLETE');write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active and 'method' in locals():save(root/'failure_partial.pt',cpu(snapshot(method,trainer,active['cursor'])))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));status('FAILED');write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        torch.nn.utils.clip_grad_norm_=clip_original
        for h in handles:h.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
