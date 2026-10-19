"""One original gradient, pure-tensor Adam candidates; no state application."""
import copy,json,os,time,traceback,math,inspect
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.optim.optimizer import _default_to_fused_or_foreach
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import restore_rng_state
from care_seg_study_v1.run import write
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import same


def norm(xs):return math.sqrt(sum(float(x.double().square().sum()) for x in xs if x is not None))
def dot(xs,ys):return sum(float((x.double()*y.double()).sum()) for x,y in zip(xs,ys) if x is not None and y is not None)
def describe(x):
    a=x.detach().float().flatten()
    if not a.numel():return dict(count=0)
    return dict(count=a.numel(),mean=float(a.mean()),quantiles=[float(v) for v in torch.quantile(a,torch.tensor([.05,.25,.5,.75,.95],device=a.device))],negative_fraction=float((a<0).float().mean()))


def candidate(params,clipped,states,group):
    assert group['weight_decay']==0 and not any(group[k] for k in ['amsgrad','maximize','capturable','differentiable'])
    fused,foreach=_default_to_fused_or_foreach(params,False,use_fused=False)
    assert not fused and foreach and group['foreach'] is None and group['fused'] is None
    beta1,beta2=group['betas'];rate=group['lr'];eps=group['eps']
    oldm=[s['exp_avg'].to(p.device) for s,p in zip(states,params)];oldv=[s['exp_avg_sq'].to(p.device) for s,p in zip(states,params)]
    steps=[float(s['step'])+1 for s in states]
    m=torch._foreach_lerp(oldm,clipped,1-beta1)
    v=torch._foreach_addcmul(torch._foreach_mul(oldv,beta2),clipped,clipped,1-beta2)
    bc1=[1-beta1**t for t in steps];bc2sqrt=[math.sqrt(1-beta2**t) for t in steps]
    denominator=torch._foreach_add(torch._foreach_div(torch._foreach_sqrt(v),bc2sqrt),eps)
    sizes=[-rate/b for b in bc1]
    values=torch._foreach_addcdiv([p.detach() for p in params],m,denominator,sizes)
    delta=[new-p.detach() for new,p in zip(values,params)]
    history=[size*beta1*a/d for size,a,d in zip(sizes,oldm,denominator)]
    current=[size*(1-beta1)*g/d for size,g,d in zip(sizes,clipped,denominator)]
    unrounded=[size*a/d for size,a,d in zip(sizes,m,denominator)]
    differences=[]
    for p,g,s,t,value in zip(params,clipped,states,steps,values):
        mm=torch.lerp(s['exp_avg'].to(p.device),g,1-beta1)
        vv=torch.addcmul(s['exp_avg_sq'].to(p.device)*beta2,g,g,value=1-beta2)
        dd=vv.sqrt()/math.sqrt(1-beta2**t)+eps
        separate=torch.addcdiv(p.detach(),mm,dd,value=-rate/(1-beta1**t))
        differences.append(float((value-separate).abs().max()))
    assert max(differences)<=1e-7
    return delta,history,current,unrounded,dict(dispatch='CUDA foreach, noncapturable, nonfused',torch_version=torch.__version__,old_steps=sorted({int(s['step']) for s in states}),next_steps=sorted({int(t) for t in steps}),
                                             lr=rate,betas=group['betas'],eps=eps,scalar_vs_foreach_max_candidate_difference=max(differences),
                                             rounding_delta_norm=norm([a-b for a,b in zip(delta,unrounded)]),formula_decomposition_residual=norm([a-b-c for a,b,c in zip(unrounded,history,current)]),
                                             verification_scope='pure scalar vs foreach formula; no optimizer.step comparison or model application')


def run(settings):
    root=Path(settings['output']);root.mkdir(exist_ok=False);started=time.time()
    def forbid(*args,**kwargs):raise AssertionError('Optimizer invocation forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(settings['origin']);payload=torch.load(origin/'J'/'initial.pt',map_location='cpu',weights_only=False);before=copy.deepcopy(payload)
        assert payload['cursor']==3000 and payload['population']==dict(labeled=2,unlabeled=0)
        ds=H5LabeledDataset(settings['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        selection=json.loads((origin/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in selection['train_indices']]).to('cuda')
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=Model().cuda();model.load_state_dict(payload['model']);model.train()
        method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600)
        assert all(p.requires_grad for p in model.parameters());named=list(model.named_parameters());params=[p for _,p in named];assert len(params)==100
        stash={};hook=model.logit_encoder.register_forward_hook(lambda mod,inp,out:stash.update(raw=out))
        original=model.log_joint
        def capture(image,h,label=None,**kwargs):
            out=original(image,h,label,**kwargs);stash.update(terms=out[1],ell=h['ell']);return out
        model.log_joint=capture
        restore_rng_state(payload['rng']);result=method.training_step(lb,None,3000,3000);hook.remove()
        mu=stash['raw'].chunk(2,1)[0]
        training_ce=-stash['terms']['labels'].mean()/128**2
        z=torch.cat([torch.zeros_like(mu[:,:1]),mu],1)
        mean_ce=F.cross_entropy(z,y,reduction='none').mean()
        reverse=0
        def grad(*args,**kwargs):
            nonlocal reverse
            reverse+=1;assert reverse<=8
            return torch.autograd.grad(*args,**kwargs)
        full=grad(result.total_loss,params,retain_graph=True,allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in full)
        total_norm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2)
        coefficient=(1/(total_norm+1e-6)).clamp(max=1.)
        clipped=torch._foreach_mul(full,coefficient)
        history_row=json.loads((origin/'J'/'telemetry.jsonl').read_text().splitlines()[0]);assert history_row['added_step']==1
        assert float(result.total_loss)==history_row['loss']
        assert abs(float(total_norm)-history_row['gradient_norm_before_clip'])<=1e-4
        for key in ['shape_rms','appearance_rms','semantic_feature_rms','physical_x_rms','label_ce_per_pixel','global_kl']:
            assert result.scalars[key]==history_row[key]
        optimizer=payload['trainer']['optimizer_state'];group=optimizer['param_groups'][0]
        assert group['params']==list(range(100))
        states=[optimizer['state'][i] for i in group['params']]
        with torch.no_grad():
            delta,hist,cur,unrounded,adam_info=candidate(params,clipped,states,group)
            reference=[-group['lr']*g for g in clipped]
        glabel=grad(training_ce,params,retain_graph=True,allow_unused=True);gmean=grad(mean_ce,params,retain_graph=True,allow_unused=True)
        scopes={'all':list(range(100)),'semantic':[i for i,(n,p) in enumerate(named) if semantic(n)]}
        directions={'adam_candidate':delta,'clipped_gradient_reference':reference}
        direction_stats={}
        for name,d in directions.items():
            row={}
            for scope,ix in scopes.items():
                vec=[d[i] for i in ix];g=[full[i] for i in ix];nn=norm(vec)
                row[scope]=dict(displacement_norm=nn,cosine_with_negative_gradient=-dot(g,vec)/(norm(g)*nn),
                               label_derivative=dot([glabel[i] for i in ix],vec),mean_label_derivative=dot([gmean[i] for i in ix],vec),
                               label_derivative_unit_direction=dot([glabel[i] for i in ix],vec)/nn)
            direction_stats[name]=row
        decomposition={k:dict(training_label_derivative=dot(glabel,v),mean_label_derivative=dot(gmean,v),norm=norm(v)) for k,v in [('history_moment',hist),('current_gradient',cur),('unrounded_total',unrounded)]}
        mask=y==2;foreground=torch.stack([mu[:,0],mu[:,2]],1);fgmax,fgwhich=foreground.max(1)
        competitor_fg=torch.where(fgwhich==0,torch.ones_like(y),torch.full_like(y,3))
        allwhich=z.masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf).argmax(1)
        values=dict(background=mu[:,1],foreground=mu[:,1]-fgmax,overall=mu[:,1]-z.gather(1,allwhich[:,None])[:,0])
        pad=F.pad(mask[:,None].float(),(1,1,1,1),mode='replicate')[:,0]
        interior=mask & (pad[:,1:-1,:-2].bool()) & (pad[:,1:-1,2:].bool()) & (pad[:,:-2,1:-1].bool()) & (pad[:,2:,1:-1].bool())
        subsets=dict(all_class2=mask,near_decision=mask&(values['overall'].abs()<=1),near_background=mask&(values['background'].abs()<=1),spatial_inner_boundary=mask&~interior)
        probe=torch.zeros_like(mu,requires_grad=True)
        jt=grad(mu,params,grad_outputs=probe,create_graph=True,retain_graph=True,allow_unused=True)
        margin_results={}
        for name,d in directions.items():
            product=sum((a*b.detach()).sum() for a,b in zip(jt,d) if a is not None)
            dmu=grad(product,probe,retain_graph=True)[0].detach()
            dz=torch.cat([torch.zeros_like(dmu[:,:1]),dmu],1)
            changes=dict(background=dmu[:,1],foreground=dmu[:,1]-dz.gather(1,competitor_fg[:,None])[:,0],overall=dmu[:,1]-dz.gather(1,allwhich[:,None])[:,0])
            row={}
            for subset,chosen in subsets.items():
                row[subset]={}
                for margin,value in values.items():
                    variation=changes[margin];entry=dict(initial=describe(value[chosen]),directional_change=describe(variation[chosen]),unit_displacement_mean_change=float(variation[chosen].mean())/norm(d) if chosen.any() else None)
                    entry['linear_positive_to_nonpositive_count']=int((chosen&(value>0)&(value+variation<=0)).sum())
                    row[subset][margin]=entry
            margin_results[name]=row
        assert all(p.grad is None for p in params)
        assert all(torch.equal(v.cpu(),payload['model'][k]) for k,v in model.state_dict().items()) and same(before,payload)
        assert len((origin/'physical_updates.jsonl').read_text().splitlines())==1204
        receipt=dict(status='COMPLETE',optimizer_updates=0,optimizer_constructed=False,parameter_application=False,code_adaptation=False,model_and_loaded_checkpoint_state_unchanged=True,
                     original_first_loss_reproduced=True,original_first_norm_reproduced=True,loss=float(result.total_loss),training_ce=float(training_ce),mean_ce=float(mean_ce),
                     full_parameter_count=100,original_global_norm=float(total_norm),clip_coefficient=float(coefficient),clipped_norm=norm(clipped),adam=adam_info,
                     direction_stats=direction_stats,moment_decomposition=decomposition,margins=margin_results,
                     reverse_calls=reverse,model_calls=model.calls,started=started,finished=time.time())
        write(root/'RESULTS.json',receipt);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(optimizer_updates=0,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
