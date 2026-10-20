"""304 formal plus32 engineering physical SGD calls; source-only finite-step utility."""
import copy,json,time,traceback
from pathlib import Path
import h5py,numpy as np,torch
from . import core,runtime
from .bounded_runtime import install,ForwardBudget
from .module import unit
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write
from ..di_dmpa_gate1.binding import safe_asset
from ..five_frameworks_v1.recipes import generator
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..siemens_lonly_v1.run import dice


def optimizer(model,cfg,group,phase):
    groups=[dict(params=[p for p in model.parent.parameters() if p.requires_grad and (p.ndim>1)==matrix],lr=.001,weight_decay=.0005 if matrix else 0.) for matrix in (True,False)]
    opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True)
    runtime.attach(opt,dict(cfg,group=group),group,phase);return opt


def snapshot(model,opt):
    return copy.deepcopy(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))


def restore(state,model,opt):
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);model.zero_grad(set_to_none=True)


class Source:
    def __init__(self,cfg):
        self.access=Accessor(cfg['data'],'mnms','Siemens','train_labeled');a=self.access
        assert len(a.patients)==10 and len(a.rows)==20 and sum(a.depths)==230
        perm=torch.randperm(10,generator=torch.Generator().manual_seed(20261021)).tolist()
        self.support=[a.patients[i] for i in perm[:7]];self.query=[a.patients[i] for i in perm[7:]];assert not set(self.support)&set(self.query)
        self.query_indices=[i for i,r in enumerate(a.rows) if r['patient_id'] in self.query];self.q=sum(a.depths[i] for i in self.query_indices)
        write(Path(cfg['root'])/'PARTITION.private.json',dict(support=self.support,query=self.query,query_slices=self.q,source_prior_exposure_all10=True))
        allowed={str(safe_asset(Path(cfg['data']),r[k]).resolve()) for r in a.rows for k in ('image_h5_relpath','label_h5_relpath')}
        real=h5py.File
        def source_only(name,*args,**kw):
            if str(Path(name).resolve()) not in allowed:raise PermissionError('source diagnostic outside Siemens train-L')
            return real(name,*args,**kw)
        h5py.File=source_only;self.reads=dict(support_batches=0,query_gradient_batches=0,query_evaluations=0)
    def batch(self,step,role='support'):
        pool=self.support if role=='support' else self.query;a=self.access;g=generator(20261021,1,0,step,'source_'+role);xs=[];ys=[]
        for p in torch.randperm(len(pool),generator=g)[:2].tolist():
            ii=a.by_patient[pool[p]];i=ii[int(torch.randint(len(ii),(),generator=g))];z=int(torch.randint(a.depths[i],(),generator=g));item=a.read(i,z);xs.append(item['image']);ys.append(item['label'])
        self.reads['support_batches' if role=='support' else 'query_gradient_batches']+=1
        return torch.stack(xs).cuda(),torch.stack(ys).cuda()


def main(cfg):
    root=Path(cfg['root']);guard=install(cfg)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        data=Source(cfg);budget=ForwardBudget(cfg,dict(real_support=512,real_query=35*data.q+64,synthetic_native=704,generator=64 if cfg.get('floor_repair') else 32))
        if cfg.get('floor_repair'):
            old=json.loads((root/'forward_counter.json').read_text());assert old['pending'] is None;budget.count=old['executed']
        model=core.build(cfg);memory=torch.load(cfg['memory'],map_location='cpu',weights_only=False)
        model.projection.load_state_dict(memory['projection']);model.generator.load_state_dict(memory['generator']);model.code.data.copy_(memory['code']);model.frames.copy_(memory['frames'])
        model.parent.configure_stage_training();model.projection.requires_grad_(False).eval();model.generator.requires_grad_(False).eval();model.code.requires_grad_(False)
        fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'};assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
        names=[n for n,p in model.parent.named_parameters() if p.requires_grad];pars=[p for p in model.parent.parameters() if p.requires_grad]
        def forward(kind,x):return budget.call(kind,len(x),model,x)
        def ce_step(opt,x,y,synthetic=False):
            opt.zero_grad(set_to_none=True);v=core.soft_ce(forward('synthetic_native',x),y) if synthetic else pixel_ce(forward('real_support',x),y).mean()
            if not torch.isfinite(v):raise FloatingPointError('nonfinite task CE')
            v.backward();opt.step();return float(v.detach())
        def index(step):
            g=generator(20261021,1,0,step//4,'synthetic_code');return torch.randperm(32,generator=g)[(step%4)*8:(step%4+1)*8].cuda()
        def evaluate():
            model.eval();records=[];ce_sum=fg_sum=0.;pixels=fg_pixels=0
            with torch.no_grad():
                for i in data.query_indices:
                    item=data.access.read(i);x=item['image'];y=item['label'];pred=[]
                    for z in range(0,len(x),2):
                        yy=y[z:z+2].cuda();logits=forward('real_query',x[z:z+2,None].cuda());v=torch.nn.functional.cross_entropy(logits,yy,reduction='none');fg=yy>0
                        ce_sum+=float(v.double().sum());pixels+=v.numel();fg_sum+=float(v[fg].double().sum());fg_pixels+=int(fg.sum());pred.append(logits.argmax(1).cpu().numpy())
                    records.append(dice(np.concatenate(pred),y.numpy()))
            data.reads['query_evaluations']+=1;values=np.mean(records,axis=0);model.train();model.projection.eval();model.generator.eval()
            return dict(CE=ce_sum/pixels,foreground_CE=fg_sum/fg_pixels if fg_pixels else None,Dice=values.tolist(),mean_foreground_Dice=float(values.mean()))
        # All32 qualification updates count and are then discarded.
        opt=optimizer(model,cfg,'qualification','engineering');initial=snapshot(model,opt);assert not opt.state;x,y=data.batch(0)
        with torch.no_grad():first=float(pixel_ce(forward('real_support',x),y).mean())
        if cfg.get('floor_repair'):
            evidence=json.loads((root/'ENGINEERING_ZERO_UPDATE_CHECK.json').read_text())
            assert evidence['numerically_impossible_decrease'] and evidence['foreground_pixels']==0 and first<1e-4
            assert json.loads((root/'budget_counter.json').read_text())['by_phase']=={'qualification:engineering':32}
            v=pixel_ce(forward('real_support',x),y).mean();v.backward();gold=[p.grad.clone() for p in pars]
            restore(initial,model,opt);v2=pixel_ce(forward('real_support',x),y).mean();v2.backward()
            assert float(v)==float(v2) and all(torch.equal(p.grad,g) and torch.isfinite(g).all() for p,g in zip(pars,gold))
            write(root/'ENGINEERING.json',dict(status='PASS_WITH_RECORDED_FLOOR_REPAIR',actual_updates=32,discarded=True,initial_CE=first,CE_decrease_test='not_applicable_below_absolute_threshold',nonempty_restore_exact_passed_before_failure=True,zero_update_golden_exact=True,failed_attempt_preserved=True))
        else:
            for i in range(30):
                before=snapshot(model,opt) if i==15 else None;v=ce_step(opt,x,y)
                if i==15:
                    expected=snapshot(model,opt);restore(before,model,opt);assert ce_step(opt,x,y)==v and runtime.exact(expected,snapshot(model,opt))
            ce_step(opt,x,y)
            with torch.no_grad():last=float(pixel_ce(forward('real_support',x),y).mean())
            write(root/'ENGINEERING_MEASUREMENT.json',dict(first=first,final=last))
            assert last<first-1e-4;write(root/'ENGINEERING.json',dict(status='PASS',actual_updates=32,discarded=True,first=first,final=last,nonempty_restore_exact=True))
        restore(initial,model,opt);runtime.save(cfg,initial,root/'entry.pt')
        with torch.no_grad():images,masks=budget.call('generator',32,model.generator,unit(model.code),(384,384))
        runtime.finite([images,masks])
        # Construct one source-support trajectory, carrying its actual SGD momentum.
        opt=optimizer(model,cfg,'construction','formal');states={'s0':snapshot(model,opt)}
        for step in range(64):
            x,y=data.batch(step);ce_step(opt,x,y)
            if step+1 in (16,64):states['s'+str(step+1)]=snapshot(model,opt)
        for state,which,seed in [('sEnc','encoder',20261022),('sDec','decoder',20261023)]:
            opt=optimizer(model,cfg,'unused','formal');restore(states['s0'],model,opt);rng=torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for n,p in model.parent.named_parameters():
                    selected=n.startswith(('native.enc','native.bottleneck')) if which=='encoder' else n.startswith('native.decoder')
                    if p.requires_grad and selected:
                        noise=torch.randn(p.shape,generator=rng).to(p.device);p.add_(noise/noise.norm().clamp_min(1e-30)*(.001*p.norm()))
            states[state]=snapshot(model,opt)
        for name,state in states.items():runtime.save(cfg,state,root/(name+'.pt'))
        results={}
        for state_name,state in states.items():
            guard(cfg,new_stage=True);opt=optimizer(model,cfg,'unused','formal');restore(state,model,opt);base=evaluate()
            # No optimizer mutation: compare task directions at the same fixed state.
            qx,qy=data.batch(0,'query');sx,sy=data.batch(0);ii=index(0)
            qg=torch.autograd.grad(pixel_ce(forward('real_query',qx),qy).mean(),pars,allow_unused=False)
            mg=torch.autograd.grad(core.soft_ce(forward('synthetic_native',images[ii]),masks[ii]),pars,allow_unused=False)
            rg=torch.autograd.grad(pixel_ce(forward('real_support',sx),sy).mean(),pars,allow_unused=False)
            def cosine(aa,bb,selection):
                pairs=[(a,b) for n,a,b in zip(names,aa,bb) if (n.startswith(('native.enc','native.bottleneck')) if selection=='encoder' else n.startswith('native.decoder'))]
                dot=sum(float((a.double()*b.double()).sum()) for a,b in pairs);na=sum(float(a.double().square().sum()) for a,b in pairs);nb=sum(float(b.double().square().sum()) for a,b in pairs)
                return dot/(na*nb)**.5 if na*nb>0 else None
            directions={k:dict(M_query=cosine(mg,qg,k),R_query=cosine(rg,qg,k)) for k in ('encoder','decoder')};del qg,mg,rg
            branches={}
            for branch in ('B0','BM','BR'):
                opt=optimizer(model,cfg,state_name+'_'+branch,'formal');restore(state,model,opt);horizons={}
                for step in range(16):
                    if branch=='B0':
                        opt.zero_grad(set_to_none=True)
                        for p in pars:p.grad=torch.zeros_like(p)
                        opt.step()
                    elif branch=='BM':
                        ii=index(step);ce_step(opt,images[ii],masks[ii],True)
                    else:
                        x,y=data.batch(step);ce_step(opt,x,y)
                    if step+1 in (1,16):
                        value=evaluate();disp={}
                        for k in ('encoder','decoder'):
                            selected=[n for n in names if (n.startswith(('native.enc','native.bottleneck')) if k=='encoder' else n.startswith('native.decoder'))]
                            disp[k]=sum(float((model.parent.state_dict()[n].detach().cpu().double()-state['model']['parent.'+n].double()).square().sum()) for n in selected)**.5
                        horizons[str(step+1)]=dict(query=value,parameter_displacement=disp)
                runtime.save(cfg,dict(**snapshot(model,opt),reads=data.reads.copy(),state=state_name,branch=branch,step=16),root/(state_name+'_'+branch+'_final.pt'))
                branches[branch]=horizons
            results[state_name]=dict(baseline_query=base,directions=directions,branches=branches)
            write(root/'PARTIAL_RESULTS.json',results)
        ref=[v['branches']['B0']['16']['query']['CE']-v['branches']['BR']['16']['query']['CE'] for v in results.values()]
        utility=[v['branches']['B0']['16']['query']['CE']-v['branches']['BM']['16']['query']['CE'] for v in results.values()]
        dice_diff=np.array([np.array(v['branches']['BM']['16']['query']['Dice'])-v['branches']['B0']['16']['query']['Dice'] for v in results.values()])
        sensitive=float(np.mean(ref))>1e-5 and sum(x>1e-5 for x in ref)>=3
        gate=sum(x>1e-5 for x in utility)>=4 and dice_diff.mean()>=0 and bool((dice_diff.mean(0)>=-.005).all())
        counter=json.loads((root/'budget_counter.json').read_text());assert counter['total']==336 and counter['pending']==0 and counter['functional']==0
        assert budget.count['real_query']==35*data.q+10 and budget.count['synthetic_native']==680 and budget.count['generator']==(64 if cfg.get('floor_repair') else 32)
        write(root/'RESULTS.json',dict(status='COMPLETE',results=results,cost=counter,forward_cost=budget.count,query_slices=data.q,reads=data.reads,reference_sensitive=sensitive,memory_gate_pass=bool(gate) if sensitive else None,reference_CE_utility_h16=ref,memory_CE_utility_h16=utility,organ_Dice_mean_difference=dice_diff.mean(0).tolist(),independent_generalization=False))
        write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        if 'model' in locals() and 'opt' in locals():
            try:
                from ..five_frameworks_v1.checkpoint import atomic_save
                guard(cfg,200_000_000);atomic_save(dict(**snapshot(model,opt),dirty=True,reads=getattr(data,'reads',{})),root/'exception_state.pt')
            except BaseException:pass
        write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED'));raise
