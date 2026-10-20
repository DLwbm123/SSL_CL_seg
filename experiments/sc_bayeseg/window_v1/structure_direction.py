"""Read-only F data/KL gradient decomposition and unapplied Adam directions."""
import copy,json,math,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.run import same
from sc_bayeseg.model import sar,student_log_prob,normal_log_prob
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.factorized_p1 import entropy
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.adam_audit_v1.run import candidate,norm,dot


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,pure_Adam=0,density=0,Theta=0,structure_forward=0,optimizer=0,parameter_applications=0,candidate_forward=0)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        p1=Path(s['factorized_p1']);p2=Path(s['factorized_p2']);old=Path(s['structure_only']);components=Path(s['components'])
        assert json.loads((components/'EXIT.json').read_text())['exit_code']==0
        paths=[old/'RESULTS.json',old/'physical_updates.jsonl',p1/'RESULTS.json',p2/'RESULTS.json',components/'RESULTS.json'];history={p:p.read_text() for p in paths}
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(sel['train_indices'])==2
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None);lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();N=128**2
        labels=dict(GT=gt,BG=torch.zeros_like(gt),CROSS=gt.flip(0),SHIFT=torch.roll(gt,(32,32),(-2,-1)),DELETE2=torch.where(gt==2,0,gt));qs={};qrefs={}
        for name in labels:
            f=(p1 if name in ('GT','BG') else p2)/f'B_init0_{name}/S0200.pt';q=torch.load(f,map_location='cpu',weights_only=False)['q'];qrefs[f]=q;qs[name]=tuple(v.to('cuda') for v in q)
        model=SpatialResidualModel().cuda();named=list(model.named_parameters());ix=[i for i,(n,p) in enumerate(named) if n.startswith('structure.')];params=[named[i][1] for i in ix];assert len(named)==100 and len(params)==12
        for n,p in named:p.requires_grad_(n.startswith('structure.'))
        scopes=dict(mean=[j for j,i in enumerate(ix) if '.mean_' in named[i][0]],sd=[j for j,i in enumerate(ix) if '.logsd_' in named[i][0]],all=list(range(12)));initial_rng=capture_rng_state();rows=[];vectors={}
        first_loss=json.loads((old/'steps.jsonl').read_text().splitlines()[0])['loss'];replay_error=None
        for state,file in [('source',old/'S0000.pt'),('trained',old/'S0200.pt')]:
            payload=torch.load(file,map_location='cpu',weights_only=False);before=copy.deepcopy(payload);model.load_state_dict(payload['model']);restore_rng_state(payload['rng'])
            op=payload['trainer']['optimizer_state'];group=op['param_groups'][0];assert group['params']==list(range(100))
            for stream in (0,1):
                values={n:[] for n in labels}
                for _ in range(2):
                    model.draw();costs['Theta']+=1
                    try:
                        fz={n:model.structure_mean(F.one_hot(z,4).permute(0,3,1,2).to(image.dtype)) for n,z in labels.items()};costs['structure_forward']+=5
                        for _ in range(4):
                            e1=torch.randn_like(qs['GT'][0]);e2=torch.randn_like(qs['GT'][1])
                            for n,q in qs.items():
                                mx,mm,lx,lm,b=q;x=mx+lx.exp()*e1;m=mm+b*e1+lm.exp()*e2
                                terms=[student_log_prob(image-x-m,.1),student_log_prob(sar(x-fz[n]),.2)+model.logdet,normal_log_prob(m,torch.zeros_like(m),torch.full_like(m,math.log(.5))),entropy(q)]
                                values[n].append(sum(terms)/N);costs['density']+=1;append(root/'density_calls.jsonl',dict(index=costs['density'],state=state,stream=stream,candidate=n))
                    finally:model.clear_draw()
                b={n:torch.stack(a).mean(0) for n,a in values.items()};data=-b['GT'].mean();kl=model.kl()/(2*N);objective=data+kl
                targets=dict(total=objective,data=data,KL=kl,Gamma=(b['GT']-b['CROSS']).mean(),BG=(b['GT']-b['BG']).mean(),SHIFT=(b['GT']-b['SHIFT']).mean(),DELETE2=(b['GT']-b['DELETE2']).mean())
                if state=='source' and stream==0:replay_error=abs(float(objective.detach())-first_loss);assert replay_error<=1e-6
                gradients={}
                for j,(n,value) in enumerate(targets.items()):
                    gradients[n]=torch.autograd.grad(value,params,retain_graph=j<len(targets)-1);assert all(torch.isfinite(v).all() for v in gradients[n])
                    costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],state=state,stream=stream,target=n))
                gj=gradients['total'];residual=norm([a-b-c for a,b,c in zip(gj,gradients['data'],gradients['KL'])])/(norm(gj)+1e-30);assert residual<1e-5
                nn=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gj,2)),2);coef=(1/(nn+1e-6)).clamp(max=1.);clipped=torch._foreach_mul(gj,coef)
                with torch.no_grad():delta,hist,cur,unrounded,meta=candidate(params,clipped,[op['state'][i] for i in ix],group)
                costs['pure_Adam']+=1;append(root/'adam_candidates.jsonl',dict(index=costs['pure_Adam'],state=state,stream=stream,applied=False))
                directions={f'negative_{n}':[-g for g in gradients[n]] for n in ('total','data','KL')};directions['Adam']=delta;report={}
                for scope,ids in scopes.items():
                    cut=lambda vs:[vs[i] for i in ids]
                    a=cut(gradients['data']);k=cut(gradients['KL']);entry=dict(norms={n:norm(cut(g)) for n,g in gradients.items()},data_KL_cosine=dot(a,k)/(norm(a)*norm(k)+1e-30),directions={})
                    for name,d in directions.items():
                        dd=cut(d);size=norm(dd);deriv={n:dot(cut(g),dd) for n,g in gradients.items()};entry['directions'][name]=dict(norm=size,derivatives=deriv,unit_derivatives={n:v/(size+1e-30) for n,v in deriv.items()})
                    report[scope]=entry
                row=dict(state=state,stream=stream,values={n:float(v.detach()) for n,v in targets.items()},gradient_decomposition_relative_error=residual,clip_coefficient=float(coef),scopes=report,Adam=meta);rows.append(row)
                vectors[f'{state}_{stream}']=dict(gradients={k:[x.detach().cpu() for x in g] for k,g in gradients.items()},Adam_delta=[x.detach().cpu() for x in delta])
                write(root/'STATUS.json',dict(status='RUNNING',state=state,stream=stream,costs=costs));assert same(model.state_dict(),payload['model']) and all(p.grad is None for n,p in named) and same(before,payload)
        restore_rng_state(initial_rng);assert same(initial_rng,capture_rng_state())
        for f,text in history.items():assert f.read_text()==text
        for f,q in qrefs.items():assert same(q,torch.load(f,map_location='cpu',weights_only=False)['q'])
        assert costs==dict(reverse=28,pure_Adam=4,density=160,Theta=8,structure_forward=40,optimizer=0,parameter_applications=0,candidate_forward=0)
        storage(s);save(root/'private_vectors.pt',vectors)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,first_training_loss_replay_error=replay_error,rows=rows,model_payload_RNG_grad_history_unchanged=True,space=storage(s)));write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED',costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
