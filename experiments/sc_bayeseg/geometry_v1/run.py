"""Twenty reverses, four unapplied Adam tensors; no optimizer or candidate forwards."""
import copy,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import same
from sc_bayeseg.adam_audit_v1.run import candidate,norm
from sc_bayeseg.finite_replay_v1.run import logits,cpu
from sc_bayeseg.geometry_v1.geometry import solve,selfcheck,unit,row_units


def losses(model,image,y):
    rng=capture_rng_state()
    try:
        probs=[logits(model,image).softmax(1)];seed_everything(50163);sampled=[]
        for _ in range(4):
            h,_=model.proposal(image);z=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1);sampled.append(z.softmax(1))
        probs.append(torch.stack(sampled).mean(0));vals=[]
        for p in probs:
            loss=-p.clamp_min(1e-30).log().gather(1,y[:,None])[:,0]
            for i in range(2):
                vals.append(loss[i].mean())
                vals.extend(loss[i][y[i]==c].mean() for c in [1,2,3])
        return vals
    finally:restore_rng_state(rng)


def flat(grads,params):
    return np.concatenate([(np.zeros(p.numel(),dtype=np.float64) if g is None else g.detach().double().cpu().numpy().reshape(-1)) for g,p in zip(grads,params)])


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time();costs=dict(extra_reverse=0,adam_candidates=0,optimizer=0,candidate_forwards=0,parameter_applications=0)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    def grad(label,*a,**kw):
        assert costs['extra_reverse']<20
        costs['extra_reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['extra_reverse'],label=label))
        return torch.autograd.grad(*a,**kw)
    def status(phase):write(root/'STATUS.json',dict(phase=phase,costs=costs.copy()))
    try:
        write(root/'SOLVER_SELF_CHECK.json',selfcheck())
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(s['origin']);guard=Path(s['guard']);replay=Path(s['replay'])
        assert json.loads((guard/'EXIT.json').read_text())['exit_code']==0 and not (guard/'FAILURE.json').exists()
        sources={k:torch.load(guard/'G'/f'S{k:02}.pt',map_location='cpu',weights_only=False) for k in [0,24]}
        originals=copy.deepcopy(sources)
        assert all(torch.equal(v,sources[24]['model'][n]) for n,v in sources[0]['model'].items() if semantic(n))
        assert all(p['population']==dict(labeled=2,unlabeled=0) for p in sources.values())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        selection=json.loads((origin/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in selection['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();assert int((y==2).sum())==490
        model=Model().cuda();named=list(model.named_parameters());params=[p for _,p in named];ix=[i for i,(n,p) in enumerate(named) if semantic(n)];sp=[params[i] for i in ix]
        assert len(params)==100 and len(sp)==40
        method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600)
        names=[f'{ch}_image{i}_{c}' for ch in ['mean','stochastic'] for i in [0,1] for c in ['all','C1','C2','C3']]
        oldmetrics=[json.loads(l) for l in (guard/'G'/'metrics.jsonl').read_text().splitlines()]
        oldsteps=[json.loads(l) for l in (guard/'G'/'steps.jsonl').read_text().splitlines()]
        model.load_state_dict(sources[0]['model']);restore_rng_state(sources[0]['rng']);base=losses(model,image,y);basevalues=[float(v.detach()) for v in base]
        assert basevalues==oldmetrics[0]['vector']
        if s.get('preflight_only'):
            assert same(cpu(model.state_dict()),sources[0]['model']) and same(capture_rng_state(),sources[0]['rng'])
            write(root/'ZERO_CHECK.json',dict(status='PASS',costs=costs,original_loss_vector_equal=True,semantic_source_states_equal=True))
            write(root/'EXIT.json',dict(exit_code=0));return
        constraint_grads=[]
        for i,(name,value) in enumerate(zip(names,base)):
            gg=grad(name,value,sp,allow_unused=True,retain_graph=i<15);constraint_grads.append(flat(gg,sp))
        C=np.stack(constraint_grads);assert np.isfinite(C).all()
        model.load_state_dict(sources[24]['model']);restore_rng_state(sources[24]['rng'])
        with torch.no_grad():shared=[float(v) for v in losses(model,image,y)]
        assert shared==basevalues and shared==oldmetrics[24]['vector']
        write(root/'SHARING_AUDIT.json',dict(status='PASS',semantic_parameters_exact=True,constraint_values_both_states_exact=True,original_G_constraint_values_exact=True,
              dependency='semantic logits use only shape_encoder/logit_encoder; fixed Gaussian draw shapes and order; physical/appearance/code values do not enter ell',extra_reverse=16))
        status('CONSTRAINT_GRADIENTS_COMPLETE')
        results={};private={};semC,cn=row_units(C)
        for sk in [0,24]:
            payload=sources[sk];model.load_state_dict(payload['model']);joint=[];directions=[];objectives=[]
            for noise in [1,25]:
                restore_rng_state(torch.load(replay/f'S{noise-1:02}.pt',map_location='cpu',weights_only=False)['rng'])
                r=method.training_step(lb,None,3000+sk,3000+sk)
                full=grad(f'joint_S{sk}_xi{noise}',r.total_loss,params,allow_unused=True)
                assert all(g is not None and torch.isfinite(g).all() for g in full)
                total_norm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2);coef=(1/(total_norm+1e-6)).clamp(max=1.)
                clipped=torch._foreach_mul(full,coef)
                assert costs['adam_candidates']<4
                costs['adam_candidates']+=1;append(root/'adam_candidates.jsonl',dict(index=costs['adam_candidates'],state=sk,noise=noise,applied=False,forward_evaluated=False))
                opt=payload['trainer']['optimizer_state'];group=opt['param_groups'][0]
                assert group['params']==list(range(100))
                with torch.no_grad():d,_,_,_,info=candidate(params,clipped,[opt['state'][i] for i in group['params']],group)
                j=flat([full[i] for i in ix],sp);adam=flat([d[i] for i in ix],sp);objectives.append(j)
                directions.extend([(f'negative_joint_xi{noise}',-j),(f'adam_xi{noise}',adam)])
                comparison=None
                if (sk,noise) in [(0,1),(24,25)]:
                    row=oldsteps[sk];comparison=dict(loss_equal=float(r.total_loss)==row['loss_before'],semantic_candidate_norm_error=abs(norm([d[i] for i in ix])-row['nominal_semantic_delta_norm']))
                    assert comparison['loss_equal'] and comparison['semantic_candidate_norm_error']==0
                joint.append(dict(noise=noise,loss=float(r.total_loss),full_gradient_norm=float(total_norm),clip_coefficient=float(coef),semantic_gradient_norm=float(np.linalg.norm(j)),adam_semantic_norm=float(np.linalg.norm(adam)),original_G_comparison=comparison,adam_old_steps=info['old_steps']))
                assert all(p.grad is None for p in params) and same(cpu(model.state_dict()),payload['model'])
                restore_rng_state(payload['rng'])
            J=np.stack(objectives);solution,solution_dirs=solve(C,J)
            dirs={}
            for name,direction in directions+list(solution_dirs.items()):
                u,n=unit(direction)
                dirs[name]=dict(norm=n,constraint_raw_dot=(C@direction).tolist(),constraint_dot_per_unit_direction=(C@u).tolist(),constraint_cosines=(semC@u).tolist(),
                               joint_raw_dot=(J@direction).tolist(),joint_dot_per_unit_direction=(J@u).tolist(),semantic_only=True)
            results[str(sk)]=dict(joint=joint,directions=dirs,geometry=solution)
            private[f'joint_{sk}']=J
            for name,d in directions+list(solution_dirs.items()):private[f'S{sk}_{name}']=d
            assert same(cpu(model.state_dict()),payload['model']) and same(capture_rng_state(),payload['rng'])
            status(f'S{sk}_COMPLETE')
        private['constraint_gradients']=C
        np.savez(root/'PRIVATE_GRADIENTS_AND_DIRECTIONS.npz',**private)
        assert costs==dict(extra_reverse=20,adam_candidates=4,optimizer=0,candidate_forwards=0,parameter_applications=0)
        assert same(originals,sources) and all(p.grad is None for p in params)
        assert len((guard/'physical_updates.jsonl').read_text().splitlines())==54
        assert [v['index'] for v in map(json.loads,(root/'reverse_calls.jsonl').read_text().splitlines())]==list(range(1,21))
        assert [v['index'] for v in map(json.loads,(root/'adam_candidates.jsonl').read_text().splitlines())]==list(range(1,5))
        result=dict(status='COMPLETE',costs=costs,started=start,finished=time.time(),constraints=names,baseline_values=basevalues,constraint_norms=cn.tolist(),states=results,
                    source_payloads_unchanged=True,last_loaded_model_unchanged=True,all_parameter_grad_buffers_none=True,source_physical_ledger_unchanged=True,model_calls=model.calls)
        write(root/'RESULTS.json',result);status('COMPLETE');write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));status('FAILED');write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
