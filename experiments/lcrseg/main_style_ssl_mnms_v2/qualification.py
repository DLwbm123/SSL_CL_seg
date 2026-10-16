"""Bounded GPU engineering only: 110 maximum Adam calls, no formal/val/test."""
import copy,json,os,time,traceback,random
from pathlib import Path
import torch
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write,append
from ..five_frameworks_v1.semantics import binding
from .protocol import contract,VERSION,ready_identity
from .runtime import bootstrap_source,setup,seal,validate_predecessor
from .checkpoint import save,restore,same_tree


def state(t):
    return dict(student=copy.deepcopy(t.model.state_dict()),ema=copy.deepcopy(t.ema.state_dict()),
        optimizer=copy.deepcopy(t.optimizer.state_dict()),scheduler=copy.deepcopy(t.scheduler.state_dict()),
        scaler=copy.deepcopy(t.scaler.state_dict()),bank=[t.prototypes.values.clone(),t.prototypes.support.clone()],
        step=t.step,cursor=t.cursor,history=t.history.state(),precision=t.precision_counts.clone(),
        guard=copy.deepcopy(t.guard_totals),last=copy.deepcopy(t.last),telemetry=copy.deepcopy(t.telemetry),
        reads=[t.provider.l_reads,t.provider.u_reads],python_rng=random.getstate(),torch_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state_all())


def route(t,node,mode):
    original=t.losses
    def routed():
        step=t.step
        if mode=='full_U':t.step=node['steps']//2+step
        elif mode=='L_only':t.step=0
        try:return original()
        finally:t.step=step
    t.losses=routed


def main():
    c=json.loads(Path(os.environ['TASK_CONFIG']).read_text());out=Path(c['output'])
    if not c.get('engineering_only') or c.get('maximum_optimizer_calls')!=110 or not any(p.name.startswith('main_style_ssl_mnms_v2_engineering_') for p in out.parents):raise PermissionError('bounded isolated engineering authority required')
    out.mkdir(exist_ok=False);started=time.time();calls=0;accesses={'train_labeled':0,'train_unlabeled':0};slices=dict(accesses);checks={};result=dict(version=VERSION,status='RUNNING',engineering_only=True,formal_calls=0,independent_test_access=False,ablation=False,ready_identity=ready_identity(),checks=checks)
    original_step=torch.optim.Adam.step;original_read=Accessor.read
    def adam(self,*args,**kwargs):
        nonlocal calls
        if calls>=110:raise PermissionError('engineering optimizer cap reached')
        try:return original_step(self,*args,**kwargs)
        finally:
            calls+=1;append(out/'physical_updates.jsonl',dict(engineering_call_index=calls,optimizer_executed=True,formal=False))
    def read(self,*args,**kwargs):
        if self.role not in accesses:raise PermissionError('qualification forbids validation/test reads')
        accesses[self.role]+=1
        item=original_read(self,*args,**kwargs)
        z=args[1] if len(args)>1 else kwargs.get('slice_index')
        slices[self.role]+=len(item['image']) if z is None else 1
        return item
    torch.optim.Adam.step=adam;Accessor.read=read
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        torch.cuda.reset_peak_memory_stats();seed=c['seed'];nodes=[n for n in contract()['nodes'] if n['seed']==seed]
        source=nodes[0];source_out=Path(c['source_cache'])/f's{seed}.pt'
        def progress(row):print(json.dumps(dict(kind='history_collector',seed=seed,**row)),flush=True)
        if not source_out.exists():bootstrap_source(Path(c['source_root'])/'jobs'/source['id']/'sealed.pt',source,c['data'],c['reference'],'cuda',source_out,engineering=True,progress=progress)
        else:result['source_summary_reused']=True
        src=torch.load(source_out,map_location='cpu',weights_only=False);assert src['new_optimizer_updates']==0 and src['collector_cost']['L_batch_reads']==32
        checks['source_summary_32_fixed_training_L_batches']=True
        predecessor=source_out;stage_summaries=[]
        for node in nodes[1:]:
            factory=lambda initialize=True:setup(node,c['data'],c['reference'],'cuda',predecessor,initialize=initialize,engineering=True)
            t=factory();initial=copy.deepcopy(t.model.entry_anchor.state_dict());stem=copy.deepcopy(t.model.parent.adapters[0].layer.state_dict())
            assert t.precision_counts[0].div(t.precision_counts[1]).eq(.5).all()
            if node['stage']==1:
                t.update();assert not t.last['active_U'];route(t,node,'full_U');grads=t.update()
                assert t.last['active_U'] and same_tree(initial,t.model.entry_anchor.state_dict()) and same_tree(stem,t.model.parent.adapters[0].layer.state_dict())
                for token in ('.A','.B','sidecar.r'):
                    selected=[grads[id(p)] for name,p in t.model.named_parameters() if token in name and p.requires_grad]
                    assert any(g is not None and g.norm()>0 for d in selected for g in d.values()),token
                checks['GPU_L_U_backward_native_stem_entry_EMA']=True
                identity={k:node[k] for k in ('id','dataset','seed','order','stage','domain')};save(t,out/'resume.pt',identity)
                # Invalid history must fail before even RNG, provider or live tensors change.
                payload=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
                payload['mnms_v2']['history'][0]['anchor']['__F__'][0,0]+=.1
                bad=out/'tampered_history.pt';torch.save(payload,bad);before=state(t)
                try:restore(t,bad,identity)
                except ValueError:pass
                else:raise AssertionError('tampered memory accepted')
                assert same_tree(before,state(t));checks['tamper_refused_before_mutation']=True
                t.update();expected=state(t);del t,grads,payload,before;torch.cuda.empty_cache()
                t=factory(False);restore(t,out/'resume.pt',identity);route(t,node,'full_U');t.update()
                assert same_tree(expected,state(t)),'GPU exact next-update resume mismatch'
                checks['full_GPU_Adam_scheduler_RNG_cursor_bank_history_resume']=True
                del expected
            elif node['stage']==2:
                t.update();route(t,node,'full_U');t.update()
            else:
                route(t,node,'full_U');t.update()
                assert len(t.last['elastic_guard']['proxy_accepted'])==12
                checks['three_historical_domains_12_separate_class_terms']=True
            guard=t.last['elastic_guard'];assert guard['converged'] and guard['spectral_backtracks']==0
            if guard['continuous_retention'] is not None:assert guard['continuous_retention']>=.5-1e-9
            assert t.model.sidecar.valid() and t.ema.sidecar.valid()
            baseline=state(t);destination=out/(node['domain']+'_engineering_sealed.pt')
            seal(t,node,destination,engineering=True,progress=progress)
            assert same_tree(baseline,state(t)),'collector changed training state/RNG'
            child=torch.load(destination,map_location='cpu',weights_only=False)
            assert same_tree(child['memory'][:-1],t.history.state()) and child['collector_cost']['L_batch_reads']==32
            stage_summaries.append(dict(domain=node['domain'],engineering_endpoint_step=t.step,history_domains=[s['domain'] for s in child['memory']],guard=guard,collector_cost=child['collector_cost'],class_support=[v['support'] for v in child['memory'][-1]['classes']]))
            predecessor=destination;del t,baseline,child;torch.cuda.empty_cache()
        checks['all_stage_fixed_L_collectors_and_immutable_handoff']=True
        result['stages']=stage_summaries
        if seed==163:
            node=nodes[1];fit=setup(node,c['data'],c['reference'],'cuda',source_out,engineering=True)
            access=fit.provider._l;xs=[];ys=[]
            for patient in access.patients[:2]:
                index=access.by_patient[patient][0];volume=access.read(index);z=int((volume['label']>0).flatten(1).sum(1).argmax());item=access.read(index,z);xs.append(item['image']);ys.append(item['label'])
            x=torch.stack(xs).cuda();y=torch.stack(ys).cuda();provider=fit.provider
            def fixed(step,stream='labeled'):provider.l_reads+=1;return x.clone(),y.clone(),('fit_patient1','fit_patient2')
            provider.labeled=fixed;route(fit,node,'L_only');losses=[]
            for i in range(96):
                fit.update();losses.append(fit.last['labeled_loss'])
                if i%16==0:print(json.dumps(dict(kind='two_case_L_fit',step=i+1,loss=losses[-1],optimizer_calls=calls)),flush=True)
            before=sum(losses[:4])/4;after=sum(losses[-4:])/4
            result['two_case_L_fit']=dict(steps=96,initial_loss=before,final_loss=after,U_reads=provider.u_reads,threshold='final < 0.8 initial; inherited engineering rule')
            assert after<.8*before and provider.u_reads==0
            checks['two_case_training_L_overfit']=True
        assert calls==(103 if seed==163 else 7)
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        torch.optim.Adam.step=original_step;Accessor.read=original_read
        result.update(optimizer_calls=calls,image_read_operations=accesses,successful_image_slice_reads=slices,elapsed_seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated())
        write(out/'RESULT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
