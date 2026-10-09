"""Frozen ENTRY100 current-label cross-image prototype diagnosis; no optimization."""
import fcntl
import importlib.util
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def main(root,cfg):
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=Path(cfg['v121']);assert json.loads((old/'COMPLETION_AUDIT.json').read_text())['status']=='PASS'
    for n in ('FINAL_PUBLICATION.json','POSTHOC_PUBLICATION.json'):assert json.loads((old/n).read_text())['anonymous_http']=='200'
    Q=load('quarter',cfg['quarter_entry']);Q.configure(cfg,root);N=Q.N;c=N.c;b=N.b;N.STAGE='V122'
    S=load('selector',cfg['selector_entry']);P=load('target',cfg['target_entry']);M=load('metrics',cfg['metrics_entry'])
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(119)
    original=N.E.install_actions(c);N.E.action_check(c,original)
    roles=c.split_roles(cfg['data']);assert roles==N.D.read(Path(cfg['action_root'])/'ROLES.private.json')
    counts=Counter();ledger=b.JobLedger(root,{})
    def deny(*a,**kw):raise PermissionError('V122 forbids optimizer updates')
    c.Trainer.update=deny;torch.optim.Adam.step=deny;torch.optim.SGD.step=deny
    sample=c.e.primitive.CurrentData.__getitem__;allowed=set()
    def guarded(ds,i):
        assert ds.role=='train_labeled' and ds.rows[i]['case_id'] in allowed
        counts['image_attempts']+=1
        value=sample(ds,i);counts['image_success']+=1;return value
    c.e.primitive.CurrentData.__getitem__=guarded
    rows=[];private=[]
    for i,ctx in enumerate(b.contexts()):
        t=b.make(cfg,roles,ledger,ctx,300);t.provider.seed=10168
        entry=torch.load(Path(cfg['v116'])/f'jobs/learn601/GROUP_{i:02d}_ENTRY.private.pt',map_location='cpu',weights_only=False)['student']
        c.restore(t,entry);assert t.step==100
        ds=t.provider._l;n=len(ds);assert n==ctx[1] and n in (2,8);allowed=set(r['case_id'] for r in ds.rows)
        before={name:c.e.cpu(model.state_dict()) for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]}
        cache=[];captured={}
        def capture(module,args):captured['feature']=args[0].detach()
        handle=t.ema.parent.native.decoder.conv_logit.register_forward_pre_hook(capture)
        def forward(kind,fn):
            counts['forward_attempts']+=1;ordinal=counts['forward_attempts'];record=dict(ordinal=ordinal,context=i,image_index=j,kind=kind)
            c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='attempt',**record))
            try:out=fn()
            except BaseException:c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='failure',**record));raise
            c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='success',**record));counts['forward_success']+=1;counts[kind]+=1
            return out
        with torch.no_grad():
            for j in range(n):
                item=ds[j];x=c.photo(item['image'][None].cuda(),ctx[2]);y=item['label'][None].cuda();valid=item['geometry'][None].cuda()&(y!=255)
                q=forward('ema',lambda:t.ema(x,mode='teacher').softmax(1));h=t.ema.parent.feature_to_output(captured['feature'],q.shape[-2:]).cpu()
                qm=forward('memory',lambda:t.memory(x,mode='teacher').softmax(1));qf=forward('memory_flip',lambda:t.memory(x.flip(-1),mode='teacher').softmax(1).flip(-1))
                p=forward('student',lambda:t.clean(x))
                _,_,target=c.transfer_loss(p,q,valid,11,qm,qf)
                features,eligible,classes=S.features(p,q,qm,qf,target,valid,100);mask,selection=S.choose(features,eligible,classes,'COVERAGE',None,None)
                assert selection[0]['selected']==selection[0]['eligible']//2
                memory_gate=valid&(qm.max(1).values>.7)&(qm.argmax(1)==qf.argmax(1))
                cache.append(dict(feature=h,label=y.cpu(),target=target.cpu(),q=q.cpu(),mask=mask,valid=valid.cpu(),memory_gate=memory_gate.cpu()))
            for j,v in enumerate(cache):
                support=[(j+k)%n for k in range(1,min(2,n-1)+1)];assert j not in support and len(set(support))==len(support)
                assert ds.rows[j]['case_id'] not in {ds.rows[k]['case_id'] for k in support}
                lf=torch.cat([cache[k]['feature'] for k in support]);ly=torch.cat([cache[k]['label'] for k in support])
                chosen,pixel_support,present=P.assignments(lf,ly,v['feature'],v['target'].argmax(1))
                centers=torch.stack([(F.normalize(lf.float(),dim=1,eps=1e-8)*(ly==k)[:,None]).sum((0,2,3))/max(pixel_support[k],1) for k in range(3)])
                norms=centers.norm(dim=1).tolist()
                target=P.permute(v['target'],chosen,v['mask']);assert torch.equal(target.sort(1).values,v['target'].sort(1).values)
                assert torch.equal(target.masked_select(~v['mask'][:,None]),v['target'].masked_select(~v['mask'][:,None]))
                qcls=v['q'].argmax(1);weight=v['q'].new_tensor([.5,1.,1.5])[qcls]
                assert all(x>0 for x in pixel_support) and all(x>1e-8 for x in norms),'fixed prototype requires all supported classes'
                prototype_scores=torch.einsum('bchw,kc->bkhw',F.normalize(v['feature'].float(),dim=1,eps=1e-8),F.normalize(centers,dim=1,eps=1e-8))
                outputs,control_counts=M.controls(v['target'],v['q'],v['mask'],prototype_scores,P.permute,122000+100*i+j)
                detail=[]
                for scope,value in outputs.items():
                    rr=dict(context=i,source_step=ctx[0],labeled_images=n,scope=scope,**M.measure(v['target'],value,v['label'],v['mask'],weight));detail.append(rr)
                private.append(dict(context=i,image_index=j,support_indices=support,support_pixel_counts=pixel_support,prototype_norms=norms,present=present,control_counts=control_counts,metrics=detail))
                counts['held_out_images']+=1;counts['prototype_assignments']+=1
        handle.remove()
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:assert c.e.same(before[name],model.state_dict()),name
        assert c.e.same(entry['optimizer'],t.optimizer.state_dict()) and not ledger.count and t.step==100
        for scope in outputs:
            group=[rr for v in private if v['context']==i for rr in v['metrics'] if rr['scope']==scope];meta={k:group[0][k] for k in ('context','source_step','labeled_images','scope')}
            sums={k:sum(r[k] for r in group) for k in group[0] if k not in meta};row=dict(meta,**sums);rows.append(dict(row,**M.rates(row)))
        N.B.write(root/'STATUS.json',dict(status='RUNNING',completed_contexts=i+1,counts=dict(counts),time=time.time()))
        del t,cache,before;torch.cuda.empty_cache()
    assert counts['image_attempts']==counts['image_success']==counts['held_out_images']==counts['prototype_assignments']==40
    assert counts['forward_attempts']==counts['forward_success']==160 and all(counts[k]==40 for k in ('ema','memory','memory_flip','student'))
    assert len(rows)==40 and len(private)==40 and not ledger.count
    write(root/'DETAILS.private.json',private);write(root/'RESULTS.json',rows);N.D.table(root/'RESULTS.csv',rows)
    reference=json.loads((old/'RESULTS.json').read_text())
    for i in range(8):
        prior=next(r for r in reference if r['context']==i and r['scope']=='BASE')
        current=next(r for r in rows if r['context']==i and r['scope']=='BASE')
        assert current==prior,'unchanged BASE must reproduce V121'
    summary=[dict(context=i,**{k:sum(v['control_counts'][k] for v in private if v['context']==i) for k in private[0]['control_counts']}) for i in range(8)]
    write(root/'CONTROL_SUMMARY.json',summary)
    write(root/'REFERENCE_CHECK.json',dict(status='PASS',BASE_matches_V121=True,new_inference_charged=True))
    write(root/'DECISION.json',M.analyze(rows))
    write(root/'COSTS.json',dict(counts=dict(counts),native_optimizer_updates=0,actor_optimizer_updates=0,linear_solves=0,Q_train_query_calls=0,Q_dev_query_calls=0,U_images_read=0,new_annotation_cases=0,current_labeled_diagnostic_images=40,model_image_forwards=160))
    write(root/'QUALIFICATION.json',dict(status='PASS',all_support_images_disjoint=True,all_probability_multisets_exact=True,all_unselected_targets_unchanged=True,all_model_and_optimizer_states_unchanged=True,all_half_pixel_budgets_exact=True,held_out_labels_used_only_for_validity_and_scoring=True))
    write(root/'FINAL.json',dict(status='COMPLETE',time=time.time()));N.B.write(root/'STATUS.json',dict(status='COMPLETE',time=time.time()))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text())
    if os.environ.get('EXEC_SELFCHECK')=='1':
        M=load('metrics',cfg['metrics_entry']);P=load('target',cfg['target_entry']);result=M.selfcheck(P.permute);P.selfcheck();print(json.dumps(result))
    else:
        write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:main(root,cfg)
        except BaseException as exc:write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
