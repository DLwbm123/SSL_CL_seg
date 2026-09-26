"""Bounded native qualification and fixed zero-update forensic readout."""
from .runner import *
from qprompt.models import semantic_probabilities
from torch.nn import functional as F


def qualify():
    setup(261);provenance();backbone=os.environ['EXEC_BACKBONE'];folder=RUN/'qualification'/backbone;folder.mkdir(parents=True,exist_ok=True);rows=[]
    for arm in ('B0','B1','B2','B3'):
        task='synthetic__'+backbone+'__'+arm;token=claim(task)
        model,init=c.build(backbone,False,r.ASSETS,r.SOURCE);model.cuda().train();opt,sch=c.optimizer_for(model,backbone)
        x=torch.rand(2,3,384,384,device='cuda');y=torch.zeros(2,384,384,device='cuda',dtype=torch.long);y[:,64:256,64:256]=1;y[:,128:224,128:224]=2
        bank=bank_init(model,[dict(image=x[i].cpu(),label=y[i].cpu()) for i in range(2)],backbone,folder) if arm in ('B1','B3') else None
        initial=folder/'initial.pt';c.atomic(initial,c.snapshot(model,opt,sch,None,bank,0,**provenance()),binary=True);torch.cuda.reset_peak_memory_stats()
        # Actual one-step equivalence after restoring identical pre-update state, including RNG.
        first=update(model,opt,sch,bank,x,y,arm,task,token,0,folder,kind='synthetic',diagnose=False)
        expected_path=folder/'expected.pt';c.atomic(expected_path,c.snapshot(model,opt,sch,None,bank,1),binary=True)
        state=torch.load(initial,map_location='cpu',weights_only=False);c.restore(state,model,opt,sch,None,bank)
        assert c.tensor_sha(model.state_dict())==c.tensor_sha(state['student']) and sch.state_dict()==state['scheduler']
        assert torch.equal(torch.get_rng_state(),state['cpu_rng']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng']))
        if bank is not None:assert c.tensor_sha(bank.state_dict())==c.tensor_sha(state['bank'])
        del state
        second=update(model,opt,sch,bank,x,y,arm,task,token,0,folder,kind='synthetic',diagnose=True)
        expected=torch.load(expected_path,map_location='cpu',weights_only=False);actual=c.snapshot(model,opt,sch,None,bank,1);errors=[]
        def compare(a,b,path='state'):
            if torch.is_tensor(a):
                aa=a.detach().cpu();bb=b.detach().cpu()
                if aa.is_floating_point():
                    errors.append(float((aa-bb).abs().max()) if aa.numel() else 0.)
                    torch.testing.assert_close(aa,bb,rtol=1e-5,atol=1e-7,msg=lambda m:path+': '+m)
                else:assert torch.equal(aa,bb),path
            elif isinstance(a,dict):
                assert set(a)==set(b)
                for key in a:compare(a[key],b[key],path+'/'+str(key))
            elif isinstance(a,(tuple,list)):
                assert len(a)==len(b)
                for i,(xv,yv) in enumerate(zip(a,b)):compare(xv,yv,path+'/'+str(i))
            elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
            else:assert a==b,path
        compare(actual,expected)
        max_error=max(errors,default=0.);del actual,expected
        expected_path.unlink()
        write_diagnostic(folder/(arm+'.jsonl'),dict(step=1,arm=arm),**second,runtime=dict(resume_equal=True))
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            before=model(x)['semantic'];bank=None;after=model(x)['semantic'];assert torch.equal(before,after)
        rows.append(dict(arm=arm,resume_equal=True,deployment_equal=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),synthetic_optimizer_calls=2,update_comparison_max_abs=max_error,comparison_rtol=1e-5,comparison_atol=1e-7,restore_exact=True,bitwise_update_claim=False))
        initial.unlink();rpc(RUN,action='release',task=task,token=token)
        del model,opt,sch,x,y,before,after;gc.collect();torch.cuda.empty_cache()
    c.atomic(folder/'PASSED.json',dict(results=rows,**provenance()))


def smoke():
    setup(261);provenance();backbone=os.environ['EXEC_BACKBONE'];folder=RUN/'smoke'/backbone;folder.mkdir(parents=True,exist_ok=True);rows=[]
    for domain in PLAN['domains']:
        ds=r.dataset(domain,'train_labeled');cache=[ds[i] for i in range(len(ds))];doc,_=schedule(261,domain)
        for arm in ('B0','B1','B2','B3'):
            task='smoke__'+backbone+'__'+domain+'__'+arm;token=claim(task)
            model,_=c.build(backbone,False,r.ASSETS,r.SOURCE);model.cuda().train();opt,sch=c.optimizer_for(model,backbone)
            bank=bank_init(model,cache,backbone,folder) if arm in ('B1','B3') else None
            x,y=r.load_batch(cache,doc['steps'][0],torch.device('cuda:0'));info=update(model,opt,sch,bank,x,y,arm,task,token,0,folder,kind='smoke',diagnose=True)
            write_diagnostic(folder/'DIAGNOSTICS.jsonl',dict(domain=domain,arm=arm,step=1),**info,runtime=dict(discarded=True))
            rows.append(dict(domain=domain,arm=arm,optimizer_calls=1,discarded=True));rpc(RUN,action='release',task=task,token=token)
            del model,opt,sch,bank,x,y;gc.collect();torch.cuda.empty_cache()
    c.atomic(folder/'PASSED.json',dict(results=rows,**provenance()))


def forensic():
    setup(261);provenance();backbone=os.environ['EXEC_BACKBONE'];folder=RUN/'forensic'/backbone;folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for domain in PLAN['domains']:
        entry=json.loads((RUN/'PREFIX_AUDIT.private.json').read_text())[f's261__{backbone}__{domain}__QUERY_PREFIX']
        if not entry['valid']:continue
        state=torch.load(entry['path'],map_location='cpu',weights_only=False)
        model,_=c.build(backbone,False,r.ASSETS,r.SOURCE);model.cuda().train();opt,sch=c.optimizer_for(model,backbone);c.restore(state,model,opt,sch);del state,opt,sch
        doc,_=schedule(261,domain);ds=r.dataset(domain,'train_labeled');cache=[ds[i] for i in range(len(ds))];bank=bank_init(model,cache,backbone,folder)
        model_hash=c.tensor_sha(model.state_dict());bank_hash=c.tensor_sha(bank.state_dict());rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        for j in range(16):
            step=2000+j*1000//16;x,y=r.load_batch(cache,doc['steps'][step],torch.device('cuda:0'))
            c.append(folder/'ATTEMPTS.jsonl',dict(domain=domain,batch=j,optimizer_calls=0))
            with torch.autocast('cuda',dtype=torch.bfloat16):total,terms,read,current,support,out=objective(model,x,y,'B3',bank)
            gradients=gradient_blocks(model,terms)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                classes=out['class_logits'].float();masks=out['mask_logits'].float().sigmoid();prob=classes.softmax(-1);sem=out['semantic'];assignment=matched_assignments(classes,out['mask_logits'].float(),y)['assignments']
                top=(F.normalize(out['queries'].float(),dim=-1)@bank.vectors.T).masked_fill(~bank.supported[None,None],-torch.inf).argmax(-1)
                samples=[]
                for b in range(len(y)):
                    valid=y[b]!=255;unmatched=assignment[b]==3;sample=dict(image_in_batch=b,assignments=assignment[b].tolist(),matched_indices=(~unmatched).nonzero().flatten().tolist(),unmatched_indices=unmatched.nonzero().flatten().tolist(),prototype_top1=top[b].tolist(),classes=[])
                    for cls in (0,1,2):
                        region=(y[b]==cls)&valid
                        if not region.any():sample['classes'].append(dict(gt_class=cls,supported=False,eta=None,null_reason='NO_GT_PIXELS'));continue
                        contributions=prob[b,:,:3].sum(-1)[:,None]*masks[b,:,region];den=contributions.sum(0);valid_den=den>0;eta=float((contributions[unmatched].sum(0)[valid_den]/den[valid_den]).mean()) if valid_den.any() else None
                        dice=(2*(masks[b,:,valid]*(y[b,valid]==cls)).sum(-1)+1)/(masks[b,:,valid].sum(-1)+(y[b,valid]==cls).sum()+1)
                        per=[]
                        for group,selected in [('matched',assignment[b]==cls),('unmatched',unmatched)]:
                            per.append(dict(group=group,count=int(selected.sum()),mask_dice=float(dice[selected].mean()) if selected.any() else None,class_probability=float(prob[b,selected,cls].mean()) if selected.any() else None,no_object_probability=float(prob[b,selected,3].mean()) if selected.any() else None))
                        sample['classes'].append(dict(gt_class=cls,supported=True,eta=eta,null_reason='ZERO_DENOMINATOR' if eta is None else None,semantic_error=float((sem[b].argmax(0)[region]!=cls).float().mean()),groups=per))
                    samples.append(sample)
                changed=classes.clone();changed[...,0]+=2
                changed_sem=semantic_probabilities(changed,out['mask_logits'].float())
                before_grqa=c.grqa_loss(out['queries'].float(),out['queries'].float(),bank)['total'];after_grqa=c.grqa_loss(out['queries'].float(),out['queries'].float(),bank)['total']
                structural=dict(semantic_changed=not torch.equal(changed_sem,sem),readout_loss_changed=float(readout_loss(changed_sem,y)['total'])!=float(terms['Lreadout']),fixed_query_bank_grqa_unchanged=bool(torch.equal(before_grqa,after_grqa)))
            row=dict(backbone=backbone,domain=domain,batch=j,schedule_step=step,losses={k:float(v.detach()) for k,v in terms.items()},readout=read,gradients=gradients,samples=samples,structural=structural)
            c.append(folder/'READOUT_FORENSIC.private.jsonl',row);records.append(row)
            del total,terms,out,x,y
        assert c.tensor_sha(model.state_dict())==model_hash and c.tensor_sha(bank.state_dict())==bank_hash
        assert torch.equal(torch.get_rng_state(),rng[0]) and torch.equal(torch.cuda.get_rng_state(),rng[1])
        del model,bank;gc.collect();torch.cuda.empty_cache()
    c.atomic(folder/'PASSED.json',dict(batches=len(records),optimizer_calls=0,model_bank_rng_unchanged=True,**provenance()))
