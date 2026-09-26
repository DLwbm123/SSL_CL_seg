"""CPU contract checks, no optimizer calls; run as a module before E0."""
import tempfile,json,math,os,uuid
from unittest.mock import patch
from pathlib import Path
import torch
from qprompt.models import semantic_probabilities
from qprompt.losses import supervised_query_loss,PrototypeBank,image_prototypes,image_alignment_loss,grqa_loss
from .method import readout_loss,objective,matched_assignments
from .runtime import PLAN,TASKS,CONFIG_SHA,seed_value,write_diagnostic,diagnostic_rows,save_checkpoint,Budget,broker,rpc,process_identity
from r1_12h import core as c


def main():
    torch.set_num_threads(2);torch.manual_seed(261)
    p=torch.tensor([[[[.6,.1]],[[.3,.7]],[[.1,.2]]]],requires_grad=True);y=torch.tensor([[[0,1]]])
    expected=-(math.log(.6)+math.log(.7))/2+((1-(2*.7+1)/(1.+1+1))+(1-1/(.3+1)))/2
    assert abs(float(readout_loss(p,y)['total'].detach())-expected)<1e-6
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        cc=torch.randn(2,12,4,dtype=dtype,requires_grad=True);mm=torch.randn(2,12,4,4,dtype=dtype,requires_grad=True);labels=torch.randint(0,3,(2,4,4));labels[1]=255
        with torch.autocast('cpu',dtype=torch.bfloat16):prob=semantic_probabilities(cc,mm);loss=readout_loss(prob,labels)['total']
        assert loss.dtype==torch.float32 and torch.isfinite(loss)
        loss.backward();assert cc.grad[0].abs().sum()>0 and mm.grad[0].abs().sum()>0 and cc.grad[1].abs().sum()==0 and mm.grad[1].abs().sum()==0
    for label in (0,1,2,255):
        z=p.detach().clone().requires_grad_();v=readout_loss(z,torch.full_like(y,label))['total'];v.backward();assert torch.isfinite(z.grad).all()
        if label==255:assert float(v)==0 and z.grad.abs().sum()==0
    mixed=torch.cat([p.detach(),p.detach()]);labels=torch.cat([y,torch.full_like(y,255)])
    torch.testing.assert_close(readout_loss(mixed,labels)['total'],readout_loss(p,y)['total'])
    for bad in (-1,3,254):
        try:readout_loss(p,torch.full_like(y,bad))
        except ValueError:pass
        else:raise AssertionError('invalid label accepted')
    for z in (p.detach()*2,torch.full_like(p,float('nan')),p.detach()[:,0]):
        try:readout_loss(z,y)
        except ValueError:pass
        else:raise AssertionError('invalid probability accepted')
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.classes=torch.nn.Parameter(torch.randn(2,12,4));self.masks=torch.nn.Parameter(torch.randn(2,12,4,4));self.pixels=torch.nn.Parameter(torch.randn(2,8,4,4))
        def forward(self,x):return dict(class_logits=self.classes,mask_logits=self.masks,pixels=self.pixels,semantic=semantic_probabilities(self.classes,self.masks))
    model=Tiny();labels=torch.randint(0,3,(2,4,4));bank=PrototypeBank(8);current,support=image_prototypes(model.pixels,labels);bank.update(current,support)
    for arm in ('B0','B1'):
        total,*_=objective(model,torch.zeros(1),labels,arm,bank if arm=='B1' else None)
        original=supervised_query_loss(model.classes,model.masks,labels)['total']
        if arm=='B1':original=original+10*image_alignment_loss(*image_prototypes(model.pixels,labels),bank)
        torch.testing.assert_close(total,original,rtol=0,atol=0)
        ga=torch.autograd.grad(total,tuple(model.parameters()),allow_unused=True,retain_graph=True);gb=torch.autograd.grad(original,tuple(model.parameters()),allow_unused=True)
        for a,b in zip(ga,gb):
            if a is None:assert b is None
            else:torch.testing.assert_close(a,b,rtol=0,atol=0)
    matched=matched_assignments(model.classes,model.masks,labels);original=supervised_query_loss(model.classes,model.masks,labels)
    for k,v in original.items():torch.testing.assert_close(matched[k],v,rtol=0,atol=0)
    q=torch.randn(2,12,8);old=grqa_loss(q,q,bank)['total'];changed=model.classes.detach().clone();changed[...,0]+=2
    assert not torch.equal(semantic_probabilities(changed,model.masks),semantic_probabilities(model.classes,model.masks))
    torch.testing.assert_close(grqa_loss(q,q,bank)['total'],old,rtol=0,atol=0)
    assert len(TASKS)==56 and sum(t['new_valid_updates'] for t in TASKS.values())==64000 and sum(t['is_final_endpoint'] for t in TASKS.values())==48
    assert seed_value(261,'x')==c.seed('x') and len({seed_value(s,'x') for s in (261,262,263)})==3
    tested=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);server,budget=broker(root)
        args=dict(task='mock-owner',token='one',pid=os.getpid(),identity=process_identity(os.getpid()))
        rpc(root,action='claim',**args)
        try:rpc(root,action='claim',**dict(args,token='two'))
        except RuntimeError:pass
        else:raise AssertionError('duplicate ownership accepted')
        # Charge precedes the simulated optimizer exception; durable accounting survives reload.
        rpc(root,action='attempt',task=args['task'],token='one',kind='mock',step=0,request_id=uuid.uuid4().hex,device='cpu')
        assert Budget(root).state['mock']==1
        budget.state['synthetic']=64
        try:rpc(root,action='attempt',task=args['task'],token='one',kind='synthetic',step=0,request_id=uuid.uuid4().hex,device='cpu')
        except RuntimeError:pass
        else:raise AssertionError('synthetic cap bypassed')
        server.shutdown();server.server_close()

    with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.get_rng_state_all',return_value=[]):
        root=Path(tmp);folder=root/'task';folder.mkdir()
        for start,end in ((2000,3000),(0,2000)):
            for local in (1,99,100,101,249,250,251,1000,end-start):
                step=start+local
                for arm in ('B0','B1','B2','B3'):
                    state=c.snapshot(model,torch.optim.AdamW(model.parameters()),torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(model.parameters()),lambda _:1),None,bank if arm in ('B1','B3') else None,step,code_commit='mock',config_digest=CONFIG_SHA)
                    state['scheduler']['last_epoch']=step
                    # State-machine check uses real serializer and aggregate reader, not optimization.
                    save_checkpoint(folder,state,root)
                    restored=torch.load(folder/'latest.pt',weights_only=False)
                    assert restored['global_step']==restored['data_cursor']==restored['scheduler']['last_epoch']==step and (restored['bank'] is not None)==(arm in ('B1','B3'))
                    write_diagnostic(folder/'diagnostic.jsonl',dict(step=step,arm=arm),dict(Lseg=1.,categorical_kl=0.),{},dict(Lseg=2.,categorical_kl=3.),dict(code='mock'))
                    assert diagnostic_rows(folder/'diagnostic.jsonl')[-1]['meta']['step']==step
                    tested.append([arm,step])
    from .analysis import aggregate
    from .closeout import closeout
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'tasks').mkdir();c.atomic(root/'BUDGET.json',{})
        queue={task:dict(status='PENDING') for task in TASKS}
        aggregate(root,queue,{},False)
        assert json.loads((root/'reports/FRESH_SEED_REPLICATION.json').read_text())['decision']=='INCOMPLETE_ENGINEERING_OR_BUDGET'
        for task,t in TASKS.items():
            if not t['is_final_endpoint']:continue
            value=.5+(.01 if t['arm'] in ('B2','B3') else 0.)
            c.atomic(root/'tasks'/task/'EVALUATION.json',dict(task=task,optimization_seed=t['seed'],backbone=t['backbone'],domain=t['domain'],arm=t['arm'],rim=value,cup=value,macro=value,disc_union=value))
        aggregate(root,queue,{},False)
        assert json.loads((root/'reports/FRESH_SEED_REPLICATION.json').read_text())['decision']=='READOUT_BASELINE_GAIN_REPLICATED'
        c.atomic(root/'FINAL.json',{})
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):closeout(root,root/'old')
        assert (root/'reports/HISTORICAL_REPRODUCTION.csv').exists()
    print(json.dumps(dict(status='PASSED',optimizer_calls=0,mock_serializations=len(tested),boundaries=tested,matcher_exact=True,legacy_equivalence=True,structural_classifier_decoupling=True)))

if __name__=='__main__':main()
