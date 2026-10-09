"""Restrict memory transfer to class-agreeing pixels; reuse the native loss."""
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path


def admitted_memory(q, qm):
    import torch
    assert q.shape==qm.shape and q.shape[1]==3
    agree=q.detach().argmax(1)==qm.detach().argmax(1)
    return torch.where(agree[:,None],qm.detach(),torch.full_like(qm,1/q.shape[1])),agree


def selfcheck(c):
    import torch
    def field(x):return torch.tensor(x).reshape(1,3,1,1).expand(1,3,2,2).clone().requires_grad_()
    p,q,m,f=field([.2,.3,.5]),field([.8,.1,.1]),field([.75,.15,.1]),field([.8,.1,.1])
    valid=torch.ones(1,2,2,dtype=torch.bool)
    with torch.no_grad():m[:,:,0,0]=m.new_tensor([.1,.8,.1]);f[:,:,0,0]=m[:,:,0,0]
    gated,agree=admitted_memory(q,m);assert not gated.requires_grad and int(agree.sum())==3
    for action in (3,8,9,11):
        loss,stats,target=c.transfer_loss(p,q,valid,action,gated,f)
        old,_,oldtarget=c.transfer_loss(p,q,valid,action,m,f)
        assert torch.equal(target[:,:,0,0],q.detach()[:,:,0,0]) and torch.equal(target[:,:,agree[0]],oldtarget[:,:,agree[0]])
        gradients=torch.autograd.grad(loss,(p,q,m,f),allow_unused=True)
        assert torch.isfinite(gradients[0]).all() and all(v is None for v in gradients[1:])
        loss2,_,_=c.transfer_loss(p,q,agree,action,gated,f);old2,_,_=c.transfer_loss(p,q,agree,action,m,f)
        assert torch.equal(loss2,old2) and torch.equal(torch.autograd.grad(loss2,p)[0],torch.autograd.grad(old2,p)[0])
        empty=c.transfer_loss(p,q,~valid,action,gated,f)[0];assert torch.isfinite(empty) and empty==0
    for action,mem in ((13,m),(11,None)):
        try:c.transfer_loss(p,q,valid,action,mem,f)
        except (ValueError,PermissionError):pass
        else:raise AssertionError('invalid input accepted')


def configure(cfg,root):
    global Q,N
    spec=importlib.util.spec_from_file_location('quarter',cfg['quarter_entry']);Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q);Q.configure(cfg,root);N=Q.N
    N.STAGE='V115';N.METHODS=('GLOBAL','TIME');N.CAPS=dict(qualification=8,development=4800);N.c.CAPS['development']=400
    original_installer=N.EXTRA_ACTION_INSTALLER
    def install(c):
        original_installer(c);selfcheck(c);original=c.transfer_loss;ordinal=0
        def transfer(p,q,valid,action,qm=None,qflip=None):
            nonlocal ordinal
            if qm is None or qflip is None:return original(p,q,valid,action,qm,qflip)
            filtered,agree=admitted_memory(q,qm)
            loss,stats,target=original(p,q,valid,action,filtered,qflip)
            eligible=valid.bool()&(qm.detach().max(1).values>.7)&(qm.detach().argmax(1)==qflip.detach().argmax(1))
            admitted=valid.bool()&(q.detach().max(1).values>.7)
            ordinal+=1
            row=dict(ordinal=ordinal,action=action,valid=int(valid.sum()),admitted=int(admitted.sum()),eligible=int((eligible&admitted).sum()),rejected=int((eligible&admitted&~agree).sum()),transferred=int((eligible&admitted&agree).sum()))
            for cls in range(3):
                mask=q.detach().argmax(1)==cls
                row['class'+str(cls)]={k:int((v&mask).sum()) for k,v in dict(valid=valid.bool(),admitted=admitted,eligible=eligible&admitted,rejected=eligible&admitted&~agree,transferred=eligible&admitted&agree).items()}
            assert row['eligible']==row['rejected']+row['transferred'];c.e.append(root/'GATE_LEDGER.jsonl',row)
            return loss,stats,target
        c.transfer_loss=transfer
    N.EXTRA_ACTION_INSTALLER=install


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN']);configure(cfg,root)
    if os.environ.get('EXEC_SELFCHECK')=='1':
        original=N.E.install_actions(N.c);N.E.action_check(N.c,original);selfcheck(N.c);Q.selfcheck();print('PASS gate targets/gradients/detach/empty/error checks; zero model/optimizer calls')
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':N.gpu_job(root,cfg)
            else:
                assert N.D.read(Path(cfg['v114'])/'COMPLETION_AUDIT.json')['status']=='PASS' and N.D.read(Path(cfg['v114'])/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
                Q.coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
