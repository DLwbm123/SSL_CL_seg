"""CPU, generated tensors only; every actual optimizer invocation is forbidden."""
import copy
import os
from pathlib import Path
import pytest
import torch
from .protocol import contract, nodes, MAPS, SHAPES, ORDERS, frozen_identity, verify_frozen
from .protection import BoundedSidecar, bounded, constrain_displacement
from .memory import SiteMemory
from .model import independent_reference
from .checkpoint import save, restore, same_tree
from .runtime import make_trainer, worker, require_v2_destination
from ..main_style_ssl_v1.runtime import permit
from ..five_frameworks_v1.native_parent import build
from ..five_frameworks_v1.recipes import generator


@pytest.fixture(autouse=True)
def no_optimizer(monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs):
        raise AssertionError('actual optimizer invocation forbidden in no-training qualification')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    monkeypatch.setattr(torch.optim.SGD,'step',forbidden)


def test_frozen_budget_and_no_launch(tmp_path):
    p=contract()
    assert len(p['nodes'])==30 and p['reused_source_updates']==130400
    assert p['new_formal_optimizer_calls']==381200 and not p['test_accessor'] and not p['ablation']
    with pytest.raises(PermissionError,match='FROZEN_NOT_LAUNCHED'):
        worker(dict(output=str(tmp_path/'forbidden')))
    assert not (tmp_path/'forbidden').exists()


def test_freeze_detects_source_change(tmp_path):
    import json
    (tmp_path/'candidate.py').write_text('x=1\n')
    receipt=dict(version=contract()['version'],**frozen_identity(tmp_path),execution_authorized=False)
    (tmp_path/'FREEZE.json').write_text(json.dumps(receipt))
    assert verify_frozen(tmp_path)==receipt
    (tmp_path/'candidate.py').write_text('x=2\n')
    with pytest.raises(ValueError,match='frozen V2'):
        verify_frozen(tmp_path)


def test_destination_cannot_hide_inside_live_v1(tmp_path):
    with pytest.raises(PermissionError,match='inside V1'):
        require_v2_destination(tmp_path/'main_style_ssl_v1_live'/'main_style_ssl_v2_candidate'/'jobs'/'node')
    with pytest.raises(PermissionError):
        require_v2_destination(Path('/uncreated_engineering_namespace/other/jobs/node'))
    require_v2_destination(tmp_path/'main_style_ssl_v2_candidate'/'jobs'/'node')


def test_near_singular_and_cumulative_guard():
    q=torch.eye(4)[:,:2];s=BoundedSidecar(q);r=s.r;before={id(r):r.detach().clone()}
    with torch.no_grad():r[0,0]=-.995
    assert not s.valid()
    stats=constrain_displacement([r],before,[],s.valid)
    assert stats['scale']<1 and s.valid()
    s=BoundedSidecar(q,torch.eye(4)*.6);before={id(s.r):s.r.detach().clone()}
    with torch.no_grad():s.r.copy_(torch.eye(2)*-.4)
    assert bounded(s.matrices()[0]) and not bounded(s.matrices()[1])
    assert constrain_displacement([s.r],before,[],s.valid)['scale']<1 and s.valid()
    with pytest.raises(ValueError,match='inherited F'):
        BoundedSidecar(q,torch.eye(4)*.006)


def test_actual_direction_after_preconditioning_and_conflicting_cones():
    p=torch.nn.Parameter(torch.zeros(2));before={id(p):p.detach().clone()}
    # Raw g dot reference is positive; an anisotropic Adam-like displacement violates it.
    raw=torch.tensor([1.,-.1]);ref=torch.tensor([1.,1.])
    assert raw@ref>0
    with torch.no_grad():p.copy_(torch.tensor([-.01,1.]))
    stats=constrain_displacement([p],before,[[ref]],lambda:True)
    assert stats['dots_before'][0]>0 and stats['dots_after'][0]<=1e-7
    p=torch.nn.Parameter(torch.ones(2));before={id(p):torch.zeros(2)}
    stats=constrain_displacement([p],before,[[torch.tensor([1.,0.])],[torch.tensor([-1.,0.])]],lambda:True)
    assert abs(float(p[0]))<1e-6
    bad=torch.nn.Parameter(torch.full((2,),float('nan')))
    with pytest.raises(FloatingPointError,match='nonfinite Adam'):
        constrain_displacement([bad],{id(bad):torch.zeros(2)},[],lambda:True)


def test_teacher_guard_and_seal():
    student=BoundedSidecar(torch.eye(4)[:,:2]);teacher=copy.deepcopy(student)
    with torch.no_grad():student.r.copy_(-2*torch.eye(2))
    assert student.valid()
    student.update_teacher(teacher,decay=.5)
    assert teacher.valid() and teacher.ema_guard['scale']<1
    x=torch.randn(2,4,5,5)
    assert torch.allclose(student(x),torch.einsum('ij,bjhw->bihw',student.seal(),x))


@pytest.mark.parametrize('classes',[2,3,4])
def test_foreground_conflicts_and_detach(classes):
    q=torch.full((1,classes,2,2),.01,requires_grad=True)
    with torch.no_grad():q[:,classes-1]=1-.01*(classes-1)
    z=torch.zeros_like(q,requires_grad=True);geometry=torch.ones(1,2,2,dtype=torch.bool)
    reliability=torch.ones(classes,requires_grad=True);evidence=torch.ones_like(geometry,dtype=torch.float32,requires_grad=True)
    loss,counts,support=independent_reference(z,q,geometry,reliability,evidence)
    assert counts[-1]==4 and support.all()  # EMA could be all-background, admission could be empty.
    loss.backward();assert z.grad is not None and q.grad is None and reliability.grad is None and evidence.grad is None
    _,counts,_=independent_reference(z,q,torch.zeros_like(geometry),reliability,evidence)
    assert not any(counts)


def test_memory_validation_and_missing_class():
    m=SiteMemory(2,4)
    with pytest.raises(ValueError,match='missing training-L'):
        m.export_site('site',{'W':torch.zeros(2,2)})
    m.commit(dict(classes=[(1,{'W':torch.ones(2,2)},torch.ones(2,4))],calibration=None))
    h=m.export_site('site',{'W':torch.zeros(2,2)})
    frozen=copy.deepcopy(h);nextm=SiteMemory(2,4,h)
    value=torch.ones(2,2,requires_grad=True)
    loss=nextm.penalty({'W':value});loss.backward();assert value.grad is not None
    assert same_tree(frozen,nextm.history)
    e=nextm.evidence(torch.randn(1,4,3,3),torch.ones(1,3,3,dtype=torch.long))
    assert (e>=.5).all() and (e<=1).all()
    corrupt=nextm.state();corrupt['history'][0]['importance'][1]['W'].fill_(-1)
    with pytest.raises(ValueError):nextm.load(corrupt)
    assert same_tree(nextm.history,frozen)


class GeneratedDomain:
    def __init__(self,node,size=64):
        self.__dict__.update({k:node[k] for k in ('seed','order','stage','dataset')})
        self.size=size;self.l_reads=self.u_reads=0
        self.stage_source=dict(kind='generated_engineering_only',dataset=self.dataset,domain=node['domain'],seed=self.seed)
    def semantic_metadata(self):
        return dict(provider='generated_engineering_only',seed=self.seed,order=self.order,stage=self.stage,
                    dataset=self.dataset,size=self.size,stage_source=self.stage_source,labels='generated L only; no U-label capability')
    def labeled(self,step,stream='labeled'):
        self.l_reads+=1;g=generator(self.seed,self.order,self.stage,step,stream)
        x=torch.randn(2,SHAPES[self.dataset][0],self.size,self.size,generator=g).clamp(-1,1)
        if self.dataset=='fundus':x=(x+1)/2
        y=(torch.arange(self.size)*len(MAPS[self.dataset])//self.size)[:,None].repeat(1,self.size)
        return x,y[None].repeat(2,1,1),('generated_L0','generated_L1')
    def unlabeled(self,step):
        self.u_reads+=1;g=generator(self.seed,self.order,self.stage,step,'U_generated')
        x=torch.randn(2,SHAPES[self.dataset][0],self.size,self.size,generator=g).clamp(-1,1)
        if self.dataset=='fundus':x=(x+1)/2
        return x,torch.ones(2,self.size,self.size,dtype=torch.bool),('generated_U0','generated_U1')


def factory(node,previous=None,history=(),initialize=True,native=None):
    reference=os.environ['SSLCL_V2_REFERENCE']
    if native is None:
        native=build(reference,torch.device('cpu'),node['seed'],input_channels=SHAPES[node['dataset']][0],
                     num_classes=len(MAPS[node['dataset']]),domain=ORDERS[node['dataset']][0][0])
    return make_trainer(native,node,GeneratedDomain(node),previous,history,initialize,permit(node,'smoke'))


@pytest.mark.parametrize('dataset',['fundus','prostate','mnms'])
def test_native_backward_memory_transition_resume_and_merge(dataset,tmp_path):
    source_node=next(n for n in nodes() if n['dataset']==dataset and n['seed']==163 and n['stage']==0)
    source=factory(source_node);initial=copy.deepcopy(source.model.state_dict());source.bootstrap=True
    source.losses();source.memory.commit(source.draft);source.draft=None
    assert same_tree(initial,source.model.state_dict()) and source.physical_optimizer_updates==0
    history=source.memory.export_site(source_node['domain'],source.model.native_values())
    deployment=source.model.deploy()
    node=next(n for n in nodes() if n['dataset']==dataset and n['seed']==163 and n['stage']==1 and n['order']==1)
    target=factory(node,deployment.transform,history,native=copy.deepcopy(deployment.parent.native))
    if dataset!='fundus':
        assert not target.model.parent.adapters[0].a.requires_grad and not target.model.parent.adapters[0].b.requires_grad
        optimizer_ids={id(p) for g in target.optimizer.param_groups for p in g['params']}
        assert id(target.model.parent.adapters[0].a) not in optimizer_ids
    # Zero-initialized B makes the entry A gradient exactly zero. A fixed synthetic
    # displacement exercises A as well; it is neither an optimizer call nor a fit.
    trainable=[p for p in target.model.parameters() if p.requires_grad]
    proposal_start={id(p):p.detach().clone() for p in trainable}
    with torch.no_grad():
        for a in target.model.parent.adapters:
            if a.b.requires_grad:a.b.add_(1e-4)
        target.model.sidecar.r.add_(torch.eye(len(target.model.sidecar.r))*1e-4)
    constrain_displacement(trainable,proposal_start,[],target.model.sidecar.valid)
    full_u=node['steps']//2
    target.step=target.cursor=full_u  # Enter full-U synthetically; no optimizer invocation or replay.
    state=copy.deepcopy(target.model.state_dict());ema=copy.deepcopy(target.ema.state_dict())
    split=target.update(skip=True)
    assert target.last['active_U'] and target.provider.u_reads>0 and target.physical_optimizer_updates==0
    assert same_tree(state,target.model.state_dict()) and same_tree(ema,target.ema.state_dict())
    assert all(p.grad is None for p in target.model.entry_anchor.parameters())
    for token in ('.A','.B','sidecar.r'):
        selected=[split[id(p)] for n,p in target.model.named_parameters() if token in n and p.requires_grad]
        assert any(g is not None and float(g.norm())>0 for d in selected for g in d.values())
    assert all(g is None or torch.isfinite(g).all() for g in target.protect_grads)
    # Exercise the real trainer hook with a fixed proposed displacement, not Adam.step.
    before={id(p):p.detach().clone() for p in target.model.parameters() if p.requires_grad}
    optimizer_before=copy.deepcopy(target.optimizer.state_dict())
    with torch.no_grad():
        for p in target.model.parameters():
            if id(p) in before:p.add_(1e-5)
    target.transform_update(before)
    assert target.model.sidecar.valid() and same_tree(optimizer_before,target.optimizer.state_dict())
    assert target.guard_totals['calls']==1 and target.physical_optimizer_updates==0
    # Save at cursor zero: the fabricated phase cursor is not an actual execution count.
    target.step=target.cursor=0
    identity=dict(dataset=dataset,seed=163,stage=1)
    path=tmp_path/'resume.pt';save(target,path,identity)
    resumed=factory(node,deployment.transform,history,initialize=False,native=copy.deepcopy(deployment.parent.native))
    restore(resumed,path,identity)
    assert same_tree(target.model.state_dict(),resumed.model.state_dict())
    assert same_tree(target.memory.state(),resumed.memory.state())
    assert same_tree(target.optimizer.state_dict(),resumed.optimizer.state_dict())
    assert target.guard_totals==resumed.guard_totals
    for t in (target,resumed):t.step=t.cursor=full_u;t.update(skip=True)
    assert target.last==resumed.last and same_tree(target.protect_grads,resumed.protect_grads)
    assert same_tree(target.ema.state_dict(),resumed.ema.state_dict())
    assert target.provider.l_reads==resumed.provider.l_reads and target.provider.u_reads==resumed.provider.u_reads
    with torch.no_grad():
        target.model.eval().requires_grad_(False);target.model.role='eval'
        x,_,_=target.provider.labeled(2)
        a=target.model(x,mode='eval');sealed=target.model.deploy();b=sealed(x)
        assert torch.allclose(a,b,rtol=2e-4,atol=2e-5)
    target.step=target.cursor=0
    bad=torch.load(path,weights_only=False);bad['v2']['memory']['history'][0]['means'].add_(1)
    corrupted=tmp_path/'bad.pt';torch.save(bad,corrupted)
    before=copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError,match='immutable inherited'):
        restore(resumed,corrupted,identity)
    assert same_tree(before,resumed.model.state_dict())
    bad=torch.load(path,weights_only=False);bad['student']['sidecar.r'].copy_(-.999*torch.eye(len(bad['student']['sidecar.r'])))
    torch.save(bad,corrupted)
    with pytest.raises(ValueError,match='unbounded checkpoint'):
        restore(resumed,corrupted,identity)
    assert same_tree(before,resumed.model.state_dict())
    # Atomic write fault leaves the previously complete checkpoint usable.
    original=torch.load(path,weights_only=False)
    with pytest.raises(RuntimeError,match='during_checkpoint'):
        save(resumed,path,identity,fault=True)
    assert same_tree(original,torch.load(path,weights_only=False))


@pytest.mark.parametrize('dataset',['fundus','prostate','mnms'])
def test_native_full_resolution_and_two_domain_memory(dataset):
    node0=next(n for n in nodes() if n['dataset']==dataset and n['seed']==163 and n['stage']==0)
    source=factory(node0);source.bootstrap=True;source.losses();source.memory.commit(source.draft);source.draft=None
    history=source.memory.export_site(node0['domain'],source.model.native_values());d=source.model.deploy()
    node=next(n for n in nodes() if n['dataset']==dataset and n['seed']==163 and n['stage']==1 and n['order']==1)
    target=factory(node,d.transform,history,native=copy.deepcopy(d.parent.native))
    target.provider.size=SHAPES[dataset][1]
    target.step=target.cursor=0
    split=target.update(skip=True)
    assert all(g is None or torch.isfinite(g).all() for part in split.values() for g in part.values())
    # Observe the second site's training-L-only summaries without any optimizer call.
    target.bootstrap=True;target.losses();target.memory.commit(target.draft);target.draft=None
    second=target.memory.export_site(node['domain'],target.model.native_values())
    assert [h['domain'] for h in second]==[node0['domain'],node['domain']]
    assert same_tree(second[0],history[0]) and target.physical_optimizer_updates==0
    assert all((v>=0).all() for c in target.memory.sensitivity for v in c.values())
    assert any(float(v.sum())>0 for c in target.memory.sensitivity for v in c.values())
