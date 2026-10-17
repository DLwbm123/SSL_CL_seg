import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from lcrseg.contracts import SegModelOutput, LabeledBatch, UnlabeledBatch
from lcrseg.methods import build_method
from lcrseg.methods.care_seg_v1.core import DecisionTree, Repair, geometry_basis
from lcrseg.methods.care_seg_v1.statistics import StateGMM, TemporalRisk, CandidateRisk, choose
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler

TREE = [dict(name='root', children=[[0], [1]])]

class Tiny(nn.Module):
    num_classes = 2
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
    def forward(self, image):
        logits = self.conv(image)
        return SegModelOutput(logits, torch.nn.functional.normalize(logits, dim=1), {'dec1': logits})

class Data:
    transform = None
    def __init__(self, patients, labeled):
        self.patients, self.labeled = patients, labeled
        self.samples = [SimpleNamespace(row=dict(patient_id=p, case_id=p), slice_index=None) for p in patients]
    def __len__(self):
        return len(self.patients)
    def _metadata(self, s):
        return dict(site='s')
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        image = torch.linspace(-1, 1, 64).reshape(1, 8, 8)
        valid = torch.ones_like(image, dtype=torch.bool)
        meta = dict(case_id=self.patients[index], patient_id=self.patients[index], site='s', slice_index=None)
        if self.labeled:
            return dict(image=image, label=(image[0] > 0).long(), valid_mask=valid, **meta)
        return dict(weak_image=image, strong_image=image, strong_valid_mask=valid,
                    geometry_record=dict(hflip=False, vflip=False), **meta)


def fixture_method():
    config = dict(tree=TREE, roles={'s': {'A': ['a1', 'a2'], 'B': ['b1', 'b2']}}, grid=[8, 8], repair_steps=3, em_iterations=3, block_steps=2)
    m = build_method('care_seg_v1', Tiny(), config=config)
    m.prepare_datasets(Data(['a1','a2','b1','b2'], True), Data(['u1'], False), 's')
    m.begin_site('s', None, 100)
    m.before_training_step(0)
    return m


def batches():
    image = torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8).repeat(2, 1, 1, 1)
    label = (image[:, 0] > 0).long()
    valid = torch.ones_like(image, dtype=torch.bool)
    l = LabeledBatch(image, label, valid, ['a1','a2'], ['a1','a2'], ['s','s'], [None,None])
    u = UnlabeledBatch(image[:1], image[:1], valid[:1], ['u1'], ['u1'], ['s'], [None], [dict(hflip=False, vflip=False)])
    return l, u


def trainer(m):
    opt = build_optimizer(m, lr=.1, weight_decay=0)
    return Trainer(m, optimizer=opt, scheduler=build_scheduler(opt,total_steps=100), device='cpu', amp=False)


def test_hierarchy_golden_and_partial_gradient():
    tree = DecisionTree(3, [dict(name='root', children=[[0],[1,2]]), dict(name='nested', children=[[1],[2]])])
    logits = torch.tensor([[[[0.]], [[1.]], [[2.]]]], requires_grad=True)
    q = logits.softmax(1)[0].detach()
    entries = [[dict(target=tree.read(q,n)[0], weight=tree.read(q,n)[1]) for n in tree.nodes]]
    got = tree.loss(logits,entries,torch.ones(1,1,1,1))
    assert torch.allclose(got, -(q*q.log()).sum())
    entries[0][1]['weight'].zero_()
    got = tree.loss(logits, entries, torch.ones(1,1,1,1)); got.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        DecisionTree(3,[dict(name='bad',children=[[0],[1,2]])])


def test_repair_bound_and_real_label_improvement():
    repair = Repair(3,.5,30,.5)
    p = torch.full((20,),.4)
    basis = torch.zeros(20,3); basis[:,0]=1
    repair.fit(p,basis,torch.ones(20),torch.ones(20))
    assert repair.coefficient.abs().sum() <= .500001
    out = repair.apply(torch.stack([1-p,p]),basis.T)
    assert out[1].mean() > p.mean() and not out.requires_grad
    taper, basis3d = geometry_basis(np.zeros((2,3,4),bool),spacing=(2,1,1),radius=3)
    assert taper.sum() == 0 and basis3d.shape == (4,2,3,4)


def test_gmm_illegal_and_fail_closed():
    g = StateGMM()
    with pytest.raises(ValueError):
        g.fit(np.zeros((4,2)),[0,1,2,3],[False]*4,['a','b','a','b'],np.zeros((1,2)),[True],['u'],classes=2)
    g.fit(np.zeros((2,2)),[0,3],[True]*2,['a','b'],np.zeros((1,2)),[True],['u'],classes=2)
    post, valid = g.predict([[0,0]],[True])
    assert not valid.any() and post.sum() == 0


def test_label_anchored_modes_and_risk_selection():
    x = np.array([[-2,-2],[-2.1,-2.1],[2,2],[2.1,2.1]])
    g = StateGMM(iterations=5).fit(x,[0,0,3,3],[True]*4,['a','b','a','b'],x,[True]*4,['u']*4,classes=2)
    post, valid = g.predict(x,[True]*4)
    assert valid.all() and post[:2,0].min() > .9 and post[2:,3].min() > .9
    assert not g.predict([[100,100]],[True])[1].any()
    risk = np.array([[.2,.05,.12],[.1,.2,np.inf],[.5,.4,.6]])
    assert choose(risk,np.isfinite(risk),maximum=.2,margin=.02).tolist() == [1,0,-1]
    temporal = TemporalRisk(.5)
    temporal.update('x',risk,np.zeros((3,3)))
    lowered = temporal.update('x',risk/2,np.zeros((3,3)))
    assert lowered[0,0] == pytest.approx(.15)
    assert temporal.update('x',risk/2,np.ones((3,3)))[0,0] == .1


def test_roles_and_geometry_alignment():
    m=fixture_method()
    with pytest.raises(ValueError):
        m.prepare_datasets(Data(['a1','a2','b1','b2'],True),Data(['a1'],False),'s')
    _,u=batches()
    m.maps[('u1',None)][0]['weight'] = torch.arange(64).reshape(8,8).float()
    u.geometry_record=[dict(hflip=True,vflip=True)]
    assert torch.equal(m.aligned_entries(u)[0][0]['weight'], m.maps[('u1',None)][0]['weight'].flip((-1,-2)))


def test_backward_checkpoint_exact_resume_and_overfit():
    torch.manual_seed(3)
    m=fixture_method(); t=trainer(m); l,u=batches()
    m._make_old_model()
    # Exercise the full post-optimizer diagnostic with known fixed historical support.
    with torch.no_grad():
        h=m.old_model(u.weak_image).logits[0].softmax(0)
    m.maps[('u1',None)][0].update(historical=h,keep=torch.ones(8,8,dtype=torch.bool))
    first=t.train_step(l,u,state=TrainerState())
    assert first.scalars['drift_coverage']==1 and 'drift_after' in first.scalars
    assert all(p.grad is None for p in m.old_model.parameters())
    assert all(p.grad is None for p in m.ema_model.parameters())
    state=copy.deepcopy(m.method_state_dict()); model=copy.deepcopy(m.model.state_dict()); ts=copy.deepcopy(t.state_dict())
    resumed=fixture_method(); resumed.model.load_state_dict(model); resumed.load_method_state_dict(state)
    rt=trainer(resumed); rt.load_state_dict(ts)
    left=t.train_step(l,u,state=TrainerState(site_step=1))
    right=rt.train_step(l,u,state=TrainerState(site_step=1))
    assert torch.equal(left.total_loss,right.total_loss)
    for p,q in zip(m.model.parameters(),resumed.model.parameters()):
        assert torch.equal(p,q)
    initial=float(first.losses['loss_sup'].detach())
    for i in range(2,70):
        m.last_refresh=i//m.config['block_steps']
        result=t.train_step(l,None,state=TrainerState(site_step=i))
    assert float(result.losses['loss_sup'].detach()) < initial*.4


def test_checkpoint_preserves_controller_through_shared_schema(tmp_path):
    from lcrseg.engine.checkpoint import checkpoint_payload,save_checkpoint,load_checkpoint
    m=fixture_method(); state=m.method_state_dict(); t=trainer(m)
    payload=checkpoint_payload(method_name=m.method_name,method_version=m.method_version,
        git_commit='local',config_resolved=m.config,site_id='s',site_index=0,epoch=0,site_step=0,global_step=0,
        current_model_state=m.model.state_dict(),**t.state_dict(),**state,
        data_split_hash='synthetic',manifest_hash='synthetic')
    path=tmp_path/'state.pt';save_checkpoint(path,payload)
    other=fixture_method();other.load_method_state_dict(load_checkpoint(path))
    assert other.last_refresh==m.last_refresh and other.maps.keys()==m.maps.keys()


def test_physical_geometry_rejects_missing_slices():
    from lcrseg.methods.care_seg_v1.controller import add_geometry
    tree=DecisionTree(2,TREE)
    row=dict(key=('v',0),p=torch.ones(2,4,4)/2,shape=(4,4))
    with pytest.raises(ValueError,match='every volume slice'):
        add_geometry([row],tree,tree.nodes[0],dict(geometry='physical3d',volumes={'v':dict(depth=2,spacing=[2,1,1])},repair_radius=2))


def test_candidate_risk_uses_action_specific_losses():
    gamma=np.tile([.1,.2,.3,.4],(6,1));desc=np.zeros((6,3,3))
    losses=np.tile([.3,.1,.2],(6,1));eligible=np.ones((6,3),bool)
    model=CandidateRisk().fit(gamma,desc,losses,eligible,['a','b','c','a','b','c'])
    prediction=model.predict(gamma,desc)
    np.testing.assert_allclose(prediction,losses)
    d01=prediction[:,0]-prediction[:,1];d12=prediction[:,1]-prediction[:,2]
    np.testing.assert_allclose(d01+d12,prediction[:,0]-prediction[:,2])


def test_shared_runner_refresh_and_resume(tmp_path):
    import csv
    from .conftest import make_synthetic_root
    from lcrseg.engine.continual_runner import ContinualRunner
    root=make_synthetic_root(tmp_path,records=6)
    manifest=root/'manifests/training/lcrseg_v1_seed0.csv'
    rows=list(csv.DictReader(manifest.open()))
    for i,row in enumerate(rows):
        row['site_or_vendor']='REFUGE'
        row['primary_20pct_split']='train_labeled' if i<3 else ('val' if i==5 else 'train_unlabeled')
        row['label_h5_relpath']=f'labels/fundus/SITE/case{i}.h5' if i<3 or i==5 else ''
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    cfg=ContinualRunner.default_config(data_root=root,run_root=tmp_path/'runs',dataset='fundus',method_name='care_seg_v1',seed=0,site_order=['REFUGE'],run_name='check',device='cpu')
    cfg['data'].update(require_readonly=False,evaluation_role='val',evaluation_sites=['REFUGE'])
    cfg['training'].update(steps_per_site=2,max_steps_this_invocation=1,amp=False,gradient_cosine_interval=0)
    cfg['method'].update(tree=[dict(name='root',children=[[0],[1,2]]),dict(name='fine',children=[[1],[2]])],roles={'REFUGE':{'A':['case0'],'B':['case1','case2']}},grid=[8,8],repair_steps=2,em_iterations=2)
    runner=ContinualRunner(cfg)
    result=runner.run()
    checkpoints=list((tmp_path/'runs/check').glob('*.pt'))
    assert checkpoints
    cfg['training']['max_steps_this_invocation']=None
    from lcrseg.engine.checkpoint import load_checkpoint
    path=max(checkpoints,key=lambda p:load_checkpoint(p)['global_step'])
    payload=load_checkpoint(path)
    assert 'care' in payload['method_statistics']
    ContinualRunner(cfg).run(resume_checkpoint=path)
    assert list((tmp_path/'runs/check').glob('checkpoint_final*'))
