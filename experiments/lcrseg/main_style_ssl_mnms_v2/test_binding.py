"""Generated-input collector and simulated scheduler boundary checks; no Adam."""
import copy,json
from pathlib import Path
import pytest
import torch
from .test_engineering import summary
from .protocol import contract
from .memory import History
from .checkpoint import same_tree
from . import collector,runtime,controller as orchestration


class Deployment(torch.nn.Module):
    def __init__(self):
        super().__init__();self.parent=torch.nn.Module();self.parent.native=torch.nn.Conv2d(1,4,1)
        self.transform=torch.nn.Parameter(torch.eye(4),requires_grad=False)
    def forward(self,x):return torch.einsum('ij,bjhw->bihw',self.transform,self.parent.native(x))


@pytest.mark.parametrize('missing',[False,True])
def test_collector_exact_fixed_anchor_independent_stream(monkeypatch,missing):
    torch.set_num_threads(1);torch.manual_seed(163);m=Deployment().eval().requires_grad_(False)
    original=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();reads=[]
    def access(data,ds,domain,role):assert (ds,domain,role)==('mnms','Siemens','train_labeled');return object()
    def batch(provider,access,cursor,stream):
        assert stream=='history_final';reads.append(cursor)
        x=torch.arange(32).reshape(2,1,4,4).float()/32
        y=torch.zeros(2,4,4,dtype=torch.long) if missing else torch.arange(4).reshape(1,4,1).expand(2,4,4)
        return x,y,('anonymous1','anonymous2')
    monkeypatch.setattr(collector,'Accessor',access);monkeypatch.setattr(collector.CurrentDomain,'_batch',batch)
    node=next(n for n in contract()['nodes'] if n['stage']==0)
    anchor={n:p for n,p in m.parent.native.named_parameters()};anchor['__F__']=m.transform
    if missing:
        with pytest.raises(ValueError,match='missing training-L class'):collector.collect(m,anchor,node,'unused')
    else:
        site,cost=collector.collect(m,anchor,node,'unused');History([site],['Siemens'])
        assert cost['L_batch_reads']==32 and cost['optimizer_calls']==0
        assert all(c['observations']==32 for c in site['classes'])
    assert reads==list(range(32)) and same_tree(original,m.state_dict()) and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())


def test_worker_and_controller_refuse_before_output(tmp_path):
    root=tmp_path/'main_style_ssl_mnms_v2_run_forbidden';config={'output':str(root),'root':str(root)}
    with pytest.raises(PermissionError):runtime.worker(config)
    with pytest.raises(PermissionError):orchestration.controller(config)
    assert not root.exists()


@pytest.mark.parametrize('case',['nfs','local_disk','outside'])
def test_formal_authority_also_requires_NAS(tmp_path,monkeypatch,case):
    from . import protocol
    storage=tmp_path/'nas';storage.mkdir();output=(storage if case!='outside' else tmp_path/'other')/'main_style_ssl_mnms_v2_run_test'
    path=tmp_path/'synthetic_authority.json'
    path.write_text(json.dumps(dict(version=protocol.VERSION,operation='authorized_formal_targets',user_instruction='synthetic fixture only',ready_identity='fixture',new_formal_budget=209800,datasets=['mnms'],seeds=[163,164])))
    monkeypatch.setattr(protocol,'verify_ready',lambda:dict(ready_identity='fixture'))
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda *a,**k:'ext4' if case=='local_disk' else 'nfs4')
    config=dict(authority_receipt=str(path),root=str(output),storage_root=str(storage))
    if case=='nfs':assert runtime.authorize(config)['new_formal_budget']==209800
    else:
        with pytest.raises(PermissionError,match='NAS'):runtime.authorize(config)
    assert not output.exists() and not list(storage.iterdir())


@pytest.mark.parametrize('failed_collection',[False,True])
def test_worker_keeps_summary_exposure_on_success_or_failure(tmp_path,monkeypatch,failed_collection):
    node=next(n for n in contract()['nodes'] if n['seed']==163 and n['stage']==1)
    out=tmp_path/'main_style_ssl_mnms_v2_run_worker'/'jobs'/node['id']
    monkeypatch.setattr(runtime,'authorize',lambda c:None)
    def shared_worker(config):
        out.mkdir(parents=True)
        runtime.base.write(out/'HISTORY_COLLECTION_STATUS.json',dict(batch=15 if failed_collection else 32,L_image_exposures=30 if failed_collection else 64,optimizer_calls=0))
        if not failed_collection:(out/'sealed.pt').touch()
        runtime.base.write(out/'COSTS.json',dict(physical_optimizer_updates=0))
        if failed_collection:raise ValueError('collector missing support')
        return 'shared engine delegated'
    monkeypatch.setattr(runtime.base,'worker',shared_worker)
    config=dict(node=node,output=str(out))
    if failed_collection:
        with pytest.raises(ValueError,match='missing support'):runtime.worker(config)
    else:assert runtime.worker(config)=='shared engine delegated'
    cost=json.loads((out/'COSTS.json').read_text())
    assert cost['final_history_collection']['L_batch_reads']==(15 if failed_collection else 32)
    assert cost['final_history_collection']['complete'] is (not failed_collection)
    assert cost['physical_optimizer_updates']==0


def test_engineering_target_cannot_enter_formal_chain():
    node=next(n for n in contract()['nodes'] if n['seed']==163 and n['stage']==2)
    predecessor=next(n for n in contract()['nodes'] if n['id']==node['predecessor'])
    payload=dict(**predecessor,version=contract()['version'],sealed=True,engineering_only=True,updates=4,native={'W':torch.ones(1)},transform=torch.eye(16),memory=[summary(),summary('Philips')])
    runtime.validate_predecessor(payload,node,engineering=True)
    with pytest.raises(ValueError,match='nonformal'):runtime.validate_predecessor(payload,node)
    payload['memory'][0]['domain']='GE'
    with pytest.raises(ValueError):runtime.validate_predecessor(payload,node,engineering=True)


@pytest.mark.parametrize('fail',[False,True])
def test_controller_finite_DAG_and_no_retry(tmp_path,monkeypatch,fail):
    root=tmp_path/'main_style_ssl_mnms_v2_run_mock';sources=tmp_path/'sources';launched=[]
    for n in contract()['nodes']:
        if n['stage']==0:
            p=sources/'jobs'/n['id'];p.mkdir(parents=True);(p/'VALIDATION.json').write_text(json.dumps({'Siemens':{'primary_dice':.5}}))
    monkeypatch.setattr(orchestration,'authorize',lambda c:None)
    monkeypatch.setattr(orchestration,'birth',lambda p:1)
    monkeypatch.setattr(orchestration.time,'sleep',lambda s:None)
    def bootstrap(*args,**kwargs):Path(args[5]).write_text('mock');return {'optimizer_calls':0}
    monkeypatch.setattr(orchestration,'bootstrap_source',bootstrap)
    monkeypatch.setattr(orchestration.subprocess,'check_output',lambda *a,**k:'40000')
    class Process:
        def __init__(self,cmd,**kwargs):
            assert cmd==['python','/neutral/n.py']
            c=json.loads(Path(kwargs['env']['TASK_CONFIG']).read_text());n=c['node'];launched.append(n['id']);self.pid=1
            self.code=1 if fail and n['seed']==163 and n['stage']==1 else 0
            p=Path(c['output']);p.mkdir();(p/'STATUS.json').write_text(json.dumps({'status':'FAILED' if self.code else 'COMPLETE'}))
            if not self.code:(p/'VALIDATION.json').write_text(json.dumps({d:{'primary_dice':.5} for d in ['Siemens','Philips','GE','Canon'][:n['stage']+1]}))
        def poll(self):return self.code
    monkeypatch.setattr(orchestration.subprocess,'Popen',Process)
    orchestration.controller(dict(root=str(root),storage_root=str(tmp_path),gpus={'163':0,'164':1},source_root=str(sources),data='unused',reference='unused',python='python',ops='/neutral',authority_receipt='unused'))
    final=json.loads((root/'FINAL.json').read_text());assert len(final['nodes'])==8 and len(launched)==len(set(launched))
    assert len(launched)==(4 if fail else 6)
    assert sum(v=='BLOCKED' for v in final['nodes'].values())==(2 if fail else 0)
