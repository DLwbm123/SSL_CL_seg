"""Factories and lifecycle bindings; one shared training loop, no implicit launch."""
import json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from ..main_style_ssl_v1 import runtime as base
from ..main_style_ssl_v1.protocol import options, ORDERS
from ..main_style_ssl_v2.runtime import validate_source
from ..five_frameworks_v1.checkpoint import atomic_save
from .protocol import VERSION, contract
from .model import Parent,MainModel,MainTrainer
from .memory import History,finite
from .collector import collect
from .protection import require_rotation
from .checkpoint import save,restore


def destination(path,engineering=False):
    prefix='main_style_ssl_mnms_v2_engineering_' if engineering else 'main_style_ssl_mnms_v2_run_'
    p=Path(path).resolve()
    if not any(a.name.startswith(prefix) for a in (p,*p.parents)):raise PermissionError('isolated M&Ms V2 destination required')
    return p


def validate_predecessor(payload,node,engineering=False):
    expected=next(n for n in contract()['nodes'] if n['id']==node['predecessor'])
    if payload.get('version')!=VERSION or payload.get('sealed')is not True or any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('wrong predecessor identity')
    if engineering:
        if payload.get('engineering_only')is not True:raise ValueError('engineering predecessor required')
    elif payload.get('engineering_only')is not False or payload.get('updates')!=expected['steps']:raise ValueError('incomplete/nonformal predecessor')
    if payload.get('updates')!=expected['steps'] and not (engineering and expected['stage']>0 and 0<payload.get('updates',0)<expected['steps']):raise ValueError('wrong predecessor updates')
    History(payload['memory'],ORDERS['mnms'][0][:node['stage']]);require_rotation(payload['transform'])
    if not finite(payload['native']):raise ValueError('nonfinite predecessor')
    return expected


def setup(node,data,reference,device,predecessor=None,initialize=True,engineering=False):
    if node not in contract()['nodes'] or not node['stage']:raise PermissionError('exact M&Ms target required; no source training')
    device=torch.device(device);sealed=torch.load(predecessor,map_location=device,weights_only=False)
    validate_predecessor(sealed,node,engineering)
    native=base.build(reference,device,node['seed'],input_channels=1,num_classes=4,domain='Siemens');native.load_state_dict(sealed['native'])
    lineage=dict(kind=VERSION,dataset='mnms',domain=sealed['domain'],seed=node['seed'],stage=sealed['stage'],order=node['order'],node=sealed['id'],engineering_only=engineering)
    execution=base.permit(node,'smoke' if engineering else 'formal')
    provider=base.CurrentDomain(data,'mnms',node['domain'],node['seed'],node['order'],node['stage'],lineage,device,execution)
    model=MainModel(Parent(native,node['seed'],lineage,False),previous=sealed['transform'],initialize_basis=initialize)
    return MainTrainer(model,provider,options(node['steps'],False),history=sealed['memory'],initialize=initialize,execution=execution)


def bootstrap_source(source,node,data,reference,device,output,engineering=True,progress=None):
    if node not in contract()['nodes'] or node['stage']:raise PermissionError('exact Siemens source required')
    path=destination(output,engineering)
    if path.exists():raise FileExistsError('create-only source import')
    device=torch.device(device);payload=torch.load(source,map_location=device,weights_only=False);validate_source(payload,node,device)
    native=base.build(reference,device,node['seed'],input_channels=1,num_classes=4,domain='Siemens');native.load_state_dict(payload['native'])
    from ..five_frameworks_v1.model import Deployment
    deployment=Deployment(Parent(native,node['seed'],{'domain':'Siemens'},True),payload['transform']).eval().requires_grad_(False)
    anchor=deployment.parent.native_values();anchor['__F__']=deployment.transform
    site,cost=collect(deployment,anchor,node,data,progress)
    from types import SimpleNamespace
    from ..main_style_ssl_v1.data import Accessor,CurrentDomain
    from ..five_frameworks_v1.reliability import CurrentPrototypes
    source_model=MainModel(deployment.parent,previous=payload['transform'])
    teacher=source_model.teacher();access=Accessor(data,'mnms','Siemens','train_labeled')
    provider=SimpleNamespace(seed=node['seed'],order=1,stage=0,dataset='mnms',device=device)
    x,y,_=CurrentDomain._batch(provider,access,0,'prototype_initialization')
    with torch.no_grad():bank,support,*_=CurrentPrototypes(4,16).estimate(teacher,x,y)
    atomic_save(dict(**node,version=VERSION,sealed=True,native=payload['native'],transform=payload['transform'],memory=[site],updates=node['steps'],new_optimizer_updates=0,inherited_optimizer_updates=node['steps'],engineering_only=engineering,collector_cost=cost,source_bank=dict(values=bank,support=support),source_bank_cost=dict(L_batch_reads=1,L_image_exposures=2,optimizer_calls=0)),path)
    return cost


def seal(trainer,node,path,engineering=False,progress=None):
    destination(path,engineering)
    if trainer.requires_restore or (not engineering and trainer.step!=node['steps']) or trainer.step<=0:raise ValueError('only committed fixed endpoint may be sealed')
    if Path(path).exists():raise FileExistsError('create-only seal')
    if progress is None:
        progress=lambda value:base.write(Path(path).parent/'HISTORY_COLLECTION_STATUS.json',value)
    deployment=trainer.model.deploy();site,cost=collect(deployment,trainer.model.native_values(),node,trainer.provider._l.data,progress)
    history=trainer.history.state()+[site];History(history,ORDERS['mnms'][0][:node['stage']+1])
    atomic_save(dict(**node,version=VERSION,sealed=True,native=deployment.parent.native.state_dict(),transform=deployment.transform.detach(),memory=history,updates=trainer.step,engineering_only=engineering,collector_cost=cost),path)
    trainer.final_summary_cost=cost
    return deployment


def authorize(config):
    receipt_path=config.get('authority_receipt')
    if not receipt_path:raise PermissionError('separate explicit formal training authorization required')
    receipt=json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    from .protocol import verify_ready
    ready=verify_ready()
    if receipt.get('version')!=VERSION or receipt.get('operation')!='authorized_formal_targets' or not receipt.get('user_instruction') or receipt.get('ready_identity')!=ready['ready_identity'] or receipt.get('new_formal_budget')!=209800 or receipt.get('datasets')!=['mnms'] or receipt.get('seeds')!=[163,164]:raise PermissionError('invalid formal authority')
    storage=Path(config['storage_root']).resolve();output=Path(config.get('root',config.get('output'))).resolve()
    if not output.is_relative_to(storage) or subprocess.check_output(['findmnt','-T',str(storage),'-no','FSTYPE'],text=True).strip() not in ('nfs','nfs4'):raise PermissionError('verified NAS output/storage required')
    with tempfile.TemporaryFile(dir=storage) as probe:
        probe.write(b'NAS');probe.flush();probe.seek(0)
        if probe.read()!=b'NAS':raise OSError('NAS write/read admission failed')
    return receipt


def worker(config):
    authorize(config);node=config['node'];out=destination(config['output'])
    if node not in contract()['nodes'] or not node['stage']:raise PermissionError('source retraining forbidden')
    original_write=base.write
    def write(path,value):
        if Path(path).name=='COSTS.json':
            progress=out/'HISTORY_COLLECTION_STATUS.json'
            if progress.exists():
                row=json.loads(progress.read_text())
                value['final_history_collection']=dict(L_batch_reads=row['batch'],L_image_exposures=row['L_image_exposures'],optimizer_calls=0,complete=(out/'sealed.pt').exists())
        return original_write(path,value)
    with patch.multiple(base,setup=setup,seal=seal,save=save,restore=restore,write=write):return base.worker(config)
