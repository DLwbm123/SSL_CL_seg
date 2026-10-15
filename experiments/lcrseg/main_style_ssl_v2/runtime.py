"""Factory/lifecycle only; the V1 worker and StageTrainer loop are reused."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from ..main_style_ssl_v1 import runtime as v1
from ..five_frameworks_v1.checkpoint import atomic_save
from ..five_frameworks_v1.numerics import finite
from .protocol import VERSION, MEMORY, nodes, options, contract, verify_frozen, MAPS, SHAPES, ORDERS
from .model import Parent, MainModel, MainTrainer
from .checkpoint import save, restore


def make_trainer(native, node, provider, previous=None, history=(), initialize=True, execution=None):
    parent = Parent(native, node['seed'], provider.stage_source, node['stage'] == 0)
    model = MainModel(parent, MAPS[node['dataset']], previous, initialize)
    return MainTrainer(model, provider, options(node['steps'], node['stage']==0),
                       history=history, initialize=initialize, execution=execution)


def require_v2_destination(path):
    parents=Path(path).resolve().parents
    if any(p.name.startswith('main_style_ssl_v1_') for p in parents) or not any(p.name.startswith('main_style_ssl_v2_') for p in parents):
        raise PermissionError('isolated V2 root required; writes inside V1 roots forbidden')


def validate_source(payload, node, device):
    if node not in nodes() or node['stage'] != 0:
        raise ValueError('not a frozen source node')
    if payload.get('sealed') is not True or payload['updates'] != node['steps'] or any(payload.get(k)!=v for k,v in node.items()):
        raise ValueError('source is not the completed exact V1 source')
    finite(payload['native'],'reused source')
    if not torch.equal(payload['transform'],torch.eye(16,device=device)):
        raise ValueError('source transform must be identity; no target imports')


def setup(node, data, reference, device, predecessor=None, initialize=True):
    device = torch.device(device); ds = node['dataset']
    if node['stage'] == 0:
        raise PermissionError('V2 does not retrain sources; use explicit training-L-only source import')
    if predecessor is None:
        raise ValueError('missing V2 sealed predecessor')
    sealed = torch.load(predecessor, map_location=device, weights_only=False)
    expected = next(n for n in nodes() if n['id']==node['predecessor'])
    if sealed.get('version') != VERSION or sealed.get('sealed') is not True or sealed['updates'] != expected['steps']:
        raise ValueError('wrong version/incomplete V2 predecessor')
    if any(sealed.get(k) != v for k,v in expected.items()):
        raise ValueError('wrong predecessor identity')
    if [h['domain'] for h in sealed.get('memory',[])] != list(ORDERS[ds][node['order']-1][:node['stage']]):
        raise ValueError('missing/wrong-order historical domain memory')
    native = v1.build(reference,device,node['seed'],input_channels=SHAPES[ds][0],
                      num_classes=len(MAPS[ds]),domain=ORDERS[ds][0][0])
    native.load_state_dict(sealed['native'])
    lineage = dict(kind='sealed_V2_predecessor', dataset=ds, domain=sealed['domain'],
                   seed=node['seed'], stage=sealed['stage'], order=sealed['order'], node=sealed['id'])
    execution = v1.permit(node,node.get('scope','formal'))
    provider = v1.CurrentDomain(data,ds,node['domain'],node['seed'],node['order'],node['stage'],lineage,device,execution)
    return make_trainer(native,node,provider,sealed['transform'],sealed['memory'],initialize,execution)


def seal(trainer, node, path):
    if trainer.requires_restore or trainer.step != node['steps']:
        raise ValueError('only the committed fixed final stage may be sealed')
    history = trainer.memory.export_site(node['domain'], trainer.model.native_values())
    deployment = trainer.model.deploy()
    atomic_save(dict(**node, native=deployment.parent.native.state_dict(), transform=deployment.transform.detach(),
                     version=VERSION, sealed=True, precision='FP32', updates=trainer.step,
                     memory=history, guard_totals=trainer.guard_totals),path)
    return deployment


def bootstrap_source(source, node, data, reference, device, output):
    """Explicit future operation, 32 training-L batches, no optimizer or U/val access."""
    if node not in nodes() or node['stage'] != 0:
        raise ValueError('not a frozen source node')
    output = Path(output)
    require_v2_destination(output)
    if output.exists():
        raise FileExistsError('source import is create-only')
    device = torch.device(device); ds = node['dataset']
    payload = torch.load(source,map_location=device,weights_only=False)
    validate_source(payload,node,device)
    native = v1.build(reference,device,node['seed'],input_channels=SHAPES[ds][0],num_classes=len(MAPS[ds]),domain=node['domain'])
    native.load_state_dict(payload['native'])
    lineage = dict(kind='reused_healthy_V1_source',dataset=ds,domain=node['domain'],seed=node['seed'],node=node['id'])
    execution = v1.permit(node,'smoke')
    provider = v1.CurrentDomain(data,ds,node['domain'],node['seed'],1,0,lineage,device,execution)
    trainer = make_trainer(native,node,provider,execution=execution)
    trainer.bootstrap = True
    for cursor in range(MEMORY['bootstrap_batches']):
        trainer.cursor = cursor
        trainer.losses()
        trainer.memory.commit(trainer.draft)
        trainer.draft = None
    assert trainer.step == trainer.physical_optimizer_updates == provider.u_reads == 0
    history = trainer.memory.export_site(node['domain'],trainer.model.native_values())
    atomic_save(dict(**node,native=payload['native'],transform=payload['transform'],sealed=True,
                     version=VERSION,precision='FP32',updates=node['steps'],memory=history,
                     inherited_optimizer_updates=node['steps'],new_optimizer_updates=0,
                     bootstrap_L_reads=provider.l_reads,bootstrap_U_reads=0),output)


def worker(config):
    # Frozen implementation has no launch authority. A later explicit user instruction
    # must produce a separately recorded authority receipt; it is absent in this delivery.
    authority = config.get('authority_receipt')
    if not authority:
        raise PermissionError('V2 FROZEN_NOT_LAUNCHED: separate user launch authorization required')
    frozen=verify_frozen()
    receipt = json.loads(Path(authority).read_text())
    if receipt.get('version') != VERSION or receipt.get('operation') != 'authorized_formal_targets' or not receipt.get('user_instruction') or receipt.get('protocol_digest') != frozen['protocol_digest'] or receipt.get('implementation_digest') != frozen['implementation_digest']:
        raise PermissionError('invalid future launch authority')
    node = config['node']; out = Path(config['output'])
    require_v2_destination(out)
    if node not in nodes() or node['stage'] == 0:
        raise PermissionError('fresh V2 target root required; V1 roots and source retraining forbidden')
    if out.exists():
        raise FileExistsError('one formal attempt per node; no automatic retry')
    with patch.multiple(v1, setup=setup, seal=seal, save=save, restore=restore):
        return v1.worker(config)
