"""One M&Ms protocol; constants are design choices, never validation-selected."""
from ..main_style_ssl_v1.protocol import nodes as v1_nodes

VERSION = 'MAIN_STYLE_SSL_MNMS_V2'
MEMORY_BATCHES = 32
MEMORY_WEIGHT = .1
ORTHOGONAL_TOLERANCE = 2e-5
ELASTIC_STRENGTH = 1.
DUAL_SWEEPS = 64


def contract():
    nodes = [n for n in v1_nodes() if n['dataset'] == 'mnms']
    inherited = sum(n['steps'] for n in nodes if n['stage'] == 0)
    target = sum(n['steps'] for n in nodes if n['stage'] > 0)
    assert (len(nodes), inherited, target) == (8, 87200, 209800)
    return dict(version=VERSION, nodes=nodes, inherited_source_updates=inherited,
                new_formal_optimizer_calls=target, nominal_updates=inherited+target,
                execution_authorized=False, training_runner_implemented=True,
                engineering_costs='separate actual-call ledger; never counted as formal updates', test_accessor=False, ablation=False,
                source_policy='reuse exact V1 Siemens sources; never import a target',
                transform='orthogonal rank4 Cayley G; cumulative F orthogonal',
                memory='immutable per-domain/class final-anchor CE Jacobian and squared-gradient summary',
                memory_batches_per_domain=MEMORY_BATCHES, memory_weight=MEMORY_WEIGHT,
                elastic_strength=ELASTIC_STRENGTH, dual_sweeps=DUAL_SWEEPS,
                orthogonality_numerical_tolerance=ORTHOGONAL_TOLERANCE,
                future_memory_L_batch_reads=2*4*MEMORY_BATCHES,
                future_memory_L_image_exposures=2*2*4*MEMORY_BATCHES,
                extra_source_bank_L_image_exposures=4,
                selection='fixed final; full stage by seen-domain validation; no best selection')


def worker(*args, **kwargs):
    from .runtime import worker as run
    if args:return run(*args,**kwargs)
    return run(kwargs)


def ready_identity():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).parent;files={}
    for package in ('main_style_ssl_mnms_v2','main_style_ssl_v2','main_style_ssl_v1','five_frameworks_v1','lctx_weight_memory_v0_1'):
        files.update({package+'/'+p.name:p.read_text(encoding='utf-8') for p in sorted((root.parent/package).glob('*.py'))})
    return hashlib.sha256(json.dumps(dict(protocol=contract(),files=files),sort_keys=True).encode()).hexdigest()


def verify_ready():
    import json
    from pathlib import Path
    receipt=json.loads((Path(__file__).parent/'TRAIN_READY.json').read_text(encoding='utf-8'))
    if receipt.get('status')!='TRAIN_READY_NOT_LAUNCHED' or receipt.get('ready_identity')!=ready_identity():raise PermissionError('training-ready code/qualification mismatch')
    return receipt
