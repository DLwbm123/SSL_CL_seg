"""One method, inherited splits/orders, reused healthy sources, no ablation."""
from ..main_style_ssl_v1.protocol import MAPS, SHAPES, CLASSES, ORDERS, SEEDS, nodes, options

VERSION = 'MAIN_STYLE_SSL_V2'
GUARD = dict(singular_min=.5, singular_max=2., backtracks=16,
             cone_passes=16, dot_tolerance=1e-7)
MEMORY = dict(stride=100, bootstrap_batches=32, weight=.1,
              prototype_temperature=.1, reliability_prior=(1., 1.))


def contract():
    plan = nodes()
    inherited = sum(n['steps'] for n in plan if n['stage'] == 0)
    return dict(version=VERSION, nodes=plan, guard=GUARD, memory=MEMORY,
                nominal_source_and_target_updates=511600,
                reused_source_updates=inherited, new_formal_optimizer_calls=511600-inherited,
                source_policy='reuse healthy V1 sealed sources; training-L-only memory bootstrap, zero optimizer calls',
                execution_authorized=False, ablation=False, test_accessor=False,
                selection='fixed final checkpoints', seeds=list(SEEDS),
                target_mri_first_conv='frozen effective weight; A/B absent from optimizer',
                native_rank_admission='certified c234bcd retained-rank repair for all V2 targets; unchanged SVD tolerance and protected dimension',
                pseudo_confidence=.7, style_alpha=.2, warmup_fraction=.2, U_ramp_fraction=.2,
                entry_support='geometry and confident entry, independent of EMA/admission; class balanced',
                historical_memory='per-domain/class native-coordinate sensitivity and post-F prototypes; no images',
                update='one Adam invocation, constrain actual displacement before EMA/commit; no retry',
                zero_displacement='counts as one optimizer invocation; report separately, never replay')


def frozen_identity(directory=None):
    import hashlib
    import json
    from pathlib import Path
    root=Path(directory) if directory is not None else Path(__file__).parent
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dependencies={}
    for package in ('main_style_ssl_v1','five_frameworks_v1','lctx_weight_memory_v0_1'):
        dependencies.update({package+'/'+p.name:p.read_text() for p in sorted((root.parent/package).glob('*.py'))})
    return dict(protocol_digest=digest(contract()),
                implementation_digest=digest({p.name:p.read_text() for p in sorted(root.glob('*.py'))}),
                dependency_digest=digest(dependencies))


def verify_frozen(directory=None):
    import json
    from pathlib import Path
    root=Path(directory) if directory is not None else Path(__file__).parent
    receipt=json.loads((root/'FREEZE.json').read_text())
    if receipt.get('version')!=VERSION or any(receipt.get(k)!=v for k,v in frozen_identity(root).items()):
        raise ValueError('frozen V2 implementation/protocol mismatch')
    return receipt
