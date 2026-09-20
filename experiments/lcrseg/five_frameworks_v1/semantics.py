"""Executable resume binding derived from the actual model and provider."""
import hashlib
import math
from pathlib import Path
import torch
from .gate import digest

DEFAULTS={'total_steps':5,'warmup_fraction':.2,'U_ramp_fraction':.2,'lr':.01,'weight_decay':0.,
          'parent_lr_multiplier':1.,'lr_B_over_A':1.,'feature_lr_multiplier':1.,
          'feature_weight_decay_multiplier':1.,'lambda_U':.5,'lambda_JML':.25,
          'lambda_structure':.1,'lambda_shape':.1,'lambda_SWD':.05,
          'probe_layers':'last_two_parent_adapters','PAS_confidence':.7,'PAS_cosine':.5,
          'kappa':.5,'prototype_uncertainty_mix':0.,'inner_steps':3,'trust_fraction':.05}
NON_SEMANTIC={'output_path','log_every'}  # No device/precision migration qualified yet.


def resolve_options(options):
    options=dict(options or {})
    unknown=set(options)-set(DEFAULTS)-NON_SEMANTIC-{'rank_ratio'}
    if unknown:raise ValueError('unknown semantic options: '+str(sorted(unknown)))
    result={**DEFAULTS,**options}
    for k,v in result.items():
        if k in NON_SEMANTIC or isinstance(v,str):continue
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('invalid option '+k)
        if v<0 and k!='PAS_cosine':raise ValueError('negative option '+k)
    if type(result['total_steps']) is not int or result['total_steps']<=0:raise ValueError('invalid total_steps')
    if type(result['inner_steps']) is not int:raise ValueError('invalid inner_steps')
    for key in ('warmup_fraction','U_ramp_fraction','PAS_confidence','prototype_uncertainty_mix'):
        if not 0<=result[key]<=1:raise ValueError('fraction out of range: '+key)
    if not -1<=result['PAS_cosine']<=1:raise ValueError('invalid cosine threshold')
    if result['probe_layers'] not in ('last_two_parent_adapters','all_parent_adapters'):raise ValueError('invalid probe layer choice')
    return result


def tensor_fingerprint(state):
    """Synthetic entry tensors only; never deserialize a real checkpoint here."""
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(str((name,str(x.dtype),tuple(x.shape))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def runtime_fingerprint():
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})


def binding(trainer):
    model=trainer.model;parent=model.parent
    backend=trainer.cwmi
    backend_id=None if backend is None else (backend.semantic_identity() if hasattr(backend,'semantic_identity')
                                            else {'fixture_class':type(backend).__module__+'.'+type(backend).__qualname__})
    return {**({'loss_contract':trainer.loss_contract()} if hasattr(trainer,'loss_contract') else {}),
            'schema':2,'code':runtime_fingerprint(),'options':{k:v for k,v in trainer.options.items() if k not in NON_SEMANTIC},
            'family':model.family,'rank_ratio':model.rank_ratio,
            'sidecar_rank':None if model.sidecar is None else model.sidecar.q.shape[1],
            'provider':trainer.provider.semantic_metadata(),
            'parent':parent.semantic_metadata(),'stage_entry_tensors':trainer.entry_fingerprint,
            'stage_source':trainer.provider.stage_source,'backend':backend_id,
            'parameters':{n:{'shape':list(p.shape),'dtype':str(p.dtype),'trainable':p.requires_grad} for n,p in model.named_parameters()},
            'student_modes':{n:m.training for n,m in model.named_modules()},
            'teacher_modes':{n:m.training for n,m in trainer.ema.named_modules()},
            'optimizer':{'class':type(trainer.optimizer).__module__+'.'+type(trainer.optimizer).__qualname__,
                         'groups':optimizer_group_binding(trainer)}}


def optimizer_group_binding(trainer):
    names={id(p):n for n,p in trainer.model.named_parameters()}
    return [{**{k:v for k,v in g.items() if k not in ('params','lr')},
             'initial_lr':g.get('initial_lr',g['lr']),'params':[names[id(p)] for p in g['params']]}
            for g in trainer.optimizer.param_groups]
