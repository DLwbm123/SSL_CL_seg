import torch


def validate_public_summary(value):
    forbidden={'patient','patients','patient_id','case_id','image','images','label','labels','checkpoint','raw_log'}
    if isinstance(value,dict):
        if forbidden.intersection(value):
            raise PermissionError('private sample content in public summary')
        for v in value.values():validate_public_summary(v)
    elif isinstance(value,(list,tuple)):
        for v in value:validate_public_summary(v)
    elif isinstance(value,str) and any(s in value for s in ('/home/','/Users/','/data_nas/','/data/bmw/','BEGIN PRIVATE KEY')):
        raise PermissionError('private path/credential in public summary')
    elif isinstance(value,torch.Tensor):
        raise PermissionError('raw tensor in public summary')


def consistency(logweights):
    if not torch.isfinite(logweights).all():
        raise FloatingPointError('nonfinite diagnostic')
    v = logweights.detach()
    return dict(delta_variance=v.var(-1, unbiased=False),
                local_ess=1 / v.softmax(-1).square().sum(-1),
                scope='fixed_theta_and_a', contributes_to_training=False)


def parameter_audit(state):
    from .stream_state import digest
    groups = {'global-generative': state.q, 'amortizer': state.phi}
    if state.current is not None:
        groups['current-domain-variational'] = state.current
    report = {}; seen = set()
    for name, module in groups.items():
        rows = []
        for n, p in module.named_parameters():
            if id(p) in seen:
                raise RuntimeError('parameter belongs to multiple groups')
            seen.add(id(p))
            rows.append(dict(name=n, shape=list(p.shape), count=p.numel(), bytes=p.numel() * p.element_size(), trainable=p.requires_grad))
        report[name] = rows
    if list(state.model.parameters()):
        raise RuntimeError('unregistered generative parameters')
    return dict(groups=report, global_random_dimensions=state.q.mean.numel(),
                global_layout=state.q.layout, projection_hash=state.projection.fingerprint(),
                fixed_basis_hash=digest(state.model.basis), fixed_basis_order='u-major,v-minor,exclude-DC,orthonormal-DCT-II',
                global_scale_floor_fraction=float((state.q.sd < 1.01 * state.config.scale_floor).double().mean()),
                domain_memory_bytes={k: sum(v.numel()*v.element_size() for v in m.state_dict().values()) for k,m in state.memories.items()},
                no_generative_bypass=True)
