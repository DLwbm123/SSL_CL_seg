"""Route each declared teacher timescale through the shared dense EMA update."""


def bind(t):
    original=t.model.parent.update_dense_ema
    def update(teacher,validate_only=False):
        freeze=t.options.get('freeze_U_teacher',False)
        if type(freeze) is not bool:raise ValueError('freeze_U_teacher must be bool')
        decay=t.options.get('teacher_decay',.99)
        if not 0<=decay<=1:raise ValueError('invalid teacher decay')
        if validate_only:return original(teacher,validate_only=True,decay=decay)
        key='fixed_teacher_skips' if freeze else 'dynamic_teacher_updates'
        t.telemetry[key]=t.telemetry.get(key,0)+1
        if not freeze:return original(teacher,decay=decay)
    t.model.parent.update_dense_ema=update
    return t


def check():
    from types import SimpleNamespace,MethodType
    import copy
    import torch
    from torch import nn
    from experiments.lcrseg.five_frameworks_v1.native_parent import NativeLRParent
    student=nn.Sequential(nn.Linear(2,1));teacher=copy.deepcopy(student)
    with torch.no_grad():
        for p in student.parameters():p.fill_(3.)
        for p in teacher.parameters():p.fill_(1.)
    for model in (student,teacher):
        model.register_buffer('counter',torch.tensor(7 if model is student else 2))
        model.register_buffer('grad_update',torch.tensor(4. if model is student else 1.))
    parent=SimpleNamespace(native=student)
    parent.update_dense_ema=MethodType(NativeLRParent.update_dense_ema,parent)
    target=SimpleNamespace(native=teacher);start=copy.deepcopy(teacher.state_dict())
    for decay in (.99,.999):
        teacher.load_state_dict(start)
        parent.update_dense_ema(target,validate_only=True,decay=decay)
        assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in start.items())
        parent.update_dense_ema(target,decay=decay)
        for k,v in teacher.state_dict().items():
            expected=student.state_dict()[k] if k in ('counter','grad_update') else start[k].clone().mul_(decay).add_(student.state_dict()[k],alpha=.01 if decay==.99 else 1-decay)
            assert torch.equal(v,expected)
    teacher.load_state_dict(start);parent.update_dense_ema(target)
    assert torch.equal(teacher[0].weight,start['0.weight'].clone().mul_(.99).add_(student[0].weight,alpha=.01))
    for decay in (float('nan'),float('inf'),-1.,1.1):
        try:parent.update_dense_ema(target,decay=decay)
        except ValueError:pass
        else:raise AssertionError('invalid decay accepted')
    t=bind(SimpleNamespace(model=SimpleNamespace(parent=parent),options=dict(freeze_U_teacher=True,teacher_decay=.999),telemetry={}))
    teacher.load_state_dict(start);t.model.parent.update_dense_ema(target,validate_only=True);t.model.parent.update_dense_ema(target)
    assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in start.items()) and t.telemetry['fixed_teacher_skips']==1
    t.options['freeze_U_teacher']=False;t.model.parent.update_dense_ema(target)
    assert torch.equal(teacher[0].weight,start['0.weight'].clone().mul_(.999).add_(student[0].weight,alpha=1-.999))
    assert t.telemetry['dynamic_teacher_updates']==1
    print('PASS actual dense EMA default/slow equations, buffers, validation-only, fixed gate and invalid decay')


if __name__=='__main__':check()
