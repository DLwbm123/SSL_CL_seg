"""Bind the one experimental gate to an already constructed native teacher."""


def bind(t):
    original=t.model.parent.update_dense_ema
    def update(teacher,validate_only=False):
        freeze=t.options.get('freeze_U_teacher',False)
        if type(freeze) is not bool:raise ValueError('freeze_U_teacher must be bool')
        # Existing commit validation remains active in both arms.
        if validate_only:return original(teacher,validate_only=True)
        key='fixed_teacher_skips' if freeze else 'dynamic_teacher_updates'
        t.telemetry[key]=t.telemetry.get(key,0)+1
        if not freeze:return original(teacher)
    t.model.parent.update_dense_ema=update
    return t


def check():
    from types import SimpleNamespace
    import torch
    calls=[]
    def native(teacher,validate_only=False):
        calls.append(validate_only)
        if not validate_only:teacher.mul_(.99).add_(1.,alpha=.01)
    t=bind(SimpleNamespace(model=SimpleNamespace(parent=SimpleNamespace(update_dense_ema=native)),options={},telemetry={}))
    q=torch.tensor([.2]);expected=q.clone().mul_(.99).add_(1.,alpha=.01)
    t.model.parent.update_dense_ema(q);assert torch.equal(q,expected)
    t.options['freeze_U_teacher']=True;before=q.clone()
    t.model.parent.update_dense_ema(q,validate_only=True);t.model.parent.update_dense_ema(q)
    assert torch.equal(q,before) and calls==[False,True]
    assert t.telemetry==dict(dynamic_teacher_updates=1,fixed_teacher_skips=1)
    t.options['freeze_U_teacher']='yes'
    try:t.model.parent.update_dense_ema(q)
    except ValueError:pass
    else:raise AssertionError('invalid gate accepted')
    print('PASS default native update, fixed-state gate, commit validation and invalid option')


if __name__=='__main__':check()
