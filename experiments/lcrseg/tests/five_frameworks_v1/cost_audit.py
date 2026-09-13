"""Test-only instrumentation; never imported by production training modules."""
import functools,json,os
import torch


def install():
    path=os.environ.get('SSLCL5_TEST_COST_LOG')
    if not path:return
    for obj,name,event in [(torch.optim.Adam,'step','Adam_step'),(torch.autograd,'grad','autograd_grad'),(torch.autograd,'backward','autograd_backward')]:
        original=getattr(obj,name)
        if getattr(original,'_cost_audited',False):continue
        def wrap(fn,event):
            @functools.wraps(fn)
            def call(*a,**kw):
                try:
                    result=fn(*a,**kw);status='completed';return result
                except BaseException:
                    status='raised';raise
                finally:
                    with open(path,'a') as f:f.write(json.dumps({'pid':os.getpid(),'event':event,'status':status})+'\n')
            call._cost_audited=True
            return call
        setattr(obj,name,wrap(original,event))
