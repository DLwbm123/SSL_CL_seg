"""Exercise the actual shared coefficient gate, including invalid windows."""
from types import SimpleNamespace
from experiments.qprompt_rl_v1.v3b_lrref_endpoint.engine import Trainer
from .protocol import check,WEIGHTS,WINDOWS,ACTIVE,H


def main():
    check()
    for method,weight in WEIGHTS.items():
        obj=SimpleNamespace(options=dict(lambda_U=weight,**WINDOWS[method]),step=0)
        actual=[]
        for step in range(H):
            obj.step=step;actual.append(Trainer.u_weight(obj))
        assert sum(v>0 for v in actual)==ACTIVE[method]
        assert sum(actual)==weight*ACTIVE[method]
        if method=='EARLY_05':assert actual[599]==.5 and actual[600]==0
        if method=='LATE_05':assert actual[599]==0 and actual[600]==.5
    assert Trainer.u_weight(SimpleNamespace(options={},step=900))==.5
    for options in ({'u_start':-1},{'u_start':600,'u_stop':600},{'lambda_U':float('nan')}):
        try:Trainer.u_weight(SimpleNamespace(options=options,step=0))
        except ValueError:pass
        else:raise AssertionError('invalid schedule accepted')
    print('PASS actual Trainer U windows, 599/600 boundary, default compatibility and invalid input rejection')


if __name__=='__main__':main()
