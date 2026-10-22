import argparse
import json
from pathlib import Path
import sys
import torch
from . import budgets
from .budgets import ResourceLedger, Counters
from .config import Config


def main(argv=None):
    parser=argparse.ArgumentParser(description='HBSI-Seg bounded implementation qualification')
    parser.add_argument('command',choices=('validate-spec','unit-tests','synthetic-qualification','real-smoke','train-source','train-stream','evaluate'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run-id',default='initial')
    parser.add_argument('--test-id',action='append',choices=[f'T{i:02d}' for i in range(1,19)])
    parser.add_argument('--authorize-real-training',action='store_true')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--budget-seconds',type=float)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg=Config.load(args.config) if args.config else Config()
    if args.command=='validate-spec':
        from .stream_state import StreamState
        from .diagnostics import parameter_audit
        s=StreamState(cfg);s.begin_task('schema-check','0'*64)
        print(json.dumps({'status':'SPECIFIED','config':cfg.to_dict(),'audit':parameter_audit(s)},indent=2))
        return 0
    if args.command in ('train-source','train-stream'):
        if not args.authorize_real_training or args.protocol is None or args.budget_seconds is None:
            parser.error('real training requires explicit authorization, protocol and wall-clock budget')
    if args.output is None:
        parser.error('--output required')
    counter=Counters();budgets.ACTIVE_COUNTER=counter
    ledger=ResourceLedger(args.output)
    try:
        if args.command=='unit-tests':
            from .tests import run_tests
            result=run_tests(ledger,args.run_id,args.test_id)
            return 0 if all(r['status']=='PASS' for r in result) else 1
        from .runner import dispatch
        return dispatch(args,cfg,ledger,counter)
    finally:
        previous=ledger.data.get('counters',{})
        for k,v in counter.export().items():previous[k]=previous.get(k,0)+v
        ledger.data['counters']=previous
        ledger.flush()


if __name__=='__main__':
    sys.exit(main())
