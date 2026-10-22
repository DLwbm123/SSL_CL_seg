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
    parser.add_argument('command',choices=('validate-spec','unit-tests','revision-checks','fixed-parameter-qualification','synthetic-qualification','real-smoke','train-source','train-stream','evaluate'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run-id',default='initial')
    parser.add_argument('--test-id',action='append',choices=[f'T{i:02d}' for i in range(1,19)])
    parser.add_argument('--authorize-real-training',action='store_true')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--budget-seconds',type=float)
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--unit-receipt',type=Path)
    parser.add_argument('--revision-receipt',type=Path)
    parser.add_argument('--revision-id',action='append',choices=[f'R{i:02d}' for i in range(1,10)])
    parser.add_argument('--max-outer-updates',type=int,default=1000)
    args=parser.parse_args(argv)
    if args.resume_checkpoint is not None and args.command not in ('train-source','train-stream'):
        parser.error('--resume-checkpoint is only supported for an explicitly authorized training stage')
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
    if not 0<=args.max_outer_updates<=1000:parser.error('invalid outer update cap')
    ledger=ResourceLedger(args.output,max_updates=args.max_outer_updates)
    try:
        if args.command=='revision-checks':
            from .revision_checks import run_checks
            from .qualification import write_receipt, fingerprint
            result=run_checks(ledger,args.run_id,args.revision_id)
            complete={r['id'] for r in result}=={f'R{i:02d}' for i in range(1,10)} and all(r['status']=='PASS' for r in result)
            report=ledger.root/f'{args.run_id}_REVISION_REPORT.json'
            with report.open('x') as f:
                json.dump({'scientific_state':'REVISION_PASS' if complete else 'REVISION_FAIL','results':result},f,indent=2)
            write_receipt(ledger.root/f'{args.run_id}_REVISION_RECEIPT.json','REVISION',cfg,report,fingerprint(result))
            return 0 if all(r['status']=='PASS' for r in result) else 1
        if args.command=='fixed-parameter-qualification':
            from .qualification import require_receipt
            for kind,path in (('UNIT',args.unit_receipt),('REVISION',args.revision_receipt)):
                if path is None:raise PermissionError('unit and revision receipts required before oracle diagnosis')
                receipt=json.loads(path.read_text())
                require_receipt(path,kind,cfg,receipt.get('evidence_fingerprint'))
            from .fixed_inference import run
            result=run(ledger,counter,args.run_id)
            with (ledger.root/f'{args.run_id}_FIXED_PARAMETER_REPORT.json').open('x') as f:json.dump(result,f,indent=2)
            return 0 if result['scientific_state']=='ORACLE_LOCAL_QUALIFIED' else 2
        if args.command=='unit-tests':
            from .tests import run_tests
            result=run_tests(ledger,args.run_id,args.test_id)
            from .qualification import write_receipt, fingerprint
            complete={r['id'] for r in result}=={f'T{i:02d}' for i in range(1,19)} and all(r['status']=='PASS' for r in result)
            report=ledger.root/f'{args.run_id}_UNIT_REPORT.json'
            with report.open('x') as f:
                json.dump({'scientific_state':'UNIT_PASS' if complete else 'UNIT_FAIL','results':result},f,indent=2)
            write_receipt(ledger.root/f'{args.run_id}_UNIT_RECEIPT.json','UNIT',cfg,report,fingerprint(result))
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
