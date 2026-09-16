"""Actual entry points. Production requires a fresh external review and user launch."""
import argparse
import json
import os
from .protocol import read,DOC,prepare,validate_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','plan','qualify','run','report'])
    parser.add_argument('--mode',choices=['cpu','cuda','smoke'],default='cpu')
    parser.add_argument('--config',default=os.environ.get('EXEC_CONFIG'))
    args=parser.parse_args()
    if args.command=='prepare':result=prepare()
    elif args.command=='plan':
        p=validate_plan(read(DOC/'PLAN.json'));result={k:p[k] for k in ('state','counts','caps','plan_sha256')}
    elif args.command=='qualify' and args.mode=='cpu':
        from .tests import run_tests
        result=run_tests()
    else:
        if not args.config:parser.error('private --config/EXEC_CONFIG required; no production authority in repository')
        from .execution import qualify,run,report
        config=read(args.config)
        if args.command=='qualify':result=qualify(config,args.mode)
        elif args.command=='run':result=run(config)
        else:result=report(config['run_root'],config['execution_commit'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
