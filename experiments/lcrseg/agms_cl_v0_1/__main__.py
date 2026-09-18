"""Use a neutral stdin/env entry for long processes; see RUNBOOK."""
import argparse
import json
from .protocol import freeze,canonical_plan,read


def main(argv=None):
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('plan')
    test=sub.add_parser('test');test.add_argument('--reference',required=True);test.add_argument('--evidence',required=True)
    for name in ('p0','qualify','run','report'):
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name=='qualify':p.add_argument('--mode',choices=['cuda','smoke'],required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=freeze()
    elif args.command=='plan':result=canonical_plan()
    elif args.command=='test':
        from .tests import run_tests
        result=run_tests(args.reference,args.evidence)
    else:
        config=read(args.config)
        if args.command=='p0':
            from .p0 import run
            result=run(config)
        elif args.command=='qualify':
            from .qualification import qualify
            result=qualify(config,'CUDA' if args.mode=='cuda' else 'smoke')
        elif args.command=='run':
            from .execution import run
            result=run(config)
        else:
            from .authority import preflight
            preflight(config)
            from .reporting import report
            result=report(config['run_root'],config)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
