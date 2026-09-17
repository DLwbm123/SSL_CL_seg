"""New-study entry points; production commands require fresh bound authority."""
import argparse
import json
from .protocol import freeze,read,validate_plan,DOC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','plan','test','integration-test','qualify','run','report'])
    p.add_argument('--reference',help='pinned upstream source code, never a checkpoint')
    p.add_argument('--config',help='private runtime JSON outside reviewed checkout')
    p.add_argument('--mode',choices=['cuda','smoke'])
    args=p.parse_args()
    if args.command=='freeze':result=freeze()
    elif args.command=='plan':result=validate_plan(read(DOC/'D1_MATRIX.json'))
    elif args.command in ('test','integration-test'):
        if not args.reference:p.error('--reference required for CPU generated tests')
        if args.command=='test':
            from .tests import run_tests
        else:
            from .integration_tests import run_tests
        result=run_tests(args.reference)
    else:
        if not args.config:p.error('--config required')
        config=read(args.config)
        if args.command=='qualify':
            if not args.mode:p.error('--mode required')
            from .qualification import qualify
            result=qualify(config,'CUDA' if args.mode=='cuda' else 'smoke')
        elif args.command=='run':
            from .execution import run
            result=run(config)
        else:
            from .authority import preflight
            from .reporting import report
            preflight(config)
            result=report(config['run_root'],config)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
