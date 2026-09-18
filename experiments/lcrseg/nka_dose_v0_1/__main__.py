"""Dose preparation, generated tests and gated finite execution."""
import argparse,json
from .protocol import freeze,read,validate_plan,DOC


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','plan','test','qualify','run','report'])
    p.add_argument('--reference');p.add_argument('--evidence');p.add_argument('--config');p.add_argument('--mode',choices=['cuda','smoke']);args=p.parse_args()
    if args.command=='prepare':
        from .authority import execution_plan
        from .protocol import write,code_manifest
        result=freeze();write(DOC/'EXECUTION_PLAN.json',execution_plan());write(DOC/'CODE_MANIFEST.json',code_manifest())
    elif args.command=='plan':result=validate_plan(read(DOC/'PLAN.json'))
    elif args.command=='test':
        if not args.reference or not args.evidence:p.error('test requires pinned source --reference and isolated NAS --evidence')
        from .tests import run_tests
        result=run_tests(args.reference,args.evidence)
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
            preflight(config);result=report(config['run_root'],config)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
