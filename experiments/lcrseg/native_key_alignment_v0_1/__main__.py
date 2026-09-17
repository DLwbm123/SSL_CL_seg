"""Code-only preparation. There is deliberately no CUDA/smoke/run command."""
import argparse
import json
from .protocol import freeze,read,validate_plan,DOC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','plan','test'])
    p.add_argument('--reference',help='pinned upstream SOURCE CODE, never a weight checkpoint')
    args=p.parse_args()
    if args.command=='freeze':result=freeze()
    elif args.command=='plan':result=validate_plan(read(DOC/'D1_MATRIX.json'))
    else:
        if not args.reference:p.error('--reference is required for native CPU generated tests')
        from .tests import run_tests
        result=run_tests(args.reference)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
