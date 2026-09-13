import argparse,json,os,subprocess
from pathlib import Path
from .parent_bridge import bind_real_parent
from .gate import ReviewRequired
from .planner import write


ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'experiments/lcrseg/docs/five_frameworks_v1'


def main():
    p=argparse.ArgumentParser(description='Five-framework CODE_ONLY review interface')
    sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('inspect');i.add_argument('--metadata-only',action='store_true',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--emit',type=Path,required=True)
    q=sub.add_parser('qualify');q.add_argument('--synthetic',action='store_true',required=True);q.add_argument('--device',choices=['cpu'],required=True)
    r=sub.add_parser('run');r.add_argument('--approval',type=Path,required=True);r.add_argument('--phases',nargs='+',required=True)
    report=sub.add_parser('report');report.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args()
    if args.command=='inspect':print((DOC/'review/PARENT_BINDING.json').read_text())
    elif args.command=='plan':write(json.loads((DOC/'delivery/configs/study_plan.json').read_text()),args.emit)
    elif args.command=='qualify':
        result=subprocess.run([os.sys.executable,'-m','pytest','-q','experiments/lcrseg/tests/five_frameworks_v1'],cwd=ROOT)
        raise SystemExit(result.returncode)
    elif args.command=='report':
        # JSON summaries only; report does not load checkpoint tensors or images.
        for path in sorted(args.run_root.glob('**/summary.json')):print(path.read_text())
    else:
        # No approval payload can turn this unresolved code-only bridge into a
        # real experiment. Gate unit tests separately exercise every binding/cap.
        raise ReviewRequired('CODE_ONLY / PARENT_BINDING_REQUIRED; no real data opened')

if __name__=='__main__':main()
