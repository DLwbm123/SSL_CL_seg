"""One fixed, non-resuming sequence; actual child exit receipts for every stage."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from care_hr_v0_7_1.io_r1 import write_json
from care_hr_v0_7_1.execute_r1 import source_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();base=out.parent
    if out.exists():raise FileExistsError('new output required; no implicit resume')
    write_json(base/'STUDY_SEQUENCE_RESERVATION.json',dict(source_commit=source_state(),created_unix=time.time(),
        outer_results_previously_viewed_in_this_attempt=False,output=str(out)))
    phases=[('prepare',None),('targets',None)]
    phases += [(s,f) for f in range(5) for s in ('train','deploy')]
    phases += [('seal',None),('evaluate',None)]
    for stage,fold in phases:
        name=stage if fold is None else stage+str(fold)
        command=[sys.executable,'-m','shor_uv_v0_8.pipeline',stage,'--output',str(out)]
        if stage=='prepare':command+=['--qualification',str(args.qualification)]
        if fold is not None:command+=['--fold',str(fold)]
        parent=[sys.executable,'-m','care_hr_v0_7_1.parent_r1','--receipt',str(base/(name+'.parent.json')),
                '--log',str(base/(name+'.log')),'--',*command]
        result=subprocess.run(parent,check=False)
        print(name+': actual child exit '+str(result.returncode),flush=True)
        if result.returncode:return result.returncode
    write_json(base/'STUDY_SEQUENCE_COMPLETE.json',dict(source_commit=source_state(),stages=len(phases),
        actual_child_exit_codes_all_zero=True,completed_unix=time.time(),next_stage_started=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
