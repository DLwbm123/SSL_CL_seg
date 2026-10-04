"""Val-only child process with its own access and accelerator cost receipts."""
import os,time
from pathlib import Path
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import evaluator as native
from experiments.qprompt_rl_v1.v3b_lrref_endpoint.engine import write
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations

def main():
    target=Path(os.environ['EVAL_INPUT']);root=Path(os.environ['EXEC_RUN'])/'evaluation_costs'/target.stem
    root.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_device(0);start=time.time()
    first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);first.record()
    with NativeOperations(root/'operations'):native.main()
    last.record();last.synchronize()
    write(root/'RESOURCE.json',dict(wall_seconds=time.time()-start,cuda_interval_seconds=first.elapsed_time(last)/1000,peak_cuda_allocated=torch.cuda.max_memory_allocated(),scope='separate val evaluator; GPU command interval includes host gaps, not kernel busy time'))
if __name__=='__main__':main()
