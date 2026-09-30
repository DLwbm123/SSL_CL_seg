"""Separate process: val only, no controller or training feedback."""
import os,time
from pathlib import Path
import torch
from .engine import read,write,build,NativeLRParent
from experiments.lcrseg.five_frameworks_v1.model import Deployment
from experiments.lcrseg.five_frameworks_v1.native_runner import evaluate

def main():
    torch.set_num_threads(2);c=read(os.environ['EXEC_CONFIG']);p=Path(os.environ['EVAL_INPUT']);v=torch.load(p,map_location='cuda:0',weights_only=False)
    native=build(c['reference'],'cuda:0',v['seed']);native.load_state_dict(v['student'])
    model=Deployment(NativeLRParent(native,v['seed'],{},adapt=False),torch.eye(16,device='cuda:0')).cuda()
    start=time.time();scores,private=evaluate(model,c['data'],['REFUGE',v['domain']],'cuda:0')
    write(p.with_suffix('.scores.json'),dict(scores=scores,seconds=time.time()-start,step=v['step'],seed=v['seed'],domain=v['domain']))
    write(p.with_suffix('.private.json'),private)
if __name__=='__main__':main()
