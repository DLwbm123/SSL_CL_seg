"""Same fixed final-student scoring, with target-seal admission and source-only evaluation."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from . import contract as ct,core as c
from experiments.lcrseg.ams_seq_transfer_v0_1 import evaluate as old
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations

def evaluate(base,task_id,data,reference,device,fixture=None):
    if fixture is None:
        source=ct.verify();task=ct.admit(base,task_id,source)
        if str(Path(data).resolve())!=ct.read(Path(base)/'private_inputs.json')['data']:raise PermissionError('data identity')
        if task['source_task_id'] is not None:
            seal=ct.read(Path(base)/'TARGET_WEIGHT_SEAL.json');assert seal['source']==source and len(seal['students'])==12
            assert seal['students'][task_id]['student_hash']==ct.read(Path(base)/'tasks'/task_id/'receipt.json')['student_hash']
    with patch.multiple(old,verify=ct.verify,admit=ct.admit,read=ct.read):return old.evaluate(base,task_id,data,reference,device,fixture)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','task_id','data','reference'):p.add_argument('--'+k.replace('_','-'),required=True)
    a=vars(p.parse_args());ct.neutral_subprocess_paths();root=Path(a['base'])/'tasks'/a['task_id']
    with Operations(root.parent/(root.name+'_eval_operations')):evaluate(**a,device=torch.device('cuda:0'))
