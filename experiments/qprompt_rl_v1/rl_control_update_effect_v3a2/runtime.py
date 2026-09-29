"""Serial processes, immutable receipts, bounded physical attempts."""
import os,json,time,hashlib,fcntl,subprocess
from pathlib import Path
from collections import Counter
import torch
from r1_12h import core as c,runner as old
ROOT=Path(os.environ['EXEC_RUN']);PREVIOUS=Path(os.environ['EXEC_V2']);SOURCE=Path(__file__).parent
PLAN=json.loads((SOURCE/'EXECUTION_PLAN.json').read_text());CODE=old.CODE;CONFIG=c.file_sha(SOURCE/'EXECUTION_PLAN.json')
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):c.atomic(Path(p),v)
def digest(p):return c.file_sha(Path(p))
def receipt(p):
    v=read(p);assert v['code_commit']==CODE and v['config_sha']==CONFIG,'explicit versioned recovery required';return v

def save(p,**v):write(p,dict(code_commit=CODE,config_sha=CONFIG,**v))
def stable(*x):return int(hashlib.sha256(repr(x).encode()).hexdigest()[:12],16)%(2**31)
def deadline():assert time.time()<read(ROOT/'SESSION.json')['deadline'],'DEADLINE'

class Ledger:
    caps={'student':8640,'student_replay':864,'controller':15360,'controller_replay':1536,'preview':1728,'preview_replay':173,'extraction':576,'extraction_replay':58,'vjp':2304,'vjp_replay':232,'student_synthetic':64,'controller_synthetic':4096,'preview_synthetic':32}
    def __init__(self):
        self.lock=(ROOT/'LEDGER.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.count=Counter();self.seen=set()
        for x in c.events(ROOT/'PHYSICAL_LEDGER.jsonl'):
            if x['event']=='attempt':self.count[x['category']]+=1;self.seen.add(x['key'])
    def call(self,kind,key,fn):
        deadline();key=kind+'|'+key
        replay=('student_replay' if kind=='student_synthetic' else 'controller_replay' if kind=='controller_synthetic' else kind+'_replay')
        cat=replay if key in self.seen else kind;assert self.count[cat]<self.caps[cat],f'budget {cat}'
        row=dict(key=key,category=cat,code_commit=CODE);c.append(ROOT/'PHYSICAL_LEDGER.jsonl',dict(event='attempt',**row));self.seen.add(key);self.count[cat]+=1
        try:value=fn()
        except BaseException:c.append(ROOT/'PHYSICAL_LEDGER.jsonl',dict(event='failed',**row));raise
        c.append(ROOT/'PHYSICAL_LEDGER.jsonl',dict(event='success',**row))
        if sum(self.count.values())%50==0:write(ROOT/'BUDGET.json',dict(self.count))
        return value
    def step(self,opt,kind,key):
        assert all(p.grad is None or torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params'])
        return self.call(kind,key,opt.step)
