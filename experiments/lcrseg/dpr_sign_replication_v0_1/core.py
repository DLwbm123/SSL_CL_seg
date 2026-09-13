"""New task registry only. SIGN math and random-key prefix remain frozen imports."""
from experiments.lcrseg.dpr_finite_v0_2.core import *
from experiments.lcrseg.dpr_finite_v0_2 import core as sign
ARMS=('F_CONV','DPR_FINITE_SIGN_U')
MAIN=ARMS[1]
RUN_ARM=None

def strength(epoch):return sign.strength(epoch) if RUN_ARM==MAIN else 0.

def step(model,ema,opt,l,probe,task,epoch,index,counts,patients,diagnostic=False):
    if task['arm']==MAIN:return sign.step(model,ema,opt,l,probe,task,epoch,index,counts,patients,diagnostic)
    if task['arm']!='F_CONV' or probe is not None:raise PermissionError('native F_CONV forbids U response')
    row=parent.step(model,ema,opt,l,None,'T_LCTX',task['seed'],task['domain'],epoch,index,counts,patients)
    row.update(checks={'native_F_CONV':True},lambda_response=0.,degeneracy='native_baseline',candidate_accepted=0,guard_rejected=0,response_images=0,response_VJP=0,response_forwards=0,pseudo_labels=0,EMA_updates=1)
    return row,{}
