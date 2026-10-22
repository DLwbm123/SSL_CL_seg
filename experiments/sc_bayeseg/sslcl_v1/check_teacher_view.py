"""Real U teacher/student view identity, gradient and default-path regression."""
import copy,importlib.util,json,os
from pathlib import Path
import torch
from .data import Schedule
from .model import initialize
from .train import setup,step_loss
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();assert len(s['tasks'])==len(s['source_paths'])==6 and not s['training_seeds']
    for name,f,stage,use_u,hist in s['tasks']:
        o=s['task_options'][name];seed=o['schedule_seed'];suffix='' if seed==163 else f'_s{seed}'
        assert seed in (163,164,165) and o['initialization_seed']==seed and f=='U' and stage=='target' and use_u
        assert hist==('SAMEH' in name) and not o['history_l'] and o['history_u'] and o['teacher_on_strong'] and o['history_strength']==1.
        assert Path(s['source_paths'][name]).parent.name=='source_U'+suffix
        ck=torch.load(s['source_paths'][name],map_location='cpu',weights_only=False);assert ck['step']==5000 and ck['family']=='U'
    spec=importlib.util.spec_from_file_location('sc_bayeseg.sslcl_v1.legacy_train',s['legacy_train']);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    ck=torch.load(s['source_paths']['USAMEH_s163'],map_location='cpu',weights_only=False)
    m=initialize(s,'U');m.load_state_dict(ck['model']);t=copy.deepcopy(m).eval().requires_grad_(False);t.load_state_dict(ck['teacher'])
    x,y,_=Schedule(s['data'],'BMC',dataset='prostate').batch(5001);u,uy,_=Schedule(s['data'],'BMC',True,dataset='prostate').batch(5001);assert uy is None
    current=s['styles']['BMC'];history=s['styles']['RUNMC']
    a,_=legacy.step_loss(copy.deepcopy(m),copy.deepcopy(t),x,y,u,current,history,1000,history_l=False)
    b,_=step_loss(copy.deepcopy(m),copy.deepcopy(t),x,y,u,current,history,1000,history_l=False)
    assert torch.equal(a,b);del a,b
    result={}
    for mode,h in [('NONE',None),('HISTORY',history)]:
        m.zero_grad(set_to_none=True);student=[];teacher=[]
        hm=m.register_forward_pre_hook(lambda model,args:student.append(args[0].detach().clone()))
        ht=t.register_forward_pre_hook(lambda model,args:teacher.append(args[0].detach().clone()))
        loss,parts=step_loss(m,t,x,y,u,current,h,1000,history_l=False,teacher_on_strong=True)
        hm.remove();ht.remove();assert len(student)==3 and len(teacher)==1 and torch.equal(student[2],teacher[0])
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad);assert all(p.grad is None for p in t.parameters())
        result[mode]=dict(loss=float(loss.detach()),parts=parts)
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,source_mappings=6,default_loss_exact_legacy=True,same_teacher_student_strong_input=True,teacher_detached=True,finite_backward=result))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
