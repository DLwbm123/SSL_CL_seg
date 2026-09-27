"""Finite native student/controller and current-L-only smoke qualification."""
from .runner import *


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def qualify():
    backbone=os.environ['EXEC_BACKBONE'];spec=next(dict(s) for s in PLAN['R3a_jobs'] if s['backbone']==backbone and s['domain']=='RIM_ONE_r3');spec.update(id='QUAL__'+backbone,arm='RL');w=Worker(spec,kind='synthetic_student');torch.cuda.reset_peak_memory_stats()
    x=torch.rand(2,3,384,384,device=DEVICE);y=torch.zeros(2,384,384,device=DEVICE,dtype=torch.long);y[:,64:300,64:300]=1;y[:,128:224,128:224]=2;us=x[:1].clone();q=torch.zeros(1,3,384,384,device=DEVICE);q[:,0]=.95;q[:,1]=.025;q[:,2]=.025;v=torch.ones_like(q[:,0],dtype=torch.bool)
    initial=w.snapshot();states=[]
    for j,a in enumerate((0,0,1,2)):
        w.restore(initial);info=w.student_step(x,y,us,q,v,a,'native/'+str(j),temporary=True)
        if j<2:states.append(cpu_copy(w.model.state_dict()))
        if a:assert info['U_grad_norm'] is not None and info['U_grad_norm']>0
    errors=[]
    for k in states[0]:
        if states[0][k].is_floating_point():
            torch.testing.assert_close(states[0][k],states[1][k],atol=1e-7,rtol=1e-5);errors.append(float((states[0][k]-states[1][k]).abs().max()))
    w.restore(initial);assert c.tensor_sha(w.model.state_dict())==c.tensor_sha(initial['student']) and c.tensor_sha(w.teacher.state_dict())==c.tensor_sha(initial['teacher']) and w.sch.state_dict()==initial['scheduler']
    assert same(initial,w.snapshot()),'full native rollback state mismatch'
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):feedback=quality(w.model(x)['semantic'],y)
    assert feedback is not None and not feedback.requires_grad
    z=torch.linspace(0,1,16,device=DEVICE)
    for arm in ADAPTIVE:
        policy=Controller(arm).cuda();reference=make_reference(policy);opt=torch.optim.Adam(policy.parameters(),lr=3e-4);behavior=policy.distribution(z).detach().clone();a=draw(behavior,123)[0]
        for j in range(4):
            opt.zero_grad(set_to_none=True);loss,_=controller_loss(policy,reference,z,torch.tensor([0,1,2,2],device=DEVICE),torch.tensor([0.,-.01,.02,.02],device=DEVICE),behavior,.01);loss.backward();w.charge_step(opt,'synthetic_controller',arm+'/'+str(j))
        assert a==draw(behavior,123)[0] and not torch.equal(policy.distribution(z),behavior)
    del states,initial,x,y,us,q,v;gc.collect();torch.cuda.empty_cache()
    # Exact 16 current-L-only smoke calls over four backbone/domain cells; no U objective.
    w.kind='smoke'
    for domain in ('RIM_ONE_r3','Drishti_GS'):
        binding=json.loads((RUN/'PREFIX_BINDINGS.private.json').read_text())[f'FROZEN_PREFIX__S261__{backbone}__{domain}'];prefix=torch.load(binding['path'],map_location='cpu',weights_only=False);w.opt,source_sch=c.optimizer_for(w.model,backbone);c.restore(prefix,w.model,w.opt,source_sch);w.sch=LocalSchedule(w.opt);del prefix
        ds=old.dataset(domain,'train_labeled');cache=[ds[i] for i in range(len(ds))];doc=make_schedule(261,domain,len(cache),63 if domain=='RIM_ONE_r3' else 41)
        for j in range(4):
            x,y=old.load_batch(cache,doc['steps'][j],DEVICE);w.student_step(x,y,None,None,None,0,domain+'/'+str(j),temporary=True)
    c.atomic(RUN/'qualification'/backbone/'PASSED.json',dict(status='PASSED',synthetic_student=4,synthetic_controller=12,smoke=8,restore_exact=True,anchor_max_abs=max(errors),feedback_no_grad=True,peak_reserved=torch.cuda.max_memory_reserved(),code_commit=CODE))
    rpc(RUN,action='release',task=w.task,token=w.token)
