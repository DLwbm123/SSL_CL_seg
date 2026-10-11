
import os,json
from pathlib import Path
import torch
from experiments.lcrseg.f5_class_cone_v15 import protocol
from experiments.lcrseg.f5_class_cone_v15.run import engine,make_model,ClassConeTrainer,bind
from experiments.lcrseg.f5_module_pilots_v1.tests import GeneratedDomain
from experiments.lcrseg.f5_confirmation_v1.native_qualification import foreground_fixture
from experiments.lcrseg.five_frameworks_v1.native_parent import build,NativeLRParent
from experiments.lcrseg.five_frameworks_v1.recipes import generator
from experiments.lcrseg.five_frameworks_v1 import checkpoint
bind();c=protocol.read(os.environ['EXEC_CONFIG']);root=Path(c['run_root']);torch.set_num_threads(2);device=torch.device('cuda:0')
try:
 permit=engine.admission(c,'qualification');p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'));native=build(c['reference'],device,163);foreground_fixture(native)
 options={**protocol.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
 m=make_model(NativeLRParent(native,163,p.stage_source),options,None,generator(163,1,1,0,'pilot_adapter_initialization')).to(device);t=ClassConeTrainer(m,p,options,execution=permit)
 checkpoint.restore(t,root/'qualification.pt',dict(family='F5',seed=163,order=1,stage=1,arm='CLASS_CONE'))
 x=torch.randn(2,3,384,384,device=device);z=t.model(x);labels=torch.arange(384,device=device).remainder(3).view(1,1,384).expand(2,384,384);q=torch.nn.functional.one_hot(labels,3).permute(0,3,1,2).to(z.dtype)*.85+.05
 support=torch.ones_like(labels,dtype=torch.bool);t.entry_reference_loss=t.entry_reference_objective(z,q,support);before=t.telemetry['entry_reference_vjps'];t.gradient_audit(z.square().mean(),z.square().mean())
 assert t.entry_class_ids==[0,1,2] and t.telemetry['entry_reference_vjps']-before==3
 params=[v for v in t.model.parameters() if v.requires_grad];groups=t.model.parent.parameter_groups();norms=[]
 for ref in t.entry_class_grads:
  gs={id(v):g for v,g in zip(params,ref)};row={name:sum(float(gs[id(v)].square().sum()) for v in ps if gs[id(v)] is not None)**.5 for name,ps in [('A',groups['input_factors']),('B',groups['output_factors']),('R',t.model.sidecar.optimizer_parameters())]};assert all(v>0 for v in row.values());norms.append(row)
 aggregate=[torch.zeros_like(v) if g is None else g.detach() for v,g in zip(params,t.entry_reference_grads)];grads={}
 for v,g in zip(params,aggregate):v.grad=-g;grads[id(v)]={'L':-g,'U':torch.zeros_like(g)}
 t.transform_gradients(grads);assert len(t.module_stats['entry_dots_after'])==3 and all(v>=-1e-6 for v in t.module_stats['entry_dots_after']) and t.entry_class_grads is None
 assert sum(float(v.grad.square().sum()) for v in params)<1e-10
 result=dict(status='PASS',execution_commit=c['execution_commit'],present_classes=[0,1,2],class_ABR_reference_norms=norms,dots_before=t.module_stats['entry_dots_before'],dots_after=t.module_stats['entry_dots_after'],active_constraints=t.module_stats['entry_active_constraints'],costs=dict(formal_updates=0,cpu_optimizer_updates=0,synthetic_cuda_updates=0,real_L_smoke_updates=0,generated_reference_vjps=3,generated_student_full_forwards=1),scope='generated stripe targets and restored qualification state; no patient, no optimizer, no formal efficacy',frozen_PASS_checks_not_repeated=True)
 protocol.write(root/'QUALIFICATION_THREE_CLASS.json',result);print(json.dumps(result))
except BaseException:
 protocol.write(root/'QUALIFICATION_THREE_CLASS_FAILURE.private.json',dict(traceback=__import__('traceback').format_exc()));raise
