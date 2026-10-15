import os,json,pathlib,torch,time,copy,traceback
c=json.loads(pathlib.Path(os.environ['TASK_SETTINGS']).read_text());root=pathlib.Path(c['root']);seed=int(os.environ['QUAL_SEED']);out=root/'engineering'/('attempt_'+os.environ['QUAL_ATTEMPT']+'_s'+str(seed));out.mkdir(exist_ok=False)
from experiments.lcrseg.main_style_ssl_v2.protocol import nodes,verify_frozen
from experiments.lcrseg.main_style_ssl_v2.runtime import bootstrap_source,setup
from experiments.lcrseg.main_style_ssl_v2.checkpoint import save,restore,same_tree
from experiments.lcrseg.main_style_ssl_v1.runtime import permit
from experiments.lcrseg.five_frameworks_v1.checkpoint import PhysicalLedger
calls=0;original=torch.optim.Adam.step;started=time.time();result={'seed':seed,'status':'RUNNING','formal_updates':0,'test_access':0,'optimizer_calls':0,'checks':{}}
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ledger=PhysicalLedger(out/'physical_updates.jsonl')
def counted(self,*a,**kw):
 global calls
 if calls>=165:raise RuntimeError('engineering per-seed cap')
 calls+=1;ledger.append(calls)
 return original(self,*a,**kw)
torch.optim.Adam.step=counted
source=next(n for n in nodes() if n['dataset']=='prostate' and n['seed']==seed and n['stage']==0);target=next(n for n in nodes() if n['dataset']=='prostate' and n['seed']==seed and n['stage']==1)
source_out=root/'jobs'/source['id'];source_out.mkdir(exist_ok=True)
def new(initialize=True):
 t=setup(target,c['data'],c['reference'],'cuda',source_out/'sealed.pt',initialize);t.execution=permit(target,'smoke');return t
# Synthetic phase routing is restricted to qualification losses. Execution counters
# remain actual engineering calls, not fabricated scientific updates.
def full_phase(t):
 old=t.losses
 def routed():
  actual=t.step;t.step=target['steps']//2+actual
  try:return old()
  finally:t.step=actual
 t.losses=routed
try:
 frozen=verify_frozen();result['frozen']=frozen['protocol_digest'];torch.cuda.reset_peak_memory_stats()
 if not (source_out/'sealed.pt').exists():bootstrap_source(pathlib.Path(c['source_root'])/'jobs'/source['id']/'sealed.pt',source,c['data'],c['reference'],'cuda',source_out/'sealed.pt')
 else:result['source_memory_reused_from_attempt02']=True
 src=torch.load(source_out/'sealed.pt',map_location='cpu',weights_only=False);assert src['new_optimizer_updates']==0 and src['bootstrap_U_reads']==0 and src['bootstrap_L_reads']==33 and src['memory'][0]['domain']=='RUNMC'
 for f in ['VALIDATION.json']:
  (source_out/f).write_bytes((pathlib.Path(c['source_root'])/'jobs'/source['id']/f).read_bytes())
 (source_out/'STATUS.json').write_text(json.dumps({'status':'REUSED_SOURCE','node':source,'step':13600,'new_optimizer_updates':0,'L_batch_reads':33,'L_image_exposure':66,'U_batch_reads':0}))
 (source_out/'COSTS.json').write_text(json.dumps({'physical_optimizer_updates':0,'inherited_updates':13600,'L_batch_reads':33,'L_image_exposure':66,'U_batch_reads':0}))
 (source_out/'EXIT.json').write_text(json.dumps({'exit_code':0,'source_retrained':False}));result['checks']['source_import_training_L_only']=True
 t=new();entry=copy.deepcopy(t.model.entry_anchor.state_dict());stem=copy.deepcopy(t.model.parent.adapters[0].layer.state_dict());t.update();assert not t.last['active_U'];full_phase(t);grads=t.update();assert t.last['active_U'] and t.model.sidecar.valid() and t.ema.sidecar.valid();assert same_tree(entry,t.model.entry_anchor.state_dict()) and same_tree(stem,t.model.parent.adapters[0].layer.state_dict())
 for token in ['.A','.B','sidecar.r']:
  selected=[grads[id(p)] for name,p in t.model.named_parameters() if token in name and p.requires_grad];assert any(g is not None and float(g.norm())>0 for d in selected for g in d.values()),token
 result['checks']['GPU_real_L_U_backward_Adam_guard_EMA']=True;result['checks']['MRI_stem_and_entry_immutable']=True
 identity={k:target[k] for k in ['id','dataset','seed','order','stage','domain']};save(t,out/'resume.pt',identity);saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
 t.update();expected={'student':copy.deepcopy(t.model.state_dict()),'ema':copy.deepcopy(t.ema.state_dict()),'optimizer':copy.deepcopy(t.optimizer.state_dict()),'memory':t.memory.state(),'last':copy.deepcopy(t.last),'guard':copy.deepcopy(t.guard_totals),'reads':[t.provider.l_reads,t.provider.u_reads]}
 del t;torch.cuda.empty_cache();resumed=new(False);restore(resumed,out/'resume.pt',identity);full_phase(resumed);assert resumed.step==2 and resumed.physical_optimizer_updates==2;resumed.update()
 actual={'student':resumed.model.state_dict(),'ema':resumed.ema.state_dict(),'optimizer':resumed.optimizer.state_dict(),'memory':resumed.memory.state(),'last':resumed.last,'guard':resumed.guard_totals,'reads':[resumed.provider.l_reads,resumed.provider.u_reads]};assert same_tree(expected,actual),'GPU checkpoint next-update mismatch';result['checks']['exact_GPU_next_update_resume']=True
 del resumed,expected,actual,saved;torch.cuda.empty_cache()
 if seed==163:
  fit=new();access=fit.provider._l;xs=[];ys=[];ids=access.patients[:2]
  for patient in ids:
   idx=access.by_patient[patient][0];volume=access.read(idx);z=int((volume['label']>0).flatten(1).sum(1).argmax());item=access.read(idx,z);xs.append(item['image']);ys.append(item['label'])
  x=torch.stack(xs).cuda();y=torch.stack(ys).cuda();assert (y>0).flatten(1).any(1).all();provider=fit.provider
  def fixed(step,stream='labeled'):
   provider.l_reads+=1;return x.clone(),y.clone(),tuple(ids)
  def forbidden(step):raise PermissionError('two-case engineering is training-L-only')
  provider.labeled=fixed;provider.unlabeled=forbidden;losses_fn=fit.losses
  def l_only():
   actual=fit.step;fit.step=0
   try:return losses_fn()
   finally:fit.step=actual
  fit.losses=l_only;values=[]
  for i in range(96):
   fit.update();values.append(fit.last['labeled_loss'])
   if i%16==0:print(json.dumps({'engineering_overfit_step':i+1,'loss':values[-1],'optimizer_calls':calls}),flush=True)
  before=sum(values[:4])/4;after=sum(values[-4:])/4;result['two_case_L_fit']={'steps':96,'initial_loss':before,'final_loss':after,'threshold':'final < 0.8 initial; inherited V1 engineering rule','U_reads_during_fit':0};assert after<.8*before,f'two-case loss did not fall enough: {before} -> {after}';result['checks']['two_case_training_L_fit']=True
 result['status']='PASS'
except BaseException as exc:
 result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());raise
finally:
 result.update(optimizer_calls=calls,formal_updates=0,test_access=0,elapsed_seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated());(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
