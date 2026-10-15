import pathlib,json,os,time,subprocess,datetime
from experiments.lcrseg.main_style_ssl_v2.protocol import nodes,verify_frozen
os.nice(10)
c=json.loads(pathlib.Path(os.environ['TASK_SETTINGS']).read_text());r=pathlib.Path(c['root']);ops=pathlib.Path(c['ops']);plan=[n for n in nodes() if n['dataset']=='prostate'];targets=[n for n in plan if n['stage']];running={};states={n['id']:'REUSED_SOURCE' for n in plan if not n['stage']};started=time.time()
def write(p,q):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(q,indent=2));os.replace(tmp,p)
def birth(pid):return int((pathlib.Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19])
def read(p):return json.loads(p.read_text()) if p.exists() else None
f=verify_frozen();qual=read(r/'QUALIFICATION.json');assert qual['status']=='PASS' and qual['protocol_digest']==f['protocol_digest'];assert sum(n['steps'] for n in targets)==150200 and len(targets)==10
write(r/'PLAN.json',{'version':'MAIN_STYLE_SSL_V2','nodes':plan,'reused_source_updates':27200,'new_formal_optimizer_updates':150200,'nominal_total_updates':177400,'datasets':['prostate'],'seeds':[163,164],'ablation':False,'test_accessor':False,'selection':'fixed final checkpoints','gpus':c['gpus'],'protocol_digest':f['protocol_digest']})
write(r/'CONTROLLER_PID.json',{'pid':os.getpid(),'birth':birth(os.getpid()),'host':'GPU_HOST','gpus':[2,3],'max_workers_per_gpu':1})
while len(states)<len(plan):
 for key,v in list(running.items()):
  p,log,info=v;code=p.poll()
  if code is None:continue
  log.close();del running[key];s=read(r/'jobs'/key/'STATUS.json') or {};states[key]='COMPLETE' if code==0 and s.get('status')=='COMPLETE' else 'FAILED'
 for n in targets:
  key=n['id'];pred=n['predecessor'];gpu=c['gpus'][str(n['seed'])]
  if key in states or key in running:continue
  if states.get(pred) in ['FAILED','BLOCKED']:
   out=r/'jobs'/key;out.mkdir(exist_ok=False);write(out/'STATUS.json',{'status':'BLOCKED','node':n,'reason':'predecessor failed; no automatic retry'});write(out/'EXIT.json',{'exit_code':None,'status':'BLOCKED'});states[key]='BLOCKED';continue
  if states.get(pred) not in ['COMPLETE','REUSED_SOURCE'] or any(info['gpu']==gpu for _,_,info in running.values()):continue
  free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free<14000:continue
  cfg={'node':n,'output':str(r/'jobs'/key),'data':c['data'],'reference':c['reference'],'predecessor':str(r/'jobs'/pred/'sealed.pt'),'authority_receipt':str(r/'AUTHORITY.json')};path=r/'configs'/(key+'.json');write(path,cfg)
  env=dict(os.environ,TASK_ACTION='worker',TASK_CONFIG=str(path),HOST_SCRIPT=str(ops/'w.py'),CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(r/'engineering'),CUDA_CACHE_PATH=str(r/'cache/cuda'),TORCH_HOME=str(r/'cache/torch'),XDG_CACHE_HOME=str(r/'cache'),HF_HOME=str(r/'cache/hf'),PYTHONUNBUFFERED='1');log=(r/'configs'/(key+'.log')).open('x');p=subprocess.Popen([c['python'],str(ops/'n.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ops));info={'pid':p.pid,'birth':birth(p.pid),'gpu':gpu,'seed':n['seed']};running[key]=(p,log,info)
 write(r/'STATUS.json',{'status':'RUNNING','completed':states,'running':{k:v[2] for k,v in running.items()},'total_nodes':12,'formal_target_nodes':10,'new_formal_budget':150200,'reused_source_updates':27200,'elapsed_seconds':time.time()-started,'host':'GPU_HOST'})
 time.sleep(2)
summary=[]
for seed in [163,164]:
 chain=sorted([n for n in plan if n['seed']==seed],key=lambda n:n['stage']);matrix=[read(r/'jobs'/n['id']/'VALIDATION.json') for n in chain];row={'seed':seed,'matrix':matrix,'status':'COMPLETE' if all(x is not None for x in matrix) else 'INCOMPLETE'}
 if all(x is not None for x in matrix):
  final=matrix[-1];domains=list(final);row.update(Final=sum(final[d]['primary_dice'] for d in domains)/len(domains),Old=sum(final[d]['primary_dice'] for d in domains[:-1])/len(domains[:-1]),Incoming=final[domains[-1]]['primary_dice'],Forgetting=sum(matrix[i][d]['primary_dice']-final[d]['primary_dice'] for i,d in enumerate(domains[:-1]))/len(domains[:-1]))
 summary.append(row)
write(r/'SUMMARY.json',{'prostate':summary});cost=[read(r/'jobs'/n['id']/'COSTS.json') or {} for n in targets];write(r/'COSTS.json',{'new_formal_optimizer_updates':sum(x.get('physical_optimizer_updates',0) for x in cost),'formal_budget':150200,'reused_source_updates':27200,'engineering_optimizer_calls':qual['optimizer_calls'],'elapsed_seconds':time.time()-started,'source_bootstrap_L_image_exposure':132,'independent_test_access':False})
final='COMPLETE' if all(x in ['COMPLETE','REUSED_SOURCE'] for x in states.values()) else 'COMPLETE_WITH_FAILURES';write(r/'STATUS.json',{'status':final,'completed':states,'running':{},'total_nodes':12,'new_formal_budget':150200});write(r/'EXIT.json',{'exit_code':0 if final=='COMPLETE' else 1,'status':final,'public_delivery_verified':False})
