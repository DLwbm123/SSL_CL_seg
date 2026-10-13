import json,pathlib,statistics
base=pathlib.Path('experiments/lcrseg/docs');out=base/'stage_retention_analysis_20261013'
read=lambda p:json.loads(p.read_text())
refs=read(out/'REFERENCE_STAGES.json');sets={}
folders={'F5':'f5_module_pilots_v1','V13':'f5_class_reference_v13','V15':'f5_class_cone_v15','V17':'f5_sard_v17','V19':'f5_care_v19','V20':'f5_block_guard_v20','V21':'f5_hierarchy_proto_v21','V22':'f5_tangent_proto_v22','V23':'f5_hierarchy_relation_v23','V24':'f5_hierarchy_orientation_v24','V25':'f5_hierarchy_dense_v25','V26':'f5_hierarchy_u_dense_v26','V27':'f5_hierarchy_contrast_v27','B1_SINGLE':'simple_b1_conf_kl_v1'}
for name,folder in folders.items():
 p=next((base/folder).rglob('RESULTS.json'));j=read(p);stages=j['stages'];stages=[x for x in stages if name!='F5' or x['arm']=='F5'];sets[name]=stages
for name,family in [('B0','B0_PARENT_LCTX'),('B2','B2_PARENT_PAS_KL')]:
 sets[name]=[dict(seed=x['identity']['seed'],order=x['identity']['order'],stage=x['identity']['stage'],scores=x['scores']) for x in refs['stages'] if x['identity']['family']==family]
orders={1:['REFUGE','RIM_ONE_r3','Drishti_GS'],2:['REFUGE','Drishti_GS','RIM_ONE_r3']};fields=['macro_Dice','rim','cup','disc_union'];rows=[];traj=[]
for method,stages in sets.items():
 keys=sorted({(s['seed'],s['order']) for s in stages});assert len(keys)==(1 if method=='B1_SINGLE' else 4)
 for seed,order in keys:
  ss={s['stage']:s['scores'] for s in stages if (s['seed'],s['order'])==(seed,order)};assert set(ss)=={1,2};domains=orders[order];src=refs['sources'][str(seed)]['REFUGE']
  for f in fields:
   vals=[]
   for position,d in enumerate(domains):
    early=src[f] if position==0 else ss[1][d][f] if position==1 else ss[2][d][f];final=ss[2][d][f];forget=early-final if position<2 else None
    row=dict(method=method,seed=seed,order=order,domain=d,role=['source','first_target','incoming'][position],metric=f,early=early*100,after_stage1=ss[1].get(d,{}).get(f),final=final*100,forget_pp=None if forget is None else forget*100)
    if row['after_stage1'] is not None:row['after_stage1']*=100
    if position==0:row.update(source_to_stage1_forget_pp=(early-ss[1][d][f])*100,stage1_to_stage2_forget_pp=(ss[1][d][f]-final)*100)
    if forget is not None:assert abs(early-forget-final)<1e-12
    rows.append(row);vals.append(row)
   early_old=sum(x['early'] for x in vals[:2])/2;old=sum(x['final'] for x in vals[:2])/2;forget=sum(x['forget_pp'] for x in vals[:2])/2;assert abs(early_old-old-forget)<1e-10
   traj.append(dict(method=method,seed=seed,order=order,metric=f,Final=sum(x['final'] for x in vals)/3,Old=old,Incoming=vals[2]['final'],Forget=forget,EarlyOld=early_old,FirstAcquisition=vals[1]['early'],FirstFinal=vals[1]['final'],FirstForget=vals[1]['forget_pp'],SourceFinal=vals[0]['final'],SourceForget=vals[0]['forget_pp']))
summary={}
for method in sets:
 summary[method]={}
 for f in fields:
  rr=[r for r in traj if r['method']==method and r['metric']==f];summary[method][f]={k:statistics.mean(r[k] for r in rr) for k in ['Final','Old','Incoming','Forget','EarlyOld','FirstAcquisition','FirstFinal','FirstForget','SourceFinal','SourceForget']}
paired=[]
for ref in ['B2','V13','V21']:
 table={(r['seed'],r['order'],r['domain'],r['metric']):r for r in rows if r['method']==ref}
 for r in rows:
  if r['method']==ref:continue
  b=table[r['seed'],r['order'],r['domain'],r['metric']];v={k:r[k] for k in ['method','seed','order','domain','role','metric']};v.update(reference=ref,delta_final_pp=r['final']-b['final'],delta_early_pp=r['early']-b['early'],delta_forget_pp=None if r['forget_pp'] is None else r['forget_pp']-b['forget_pp'])
  if v['delta_forget_pp'] is not None:assert abs(v['delta_final_pp']-(v['delta_early_pp']-v['delta_forget_pp']))<1e-9
  paired.append(v)
for name,v in [('STAGE_DECOMPOSITION.json',rows),('TRAJECTORY_METRICS.json',traj),('PAIRED_DECOMPOSITION.json',paired),('SUMMARY.json',summary)]:
 (out/name).write_text(json.dumps(v,indent=2)+'\n')
print('All coverage/decomposition assertions PASS; no model or evaluation calls')
