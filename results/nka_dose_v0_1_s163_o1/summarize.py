"""Metadata-only report for the explicitly user-reduced experiment; no training."""
import csv,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[1];sys.path.insert(0,str(ROOT))
from experiments.lcrseg.f5_confirmation_v1.protocol import metrics,digest
read=lambda p:json.loads(p.read_text())
def write(name,v):(D/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def table(name,rows):
    with (D/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def metric(r):return metrics(dict(identity=r['identity'],scores=r['scores'],timeline={'0':r['prefix_timeline']['0'],'1':r['prefix_scores']}))
x=read(D/'AUDIT_EVIDENCE.json');assert x['status']=='COMPLETE_USER_REDUCED_SCOPE'
old=read(ROOT/'experiments/lcrseg/docs/nka_dose_v0_1/inputs/PUBLIC_RESULTS.json')
rows=[]
for r in old['rows']:
    i=r['identity']
    if i['seed']==163 and i['order']==1 and i['arm'] in ('C0','C2','C3'):
        rows.append(dict(r,origin='historical_import',lambda_align=0. if i['arm']=='C0' else .05,source_commit='0da596f4050a3701f41ff74248ca9d4ba21222e2',adam_diagnostics='NOT_MEASURED_HISTORICAL'))
for r in x['rows']:
    proof=x['integrity'][r['node_id']]
    assert proof['status']=='VERIFIED' and proof['receipt_sha256']==digest(r)
    assert r['step']==r['physical_optimizer_calls']==2100 and r['status']=='SEALED'
    rows.append(dict(r,lambda_align=r['identity']['lambda_align']))
rows.sort(key=lambda r:(r['lambda_align'],r['identity']['arm']))
assert len(rows)==7 and len(x['rows'])==4 and x['actual_formal_calls']==8400
index={(r['identity']['arm'],r['lambda_align']):r for r in rows}
assert set(index)=={('C0',0.),('C2',.05),('C3',.05),('C2',.5),('C3',.5),('C2',2.),('C3',2.)}
final=[];domain=[];pairs=[];adam={};gradient={}
for r in rows:
    i=r['identity'];m=dict(node_id=r['node_id'],arm=i['arm'],lambda_align=r['lambda_align'],seed=163,order=1,origin=r['origin'])
    final.append(dict(m,**metric(r)))
    for d,v in r['scores'].items():domain.append(dict(m,domain=d,**v))
    if r['origin']=='historical_import':continue
    assert set(r['diagnostics'])=={'525','1050','1575','2100'}
    for step,d in r['diagnostics'].items():
        assert d['prediction']['passed'] and d['extra_vjps']==4 and d['read_only_candidates']==2
        key=r['node_id']+'/'+step;adam[key]=dict(m,**{k:v for k,v in d.items() if k not in ('gradients','alignment')});gradient[key]=dict(m,gradients=d['gradients'],alignment=d['alignment'],losses=d['losses'])
for dose in (.05,.5,2.):
    for left,right in [('C3','C0'),('C2','C0'),('C3','C2')]:
        a=index[left,dose];b=index[right,0. if right=='C0' else dose]
        assert a['identity']['prefix_binding_sha256']==b['identity']['prefix_binding_sha256'] and a['prefix_scores']==b['prefix_scores'] and a['prefix_timeline']==b['prefix_timeline']
        ma,mb=metric(a),metric(b);delta={k:ma[k]-mb[k] for k in ma}
        assert abs(delta['Forget']+delta['Old'])<1e-10
        pairs.append(dict(lambda_align=dose,contrast=left+'-'+right,seed=163,order=1,**delta))
assert (len(final),len(domain),len(pairs),len(adam))==(7,21,9,16)
new=x['rows'];alignment={k:sum(r['alignment_cost'][k] for r in new) for k in new[0]['alignment_cost']}
assert alignment['diagnostic_vjps']==64 and alignment['adam_candidates']==32 and alignment['clean_U_forwards']==6720
assert all(r['support']['valid_steps']==r['support']['committed_active_steps']==1680 for r in new)
training=[x['costs'][r['node_id']]['training'] for r in new]
assert all(c['failed_sessions']==0 for c in training)
cost=dict(formal_scientific=8400,formal_physical=8400,smoke=32,real_total=8432,synthetic_CUDA=40,CUDA_prespecified_failure_calls=4,formal_failed_sessions=0,new_source_updates=0,new_first_target_updates=0,old_CPU=64,new_CPU=70,new_CPU_attempts=3,alignment=alignment,formal_worker_seconds=sum(c['worker_seconds'] for c in training),peak_allocated_bytes=max(c['peak_cuda_allocated'] for c in training),sessions=x['costs'])
table('FINAL_METRICS.csv',final);table('DOMAIN_METRICS.csv',domain);table('PAIRED_COMPARISONS.csv',pairs)
write('ADAM_COUNTERFACTUAL.json',adam);write('GRADIENT_DIAGNOSTICS.json',gradient)
write('COST_AND_COMPLETION.json',dict(status=x['status'],costs=cost,counts=dict(final=7,domain=21,paired=9,adam=16),cancelled_nodes=x['cancelled_nodes'],original_pipeline_status=x['original_pipeline_status'],scope_change=x['scope_change'],stop_reason=x['stop_reason']))
write('PUBLIC_RESULTS.json',dict(status=x['status'],execution_commit=x['execution_commit'],seed=163,order=1,rows=rows,paired=pairs,gates='NOT_ASSESSED_REDUCED_SCOPE: requires two seeds and O2',automatic_followup=False))
write('COMPLETION_PROOF.json',dict(status='PASS_USER_REDUCED_SCOPE',execution_commit=x['execution_commit'],verified_nodes=4,formal_calls=8400,real_total=8432,diagnostic_points=16,integrity=x['integrity'],qualification={k:dict(status=v['status'],physical_calls=v['physical_calls'],records=len(v['rows'])) for k,v in x['qualification'].items()},original_16_node_completion=False))
for k,v in x['qualification'].items():write(k+'_QUALIFICATION.json',v)
lines=['# NKA_DOSE：seed163 / O1 快速比较','',
'**结论：原生键 C3 优于同剂量随机键 C2，但提高权重至0.5或2.0均未改善三域平均 Dice。** 权重2.0提高当前域表现，同时损失旧域表现。','',
'用户在运行中明确要求仅保留一个seed和一个域顺序，因此保留已执行的seed163/O1（REFUGE→RIM_ONE_r3→Drishti_GS）。4个新节点全部SEALED且VERIFIED，合计8400次正式更新；另3条历史结果只导入。其余12个节点取消。选择依据是既有执行顺序，不按最终成绩选择。','',
'## 指标','',
'Dice以百分数显示；Final为三域平均，Old为REFUGE与RIM平均，Incoming为Drishti。差值以百分点表示。', '',
'|组别|权重|来源|Final|Old|Incoming|相对C0 Final|','|---|---:|---|---:|---:|---:|---:|']
base=metric(index['C0',0.])
for r in final:lines.append(f"|{r['arm']}|{r['lambda_align']:g}|{'历史导入' if r['origin']=='historical_import' else '新执行'}|{r['Final']*100:.5f}|{r['Old']*100:.5f}|{r['Incoming']*100:.5f}|{(r['Final']-base['Final'])*100:+.5f}|")
lines+=['','## 配对结果','','|权重|比较|ΔFinal|ΔOld|ΔIncoming|','|---:|---|---:|---:|---:|']
for p in pairs:lines.append(f"|{p['lambda_align']:g}|{p['contrast']}|{p['Final']*100:+.5f}|{p['Old']*100:+.5f}|{p['Incoming']*100:+.5f}|")
lines+=['','## 工程与诊断','',
'CUDA40/40（18条记录）与L-only smoke32/32通过；4次预设CUDA失败计入物理成本。正式4/4完成、验收通过，正式失败session为0。真实总更新8432；源模型和第一目标新增训练0。旧CPU64、新CPU70单列，未再运行CPU suite。',
f"正式工作session合计{cost['formal_worker_seconds']/3600:.3f}小时，包含训练与评价；峰值分配显存{cost['peak_allocated_bytes']/1024**3:.3f}GiB。共享GPU环境下，这不是独占GPU耗时。",
'固定诊断16点全部通过：64额外VJP、32只读Adam候选、6720 clean-U前向。每节点1680个active-U步骤均有合法辅助支持；所有诊断点候选与真实更新的最大参数误差为0。',
'原生键C3的上游有效权重局部变化比例rho_W：lambda0.5约0.0031–0.0452，lambda2.0约0.0113–0.1163。辅助项确实改变了更新，但更大的局部扰动未转化为总体Dice收益。这只是当前剂量轨迹上的一步反事实，不代表完整无辅助轨迹；历史Adam计量未补造。','',
'## 停止与解释边界','',
'第四个节点完成模型验收后，下一节点入口的权限屏障按用户缩减要求触发PermissionError。旧控制器将退出统一记作ENGINEERING_STOP，该原始状态保留；本报告据用户范围变更及四个完整验收记录单独登记COMPLETE_USER_REDUCED_SCOPE。没有OOM或非有限值导致的正式训练失败。未启动被取消节点，不重放任何更新。',
'本结果仅来自一个seed和一个域顺序。原G_perf/G_key要求两个seed及O2，因此记NOT_ASSESSED_REDUCED_SCOPE，不伪报原16节点矩阵完成，也不做显著性或跨seed稳定性声明。不能据此宣称独立患者泛化、完整新方法轨迹、原始KI复现或SOTA。没有追加实验或自动进入后续完整轨迹。','',
'执行代码：`'+x['execution_commit']+'`。历史聚合来源：`0da596f4050a3701f41ff74248ca9d4ba21222e2`；B2前缀公开来源：`3034b2199aa7d6e54ea67499f391a0dd4ea3d21e`。结果在独立分支发布，原执行checkout、审批及完整计划保持不变。','',
'复核：在仓库内执行 `python3 results/nka_dose_v0_1_s163_o1/summarize.py`，仅读取公开聚合元数据，重建表格并检查范围、配对、诊断计数与已保存验收摘要；不读取模型或患者数据、不运行训练，也不声称该离线脚本重新验收了真实模型文件。']
(D/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='PASS',rows=len(final),pairs=len(pairs),formal_hours=cost['formal_worker_seconds']/3600)))
