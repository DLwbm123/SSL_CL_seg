"""One final receipt/checkpoint audit and anonymous CPU analysis; no model forwards."""
import json,collections,subprocess
from pathlib import Path
import numpy as np,torch
from .runtime import finite,exact
from .single_target_audit import paired
from ..main_style_ssl_v1.runtime import write


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def retired(root):
    launch=read(root/'LAUNCH.private.json');pg=int(launch.get('process_group',launch['pid']))
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True).splitlines()[1:]:
        fields=line.split();assert int(fields[1])!=pg or fields[2].startswith('Z'),'managed group still active'
    for path in [root/'LAUNCH.private.json',root/'WORKER.private.json',*root.glob('ACTIVE_*.private.json')]:
        if not path.exists():continue
        r=read(path);p=Path('/proc')/str(r['pid'])/'stat'
        if p.exists():
            s=p.read_text().split();assert s[21]!=str(r['birth']) or s[2]=='Z',path.name


def source(cfg):
    root=Path(cfg['root']);assert not (root/'AUDIT.json').exists();assert read(root/'STATUS.json')['status']=='COMPLETE';retired(root);assert read(root/'EXIT.json')['exit_code']==0
    result=read(root/'RESULTS.json');counter=read(root/'budget_counter.json');events=rows(root/'budget_events.jsonl');fw=read(root/'forward_counter.json')
    assert counter['total']==counter['optimizer']==336 and counter['functional']==counter['pending']==0
    expected={'qualification:engineering':32,'construction:formal':64,**{s+'_'+b+':formal':16 for s in ['s0','s16','s64','sEnc','sDec'] for b in ['B0','BM','BR']}}
    assert counter['by_phase']==expected and collections.Counter(e['key'] for e in events if e['kind']=='optimizer')==expected
    assert sum(e['kind']=='reserved' for e in events)==336 and fw['pending'] is None and all(fw['executed'][k]<=v for k,v in fw['caps'].items())
    q=read(root/'PARTITION.private.json');assert len(q['support'])==7 and len(q['query'])==3 and not set(q['support'])&set(q['query'])
    assert fw['executed']['real_query']==35*q['query_slices']+10 and fw['executed']['synthetic_native']==680
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    entry=load(root/'s0.pt');assert not entry['optimizer']['state'];finite(entry)
    for s in ['s0','s16','s64','sEnc','sDec']:
        state=load(root/(s+'.pt'));finite(state)
        assert bool(state['optimizer']['state'])==(s in ['s16','s64'])
        for b in ['B0','BM','BR']:
            v=load(root/(s+'_'+b+'_final.pt'));finite(v);assert v['step']==16 and all(k in v for k in ['model','optimizer','torch_rng','cuda_rng','reads'])
            assert all(torch.equal(t,entry['model'][k]) for k,t in v['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
    audit=dict(status='PASS',updates=counter,forward=fw,engineering_status=read(root/'ENGINEERING.json')['status'],source_train_L_only=True,query_not_optimized=True,query_not_independent_of_prior_P_G_M=True,workers_retired=True,no_audit_model_forward=True,old_endpoints_unchanged=True)
    write(root/'AUDIT.json',audit)
    lines=['# 源功能有限步诊断结果','',f"实际336次更新（工程32，正式304），0 functional；参考灵敏度：{result['reference_sensitive']}，预设记忆门槛：{result['memory_gate_pass']}。",'', '|状态|BR query CE收益(h16)|BM query CE收益(h16)|BM−B0 前景Dice(pp)|','|---|---:|---:|---:|']
    for i,(s,v) in enumerate(result['results'].items()):
        d=100*(v['branches']['BM']['16']['query']['mean_foreground_Dice']-v['branches']['B0']['16']['query']['mean_foreground_Dice'])
        lines.append(f"|{s}|{result['reference_CE_utility_h16'][i]:.7f}|{result['memory_CE_utility_h16'][i]:.7f}|{d:+.4f}|")
    lines+=['','效用为相对B0的query CE降低，正值有利。各状态h1/h16、条件前景CE、LV/MYO/RV、梯度方向与实际encoder/decoder位移完整保存于匿名RESULTS。方向相似不能替代有限步效用。','', '工程首轮固定两例全背景、入口CE低于绝对下降阈值，原32次已计并保留失败；零更新检查证实阈值在非负CE下不可达，按启动正式前修订改为golden精确/finite与既有非空恢复证据。没有重复资格更新，generator易失缓存恢复额外32次全部计入64硬cap，时钟不重置。','', '这是源train-L开发功能诊断；P/G/M已见过全部10患者，7/3划分不构成独立泛化。没有源val、目标验证或U隐藏GT参与状态/更新；未改旧终点，未启动条件后续。','',f"实际前向：{json.dumps(fw['executed'],ensure_ascii=False)}。一次必要审计PASS，未额外模型评分。"]
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write(root/'PUBLIC_SUMMARY.json',dict(audit=audit,results=result))


def contribution(cfg):
    root=Path(cfg['root']);old=Path(cfg['comparison_root']);assert not (root/'AUDIT.json').exists();assert read(root/'STATUS.json')['status']=='COMPLETE';retired(root);assert read(root/'EXIT.json')['exit_code']==0
    groups=['S','G00','G10','G01','G11'];counter=read(root/'budget_counter.json');events=rows(root/'budget_events.jsonl')
    expected={g+':'+p:n for g in groups for p,n in [('engineering',32),('formal',2000)]}
    assert counter==dict(total=10160,optimizer=10160,functional=0,pending=0,by_phase=expected)
    assert collections.Counter(e['key'] for e in events if e['kind']=='optimizer')==expected and sum(e['kind']=='reserved' for e in events)==10160
    assert read(root/'STREAM_SELF_CHECK.json')['status']=='PASS'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False);base=load(old/'jobs/A/initial.pt');metrics={};sampling={};paired_groups={}
    for g in groups:
        out=root/'jobs'/g
        for stage in ['qualify','train','evaluate']:assert read(root/('EXIT_'+g+'_'+stage+'.json'))['exit_code']==0
        for n in ['EXIT.json','EVALUATION_EXIT.json']:assert read(out/n)['exit_code']==0
        assert read(root/'qualification'/g/'ENGINEERING.json')['actual_calls']==32 and len(rows(out/'physical_updates.jsonl'))==2000
        initial=load(out/'initial.pt');final=load(out/'final.pt');latest=load(out/'latest.pt');finite(final)
        assert exact(final,latest) and final['step']==final['physical_updates']==2000 and not initial['optimizer']['state']
        assert all(torch.equal(v,base['model'][k]) for k,v in initial['model'].items() if k.startswith('parent.'))
        assert not any(k=='code' or k.startswith('generator.') for k in final['model'])
        assert all(torch.equal(v,initial['model'][k]) and torch.equal(v,final['ema'][k]) for k,v in final['model'].items() if not k.startswith('parent.') or k.endswith(('sigma.weight','grad_update')))
        metrics[g]=read(out/'RESULTS.json');assert metrics[g]['source_train']['patients']==10 and metrics[g]['target_train']['patients']==13
        if g!='S':
            ss=rows(out/'sampling_aggregates.jsonl');assert len(ss)==2000 and {s['step'] for s in ss}==set(range(2000)) and all(s['images']==16 and s['fill_labels_read']==0 for s in ss)
            sampling[g]=dict(mean_distinct_images=float(np.mean([s['distinct_images'] for s in ss])),mean_repeat_fraction=float(np.mean([s['repeat_fraction'] for s in ss])),fill_pool=ss[0]['fill_pool'])
    comparisons={};contrasts={'U':{'G11':1,'G01':-1},'frame':{'G11':1,'G10':-1},'full':{'G11':1,'S':-1},'interaction':{'G11':1,'G01':-1,'G10':-1,'G00':1}}
    for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
        comparisons[mode]={}
        for domain,n in [('Siemens',23),('Philips',31)]:
            rr={g:rows(root/'jobs'/g/(prefix+domain+'_val.jsonl')) for g in groups};keys=lambda a:{(r['patient'],r['phase']) for r in a}
            assert all(keys(v)==keys(rr['S']) and len(v)==2*n for v in rr.values())
            d={}
            for name,coefs in contrasts.items():
                tables={g:{(r['patient'],r['phase']):np.array(r['dice']) for r in rr[g]} for g in coefs};patient=sorted({k[0] for k in tables[next(iter(coefs))]})
                values=np.array([sum(c*np.mean([tables[g][p,phase] for phase in ['ED','ES']],axis=0) for g,c in coefs.items()) for p in patient])*100
                rng=np.random.default_rng(163);mean=values.mean(1)
                d[name]=dict(delta_pp=float(mean.mean()),classes_pp=dict(zip(['LV','MYO','RV'],values.mean(0).tolist())),patient_bootstrap95_pp=np.quantile(mean[rng.integers(n,size=(10000,n))].mean(1),[.025,.975]).tolist(),patients=n,increased=int((mean>0).sum()),decreased=int((mean<0).sum()),unchanged=int((mean==0).sum()),median_patient_delta_pp=float(np.median(mean)),phases_pp={phase:float(np.mean([sum(c*tables[g][p,phase] for g,c in coefs.items()) for p in patient])*100) for phase in ['ED','ES']})
            comparisons[mode][domain]=d
            paired_groups[mode+':'+domain]={g:paired(rr['S'],rr[g]) for g in groups if g!='S'}
    gates={}
    for mode,d in comparisons.items():
        a,b=d['Siemens'],d['Philips'];gates[mode]=dict(U=b['U']['delta_pp']>=.5 and a['U']['delta_pp']>=-.5,frame=a['frame']['delta_pp']>=.3 and b['frame']['delta_pp']>=-.3,full=(a['full']['delta_pp']>=.5 and b['full']['delta_pp']>=-.5) or (b['full']['delta_pp']>=.5 and a['full']['delta_pp']>=-.5))
    audit=dict(status='PASS',cost=counter,common_native_entry=True,empty_optimizer=True,no_generator_or_code=True,frozen_P_frame_constants=True,finite_final_latest_opt_RNG_reads=True,patient_EDES_exact=True,workers_retired=True,stream_self_check=True,audit_model_forwards=0)
    summary=dict(metrics=metrics,comparisons=comparisons,paired_vs_S=paired_groups,sampling=sampling,gates=gates,audit=audit,primary='fixed last student',EMA_uniform_secondary=True,single_target_stream=20261020)
    write(root/'AUDIT.json',audit);write(root/'PUBLIC_SUMMARY.json',summary)
    lines=['# N五组贡献分解完整结果','', '统一新目标随机流20261020；固定最后2000 student为主，EMA .999统一辅助，旧A/B/C/M/N终点不变。','', '|组|student旧/新Dice %|EMA旧/新Dice %|','|---|---:|---:|']
    for g,v in metrics.items():lines.append(f"|{g}|{100*v['Siemens']['primary_dice']:.6f}/{100*v['Philips']['primary_dice']:.6f}|{100*v['EMA_diagnostic']['Siemens']['primary_dice']:.6f}/{100*v['EMA_diagnostic']['Philips']['primary_dice']:.6f}|")
    lines+=['','|口径/对比|旧域pp [患者95%区间]|新域pp [患者95%区间]|','|---|---:|---:|']
    for mode,dom in comparisons.items():
        for name in contrasts:
            a,b=dom['Siemens'][name],dom['Philips'][name]
            lines.append(f"|{mode}/{name}|{a['delta_pp']:+.6f} {a['patient_bootstrap95_pp']}|{b['delta_pp']:+.6f} {b['patient_bootstrap95_pp']}|")
    lines+=['',f"冻结门槛判定：{json.dumps(gates,ensure_ascii=False)}。",'', 'S算力较低；无U使用L池额外14图但不读取其标签，无frame仍用2/3*c及原分母，原始几何loss不能横比。保留源训练固定P，不能称完全无源信息。患者损益、类别/EDES、bootstrap、中位差、训练口径、入口变化及图像重复率在PUBLIC_SUMMARY完整保留；不使用双域均值掩盖代价。','', '这是一个目标训练随机流的开发验证，患者bootstrap不覆盖训练随机性或反复开发偏差。EMA/student分别完整报告，不事后择优；交互可解释为本操作定义下的差分，不能无依据扩展为普遍因果协同。未启动第二流、SG/T、F或其他条件后续。一次收据/CPU审计PASS，无审计模型前向。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
