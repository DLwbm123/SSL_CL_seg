"""V2 fine-relative reanalysis and grouped OOF analysis; no validation data access."""
import csv,json,math
from collections import defaultdict,Counter
import numpy as np
import torch
from .execution import ROOT,PREVIOUS,CONFIG,CONFIG_SHA,CODE,c,read,write,records,compatible
from . import control

def table(name,rows,private=False):
    p=ROOT/name if private else ROOT/'reports'/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}),lineterminator='\n');w.writeheader();w.writerows(rows)

def sign(x,delta=0.):return 1 if x>delta else -1 if x< -delta else 0

def paired(row):
    grouped=defaultdict(list)
    for a,o,u in row:grouped[a].append((o,u))
    duplicate=max([max(v[i] for v in vs)-min(v[i] for v in vs) for vs in grouped.values() for i in (0,1)]+[0.]);means={a:np.mean(v,axis=0) for a,v in grouped.items()}
    if 2 not in means:return dict(fine=False,complete=False,duplicate_max=duplicate,duplicate_records=sum(len(v)-1 for v in grouped.values()))
    d={a:v-means[2] for a,v in means.items()};out=dict(fine=True,complete=len(d)==3,duplicate_max=duplicate,duplicate_records=sum(len(v)-1 for v in grouped.values()))
    for a in (0,1):
        if a in d:out.update({f'online_{a}':float(d[a][0]),f'audit_{a}':float(d[a][1]),f'rank_agreement_{a}':float(sign(d[a][0])==sign(d[a][1]))})
    if len(d)==3:
        online=[d[a][0] for a in range(3)];audit=[d[a][1] for a in range(3)];a=control.choose(online);out.update(oracle=max(audit),transfer=audit[a],deviate=int(a!=2),wrong=int(a!=2 and audit[a]<0),selected_positive=int(a!=2 and audit[a]>0),selected_uncertain=int(a!=2 and audit[a]==0),chosen=a)
    return out

def p0():
    out=ROOT/'reports';out.mkdir(exist_ok=True);rows=[];sources={};audit={x['task']:x for x in read(PREVIOUS/'reports/FINAL_STATE_AUDIT.json')};binding=read(PREVIOUS/'DATA_BINDING.json')
    for p in sorted((PREVIOUS/'tasks').glob('R3B*/EVALUATION.json')):
        e=read(p)
        if e['arm'] not in ('RL','NC_RL','REG'):continue
        assert e['complete'] and audit[e['task']]['passed'] and all(e[k]==audit[e['task']][k if k!='code_commit' else 'training_commit'] for k in ('code_commit','prefix_sha','schedule_sha','config_sha'));assert e['schedule_sha']==binding['schedules'][f"{e['seed']}__{e['domain']}"]
        successful={x['transaction'] for x in records(p.parent/'TRANSACTIONS.jsonl') if x['event']=='optimizer_success'};sources[e['task']]=dict(evaluation_sha=c.file_sha(p),training_commit=e['code_commit'],prefix_sha=e['prefix_sha'],schedule_sha=e['schedule_sha'])
        for f in sorted((p.parent/'decisions').glob('*.private.json'),key=lambda p:int(p.name.split('.')[0])):
            d=read(f);t=d['meta']['t'];assert int(f.name.split('.')[0])==t and d['meta']['arm']==e['arm'];b=d['feedback'];assert f'{t}/anchor' in successful and all(f'{t}/probe/{j}' in successful for j in range(4));raw=[(x['action'],x['online'],x['audit']) for x in [b['anchor']]+b['candidates'] if x['online'] is not None and x['audit'] is not None]
            pair=paired(raw);non_skip=[(o-b['anchor']['online'],u-b['anchor']['audit']) for a,o,u in raw if a!=0 and b['anchor']['online'] is not None and b['anchor']['audit'] is not None];pair['non_skip_sign_agreement']=sum(sign(o)==sign(u) for o,u in non_skip)/len(non_skip) if non_skip else None;rows.append(dict(method=e['arm'],seed=e['seed'],backbone=e['backbone'],domain=e['domain'],t=t,**pair))
    for f in sorted((PREVIOUS/'tasks').glob('R3A*/R3A.private.json')):
        done=read(f.parent/'R3A_DONE.json');assert done['contexts']==16 and done['retained']==0;successful={x['transaction'] for x in records(f.parent/'TRANSACTIONS.jsonl') if x['event']=='optimizer_success'};sources[f.parent.name]=dict(source_sha=c.file_sha(f),training_commit=done['code_commit'],prefix_sha=done['prefix_sha'],schedule_sha=done['schedule_sha'])
        for row in read(f):
            j=row['context'];assert all(f'audit/{j}/{k}' in successful for k in range(5));raw=[(p['action'],row['online_rewards'][k],row['audit_rewards'][k]) for k,p in enumerate(row['candidates']) if row['online_rewards'][k] is not None and row['audit_rewards'][k] is not None];rows.append(dict(method='R3A',seed=261,backbone=done['backbone'],domain=done['domain'],t=row['schedule_t'],**paired(raw)))
    summary=[];coverage=[]
    for key in sorted({(x['method'],x['seed'],x['backbone'],x['domain']) for x in rows}):
        v=sorted([x for x in rows if (x['method'],x['seed'],x['backbone'],x['domain'])==key],key=lambda x:x['t']);base=dict(zip(('method','seed','backbone','domain'),key));coverage.append(dict(**base,total=len(v),fine_observed=sum(x['fine'] for x in v),complete_panels=sum(x['complete'] for x in v),duplicate_max=max(x['duplicate_max'] for x in v),duplicate_records=sum(x['duplicate_records'] for x in v)))
        for part,w in [('all',v),('early10',v[:10]),('late10',v[-10:])]:
            stats={}
            for k in ('online_0','online_1','audit_0','audit_1','oracle','transfer','deviate','wrong','selected_positive','selected_uncertain','rank_agreement_0','rank_agreement_1','non_skip_sign_agreement'):
                a=[x[k] for x in w if x.get(k) is not None];stats[k]=float(np.mean(a)) if a else None;stats[k+'_n']=len(a)
            n=sum(x.get('deviate',0) for x in w)
            for name in ('wrong','selected_positive','selected_uncertain'):stats[name+'_among_deviations']=sum(x.get(name,0) for x in w)/n if n else None
            summary.append(dict(**base,segment=part,**stats))
    table('PAIR_COVERAGE.csv',coverage);table('FINE_RELATIVE_LOG_SUMMARY.csv',summary);write(ROOT/'P0_SOURCES.json',sources)
    (out/'P0_REANALYSIS.md').write_text('# P0: valid V2 logs only\n\nAll 36 adaptive trajectories and four valid R3a panels included. Numeric t ordering defines early/late. Fine-unobserved contrasts remain missing; repeated actions are averaged only within one decision. Complete-panel statistics are conditional on observed coverage. No IPS/DR and no independent-sample claim. Non-skip sign agreement excludes skip self-zero. Medical forward/optimizer/controller calls: 0.\n')

def panels():
    rows=[];summary=[];rest=[];diagnostics=[];manifest=read(ROOT/'CONTEXT_MANIFEST.private.json');values={}
    for cell in manifest:
        f=ROOT/'panels'/cell;paths=list(f.glob('*.private.json'))
        if len(paths)!=52:rest.append(dict(cell=cell,complete=False,branches=len(paths),reason='missing complete branch receipts'));continue
        raw={(x['scene'],x['action'],x['repeat']):x for x in map(compatible,paths)};assert all(x['root_restored'] and x['retained_after_discard']==0 for x in raw.values());cellvalues=[]
        missing=sum(v is None or not np.isfinite(v) for x in raw.values() for q in x['horizons'].values() for v in q.values())
        if missing:rest.append(dict(cell=cell,complete=False,branches=52,missing_feedback=missing));continue
        for a in range(3):
            steps=[s for x in raw.values() if x['action']==a and x['repeat']==0 for s in x['steps']];diag=dict(cell=cell,action=a,updates=len(steps),teacher_student_l2_mean=float(np.mean([s['teacher_student_l2'] for s in steps])))
            for k in ('LU','U_grad_norm','grad_norm'):
                v=[s['info'][k] for s in steps if s['info'].get(k) is not None];diag[k+'_mean']=float(np.mean(v)) if v else None
            for k in ('1','2'):
                v=[s['info']['coverage'][k] for s in steps if k in s['info']['coverage']];diag['coverage_'+k]=float(np.mean(v)) if v else None
            diagnostics.append(diag)
        for h in (1,5):
            e=max(abs(raw[(j,a,0)]['horizons'][str(h)][role]-raw[(j,a,1)]['horizons'][str(h)][role]) for j in (0,8) for a in (0,2) for role in ('online','audit'));delta=max(1e-6,5*e);ds=[]
            for j in range(16):
                online=np.array([raw[(j,a,0)]['horizons'][str(h)]['online'] for a in range(3)],float);audit=np.array([raw[(j,a,0)]['horizons'][str(h)]['audit'] for a in range(3)],float);assert np.isfinite(online).all() and np.isfinite(audit).all();online-=online[2];audit-=audit[2];a=control.choose(online,delta);a_raw=control.choose(online);z=raw[(j,2,0)]['state'];ds.append((online,audit,a,a_raw));cellvalues.append(dict(scene=j,h=h,state=z,online=online.tolist(),audit=audit.tolist(),delta=delta))
                for action in range(3):rows.append(dict(cell=cell,scene=j,h=h,action=action,online=float(online[action]),audit=float(audit[action])))
            row=dict(cell=cell,h=h,scenes=16,delta_numeric=delta,null_max=e,oracle=float(np.mean([max(a) for _,a,_,_ in ds])),transfer=float(np.mean([u[a] for _,u,a,_ in ds])),raw_transfer=float(np.mean([u[a] for _,u,_,a in ds])),deviations=sum(a!=2 for _,_,a,_ in ds),wrong_interventions=sum(a!=2 and u[a]<-delta for _,u,a,_ in ds),uncertain_nonfine=sum(max(o[:2])<=delta and max(o[:2])>=-delta for o,_,_,_ in ds))
            for action in (0,1):
                v=np.array([u[action] for _,u,_,_ in ds]);selected=[u[action] for _,u,a,_ in ds if a==action];row.update({f'a{action}_mean':float(v.mean()),f'a{action}_q25':float(np.quantile(v,.25)),f'a{action}_median':float(np.median(v)),f'a{action}_q75':float(np.quantile(v,.75)),f'a{action}_raw_positive':int((v>0).sum()),f'a{action}_raw_negative':int((v<0).sum()),f'a{action}_raw_zero':int((v==0).sum()),f'a{action}_wrong_rate':float(np.mean(np.array(selected)<-delta)) if selected else None,f'a{action}_positive':int((v>delta).sum()),f'a{action}_negative':int((v< -delta).sum()),f'a{action}_uncertain':int((np.abs(v)<=delta).sum()),f'a{action}_selected':len(selected),f'a{action}_value_per_intervention':float(np.mean(selected)) if selected else None})
            summary.append(row)
        values[cell]=cellvalues;change=sum(sign(cellvalues[j]['audit'][a])!=sign(cellvalues[16+j]['audit'][a]) for j in range(16) for a in (0,1));rest.append(dict(cell=cell,branches=52,root_restored=True,retained_updates=0,teacher_per_step=True,adam_steps=[2001,2002,2003,2004,2005],scheduler=[1,2,3,4,5],nonfine_horizon_sign_changes=change,compared_pairs=32,coarse_skip_rank_changes=sum(sign(cellvalues[j]['audit'][0]-cellvalues[j]['audit'][1])!=sign(cellvalues[16+j]['audit'][0]-cellvalues[16+j]['audit'][1]) for j in range(16)),complete=True))
    table('COUNTERFACTUAL_VALUES.private.csv',rows,True);write(ROOT/'P1_VALUES.private.json',values);table('HORIZON_TRANSFER.csv',summary);table('BRANCH_DIAGNOSTICS.csv',diagnostics);write(ROOT/'reports/NULL_AND_RESTORE_AUDIT.json',rest)

def p2(ledger):
    values=read(ROOT/'P1_VALUES.private.json');split=read(ROOT/'OOF_SPLITS.private.json');predictions=[];scales=read(ROOT/'SCALE_BINDINGS.json');analytic=Counter()
    for cell,rows in values.items():
        if not split[cell]['eligible']:continue
        scale_path=PREVIOUS/'tasks'/('R3A__'+cell)/'R3A_DONE.json';assert c.file_sha(scale_path)==scales[cell]['sha256'];sc=scales[cell]['scale']
        for h in (1,5):
            hs=sorted([x for x in rows if x['h']==h],key=lambda x:x['scene']);z=np.array([x['state'] for x in hs]);online=np.array([x['online'] for x in hs]);fold=np.array(split[cell]['fold'])
            for k in range(4):
                tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);train=online[tr]/sc;fit_seed=control.seed(cell,h,k);probs={'FINE':np.tile([0.,0.,1.],(len(te),1)),'STATIC_PRIOR':np.tile(control.PRIOR,(len(te),1)),'NC_OPT':np.tile(control.optimum(train.mean(0)),(len(te),1)),'LIN_VALUE':control.ridge(z[tr],train,z[te])}
                analytic['NC_OPT_solutions']+=1;analytic['LIN_VALUE_fits']+=1;analytic['LIN_VALUE_heldout_solutions']+=len(te)
                for method in ('FI_POLICY','FI_POLICY_SHUFFLE'):
                    name=f'{cell}/{h}/{k}/{method}';p=ROOT/'fits'/(name.replace('/','_')+'.private.json')
                    if p.exists():v=compatible(p);assert v['config_sha']==CONFIG_SHA;probs[method]=np.array(v['probabilities'])
                    else:
                        model,diag=control.fit(z[tr],train,fit_seed,ledger.step,name,method.endswith('SHUFFLE'));prob=model(torch.tensor(z[te],dtype=torch.float32)).detach().numpy();write(p,dict(code_commit=CODE,config_sha=CONFIG_SHA,train_indices=tr.tolist(),test_indices=te.tolist(),scale_sha=scales[cell]['sha256'],probabilities=prob.tolist(),diagnostics=diag));probs[method]=prob
                # Independent scoring only after every held-out prediction has been frozen.
                for method,pr in probs.items():
                    for i,p in zip(te,pr):
                        u=np.array(hs[i]['audit']);predictions.append(dict(cell=cell,h=h,fold=k,scene=int(i),method=method,value=float(p@u),intervention_probability=float(p[:2].sum()),negative_mass=float(p[:2][u[:2]< -hs[i]['delta']].sum()),p_skip=float(p[0]),p_coarse=float(p[1]),p_fine=float(p[2])))
    write(ROOT/'reports/ANALYTIC_COSTS.json',dict(analytic));table('OOF_PREDICTIONS.private.csv',predictions,True);summary=[]
    for key in sorted({(x['cell'],x['h'],x['method']) for x in predictions}):
        v=[x for x in predictions if (x['cell'],x['h'],x['method'])==key];mass=sum(x['intervention_probability'] for x in v);summary.append(dict(cell=key[0],h=key[1],method=key[2],scenes=len(v),U_context_groups=len(set(split[key[0]]['groups'])),value=float(np.mean([x['value'] for x in v])),intervention_probability=mass/len(v),value_per_intervention=sum(x['value'] for x in v)/mass if mass else None,negative_mass=float(np.mean([x['negative_mass'] for x in v])),p_skip=float(np.mean([x['p_skip'] for x in v])),p_coarse=float(np.mean([x['p_coarse'] for x in v])),p_fine=float(np.mean([x['p_fine'] for x in v]))))
    table('OOF_POLICY_VALUES.csv',summary);contrasts=[]
    for cell in values:
        for h in (1,5):
            lookup={x['method']:x['value'] for x in summary if x['cell']==cell and x['h']==h}
            if 'FI_POLICY' in lookup:
                for method in ('FINE','STATIC_PRIOR','NC_OPT','LIN_VALUE','FI_POLICY_SHUFFLE'):contrasts.append(dict(cell=cell,h=h,contrast='FI_POLICY-'+method,delta=lookup['FI_POLICY']-lookup[method]))
    table('POLICY_CONTROL_SUMMARY.csv',contrasts)

def report():
    out=ROOT/'reports';policy=list(csv.DictReader((out/'OOF_POLICY_VALUES.csv').open()));horizon=list(csv.DictReader((out/'HORIZON_TRANSFER.csv').open()));primary=[x for x in policy if x['method']=='FI_POLICY' and x['h']=='5'];cells={x['cell'] for x in primary}
    means={method:float(np.mean([float(x['value']) for x in policy if x['method']==method and x['h']=='5'])) for method in ('FINE','STATIC_PRIOR','NC_OPT','LIN_VALUE','FI_POLICY','FI_POLICY_SHUFFLE') if any(x['method']==method and x['h']=='5' for x in policy)}
    hs={x['cell']:x for x in horizon if x['h']=='5'};above=sum(float(x['value'])>float(hs[x['cell']]['delta_numeric']) for x in primary);backbones={b:float(np.mean([float(x['value']) for x in primary if x['cell'].startswith(b+'__')])) for b in CONFIG['backbones'] if any(x['cell'].startswith(b+'__') for x in primary)}
    pilot=bool(len(cells)==4 and above>=3 and len(backbones)==2 and all(v>0 for v in backbones.values()) and all(means['FI_POLICY']>means[m] for m in ('STATIC_PRIOR','NC_OPT','FI_POLICY_SHUFFLE')))
    summary=dict(primary_horizon=5,cells=len(primary),mean_value=means.get('FI_POLICY'),method_values=means,backbone_mean=backbones,cells_above_numeric_dead_zone=above,closed_loop_proposal_criterion=pilot,unit='feedback quality difference, NOT Dice or Dice percentage points',student_endpoints=0,persistent_student_updates=0);write(out/'PRIMARY_RESULT.json',summary)
    opportunity=float(np.mean([float(x['oracle']) for x in hs.values()])) if hs else None;transfer=float(np.mean([float(x['transfer']) for x in hs.values()])) if hs else None
    text=['# V3A final interpretation','', 'Primary horizon: h5. Values are feedback quality differences, never Dice percentage points. Missing cells remain missing; an incomplete four-cell matrix cannot meet the proposal criterion.', '', '## 1. Measured opportunity', f'Mean finite-panel post-hoc optimistic oracle: {opportunity}. Compare each cell with its recorded numeric dead zone. This noisy observed maximum is neither a generalization bound nor long-term policy value.', '', '## 2. Feedback transfer', f'Online-selected, audit-scored value: {transfer}. HORIZON_TRANSFER reports raw and dead-zone choices, wrong interventions and per-action counts. Positive oracle with nonpositive transfer points to feedback selection limitations.', '', '## 3. State predictability', f'h5 method values: {json.dumps(means,sort_keys=True)}. Compare LIN_VALUE and FI_POLICY against NC_OPT and SHUFFLE before attributing value to state. A single shuffle is a negative control, not a permutation test.', '', '## 4. Policy optimization', 'SYNTHETIC_CONTROLLABILITY records all four seeds, initial/final probabilities and true expected values. OOF_POLICY_VALUES evaluates fixed 256-step fits. Better than prior while below FINE remains a loss; a stronger linear control does not establish an RL-specific advantage.', '', '## 5. Future closed-loop evidence', f'Cells exceeding their numeric dead zone: {above}/4. Backbone means: {json.dumps(backbones,sort_keys=True)}. Frozen proposal criterion met: {pilot}. This only assesses grounds for proposing a separately authorized pilot; no V3B or R3c is launched.', '', '## Limits and completeness', 'U-context grouped OOF is not patient or L-feedback identity independence. Shared feedback images and one entry prefix per cell limit inference about later training states. h1 is explanatory only. Old R3a scales are frozen and not tuned. No statistical significance, independent confirmation, retained medical student update or new student endpoint is claimed. See NULL_AND_RESTORE_AUDIT, RESOURCE_REPORT and FINAL for missing feedback, engineering failures and charged physical calls.']
    (out/'FINAL_INTERPRETATION.md').write_text('\n'.join(text)+'\n')
