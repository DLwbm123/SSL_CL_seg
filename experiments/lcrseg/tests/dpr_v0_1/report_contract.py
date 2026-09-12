"""Zero-training synthetic report/queue check. No real data, model, or remote access."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
from experiments.lcrseg.dpr_v0_1 import contract as ct,results as r,execute as ex

def run():
    p=ct.protocol();source='SYNTHETIC_REPORT_FIXTURE';docs={};tables={}
    def read(path):return docs[str(path)] if str(path) in docs else json.loads(Path(path).read_text())
    def put(path,obj):docs[str(path)]=obj
    with tempfile.TemporaryDirectory() as tmp:
        b=Path(tmp);inputs=dict(parent=str(b/'parent'),baseline_original=str(b/'old'));sb=dict(sources={});bb=dict(baselines={});students={}
        def score(path,tid,arm,domains,src):
            rows=[];summary=[]
            for domain in domains:
                v=.68 if arm=='SRC_CE' or (arm=='STATIC_SOURCE' and domain==src) else (.5 if domain==src else .7)
                v+={'DPR_U':.01,'RESPONSE_U':.003,'DPR_L':.005}.get(arm,0)
                rr=[dict(task_id=tid,domain=domain,patient_index=i,**{m:v for m in r.METRICS}) for i in range(r.c.COUNTS[domain][2])];rows+=rr
                summary.append(dict(task_id=tid,arm=arm,domain=domain,role='old' if domain==src else 'current',**{m:v for m in r.METRICS}))
            tables[str(path/'private_patient_metrics.csv')]=rows;tables[str(path/'metrics.csv')]=summary
        for src in p['sources']:
            tid=src['task_id'];sb['sources'][tid]=dict(student_hash=src['student_hash'],source_scores_sha256='fixture')
            score(b/'parent/tasks'/tid/'evaluation',tid,'SRC_CE',[src['domain']],src['domain'])
            for arm in ('F_FULL','F_CONV','STATIC_SOURCE'):
                bid=tid.replace('SRC_CE',arm);bb['baselines'][bid]=dict(storage='original',score_sha256='fixture',label_order_hash='fixture')
                domains=[next(d for d in r.c.DOMAINS if d!=src['domain'])] if arm=='STATIC_SOURCE' else list(r.c.DOMAINS)
                score(b/'old/tasks'/bid/'evaluation',bid,arm,domains,src['domain'])
        for task in p['tasks']:
            tid=task['task_id'];root=b/'tasks'/tid;root.mkdir(parents=True);(root/'deploy_student.pt').write_bytes(b'SYNTHETIC_NOT_A_MODEL')
            (root/('checkpoint_'+str(task['updates'])+'.pt')).write_bytes(b'SYNTHETIC_NOT_A_MODEL')
            src=next(x for x in p['sources'] if x['task_id']==task['source_task_id']);students[tid]=dict(student_hash='fixture')
            put(root/'receipt.json',dict(status='TRAINING_COMPLETE',source=source,task=task,updates=task['updates'],student_hash='fixture',memory=dict(full_models=2),boundary=dict(source_student_hash=src['student_hash']),fixed_subset_unchanged=True,EMA_updates=task['updates'],pseudo_labels=0,U_opens=0 if task['arm']=='DPR_L' else 80*r.c.COUNTS[task['domain']][1],L_opens=task['updates']*2,seconds=0,label_order_hash='fixture'))
            put(root/'evaluation/receipt.json',dict(source=source,student_hash='fixture',models=1))
            put(root/('diagnostics_pass_'+str(task['updates'])+'.json'),dict(status='PASS'))
            score(root/'evaluation',tid,task['arm'],r.c.DOMAINS,src['domain'])
            for pos in task['actual_diagnostic_positions']:put(root/('actual_'+str(pos)+'.json'),dict(position=pos,raw_response_RMS=1.,corrected_response_RMS=.9))
            for phase in ('train','eval'):
                put(b/'logs'/(tid+'_'+phase+'_exit.json'),dict(exit_code=0))
                counts=dict(optimizer_steps=task['updates'],backward=task['updates'],ema_updates=task['updates'],autograd_grad=int(task['updates']*1.6),sample_train_labeled=task['updates']*2,sample_train_unlabeled=0 if task['arm']=='DPR_L' else 80*r.c.COUNTS[task['domain']][1]) if phase=='train' else dict(sample_val=65)
                put(b/'tasks'/(tid+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json',dict(status='PASS',counts=counts))
            columns=('loss','CE','GT_Dice','anchor_label_records','response_images','response_VJP','donor_context_records','lambda_response','g_norm','d0_norm','correction_ratio','g_dot_d0','g_dot_dstar','g_dot_applied','Jd0_norm','Jdstar_norm','Japplied_norm','raw_non_descent','condition_number','zero_probe','zero_task_gradient','applied_progress_residual','applied_progress_budget')
            with (root/'steps.jsonl').open('w') as f:
                for pos in range(1,task['updates']+1):
                    row=dict.fromkeys(columns,0);row.update(position=pos,epoch=(pos-1)//task['steps_per_epoch']+1,labeled_source_scoring_multiplicity=1,donor_extra_image_reads=0,pseudo_labels=0,EMA_updates=1,checks={'fixture':True},U_image_records=0)
                    f.write(json.dumps(row)+'\n')
        for n,d in [('private_inputs.json',inputs),('SOURCE_BINDING.json',sb),('BASELINE_BINDING.json',bb),('SOURCE_AND_BASELINE_BINDING.json',{}),('TARGET_WEIGHT_SEAL.json',dict(source=source,students=students)),('qualification_ledger.json',{'synthetic':True}),('smoke/receipt.json',{'synthetic':True})]:put(b/n,d)
        with patch.object(ct,'verify',return_value=source),patch.object(ct,'read',side_effect=read),patch.object(ct,'check_hash'),patch.object(r,'csv_read',side_effect=lambda path:tables[str(path)]):
            terminal=r.finish(b)
            assert terminal['scientific_state']=='PRIMARY_FINAL_SIGNAL_AT_LEAST_0_005'
            assert abs(terminal['primary']['mean']['Final']-.01)<1e-12 and len(terminal['COMPONENT_EFFECT'])==2
            assert (b/'public_results/ACTUAL_STEP_RESPONSE.csv').is_file() and (b/'public_results/RESOURCE_ACCOUNTING.json').is_file()
        # Exercise the actual parent queue and weight-seal path with zero-training stub children.
        q=b/'queue';q.mkdir()
        for t in p['tasks']:
            root=q/'tasks'/t['task_id'];root.mkdir(parents=True)
            (root/('checkpoint_'+str(t['updates'])+'.pt')).write_bytes(b'SYNTHETIC_NOT_A_MODEL')
            put(root/'receipt.json',read(b/'tasks'/t['task_id']/'receipt.json'));put(root/('diagnostics_pass_'+str(t['updates'])+'.json'),dict(status='PASS'))
        for n in ('SOURCE_BINDING.json','BASELINE_BINDING.json','qualification_local/qualification.json','qualification_server/qualification.json','smoke/receipt.json'):put(q/n,dict(status='PASS',source=source,admission_mib=1))
        put(q/'qualification_ledger.json',dict(synthetic_updates=0,real_smoke_updates=12))
        dispatched=[]
        with patch.object(ct,'verify',return_value=source),patch.object(ct,'read',side_effect=read),patch.object(ct,'nas',side_effect=Path),patch.object(ct,'neutral_subprocess_paths'),patch.object(ct,'sha256',return_value='fixture'),patch.object(ex.subprocess,'check_output',return_value='9999'),patch.object(ex,'child',side_effect=lambda base,tid,phase,*a:dispatched.append((tid,phase))),patch.object(r,'finish',return_value={'synthetic':True}):
            ex.execute(q,'synthetic','synthetic')
        assert len(dispatched)==36 and len(set(dispatched))==36
        seal=json.loads((q/'TARGET_WEIGHT_SEAL.json').read_text());assert len(seal['students'])==18
    result=dict(status='PASS',scope='synthetic report and executor fixture only; no model or data',formal_updates=0,synthetic_optimizer_updates=0,report_branches=3,queue_dispatches=36)
    for value,state in [(-.001,'NO_PRIMARY_ACCURACY_GAIN'),(.001,'SMALL_POSITIVE_PRIMARY_SIGNAL'),(.006,'PRIMARY_FINAL_SIGNAL_AT_LEAST_0_005')]:
        assert r.primary_decision(r.np.full((2,3,4),value),dict(contains_zero=True))['scientific_state']==state
    return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
