"""Read-only terminal audit of saved artifacts; no inference or patient-data IO."""
import datetime
import json
import math
import os
from pathlib import Path
import torch


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def ledger(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for index,row in enumerate(rows,1):
        assert row['optimizer_executed'] is True and row['scientific_step']==index, 'nonsequential physical ledger'
    return len(rows)


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def spectrum(state):
    if 'sidecar.q' not in state:return None
    q=state['sidecar.q'].double();r=state['sidecar.r'].double()
    g=torch.eye(len(q))+q@r@q.T;f=state['sidecar.previous'].double()@g
    result={}
    for name,m in [('G',g),('F',f)]:
        s=torch.linalg.svdvals(m);result[name]=dict(min=float(s.min()),max=float(s.max()),condition=float(s.max()/s.min()))
    return result


def phase_interval(first,last,total):
    warmup=math.ceil(.2*total);ramp=warmup+math.ceil(.2*total)
    return 'warmup' if last<=warmup else 'full_U' if first>=ramp else 'U_ramp' if first>=warmup and last<=ramp else 'mixed_phase'


def audit(root):
    plan=read(root/'PLAN.json');stop=read(root/'TERMINATION_RECEIPT.json')
    assert stop['status']=='STOPPED_BY_USER' and all(p['terminated'] for p in stop['processes'])
    assert len(plan['nodes'])==12 and plan['new_formal_optimizer_updates']==150200
    nodes=[];metrics={};issues=[];endpoints={};inventory=[]
    for n in plan['nodes']:
        key=n['id'];d=root/'jobs'/key;status=read(d/'STATUS.json');cost=read(d/'COSTS.json');exit_=read(d/'EXIT.json')
        calls=ledger(d/'physical_updates.jsonl')
        if n['stage']==0: terminal='REUSED_SOURCE'
        elif status and status['status']=='COMPLETE':terminal='COMPLETE'
        elif status and status.get('status')=='FAILED' and 'KeyboardInterrupt' in status.get('traceback',''):terminal='USER_TERMINATED'
        elif not status or status.get('status')=='CANCELLED_BY_USER':terminal='CANCELLED_BY_USER'
        else:terminal='FAILED';issues.append(dict(node=key,issue='unexpected terminal state'))
        if cost:assert cost['physical_optimizer_updates']==calls
        if terminal=='CANCELLED_BY_USER':assert calls==0
        committed=0 if n['stage']==0 or not status else int(status.get('step',0))
        if cost and n['stage']>0:assert cost['committed_scientific_updates']==committed
        checkpoints=[];latest_step=0;checkpoint_guard=None;checkpoint_telemetry=None
        for name in ('latest.pt','final.pt','sealed.pt'):
            p=d/name
            if not p.exists():continue
            receipt=read(d/(name+'.receipt.json'));payload=torch.load(p,map_location='cpu',weights_only=False)
            step=payload.get('step',payload.get('updates'))
            assert step is not None and receipt and receipt['committed'] and receipt['bytes']==p.stat().st_size
            if receipt.get('step') is not None:assert receipt['step']==step
            assert finite_tree({k:payload[k] for k in ('student','ema','optimizer','native','transform') if k in payload})
            cp=dict(file=name,bytes=p.stat().st_size,step=step,committed_receipt=True,size_matches=True,tensor_payload_readable_and_finite=True)
            if 'identity' in payload:assert payload['identity']['id']==key
            if 'student' in payload:
                cp['student_spectrum']=spectrum(payload['student']);cp['ema_spectrum']=spectrum(payload['ema'])
                extra=payload.get('v2',{});cp['guard_totals']=extra.get('guard_totals')
                cp['history_support']=[dict(domain=h['domain'],support=h['support'].tolist(),observations=h['observations']) for h in extra.get('memory',{}).get('history',[])]
                cp['provider_reads']=payload.get('provider_reads')
                if step>=latest_step:
                    latest_step=step;checkpoint_guard=cp['guard_totals'];checkpoint_telemetry=payload.get('telemetry',{})
                if cp['guard_totals']:endpoints[(key,step)]=dict(node=key,step=step,guard_totals=cp['guard_totals'],source='checkpoint',durable=True)
            elif n['stage']==0:
                assert payload.get('new_optimizer_updates')==0 and payload['updates']==n['steps']
                cp['inherited_updates']=n['steps'];cp['new_optimizer_updates']=0
            checkpoints.append(cp);inventory.append(dict(node=key,**cp));del payload
        if terminal=='COMPLETE':assert latest_step==n['steps'] and any(c['file']=='sealed.pt' for c in checkpoints)
        if terminal=='USER_TERMINATED':assert latest_step>0 and not any(c['file']=='final.pt' for c in checkpoints)
        if cost and 'v2_guard_calls' in cost.get('telemetry',{}):
            t=cost['telemetry'];g={k:t['v2_guard_'+k] for k in ('calls','damped','cone_zero','zero_displacement')}
            assert g['calls']==committed
            endpoints.setdefault((key,committed),dict(node=key,step=committed,guard_totals=g,source='terminal_cost_receipt',durable=False))
        validation=read(d/'VALIDATION.json')
        if validation is not None:metrics[key]=dict(seed=n['seed'],stage=n['stage'],domain=n['domain'],inherited_source_validation=n['stage']==0,validation=validation)
        raw=dict(status=None if not status else status.get('status'),exit_code=None if not exit_ else exit_.get('exit_code'),interruption='KeyboardInterrupt' if terminal=='USER_TERMINATED' else None)
        nodes.append(dict(**n,terminal_status=terminal,raw_worker_receipt=raw,physical_optimizer_calls=calls,inherited_source_updates=n['steps'] if n['stage']==0 else 0,successfully_committed_updates=committed,latest_recoverable_checkpoint_step=latest_step if n['stage'] else n['steps'],executed_after_last_checkpoint=0 if n['stage']==0 else calls-latest_step,successful_but_not_checkpointed=0 if n['stage']==0 else committed-latest_step,executed_but_uncommitted=0 if n['stage']==0 else calls-committed,checkpoints=checkpoints,cost_receipt=cost,validation_available=validation is not None))
    # Monitoring history stores anonymous checkpoint totals; duplicate observations are one endpoint.
    for p in sorted((root/'monitoring').glob('2*.json')):
        x=read(p)
        for key,t in x.get('tasks',{}).items():
            cp=t.get('checkpoint_review') or {};g=cp.get('guard_totals')
            if g:endpoints.setdefault((key,cp['step']),dict(node=key,step=cp['step'],guard_totals=g,source='previously_committed_checkpoint_observation',durable=True))
    intervals=[]
    for n in plan['nodes']:
        if not n['stage']:continue
        first=0;previous=dict(calls=0,damped=0,cone_zero=0,zero_displacement=0)
        for e in sorted([v for (key,_),v in endpoints.items() if key==n['id']],key=lambda e:e['step']):
            g=e['guard_totals'];delta={k:g[k]-previous[k] for k in previous};assert all(v>=0 for v in delta.values()) and delta['calls']==e['step']-first
            if delta['calls']:
                intervals.append(dict(node=n['id'],seed=n['seed'],domain=n['domain'],from_step=first,to_step=e['step'],phase=phase_interval(first,e['step'],n['steps']),delta_guard=delta,zero_ratio=delta['zero_displacement']/delta['calls'],damped_ratio=delta['damped']/delta['calls'],endpoint_source=e['source'],endpoint_model_durable=e['durable']))
            first=e['step'];previous=g
    engineering=[]
    for p in sorted((root/'engineering').glob('attempt_*/RESULT.json')):
        x=read(p);count=ledger(p.parent/'physical_updates.jsonl');assert count==x['optimizer_calls']
        engineering.append(dict(attempt=p.parent.name,status=x['status'],optimizer_calls=count,formal_updates=x['formal_updates'],test_access=x['test_access'],checks=x.get('checks',{}),source_memory_reused=x.get('source_memory_reused_from_attempt02',False),error_category='missing_CUBLAS_workspace' if 'CUBLAS_WORKSPACE_CONFIG' in x.get('traceback','') else 'qualification_helper_Adapter_interface' if x['status']=='FAIL' else None,two_case_L_fit=x.get('two_case_L_fit'),elapsed_seconds=x.get('elapsed_seconds')))
    assert sum(e['optimizer_calls'] for e in engineering)==104
    targets=[n for n in nodes if n['stage']];physical=sum(n['physical_optimizer_calls'] for n in targets);committed=sum(n['successfully_committed_updates'] for n in targets);covered=sum(n['latest_recoverable_checkpoint_step'] for n in targets)
    costs=dict(formal_budget=150200,new_formal_optimizer_calls=physical,unused_formal_budget=150200-physical,successfully_committed_updates=committed,durable_checkpoint_covered_updates=covered,physical_calls_after_latest_durable_checkpoints=physical-covered,successful_updates_not_checkpointed=committed-covered,executed_but_uncommitted_calls=physical-committed,reused_source_updates=27200,new_source_optimizer_calls=0,engineering_optimizer_calls=104,observed_new_formal_plus_GPU_engineering_calls=physical+104,source_bootstrap_successful_L_image_exposures=132,completed_target_L_batch_reads=sum(n['cost_receipt'].get('L_batch_reads',0) for n in targets if n['terminal_status']=='COMPLETE'),completed_target_U_batch_reads=sum(n['cost_receipt'].get('U_batch_reads',0) for n in targets if n['terminal_status']=='COMPLETE'),worker_elapsed_seconds_sum=sum(n['cost_receipt'].get('elapsed_seconds',0) for n in targets if n['cost_receipt']),controller_receipt_creation_utc=datetime.datetime.fromtimestamp((root/'CONTROLLER_PID.json').stat().st_mtime,datetime.timezone.utc).isoformat(),termination_utc=stop['utc'],missing_costs=['complete active GPU-hours and energy','all failed source-bootstrap and other engineering image exposures','interrupted workers terminal provider L/U read counts; saved checkpoint counts retained','optimizer invocation alone does not imply accepted nonzero learning'])
    summaries=[]
    for seed in (163,164):
        chain=[n for n in nodes if n['seed']==seed];available=[metrics[n['id']] for n in chain if n['id'] in metrics];bmc=next(m for m in available if m['domain']=='BMC');v=bmc['validation']
        summaries.append(dict(seed=seed,status='USER_TERMINATED_INCOMPLETE_CHAIN',available_stage_results=available,after_BMC_seen_domain_mean=sum(z['primary_dice'] for z in v.values())/len(v),full_chain_Final=None,full_chain_Old=None,full_chain_Incoming=None,full_chain_Forgetting=None,missing_stages=[n['domain'] for n in chain if not n['validation_available']]))
    audit_=dict(status='AUDITED_EARLY_TERMINATION_WITH_DECLARED_MISSING_RESULTS_AND_COSTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),study='Prostate MAIN_STYLE_SSL_V2',termination_reason='explicit user stop after sustained guard/spectral-boundary learning suppression',terminal_counts={s:sum(n['terminal_status']==s for n in nodes) for s in ('REUSED_SOURCE','COMPLETE','USER_TERMINATED','CANCELLED_BY_USER','FAILED')},all_target_ledgers_sequential=True,checkpoint_checks='committed receipt, file presence/size, readable finite payload and step; no bytewise identity claim',process_termination_verified=all(p['terminated'] for p in stop['processes']),new_training_or_inference_or_optimizer_calls=0,independent_test_access=False,ablation=False,retry=False,issues=issues,public_delivery='PENDING')
    return {'AUDIT.json':audit_,'PLAN.json':plan,'NODES.json':nodes,'CHECKPOINT_INVENTORY.json':inventory,'COSTS.json':costs,'AVAILABLE_RESULTS.json':dict(prostate=summaries),'GUARD_INTERVALS.json':intervals,'ENGINEERING_ATTEMPTS.json':engineering}


if __name__=='__main__':
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    root=Path(os.environ['RUN_ROOT']);out=Path(os.environ['AUDIT_OUTPUT']);out.mkdir(exist_ok=False)
    result=audit(root)
    for name,value in result.items():(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['AUDIT.json']['status'],counts=result['AUDIT.json']['terminal_counts'],costs=result['COSTS.json'],available_results=[{k:v for k,v in row.items() if k!='available_stage_results'} for row in result['AVAILABLE_RESULTS.json']['prostate']]),ensure_ascii=False),flush=True)
