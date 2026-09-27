"""Aggregate evidence only; private state, identifiers and raw transactions stay private."""
import json,csv,io,math
from pathlib import Path
from collections import Counter,defaultdict
from .runtime import TASKS,PLAN
from r1_12h.core import atomic,events


def table(folder,name,rows):
    stream=io.StringIO();fields=sorted({k for r in rows for k in r})
    if fields:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (folder/name).write_text(stream.getvalue())


def aggregate(run,queue,budget,final=False):
    run=Path(run);root=run/'reports';root.mkdir(exist_ok=True);results=[];ledger=[]
    for task,spec in TASKS.items():
        folder=run/'tasks'/task;row=dict(task=task,status=queue[task]['status'],seed=spec['seed'],backbone=spec['backbone'],domain=spec['domain'],arm=spec.get('arm','R3A'))
        if (folder/'PROGRESS.json').exists():row.update(json.loads((folder/'PROGRESS.json').read_text()))
        if (folder/'EVALUATION.json').exists():
            value=json.loads((folder/'EVALUATION.json').read_text())
            if value.get('complete'):results.append(value)
        ledger.append(row)
    table(root,'RESULTS.csv',results);(root/'TASK_LEDGER.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ledger))
    by={(x['seed'],x['backbone'],x['domain'],x['arm']):x for x in results};deltas=[]
    for seed in (261,262,263):
        for backbone in ('UNET_QUERY_128','DINOV2_VITS14_QUERY'):
            for domain in ('RIM_ONE_r3','Drishti_GS'):
                a=by.get((seed,backbone,domain,'RL'))
                for arm in ('FIX_FINE','SUP','FIX_COARSE','RANDOM','RULE','REG','NC_RL','FIX_FINE_EXTRA_L'):
                    b=by.get((seed,backbone,domain,arm))
                    for metric in ('macro','rim','cup'):
                        delta=a[metric]-b[metric] if a and b and a[metric] is not None and b[metric] is not None else None
                        deltas.append(dict(seed=seed,backbone=backbone,domain=domain,contrast='RL-'+arm,metric=metric,delta=delta,delta_pp=None if delta is None else delta*100))
    table(root,'PAIRED_DELTAS.csv',deltas);checks={};decision='INCOMPLETE'
    if len(results)==108:
        fresh=[x for x in deltas if x['seed'] in (262,263) and x['contrast']=='RL-FIX_FINE'];macro=[x for x in fresh if x['metric']=='macro'];mean=lambda xs:sum(x['delta'] for x in xs)/len(xs)
        checks=dict(mean_macro=mean(macro),positive_cells=sum(x['delta']>0 for x in macro),worst_macro=min(x['delta'] for x in macro),backbone_macro={b:mean([x for x in macro if x['backbone']==b]) for b in ('UNET_QUERY_128','DINOV2_VITS14_QUERY')},worst_class=min(x['delta'] for x in fresh if x['metric']!='macro'))
        passed=checks['mean_macro']>=.003 and checks['positive_cells']>=6 and checks['worst_macro']>=-.005 and checks['worst_class']>=-.01 and all(x>0 for x in checks['backbone_macro'].values());decision='PRIMARY_INVESTMENT_TARGET_MET' if passed else 'PRIMARY_GAIN_NOT_ESTABLISHED'
    atomic(root/'PRIMARY_DECISION.json',dict(decision=decision,checks=checks,endpoints=len(results),criteria=PLAN['primary']))
    if not final:return
    audits=[];curves=[];counts=[];actions=[]
    for task,spec in TASKS.items():
        folder=run/'tasks'/task;ev=events(folder/'TRANSACTIONS.jsonl');cost=events(folder/'COST_EVENTS.jsonl');purposes=Counter()
        for x in cost:purposes[x['purpose']]+=x['images']
        attempts=[x for x in ev if x['event']=='attempt'];commits=[x for x in ev if x['event']=='student_commit'];valid_commits=sum(not x['temporary'] for x in commits);successful=sum(x['event']=='optimizer_success' for x in ev)
        counts.append(dict(task=task,optimizer_attempts=len(attempts),optimizer_successes=successful,optimizer_failures=sum(x['event']=='optimizer_failed' for x in ev),physical_retained_commits=valid_commits,physical_disposable_commits=sum(x['temporary'] for x in commits),replay_attempts=sum(x['category'] in ('student_replay','controller_replay') for x in attempts),python_exceptions=len(list(folder.glob('FAILURE_*.private.json'))),student_forwards=sum(x['student_forwards'] for x in cost),teacher_forwards=sum(x['teacher_forwards'] for x in cost),**{'label_images_'+k:v for k,v in purposes.items()}))
        if (folder/'R3A.private.json').exists():
            rows=json.loads((folder/'R3A.private.json').read_text());reward=[r['online_rewards'][a] for r in rows for a in (1,2)];null0=[r['online_rewards'][3] for r in rows];null2=[r['online_rewards'][4]-r['online_rewards'][2] for r in rows];top=lambda v:max(range(3),key=lambda i:v[i]);agreement=sum(top(r['online_rewards'])==top(r['audit_rewards']) for r in rows)
            audits.append(dict(task=task,scale=json.loads((folder/'R3A_DONE.json').read_text())['scale'],contexts=len(rows),mean_reward=sum(reward)/len(reward),reward_rms=math.sqrt(sum(x*x for x in reward)/len(reward)),null_skip_max=max(map(abs,null0)),null_fine_max=max(map(abs,null2)),online_audit_top_agreement=agreement/len(rows),best_action_counts=dict(Counter(top(r['online_rewards']) for r in rows)),mean_action_spread=sum(max(r['online_rewards'][:3])-min(r['online_rewards'][:3]) for r in rows)/len(rows)))
        for p in sorted((folder/'decisions').glob('*.private.json')):
            row=json.loads(p.read_text());f=row.get('feedback')
            if f:
                rewards=f['raw_rewards'];curves.append(dict(task=task,t=row['meta']['t'],mean_raw_reward=sum(rewards)/len(rewards) if rewards else None,scale=f['scale'],controller_updates=len(f['controller']),mean_clip=sum(x['clip_fraction'] for x in f['controller'])/len(f['controller']) if f['controller'] else None,mean_kl=sum(x['kl'] for x in f['controller'])/len(f['controller']) if f['controller'] else None,entropy=-sum(p*math.log(p) for p in f['behavior'])))
        latest={x['t']:x for x in events(folder/'ACTION_LOG.jsonl')}
        if latest:
            dist=Counter(x['action'] for x in latest.values());actions.append(dict(task=task,ordinary_steps=len(latest),skip=dist[0],coarse=dist[1],fine=dist[2]))
    state_audits=[json.loads(p.read_text()) for p in (run/'tasks').glob('*/STATE_AUDIT.json')]
    atomic(root/'STATE_AUDIT.json',state_audits)
    grants=events(run/'BUDGET_EVENTS.jsonl');atomic(root/'PHYSICAL_BUDGET_AUDIT.json',dict(categories=dict(Counter(x['category'] for x in grants)),grants=len(grants),ledger_matches=dict(Counter(x['category'] for x in grants))=={k:v for k,v in budget.items() if v},endpoint_states=len(state_audits)))
    atomic(root/'R3A_REWARD_AUDIT.json',audits);table(root,'R3A_REWARD_AUDIT.csv',audits);table(root,'POLICY_LEARNING_CURVES.csv',curves);table(root,'ACTION_DISTRIBUTIONS.csv',actions);table(root,'COST_AND_FEEDBACK_COUNTS.csv',counts)
    atomic(root/'TRANSACTION_AUDIT.json',dict(budget=budget,tasks=counts,total_optimizer_attempts=sum(x['optimizer_attempts'] for x in counts),note='Qualification has separate task ledgers. Retained and disposable physical commits include replay; final unique path counts are in task ledger.'))
    (root/'FINAL_INTERPRETATION.md').write_text('# RL-Control Bandit V2\n\n'+f'{decision}. Endpoints {len(results)}/108.\n\n'+json.dumps(checks,indent=2)+'\n\nThe primary contrast remains fresh-seed RL−FIX_FINE. All registered secondary contrasts are reported without switching candidates. This is a single-domain contextual-bandit pilot using previously exposed training/validation cohorts. Reward improvements, policy movement, or a met development threshold do not establish statistical significance, independent-patient confirmation, or continual-learning benefit. Two fresh optimization seeds are not eight independent patient trials. Original KI remains unbound; R3c is not authorized.\n\nFixed COARSE tests harmful fine supervision; REG tests non-policy-gradient adaptation; NC_RL tests context value; EXTRA_L matches student optimizer calls but not GPU time or feedback forward cost. Interpretation must compare each of these fixed controls. See complete costs, audits, action distributions and paired deltas.\n')
