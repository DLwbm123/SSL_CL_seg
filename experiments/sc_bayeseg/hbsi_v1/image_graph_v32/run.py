"""Two graph-coupled target arms with sealed A/N controls."""
import json,os,time
from pathlib import Path
import torch
from ..stream_state import StreamState,digest
from ..paired_semantic_v19.run import data,evaluate
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..posterior_teacher_v25.teacher import CachedData
from ..posterior_teacher_v25.method import Objective as SoftObjective
from .targets import build_cache,fixture
Objective=SoftObjective


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='IMAGE_GRAPH_V32' and p['U_coefficient']==1 and p['teacher']=='v23_L1600'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=12000,max_updates=800,max_bytes=3_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    checkpoint=Path(p['v23_root'])/'results/L_attempt001/ACTIVE_STEP_001600.pt'
    def load():
        s=StreamState.load(checkpoint);assert s.steps==1600 and s.active['id']=='RUNMC'
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s
    try:
        previous=Path(p['v25_root'])
        old=json.loads((previous/'PRIVATE_PROTOCOL.json').read_text())
        assert json.loads((previous/'STATUS.json').read_text())['stage']=='COMPLETE'
        assert json.loads((previous/'EXIT.json').read_text())['exit_code']==0
        results=json.loads((previous/'RESULTS.json').read_text())
        assert results['new_formal_updates']==1200 and all(c['complete'] for c in results['cells'].values())
        oldledger=json.loads((previous/'results/RESOURCE_LEDGER.json').read_text())
        assert len(oldledger['training_sessions'])==3 and all(v['status']=='COMPLETE' and v['completed_steps']==2000 for v in oldledger['training_sessions'].values())
        assert all(p[k]==old[k] for k in ('config','data_root','data_version','v19_root','v23_root','U_coefficient','teacher'))
        dependencies=('training.py','stream_state.py','data_adapter.py','paired_semantic_v19/run.py','posterior_teacher_v25/method.py','semantic_alignment_v23/method.py','semantic_evidence_v22/method.py')
        for dependency in dependencies:
            assert (root/'source/sc_bayeseg/hbsi_v1'/dependency).read_bytes()==(previous/'source/sc_bayeseg/hbsi_v1'/dependency).read_bytes()
        cache=torch.load(previous/'PRIVATE_TEACHER.pt',map_location='cpu',weights_only=False)
        reference=json.loads((previous/'BRANCH_A.json').read_text())
        assert all(json.loads((previous/f'BRANCH_{a}.json').read_text())['common_payload']==reference['common_payload'] for a in 'ABC')
        assert cache['teacher_payload']==reference['common_payload']
        write(root/'MATCHING.json',dict(status='PASS',reused_controls=['v25_A2000','v30_N2000'],same_config_and_data=True,critical_training_dependencies_match_v25=True,common_payload=reference['common_payload'],prior_hash=reference['prior_hash'],old_teacher_cache_reused_for_matching=True,old_teacher_new_forwards=0,expected_new_target_backbone_forwards=0,old_endpoints_rescored=False))
        neighbor=Path(p['v30_root'])
        assert json.loads((neighbor/'STATUS.json').read_text())['stage']=='COMPLETE' and json.loads((neighbor/'EXIT.json').read_text())['exit_code']==0
        basecache=torch.load(neighbor/'PRIVATE_NEIGHBORS.pt',map_location='cpu',weights_only=False)
        assert basecache['teacher_payload']==reference['common_payload']
        status('TARGET_CACHE')
        with ledger.attempt('target_cache',seconds=1200,updates=0) as out:
            s=load();d=data(p,s.config,'RUNMC','LU');assert s.active['evidence']==d.evidence
            fixtures=fixture()
            assert digest(s.payload())==basecache['teacher_payload'] and basecache['evidence_fingerprint']==d.evidence['fingerprint']
            newcaches,summary=build_cache(d,basecache,root)
            assert digest(s.payload())==basecache['teacher_payload']
            write(root/'TARGET_SUMMARY.json',summary);del s,d
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=600,updates=0) as out:
            s=load();d=CachedData(data(p,s.config,'RUNMC','LU'),cache)
            assert digest(s.payload())==cache['teacher_payload'];before=digest(s.payload());rng=s.generator.get_state()
            L=d.batch('L',1,torch.Generator().manual_seed(166200));U=d.batch('U',1,torch.Generator().manual_seed(166201))
            for b in (L,U):
                b['x']=b['x'].cuda()
                if 'observations' in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            s.generator.set_state(rng)
            _,refparts=SoftObjective(s,Refiner(s.model,s.phi,counter),'B').estimate(L,U,d.counts,s.generator)
            refend=s.generator.get_state()
            frozen=json.loads((previous/'PREFLIGHT.json').read_text())['real_batch'][1]['parts']
            assert all(abs(float(refparts[k])-frozen[k])<1e-10 for k in ('task','anchor','U_risk'))
            parts=[]
            nd=CachedData(d.base,basecache);NU=nd.batch('U',1,torch.Generator().manual_seed(166201));NU['x']=NU['x'].cuda()
            s.generator.set_state(rng)
            _,np=SoftObjective(s,Refiner(s.model,s.phi,counter),'N').estimate(L,NU,d.counts,s.generator)
            nref=json.loads((neighbor/'PREFLIGHT.json').read_text())['real_batch'][0]['parts']
            assert all(abs(float(np[k])-nref[k])<1e-10 for k in ('task','anchor','U_risk'))
            for arm in 'IE':
                nd=CachedData(d.base,newcaches[arm]);NU=nd.batch('U',1,torch.Generator().manual_seed(166201));NU['x']=NU['x'].cuda()
                assert torch.equal(NU['x'],U['x'])
                s.generator.set_state(rng);loss,record=Objective(s,Refiner(s.model,s.phi,counter),arm).estimate(L,NU,d.counts,s.generator);loss.backward()
                assert all(q.grad is None or torch.isfinite(q.grad).all() for group in s.optimizer.param_groups for q in group['params'])
                assert s.q.mean.grad is not None and s.phi.c_head.weight.grad is not None
                assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
                assert all(torch.equal(record[k],refparts[k]) for k in ('task','anchor'))
                assert torch.equal(refend,s.generator.get_state())
                parts.append(dict(arm=arm,parts={k:float(v) for k,v in record.items()}));s.optimizer.zero_grad(set_to_none=True)
            s.generator.set_state(rng);assert digest(s.payload())==before==reference['common_payload']
            assert digest(s.prior.state_dict())==reference['prior_hash']
            write(root/'PREFLIGHT.json',dict(status='PASS',real_batch=parts,source_unchanged=True,paired_L_task_anchor_rng=True,v25_fixed_batch_reproduced=True,graph_fixture=fixtures,v30_fixed_batch_reproduced=True,reverse_calls=2,optimizer_updates=0));del s,d,L,U
        common=None;cells={}
        for arm in 'IE':
            newcache=newcaches[arm]
            s=load();d=CachedData(data(p,s.config,'RUNMC','LU'),newcache);initial=digest(s.payload())
            if common is None:common=initial
            else:assert initial==common
            assert initial==cache['teacher_payload']==newcache['teacher_payload']
            unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            write(root/f'BRANCH_{arm}.json',dict(common_payload=initial,prior_hash=digest(s.prior.state_dict()),initial_steps=1600,Adam_inherited=True,teacher_payload=cache['teacher_payload']))
            status('TRAIN',arm=arm);holder={}
            def factory(st,ref):
                holder['obj']=Objective(st,ref,arm,root/f'PRIVATE_AUDIT_{arm}.jsonl');return holder['obj']
            run_stage(s,d,ledger,counter,arm,2000,3600,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=factory,after_checkpoint=lambda step:holder['obj'].after_update(step))
            assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_'+domain,seconds=900,updates=0) as out:cells[arm+'_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in cells.items() if k.startswith(arm+'_')});del s,d
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=800,preflight_reverse_calls=2,training_audit_reverse_calls=12,source_updates=0,teacher_updates=0,common_entrance=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
