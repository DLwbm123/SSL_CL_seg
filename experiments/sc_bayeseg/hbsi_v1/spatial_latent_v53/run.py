"""One R-resolution change in the W structured posterior, source to target."""
import json,os,time
from dataclasses import replace
from pathlib import Path
import torch
from ..config import Config
from ..stream_state import digest
from ..multiscale_recognition_v46.model import MultiscaleState
from ..paired_semantic_v19.run import data,evaluate
from ..semantic_alignment_v23.method import Objective
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..pixel_ce_v49.diagnostics import prepare,train_evaluate


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='SPATIAL_LATENT_V53'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base_cfg=replace(Config(**p['config']),seed=20261023);cfg=replace(base_cfg,r_height=64,r_width=64)
    assert (base_cfg.height,base_cfg.width,base_cfg.r_height,base_cfg.r_width,base_cfg.hidden)==(128,128,32,32,16)
    ledger=ResourceLedger(root/'results',max_seconds=12900,max_updates=2400,max_bytes=3_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));f.replace(root/'STATUS.json')
    def freeze(s):s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
    def objective(s,ref):return Objective(s,ref,False)
    def initial(configuration):
        s=MultiscaleState(configuration);d=data(p,configuration,'RUNMC','L');assert d.counts==dict(L=80,U=0)
        s.begin_task('RUNMC',d.manifest_hash,d.evidence);s.to_device('cuda');freeze(s);return s,d
    def backward(s,d,seed):
        b=d.batch('L',1,torch.Generator().manual_seed(seed));b['x']=b['x'].cuda();b['observations']={k:v.cuda() for k,v in b['observations'].items()}
        before=digest(s.payload());rng=s.generator.get_state()
        loss,_=objective(s,Refiner(s.model,s.phi,counter)).estimate(b,None,d.counts,s.generator);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for g in s.optimizer.param_groups for v in g['params'])
        assert all(any(v.grad is not None and float(v.grad.norm())>0 for v in g['params']) for g in s.optimizer.param_groups)
        assert any(v.grad is not None and float(v.grad.norm())>0 for v in s.phi.backbone.parameters())
        assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
        assert all(v.grad is None for m in s.memories.values() for v in m.parameters())
        s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==before
        return float(loss)
    try:
        source=Path(p['v46_root']);assert json.loads((source/'STATUS.json').read_text())['stage']=='COMPLETE' and json.loads((source/'EXIT.json').read_text())['exit_code']==0
        oldp=json.loads((source/'PRIVATE_PROTOCOL.json').read_text());assert all(p[k]==oldp[k] for k in ('config','data_root','data_version','v19_root'))
        for dep in ('training.py','stream_state.py','data_adapter.py','paired_semantic_v19/run.py','paired_semantic_v19/method.py','semantic_alignment_v23/method.py','semantic_alignment_v23/candidates.py','semantic_evidence_v22/method.py','unified_sslcl_v18/method.py','multiscale_recognition_v46/model.py'):
            assert (root/'source/sc_bayeseg/hbsi_v1'/dep).read_bytes()==(source/'source/sc_bayeseg/hbsi_v1'/dep).read_bytes(),dep
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0) as out:
            baseline,_=initial(base_cfg);s,d=initial(cfg)
            matching=dict(q=digest(baseline.q.state_dict())==digest(s.q.state_dict()),prior=digest(baseline.prior.state_dict())==digest(s.prior.state_dict()),current=digest(baseline.current.state_dict())==digest(s.current.state_dict()),rng=digest(baseline.generator.get_state())==digest(s.generator.get_state()),phi_weights=digest(baseline.phi.state_dict())==digest(s.phi.state_dict()),evidence=d.evidence==baseline.active['evidence'])
            assert all(matching.values())
            source_loss=backward(s,d,163000);cp=out/'PRIVATE_ENTRY.pt';s.save(cp)
            restored=MultiscaleState.load(cp).to_device('cuda');freeze(restored);assert digest(s.payload())==digest(restored.payload())
            s.finalize_task();target_data=data(p,cfg,'BMC','L');assert target_data.counts==dict(L=134,U=0)
            s.begin_task('BMC',target_data.manifest_hash,target_data.evidence);freeze(s);s.to_device('cuda')
            target_loss=backward(s,target_data,263000)
            write(root/'PREFLIGHT.json',dict(status='PASS',initial_weights_q_prior_domain_rng_matching=matching,only_resolution_changed=True,r32=[32,32],r64=[64,64],source_target_model_reverse_calls=2,Adam_updates=0,source_loss=source_loss,target_loss=target_loss,checkpoint_roundtrip=True,target_fixture_untrained_source=True))
            del baseline,restored,s,d,target_data
        s,d=initial(cfg);unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
        write(root/'MATCHING.json',dict(status='PASS',source_initial=digest(s.payload()),prior=digest(s.prior.state_dict()),initial_weights_q_prior_domain_rng_matching=True,dependencies_equal=True,same_split_data_stream_budget=True,old_controls_rescored=False))
        status('TRAIN_SOURCE')
        run_stage(s,d,ledger,counter,'R64_source',1600,6000,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=objective)
        assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
        cells={}
        for domain in ('RUNMC','BMC'):
            status('EVALUATE_SOURCE',domain=domain)
            with ledger.attempt('eval_source_'+domain,seconds=450,updates=0) as out:cells['source_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
        path=root/'results/R64_source_attempt001/PRIVATE_SEALED_STATE.pt';source_digest=digest(s.payload())
        s=MultiscaleState.load(path);s.steps=s.receipts['RUNMC']['steps'];assert digest(s.payload())==source_digest
        old=digest(s.memories['RUNMC'].state_dict());d=data(p,cfg,'BMC','L');s.begin_task('BMC',d.manifest_hash,d.evidence);freeze(s);s.to_device('cuda')
        assert not s.optimizer.state and d.counts==dict(L=134,U=0)
        definitions=prepare(d,root)
        assert json.loads((root/'TRAIN_DEFINITION.json').read_text())==json.loads((Path(p['v49_root'])/'TRAIN_DEFINITION.json').read_text())
        status('TRAIN_TARGET')
        run_stage(s,d,ledger,counter,'R64_target',800,3600,fingerprint(p),data_seed=263000,checkpoint_stride=400,objective_factory=objective)
        assert digest(s.memories['RUNMC'].state_dict())==old and all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
        status('TRAIN_DIAGNOSTIC')
        with ledger.attempt('train_diagnostic',seconds=1200,updates=0) as out:train=train_evaluate(s,d,definitions,out/'train')
        for domain in ('RUNMC','BMC'):
            status('EVALUATE_TARGET',domain=domain)
            with ledger.attempt('eval_target_'+domain,seconds=450,updates=0) as out:cells['target_'+domain]=evaluate(p,s,domain,domain,out/'validation')
        write(root/'RESULTS.json',dict(cells=cells,train_diagnostics=dict(R64=train),formal_updates=2400,preflight_reverse_calls=2,train_diagnostic_semantic_sample_forwards=12864,old_memory_and_Sb_unchanged=True,source_target_pair=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
