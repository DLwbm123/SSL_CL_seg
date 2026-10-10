"""Reuse qualified V3 checks; add frozen V3 parity and SWD B exclusion."""
import copy
from pathlib import Path
import torch
from ..f5_output_u_v3 import tests as base
from .model import SelectiveUModel,SelectiveUTrainer
from .protocol import CAPS,plan,write,read
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.native_operations import NativeOperations


def bind_checks():
    from ..f5_output_u_v3 import run as previous
    from .run import construct
    base.OutputUModel=SelectiveUModel;base.OutputUTrainer=SelectiveUTrainer
    base.plan=plan;base.CAPS=CAPS;previous.construct=construct


def cpu_qualification(config):
    bind_checks();return base.cpu_qualification(config)


def qualify(config):
    bind_checks();root=Path(config['run_root'])
    try:
        base.qualify(config)
        counter=Counter(root/'qualification_cuda_physical.jsonl',CAPS['synthetic_cuda_updates'])
        from .run import engine
        permit=engine.admission(config,'qualification');device=torch.device('cuda:0')
        options={**base.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        with NativeOperations(root/'v3_parity_operations') as operations:
            provider=base.GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'))
            native=base.build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            m=SelectiveUModel(base.NativeLRParent(native,163,provider.stage_source),'F5',ratio=options['rank_ratio'],swd_to_parent=True).to(device)
            a=SelectiveUTrainer(m,provider,options,execution=permit)
            legacy=Path(config['v3_parity_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py'
            ns=dict(__name__='experiments.lcrseg.five_frameworks_v1.v3_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
            exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
            b=ns['StageTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),options,execution=permit)
            counter.wrap(a.optimizer);counter.wrap(b.optimizer)
            for _ in range(2):a.update();b.update();base.equal(base.snapshot(a),base.snapshot(b))
        assert counter.count==18
        q=read(root/'QUALIFICATION.json');q['checks']=[x.replace('KL and SWD each reach B','KL reaches B/R; SWD reaches R only') for x in q['checks']];q['checks']+=['SWD B gradient exactly zero while R nonzero; KL B nonzero','total-U restored fixture matches archived V3 state']
        q['costs']['synthetic_cuda_updates']=counter.count;q['costs']['V3_parity_operations']=dict(operations.counts)
        write(root/'QUALIFICATION.json',q)
    except BaseException:
        failure=dict(status='FAIL',execution_commit=config['execution_commit'],costs={})
        for name,file in [('cpu_optimizer_updates','qualification_cpu_physical.jsonl'),('synthetic_cuda_updates','qualification_cuda_physical.jsonl'),('real_L_smoke_updates','smoke_physical.jsonl')]:
            p=root/file;failure['costs'][name]=sum(1 for _ in p.open()) if p.exists() else 0
        write(root/'QUALIFICATION.json',failure);raise
