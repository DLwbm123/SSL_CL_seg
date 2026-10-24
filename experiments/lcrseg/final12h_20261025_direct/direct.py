"""Exact strong-source reuse, then the existing complete B/C stage runner."""
import copy
import json
import random
import shutil
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import campaign as c


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def prepare_source(dataset):
    root = Path(c.CFG['root']); out = root/f'{dataset}_0_SOURCE'; out.mkdir(exist_ok=False)
    original = Path(c.CFG['previous_root'])/f'{dataset}_0_SOURCE'
    baseline = original/'warm_reference'
    assert (baseline/'METRICS.json').exists() and (baseline/'PRIVATE_ROWS.json').exists()
    # The failed retrained source is never loaded: only the step-zero restored initial.
    initial = torch.load(original/'initial.pt', map_location='cpu', weights_only=False)
    assert initial['step'] == 0
    torch.manual_seed(20272401); np.random.seed(20272401); random.seed(20272401)
    model = c.warm(dataset)
    assert equal(model.state_dict(), initial['model']) and equal(model.state_dict(), initial['teacher'])
    method = c.Method(model, dataset, 'SOURCE')
    before_model = copy.deepcopy(method.model.state_dict()); before_teacher = copy.deepcopy(method.teacher.state_dict())
    l = c.Schedule(c.CFG['data'], c.CFG['sites'][dataset][0], seed=20272401, dataset=dataset, image_size=384, batch_size=2)
    u = c.Schedule(c.CFG['data'], c.CFG['sites'][dataset][0], True, seed=20272401, dataset=dataset, image_size=384, batch_size=4)
    rng = c.capture_rng_state(); imp, stats = c.summarize_current(method, l, u, {})
    assert equal(rng, c.capture_rng_state())
    assert equal(before_model, method.model.state_dict()) and equal(before_teacher, method.teacher.state_dict())
    assert all(p.grad is None for p in method.model.parameters()) and all(p.grad is None for p in method.teacher.parameters())
    reference = out/'warm_reference'; reference.mkdir()
    for name in ('METRICS.json', 'PRIVATE_ROWS.json'):
        shutil.copyfile(baseline/name, reference/name); shutil.copyfile(baseline/name, out/name)
    metrics = json.loads((out/'METRICS.json').read_text()); site = c.CFG['sites'][dataset][0]
    floor = 80.06740663770262 if dataset == 'mnms' else 89.29755965874404
    assert metrics[site]['Dice_percent'] >= floor
    n = metrics[site]['n']; difference = dict(mean_pp=0., CI95_pp=[0.,0.], over1=0, over2=0, p10_pp=0., n=n)
    c.write(out/'SOURCE_GATE.json', dict(status='PASS_IDENTITY_REUSE', parameter_and_function_exact=True,
            reused_original_evaluation=True, new_source_optimizer=0, source_training_sufficiency='NOT_APPLICABLE_NO_NEW_TRAINING',
            contrast=difference, reference_Dice_percent=metrics[site]['Dice_percent'], RNG_unchanged=True))
    c.save(out/'final.pt', dict(model=method.model.state_dict(), teacher=method.teacher.state_dict(), trainer={},
                              rng=c.capture_rng_state(), step=0, bank=[stats], importance=imp, counts=dict(c.COUNTS)))
    c.write(out/'EXIT.json', dict(exit_code=0, source_pass=True, step=0, counts=dict(c.COUNTS)))
    return out


def main(cfg):
    c.CFG = cfg
    def stop(*args): c.STOP = True
    signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    root=Path(cfg['root']); failed={}; current={}; source_ok={}
    c.audit_data()
    # Production objectives and shared step code are unchanged; preserve and reuse their prior synthetic receipt.
    receipt=json.loads((Path(cfg['previous_root'])/'SYNTHETIC.json').read_text()); assert receipt['status']=='PASS'
    c.write(root/'REUSED_SYNTHETIC.json', dict(receipt=receipt, new_synthetic_native=0, new_synthetic_reverse=0))
    for dataset in ('mnms','fundus'):
        try:
            p=prepare_source(dataset);source_ok[dataset]=True
            current[(dataset,'B')]=current[(dataset,'C')]=p
        except BaseException:
            source_ok[dataset]=False;failed[dataset+'_source_identity']=traceback.format_exc();c.write(root/'INCIDENTS.json',failed)
    for depth in (1,2,3):
        for dataset in ('mnms','fundus'):
            if not source_ok.get(dataset) or depth>=len(cfg['sites'][dataset]):continue
            for arm in ('B','C'):
                if dataset+'_'+arm in failed:continue
                try:
                    p,_=c.run_stage(dataset,arm,depth,current[(dataset,arm)]);current[(dataset,arm)]=p
                except BaseException:
                    failed[dataset+'_'+arm]=traceback.format_exc();c.write(root/'INCIDENTS.json',failed)
    terminal=dict(status='FINISHED_WITH_STOPS' if failed or not all(source_ok.values()) else 'COMPLETE',source_ok=source_ok,
                  failed=failed,counts=dict(c.COUNTS),previous_costs=cfg['previous_costs'],
                  window_optimizer=cfg['previous_costs']['optimizer']+c.COUNTS['optimizer'],
                  end_epoch=time.time(),elapsed_seconds=time.time()-cfg['started_epoch'],
                  round_elapsed_seconds=time.time()-cfg['round_started_epoch'],
                  allocated_bytes=c.allocated(root),window_allocated_bytes=sum(c.allocated(p) for p in cfg['window_roots']))
    c.write(root/'TERMINAL.json',terminal);c.write(root/'STATUS.json',terminal)
