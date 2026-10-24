"""Only zero-update completion of the exact final-count evaluation-boundary incident."""
import copy
import json
from pathlib import Path
import time
import torch
import campaign as c


def main(cfg):
    root = Path(cfg['root']); out = root/'mnms_3_C'
    assert not (root/'FINALIZATION_RECOVERY.json').exists()
    term = json.loads((root/'TERMINAL.json').read_text())
    fail = json.loads((out/'FAILURE.json').read_text())
    assert term['counts']['optimizer'] == 80096 and fail['step'] == 6000
    assert 'Frozen physical call cap' in fail['traceback']
    assert not (out/'METRICS.json').exists() and not (out/'final.pt').exists()
    saved = torch.load(out/'failure_partial.pt', map_location='cpu', weights_only=False)
    assert saved['step'] == 6000 and saved['counts']['optimizer'] == 80096
    assert all(torch.isfinite(v).all() for s in ('model', 'teacher') for v in saved[s].values())
    assert time.time() < cfg['deadline_epoch']-180
    c.CFG = cfg; c.COUNTS.update(saved['counts'])
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = c.Network('mnms').cuda(); model.load_state_dict(saved['model'])
    method = c.Method(model, 'mnms', 'C'); method.teacher.load_state_dict(saved['teacher'])
    c.restore_rng_state(saved['rng'])
    metrics, _ = c.evaluate(method.teacher, 'mnms', cfg['sites']['mnms'], out)
    l = c.Schedule(cfg['data'], 'Canon', seed=20272401, dataset='mnms', image_size=384, batch_size=2)
    u = c.Schedule(cfg['data'], 'Canon', True, seed=20272401, dataset='mnms', image_size=384, batch_size=4)
    imp, stats = c.summarize_current(method, l, u, {k:v.cuda() for k,v in saved['importance'].items()})
    saved.update(bank=saved['bank']+[stats], importance=imp, counts=dict(c.COUNTS))
    c.save(out/'final.pt', saved)
    # Keep all original failure files and original terminal receipts.
    c.write(root/'TERMINAL_BEFORE_FINALIZATION_RECOVERY.json', term)
    c.write(out/'EXIT_BEFORE_FINALIZATION_RECOVERY.json', json.loads((out/'EXIT.json').read_text()))
    c.write(out/'EXIT.json', dict(exit_code=0, source_pass=True, step=6000, counts=dict(c.COUNTS), recovered_without_optimizer=True))
    term['failed'].pop('mnms_C', None)
    term.update(status='COMPLETE' if not term['failed'] and all(term['source_ok'].values()) else 'FINISHED_WITH_STOPS', counts=dict(c.COUNTS), end_epoch=time.time(), elapsed_seconds=time.time()-cfg['started_epoch'], allocated_bytes=c.allocated(root))
    c.write(root/'TERMINAL.json', term); c.write(root/'STATUS.json', term)
    c.write(root/'FINALIZATION_RECOVERY.json', dict(status='PASS', reason='optimizer cap was incorrectly applied to zero-update evaluation at equality', optimizer=0, repeated_evaluation=False, counts=dict(c.COUNTS), original_failures_retained=True))
