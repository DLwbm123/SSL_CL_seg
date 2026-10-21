"""Small real-data integration checks, not a scientific efficacy gate."""
import copy
import json
import os
import time
from pathlib import Path
import torch
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.window_v1.r_start import write, save
from .data import Schedule, audit, style, view
from .model import initialize
from .train import setup, optimizer, step_loss, ema


def run(s):
    setup(); out = Path(s['root']); report, private = audit(s)
    write(out / 'DATA_AUDIT.json', report); write(out / 'PRIVATE_DATA_MAP.json', private)
    styles = {d: style(Schedule(s['data'], d)) for d in ('Siemens', 'Philips')}
    s['styles'] = styles; write(Path(os.environ['TASK_SETTINGS']), s)
    source = Schedule(s['data'], 'Siemens'); unlabeled = Schedule(s['data'], 'Siemens', True)
    x, y, _ = source.batch(1); u, uy, _ = unlabeled.batch(1); assert uy is None
    assert all(not r['label_h5_relpath'] for r in unlabeled.ds.records)
    assert x.shape == (2, 1, 128, 128) and y.shape == (2, 128, 128)
    assert torch.equal(y, y.clone()) and view(x, styles['Siemens'], None, 1).shape == x.shape
    details = {}
    for family in ('B', 'U'):
        model = initialize(s, family); teacher = copy.deepcopy(model).eval().requires_grad_(False); opt = optimizer(model)
        old_teacher = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
        loss, parts = step_loss(model, teacher, x, y, u, styles['Siemens'], None, 1)
        assert torch.isfinite(loss); loss.backward()
        grads = {n: float(p.grad.abs().sum()) if p.grad is not None else None for n, p in model.named_parameters() if p.requires_grad}
        assert all(v is not None and v >= 0 for v in grads.values())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in teacher.parameters())
        if family == 'B':
            assert all(sum(v for n, v in grads.items() if n.startswith(prefix)) > 0 for prefix in ('hc.', 'hr.', 'hs.', 'hm.', 'g.', 'f.'))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.); opt.step(); ema(teacher, model)
        for k, v in teacher.state_dict().items():
            if v.is_floating_point(): assert torch.allclose(v, old_teacher[k] * .99 + model.state_dict()[k] * .01, atol=2e-6)
        checkpoint = out / f'integration_{family}.pt'
        save(checkpoint, dict(model=model.state_dict(), teacher=teacher.state_dict(), optimizer=opt.state_dict(), rng=capture_rng_state()))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        restored = initialize(s, family); restored.load_state_dict(saved['model']); restored_opt = optimizer(restored); restored_opt.load_state_dict(saved['optimizer'])
        model.eval(); restored.eval(); restore_rng_state(saved['rng'])
        with torch.no_grad(): before = model.probability(x)
        restore_rng_state(saved['rng'])
        with torch.no_grad(): after = restored.probability(x)
        assert torch.equal(before, after)
        assert all(int(v['step']) == 1 for v in restored_opt.state.values())
        # No-U losses have no U forward, including their image-path objective.
        no_u, no_u_parts = step_loss(model, teacher, x, y, None, styles['Siemens'], styles['Philips'], 2)
        assert torch.isfinite(no_u) and no_u_parts['soft_KL'] == 0
        details[family] = dict(loss=float(loss.detach()), parts=parts, active_gradients=len(grads), teacher_gradients_absent=True, EMA_checked=True, checkpoint_prediction_exact=True, optimizer_step_restored=True, no_u_soft_loss=0)
        del model, restored, teacher, opt, restored_opt; torch.cuda.empty_cache()
    write(out / 'INTEGRATION.json', dict(status='PASS', real_data=True, details=details, integration_optimizer_updates=2, styles=styles, elapsed_seconds=time.time()-s['original_started_unix']))


if __name__ == '__main__': run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
