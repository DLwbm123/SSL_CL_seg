"""Create-only two-domain factorial pilot, bounded actual optimizer receipts."""
import copy
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from lcrseg.data import collate_labeled, collate_unlabeled
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.engine.trainer import Trainer, TrainerState, seed_everything
from care_seg_study_v1.data import evaluate
from care_seg_study_v1.run import write, append, save
from rcf_seg_study_v1.data import datasets
from sc_bayeseg.model import Model
from sc_bayeseg.method import Method


class UniformBatcher:
    def __init__(self, dataset, labeled):
        self.dataset, self.labeled = dataset, labeled
        self.dataset.transform = None

    def batch_at(self, step):
        generator = torch.Generator().manual_seed(163 + 100003 * step + (0 if self.labeled else 700001))
        indices = torch.randint(len(self.dataset), (2,), generator=generator).tolist()
        values = [self.dataset[i] for i in indices]
        return (collate_labeled if self.labeled else collate_unlabeled)(values)


def snapshot(method, trainer, cursor):
    return copy.deepcopy(dict(model=method.model.state_dict(), trainer=trainer.state_dict(),
                             rng=capture_rng_state(), cursor=cursor,
                             population=dict(labeled=method.nl, unlabeled=method.nu)))


def restore(payload, method, trainer):
    assert payload['population'] == dict(labeled=method.nl, unlabeled=method.nu)
    method.model.load_state_dict(payload['model'])
    method.model.clear_draw()
    trainer.load_state_dict(copy.deepcopy(payload['trainer']))
    restore_rng_state(payload['rng'])


def same(first, second):
    import numpy as np
    if torch.is_tensor(first):
        return torch.equal(first.cpu(), second.cpu())
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(same(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(same(a, b) for a, b in zip(first, second))
    return first == second


class View(nn.Module):
    def __init__(self, model, corrected):
        super().__init__()
        self.model, self.corrected, self.diagnostics = model, corrected, []

    def forward(self, image):
        from lcrseg.contracts import SegModelOutput
        p, diagnostics = self.model.prediction(image, corrected=self.corrected)
        if diagnostics:
            self.diagnostics.append((len(image), diagnostics))
        return SegModelOutput(p.clamp_min(1e-30).log(), p)


def run(settings):
    root, data = Path(settings['output']), Path(settings['data'])
    root.mkdir(exist_ok=False)
    costs = dict(formal_updates=0, engineering_updates=0)
    active = {}
    started = time.time()
    write(root / 'CONTEXT.json', settings)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        populations = {site: datasets(data, site) for site in ('Siemens', 'Philips')}
        batches = {site: (UniformBatcher(l, True), UniformBatcher(u, False))
                   for site, (l, u) in populations.items()}
        write(root / 'POPULATIONS.json', {site: dict(labeled_slices=len(l), unlabeled_slices=len(u),
              labeled_patients=len({s.row['patient_id'] for s in l.samples}),
              unlabeled_patients=len({s.row['patient_id'] for s in u.samples}))
              for site, (l, u) in populations.items()})
        source_paths, results = {}, {}

        def build(site, use_u):
            seed_everything(163)
            model = Model().cuda()
            l, u = populations[site]
            method = Method(model, len(l), len(u) if use_u else 0)
            method.begin_site(site, None, 2000 if site == 'Siemens' else 1500)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            return method, Trainer(method, optimizer=optimizer, scheduler=scheduler,
                                   device='cuda', amp=False, grad_clip_norm=1.)

        def step(method, trainer, index):
            lb, ub = batches[method.site_id]
            return trainer.train_step(lb.batch_at(index), ub.batch_at(index) if method.nu else None,
                                      state=TrainerState(global_step=index, site_step=index))

        def qualified(method, trainer, out, overfit=False):
            initial = snapshot(method, trainer, 0)
            local = 0
            def receipt(opt, args, kwargs):
                nonlocal local
                local += 1
                costs['engineering_updates'] += 1
                append(root / 'engineering_updates.jsonl', dict(physical=costs['engineering_updates'],
                       stage=out.name, local=local, time=time.time()))
            def guard(opt, args, kwargs):
                if costs['engineering_updates'] >= 24:
                    raise RuntimeError('engineering update budget exhausted')
            pre = trainer.optimizer.register_step_pre_hook(guard)
            post = trainer.optimizer.register_step_post_hook(receipt)
            first = step(method, trainer, 0)
            expected = snapshot(method, trainer, 1)
            restore(initial, method, trainer)
            second = step(method, trainer, 0)
            assert torch.equal(first.total_loss, second.total_loss)
            assert same(expected, snapshot(method, trainer, 1)), 'full-state golden replay mismatch'
            restore(initial, method, trainer)
            overfit_receipt = None
            if overfit:
                fixed_rng = capture_rng_state()
                def fixed_step():
                    restore_rng_state(fixed_rng)
                    return step(method, trainer, 0)
                before = float(first.total_loss.detach())
                for _ in range(8):
                    fixed_step()
                restore_rng_state(fixed_rng)
                lb, ub = batches[method.site_id]
                with torch.no_grad():
                    after = float(method.training_step(lb.batch_at(0).to('cuda'),
                                  ub.batch_at(0).to('cuda') if method.nu else None, 0, 0).total_loss)
                assert after < before, 'fixed-noise two-case joint objective did not decrease'
                overfit_receipt = dict(before=before, after=after, updates=8,
                                       same_visible_cases=True, fixed_noise=True)
                restore(initial, method, trainer)
            pre.remove(); post.remove()
            write(out / 'QUALIFICATION.json', dict(status='PASS', full_state_resume=True,
                  finite_backward=True, engineering_updates=local, two_case=overfit_receipt))

        def train_stage(name, site, use_u, recursive=None, source=None):
            out = root / name
            out.mkdir(exist_ok=False)
            method, trainer = build(site, use_u)
            active.update(method=method, trainer=trainer, out=out, cursor=0)
            if source is not None:
                payload = torch.load(source, map_location='cpu', weights_only=False)
                method.model.load_state_dict(payload['model'])
                assert same(payload['model'], method.model.state_dict()), 'source fork state mismatch'
                method.model.set_prior(recursive)
                write(out / 'FORK.json', dict(source=Path(source).parent.name,
                      inherited_model=True, full_model_equal_before_prior_change=True,
                      recursive_prior=recursive, optimizer_reset=True))
            prior = {k: v.clone() for k, v in method.model.state_dict().items()
                     if 'prior_' in k or 'initial_' in k}
            qualified(method, trainer, out, overfit=name == 'source_L')
            updates = 2000 if source is None else 1500
            actual = 0
            def guard(opt, args, kwargs):
                if costs['formal_updates'] >= 10000 or actual >= updates:
                    raise RuntimeError('formal budget exhausted')
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in method.model.parameters()):
                    raise FloatingPointError('nonfinite gradient')
            def receipt(opt, args, kwargs):
                nonlocal actual
                actual += 1; costs['formal_updates'] += 1
                append(root / 'physical_updates.jsonl', dict(physical=costs['formal_updates'],
                       stage=name, local=actual, time=time.time()))
            pre=trainer.optimizer.register_step_pre_hook(guard)
            post=trainer.optimizer.register_step_post_hook(receipt)
            for index in range(updates):
                result = step(method, trainer, index)
                active['cursor'] = index + 1
                if index == 0 or (index + 1) % 25 == 0:
                    row=dict(status='RUNNING', stage=name, step=index + 1, budget=updates,
                             loss=float(result.total_loss.detach()), costs=costs.copy(),
                             calls=method.model.calls.copy(), time=time.time(), **result.scalars)
                    append(out / 'telemetry.jsonl', row); write(root / 'STATUS.json', row)
                    print(json.dumps(row), flush=True)
                if (index + 1) % 100 == 0:
                    save(out / 'latest.pt', snapshot(method, trainer, index + 1))
            pre.remove(); post.remove()
            assert actual == updates
            assert all(torch.equal(v, method.model.state_dict()[k]) for k, v in prior.items())
            save(out / 'final.pt', snapshot(method, trainer, updates))
            write(out / 'TRAINING_COMPLETE.json', dict(formal_updates=actual, prior_unchanged=True,
                   status='COMPLETE', calls=method.model.calls.copy(),
                   parameters=sum(p.numel() for p in method.model.parameters())))
            rows = {}
            for seen in (('Siemens',) if source is None else ('Siemens', 'Philips')):
                rows[seen] = {}
                for corrected in (False, True):
                    mode = 'corrected' if corrected else 'direct'
                    write(root / 'STATUS.json', dict(status='EVALUATING', stage=name,
                          site=seen, prediction=mode, costs=costs.copy(), time=time.time()))
                    seed_everything(50163 if seen == 'Siemens' else 50164)
                    before = copy.deepcopy(method.model.state_dict())
                    view = View(method.model, corrected)
                    metrics, private = evaluate(view, data, seen)
                    assert same(before, method.model.state_dict()), 'evaluation changed persistent model'
                    if view.diagnostics:
                        count = sum(n for n, _ in view.diagnostics)
                        metrics['importance_diagnostics'] = {
                            k: sum(n * d[k] for n, d in view.diagnostics) / count
                            for k in view.diagnostics[0][1]}
                    rows[seen][mode] = metrics
                    write(out / f'private_{seen}_{mode}.json', private)
            write(out / 'RESULTS.json', rows)
            write(out / 'COSTS.json', method.model.calls)
            results[name] = rows
            del method, trainer
            gc.collect(); torch.cuda.empty_cache()
            return out / 'final.pt'

        for use_u in (False, True):
            source_paths[use_u] = train_stage('source_LU' if use_u else 'source_L', 'Siemens', use_u)
        for name, use_u, recursive in [('A', False, False), ('B', True, False),
                                       ('C', False, True), ('D', True, True)]:
            train_stage(name, 'Philips', use_u, recursive, source_paths[use_u])
        assert costs == dict(formal_updates=10000, engineering_updates=20)
        complete = dict(status='COMPLETE', started=started, finished=time.time(), costs=costs,
                        results=results, scope='single-seed two-domain validation development pilot')
        write(root / 'RESULTS.json', complete); write(root / 'STATUS.json', complete)
    except BaseException:
        failure=dict(status='FAILED', costs=costs, traceback=traceback.format_exc(), time=time.time())
        if active:
            try:
                save(active['out'] / 'failure_partial.pt', snapshot(active['method'],
                     active['trainer'], active['cursor']))
                failure['partial_saved'] = True
            except BaseException as exc:
                failure['partial_save_error'] = str(exc)
        write(root / 'FAILURE.json', failure); write(root / 'STATUS.json', failure)
        raise


if __name__ == '__main__':
    settings = json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
    root = Path(settings['output'])
    if root.exists():
        raise FileExistsError('create-only output already exists')
    code=1
    try:
        run(settings); code=0
    finally:
        if root.exists():
            write(root / 'EXIT.json', dict(exit_code=code, time=time.time()))
