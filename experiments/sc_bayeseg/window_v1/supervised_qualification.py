"""Matched continuation: pixel CE versus four-class mean CE."""
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import Interface, cpu, metrics, save, write, append
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip


ARMS = ('CE', 'CLASS_MEAN_CE')


def objective(logits, label, class_mean):
    pixels = F.cross_entropy(logits, label, reduction='none')
    if not class_mean:
        return pixels.mean()
    return torch.stack([pixels[label == k].mean() for k in range(4)]).mean()


def loss_check():
    logits = torch.arange(48, dtype=torch.float64).reshape(2, 4, 2, 3) / 20
    y = torch.tensor([[[0, 0, 0], [1, 2, 3]], [[0, 0, 1], [1, 2, 3]]])
    counts = torch.bincount(y.flatten(), minlength=4).double()
    per = F.cross_entropy(logits, y, reduction='none')
    weighted = (per / counts[y]).sum() / 4
    error = float((objective(logits, y, True) - weighted).abs())
    assert error < 1e-12
    assert objective(logits, y, False) == per.mean()
    return dict(status='PASS', class_mean_identity_error=error, optimizer=0, reverse=0)


def run(s):
    root = Path(s['output'])
    root.mkdir(exist_ok=False)
    costs = dict(optimizer=0, reverse=0, train_image_forward=0, prediction_image_forward=0,
                 generator_forward=0, qualification_optimizer=0, qualification_reverse=0)
    active = {}
    began = time.time()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        storage(s, reserve=1_000_000_000)
        write(root / 'CONTEXT.json', s)
        write(root / 'LOSS_CHECK.json', loss_check())
        src = Path(s['interface_source'])
        prior = src.parent.parent
        assert json.loads((prior / 'AUDIT.json').read_text())['status'] == 'PASS'
        payload = torch.load(src, map_location='cpu', weights_only=False)
        assert payload['arm'] == 'R0' and payload['step'] == 600 and payload['source_cursor'] == 8675
        assert all(int(v['step']) == 600 for v in payload['optimizer']['state'].values())
        source_stat = src.stat()
        selection = json.loads((Path(s['origin']) / 'PRIVATE_SELECTION.json').read_text())
        assert len(selection['train_indices']) == 2
        assert not set(selection['train_indices']) & set(selection['heldout_indices'])
        ds = H5LabeledDataset(s['data'], seed=0, dataset='mnms', sites=['Siemens'], transform=None)
        lb = collate_labeled([ds[i] for i in selection['train_indices']]).to('cuda')
        image = F.interpolate(lb.image, (128, 128), mode='bilinear', align_corners=False)
        y = F.interpolate(lb.label[:, None].float(), (128, 128), mode='nearest')[:, 0].long()
        image = torch.cat([flip(image, k) for k in range(4)])
        y = torch.cat([flip(y, k) for k in range(4)])
        counts = torch.bincount(y.flatten(), minlength=4).cpu().tolist()
        assert all(n > 0 for n in counts)
        write(root / 'DESIGN.json', dict(source_arm='R0', source_step=600, source_generator_cursor=8675,
              class_pixel_counts_all_8_views=counts, source_optimizer_steps=600,
              formal_updates_per_arm=1000, data_views=8, no_generator=True))
        # Immutable references are small ledgers, not byte scans of historical datasets.
        history = {p: p.read_text() for p in (prior / 'physical_updates.jsonl', prior / 'RESULTS.json', prior / 'AUDIT.json')}
        initial_prediction = None
        results = {}
        for arm in ARMS:
            out = root / arm
            out.mkdir(exist_ok=False)
            interface = Interface().cuda()
            interface.load_state_dict(payload['model'])
            for name, p in interface.named_parameters():
                p.requires_grad_(name.startswith(('e.', 'd.')))
            opt = torch.optim.Adam([p for p in interface.parameters() if p.requires_grad], lr=3e-4,
                                   weight_decay=0, foreach=False)
            opt.load_state_dict(payload['optimizer'])
            restore_rng_state(payload['rng'])
            assert same(interface.state_dict(), payload['model'])
            assert same(opt.state_dict(), payload['optimizer'])
            assert same(capture_rng_state(), payload['rng'])
            source_h = {k: v for k, v in payload['model'].items() if k.startswith(('hc.', 'hr.'))}
            count = [0]
            def post(opt, args, kwargs):
                count[0] += 1
                costs['optimizer'] += 1
                assert count[0] <= 1000 and costs['optimizer'] <= 2000
                append(root / 'physical_updates.jsonl', dict(index=costs['optimizer'], arm=arm,
                       step=count[0], total_interface_step=600 + count[0]))
            hook = opt.register_step_post_hook(post)
            active = dict(arm=arm, interface=interface, optimizer=opt, count=count)
            def checkpoint(step):
                return dict(arm=arm, step=step, total_interface_step=600 + step, model=interface.state_dict(),
                            optimizer=opt.state_dict(), rng=capture_rng_state(), costs=costs)
            def evaluate(step):
                nonlocal initial_prediction
                with torch.no_grad():
                    probability = interface.d(interface.e(image)).softmax(1)
                costs['prediction_image_forward'] += 8
                save(out / f'PRED_{step:04}.pt', dict(probability=probability))
                if step == 0:
                    if initial_prediction is None:
                        initial_prediction = cpu(probability)
                        old_prediction = torch.load(prior / 'R0/PRED_0600.pt', map_location='cpu', weights_only=False)['direct']
                        assert same(initial_prediction, old_prediction)
                    else:
                        assert same(initial_prediction, probability)
                # Score on CPU after sealing; only the evaluator receives labels.
                row = dict(arm=arm, step=step, total_interface_step=600 + step,
                           metrics=metrics(probability.cpu(), y.cpu()))
                append(root / 'metrics.jsonl', row)
                return row
            save(out / 'S0000.pt', checkpoint(0))
            first = evaluate(0)
            for step in range(1, 1001):
                assert time.time() - s['original_started_unix'] < 85500
                opt.zero_grad(set_to_none=True)
                logits = interface.d(interface.e(image))
                costs['train_image_forward'] += 8
                loss = objective(logits, y, arm == 'CLASS_MEAN_CE')
                assert torch.isfinite(loss)
                loss.backward()
                costs['reverse'] += 1
                norms = {}
                for name, module in [('e', interface.e), ('d', interface.d)]:
                    params = list(module.parameters())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    norms[name] = float(torch.nn.utils.clip_grad_norm_(params, 1.))
                assert all(p.grad is None for module in (interface.hc, interface.hr) for p in module.parameters())
                opt.step()
                append(root / 'steps.jsonl', dict(arm=arm, step=step, loss=float(loss.detach()),
                       gradient_norms_before_clip=norms))
                if step % 25 == 0:
                    write(root / 'STATUS.json', dict(status='RUNNING', arm=arm, step=step, costs=costs,
                          original_elapsed_seconds=time.time() - s['original_started_unix']))
                if step % 100 == 0:
                    storage(s, reserve=1_000_000_000)
                    save(out / f'S{step:04}.pt', checkpoint(step))
                if step in (250, 500, 1000):
                    last = evaluate(step)
                if step % 250 == 0:
                    assert all(same(v, interface.state_dict()[k]) for k, v in source_h.items())
            assert count[0] == 1000
            hook.remove()
            assert all(int(v['step']) == 1600 for v in opt.state_dict()['state'].values())
            assert same(capture_rng_state(), payload['rng'])
            results[arm] = dict(initial=first, final=last, optimizer=1000, image_only=True,
                                unused_H_frozen=True, final_Adam_step=1600, rng_unchanged=True)
            write(out / 'RESULTS.json', results[arm])
            write(out / 'EXIT.json', dict(exit_code=0))
            active = {}
            del opt, interface
            torch.cuda.empty_cache()
        assert costs == dict(optimizer=2000, reverse=2000, train_image_forward=16000,
                             prediction_image_forward=64, generator_forward=0,
                             qualification_optimizer=0, qualification_reverse=0)
        for p, text in history.items():
            assert p.read_text() == text
        stat = src.stat()
        assert (source_stat.st_size, source_stat.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        result = dict(status='COMPLETE', costs=costs, arms=results, current_elapsed_seconds=time.time()-began,
                      original_elapsed_seconds=time.time()-s['original_started_unix'],
                      source_steps_and_optimizer_preserved=True, old_history_unchanged=True,
                      no_heldout_U_target_generator=True, space=storage(s))
        write(root / 'RESULTS.json', result)
        write(root / 'STATUS.json', dict(status='COMPLETE', costs=costs))
        write(root / 'EXIT.json', dict(exit_code=0))
    except BaseException:
        if active:
            save(root / 'failure_partial.pt', dict(arm=active['arm'], step=active['count'][0],
                 model=active['interface'].state_dict(), optimizer=active['optimizer'].state_dict(),
                 rng=capture_rng_state(), costs=costs))
        write(root / 'FAILURE.json', dict(costs=costs, traceback=traceback.format_exc()))
        write(root / 'STATUS.json', dict(status='FAILED', costs=costs))
        write(root / 'EXIT.json', dict(exit_code=1))
        raise


if __name__ == '__main__':
    run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
