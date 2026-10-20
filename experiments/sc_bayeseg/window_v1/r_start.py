"""Bounded image-only interfaces to the frozen categorical generator."""
import copy
import inspect
import json
import math
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state
from sc_bayeseg.model import inference_net
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel


ARMS = ('R0', 'R1', 'R2', 'R2-SG')
STREAMS = (620163, 630163, 640163, 650163)


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(cpu(value), tmp)
    tmp.replace(path)


class Interface(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = inference_net(1, 16, width=16, blocks=4)
        self.d = nn.Conv2d(16, 4, 1)
        self.hc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32))
        self.hr = nn.Conv2d(16, 32, 1)

    def posterior(self, feature):
        b, channels, h, w = feature.shape
        assert (h, w) == (128, 128)
        spatial = feature.reshape(b, channels, 16, 8, 16, 8).mean((3, 5))
        cm, cs = self.hc(feature.mean((2, 3))).chunk(2, 1)
        rm, rs = self.hr(spatial).chunk(2, 1)
        return cm, -4 + 3 * torch.tanh(cs / 3), rm, -4 + 3 * torch.tanh(rs / 3)


def noise_like(value, generator):
    return torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)


def draw_g(model, generator):
    # Freeze weights, while retaining the generator's derivative with respect to c/R.
    for layer in model.anatomy:
        layer.sampled = tuple(getattr(layer, 'mean_' + p).detach()
                              + getattr(layer, 'logsd_' + p).detach().exp()
                              * noise_like(getattr(layer, 'mean_' + p), generator)
                              for p in ('w', 'b'))


def local_sample(q, generator):
    cm, cl, rm, rl = q
    return cm + cl.exp() * noise_like(cm, generator), rm + rl.exp() * noise_like(rm, generator)


def local_kl(q):
    cm, cl, rm, rl = q
    kl = lambda mu, ls: .5 * (mu.square() + (2 * ls).exp() - 1 - 2 * ls).flatten(1).sum(1)
    return kl(cm, cl), kl(rm, rl)


def semantic_loss(model, q, labels, step, counters):
    tg = torch.Generator(device='cuda').manual_seed(720163 + step)
    lg = torch.Generator(device='cuda').manual_seed(820163 + step)
    categories = []
    for _ in range(2):
        draw_g(model, tg)
        counters['train_theta'] += 1
        for _ in range(2):
            c, r = local_sample(q, lg)
            categories.append(F.cross_entropy(model.spatial_logits(c, r), labels, reduction='none').mean((1, 2)))
            counters['train_g_forward'] += 1
            counters['train_g_image_forward'] += len(labels)
    kc, kr = local_kl(q)
    cat = torch.stack(categories).mean()
    kc, kr = kc.mean() / (128 ** 2), kr.mean() / (128 ** 2)
    return cat + kc + kr, dict(category=float(cat.detach()), KLc=float(kc.detach()), KLR=float(kr.detach()))


@torch.no_grad()
def predict(interface, generator_model, image, seeds, theta_draws, local_draws):
    """Image-only API: no label, dataset, identity or label-conditioned state."""
    feature = interface.e(image)
    direct = interface.d(feature).softmax(1)
    if not seeds:
        return direct, []
    q = interface.posterior(feature)
    flows = []
    try:
        for seed in seeds:
            tg = torch.Generator(device=image.device).manual_seed(seed)
            lg = torch.Generator(device=image.device).manual_seed(seed + 1000000)
            total = torch.zeros_like(direct)
            for _ in range(theta_draws):
                draw_g(generator_model, tg)
                for _ in range(local_draws):
                    c, r = local_sample(q, lg)
                    total += generator_model.spatial_logits(c, r).softmax(1)
            flows.append(total / (theta_draws * local_draws))
        return direct, flows
    finally:
        generator_model.clear_draw()


def metrics(probabilities, labels):
    assert probabilities.shape == (8, 4, 128, 128)
    assert torch.isfinite(probabilities).all()
    assert (probabilities.sum(1) - 1).abs().max() < 2e-6
    pred = probabilities.argmax(1)
    rows = []
    for i in range(8):
        ds, areas = [], []
        for k in range(4):
            a, b = pred[i] == k, labels[i] == k
            denom = int(a.sum() + b.sum())
            ds.append(float(2 * (a & b).sum() / denom) if denom else None)
            areas.append(float(a.float().mean()))
        a, b = pred[i] == 2, labels[i] == 2
        tp = int((a & b).sum())
        onehot = F.one_hot(labels[i], 4).permute(2, 0, 1).to(probabilities.dtype)
        rows.append(dict(view=i // 2, image=i % 2, dice=ds,
                         foreground_macro_dice=sum(ds[1:]) / 3,
                         class2_precision=tp / int(a.sum()) if a.any() else None,
                         class2_recall=tp / int(b.sum()) if b.any() else None,
                         predicted_area_fraction=areas,
                         pixel_mixture_NLL=float(F.nll_loss(probabilities[i:i+1].clamp_min(1e-30).log(), labels[i:i+1], reduction='none').mean()),
                         Brier=float((probabilities[i] - onehot).square().sum(0).mean())))
    return dict(per_view_image=rows, foreground_macro_dice=sum(r['foreground_macro_dice'] for r in rows) / 8,
                qualified_24=all(d is not None and d >= .8 for r in rows for d in r['dice'][1:]))


def qualification(model, initial, image, labels, counters):
    interface = Interface().cuda()
    interface.load_state_dict(initial)
    feature = interface.e(image)
    q = interface.posterior(feature)
    assert q[0].shape == (8, 16) and q[2].shape == (8, 16, 16, 16)
    tg = torch.Generator(device='cuda').manual_seed(920163)
    lg = torch.Generator(device='cuda').manual_seed(930163)
    draw_g(model, tg)
    c, r = local_sample(q, lg)
    counters['qualification_g_forward'] += 1
    v = F.cross_entropy(model.spatial_logits(c, r), labels, reduction='none').mean()
    grads = torch.autograd.grad(v, (c, r) + tuple(interface.e.parameters()))
    counters['qualification_reverse'] += 1
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(float(g.norm()) > 0 for g in grads[:2]) and sum(float(g.norm()) for g in grads[2:]) > 0
    detached = interface.posterior(interface.e(image).detach())
    c, r = local_sample(detached, lg)
    counters['qualification_g_forward'] += 1
    v = F.cross_entropy(model.spatial_logits(c, r), labels, reduction='none').mean()
    gs = torch.autograd.grad(v, tuple(interface.e.parameters()) + tuple(interface.hr.parameters()), allow_unused=True)
    counters['qualification_reverse'] += 1
    ne = len(tuple(interface.e.parameters()))
    assert all(g is None for g in gs[:ne]) and any(g is not None and g.abs().max() > 0 for g in gs[ne:])
    model.clear_draw()
    assert 'label' not in str(inspect.signature(predict))
    # Change the evaluator's labels in memory; inference only receives images.
    first = predict(interface, model, image, [940163], 1, 1)
    replaced_labels = torch.zeros_like(labels)
    assert not torch.equal(replaced_labels, labels)
    second = predict(interface, model, image, [940163], 1, 1)
    counters['qualification_g_forward'] += 2
    assert same(first, second)
    kc, kr = local_kl(tuple(torch.zeros_like(v) for v in q))
    assert kc.eq(0).all() and kr.eq(0).all()
    assert all(p.grad is None for p in model.parameters())
    return dict(status='PASS', input_gradients_preserved=True, SG_encoder_gradient_absent=True,
                image_only_signature=str(inspect.signature(predict)), fixed_stream_label_context_invariance=True,
                Gaussian_prior_KL_zero=True, engineering_optimizer_applications=0)


def run(s):
    root = Path(s['output'])
    root.mkdir(exist_ok=False)
    began = time.time()
    counters = dict(optimizer=0, training_reverse=0, train_theta=0, train_g_forward=0,
                    train_g_image_forward=0, train_encoder_image_forward=0, train_direct_image_forward=0,
                    evaluation_theta=0, evaluation_g_forward=0, evaluation_encoder_image_forward=0,
                    evaluation_direct_image_forward=0, qualification_g_forward=0, qualification_reverse=0)
    active = {}
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        space = storage(s, reserve=2_000_000_000)
        write(root / 'CONTEXT.json', s)
        payload = torch.load(s['checkpoint'], map_location='cpu', weights_only=False)
        assert payload['cursor'] == 8675 and payload['population'] == dict(labeled=2, unlabeled=0)
        source_stat = Path(s['checkpoint']).stat()
        model = SpatialResidualModel().cuda()
        model.load_state_dict(payload['model'])
        model.requires_grad_(False)
        model.eval()
        source = cpu(model.state_dict())
        selection = json.loads((Path(s['origin']) / 'PRIVATE_SELECTION.json').read_text())
        assert len(selection['train_indices']) == 2
        assert not set(selection['train_indices']) & set(selection['heldout_indices'])
        dataset = H5LabeledDataset(s['data'], seed=0, dataset='mnms', sites=['Siemens'], transform=None)
        lb = collate_labeled([dataset[i] for i in selection['train_indices']]).to('cuda')
        image = F.interpolate(lb.image, (128, 128), mode='bilinear', align_corners=False)
        label = F.interpolate(lb.label[:, None].float(), (128, 128), mode='nearest')[:, 0].long()
        image = torch.cat([flip(image, k) for k in range(4)])
        label = torch.cat([flip(label, k) for k in range(4)])
        assert all(int((label[i] == k).sum()) > 0 for i in range(8) for k in range(4))
        torch.manual_seed(20261020)
        initial = cpu(Interface().state_dict())
        if s.get('initial_interface'):
            previous_initial = torch.load(s['initial_interface'], map_location='cpu', weights_only=False)
            assert same(initial, previous_initial['model'])
            initial = previous_initial['model']
        write(root / 'PREVIOUS_ATTEMPT.json', s.get('previous_attempt', {}))
        save(root / 'INITIAL_INTERFACE.pt', dict(model=initial, seed=20261020, rng=capture_rng_state()))
        write(root / 'QUALIFICATION.json', qualification(model, initial, image, label, counters))
        results = {}
        for arm in ARMS:
            out = root / arm
            out.mkdir(exist_ok=False)
            interface = Interface().cuda()
            interface.load_state_dict(initial)
            modules = [interface.e]
            if arm != 'R1':
                modules.append(interface.d)
            if arm != 'R0':
                modules.extend([interface.hc, interface.hr])
            allowed = {id(p) for m in modules for p in m.parameters()}
            for p in interface.parameters():
                p.requires_grad_(id(p) in allowed)
            opt = torch.optim.Adam([p for p in interface.parameters() if p.requires_grad], lr=3e-4,
                                   weight_decay=0, foreach=False)
            local_steps = [0]
            def post_hook(optimizer, args, kwargs):
                counters['optimizer'] += 1
                local_steps[0] += 1
                assert counters['optimizer'] <= 2400 and local_steps[0] <= 600
                append(root / 'physical_updates.jsonl', dict(index=counters['optimizer'], arm=arm, step=local_steps[0]))
            hook = opt.register_step_post_hook(post_hook)
            active = dict(arm=arm, interface=interface, optimizer=opt, steps=local_steps)
            def checkpoint(step):
                return dict(arm=arm, step=step, model=interface.state_dict(), optimizer=opt.state_dict(),
                            source_cursor=8675, counters=counters, rng=capture_rng_state())
            def parity(step):
                path = root / 'R0' / f'PARITY_{step:04}.pt'
                ed = {k: v for k, v in interface.state_dict().items() if k.startswith(('e.', 'd.'))}
                ed_states = [opt.state.get(p, {}) for m in (interface.e, interface.d) for p in m.parameters()]
                current = cpu(dict(model=ed, Adam=ed_states))
                if arm == 'R0':
                    save(path, current)
                elif arm == 'R2-SG':
                    assert same(torch.load(path, map_location='cpu', weights_only=False), current), f'R0/SG mismatch at {step}'
                    append(root / 'SG_PARITY.jsonl', dict(step=step, model_and_Adam_exact=True))
            def evaluate(step):
                seeds = STREAMS if arm != 'R0' else ()
                direct, flows = predict(interface, model, image, seeds, 8, 8)
                counters['evaluation_encoder_image_forward'] += 8
                counters['evaluation_direct_image_forward'] += 8
                counters['evaluation_theta'] += len(seeds) * 8
                counters['evaluation_g_forward'] += len(seeds) * 64
                # Seal predictions before the separate evaluator sees labels.
                file = out / f'PRED_{step:04}.pt'
                save(file, dict(direct=direct if arm != 'R1' else None, generator_flows=flows,
                                seeds=list(seeds), source_cursor=8675))
                row = dict(arm=arm, step=step)
                if arm != 'R1':
                    row['direct'] = metrics(direct, label)
                if flows:
                    row['generator'] = metrics(torch.stack(flows).mean(0), label)
                    row['generator_flow_metrics'] = [metrics(p, label) for p in flows]
                append(root / 'metrics.jsonl', row)
                return row
            save(out / 'S0000.pt', checkpoint(0))
            if arm in ('R0', 'R2-SG'):
                parity(0)
            first = evaluate(0)
            for step in range(1, 601):
                assert time.time() - s['original_started_unix'] < 85500
                opt.zero_grad(set_to_none=True)
                feature = interface.e(image)
                counters['train_encoder_image_forward'] += 8
                loss = None
                row = dict(arm=arm, step=step)
                if arm != 'R1':
                    c_loss = F.cross_entropy(interface.d(feature), label, reduction='none').mean()
                    counters['train_direct_image_forward'] += 8
                    row['direct_CE'] = float(c_loss.detach())
                    loss = c_loss
                if arm != 'R0':
                    q = interface.posterior(feature.detach() if arm == 'R2-SG' else feature)
                    a_loss, parts = semantic_loss(model, q, label, step, counters)
                    row.update(parts)
                    loss = a_loss if loss is None else loss + a_loss
                assert torch.isfinite(loss)
                loss.backward()
                counters['training_reverse'] += 1
                norms = {}
                for name, group in [('e', list(interface.e.parameters())), ('d', list(interface.d.parameters())),
                                    ('h', list(interface.hc.parameters()) + list(interface.hr.parameters()))]:
                    enabled = [p for p in group if p.grad is not None]
                    assert all(torch.isfinite(p.grad).all() for p in enabled)
                    if enabled:
                        norms[name] = float(nn.utils.clip_grad_norm_(enabled, 1.))
                assert all(p.grad is None for p in model.parameters())
                opt.step()
                model.clear_draw()
                row.update(loss=float(loss.detach()), gradient_norms_before_clip=norms)
                append(root / 'steps.jsonl', row)
                if step % 25 == 0:
                    if arm in ('R0', 'R2-SG'):
                        parity(step)
                    write(root / 'STATUS.json', dict(status='RUNNING', arm=arm, step=step, costs=counters,
                                                    elapsed_seconds=time.time() - s['original_started_unix']))
                if step % 100 == 0:
                    space = storage(s, reserve=2_000_000_000)
                    save(out / f'S{step:04}.pt', checkpoint(step))
                if step in (200, 600):
                    last = evaluate(step)
                    assert same(source, model.state_dict())
            assert local_steps[0] == 600
            hook.remove()
            results[arm] = dict(initial=first, final=last, optimizer_applications=600)
            write(out / 'RESULTS.json', results[arm])
            write(out / 'EXIT.json', dict(exit_code=0))
            active = {}
            del interface, opt
            torch.cuda.empty_cache()
        assert counters['optimizer'] == counters['training_reverse'] == 2400
        assert counters['train_theta'] == 3600 and counters['train_g_forward'] == 7200
        assert counters['train_g_image_forward'] == 57600
        assert counters['evaluation_theta'] == 288 and counters['evaluation_g_forward'] == 2304
        assert same(source, model.state_dict()) and all(p.grad is None for p in model.parameters())
        end_stat = Path(s['checkpoint']).stat()
        assert (source_stat.st_size, source_stat.st_mtime_ns) == (end_stat.st_size, end_stat.st_mtime_ns)
        assert same(results['R0']['final']['direct'], results['R2-SG']['final']['direct'])
        write(root / 'R_RESULTS.json', dict(status='COMPLETE', arms=results, costs=counters,
              SG_parity_checks=25, frozen_generator_100_parameters=True, original_Adam_not_loaded_or_updated=True,
              source_cursor=8675, no_heldout_U_target=True, space=space, elapsed_seconds=time.time() - began))
        write(root / 'STATUS.json', dict(status='R_COMPLETE_E0_RUNNING', costs=counters))
        from sc_bayeseg.window_v1.student_e0 import run as run_e0
        e0 = run_e0(root / 'E0')
        status = 'COMPLETE' if e0['status'] == 'PASS' else 'R_COMPLETE_E0_FAILED'
        write(root / 'RESULTS.json', dict(status=status, R='R_RESULTS.json', E0=e0, costs=counters,
              elapsed_seconds=time.time() - s['original_started_unix'], space=storage(s)))
        write(root / 'STATUS.json', dict(status=status, costs=counters))
        write(root / 'EXIT.json', dict(exit_code=0 if e0['status'] == 'PASS' else 1))
    except BaseException:
        if active:
            save(root / 'failure_partial.pt', dict(arm=active['arm'], step=active['steps'][0],
                 model=active['interface'].state_dict(), optimizer=active['optimizer'].state_dict(),
                 counters=counters, rng=capture_rng_state()))
        write(root / 'FAILURE.json', dict(costs=counters, traceback=traceback.format_exc()))
        write(root / 'STATUS.json', dict(status='FAILED', costs=counters))
        write(root / 'EXIT.json', dict(exit_code=1))
        raise


if __name__ == '__main__':
    run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
