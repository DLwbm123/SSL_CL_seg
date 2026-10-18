"""Bounded Canon foreground selection by decoder factorial diagnostic."""
import copy
import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from care_seg_study_v1.data import PatientBatcher, evaluate
from care_seg_study_v1.run import append, save, write
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.engine.trainer import TrainerState, seed_everything
from rcf_seg_study_v1.data import datasets
from rcf_seg_study_v1.method import RCFMethod
from rcf_seg_study_v1.model import Segmenter
from rcf_seg_study_v1.run import snapshot, trainer_for


ARMS = (('A', False, False), ('B', True, False),
        ('C', False, True), ('D', True, True))


def foreground_half(targets, scores, original):
    """Exactly ceil(n/2) per predicted foreground class; stable population-order ties."""
    targets = np.asarray(targets)
    scores = np.asarray(scores)
    result = np.asarray(original).copy() & (targets == 0)
    for c in (1, 2, 3):
        indices = np.flatnonzero(targets == c)
        order = np.argsort(-scores[indices], kind='stable')
        result[indices[order[:(len(indices) + 1) // 2]]] = True
    return result


def aligned_targets(method, batch, device):
    ys, masks = [], []
    for case, z, record in zip(batch.case_id, batch.slice_index, batch.geometry_record):
        y, mask = method.targets[(case, z)], method.selected[(case, z)]
        for field, dim in (('hflip', -1), ('vflip', -2)):
            if record.get(field):
                y, mask = y.flip(dim), mask.flip(dim)
        ys.append(y)
        masks.append(mask)
    y = torch.stack(ys).to(device).long()
    mask = torch.stack(masks).to(device) & batch.strong_valid_mask.to(device)[:, 0]
    return y, mask


class FactorMethod(RCFMethod):
    def __init__(self, model, labeled, unlabeled, foreground):
        super().__init__(model, labeled, unlabeled, 'rcf')
        self.foreground = foreground
        self.diagnostics_enabled = False
        self.derivative_calls = 0

    def prepare(self, step):
        previous = self.last_epoch
        super().prepare(step)
        if self.last_epoch == previous or not self.targets:
            return
        keys = list(self.targets)
        labels = np.concatenate([self.targets[k].numpy().ravel() for k in keys])
        original = np.concatenate([self.selected[k].numpy().ravel() for k in keys])
        if self.foreground:
            scores = np.concatenate([self.history[k].ravel() for k in keys])
            selected = foreground_half(labels, scores, original)
            offset = 0
            for k in keys:
                n = self.targets[k].numel()
                self.selected[k] = torch.from_numpy(selected[offset:offset+n].reshape(self.targets[k].shape).copy())
                offset += n
            assert np.array_equal(selected[labels == 0], original[labels == 0])
        else:
            selected = original
        for c in range(4):
            self.accounting['pseudo_count_' + str(c)] = int((labels == c).sum())
            self.accounting['selected_count_' + str(c)] = int(((labels == c) & selected).sum())
        self.accounting['selected_fraction'] = float(selected.mean())
        self.accounting['foreground_half'] = self.foreground

    def training_step(self, labeled_batch, unlabeled_batch, global_step, site_step):
        diagnose = self.diagnostics_enabled and site_step in (300, 900) and bool(self.targets)
        outputs = []
        hook = self.model.register_forward_hook(lambda module, args, out: outputs.append(out.logits)) if diagnose else None
        try:
            result = super().training_step(labeled_batch, unlabeled_batch, global_step, site_step)
        finally:
            if hook is not None:
                hook.remove()
        if diagnose:
            assert len(outputs) == 2
            y, mask = aligned_targets(self, unlabeled_batch, outputs[-1].device)
            pixel = F.cross_entropy(outputs[-1], y, reduction='none')
            params = [p for p in self.model.parameters() if p.requires_grad]
            losses = {'supervised': result.losses['loss_sup']}
            losses.update({'U_class_' + str(c): (pixel * mask * (y == c)).sum() / mask.sum().clamp_min(1)
                           for c in range(4)})
            for name, loss in losses.items():
                gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                self.derivative_calls += 1
                norm = sum(float(g.detach().float().square().sum()) for g in gradients if g is not None) ** .5
                result.scalars['gradient_norm_' + name] = norm
        return result


def restore(payload, method, trainer):
    method.model.load_state_dict(payload['model'])
    method.load_method_state_dict(payload['method'])
    # Optimizer load can alias tensor states on the same device.
    trainer.load_state_dict(copy.deepcopy(payload['trainer']))
    restore_rng_state(payload['rng'])


def equal(first, second):
    if torch.is_tensor(first):
        return torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(equal(x, y) for x, y in zip(first, second))
    return first == second


def decoder_parameters(model):
    parameters = []
    for module in (model.net.decoder, model.net.segmentation_head):
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                for p in child.parameters(recurse=False):
                    p.requires_grad_(True)
                    parameters.append(p)
    assert parameters
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert all(not p.requires_grad for p in module.parameters())
    return parameters


def run(settings):
    root = Path(settings['output'])
    root.mkdir(exist_ok=False)
    started = time.time()
    costs = dict(formal_updates=0, engineering_updates=0, diagnostic_derivative_calls=0,
                 segmentation_image_forwards=0, encoder_image_forwards=0, evaluation_slice_forwards=0)
    write(root / 'CONTEXT.json', settings)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        labeled, unlabeled = datasets(Path(settings['data']), 'Canon')
        assert len(labeled.samples) == 116 and len(unlabeled.samples) == 530
        lb = PatientBatcher(labeled, {s.row['patient_id'] for s in labeled.samples}, True, 163)
        ub = PatientBatcher(unlabeled, {s.row['patient_id'] for s in unlabeled.samples}, False, 163)
        source = torch.load(settings['source'], map_location='cpu', weights_only=False)

        def build(foreground):
            seed_everything(163)
            model = Segmenter().cuda()
            model.load_state_dict(source['model'], strict=True)
            model.isolate()
            seed_everything(166)
            model.add_task('Canon')
            def segmentation(module, inputs):
                costs['segmentation_image_forwards'] += len(inputs[0])
            def encoder(module, inputs):
                costs['encoder_image_forwards'] += len(inputs[0])
            model.register_forward_pre_hook(segmentation)
            model.net.transformer.register_forward_pre_hook(encoder)
            method = FactorMethod(model, labeled, unlabeled, foreground)
            method.begin_site('Canon', None, 1500)
            return method, trainer_for(method, 1500)

        def step(method, trainer, index):
            return trainer.train_step(lb.batch_at(index), ub.batch_at(index),
                                      state=TrainerState(global_step=index, site_step=index, epoch=index//150))

        def qualify(method, trainer, out, index, overfit=False):
            out.mkdir(exist_ok=False)
            method.prepare(index)
            initial = snapshot(method, trainer, index)
            actual = 0
            def receipt(opt, args, kwargs):
                nonlocal actual
                actual += 1
                costs['engineering_updates'] += 1
                append(root / 'engineering_updates.jsonl', dict(physical=costs['engineering_updates'],
                       stage=out.name, local=actual, time=time.time()))
            def guard(opt, args, kwargs):
                if costs['engineering_updates'] >= 12:
                    raise RuntimeError('engineering budget exhausted')
            pre = trainer.optimizer.register_step_pre_hook(guard)
            post = trainer.optimizer.register_step_post_hook(receipt)
            first = step(method, trainer, index)
            expected = snapshot(method, trainer, index)
            restore(initial, method, trainer)
            second = step(method, trainer, index)
            assert torch.equal(first.total_loss, second.total_loss)
            assert equal(expected, snapshot(method, trainer, index)), 'full-state exact resume mismatch'
            restore(initial, method, trainer)
            overfit_result = None
            if overfit:
                batch = lb.batch_at(0).to('cuda')
                def loss():
                    method.model.eval()
                    with torch.no_grad():
                        return float(method._supervised_losses(method.model(batch.image), batch)['loss_sup'])
                before = loss()
                step(method, trainer, 0)
                step(method, trainer, 0)
                after = loss()
                assert after < before, 'two-case qualification loss did not decrease'
                overfit_result = dict(before=before, after=after, updates=2)
                restore(initial, method, trainer)
            pre.remove()
            post.remove()
            trainer.optimizer.zero_grad(set_to_none=True)
            assert actual == (4 if overfit else 2)
            write(out / 'QUALIFICATION.json', dict(status='PASS', engineering_updates=actual,
                  exact_resume_full_state=True, finite_backward=True, two_case=overfit_result))
            del initial, expected, first, second
            gc.collect()

        def train(method, trainer, out, start, stop):
            local = 0
            allowed = {n for n, p in method.model.named_parameters() if p.requires_grad}
            frozen = {n: v.detach().cpu().clone() for n, v in method.model.state_dict().items() if n not in allowed}
            def guard(opt, args, kwargs):
                if costs['formal_updates'] >= 5100 or local >= stop-start:
                    raise RuntimeError('formal update budget exhausted')
                for name, p in method.model.named_parameters():
                    if name not in allowed and p.grad is not None:
                        raise AssertionError('gradient on frozen parameter: ' + name)
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError('nonfinite gradient')
            def receipt(opt, args, kwargs):
                nonlocal local
                local += 1
                costs['formal_updates'] += 1
                append(root / 'physical_updates.jsonl', dict(physical=costs['formal_updates'],
                       stage=out.name, local=local, site_step=start+local, time=time.time()))
            pre = trainer.optimizer.register_step_pre_hook(guard)
            post = trainer.optimizer.register_step_post_hook(receipt)
            method.diagnostics_enabled = start == 300
            for index in range(start, stop):
                if index % 150 == 0:
                    write(root / 'STATUS.json', dict(status='CALIBRATING', stage=out.name, step=index, costs=costs))
                method.prepare(index)
                result = step(method, trainer, index)
                if index == start or (index+1) % 25 == 0:
                    row = dict(stage=out.name, step=index+1, loss=float(result.total_loss.detach()),
                               loss_sup=float(result.losses['loss_sup'].detach()),
                               loss_U=float(result.losses['loss_assim'].detach()), time=time.time(), **result.scalars)
                    append(out / 'telemetry.jsonl', row)
                    write(root / 'STATUS.json', dict(status='RUNNING', **row, costs=costs))
                    print(json.dumps(row), flush=True)
                if (index+1) % 150 == 0:
                    save(out / 'latest.pt', snapshot(method, trainer, index+1))
            pre.remove()
            post.remove()
            assert local == stop-start
            assert all(torch.equal(v.cpu(), frozen[n]) for n, v in method.model.state_dict().items() if n in frozen)
            costs['diagnostic_derivative_calls'] += method.derivative_calls
            assert method.derivative_calls == (10 if start == 300 else 0)
            write(out / 'TRAINING_COMPLETE.json', dict(status='COMPLETE', formal_updates=local,
                  frozen_parameters_and_buffers_unchanged=True, trainable_parameters=sum(p.numel() for p in method.model.parameters() if p.requires_grad),
                  diagnostic_derivative_calls=method.derivative_calls))

        method, trainer = build(False)
        warm = root / 'warmup'
        qualify(method, trainer, warm, 0, overfit=True)
        train(method, trainer, warm, 0, 300)
        save(root / 'shared_300.pt', snapshot(method, trainer, 300))
        del method, trainer
        gc.collect()
        torch.cuda.empty_cache()
        metrics = {}
        for arm, foreground, decoder in ARMS:
            method, trainer = build(foreground)
            shared = torch.load(root / 'shared_300.pt', map_location='cuda', weights_only=False)
            # RNG tensors must be on CPU even when model and Adam states are loaded to CUDA.
            for k in ('rng_torch_cpu',):
                shared['rng'][k] = shared['rng'][k].cpu()
            shared['rng']['rng_torch_cuda'] = [x.cpu() for x in shared['rng']['rng_torch_cuda']]
            restore(shared, method, trainer)
            assert equal(shared, snapshot(method, trainer, 300)), 'shared warmup fork mismatch'
            if decoder:
                params = decoder_parameters(method.model)
                trainer.optimizer.add_param_group(dict(params=params, lr=5e-4, weight_decay=1e-4, initial_lr=5e-4))
                trainer.scheduler.base_lrs.append(5e-4)
                trainer.scheduler.lr_lambdas.append(trainer.scheduler.lr_lambdas[0])
                assert all(p not in trainer.optimizer.state for p in params)
            del shared
            out = root / arm
            qualify(method, trainer, out, 300)
            write(out / 'FORK.json', dict(shared_step=300, shared_full_state_equal=True,
                  foreground_half=foreground, decoder_convolutions_trainable=decoder,
                  inherited_lora_optimizer=True, new_decoder_optimizer_empty=decoder))
            train(method, trainer, out, 300, 1500)
            method.model.requires_grad_(False)
            method.model.eval()
            method.model.activate('Canon')
            before = costs['segmentation_image_forwards']
            result, private = evaluate(method.model, Path(settings['data']), 'Canon')
            actual = costs['segmentation_image_forwards']-before
            assert actual == 294
            costs['evaluation_slice_forwards'] += actual
            metrics[arm] = result
            write(out / 'RESULTS.json', result)
            write(out / 'private_patients.json', private)
            save(out / 'final.pt', dict(model=method.model.state_dict(), tasks=['Canon'], routers={}, steps=1500))
            del method, trainer
            gc.collect()
            torch.cuda.empty_cache()
        assert costs['formal_updates'] == 5100 and costs['engineering_updates'] == 12
        assert costs['evaluation_slice_forwards'] == 1176 and costs['diagnostic_derivative_calls'] == 40
        a, b, c, d = [100 * metrics[name]['macro_dice'] for name in ('A', 'B', 'C', 'D')]
        effects = dict(foreground_frozen=b-a, foreground_open=d-c,
                       decoder_original=c-a, decoder_foreground=d-b,
                       interaction=(d-c)-(b-a), foreground_main=((b-a)+(d-c))/2,
                       decoder_main=((c-a)+(d-b))/2)
        results = dict(status='COMPLETE', metrics=metrics, costs=costs, started=started, finished=time.time(),
                       effects_percentage_points=effects, code_commit=settings['code_commit'],
                       scope='single-seed Canon validation factorial; fixed domain expert')
        write(root / 'RESULTS.json', results)
        report = ['# Canon 前景筛选与解码器四格结果', '',
                  '本轮固定 Canon 专家评价，单种子已见验证集诊断。全部四格终态均保留，无挑选最佳 checkpoint。', '',
                  '| 分支 | 前景干预 | 解码器卷积可训练 | Dice % |', '|---|---|---|---:|']
        for name, foreground, decoder in ARMS:
            report.append(f'| {name} | {foreground} | {decoder} | {100*metrics[name]["macro_dice"]:.4f} |')
        report += ['', '条件效应及交互，单位百分点：', '']
        report += [f'- {name}: {value:+.4f}' for name, value in effects.items()]
        report += ['', '成本与完整结果见 RESULTS.json；资格、分叉、冻结检查和逐阶段物理更新账本独立保留。',
                   '前景干预同时改变入选数量和组成；解码器干预改变可训练参数空间。不能据单种子验证差值声称统计显著、独立泛化或无任务 ID 部署收益。',
                   '此为后台自动生成汇总，公开交付前仍需核对实际账本、完整阶段回执及结果覆盖。', '']
        (root / 'REPORT_ZH.md').write_text('\n'.join(report))
        write(root / 'STATUS.json', results)
    except BaseException:
        write(root / 'FAILURE.json', dict(traceback=traceback.format_exc(), costs=costs, time=time.time()))
        write(root / 'STATUS.json', dict(status='FAILED', costs=costs))
        raise


if __name__ == '__main__':
    settings = json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
    if Path(settings['output']).exists():
        raise FileExistsError('create-only output already exists')
    code = 1
    try:
        run(settings)
        code = 0
    finally:
        root = Path(settings['output'])
        if root.exists():
            write(root / 'EXIT.json', dict(exit_code=code, time=time.time()))
