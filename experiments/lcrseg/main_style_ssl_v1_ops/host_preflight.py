"""Disposable checkpoint restore/backward check with zero optimizer calls."""
import json
import os
import random
import runpy
import traceback
from pathlib import Path
import torch


def check(config):
    result = dict(formal_optimizer_updates=0, engineering_optimizer_updates=0)
    try:
        if config.get('compatibility'):
            operations = runpy.run_path(str(Path(__file__).with_name('host_migration.py')))
            operations['install_compatibility']()
            operations['install_entry'](config)
        from experiments.lcrseg.main_style_ssl_v1 import runtime
        from experiments.lcrseg.five_frameworks_v1.semantics import binding
        from experiments.lcrseg.tests.five_frameworks_v1.test_integration import assert_state_equal
        def cpu(value):
            if isinstance(value, torch.Tensor): return value.cpu()
            if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
            if isinstance(value, list): return [cpu(v) for v in value]
            if isinstance(value, tuple): return tuple(cpu(v) for v in value)
            return value
        batch_trace = []
        if config.get('batch_trace') or config.get('batch_reference'):
            original_batch = runtime.CurrentDomain._batch
            def captured_batch(provider, access, step, stream):
                value = original_batch(provider, access, step, stream)
                batch_trace.append(dict(role=access.role, step=step, stream=stream,
                                        images=value[0].detach().cpu(), targets=value[1].detach().cpu(), patients=value[2]))
                return value
            runtime.CurrentDomain._batch = captured_batch
        worker = config['worker']
        node = worker['node']
        trainer = runtime.setup(node, worker['data'], worker['reference'], 'cuda', worker.get('predecessor'), initialize=False)
        saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        actual = binding(trainer)
        diff = [key for key in actual if actual[key] != saved['semantic'][key]]
        result['semantic_differences'] = diff
        if diff:
            raise ValueError('checkpoint semantic mismatch: '+str(diff))
        runtime.restore(trainer, config['checkpoint'], {k:node[k] for k in ('id','dataset','seed','order','stage','domain')})
        assert trainer.step == trainer.cursor == saved['step']
        assert torch.equal(torch.get_rng_state(), saved['torch_rng'])
        assert random.getstate() == saved['python_rng']
        if saved['cuda_rng'] is not None:
            assert all(torch.equal(a.cpu(), b.cpu()) for a,b in zip(torch.cuda.get_rng_state_all(), saved['cuda_rng']))
        assert [trainer.provider.l_reads, trainer.provider.u_reads] == saved['provider_reads']
        assert trainer.probe == saved['probe'] and trainer.telemetry == saved['telemetry']
        for expected, restored in ((saved['student'],trainer.model.state_dict()), (saved['ema'],trainer.ema.state_dict()),
                                   (saved['scheduler'],trainer.scheduler.state_dict()), (saved['scaler'],trainer.scaler.state_dict()),
                                   (saved['prototypes'],trainer.prototypes.values), (saved['support'],trainer.prototypes.support)):
            assert_state_equal(cpu(expected), cpu(restored))
        for key, state in saved['optimizer']['state'].items():
            assert_state_equal(cpu(state),cpu(trainer.optimizer.state_dict()['state'][key]))
        labeled, unlabeled, _ = trainer.losses()
        trainer.gradient_audit(labeled, unlabeled)
        total = labeled + (0 if unlabeled is None else unlabeled)
        gradients = torch.autograd.grad(total, [p for p in trainer.model.parameters() if p.requires_grad], allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in gradients)
        if config.get('batch_reference'):
            assert_state_equal(torch.load(config['batch_reference'], map_location='cpu', weights_only=False), batch_trace)
            result['continuation_batch_data_order_exact'] = True
        if config.get('batch_trace'):
            torch.save(batch_trace, config['batch_trace'])
        result.update(status='PASS', step=trainer.step, cursor=trainer.cursor, labeled=float(labeled.detach()),
                      unlabeled=None if unlabeled is None else float(unlabeled.detach()), finite_gradients=True,
                      exact_student_ema_optimizer_scheduler_scaler_prototypes=True, active_U=trainer.last['active_U'],
                      rng_probe_telemetry_provider_reads_exact=True,
                      torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name())
    except BaseException as exc:
        result.update(status='FAIL', error=str(exc), traceback=traceback.format_exc())
    Path(config['result']).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    check(json.loads(Path(os.environ['HOST_CONFIG']).read_text()))
