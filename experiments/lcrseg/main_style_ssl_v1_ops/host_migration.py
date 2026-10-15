"""Host handoff operations; the frozen training loop and checkpoints stay intact."""
import json
import importlib.util
import os
import subprocess
import time
from pathlib import Path
spec = importlib.util.spec_from_file_location('_handoff_resource', Path(__file__).with_name('resource.py'))
resource = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource)


def install_compatibility():
    # Torch 2.7 adds this false Adam default; the executed coupled decay is unchanged.
    from experiments.lcrseg.five_frameworks_v1 import semantics
    original = semantics.optimizer_group_binding
    def groups(trainer):
        result = original(trainer)
        for group in result:
            if 'decoupled_weight_decay' in group:
                if group['decoupled_weight_decay'] is not False:
                    raise ValueError('migration cannot change the Adam decay equation')
                del group['decoupled_weight_decay']
        return result
    semantics.optimizer_group_binding = groups
    import torch
    original_scheduler_state = torch.optim.lr_scheduler.LambdaLR.state_dict
    def scheduler_state(scheduler):
        return normalize_scheduler_state(original_scheduler_state(scheduler))
    torch.optim.lr_scheduler.LambdaLR.state_dict = scheduler_state


def normalize_scheduler_state(state):
    # Torch 2.6 retains a disabled print flag removed in 2.7; step and LR stay intact.
    if 'verbose' in state:
        if state['verbose'] is not False:
            raise ValueError('unexpected enabled scheduler verbosity')
        state = {k:v for k,v in state.items() if k != 'verbose'}
    return state


def install_entry(config):
    """Restore the authenticated old entry, retaining strict checkpoint binding."""
    if not config.get('entry_state'):
        return
    import torch
    from experiments.lcrseg.main_style_ssl_v1 import runtime
    from experiments.lcrseg.five_frameworks_v1.semantics import tensor_fingerprint
    entry = torch.load(config['entry_state'], map_location='cpu', weights_only=False)
    saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    assert entry['identity'] == saved['identity']
    assert entry['checkpoint_semantic_sha256'] == saved['semantic_sha256']
    state = dict(entry['parent'], __F_prev__=entry['previous'])
    assert tensor_fingerprint(state) == saved['semantic']['stage_entry_tensors'] == entry['stage_entry_fingerprint']
    original = runtime.Parent
    class HistoricalParent(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            current = self.state_dict()
            assert current.keys() == entry['parent'].keys()
            for key, value in current.items():
                if key.startswith('V64_') or key.endswith(('.A', '.right')):
                    continue
                assert torch.equal(value.cpu(), entry['parent'][key]), 'frozen predecessor changed: '+key
            self.load_state_dict(entry['parent'], strict=True)
    runtime.Parent = HistoricalParent


def validate_handoffs(config, plan):
    assert config['gpus'] in ([4, 6], [0, 1]) and config['capacity'] == 2
    assert sum(n['steps'] for n in plan) == 511600
    expected = set(config.get('expected_handoffs', (
        'prostate_s163_o1_t1', 'prostate_s164_o1_t1', 'mnms_s163_source', 'mnms_s164_source')))
    assert len(expected) == 4 and expected <= {n['id'] for n in plan}
    pending = set(config['pending_handoffs'])
    assert set(config['handoffs']) | pending == expected
    assert not set(config['handoffs']) & pending


def available_cards(config):
    if not config.get('minimum_free_mib'):
        return config['gpus']
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(i):int(m) for i,m in (row.split(',') for row in rows.splitlines())}
    return [g for g in config['gpus'] if free.get(g, 0) >= config['minimum_free_mib']]


def run(config):
    from experiments.lcrseg.main_style_ssl_v1.protocol import nodes
    from experiments.lcrseg.main_style_ssl_v1.controller import summarize
    root = Path(config['root'])
    change = Path(config['change'])
    plan = nodes()
    states = config['states'].copy()
    pending = set(config['pending_handoffs'])
    running = {}
    logs = {}
    started = time.time()
    validate_handoffs(config, plan)
    frozen = json.loads((root/'PLAN.json').read_text())
    assert frozen['nodes'] == plan and frozen['total_optimizer_updates'] == 511600

    def launch(node, gpu, handoff=None):
        key = node['id']
        predecessor = node['predecessor']
        worker = dict(node=node, output=str(root/'jobs'/key), data=config['data'],
                      reference=config['reference'], predecessor=None if predecessor is None
                      else str(root/'jobs'/predecessor/'sealed.pt'))
        code = config['worker_code_roots'].get(key, config['default_code_root'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=code,
                   HOST_SCRIPT=config['script'], HOST_CONFIG=str(change/(key+'.config.json')))
        if handoff:
            resource.write(change/(key+'.config.json'), dict(worker=worker, handoff=handoff))
            env['HOST_ACTION'] = 'resume'
        else:
            resource.write(change/(key+'.config.json'), worker)
            env['HOST_ACTION'] = 'worker'
        log = (change/(key+'.log')).open('x')
        p = subprocess.Popen(['bash', './s', config['python'], './n.py'], cwd=config['ops'],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        running[key] = dict(pid=p.pid, birth=resource.process(p.pid)['birth'], gpu=gpu,
                            resumed_from=None if handoff is None else handoff['step'])
        logs[key] = log

    for key, handoff in config['handoffs'].items():
        launch(next(n for n in plan if n['id'] == key), config['placements'][key], handoff)
    while len(states) < len(plan):
        for key in list(pending):
            receipt = change/(key+'.ADMITTED.json')
            if not receipt.exists():
                continue
            handoff = json.loads(receipt.read_text())
            assert handoff['preflight_status'] == 'PASS'
            if config['placements'][key] not in available_cards(config):
                continue
            assert sum(e['gpu'] == config['placements'][key] for e in running.values()) < config['capacity']
            launch(next(n for n in plan if n['id'] == key), config['placements'][key], handoff)
            pending.remove(key)
        for key, entry in list(running.items()):
            if resource.alive(entry):
                continue
            folder = root/'jobs'/key
            status = json.loads((folder/'STATUS.json').read_text()) if (folder/'STATUS.json').exists() else {}
            exit_value = json.loads((folder/'EXIT.json').read_text()) if (folder/'EXIT.json').exists() else {}
            states[key] = 'COMPLETE' if status.get('status') == 'COMPLETE' and exit_value.get('exit_code') == 0 else 'FAILED'
            logs.pop(key).close()
            del running[key]
        for node in plan:
            key, predecessor = node['id'], node['predecessor']
            if key in states or key in running or key in pending:
                continue
            if predecessor and states.get(predecessor) in ('FAILED', 'BLOCKED'):
                states[key] = 'BLOCKED'
                out = root/'jobs'/key
                out.mkdir(exist_ok=False)
                resource.write(out/'STATUS.json', dict(status='BLOCKED', node=node, reason='predecessor failed; no automatic retry'))
                continue
            gpu = resource.slot(running, available_cards(config), config['capacity'])
            if gpu is None or predecessor and states.get(predecessor) != 'COMPLETE':
                continue
            launch(node, gpu)
        resource.write(root/'STATUS.json', dict(status='RUNNING', completed=states, running=running,
                       elapsed_seconds=config['previous_elapsed']+time.time()-started, total_nodes=len(plan),
                       scheduling=dict(gpus=config['gpus'], capacity_per_gpu=config['capacity']),
                       active_host=config.get('active_host', 'bridge-gpu'), pending_host_handoffs=sorted(pending)))
        if len(states) < len(plan):
            time.sleep(2)
    summary = summarize(root, plan)
    failed = {k:v for k,v in states.items() if v != 'COMPLETE'}
    resource.write(root/'STATUS.json', dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',
                   nodes=states, elapsed_seconds=config['previous_elapsed']+time.time()-started))
    resource.write(root/'FINAL.json', dict(completed=len(states), failures=failed, summary=summary,
                   public_delivery='PENDING_COMPLETION_DELIVERY', scientific_success='NOT_ASSUMED',
                   attempt_history=config['attempt_history'], host_migration=config['migration_receipt']))


def main():
    install_compatibility()
    config = json.loads(Path(os.environ['HOST_CONFIG']).read_text())
    action = os.environ['HOST_ACTION']
    if action == 'run':
        run(config)
    elif action == 'resume':
        install_entry(dict(entry_state=config['handoff'].get('entry_state'), checkpoint=config['handoff']['checkpoint']))
        resource.resume(config)
    elif action == 'worker':
        from experiments.lcrseg.main_style_ssl_v1.runtime import worker
        worker(config)
    else:
        raise ValueError('unsupported migration action')


if __name__ == '__main__':
    main()
