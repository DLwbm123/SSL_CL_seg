"""Budget-scoped data overlay and V6 jobs over the existing native engine."""
import copy
import fcntl
import gc
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from experiments.qprompt_rl_v1.v5_grpo_group_control import worker as w

C, ROOT, e, v4, q, p = w.C, w.ROOT, w.e, w.v4, w.q, w.p
BASE_PROVIDER = e.Provider
BASE_FEEDBACK = e.Trainer.feedback_score
BUDGETS = {20: {'RIM_ONE_r3': 16, 'Drishti_GS': 10},
           10: {'RIM_ONE_r3': 8, 'Drishti_GS': 5},
           5: {'RIM_ONE_r3': 4, 'Drishti_GS': 3}}


class BudgetProvider(BASE_PROVIDER):
    def __init__(self, *args, development=False, **kwargs):
        super().__init__(*args, development=False, **kwargs)
        self.development = development
        original = self._l
        patients = sorted(set(self._patients.values()))
        assert len(patients) == len(original), 'V6 expects one image per patient'
        order = np.random.RandomState(e.stable('V6/subset', C['subset_seed'], self.domain)).permutation(len(patients))
        count = BUDGETS[C['label_percent']][self.domain]
        chosen = [patients[i] for i in order[:count]]
        selected = set(chosen)
        online = set(chosen[:1])
        fit = selected - online if development else selected
        assert len(fit) >= 2
        def subset(ids):
            ds = copy.copy(original)
            ds.rows = [r for r in original.rows if self._patients[r['case_id']] in ids]
            ds.checked = set()
            return ds
        self._l = subset(fit)
        # Excluded labels never enter the unlabeled accessor, even in its metadata.
        self._u = copy.copy(self._u)
        self._u.rows = list(self._u.rows) + [
            {k: r[k] for k in ('case_id', 'image_h5_relpath', 'image_sha256')}
            for r in original.rows if self._patients[r['case_id']] not in selected]
        self._u.checked = set()
        assert all(not any('label' in k for k in r) for r in self._u.rows)
        self.roles = dict(fit=sorted(fit), online=sorted(online) if development else [], audit=[])
        self.feedback = dict(online=subset(online), audit=subset(set())) if development else {}
        self._identity.update(label_percent=C['label_percent'], label_count=count,
                              subset_seed=C['subset_seed'], selected_patients=sorted(selected),
                              reward_patients=sorted(online) if development else [],
                              overlay='V6_nested_label_budget_v1')
        self.budget_receipt = dict(label_percent=C['label_percent'], labeled_patients=count,
                                  fit_patients=len(fit), reward_patients=1 if development else 0,
                                  audit_patients=0, unlabeled_images=len(self._u),
                                  total_train_images=len(original) + len(self._u) - (len(original)-count),
                                  student_training_images=len(self._l)+len(self._u),
                                  original_steps_per_epoch=self.steps_per_epoch,
                                  hidden_label_fields_in_U=0, development=development)


def feedback_score(t, role):
    if role == 'audit':
        assert not t.provider.roles['audit']
        return None
    return BASE_FEEDBACK(t, role)


def create(seed, domain, development, ledger, **unused):
    assert seed == 168, 'fixed source model and segmentation seed'
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    e.random.seed(seed); np.random.seed(seed)
    t = e.create(dict(C, source=C['reused_source']), seed, domain, development, ledger)
    assert t.options['total_steps'] == p.HORIZONS[domain]
    e.write(ROOT / 'LABEL_BUDGET.json', t.provider.budget_receipt)
    e.write(ROOT / 'ROLES.private.json', t.provider.roles)
    return t


def initialize(ledger):
    d = C['domain']
    rows = e.read(w.CAMPAIGN/'jobs'/f'panel_168_{d}'/'PANELS.private.json')
    x = np.array([r['x'] for r in rows]); scaler = v4.policy.scaler(x)
    xx = q.normalize(x, scaler)
    outcomes = np.array([[a['25']['online'] for a in r['outcomes']] for r in rows])
    advantage = outcomes - outcomes[:, 2:3]
    scale = max(1e-4, float(np.sqrt(np.mean(advantage**2))))
    model = v4.policy.offline(xx, advantage/scale, e.stable('V6/offline', d, C['subset_seed']),
                             lambda opt,i: ledger.step(opt,'controller',str(i)))
    v4.save_policy(model, ROOT/'OFFLINE_25.pt', scaler=scaler, reward_scale=scale, horizon=25)
    softened = q.Policy(model.state_dict()); diag = q.soften(softened, xx)
    state = copy.deepcopy(model.state_dict())
    state.update({'actor.'+k:v for k,v in softened.actor.state_dict().items()})
    e.atomic_save(dict(state=state, scaler=scaler, panel=xx,
                       initial_logits=softened.actor(xx).detach()), ROOT/'INITIAL.private.pt')
    e.write(ROOT/'INITIALIZATION.json', dict(domain=d, contexts=len(rows),
        label_percent=C['label_percent'], subset_seed=C['subset_seed'],
        historical_target_label_policy_reused=False, audit_disabled=True, **diag))


def load_final(d, m, controller):
    variant = C['variant']
    root = w.CAMPAIGN/'jobs'/f'learn_{controller}_{d}_{variant}'
    raw = torch.load(root/'FROZEN_POLICY.private.pt', map_location='cpu', weights_only=False)
    model, _ = w.load_init(d, m)
    model.load_state_dict(raw['state']); model.eval()
    return model, raw['scaler']


def evaluate_file(path):
    env = dict(os.environ, EVAL_INPUT=str(path),
               EXEC_MODULE='experiments.qprompt_rl_v1.v5_grpo_group_control.evaluator')
    with (ROOT/'evaluator.log').open('a') as log:
        subprocess.run([sys.executable, '-c', 'import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def qualification(ledger):
    t = create(168, C['domain'], True, ledger)
    initial = w.snapshot(t)
    z = w.features(t)
    assert np.isfinite(z).all() and e.same(initial, w.snapshot(t))
    assert np.isfinite(w.feedback(t, 'online')) and w.feedback(t, 'audit') is None
    assert e.same(initial, w.snapshot(t))
    for action in (0, 2):
        w.restore(t, initial)
        w.advance(t, ledger, 'smoke', f'{action}/a', action, 2)
        expected = w.snapshot(t)
        w.restore(t, initial)
        w.advance(t, ledger, 'smoke', f'{action}/b', action, 2)
        assert e.same(expected, w.snapshot(t)), 'exact replay failed'
    w.save_student(t, ROOT/'resume.private.pt')
    saved = torch.load(ROOT/'resume.private.pt', map_location='cpu', weights_only=False)
    w.restore(t, initial); w.restore(t, saved['state'])
    assert e.same(saved['state'], w.snapshot(t)), 'checkpoint resume failed'
    e.write(ROOT/'QUALIFICATION.json', dict(status='PASS', student_calls=8,
        feature_immutable=True, feedback_immutable=True, exact_action_replay=True,
        checkpoint_restore=True, label_budget=t.provider.budget_receipt))


def configure():
    e.Provider = BudgetProvider; e.Trainer.feedback_score = feedback_score
    w.create = create; v4.create = create
    w.initialize = initialize; w.load_final = load_final; w.evaluate_file = evaluate_file
    p.DEVELOPMENT_SEEDS = (168,); p.LEARNERS = ('GRPO_FS', 'PPO_MATCHED')
    p.GROUP_SIZE = C.get('group_size', 4); p.GROUPS = C.get('trajectories',8)//p.GROUP_SIZE
    p.budget = lambda: dict(protocol='V6_LOW_LABEL', wall_clock_limit=None)
    # All groups start at the same source/seed/data/RNG state. Its complete state is
    # checked by the existing learner on every use; group index is not a new cache key.
    old_identity = w.reference_identity
    w.reference_identity = lambda t,g: dict(old_identity(t,0), reference_reuse='same_complete_entry_V6')
    old_update = q.update
    def update(*args, **kwargs):
        kwargs.update(minibatches=p.GROUP_SIZE, epochs=4, entropy_weight=C.get('entropy_weight',.01))
        kwargs['clip_low'] = C.get('clip',.2); kwargs['clip_high'] = C.get('clip',.2)
        return old_update(*args, **kwargs)
    q.update = update
    old_optimizers = q.optimizers
    q.optimizers = lambda model: old_optimizers(model, C.get('actor_lr',.0003))


def main():
    configure(); torch.set_num_threads(2); torch.cuda.set_device(0)
    lock = (ROOT/'EXECUTOR.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'archive failed attempt before explicit repair retry'
    ledger = v4.Ledger(ROOT); ledger.caps = dict(C['caps']); ledger.seen = set()
    v4.status('STARTING', job=C['job'])
    jobs = dict(qualification=qualification, panel=v4.panel, initialize=initialize,
                reference=w.reference, calibration=w.calibration, scale=w.scale,
                learn=w.learn, endpoint=w.endpoint, evaluate=w.evaluate)
    try:
        with w.NativeOperations(ROOT/'operations'): jobs[C['job']](ledger)
        w.audit(ledger)
        e.write(ROOT/'FINAL.json', dict(status='COMPLETE', time=time.time(), commit=C['commit'],
            physical_calls=dict(ledger.count), peak_cuda_allocated=torch.cuda.max_memory_allocated()))
        v4.status('COMPLETE', job=C['job'])
    except BaseException as exc:
        v4.status('FAILED', error=repr(exc), traceback=traceback.format_exc(), physical_calls=dict(ledger.count))
        raise
    finally:
        e.write(ROOT/'MEASURED_COSTS.json',dict(w.COST))

if __name__ == '__main__': main()
