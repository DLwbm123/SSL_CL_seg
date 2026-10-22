"""One resumable stage, with a fixed prior, durable step boundaries and one original deadline."""
import json
from pathlib import Path
import time
import torch
from .budgets import BudgetExceeded
from .local_inference import Refiner
from .objective import SequentialObjective
from .qualification import code_fingerprint, fingerprint, file_fingerprint
from .stream_state import StreamState


def session_identity(state, ledger, name, steps, seconds, protocol_fingerprint):
    return dict(name=name, ledger_root=str(ledger.root.resolve()), target_steps=steps,
                original_budget_seconds=seconds, protocol_fingerprint=protocol_fingerprint,
                code_fingerprint=code_fingerprint(), config_fingerprint=fingerprint(state.config.to_dict()),
                stage=state.active['id'], evidence_fingerprint=state.active['evidence']['fingerprint'],
                prior_hash=state.prior_hash)


def require_resume(state, ledger, name, steps, seconds, protocol_fingerprint, checkpoint):
    if state.active is None or state.runtime is None:
        raise PermissionError('resume requires an active checkpoint with an original session')
    expected = session_identity(state, ledger, name, steps, seconds, protocol_fingerprint)
    saved = ledger.data.get('training_sessions', {}).get(name)
    if saved is None or saved['status'] == 'COMPLETE' or any(state.runtime.get(k) != v for k, v in expected.items()):
        raise PermissionError('resume cannot change stage, evidence, protocol, prior, code or budget root')
    if (saved['identity'] != state.runtime or saved['completed_steps'] != state.steps or
            saved['checkpoint_fingerprint'] != file_fingerprint(checkpoint)):
        raise PermissionError('resume must use the latest committed checkpoint')
    if time.time() >= state.runtime['deadline']:
        raise BudgetExceeded('original training deadline expired')
    return saved


def run_stage(state, data, ledger, counter, name, steps, seconds, protocol_fingerprint,
              resume_checkpoint=None, after_checkpoint=None, real_data=True, batch_size=1,
              data_seed=None, checkpoint_stride=None):
    if checkpoint_stride is not None and (type(checkpoint_stride) is not int or checkpoint_stride < 1):
        raise ValueError('checkpoint stride')
    if state.active is None or state.active.get('evidence') != data.evidence:
        raise PermissionError('active evidence differs from training data')
    if resume_checkpoint is not None:
        session = require_resume(state, ledger, name, steps, seconds, protocol_fingerprint, resume_checkpoint)
    else:
        sessions = ledger.data.setdefault('training_sessions', {})
        if state.runtime is not None or name in sessions:
            raise PermissionError('existing session requires explicit resume')
        started = time.time()
        state.runtime = dict(session_identity(state, ledger, name, steps, seconds, protocol_fingerprint),
                             started=started, deadline=started + seconds)
        session = dict(identity=state.runtime, status='RUNNING', completed_steps=0, attempts=0)
        sessions[name] = session
        ledger.flush()
    remaining = state.runtime['deadline'] - time.time()
    if remaining <= 0:
        raise BudgetExceeded('original training deadline expired')
    session['attempts'] += 1
    attempt = f'{name}_attempt{session["attempts"]:03}'
    with ledger.attempt(attempt, seconds=remaining, updates=steps-state.steps) as out:
        def checkpoint(current, filename):
            path = out / filename
            current.save(path)
            previous_path = session.get('checkpoint')
            session.update(completed_steps=current.steps, checkpoint=str(path.resolve()),
                           checkpoint_fingerprint=file_fingerprint(path))
            ledger.flush()
            # Delete only this attempt's superseded temporary recovery snapshot,
            # after the new snapshot and ledger have been durably published.
            if checkpoint_stride is not None and previous_path is not None:
                previous = Path(previous_path)
                if previous.parent == out.resolve() and previous.name.startswith('ACTIVE_STEP_'):
                    step = int(previous.stem.removeprefix('ACTIVE_STEP_'))
                    if step % checkpoint_stride and previous != path.resolve():
                        previous.unlink()
            return path
        latest = checkpoint(state, 'ACTIVE_INITIAL.pt')
        objective = SequentialObjective(state, Refiner(state.model, state.phi, counter))
        try:
            with (out / 'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                while state.steps < steps:
                    ledger.check()
                    if time.time() >= state.runtime['deadline']:
                        raise BudgetExceeded('original training deadline expired')
                    # Independent, step-indexed data streams keep L exposure paired across arms
                    # and reproduce the schedule after resume despite different model RNG use.
                    lg = state.generator if data_seed is None else torch.Generator().manual_seed(data_seed + 2 * state.steps)
                    ug = state.generator if data_seed is None else torch.Generator().manual_seed(data_seed + 2 * state.steps + 1)
                    L = data.batch('L', batch_size, lg)
                    U = data.batch('U', batch_size, ug)
                    for batch in (L, U):
                        if batch is not None:
                            batch['x'] = batch['x'].to(state.q.mean.device)
                            if 'observations' in batch:
                                batch['observations'] = {k:v.to(state.q.mean.device) for k,v in batch['observations'].items()}
                    loss, parts = objective.estimate(L, U, data.counts, state.generator)
                    state.optimizer.zero_grad()
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for g in state.optimizer.param_groups for p in g['params']):
                        raise FloatingPointError('outer gradients')
                    state.optimizer.step()
                    state.steps += 1
                    for k in ('global_updates', 'domain_updates', 'phi_updates', 'outer_backward_calls'):
                        counter.add(k)
                    n = sum(len(b['x']) for b in (L, U) if b is not None)
                    counter.add('real_train_images' if real_data else 'fixture_train_images', n)
                    counter.add('current_train_images', n)
                    latest = checkpoint(state, f'ACTIVE_STEP_{state.steps:06}.pt')
                    log.write(json.dumps({'step': state.steps, **{k: float(v) for k, v in parts.items()}}) + '\n')
                    log.flush()
                    if after_checkpoint is not None:
                        after_checkpoint(state.steps)
            state.finalize_task(counter.export())
            state.save(out / 'PRIVATE_SEALED_STATE.pt')
            session['status'] = 'COMPLETE'
            (out / 'EXECUTION_RECEIPT.json').write_text(json.dumps(
                dict(execution='COMPLETE', scientific_state='NOT_QUALIFIED', completed_steps=state.steps,
                     original_deadline=state.runtime['deadline']), indent=2))
            ledger.flush()
        except BaseException as exc:
            # Resume the last completed transaction, never an interrupted batch's advanced RNG or optimizer.
            partial = StreamState.load(latest)
            checkpoint(partial, 'PARTIAL_STATE.pt')
            session['status'] = 'FAILED_RECOVERABLE'
            (out / 'FAILURE_RECEIPT.json').write_text(json.dumps(
                dict(execution='FAILED', error_type=type(exc).__name__, completed_steps=partial.steps,
                     scientific_state='NOT_QUALIFIED', original_deadline=partial.runtime['deadline']), indent=2))
            ledger.flush()
            raise
    return state
