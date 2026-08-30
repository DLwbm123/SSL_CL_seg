"""Finite same-pair, same-Gaussian FP64 reference; never rescue a native guard."""
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import inspect_gate1c_decomposition as inspector
from di_dmpa_gate1.feature_extraction import state_groups, state_hash
from di_dmpa_jascl.checkpoint import capture_rng_state
from di_dmpa_gate1c_v2 import binding as b, gradients as g, execution as e
from di_dmpa_gate1c_v2.runner import disk_hashes

PREREG = '136f19fd9b4ba75dc8f4891e4d7601c58d7d90fb'
NAME = 'GATE1C_V21_FP64_REFERENCE_PREREGISTRATION'
HASHES = {'md': 'd6950a517c540a83d0018972f67ee9462bb7d619d58b59a042dcc3181a691104',
          'json': 'b6933c33ae5425d73818db9c22859bdcd4f3b84c5ac5363412dd36a2e89c0824'}


def trace_values(error, module, function):
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if Path(frame.f_code.co_filename).resolve() == Path(module.__file__).resolve() and frame.f_code.co_name == function:
            return dict(frame.f_locals)
        tb = tb.tb_next
    raise b.ProtocolError('expected native traceback frame missing: '+function)


def rng_hash():
    state = capture_rng_state(); npstate = state['numpy']
    return b.H(dict(python=state['python'], numpy=[npstate[0], npstate[1].tolist(), *npstate[2:]],
        torch_cpu=b.tensor_hash(state['torch_cpu']), torch_cuda=[b.tensor_hash(x) for x in state['torch_cuda']]))


@contextmanager
def observe_draws(shape):
    original = torch.randn_like; draws = []

    def observe(value, *args, **kwargs):
        b.require(len(draws) < 3 and list(value.shape) == list(shape) and value.dtype == torch.float32, 'unexpected native Gaussian call')
        result = original(value, *args, **kwargs)
        b.require(result.dtype == torch.float32 and not result.requires_grad, 'native Gaussian dtype/detach changed')
        b.finite(result); draws.append(result.detach().clone())
        return result

    with patch.object(torch, 'randn_like', observe):
        yield draws


@contextmanager
def replay_draw(draw):
    calls = 0
    b.require(draw.dtype == torch.float32 and not draw.requires_grad, 'source draw must be detached float32')

    def replay(value, *args, **kwargs):
        nonlocal calls
        b.require(calls == 0 and not args and not kwargs and value.shape == draw.shape and value.dtype == torch.float64, 'unexpected reference Gaussian call')
        calls += 1
        return draw.to(device=value.device, dtype=value.dtype)

    with patch.object(torch, 'randn_like', replay):
        yield
    b.require(calls == 1, 'reference did not consume exactly one captured draw')


def compare(native, reference):
    native = np.asarray(native, np.float64).reshape(-1); reference = np.asarray(reference, np.float64).reshape(-1)
    b.require(native.shape == reference.shape and native.size > 0, 'reference comparison geometry')
    b.finite(native, reference)
    nn = float(np.linalg.norm(native)); rn = float(np.linalg.norm(reference)); delta = native-reference
    return dict(native_sha256=b.array_hash(native), reference_sha256=b.array_hash(reference),
        max_abs_error=float(np.abs(delta).max()), native_l2_norm=nn, reference_l2_norm=rn,
        relative_l2=None if rn == 0 else float(np.linalg.norm(delta))/rn,
        cosine=None if nn == 0 or rn == 0 else float(np.clip(np.dot(native, reference)/(nn*rn), -1, 1)))


def after_error_isolation(unit):
    result = g.isolation(unit['models'], unit['legacy'], unit['before'], unit['current'], unit['history'])
    legacy_after = b.tensor_hash(unit['legacy'])
    b.require(legacy_after == unit['legacy_before'], 'legacy bank changed after exception')
    return dict(**result, legacy_before=unit['legacy_before'], legacy_after=legacy_after,
        current_history_before=list(unit['before']),
        current_history_after=[b.array_hash(unit['current']), b.array_hash(unit['history'])],
        model_state={key: state_groups(model) for key, model in unit['models'].items()})


def reference(pair, native, draw):
    student = pair['models']['student']
    b.require(all(not m.training for m in student.modules()), 'reference mode must match native eval')
    original_state = state_groups(student); rng_before = rng_hash()
    frozen_inputs = {key: b.tensor_hash(native[key]) for key in ('target', 'weights', 'predicted')}
    shadow = copy.deepcopy(student).double()
    expected = {key: value.double() if value.is_floating_point() else value for key, value in student.state_dict().items()}
    b.require(state_hash(shadow.state_dict()) == state_hash(expected), 'shadow conversion changed source values')
    b.require(all(x.data_ptr() != y.data_ptr() for x, y in zip(student.parameters(), shadow.parameters())), 'shadow aliases original parameters')
    before = state_groups(shadow); parts = g.partition(shadow)
    with replay_draw(draw):
        logits, features = shadow(pair['xu'].double(), stochastic_classifier=True)
    b.finite(logits, features)
    probability = logits.softmax(1)
    target = native['target'].detach().double(); weights = native['weights'].detach().double()
    predicted = native['predicted'].detach()
    b.require(torch.equal(target.to(native['target']), native['target']) and
        torch.equal(weights, native['weights']) and torch.equal(predicted, native['predicted']), 'frozen reference inputs changed')
    kwargs = dict(probability=probability, target=target, weights=weights, predicted=predicted, normalization='class_balanced')
    loss = g.objective(**kwargs); vector = g.vectors(g.grad(loss, parts), parts)
    components = [g.vectors(g.grad(g.objective(**kwargs, class_component=c), parts), parts) for c in range(3)]
    blocks = {}; comparisons = {}
    for block in ('global', *g.BLOCKS):
        row = inspector.decomposition_summary(vector[block], [x[block] for x in components])
        row['worst_parameter'] = inspector.parameter_location(parts, block, row['worst_flat_index'])
        blocks[block] = row
        comparisons[block] = dict(total=compare(native['vector'][block], vector[block]),
            classes=[compare(native['class_vectors'][c][block], components[c][block]) for c in range(3)])
    after = state_groups(shadow); rng_after = rng_hash()
    b.require(before == after and original_state == state_groups(student), 'original or shadow model state changed')
    b.require(rng_before == rng_after, 'reference advanced RNG')
    b.require(frozen_inputs == {key: b.tensor_hash(native[key]) for key in frozen_inputs}, 'reference changed frozen target/weight/stratum')
    b.require(all(p.grad is None for model in (student, shadow) for p in model.parameters()), 'reference parameter.grad write')
    forward = {key: compare(a.detach().double().cpu().numpy(), z.detach().cpu().numpy()) for key, a, z in
        (('logits', pair['sl'], logits), ('features', pair['sf'], features), ('probabilities', native['student_probability'], probability))}
    return dict(dtype='torch.float64', candidate='R2', normalization='class_balanced', model_forwards=1, autograd_calls=4,
        total_loss=float(loss.detach()), block_details=blocks, native_reference_gradient_comparisons=comparisons,
        native_reference_forward_comparisons=forward, rng_before=rng_before, rng_after=rng_after,
        shadow_state_before=before, shadow_state_after=after, shadow_isolated_copy=True,
        inactive_gradients_verified_none=sorted(g.INACTIVE), parameter_grad_fields='None',
        native_draw_sha256=b.tensor_hash(draw), replayed_float64_draw_sha256=b.tensor_hash(draw.double()),
        frozen_input_sha256=frozen_inputs, frozen_target_weights_strata_unchanged=True)


def interpretation(details, spec):
    blocks = details['block_details']; residual = blocks['global']['relative_residual_l2']
    if not all(row['original_predicate_pass'] for row in blocks.values()) or residual is None or residual > 1e-10:
        return 'FP64_REFERENCE_NOT_SUPPORTING_HYPOTHESIS'
    row = details['native_reference_gradient_comparisons']['global']['total']; limits = spec['interpretation']['comparability_condition']
    if (row['relative_l2'] is None or row['cosine'] is None or
        row['relative_l2'] > limits['native_reference_global_gradient_relative_l2_max'] or
        row['cosine'] < limits['native_reference_global_gradient_cosine_min']):
        return 'HIGH_PRECISION_DECOMPOSITION_ONLY_NONCOMPARABLE'
    return 'SAME_PAIR_FP64_NUMERICAL_REFERENCE_SUPPORTED'


def run(code_commit, gpu):
    repo = ROOT.parents[1]; docs = ROOT/'docs/di_dmpa_jascl'
    for suffix, digest in HASHES.items():
        path = docs/f'{NAME}.{suffix}'; b.check_hash(path, digest)
        blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{PREREG}:{path.relative_to(repo)}'])
        b.require(hashlib.sha256(blob).hexdigest() == digest, 'reference preregistration blob changed')
    b.verify_ancestor(repo, PREREG, code_commit)
    spec = b.read_json(docs/f'{NAME}.json'); parent = spec['parent']
    for path, digest in [(parent['report_path'], parent['report_sha256']),
                         (parent['investigation_json_path'], parent['investigation_json_sha256'])] + [(r['path'], r['sha256']) for r in parent['replica_outcomes']]:
        b.check_hash(repo/path, digest)
    engines = ['experiments/lcrseg/'+x for x in ('di_dmpa_gate1', 'di_dmpa_gate1_v2', 'di_dmpa_gate1b_v2', 'di_dmpa_gate1c_v2', 'di_dmpa_jascl')]
    b.require(not b.git(repo, 'diff', parent['formal_code_commit'], 'HEAD', '--', *engines), 'shared engine changed')
    b.require(gpu in spec['fixed_inputs']['physical_gpus'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu), 'unexpected physical GPU')
    b.require(torch.cuda.device_count() == 1, 'one visible device per replica required')
    upstream = ROOT/'third_party/JASCL_REFERENCE'
    b.require(b.git(upstream, 'rev-parse', 'HEAD') == '3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53' and
        not b.git(upstream, 'diff', '--name-only', 'HEAD'), 'official tracked source changed')
    torch.set_num_threads(1)
    p, freeze, metadata = b.verify(ROOT, code_commit, input_contract='v2.1')
    prior = b.read_json(repo/parent['investigation_json_path']); pair = prior['pair']
    b.require(pair in p['gradient_diagnostic']['batch_pairs'], 'original pair changed')
    previous = b.read_json(repo/next(r['path'] for r in parent['replica_outcomes'] if r['gpu'] == gpu))
    formal = Path(prior['parent']['formal_attempt'])
    frozen_files = {formal/name: prior['parent'][field] for name, field in
        (('GATE1C_V2_ARTIFACT_MANIFEST.json', 'artifact_manifest_sha256'),
         ('GATE1C_V2_FAILURE.json', 'failure_sha256'), ('GATE1C_V21_STATUS.json', 'status_sha256'))}
    for path, digest in frozen_files.items(): b.check_hash(path, digest)
    output = Path(spec['execution']['output_root_prefix'])/PREREG/f'replica_gpu{gpu}'
    output.mkdir(parents=True, exist_ok=False)
    metadata.update(reference_registration_id=spec['registration_id'], reference_preregistration_commit=PREREG,
        reference_preregistration_file_sha256=HASHES, execution_scope='SAME_PAIR_FP64_REFERENCE_ONLY',
        reference_code_commit=code_commit, reference_parent_report_commit=parent['report_commit'],
        official_classifier_source_sha256=b.sha256(upstream/'Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT/models/deeplabv3/deeplab.py'),
        physical_gpu=gpu, started_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), exact_command=sys.argv,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0),
        original_formal_status=parent['formal_status'], additional_full_gate_attempt=False, model_forward_call_bound=4)
    b.write_json(output/'RUN_METADATA.json', metadata); before = disk_hashes(p)

    def timeout(signum, frame):
        raise TimeoutError('preregistered reference replica time limit exceeded')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(spec['execution']['finite_time_limit_minutes_per_replica']*60)
    try:
        with b.no_updates(), observe_draws(spec['native_observation']['expected_draw_shape']) as draws:
            try:
                e.probe_unit(ROOT, '/root/LCRSeg', p, freeze, metadata, pair['seed'], pair['stage_index'], output,
                    'cuda:0', 'draw0', pair_indices=[pair['pair_index']])
            except b.ProtocolError as error:
                observed = inspector.capture_failure(error)
                b.require(observed is not None, 'unexpected native protocol failure')
                b.write_json(output/'NATIVE_OBSERVATION.json', observed)
                b.require(observed == previous['details'], 'native observation differs from bound device receipt')
                values = trace_values(error, g, 'consistency_gradients')
                pair_values = trace_values(error, e, 'gradient_pair')
                unit = trace_values(error, e, 'probe_unit')
            else:
                raise b.ProtocolError('original native failure did not reproduce')
        b.require(len(draws) == 3, 'native draw count changed')
        isolation_before = after_error_isolation(unit)
        b.write_json(output/'ISOLATION_BEFORE_REFERENCE.json', isolation_before)
        with b.no_updates(): details = reference(pair_values, values, draws[0])
        b.write_json(output/'REFERENCE_DETAILS.json', details)
        isolation_after = after_error_isolation(unit)
        b.write_json(output/'ISOLATION_AFTER_REFERENCE.json', isolation_after)
        b.require(isolation_before == isolation_after, 'post-reference original isolation changed')
        guards = [b.read_json(path) for path in (output/'probe_models').rglob('*.json')]
        b.require(len(guards) == 1 and guards[0]['bitwise_unchanged'] and guards[0]['before'] == isolation_before['model_state'], 'native model immutability receipt mismatch')
        b.require(disk_hashes(p) == before, 'original checkpoint changed')
        for path, digest in frozen_files.items(): b.check_hash(path, digest)
        result = dict(metadata=metadata, status=interpretation(details, spec), reference=details,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), native_failure_reproduced=True,
            native_draws=[dict(role=role, shape=list(draw.shape), dtype=str(draw.dtype), sha256=b.tensor_hash(draw))
                for role, draw in zip(spec['native_observation']['expected_draw_order'], draws)],
            isolation_before_reference=isolation_before, isolation_after_reference=isolation_after,
            all9_checkpoint_hashes_unchanged=True, original_failure_manifest_unchanged=True,
            original_formal_status='BLOCKED_INCOMPLETE_EVIDENCE', scientific_admission=None,
            method_registered=False, model_optimizer_steps=0, transport_optimizer_steps=0,
            hidden_gt_training_usage='none', test_gt_usage='none',
            next_action='ANALYZE_BOTH_REFERENCE_REPLICAS_NO_AUTOMATIC_FORMAL_RETRY')
        b.write_json(output/'REFERENCE_OUTCOME.json', result)
        print(dict(output=str(output), status=result['status']), flush=True)
    except Exception as error:
        b.write_json(output/'UNEXPECTED_FAILURE.json', dict(metadata=metadata, failed_at_utc=datetime.now(timezone.utc).isoformat(),
            error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--gpu', type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    run(args.code_commit, args.gpu)
