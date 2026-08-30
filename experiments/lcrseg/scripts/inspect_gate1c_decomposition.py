"""Observe one preregistered native failure; never alter the shared gradient guard."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from di_dmpa_gate1c_v2 import binding as b, gradients as g
from di_dmpa_gate1c_v2.execution import probe_unit
from di_dmpa_gate1c_v2.runner import disk_hashes

PREREG = '6477a1c240a49c0c365217c16c6ff7ca0a5163e8'
NAME = 'GATE1C_V21_DECOMPOSITION_INVESTIGATION_PREREGISTRATION'
HASHES = {'md': '25f7ed58cc1aec5f18d6ff370acfb357b2715fd5878a8e2850dcbd20dee8de10',
          'json': '26c46ae673b5d5831a057598dbfe4fd0e70ca1be809e0d42c84b8f29360f90d5'}
ERROR = 'class gradient decomposition outside preregistered tolerance'


def decomposition_summary(total, components):
    total = np.asarray(total, np.float64).reshape(-1)
    components = [np.asarray(x, np.float64).reshape(-1) for x in components]
    b.require(total.size > 0 and len(components) == 3 and all(x.shape == total.shape for x in components), 'component geometry')
    b.finite(total, *components)
    summed = sum(components)
    residual = np.abs(total-summed)
    bound = 1e-6+1e-4*np.abs(summed)
    ratio = residual/bound
    index = int(np.argmax(ratio))
    norm = float(np.linalg.norm(total)); residual_norm = float(np.linalg.norm(total-summed))
    return dict(numel=total.size, atol=1e-6, rtol=1e-4,
        original_predicate_pass=bool(np.allclose(total, summed, atol=1e-6, rtol=1e-4)),
        violating_coordinates=int(np.count_nonzero(residual > bound)), max_abs_error=float(residual.max()),
        max_tolerance_ratio=float(ratio[index]), total_l2_norm=norm, component_sum_l2_norm=float(np.linalg.norm(summed)),
        residual_l2_norm=residual_norm, relative_residual_l2=residual_norm/norm if norm else None,
        worst_flat_index=index, worst_total=float(total[index]), worst_component_sum=float(summed[index]),
        worst_components=[float(x[index]) for x in components], worst_bound=float(bound[index]),
        total_sha256=b.array_hash(total), component_sum_sha256=b.array_hash(summed),
        component_sha256=[b.array_hash(x) for x in components])


def parameter_location(parts, block, offset):
    indices = [i for key in g.BLOCKS for i in parts['blocks'][key]] if block == 'global' else parts['blocks'][block]
    for i in indices:
        parameter = parts['params'][i]
        if offset < parameter.numel():
            return dict(name=parts['names'][i], shape=list(parameter.shape),
                        index=[int(x) for x in np.unravel_index(offset, tuple(parameter.shape))])
        offset -= parameter.numel()
    raise ValueError('gradient coordinate outside parameter partition')


def leaf_checks(locals_):
    """Separate algebra check on detached copies, not replacement model gradients."""
    result = []
    for dtype in (torch.float32, torch.float64):
        probability = locals_['student_probability'].detach().to(device='cpu', dtype=dtype).clone().requires_grad_(True)
        target = locals_['target'].detach().to(device='cpu', dtype=dtype)
        weights = locals_['weights'].detach().to(device='cpu', dtype=torch.float64)
        predicted = locals_['predicted'].detach().cpu()
        kwargs = dict(probability=probability, target=target, weights=weights,
                      predicted=predicted, normalization=locals_['normalization'])
        loss = g.objective(**kwargs)
        full = torch.autograd.grad(loss, probability)[0]
        component_losses = []; components = []
        for c in range(3):
            part = g.objective(**kwargs, class_component=c)
            component_losses.append(float(part.detach()))
            components.append(torch.autograd.grad(part, probability)[0].detach().numpy())
        b.require(probability.grad is None, 'leaf parameter.grad write')
        result.append(dict(dtype=str(dtype), device='cpu', gradient_receiver='detached_probability_leaf_only',
            model_forwards=0, model_parameter_gradients=0, total_loss=float(loss.detach()),
            class_losses=component_losses, scalar_sum_abs_error=abs(float(loss.detach())-sum(component_losses)),
            gradient_sum=decomposition_summary(full.detach().numpy(), components)))
    return result


def capture_failure(error):
    if not isinstance(error, b.ProtocolError) or str(error) != ERROR:
        return None
    frame = None; tb = error.__traceback__
    while tb is not None:
        if (Path(tb.tb_frame.f_code.co_filename).resolve() == Path(g.__file__).resolve()
                and tb.tb_frame.f_code.co_name == 'consistency_gradients'):
            frame = tb.tb_frame
        tb = tb.tb_next
    b.require(frame is not None, 'native failure traceback missing')
    values = frame.f_locals
    summaries = {}
    for block in ('global', *g.BLOCKS):
        row = decomposition_summary(values['vector'][block], [x[block] for x in values['class_vectors']])
        row['worst_parameter'] = parameter_location(values['parts'], block, row['worst_flat_index'])
        summaries[block] = row
    b.require(not summaries[values['block']]['original_predicate_pass'], 'native failing predicate did not reproduce in observation')
    b.require(all(p.grad is None for p in values['parts']['params']), 'native parameter.grad write')
    with b.no_updates():
        leaves = leaf_checks(values)
    return dict(exception_type=type(error).__name__, exception_message=str(error),
        candidate=values['candidate'], normalization=values['normalization'], first_failed_block=values['block'],
        native_autograd_dtype=str(values['parts']['params'][0].dtype), observed_numpy_gradient_dtype=str(values['vector']['global'].dtype),
        parameter_grad_fields='None', block_details=summaries, leaf_only_audit=leaves,
        original_guard_raised=True, original_guard_replaced=False, native_failure_rescued=False,
        probability_sha256=b.tensor_hash(values['student_probability']), target_sha256=b.tensor_hash(values['target']),
        weights_sha256=b.tensor_hash(values['weights']), predicted_sha256=b.tensor_hash(values['predicted']),
        backend_flags=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))


def run(code_commit, gpu):
    docs = ROOT/'docs/di_dmpa_jascl'; repo = ROOT.parents[1]
    for suffix, digest in HASHES.items():
        path = docs/f'{NAME}.{suffix}'
        b.check_hash(path, digest)
        blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{PREREG}:{path.relative_to(repo)}'])
        b.require(hashlib.sha256(blob).hexdigest() == digest, 'investigation registration blob changed')
    b.verify_ancestor(repo, PREREG, code_commit)
    spec = b.read_json(docs/f'{NAME}.json'); parent = spec['parent']
    paths = ['experiments/lcrseg/'+x for x in ('di_dmpa_gate1', 'di_dmpa_gate1_v2', 'di_dmpa_gate1b_v2', 'di_dmpa_gate1c_v2', 'di_dmpa_jascl')]
    b.require(not b.git(repo, 'diff', parent['diagnostic_code_commit'], 'HEAD', '--', *paths), 'shared engine changed')
    b.require(gpu in spec['native_probe']['physical_gpus'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu), 'unexpected physical GPU')
    b.require(torch.cuda.device_count() == 1, 'single visible device required per original worker')
    torch.set_num_threads(1)
    p, freeze, metadata = b.verify(ROOT, code_commit, input_contract='v2.1')
    pair = spec['pair']; b.require(pair in p['gradient_diagnostic']['batch_pairs'], 'original fixed pair changed')
    formal = Path(parent['formal_attempt'])
    b.check_hash(formal/'GATE1C_V2_ARTIFACT_MANIFEST.json', parent['artifact_manifest_sha256'])
    b.check_hash(formal/'GATE1C_V2_FAILURE.json', parent['failure_sha256'])
    b.check_hash(formal/'GATE1C_V21_STATUS.json', parent['status_sha256'])
    b.require(b.read_json(formal/'GATE1C_V21_STATUS.json')['status'] == 'BLOCKED_INCOMPLETE_EVIDENCE', 'parent failure changed')
    output = Path(spec['execution']['output_root_prefix'])/PREREG/f'replica_gpu{gpu}'
    output.mkdir(parents=True, exist_ok=False)
    before = disk_hashes(p)
    metadata.update(investigation_registration_id=spec['registration_id'], investigation_preregistration_commit=PREREG,
        investigation_preregistration_file_sha256=HASHES, source_parent_attempt=parent,
        execution_scope='SINGLE_PAIR_DECOMPOSITION_INVESTIGATION_ONLY', physical_gpu=gpu,
        started_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), exact_command=sys.argv,
        additional_full_gate_attempt=False, model_forward_call_bound=3)
    b.write_json(output/'RUN_METADATA.json', metadata)
    try:
        with b.no_updates():
            probe_unit(ROOT, '/root/LCRSeg', p, freeze, metadata, pair['seed'], pair['stage_index'], output,
                       'cuda:0', 'draw0', pair_indices=[pair['pair_index']])
        details = None
        status = 'ORIGINAL_FAILURE_NOT_REPRODUCED'
    except Exception as error:
        details = capture_failure(error)
        if details is None:
            b.write_json(output/'UNEXPECTED_FAILURE.json', dict(metadata=metadata, error=str(error), traceback=traceback.format_exc()))
            raise
        status = 'NATIVE_DECOMPOSITION_FAILURE_REPRODUCED'
    guards = [b.read_json(path) for path in (output/'probe_models').rglob('*.json')]
    b.require(len(guards) == 1 and guards[0]['bitwise_unchanged'] and guards[0]['before'] == guards[0]['after'], 'model mutated during investigation')
    b.require(disk_hashes(p) == before, 'original checkpoint changed')
    b.check_hash(formal/'GATE1C_V2_ARTIFACT_MANIFEST.json', parent['artifact_manifest_sha256'])
    result = dict(metadata=metadata, status=status, details=details, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        all9_checkpoint_hashes_unchanged=True, model_guard_bitwise_unchanged=True,
        original_formal_status='BLOCKED_INCOMPLETE_EVIDENCE', scientific_admission=None,
        method_registered=False, model_optimizer_steps=0, transport_optimizer_steps=0,
        hidden_gt_training_usage='none', test_gt_usage='none', next_action='ANALYZE_BOTH_FIXED_DEVICE_OUTCOMES_NO_AUTOMATIC_RETRY')
    b.write_json(output/'INVESTIGATION_OUTCOME.json', result)
    print(dict(output=str(output), status=status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--gpu', type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    run(args.code_commit, args.gpu)
