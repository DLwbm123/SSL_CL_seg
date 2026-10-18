"""Read-only fixed-expert validation of an already completed transfer run."""
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch

from care_seg_study_v1.data import evaluate
from rcf_seg_study_v1.model import Segmenter, load_deployment


SLICES = {'Philips': 660, 'GE': 288, 'Canon': 294}
BUDGET = 3 * sum(SLICES.values())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(settings):
    output = Path(settings['output'])
    output.mkdir(exist_ok=False)
    started = time.time()
    physical = 0
    rows = {}
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        trained = Path(settings['trained'])
        prior = json.loads((trained / 'RESULTS.json').read_text())
        assert prior['status'] == 'COMPLETE'
        assert json.loads((trained / 'EXIT.json').read_text())['exit_code'] == 0
        assert not (trained / 'FAILURE.json').exists()
        write(output / 'CONTEXT.json', settings)

        def count(module, inputs):
            nonlocal physical
            batch = inputs[0].shape[0]
            if physical + batch > BUDGET:
                raise RuntimeError('fixed inference budget exhausted')
            if module.training or torch.is_grad_enabled():
                raise RuntimeError('evaluation requires eval mode and disabled gradients')
            physical += batch

        for arm in ('source', 'confidence', 'rcf'):
            checkpoint = trained / ('source/sealed.pt' if arm == 'source' else arm + '_Canon/deployment.pt')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if arm == 'source':
                model = Segmenter().cuda()
                model.load_state_dict(payload['model'], strict=True)
                model.requires_grad_(False)
                model.eval()
            else:
                model = load_deployment(payload, device='cuda').model
            del payload
            assert all(not p.requires_grad for p in model.parameters())
            versions = {n: v._version for n, v in model.named_parameters()}
            versions.update({'buffer:' + n: v._version for n, v in model.named_buffers()})
            hook = model.register_forward_pre_hook(count)
            rows[arm] = {}
            for site, expected in SLICES.items():
                model.activate('source' if arm == 'source' else site)
                model.eval()
                before = physical
                write(output / 'STATUS.json', dict(status='EVALUATING', arm=arm, site=site,
                                                   slice_forwards=physical, budget=BUDGET))
                with torch.no_grad():
                    metrics, private = evaluate(model, Path(settings['data']), site)
                assert physical - before == expected
                assert all(p.grad is None and p._version == versions[n]
                           for n, p in model.named_parameters())
                assert all(v._version == versions['buffer:' + n] for n, v in model.named_buffers())
                row = dict(arm=arm, site=site, expert=model.active, slice_forwards=expected,
                           metrics=metrics, parameters_and_buffers_unmodified=True)
                rows[arm][site] = row
                write(output / ('private_' + arm + '_' + site + '.json'), private)
                write(output / (arm + '_' + site + '.json'), row)
                print(json.dumps(row), flush=True)
            hook.remove()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert physical == BUDGET == 3726
        result = dict(status='COMPLETE', fixed=rows,
                      reused_routed={arm: prior['trajectories'][arm][-1] for arm in ('confidence', 'rcf')},
                      source_validation_reused=prior['source'], slice_forwards=physical,
                      optimizer_updates=0, backward_calls=0, routed_forwards=0,
                      parameters_and_buffers_unmodified=True, training_commit=settings['training_commit'],
                      diagnostic_commit=settings['diagnostic_commit'], started=started, finished=time.time(),
                      scope='post-hoc validation diagnostic; domain IDs used only to select a fixed expert')
        write(output / 'RESULTS.json', result)
        write(output / 'STATUS.json', dict(status='COMPLETE', slice_forwards=physical))
    except BaseException:
        write(output / 'FAILURE.json', dict(traceback=traceback.format_exc(), slice_forwards=physical))
        raise


if __name__ == '__main__':
    code = 1
    settings = json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
    if Path(settings['output']).exists():
        raise FileExistsError('diagnostic output must be a new directory')
    try:
        main(settings)
        code = 0
    finally:
        output = Path(settings['output'])
        if output.exists():
            write(output / 'EXIT.json', dict(exit_code=code, time=time.time()))
