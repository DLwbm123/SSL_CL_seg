import os
from pathlib import Path
from ..f5_module_pilots_v1 import run as engine
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.native_data import NativeCurrentDomain
from ..five_frameworks_v1.native_parent import NativeLRParent
from . import protocol


class BaselineTrainer(StageTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_stats = {}
        self.gradient_rows = []

    def kl_objective(self, images, logits, target, valid):
        self.module_stats = dict(confidence_only=True, accepted=int(valid.sum()),
                                 teacher_detached=not target.requires_grad,
                                 prototype_filter_used=False)
        self.telemetry['confidence_only_calls'] = self.telemetry.get('confidence_only_calls', 0) + 1
        self.telemetry['confidence_accepted_pixels'] = self.telemetry.get('confidence_accepted_pixels', 0) + int(valid.sum())
        return super().kl_objective(images, logits, target, valid)


def construct(config, j, stage, device, permit, previous=None, native=None, source_id=None, allow_u=True):
    if native is None:
        native, receipt = engine.source(config, j['seed'], device)
        source_id = dict(node_id=receipt['node_id'], student_hash=receipt['student_hash'], seed=j['seed'], domain='REFUGE')
    provider = NativeCurrentDomain(config['data'], j['seed'], j['order'], stage, source_id, device, permit, allow_u=allow_u)
    options = {**protocol.OPTIONS, 'total_steps': protocol.STEPS[provider.domain]}
    model = Model(NativeLRParent(native, j['seed'], source_id), protocol.B1, previous=previous).to(device)
    return BaselineTrainer(model, provider, options, execution=permit)


def bind():
    for name in ('plan', 'jobs', 'CAPS', 'ARMS', 'compare'):
        setattr(engine, name, getattr(protocol, name))
    engine.construct = construct
    original_readout = engine.final_readout

    def readout(config):
        original_readout(config)
        root = Path(config['run_root']); result = protocol.read(root/'RESULTS.json')
        for version, arm in ((13, 'CLASS_REFERENCE'), (17, 'SARD')):
            rows = protocol.read(config[f'v{version}_comparison_results'])['rows']
            result[f'vs_V{version}'] = protocol.paired_compare(result['rows']+rows, arm, arms=protocol.ARMS)
        protocol.write(root/'RESULTS.json', result)

    engine.final_readout = readout


def main():
    bind()
    config = protocol.read(os.environ['EXEC_CONFIG'])
    if os.environ.get('EXEC_MODE') == 'qualify':
        from .tests import qualify
        qualify(config)
    else:
        engine.main()
