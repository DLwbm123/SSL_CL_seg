"""CPU preparation adapter; native execution deliberately unavailable."""
import hashlib
from pathlib import Path
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.parent_bridge import SyntheticParentBridge
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
from ..five_frameworks_v1.native_parent import NativeLRParent
from ..five_frameworks_v1.native_data import NativeCurrentDomain
from .core import hierarchical_kl, ARMS


class HierarchicalTrainer(StageTrainer):
    def __init__(self, model, provider, options=None, arm='C2', initialize=True, execution=None):
        native = type(model.parent) is NativeLRParent and type(provider) is NativeCurrentDomain
        synthetic = (type(model.parent) is SyntheticParentBridge and type(provider) is SyntheticCurrentDomain
                     and all(p.device.type == 'cpu' for p in model.parameters()))
        if not native and not synthetic:
            raise PermissionError('HKL requires the bound native B2 backend or the synthetic CPU fixture')
        if model.family != 'B2_PARENT_PAS_KL' or model.sidecar is not None or arm not in ARMS:
            raise ValueError('B2 main-only and registered arm required')
        self.arm = arm
        super().__init__(model, provider, options, initialize=initialize, execution=execution)

    @classmethod
    def for_resume(cls, model, provider, options=None, *, arm, execution=None):
        """Resume only through the arm-aware preparation entry point."""
        return cls(model, provider, options=options, arm=arm, initialize=False, execution=execution)

    def fine_kl(self, logits, target, valid):
        return hierarchical_kl(logits, target, valid, self.arm)

    def loss_contract(self):
        return dict(study='MAIN_HEAD_HIERARCHICAL_KL_V1', arm=self.arm, parent_threshold=.9,
                    maximum_attenuation=.5, reduction='original accepted-pixel count',
                    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))})
