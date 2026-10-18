"""SPACE-LoRA input-space allocation and post-SGD A projection, adapted to Conv2d.

Reference: OpenReview thVLeQiv87; independently written from equations and
checked against the authors' statistics.py/protection.py/space_lora.py.
"""
import copy
import math
import torch
from ..lctx_weight_memory_v0_1 import core as lr
from ..five_frameworks_v1.native_parent import NativeLRParent
from ..main_style_ssl_v1.model import Parent as DenseParent
from ..five_frameworks_v1.model import Model
from ..siemens_ssl_reproduction_v1.methods import Segmenter, optimizer

ARMS = ('dense_fixmatch', 'free_lora', 'space_lora')
STEPS = 4000
RANK = 8
RHO = .90


def allocate(h, fisher, rank=RANK, rho=RHO):
    h, fisher = h.double(), fisher.double()
    values, vectors = torch.linalg.eigh((h+h.T)/2)
    vectors = vectors[:, torch.argsort(values, descending=True)]
    mass = (vectors*(fisher@vectors)).sum(0).clamp_min(0)
    total = mass.sum()
    cap = h.shape[0]-rank
    if cap < 0 or not torch.isfinite(mass).all():
        raise ValueError('invalid SPACE geometry/statistics')
    requested = 0 if total == 0 else int(torch.searchsorted(mass.cumsum(0), rho*total))+1
    k = min(requested, cap)
    covered = float(mass[:k].sum()/total) if total > 0 else 1.
    return vectors[:, :k].float().contiguous(), dict(input_width=h.shape[0], protected=k,
        requested=requested, cap=cap, fisher_coverage=covered, target_met=covered>=rho-1e-12)


class Parent(NativeLRParent):
    identity = 'SPACE_CONV_ADAPTATION_V1'

    def __init__(self, native, bases, protected):
        super().__init__(native, 163, {'domain':'Siemens','kind':'completed_FixMatch_EMA'}, adapt=False)
        tensors = {}
        for i, name in enumerate(lr.LAYERS):
            w = native.get_parameter(name); d = w.flatten(1).shape[1]
            generator = torch.Generator().manual_seed(163+i)
            tensors[name] = dict(W0=w, V=torch.empty(d,0),
                                 G=torch.empty(RANK,d).uniform_(-1/math.sqrt(d),1/math.sqrt(d),generator=generator))
            basis = bases[name] if protected else torch.empty(d,0)
            self.register_buffer('space_basis_'+str(i), basis.to(w))
        lr.configure(native, 'LR_FREE', lambda name:tensors[name])
        self.project()
        self.configure_stage_training()

    @torch.no_grad()
    def project(self):
        for i, name in enumerate(lr.LAYERS):
            a = self.native.get_submodule(name[:-7]).A
            basis = getattr(self, 'space_basis_'+str(i))
            a.sub_((a@basis)@basis.T)
            residual = float((a@basis).norm())
            if residual > 2e-5*(float(a.norm())+1e-12):
                raise FloatingPointError('SPACE A projection residual')

    def stage_exit(self):
        super().stage_exit()
        for key in list(self._buffers):
            if key.startswith('space_basis_'): delattr(self, key)

    def semantic_metadata(self):
        return dict(identity=self.identity, rank=RANK, rho=RHO, layers=list(lr.LAYERS),
                    head='frozen', teacher='dense_effective_weight_EMA',
                    protection='post-SGD A <- A-(A@V)@V.T; momentum not projected')


def source_model(cfg):
    source = Segmenter(cfg['reference']).cuda()
    saved = torch.load(cfg['source_checkpoint'], map_location='cpu', weights_only=False)
    if (saved['method'],saved['step'],saved['physical_updates']) != ('fixmatch',43600,43600):
        raise ValueError('wrong completed source')
    source.load_state_dict(saved['ema'], strict=True)
    return source.eval().requires_grad_(False)


def setup(cfg):
    if cfg['arm'] not in ARMS: raise PermissionError('unregistered arm')
    torch.manual_seed(163); torch.cuda.manual_seed_all(163)
    source = source_model(cfg)
    native = copy.deepcopy(source.parent.native)
    if cfg['arm']=='dense_fixmatch':
        parent = DenseParent(native,163,{'domain':'Siemens','kind':'completed_FixMatch_EMA'},source=True)
    else:
        stats = torch.load(cfg['statistics'], map_location='cpu', weights_only=False)
        parent = Parent(native,stats['bases'],cfg['arm']=='space_lora')
    model = Model(parent,'MAIN_SOURCE').cuda()
    model.configure_training()
    return source,model,model.teacher(),optimizer(model,'fixmatch')


def rate(step):
    if not 0 <= step < STEPS: raise PermissionError('outside frozen pilot')
    return .03*math.cos(7*math.pi*step/(16*53300))


@torch.no_grad()
def update_ema(ema, model):
    # The shared engine records the executed optimizer call before this hook.
    if isinstance(model.parent,Parent): model.parent.project()
    model.parent.update_dense_ema(ema.parent,decay=.999)
