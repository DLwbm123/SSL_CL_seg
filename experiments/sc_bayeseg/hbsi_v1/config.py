from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class Config:
    schema: int = 1
    channels: int = 1
    classes: int = 2
    label_groups: tuple = ()
    height: int = 16
    width: int = 16
    c_dim: int = 2
    r_channels: int = 2
    r_height: int = 4
    r_width: int = 4
    hidden: int = 4
    projection_dim: int = 4
    projection_seed: int = 1729
    dct_order: int = 4
    alpha: float = .9
    sigma_s: float = .2
    tau_b: float = .1
    nu: float = 4.
    bias_prior_sd: float = .1
    scale_floor: float = 1e-6
    initial_sd_ratio: float = .1
    inner_lr: float = 1e-4
    inner_steps: int = 3
    outer_lr: float = 3e-4
    global_mc: int = 2
    local_mc: int = 2
    seed: int = 20261022
    dtype: str = 'float64'
    common_scale: float = 1.

    def __post_init__(self):
        object.__setattr__(self,'label_groups',tuple(tuple(v) for v in self.label_groups))
        if any(not v or len(set(v))!=len(v) or any(type(k) is not int or not 0<=k<self.classes for k in v) for v in self.label_groups):
            raise ValueError('invalid parent-to-leaf label mapping')
        if self.schema != 1 or self.channels not in (1, 3) or self.classes not in (2, 3, 4):
            raise ValueError('unsupported schema/channels/classes')
        for k in ('height', 'width', 'c_dim', 'r_channels', 'r_height', 'r_width', 'hidden',
                  'projection_dim', 'dct_order', 'global_mc', 'local_mc'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k)
        if not 2 <= self.dct_order <= min(self.height, self.width):
            raise ValueError('DCT order outside grid')
        if not 0 < self.alpha < 1 or self.nu <= 2 or self.dtype not in ('float32', 'float64'):
            raise ValueError('invalid density configuration')
        for k in ('sigma_s', 'tau_b', 'bias_prior_sd', 'scale_floor', 'initial_sd_ratio',
                  'inner_lr', 'outer_lr', 'common_scale'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(k)
        if type(self.inner_steps) is not int or self.inner_steps < 0:
            raise ValueError('inner_steps')

    @property
    def b_dim(self):
        return self.dct_order ** 2 - 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        # JSON is a YAML subset; no extra parser dependency or executable tags.
        return cls(**json.loads(path.read_text()))
