"""Metadata contracts; none of these objects grants real data access."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ParentContract:
    status: str
    identity: str | None
    multiplication: str = 'delta_W = B @ A'
    constraint_kind: str | None = None
    feature_coordinate: str = 'post_last_nonlinearity_pre_native_readout'
    native_loss_verified: bool = False
    native_geometry_verified: bool = False
    optimizer_verified: bool = False

    def require_real_ready(self):
        if self.status!='BOUND_VERIFIED' or not self.identity or not all((self.native_loss_verified,self.native_geometry_verified,self.optimizer_verified)):
            raise RuntimeError('PARENT_BINDING_REQUIRED')
        if self.constraint_kind not in ('hard_input_projection','soft_native','none_verified'):
            raise ValueError('unknown parent constraint; no implicit hardening')


class CurrentDomainProvider(Protocol):
    seed: int
    order: int
    stage: int
    def labeled(self,step,stream='labeled'): ...
    def unlabeled(self,step): ...  # Images, geometry, source IDs; never labels.


@dataclass(frozen=True)
class DataScope:
    domain: str
    mode: str = 'CODE_ONLY'
    labeled: bool = True
    unlabeled_images: bool = False

    def require(self,domain,kind):
        if self.mode!='SYNTHETIC' or domain!=self.domain:
            raise PermissionError('no authorized real/current-domain capability')
        if kind=='L' and self.labeled:return
        if kind=='U_IMAGE' and self.unlabeled_images:return
        raise PermissionError('forbidden payload category: '+kind)
