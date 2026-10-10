"""OFT-inspired Cayley rotation inside the existing F5 frozen Q subspace."""
import torch
from ..five_frameworks_v1.kernels import StageSubspaceAdapter

class CayleyAdapter(StageSubspaceAdapter):
    def __init__(self,q,previous=None,*,cayley_enabled=True):
        super().__init__(q,previous);self.cayley_enabled=cayley_enabled
        if cayley_enabled:
            f=self.previous.detach();eye=torch.eye(f.shape[0],device=f.device,dtype=f.dtype)
            if not torch.allclose(f.T@f,eye,atol=2e-5,rtol=2e-5):raise ValueError('Cayley requires own orthogonal predecessor, not an archived linear F5 prefix')
    def current_matrix(self):
        if not self.cayley_enabled:return self.r
        if not torch.isfinite(self.r).all():raise FloatingPointError('nonfinite Cayley parameter')
        skew=(self.r-self.r.T)/2;eye=torch.eye(skew.shape[0],device=skew.device,dtype=skew.dtype)
        rotation=torch.linalg.solve(eye-skew,eye+skew)
        if not torch.isfinite(rotation).all():raise FloatingPointError('nonfinite Cayley rotation')
        return rotation-eye
    @torch.no_grad()
    def update_teacher(self,teacher,decay=.99):
        if teacher.cayley_enabled!=self.cayley_enabled:raise ValueError('student/teacher Cayley contract mismatch')
        # EMA the raw generator; averaging rotations would lose orthogonality.
        teacher.r.mul_(decay).add_(self.r,alpha=1-decay)
