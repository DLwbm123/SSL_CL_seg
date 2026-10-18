"""Adapter to the existing shared optimizer engine; no additional loss terms."""
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput
from lcrseg.methods.base import ContinualSegMethod


class Method(ContinualSegMethod):
    method_name = 'sc_bayeseg_collapsed'

    def __init__(self, model, labeled_count, unlabeled_count):
        super().__init__(model)
        self.nl, self.nu = labeled_count, unlabeled_count

    def training_step(self, labeled_batch, unlabeled_batch, global_step, site_step):
        size = (self.model.size, self.model.size)
        if not labeled_batch.valid_mask.all():
            raise ValueError('pilot requires complete observed L images without cutout')
        image = F.interpolate(labeled_batch.image, size, mode='bilinear', align_corners=False)
        label = F.interpolate(labeled_batch.label[:, None].float(), size, mode='nearest')[:, 0].long()
        self.model.draw()
        try:
            bound_l, terms, diag = self.model.bound(image, label)
            nll_l = -self.nl * bound_l.mean()
            nll_u = nll_l * 0
            if self.nu:
                if unlabeled_batch is None:
                    raise ValueError('unlabeled observation batch missing')
                u = F.interpolate(unlabeled_batch.weak_image, size, mode='bilinear', align_corners=False)
                bound_u, _, udiag = self.model.bound(u)
                nll_u = -self.nu * bound_u.mean()
                diag.update({'U_' + k: v for k, v in udiag.items()})
            kl = self.model.kl()
            # One common normalization of the entire statistical objective.
            scale = (self.nl + self.nu) * self.model.size ** 2
            total = (nll_l + nll_u + kl) / scale
            diag.update(negative_bound_L_per_pixel=float((-bound_l.mean() / self.model.size ** 2).detach()),
                        label_ce_per_pixel=float((-terms['labels'].mean() / self.model.size ** 2).detach()),
                        global_kl=float(kl.detach()), labeled_slices=self.nl, unlabeled_slices=self.nu)
            return MethodStepOutput(total, dict(joint_L=nll_l / scale,
                                    marginal_U=nll_u / scale, posterior_kl=kl / scale), diag)
        finally:
            self.model.clear_draw()
