"""Fixed discrete spatial L2 metric for local variational inference."""
import torch
from torch.nn import functional as F
from sc_bayeseg.split_v1.model import Model as SplitModel
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state


class Model(SplitModel):
    def audit_elbo(self,image,shifts):
        rng=capture_rng_state();values=[]
        try:
            seed_everything(50163)
            with torch.no_grad():
                for _ in range(2):
                    SplitModel.draw(self)
                    for _ in range(4):
                        h,logq=self.proposal(image,shifts)
                        joint,_,_=self.log_joint(image,h)
                        values.append((joint-logq)/self.size**2)
            result=torch.stack(values).mean(0)
            assert torch.isfinite(result).all()
            return dict(per_image=[float(v) for v in result],mean=float(result.mean()))
        finally:
            self.clear_draw();restore_rng_state(rng)

    def prediction(self, image, corrected=True, refine=3, parameter_samples=2, local_samples=4):
        native = image.shape[-2:]
        image = F.interpolate(image, (self.size, self.size), mode='bilinear', align_corners=False)
        self.calls['predictive_images'] += len(image)
        if not corrected:
            h, _ = self.proposal(image, sample=False)
            p = torch.cat([torch.zeros_like(h['ell'][:, :1]), h['ell']], 1).softmax(1)
            return F.interpolate(p, native, mode='bilinear', align_corners=False), {}
        flags = {name: p.requires_grad for name, p in self.named_parameters()}
        self.requires_grad_(False)
        try:
            with torch.enable_grad():
                shifts = {key: (torch.zeros(shape, device=image.device, requires_grad=True),
                                torch.zeros(shape, device=image.device, requires_grad=True))
                          for key, shape in dict(x=(len(image), 1, self.size, self.size),
                                                 m=(len(image), 1, self.size, self.size),
                                                 ell=(len(image), 3, self.size, self.size),
                                                 c=(len(image), 16)).items()}
                self.mesh_diagnostics={"initial_elbo":self.audit_elbo(image,shifts),"trace":[]}
                for local_step in range(refine):
                    self.draw()
                    h, logq = self.proposal(image, shifts)
                    joint, _, _ = self.log_joint(image, h)
                    # Sum retains per-instance update scale, independent of batch size.
                    loss = -(joint - logq).sum() / (self.size ** 2)
                    flat = [value for pair in shifts.values() for value in pair]
                    self.audit_callback("reverse",dict(local_step=local_step+1))
                    gradients = torch.autograd.grad(loss, flat)
                    self.calls['refinement_derivatives'] += 1
                    scales=[self.size**2 if p.ndim==4 else 1 for p in flat]
                    updated = [(p - .05 * g * scale).detach().requires_grad_() for p,g,scale in zip(flat,gradients,scales)]
                    assert all(torch.isfinite(v).all() for v in updated)
                    labels=[k+"_"+part for k in shifts for part in ["mean","logsd"]]
                    row=dict(step=local_step+1,loss=float(loss.detach()),blocks={k:dict(gradient_rms=float(g.square().mean().sqrt()),delta_rms=float((a-p).square().mean().sqrt()),shift_rms=float(a.square().mean().sqrt()),scale=scale) for k,p,g,a,scale in zip(labels,flat,gradients,updated,scales)})
                    self.mesh_diagnostics['trace'].append(row)
                    self.audit_callback("local_update",row)
                    shifts = {key: tuple(updated[2*i:2*i+2]) for i, key in enumerate(shifts)}
            self.mesh_diagnostics["final_elbo"]=self.audit_elbo(image,shifts)
            weights, predictions = [], []
            with torch.no_grad():
                for _ in range(parameter_samples):
                    self.draw()
                    for _ in range(local_samples):
                        h, logq = self.proposal(image, shifts)
                        joint, _, p = self.log_joint(image, h)
                        weights.append(joint - logq)
                        predictions.append(p)
                weights = torch.stack(weights).softmax(0)
                p = (weights[:, :, None, None, None] * torch.stack(predictions)).sum(0)
                ess = 1 / weights.square().sum(0)
                aggregate = weights.reshape(parameter_samples, local_samples, -1).sum(1)
                diagnostics = dict(ess=float(ess.mean()), maximum_weight=float(weights.max(0).values.mean()),
                                   parameter_ess=float((1 / aggregate.square().sum(0)).mean()))
                return F.interpolate(p, native, mode='bilinear', align_corners=False), diagnostics
        finally:
            self.clear_draw()
            for name, p in self.named_parameters():
                p.requires_grad_(flags[name])

